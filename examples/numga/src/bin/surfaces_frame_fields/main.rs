//! numga's `surfaces/frame_fields`: rotations attached to a mesh, blended into a continuous field
//! of orientations (VGA3D).
//!
//! Blending the rotors at a triangle's corners by their hats traces a chord inside the unit rotor
//! sphere. Its Darboux derivative, `2 ~chord dchord`, a map from a step to an even multivector,
//! holds the changing orientation and the changing length in different grades: in grade two the
//! edges' Whitney 1-forms times the bivectors of the rotors' transitions along them, in grade zero
//! the symmetric products of the ends' gradients times the transitions' sagittas. Over the chord's
//! squared size the bivector part is the Maurer–Cartan form of the normalized rotors, their exact
//! turning rate. On a disk a radial texture of rotors wraps the sphere of directors once: the
//! commutator of a face's two turning rates, read against the director's equatorial plane, is the
//! area it sweeps, and summed over the faces, over four pi, the charge is one. On a curved saddle,
//! frames pinned on the rim spread inward as a harmonic chord field in the even subalgebra (one
//! sparse solve on the rotors' cotangent Laplacian, `d0.adjoint() W d0`), made unit afterwards;
//! a twisted start relaxes by projected steps, each lowering the energy.
//!
//! The animation shows numga's four: frame patches, turning arcs and the chord's bowl on the
//! triangle; the disk's matched paths and their images on the director sphere; ribbon
//! cross-sections of the harmonic frames for three boundary prescriptions; and the relaxation,
//! forward and back.

mod frames;
#[cfg(test)]
mod tests;

use frames::*;
use gax::field::Field;
use gax::mesh::Topology;
use gax_numga_examples::scene3::{Scene3, panel3};
use gax_numga_examples::shared::simplicial;
use gax_numga_examples::shared::surfaces_plane;
use gax_numga_examples::signal::{frame, linspace};
use gax_numga_examples::{
    Anim, Camera, Canvas, Lens, Light, Marker, ORIGIN3, Point3, Rect, backdrop, caption,
    label_under, palette, run_with,
};
use simplicial::{exterior, faces, midpoint_split};
use surfaces_plane::Flat;

/// The still figures' time on screen, and the relaxation's (numga's 46 frames of 40 ms, out and
/// back, slowed).
const STILL: f32 = 3.5;
const RELAXING: f32 = 6.0;
const SECONDS: f32 = 3.0 * STILL + RELAXING;
/// Samples along each drawn curve, across each ribbon, the ribbons, and the meridians.
const CURVE_SAMPLES: usize = 96;
const PROFILE_SAMPLES: usize = 7;
const RIBBONS: usize = 5;
const MERIDIANS: usize = 6;
/// The views, orthographic as numga's: elevation, azimuth, and half the panel's height in
/// world units.
const TRIANGLE_VIEW: (f32, f32, f32) = (0.66, -0.84, 1.1);
const DISK_VIEW: (f32, f32, f32) = (0.70, -0.73, 1.3);
const SPHERE_VIEW: (f32, f32, f32) = (0.35, -0.73, 1.35);
const SADDLE_VIEW: (f32, f32, f32) = (0.56, -0.66, 1.5);

/// The patches' faces' colours, round each patch.
const PATCH: [Light; 4] = [palette::RED, palette::YELLOW, palette::GREEN, palette::BLUE];

struct Triangle {
    corners: [Point; 3],
    rotors: [Rotor; 3],
    /// The four samples shown: each point, its unit rotor and its rates along the probe.
    samples: Vec<(Point, Rotor, Bivector)>,
    probe: Point,
    /// The unit direction square to the triangle.
    lift: Point,
    fine: Topology<3>,
    fine_vertices: Field<Point>,
    /// The chord's squared size at each fine vertex.
    fine_norms: Vec<f64>,
}

struct Disk {
    mesh: Topology<3>,
    vertices: Field<Point>,
    /// The drawn paths on the disk and their images on the director sphere: the rings, then the
    /// meridians.
    paths: Vec<Vec<Point3>>,
    images: Vec<Vec<Point3>>,
}

/// The saddle, the rings its ribbons follow, and each ring sample's weights on the saddle's
/// vertices.
struct Saddled {
    mesh: Topology<3>,
    vertices: Field<Point>,
    rings: Vec<Vec<Point>>,
    weights: Vec<Vec<[(usize, f64); 3]>>,
    harmonic: [Field<Rotor>; 3],
    relaxed: Vec<(Field<Rotor>, f64)>,
}

struct Shown {
    triangle: Triangle,
    disk: Disk,
    saddle: Saddled,
}

/// A point as drawn.
fn drawn(p: Point) -> Point3 {
    let [x, y, z] = coordinates(p).map(|c| c as f32);
    Point3::xyz(x, y, z)
}

fn triangle() -> Triangle {
    let (corners, rotors, points) = single_triangle(TRIANGLE_SUBDIVISIONS);
    let probe = (corners[1] - corners[0]).normalized();
    let chosen = linspace(0.0, (points.len() - 1) as f64, 4);
    let samples = chosen
        .iter()
        .map(|&i| {
            let p = points[i as usize];
            let d = triangle_darboux(corners, p, rotors);
            (p, d.unit, d.whitney_rates.of(probe))
        })
        .collect();
    // The direction square to the triangle: its two sides' join, a direction of space.
    let normal = (corners[1] - corners[0]) & (corners[2] - corners[0]);
    let lift = normal.normalized().hodge();
    let flat = Topology::new(3, &[[0, 1, 2]]).expect("the triangle");
    let (fine, fine_vertices) = midpoint_split(&flat, &corners.into_iter().collect(), FINE_LEVELS);
    let fine_norms = fine_vertices
        .iter()
        .map(|p| triangle_darboux(corners, p, rotors).chord.norm_squared())
        .collect();
    Triangle {
        corners,
        rotors,
        samples,
        probe,
        lift: lift.with_orientation(),
        fine,
        fine_vertices,
        fine_norms,
    }
}

/// The level plane's mesh under a surface: each vertex's shadow.
fn shadow(mesh: &Topology<3>, vertices: &Field<Point>) -> Flat {
    let points: Vec<[f64; 2]> = vertices
        .iter()
        .map(|p| {
            let [x, y, _] = coordinates(p);
            [x, y]
        })
        .collect();
    Flat::new(&points, &faces(mesh))
}

/// The weights of a point of the level plane on the vertices of the face of the shadow holding
/// it (or the nearest's, a little outside).
fn weights_at(flat: &Flat, x: f64, y: f64) -> [(usize, f64); 3] {
    let q = gax_numga_examples::Point2::xy(x as f32, y as f32);
    let (f, w) = flat
        .locate(q)
        .or_else(|| {
            flat.locate(q.unitized() + (gax_numga_examples::ORIGIN2 - q.unitized()).gp(0.01))
        })
        .expect("on the disk");
    let face = flat.faces[f];
    core::array::from_fn(|k| (face[k], f64::from(w[k])))
}

/// Circles of the level plane, `RIBBONS` of them from 0.18 to 0.9 of `radius`.
fn rings(radius: f64) -> Vec<Vec<Point>> {
    linspace(0.18 * radius, 0.9 * radius, RIBBONS)
        .into_iter()
        .map(|r| {
            (0..CURVE_SAMPLES)
                .map(|k| {
                    let d = gax::vga2d::Vector::from_angle(
                        core::f64::consts::TAU * k as f64 / (CURVE_SAMPLES - 1) as f64,
                    );
                    point(r * d.e1(), r * d.e2(), 0.0)
                })
                .collect()
        })
        .collect()
}

fn disk() -> Disk {
    let (mesh, vertices) = triangular_disk(DISK_RADIUS, SKYRMION_DIVISIONS);
    let s = skyrmion_winding(&mesh, &vertices, SKYRMION_TWIST);
    let flat = shadow(&mesh, &vertices);
    // The meridians run out to the rim vertices nearest in direction to even turns about the
    // vertical.
    let rim: Vec<Point> = exterior(&mesh, 0)
        .into_iter()
        .map(|v| vertices.site(v))
        .collect();
    let meridians = (0..MERIDIANS).map(|k| {
        let angle = core::f64::consts::TAU * (k as f64 / MERIDIANS as f64 - 0.5);
        let d = gax::vga2d::Vector::from_angle(angle);
        let heading = point(d.e1(), d.e2(), 0.0);
        let end = rim
            .iter()
            .copied()
            .max_by(|a, b| dot(heading, a.normalized()).total_cmp(&dot(heading, b.normalized())))
            .expect("a rim");
        linspace(0.0, 0.999, CURVE_SAMPLES)
            .into_iter()
            .map(|t| end * t)
            .collect::<Vec<_>>()
    });
    let paths: Vec<Vec<Point>> = rings(0.95).into_iter().chain(meridians).collect();
    let sample = |p: Point, field: &Field<Point>| {
        let [x, y, _] = coordinates(p);
        weights_at(&flat, x, y)
            .iter()
            .map(|&(v, w)| field.site(v) * w)
            .reduce(|a, b| a + b)
            .expect("three corners")
    };
    Disk {
        paths: paths
            .iter()
            .map(|path| {
                path.iter()
                    .map(|&p| drawn(sample(p, &vertices) + vertical() * 0.012))
                    .collect()
            })
            .collect(),
        images: paths
            .iter()
            .map(|path| {
                path.iter()
                    .map(|&p| drawn(sample(p, &s.directors).normalized() * 1.012))
                    .collect()
            })
            .collect(),
        mesh,
        vertices,
    }
}

fn saddle() -> Saddled {
    let saddle = saddle_surface(DISK_RADIUS, SADDLE_DIVISIONS, SADDLE_HEIGHT);
    let flat = shadow(&saddle.mesh, &saddle.vertices);
    let rings = rings(DISK_RADIUS);
    let weights = rings
        .iter()
        .map(|ring| {
            ring.iter()
                .map(|&p| {
                    let [x, y, _] = coordinates(p);
                    weights_at(&flat, x, y)
                })
                .collect()
        })
        .collect();
    let harmonic = core::array::from_fn(|case| {
        harmonic_rotor_frames(
            &saddle.mesh,
            &saddle.vertices,
            &saddle.rim,
            &saddle.prescribed[case],
        )
        .1
    });
    let relaxed = relax_rotor_frames(
        &saddle.mesh,
        &saddle.vertices,
        &saddle.start,
        RELAXATION_STEP,
        RELAXATION_STEPS,
    );
    Saddled {
        mesh: saddle.mesh,
        vertices: saddle.vertices,
        rings,
        weights,
        harmonic,
        relaxed,
    }
}

fn shown() -> Shown {
    Shown {
        triangle: triangle(),
        disk: disk(),
        saddle: saddle(),
    }
}

/// An orthographic camera on a panel from a view.
fn camera(rect: Rect, (elevation, azimuth, half): (f32, f32, f32), target: Point3) -> Camera {
    Camera::orbit(rect, target, 20.0, azimuth, elevation, Lens::Parallel(half))
}

/// A surface's faces, lit, and its rim.
fn sheet(s: &mut Scene3, mesh: &Topology<3>, vertices: &Field<Point>, light: Light, opacity: f32) {
    for f in faces(mesh) {
        let [a, b, c] = f.map(|v| drawn(vertices.site(v)));
        let lit = s.lit(a, b, c, light);
        s.tri(a, b, c, lit, opacity);
    }
    let rim: Vec<[Point3; 2]> = exterior(mesh, 1)
        .into_iter()
        .map(|e| {
            let ends = mesh.simplex(1, e);
            [drawn(vertices.site(ends[0])), drawn(vertices.site(ends[1]))]
        })
        .collect();
    s.stroke(&rim, 1.1, palette::INK.faded(0.5));
}

/// A frame patch: a square in the frame's own `x` and `y` about `start`, with a shallow ridge up
/// its `z`, its four faces in four colours.
fn patch(s: &mut Scene3, start: Point, rotor: Rotor, size: f64) {
    let profile = [
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [0.0, 0.0, 0.16],
    ];
    let places =
        profile.map(|[x, y, z]| drawn(start + (rotor >> point(x * size, y * size, z * size))));
    for (k, light) in PATCH.iter().enumerate() {
        let (a, b, c) = (places[k], places[(k + 1) % 4], places[4]);
        let lit = s.lit(a, b, c, light.faded(0.8));
        s.tri(a, b, c, lit, 1.0);
    }
    s.polyline(
        &[places[0], places[1], places[2], places[3], places[0]],
        0.8,
        palette::INK.faded(0.6),
    );
}

/// The first figure: the frames at the corners and samples, the turning arcs of the rates, and
/// the chord's bowl.
fn draw_triangle(c: &mut Canvas, t: &Triangle) {
    let screen = c.rect();
    let below = Rect::new(
        0.0,
        screen.height() * 0.14,
        screen.width(),
        screen.height() * 0.93,
    );
    let centroid = drawn((t.corners[0] + t.corners[1] + t.corners[2]) * (1.0 / 3.0));
    let panels = [0, 1, 2].map(|i| below.column(i, 3).inset(4.0, 4.0, 4.0, 4.0));
    let base = palette::BLUE.faded(0.35);
    panel3(
        c,
        panels[0],
        camera(panels[0], TRIANGLE_VIEW, centroid),
        |s| {
            sheet(s, &t.fine, &t.fine_vertices, base, 0.6);
            for (corner, rotor) in t.corners.iter().zip(t.rotors) {
                patch(s, *corner + t.lift * 0.04, rotor, 0.13);
            }
            for &(p, unit, _) in &t.samples {
                patch(s, p + t.lift * 0.07, unit, 0.085);
            }
        },
    );
    label_under(
        c,
        panels[0],
        "CORNER FRAMES AND THEIR UNIT BLEND",
        16.0,
        0.9,
    );
    panel3(
        c,
        panels[1],
        camera(panels[1], TRIANGLE_VIEW, centroid),
        |s| {
            sheet(s, &t.fine, &t.fine_vertices, base, 0.6);
            for &(p, _, rate) in &t.samples {
                let start = p + t.lift * 0.025 - t.probe * 0.08;
                let tip = start + t.probe * 0.18;
                s.seg(drawn(start), drawn(tip), 2.0, palette::SKY);
                s.dot(drawn(tip), Marker::Dot, 5.0, palette::SKY);
                // An arc turning a ray in the rate's plane, through 2.4 times the rate.
                let ray = t.probe.commutator(rate).normalized();
                let centre = p + t.lift * 0.14;
                let arc: Vec<Point3> = linspace(0.0, 1.0, CURVE_SAMPLES)
                    .into_iter()
                    .map(|phase| {
                        let turn = (rate * (1.2 * phase)).exp();
                        drawn(centre + (turn >> ray) * 0.115)
                    })
                    .collect();
                s.polyline(&arc, 2.4, palette::ORANGE);
                s.dot(arc[arc.len() - 1], Marker::Dot, 5.0, palette::YELLOW);
            }
        },
    );
    label_under(
        c,
        panels[1],
        "THE WHITNEY RATES TURN IN THEIR PLANES",
        16.0,
        0.9,
    );
    panel3(
        c,
        panels[2],
        camera(panels[2], TRIANGLE_VIEW, centroid),
        |s| {
            let lowest = t.fine_norms.iter().copied().fold(1.0, f64::min);
            let lifted: Field<Point> = t
                .fine_vertices
                .iter()
                .zip(&t.fine_norms)
                .map(|(p, n)| p + t.lift * (1.3 * (n - 1.0)))
                .collect();
            for f in faces(&t.fine) {
                let [a, b, d] = f.map(|v| drawn(lifted.site(v)));
                let depth = f
                    .iter()
                    .map(|&v| (1.0 - t.fine_norms[v]) / (1.0 - lowest))
                    .sum::<f64>()
                    / 3.0;
                let colour = palette::SKY
                    .faded(0.45)
                    .blend(palette::BLUE.faded(0.55), depth.clamp(0.0, 1.0) as f32);
                let lit = s.lit(a, b, d, colour);
                s.tri(a, b, d, lit, 1.0);
            }
            for &corner in &t.corners {
                s.dot(drawn(corner), Marker::Dot, 7.0, palette::YELLOW);
            }
        },
    );
    label_under(
        c,
        panels[2],
        "THE CHORD'S SIZE: A BOWL PINNED AT THE CORNERS",
        16.0,
        0.9,
    );
}

/// The second figure: the disk with its paths, and their images on the sphere of directors.
fn draw_disk(c: &mut Canvas, d: &Disk) {
    let screen = c.rect();
    let below = Rect::new(
        screen.width() * 0.08,
        screen.height() * 0.14,
        screen.width() * 0.92,
        screen.height() * 0.93,
    );
    let panels = [0, 1].map(|i| below.column(i, 2).inset(6.0, 4.0, 6.0, 4.0));
    let colour = |k: usize| {
        if k < RIBBONS {
            palette::INK.faded(0.5)
        } else {
            palette::series(k - RIBBONS)
        }
    };
    panel3(c, panels[0], camera(panels[0], DISK_VIEW, ORIGIN3), |s| {
        sheet(s, &d.mesh, &d.vertices, palette::BLUE.faded(0.35), 0.7);
        for (k, path) in d.paths.iter().enumerate() {
            s.polyline(path, 2.0, colour(k));
        }
        s.dot(drawn(vertical() * 0.018), Marker::Dot, 9.0, palette::ORANGE);
    });
    label_under(c, panels[0], "PATHS ON THE DISK", 16.0, 0.9);
    panel3(c, panels[1], camera(panels[1], SPHERE_VIEW, ORIGIN3), |s| {
        s.sphere_wire(ORIGIN3, 1.0, 12, palette::GRID);
        for (k, path) in d.images.iter().enumerate() {
            s.polyline(path, 2.4, colour(k));
        }
        s.dot(drawn(vertical() * 1.02), Marker::Dot, 9.0, palette::ORANGE);
        s.dot(drawn(vertical() * -1.02), Marker::Dot, 9.0, palette::BLUE);
    });
    label_under(
        c,
        panels[1],
        "THEIR DIRECTORS: THE CENTRE AT THE NORTH POLE, THE RIM AT THE SOUTH",
        16.0,
        0.95,
    );
}

/// Ribbons along the rings over the saddle, their cross-sections turned by the rotor field: the
/// radial direction turned gives the width, the vertical turned the height.
fn ribbons(s: &mut Scene3, saddled: &Saddled, rotors: &Field<Rotor>) {
    sheet(
        s,
        &saddled.mesh,
        &saddled.vertices,
        palette::BLUE.faded(0.3),
        0.35,
    );
    let profile = linspace(-1.0, 1.0, PROFILE_SAMPLES);
    let across = |p: f64| {
        palette::RED
            .blend(palette::BLUE, ((p + 1.0) / 2.0) as f32)
            .faded(0.8)
    };
    for (ring, weights) in saddled.rings.iter().zip(&saddled.weights) {
        let sections: Vec<Vec<Point3>> = ring
            .iter()
            .zip(weights)
            .map(|(&on_plane, w)| {
                let blend = |f: &dyn Fn(usize) -> Rotor| {
                    w.iter()
                        .map(|&(v, x)| f(v) * x)
                        .reduce(|a, b| a + b)
                        .expect("corners")
                };
                let centre = w
                    .iter()
                    .map(|&(v, x)| saddled.vertices.site(v) * x)
                    .reduce(|a, b| a + b)
                    .expect("corners")
                    + vertical() * 0.09;
                let frame = blend(&|v| rotors.site(v)).normalized();
                let width = frame >> on_plane.normalized();
                let height = frame >> vertical();
                profile
                    .iter()
                    .map(|&p| {
                        drawn(centre + width * (0.065 * p) + height * (0.022 * (1.0 - p * p)))
                    })
                    .collect()
            })
            .collect();
        for (a, b) in sections.iter().zip(&sections[1..]) {
            for k in 0..PROFILE_SAMPLES - 1 {
                let light = across(0.5 * (profile[k] + profile[k + 1]));
                let lit = s.lit(a[k], a[k + 1], b[k], light);
                s.quad(a[k], a[k + 1], b[k + 1], b[k], lit, 1.0);
            }
        }
        let edge = |k: usize| {
            sections
                .iter()
                .map(|section| section[k])
                .collect::<Vec<_>>()
        };
        s.polyline(&edge(0), 1.2, palette::YELLOW);
        s.polyline(&edge(PROFILE_SAMPLES - 1), 1.2, palette::GREEN);
    }
}

/// The third figure: the harmonic frames of the three prescriptions.
fn draw_frames(c: &mut Canvas, saddled: &Saddled) {
    let screen = c.rect();
    let below = Rect::new(
        0.0,
        screen.height() * 0.14,
        screen.width(),
        screen.height() * 0.93,
    );
    let labels = [
        "A GENTLE TILT",
        "TILTED AND TWISTED",
        "TILTED AND TWISTED BACK",
    ];
    for (case, label) in labels.iter().enumerate() {
        let rect = below.column(case, 3).inset(4.0, 4.0, 4.0, 4.0);
        panel3(c, rect, camera(rect, SADDLE_VIEW, ORIGIN3), |s| {
            ribbons(s, saddled, &saddled.harmonic[case]);
        });
        label_under(c, rect, label, 16.0, 0.9);
    }
}

/// The fourth figure: the relaxation at `phase`, out and back.
fn draw_relaxation(c: &mut Canvas, saddled: &Saddled, phase: f32) {
    let screen = c.rect();
    let rect = Rect::new(
        screen.width() * 0.2,
        screen.height() * 0.13,
        screen.width() * 0.8,
        screen.height() * 0.95,
    );
    let there = 1.0 - (2.0 * phase - 1.0).abs();
    let (rotors, energy) = &saddled.relaxed[frame(there, saddled.relaxed.len())];
    panel3(c, rect, camera(rect, SADDLE_VIEW, ORIGIN3), |s| {
        ribbons(s, saddled, rotors)
    });
    label_under(c, rect, &format!("ENERGY {energy:.3}"), 24.0, 0.5);
}

fn draw(c: &mut Canvas, s: &Shown, t: f32) {
    backdrop(c);
    if t < STILL {
        draw_triangle(c, &s.triangle);
        caption(
            c,
            "FRAME FIELDS: THE DARBOUX DERIVATIVE OF BLENDED ROTORS",
            "2 ~R dR: WHITNEY 1-FORMS IN GRADE 2, THE CHORD'S STRETCH IN GRADE 0 (VGA3D)",
        );
    } else if t < 2.0 * STILL {
        draw_disk(c, &s.disk);
        caption(
            c,
            "FRAME FIELDS: A DISK WRAPS THE SPHERE OF DIRECTORS",
            "THE COMMUTATOR OF EACH FACE'S TURNING RATES IS THE AREA IT SWEEPS: CHARGE ONE (VGA3D)",
        );
    } else if t < 3.0 * STILL {
        draw_frames(c, &s.saddle);
        caption(
            c,
            "FRAME FIELDS: HARMONIC FRAMES ON A SADDLE",
            "BOUNDARY ROTORS SPREAD BY ONE SOLVE OF THE ROTORS' COTANGENT LAPLACIAN, THEN MADE UNIT (VGA3D)",
        );
    } else {
        draw_relaxation(c, &s.saddle, (t - 3.0 * STILL) / RELAXING);
        caption(
            c,
            "FRAME FIELDS: A TWISTED FIELD RELAXES",
            "PROJECTED COTANGENT STEPS ON UNIT ROTORS, THE RIM HELD; EVERY STEP LOWERS THE ENERGY (VGA3D)",
        );
    }
}

fn main() {
    run_with(
        Anim::new("frame fields", SECONDS).size(960, 540),
        shown,
        draw,
    );
}
