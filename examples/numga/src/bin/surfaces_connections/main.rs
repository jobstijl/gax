//! numga's `surfaces/connections`: connections on curved surfaces in VGA3D: parallel transport and
//! its holonomy, the connection Laplacian, trivial connections that design vector fields with
//! chosen singularities, and stripe patterns along those fields.
//!
//! Parallel transport carries a tangent vector along a surface keeping it tangent and turning it
//! no more than the surface bends. Between neighbouring unit normals the shortest turn carrying
//! the first onto the second is the square root of their product, and its action on a tangent
//! vector is the transport. Chained around a closed loop, the vector comes back turned in its
//! tangent plane by the solid angle the normals sweep: around the equator of a sphere not at all,
//! around the circle sixty degrees from the pole by half a turn.
//!
//! A connection on a mesh is a rotor from each end of each edge to its head. Its covariant
//! derivative turns each edge's tail vector into the head's tangent plane and subtracts it from
//! the head's, and with the cotangent weights its adjoint makes the connection Laplacian
//! (`covariant.adjoint() W covariant`, gax's `SparseMap::adjoint`). Because each rotor carries
//! the tail's normal onto the head's, neither ever mixes tangent with normal vectors, with no
//! charts. One backward-Euler step from a single tangent vector diffuses it along all paths at
//! once (vector heat), and the lowest eigenfields are the smoothest tangent fields, in pairs, a
//! field and its quarter turn.
//!
//! Each face's holonomy is its curvature, and the holonomies add up to two pi times the Euler
//! characteristic. Extra turns on the edges, in the heads' tangent planes, add their circulation
//! to every loop's holonomy; asking every face to turn by two pi times a chosen index and every
//! generator of the surface's handles by nothing is a linear system, solvable when the indices add
//! up to the Euler characteristic (Crane, Desbrun and Schröder, 2010). Its connection is trivial,
//! and its null space a unit field with exactly those singularities. Stripes across such a field
//! (Knöppel, Crane, Pinkall and Schröder, 2015) are the smoothest field of the same connection
//! turned further by the frequency times the field along each edge; where no phase can follow,
//! they fork. A cross field, carried out from one vertex along a tree, carries two families of
//! stripes coupled where its lines trade places.
//!
//! The animation shows numga's figures in turn: the holonomy around three latitude circles, a
//! vector heat pulse on the torus beside two of its smoothest fields, designed fields on spheres
//! and tori turning about every normal (still parallel), their stripes sliding along them, and the
//! stripes of two cross fields. numga's least-norm `lstsq` for the trivial connections is a
//! conjugate-gradient solve on `C Cᵀ` here (to the `1e-12` its checks ask), and its `eigh` a
//! shift-invert subspace iteration on gax's sparse LU (gax's `SparseMap::eigh` took seconds to a
//! minute on these Laplacians).

mod connections;
mod scenes;
#[cfg(test)]
mod tests;

use connections::*;
use gax::field::Field;
use gax_numga_examples::mesh::{Mesh, Vector3};
use gax_numga_examples::scene3::{Scene3, panel3};
use gax_numga_examples::shared::simplicial;
use gax_numga_examples::shared::surfaces_spin::{Hit, render};
use gax_numga_examples::signal::{frame, phasor};
use gax_numga_examples::{
    Anim, Camera, Canvas, Lens, Light, Marker, ORIGIN3, Point3, Rect, backdrop, caption,
    label_under, palette, run_with,
};
use scenes::*;
use simplicial::{boundary, faces};

/// Each figure's time on screen.
const FIGURE: f32 = 4.0;
const FIGURES: usize = 5;
const SECONDS: f32 = FIGURE * FIGURES as f32;
/// The steps between the holonomy's frames, and the length of its needles; the needles of the
/// other fields, as shares of the surface's radius, and the share of vertices they stand on.
const RING_STRIDE: usize = 2;
const HOLONOMY_NEEDLE: f64 = 0.25;
const NEEDLE: f64 = 0.1;
/// The views: elevation and azimuth (radians), numga's.
const TRANSPORT_VIEW: (f32, f32) = (0.489, -0.628);
const VIEW: (f32, f32) = (0.419, -0.628);
/// The stripes: how sharply dark turns into light.
const SHARPNESS: f64 = 12.0;

/// The ring colours of the holonomy.
const RINGS: [Light; 3] = [palette::RED, palette::YELLOW, palette::GREEN];

/// A surface ready to draw: its scene's quantities and the port's mesh of its coordinates.
struct Drawn {
    surface: Surface,
    mesh: Mesh,
    radius: f64,
    /// The vertices that carry a field's needles.
    needles: Vec<usize>,
}

impl Drawn {
    /// A sphere's needles stand on the vertices of the icosphere one level coarser, its first
    /// ones.
    fn sphere(surface: Surface, levels: usize) -> Drawn {
        let coarser = 10 * 4usize.pow(levels.saturating_sub(1) as u32) + 2;
        Drawn::new(surface, (0..coarser).collect())
    }

    /// A torus's needles stand on every other vertex around both circles.
    fn torus(surface: Surface, minor: usize) -> Drawn {
        let n = surface.vertices.len();
        Drawn::new(
            surface,
            (0..n)
                .filter(|v| (v / minor).is_multiple_of(2) && (v % minor).is_multiple_of(2))
                .collect(),
        )
    }

    fn new(surface: Surface, needles: Vec<usize>) -> Drawn {
        let coordinates: Field<Vector3> = surface
            .vertices
            .iter()
            .map(|p| {
                let [x, y, z] = coordinates(p);
                Vector3::new(x, y, z)
            })
            .collect();
        let radius = coordinates.iter().map(|v| v.norm()).fold(0.0, f64::max);
        let mesh = Mesh::new(coordinates, faces(&surface.mesh));
        Drawn {
            surface,
            mesh,
            radius,
            needles,
        }
    }
}

struct Shown {
    rings: Vec<Vec<Normal>>,
    carried: Vec<Vec<Point>>,
    torus: TorusFields,
    heat_torus: Drawn,
    sphere: Drawn,
    sphere_designs: Vec<(Design, Vec<usize>)>,
    field_torus: Drawn,
    torus_designs: Vec<(Design, Vec<usize>)>,
    stripe_sphere: Drawn,
    sphere_stripes: Vec<[Stripes; 2]>,
    stripe_torus: Drawn,
    torus_stripes: Vec<[Stripes; 2]>,
    crosses: Vec<(Vec<f64>, Cross)>,
}

fn shown() -> Shown {
    // The scenes are independent: each on its own thread.
    std::thread::scope(|scope| {
        let torus = scope.spawn(torus_fields);
        let spheres = scope.spawn(field_spheres);
        let tori = scope.spawn(field_tori);
        let sphere_stripes = scope.spawn(stripe_spheres);
        let torus_stripes = scope.spawn(stripe_tori);
        let crossed = scope.spawn(crosses);
        let (rings, starts) = holonomy();
        let carried = carried(&rings, &starts);
        let torus = torus.join().expect("the torus");
        let heat_torus = Drawn::torus(Surface::torus(HEAT_MAJOR, HEAT_MINOR), HEAT_MINOR);
        let (sphere, sphere_designs) = spheres.join().expect("the spheres");
        let (field_torus, torus_designs) = tori.join().expect("the tori");
        let (stripe_sphere, sphere_stripes) = sphere_stripes.join().expect("the striped spheres");
        let (stripe_torus, torus_stripes) = torus_stripes.join().expect("the striped tori");
        let (_, crosses) = crossed.join().expect("the crosses");
        Shown {
            rings,
            carried,
            torus,
            heat_torus,
            sphere: Drawn::sphere(sphere, SPHERE_LEVELS),
            sphere_designs,
            field_torus: Drawn::torus(field_torus, FIELD_MINOR),
            torus_designs,
            stripe_sphere: Drawn::sphere(stripe_sphere, STRIPE_SPHERE_LEVELS),
            sphere_stripes,
            stripe_torus: Drawn::torus(stripe_torus, STRIPE_MINOR),
            torus_stripes,
            crosses,
        }
    })
}

/// A point as drawn.
fn drawn(p: Point) -> Point3 {
    let [x, y, z] = coordinates(p).map(|c| c as f32);
    Point3::xyz(x, y, z)
}

/// An orthographic camera on a panel showing a surface of the radius.
fn camera(rect: Rect, (elevation, azimuth): (f32, f32), radius: f64) -> Camera {
    let half = 1.12 * radius as f32 * (rect.width() / rect.height()).recip().max(1.0);
    Camera::orbit(
        rect,
        ORIGIN3,
        30.0,
        azimuth,
        elevation,
        Lens::Parallel(half),
    )
}

/// A needle along a tangent vector at a point: a shaft and two strokes of its head, in the
/// tangent plane, the head spread to either side by the vector's quarter turn.
fn needle(
    s: &mut Scene3,
    at: Point,
    along: Point,
    normal: Normal,
    length: f64,
    width: f32,
    light: Light,
) {
    let tip = at + along * length;
    let side = along.commutator(plane(normal)) * (0.25 * length);
    let base = at + along * (0.62 * length);
    s.seg(drawn(at), drawn(tip), width, light);
    s.seg(drawn(base + side), drawn(tip), width, light);
    s.seg(drawn(base - side), drawn(tip), width, light);
}

/// The first figure: the holonomy around the latitude circles, at `phase` of the way round.
fn draw_holonomy(c: &mut Canvas, s: &Shown, phase: f32) {
    let screen = c.rect();
    let rect = Rect::new(
        screen.width() * 0.25,
        screen.height() * 0.12,
        screen.width() * 0.75,
        screen.height(),
    );
    let steps = s.rings.len();
    let now = frame(phase, (steps - 1) / RING_STRIDE + 1) * RING_STRIDE;
    // On the unit sphere about the origin each point is its own normal's direction.
    let at = towards;
    panel3(c, rect, camera(rect, TRANSPORT_VIEW, 1.3), |sc| {
        sc.sphere_wire(ORIGIN3, 1.0, 16, palette::GRID.faded(0.7));
        for (r, light) in RINGS.iter().enumerate() {
            let path: Vec<Point3> = s.rings.iter().map(|step| drawn(at(step[r]))).collect();
            sc.polyline(&path, 2.0, light.faded(0.6));
            for k in (0..=now).step_by(5 * RING_STRIDE) {
                let n = s.rings[k][r];
                needle(
                    sc,
                    at(n),
                    s.carried[r][k],
                    n,
                    HOLONOMY_NEEDLE,
                    1.0,
                    light.faded(0.3),
                );
            }
            let n = s.rings[now][r];
            needle(
                sc,
                at(n),
                s.carried[r][now],
                n,
                HOLONOMY_NEEDLE * 1.15,
                2.6,
                *light,
            );
        }
    });
    label_under(
        c,
        rect,
        &format!("CARRIED {now} OF {} STEPS ROUND", steps - 1),
        24.0,
        0.6,
    );
}

/// A surface rendered in a panel in the light `colour` gives each hit, with needles of a field at
/// its vertices on the side facing the camera.
#[allow(clippy::too_many_arguments)]
fn field_panel(
    c: &mut Canvas,
    rect: Rect,
    d: &Drawn,
    view: (f32, f32),
    field: Option<&Field<Point>>,
    marks: &[(usize, Light)],
    colour: impl Fn(&Hit) -> Light + Sync,
) {
    let cam = camera(rect, view, d.radius);
    render(c, &cam, rect, &d.mesh, colour);
    let eye = cam.eye().to_euclidean().map(f64::from);
    let eye = point(eye[0], eye[1], eye[2]);
    let s = &d.surface;
    panel3(c, rect, cam, |sc| {
        if let Some(field) = field {
            for &v in &d.needles {
                let n = s.normals.site(v);
                // Only the vertices whose tangent plane faces the camera.
                if dot(towards(n), eye) > 0.25 * eye.norm() {
                    let lift = at_normal(s.vertices.site(v), n, 0.01 * d.radius);
                    needle(
                        sc,
                        lift,
                        field.site(v).normalized(),
                        n,
                        NEEDLE * d.radius,
                        1.0,
                        palette::INK.faded(0.8),
                    );
                }
            }
        }
        for &(f, light) in marks {
            let centre = (0..3)
                .map(|k| s.vertices.site(s.mesh.simplex(2, f)[k]))
                .reduce(|a, b| a + b)
                .expect("corners");
            sc.dot(drawn(centre * (1.0 / 3.0)), Marker::Dot, 9.0, light);
        }
    });
}

/// A point lifted off the surface along its normal.
fn at_normal(p: Point, n: Normal, by: f64) -> Point {
    p + towards(n) * by
}

/// The direction a normal points in, as a point: the same coordinates.
fn towards(n: Normal) -> Point {
    point(n.c[0], n.c[1], n.c[2])
}

/// The light of a surface: the blue of the backdrop's grid, faint.
fn plain(_: &Hit) -> Light {
    palette::BLUE.faded(0.2)
}

/// The second figure: vector heat spreading on the torus at `phase`, and two smoothest fields.
fn draw_heat(c: &mut Canvas, s: &Shown, phase: f32) {
    let screen = c.rect();
    let below = Rect::new(
        0.0,
        screen.height() * 0.14,
        screen.width(),
        screen.height() * 0.95,
    );
    let t = &s.torus;
    let k = frame(phase, t.waves.len());
    let wave = &t.waves[k];
    let largest = wave.iter().map(|p| p.norm()).fold(0.0, f64::max);
    let amplitude: Vec<f64> = wave.iter().map(|p| p.norm() / largest).collect();
    let panels = [0, 1, 2].map(|i| below.column(i, 3).inset(4.0, 4.0, 4.0, 4.0));
    let mesh = &s.heat_torus.mesh;
    field_panel(
        c,
        panels[0],
        &s.heat_torus,
        TRANSPORT_VIEW,
        Some(wave),
        &[],
        |hit| {
            let a = hit
                .weights
                .iter()
                .zip(mesh.faces[hit.face])
                .map(|(w, v)| w * amplitude[v])
                .sum::<f64>();
            palette::BLUE.faded(0.3) + palette::ORANGE.faded(0.6 * a as f32)
        },
    );
    label_under(
        c,
        panels[0],
        &format!("VECTOR HEAT, STEP {} OF {HEAT_STEPS}", k + 1),
        16.0,
        0.9,
    );
    for (i, m) in [0, 2].into_iter().enumerate() {
        field_panel(
            c,
            panels[i + 1],
            &s.heat_torus,
            TRANSPORT_VIEW,
            Some(&t.modes[m]),
            &[],
            plain,
        );
        label_under(
            c,
            panels[i + 1],
            &format!("SMOOTHEST FIELD {}, ENERGY {:.3}", m + 1, t.energies[m]),
            16.0,
            0.9,
        );
    }
}

/// The singular faces of a design marked: orange where the field turns forward, sky backward.
fn marks(singular: &[usize], indices: &[f64]) -> Vec<(usize, Light)> {
    singular
        .iter()
        .zip(indices)
        .map(|(&f, &k)| {
            (
                f,
                if k > 0.0 {
                    palette::ORANGE
                } else {
                    palette::SKY
                },
            )
        })
        .collect()
}

/// Five panels in a row: three spheres and two tori.
fn row_of_five(c: &Canvas) -> [Rect; 5] {
    let screen = c.rect();
    let below = Rect::new(
        0.0,
        screen.height() * 0.2,
        screen.width(),
        screen.height() * 0.88,
    );
    [0, 1, 2, 3, 4].map(|i| below.column(i, 5).inset(2.0, 2.0, 2.0, 2.0))
}

/// The third figure: the designed fields, turned about every normal by `phase` of a turn.
fn draw_fields(c: &mut Canvas, s: &Shown, phase: f32) {
    let panels = row_of_five(c);
    let angle = core::f64::consts::TAU * f64::from(phase);
    let scenes = s
        .sphere_designs
        .iter()
        .zip(SPHERE_INDICES)
        .map(|(d, k)| (&s.sphere, d, k))
        .chain(
            s.torus_designs
                .iter()
                .zip(TORUS_INDICES)
                .map(|(d, k)| (&s.field_torus, d, k)),
        );
    for (rect, (surface, (design, singular), indices)) in panels.into_iter().zip(scenes) {
        let field = turned(&surface.surface, &design.field, angle);
        field_panel(
            c,
            rect,
            surface,
            VIEW,
            Some(&field),
            &marks(singular, indices),
            plain,
        );
        let label = if indices.is_empty() {
            "NO SINGULARITY".to_string()
        } else {
            let list: Vec<String> = indices.iter().map(|k| format!("{k:+}")).collect();
            format!("INDICES {}", list.join(" "))
        };
        label_under(c, rect, &label, 14.0, 0.9);
    }
}

/// The phase of each family at each corner of each face and the whole turns it winds around the
/// face, numga's: each corner's lines matched to the first corner's (the line nearest each, and
/// the sense it points in), the walk around the face along the edges' targets taking each
/// corner's phase nearest the one before it plus the target; a face the walk comes back to off by
/// whole turns holds a fork, and the mismatch is spread over its three edges.
fn corner_phases(
    s: &Surface,
    families: &[(&Field<Point>, Field<Point>, &[f64])],
) -> (Vec<Vec<[f64; 3]>>, Vec<Vec<f64>>) {
    let facing = boundary(&s.mesh, 2);
    let n = families.len();
    let tau = core::f64::consts::TAU;
    let wrap = |x: f64| -((-x + core::f64::consts::PI).rem_euclid(tau) - core::f64::consts::PI);
    let mut spreads = vec![Vec::new(); n];
    let mut turns = vec![Vec::new(); n];
    for f in 0..s.mesh.len(2) {
        let v: Vec<usize> = s.mesh.simplex(2, f).to_vec();
        let normal = |c: usize| s.normals.site(v[c]);
        let phase = |c: usize, g: usize| {
            let (line, stripes, _) = &families[g];
            let between = (stripes.site(v[c]) * line.site(v[c]).reverse()).normalized();
            between.unit().log().scalar_product(plane(normal(c))).s()
        };
        // For each corner and family, the family there nearest the first corner's, and its sense.
        let pick = |c: usize, fam: usize| {
            let first = families[fam].0.site(v[0]);
            let alignment = |g: usize| dot(first, families[g].0.site(v[c]));
            let g = (0..n)
                .max_by(|&a, &b| alignment(a).abs().total_cmp(&alignment(b).abs()))
                .expect("a family");
            (g, alignment(g).signum())
        };
        for fam in 0..n {
            let local = |c: usize| {
                let (g, sense) = pick(c, fam);
                phase(c, g) * sense
            };
            let mut walked = [local(0), 0.0, 0.0, 0.0];
            for side in 0..3 {
                // From corner `side` to the next: the edge facing the corner after them.
                let k = (side + 2) % 3;
                let (edge, forward) = (facing.sites[f][k], facing.signs[f][k] > 0);
                let head = if forward { (side + 1) % 3 } else { side };
                let (g, sense) = pick(head, fam);
                let target = families[g].2[edge] * sense * if forward { 1.0 } else { -1.0 };
                let expected = walked[side] + target;
                walked[side + 1] = expected + wrap(local((side + 1) % 3) - expected);
            }
            let whole = ((walked[3] - walked[0]) / tau).round();
            spreads[fam].push(core::array::from_fn(|c| {
                walked[c] - tau * whole * c as f64 / 3.0
            }));
            turns[fam].push(whole);
        }
    }
    (spreads, turns)
}

/// An angle about a triangle's middle from a point's barycentric weights, rising by a whole turn
/// once around (numga's shader): piecewise linear on the three sectors nearest each side.
fn winding(w: [f64; 3]) -> f64 {
    let least = w[0].min(w[1]).min(w[2]);
    let (start, along) = if w[0] == least {
        (3.0, w[2] - w[1])
    } else if w[1] == least {
        (5.0, w[0] - w[2])
    } else {
        (1.0, w[1] - w[0])
    };
    core::f64::consts::FRAC_PI_3 * (start + along / (1.0 - 3.0 * least).max(1e-6))
}

/// The stripes' light at a hit: light where every family's stripe is light.
fn stripes_light(hit: &Hit, spreads: &[Vec<[f64; 3]>], turns: &[Vec<f64>]) -> Light {
    let share: f64 = spreads
        .iter()
        .zip(turns)
        .map(|(spread, turn)| {
            let corners = spread[hit.face];
            let phase = (0..3).map(|k| hit.weights[k] * corners[k]).sum::<f64>()
                + winding(hit.weights) * turn[hit.face];
            0.5 - 0.5 * (SHARPNESS * phasor(phase).e20()).tanh()
        })
        .product();
    palette::BLUE
        .faded(0.12)
        .blend(palette::SKY.faded(0.6), share as f32)
}

/// The fourth figure: the stripes along the first family of each pattern, slid by `phase` of a
/// stripe (their stripe fields turned about every normal).
fn draw_stripes(c: &mut Canvas, s: &Shown, phase: f32) {
    let panels = row_of_five(c);
    let angle = core::f64::consts::TAU * f64::from(phase);
    let scenes = s
        .sphere_stripes
        .iter()
        .map(|p| (&s.stripe_sphere, p))
        .chain(s.torus_stripes.iter().map(|p| (&s.stripe_torus, p)));
    let labels = [
        "TWO POLES",
        "ONE OF INDEX TWO",
        "THREE AND ONE BACK",
        "THE TORUS",
        "TWO OPPOSITE",
    ];
    for ((rect, (d, pattern)), label) in panels.into_iter().zip(scenes).zip(labels) {
        let family = &pattern[0];
        let slid = turned(&d.surface, &family.stripes, angle);
        let (spreads, turns) = corner_phases(&d.surface, &[(&family.line, slid, &family.advance)]);
        field_panel(c, rect, d, VIEW, None, &[], |hit| {
            stripes_light(hit, &spreads, &turns)
        });
        label_under(c, rect, label, 14.0, 0.9);
    }
}

/// The fifth figure: the two families of stripes across each cross field.
fn draw_crosses(c: &mut Canvas, s: &Shown) {
    let screen = c.rect();
    let below = Rect::new(
        screen.width() * 0.1,
        screen.height() * 0.14,
        screen.width() * 0.9,
        screen.height() * 0.95,
    );
    let labels = [
        "A QUARTER TURN AT EACH CORNER OF A CUBE",
        "A HALF TURN AT EACH CORNER OF A TETRAHEDRON",
    ];
    for (i, ((_, cross), label)) in s.crosses.iter().zip(labels).enumerate() {
        let rect = below.column(i, 2).inset(6.0, 4.0, 6.0, 4.0);
        let families: Vec<(&Field<Point>, Field<Point>, &[f64])> = (0..2)
            .map(|a| {
                (
                    &cross.lines[a],
                    cross.stripes[a].clone(),
                    cross.advance[a].as_slice(),
                )
            })
            .collect();
        let (spreads, turns) = corner_phases(&s.stripe_sphere.surface, &families);
        field_panel(c, rect, &s.stripe_sphere, VIEW, None, &[], |hit| {
            stripes_light(hit, &spreads, &turns)
        });
        label_under(c, rect, label, 18.0, 0.9);
    }
}

fn draw(c: &mut Canvas, s: &Shown, t: f32) {
    backdrop(c);
    let figure = ((t / FIGURE).floor() as usize).min(FIGURES - 1);
    let phase = (t - figure as f32 * FIGURE) / FIGURE;
    let (title, sub) = match figure {
        0 => {
            draw_holonomy(c, s, phase);
            (
                "CONNECTIONS: HOLONOMY AROUND LATITUDE CIRCLES",
                "CARRIED BY THE SQUARE ROOTS OF NEIGHBOURING NORMALS' PRODUCTS, A VECTOR TURNS BY THE CAP'S SOLID ANGLE (VGA3D)",
            )
        }
        1 => {
            draw_heat(c, s, phase);
            (
                "CONNECTIONS: VECTOR HEAT AND THE SMOOTHEST FIELDS",
                "THE CONNECTION LAPLACIAN, COVARIANT.ADJOINT() W COVARIANT, NEVER MIXES TANGENT AND NORMAL (VGA3D)",
            )
        }
        2 => {
            draw_fields(c, s, phase);
            (
                "CONNECTIONS: FIELDS DESIGNED BY TRIVIAL CONNECTIONS",
                "EXTRA TURNS MAKE EVERY FACE TURN BY ITS INDEX: THE NULL SPACE IS A UNIT FIELD, PARALLEL AT EVERY TURN (VGA3D)",
            )
        }
        3 => {
            draw_stripes(c, s, phase);
            (
                "CONNECTIONS: STRIPES ALONG THE FIELDS",
                "THE SMOOTHEST FIELD OF THE CONNECTION TURNED BY THE FREQUENCY ALONG EACH EDGE; TURNED, THE STRIPES SLIDE (VGA3D)",
            )
        }
        _ => {
            draw_crosses(c, s);
            (
                "CONNECTIONS: STRIPES ACROSS CROSS FIELDS",
                "TWO FAMILIES ON ONE DOUBLED FIELD, COUPLED WHERE THE CROSS'S LINES TRADE PLACES (VGA3D)",
            )
        }
    };
    caption(c, title, sub);
}

fn main() {
    run_with(
        Anim::new("connections", SECONDS).size(960, 540),
        shown,
        draw,
    );
}
