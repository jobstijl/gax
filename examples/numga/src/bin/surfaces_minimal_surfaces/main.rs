//! numga's `surfaces/minimal_surfaces`: minimal surfaces relaxed from triangle meshes, their
//! harmonic conjugates, and Bonnet's family between the two (VGA3D).
//!
//! A soap film settles where no small displacement of its interior can shrink its area, and
//! there its mean curvature vanishes. Turning every small piece of a minimal surface a quarter
//! turn in its own tangent plane leaves the pieces fitting together into a second minimal
//! surface, its harmonic conjugate; turning them by any other angle gives Bonnet's family between
//! the two, all isometric, all with the same normals.
//!
//! Points are antivectors. A triangle's patch is the join of two of its sides, the direction
//! square to it as long as twice its area, and each corner's hat gradient is the facing side
//! contracted with the patch's inverse. The quarter turn in the triangle's plane is the
//! commutator with the plane: it is both the area gradient, whose sums at the vertices are the
//! cotangent Laplacian's pull (the corner gradients paired over the areas), and the turned
//! differential that integrates to the conjugate. Read through the corner gradients, that
//! differential is half the bare facing side, and those sums cancel at every interior vertex,
//! where neighbouring triangles run their shared edge in opposite senses: the conjugate is
//! solved from the boundary alone (Pinkall and Polthier, 1993), a least-squares solve on the
//! Laplacian. The relaxation moves part of the way to the harmonic surface of its own
//! triangles at each step, the wire held, by conjugate gradients on the interior.
//!
//! The animation shows numga's three: a dome sinking onto the soap film of a four-arch wire,
//! its mean curvature tinting it until it fades; the film and Enneper's surface beside their
//! recovered conjugates, with matched domain curves (rings and spokes of the disk lifted onto
//! each surface); and a slit catenoid turned through Bonnet's family into the helicoid, its
//! rings and meridians drawn and its normals pinned where they cross, the pins keeping their
//! directions.

mod minimal;
#[cfg(test)]
mod tests;

use gax_numga_examples::scene3::{Scene3, panel3};
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Light, Marker, ORIGIN3, Point2, Point3, Rect, backdrop, caption,
    palette, reach3, run_with,
};
use minimal::*;

/// The loop: the relaxation (numga's 41 frames of 40 ms, out and back) and Bonnet's family (36
/// frames), both stretched to it.
const SECONDS: f32 = 6.56;
/// Samples along each domain curve, the rings' radii as shares of the disk's, and the spokes.
const CURVE_SAMPLES: usize = 80;
const RING_SHARES: [f64; 4] = [0.22, 0.4333333333333333, 0.6466666666666667, 0.86];
const SPOKES: usize = 6;
/// The views as numga's (elevation, azimuth from `x`), in degrees.
const PLATEAU_VIEW: (f32, f32) = (28.0, -38.0);
const CONJUGATE_VIEW: (f32, f32) = (30.0, -36.0);
const BONNET_VIEW: (f32, f32) = (24.0, -36.0);
/// How far a normal pin reaches behind and in front of its vertex.
const PIN_DEPTH: f64 = 0.16;
const PIN_REACH: f64 = 0.30;

/// Where a sample of a domain curve lies: the three corners of its triangle and their hats.
type Lift = [(usize, f64); 3];

/// The rings and spokes of the flat disk: each ring a point on the x axis turned through a full
/// turn, each spoke a rim vertex drawn in towards the centre, the vertex closest in angle to the
/// spoke.
fn domain_curves(mesh: &gax_numga_examples::mesh::Mesh<Point>, flat: &[Point]) -> Vec<Vec<Point>> {
    let (_, rim) = interior_and_wire(mesh);
    let radius = rim.iter().map(|v| flat[*v].norm()).sum::<f64>() / rim.len() as f64;
    let xy = gax::vga3d::Bivector::new(0.0, 0.0, 1.0);
    let turned = |angle: f64| (xy * (-angle / 2.0)).exp() >> x_point();
    let samples = gax_numga_examples::signal::linspace(0.0, 1.0, CURVE_SAMPLES);
    let rings = RING_SHARES.iter().map(|share| {
        let around = samples.iter().map(|s| turned(core::f64::consts::TAU * s));
        around.map(|p| p * (radius * share)).collect()
    });
    let spokes = (0..SPOKES).map(|k| {
        let direction =
            turned(-core::f64::consts::PI + core::f64::consts::TAU * k as f64 / SPOKES as f64);
        let alignment = |v: &usize| -(direction | flat[*v].normalized()).s();
        let end = rim
            .iter()
            .max_by(|a, b| alignment(a).total_cmp(&alignment(b)));
        let end = flat[*end.expect("a rim")];
        samples.iter().map(|s| end * *s).collect()
    });
    rings.chain(spokes).collect()
}

/// Each sample's triangle in the flat disk and its hats there: the triangle whose hats' sizes
/// sum least, one in a triangle that holds the sample.
fn lifting(faces: &[[usize; 3]], flat: &[Point], curves: &[Vec<Point>]) -> Vec<Vec<Lift>> {
    let flat_triangles = triangles(faces, flat);
    let locate = |p: Point| -> Lift {
        let hats = |f: usize| flat_triangles[f].hats(faces[f].map(|v| flat[v]), p);
        let size = |f: usize| hats(f).iter().map(|h| h.abs()).sum::<f64>();
        let holding = (0..faces.len())
            .min_by(|a, b| size(*a).total_cmp(&size(*b)))
            .expect("a face");
        let h = hats(holding);
        [0, 1, 2].map(|k| (faces[holding][k], h[k]))
    };
    curves
        .iter()
        .map(|c| c.iter().map(|p| locate(*p)).collect())
        .collect()
}

/// A lifted sample on a surface over the disk.
fn lifted(lift: &Lift, surface: &[Point]) -> Point {
    lift.iter().map(|(v, h)| surface[*v] * *h).sum()
}

/// The relaxation, the conjugate pairs and Bonnet's family, computed once.
struct Scenes {
    disk: Sheet,
    states: Vec<State>,
    /// The tint's scale: most of the dome's curvature reaches full colour.
    curvature_scale: f64,
    film_conjugate: Vec<Point>,
    enneper: Sheet,
    enneper_surface: Vec<Point>,
    enneper_conjugate: Vec<Point>,
    catenoid: Catenoid,
    family: Vec<(Vec<Point>, Vec<Patch>)>,
}

/// A disk's faces, rim edges and lifted domain curves.
struct Sheet {
    faces: Vec<[usize; 3]>,
    rim: Vec<[usize; 2]>,
    lift: Vec<Vec<Lift>>,
}

fn sheet_of(mesh: &gax_numga_examples::mesh::Mesh<Point>, flat: &[Point]) -> Sheet {
    let curves = domain_curves(mesh, flat);
    Sheet {
        faces: mesh.faces.clone(),
        rim: mesh
            .boundary_edges()
            .iter()
            .map(|e| mesh.edges[*e])
            .collect(),
        lift: lifting(&mesh.faces, flat, &curves),
    }
}

fn scenes() -> Scenes {
    let (mesh, flat, initial) =
        four_arch_dome(DISK_RADIUS, DISK_DIVISIONS, ARCH_HEIGHT, DOME_HEIGHT);
    let states = relax_plateau(&mesh, &initial, RELAXATION_STEP, RELAXATION_STEPS);
    let mut sizes: Vec<f64> = states[0].curvature.iter().map(|c| c.abs()).collect();
    sizes.sort_by(f64::total_cmp);
    let curvature_scale = sizes[(sizes.len() - 1) * 95 / 100].max(1e-6);
    let film = &states[states.len() - 1].vertices;
    let (film_conjugate, _) = conjugate_minimal_surface(&mesh.faces, film);
    let (enneper_mesh, enneper_flat, enneper_surface, _) =
        enneper_disk(ENNEPER_RADIUS, ENNEPER_DIVISIONS);
    let (enneper_conjugate, _) = conjugate_minimal_surface(&enneper_mesh.faces, &enneper_surface);
    let catenoid = slit_catenoid(
        CATENOID_HEIGHT,
        CATENOID_SPAN,
        CATENOID_HEIGHTS,
        CATENOID_AZIMUTHS,
    );
    let (helicoid, _) = conjugate_minimal_surface(&catenoid.mesh.faces, &catenoid.catenoid);
    let family = (0..BONNET_FRAMES)
        .map(|k| {
            let angle = core::f64::consts::FRAC_PI_2 * k as f64 / (BONNET_FRAMES - 1) as f64;
            let (surface, normals) =
                bonnet(&catenoid.mesh.faces, &catenoid.catenoid, &helicoid, angle);
            (centred(&surface), normals)
        })
        .collect();
    Scenes {
        disk: sheet_of(&mesh, &flat),
        states,
        curvature_scale,
        film_conjugate,
        enneper: sheet_of(&enneper_mesh, &enneper_flat),
        enneper_surface,
        enneper_conjugate,
        catenoid,
        family,
    }
}

/// A point drawn: where its dual direction reaches.
fn place(p: Point) -> Point3 {
    reach3(p.hodge())
}

/// The surface's light, and the tint of mean curvature.
fn surface_light() -> Light {
    palette::SKY.faded(0.3)
}
fn tint() -> Light {
    palette::ORANGE.faded(0.6)
}

/// A parallel camera at (elevation, azimuth) in degrees seeing points of reach `reach` whole.
fn camera(rect: Rect, (elevation, azimuth): (f32, f32), reach: f32) -> Camera {
    let scale = 0.5 * rect.height().min(rect.width()) / reach;
    Camera::parallel(rect, scale, azimuth.to_radians(), elevation.to_radians())
}

/// The largest distance of points from the origin.
fn reach_of<'a>(points: impl IntoIterator<Item = &'a Point>) -> f32 {
    points
        .into_iter()
        .map(|p| p.norm() as f32)
        .fold(0.0, f32::max)
}

/// A small step towards the viewer, so that curves on a sheet draw over it.
fn lift_towards(cam: &Camera) -> Point3 {
    let d = cam.eye().unitized() - ORIGIN3;
    d.gp(0.02 / d.ideal_norm())
}

/// A lit sheet, each face the mean of its corners' lights, and its rim.
fn draw_sheet(
    s: &mut Scene3,
    faces: &[[usize; 3]],
    rim: &[[usize; 2]],
    surface: &[Point],
    light: impl Fn(usize) -> Light,
) {
    for f in faces {
        let [a, b, c] = f.map(|v| place(surface[v]));
        let mean = light(f[0])
            .mix_light(light(f[1]), 0.5)
            .mix_light(light(f[2]), 1.0 / 3.0);
        let lit = s.lit(a, b, c, mean);
        s.tri(a, b, c, lit, 0.96);
    }
    let up = lift_towards(&s.cam);
    for [a, b] in rim {
        s.seg(
            place(surface[*a]) + up,
            place(surface[*b]) + up,
            1.6,
            palette::INK.faded(0.8),
        );
    }
}

/// The domain curves lifted onto a sheet: the rings green, the spokes in the series' lights.
fn draw_curves(s: &mut Scene3, lift: &[Vec<Lift>], surface: &[Point]) {
    let up = lift_towards(&s.cam);
    for (k, curve) in lift.iter().enumerate() {
        let light = if k < RING_SHARES.len() {
            palette::GREEN
        } else {
            palette::series(k - RING_SHARES.len())
        };
        let points: Vec<Point3> = curve
            .iter()
            .map(|l| place(lifted(l, surface)) + up)
            .collect();
        s.polyline(&points, 1.3, light.faded(0.8));
    }
}

/// The step `fraction` of the way through `count` frames, out and back.
fn out_and_back(t: f32, count: usize) -> usize {
    let phase = (t / SECONDS).rem_euclid(1.0);
    let there = 1.0 - (2.0 * phase - 1.0).abs();
    (there * (count - 1) as f32).round() as usize
}

/// The dome relaxing: the surface moved by the film's mean point, tinted by its mean curvature.
fn draw_plateau(c: &mut Canvas, rect: Rect, s: &Scenes, frame: usize) {
    let centre = mean(&s.states[s.states.len() - 1].vertices);
    let moved =
        |state: &State| -> Vec<Point> { state.vertices.iter().map(|p| *p - centre).collect() };
    let reach = s
        .states
        .iter()
        .map(|st| reach_of(&moved(st)))
        .fold(0.0, f32::max);
    let cam = camera(rect, PLATEAU_VIEW, 1.05 * reach);
    let state = &s.states[frame];
    let surface = moved(state);
    let shade = |v: usize| {
        let level = (state.curvature[v].abs() / s.curvature_scale).min(1.0) as f32;
        surface_light().mix_light(tint(), level)
    };
    panel3(c, rect, cam, |scene| {
        draw_sheet(scene, &s.disk.faces, &s.disk.rim, &surface, shade);
        draw_curves(scene, &s.disk.lift, &surface);
    });
    let area = format!("AREA {:.4}", state.area);
    c.text(
        &area,
        rect.bottom_middle() - Point2::direction(0.0, 6.0),
        9.0,
        palette::INK,
        Align::Center,
    );
}

/// A surface beside its conjugate, each moved by its own mean point, framed together.
fn draw_pair(c: &mut Canvas, rect: Rect, sheet: &Sheet, surface: &[Point], conjugate: &[Point]) {
    let surfaces = [centred(surface), centred(conjugate)];
    let reach = surfaces.iter().map(reach_of).fold(0.0, f32::max);
    for (k, surface) in surfaces.iter().enumerate() {
        let panel = rect.column(k, 2).inset(2.0, 0.0, 2.0, 0.0);
        let cam = camera(panel, CONJUGATE_VIEW, 1.05 * reach);
        panel3(c, panel, cam, |scene| {
            draw_sheet(scene, &sheet.faces, &sheet.rim, surface, |_| {
                surface_light()
            });
            draw_curves(scene, &sheet.lift, surface);
        });
    }
}

/// Bonnet's family at a frame: the sheet, its rings and meridians, and the normals pinned where
/// they cross.
fn draw_bonnet(c: &mut Canvas, rect: Rect, s: &Scenes, frame: usize) {
    let reach = s
        .family
        .iter()
        .map(|(p, _)| reach_of(p))
        .fold(0.0, f32::max);
    let cam = camera(rect, BONNET_VIEW, 0.8 * reach);
    let (surface, normals) = &s.family[frame];
    let grid = &s.catenoid.grid;
    let (heights, azimuths) = (grid.len(), grid[0].len());
    let spread = |from: usize, to: usize, count: usize| -> Vec<usize> {
        (0..count)
            .map(|k| from + (to - from) * k / (count - 1))
            .collect()
    };
    let rim: Vec<[usize; 2]> = s
        .catenoid
        .mesh
        .boundary_edges()
        .iter()
        .map(|e| s.catenoid.mesh.edges[*e])
        .collect();
    panel3(c, rect, cam, |scene| {
        draw_sheet(scene, &s.catenoid.mesh.faces, &rim, surface, |_| {
            surface_light()
        });
        let up = lift_towards(&scene.cam);
        for i in spread(2, heights - 3, 5) {
            let ring: Vec<Point3> = grid[i].iter().map(|v| place(surface[*v]) + up).collect();
            scene.polyline(&ring, 1.3, palette::GREEN.faded(0.8));
        }
        for j in spread(3, azimuths - 4, 6) {
            let meridian: Vec<Point3> =
                grid.iter().map(|row| place(surface[row[j]]) + up).collect();
            scene.polyline(&meridian, 1.3, palette::ORANGE.faded(0.8));
        }
        for i in spread(3, heights - 4, 3) {
            for j in spread(5, azimuths - 6, 4) {
                let v = grid[i][j];
                // The direction square to the patch, as a step from the vertex: its dual,
                // read in the vertex's orientation (numga multiplies by the unit volume).
                let n: Point = normals[v].hodge().with_orientation();
                let (back, front) = (surface[v] - n * PIN_DEPTH, surface[v] + n * PIN_REACH);
                scene.seg(place(back), place(front), 1.4, palette::INK);
                scene.dot(place(front), Marker::Dot, 6.0, palette::YELLOW);
            }
        }
    });
}

fn draw(c: &mut Canvas, s: &Scenes, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let body = Rect::new(0.0, h * 0.14, w, h);
    let label = |c: &mut Canvas, rect: Rect, text: &str| {
        let at = rect.top_middle() - Point2::direction(0.0, 4.0);
        c.text(text, at, 9.0, palette::INK, Align::Center);
    };
    let plateau = body.part(0.0, 0.0, 0.34, 1.0).inset(6.0, 16.0, 6.0, 12.0);
    draw_plateau(c, plateau, s, out_and_back(t, s.states.len()));
    label(c, plateau, "A DOME RELAXES TO A SOAP FILM");
    let bonnet = body.part(0.34, 0.0, 0.66, 1.0).inset(6.0, 16.0, 6.0, 12.0);
    draw_bonnet(c, bonnet, s, out_and_back(t, s.family.len()));
    label(c, bonnet, "BONNET: CATENOID TO HELICOID");
    let pairs = body.part(0.66, 0.0, 1.0, 1.0).inset(6.0, 16.0, 6.0, 12.0);
    let (top, bottom) = (
        pairs.part(0.0, 0.0, 1.0, 0.5),
        pairs.part(0.0, 0.5, 1.0, 1.0).inset(0.0, 14.0, 0.0, 0.0),
    );
    let film = &s.states[s.states.len() - 1].vertices;
    draw_pair(c, top, &s.disk, film, &s.film_conjugate);
    label(c, top, "THE FILM, ITS CONJUGATE");
    draw_pair(
        c,
        bottom,
        &s.enneper,
        &s.enneper_surface,
        &s.enneper_conjugate,
    );
    label(c, bottom, "ENNEPER, ITS CONJUGATE");
    caption(
        c,
        "MINIMAL SURFACES, THEIR CONJUGATES AND BONNET'S FAMILY",
        "THE QUARTER TURN IN A TRIANGLE IS THE COMMUTATOR WITH ITS PLANE: AREA GRADIENT AND CONJUGATE (VGA3D)",
    );
}

fn main() {
    run_with(
        Anim::new("minimal_surfaces", SECONDS).size(960, 540),
        scenes,
        draw,
    );
}
