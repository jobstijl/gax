//! numga's `electromagnetism/eddy_brake`: currents, braking torque and heat in a conducting
//! disc turning under a local magnetic field, on a triangle mesh in VGA3D.
//!
//! The disc is the mesh of numga's `examples/mesh.py` (`gax_numga_examples::mesh`), rings of
//! nearly equilateral cells on the fibres' edges. Its material is a field of conductance maps
//! `Vector <- Vector`, one per face, a sum of dyads along and across fibres. A magnet fixed in
//! the world pushes the moving charges; the charge balance on the vertices gives the potential,
//! the face currents, the torque on the dual-edge fluxes and the Joule heat (`brake`): the disc
//! brakes, and the heat stays on the material faces as they turn. With the current's own
//! magnetic field (`inductive`) the currents are closed patterns with a memory, a generalized
//! eigenproblem of resistance against inductance, driven by the magnet through one revolution's
//! Fourier series, and a motor pulse spins the disc up before it brakes.
//!
//! numga batches the three materials and the time steps; here each material is its own run,
//! and the three run in parallel at start. gax's sparse field maps compose the face gradient,
//! the material and its transpose (the metric is Euclidean, so numga's adjoint is the
//! coefficients' transpose), factor the charge balance once per disc (it does not change as
//! the disc turns), and solve the eigenproblem. The magnetic coupling's cells are scalars,
//! which keeps the whole face-to-face map small, where numga sums it in blocks of 64 faces.
//!
//! The animation shows the isotropic, radial-fibre and circumferential-fibre discs: above,
//! braking resistively from 2 rad/s over 12 s; below, spun to 600 rad/s by a motor pulse under
//! a stronger magnet, with inductance, over 0.1 s. Each face glows with the heat it has
//! gathered, the arrows are the sheet current as the lab sees it, the guides the fibres, the
//! pink knob a mark on the rim, and the dashed circle the magnet's width.

mod brake;
mod inductive;
#[cfg(test)]
mod inductive_tests;
mod sample;
mod scenes;
#[cfg(test)]
mod tests;

use brake::Motion;
use gax::field::Field;
use gax_colour::Srgb;
use gax_numga_examples::mesh::{Bivector3, Mesh, Scalar3, Vector3, direction};
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, Point2, Rect, backdrop, caption, colormap,
    from_above, palette, reach3, run,
};
use inductive::{InductiveMotion, Rotation};
use sample::{Located, locate};
use scenes::*;
use std::sync::OnceLock;

/// One loop of both runs.
const SECONDS: f32 = 12.0;
/// The power on the heat's and the currents' magnitudes before colouring, which keeps weak
/// values visible while they still fade to nothing.
const HEAT_GAMMA: f32 = 0.5;
const CURRENT_GAMMA: f32 = 0.35;
/// Where the current arrows sit in the material: rings at these shares of the radius, each
/// with this many arrows per unit of its share.
const ARROW_RINGS: [f64; 5] = [0.14, 0.32, 0.5, 0.68, 0.86];
const ARROWS_PER_SHARE: f64 = 26.0;
/// The longest arrow, a share of the radius.
const ARROW_LENGTH: f64 = 0.16;
/// The fibre guides: rays and rings between the band's radii.
const GUIDES: usize = 12;
/// The points along a drawn circle.
const CIRCLE_SAMPLES: usize = 64;
/// The rim knob's distance from the centre, a share of the radius.
const KNOB: f64 = 0.94;

/// What one turning disc looks like at one moment, in its material frame.
struct Snapshot {
    orientation: brake::Turn,
    speed: f64,
    heat: Field<Scalar3>,
    current: Field<Vector3>,
}

impl Snapshot {
    /// The disc turned by `orientation` and spinning at `spin` in the discs' plane.
    fn new(
        orientation: brake::Turn,
        spin: Bivector3,
        heat: &Field<Scalar3>,
        current: &Field<Vector3>,
    ) -> Snapshot {
        Snapshot {
            orientation,
            speed: -(brake::XY | spin).s(),
            heat: heat.clone(),
            current: current.clone(),
        }
    }
}

impl From<&Motion> for Snapshot {
    fn from(m: &Motion) -> Self {
        Snapshot::new(m.orientation, m.spin, &m.heat, &m.current)
    }
}

impl From<&InductiveMotion> for Snapshot {
    fn from(m: &InductiveMotion) -> Self {
        let Rotation { orientation, spin } = m.motion;
        Snapshot::new(orientation, spin, &m.heat, &m.current)
    }
}

/// A run of the three materials, with the scales its colours share.
struct Run {
    /// The title, in two lines.
    title: [&'static str; 2],
    /// The simulated time per state.
    step: f64,
    /// `[material][state]`.
    states: [Vec<Snapshot>; 3],
    heat_peak: f64,
    current_peak: f64,
}

impl Run {
    fn new(
        title: [&'static str; 2],
        step: f64,
        states: [Vec<Snapshot>; 3],
        areas: &Field<Scalar3>,
    ) -> Run {
        let all = || states.iter().flatten();
        let densities = |s: &Snapshot| heat_densities(&s.heat, areas);
        let heat_peak = all().flat_map(densities).fold(0.0, f64::max);
        let current_peak = all()
            .flat_map(|s| s.current.iter())
            .map(|j| j.norm())
            .fold(0.0, f64::max);
        Run {
            title,
            step,
            states,
            heat_peak,
            current_peak,
        }
    }
}

/// The heat per area of each face.
fn heat_densities(heat: &Field<Scalar3>, areas: &Field<Scalar3>) -> Vec<f64> {
    heat.iter()
        .zip(areas.iter())
        .map(|(h, a)| h.s() / a.s())
        .collect()
}

/// The mesh, its faces' areas, its sampling sites, and both runs, computed once: the six runs
/// in parallel. Every frame shows a state of all six, coloured to the scales of their whole
/// runs, so the first frame needs them all.
struct Scene {
    mesh: Mesh,
    areas: Field<Scalar3>,
    arrows: Vec<Located>,
    resistive: Run,
    inductive: Run,
}

fn scene() -> &'static Scene {
    static SCENE: OnceLock<Scene> = OnceLock::new();
    SCENE.get_or_init(|| {
        let (mesh, conductivity) = materials();
        let (braked, spun) = std::thread::scope(|s| {
            let disc = &mesh;
            let braked = conductivity
                .each_ref()
                .map(|c| s.spawn(move || materials_braking(disc, c)));
            let spun = conductivity
                .each_ref()
                .map(|c| s.spawn(move || inductive_material(disc, c)));
            (
                braked.map(|run| run.join().expect("a resistive run")),
                spun.map(|run| run.join().expect("an inductive run")),
            )
        });
        let areas = mesh.triangle_areas();
        let resistive = Run::new(
            ["RESISTIVE BRAKING", "FROM 2 RAD/S"],
            STEP,
            braked.map(|r| r.iter().map(Snapshot::from).collect()),
            &areas,
        );
        let inductive = Run::new(
            ["SPUN TO 600 RAD/S", "WITH INDUCTANCE"],
            INDUCTIVE_STEP,
            spun.map(|r| r.iter().map(Snapshot::from).collect()),
            &areas,
        );
        let arrows = arrow_sites(&mesh);
        Scene {
            mesh,
            areas,
            arrows,
            resistive,
            inductive,
        }
    })
}

/// The material points where current arrows sit: rings of evenly spaced points.
fn arrow_sites(mesh: &Mesh) -> Vec<Located> {
    let points = ARROW_RINGS.iter().flat_map(|&share| {
        let count = (ARROWS_PER_SHARE * share).round() as usize;
        let turn = move |k: usize| core::f64::consts::TAU * (k as f64 + 0.5) / count as f64;
        (0..count).map(move |k| direction(turn(k)) * (share * RADIUS))
    });
    points.filter_map(|p| locate(mesh, p)).collect()
}

/// A point of the plane, as drawn.
fn planar(v: Vector3) -> Point2 {
    from_above(reach3(v))
}

/// Heat as light: from the disc's dark slate through ember red and orange to pale yellow.
fn heat_light(t: f32) -> Light {
    colormap::stops(
        t,
        &[
            Srgb::rgb(0.07, 0.09, 0.2),
            Srgb::rgb(0.5, 0.08, 0.14),
            Srgb::rgb(0.95, 0.42, 0.08),
            Srgb::rgb(1.0, 0.92, 0.6),
        ],
    )
}

/// The points of a circle about `centre` in the discs' plane, the first repeated last.
fn circle(centre: Vector3, radius: f64) -> Vec<Vector3> {
    let turn = |k: usize| core::f64::consts::TAU * k as f64 / CIRCLE_SAMPLES as f64;
    (0..=CIRCLE_SAMPLES)
        .map(|k| centre + direction(turn(k)) * radius)
        .collect()
}

/// The fibre guides of a material (its index in `MATERIALS`), in its frame: rays for radial
/// fibres, rings for circumferential ones, none for the isotropic disc.
fn guides(material: usize) -> Vec<Vec<Vector3>> {
    let (inner, outer) = (FIBRE_INNER_RADIUS, FIBRE_OUTER_RADIUS);
    match material {
        1 => (0..2 * GUIDES)
            .map(|k| {
                let d = direction(core::f64::consts::TAU * k as f64 / (2 * GUIDES) as f64);
                vec![d * inner, d * outer]
            })
            .collect(),
        2 => (0..GUIDES)
            .map(|k| {
                let share = k as f64 / (GUIDES - 1) as f64;
                circle(Vector3::zero(), inner + share * (outer - inner))
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// One disc at one moment: its faces glowing with their heat, the fibres, the current arrows,
/// the rim, the knob and the magnet.
fn disc(c: &mut Canvas, ax: &Axes, scene: &Scene, run: &Run, material: usize, state: &Snapshot) {
    let world = |v: Vector3| planar(state.orientation >> v);
    heated_faces(c, ax, scene, run, state, &world);
    for guide in guides(material) {
        let line: Vec<Point2> = guide.into_iter().map(world).collect();
        ax.polyline(c, &line, 0.8, palette::blue().faded(0.15));
    }
    current_arrows(c, ax, scene, run, state, &world);
    let mesh = &scene.mesh;
    let rim: Vec<[Point2; 2]> = mesh
        .boundary_edges()
        .iter()
        .map(|&e| mesh.edges[e].map(|v| world(mesh.vertices.site(v))))
        .collect();
    ax.stroke(c, &rim, 1.0, palette::grid().faded(3.0));
    let knob = world(Vector3::new(KNOB * RADIUS, 0.0, 0.0));
    ax.scatter(c, &[knob], Marker::Dot, 9.0, palette::purple());
    let footprint = circle(magnet(FIELD_STRENGTH).centre, FIELD_WIDTH);
    let footprint: Vec<Point2> = footprint.into_iter().map(planar).collect();
    ax.dashed(c, &footprint, 1.2, 4.0, palette::yellow().faded(0.8));
}

/// The disc's faces, each glowing with the heat it has gathered per area.
fn heated_faces(
    c: &mut Canvas,
    ax: &Axes,
    scene: &Scene,
    run: &Run,
    state: &Snapshot,
    world: &impl Fn(Vector3) -> Point2,
) {
    let mesh = &scene.mesh;
    let densities = heat_densities(&state.heat, &scene.areas);
    for (corners, density) in mesh.faces.iter().zip(densities) {
        let t = (density / run.heat_peak.max(f64::MIN_POSITIVE)) as f32;
        let triangle = corners.map(|v| world(mesh.vertices.site(v)));
        ax.fill(c, &triangle, heat_light(t.powf(HEAT_GAMMA)), 1.0);
    }
}

/// The sheet current at the arrows' sites: each arrow centred on its site, longer and brighter
/// the stronger the current.
fn current_arrows(
    c: &mut Canvas,
    ax: &Axes,
    scene: &Scene,
    run: &Run,
    state: &Snapshot,
    world: &impl Fn(Vector3) -> Point2,
) {
    let mesh = &scene.mesh;
    for site in &scene.arrows {
        let current = state.current.site(site.face);
        let strength = (current.norm() / run.current_peak.max(f64::MIN_POSITIVE)) as f32;
        // Too faint to see.
        if strength < 1e-3 {
            continue;
        }
        let corners = mesh.faces[site.face].map(|v| mesh.vertices.site(v));
        let centre = (0..3)
            .map(|k| corners[k] * site.weights[k])
            .sum::<Vector3>();
        let shown = strength.powf(CURRENT_GAMMA);
        let length = ARROW_LENGTH * RADIUS * f64::from(shown);
        let half = current.normalized().into_inner() * (length / 2.0);
        let light = palette::sky().faded(shown);
        ax.arrow(
            c,
            world(centre - half),
            world(centre + half),
            1.4,
            5.0,
            light,
        );
    }
}

/// The share of the canvas's width the rows' titles take, at the left.
const TITLES: f32 = 0.2;

/// A positive number to three figures.
fn three_figures(x: f64) -> String {
    match x {
        x if x >= 100.0 => format!("{x:.0}"),
        x if x >= 10.0 => format!("{x:.1}"),
        x if x >= 0.1 => format!("{x:.2}"),
        x => format!("{x:.2E}"),
    }
}

/// A row of discs at the share of the run: its title and time at the left, each disc's speed
/// and heat under it.
fn row(c: &mut Canvas, rect: Rect, scene: &Scene, run: &Run, share: f64) {
    let last = run.states[0].len() - 1;
    let index = ((share * last as f64).round() as usize).min(last);
    let size = (rect.height() / 22.0).clamp(6.0, 10.0);
    let time = format!("T = {:.3} S", run.step * index as f64);
    let lines = [run.title[0], run.title[1], &time];
    for (k, line) in lines.iter().enumerate() {
        let at = rect.left_middle() + Point2::direction(size, size * (1.6 * k as f32 - 1.6));
        c.text(line, at, size, palette::ink(), Align::Left);
    }
    let discs = rect.part(TITLES, 0.0, 1.0, 1.0);
    for (material, states) in run.states.iter().enumerate() {
        let panel = discs.column(material, 3);
        let state = &states[index];
        let ax = Axes::equal(
            panel.inset(0.0, 0.0, 0.0, size * 1.6),
            Point2::xy(0.0, 0.0),
            (1.06 * RADIUS) as f32,
        );
        disc(c, &ax, scene, run, material, state);
        let (speed, heat) = (state.speed, state.heat.sum().s());
        let label = format!("{} RAD/S, {} J", three_figures(speed), three_figures(heat));
        let at = panel.bottom_middle() - Point2::direction(0.0, size * 0.4);
        c.text(&label, at, size * 0.9, palette::ink(), Align::Center);
    }
}

/// The materials' names over their columns.
fn headers(c: &mut Canvas, rect: Rect) {
    let discs = rect.part(TITLES, 0.0, 1.0, 1.0);
    for (material, (name, _)) in MATERIALS.iter().enumerate() {
        let at = discs.column(material, 3).bottom_middle();
        c.text(name, at, 10.0, palette::ink(), Align::Center);
    }
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let share = f64::from(t / SECONDS).rem_euclid(1.0);
    let scene = scene();
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let top = (h / 30.0).clamp(10.0, 22.0) * 3.6;
    headers(c, Rect::new(0.0, top - 4.0, w, top + 12.0));
    let middle = top + 12.0 + (h - top - 12.0) / 2.0;
    row(
        c,
        Rect::new(0.0, top + 14.0, w, middle),
        scene,
        &scene.resistive,
        share,
    );
    row(
        c,
        Rect::new(0.0, middle, w, h),
        scene,
        &scene.inductive,
        share,
    );
    caption(
        c,
        "EDDY BRAKE: CURRENTS AND HEAT IN TURNING CONDUCTING DISCS",
        "CHARGE BALANCE OF WHITNEY-RECONSTRUCTED FACE CURRENTS; INDUCTIVE PATTERNS FROM RESISTANCE AGAINST INDUCTANCE (VGA3D)",
    );
}

fn main() {
    run(Anim::new("eddy_brake", SECONDS).size(960, 540), draw);
}
