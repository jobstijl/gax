//! numga's `mechanics/bending`: bending and twisting of thin-walled beams, in Euler-Bernoulli
//! beam theory and Saint-Venant's theory of torsion. A cantilever loaded away from its
//! principal bending directions bends sideways as well as along the load, and one loaded beside
//! its shear centre twists.
//!
//! The beam runs along `z` from its clamp at `z = 0`. A cross-section is thin-walled: walls of a
//! thickness along a midline of straight pieces, each a pair of points of the section's plane
//! (PGA2D; gax's homomorphism into PGA3D takes a point of the plane to the vertical line through
//! it, its fibre, and a line to the vertical plane). Bending keeps each cross-section plane, and
//! its strain is a line of the section, the neutral axis: each fibre, through a point `p`,
//! stretches by their pairing `strain & p`, and pulls along the beam with the modulus times its
//! stretch. Summed over the walls, exactly by each wall's two lumped mass points (the simplex's),
//! the fibres' pulls make the bending stiffness, a map `Point <- Line` that holds the area and the
//! first and second moments of area, with no centroid and no arms.
//!
//! Under a twist the walls warp: each point of the midline moves along the beam by the area
//! that the line from a pole sweeps along the walls to it, the join of the pole with each
//! wall's ends added up from node to node. The shear centre is the pole whose warping no
//! bending strain sees, its warped fibres pulling with no couple across the beam; the warping is
//! linear in the pole, so it is one solve. The walls resist a twist by the shear modulus times a
//! third of their length times the cube of their thickness, and a slide of the section along a
//! wall by the shear modulus times the wall's area, the force of the slide passing through the
//! shear centre.
//!
//! Any strain is a line of space, and the whole stiffness, `Forque <- Line`, takes it to the
//! resultant line of everything the section exerts: a sum of dyads of lines, the fibres' for
//! the stretch and the bending, the couple about the beam for the twist, and the slides' forces.
//! A twist about the line along the beam through the shear centre meets every slide's force and
//! pairs with none, so the twisting axis comes out of the solve. A load is a line: seen from a
//! station, shifted back to the clamp, one solve gives the strain there, and the running
//! product of the twists of the pieces, each the shift along the beam plus the strain, places
//! every cross-section (a small-deflection load with a finite-motion reconstruction, the clamp
//! free to warp). The bending stiffness is a polarity: the strain of a pull through a point is
//! that point's polar line, and the points whose push squeezes every fibre form the kern, cut
//! out here exactly by the nodes' polar lines (numga's kern is the same construction).
//!
//! In the matrix notation of beam theory the curvatures read as
//! `κ = E⁻¹ [[I_yy, −I_xy], [−I_xy, I_xx]]⁻¹ M`, the rate of twist as `θ' = T / (G J)` with
//! `J = Σ b t³ / 3` and `T` the torque about the shear centre, and the slide as `γ = V / (G A_s)`.
//!
//! The animation: the I-section, the turned I-section and the Z-section with their tip loads
//! turning once around through the centroid, which is their shear centre, and the slotted tube
//! with its load moved across, from the slot's side past the shear centre and back, twisting it
//! one way, then not at all, then the other. Each section at the clamp shows its stress over the
//! walls (blue squeezed, red stretched), its walls at the tip as the tip has moved, its principal
//! axes (dashed), neutral axis, kern, centroid (dot), shear centre (ring), load, and numga's two
//! pushes along the beam (purple where the push squeezes every fibre, inside the kern; red
//! outside). Beside them the beams bend and twist in space over the straight ones, with the line
//! of shear centres; below, numga's stubby I-beam from the side, with the walls' shear stiffness
//! and with the slides held rigid.

mod beam;
mod sections;
use gax_numga_examples::shared::mechanics_simplex as simplex;

use beam::{Beam, Body, LENGTH, Load, P3, PIECES, Strain2};
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Marker, ORIGIN2, Point2, Rect, Scene3, backdrop,
    caption, colormap, palette, run_with, signal,
};
use sections::{NAMES, P2};

const SECONDS: f32 = 12.0;
/// Loads sampled over the loop for the stress colours' range.
const LOAD_SAMPLES: usize = 48;
/// The drawn length of the load's arrow in a section, and at the tip.
const ARROW: f64 = 0.45;
const TIP_ARROW: f64 = 1.4;
/// The drawn width of a wall, and the pieces each wall's stress is drawn in.
const WALL_WIDTH: f32 = 4.0;
const WALL_PIECES: usize = 12;
/// The half height of the section views (numga's): it holds the tips' moves, the tube's shear
/// centre and its load's way across.
const SECTION_REACH: f32 = 1.2;
/// The tube's nodes between drawn edges along the straight beam.
const EDGE_EVERY: usize = 8;
/// The beams' spacing in the view of space, the view's azimuth and elevation, and its half
/// height.
const BEAM_SPACING: f64 = 3.2;
const BEAM_AZIMUTH: f32 = -2.0;
const BEAM_ELEVATION: f32 = 0.6;
const BEAM_VIEW: f32 = 6.5;
/// The pieces between the stubby beam's drawn cross-sections, and the half height of its view.
const ELEVATION_EVERY: usize = 5;
const STUBBY_REACH: f32 = 0.8;

/// The sections' bodies and kerns, the stubby I-beam with its walls' shear stiffness and with
/// its slides held rigid, and the largest stress at the clamp under any of the loop's loads,
/// the colour scale's end.
struct Scene {
    bodies: Vec<Body>,
    kerns: Vec<Vec<P2>>,
    stubs: [Beam; 2],
    largest: f64,
}

/// The tip load of section `k` at the loop's phase, as numga's scenes have it: straight down
/// through the centroid on the I-sections; turning once around through the centroid on the
/// Z-section; and on the slotted tube straight down through a point moved across and back.
/// Returns the point it goes through, its direction, and the load.
fn load(body: &Body, k: usize, phase: f64) -> (P2, P3, Load) {
    let down = beam::up_direction() * -beam::LOAD;
    match k {
        2 => {
            let turn = gax::pga3d::Motor::rotation_about(P3::direction(0.0, 0.0, 1.0), phase);
            (body.centroid, turn >> down, beam::turned_load(body, phase))
        }
        3 => {
            let through = beam::across(signal::haversine(phase));
            (through, down, beam::downward(through, beam::LOAD, LENGTH))
        }
        _ => (
            body.centroid,
            down,
            beam::downward(body.centroid, beam::LOAD, LENGTH),
        ),
    }
}

fn scene() -> Scene {
    let bodies: Vec<Body> = sections::sections()
        .into_iter()
        .map(|s| beam::body(s, beam::SHEAR_MODULUS))
        .collect();
    let kerns = bodies.iter().map(|b| beam::kern(b, 1.2)).collect();
    let largest = bodies
        .iter()
        .enumerate()
        .flat_map(|(k, b)| (0..LOAD_SAMPLES).map(move |i| (k, b, i)))
        .map(|(k, b, i)| {
            let phase = core::f64::consts::TAU * i as f64 / LOAD_SAMPLES as f64;
            let clamp = beam::cantilever(b, load(b, k, phase).2, LENGTH).strains[0];
            let ends = b.section.ends().flatten();
            ends.map(|p| beam::stress(clamp, p).abs())
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max);
    let stubby = |shear: f64| {
        let body = beam::body(sections::sections()[0].clone(), shear);
        let tip = beam::downward(body.centroid, beam::STUBBY_LOAD, beam::STUBBY_LENGTH);
        beam::cantilever(&body, tip, beam::STUBBY_LENGTH)
    };
    Scene {
        kerns,
        stubs: [stubby(beam::SHEAR_MODULUS), stubby(beam::RIGID_SHEAR)],
        largest,
        bodies,
    }
}

/// The direction of a line of the section, as the displacement along it: its meet with the
/// line at infinity.
fn heading(line: Strain2) -> P2 {
    line.direction()
}

/// The point `reach` from `from` towards a direction.
fn towards(from: P2, direction: P2, reach: f64) -> P2 {
    from + direction * (reach / direction.ideal_norm())
}

/// A line of the section through the foot of `centre` on it, `reach` either way.
fn through(line: Strain2, centre: P2, reach: f64) -> [P2; 2] {
    let foot = ((centre | line) ^ line).unitized();
    let d = heading(line);
    [towards(foot, d, -reach), towards(foot, d, reach)]
}

/// The walls coloured by a stress, linear along each wall between its ends.
fn draw_stress(c: &mut Canvas, ax: &Axes, body: &Body, stress: impl Fn(P2) -> f64, largest: f64) {
    for [a, b] in body.section.ends() {
        let (sa, sb) = (stress(a), stress(b));
        for i in 0..WALL_PIECES {
            let [f0, f1] = [i, i + 1].map(|j| j as f64 / WALL_PIECES as f64);
            let middle = (f0 + f1) / 2.0;
            let level = 0.5 + 0.5 * (sa * (1.0 - middle) + sb * middle) / largest;
            let ends = [a * (1.0 - f0) + b * f0, a * (1.0 - f1) + b * f1];
            ax.polyline(
                c,
                &ends,
                WALL_WIDTH,
                colormap::rdbu(level as f32).faded(0.7),
            );
        }
    }
}

/// One section at the clamp under a load through a point: its stress, the walls at the tip,
/// its kern, principal and neutral axes, centroid and shear centre, the load and the pushes.
fn draw_section(c: &mut Canvas, rect: Rect, sc: &Scene, k: usize, at: P2, load: P3, bent: &Beam) {
    let body = &sc.bodies[k];
    let ax = Axes::equal(rect, ORIGIN2, SECTION_REACH);
    ax.clip(c);
    let strain = bent.strains[0];
    ax.fill(c, &sc.kerns[k], palette::PURPLE.faded(0.2), 0.4);
    let closed = [&sc.kerns[k][..], &sc.kerns[k][..1]].concat();
    ax.polyline(c, &closed, 1.2, palette::PURPLE);
    draw_stress(c, &ax, body, |p| beam::stress(strain, p), sc.largest);
    let tip = *bent.frames.last().expect("the tip");
    for ends in body.section.ends() {
        let moved = ends.map(|p| beam::chart().of(tip >> beam::in_space(p)));
        ax.polyline(c, &moved, 1.8, palette::SKY);
    }
    let reach = 4.0 * f64::from(SECTION_REACH);
    for axis in body.principal {
        let line = through(axis, body.centroid, reach);
        ax.dashed(c, &line, 1.0, 5.0, palette::INK.faded(0.5));
    }
    let neutral = through(beam::neutral_axis(strain), body.centroid, reach);
    ax.polyline(c, &neutral, 2.0, palette::GREEN);
    let end = towards(at, beam::chart().of(load), ARROW);
    ax.arrow(c, at, end, 2.0, 8.0, palette::ORANGE);
    ax.scatter(c, &[body.centroid], Marker::Dot, 4.0, palette::INK);
    ax.scatter(c, &[body.shear_centre], Marker::Ring, 8.0, palette::INK);
    for (p, push) in beam::pushes(body) {
        let mut ends = body.section.ends().flatten();
        let squeezes_all = ends.all(|q| beam::bending_stress(push, q) < 0.0);
        let colour = if squeezes_all {
            palette::PURPLE
        } else {
            palette::RED
        };
        ax.scatter(c, &[p], Marker::Dot, 6.0, colour);
    }
    c.unclip();
    let title = ax.px(ax.at(0.5, 1.0)) + Point2::direction(0.0, -6.0);
    c.text(NAMES[k], title, 11.0, palette::INK, Align::Center);
    let turned = format!("TIP TWIST {:+.1} DEG", beam::twist(tip).to_degrees());
    let under = ax.px(ax.at(0.5, 0.0)) + Point2::direction(0.0, -8.0);
    c.text(&turned, under, 9.0, palette::INK.faded(0.7), Align::Center);
}

/// The view of space, a map on points: along the beam (`z`) to the view's `x`, across (`x`) to
/// its `y`, up (`y`) to its `z`, so that its camera, with `z` up, sees the beam's up as up.
fn view() -> gax::pga3d::Point<(gax::pga3d::Point,)> {
    let image = gax::pga3d::Point::new;
    gax::pga3d::Point::from_images([
        image(0.0, 1.0, 0.0, 0.0),
        image(0.0, 0.0, 1.0, 0.0),
        image(1.0, 0.0, 0.0, 0.0),
        image(0.0, 0.0, 0.0, 1.0),
    ])
}

/// The four cantilevers in space side by side under their loads, each the straight beam
/// faintly, the bent one by its walls swept along it, its line of shear centres, its clamp and
/// the load at its tip; their names at their clamps.
fn draw_beams(c: &mut Canvas, rect: Rect, sc: &Scene, loads: &[(P2, P3, Load)], beams: &[Beam]) {
    let view = view();
    // Each beam's place across the view, and its clamp's centre there.
    let offset = |k: usize| P3::direction(0.0, BEAM_SPACING * k as f64, 0.0);
    let clamp = |k: usize| P3::origin() + offset(k);
    let middle = (clamp(0) + clamp(3)) * 0.5 + P3::direction(LENGTH / 2.0 + 2.0, 0.0, -0.5);
    let lens = Lens::Parallel(BEAM_VIEW);
    let cam = Camera::orbit(rect, middle, 30.0, BEAM_AZIMUTH, BEAM_ELEVATION, lens);
    let cam = panel3(c, rect, cam, |s| {
        for (k, (body, bent)) in sc.bodies.iter().zip(beams).enumerate() {
            draw_beam(s, body, loads[k], bent, |p| view.of(p) + offset(k));
        }
    });
    c.clip(rect);
    for (k, name) in NAMES.iter().enumerate() {
        if let Some(at) = cam.px(clamp(k) + P3::direction(LENGTH + 0.8, 0.0, 0.0)) {
            c.text(name, at, 9.0, palette::INK.faded(0.8), Align::Left);
        }
    }
    c.unclip();
}

/// One cantilever under its load, its points placed in the view by `place`.
fn draw_beam(
    s: &mut Scene3,
    body: &Body,
    load: (P2, P3, Load),
    bent: &Beam,
    place: impl Fn(P3) -> P3,
) {
    let section = &body.section;
    let at = |frame: beam::Frame, p: P2| place(frame >> beam::in_space(p));
    let straight = beam::bend(&[beam::Strain::zero(); PIECES], LENGTH / PIECES as f64);
    let ends: [_; 2] = [straight[0], straight[PIECES]];
    for frame in ends {
        for [a, b] in section.ends() {
            s.seg(at(frame, a), at(frame, b), 1.0, palette::GRID);
        }
    }
    for p in section
        .nodes
        .iter()
        .step_by(EDGE_EVERY.min(section.nodes.len() - 1))
    {
        s.seg(
            at(ends[0], *p),
            at(ends[1], *p),
            0.8,
            palette::GRID.faded(0.6),
        );
    }
    for pair in bent.frames.windows(2) {
        for [a, b] in section.ends() {
            let corners = [
                at(pair[0], a),
                at(pair[0], b),
                at(pair[1], b),
                at(pair[1], a),
            ];
            let colour = s.lit(corners[0], corners[1], corners[2], palette::SKY.faded(0.5));
            s.quad(corners[0], corners[1], corners[2], corners[3], colour, 0.85);
        }
    }
    let shears: Vec<P3> = bent
        .frames
        .iter()
        .map(|f| at(*f, body.shear_centre))
        .collect();
    s.polyline(&shears, 1.0, palette::INK.faded(0.5));
    let square = [
        (-1.0, -1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
    ];
    let wall: Vec<P3> = square
        .iter()
        .map(|(x, y)| place(P3::xyz(*x, *y, 0.0)))
        .collect();
    s.polyline(&wall, 1.4, palette::INK.faded(0.4));
    let (through, direction, _) = load;
    let tip = at(*bent.frames.last().expect("the tip"), through);
    let head = place(beam::in_space(through) + direction * (TIP_ARROW / beam::LOAD));
    s.arrow(
        tip,
        head - place(beam::in_space(through)),
        2.0,
        8.0,
        palette::ORANGE,
    );
}

/// The side view, a map on points: along the beam (`z`) to the right, up (`y`) up.
fn side() -> gax::pga2d::Point<(gax::pga3d::Point,)> {
    gax::pga2d::Point::from_images([
        P2::new(0.0, 0.0, 0.0),
        P2::new(0.0, 1.0, 0.0),
        P2::new(1.0, 0.0, 0.0),
        P2::new(0.0, 0.0, 1.0),
    ])
}

/// numga's stubby I-beam from the side: the lines along it through its walls' ends, and its
/// cross-sections, solid with the walls' shear stiffness and dashed with the slides held rigid,
/// beside the straight beam, with the clamp and the load at the tip.
fn draw_stubby(c: &mut Canvas, rect: Rect, sc: &Scene) {
    let body = &sc.bodies[0];
    let length = beam::STUBBY_LENGTH;
    let centre = P2::xy(length / 2.0, -0.15);
    let ax = Axes::equal(rect, centre, STUBBY_REACH);
    let side = side();
    let seen = |frame: beam::Frame, p: P2| side.of(frame >> beam::in_space(p));
    for p in &body.section.nodes {
        let start = side.of(beam::in_space(*p));
        let end = start + P2::direction(length, 0.0);
        ax.polyline(c, &[start, end], 1.0, palette::GRID);
    }
    for (bent, rigid) in sc.stubs.iter().zip([false, true]) {
        let colour = if rigid {
            palette::INK.faded(0.6)
        } else {
            palette::SKY
        };
        let draw = |c: &mut Canvas, pts: &[P2]| {
            if rigid {
                ax.dashed(c, pts, 1.4, 5.0, colour);
            } else {
                ax.polyline(c, pts, 1.4, colour);
            }
        };
        for p in &body.section.nodes {
            let along: Vec<P2> = bent.frames.iter().map(|f| seen(*f, *p)).collect();
            draw(c, &along);
        }
        for frame in bent.frames.iter().step_by(ELEVATION_EVERY) {
            for [a, b] in body.section.ends() {
                draw(c, &[seen(*frame, a), seen(*frame, b)]);
            }
        }
    }
    ax.polyline(
        c,
        &[P2::xy(0.0, -0.7), P2::xy(0.0, 0.7)],
        3.0,
        palette::INK.faded(0.6),
    );
    let tip = seen(*sc.stubs[0].frames.last().expect("the tip"), body.centroid);
    let end = towards(tip, P2::direction(0.0, -1.0), ARROW);
    ax.arrow(c, tip, end, 2.0, 8.0, palette::ORANGE);
    let title = ax.px(ax.at(0.5, 1.0)) + Point2::direction(0.0, -6.0);
    let name = "STUBBY I-BEAM: SLIDING (CYAN), RIGID (DASHED)";
    c.text(name, title, 10.0, palette::INK, Align::Center);
}

fn draw(c: &mut Canvas, sc: &Scene, t: f32) {
    backdrop(c);
    let phase = core::f64::consts::TAU * f64::from(t / SECONDS);
    let loads: Vec<(P2, P3, Load)> = sc
        .bodies
        .iter()
        .enumerate()
        .map(|(k, b)| load(b, k, phase))
        .collect();
    let beams: Vec<Beam> = sc
        .bodies
        .iter()
        .zip(&loads)
        .map(|(b, (_, _, l))| beam::cantilever(b, *l, LENGTH))
        .collect();
    let below = c.rect().part(0.0, 0.12, 1.0, 1.0);
    for (k, bent) in beams.iter().enumerate() {
        let (column, row) = ((k % 2) as f32, (k / 2) as f32);
        let cell = below.part(
            0.25 * column,
            0.5 * row,
            0.25 * (column + 1.0),
            0.5 * (row + 1.0),
        );
        let (through, direction, _) = loads[k];
        draw_section(
            c,
            cell.inset(6.0, 18.0, 6.0, 4.0),
            sc,
            k,
            through,
            direction,
            bent,
        );
    }
    draw_beams(c, below.part(0.5, 0.0, 1.0, 0.62), sc, &loads, &beams);
    draw_stubby(
        c,
        below.part(0.5, 0.62, 1.0, 1.0).inset(6.0, 18.0, 6.0, 4.0),
        sc,
    );
    caption(
        c,
        "THIN-WALLED BEAMS: BENDING, TWISTING ABOUT THE SHEAR CENTRE, AND SLIDING",
        "STRESS AT THE CLAMP (BLUE SQUEEZED, RED STRETCHED), TIP (CYAN), NEUTRAL AXIS (GREEN), \
         KERN (PURPLE), CENTROID (DOT), SHEAR CENTRE (RING), LOAD (ORANGE)",
    );
}

fn main() {
    run_with(Anim::new("bending", SECONDS).size(960, 540), scene, draw);
}

#[cfg(test)]
mod tests {
    use super::beam::*;
    use super::sections::*;
    use gax::vga2d;

    type Map2 = vga2d::Vector<(vga2d::Vector,)>;

    fn map(c: [[f64; 2]; 2]) -> Map2 {
        Map2::from_coeffs(c)
    }

    /// A map turned in the plane by `degrees`: moved by the rotor, its input and its output.
    fn turned(m: Map2, degrees: f64) -> Map2 {
        let turn = vga2d::Pseudoscalar::new(-degrees.to_radians() / 2.0).exp();
        turn >> m.of(turn << vga2d::Vector::slot())
    }

    /// numga's closed-form thin-walled second moments of the two flanges and the web along
    /// their midlines, per section (the I-section, turned, the Z-section), as maps on the
    /// plane's vectors: across and up.
    fn second_moments() -> [Map2; 3] {
        let flange_area = FLANGE * THICKNESS;
        let across = 2.0 * THICKNESS * FLANGE.powi(3) / 12.0;
        let up = 2.0 * flange_area * (HEIGHT / 2.0).powi(2) + THICKNESS * HEIGHT.powi(3) / 12.0;
        let product = 2.0 * flange_area * (FLANGE / 2.0) * (HEIGHT / 2.0);
        let aligned = map([[across, 0.0], [0.0, up]]);
        let z_section = map([
            [across + 2.0 * flange_area * (FLANGE / 2.0).powi(2), product],
            [product, up],
        ]);
        [aligned, turned(aligned, TURNS[1]), z_section]
    }

    /// numga's closed-form shear areas: each wall carries a slide along itself, so the flanges
    /// carry a slide across and the web a slide up.
    fn shear_areas() -> [Map2; 3] {
        let aligned = map([[2.0 * FLANGE * THICKNESS, 0.0], [0.0, HEIGHT * THICKNESS]]);
        [aligned, turned(aligned, TURNS[1]), aligned]
    }

    /// A map's entry: `row | map(column)`.
    fn entry(m: Map2, row: usize, column: usize) -> f64 {
        let axis =
            |k: usize| vga2d::Vector::new(f64::from(u8::from(k == 0)), f64::from(u8::from(k == 1)));
        (axis(row) | m.of(axis(column))).s()
    }

    /// The lines through the centroid across and up, as strains.
    fn turns(body: &Body) -> [Strain2; 2] {
        [P2::direction(1.0, 0.0), P2::direction(0.0, 1.0)]
            .map(|d| Strain2::through(body.centroid, d))
    }

    /// A bending strain's energy, `strain & bending(strain)`.
    fn energy(body: &Body, strain: Strain2) -> f64 {
        (strain & body.bending.of(strain)).s()
    }

    fn bodies() -> Vec<Body> {
        sections()
            .into_iter()
            .map(|s| body(s, SHEAR_MODULUS))
            .collect()
    }

    fn relative(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs()
    }

    fn down(body: &Body) -> Beam {
        cantilever(body, downward(body.centroid, LOAD, LENGTH), LENGTH)
    }

    /// numga's `test_stiffness_holds_the_area_and_the_second_moments`.
    #[test]
    fn stiffness_holds_the_area_and_the_second_moments() {
        let area = (2.0 * FLANGE + HEIGHT) * THICKNESS;
        for (body, spread) in bodies().iter().zip(second_moments()) {
            assert!(relative(energy(body, Strain2::infinity()), MODULUS * area) < 1e-12);
            let [about_x, about_y] = turns(body).map(|l| energy(body, l));
            assert!(relative(about_x, MODULUS * entry(spread, 1, 1)) < 1e-12);
            assert!(relative(about_y, MODULUS * entry(spread, 0, 0)) < 1e-12);
        }
    }

    /// numga's `test_small_tip_load_deflects_by_the_cantilever_formula`: bending along the
    /// inverse second moment and the slide along the inverse shear area, on the slender beams
    /// and on the stubby one, where the slide is a large part of the deflection. The tolerance
    /// tests the approximation by pieces of constant strain.
    #[test]
    fn small_tip_load_deflects_by_the_cantilever_formula() {
        let load = 1e-3 * LOAD;
        for length in [LENGTH, STUBBY_LENGTH] {
            let cases = bodies()
                .into_iter()
                .zip(second_moments())
                .zip(shear_areas());
            for ((body, spread), shear) in cases {
                let bent = cantilever(&body, downward(body.centroid, load, length), length);
                let tip = *bent.frames.last().expect("the tip") >> in_space(body.centroid);
                let [x, y] = flat(tip);
                let [cx, cy] = body.centroid.to_euclidean();
                let force = vga2d::Vector::new(0.0, -load);
                let expected = spread.solve(force) * (length.powi(3) / (3.0 * MODULUS))
                    + shear.solve(force) * (length / SHEAR_MODULUS);
                let moved = vga2d::Vector::new(x - cx, y - cy);
                let error = (moved - expected).norm() / expected.norm();
                assert!(error < 1e-3, "{length} {error}");
            }
        }
    }

    /// numga's `test_slotted_tube_matches_a_thin_circular_arc`: the area, the centroid, the
    /// second moments and the shear centre of a thin arc; the tolerances measure the polygon
    /// against the arc.
    #[test]
    #[allow(clippy::disallowed_methods)] // the closed forms' sines and cosines of the slot angle
    fn slotted_tube_matches_a_thin_circular_arc() {
        let tube = &bodies()[3];
        let sweep = core::f64::consts::TAU - SLOT_ANGLE.to_radians();
        let half = sweep / 2.0;
        let area = sweep * TUBE_RADIUS * TUBE_WALL;
        let centre_x = -TUBE_RADIUS * half.sin() / half;
        let radial = TUBE_RADIUS.powi(3) * TUBE_WALL;
        let slot = SLOT_ANGLE.to_radians().sin();
        let across = radial * (sweep - slot) / 2.0 - area * centre_x.powi(2);
        let up = radial * (sweep + slot) / 2.0;
        let shear_x = -2.0 * TUBE_RADIUS * (half.sin() - half * half.cos())
            / (half - half.sin() * half.cos());
        assert!(relative(super::beam::area(&tube.section), area) < 1e-2);
        let [x, y] = tube.centroid.to_euclidean();
        assert!(relative(x, centre_x) < 1e-2 && y.abs() < 1e-12);
        let [about_x, about_y] = turns(tube).map(|l| energy(tube, l));
        assert!(relative(about_x, MODULUS * up) < 1e-2);
        assert!(relative(about_y, MODULUS * across) < 1e-2);
        let [sx, sy] = tube.shear_centre.to_euclidean();
        assert!(relative(sx, shear_x) < 1e-2 && sy.abs() < 1e-12);
    }

    /// numga's `test_warping_from_the_shear_centre_is_invisible_to_bending`: the warped pull
    /// from the shear centre runs through the origin, no couple; and mirror or point symmetry
    /// puts the shear centre on the centroid.
    #[test]
    fn warping_from_the_shear_centre_is_invisible_to_bending() {
        for (k, body) in bodies().iter().enumerate() {
            let warped = warping(&body.section).of(body.shear_centre);
            assert!((warped & P2::origin()).norm() < 1e-12);
            if k < 3 {
                let [a, b] = [body.shear_centre, body.centroid].map(|p| p.to_euclidean());
                assert!((a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12);
            }
        }
    }

    /// numga's `test_torque_twists_about_the_shear_centre`: a torque twists the tip by the
    /// torque times the length over the torsional stiffness, about the shear centre, which stays
    /// where it is; a load through the shear centre bends the beam without twisting it.
    #[test]
    fn torque_twists_about_the_shear_centre() {
        let torque = 1e-5;
        for body in bodies() {
            let stiffness = twisting(&body.section, SHEAR_MODULUS);
            let twisted = cantilever(&body, couple_about_beam() * torque, LENGTH);
            let tip = *twisted.frames.last().expect("the tip");
            assert!(relative(twist(tip), torque * LENGTH / stiffness) < 1e-12);
            let [a, b] = [
                flat(tip >> in_space(body.shear_centre)),
                body.shear_centre.to_euclidean(),
            ];
            assert!((a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12);
            let through = cantilever(&body, downward(body.shear_centre, LOAD, LENGTH), LENGTH);
            assert!(through.strains.iter().all(|s| twist_rate(*s).abs() < 1e-12));
        }
    }

    /// numga's `test_neutral_axes_and_principal_bending_directions`: under the downward load
    /// through the centroid, the bending part of the strain, its neutral axis, and the principal
    /// axes pass through the centroid.
    #[test]
    fn neutral_axes_and_principal_bending_directions() {
        for body in &bodies() {
            let strain = down(body).strains[0];
            assert!((neutral_axis(strain) & body.centroid).s().abs() < 1e-12);
            for axis in body.principal {
                assert!((body.centroid & axis).s().abs() < 1e-12);
            }
        }
    }

    /// The bending part of a strain is what the fibres see: its neutral axis pairs with each
    /// point as the strain with the point's fibre; and a pull along the beam through a point
    /// inside the kern stretches every fibre, its neutral axis the point's polar line in the
    /// bending stiffness.
    #[test]
    fn the_fibres_see_the_bending_part() {
        for body in &bodies() {
            let strain = down(body).strains[0];
            for p in body.section.ends().flatten() {
                let stretch = (neutral_axis(strain) & p).s();
                assert!((stretch - (strain & fibre(p)).s()).abs() < 1e-15);
            }
            let [x, y] = PUSH_POINTS[0];
            let p = P2::xy(x, y);
            let along = P3::direction(0.0, 0.0, PUSH);
            let pulled = cantilever(body, tip(p, LENGTH) & along, LENGTH).strains[0];
            assert!(
                body.section
                    .ends()
                    .flatten()
                    .all(|q| stress(pulled, q) > 0.0)
            );
            let polar = pull(body.bending, p);
            assert!((neutral_axis(pulled) ^ polar).norm() < 1e-12 * polar.norm());
        }
    }

    /// numga's `test_kern_and_reciprocity`: a pull through the centroid does not bend; the push
    /// inside the kern squeezes every fibre, the one outside stretches some; each node's polar
    /// stretches the fibre through a point as much as a pull through the point stretches the
    /// node's; and a push through a corner of the kern leaves some fibre unstretched and squeezes
    /// the rest.
    #[test]
    fn kern_and_reciprocity() {
        let scene = super::scene();
        for (body, kern) in scene.bodies.iter().zip(&scene.kerns) {
            assert!(pull(body.bending, body.centroid).norm() < 1e-12);
            let [(p0, push0), (p1, push1)] = pushes(body);
            let ends: Vec<P2> = body.section.ends().flatten().collect();
            assert!(ends.iter().all(|q| bending_stress(push0, *q) < 0.0));
            assert!(ends.iter().any(|q| bending_stress(push1, *q) > 0.0));
            for p in [p0, p1] {
                for q in &body.section.nodes {
                    let (polar, direct) = (pull(body.bending, *q) & p, pull(body.bending, p) & *q);
                    assert!((polar.s() - direct.s()).abs() < 1e-12);
                }
            }
            for corner in kern {
                let stretches = body.section.nodes.iter();
                let least = stretches
                    .map(|q| (pull(body.bending, *corner) & *q).s())
                    .fold(f64::MAX, f64::min);
                assert!(least.abs() < 1e-9, "{least}");
            }
        }
    }

    /// The checks of numga's scenario: every tip under the downward load moves down; the
    /// I-section's, mirror-symmetric about the load, straight down; the turned I-section's and
    /// the Z-section's sideways as well, by more than half the most either moves down. The
    /// slotted tube's load, moved across, twists it one way, then the other, and through the
    /// shear centre hardly at all. The stubby beam's slides add a deflection comparable to its
    /// bending's. For each section the push inside the kern squeezes every fibre and the one
    /// outside stretches some, and a pull through one push point stretches the fibre through the
    /// other as much as the exchanged pull.
    #[test]
    fn scenes_pass_their_checks() {
        let bodies = bodies();
        let tips: Vec<[f64; 2]> = bodies.iter().map(|b| deflection(b, &down(b))).collect();
        let [sideways, falls]: [Vec<f64>; 2] = [0, 1].map(|i| tips.iter().map(|d| d[i]).collect());
        assert!(falls.iter().all(|d| *d < 0.0) && sideways[0].abs() < 1e-12);
        let most = falls[1].abs().max(falls[2].abs());
        assert!(sideways[1].abs().min(sideways[2].abs()) > 0.5 * most);
        let tube = &bodies[3];
        let frames = 48;
        let rates: Vec<f64> = (0..frames)
            .map(|k| {
                let through = across(k as f64 / (frames - 1) as f64);
                let bent = cantilever(tube, downward(through, LOAD, LENGTH), LENGTH);
                twist_rate(bent.strains[0])
            })
            .collect();
        assert!(rates[0] * rates[frames - 1] < 0.0);
        let largest = rates.iter().map(|r| r.abs()).fold(0.0, f64::max);
        let shear_x = tube.shear_centre.to_euclidean()[0];
        let passing = (0..frames)
            .min_by(|a, b| {
                let off = |k: usize| {
                    (across(k as f64 / (frames - 1) as f64).to_euclidean()[0] - shear_x).abs()
                };
                off(*a).total_cmp(&off(*b))
            })
            .expect("frames");
        assert!(rates[passing].abs() < 0.1 * largest);
        let scene = super::scene();
        let [sliding, rigid] = scene.stubs.each_ref().map(|b| deflection(&bodies[0], b)[1]);
        assert!(sliding < rigid && rigid < 0.0 && sliding < 1.2 * rigid);
        for body in &bodies {
            let [(p0, push0), (p1, push1)] = pushes(body);
            let mut ends = body.section.ends().flatten();
            assert!(ends.all(|q| bending_stress(push0, q) < 0.0));
            assert!(
                body.section
                    .ends()
                    .flatten()
                    .any(|q| bending_stress(push1, q) > 0.0)
            );
            let (first, second) = (pull(body.bending, p0), pull(body.bending, p1));
            assert!(((first & p1).s() - (second & p0).s()).abs() < 1e-12);
        }
    }

    /// numga 7dff347's numbers: the sections' centroids and shear centres, torsional
    /// stiffnesses, the tips and twists under the downward loads, the stress at the I-section's
    /// clamp, the Z-section's tip as its load turns, the slotted tube's rate of twist and the
    /// twist of its tip as its load moves across, and the stubby beam's tips.
    #[test]
    fn numbers_match_numga() {
        let bodies = bodies();
        let near = |a: f64, b: f64, tolerance: f64| (a - b).abs() < tolerance;
        let [x, y] = bodies[3].centroid.to_euclidean();
        assert!(near(x, -8.7766396346986e-02, 1e-12) && y.abs() < 1e-12);
        let [x, y] = bodies[3].shear_centre.to_euclidean();
        assert!(
            near(x, -8.3360810002555e-01, 1e-12) && y.abs() < 1e-12,
            "{x}"
        );
        let stiffness = bodies.iter().map(|b| twisting(&b.section, SHEAR_MODULUS));
        for (g, numga) in stiffness.zip([
            1.3915897435897437e-4,
            1.3915897435897437e-4,
            1.3915897435897437e-4,
            3.086359477351744e-4,
        ]) {
            assert!(relative(g, numga) < 1e-13);
        }
        let numga = [
            ([0.0, -1.3420016725074e-01], 0.0),
            ([-0.4437532810023, -0.3902896831481], -0.001940499224886175),
            ([0.3729724311547, -0.3278564618737], 0.0013645131312801273),
            ([-0.0287675750824, -0.2932122857153], -0.24163953427373178),
        ];
        for (body, ([dx, dy], turn)) in bodies.iter().zip(numga) {
            let bent = down(body);
            let [x, y] = deflection(body, &bent);
            assert!(near(x, dx, 1e-12) && near(y, dy, 1e-12), "{x} {y}");
            let tip = *bent.frames.last().expect("the tip");
            assert!(near(twist(tip), turn, 1e-13), "{}", twist(tip));
        }
        let clamp = down(&bodies[0]).strains[0];
        let stresses = bodies[0].section.nodes.iter().map(|p| stress(clamp, *p));
        for (s, side) in stresses.zip([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]) {
            assert!(near(s, side * 1.7810335706e-3, 1e-13), "{s}");
        }
        let z = &bodies[2];
        let turning = [
            [0.3729724311547, -0.3278564618737],
            [0.7199872333083, -0.3725765736692],
            [-0.3729724311547, 0.3278564618737],
            [-0.7199872333083, 0.3725765736692],
        ];
        for (k, [dx, dy]) in [0, 12, 24, 36].into_iter().zip(turning) {
            let angle = core::f64::consts::TAU * k as f64 / 48.0;
            let [x, y] = deflection(z, &cantilever(z, turned_load(z, angle), LENGTH));
            assert!(near(x, dx, 1e-12) && near(y, dy, 1e-12), "{k}: {x} {y}");
        }
        let tube = &bodies[3];
        let moving = [
            (0, -0.0399696830223, -0.3996685119264),
            (24, -0.014324926878, -0.143238484901),
            (47, 0.0102512977602, 0.1025052271037),
        ];
        for (k, rate, turn) in moving {
            let through = across(k as f64 / 47.0);
            let bent = cantilever(tube, downward(through, LOAD, LENGTH), LENGTH);
            assert!(near(twist_rate(bent.strains[0]), rate, 1e-12), "{k}");
            let tip = *bent.frames.last().expect("the tip");
            assert!(near(twist(tip), turn, 1e-12), "{k}: {}", twist(tip));
        }
        let scene = super::scene();
        let [sliding, rigid] = scene.stubs.each_ref().map(|b| deflection(&bodies[0], b));
        assert!(sliding[0].abs() < 1e-12 && near(sliding[1], -2.6214833486910e-01, 1e-12));
        assert!(rigid[0].abs() < 1e-12 && near(rigid[1], -1.5656073963774e-01, 1e-11));
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scene(), 0.5);
    }
}
