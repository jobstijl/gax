//! numga's `mechanics/bending`: Euler-Bernoulli beam bending. A cantilever loaded away from its
//! principal bending directions bends sideways as well as along the load.
//!
//! The beam runs along `z` from its clamp at `z = 0`, and a cross-section is a plane of PGA2D
//! (numga works in PGA3D with the lines of the section's plane as a kind of their own; gax's
//! PGA2D has them as its lines, and its homomorphism into PGA3D, a point to the vertical line
//! through it and a line to the vertical plane, carries them into space). Bending keeps each
//! cross-section plane, and the strain is a line of the section: each fibre, through a point
//! `p` of the section, stretches by their pairing `strain & p`. The line at infinity stretches
//! every fibre by one; any other line is the neutral axis, the fibres neither stretched nor
//! squeezed, about which the section turns. Each fibre pulls along the beam through its point
//! with the modulus times its stretch, so the fibres' resultant is a point weighted by the
//! force (a direction, for a couple), and the stiffness a map `Point <- Line`: summed exactly
//! over the triangles by the simplex's lumped mass points, it holds the area and the first and
//! second moments of area, with no centroid and no arms.
//!
//! A load is a line of space. What the fibres at a station carry is the load seen from that
//! cross-section, met with its plane: a couple for a load across the beam, the load itself for
//! a push along it. The strain is one solve with the stiffness, and the running product of the
//! twists of the pieces, each the line at infinity plus the strain lifted into space, gives the
//! frame of each cross-section (a small-deflection load with a finite-motion reconstruction).
//!
//! For a load across the beam the neutral axis runs through the centroid. With a mirror
//! symmetry along the load it lies square to the load and the beam bends along it; for a
//! Z-section, or an I-section turned in its plane, it tilts, and the beam bends sideways as
//! well as down. The stiffness is a polarity: the strain of a pull through a point is that
//! point's polar line, and the points whose push squeezes every fibre form the kern, cut out
//! here exactly by the boundary points' polar lines (numga samples it on a grid).
//!
//! The animation turns each beam's tip load once around: the four sections at the clamp with
//! their stress (blue squeezed, red stretched), principal axes (dashed), neutral axis, kern,
//! load and the tip's motion, and numga's two pushes along the beam (purple where the push
//! squeezes every fibre, inside the kern; red outside); beside them the beams bending in space
//! over the straight ones.

mod beam;
mod sections;
use gax_numga_examples::shared::mechanics_simplex as simplex;

use beam::{Beam, Body, LENGTH, P3, Strain};
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Marker, ORIGIN2, Point2, Rect, Scene3, backdrop,
    caption, colormap, palette, run_with,
};
use sections::{NAMES, P2};

const SECONDS: f32 = 12.0;
/// Load directions sampled for the stress colours' range.
const LOAD_SAMPLES: usize = 48;
/// The drawn length of the arrows in a section, and of the load at the tip.
const ARROW: f64 = 0.45;
const TIP_ARROW: f64 = 1.4;
/// The pieces between drawn cross-sections, and about how many edges are drawn along a beam.
const OUTLINE_EVERY: usize = 10;
const EDGES: usize = 12;
/// The half height of the section views.
const SECTION_REACH: f32 = 0.72;
/// The beams' spacing in the view of space, the view's azimuth and elevation, and its half
/// height.
const BEAM_SPACING: f64 = 3.2;
const BEAM_AZIMUTH: f32 = -2.0;
const BEAM_ELEVATION: f32 = 0.6;
const BEAM_VIEW: f32 = 8.0;

/// The sections' bodies, their triangles' edges and their kerns, and the largest stress at the
/// clamp under any turn of the load, the colour scale's end.
struct Scene {
    bodies: Vec<Body>,
    edges: Vec<Vec<[Edge; 3]>>,
    kerns: Vec<Vec<P2>>,
    largest: f64,
}

fn scene() -> Scene {
    let bodies: Vec<Body> = sections::sections().into_iter().map(beam::body).collect();
    let edges = bodies.iter().map(|b| edges(&b.section)).collect();
    let kerns = bodies.iter().map(beam::kern).collect();
    let largest = bodies
        .iter()
        .flat_map(|b| (0..LOAD_SAMPLES).map(move |k| (b, k)))
        .map(|(b, k)| {
            let angle = core::f64::consts::TAU * k as f64 / LOAD_SAMPLES as f64;
            let clamp = beam::cantilever(b, beam::turned_load(angle)).strains[0];
            corner_stresses(b, clamp).map(f64::abs).fold(0.0, f64::max)
        })
        .fold(0.0, f64::max);
    Scene {
        bodies,
        edges,
        kerns,
        largest,
    }
}

/// A triangle's edge: the join of two of its corners, an intrinsic line.
type Edge = gax::pga2d::intrinsic::Line;

/// Each triangle's edges as lines, the triangle on their positive sides.
fn edges(section: &sections::Section) -> Vec<[Edge; 3]> {
    let triangles = section.triangles.iter();
    triangles
        .map(|[a, b, c]| [*a & *b, *b & *c, *c & *a])
        .collect()
}

/// Whether a point lies in one of the triangles with these edges.
fn inside(edges: &[[Edge; 3]], p: P2) -> bool {
    edges
        .iter()
        .any(|lines| lines.iter().all(|l| (*l & p).s() >= 0.0))
}

/// The stresses a strain gives at the section's corners.
fn corner_stresses(body: &Body, strain: Strain) -> impl Iterator<Item = f64> + '_ {
    let corners = body.section.triangles.iter().flatten();
    corners.map(move |p| beam::stress(strain, *p))
}

/// The direction of a line of the section, as the displacement along it: its meet with the
/// line at infinity.
fn heading(line: Strain) -> P2 {
    line.direction()
}

/// The point `reach` from `from` towards a direction.
fn towards(from: P2, direction: P2, reach: f64) -> P2 {
    from + direction * (reach / direction.ideal_norm())
}

/// A line of the section through the foot of `centre` on it, `reach` either way.
fn through(line: Strain, centre: P2, reach: f64) -> [P2; 2] {
    let foot = ((centre | line) ^ line).unitized();
    let d = heading(line);
    [towards(foot, d, -reach), towards(foot, d, reach)]
}

/// One section at the clamp under a load: its stress, kern, principal and neutral axes, the
/// load's direction and the way the tip moves.
fn draw_section(c: &mut Canvas, rect: Rect, sc: &Scene, k: usize, load: P3, beam: &Beam) {
    let body = &sc.bodies[k];
    let ax = Axes::equal(rect, ORIGIN2, SECTION_REACH);
    let strain = beam.strains[0];
    let largest = sc.largest;
    ax.image(c, 1, |q| {
        let p = q.map_coefs(f64::from);
        let level = 0.5 + 0.5 * beam::stress(strain, p) / largest;
        inside(&sc.edges[k], p).then(|| colormap::rdbu(level as f32).faded(0.9))
    });
    let closed = |pts: &[P2]| [pts, &pts[..1]].concat();
    ax.polyline(
        c,
        &closed(&body.section.boundary),
        1.2,
        palette::INK.faded(0.7),
    );
    ax.fill(c, &sc.kerns[k], palette::PURPLE.faded(0.2), 0.4);
    ax.polyline(c, &closed(&sc.kerns[k]), 1.2, palette::PURPLE);
    let reach = 2.0 * f64::from(SECTION_REACH);
    for axis in body.principal {
        ax.dashed(
            c,
            &through(axis, body.centroid, reach),
            1.0,
            5.0,
            palette::INK.faded(0.5),
        );
    }
    ax.polyline(
        c,
        &through(strain, body.centroid, reach),
        2.0,
        palette::GREEN,
    );
    let load = beam::chart().of(load);
    let [dx, dy] = beam::deflection(body, beam);
    let tip = P2::direction(dx, dy);
    for (direction, colour) in [(load, palette::ORANGE), (tip, palette::SKY)] {
        let end = towards(body.centroid, direction, ARROW);
        ax.arrow(c, body.centroid, end, 2.0, 8.0, colour);
    }
    for (p, push) in beam::pushes(body) {
        let squeezes_all = corner_stresses(body, push).all(|s| s < 0.0);
        let colour = if squeezes_all {
            palette::PURPLE
        } else {
            palette::RED
        };
        ax.scatter(c, &[p], Marker::Dot, 6.0, colour);
    }
    let title = ax.px(ax.at(0.5, 1.0)) + Point2::direction(0.0, -6.0);
    c.text(NAMES[k], title, 11.0, palette::INK, Align::Center);
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

/// The four cantilevers in space side by side under a load, each the straight beam faintly, the
/// bent one by a few cross-sections and the edges along it, and the load at its tip; their names
/// at their clamps.
fn draw_beams(c: &mut Canvas, rect: Rect, sc: &Scene, load: P3, beams: &[Beam]) {
    let view = view();
    // Each beam's place across the view, and its clamp's centre there.
    let offset = |k: usize| P3::direction(0.0, BEAM_SPACING * k as f64, 0.0);
    let clamp = |k: usize| P3::origin() + offset(k);
    let middle = (clamp(0) + clamp(3)) * 0.5 + P3::direction(LENGTH / 2.0, 0.0, -0.5);
    let lens = Lens::Parallel(BEAM_VIEW);
    let cam = Camera::orbit(rect, middle, 30.0, BEAM_AZIMUTH, BEAM_ELEVATION, lens);
    let cam = panel3(c, rect, cam, |s| {
        for (k, (body, beam)) in sc.bodies.iter().zip(beams).enumerate() {
            draw_beam(s, body, load, beam, |p| view.of(p) + offset(k));
        }
    });
    c.clip(rect);
    for (k, name) in NAMES.iter().enumerate() {
        if let Some(at) = cam.px(clamp(k) + P3::direction(1.5, 0.0, -1.3)) {
            c.text(name, at, 10.0, palette::INK.faded(0.8), Align::Center);
        }
    }
    c.unclip();
}

/// One cantilever under its load, its points placed in the view by `place`.
fn draw_beam(s: &mut Scene3, body: &Body, load: P3, beam: &Beam, place: impl Fn(P3) -> P3) {
    let boundary = &body.section.boundary;
    let edges: Vec<usize> = (0..boundary.len())
        .step_by(boundary.len().div_ceil(EDGES))
        .collect();
    let straight = beam::bend(
        &[Strain::zero(); beam::PIECES],
        LENGTH / beam::PIECES as f64,
    );
    for (frames, colour) in [(&straight, palette::GRID), (&beam.frames, palette::SKY)] {
        let at = |k: usize, i: usize| place(frames[k] >> beam::in_space(boundary[i]));
        for k in (0..frames.len()).step_by(OUTLINE_EVERY) {
            let ring: Vec<_> = (0..=boundary.len())
                .map(|i| at(k, i % boundary.len()))
                .collect();
            s.polyline(&ring, 1.2, colour);
        }
        for &i in &edges {
            let edge: Vec<_> = (0..frames.len()).map(|k| at(k, i)).collect();
            s.polyline(&edge, 0.8, colour.faded(0.6));
        }
    }
    let tip = *beam::centres(body, beam).last().expect("the tip");
    let head = tip + load * (TIP_ARROW / beam::LOAD);
    s.arrow(
        place(tip),
        place(head) - place(tip),
        2.0,
        8.0,
        palette::ORANGE,
    );
}

fn draw(c: &mut Canvas, sc: &Scene, t: f32) {
    backdrop(c);
    let angle = core::f64::consts::TAU * f64::from(t / SECONDS);
    let load = beam::turned_load(angle);
    let beams: Vec<Beam> = sc
        .bodies
        .iter()
        .map(|b| beam::cantilever(b, load))
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
        draw_section(c, cell.inset(6.0, 18.0, 6.0, 4.0), sc, k, load, bent);
    }
    draw_beams(c, below.part(0.5, 0.0, 1.0, 1.0), sc, load, &beams);
    caption(
        c,
        "EULER-BERNOULLI BEAM BENDING: A TIP LOAD TURNING ONCE AROUND",
        "STRESS AT THE CLAMP (BLUE SQUEEZED, RED STRETCHED), NEUTRAL AXIS (GREEN), KERN (PURPLE), \
         LOAD (ORANGE), TIP MOTION (CYAN)",
    );
}

fn main() {
    run_with(Anim::new("bending", SECONDS).size(960, 540), scene, draw);
}

#[cfg(test)]
mod tests {
    use super::beam::*;
    use super::scene;
    use super::sections::*;
    use gax::vga2d;

    /// numga's closed-form second moments of area of the two flanges and the web, per section
    /// (the I-section, turned, the Z-section), as maps on the plane's vectors: across and up.
    fn second_moments() -> [vga2d::Vector<(vga2d::Vector,)>; 3] {
        let flange_area = FLANGE * THICKNESS;
        let web_height = HEIGHT - 2.0 * THICKNESS;
        let web_area = web_height * THICKNESS;
        let offset_y = (HEIGHT - THICKNESS) / 2.0;
        let offset_x = (FLANGE - THICKNESS) / 2.0;
        let across =
            2.0 * flange_area * FLANGE.powi(2) / 12.0 + web_area * THICKNESS.powi(2) / 12.0;
        let up = 2.0 * flange_area * (offset_y.powi(2) + THICKNESS.powi(2) / 12.0)
            + web_area * web_height.powi(2) / 12.0;
        let product = 2.0 * flange_area * offset_x * offset_y;
        let map = |c: [[f64; 2]; 2]| vga2d::Vector::<(vga2d::Vector,)>::from_coeffs(c);
        let aligned = map([[across, 0.0], [0.0, up]]);
        // Turned in the plane: the map moved by the rotor, its input and its output.
        let turn = vga2d::Pseudoscalar::new(-TURN.to_radians() / 2.0).exp();
        let turned = turn >> aligned.of(turn << vga2d::Vector::slot());
        let z_section = map([
            [across + 2.0 * flange_area * offset_x.powi(2), product],
            [product, up],
        ]);
        [aligned, turned, z_section]
    }

    /// A map's entry: `row | map(column)`.
    fn entry(map: vga2d::Vector<(vga2d::Vector,)>, row: usize, column: usize) -> f64 {
        let axis =
            |k: usize| vga2d::Vector::new(f64::from(u8::from(k == 0)), f64::from(u8::from(k == 1)));
        (axis(row) | map.of(axis(column))).s()
    }

    /// The lines through the centroid across and up, as strains: joins taken as the axes of
    /// turns.
    fn turns(body: &Body) -> [Strain; 2] {
        [P2::direction(1.0, 0.0), P2::direction(0.0, 1.0)]
            .map(|d| Strain::through(body.centroid, d))
    }

    /// A strain's energy, `strain & stiffness(strain)`.
    fn energy(body: &Body, strain: Strain) -> f64 {
        (strain & body.stiffness.of(strain)).s()
    }

    fn bodies() -> Vec<Body> {
        sections().into_iter().map(body).collect()
    }

    fn relative(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs()
    }

    /// numga's `test_stiffness_holds_the_area_and_the_second_moments`.
    #[test]
    fn stiffness_holds_the_area_and_the_second_moments() {
        let area = 2.0 * FLANGE * THICKNESS + (HEIGHT - 2.0 * THICKNESS) * THICKNESS;
        for (body, spread) in bodies().iter().zip(second_moments()) {
            assert!(relative(energy(body, shift_along_beam()), MODULUS * area) < 1e-12);
            let [about_x, about_y] = turns(body).map(|l| energy(body, l));
            assert!(relative(about_x, MODULUS * entry(spread, 1, 1)) < 1e-12);
            assert!(relative(about_y, MODULUS * entry(spread, 0, 0)) < 1e-12);
        }
    }

    /// numga's `test_small_tip_load_deflects_by_the_cantilever_formula`: the tolerance tests
    /// the approximation by pieces of constant strain.
    #[test]
    fn small_tip_load_deflects_by_the_cantilever_formula() {
        let load = 1e-3 * LOAD;
        for (body, spread) in bodies().iter().zip(second_moments()) {
            let beam = cantilever(body, P3::direction(0.0, -load, 0.0));
            let [x, y] = deflection(body, &beam);
            let curvature = spread.solve(vga2d::Vector::new(0.0, -load));
            let expected = curvature * (LENGTH.powi(3) / (3.0 * MODULUS));
            let error = (vga2d::Vector::new(x, y) - expected).norm() / expected.norm();
            assert!(error < 1e-3, "{error}");
        }
    }

    /// numga's `test_slotted_tube_matches_annular_sector_moments`: the tolerance measures the
    /// polygon against the curved annular sector.
    #[test]
    #[allow(clippy::disallowed_methods)] // the closed form's sines of the slot angle
    fn slotted_tube_matches_annular_sector_moments() {
        let tube = &bodies()[3];
        let inner = TUBE_RADIUS - TUBE_WALL;
        let slot = SLOT_ANGLE.to_radians();
        let sweep = core::f64::consts::TAU - slot;
        let area = sweep * (TUBE_RADIUS.powi(2) - inner.powi(2)) / 2.0;
        let centre_x =
            -2.0 * (slot / 2.0).sin() * (TUBE_RADIUS.powi(3) - inner.powi(3)) / (3.0 * area);
        let radial = (TUBE_RADIUS.powi(4) - inner.powi(4)) / 4.0;
        let across = radial * (sweep - slot.sin()) / 2.0 - area * centre_x.powi(2);
        let up = radial * (sweep + slot.sin()) / 2.0;
        let polygon: f64 = tube.section.triangles.iter().map(super::beam::area).sum();
        assert!(relative(polygon, area) < 0.01);
        let [x, y] = tube.centroid.to_euclidean();
        assert!(relative(x, centre_x) < 0.01 && y.abs() < 1e-12);
        let [about_x, about_y] = turns(tube).map(|l| energy(tube, l));
        assert!(relative(about_x, MODULUS * up) < 0.01);
        assert!(relative(about_y, MODULUS * across) < 0.01);
    }

    /// numga's `test_neutral_axes_and_principal_bending_directions`.
    #[test]
    fn neutral_axes_and_principal_bending_directions() {
        let quarter = gax::pga2d::Motor::rotation(P2::origin(), core::f64::consts::FRAC_PI_2);
        for body in &bodies() {
            let beam = cantilever(body, downward());
            assert!((body.centroid & beam.strains[0]).s().abs() < 1e-12);
            for axis in body.principal {
                assert!((body.centroid & axis).s().abs() < 1e-12);
            }
            // A load square to the weakest axis bends the beam along the load alone: the tip
            // stays in the plane of the beam's axis and the load.
            let weakest = body.principal[0].normalized();
            let load = in_space(quarter >> super::heading(weakest)) * LOAD;
            let along_principal = cantilever(body, load);
            let along = P3::direction(0.0, 0.0, 1.0);
            let plane = (in_space(body.centroid) & along) & load;
            let tip = centres(body, &along_principal)
                .last()
                .expect("the tip")
                .unitized();
            assert!((tip & plane).s().abs() < 1e-12, "{}", (tip & plane).s());
        }
    }

    /// numga's `test_kern_and_reciprocity`, with the exact kern: a pull through the centroid
    /// does not bend; the push inside the kern squeezes every fibre, the one outside stretches
    /// some; and the first push point is in the kern, the second not.
    #[test]
    fn kern_and_reciprocity() {
        let scene = scene();
        for (body, kern) in scene.bodies.iter().zip(&scene.kerns) {
            assert!(pull(body.stiffness, body.centroid).norm() < 1e-12);
            let [(p0, push0), (p1, push1)] = pushes(body);
            assert!(super::corner_stresses(body, push0).all(|s| s < 0.0));
            assert!(super::corner_stresses(body, push1).any(|s| s > 0.0));
            let contains = |p: P2| {
                let n = kern.len();
                (0..n).all(|k| orientation(kern[k], kern[(k + 1) % n], p) >= 0.0)
            };
            assert!(contains(p0) && !contains(p1));
        }
    }

    /// The checks of numga's scenario: every tip under the downward load moves down; the
    /// I-section's and the slotted tube's, mirror-symmetric about the load, straight down; the
    /// turned I-section's and the Z-section's sideways as well, by more than half the most
    /// either moves down. For each section the push inside the kern squeezes every fibre and the
    /// one outside stretches some, and a pull through one push point stretches the fibre through
    /// the other as much as the exchanged pull.
    #[test]
    fn scenes_pass_their_checks() {
        let tips: Vec<[f64; 2]> = bodies()
            .iter()
            .map(|body| deflection(body, &cantilever(body, downward())))
            .collect();
        let [sideways, down]: [Vec<f64>; 2] = [0, 1].map(|i| tips.iter().map(|d| d[i]).collect());
        assert!(down.iter().all(|d| *d < 0.0));
        assert!(sideways[0].abs() < 1e-12 && sideways[3].abs() < 1e-12);
        let most_down = down[1].abs().max(down[2].abs());
        assert!(sideways[1].abs().min(sideways[2].abs()) > 0.5 * most_down);
        for body in &bodies() {
            let [(p0, push0), (p1, push1)] = pushes(body);
            assert!(super::corner_stresses(body, push0).all(|s| s < 0.0));
            assert!(super::corner_stresses(body, push1).any(|s| s > 0.0));
            let (first, second) = (pull(body.stiffness, p0), pull(body.stiffness, p1));
            assert!(((first & p1).s() - (second & p0).s()).abs() < 1e-12);
        }
    }

    /// numga's `test_scenes_draw_and_opposite_loads_bend_oppositely` (the drawing is
    /// `a_frame_draws`): the Z-section's tip, under a load turning once around in numga's 48
    /// frames, moves opposite under opposite loads.
    #[test]
    fn opposite_loads_bend_oppositely() {
        let z = &bodies()[2];
        let frames = 48;
        let tip = |k: usize| {
            let angle = core::f64::consts::TAU * k as f64 / frames as f64;
            let [x, y] = deflection(z, &cantilever(z, turned_load(angle)));
            gax::vga2d::Vector::new(x, y)
        };
        for k in 0..frames / 2 {
            let (a, b) = (tip(k), tip(k + frames / 2));
            assert!((a + b).norm() / a.norm() < 2e-2);
        }
    }

    /// The triangulation is numga's: as many triangles, none clockwise (the last three corners
    /// left by the ear clipping may lie on a line, as in numga's); and the tips under the
    /// downward load are numga's to rounding.
    #[test]
    fn meshes_and_tips_match_numga() {
        let numga = [
            (10, [0.0, -0.5478228449592]),
            (10, [-1.718035701117, -1.53250704122]),
            (6, [1.563097266154, -1.280450246567]),
            (96, [0.0, -0.5586844827348]),
        ];
        for (body, (count, tip)) in bodies().iter().zip(numga) {
            let triangles = &body.section.triangles;
            assert_eq!(triangles.len(), count);
            assert!(
                triangles
                    .iter()
                    .all(|[a, b, c]| orientation(*a, *b, *c) > -1e-15)
            );
            let [x, y] = deflection(body, &cantilever(body, downward()));
            assert!(
                (x - tip[0]).abs() < 1e-10 && (y - tip[1]).abs() < 1e-10,
                "{x} {y}"
            );
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scene(), 0.5);
    }
}
