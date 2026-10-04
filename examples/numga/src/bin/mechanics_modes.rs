//! numga's `mechanics/modes`: the normal modes of a plate on springs, from stiffness and inertia
//! assembled with an open twist (PGA2D). A spring's normalized line is both its line of action
//! and a measurement: paired with a small rigid motion (a twist, a point in PGA2D) it gives the
//! spring's extension. Multiplying that by the line and the spring constant and summing gives the
//! stiffness, a map from twists to forques (lines); the inertia is the same kind of map, from the
//! momenta of the plate's mass points. Paired with another open twist both are energy forms, and
//! their generalized eigenvectors are the modes, as twists. The animation releases every mode
//! from rest at once: two vertical springs (a free slide, a bounce and a rocking), then the same
//! with an off-centre angled spring that couples them. Springs are orange when stretched and
//! blue when compressed (the linear change in length).

use gax::motions::{Motions, Pga2d};
use gax::pga2d::{Line, Motor, Point, Scalar};

use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, Point2, Rect, backdrop, caption, palette, run,
};
use std::sync::OnceLock;

mod modes {
    use super::*;

    pub type P = Point<(), f64>;
    pub type L = Line<(), f64>;
    /// A map from twists to forques: a stiffness or an inertia.
    pub type Response = Line<(Point,), f64>;
    /// A linear form on twists: a spring's extension.
    pub type Extension = Scalar<(Point,), f64>;
    /// A bilinear form on twists: an energy.
    pub type Energy = Scalar<(Point, Point), f64>;

    /// One suspension: the plate's corners, the springs' attachments and anchors, their
    /// constants, and the mass points with their masses.
    #[derive(Clone, Debug)]
    pub struct Suspension {
        pub body: Vec<P>,
        pub attachments: Vec<P>,
        pub anchors: Vec<P>,
        pub spring_constants: Vec<f64>,
        pub mass_points: Vec<P>,
        pub masses: Vec<f64>,
    }

    /// The modes of one suspension: frequencies in Hz, the modes as twists, and the
    /// displacements of the corners and attachments and the springs' extensions per mode.
    #[derive(Clone, Debug)]
    pub struct ModeCase {
        pub system: Suspension,
        pub frequencies: [f64; 3],
        pub modes: [P; 3],
        pub body_offsets: [Vec<P>; 3],
        pub attachment_offsets: [Vec<P>; 3],
        pub extensions: [Vec<f64>; 3],
    }

    /// A uniform 2 by 1 plate of mass 1 on the first `springs` springs of stiffness 6. The first
    /// two hang vertically; the third is angled and off-centre.
    pub fn suspension(springs: usize) -> Suspension {
        let points = |v: &[[f64; 2]]| v.iter().map(|p| Point::xy(p[0], p[1])).collect::<Vec<P>>();
        let body = points(&[[0.0, 0.5], [2.0, 0.5], [2.0, 1.5], [0.0, 1.5]]);
        let attachments = points(&[[0.2, 1.5], [1.8, 1.5], [2.0, 1.0]][..springs]);
        let anchors = points(&[[0.2, 2.55], [1.8, 2.55], [2.9, 1.85]][..springs]);
        // Tensor-product two-point Gauss quadrature on the uniform plate: the corners pulled
        // towards the centre by `1 / sqrt(3)`.
        let center = body.iter().fold(Point::zero(), |s: P, p| s + *p).gp(0.25);
        let mass_points = body
            .iter()
            .map(|p| center + (*p - center).gp(1.0 / 3f64.sqrt()))
            .collect();
        Suspension {
            body,
            attachments,
            anchors,
            spring_constants: vec![6.0; springs],
            mass_points,
            masses: vec![0.25; 4],
        }
    }

    /// The springs' lines of action, joining anchor to attachment, normalized.
    pub fn lines(s: &Suspension) -> Vec<L> {
        s.anchors
            .iter()
            .zip(&s.attachments)
            .map(|(a, b)| (*a & *b).normalized().into_inner())
            .collect()
    }

    /// The stiffness and each spring's extension form, from the springs' lines: pairing an open
    /// twist with a line measures the stretch, and Hooke's law scales the line by it.
    pub fn spring_stiffness(lines: &[L], constants: &[f64]) -> (Response, Vec<Extension>) {
        let extension: Vec<Extension> = lines.iter().map(|l| Point::slot() & *l).collect();
        let stiffness = lines
            .iter()
            .zip(&extension)
            .zip(constants)
            .fold(Line::zero(), |sum: Response, ((l, e), k)| {
                sum + (*l * *e).gp(*k)
            });
        (stiffness, extension)
    }

    /// The inertia from lumped masses: each point's velocity under an open twist is its
    /// commutator with it, and the point joined with its velocity is its momentum
    /// (`Motions::point_inertia`, per unit mass).
    pub fn body_inertia(points: &[P], masses: &[f64]) -> Response {
        points
            .iter()
            .zip(masses)
            .fold(Line::zero(), |sum: Response, (p, m)| {
                sum + Pga2d::point_inertia(*p).gp(*m)
            })
    }

    /// The generalized eigenproblem between the energy forms: the modes as twists (normalized
    /// to unit kinetic energy form) and the frequencies in Hz.
    pub fn normal_modes(stiffness: Response, inertia: Response) -> ([P; 3], [f64; 3]) {
        let pe: Energy = Point::slot() & stiffness;
        let ke: Energy = Point::slot() & inertia;
        let (values, modes) = pe.eigh_with(ke);
        let frequencies = values.map(|v| v.max(0.0).sqrt() / core::f64::consts::TAU);
        (modes, frequencies)
    }

    /// A suspension's modes and what they do to the plate and the springs.
    pub fn mode_case(system: Suspension) -> ModeCase {
        let (stiffness, extension) = spring_stiffness(&lines(&system), &system.spring_constants);
        let inertia = body_inertia(&system.mass_points, &system.masses);
        let (modes, frequencies) = normal_modes(stiffness, inertia);
        // The displacement of a point under a twist is their commutator (a direction).
        let offsets = |pts: &[P], m: P| pts.iter().map(|p| p.commutator(m)).collect::<Vec<P>>();
        ModeCase {
            body_offsets: modes.map(|m| offsets(&system.body, m)),
            attachment_offsets: modes.map(|m| offsets(&system.attachments, m)),
            extensions: modes.map(|m| extension.iter().map(|e| e.of(m).s()).collect()),
            frequencies,
            modes,
            system,
        }
    }

    /// Two vertical springs, then the same with the off-centre angled spring.
    pub fn suspensions() -> [ModeCase; 2] {
        [mode_case(suspension(2)), mode_case(suspension(3))]
    }
}

use modes::*;

fn cases() -> &'static [ModeCase; 2] {
    static C: OnceLock<[ModeCase; 2]> = OnceLock::new();
    C.get_or_init(suspensions)
}

/// The loop: twice the period of the slowest oscillating mode, as numga's animation.
fn seconds() -> f32 {
    let slowest = cases()
        .iter()
        .flat_map(|c| c.frequencies)
        .filter(|f| *f > 1e-6)
        .fold(f64::MAX, f64::min);
    (2.0 / slowest) as f32
}

/// A direction turned a quarter turn counterclockwise, scaled to unit length.
fn across(d: P) -> P {
    let quarter = Motor::rotation(Point::xy(0.0, 0.0), core::f64::consts::FRAC_PI_2);
    (quarter >> d).gp(1.0 / d.ideal_norm())
}

/// A coil with straight leads from `a` to `b`: points along the axis, offset across it.
fn spring_path(a: P, b: P) -> Vec<P> {
    let axis = b - a;
    let side = across(axis);
    let turns = 15;
    let mut along = vec![(0.0, 0.0), (0.16, 0.0)];
    for k in 0..turns {
        let s = 0.20 + 0.60 * k as f64 / (turns - 1) as f64;
        along.push((s, if k % 2 == 0 { 0.048 } else { -0.048 }));
    }
    along.extend([(0.84, 0.0), (1.0, 0.0)]);
    along
        .iter()
        .map(|&(s, q)| a + axis.gp(s) + side.gp(q))
        .collect()
}

/// A wall at the anchor, across the spring, hatched on the far side.
fn support(ax: &Axes, c: &mut Canvas, anchor: P, attachment: P) {
    let out = anchor - attachment;
    let out = out.gp(1.0 / out.ideal_norm());
    let side = across(out);
    let at = |s: f64| anchor + side.gp(s);
    ax.line(c, at(-0.14), at(0.14), 2.0, palette::grid());
    for k in 0..5 {
        let s = at(-0.12 + 0.06 * k as f64);
        ax.line(c, s, s + (out + side).gp(0.075), 1.0, palette::grid());
    }
}

fn spring_colour(extension: f64) -> Light {
    if extension > 1e-9 {
        palette::orange()
    } else if extension < -1e-9 {
        palette::sky()
    } else {
        palette::grid()
    }
}

/// One mode's panel at `phase` (the cosine of the oscillation).
fn panel(c: &mut Canvas, rect: Rect, case: &ModeCase, mode: usize, phase: f64, title: &str) {
    let ax = Axes::equal(rect, P::xy(1.35, 1.5), 1.4);
    let s = &case.system;
    // The mode's shape, enlarged so that the largest corner moves 0.2.
    let largest = case.body_offsets[mode]
        .iter()
        .map(|o| o.ideal_norm())
        .fold(0.0, f64::max);
    let k = 0.2 / largest * phase;
    let moved = |p: P, o: P| p + o.gp(k);
    let body: Vec<P> = s
        .body
        .iter()
        .zip(&case.body_offsets[mode])
        .map(|(p, o)| moved(*p, *o))
        .collect();
    let closed = |pts: &[P]| [pts, &pts[..1]].concat();
    ax.dashed(c, &closed(&s.body), 1.2, 4.0, palette::grid());
    // The plate's fill covers what is below, a dim blue: its outline glows.
    ax.fill(c, &body, palette::blue().faded(0.1), 0.55);
    ax.polyline(c, &closed(&body), 2.0, palette::sky());
    for (j, (anchor, attachment)) in s.anchors.iter().zip(&s.attachments).enumerate() {
        let b = moved(*attachment, case.attachment_offsets[mode][j]);
        support(&ax, c, *anchor, *attachment);
        let colour = spring_colour(case.extensions[mode][j] * k);
        ax.polyline(c, &spring_path(*anchor, b), 2.0, colour);
        ax.scatter(c, &[b], Marker::Dot, 7.0, palette::ink());
    }
    // The mode's centre of rotation. A translation's is at infinity (its weight is zero), so an
    // arrow from the plate's centre shows the direction it slides in instead.
    let m = case.modes[mode];
    if m.e12().abs() > 1e-9 {
        ax.scatter(c, &[m], Marker::Cross, 9.0, palette::yellow().faded(0.9));
    } else {
        let centre = Point::xy(1.0, 1.0);
        let slide = centre.commutator(m);
        ax.arrow(
            c,
            centre,
            centre + slide.gp(0.4 / slide.ideal_norm()),
            1.5,
            7.0,
            palette::yellow(),
        );
    }
    // The title just inside the lower left corner.
    let corner = ax.at(0.0, 0.0) + Point2::direction(0.05, 0.12);
    ax.text(c, corner, title, 11.0, palette::ink(), Align::Left);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let top = h * 0.12;
    let row_h = (h - top) / 2.0;
    for (r, (case, label)) in cases()
        .iter()
        .zip(["TWO SPRINGS", "THREE SPRINGS"])
        .enumerate()
    {
        // The row of the case's three modes, under the caption.
        let row = Rect::new(0.0, top + r as f32 * row_h, w, top + (r + 1) as f32 * row_h);
        for mode in 0..3 {
            let rect = row.column(mode, 3).inset(6.0, 6.0, 6.0, 6.0);
            let f = case.frequencies[mode];
            let phase = (core::f64::consts::TAU * f * f64::from(t)).cos();
            let title = format!("{label}: {:.3} HZ", f);
            panel(c, rect, case, mode, phase, &title);
        }
    }
    caption(
        c,
        "NORMAL MODES OF A PLATE ON SPRINGS",
        "STIFFNESS, INERTIA: TWISTS TO FORQUES. MODES: EIGENPAIRS OF BOTH (PGA2D)",
    );
}

fn main() {
    run(Anim::new("modes", seconds()).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::modes::*;
    use gax::pga2d::Point;

    /// numga's `test_scenario_has_analytic_modes_and_renders` and
    /// `test_two_springs_have_analytic_slide_bounce_and_rock_frequencies`: mass 1, polar
    /// inertia `(2² + 1²) / 12`, springs 0.8 either side of the centre.
    #[test]
    fn two_springs_have_analytic_slide_bounce_and_rock_frequencies() {
        let [free, restrained] = suspensions();
        let expected = [0.0, 2.0 * 6.0, 2.0 * 6.0 * 0.8f64.powi(2) / (5.0 / 12.0)];
        for (f, e) in free.frequencies.iter().zip(expected) {
            let w2 = (core::f64::consts::TAU * f).powi(2);
            assert!((w2 - e).abs() < 1e-12, "{w2} vs {e}");
        }
        assert!(restrained.frequencies.iter().all(|f| *f > 0.0));
    }

    /// The extension forms against lengths measured after exact finite motions, and the
    /// stiffness's energy against the springs' energy.
    #[test]
    fn spring_extension_and_energy_match_finite_rigid_displacements() {
        let s = suspension(3);
        let (stiffness, extension) = spring_stiffness(&lines(&s), &s.spring_constants);
        let q: P = Point::new(0.3, -0.4, 0.25);
        let h = 1e-4;
        // The distance between two (unit) points: the norm of their join.
        let dist = |a: P, b: P| (a & b).norm();
        let plus = q.gp(-h / 2.0).exp();
        let minus = q.gp(h / 2.0).exp();
        let mut actual = 0.0;
        for (j, (anchor, att)) in s.anchors.iter().zip(&s.attachments).enumerate() {
            let rest = dist(*att, *anchor);
            let ep = dist(plus >> *att, *anchor) - rest;
            let em = dist(minus >> *att, *anchor) - rest;
            let measured = (ep - em) / (2.0 * h);
            let predicted = extension[j].of(q).s();
            assert!(
                (measured - predicted).abs() < 2e-7,
                "{measured} {predicted}"
            );
            actual += s.spring_constants[j] * (ep * ep + em * em) / (4.0 * h * h);
        }
        let predicted = 0.5 * (q & stiffness.of(q)).s();
        assert!((actual - predicted).abs() < 2e-6 * predicted.abs());
    }

    /// The modes are orthonormal in the kinetic energy, diagonalize the elastic energy with the
    /// squared angular frequencies, and that energy is the springs' work.
    #[test]
    fn mass_normalized_modes_and_spring_work() {
        let s = suspension(3);
        let (stiffness, extension) = spring_stiffness(&lines(&s), &s.spring_constants);
        let inertia = body_inertia(&s.mass_points, &s.masses);
        let (modes, frequencies) = normal_modes(stiffness, inertia);
        for i in 0..3 {
            for j in 0..3 {
                let mass = (modes[i] & inertia.of(modes[j])).s();
                let elastic = (modes[i] & stiffness.of(modes[j])).s();
                let w2 = (core::f64::consts::TAU * frequencies[i]).powi(2);
                let (m, e) = if i == j { (1.0, w2) } else { (0.0, 0.0) };
                assert!((mass - m).abs() < 1e-12, "{i} {j} {mass}");
                assert!((elastic - e).abs() < 1e-12, "{i} {j} {elastic}");
                let work: f64 = extension
                    .iter()
                    .zip(&s.spring_constants)
                    .map(|(x, k)| x.of(modes[i]).s() * k * x.of(modes[j]).s())
                    .sum();
                assert!((work - elastic).abs() < 1e-12);
            }
        }
    }

    /// Moving the whole suspension leaves the frequencies unchanged.
    #[test]
    fn modes_do_not_depend_on_world_pose() {
        let s = suspension(3);
        // numga's `exp(-0.9 xw + 0.3 yw) exp(0.37 xy)`: with `w = e0`, `xw = -e01`, `yw = e20`.
        let m = Point::new(0.3, 0.9, 0.0).exp() * Point::new(0.0, 0.0, 0.37).exp();
        let lines = lines(&s);
        let moved_lines: Vec<L> = lines.iter().map(|l| m >> *l).collect();
        let moved_points: Vec<P> = s.mass_points.iter().map(|p| m >> *p).collect();
        let (k0, _) = spring_stiffness(&lines, &s.spring_constants);
        let (k1, _) = spring_stiffness(&moved_lines, &s.spring_constants);
        let (_, before) = normal_modes(k0, body_inertia(&s.mass_points, &s.masses));
        let (_, after) = normal_modes(k1, body_inertia(&moved_points, &s.masses));
        for (a, b) in after.iter().zip(before) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
