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

use gax::pga2d::{Line, Point, Scalar};
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Marker, Rgb, backdrop, canvas, caption, palette, plot, run,
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
        let xy = |v: &[[f64; 2]]| v.iter().map(|p| Point::xy(p[0], p[1])).collect::<Vec<P>>();
        let body = xy(&[[0.0, 0.5], [2.0, 0.5], [2.0, 1.5], [0.0, 1.5]]);
        let attachments = xy(&[[0.2, 1.5], [1.8, 1.5], [2.0, 1.0]][..springs]);
        let anchors = xy(&[[0.2, 2.55], [1.8, 2.55], [2.9, 1.85]][..springs]);
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
    /// commutator with it, and the point joined with its velocity is its momentum.
    pub fn body_inertia(points: &[P], masses: &[f64]) -> Response {
        points
            .iter()
            .zip(masses)
            .fold(Line::zero(), |sum: Response, (p, m)| {
                sum + (*p & p.commutator(Point::slot())).gp(*m)
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

fn xy(p: P) -> [f32; 2] {
    let [x, y] = p.to_euclidean();
    [x as f32, y as f32]
}

fn ideal(p: P) -> [f32; 2] {
    [p.e20() as f32, p.e01() as f32]
}

/// A coil with straight leads between two points.
fn spring_path(a: [f32; 2], b: [f32; 2]) -> Vec<[f32; 2]> {
    let axis = [b[0] - a[0], b[1] - a[1]];
    let len = (axis[0] * axis[0] + axis[1] * axis[1]).sqrt();
    let normal = [-axis[1] / len, axis[0] / len];
    let turns = 15;
    let mut along = vec![(0.0, 0.0), (0.16, 0.0)];
    for k in 0..turns {
        let s = 0.20 + 0.60 * k as f32 / (turns - 1) as f32;
        along.push((s, if k % 2 == 0 { 0.048 } else { -0.048 }));
    }
    along.extend([(0.84, 0.0), (1.0, 0.0)]);
    along
        .iter()
        .map(|(s, q)| {
            [
                a[0] + s * axis[0] + q * normal[0],
                a[1] + s * axis[1] + q * normal[1],
            ]
        })
        .collect()
}

/// A wall at the anchor, across the spring, hatched on the far side.
fn support(ax: &Axes, c: &mut Canvas, anchor: [f32; 2], attachment: [f32; 2]) {
    let d = [anchor[0] - attachment[0], anchor[1] - attachment[1]];
    let n = (d[0] * d[0] + d[1] * d[1]).sqrt();
    let dir = [d[0] / n, d[1] / n];
    let tan = [-dir[1], dir[0]];
    let at = |s: f32| [anchor[0] + s * tan[0], anchor[1] + s * tan[1]];
    ax.line(c, at(-0.14), at(0.14), 2.0, palette::grid(), 1.0);
    for k in 0..5 {
        let s = at(-0.12 + 0.06 * k as f32);
        let e = [
            s[0] + 0.075 * (dir[0] + tan[0]),
            s[1] + 0.075 * (dir[1] + tan[1]),
        ];
        ax.line(c, s, e, 1.0, palette::grid(), 1.0);
    }
}

fn spring_colour(extension: f64) -> Rgb {
    if extension > 1e-9 {
        palette::orange()
    } else if extension < -1e-9 {
        palette::sky()
    } else {
        palette::grid()
    }
}

/// One mode's panel at `phase` (the cosine of the oscillation).
fn panel(c: &mut Canvas, rect: [f32; 4], case: &ModeCase, mode: usize, phase: f64, title: &str) {
    let ax = Axes::equal(rect, [1.35, 1.5], 1.4);
    let s = &case.system;
    let reference: Vec<[f32; 2]> = s.body.iter().map(|p| xy(*p)).collect();
    // The mode's shape, enlarged so that the largest corner moves 0.2.
    let largest = case.body_offsets[mode]
        .iter()
        .map(|o| o.ideal_norm())
        .fold(0.0, f64::max);
    let k = 0.2 / largest * phase;
    let moved = |p: P, o: P| xy(p + o.gp(k));
    let body: Vec<[f32; 2]> = s
        .body
        .iter()
        .zip(&case.body_offsets[mode])
        .map(|(p, o)| moved(*p, *o))
        .collect();
    let mut closed = reference.clone();
    closed.push(reference[0]);
    ax.dashed(c, &closed, 1.2, 4.0, palette::grid(), 1.0);
    ax.fill(c, &body, canvas::scale(palette::blue(), 0.8), 0.55);
    let mut outline = body.clone();
    outline.push(body[0]);
    ax.polyline(c, &outline, 2.0, palette::sky(), 1.0);
    for (j, (anchor, attachment)) in s.anchors.iter().zip(&s.attachments).enumerate() {
        let a = xy(*anchor);
        let b = moved(*attachment, case.attachment_offsets[mode][j]);
        support(&ax, c, a, xy(*attachment));
        let colour = spring_colour(case.extensions[mode][j] * k);
        ax.polyline(c, &spring_path(a, b), 2.0, colour, 1.0);
        ax.scatter(c, &[b], Marker::Dot, 7.0, palette::ink(), 1.0);
    }
    // The mode's centre of rotation (a translation has none in view: its weight is zero).
    let m = case.modes[mode];
    if m.e12().abs() > 1e-9 {
        ax.scatter(c, &[xy(m)], Marker::Cross, 9.0, palette::yellow(), 0.9);
    } else {
        let d = ideal(m);
        let n = (d[0] * d[0] + d[1] * d[1]).sqrt();
        let centre = [1.0, 1.0];
        ax.arrow(
            c,
            centre,
            [centre[0] + 0.4 * d[0] / n, centre[1] + 0.4 * d[1] / n],
            1.5,
            7.0,
            palette::yellow(),
        );
    }
    ax.text(
        c,
        [ax.x[0] + 0.05, ax.y[0] + 0.12],
        title,
        11.0,
        palette::ink(),
        Align::Left,
    );
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.width as f32, c.height as f32);
    let top = h * 0.12;
    let row_h = (h - top) / 2.0;
    for (r, (case, label)) in cases()
        .iter()
        .zip(["TWO SPRINGS", "THREE SPRINGS"])
        .enumerate()
    {
        for mode in 0..3 {
            let rect = [
                w * mode as f32 / 3.0,
                top + r as f32 * row_h,
                w * (mode + 1) as f32 / 3.0,
                top + (r + 1) as f32 * row_h,
            ];
            let f = case.frequencies[mode];
            let phase = (core::f64::consts::TAU * f * f64::from(t)).cos();
            let title = format!("{label}: {:.3} HZ", f);
            panel(
                c,
                plot::inset(rect, 6.0, 6.0, 6.0, 6.0),
                case,
                mode,
                phase,
                &title,
            );
        }
    }
    caption(
        c,
        "NORMAL MODES OF A PLATE ON SPRINGS",
        "STIFFNESS AND INERTIA AS MAPS FROM TWISTS TO FORQUES; MODES BY THE GENERALIZED EIGENPROBLEM (PGA2D)",
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
        let dist = |a: P, b: P| {
            let [x, y] = a.to_euclidean();
            let [u, v] = b.to_euclidean();
            (x - u).hypot(y - v)
        };
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
        let mut draw = super::draw;
        let c = gax_numga_examples::app::frame(
            &gax_numga_examples::Anim::new("t", 1.0).size(320, 180),
            0.5,
            &mut draw,
        );
        assert!(c.mean()[0] > 0.0);
    }
}
