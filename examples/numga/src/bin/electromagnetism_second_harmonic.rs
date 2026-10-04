//! numga's `electromagnetism/second_harmonic`: frequency doubling in a tetrahedral crystal, in
//! spacetime algebra. Electric fields are bivectors `t ^ e`, and the crystal's nonlinear response
//! is a bilinear map `Bivector <- (Bivector, Bivector)`: each of four directed bonds `b` answers
//! the product of the two fields along it, `Σ b (b · E₁)(b · E₂)`. A pump `E cos ωt` drives
//! `E E cos² ωt`, half static and half at twice the frequency. Turning the crystal conjugates
//! the map by a rotor, binding the pump into one slot leaves the linear map that mixes a weak
//! probe, and the doubled light grows along the crystal as a sum of slices with their temporal
//! phase: steadily when matched, not at all when the slip is whole turns, and quasi-matched when
//! the crystal flips every half turn of slip.
//!
//! The animation turns the crystal once about the beam (numga's animation: the bonds, the
//! doubled polarization for every pump direction, and its power), while below a time cursor
//! runs over the waveform, a probe circles a fixed pump, and the light grows along three
//! crystals.

use gax::pga2d::Point;

use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Dir3, Light, Marker, ORIGIN3, Point2, Rect, Scene3,
    backdrop, caption, palette, run,
};

mod shg {
    // numga's API in full: the tests use parts the animation does not.
    #![cfg_attr(not(test), allow(dead_code))]

    use gax::sta::{Bivector, Even, Vector};

    pub type B = Bivector<(), f64>;
    /// The crystal's response: two electric fields in, a polarization source out.
    pub type Response = Bivector<(Bivector, Bivector), f64>;
    /// A linear map on fields.
    pub type Linear = Bivector<(Bivector,), f64>;
    pub type Rotor = gax::Unit<Even<(), f64>>;

    /// A field from numga's blade coefficients `tx, ty, tz, yz, zx, xy` (numga's `tx = t ^ x`
    /// is gax's `-e10`).
    pub fn field(tx: f64, ty: f64, tz: f64, yz: f64, zx: f64, xy: f64) -> B {
        Bivector::new(-tx, -ty, -tz, yz, zx, xy)
    }

    /// The observer's time direction.
    pub fn t() -> Vector<(), f64> {
        Vector::new(1.0, 0.0, 0.0, 0.0)
    }

    /// The open field.
    pub fn open() -> Linear {
        Bivector::slot()
    }

    /// Electric planes across a beam along a face diagonal of the crystal: horizontal,
    /// vertical, and along the beam.
    pub fn horizontal() -> B {
        let h = core::f64::consts::FRAC_1_SQRT_2;
        field(-h, h, 0.0, 0.0, 0.0, 0.0)
    }
    pub fn vertical() -> B {
        field(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    }
    pub fn longitudinal() -> B {
        let h = core::f64::consts::FRAC_1_SQRT_2;
        field(h, h, 0.0, 0.0, 0.0, 0.0)
    }

    /// The projector on electric planes, `(F - t F t) / 2` (the time reflection keeps the
    /// magnetic planes and negates the electric ones).
    pub fn electric() -> Linear {
        (open() - (t() >> open())).gp(0.5)
    }

    /// The projector on electric planes across the beam.
    pub fn transverse() -> Linear {
        electric() - longitudinal() * (longitudinal() | open())
    }

    /// Four directed bond planes, `t ^ b` for the bonds to a tetrahedron's corners.
    pub fn bonds() -> [B; 4] {
        let s = 1.0 / 3f64.sqrt();
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
        .map(|[a, b, c]| field(a * s, b * s, c * s, 0.0, 0.0, 0.0))
    }

    /// The response coefficient, chosen so that the six permutations of three distinct axes
    /// have a unit coefficient.
    pub const STRENGTH: f64 = 1.299_038_105_676_658; // 3 √3 / 4

    /// Each bond responds to the product of the two electric fields along it.
    pub fn response(bonds: &[B]) -> Response {
        bonds.iter().fold(Response::zero(), |acc, b| {
            acc + (*b * (*b | open()) * (*b | open())).gp(STRENGTH)
        })
    }

    /// Pull both fields into the crystal and turn its response forward:
    /// `R crystal(~R E₁ R, ~R E₂ R) ~R`. Filling the second slot goes through `at::<1>`, which
    /// brings it to the front, and `swap` restores the order.
    pub fn turned(crystal: Response, rotation: Rotor) -> Response {
        let pulled = rotation << open();
        rotation >> crystal.of(pulled).at::<1>().of(pulled).swap()
    }

    /// The transverse polarization source at twice the pump frequency, `½ P(χ(E, E))`.
    pub fn doubled(crystal: Response, pump: B) -> B {
        transverse().of(crystal.of(pump).of(pump)).gp(0.5)
    }

    /// The source fields accumulated slice by slice with their temporal phase: after each
    /// slice, the in-phase and quadrature amplitudes.
    pub fn growth(source: B, orientation: &[f64], mismatch: f64, depths: &[f64]) -> Vec<[B; 2]> {
        let thickness = depths[1] - depths[0];
        let mut sum = [B::zero(); 2];
        depths
            .iter()
            .zip(orientation)
            .map(|(d, o)| {
                let phase = mismatch * d;
                sum[0] += source.gp(phase.cos() * o * thickness);
                sum[1] += source.gp(phase.sin() * o * thickness);
                sum
            })
            .collect()
    }

    /// The generated power of the two quadratures.
    pub fn power(a: [B; 2]) -> f64 {
        a[0].dot(a[0]).s() + a[1].dot(a[1]).s()
    }

    /// The rotor turning the crystal about the beam by `angle`, in the screen's plane.
    pub fn about_beam(angle: f64) -> Rotor {
        (horizontal().commutator(vertical()) * (-angle / 2.0)).exp()
    }

    // --- scenes -------------------------------------------------------------------------

    /// Two pump periods: the phase, the pump and the doubled-frequency polarization it drives
    /// across the beam, its static half removed.
    pub fn waveform(samples: usize) -> (Vec<f64>, Vec<B>, Vec<B>) {
        let crystal = response(&bonds());
        let phase: Vec<f64> = (0..samples)
            .map(|i| 4.0 * core::f64::consts::PI * i as f64 / (samples - 1) as f64)
            .collect();
        let pump: Vec<B> = phase.iter().map(|p| horizontal().gp(p.cos())).collect();
        // The square of the cosine is half static and half at twice the frequency.
        let static_half = doubled(crystal, horizontal());
        let harmonic = pump
            .iter()
            .map(|p| transverse().of(crystal.of(*p).of(*p)) - static_half)
            .collect();
        (phase, pump, harmonic)
    }

    /// Pump polarizations all around the beam, from horizontal.
    pub fn pumps(samples: usize) -> Vec<B> {
        (0..samples)
            .map(|i| {
                let heading = core::f64::consts::TAU * i as f64 / (samples - 1) as f64;
                about_beam(heading) >> horizontal()
            })
            .collect()
    }

    /// The crystal turned about the beam by `angle`: its bonds and the doubled-frequency
    /// polarization each pump drives.
    pub fn turning(angle: f64, pumps: &[B]) -> ([B; 4], Vec<B>) {
        let rotation = about_beam(angle);
        let crystal = turned(response(&bonds()), rotation);
        (
            bonds().map(|b| rotation >> b),
            pumps.iter().map(|p| doubled(crystal, *p)).collect(),
        )
    }

    /// Two fixed pumps, a circle of weak probes, and the doubled-frequency polarization the
    /// linear map each pump leaves makes of the probes.
    pub fn mixing(samples: usize) -> ([B; 2], Vec<B>, [Vec<B>; 2]) {
        let crystal = response(&bonds());
        let pumps = [
            horizontal(),
            (horizontal() + vertical()).gp(core::f64::consts::FRAC_1_SQRT_2),
        ];
        let probes: Vec<B> = self::pumps(samples).iter().map(|p| p.gp(0.2)).collect();
        // Binding a pump into one input leaves a linear map on the other.
        let generated = pumps.map(|pump| {
            let probe_map: Linear = transverse().of(crystal.of(pump));
            probes.iter().map(|q| probe_map.of(*q)).collect()
        });
        (pumps, probes, generated)
    }

    /// The light grown along a matched crystal, a mismatched one, and a mismatched one flipped
    /// every time the slip reaches half a turn: the slice depths and the amplitudes.
    pub fn phase_matching(slices: usize) -> (Vec<f64>, [Vec<[B; 2]>; 3]) {
        let depths: Vec<f64> = (0..slices)
            .map(|i| (i as f64 + 0.5) / slices as f64)
            .collect();
        // The slip over the whole crystal: none, and two full turns.
        let turns = 4.0 * core::f64::consts::PI;
        // Inverting the bonds reverses the response, so a flipped slice adds with the opposite
        // sign; the slip reaches half a turn every quarter of the crystal.
        let flips: Vec<f64> = depths
            .iter()
            .map(|d| {
                if (d * 4.0).floor() as i64 % 2 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        let ones = vec![1.0; slices];
        let amplitude = [
            growth(vertical(), &ones, 0.0, &depths),
            growth(vertical(), &ones, turns, &depths),
            growth(vertical(), &flips, turns, &depths),
        ];
        (depths, amplitude)
    }

    /// Where a field reaches on the observer's screen across the beam: its horizontal and
    /// vertical components, drawn from the screen's centre.
    pub fn screen(f: B) -> gax::pga2d::Point<(), f64> {
        gax::pga2d::Point::xy((f | horizontal()).s(), (f | vertical()).s())
    }

    /// An electric plane's direction in space, `t · b`, as a vector of space.
    pub fn direction(b: B) -> gax::vga3d::Vector<(), f64> {
        let v = t() | b;
        gax::vga3d::Vector::new(v.e1(), v.e2(), v.e3())
    }
}

use shg::*;

/// A spot on the screen across the beam.
type Spot = gax::pga2d::Point<(), f64>;

/// The screen's centre, where the beam passes.
fn centre() -> Spot {
    Spot::xy(0.0, 0.0)
}

const SECONDS: f32 = 9.0;
/// Pump directions sampled around the beam.
const HEADINGS: usize = 241;

/// Precomputed scenes that do not change while the crystal turns.
struct Still {
    pumps: Vec<B>,
    waveform: (Vec<f64>, Vec<B>, Vec<B>),
    mixing: ([B; 2], Vec<B>, [Vec<B>; 2]),
    growth: (Vec<f64>, [Vec<[B; 2]>; 3]),
}

fn still() -> &'static Still {
    static S: std::sync::OnceLock<Still> = std::sync::OnceLock::new();
    S.get_or_init(|| Still {
        pumps: pumps(HEADINGS),
        waveform: waveform(401),
        mixing: mixing(HEADINGS),
        growth: phase_matching(256),
    })
}

/// A panel's title, centred under its top edge.
fn title(c: &mut Canvas, rect: Rect, s: &str) {
    let top_middle = rect.top_middle();
    let at = top_middle + Point2::direction(0.0, 12.0);
    c.text(s, at, 11.0, palette::ink(), Align::Center);
}

/// The crystal's bonds in space, the beam along the face diagonal and the screen's axes.
fn draw_bonds(c: &mut Canvas, rect: Rect, bonds: &[B; 4], spin: f32) {
    let view = rect.inset(0.0, 16.0, 0.0, 0.0);
    let cam = Camera::parallel(view, rect.height() * 0.32, -0.98 + spin, 0.40);
    let mut sc = Scene3::new(cam);
    // The point an electric plane's direction reaches from the crystal's centre.
    let tip = |b: B| ORIGIN3 + direction(b).dir3();
    let beam = direction(longitudinal()).dir3();
    sc.seg(
        ORIGIN3 - beam.gp(1.4),
        ORIGIN3 + beam.gp(1.4),
        1.0,
        palette::grid(),
    );
    sc.arrow(ORIGIN3 + beam, beam.gp(0.5), 1.5, 8.0, palette::yellow());
    for axis in [horizontal(), vertical()] {
        sc.arrow(ORIGIN3, direction(axis), 1.0, 6.0, palette::grid());
    }
    let tips = bonds.map(tip);
    for (i, a) in tips.iter().enumerate() {
        sc.seg(ORIGIN3, *a, 2.5, palette::ink().faded(0.8));
        for b in &tips[i + 1..] {
            sc.seg(*a, *b, 1.0, palette::sky().faded(0.35));
        }
        sc.dot(*a, Marker::Dot, 9.0, palette::orange());
    }
    sc.dot(ORIGIN3, Marker::Dot, 7.0, palette::grid());
    sc.draw(c);
    title(c, rect, "CRYSTAL BONDS");
    let corner = rect.hi - Point2::direction(34.0, 10.0);
    c.text("BEAM", corner, 10.0, palette::yellow(), Align::Center);
}

/// The screen: pump directions all around and the doubled polarization each drives.
fn draw_polarization(c: &mut Canvas, rect: Rect, pumps: &[B], harmonic: &[B]) {
    let ax = Axes::equal(rect.inset(34.0, 30.0, 10.0, 30.0), centre(), 1.15);
    ax.frame(c, "TRANSVERSE POLARIZATION", "HORIZONTAL", "");
    let faint = palette::grid().faded(0.6);
    ax.line(c, Spot::xy(-1.15, 0.0), Spot::xy(1.15, 0.0), 1.0, faint);
    ax.line(c, Spot::xy(0.0, -1.15), Spot::xy(0.0, 1.15), 1.0, faint);
    let circle: Vec<Spot> = pumps.iter().map(|p| screen(*p)).collect();
    ax.polyline(c, &circle, 1.0, palette::sky().faded(0.35));
    let curve: Vec<Spot> = harmonic.iter().map(|p| screen(*p)).collect();
    ax.polyline(c, &curve, 2.0, palette::orange().faded(0.8));
    ax.arrow(c, centre(), circle[0], 2.0, 8.0, palette::sky());
    ax.arrow(c, centre(), curve[0], 2.0, 8.0, palette::orange());
}

/// A polar plot of the power over the pump's direction. (numga uses matplotlib's polar
/// axes; this draws the rings, the spokes and the curve itself.) The pumps go once around the
/// beam at unit strength, so scaled they draw the rings, and every eighth of the way a spoke.
fn draw_power(c: &mut Canvas, rect: Rect, pumps: &[B], harmonic: &[B]) {
    let intensity: Vec<f64> = harmonic.iter().map(|h| h.dot(*h).s()).collect();
    let top = intensity.iter().cloned().fold(0.0, f64::max).max(1e-9) * 1.12;
    let ax = Axes::equal(rect.inset(10.0, 30.0, 10.0, 30.0), centre(), 1.0);
    title(c, rect, "POWER BY PUMP DIRECTION");
    let faint = palette::grid().faded(0.8);
    for k in 1..=4 {
        let ring: Vec<Spot> = pumps.iter().map(|p| screen(p.gp(k as f64 / 4.0))).collect();
        ax.polyline(c, &ring, 1.0, faint);
    }
    let below = gax::pga2d::Motor::translation(0.0, -0.04);
    for (k, p) in pumps.iter().step_by(pumps.len() / 8).take(8).enumerate() {
        ax.line(c, centre(), screen(*p), 1.0, faint);
        ax.text(
            c,
            below >> screen(p.gp(1.1)),
            &format!("{}", 45 * k),
            9.0,
            palette::grid(),
            Align::Center,
        );
    }
    let pts: Vec<Spot> = pumps
        .iter()
        .zip(&intensity)
        .map(|(p, i)| screen(p.gp(i / top)))
        .collect();
    // The shared fill tests every edge at 16 samples per pixel, so the shading takes every
    // fourth point of the curve.
    let coarse: Vec<Spot> = pts.iter().step_by(4).copied().collect();
    ax.fill(c, &coarse, palette::orange(), 0.12);
    ax.polyline(c, &pts, 2.0, palette::orange());
    ax.scatter(c, &pts[..1], Marker::Dot, 9.0, palette::sky());
    c.text(
        &format!("MAX {:.2}", top / 1.12),
        rect.bottom_left() + Point2::direction(12.0, -10.0),
        10.0,
        palette::grid(),
        Align::Left,
    );
}

fn draw_waveform(c: &mut Canvas, rect: Rect, at: f32) {
    let (phase, pump, harmonic) = &still().waveform;
    let ax = Axes::new(rect.inset(34.0, 30.0, 10.0, 34.0), [0.0, 2.0], [-1.2, 1.2]);
    ax.frame(c, "WAVEFORM", "PUMP PERIODS", "");
    let periods = |p: f64| p / core::f64::consts::TAU;
    let series = |v: &[B], pick: B| -> Vec<Point<(), f64>> {
        phase
            .iter()
            .zip(v)
            .map(|(p, f)| Point::xy(periods(*p), (*f | pick).s()))
            .collect()
    };
    let a = series(pump, horizontal());
    let b = series(harmonic, vertical());
    let zero = (Point::xy(0.0, 0.0), Point::xy(2.0, 0.0));
    ax.line(c, zero.0, zero.1, 1.0, palette::grid().faded(0.6));
    ax.polyline(c, &a, 2.0, palette::sky());
    ax.polyline(c, &b, 2.0, palette::orange());
    let k = ((at * (a.len() - 1) as f32) as usize).min(a.len() - 1);
    let now = periods(phase[k]);
    let cursor = palette::ink().faded(0.5);
    ax.line(c, Point::xy(now, -1.2), Point::xy(now, 1.2), 1.0, cursor);
    ax.scatter(c, &[a[k]], Marker::Dot, 8.0, palette::sky());
    ax.scatter(c, &[b[k]], Marker::Dot, 8.0, palette::orange());
}

fn draw_mixing(c: &mut Canvas, rect: Rect, at: f32) {
    let (pumps, probes, generated) = &still().mixing;
    let ax = Axes::equal(rect.inset(30.0, 30.0, 10.0, 34.0), centre(), 0.5);
    ax.frame(c, "PUMP AT 45: MIXING", "HORIZONTAL", "");
    ax.arrow(
        c,
        centre(),
        screen(pumps[1].gp(0.45)),
        3.0,
        8.0,
        palette::grid(),
    );
    let on_screen = |fields: &[B]| -> Vec<Spot> { fields.iter().map(|p| screen(*p)).collect() };
    let circle = on_screen(probes);
    let image = on_screen(&generated[1]);
    ax.dashed(c, &circle, 1.5, 6.0, palette::sky());
    let unmixed = on_screen(&generated[0]);
    ax.polyline(c, &unmixed, 1.0, palette::orange().faded(0.3));
    ax.polyline(c, &image, 2.0, palette::orange());
    let k = ((at * (circle.len() - 1) as f32) as usize).min(circle.len() - 1);
    ax.arrow(c, centre(), circle[k], 1.5, 6.0, palette::sky());
    ax.arrow(c, centre(), image[k], 1.5, 6.0, palette::orange());
}

const CASES: [&str; 3] = ["MATCHED", "MISMATCHED", "FLIPPED"];

fn case_colour(k: usize) -> Light {
    [palette::sky(), palette::orange(), palette::purple()][k]
}

fn draw_growth(c: &mut Canvas, phasor: Rect, power_rect: Rect, at: f32) {
    let (depths, amplitude) = &still().growth;
    let n = ((at * depths.len() as f32) as usize).clamp(1, depths.len());
    let ax = Axes::equal(
        phasor.inset(30.0, 30.0, 10.0, 34.0),
        Point::xy(0.45, 0.2),
        0.62,
    );
    ax.frame(c, "ACCUMULATED FIELD", "IN PHASE", "");
    for (k, path) in amplitude.iter().enumerate() {
        let pts: Vec<Point<(), f64>> = path[..n]
            .iter()
            .map(|a| Point::xy((a[0] | vertical()).s(), (a[1] | vertical()).s()))
            .collect();
        ax.polyline(c, &pts, 2.0, case_colour(k));
        ax.scatter(c, &pts[n - 1..], Marker::Dot, 7.0, case_colour(k));
    }
    let ax = Axes::new(
        power_rect.inset(34.0, 30.0, 10.0, 34.0),
        [0.0, 1.0],
        [0.0, 1.05],
    );
    ax.frame(c, "GROWTH", "DEPTH", "");
    for (k, path) in amplitude.iter().enumerate() {
        let pts: Vec<Point<(), f64>> = depths
            .iter()
            .zip(path)
            .map(|(d, a)| Point::xy(*d, power(*a)))
            .collect();
        ax.polyline(c, &pts, 1.0, (case_colour(k)).faded(0.3));
        ax.polyline(c, &pts[..n], 2.0, case_colour(k));
    }
    ax.legend(
        c,
        &[
            (CASES[0], case_colour(0)),
            (CASES[1], case_colour(1)),
            (CASES[2], case_colour(2)),
        ],
    );
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let h = screen.height();
    let at = (t / SECONDS).rem_euclid(1.0);
    let s = still();
    let angle = core::f64::consts::TAU * f64::from(at);
    let (bonds, harmonic) = turning(angle, &s.pumps);
    // Below the caption, a row of three panels over a row of four.
    let top = h * 0.11;
    let mid = top + (h - top) * 0.52;
    let upper = screen.inset(0.0, top, 0.0, h - mid);
    let lower = screen.inset(0.0, mid, 0.0, 0.0);
    let spin = 0.15 * (core::f32::consts::TAU * at).sin();
    draw_bonds(c, upper.column(0, 3), &bonds, spin);
    draw_polarization(c, upper.column(1, 3), &s.pumps, &harmonic);
    draw_power(c, upper.column(2, 3), &s.pumps, &harmonic);
    draw_waveform(c, lower.column(0, 4), at);
    draw_mixing(c, lower.column(1, 4), at);
    draw_growth(c, lower.column(2, 4), lower.column(3, 4), at);
    caption(
        c,
        "SECOND HARMONIC: A CRYSTAL THAT DOUBLES THE FREQUENCY",
        &format!(
            "THE BILINEAR RESPONSE TURNED ABOUT THE BEAM BY {:.0} DEGREES (STA)",
            angle.to_degrees()
        ),
    );
}

fn main() {
    run(Anim::new("second harmonic", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::shg::*;

    use gax::ApproxEq;
    use gax_numga_examples::rng::{Draw, Rng, rng};

    /// A field with coefficients uniform in `[-1, 1)`.
    fn bivector(r: &mut Rng) -> B {
        B::from_coeffs(core::array::from_fn(|_| r.range(-1.0, 1.0)))
    }

    fn near(a: B, b: B, tol: f64) -> bool {
        a.max_abs_diff(&b) <= tol
    }

    #[test]
    fn response_has_the_tetrahedral_component_law_bound_at_once_or_one_slot_at_a_time() {
        let crystal = response(&bonds());
        let mut rng = rng(814);
        let mut unit = || rng.range(-1.0, 1.0);
        for _ in 0..19 {
            let a = [unit(), unit(), unit()];
            let b = [unit(), unit(), unit()];
            let first = field(a[0], a[1], a[2], 0.0, 0.0, 0.0);
            let second = field(b[0], b[1], b[2], 0.0, 0.0, 0.0);
            // The six permutations of three distinct axes share one unit coefficient.
            let expected = field(
                a[1] * b[2] + a[2] * b[1],
                a[2] * b[0] + a[0] * b[2],
                a[0] * b[1] + a[1] * b[0],
                0.0,
                0.0,
                0.0,
            );
            assert!(near(crystal.of(first).of(second), expected, 1e-12));
            let partial: Linear = crystal.of(first);
            assert!(near(partial.of(second), expected, 1e-12));
            // Magnetic parts do not drive the electric-dipole response.
            let m1 = field(0.0, 0.0, 0.0, unit(), unit(), unit());
            let m2 = field(0.0, 0.0, 0.0, unit(), unit(), unit());
            assert!(near(
                crystal.of(first + m1).of(second + m2),
                expected,
                1e-12
            ));
        }
    }

    #[test]
    fn the_response_turns_with_the_crystal_and_reverses_with_its_bonds() {
        let crystal = response(&bonds());
        let mut rng = rng(177);
        let rotation = B::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.37).exp()
            * B::new(0.0, 0.0, 0.0, 0.23, 0.0, 0.0).exp();
        let turned = turned(crystal, rotation);
        let rebuilt = response(&bonds().map(|b| rotation >> b));
        let reversed = response(&bonds().map(|b| -b));
        for _ in 0..23 {
            let (f, p) = (bivector(&mut rng), bivector(&mut rng));
            assert!(near(
                turned.of(rotation >> f).of(rotation >> p),
                rotation >> crystal.of(f).of(p),
                1e-8
            ));
            assert!(near(turned.of(f).of(p), rebuilt.of(f).of(p), 1e-12));
            assert!(near(reversed.of(f).of(p), -crystal.of(f).of(p), 2e-12));
        }
    }

    #[test]
    fn cut_110_has_the_analytic_polarization_and_power() {
        let crystal = response(&bonds());
        let n = 129;
        for (i, pump) in pumps(n).iter().enumerate() {
            let heading = core::f64::consts::TAU * i as f64 / (n - 1) as f64;
            let (across, up) = (heading.cos(), heading.sin());
            let harmonic = doubled(crystal, *pump);
            let expected = -horizontal().gp(across * up) - vertical().gp(across * across / 2.0);
            assert!(near(harmonic, expected, 1e-6));
            assert!(
                (harmonic.dot(harmonic).s()
                    - across * across * (4.0 - 3.0 * across * across) / 4.0)
                    .abs()
                    < 1e-6
            );
        }
    }

    #[test]
    fn pump_bound_response_is_the_mixed_part_of_the_doubled_field() {
        let crystal = response(&bonds());
        let (pumps, probes, generated) = mixing(49);
        for (pump, images) in pumps.iter().zip(&generated) {
            for (q, g) in probes.iter().zip(images) {
                let change =
                    doubled(crystal, *pump + *q) - doubled(crystal, *pump) - doubled(crystal, *q);
                assert!(near(change, *g, 1e-12));
            }
        }
    }

    #[test]
    fn uniform_growth_follows_the_closed_form() {
        let slices = 512;
        let depths: Vec<f64> = (0..slices)
            .map(|i| (i as f64 + 0.5) / slices as f64)
            .collect();
        let source = (horizontal() + vertical()).gp(core::f64::consts::FRAC_1_SQRT_2);
        let ones = vec![1.0; slices];
        for k in [0.0, core::f64::consts::TAU, 2.0 * core::f64::consts::TAU] {
            let amplitude = growth(source, &ones, k, &depths);
            for (d, a) in depths.iter().zip(&amplitude) {
                // The running sum of midpoint slices, against the integral up to each slice's
                // far edge: (exp(i k z) - 1) / (i k).
                let edge = d + 0.5 / slices as f64;
                let (re, im) = if k == 0.0 {
                    (edge, 0.0)
                } else {
                    ((k * edge).sin() / k, (1.0 - (k * edge).cos()) / k)
                };
                assert!(near(a[0], source.gp(re), 1e-4));
                assert!(near(a[1], source.gp(im), 1e-4));
            }
        }
    }

    #[test]
    fn phase_matching_builds_while_uniform_mismatched_light_cancels() {
        let (_, amplitude) = phase_matching(256);
        let expected = [1.0, 0.0, (2.0 / core::f64::consts::PI).powi(2)];
        for (path, e) in amplitude.iter().zip(expected) {
            assert!((power(path[path.len() - 1]) - e).abs() < 1e-4);
        }
    }

    #[test]
    fn a_frame_draws_and_the_crystal_turns() {
        let mut draw = super::draw;
        let anim = gax_numga_examples::Anim::new("t", super::SECONDS).size(320, 180);
        let a = gax_numga_examples::app::frame(&anim, 0.0, &mut draw);
        let b = gax_numga_examples::app::frame(&anim, 1.0, &mut draw);
        assert!(a.mean().luma() > 0.0);
        assert!(a.mean() != b.mean());
    }
}
