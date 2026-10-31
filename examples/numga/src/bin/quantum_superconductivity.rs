//! numga's `quantum/superconductivity`: a collisionless s-wave superconductor after a sudden
//! change of the pairing strength, as Anderson pseudospins in VGA3D, and the superconducting Higgs
//! mode it rings with. Each energy level `ξ` of a band holds a pair of orbitals of opposite momenta,
//! and the pair's state is a pseudospin, modelled like a spin as the intrinsic patch of the plane
//! it turns in (an intrinsic vector): its reading by the plane `xy`, `xy & spin`, is the occupation
//! less a half, its readings by the planes `yz` and `zx` the pair coherence. The levels are the
//! sites of a field.
//!
//! The field a pseudospin turns in is an extrinsic bivector: the level's energy in the plane
//! `xy`, less the gap in the planes `yz` and `zx`. The pairing interaction is a correlation from
//! pseudospins to fields, `Bivector <- intrinsic Vector`, which reads the mean coherence by
//! pairing (`yz & spin`, `zx & spin`) into the gap, and each spin precesses in its own field: its
//! rate is the commutator with `-2 field`. The energy pairs each field with its pseudospin,
//! `field & spin`. In the ground state every spin is square to its field's plane and set against
//! it, which is found by iterating to self-consistency. Turning every spin about `z` together
//! gives another ground state, the gap's phase.
//!
//! The implicit midpoint rule follows the spins after a quench of the pairing strength to 88%:
//! the gap oscillates and settles below its old value, near the new ground state's, while the
//! energy stays where it was. Linearized about the ground state, a disturbance changes both the
//! spin turned and the field turning it: the local precession, and the feedback of the gap. For a
//! weak quench the response predicts the gap with the feedback; without it (the gap held fixed)
//! it fails.
//!
//! The animation shows the levels' coherence as arrows across the energy axis, turning as the
//! gap rings, the gap against time, the ground state's occupation and coherence, and the weak
//! quench's gap with its two linear predictions.

use gax::vga3d::Vector;

use gax_numga_examples::scene3::panel3;
use gax_numga_examples::signal::{frame, wave};
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Light, Marker, ORIGIN2, ORIGIN3, Point2, Scene3,
    backdrop, caption, colormap, palette, reach3, run_with,
};

mod condensate {
    use gax::field::Field;
    use gax::vga3d::{Bivector, intrinsic};

    /// A pseudospin, the state of a pair: the intrinsic patch of the plane it turns in. The field
    /// it turns in is an extrinsic bivector.
    pub type Spin = intrinsic::Vector;
    pub type Plane = Bivector;
    /// The pairing, a correlation from pseudospins to fields, and a map on pseudospins.
    pub type Pairing = Bivector<(Spin,)>;
    pub type Map = intrinsic::Vector<(Spin,)>;
    /// A pseudospin, and a field, at every energy level.
    pub type Spins = Field<Spin>;
    pub type Fields = Field<Plane>;

    /// The planes `yz`, `zx` and `xy`: the coherence's two and the occupation's.
    pub fn yz() -> Plane {
        Bivector::new(1.0, 0.0, 0.0)
    }
    pub fn zx() -> Plane {
        Bivector::new(0.0, 1.0, 0.0)
    }
    pub fn xy() -> Plane {
        Bivector::new(0.0, 0.0, 1.0)
    }

    fn open() -> Map {
        Spin::slot()
    }

    /// The levels' energies in the plane `xy`, and the pairing: the correlation from the mean
    /// spin to the gap.
    pub struct Condensate {
        pub dispersion: Fields,
        pub pairing: Pairing,
    }

    impl Condensate {
        /// The gap, the pairing field all levels share: read from their mean pseudospin.
        pub fn gap(&self, spins: &Spins) -> Plane {
            self.pairing.of(spins.mean())
        }

        /// Each level's field: its energy less the gap.
        fn fields(&self, gap: Plane) -> Fields {
            self.dispersion.map(|d| d - gap)
        }

        /// The precession of every spin in its field, with the gap the spins make themselves.
        pub fn rate(&self, spins: &Spins) -> Spins {
            let fields = self.fields(self.gap(spins));
            fields.zip_with(spins, |f, s| (f * -2.0).commutator(s))
        }

        /// The ground state: every spin of length a half, square to its field's plane and set
        /// against it, iterated from a seed gap to self-consistency.
        pub fn equilibrium(&self, seed: Plane, iterations: usize) -> Spins {
            let against = |gap: Plane| self.fields(gap).map(|f| -f.hodge().normalized() * 0.5);
            let mut spins = against(seed);
            for _ in 1..iterations {
                spins = against(self.gap(&spins));
            }
            spins
        }

        /// The linear response to a disturbance about `spins`.
        pub fn response(&self, spins: &Spins) -> Response {
            let local = self
                .fields(self.gap(spins))
                .map(|f| (f * -2.0).commutator(open()));
            let turning = self.pairing * 2.0;
            let feedback = spins.map(|s| turning.commutator(s));
            Response { local, feedback }
        }

        /// The energy per pair of orbitals, each field paired with its pseudospin, the shared
        /// interaction counted once.
        pub fn energy(&self, spins: &Spins) -> f64 {
            let mean = spins.mean();
            let kinetic = self.dispersion.zip_with(spins, |d, s| d & s).mean().s();
            2.0 * kinetic - (self.pairing.of(mean) & mean).s()
        }
    }

    /// The linear response about a state: the maps that act on a disturbance at each level (the
    /// precession in the unchanged field) and on its mean (the changed gap turning the unchanged
    /// spins).
    pub struct Response {
        pub local: Field<Map>,
        pub feedback: Field<Map>,
    }

    impl Response {
        /// The rate of a disturbance, with the gap's feedback at `feedback` times its strength
        /// (one: the whole response; zero: the gap held fixed).
        pub fn rate(&self, delta: &Spins, feedback: f64) -> Spins {
            let mean = delta.mean();
            self.local.zip_with(delta, |m, d| m.of(d))
                + self.feedback.map(|m| m.of(mean) * feedback)
        }
    }

    /// The pairing strength times the coherence, read by pairing with the planes `yz` and `zx`,
    /// into the gap in those planes: the occupation does not pair.
    pub fn coherence_pairing(strength: f64) -> Pairing {
        (yz() * (yz() & open()) + zx() * (zx() & open())) * strength
    }

    /// The dispersion: `count` levels evenly spread between `-cutoff` and `cutoff`, at the
    /// midpoints of their shares (a constant density of states), in the plane `xy`.
    pub fn energy_levels(count: usize, cutoff: f64) -> Fields {
        Field::from_fn(count, |k| {
            xy() * (cutoff * (2.0 * (k as f64 + 0.5) / count as f64 - 1.0))
        })
    }

    /// One implicit midpoint step, the implicit equation solved by `iterations` of fixed-point
    /// iteration. At convergence it keeps the quadratic invariants: the spins' lengths and the
    /// energy.
    pub fn midpoint(
        state: &Spins,
        rate: &impl Fn(&Spins) -> Spins,
        dt: f64,
        iterations: usize,
    ) -> Spins {
        let mut end = state.clone();
        for _ in 0..iterations {
            let middle = (state + &end).scaled(0.5);
            end = state + &rate(&middle).scaled(dt);
        }
        end
    }

    /// The state and every `steps_per_frame`-th of `steps` midpoint steps after it.
    pub fn evolution(
        state: &Spins,
        rate: impl Fn(&Spins) -> Spins,
        dt: f64,
        steps: usize,
        steps_per_frame: usize,
        iterations: usize,
    ) -> Vec<Spins> {
        let mut out = vec![state.clone()];
        let mut current = state.clone();
        for _ in 0..steps / steps_per_frame {
            for _ in 0..steps_per_frame {
                current = midpoint(&current, &rate, dt, iterations);
            }
            out.push(current.clone());
        }
        out
    }

    /// The levels, the coupling and the quench, numga's.
    pub const LEVELS: usize = 240;
    pub const CUTOFF: f64 = 5.0;
    pub const COUPLING: f64 = 4.4;
    pub const QUENCH_RATIO: f64 = 0.88;
    pub const SMALL_QUENCH_RATIO: f64 = 0.99;
    const EQUILIBRIUM_ITERATIONS: usize = 160;
    pub const DT: f64 = 0.04;
    pub const STEPS: usize = 600;
    pub const STEPS_PER_FRAME: usize = 2;
    pub const MIDPOINT_ITERATIONS: usize = 3;

    /// A quench: the condensate after it, its old ground state (where the run starts), its new
    /// one, the times of the records and the spins at each.
    pub struct Quench {
        pub after: Condensate,
        pub initial: Spins,
        pub equilibrium: Spins,
        pub times: Vec<f64>,
        pub history: Vec<Spins>,
    }

    /// The ground state at the full coupling, followed after the coupling drops to `ratio`.
    pub fn quench(ratio: f64) -> Quench {
        let before = Condensate {
            dispersion: energy_levels(LEVELS, CUTOFF),
            pairing: coherence_pairing(COUPLING),
        };
        let after = Condensate {
            dispersion: before.dispersion.clone(),
            pairing: before.pairing * ratio,
        };
        let initial = before.equilibrium(yz(), EQUILIBRIUM_ITERATIONS);
        let equilibrium = after.equilibrium(yz(), EQUILIBRIUM_ITERATIONS);
        let history = evolution(
            &initial,
            |s| after.rate(s),
            DT,
            STEPS,
            STEPS_PER_FRAME,
            MIDPOINT_ITERATIONS,
        );
        let times = (0..history.len())
            .map(|k| (k * STEPS_PER_FRAME) as f64 * DT)
            .collect();
        Quench {
            after,
            initial,
            equilibrium,
            times,
            history,
        }
    }

    /// The weak quench's gap less its new ground state's at each record: from the nonlinear
    /// evolution, and predicted by the linear response about the new ground state, with the
    /// gap's feedback and with the gap held fixed.
    pub struct WeakQuench {
        pub exact: Vec<Plane>,
        pub predicted: [Vec<Plane>; 2],
    }

    pub fn weak_quench() -> WeakQuench {
        let weak = quench(SMALL_QUENCH_RATIO);
        let model = &weak.after;
        let settled = model.gap(&weak.equilibrium);
        let response = model.response(&weak.equilibrium);
        let disturbance = &weak.initial - &weak.equilibrium;
        let predicted = [1.0, 0.0].map(|feedback| {
            let rate = |delta: &Spins| response.rate(delta, feedback);
            evolution(
                &disturbance,
                rate,
                DT,
                STEPS,
                STEPS_PER_FRAME,
                MIDPOINT_ITERATIONS,
            )
            .iter()
            .map(|delta| model.gap(delta))
            .collect()
        });
        WeakQuench {
            exact: weak
                .history
                .iter()
                .map(|s| model.gap(s) - settled)
                .collect(),
            predicted,
        }
    }
}

use condensate::*;

const SECONDS: f32 = 12.0;
/// The levels drawn as arrows, within this energy of the chemical potential.
const DISPLAY_LEVELS: usize = 33;
const DISPLAY_CUTOFF: f64 = 3.0;
/// The drawing's length per unit of energy, and per unit of spin.
const ENERGY_SCALE: f64 = 0.8;
const SPIN_SCALE: f64 = 2.0;

/// What the frames share, computed once.
struct Data {
    strong: Quench,
    gaps: Vec<f64>,
    final_gap: f64,
    shown: Vec<usize>,
    /// The weak quench's gap less the ground state's, in the plane yz: from the nonlinear
    /// evolution, and predicted with and without the gap's feedback.
    exact: Vec<f64>,
    predicted: [Vec<f64>; 2],
}

fn data() -> Data {
    let strong = quench(QUENCH_RATIO);
    let gaps = strong
        .history
        .iter()
        .map(|s| strong.after.gap(s).norm())
        .collect();
    let final_gap = strong.after.gap(&strong.equilibrium).norm();
    let energies: Vec<f64> = strong.after.dispersion.iter().map(energy).collect();
    let visible: Vec<usize> = (0..LEVELS)
        .filter(|k| energies[*k].abs() < DISPLAY_CUTOFF)
        .collect();
    let shown = (0..DISPLAY_LEVELS)
        .map(|k| visible[k * (visible.len() - 1) / (DISPLAY_LEVELS - 1)])
        .collect();
    let weak = weak_quench();
    // A gap's part in the plane yz, read across the patch square to it.
    let in_yz = |gaps: &[Plane]| -> Vec<f64> { gaps.iter().map(|g| (*g & x()).s()).collect() };
    let exact = in_yz(&weak.exact);
    let predicted = weak.predicted.each_ref().map(|p| in_yz(p));
    Data {
        strong,
        gaps,
        final_gap,
        shown,
        exact,
        predicted,
    }
}

/// The drawing's space.
type Space = Vector;

/// The patches square to the planes `yz` and `xy`: across them the planes read a field's part
/// in each.
fn x() -> Spin {
    Spin::new(1.0, 0.0, 0.0)
}
fn z() -> Spin {
    Spin::new(0.0, 0.0, 1.0)
}

/// A level's energy: its field's part in the plane `xy`.
fn energy(field: Plane) -> f64 {
    (field & z()).s()
}

/// A level's place on the energy axis, drawn along x.
fn on_axis(field: Plane) -> Space {
    Vector::new(energy(field) * ENERGY_SCALE, 0.0, 0.0)
}

/// A spin's coherence drawn across the energy axis: its reading by `yz` up, along the drawing's
/// z, its reading by `zx` along the drawing's y (its occupation, along the energy axis, is left
/// out).
fn upright(spin: Spin) -> Space {
    let read = |plane: Plane| (plane & spin).s() * SPIN_SCALE;
    Vector::new(0.0, read(zx()), read(yz()))
}

/// The coherence of the shown levels as arrows across the energy axis, with circles of the
/// largest coherence a half along it, and the curve through the arrows' tips.
fn draw_levels(s: &mut Scene3, d: &Data, spins: &Spins) {
    let dispersion = &d.strong.after.dispersion;
    let first = on_axis(dispersion.site(d.shown[0]));
    let last = on_axis(dispersion.site(*d.shown.last().expect("levels")));
    s.seg(reach3(first), reach3(last), 1.0, palette::GRID);
    // A circle at every eighth level, as numga draws them.
    for k in d.shown.iter().step_by(8) {
        let centre = on_axis(dispersion.site(*k));
        let ring: Vec<_> = (0..=64)
            .map(|j| {
                let turn = (gax::vga3d::Bivector::new(1.0, 0.0, 0.0)
                    * (-core::f64::consts::PI * j as f64 / 64.0))
                    .exp();
                reach3(centre + (turn >> upright(x() * 0.5)))
            })
            .collect();
        s.polyline(&ring, 0.8, palette::GRID.faded(0.8));
    }
    let n = d.shown.len();
    let tips: Vec<_> = d
        .shown
        .iter()
        .map(|k| reach3(on_axis(dispersion.site(*k)) + upright(spins.site(*k))))
        .collect();
    s.polyline(&tips, 0.8, palette::INK.faded(0.25));
    for (j, k) in d.shown.iter().enumerate() {
        let tone = colormap::coolwarm(0.08 + 0.84 * j as f32 / (n - 1) as f32);
        let base = reach3(on_axis(dispersion.site(*k)));
        s.arrow(base, upright(spins.site(*k)), 1.8, 6.0, tone);
    }
}

/// The ground state before the quench: the occupation and the coherence against energy.
fn draw_ground_state(c: &mut Canvas, ax: &Axes, d: &Data) {
    ax.frame(c, "GROUND STATE", "ENERGY FROM THE CHEMICAL POTENTIAL", "");
    let along = |pick: &dyn Fn(Spin) -> f64| -> Vec<Point2> {
        d.strong
            .after
            .dispersion
            .iter()
            .zip(d.strong.initial.iter())
            .map(|(e, s)| Point2::xy(energy(e) as f32, pick(s) as f32))
            .collect()
    };
    let occupation = along(&|s| (xy() & s).s() + 0.5);
    let coherence = along(&|s| (yz() & s).s());
    ax.polyline(c, &occupation, 1.6, palette::SKY);
    ax.polyline(c, &coherence, 1.6, palette::GREEN);
    ax.legend(
        c,
        &[
            ("OCCUPATION", palette::SKY),
            ("COHERENCE IN YZ", palette::GREEN),
        ],
    );
}

/// The gap's size against time up to record `now`, and the new ground state's gap.
fn draw_gap(c: &mut Canvas, ax: &Axes, d: &Data, now: usize) {
    ax.frame(c, "THE GAP AFTER THE QUENCH", "TIME", "|GAP|");
    let end = *d.strong.times.last().expect("records") as f32;
    let level = d.final_gap as f32;
    let across = [Point2::xy(0.0, level), Point2::xy(end, level)];
    ax.dashed(c, &across, 1.0, 5.0, palette::INK.faded(0.5));
    let note = Point2::xy(end * 0.98, level - 0.03);
    ax.text(c, note, "NEW GROUND STATE", 9.0, palette::INK, Align::Right);
    let curve: Vec<Point2> = d.strong.times[..=now]
        .iter()
        .zip(&d.gaps)
        .map(|(t, g)| Point2::xy(*t as f32, *g as f32))
        .collect();
    ax.polyline(c, &curve, 1.8, palette::GREEN);
    ax.scatter(
        c,
        &[*curve.last().expect("a record")],
        Marker::Dot,
        7.0,
        palette::GREEN,
    );
}

/// The weak quench's gap against time, and the two linear predictions, with a cursor at `now`.
fn draw_response(c: &mut Canvas, ax: &Axes, d: &Data, now: usize) {
    ax.frame(c, "A WEAK QUENCH: THE GAP'S CHANGE IN YZ", "TIME", "");
    let times = &d.strong.times;
    let curve = |values: &[f64]| -> Vec<Point2> {
        times
            .iter()
            .zip(values)
            .map(|(t, v)| Point2::xy(*t as f32, *v as f32))
            .collect()
    };
    let end = *times.last().expect("records") as f32;
    ax.line(c, ORIGIN2, Point2::xy(end, 0.0), 1.0, palette::GRID);
    ax.polyline(c, &curve(&d.exact), 3.5, palette::INK.faded(0.45));
    let predictions = [palette::GREEN, palette::ORANGE];
    for (values, colour) in d.predicted.iter().zip(predictions) {
        ax.dashed(c, &curve(values), 1.4, 5.0, colour);
    }
    let t = times[now] as f32;
    ax.line(
        c,
        Point2::xy(t, ax.y[0]),
        Point2::xy(t, ax.y[1]),
        1.0,
        palette::GRID,
    );
    let entries: [(&str, Light); 3] = [
        ("NONLINEAR", palette::INK),
        ("WITH FEEDBACK", predictions[0]),
        ("GAP HELD FIXED", predictions[1]),
    ];
    ax.legend(c, &entries);
}

fn draw(c: &mut Canvas, d: &Data, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let progress = (t / SECONDS).fract();
    let records = d.strong.history.len();
    let now = frame(progress, records);
    let spins = &d.strong.history[now];

    let levels_rect = screen.part(0.0, 0.12, 0.58, 0.68);
    let cam = Camera::orbit(
        levels_rect,
        ORIGIN3,
        7.0,
        (-58.0f32).to_radians() + 0.2 * wave(core::f32::consts::TAU * progress),
        24.0f32.to_radians(),
        Lens::Perspective(0.6),
    );
    let drawn = panel3(c, levels_rect, cam, |s| draw_levels(s, d, spins));
    let axis_end = on_axis(xy() * (DISPLAY_CUTOFF * 1.1));
    if let Some(p) = drawn.px(reach3(axis_end)) {
        c.text("ENERGY", p, 10.0, palette::INK, Align::Left);
    }
    let title = levels_rect.top_middle() + Point2::direction(0.0, 14.0);
    let text = "PAIR COHERENCE OF EACH LEVEL";
    c.text(text, title, 12.0, palette::INK, Align::Center);

    let ground = Axes::new(
        screen.part(0.07, 0.74, 0.55, 0.92),
        [-5.0, 5.0],
        [-0.05, 1.05],
    );
    draw_ground_state(c, &ground, d);
    let top = d.gaps.iter().fold(0.0f64, |m, g| m.max(*g)) as f32 * 1.12;
    let end = *d.strong.times.last().expect("records") as f32;
    let gap = Axes::new(screen.part(0.66, 0.16, 0.97, 0.44), [0.0, end], [0.0, top]);
    draw_gap(c, &gap, d, now);
    let reach = d.exact.iter().fold(0.0f64, |m, v| m.max(v.abs())) as f32 * 1.6;
    let weak = Axes::new(
        screen.part(0.66, 0.6, 0.97, 0.9),
        [0.0, end],
        [-reach, reach],
    );
    draw_response(c, &weak, d, now);
    let total = d.strong.after.energy(spins);
    caption(
        c,
        "SUPERCONDUCTIVITY: A QUENCHED CONDENSATE",
        &format!(
            "ANDERSON PSEUDOSPINS IN VGA3D, A FIELD OVER {LEVELS} LEVELS  T {:.1}  ENERGY {total:.5}",
            d.strong.times[now]
        ),
    );
}

fn main() {
    run_with(
        Anim::new("superconductivity", SECONDS).size(960, 540),
        data,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::condensate::*;
    use gax::vga3d::Vector;

    const ROUND_OFF: f64 = 1e-11;

    /// The largest coefficient of some pseudospins.
    fn largest(spins: impl IntoIterator<Item = Spin>) -> f64 {
        spins
            .into_iter()
            .flat_map(|v| v.c)
            .fold(0.0, |m, x| m.max(x.abs()))
    }

    /// The largest coefficient of some fields.
    fn largest_field(fields: impl IntoIterator<Item = Plane>) -> f64 {
        fields
            .into_iter()
            .flat_map(|v| v.c)
            .fold(0.0, |m, x| m.max(x.abs()))
    }

    /// numga's `test_equilibrium_gap_and_phase_symmetry`: the ground state is still, its spins
    /// are a half long, its gap is the continuum gap equation's `Λ / sinh(2 Λ / g)` (up to the
    /// quadrature), and turning every spin about z gives another ground state, the gap turned.
    #[test]
    fn the_ground_state_is_still_and_its_phase_is_free() {
        let q = quench(QUENCH_RATIO);
        let model = &q.after;
        assert!(largest(model.rate(&q.equilibrium).iter()) < ROUND_OFF);
        assert!(
            q.equilibrium
                .iter()
                .all(|s| (s.norm_squared() - 0.25).abs() < ROUND_OFF)
        );
        let coupling = COUPLING * QUENCH_RATIO;
        let continuum = CUTOFF / (2.0 * CUTOFF / coupling).sinh();
        let gap = model.gap(&q.equilibrium);
        assert!(
            (gap.norm() - continuum).abs() < 1e-5 * continuum,
            "{gap:?} vs {continuum}"
        );
        let rotor = (xy() * (-0.73 / 2.0)).exp();
        let rotated = q.equilibrium.map(|s| rotor >> s);
        assert!(largest(model.rate(&rotated).iter()) < ROUND_OFF);
        let turned = model.gap(&rotated) - (rotor >> gap);
        assert!(turned.c.iter().all(|x| x.abs() < ROUND_OFF), "{turned:?}");
    }

    /// numga's `test_response_is_the_derivative_and_contains_the_phase_zero_mode`: the rate is
    /// quadratic, so its central difference is its derivative, the response; turning every
    /// phase together moves the spins by the local precession, which the changed gap cancels.
    #[test]
    fn the_response_is_the_derivative_and_holds_the_phase_mode() {
        let q = quench(QUENCH_RATIO);
        let model = &q.after;
        let response = model.response(&q.equilibrium);
        let respond = |delta: &Spins| response.rate(delta, 1.0);
        let turn = (yz() * 0.11 + zx() * 0.07).exp();
        let delta = (&q.initial - &q.equilibrium).map(|s| turn >> s);
        let ahead = model.rate(&(&q.equilibrium + &delta));
        let behind = model.rate(&(&q.equilibrium - &delta));
        let difference = (ahead - behind).scaled(0.5);
        assert!(largest((respond(&delta) - difference).iter()) < ROUND_OFF);

        let phase: Spins = q.equilibrium.map(|s| -xy().commutator(s));
        let moved = response.local.zip_with(&phase, |m, d| m.of(d));
        assert!(largest(moved.iter()) > 0.1);
        assert!(largest(respond(&phase).iter()) < ROUND_OFF);
    }

    /// numga's `test_collective_response_predicts_a_weak_quench`: with the gap's feedback the
    /// linear response follows the weak quench's gap within half a percent; with the gap held
    /// fixed it misses by more than a fifth.
    #[test]
    fn the_collective_response_predicts_a_weak_quench() {
        let WeakQuench { exact, predicted } = weak_quench();
        let scale = largest_field(exact.iter().copied());
        let error = |p: &[Plane]| largest_field(exact.iter().zip(p).map(|(e, p)| *e - *p)) / scale;
        let (collective, frozen) = (error(&predicted[0]), error(&predicted[1]));
        assert!(collective < 0.005, "{collective}");
        assert!(frozen > 0.2, "{frozen}");
    }

    /// numga's `test_midpoint_converges_at_second_order`: halving the step quarters the change
    /// of the end state. (numga compares the norms' ratio with 3.8 to 4.2; here the squared
    /// norms' ratio with their squares.)
    #[test]
    fn the_midpoint_rule_converges_at_second_order() {
        let q = quench(QUENCH_RATIO);
        let duration = 2.0;
        let end = |count: usize| {
            let run = evolution(
                &q.initial,
                |s| q.after.rate(s),
                duration / count as f64,
                count,
                count,
                MIDPOINT_ITERATIONS,
            );
            run.last().expect("the end").clone()
        };
        let ends = [40, 80, 160].map(end);
        let squared = |a: &Spins, b: &Spins| {
            let d = a - b;
            d.dot(&d)
        };
        let ratio = squared(&ends[0], &ends[1]) / squared(&ends[1], &ends[2]);
        assert!((3.8f64 * 3.8..4.2 * 4.2).contains(&ratio), "{ratio}");
    }

    /// numga's `test_precession_is_mirror_covariant`: mirroring the condensate, its occupation
    /// planes and its pairing correlation with it (the pairing's input pulled back through the
    /// mirror), mirrors its motion. The pseudospins are intrinsic, so the mirror turns them as
    /// the patches of their planes, opposite to plain vectors.
    #[test]
    fn the_precession_is_mirror_covariant() {
        let q = quench(QUENCH_RATIO);
        let model = &q.after;
        let mirrors = [
            Vector::new(1.0, 0.0, 0.0),
            Vector::new(0.0, 0.0, 1.0),
            Vector::new(0.6, 0.0, 0.8),
        ];
        let rate = model.rate(&q.initial);
        for mirror in mirrors {
            let mirrored = Condensate {
                dispersion: model.dispersion.map(|d| mirror >> d),
                pairing: mirror >> model.pairing.of(mirror << Spin::slot()),
            };
            let moved = mirrored.rate(&q.initial.map(|s| mirror >> s));
            let want = rate.map(|r| mirror >> r);
            assert!(largest((moved - want).iter()) < ROUND_OFF, "{mirror:?}");
        }
    }

    /// The energy stays as the condensate rings (not one of numga's tests: its midpoint rule's
    /// promise).
    #[test]
    fn the_quench_keeps_the_energy() {
        let q = quench(QUENCH_RATIO);
        let start = q.after.energy(&q.history[0]);
        let last = q.after.energy(q.history.last().expect("records"));
        assert!((last - start).abs() < 1e-5 * start.abs(), "{start} {last}");
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::data(), 0.5);
    }
}
