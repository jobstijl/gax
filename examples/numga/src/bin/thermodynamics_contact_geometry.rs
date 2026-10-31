//! numga's `thermodynamics/contact_geometry`: a Legendre change of the controlled variable, from
//! a gas's energy as a function of its entropy to its Helmholtz energy as a function of its
//! temperature, in the five-dimensional metric-free algebra `u0s0v0t0p0` (declared here with
//! `gax::algebra!`; every generator squares to zero).
//!
//! The five quantities of the gas, energy, entropy, volume, temperature and pressure, are the
//! readings of an intrinsic grade-4 element (numga's antivector), `t & point`. Equilibrium lifts
//! each state of entropy and volume (a point of their own plane, `s0v0`, where numga takes the
//! subspace of the two antivectors) to the point whose energy,
//! temperature and pressure the energy function gives there, a surface in the five-dimensional
//! space; the slopes are a jet gradient. Along the surface the contact element,
//! `u - T s + p v` paired with a change, vanishes: a change of energy is the heat less the work.
//! The Legendre change to the Helmholtz energy is a map of the whole space, exchanging entropy
//! and temperature and the energy for the energy less temperature times entropy; its jet
//! derivative carries changes, and the contact element pulled back through it is the contact
//! element again, so the transformed surface is an equilibrium surface too. The contact element's
//! own derivative, the bivector `p∧v - t∧s`, wedged twice with it, fills the space: the contact
//! planes twist everywhere, and a loop lifted along them does not close but gains its enclosed
//! area in energy.
//!
//! The animation: on the left, the energy against the entropy at several volumes, the followed
//! section's tangent lines enveloping it, the moving tangent's slope the temperature and its
//! height at zero entropy the Helmholtz energy; on the right, the Helmholtz energy against the
//! temperature, traced as the tangent moves, its own tangent's slope minus the entropy and its
//! height at zero temperature the energy again. Entropy is in nR, volume and temperature in
//! reference values, energy in nRT₀.

#[path = "../shared/thermodynamics_charts.rs"]
mod charts;

gax::algebra! {
    algebra gas "Energy, entropy, volume, temperature and pressure, `u0s0v0t0p0`: every generator squares to zero.";
    basis eu = 0, es = 0, ev = 0, et = 0, ep = 0;
    kind Scalar = [1];
    kind Vector = [eu, es, ev, et, ep];
    kind Bivector = [eus, euv, eut, eup, esv, est, esp, evt, evp, etp];
    kind Trivector = [eusv, eust, eusp, euvt, euvp, eutp, esvt, esvp, estp, evtp];
    kind Antivector = [esvtp, euvtp, eustp, eusvp, eusvt];
    kind Pseudoscalar = [eusvtp];
}

gax::algebra! {
    algebra plane "The states of entropy and volume, `s0v0`, lifted into the five quantities by equilibrium.";
    basis es = 0, ev = 0;
    kind Scalar = [1];
    kind Vector = [es, ev];
    kind Pseudoscalar = [esv];
}

use charts::{BOLD, Diagram, Family, Floating, MEDIUM, THIN};
use gax_numga_examples::{Anim, Canvas, Point2, Rect, caption, run_with};

mod legendre {
    use super::gas::{Bivector, Scalar, Vector, intrinsic};
    use gax::Real;
    use gax::jet::{gradient, jacobian};

    /// A point of the five-dimensional space (numga's `Antivector-`).
    pub type Ambient<T = f64> = intrinsic::Antivector<(), T>;
    /// A state of entropy and volume: an intrinsic vector of their own plane, `s0v0`.
    pub type State<T = f64> = super::plane::intrinsic::Vector<(), T>;
    /// `Scalar <- Ambient`.
    pub type Reading = Scalar<(intrinsic::Antivector,), f64>;

    /// The five quantities' vectors.
    pub fn u() -> Vector {
        Vector::new(1.0, 0.0, 0.0, 0.0, 0.0)
    }
    pub fn s() -> Vector {
        Vector::new(0.0, 1.0, 0.0, 0.0, 0.0)
    }
    pub fn v() -> Vector {
        Vector::new(0.0, 0.0, 1.0, 0.0, 0.0)
    }
    pub fn t() -> Vector {
        Vector::new(0.0, 0.0, 0.0, 1.0, 0.0)
    }
    pub fn p() -> Vector {
        Vector::new(0.0, 0.0, 0.0, 0.0, 1.0)
    }

    /// A quantity's reading of a point with coefficients of any type (jets included).
    pub fn read<T: Real>(g: Vector, x: Ambient<T>) -> T {
        (g.map_coefs(T::from_f64) & x).s()
    }
    /// A quantity's unit change, `g.dual()`, with coefficients of any type.
    fn unit<T: Real>(g: Vector) -> Ambient<T> {
        g.dual().map_coefs(T::from_f64)
    }

    /// The point whose five readings are `[u, s, v, t, p]`.
    #[cfg_attr(not(test), allow(dead_code))] // numga's core, checked by the tests
    pub fn point(values: [f64; 5]) -> Ambient {
        [u(), s(), v(), t(), p()]
            .into_iter()
            .zip(values)
            .map(|(g, x)| g.dual() * x)
            .sum()
    }
    /// The plane of states' entropy and volume vectors.
    fn plane_s() -> super::plane::Vector {
        super::plane::Vector::new(1.0, 0.0)
    }
    fn plane_v() -> super::plane::Vector {
        super::plane::Vector::new(0.0, 1.0)
    }
    /// The state of an entropy and a volume.
    pub fn state(entropy: f64, volume: f64) -> State {
        plane_s().dual() * entropy + plane_v().dual() * volume
    }
    /// The unit changes of the state, along the entropy and the volume.
    pub fn along_entropy() -> State {
        plane_s().dual()
    }
    pub fn along_volume() -> State {
        plane_v().dual()
    }
    /// A state as a point of the five-dimensional space with no energy, temperature or
    /// pressure: its entropy and volume read by pairing and carried to the space's changes.
    fn included<T: Real>(x: State<T>) -> Ambient<T> {
        let read = |g: super::plane::Vector| (g.map_coefs(T::from_f64) & x).s();
        unit::<T>(s()).gp(read(plane_s())) + unit::<T>(v()).gp(read(plane_v()))
    }

    /// The contact element's derivative: the temperature's change wedged with the entropy's,
    /// against the pressure's with the volume's.
    #[cfg_attr(not(test), allow(dead_code))] // numga's core, checked by the tests
    pub fn contact_derivative() -> Bivector {
        p().wedge(v()) - t().wedge(s())
    }

    /// The energy of an ideal gas with a constant heat capacity at fixed volume.
    pub fn internal_energy<T: Real>(x: State<T>, capacity: f64) -> T {
        let point = included(x);
        let c = T::from_f64(capacity);
        c * ((read(s(), point) - read(v(), point).ln()) / c).exp()
    }

    /// Each state lifted to its five quantities: the energy, and its slopes along the entropy and
    /// the volume, the temperature and minus the pressure.
    pub fn equilibrium<T: Real>(x: State<T>, capacity: f64) -> Ambient<T> {
        let (energy, differential) = gradient(|y| internal_energy(y, capacity), x);
        let slope = |step: State| differential.of(step.map_coefs(T::from_f64)).s();
        included(x) + unit::<T>(u()).gp(energy) + unit::<T>(t()).gp(slope(along_entropy()))
            - unit::<T>(p()).gp(slope(along_volume()))
    }

    /// The vector whose pairing with a change vanishes when the change of energy is the heat
    /// less the work.
    #[cfg_attr(not(test), allow(dead_code))] // numga's core, checked by the tests
    pub fn contact_element(x: Ambient) -> Vector {
        u() - s() * read(t(), x) + v() * read(p(), x)
    }

    /// The Legendre change exchanging entropy and temperature: the Helmholtz energy, the energy
    /// less temperature times entropy, in place of the energy, the temperature in place of the
    /// entropy and minus the entropy in place of the temperature.
    pub fn helmholtz<T: Real>(x: Ambient<T>) -> Ambient<T> {
        let [energy, entropy, volume, temperature, pressure] =
            [u(), s(), v(), t(), p()].map(|g| read(g, x));
        unit::<T>(u()).gp(energy - temperature * entropy) + unit::<T>(s()).gp(temperature)
            - unit::<T>(t()).gp(entropy)
            + unit::<T>(v()).gp(volume)
            + unit::<T>(p()).gp(pressure)
    }

    /// The Legendre change exchanging volume and pressure, with the enthalpy as the potential.
    #[cfg_attr(not(test), allow(dead_code))] // numga's core, checked by the tests
    pub fn enthalpy<T: Real>(x: Ambient<T>) -> Ambient<T> {
        let [energy, entropy, volume, temperature, pressure] =
            [u(), s(), v(), t(), p()].map(|g| read(g, x));
        unit::<T>(u()).gp(energy + pressure * volume)
            + unit::<T>(s()).gp(entropy)
            + unit::<T>(t()).gp(temperature)
            + unit::<T>(v()).gp(pressure)
            - unit::<T>(p()).gp(volume)
    }

    /// A quantity's reading as a map on points.
    pub fn reading(g: Vector) -> Reading {
        g & Ambient::slot()
    }

    // --- the scenario: the ideal gas's surface before and after the change --------------

    /// The gas's heat capacity at fixed volume, in nR.
    pub const CAPACITY: f64 = 1.5;

    /// The equilibrium surface over volumes and entropies, as curves at each volume, and its
    /// Legendre change.
    pub fn surfaces(entropies: &[f64], volumes: &[f64]) -> [Vec<Vec<Ambient>>; 2] {
        let lifted: Vec<Vec<Ambient>> = (volumes.iter())
            .map(|&vol| {
                (entropies.iter())
                    .map(|&e| equilibrium(state(e, vol), CAPACITY))
                    .collect()
            })
            .collect();
        let changed = (lifted.iter())
            .map(|curve| curve.iter().map(|x| helmholtz(*x)).collect())
            .collect();
        [lifted, changed]
    }

    /// A point of a section at fixed volume, its change along the entropy, and both after the
    /// Legendre change.
    #[derive(Clone, Copy, Debug)]
    pub struct Section {
        pub point: Ambient,
        pub tangent: Ambient,
        pub changed_point: Ambient,
        pub changed_tangent: Ambient,
    }

    /// The equilibrium points at a fixed volume, their changes along the entropy (the
    /// embedding's jet derivative bound to the entropy's change), and both carried through the
    /// Legendre change (its jet derivative carries the change).
    pub fn section(entropies: &[f64], volume: f64) -> Vec<Section> {
        (entropies.iter())
            .map(|&e| {
                let (point, embedding) = jacobian(|y| equilibrium(y, CAPACITY), state(e, volume));
                let tangent = embedding.of(along_entropy());
                let (changed_point, change) = jacobian(helmholtz, point);
                Section {
                    point,
                    tangent,
                    changed_point,
                    changed_tangent: change.of(tangent),
                }
            })
            .collect()
    }

    /// A loop of entropy and temperature at fixed volume and pressure, lifted along the contact
    /// planes: every edge changes the energy by the temperature times the entropy's change, so
    /// the lift does not close but gains the area the loop encloses. Its four edges, 17 samples
    /// each, before and after the Legendre change.
    #[cfg_attr(not(test), allow(dead_code))] // numga's core, checked by the tests
    pub fn horizontal_lift() -> [Vec<Vec<Ambient>>; 2] {
        let (low, high) = (-0.6, 0.6);
        let (cold, hot, initial) = (0.5, 1.5, 0.1);
        let span = high - low;
        let corners = [
            point([initial, low, 1.0, cold, 1.0]),
            point([initial, low, 1.0, hot, 1.0]),
            point([initial + hot * span, high, 1.0, hot, 1.0]),
            point([initial + hot * span, high, 1.0, cold, 1.0]),
            point([initial + (hot - cold) * span, low, 1.0, cold, 1.0]),
        ];
        let path: Vec<Vec<Ambient>> = (corners.windows(2))
            .map(|w| {
                (0..17)
                    .map(|k| {
                        let f = k as f64 / 16.0;
                        w[0] * (1.0 - f) + w[1] * f
                    })
                    .collect()
            })
            .collect();
        let changed = (path.iter())
            .map(|edge| edge.iter().map(|x| helmholtz(*x)).collect())
            .collect();
        [path, changed]
    }
}

use legendre::*;

/// numga's run: 93 entropies from -0.8 to 1.2 at volume 1, 40 ms a frame; the volumes of the
/// family.
const ENTROPIES: usize = 93;
const ENTROPY_SPAN: [f64; 2] = [-0.8, 1.2];
const SECTION_VOLUME: f64 = 1.0;
const VOLUMES: [f64; 5] = [0.6, 0.8, 1.0, 1.3, 1.7];
const FRAME_SECONDS: f32 = 0.04;
const HOLD: f32 = 1.2;

/// The charts' ranges: entropy and energy, temperature and Helmholtz energy; every so many
/// samples of the section, a faint tangent, sampled at so many places along it.
const ENTROPY_RANGE: [f64; 2] = [-0.8, 1.25];
const ENERGY_RANGE: [f64; 2] = [0.0, 3.4];
const TEMPERATURE_RANGE: [f64; 2] = [0.0, 2.2];
const FREE_ENERGY_RANGE: [f64; 2] = [0.0, 3.4];
const TANGENT_EVERY: usize = 3;
const LINE_SAMPLES: usize = 60;

/// Points on a chart of their entropy across and their energy up, read by pairing; after the
/// Legendre change, of the temperature and the Helmholtz energy.
fn chart_point(x: Ambient) -> Point2 {
    charts::at(reading(s()).of(x).s(), reading(u()).of(x).s())
}

/// The directions on the chart of changes: their changes of entropy and of energy.
fn chart_direction(x: Ambient) -> Point2 {
    charts::toward(reading(s()).of(x).s(), reading(u()).of(x).s())
}

/// One of the two charts: its family of curves, its section, and the faint tangents enveloping
/// the section.
struct Chart {
    family: Vec<Vec<Point2>>,
    section: Vec<Point2>,
    faint: Vec<Vec<Point2>>,
}

impl Chart {
    fn new(
        family: &[Vec<Ambient>],
        points: &[Ambient],
        tangents: &[Ambient],
        across: [f64; 2],
    ) -> Chart {
        let crossings: Vec<f64> = (0..LINE_SAMPLES)
            .map(|k| across[0] + (across[1] - across[0]) * k as f64 / (LINE_SAMPLES - 1) as f64)
            .collect();
        let faint = (points.iter().zip(tangents).step_by(TANGENT_EVERY))
            .map(|(x, d)| charts::tangent_points(chart_point(*x), chart_direction(*d), &crossings))
            .collect();
        Chart {
            family: (family.iter())
                .map(|curve| curve.iter().map(|x| chart_point(*x)).collect())
                .collect(),
            section: points.iter().map(|x| chart_point(*x)).collect(),
            faint,
        }
    }
}

/// What is computed once: the section and the two charts.
struct Scene {
    section: Vec<Section>,
    charts: [Chart; 2],
}

fn scene() -> Scene {
    let entropies: Vec<f64> = (0..ENTROPIES)
        .map(|k| {
            ENTROPY_SPAN[0]
                + (ENTROPY_SPAN[1] - ENTROPY_SPAN[0]) * k as f64 / (ENTROPIES - 1) as f64
        })
        .collect();
    let [family, changed_family] = surfaces(&entropies, &VOLUMES);
    let section = section(&entropies, SECTION_VOLUME);
    let pick = |f: fn(&Section) -> Ambient| section.iter().map(f).collect::<Vec<_>>();
    let charts = [
        Chart::new(
            &family,
            &pick(|x| x.point),
            &pick(|x| x.tangent),
            ENTROPY_RANGE,
        ),
        Chart::new(
            &changed_family,
            &pick(|x| x.changed_point),
            &pick(|x| x.changed_tangent),
            TEMPERATURE_RANGE,
        ),
    ];
    Scene { section, charts }
}

/// A chart with the tangent at a point of the section, its crossing of the axis up (the other
/// potential: the tangent line met with the line of zero across), and the section traced bold up
/// to the point.
#[allow(clippy::too_many_arguments)]
fn picture(
    c: &mut Canvas,
    d: &Diagram,
    chart: &Chart,
    point: Ambient,
    tangent: Ambient,
    frame: usize,
    intercept_name: &str,
    note: &str,
) {
    let level = Family::Level.light();
    d.lines(c, &chart.faint, THIN, level.faded(0.6));
    let volume = Family::Volume.light();
    d.lines(c, &chart.family, THIN, volume);
    d.polyline(c, &chart.section, THIN, charts::INK);
    let names = d.place(c, &[Floating::along(&chart.family, 4, "V = CONST", volume)]);
    d.polyline(c, &chart.section[..=frame], BOLD, charts::INK);
    let (position, direction) = (chart_point(point), chart_direction(tangent));
    let crossings: Vec<f64> = (0..LINE_SAMPLES)
        .map(|k| {
            let [a, b] = [d.ax.x[0], d.ax.x[1]].map(f64::from);
            a + (b - a) * k as f64 / (LINE_SAMPLES - 1) as f64
        })
        .collect();
    let line = charts::tangent_points(position, direction, &crossings);
    d.polyline(c, &line, MEDIUM, charts::ORANGE);
    let intercept = ((position & direction) ^ charts::across_line(0.0)).position();
    d.dots(c, &[intercept], charts::GREEN);
    d.dots(c, &[position], charts::ORANGE);
    let text = charts::TEXT;
    d.label(
        c,
        intercept,
        intercept_name,
        charts::GREEN,
        text,
        charts::RM,
        [-8.0, 0.0],
    );
    d.write_names(c, &names);
    d.axes(c);
    d.note(c, note);
}

fn draw(c: &mut Canvas, sc: &Scene, t: f32) {
    gax_numga_examples::backdrop(c);
    let frame = ((t / FRAME_SECONDS) as usize).min(ENTROPIES - 1);
    let x = sc.section[frame];
    let panels = charts::row(c, &[1.0, 1.0]);
    let inset = |r: Rect| r.part(0.1, 0.0, 0.9, 1.0);
    let energy = Diagram::new(
        inset(panels[0]),
        ENTROPY_RANGE,
        ENERGY_RANGE,
        [0.0, 0.0],
        ["S", "U"],
    );
    picture(
        c,
        &energy,
        &sc.charts[0],
        x.point,
        x.tangent,
        frame,
        "F",
        "U(S): SLOPE T, HEIGHT AT S = 0 IS F",
    );
    let free = Diagram::new(
        inset(panels[1]),
        TEMPERATURE_RANGE,
        FREE_ENERGY_RANGE,
        [0.0, 0.0],
        ["T", "F"],
    );
    picture(
        c,
        &free,
        &sc.charts[1],
        x.changed_point,
        x.changed_tangent,
        frame,
        "U",
        "F(T): SLOPE -S, HEIGHT AT T = 0 IS U",
    );
    caption(
        c,
        "CONTACT GEOMETRY: THE LEGENDRE CHANGE TO THE HELMHOLTZ ENERGY",
        "THE CONTACT ELEMENT U - T S + P V PULLS BACK TO ITSELF; ITS TANGENTS STAY TANGENT (U0S0V0T0P0)",
    );
}

fn main() {
    run_with(
        Anim::new("contact_geometry", FRAME_SECONDS * ENTROPIES as f32 + HOLD).size(960, 540),
        scene,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::legendre::*;
    use gax::ApproxEq;
    use gax::jet::jacobian;

    const ROUND_OFF: f64 = 1e-11;

    /// The lifted surface lies in the contact planes, its tangent planes are isotropic for the
    /// contact element's derivative, the gas is ideal with `U = c T` and `p V = T`, and the
    /// contact element wedged twice with its derivative is twice the pseudoscalar.
    #[test]
    fn ideal_gas_equilibrium_is_legendrian() {
        for (entropy, volume) in [(-0.5, 0.6), (0.2, 1.1), (0.8, 1.8)] {
            let (x, tangent) = jacobian(|y| equilibrium(y, CAPACITY), state(entropy, volume));
            let contact = contact_element(x);
            let pulled = contact & tangent;
            assert!(
                pulled.max_abs_diff(&super::gas::Scalar::zero()) < ROUND_OFF,
                "{pulled:?}"
            );
            let area = tangent.of(along_entropy()) & tangent.of(along_volume());
            assert!((area & contact_derivative()).s().abs() < ROUND_OFF);
            let [energy, temperature] = [u(), t()].map(|g| read(g, x));
            assert!((energy - CAPACITY * temperature).abs() < ROUND_OFF);
            let [pressure, vol] = [p(), v()].map(|g| read(g, x));
            assert!((pressure * vol - temperature).abs() < ROUND_OFF);
            let twist = contact_derivative();
            let volume_form = contact.wedge(twist).wedge(twist);
            assert!(
                (volume_form.c[0] - 2.0).abs() < ROUND_OFF,
                "{volume_form:?}"
            );
        }
    }

    /// Both Legendre changes preserve the contact element at points off equilibrium: pulled back
    /// through the change's jet derivative it is the contact element again. They commute, and
    /// their composite's potential is the Gibbs energy. (numga gives the two points by their
    /// antivector coefficients; here the five numbers are the points' readings.)
    #[test]
    fn legendre_change_preserves_contact_away_from_equilibrium() {
        let points = [
            point([0.4, -0.6, 0.7, 1.1, 1.8]),
            point([1.9, 0.8, 1.6, 0.5, 0.7]),
        ];
        // The contact element pulled back through a change's derivative, against itself.
        macro_rules! preserves {
            ($change:ident, $x:expr) => {{
                let (changed, derivative) = jacobian($change, $x);
                let pullback = contact_element(changed) & derivative;
                pullback.abs_diff_eq(&(contact_element($x) & Ambient::slot()), ROUND_OFF)
            }};
        }
        for x in points {
            assert!(preserves!(helmholtz, x));
            assert!(preserves!(enthalpy, x));
            let gibbs = helmholtz(enthalpy(x));
            assert!(gibbs.abs_diff_eq(&enthalpy(helmholtz(x)), ROUND_OFF));
            let [energy, entropy, volume, temperature, pressure] =
                [u(), s(), v(), t(), p()].map(|g| read(g, x));
            let potential = read(u(), gibbs) - energy + temperature * entropy - pressure * volume;
            assert!(potential.abs() < ROUND_OFF);
        }
    }

    /// Each edge of the lifted loop lies in the contact planes, before and after the change, and
    /// the lift's gap is the enclosed area, `ΔS ΔT`, in energy.
    #[test]
    fn horizontal_lift_gains_the_enclosed_area() {
        let [path, changed] = horizontal_lift();
        for edges in [&path, &changed] {
            for edge in edges {
                let step = edge[16] - edge[0];
                assert!((contact_element(edge[8]) & step).s().abs() < ROUND_OFF);
            }
        }
        let gap = path[3][16] - path[0][0];
        let entropy_span = read(s(), path[1][16] - path[1][0]);
        let temperature_span = read(t(), path[0][16] - path[0][0]);
        let expected = u().dual() * (entropy_span * temperature_span);
        assert!(gap.abs_diff_eq(&expected, ROUND_OFF), "{gap:?}");
    }

    /// The section's tangents lie in the contact planes before and after the change; the energy's
    /// slope along the entropy is the temperature, and after the change the Helmholtz energy's
    /// slope along the temperature is minus the entropy.
    #[test]
    fn section_tangents_preserve_contact_and_trade_slopes() {
        let entropies: Vec<f64> = (0..9).map(|k| -0.8 + 2.0 * k as f64 / 8.0).collect();
        for x in section(&entropies, 1.0) {
            assert!((contact_element(x.point) & x.tangent).s().abs() < ROUND_OFF);
            let changed = contact_element(x.changed_point) & x.changed_tangent;
            assert!(changed.s().abs() < ROUND_OFF);
            let slope = read(u(), x.tangent) / read(s(), x.tangent);
            assert!((slope - read(t(), x.point)).abs() < ROUND_OFF);
            let changed_slope = read(u(), x.changed_tangent) / read(s(), x.changed_tangent);
            assert!((changed_slope + read(s(), x.point)).abs() < ROUND_OFF);
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scene(), 2.0);
    }
}
