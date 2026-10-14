//! numga's `twistors/scenarios.py`: the Robinson congruence's flow lines at time zero, the
//! Hopfion's field lines carried along its rays, and a light ray recovered from its twistor
//! through two events, at rest and boosted.

use crate::twistors::{
    Even, Event, Spatial, Twistor, act, at_time, event_of, moved_vector, point, ray, spatial, t,
    through, x, y, z,
};
use gax::vga3d;

/// Space's axes `x y z`, in VGA3D.
fn space_axes() -> [Spatial; 3] {
    [x(), y(), z()].map(spatial)
}

/// A Hopf fibre on a torus: the circle the rotor `start exp(I y phase)` traces in the
/// stereographic projection of the unit rotors onto space, `-I B / (1 + s)` for the rotor
/// `s + B`, sampled at `samples + 1` phases from `-π` to `π` (both ends, the same point).
pub fn fibre(polar: f64, azimuth: f64, samples: usize) -> Vec<Spatial> {
    let i = vga3d::Pseudoscalar::new(1.0);
    let [ex, ey, ez] = space_axes();
    let start = ((ez ^ ex) * (-azimuth / 2.0)).exp() * ((ex ^ ey) * (polar / 2.0)).exp();
    (0..=samples)
        .map(|k| {
            let phase = -core::f64::consts::PI + core::f64::consts::TAU * k as f64 / samples as f64;
            let rotor = (start * (i * ey * phase).exp()).into_inner();
            -(i * rotor.grade::<2>()) * (1.0 / (1.0 + rotor.s()))
        })
        .collect()
}

/// Linked circles tangent to the Robinson directions at time zero: `per_circle` fibres at
/// each polar angle, on nested tori.
pub fn fibres(polars: &[f64], per_circle: usize, samples: usize) -> Vec<Vec<Spatial>> {
    polars
        .iter()
        .flat_map(|polar| {
            (0..per_circle).map(move |k| {
                let azimuth = core::f64::consts::TAU * k as f64 / per_circle as f64;
                fibre(*polar, azimuth, samples)
            })
        })
        .collect()
}

/// The congruence's flow lines shown: polar angles, fibres per angle and samples per fibre.
pub const POLARS: [f64; 3] = [
    0.5 * core::f64::consts::PI,
    0.65 * core::f64::consts::PI,
    0.8 * core::f64::consts::PI,
];
pub const PER_CIRCLE: usize = 10;
pub const FIBRE_SAMPLES: usize = 240;

/// The turn of space from the flow lines to the Hopfion's electric lines, a quarter turn in
/// the `xy` plane.
pub fn electric_turn() -> gax::Unit<vga3d::Rotor<(), f64>> {
    let [ex, ey, _] = space_axes();
    ((ex ^ ey) * core::f64::consts::FRAC_PI_4).exp()
}

/// The turn from the flow lines to the magnetic lines, a quarter turn in the `yz` plane.
#[cfg_attr(not(test), allow(dead_code))] // the tests follow the magnetic lines too
pub fn magnetic_turn() -> gax::Unit<vga3d::Rotor<(), f64>> {
    let [_, ey, ez] = space_axes();
    ((ey ^ ez) * core::f64::consts::FRAC_PI_4).exp()
}

/// Curves of space turned by a rotor.
pub fn turned(curves: &[Vec<Spatial>], r: gax::Unit<vga3d::Rotor<(), f64>>) -> Vec<Vec<Spatial>> {
    curves
        .iter()
        .map(|c| c.iter().map(|p| r >> *p).collect())
        .collect()
}

/// The rapidities of the boosts that carry the incidence along, the first at rest.
pub const RAPIDITIES: [f64; 2] = [0.0, 0.8];

/// A boost along `x` by a rapidity.
pub fn boost(rapidity: f64) -> Even {
    ((x() ^ t()) * (rapidity / 2.0)).exp().into_inner()
}

/// The twistor of the light ray through two null-separated events, carried by each boost:
/// per boost, its twistor, the two events, and samples of the reconstructed ray.
pub struct Incidence {
    #[cfg_attr(not(test), allow(dead_code))] // the tests check the ray against it
    pub twistor: Twistor,
    pub events: [Event; 2],
    pub ray: Vec<Event>,
}

/// The incidence at rest and boosted. The boost acts on the twistor by its representation
/// and on the events by the sandwich.
pub fn incidence() -> Vec<Incidence> {
    let emission = x() * -0.6 + z() * 0.3;
    let light = t() + (x() + z()) * core::f64::consts::FRAC_1_SQRT_2;
    let events = [emission, emission + light * 1.4];
    let points = events.map(point);
    let twistor = through(points[0], points[1]);
    // The ray is sampled this far beyond the two events, as a share of their separation.
    let (margin, samples) = (0.6, 120);
    RAPIDITIES
        .iter()
        .map(|r| {
            let b = boost(*r);
            let twistor = act(b, twistor);
            let plane = ray(twistor, twistor);
            let events = points.map(|p| event_of(moved_vector(b, p)));
            let times = events.map(|e| -(t() | e).s());
            let ray = (0..samples)
                .map(|k| {
                    let share = -margin + (1.0 + 2.0 * margin) * k as f64 / (samples - 1) as f64;
                    at_time(plane, times[0] + (times[1] - times[0]) * share)
                })
                .collect();
            Incidence {
                twistor,
                events,
                ray,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::twistors::tests::{TOLERANCE, vanishes};
    use crate::twistors::{Faraday, at, electric, energy_flow, hopfion, magnetic, robinson};
    use gax::ApproxEq;
    use gax_numga_examples::measure::linking;

    /// Centred differences of sampled curves resolve their tangents to about this accuracy.
    const TANGENT_TOLERANCE: f64 = 0.003;
    /// The flow lines the tests sample: polar angles, fibres per angle, samples per fibre.
    const TEST_POLARS: [f64; 2] = [0.4 * core::f64::consts::PI, 0.65 * core::f64::consts::PI];
    const TEST_PER_CIRCLE: usize = 3;
    const TEST_SAMPLES: usize = 192;

    /// Whether a sampled curve runs along a direction field: at each inner sample, the centred
    /// difference's direction and the field's are parallel.
    fn runs_along(curve: &[Spatial], field: impl Fn(usize) -> Spatial) -> bool {
        (1..curve.len() - 1).all(|i| {
            let tangent = (curve[i + 1] - curve[i - 1]).normalized().into_inner();
            let along = field(i).normalized().into_inner();
            (tangent ^ along).max_abs_diff(&vga3d::Bivector::zero()) < TANGENT_TOLERANCE
        })
    }

    #[test]
    fn flow_lines_follow_the_robinson_directions() {
        for curve in fibres(&TEST_POLARS, TEST_PER_CIRCLE, TEST_SAMPLES) {
            let flow = |i: usize| spatial(robinson(at(curve[i], 0.0)));
            assert!(runs_along(&curve, flow));
        }
    }

    /// What an observer at rest reads off a field: its electric or its magnetic part.
    type FieldReading = fn(Faraday) -> Event;

    #[test]
    fn field_lines_follow_their_fields_along_the_energy_flow_rays() {
        let curves = fibres(&TEST_POLARS, TEST_PER_CIRCLE, TEST_SAMPLES);
        let kinds: [(_, FieldReading); 2] = [
            (turned(&curves, electric_turn()), electric),
            (turned(&curves, magnetic_turn()), magnetic),
        ];
        for time in [-0.8, 0.0, 0.8] {
            for (lines, field) in &kinds {
                for curve in lines {
                    // After their material points have followed the straight energy-flow rays,
                    // the lines still run along their fields.
                    let events: Vec<Event> = curve
                        .iter()
                        .map(|p| at(*p, 0.0) + robinson(at(*p, 0.0)) * time)
                        .collect();
                    let moved: Vec<Spatial> = events.iter().map(|e| spatial(*e)).collect();
                    assert!(runs_along(&moved, |i| spatial(field(hopfion(events[i])))));
                }
            }
        }
    }

    #[test]
    fn two_fibres_link_once() {
        let curves = fibres(&TEST_POLARS, TEST_PER_CIRCLE, TEST_SAMPLES);
        // The Gauss integral counts one linking of two distinct closed fibres.
        assert!((linking(&curves[0], &curves[1]).abs() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn incidence_rays_meet_their_events_at_rest_and_boosted() {
        for case in incidence() {
            let plane = ray(case.twistor, case.twistor);
            // A ray's meet with a time slice reconstructs the incident events, also when a
            // boost changes their times, and every sampled ray event sends its twistor to zero.
            for e in case.events {
                assert!(at_time(plane, -(e | t()).s()).max_abs_diff(&e) < TOLERANCE);
            }
            for r in &case.ray {
                assert!(vanishes(
                    act(point(*r), case.twistor),
                    Twistor::zero(),
                    TOLERANCE
                ));
            }
        }
    }

    /// numga's scenario checks: the Hopfion is null, and its energy flows along the Robinson
    /// ray through each point of its electric lines.
    #[test]
    fn hopfion_is_null_and_flows_along_the_robinson_rays() {
        let lines = turned(&fibres(&POLARS, PER_CIRCLE, FIBRE_SAMPLES), electric_turn());
        for p in lines.iter().flatten() {
            let e = at(*p, 0.0);
            let field = hopfion(e);
            assert!(vanishes(field * field, Even::zero(), 1e-12));
            assert!(energy_flow(field).max_abs_diff(&robinson(e)) < 1e-10);
        }
    }

    /// Points of two fibres, against numga's numbers.
    #[test]
    fn fibres_match_numga() {
        let pi = core::f64::consts::PI;
        let curves = fibres(&[0.5 * pi, 0.8 * pi], 10, 240);
        let expected = [
            (3, 0, [-0.97893185, 0.97893185, -0.71123562]),
            (3, 60, [-0.97123031, -0.97123031, 1.33678384]),
            (3, 120, [0.40410463, -0.40410463, 0.2935992]),
            (17, 30, [-0.15633541, 0.32071619, 0.98706295]),
            (17, 90, [1.35199238, -0.06957654, 0.21413456]),
        ];
        for (curve, sample, [px, py, pz]) in expected {
            let p = curves[curve][sample];
            assert!((p - Spatial::new(px, py, pz)).norm() < 1e-7, "{p:?}");
        }
    }
}
