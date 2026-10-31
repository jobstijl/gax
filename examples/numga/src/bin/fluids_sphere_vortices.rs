//! numga's `fluids/sphere_vortices`: point vortices on the sphere, in the geometric algebra of
//! the sphere (VGA3D, whose vectors are the great circles).
//!
//! A vortex is a bivector: a point of the sphere, the meet of two great circles, oriented by the
//! circulation around it. Its rotor turns the sphere about the point, and the turn leaves two
//! places fixed, the antipodal pair the bivector stands for; a vortex with both its bivector and
//! its circulation negated is the same vortex. A place on the sphere is held the same way. The
//! flow of a vortex turns every place about the vortex at its circulation over `2π` times the
//! squared norm of the join of the two, `A & B`, the great circle through them, whose norm is the
//! sine of the angle between them: along each circle about the vortex the speed is the
//! circulation over `2π` over the circle's radius, so the flow is irrotational away from the
//! vortex's two points. It is the plane's formula, where the join of two points is their
//! distance; only the signature changed. A core radius softens the rate near the points.
//!
//! The flow at a place is the sum of the other vortices' bivectors, each weighted by its rate
//! there: the generator of the turn the fluid makes there. Two vortices alone turn rigidly about
//! their circulation-weighted sum. Two steps follow: the explicit one turns the vortices pair by
//! pair, each pair by its exact rotor, over the rounds of a round-robin schedule and back; the
//! implicit one turns every vortex by the rotor of the flow at the middle of its step, found by a
//! few sweeps of fixed-point iteration. The pairs keep the impulse, the circulation-weighted sum
//! of the bivectors, exactly; the midpoint turns closely.
//!
//! The animation shows numga's two: rings of five, six and seven equal vortices around the pole
//! (the ring of seven, neutral in the plane, breaks up on the sphere), and a dipole and two
//! leapfrogging pairs travelling around the equator. Every vortex is drawn at both its points;
//! what lies behind a sphere is hidden by it, and its antipode in front shows it.

use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Light, Marker, ORIGIN3, Point2, Point3, Rect, backdrop, caption,
    on_sphere, palette, reach3, run_with, signal::frame,
};

mod vortices {
    use core::f64::consts::{PI, TAU};
    use gax::vga3d::{Bivector, Vector};

    /// A point of the sphere oriented by the turn around it, the meet of two great circles: a
    /// vortex, or a place the flow turns.
    pub type Pivot = Bivector;

    /// Vortices on the sphere.
    #[derive(Clone, Debug)]
    pub struct Vortices {
        pub pivots: Vec<Pivot>,
        /// The circulation of each vortex.
        pub circulations: Vec<f64>,
        pub core_radius: f64,
    }

    /// The point of the sphere in a direction, of unit size: on the bivectors `yz`, `zx`, `xy`,
    /// the metric dual of the direction, read as the turn about it (extrinsic, ADR-051).
    pub fn pivot(direction: Vector) -> Pivot {
        direction.normalized().hodge().with_orientation()
    }

    impl Vortices {
        /// The same vortices at other places.
        pub fn at(&self, pivots: Vec<Pivot>) -> Vortices {
            Vortices {
                pivots,
                ..self.clone()
            }
        }

        /// The softened rate of a turn about one point at another, per unit circulation:
        /// `1 / (2π (|a & b|² + core²))`, the join of the two unit points.
        fn rate(&self, a: Pivot, b: Pivot) -> f64 {
            let join = a.normalized() & b.normalized();
            1.0 / (TAU * (join.norm_squared() + self.core_radius * self.core_radius))
        }

        /// The flow of the other vortices at each vortex's place `at[k]`: their unit bivectors,
        /// each weighted by its circulation times the rate.
        pub fn flow(&self, at: &[Pivot]) -> Vec<Pivot> {
            at.iter()
                .enumerate()
                .map(|(k, place)| {
                    let others = self.pivots.iter().zip(&self.circulations).enumerate();
                    others
                        .filter(|(j, _)| *j != k)
                        .map(|(_, (p, g))| p.normalized() * (g * self.rate(*place, *p)))
                        .sum()
                })
                .collect()
        }

        /// One step turning every vortex by the rotor of the flow of the others at the middle
        /// of its step, the middles found by `sweeps` of fixed-point iteration.
        pub fn midpoint(&self, dt: f64, sweeps: usize) -> Vortices {
            let mut later = self.pivots.clone();
            for _ in 0..sweeps {
                let middles: Vec<Pivot> = self
                    .pivots
                    .iter()
                    .zip(&later)
                    .map(|(p, q)| ((*p + *q) * 0.5).normalized())
                    .collect();
                let middle = self.at(middles);
                let flow = middle.flow(&middle.pivots);
                later = flow
                    .iter()
                    .zip(&self.pivots)
                    .map(|(f, p)| (*f * (-dt / 2.0)).exp() >> *p)
                    .collect();
            }
            self.at(later)
        }

        /// One step by the exact rotors of pairs: through the rounds of partners and back, for
        /// half the step each way. A vortex its own partner sits the round out: the turn about
        /// itself leaves it in place.
        pub fn paired(&self, partners: &[Vec<usize>], dt: f64) -> Vortices {
            let g = &self.circulations;
            let mut pivots = self.pivots.clone();
            for partner in partners.iter().chain(partners.iter().rev()) {
                pivots = (0..pivots.len())
                    .map(|k| {
                        let (p, j) = (pivots[k], partner[k]);
                        let other = pivots[j];
                        // The two bivectors weighted by circulation: the point the pair turns
                        // about, the same for both.
                        let both = (p * g[k] + other * g[j]) * self.rate(p, other);
                        (both * (-dt / 4.0)).exp() >> p
                    })
                    .collect();
            }
            self.at(pivots)
        }

        /// The circulation-weighted sum of the vortices' unit bivectors.
        #[cfg_attr(not(test), allow(dead_code))] // numga's check of the two steps
        pub fn impulse(&self) -> Pivot {
            let weighted = self.pivots.iter().zip(&self.circulations);
            weighted.map(|(p, g)| p.normalized() * *g).sum()
        }

        /// The interaction energy, the log of the tangent of half the angle between every two
        /// vortices, softened by the core: with `c` the cosine of the angle, minus
        /// `log((s - c) / (s + c))` times their circulations over `4π s`, with `s` the square
        /// root of one plus the squared core radius.
        #[cfg_attr(not(test), allow(dead_code))] // numga's check of the negated vortices
        pub fn energy(&self) -> f64 {
            #[allow(clippy::disallowed_methods)] // a scalar softening, not a length
            let s = (1.0 + self.core_radius * self.core_radius).sqrt();
            let n = self.pivots.len();
            let mut sum = 0.0;
            for i in 0..n {
                for j in (0..n).filter(|j| *j != i) {
                    let (a, b) = (self.pivots[i].normalized(), self.pivots[j].normalized());
                    // The inner product of two unit bivectors is minus the cosine between them.
                    let cosine = -(a | b).s();
                    let half = ((s - cosine) / (s + cosine)).ln() / (2.0 * TAU * s);
                    sum += self.circulations[i] * self.circulations[j] * half;
                }
            }
            -sum / 2.0
        }
    }

    /// A round robin over `count` vortices by the circle method: every vortex's partner in
    /// every round, each pair meeting once. For an odd count every vortex sits one round out, as
    /// its own partner.
    pub fn schedule(count: usize) -> Vec<Vec<usize>> {
        let seats = count + count % 2; // an empty seat for odd counts
        (0..seats - 1)
            .map(|shift| {
                let order: Vec<usize> = (0..seats)
                    .map(|i| {
                        if i == 0 {
                            0
                        } else {
                            1 + (i - 1 + shift) % (seats - 1)
                        }
                    })
                    .collect();
                let mut partner: Vec<usize> = (0..=count).collect();
                for i in 0..seats / 2 {
                    let (a, b) = (order[i], order[seats - 1 - i]);
                    partner[a] = b;
                    partner[b] = a;
                }
                // A vortex facing the empty seat keeps itself.
                (0..count)
                    .map(|k| if partner[k] == count { k } else { partner[k] })
                    .collect()
            })
            .collect()
    }

    // --- scenarios

    /// Rings of equal vortices around the pole: their numbers, the angle from the pole, and the
    /// core, small against the ring.
    pub const RING_COUNTS: [usize; 3] = [5, 6, 7];
    pub const RING_ANGLE: f64 = 25.0 * PI / 180.0;
    pub const RING_CORE: f64 = 0.02;
    /// A dipole and two leapfrogging pairs on the first meridian, each an angle from the
    /// equator, positive vortices to the north.
    pub const DIPOLE_ANGLES: [f64; 2] = [0.3, -0.3];
    pub const DIPOLE_CIRCULATIONS: [f64; 2] = [1.0, -1.0];
    pub const LEAPFROG_ANGLES: [f64; 4] = [0.5, -0.5, 0.275, -0.275];
    pub const LEAPFROG_CIRCULATIONS: [f64; 4] = [1.0, -1.0, 1.0, -1.0];
    pub const PAIR_CORE: f64 = 0.05;
    /// How far each place of a ring is turned off at random, and the seed. numga's stream
    /// (numpy's, seed 0) cannot be reproduced, and the check does not need it: the ring of
    /// seven, neutral in the plane, is linearly unstable on the sphere (Polvani and Dritschel,
    /// 1993), so a nudge grows exponentially whatever its direction, and the time it takes to
    /// reach the ring's size depends on the draw only through the logarithm of its part along
    /// the growing modes. With this seed the ring of seven's squared chords change by 0.58
    /// (0.55 to 0.96 over seeds 0 to 7) against numga's bound of 0.2; the rings of five and six,
    /// stable this close to the pole, stay below 0.002 and 0.004.
    pub const DISTURBANCE: f64 = 1e-4;
    pub const SEED: u64 = 0;
    /// The time step, the midpoint step's sweeps, the steps of the rings and of the
    /// travelling pairs, and the steps compared between the two time steps.
    pub const DT: f64 = 0.05;
    pub const SWEEPS: usize = 3;
    pub const RING_STEPS: usize = 320;
    pub const TRAVEL_STEPS: usize = 640;
    pub const COMPARED_STEPS: usize = 100;

    /// The vortices' pivots at every step, the first the start.
    pub fn history(
        vortices: &Vortices,
        step: impl Fn(&Vortices) -> Vortices,
        steps: usize,
    ) -> Vec<Vec<Pivot>> {
        let mut now = vortices.clone();
        let mut pivots = vec![now.pivots.clone()];
        for _ in 0..steps {
            now = step(&now);
            pivots.push(now.pivots.clone());
        }
        pivots
    }

    /// One midpoint step of the scenes' size.
    pub fn midpoint(vortices: &Vortices) -> Vortices {
        vortices.midpoint(DT, SWEEPS)
    }

    /// The place at an angle from the pole and around it: the pole, the meet of the great
    /// circles `x = 0` and `y = 0`, turned about the point `(0, 1, 0)`, then about itself.
    pub fn turned(colatitude: f64, longitude: f64) -> Pivot {
        let (pole, across) = (Bivector::new(0.0, 0.0, 1.0), Bivector::new(0.0, 1.0, 0.0));
        let tilted = (across * (-colatitude / 2.0)).exp() >> pole;
        (pole * (-longitude / 2.0)).exp() >> tilted
    }

    /// Rings of five, six and seven equal vortices around the pole, each place slightly
    /// disturbed: the pivots of each ring at every step.
    pub fn rings() -> Vec<Vec<Vec<Pivot>>> {
        use gax_numga_examples::rng::{self, Draw};
        let mut rng = rng::rng(SEED);
        RING_COUNTS
            .iter()
            .map(|&size| {
                let pivots = (0..size)
                    .map(|k| {
                        let place = turned(RING_ANGLE, TAU * k as f64 / size as f64);
                        let nudge = (pivot(rng.direction()) * DISTURBANCE).exp();
                        nudge >> place
                    })
                    .collect();
                let ring = Vortices {
                    pivots,
                    circulations: vec![1.0; size],
                    core_radius: RING_CORE,
                };
                history(&ring, midpoint, RING_STEPS)
            })
            .collect()
    }

    /// Vortices on the first meridian at angles from the equator.
    pub fn on_meridian(angles: &[f64], circulations: &[f64]) -> Vortices {
        Vortices {
            pivots: angles.iter().map(|a| turned(PI / 2.0 - a, 0.0)).collect(),
            circulations: circulations.to_vec(),
            core_radius: PAIR_CORE,
        }
    }

    /// The leapfrogging pairs.
    pub fn frogs() -> Vortices {
        on_meridian(&LEAPFROG_ANGLES, &LEAPFROG_CIRCULATIONS)
    }

    /// A dipole and two leapfrogging pairs travelling around the equator: their pivots at every
    /// step.
    pub fn travelling() -> [Vec<Vec<Pivot>>; 2] {
        let dipole = on_meridian(&DIPOLE_ANGLES, &DIPOLE_CIRCULATIONS);
        [dipole, frogs()].map(|v| history(&v, midpoint, TRAVEL_STEPS))
    }

    /// The largest change of any squared chord between two vortices over the steps.
    #[cfg_attr(not(test), allow(dead_code))] // numga's check of the rings
    pub fn chord_change(history: &[Vec<Pivot>]) -> f64 {
        let chords = |pivots: &[Pivot]| -> Vec<f64> {
            let pairs = pivots
                .iter()
                .flat_map(|a| pivots.iter().map(move |b| (*a, *b)));
            pairs.map(|(a, b)| (a - b).norm_squared()).collect()
        };
        let first = chords(&history[0]);
        let changes = history.iter().flat_map(|p| {
            let now = chords(p);
            now.into_iter()
                .zip(first.clone())
                .map(|(a, b)| (a - b).abs())
        });
        changes.fold(0.0, f64::max)
    }

    /// The leapfrogging pairs by the rotors of pairs and by midpoint turns over the first
    /// steps.
    #[cfg_attr(not(test), allow(dead_code))] // numga's check of the two steps
    pub fn compared() -> [Vec<Vec<Pivot>>; 2] {
        let frogs = frogs();
        let rounds = schedule(frogs.pivots.len());
        let by_pairs = history(&frogs, |v| v.paired(&rounds, DT), COMPARED_STEPS);
        [by_pairs, history(&frogs, midpoint, COMPARED_STEPS)]
    }
}

use vortices::*;

/// numga's frames: 320 of 40 ms, a step each for the rings and two for the travelling pairs.
const FRAMES: usize = RING_STEPS + 1;
const SECONDS: f32 = 12.8;
/// The trails' lengths in steps: numga's 120 for the travelling pairs; the rings' a quarter of
/// that, since 120 steps take a ring once round and its trails would close into one white loop.
const TRAIL: usize = 120;
const RING_TRAIL: usize = 30;
/// The width of a trail and the size of a dot.
const TRAIL_WIDTH: f32 = 1.6;
const DOT: f32 = 6.0;
/// The views as numga's (elevation, azimuth), in degrees: the rings from above the pole, the
/// travelling pairs from just above the equator.
const RING_VIEW: (f32, f32) = (70.0, -60.0);
const TRAVEL_VIEW: (f32, f32) = (20.0, 0.0);
/// The reach of a panel's view from its centre to its top, the sphere and a margin.
const REACH: f32 = 1.12;

/// The histories, computed once.
struct Scenes {
    rings: Vec<Vec<Vec<Pivot>>>,
    travelling: [Vec<Vec<Pivot>>; 2],
}

fn scenes() -> Scenes {
    Scenes {
        rings: rings(),
        travelling: travelling(),
    }
}

/// The light of the sphere.
fn sphere() -> Light {
    palette::BLUE.faded(0.12)
}

/// The light of a vortex: one per vortex, repeating.
fn strand(k: usize) -> Light {
    palette::series(k)
}

/// The point of the unit sphere a pivot stands for, its antipode for `-1`: the direction the
/// pivot's metric dual points in.
fn place(pivot: Pivot, end: f64) -> Point3 {
    reach3((pivot.normalized() * end).hodge())
}

/// A panel's parallel camera at numga's (elevation, azimuth) in degrees.
fn camera(rect: Rect, (elevation, azimuth): (f32, f32)) -> Camera {
    let scale = 0.5 * rect.height() / REACH;
    Camera::parallel(rect, scale, azimuth.to_radians(), elevation.to_radians())
}

/// One sphere at a step: the sphere lit, then each vortex at both its points, its trail
/// through the last steps fading into the sphere, only on the side facing the viewer.
fn draw_sphere(
    c: &mut Canvas,
    rect: Rect,
    history: &[Vec<Pivot>],
    (now, trail): (usize, usize),
    view: (f32, f32),
) {
    let cam = camera(rect, view);
    panel3(c, rect, cam, |s| {
        let at = |u: f32, v: f32| {
            use core::f32::consts::{PI, TAU};
            reach3(on_sphere(TAU * u, PI * (v - 0.5)))
        };
        s.surface(at, 36, 18, |_, _| sphere(), 0.95, None);
    });
    let facing = |p: Point3| cam.depth(p) < cam.depth(ORIGIN3);
    let first = now.saturating_sub(trail);
    panel3(c, rect, cam, |s| {
        let vortices = history[0].len();
        for (k, colour) in (0..vortices).map(|k| (k, strand(k))) {
            for end in [1.0, -1.0] {
                for step in first..now {
                    let (a, b) = (
                        place(history[step][k], end),
                        place(history[step + 1][k], end),
                    );
                    if facing(a) && facing(b) {
                        let age = (step + 1 - first) as f32 / (now - first) as f32;
                        s.seg(a, b, TRAIL_WIDTH, sphere().blend(colour.faded(0.45), age));
                    }
                }
                let p = place(history[now][k], end);
                if facing(p) {
                    s.dot(p, Marker::Dot, DOT, colour.whitened(0.3));
                }
            }
        }
    });
}

fn draw(c: &mut Canvas, s: &Scenes, t: f32) {
    backdrop(c);
    let now = frame((t / SECONDS).rem_euclid(1.0), FRAMES);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let rings = Rect::new(0.0, h * 0.13, w, h * 0.55);
    let names = ["FIVE: KEEP THEIR RING", "SIX: NEARLY", "SEVEN: BREAK UP"];
    for (k, (history, name)) in s.rings.iter().zip(names).enumerate() {
        let rect = rings.column(k, 3).inset(6.0, 16.0, 6.0, 0.0);
        draw_sphere(c, rect, history, (now, RING_TRAIL), RING_VIEW);
        let at = rect.top_middle() - Point2::direction(0.0, 4.0);
        c.text(name, at, 11.0, palette::INK, Align::Center);
    }
    let travel = Rect::new(w * 0.12, h * 0.57, w * 0.88, h);
    let names = ["A DIPOLE", "TWO PAIRS LEAPFROGGING"];
    for (k, (history, name)) in s.travelling.iter().zip(names).enumerate() {
        let rect = travel.column(k, 2).inset(6.0, 16.0, 6.0, 4.0);
        draw_sphere(c, rect, history, (2 * now, TRAIL), TRAVEL_VIEW);
        let at = rect.top_middle() - Point2::direction(0.0, 4.0);
        c.text(name, at, 11.0, palette::INK, Align::Center);
    }
    caption(
        c,
        "VORTICES ON THE SPHERE",
        "A VORTEX IS A BIVECTOR, BOTH ITS POINTS; ITS FLOW TURNS A PLACE AT G / 2PI |A & B|^2 (VGA3D)",
    );
}

fn main() {
    run_with(
        Anim::new("sphere_vortices", SECONDS).size(960, 540),
        scenes,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::vortices::*;
    use core::f64::consts::{PI, TAU};
    use gax::vga3d::Vector;
    use gax_numga_examples::rng::{self, Draw};

    /// The largest coefficient of the difference of two lists of pivots.
    fn apart(a: &[Pivot], b: &[Pivot]) -> f64 {
        let each = a.iter().zip(b).flat_map(|(p, q)| (*p - *q).c);
        each.map(f64::abs).fold(0.0, f64::max)
    }

    /// The circulation around every circle about the vortex is the vortex's own.
    #[test]
    fn a_vortex_flow_is_irrotational_away_from_its_two_points() {
        let vortex = pivot(Vector::new(0.0, 0.0, 1.0));
        for angle in [0.2, 1.0, PI / 2.0, 2.6] {
            // The place at the angle from the vortex, towards x; the sine of the angle is the
            // norm of their join.
            let place = turned(angle, 0.0);
            let sine = (place & vortex).norm();
            let rate = vortex * (1.0 / (TAU * (place & vortex).norm_squared()));
            let speed = place.commutator(rate).norm();
            assert!((speed * TAU * sine - 1.0).abs() <= 1e-12, "{angle}");
        }
    }

    /// Negating some vortices' bivectors with their circulations changes neither their motion,
    /// by either step, nor their energy or impulse.
    #[test]
    fn a_vortex_negated_whole_is_the_same_vortex() {
        let mut rng = rng::rng(1);
        let vortices = Vortices {
            pivots: (0..5).map(|_| pivot(rng.direction())).collect(),
            circulations: vec![1.0, -0.7, 0.5, 1.2, -1.0],
            core_radius: 0.05,
        };
        let signs = [1.0, -1.0, 1.0, -1.0, 1.0];
        let flip =
            |p: &[Pivot]| -> Vec<Pivot> { p.iter().zip(signs).map(|(p, s)| *p * s).collect() };
        let negated = Vortices {
            pivots: flip(&vortices.pivots),
            circulations: vortices
                .circulations
                .iter()
                .zip(signs)
                .map(|(g, s)| g * s)
                .collect(),
            core_radius: vortices.core_radius,
        };
        let rounds = schedule(5);
        let steps: [&dyn Fn(&Vortices) -> Vortices; 2] =
            [&|v| v.midpoint(0.1, 4), &|v| v.paired(&rounds, 0.1)];
        for step in steps {
            let (moved, negated_moved) = (step(&vortices), step(&negated));
            assert!(apart(&moved.pivots, &flip(&negated_moved.pivots)) <= 1e-14);
        }
        assert!((vortices.energy() - negated.energy()).abs() <= 1e-14);
        assert!(apart(&[vortices.impulse()], &[negated.impulse()]) <= 1e-14);
    }

    /// The pair's rotor keeps the angle between two vortices and their impulse, over a long
    /// step.
    #[test]
    fn two_vortices_alone_turn_rigidly() {
        let pair = Vortices {
            pivots: vec![
                pivot(Vector::new(1.0, 0.0, 0.3)),
                pivot(Vector::new(1.0, 0.4, -0.2)),
            ],
            circulations: vec![1.0, 0.6],
            core_radius: 0.05,
        };
        let moved = pair.paired(&schedule(2), 3.0);
        let sine = |v: &Vortices| (v.pivots[0] & v.pivots[1]).norm();
        assert!((sine(&moved) / sine(&pair) - 1.0).abs() <= 1e-12);
        assert!(apart(&[moved.impulse()], &[pair.impulse()]) <= 1e-14);
        // The vortices did move.
        assert!(apart(&moved.pivots, &pair.pivots) > 0.1);
    }

    /// The round robin: each pair meets once, a vortex sits out only for an odd count.
    #[test]
    fn the_schedule_meets_every_pair_once() {
        for count in 2..9 {
            let rounds = schedule(count);
            let mut met = vec![vec![0; count]; count];
            for partner in &rounds {
                for (k, j) in partner.iter().enumerate() {
                    assert_eq!(partner[*j], k);
                    if *j != k {
                        met[k][*j] += 1;
                    }
                }
            }
            for (k, row) in met.iter().enumerate() {
                assert!(
                    row.iter()
                        .enumerate()
                        .all(|(j, m)| *m == usize::from(j != k))
                );
            }
        }
    }

    /// numga's scenario checks: the ring of five keeps its shape, the ring of six nearly; the
    /// ring of seven, neutral in the plane, breaks up on the sphere.
    #[test]
    fn the_ring_of_seven_breaks_up() {
        let changes: Vec<f64> = rings().iter().map(|h| chord_change(h)).collect();
        assert!(
            changes[0] < 0.01 && changes[1] < 0.05 && changes[2] > 0.2,
            "{changes:?}"
        );
    }

    /// The dipole keeps its chord closely (the midpoint turns follow the pair's rigid turn to
    /// second order); the leapfrogging pairs stay mirror images across the equator, the mirror
    /// taking each positive vortex's flow onto the negative one's beside it.
    #[test]
    fn the_dipole_keeps_its_chord_and_the_pairs_their_mirror() {
        let [dipole, frogs] = travelling();
        let chord = |p: &[Pivot]| (p[0] - p[1]).norm_squared();
        let first = chord(&dipole[0]);
        assert!(
            dipole
                .iter()
                .all(|p| (chord(p) / first - 1.0).abs() <= 1e-2)
        );
        let equator = Vector::new(0.0, 0.0, 1.0);
        let g = LEAPFROG_CIRCULATIONS;
        for p in &frogs {
            for k in [0, 2] {
                let mirrored = equator >> (p[k] * g[k]);
                let beside = p[k + 1] * g[k + 1];
                assert!(apart(&[mirrored], &[beside]) <= 1e-8);
            }
        }
    }

    /// The pairs keep the impulse to round-off, the midpoint turns closely, and both steps
    /// agree closely, being of second order.
    #[test]
    fn the_steps_compare() {
        let [by_pairs, by_midpoints] = compared();
        let frogs = frogs();
        for (history, tolerance) in [(&by_pairs, 1e-12), (&by_midpoints, 1e-2)] {
            let impulses: Vec<Pivot> = history
                .iter()
                .map(|p| frogs.at(p.clone()).impulse())
                .collect();
            assert!(
                impulses
                    .iter()
                    .all(|i| apart(&[*i], &[impulses[0]]) <= tolerance)
            );
        }
        let largest = |a: &[Pivot], b: &[Pivot]| {
            let each = a.iter().zip(b).map(|(p, q)| (*p - *q).norm());
            each.fold(0.0, f64::max)
        };
        let last = by_pairs.len() - 1;
        let parting = largest(&by_pairs[last], &by_midpoints[last]);
        let travel = largest(&by_midpoints[last], &by_midpoints[0]);
        assert!(parting < 0.05 * travel, "{parting} {travel}");
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scenes(), 0.5);
    }
}
