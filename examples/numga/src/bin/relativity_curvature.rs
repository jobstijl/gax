//! numga's `relativity/curvature`: a gravitational plane wave's curvature as a map on
//! bivectors in the spacetime algebra (STA), and the tides it produces in a ring of beads.
//!
//! In a vacuum plane wave the curvature is a nonzero bivector-to-bivector map whose
//! composition with itself vanishes: its image is null bivectors containing the wave's
//! direction, and that image lies in its kernel. All six eigenvalues vanish, yet the map does
//! not. Bind an observer twice, `R(u ∧ s)·u` with the separation `s` left open, and the result
//! is the tidal map from separation to relative acceleration: a composition of different
//! maps, not a similarity, so its eigenvalues are `±a` and zero. The cross polarization is the
//! dual of plus, `-I R`, the pattern turned an eighth of a turn: a spin-two field.
//!
//! The animation integrates a ring of beads through a Gaussian wave packet in the plus, cross
//! and circular polarizations (displacements magnified 4000 times), with the tidal
//! acceleration arrows; below, the packet's strain, the curvature map at one event (a
//! spacetime ribbon, its null image and the tidal readout), and the Doppler scaling of the
//! tide seen by observers boosted along the wave.

use std::sync::OnceLock;

use gax::{sta, vga3d};
use gax_numga_examples::canvas::srgb;
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Marker, Rgb, Scene3, backdrop, caption, palette, plot,
    run,
};

mod curvature {
    use core::ops::{Add, Mul};
    use gax::sta::{Bivector, Odd, Pseudoscalar, Vector};

    pub type V = Vector<(), f64>;
    pub type B = Bivector<(), f64>;
    /// A scalar plus a pseudoscalar: an amplitude and a phase.
    pub type Phasor = gax::sta::Phasor<(), f64>;
    /// Area bivector to curvature bivector.
    pub type Curvature = Bivector<(Bivector,), f64>;
    /// Separation to relative acceleration.
    pub type Tidal = Vector<(Vector,), f64>;
    /// Rest separation to displacement.
    pub type Strain = Vector<(Vector,), f64>;

    pub fn t() -> V {
        Vector::new(1.0, 0.0, 0.0, 0.0)
    }
    pub fn x() -> V {
        Vector::new(0.0, 1.0, 0.0, 0.0)
    }
    pub fn y() -> V {
        Vector::new(0.0, 0.0, 1.0, 0.0)
    }
    pub fn z() -> V {
        Vector::new(0.0, 0.0, 0.0, 1.0)
    }
    /// The pseudoscalar `e0123`: it squares to minus one and commutes with every bivector.
    pub fn i() -> Pseudoscalar<(), f64> {
        Pseudoscalar::new(1.0)
    }
    /// The bivector `e12 = x ∧ y`.
    pub fn xy() -> B {
        Bivector::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    }

    /// The product of two phasors, a phasor: they multiply as complex numbers.
    pub fn times(a: Phasor, b: Phasor) -> Phasor {
        (a * b).cast::<gax::sta::Phasor>()
    }

    /// Vacuum plane-wave curvature along the null direction `k`, polarized on the transverse
    /// pair `(a, b)`. The null bivectors `k ∧ a` and `k ∧ b` are mutually orthogonal, so their
    /// dyads compose to zero and the map is nilpotent; opposite weights cancel the Ricci
    /// contraction.
    pub fn plane_wave_curvature(k: V, a: V, b: V) -> Curvature {
        let (na, nb) = (k ^ a, k ^ b);
        na * (na | Bivector::slot()) - nb * (nb | Bivector::slot())
    }

    /// Unit plus and cross curvature maps for a wave travelling along `+z`. The cross
    /// polarization is the dual of plus, `-I plus`: a quarter duality turn turns the stretch
    /// and squeeze pattern of a null curvature by an eighth turn about the axis.
    pub fn polarizations() -> (Curvature, Curvature) {
        let plus = plane_wave_curvature(t() + z(), x(), y());
        (plus, -(i() * plus))
    }

    /// Bind the observer twice and leave the separation open: `u ∧ s` sweeps a separation
    /// into a spacetime ribbon, the curvature maps it to a bivector, and the commutator with
    /// the observer reads it out as a vector.
    pub fn tidal_map(curvature: Curvature, observer: V) -> Tidal {
        curvature.of(observer ^ Vector::slot()).commutator(observer)
    }

    /// Four-velocities of observers boosted along a spatial direction, one per rapidity.
    pub fn boosted_observers(rapidities: &[f64], direction: V) -> Vec<V> {
        rapidities
            .iter()
            .map(|r| ((direction ^ t()) * (r / 2.0)).exp() >> t())
            .collect()
    }

    /// Unit plus and cross strain maps for a wave along `+z`: plus stretches along `x` and
    /// squeezes along `y`; cross is the same turned by an eighth, by the transverse plane `xy`
    /// acting on each output (it makes the turn that the duality `-I` makes on the curvature).
    pub fn strain_patterns() -> (Strain, Strain) {
        let plus = y() * (y() | Vector::slot()) - x() * (x() | Vector::slot());
        (plus, xy() | plus)
    }

    /// Weak-wave strain in the plus, cross and circular polarizations: half the profile times
    /// the unit strain maps. `plus + I cross` pairs the patterns the way the phasor pairs its
    /// parts; the vector part of the product is the strain.
    pub fn polarized_strain(plus: Strain, cross: Strain, profile: Phasor) -> [Strain; 3] {
        let analytic = plus.cast::<Odd>() + (i() * cross).cast::<Odd>();
        [
            analytic * profile.s(),
            Pseudoscalar::new(profile.e0123()) * analytic,
            profile * analytic,
        ]
        .map(|w| w.cast::<Vector>() * 0.5)
    }

    // Used by the tests (part of numga's core).
    #[cfg_attr(not(test), allow(dead_code))]
    /// Curvature as a map on pairs of vectors (the plane's edges, in order), from the strain's
    /// second derivative along the wave: `k ∧ s''(a) (k·b) - (k·a) k ∧ s''(b)`.
    pub fn curvature_of_strain(k: V, second: Strain) -> Bivector<(Vector, Vector), f64> {
        (k ^ second) * (k | Vector::slot()) - (k | Vector::slot()) * (k ^ second)
    }

    /// The strain profile and its second derivative as phasors at the given times: a Gaussian
    /// about the middle of the window, a twelfth of it wide, times the carrier
    /// `exp(-I b s)`. Its rate is the phasor `-s / σ² - I b`, and its second derivative
    /// `(rate² - 1 / σ²) profile`.
    pub fn wave_packet(
        time: &[f64],
        duration: f64,
        cycles: f64,
        amplitude: f64,
    ) -> (Vec<Phasor>, Vec<Phasor>) {
        let sigma = duration / 12.0;
        let b = core::f64::consts::TAU * cycles / duration;
        time.iter()
            .map(|t| {
                let s = t - duration / 2.0;
                let envelope = amplitude * (-s * s / (2.0 * sigma * sigma)).exp();
                let profile = Pseudoscalar::new(-b * s).exp() * envelope;
                let rate = Phasor::new(-s / (sigma * sigma), -b);
                let curving = times(rate, rate) - Phasor::new(1.0 / (sigma * sigma), 0.0);
                (profile, times(curving, profile))
            })
            .unzip()
    }

    /// Weak-wave curvature in the plus, cross and circular polarizations: minus half the
    /// profile's second derivative times the unit map. The circular wave is the phasor times
    /// plus; its scalar part alone is the plus wave, its pseudoscalar part alone the cross.
    pub fn polarized_waves(plus: Curvature, second: Phasor) -> [Curvature; 3] {
        [
            plus * second.s(),
            Pseudoscalar::new(second.e0123()) * plus,
            second * plus,
        ]
        .map(|w| w * -0.5)
    }

    /// Unit reference separations in the plane transverse to the wave.
    pub fn detector_ring(count: usize) -> Vec<V> {
        (0..count)
            .map(|k| {
                let angle = core::f64::consts::TAU * k as f64 / count as f64;
                (xy() * (angle / 2.0)).exp() >> x()
            })
            .collect()
    }

    /// scipy's `cumulative_simpson` with `x` given and `initial=0`: on each interval, the
    /// integral of the parabola through it and a neighbour (the next one for even intervals,
    /// the previous one for odd intervals and the last), summed. Works on anything linear.
    pub fn cumulative_simpson<T>(y: &[T], x: &[f64]) -> Vec<T>
    where
        T: Copy + Add<Output = T> + Mul<f64, Output = T>,
    {
        let n = y.len();
        // The integral over [x[a], x[b]] (adjacent) of the parabola through a, b and c.
        let piece = |a: usize, b: usize, c: usize| {
            let (x21, x32) = ((x[b] - x[a]).abs(), (x[c] - x[b]).abs());
            let x31 = x21 + x32;
            let (q, r) = (x21 / x31, x21 / x32);
            (y[a] * (3.0 - q) + y[b] * (3.0 + q * r + q) + y[c] * (-q * r)) * (x21 / 6.0)
        };
        let mut out = vec![y[0] * 0.0];
        let mut sum = y[0] * 0.0;
        for k in 0..n - 1 {
            let part = if k % 2 == 0 && k + 2 < n {
                piece(k, k + 1, k + 2)
            } else {
                piece(k + 1, k, k - 1)
            };
            sum = sum + part;
            out.push(sum);
        }
        out
    }

    /// Four corners of a simple area element, drawn from a unit spacelike edge in its plane.
    pub fn plane_patch(area: B, edge: V) -> [V; 4] {
        let other = edge.commutator(area) * 0.5;
        let edge = edge * 0.6;
        [-edge - other, -edge + other, edge + other, edge - other]
    }

    // --- the scenes -------------------------------------------------------------------

    /// The packet window; the carrier wavelength is `DURATION / CYCLES`, two thirds.
    pub const DURATION: f64 = 6.0;
    pub const CYCLES: f64 = 9.0;
    /// Peak strain.
    pub const AMPLITUDE: f64 = 1e-4;
    pub const SAMPLES: usize = 1201;
    /// Detector radius, much smaller than the wavelength.
    pub const RADIUS: f64 = 0.01;
    pub const BEADS: usize = 24;
    /// Display magnification of the displacements.
    pub const AMPLIFICATION: f64 = 4000.0;

    /// `n` evenly spaced values from `a` to `b`, as numpy's `linspace`.
    pub fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
        let step = (b - a) / (n - 1) as f64;
        (0..n)
            .map(|i| if i == n - 1 { b } else { a + step * i as f64 })
            .collect()
    }

    /// The bead ring's response: `[polarization][time][bead]` accelerations, integrated twice
    /// from rest into displacements.
    pub struct Detector {
        pub time: Vec<f64>,
        pub reference: Vec<V>,
        pub displacement: Vec<Vec<Vec<V>>>,
        pub acceleration: Vec<Vec<Vec<V>>>,
        pub strain: Vec<Phasor>,
    }

    /// Accelerations of the beads, `[polarization][time][bead]`, from the tidal maps of the
    /// three polarized waves at each time.
    pub fn accelerations(time: &[f64], cycles: f64, reference: &[V]) -> Vec<Vec<Vec<V>>> {
        let (plus, _) = polarizations();
        let (_, second) = wave_packet(time, DURATION, cycles, AMPLITUDE);
        let mut out: Vec<Vec<Vec<V>>> = (0..3).map(|_| Vec::with_capacity(time.len())).collect();
        for s in &second {
            for (p, wave) in polarized_waves(plus, *s).into_iter().enumerate() {
                let response = tidal_map(wave, t());
                out[p].push(reference.iter().map(|r| response.of(*r)).collect());
            }
        }
        out
    }

    /// Integrate each bead's acceleration twice along time, starting at rest.
    pub fn integrate(time: &[f64], acceleration: &[Vec<Vec<V>>]) -> Vec<Vec<Vec<V>>> {
        acceleration
            .iter()
            .map(|per_time| {
                let beads = per_time[0].len();
                let columns: Vec<Vec<V>> = (0..beads)
                    .map(|b| {
                        let a: Vec<V> = per_time.iter().map(|row| row[b]).collect();
                        cumulative_simpson(&cumulative_simpson(&a, time), time)
                    })
                    .collect();
                (0..time.len())
                    .map(|k| columns.iter().map(|c| c[k]).collect())
                    .collect()
            })
            .collect()
    }

    /// The ring of beads through the packet in the plus, cross and circular polarizations.
    pub fn detector_scenario() -> Detector {
        let time = linspace(0.0, DURATION, SAMPLES);
        let reference: Vec<V> = detector_ring(BEADS)
            .into_iter()
            .map(|r| r * RADIUS)
            .collect();
        let acceleration = accelerations(&time, CYCLES, &reference);
        let displacement = integrate(&time, &acceleration);
        let (strain, _) = wave_packet(&time, DURATION, CYCLES, AMPLITUDE);
        Detector {
            time,
            reference,
            displacement,
            acceleration,
            strain,
        }
    }

    /// The tidal amplitude of the unit plus wave seen by observers boosted along the wave: the
    /// tidal map's largest singular value.
    pub fn doppler_scenario() -> (Vec<f64>, Vec<f64>) {
        let (plus, _) = polarizations();
        let rapidities = linspace(-0.7, 0.7, 15);
        let amplitudes = boosted_observers(&rapidities, z())
            .into_iter()
            .map(|o| tidal_map(plus, o).svdvals()[0])
            .collect();
        (rapidities, amplitudes)
    }
}

use curvature::*;

const SECONDS: f32 = 9.0;

fn tracking() -> [Rgb; 4] {
    [
        srgb(0.82, 0.35, 0.40),
        srgb(0.85, 0.65, 0.20),
        srgb(0.15, 0.70, 0.60),
        srgb(0.65, 0.45, 0.85),
    ]
}

/// The detector read out for drawing: bead positions and acceleration arrows in units of the
/// ring's radius, `[polarization][time][bead]`, displacements magnified.
struct Rings {
    reference: Vec<V>,
    profile: Vec<Phasor>,
    positions: Vec<Vec<Vec<V>>>,
    arrows: Vec<Vec<Vec<V>>>,
    limit: f32,
    /// Plus and cross strain over time, in units of `1e-4`.
    strain: Vec<[f64; 2]>,
    time: Vec<f64>,
    doppler: (Vec<f64>, Vec<f64>),
}

/// A separation's coordinates in the plane transverse to the wave (the ring panels' axes).
fn transverse(v: V) -> [f64; 2] {
    [v.e1(), v.e2()]
}

/// `[polarization][time][bead]` values, each mapped by `f`.
fn per_bead(values: &[Vec<Vec<V>>], f: impl Fn(V) -> V) -> Vec<Vec<Vec<V>>> {
    values
        .iter()
        .map(|per_time| {
            per_time
                .iter()
                .map(|row| row.iter().map(|v| f(*v)).collect())
                .collect()
        })
        .collect()
}

fn rings() -> &'static Rings {
    static RINGS: OnceLock<Rings> = OnceLock::new();
    RINGS.get_or_init(|| {
        let d = detector_scenario();
        let positions = d
            .displacement
            .iter()
            .map(|per_time| {
                per_time
                    .iter()
                    .map(|row| {
                        row.iter()
                            .zip(&d.reference)
                            .map(|(u, r)| (*r + *u * AMPLIFICATION) * (1.0 / RADIUS))
                            .collect()
                    })
                    .collect()
            })
            .collect::<Vec<Vec<Vec<V>>>>();
        // One arrow scale for the whole animation: the largest arrow is 0.31 radii long.
        let largest = d
            .acceleration
            .iter()
            .flatten()
            .flatten()
            .map(|a| a.norm())
            .fold(0.0, f64::max)
            .max(1e-300);
        let arrows = per_bead(&d.acceleration, |a| a * (0.31 / largest));
        let limit = positions
            .iter()
            .flatten()
            .flatten()
            .map(|p| transverse(*p).map(f64::abs).into_iter().fold(0.0, f64::max))
            .fold(1.35 - 0.31, f64::max)
            + 0.31;
        Rings {
            reference: d.reference.clone(),
            profile: d.strain.clone(),
            positions,
            arrows,
            limit: limit as f32,
            strain: d
                .strain
                .iter()
                .map(|p| {
                    // Plus is the phasor's scalar part, cross its pseudoscalar part turned back
                    // by `I`: the scalar part of `p I`.
                    let turned = *p * i();
                    [p.s() * 1e4, turned.s() * 1e4]
                })
                .collect(),
            time: d.time.clone(),
            doppler: doppler_scenario(),
        }
    })
}

/// One ring of beads at sample `k`, with the trails of four tracked beads and the tidal
/// acceleration arrows.
fn ring(c: &mut Canvas, ax: &Axes, r: &Rings, polarization: usize, k: usize) {
    let blue = palette::sky();
    let closed = |mut pts: Vec<[f64; 2]>| {
        pts.push(pts[0]);
        pts
    };
    let circle = closed(detector_ring(120).into_iter().map(transverse).collect());
    ax.dashed(c, &circle, 1.0, 4.0, palette::grid(), 1.0);
    ax.line(c, [-1.08, 0.0], [1.08, 0.0], 0.8, palette::grid(), 0.5);
    ax.line(c, [0.0, -1.08], [0.0, 1.08], 0.8, palette::grid(), 0.5);
    // The weak-wave prediction, the strain map applied to the rest separations: the integrated
    // beads land on it.
    let (plus, cross) = strain_patterns();
    let strain = polarized_strain(plus, cross, r.profile[k])[polarization];
    let predicted = closed(
        r.reference
            .iter()
            .map(|s| transverse((*s + strain.of(*s) * AMPLIFICATION) * (1.0 / RADIUS)))
            .collect(),
    );
    ax.dashed(c, &predicted, 1.0, 3.0, palette::ink(), 0.5);
    let beads = &r.positions[polarization][k];
    let now: Vec<[f64; 2]> = beads.iter().map(|p| transverse(*p)).collect();
    ax.polyline(c, &closed(now.clone()), 1.2, blue, 0.45);
    let tracked = [0, 6, 12, 18];
    let first = k.saturating_sub(r.time.len() / 7);
    for (j, bead) in tracked.iter().enumerate() {
        let trail: Vec<[f64; 2]> = r.positions[polarization][first..=k]
            .iter()
            .map(|row| transverse(row[*bead]))
            .collect();
        ax.polyline(c, &trail, 1.6, tracking()[j], 0.6);
    }
    for (p, a) in beads.iter().zip(&r.arrows[polarization][k]).step_by(3) {
        ax.arrow(c, transverse(*p), transverse(*p + *a), 1.4, 6.0, blue);
    }
    ax.scatter(c, &now, Marker::Dot, 5.0, blue, 1.0);
    for (j, bead) in tracked.iter().enumerate() {
        ax.scatter(c, &[now[*bead]], Marker::Dot, 8.0, tracking()[j], 1.0);
    }
    ax.scatter(c, &[[0.0, 0.0]], Marker::Dot, 5.0, palette::orange(), 1.0);
}

/// The curvature-map view of spacetime: `x` across, `z` along and `t` up, as a map to the
/// space the camera sees (`y`, the other transverse direction, drops out).
fn view() -> vga3d::Vector<(sta::Vector,), f64> {
    vga3d::Vector::from_images([
        vga3d::Vector::new(0.0, 0.0, 1.0),
        vga3d::Vector::new(1.0, 0.0, 0.0),
        vga3d::Vector::new(0.0, 0.0, 0.0),
        vga3d::Vector::new(0.0, 1.0, 0.0),
    ])
}

/// The curvature map at one event: the ribbon `t ∧ x`, its image under the plus curvature (a
/// null plane containing the wave direction), and the readout of that image by the observer.
fn curvature_map(c: &mut Canvas, rect: [f32; 4], azimuth: f32) {
    let (plus, _) = polarizations();
    let (observer, wave, edge) = (t(), t() + z(), x());
    let incoming = observer ^ edge;
    let outgoing = plus.of(incoming);
    let readout = outgoing.commutator(observer);
    let (cx, cy) = ((rect[0] + rect[2]) * 0.5, (rect[1] + rect[3]) * 0.5);
    let cam = Camera::orbit(
        (2.0 * cx) as usize,
        (2.0 * cy) as usize,
        [0.0, 0.0, 0.0],
        20.0,
        azimuth,
        0.3,
        Lens::Parallel(1.25 * 2.0 * cy / (rect[3] - rect[1])),
    );
    let mut s = Scene3::new(cam);
    let view = view();
    let origin = [0.0; 3];
    for axis in [x(), z(), t()] {
        s.arrow(origin, view.of(axis) * 1.05, 1.0, 6.0, palette::grid());
    }
    let ribbon = plane_patch(incoming, edge).map(|v| view.of(v));
    let image = plane_patch(outgoing, edge).map(|v| view.of(v));
    let (blue, orange) = (palette::sky(), palette::orange());
    s.quad(ribbon[0], ribbon[1], ribbon[2], ribbon[3], blue, 0.2);
    s.polyline(
        &[ribbon[0], ribbon[1], ribbon[2], ribbon[3], ribbon[0]],
        1.4,
        blue,
        1.0,
    );
    s.quad(image[0], image[1], image[2], image[3], orange, 0.12);
    s.polyline(
        &[image[0], image[1], image[2], image[3], image[0]],
        1.4,
        orange,
        1.0,
    );
    s.arrow(origin, view.of(observer), 2.0, 8.0, palette::ink());
    s.arrow(origin, view.of(wave), 2.0, 8.0, palette::yellow());
    s.arrow(origin, view.of(readout), 2.6, 8.0, palette::red());
    c.clip(rect);
    s.draw(c);
    c.unclip();
}

fn draw(c: &mut Canvas, t_now: f32) {
    backdrop(c);
    let r = rings();
    let (w, h) = (c.width as f32, c.height as f32);
    let size = (h / 40.0).clamp(7.0, 13.0);
    let phase = (t_now / SECONDS).rem_euclid(1.0);
    let k = ((phase * (r.time.len() - 1) as f32).round() as usize).min(r.time.len() - 1);
    let now = r.time[k];

    // The three rings.
    let (top, bottom) = (h * 0.13, h * 0.6);
    for (p, title) in ["PLUS", "CROSS", "CIRCULAR"].into_iter().enumerate() {
        let x0 = w * p as f32 / 3.0;
        let rect = plot::inset([x0, top, x0 + w / 3.0, bottom], 10.0, 18.0, 10.0, 0.0);
        let ax = Axes::equal(rect, [0.0, 0.0], r.limit);
        ring(c, &ax, r, p, k);
        let cx = (rect[0] + rect[2]) * 0.5;
        c.text(
            title,
            cx,
            rect[1],
            size * 1.1,
            palette::ink(),
            Align::Center,
        );
    }

    // The packet: plus and cross strain, with the present.
    let row = [0.0, bottom + h * 0.06, w, h - size * 3.0];
    let rect = plot::inset([row[0], row[1], w * 0.42, row[3]], 50.0, 18.0, 10.0, 0.0);
    let ax = Axes::new(rect, [0.0, 6.0], [-1.15, 1.15]);
    for (part, colour) in [palette::sky(), palette::orange()].into_iter().enumerate() {
        let curve: Vec<[f64; 2]> = r
            .time
            .iter()
            .zip(&r.strain)
            .map(|(t, s)| [*t, s[part]])
            .collect();
        ax.polyline(c, &curve, 1.3, colour, 0.9);
    }
    ax.line(c, [now, -1.15], [now, 1.15], 1.2, palette::yellow(), 0.9);
    ax.frame(c, "STRAIN / 1E-4: PLUS, CROSS", "TIME (C = 1)", "");

    // The curvature map at one event, the view turning.
    let rect = [w * 0.44, row[1], w * 0.72, row[3] + size * 2.0];
    curvature_map(c, rect, -1.2 + 0.6 * (phase * core::f32::consts::TAU).sin());
    let cx = (rect[0] + rect[2]) * 0.5;
    c.text(
        "RIBBON T^X",
        cx - size * 4.5,
        row[1] + size,
        size * 0.8,
        palette::sky(),
        Align::Right,
    );
    c.text(
        "ITS NULL IMAGE",
        cx - size * 3.5,
        row[1] + size,
        size * 0.8,
        palette::orange(),
        Align::Left,
    );
    c.text(
        "TIDAL READOUT",
        cx + size * 4.0,
        row[3] + size * 1.6,
        size * 0.8,
        palette::red(),
        Align::Left,
    );

    // The Doppler check: tidal amplitude against `exp(-2 rapidity)`.
    let rect = plot::inset([w * 0.74, row[1], w, row[3]], 40.0, 18.0, 14.0, 0.0);
    let ax = Axes::new(rect, [-0.75, 0.75], [0.2, 5.0]).log_y();
    let fine: Vec<[f32; 2]> = (0..=100)
        .map(|j| {
            let q = -0.7 + 1.4 * j as f32 / 100.0;
            [q, (-2.0 * q).exp()]
        })
        .collect();
    ax.polyline(c, &fine, 1.3, palette::ink(), 0.9);
    let dots: Vec<[f64; 2]> = r
        .doppler
        .0
        .iter()
        .zip(&r.doppler.1)
        .map(|(q, a)| [*q, *a])
        .collect();
    ax.scatter(c, &dots, Marker::Dot, 6.0, palette::sky(), 1.0);
    ax.frame(c, "TIDE VS BOOST", "RAPIDITY", "");

    caption(
        c,
        "CURVATURE OF A GRAVITATIONAL PLANE WAVE",
        "STA: A NILPOTENT MAP ON BIVECTORS, READ OUT AS TIDES (BEADS 4000X)",
    );
}

fn main() {
    run(Anim::new("curvature", SECONDS).size(960, 560), draw);
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::curvature::*;
    use gax::ApproxEq;
    use gax::motions::Linear;
    use gax::sta::{Bivector, Vector};

    /// numga draws normals from `np.random.default_rng(seed)`, which cannot be reproduced; the
    /// identities hold for any vectors.
    use gax_numga_examples::rng::{Draw, Rng, rng};

    /// A vector of four standard normal coefficients.
    fn vector(draws: &mut Rng) -> V {
        Vector::from_coeffs(core::array::from_fn(|_| draws.normal()))
    }

    /// The largest coefficient of a value or map, in magnitude.
    fn largest<X: Linear<f64>>(x: X) -> f64 {
        x.max_abs_diff(&X::zero())
    }

    fn close<X: ApproxEq>(a: X, b: X, tol: f64) -> bool {
        a.max_abs_diff(&b) <= tol
    }

    /// The reciprocal Lorentz basis: `basis[i] · reciprocal[j] = δij`.
    fn frames() -> ([V; 4], [V; 4]) {
        ([t(), x(), y(), z()], [t(), -x(), -y(), -z()])
    }

    #[test]
    fn nonzero_curvature_has_rank_two_and_annihilates_its_image() {
        let (plus, cross) = polarizations();
        for (a, b) in [(1.0, 0.0), (0.0, 1.0), (0.6, -0.8)] {
            let curvature = plus * a + cross * b;
            let (_, sigma, _) = curvature.svd();
            let rank = sigma.iter().filter(|s| **s > sigma[0] * 1e-12).count();
            assert_eq!(rank, 2, "{sigma:?}");
            assert!(largest(curvature.of(curvature)) <= 1e-14);
            // Null output planes are nonzero even though their Lorentz norms vanish.
            let image = curvature.of(t() ^ x());
            assert!(largest(image) > 0.5);
            assert!(image.norm_squared().abs() < 1e-14);
        }
    }

    #[test]
    fn riemann_pair_symmetry_bianchi_identity_and_vacuum_ricci() {
        let (plus, cross) = polarizations();
        let curvature = plus * 1.3 - cross * 0.7;
        let mut draws = rng(7);
        let [a, b, c, d] = core::array::from_fn(|_| vector(&mut draws));
        let (ab, cd) = (a ^ b, c ^ d);
        assert!(((ab | curvature.of(cd)).s() - (curvature.of(ab) | cd).s()).abs() < 1e-13);
        let cyclic = curvature.of(ab).commutator(c)
            + curvature.of(b ^ c).commutator(a)
            + curvature.of(c ^ a).commutator(b);
        assert!(largest(cyclic) <= 1e-13);
        // Contract one curvature slot with the reciprocal Lorentz basis.
        let (basis, reciprocal) = frames();
        let ricci = |r: Curvature| {
            basis
                .iter()
                .zip(&reciprocal)
                .map(|(e, f)| r.of(*e ^ Vector::slot()).commutator(*f))
                .fold(Tidal::zero(), |acc, m| acc + m)
        };
        assert!(largest(ricci(curvature)) <= 1e-13);
        // Frame-free: the Ricci form is the trace of the curvature against its wedge slot, with
        // the inner-product slot and the separation left open.
        let ricci_form = |r: Curvature| {
            Vector::slot()
                .commutator(r.of(Vector::slot() ^ Vector::slot()))
                .trace_at::<1>()
        };
        assert!(largest(ricci_form(curvature)) <= 1e-13);
        // The same trace on a non-vacuum curvature reproduces the frame contraction as a form.
        let tx = t() ^ x();
        let dyad = curvature + tx * (tx | Bivector::slot()) * 0.7;
        let frame_map = ricci(dyad);
        let mut draws = rng(11);
        let (a, b) = (vector(&mut draws), vector(&mut draws));
        let form = ricci_form(dyad).of(a).of(b).s();
        assert!((form - (a | frame_map.of(b)).s()).abs() < 1e-13);
        assert!(form.abs() > 1e-3);
    }

    /// The traces of the first four powers of a map on vectors: its eigenvalues' power sums,
    /// which fix the characteristic polynomial (numga compares `np.linalg.eigvals`).
    fn power_traces(m: Tidal) -> [f64; 4] {
        let m2 = m.of(m);
        [m.trace(), m2.trace(), m2.of(m).trace(), m2.of(m2).trace()]
    }

    #[test]
    fn observer_sees_opposite_transverse_tides_and_doppler_scaling() {
        let curvature = plane_wave_curvature(t() + z(), x(), y());
        for rapidity in [-0.7f64, 0.0, 0.5] {
            let observer = t() * rapidity.cosh() + z() * rapidity.sinh();
            let response = tidal_map(curvature, observer);
            // A chasing observer sees the frequency redshifted; curvature has two time slots,
            // so the tide scales with the frequency squared. Eigenvalues `[-a, 0, 0, a]`.
            let a = (-2.0 * rapidity).exp();
            let expected = [0.0, 2.0 * a * a, 0.0, 2.0 * a.powi(4)];
            let traces = power_traces(response);
            assert!(
                traces
                    .iter()
                    .zip(expected)
                    .all(|(p, e)| (p - e).abs() <= 1e-12)
            );
            assert!(largest(response.of(observer)) <= 1e-13);
            assert!(largest(response.of(t() + z())) <= 1e-13);
        }
        let (rapidities, amplitudes) = doppler_scenario();
        for (r, a) in rapidities.iter().zip(&amplitudes) {
            assert!((a - (-2.0 * r).exp()).abs() < 1e-12);
        }
    }

    #[test]
    fn curvature_and_observer_binding_are_lorentz_covariant() {
        let (plus, cross) = polarizations();
        // The duality turn -I is the eighth turn about the wave axis.
        let eighth = (xy() * (core::f64::consts::PI / 8.0)).exp();
        let turned = eighth >> plus.of(eighth << Bivector::slot());
        assert!(close(cross, turned, 1e-9));
        // numga's `exp(tx * .31) exp(yz * -.23)`, with `tx = -e10`.
        let rotor = Bivector::new(-0.31, 0.0, 0.0, 0.0, 0.0, 0.0).exp()
            * Bivector::new(0.0, 0.0, 0.0, -0.23, 0.0, 0.0).exp();
        let transformed = plane_wave_curvature(rotor >> (t() + z()), rotor >> x(), rotor >> y());
        let conjugated = rotor >> plus.of(rotor << Bivector::slot());
        assert!(close(transformed, conjugated, 1e-13));
        let observer_response = tidal_map(transformed, rotor >> t());
        let transformed_response = rotor >> tidal_map(plus, t()).of(rotor << Vector::slot());
        assert!(close(observer_response, transformed_response, 1e-13));
    }

    #[test]
    fn packet_acceleration_matches_the_strain_second_derivative() {
        let time = linspace(-0.5, 6.5, 2801);
        let (h, a) = wave_packet(&time, 6.0, 3.0, 1e-4);
        let dt = time[1] - time[0];
        for k in 2..time.len() - 2 {
            if time[k] > 0.1 && time[k] < 5.9 {
                // A five-point stencil, on the phasors.
                let numerical = (-h[k - 2] + h[k - 1] * 16.0 - h[k] * 30.0 + h[k + 1] * 16.0
                    - h[k + 2])
                    * (1.0 / (12.0 * dt * dt));
                assert!(close(numerical, a[k], 1e-11));
            }
        }
        // Outside the window only the Gaussian's tails remain.
        for (k, t) in time.iter().enumerate() {
            if *t <= 0.0 || *t >= 6.0 {
                assert!(largest(h[k]) <= 1e-11 && largest(a[k]) <= 1e-9);
            }
        }
        assert!(close(h[time.len() / 2], Phasor::new(1e-4, 0.0), 1e-16));
    }

    /// The independent prediction: half the strain applied to the separation, for the three
    /// polarizations, from the packet's envelope and carrier.
    fn expected(time: &[f64], reference: &[V]) -> Vec<Vec<Vec<V>>> {
        (0..3)
            .map(|p| {
                time.iter()
                    .map(|t| {
                        let envelope = 1e-4 * (-(t - 3.0).powi(2) / (2.0 * 0.25)).exp();
                        let hp = envelope * (core::f64::consts::PI * (t - 3.0)).cos();
                        let hc = envelope * (core::f64::consts::PI * (t - 3.0)).sin();
                        let (hp, hc) = [(hp, 0.0), (0.0, hc), (hp, hc)][p];
                        reference
                            .iter()
                            .map(|r| {
                                let (rx, ry) = (r.e1(), r.e2());
                                let (dx, dy) = (hp * rx + hc * ry, hc * rx - hp * ry);
                                Vector::new(0.0, 0.5 * dx, 0.5 * dy, 0.0)
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn integrated_detector_response_converges_to_weak_wave_displacements() {
        let reference = detector_ring(12);
        let mut errors = vec![];
        let mut last = None;
        for count in [321, 641] {
            let time = linspace(-1.0, 7.0, count);
            let acceleration = accelerations(&time, 3.0, &reference);
            let displacement = integrate(&time, &acceleration);
            let prediction = expected(&time, &reference);
            let mut error = 0.0f64;
            for p in 0..3 {
                for k in 0..count {
                    for b in 0..reference.len() {
                        let d = displacement[p][k][b];
                        error = error.max(d.max_abs_diff(&prediction[p][k][b]));
                    }
                }
            }
            errors.push(error);
            last = Some((time, acceleration, displacement));
        }
        // Fourth-order integration convergence.
        assert!(errors[1] < errors[0] / 10.0, "{errors:?}");
        assert!(errors[1] < 2e-9, "{errors:?}");
        let (time, acceleration, displacement) = last.expect("ran");
        let dt = time[1] - time[0];
        for p in 0..3 {
            for (k, t) in time.iter().enumerate() {
                for b in 0..reference.len() {
                    let d = displacement[p][k][b];
                    // At rest before the packet, and after it, to within the Gaussian's tails.
                    if *t <= 0.0 {
                        assert!(largest(d) <= 1e-11);
                    }
                    if *t >= 6.0 {
                        assert!(largest(d) <= 2e-10);
                        if k + 1 < time.len() {
                            let rate = (displacement[p][k + 1][b] - d) * (1.0 / dt);
                            assert!(largest(rate) <= 1e-10);
                        }
                    }
                }
            }
        }
        // Stopping the calculation halfway through the pulse keeps the nonzero displacement:
        // the integrator does not reset its final frame.
        let middle = time.len() / 2;
        let prefix_accel: Vec<Vec<Vec<V>>> = acceleration
            .iter()
            .map(|per_time| per_time[..=middle].to_vec())
            .collect();
        let prefix = integrate(&time[..=middle], &prefix_accel);
        let mut biggest = 0.0f64;
        for p in 0..3 {
            for b in 0..reference.len() {
                let (a, d) = (prefix[p][middle][b], displacement[p][middle][b]);
                assert!(close(a, d, 1e-15));
                biggest = biggest.max(largest(a));
            }
        }
        assert!(biggest > 4e-5);
    }

    #[test]
    fn strain_map_predicts_the_ring_and_its_second_derivative_is_the_curvature() {
        let (plus_strain, cross_strain) = strain_patterns();
        // Stretch along x, squeeze along y, nothing along time or the wave.
        for (separation, image) in [(x(), x()), (y(), -y()), (t(), t() * 0.0), (z(), z() * 0.0)] {
            assert!(close(plus_strain.of(separation), image, 1e-15));
        }
        let time = linspace(-1.0, 7.0, 641);
        let (strain, second) = wave_packet(&time, 6.0, 3.0, 1e-4);
        let reference = detector_ring(12);
        let displacement = integrate(&time, &accelerations(&time, 3.0, &reference));
        let (plus, _) = polarizations();
        let edge = vector(&mut rng(5));
        let mut biggest = 0.0f64;
        for k in 0..time.len() {
            let predicted = polarized_strain(plus_strain, cross_strain, strain[k]);
            let waves = polarized_waves(plus, second[k]);
            let acceleration_map = polarized_strain(plus_strain, cross_strain, second[k]);
            for p in 0..3 {
                for (b, r) in reference.iter().enumerate() {
                    let pr = predicted[p].of(*r);
                    assert!(close(displacement[p][k][b], pr, 2e-9));
                    biggest = biggest.max(largest(pr));
                }
                // The tidal map is the strain's second time derivative as a map on separations.
                assert!(close(tidal_map(waves[p], t()), acceleration_map[p], 1e-12));
                // Wedged with the wave vector, that second derivative is the curvature on pairs
                // of vectors.
                let two_form = curvature_of_strain(t() + z(), acceleration_map[p]);
                let bound = two_form.of(edge);
                let direct = waves[p].of(edge ^ Vector::slot());
                assert!(close(bound, direct, 1e-12));
            }
        }
        assert!(biggest > 4e-5);
    }

    /// The scenario checks: the curvature is nonzero and squares to zero, and the integrated
    /// ring lands on the strain map applied to the rest separations.
    #[test]
    fn scenarios_pass_their_checks() {
        let (plus, _) = polarizations();
        assert!(largest(plus.of(plus)) <= 1e-14);
        assert!(largest(plus) > 0.0);
        let d = detector_scenario();
        let (plus_strain, cross_strain) = strain_patterns();
        for (k, s) in d.strain.iter().enumerate() {
            let predicted = polarized_strain(plus_strain, cross_strain, *s);
            for p in 0..3 {
                for (b, r) in d.reference.iter().enumerate() {
                    assert!(close(d.displacement[p][k][b], predicted[p].of(*r), 1e-12));
                }
            }
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
