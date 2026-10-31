//! numga's `optics/specular_caustics`: the bright net of light on the floor of a pool, the
//! refraction of a rippled water surface mapped onto a flat receiver (VGA3D).
//!
//! Positions and ray directions are intrinsic antivectors, bivectors in VGA3D; the ripples'
//! wavevectors and the water's planes extrinsic vectors, read against a position by pairing,
//! `wavevector & position`. Horizontal places, on the water and on the receiver, are the
//! antivectors that contain `z`, gax's `Tilt` taken intrinsic, and the wavevectors are
//! `Horizontal`, its complement. The surface is the level set of its height above the ripples,
//! so its plane at a point is `z − gradient`, and a vertical ray refracts across it by Snell's
//! law: the part along the surface scaled by the ratio of the refractive indices, unit length
//! fixing the part across it.
//!
//! Where the ray map from the water to the receiver folds over, many rays pile up on a line: a
//! caustic. Its derivative, a map `Tilt <- Tilt` from forward-mode jets on the two coordinates
//! of a place (`jet::jacobian`), gives each patch of water the area it lands on, and the folds
//! are where that area changes sign. Newton's method on the same derivative finds the three
//! points of a single ripple that all send their rays to one point of the floor.
//!
//! The animation shows the light on the floor as four travelling waves cross the water (left),
//! the folds of the ray map carried onto the floor over its light (middle), and, in a vertical
//! section through a single ripple, a fan of rays and the three paths to one point (right).

use gax::vga3d::{Vector, intrinsic};
use gax_colour::Srgb;
use gax_numga_examples::{
    Anim, Axes, Canvas, Light, Marker, Point2, Rect, backdrop, caption, colormap, label_above,
    palette, par_map, run_with,
};

mod caustics {
    use gax::Real;
    use gax::vga3d::{Horizontal, Vector, intrinsic};
    use gax_numga_examples::signal::phasor;

    /// A position or a ray's direction: an intrinsic bivector.
    pub type Direction<T = f64> = intrinsic::Bivector<(), T>;
    /// A horizontal place: the directions that contain `z`.
    pub type Place<T = f64> = intrinsic::Tilt<(), T>;
    /// A plane: an extrinsic vector.
    pub type Plane<T = f64> = Vector<(), T>;

    /// Travelling waves on the water: their wavevectors, amplitudes and phases.
    #[derive(Clone)]
    pub struct Waves {
        pub wavevectors: Vec<Horizontal>,
        pub amplitudes: Vec<f64>,
        pub phases: Vec<f64>,
    }

    /// The plane across `z`.
    pub fn z<T: Real>() -> Plane<T> {
        Vector::new(T::zero(), T::zero(), T::one())
    }

    /// The direction straight up, the complement of the plane across `z`.
    pub fn up<T: Real>() -> Direction<T> {
        z::<T>().dual()
    }

    /// The surface position above a place, the unit plane of the surface there and the
    /// transmitted direction, for vertical illumination.
    pub fn rays<T: Real>(
        place: Place<T>,
        waves: &Waves,
        index_ratio: f64,
    ) -> (Direction<T>, Plane<T>, Direction<T>) {
        let mut height = T::zero();
        let mut gradient = Horizontal::<(), T>::zero();
        for ((k, a), p) in waves
            .wavevectors
            .iter()
            .zip(&waves.amplitudes)
            .zip(&waves.phases)
        {
            let k = k.map_coefs(T::from_f64);
            let phasor = phasor((k & place).s() - T::from_f64(*p));
            let a = T::from_f64(*a);
            height = height + a * phasor.e20();
            gradient -= k * (a * phasor.e01());
        }
        // The plane of the surface, `z − height == 0`, and the unit direction across it.
        let plane = (z::<T>() - Vector::from(gradient)).normalized();
        let normal = plane.dual();
        // Snell's law scales the part of the ray along the surface; unit length fixes the part
        // across it.
        let incoming = -up::<T>();
        let tangent = (incoming - normal * (plane & incoming).s()) * T::from_f64(index_ratio);
        let across = (T::one() - tangent.norm_squared()).sqrt();
        let outgoing = tangent - normal * across;
        let surface = place.cast::<intrinsic::Bivector>() + up::<T>() * height;
        (surface, plane, outgoing)
    }

    /// The place on the receiver, at the given depth below the water, where the ray from the
    /// water above a place lands.
    pub fn receiver<T: Real>(
        place: Place<T>,
        waves: &Waves,
        depth: f64,
        index_ratio: f64,
    ) -> Place<T> {
        let (surface, _, outgoing) = rays(place, waves, index_ratio);
        // The distance along the ray down to the receiver, at height minus the depth.
        let distance = (T::from_f64(-depth) - (z::<T>() & surface).s()) / (z::<T>() & outgoing).s();
        (surface + outgoing * distance).cast::<intrinsic::Tilt>()
    }

    /// The landing of a place and the ray map's derivative there, `Tilt <- Tilt`.
    pub fn landing_and_derivative(
        place: Place,
        waves: &Waves,
        depth: f64,
        index_ratio: f64,
    ) -> (Place, intrinsic::Tilt<(intrinsic::Tilt,)>) {
        gax::jet::jacobian(|p| receiver(p, waves, depth, index_ratio), place)
    }

    // --- scenes

    /// Four travelling waves: their headings, wavenumbers, amplitudes, phases and phase rates.
    pub const WAVE_ANGLES: [f64; 4] = [0.0, 0.7, 1.5, 2.2];
    pub const WAVENUMBERS: [f64; 4] = [6.0, 8.0, 10.0, 12.0];
    pub const AMPLITUDES: [f64; 4] = [0.045, 0.035, 0.025, 0.018];
    pub const PHASES: [f64; 4] = [0.0, 1.2, -0.7, 0.6];
    pub const PHASE_RATES: [f64; 4] = [1.0, -1.0, 2.0, -2.0];
    /// The ratio of the refractive indices of air and water, and the depth of the receiver
    /// below the surface.
    pub const INDEX_RATIO: f64 = 0.75;
    pub const DEPTH: f64 = 2.5;
    /// The square of water the rays enter through, and the square of the receiver drawn.
    pub const SOURCE_HALFWIDTH: f64 = 1.6;
    pub const RECEIVER_HALFWIDTH: f64 = 1.0;
    /// One ray in each cell of a grid over the water, a coarser grid for the map's derivative,
    /// and the frames of the waves' period.
    pub const N_RAYS: usize = 640;
    pub const N_DERIVATIVE: usize = 192;
    pub const N_FRAMES: usize = 126;
    pub const SEED: u64 = 17;
    /// A single ripple across x, for a side view of three paths to one receiver point.
    pub const RIPPLE_WAVENUMBER: f64 = 4.0;
    pub const RIPPLE_AMPLITUDE: f64 = 0.12;
    pub const SEED_OFFSETS: [f64; 3] = [-0.3, 0.0, 0.3];
    pub const SECTION_HALFWIDTH: f64 = 0.8;
    pub const N_SECTION: usize = 300;
    pub const N_FAN: usize = 45;
    pub const N_ITERATIONS: usize = 3;

    /// The place at `(x, y)`: the steps along `x` and `y`, the complements of the planes across
    /// them.
    pub fn place(x: f64, y: f64) -> Place {
        (Vector::new(1.0, 0.0, 0.0).dual() * x + Vector::new(0.0, 1.0, 0.0).dual() * y)
            .cast::<intrinsic::Tilt>()
    }

    /// The wavevectors of the four waves: `x` turned to each heading, times its wavenumber.
    pub fn wavevectors() -> Vec<Horizontal> {
        let x = Vector::new(1.0, 0.0, 0.0);
        let xy = Vector::new(1.0, 0.0, 0.0) ^ Vector::new(0.0, 1.0, 0.0);
        WAVE_ANGLES
            .iter()
            .zip(WAVENUMBERS)
            .map(|(a, k)| ((xy * (-a / 2.0)).exp() >> x).cast::<Horizontal>() * k)
            .collect()
    }

    /// The four waves at a time of their period.
    pub fn waves_at(time: f64) -> Waves {
        Waves {
            wavevectors: wavevectors(),
            amplitudes: AMPLITUDES.to_vec(),
            phases: PHASES
                .iter()
                .zip(PHASE_RATES)
                .map(|(p, r)| p + r * time)
                .collect(),
        }
    }

    /// The single ripple across `x`.
    pub fn ripple() -> Waves {
        Waves {
            wavevectors: vec![Horizontal::new(RIPPLE_WAVENUMBER, 0.0)],
            amplitudes: vec![RIPPLE_AMPLITUDE],
            phases: vec![0.0],
        }
    }

    /// One jittered ray per cell over the water, row by row, and the area each ray carries.
    pub fn water() -> (Vec<Vec<Place>>, f64) {
        use gax_numga_examples::rng::{Draw, rng};
        let spacing = 2.0 * SOURCE_HALFWIDTH / N_RAYS as f64;
        let sample = |k: usize| (k as f64 + 0.5) * spacing - SOURCE_HALFWIDTH;
        let mut r = rng(SEED);
        // A fixed jitter within each cell hides the sampling lattice.
        let mut jitter = || r.range(-0.5, 0.5) * spacing;
        let rows = (0..N_RAYS)
            .map(|row| {
                (0..N_RAYS)
                    .map(|column| place(sample(column) + jitter(), sample(row) + jitter()))
                    .collect()
            })
            .collect();
        (rows, spacing * spacing)
    }

    /// The landings of a grid over the water and the area the ray map gives each patch of it
    /// (the determinant of its derivative), row by row.
    pub fn folds(waves: &Waves) -> (Vec<Place>, Vec<f64>) {
        let n = N_DERIVATIVE;
        let sample =
            |k: usize| -SOURCE_HALFWIDTH + 2.0 * SOURCE_HALFWIDTH * k as f64 / (n - 1) as f64;
        (0..n * n)
            .map(|k| {
                let p = place(sample(k % n), sample(k / n));
                let (landing, derivative) = landing_and_derivative(p, waves, DEPTH, INDEX_RATIO);
                (landing, derivative.det())
            })
            .unzip()
    }

    /// Under a single ripple, the three points of the water whose rays reach the receiver point
    /// below the origin, by Newton's method on the ray map from three trial points.
    pub fn three_paths() -> [Place; 3] {
        let waves = ripple();
        SEED_OFFSETS.map(|offset| {
            let mut trial = place(offset, 0.0);
            for _ in 0..N_ITERATIONS {
                // The local ray map turns the miss at the receiver into a correction on the
                // water.
                let (landing, derivative) =
                    landing_and_derivative(trial, &waves, DEPTH, INDEX_RATIO);
                trial -= derivative.solve(landing);
            }
            trial
        })
    }
}

use caustics::*;

/// The loop: numga's 126 frames of 40 ms, one period of the waves.
const SECONDS: f32 = 5.04;
/// The receiver's pixels, the blur of each ray's footprint in pixels, and the exposure.
const N_PIXELS: usize = 300;
const BLUR_PIXELS: f64 = 0.95;
const EXPOSURE: f64 = 0.3;
/// The height the rays come down from in the section.
const INCIDENT_HEIGHT: f64 = 0.6;

/// numga's colours of the light on the floor, from dark water blue to warm bright folds.
fn floor_light(brightness: f32) -> Light {
    colormap::stops(
        brightness,
        &[
            Srgb::rgb(0.012, 0.063, 0.118),
            Srgb::rgb(0.027, 0.2, 0.294),
            Srgb::rgb(0.09, 0.42, 0.506),
            Srgb::rgb(0.443, 0.733, 0.749),
            Srgb::rgb(1.0, 0.957, 0.847),
        ],
    )
}

/// A place read by the planes across `x` and `y`.
fn coordinates(p: Place) -> [f64; 2] {
    let read = |v: Vector| (v & p).s();
    [
        read(Vector::new(1.0, 0.0, 0.0)),
        read(Vector::new(0.0, 1.0, 0.0)),
    ]
}

/// A point of the vertical section, read by the planes across `x` and `z`.
fn section(p: Direction) -> Point2 {
    let read = |v: Vector| (v & p).s() as f32;
    Point2::xy(
        read(Vector::new(1.0, 0.0, 0.0)),
        read(Vector::new(0.0, 0.0, 1.0)),
    )
}

/// Power per receiver area for equal-power rays, each spread over a Gaussian footprint, on the
/// receiver's pixels (row by row, from the bottom), normalized to the brightness shown.
fn irradiance(landings: &[Place], ray_area: f64) -> Vec<f32> {
    let pixel = 2.0 * RECEIVER_HALFWIDTH / N_PIXELS as f64;
    // Trace beyond the visible window, so a footprint crossing its edge still contributes.
    let pad = (4.0 * BLUR_PIXELS).ceil() as usize;
    let n = N_PIXELS + 2 * pad;
    let half = RECEIVER_HALFWIDTH + pad as f64 * pixel;
    let mut counts = vec![0.0; n * n];
    for p in landings {
        let [x, y] = coordinates(*p);
        let (i, j) = ((x + half) / pixel, (y + half) / pixel);
        if i >= 0.0 && j >= 0.0 && (i as usize) < n && (j as usize) < n {
            counts[j as usize * n + i as usize] += ray_area / (pixel * pixel);
        }
    }
    // scipy's `gaussian_filter`, truncated at four deviations, with zeros outside.
    let radius = (4.0 * BLUR_PIXELS + 0.5) as isize;
    let kernel: Vec<f64> = (-radius..=radius)
        .map(|k| (-0.5 * (k as f64 / BLUR_PIXELS).powi(2)).exp())
        .collect();
    let total: f64 = kernel.iter().sum();
    let blur = |values: &[f64], step: usize, along: usize| {
        let mut out = vec![0.0; n * n];
        for (k, o) in out.iter_mut().enumerate() {
            let at = (k / step) % n;
            *o = (-radius..=radius)
                .filter_map(|d| {
                    let to = at as isize + d;
                    (0..n as isize).contains(&to).then(|| {
                        let index = (k as isize + d * along as isize) as usize;
                        values[index] * kernel[(d + radius) as usize] / total
                    })
                })
                .sum();
        }
        out
    };
    let across = blur(&counts, 1, 1);
    let both = blur(&across, n, n);
    (0..N_PIXELS * N_PIXELS)
        .map(|k| {
            let (i, j) = (k % N_PIXELS + pad, k / N_PIXELS + pad);
            (-(-EXPOSURE * both[j * n + i]).exp_m1()) as f32
        })
        .collect()
}

/// The landings of every ray on the receiver at a time of the waves' period.
fn landings(water: &[Vec<Place>], time: f64) -> Vec<Place> {
    let waves = waves_at(time);
    par_map(water, |row| {
        row.iter()
            .map(|p| receiver(*p, &waves, DEPTH, INDEX_RATIO))
            .collect::<Vec<_>>()
    })
    .into_iter()
    .flatten()
    .collect()
}

struct Scenes {
    water: Vec<Vec<Place>>,
    ray_area: f64,
    /// The light at the start and the folds carried onto the receiver.
    still: Vec<f32>,
    folds: Vec<[Point2; 2]>,
    /// The section: the surface, the fan's paths, the three paths and the target.
    surface: Vec<Point2>,
    fan: Vec<[Point2; 3]>,
    paths: Vec<[Point2; 3]>,
    target: Point2,
}

/// The landings at a place between the samples of the chart over the water, in fractional
/// columns and rows: the four samples around it, each weighted by its bilinear share.
fn between_samples(chart: &[Place], at: Point2) -> Place {
    let n = N_DERIVATIVE;
    let [u, v] = at.to_euclidean().map(f64::from);
    let (i, j) = (
        (u.floor() as usize).min(n - 2),
        (v.floor() as usize).min(n - 2),
    );
    let (s, t) = (u - i as f64, v - j as f64);
    chart[j * n + i] * ((1.0 - s) * (1.0 - t))
        + chart[j * n + i + 1] * (s * (1.0 - t))
        + chart[(j + 1) * n + i] * ((1.0 - s) * t)
        + chart[(j + 1) * n + i + 1] * (s * t)
}

fn scenes() -> Scenes {
    let (water, ray_area) = water();
    let still = irradiance(&landings(&water, 0.0), ray_area);
    let (chart, areas) = folds(&waves_at(0.0));
    let areas: Vec<f32> = areas.iter().map(|a| *a as f32).collect();
    let to_floor = |q: Point2| {
        let [x, y] = coordinates(between_samples(&chart, q));
        Point2::xy(x as f32, y as f32)
    };
    let folds = gax_numga_examples::contour::segments(&areas, N_DERIVATIVE, N_DERIVATIVE, 0.0)
        .into_iter()
        .map(|[a, b]| [to_floor(a), to_floor(b)])
        .collect();
    // The section through the single ripple.
    let ripple = ripple();
    let along = |k: usize, n: usize| {
        -SECTION_HALFWIDTH + 2.0 * SECTION_HALFWIDTH * k as f64 / (n - 1) as f64
    };
    let surface = (0..N_SECTION)
        .map(|k| section(rays(place(along(k, N_SECTION), 0.0), &ripple, INDEX_RATIO).0))
        .collect();
    let floor = up::<f64>() * DEPTH;
    // Each ray comes straight down from the incident height to where it crosses the surface,
    // and on to the receiver.
    let path = |p: Place, end: Direction| {
        let (at, _, _) = rays(p, &ripple, INDEX_RATIO);
        let from = at + up::<f64>() * (INCIDENT_HEIGHT - (z::<f64>() & at).s());
        [section(from), section(at), section(end)]
    };
    let fan = (0..N_FAN)
        .map(|k| {
            let p = place(along(k, N_FAN), 0.0);
            let end =
                receiver(p, &ripple, DEPTH, INDEX_RATIO).cast::<intrinsic::Bivector>() - floor;
            path(p, end)
        })
        .collect();
    let target = place(0.0, 0.0).cast::<intrinsic::Bivector>() - floor;
    let paths = three_paths().iter().map(|p| path(*p, target)).collect();
    Scenes {
        water,
        ray_area,
        still,
        folds,
        surface,
        fan,
        paths,
        target: section(target),
    }
}

/// The light on the receiver, as an image over its square.
fn draw_light(c: &mut Canvas, ax: &Axes, light: &[f32]) {
    let half = RECEIVER_HALFWIDTH as f32;
    ax.image(c, 1, |q| {
        let [x, y] = q.to_euclidean();
        let (i, j) = ((x + half) / (2.0 * half), (y + half) / (2.0 * half));
        if !(0.0..1.0).contains(&i) || !(0.0..1.0).contains(&j) {
            return None;
        }
        let (i, j) = (
            (i * N_PIXELS as f32) as usize,
            (j * N_PIXELS as f32) as usize,
        );
        Some(floor_light(light[j * N_PIXELS + i]))
    });
}

/// The vertical section through the single ripple: the water, a fan of rays, the surface, the
/// three paths to one point and the receiver.
fn draw_section(c: &mut Canvas, rect: Rect, s: &Scenes) {
    let lower = s
        .fan
        .iter()
        .map(|p| p[2].to_euclidean()[1])
        .fold(f32::MAX, f32::min);
    let (left, right) = (-SECTION_HALFWIDTH as f32, SECTION_HALFWIDTH as f32);
    let corners = [
        Point2::xy(left, lower),
        Point2::xy(right, INCIDENT_HEIGHT as f32),
    ];
    let ax = Axes::fitting(rect, corners, 1.02);
    let mut water = s.surface.clone();
    water.push(Point2::xy(right, lower));
    water.push(Point2::xy(left, lower));
    ax.fill(c, &water, palette::BLUE.faded(0.3), 0.5);
    for p in &s.fan {
        ax.polyline(c, p, 0.6, palette::INK.faded(0.3));
    }
    ax.polyline(c, &s.surface, 1.4, palette::SKY);
    for p in &s.paths {
        ax.polyline(c, p, 1.6, palette::ORANGE);
    }
    ax.line(
        c,
        Point2::xy(left, lower),
        Point2::xy(right, lower),
        1.2,
        palette::INK.faded(0.6),
    );
    ax.scatter(
        c,
        &[s.target],
        Marker::Dot,
        6.0,
        palette::ORANGE.whitened(0.3),
    );
}

fn draw(c: &mut Canvas, s: &Scenes, t: f32) {
    backdrop(c);
    let frame = gax_numga_examples::signal::frame((t / SECONDS).rem_euclid(1.0), N_FRAMES);
    let time = core::f64::consts::TAU * frame as f64 / N_FRAMES as f64;
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let top = h * 0.2;
    let side = (h - top - 12.0).min(w * 0.39);
    let gap = w * 0.015;
    let square = |x0: f32| Rect::new(x0, top, x0 + side, top + side);
    let (moving, folded) = (square(gap), square(2.0 * gap + side));
    let half = RECEIVER_HALFWIDTH as f32;
    let ax = Axes::equal(moving, Point2::xy(0.0, 0.0), half);
    draw_light(c, &ax, &irradiance(&landings(&s.water, time), s.ray_area));
    label_above(c, moving, "THE LIGHT ON THE FLOOR", 11.0, 6.0);
    let ax = Axes::equal(folded, Point2::xy(0.0, 0.0), half);
    draw_light(c, &ax, &s.still);
    ax.stroke(c, &s.folds, 0.9, palette::RED.whitened(0.35));
    label_above(c, folded, "FOLDS OF THE RAY MAP", 11.0, 6.0);
    let rest = Rect::new(3.0 * gap + 2.0 * side, top, w - gap, top + side);
    draw_section(c, rest, s);
    label_above(c, rest, "THREE PATHS", 11.0, 6.0);
    caption(
        c,
        "SPECULAR CAUSTICS: LIGHT UNDER RIPPLED WATER",
        "SNELL ACROSS THE PLANE z - GRADIENT; FOLDS WHERE THE JACOBIAN Tilt <- Tilt CHANGES SIGN (VGA3D)",
    );
}

fn main() {
    run_with(
        Anim::new("specular_caustics", SECONDS).size(960, 540),
        scenes,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::caustics::*;
    use gax::vga3d::{Horizontal, Vector, intrinsic};

    const ROUND_OFF: f64 = 1e-11;

    fn three_waves() -> Waves {
        let x = Vector::new(1.0, 0.0, 0.0);
        let xy = x ^ Vector::new(0.0, 1.0, 0.0);
        Waves {
            wavevectors: [0.0, 1.1, 2.2]
                .iter()
                .zip([3.1, 4.4, 6.0])
                .map(|(a, k): (&f64, f64)| ((xy * (-a / 2.0)).exp() >> x).cast::<Horizontal>() * k)
                .collect(),
            amplitudes: vec![0.11, 0.065, 0.035],
            phases: vec![0.0, 1.2, -0.7],
        }
    }

    fn places() -> Vec<Place> {
        [(-0.6, 0.3), (-0.1, -0.5), (0.2, 0.4), (0.7, -0.2)]
            .iter()
            .map(|(x, y)| place(*x, *y))
            .collect()
    }

    /// The surface plane is unit, the transmitted ray unit, its part along the surface the
    /// incoming's scaled by the ratio of the indices, it lands on its own line, going down.
    #[test]
    fn refraction_obeys_snell_and_reaches_the_receiver() {
        let waves = three_waves();
        let incoming = -up::<f64>();
        for p in places() {
            let (surface, plane, outgoing) = rays(p, &waves, 0.75);
            let landing =
                receiver(p, &waves, 2.5, 0.75).cast::<intrinsic::Bivector>() - up::<f64>() * 2.5;
            let normal = plane.dual();
            assert!((plane.norm_squared() - 1.0).abs() < ROUND_OFF);
            assert!((outgoing.norm_squared() - 1.0).abs() < ROUND_OFF);
            let snell = (outgoing - normal * (plane & outgoing).s())
                - (incoming - normal * (plane & incoming).s()) * 0.75;
            assert!(snell.c.iter().all(|c| c.abs() < ROUND_OFF), "{snell:?}");
            let off = (landing - surface) & outgoing;
            assert!(off.c.iter().all(|c| c.abs() < ROUND_OFF), "{off:?}");
            assert!((z::<f64>() & outgoing).s() < 0.0);
        }
    }

    /// A flat surface sends every ray straight down: the map is the identity, its derivative
    /// too, of unit area.
    #[test]
    fn flat_surface_preserves_positions_and_area() {
        let flat = Waves {
            wavevectors: vec![Horizontal::new(4.0, 0.0)],
            amplitudes: vec![0.0],
            phases: vec![0.0],
        };
        for p in places() {
            let (landing, local) = landing_and_derivative(p, &flat, 2.5, 0.75);
            assert!((landing - p).c.iter().all(|c| c.abs() < ROUND_OFF));
            assert!((local.of(p) - p).c.iter().all(|c| c.abs() < ROUND_OFF));
            assert!((local.det() - 1.0).abs() < ROUND_OFF);
        }
    }

    /// The derivative predicts the change of the landing under small steps of the place.
    #[test]
    fn local_map_matches_changes_in_ray_landing() {
        let waves = three_waves();
        let step = 1e-4;
        for p in places() {
            let (_, local) = landing_and_derivative(p, &waves, 2.5, 0.75);
            for angle in [0.2, 0.9, 1.8] {
                let xy = Vector::new(1.0, 0.0, 0.0) ^ Vector::new(0.0, 1.0, 0.0);
                let d = ((xy * (-angle / 2.0)).exp() >> place(1.0, 0.0)).cast::<intrinsic::Tilt>();
                let forward = receiver(p + d * step, &waves, 2.5, 0.75);
                let backward = receiver(p - d * step, &waves, 2.5, 0.75);
                let measured = (forward - backward) * (1.0 / (2.0 * step));
                let error = local.of(d) - measured;
                assert!(error.c.iter().all(|c| c.abs() < 1e-6), "{error:?}");
            }
        }
    }

    /// Under a single ripple three points of the water, symmetric about the crest, send their
    /// rays to the one receiver point below it; the middle one's patch is turned over.
    #[test]
    fn three_refracted_paths_reach_one_receiver_point() {
        let paths = three_paths();
        let waves = ripple();
        for p in &paths {
            let landing = receiver(*p, &waves, DEPTH, INDEX_RATIO);
            assert!(landing.c.iter().all(|c| c.abs() < 1e-6), "{landing:?}");
        }
        assert!((paths[0] + paths[2]).c.iter().all(|c| c.abs() < ROUND_OFF));
        assert!(paths[1].c.iter().all(|c| c.abs() < 1e-9));
        let x = Vector::new(1.0, 0.0, 0.0);
        assert!((x & paths[0]).s() < -0.2 && (x & paths[2]).s() > 0.2);
        let area = |p: Place| {
            landing_and_derivative(p, &waves, DEPTH, INDEX_RATIO)
                .1
                .det()
        };
        assert!(area(paths[1]) < 0.0 && area(paths[0]) > 0.0 && area(paths[2]) > 0.0);
    }

    /// The ray map of the four waves folds over part of the water.
    #[test]
    fn the_ray_map_folds() {
        let (_, areas) = folds(&waves_at(0.0));
        let (lo, hi) = areas
            .iter()
            .fold((f64::MAX, f64::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
        assert!(lo < 0.0 && 0.0 < hi, "{lo} {hi}");
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scenes(), 1.0);
    }
}
