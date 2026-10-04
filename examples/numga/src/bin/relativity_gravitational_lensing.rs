//! numga's `relativity/gravitational_lensing`: point masses bending light, in the geometric
//! algebra of the plane of the sky (VGA2D). A direction on the sky is a vector, in units of the
//! Einstein angle of the total mass. Each mass pulls a sightline towards itself by the inverse
//! of the separation, weighted by its mass, so the source direction a sightline reaches is
//! `observed - Σ m (observed - p)⁻¹`.
//!
//! The derivative of that deflection is a map on vectors, built with an open slot: the
//! identity plus each mass's reflection sandwich `s⁻¹ x s⁻¹`, the change of a vector's inverse.
//! Its outermorphism on the pseudoscalar is the area ratio, zero on the critical curve; the
//! lens carries that curve to the caustic. The animation slides a small round source behind
//! two equal masses: crossing the caustic, it gains or loses a pair of images, and the light
//! the sky shows (the magnification) jumps. On the right, small round sources as the local map
//! shows them: blue where the image keeps its orientation, red where it is mirrored.

use std::sync::OnceLock;

use gax_colour::Srgb;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, ORIGIN2, Point2, Rect, backdrop, caption, colormap,
    contour, palette, reach2, run,
};

mod lensing {
    use gax::vga2d::{Pseudoscalar, Vector};
    use gax_numga_examples::Point2;

    pub type V = Vector<(), f64>;
    /// A map from a small displacement of a sightline to the displacement it makes at the
    /// source.
    pub type LocalMap = Vector<(Vector,), f64>;

    /// A direction from its `x` and `y` components.
    pub fn v(x: f64, y: f64) -> V {
        Vector::new(x, y)
    }

    /// The direction at a point of a drawing: the point's place on the sky.
    pub fn at(p: Point2) -> V {
        let [x, y] = p.to_euclidean();
        v(f64::from(x), f64::from(y))
    }

    /// The source direction each observed direction reaches.
    pub fn deflected(observed: V, positions: &[V], masses: &[f64]) -> V {
        positions
            .iter()
            .zip(masses)
            .fold(observed, |acc, (p, m)| acc - (observed - *p).inverse() * *m)
    }

    /// The map from a small displacement of the observed direction to the displacement it makes
    /// at the source: the identity plus each mass's reflection in the line of its separation,
    /// scaled by the inverse square distance (the sandwich by the separation's inverse).
    pub fn local_map(observed: V, positions: &[V], masses: &[f64]) -> LocalMap {
        positions
            .iter()
            .zip(masses)
            .fold(Vector::slot(), |acc, (p, m)| {
                acc + ((observed - *p).inverse() >> Vector::slot()) * *m
            })
    }

    /// The ratio the local map carries areas by: its outermorphism on the pseudoscalar (its
    /// determinant).
    pub fn area(local: LocalMap) -> f64 {
        local
            .outermorphism::<Pseudoscalar>()
            .of(Pseudoscalar::new(1.0))
            .e12()
    }

    /// A round Gaussian source of unit peak brightness and the given angular standard deviation.
    pub fn brightness(direction: V, centre: V, width: f64) -> f64 {
        let offset = direction - centre;
        (-offset.norm_squared() / (2.0 * width * width)).exp()
    }

    /// Directions at the centres of a square of pixels about the optical axis, row by row
    /// (rows along `y`).
    pub fn sky(half_width: f64, resolution: usize) -> Vec<V> {
        let angle =
            |i: usize| (i as f64 + 0.5) * (2.0 * half_width / resolution as f64) - half_width;
        (0..resolution * resolution)
            .map(|k| v(angle(k % resolution), angle(k / resolution)))
            .collect()
    }

    // --- the binary scene ---------------------------------------------------------------

    /// Two equal masses on the x axis, 1.1 Einstein angles apart: one caustic with six cusps.
    pub fn positions() -> [V; 2] {
        [v(-0.55, 0.0), v(0.55, 0.0)]
    }
    pub const MASSES: [f64; 2] = [0.5, 0.5];

    /// The source direction an observed direction reaches through the two masses.
    pub fn binary(observed: V) -> V {
        deflected(observed, &positions(), &MASSES)
    }

    /// The local map of the binary lens.
    pub fn binary_local(observed: V) -> LocalMap {
        local_map(observed, &positions(), &MASSES)
    }

    /// The sky's directions, the source direction each reaches, and the area ratio there.
    pub fn lens(resolution: usize) -> (Vec<V>, Vec<V>, Vec<f64>) {
        let directions = sky(1.9, resolution);
        let reached = directions.iter().map(|d| binary(*d)).collect();
        let areas = directions.iter().map(|d| area(binary_local(*d))).collect();
        (directions, reached, areas)
    }

    /// The offsets of a small round source of the given radius, 64 around.
    pub fn circle(radius: f64) -> Vec<V> {
        (0..64)
            .map(|k| {
                let angle = core::f64::consts::TAU * k as f64 / 63.0;
                (Pseudoscalar::new(-angle / 2.0).exp() >> v(1.0, 0.0)) * radius
            })
            .collect()
    }

    /// Small round sources behind a lattice of sightlines, as they appear on the sky: each
    /// circle of offsets carried by the inverse of the local map (a solve), about its
    /// sightline; with the area ratio there. Returns `(centre, outline, area)` per source.
    pub fn tissot(count: usize, radius: f64) -> Vec<(V, Vec<V>, f64)> {
        let spread = |i: usize| -1.25 + 2.5 * i as f64 / (count - 1) as f64;
        let offsets = circle(radius);
        (0..count * count)
            .map(|k| {
                let centre = v(spread(k % count), spread(k / count));
                let local = binary_local(centre);
                let seen = offsets.iter().map(|o| centre + local.solve(*o)).collect();
                (centre, seen, area(local))
            })
            .collect()
    }
}

use lensing::*;

/// Both lens panels show the directions within this distance of the optical axis.
const HALF: f32 = 1.4;
/// The source's angular width.
const WIDTH: f64 = 0.035;
/// The loop length in seconds: the source crosses and comes back.
const SECONDS: f32 = 8.0;

/// The colours of starlight, dark to bright (numga's map), with a power-law stretch.
fn starlight(t: f64) -> Light {
    let stretched = t.clamp(0.0, 1.0).powf(0.65) as f32;
    colormap::stops(
        stretched,
        &[
            Srgb::rgb(0.035, 0.051, 0.075),
            Srgb::rgb(0.255, 0.188, 0.263),
            Srgb::rgb(0.588, 0.376, 0.259),
            Srgb::rgb(0.918, 0.718, 0.447),
            Srgb::rgb(1.0, 0.949, 0.792),
        ],
    )
}

/// What the frames share: the critical curve and the caustic (as segments), the small round
/// sources, and a sky grid with the source direction each pixel reaches (for the magnification).
struct Scene {
    critical: Vec<[Point2; 2]>,
    caustic: Vec<[Point2; 2]>,
    tissot: Vec<(V, Vec<V>, f64)>,
    grid: (Vec<V>, Vec<V>),
}

fn scene() -> &'static Scene {
    static SCENE: OnceLock<Scene> = OnceLock::new();
    SCENE.get_or_init(|| {
        let ratio = |p: Point2| area(binary_local(at(p))) as f32;
        let critical = contour::of_fn(ratio, [-1.9, 1.9], [-1.9, 1.9], 380, 0.0);
        // The lens carries the critical curve to the caustic.
        let caustic = critical
            .iter()
            .map(|seg| seg.map(|p| reach2(binary(at(p)))))
            .collect();
        let (directions, reached, _) = lens(400);
        Scene {
            critical,
            caustic,
            tissot: tissot(11, 0.025),
            grid: (directions, reached),
        }
    })
}

/// The source's direction at time `t`: back and forth along a line just above the masses.
fn source_at(t: f32) -> V {
    let phase = f64::from(t / SECONDS) * core::f64::consts::TAU;
    v(-0.6 * phase.cos(), 0.12)
}

/// Axes on the sky in the largest square hanging from the middle of the top of `rect`.
fn square(rect: Rect) -> Axes {
    let side = rect.width().min(rect.height());
    let top_middle = rect.top_middle();
    let half = Point2::direction(side * 0.5, 0.0);
    let lo = top_middle - half;
    let hi = top_middle + half + Point2::direction(0.0, side);
    Axes::equal(Rect { lo, hi }, ORIGIN2, HALF)
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let s = scene();
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let critical_colour = Light::from_srgb(0.333, 0.796, 0.827, 1.6);
    let caustic_colour = Light::from_srgb(1.0, 0.53, 0.447, 1.6);
    let masses = positions().map(reach2);
    let centre = source_at(t);
    let size = (h / 34.0).clamp(7.0, 15.0);
    // Three columns between the caption and the notes below the panels.
    let row = Rect::new(0.0, h * 0.17, w, h - size * 3.2);
    let panel = |i: usize| square(row.column(i, 3).inset(w * 0.02, 0.0, w * 0.02, 0.0));
    // A note under a panel, from the middle of its bottom edge.
    let below = |c: &mut Canvas, ax: &Axes, dx: f32, text: &str, colour: Light, align: Align| {
        let bottom_middle = ax.rect.bottom_middle();
        let at = bottom_middle + Point2::direction(dx, size * 1.8);
        c.text(text, at, size, colour, align);
    };

    // The source plane: the source unlensed, with the caustic.
    let ax = panel(0);
    ax.image(c, 1, |p| Some(starlight(brightness(at(p), centre, WIDTH))));
    ax.stroke(c, &s.caustic, 1.3, caustic_colour);
    ax.frame(c, "SOURCE", "", "");
    below(c, &ax, 0.0, "CAUSTIC", caustic_colour, Align::Center);

    // The sky: each pixel shows the source's brightness where its sightline arrives.
    let ax = panel(1);
    ax.image(c, 2, |p| {
        Some(starlight(brightness(binary(at(p)), centre, WIDTH)))
    });
    ax.stroke(c, &s.critical, 1.1, critical_colour);
    ax.scatter(c, &masses, Marker::Ring, 10.0, palette::grid());
    ax.frame(c, "SKY", "", "");
    // The magnification: the light over the whole sky over the source's own.
    let (directions, reached) = &s.grid;
    let seen: f64 = reached.iter().map(|r| brightness(*r, centre, WIDTH)).sum();
    let own: f64 = directions
        .iter()
        .map(|d| brightness(*d, centre, WIDTH))
        .sum();
    let readout = format!("MAGNIFICATION {:.2}", seen / own);
    below(c, &ax, 0.0, &readout, palette::yellow(), Align::Center);

    // Small round sources, as seen.
    let ax = panel(2);
    let preserved = palette::sky();
    let reversed = Light::from_srgb(0.79, 0.41, 0.28, 1.7);
    for (_, outline, ratio) in &s.tissot {
        let colour = if *ratio >= 0.0 { preserved } else { reversed };
        let outline: Vec<Point2> = outline.iter().map(|d| reach2(*d)).collect();
        ax.fill(c, &outline, colour, 0.35);
        ax.polyline(c, &outline, 1.0, colour.faded(0.9));
    }
    ax.stroke(c, &s.critical, 1.0, palette::grid());
    ax.scatter(c, &masses, Marker::Ring, 10.0, palette::ink().faded(0.8));
    ax.frame(c, "SMALL ROUND SOURCES, AS SEEN", "", "");
    below(c, &ax, -size, "KEPT", preserved, Align::Right);
    below(c, &ax, size, "MIRRORED", reversed, Align::Left);
    caption(
        c,
        "GRAVITATIONAL LENSING: ONE STAR, SEVERAL IMAGES",
        "TWO EQUAL POINT MASSES (VGA2D): CRITICAL CURVE ON THE SKY, CAUSTIC AT THE SOURCE",
    );
}

fn main() {
    run(
        Anim::new("gravitational lensing", SECONDS).size(960, 420),
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::lensing::*;
    use gax::ApproxEq;
    use gax::vga2d::{Pseudoscalar, Vector};

    fn close_v(a: V, b: V, tol: f64) -> bool {
        a.max_abs_diff(&b) <= tol
    }

    /// One point mass: both images reach the source; the radial stretch is `1 + 1/r²` and the
    /// tangential one `1 - 1/r²`; the images have opposite orientations, and their light adds
    /// up to the analytic magnification.
    #[test]
    fn single_lens_images_have_the_analytic_stretches_orientation_and_magnification() {
        let (centre, one) = ([v(0.0, 0.0)], [1.0]);
        let source = v(0.3, 0.2);
        let radius = source.norm();
        let direction = source.normalized().into_inner();
        let root = (radius * radius + 4.0).sqrt();
        let image_radii = [(radius + root) / 2.0, (radius - root) / 2.0];
        let mut magnification = 0.0;
        let mut areas = [0.0; 2];
        for (k, r) in image_radii.into_iter().enumerate() {
            let image = direction * r;
            let local = local_map(image, &centre, &one);
            assert!(close_v(deflected(image, &centre, &one), source, 1e-13));
            assert!(close_v(
                local.of(direction),
                direction * (1.0 + 1.0 / (r * r)),
                1e-13
            ));
            let tangent: V = direction * Pseudoscalar::new(1.0);
            assert!(close_v(
                local.of(tangent),
                tangent * (1.0 - 1.0 / (r * r)),
                1e-13
            ));
            areas[k] = area(local);
            magnification += (1.0 / areas[k]).abs();
        }
        assert!(areas[0] > 0.0 && 0.0 > areas[1]);
        let expected = (radius * radius + 2.0) / (radius * root);
        assert!((magnification - expected).abs() < 1e-13);
    }

    /// The local map is the derivative of the deflection (a five-point stencil), and turns
    /// with the lens.
    #[test]
    fn local_map_is_the_derivative_and_turns_with_the_lens() {
        let positions = [v(-0.4, 0.1), v(0.6, -0.2), v(-0.1, 0.7)];
        let masses = [0.2, 0.5, 0.3];
        let observed = [v(-1.2, 0.8), v(0.3, -0.9), v(1.5, 1.2), v(-0.8, -1.4)];
        let small = [v(0.3, 0.8), v(-0.6, 0.1), v(0.4, -0.7), v(0.9, 0.2)];
        let step = 1e-4;
        let lens = |d: V| deflected(d, &positions, &masses);
        let turn = Pseudoscalar::<(), f64>::new(0.37).exp();
        for (o, s) in observed.into_iter().zip(small) {
            let numerical = (lens(o + s * (2.0 * step)) * -1.0 + lens(o + s * step) * 8.0
                - lens(o - s * step) * 8.0
                + lens(o - s * (2.0 * step)))
                * (1.0 / (12.0 * step));
            assert!(close_v(
                numerical,
                local_map(o, &positions, &masses).of(s),
                1e-9
            ));
            let turned_positions = positions.map(|p| turn >> p);
            let turned = local_map(turn >> o, &turned_positions, &masses).of(turn >> s);
            assert!(close_v(
                turned,
                turn >> local_map(o, &positions, &masses).of(s),
                1e-9
            ));
        }
    }

    /// Composite Simpson's rule with `n` (even) intervals: scipy's `quad` in numga's test.
    fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let h = (b - a) / n as f64;
        let inner: f64 = (1..n)
            .map(|i| f(a + h * i as f64) * if i % 2 == 1 { 4.0 } else { 2.0 })
            .sum();
        (f(a) + inner + f(b)) * h / 3.0
    }

    /// A finite source's light over the sky matches the analytic point-source magnification
    /// integrated over rings of the source.
    #[test]
    fn finite_source_light_integrates_to_the_radial_magnification() {
        let width = 0.15;
        let flux: f64 = sky(2.0, 128)
            .into_iter()
            .map(|d| brightness(deflected(d, &[v(0.0, 0.0)], &[1.0]), v(0.0, 0.0), width))
            .sum::<f64>()
            * (4.0f64 / 128.0).powi(2);
        // The integrand is below 1e-80 beyond 3 (the source is 0.15 wide).
        let expected = core::f64::consts::TAU
            * simpson(
                |d| (d * d + 2.0) / (d * d + 4.0).sqrt() * (-d * d / (2.0 * width * width)).exp(),
                0.0,
                3.0,
                60_000,
            );
        assert!(
            (flux - expected).abs() <= 1e-11 + 1e-11 * expected.abs(),
            "{flux} vs {expected}"
        );
    }

    /// The binary lens's checks: the local map's trace is two everywhere (reflections have no
    /// trace), it is the derivative of the deflection, and with trace two its inverse is
    /// `(2 - local) / area`.
    #[test]
    fn scenes_pass_their_checks() {
        let resolution = 97;
        let (directions, _, areas) = lens(resolution);
        for (i, d) in directions.iter().enumerate() {
            let local = binary_local(*d);
            assert!((local.trace() - 2.0).abs() < 1e-11);
            assert_eq!(areas[i], area(local));
            assert!((areas[i] - local.det()).abs() <= 1e-12 * areas[i].abs().max(1.0));
            let (row, col) = (i / resolution, i % resolution);
            if row % 97 == 0 && col % 97 == 0 {
                let step = v(1e-6, 0.0);
                let difference = (binary(*d + step) - binary(*d - step)) * (1.0 / 2e-6);
                assert!(close_v(difference, local.of(v(1.0, 0.0)), 1e-7));
            }
        }
        let offsets = circle(0.025);
        for (centre, seen, ratio) in tissot(5, 0.025) {
            let local = binary_local(centre);
            let inverse = (Vector::slot() * 2.0 - local) * (1.0 / ratio);
            for (s, o) in seen.iter().zip(&offsets) {
                assert!(close_v(*s - centre, inverse.of(*o), 1e-11));
            }
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
