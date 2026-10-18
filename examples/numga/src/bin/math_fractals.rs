//! numga's `math/fractals` (a notebook): fractals of the even subalgebras of the plane and of
//! space.
//!
//! The even multivectors of the plane, a scalar plus a multiple of `e12`, multiply among
//! themselves, and since `e12` squares to minus one they multiply as the complex numbers: they
//! are VGA2D's rotor kind. Iterating `z ↦ z² + c` on them from `z = c`, for each `c` of a grid,
//! draws the Mandelbrot set, the `c` for which the iteration stays bounded. The even
//! multivectors of space, a scalar plus a bivector, multiply as the quaternions, and the same
//! iteration with `c` fixed and the start varying draws a Julia set of four dimensions. A plane
//! of starts through the scalar axis, spanned by the scalar and a unit bivector, shows a slice.
//! While that bivector is the direction of the constant's, the scalar and it multiply as the
//! complex numbers, and the slice is a complex Julia set; turning it, by the rotor that turns
//! its normal towards `z`, shows the other faces of the set.
//!
//! The iteration is written once, generic over the even kind; a point stops where it leaves the
//! circle of radius two, and the steps it stayed inside colour it, smoothed by how far past the
//! circle its last step took it.
//!
//! The animation: the Mandelbrot set on the left, with the Julia constant's complex shadow (its
//! scalar, and its bivector's size as the imaginary part); on the right the Julia set's slice,
//! turning out of the constant's plane and back.

use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, Point2, Rect, backdrop, caption, palette, run,
    signal::haversine,
};

mod fractals {
    use gax::vga3d::Bivector;
    use gax::{Gp, Norm};

    /// An even multivector of the plane: a complex number.
    pub type Complex = gax::vga2d::Rotor<(), f64>;
    /// An even multivector of space: a quaternion.
    pub type Quaternion = gax::vga3d::Rotor<(), f64>;

    /// The square of the radius a point escapes at.
    const ESCAPE: f64 = 4.0;

    /// A point iterated by `z z + constant` until it leaves the circle of radius two, where it
    /// stops, and how many steps it stayed inside.
    pub fn escape<E>(start: E, constant: E, steps: u32) -> (E, u32)
    where
        E: Norm<Coef = f64> + Gp<E, Output = E> + core::ops::Add<Output = E>,
    {
        let mut point = start;
        for count in 0..steps {
            if point.norm_squared() >= ESCAPE {
                return (point, count);
            }
            point = point.gp(point) + constant;
        }
        (point, steps)
    }

    /// The complex number `x + y e12`.
    pub fn complex(x: f64, y: f64) -> Complex {
        gax::vga2d::Rotor::new(x, y)
    }

    /// numga's Julia constant, `-0.2 + 0.6 yz + 0.2 zx`.
    pub fn julia() -> Quaternion {
        gax::vga3d::Rotor::new(-0.2, 0.6, 0.2, 0.0)
    }

    /// The unit bivector of the slice at an angle: the constant's own direction turned by the
    /// rotor in the plane of its normal and `z`, the commutator of `xy` with it.
    pub fn slice_direction(angle: f64) -> Bivector<(), f64> {
        let direction = julia().grade::<2>().normalized();
        let xy = Bivector::new(0.0, 0.0, 1.0);
        let away = xy.commutator(direction).normalized();
        away.gp(-angle / 2.0).exp() >> direction
    }

    /// The start `x + y b` of the slice spanned by the scalar and the unit bivector `b`.
    pub fn slice_point(direction: Bivector<(), f64>, x: f64, y: f64) -> Quaternion {
        gax::vga3d::Rotor::new(x, 0.0, 0.0, 0.0) + Quaternion::from(direction.gp(y))
    }

    /// How long a point took to escape, smoothed by how far past the circle of radius two the
    /// last step took it: `steps + 1 - log2(log2 |z|)`, with `|z|` at least two.
    pub fn smooth(norm_squared: f64, steps: u32) -> f64 {
        let log_size = 0.5 * norm_squared.max(ESCAPE).log2();
        f64::from(steps) + 1.0 - log_size.log2()
    }
}

use fractals::*;

const SECONDS: f32 = 10.0;
/// numga's view of the Mandelbrot set and its steps.
const MANDELBROT_LOWER: [f32; 2] = [-2.2, -1.1];
const MANDELBROT_UPPER: [f32; 2] = [0.8, 1.1];
const MANDELBROT_STEPS: u32 = 120;
/// numga's view of the Julia slices and their steps, and the largest turn of the slice.
const JULIA_HALF: f32 = 1.5;
const JULIA_STEPS: u32 = 40;
const QUARTER: f64 = core::f64::consts::FRAC_PI_2;
/// Escape steps per pass through the colours, and the steps a point must stay to shine fully.
const PERIOD: f64 = 24.0;
const GLOW: f64 = 14.0;

/// The colours escaping points run through as they take longer to escape, repeating: numga's
/// bands, as lights.
fn bands() -> [Light; 7] {
    [
        [0.05, 0.07, 0.25],
        [0.12, 0.35, 0.65],
        [0.55, 0.80, 0.90],
        [0.98, 0.93, 0.70],
        [0.95, 0.60, 0.20],
        [0.55, 0.15, 0.25],
        [0.05, 0.07, 0.25],
    ]
    .map(|[r, g, b]| Light::from_srgb(r, g, b, 1.2))
}

/// The colour of a point that escaped, by its smoothed escape time, brighter the longer it
/// stayed, so that the boundary glows; nothing (the backdrop) for one that stayed bounded.
fn colour(norm_squared: f64, steps: u32, limit: u32) -> Option<Light> {
    if steps >= limit {
        return None;
    }
    let bands = bands();
    let time = smooth(norm_squared, steps);
    let position = (time / PERIOD).rem_euclid(1.0) * (bands.len() - 1) as f64;
    let lower = position as usize;
    let fraction = (position - lower as f64) as f32;
    let glow = (time / GLOW).clamp(0.0, 1.0) as f32;
    Some(
        bands[lower]
            .mix_light(bands[lower + 1], fraction)
            .faded(0.12 + 0.88 * glow),
    )
}

/// The slice's turn at time `t`: out to a quarter turn and back.
fn slice_angle(t: f32) -> f64 {
    let phase = core::f64::consts::TAU * f64::from(t / SECONDS);
    QUARTER * haversine(phase)
}

/// The Mandelbrot set in `rect`, and the Julia constant's complex shadow on it.
fn mandelbrot_panel(c: &mut Canvas, rect: Rect) {
    let x = [MANDELBROT_LOWER[0], MANDELBROT_UPPER[0]];
    let y = [MANDELBROT_LOWER[1], MANDELBROT_UPPER[1]];
    let ax = Axes::new(rect, x, y);
    ax.image(c, 1, |p| {
        let [x, y] = p.to_euclidean().map(f64::from);
        let constant = complex(x, y);
        let (point, steps) = escape(constant, constant, MANDELBROT_STEPS);
        colour(point.norm_squared(), steps, MANDELBROT_STEPS)
    });
    let shadow = julia();
    let size = shadow.grade::<2>().norm();
    let at = Point2::xy(shadow.s() as f32, size as f32);
    ax.scatter(c, &[at], Marker::Ring, 10.0, palette::orange());
    ax.text(
        c,
        at + Point2::direction(0.1, 0.05),
        "C",
        11.0,
        palette::orange(),
        Align::Left,
    );
    let label = "MANDELBROT: Z Z + C ON S + B E12, E12 E12 = -1";
    c.text(
        label,
        rect.bottom_left() + Point2::direction(0.0, 18.0),
        10.0,
        palette::ink(),
        Align::Left,
    );
}

/// The Julia set's slice in `rect`, turned by `angle` out of the constant's plane.
fn julia_panel(c: &mut Canvas, rect: Rect, angle: f64) {
    let ax = Axes::equal(rect, Point2::xy(0.0, 0.0), JULIA_HALF);
    let direction = slice_direction(angle);
    ax.image(c, 1, |p| {
        let [x, y] = p.to_euclidean().map(f64::from);
        let (point, steps) = escape(slice_point(direction, x, y), julia(), JULIA_STEPS);
        colour(point.norm_squared(), steps, JULIA_STEPS)
    });
    let label = format!(
        "QUATERNION JULIA SLICE, TURNED {:.0} DEG",
        angle.to_degrees()
    );
    c.text(
        &label,
        rect.bottom_left() + Point2::direction(0.0, 18.0),
        10.0,
        palette::ink(),
        Align::Left,
    );
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    mandelbrot_panel(c, screen.part(0.02, 0.16, 0.56, 0.88));
    julia_panel(c, screen.part(0.6, 0.16, 0.98, 0.88), slice_angle(t));
    caption(
        c,
        "FRACTALS OF THE EVEN SUBALGEBRAS",
        "THE PLANE'S EVEN MULTIVECTORS ARE THE COMPLEX NUMBERS, SPACE'S THE QUATERNIONS",
    );
}

fn main() {
    run(Anim::new("fractals", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::fractals::*;
    use super::{JULIA_STEPS, MANDELBROT_STEPS};

    /// numga's checks: `e12` squares to minus one; the origin stays bounded and a constant
    /// beyond the circle escapes at once; and the first slice, through the constant's own
    /// bivector, is the complex Julia set of the constant with that bivector's size as its
    /// imaginary part, on numga's 200 x 200 grid.
    #[test]
    fn the_even_subalgebras_iterate_as_the_complex_numbers_and_the_quaternions() {
        let e12 = complex(0.0, 1.0);
        assert_eq!(e12 * e12, complex(-1.0, 0.0));
        let origin = complex(0.0, 0.0);
        assert_eq!(escape(origin, origin, MANDELBROT_STEPS).1, MANDELBROT_STEPS);
        let far = complex(2.5, 0.0);
        assert_eq!(escape(far, far, MANDELBROT_STEPS).1, 0);

        let direction = slice_direction(0.0);
        let size = julia().grade::<2>().norm();
        let shadow = complex(-0.2, size);
        const GRID: usize = 200;
        let centre = |k: usize| -1.5 + 3.0 * (k as f64 + 0.5) / GRID as f64;
        for row in 0..GRID {
            for column in 0..GRID {
                let (x, y) = (centre(column), -centre(row));
                let quaternion = escape(slice_point(direction, x, y), julia(), JULIA_STEPS).1;
                let plane = escape(complex(x, y), shadow, JULIA_STEPS).1;
                assert_eq!(quaternion, plane, "{x} {y}");
            }
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 2.5);
    }
}
