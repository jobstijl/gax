//! Measures computed with gax: the roots of a quadratic as a circle, the sphere over the unit
//! disc, and the Gauss linking number of two closed curves; and the scalar tools the measures
//! of the examples share: the golden ratio and a root by bisection. (Angles between directions
//! are gax's own: `angle_between`, and the signed `angle_to` in the plane.)

use gax::vga3d::Vector;

/// The golden ratio `φ = (1 + √5) / 2`: the icosahedron's corners are `(0, ±1, ±φ)` cycled, a
/// Fibonacci spiral turns by `2π φ` per point, and a golden-section search splits its bracket at
/// `1 / φ = φ - 1`.
pub const GOLDEN: f64 = 1.618_033_988_749_895;

/// The roots of `a t² + 2 b t + c` as a circle on the line of `t`: in CGA2D, the dual circle
/// `S = eo - (b / a) e1 + c / (2 a) ei`, whose round points `X(t) = eo + t e1 + ½ t² ei` on the
/// x axis meet it, `X · S = -(a t² + 2 b t + c) / (2 a) = 0`, exactly at the roots. Its centre is
/// the roots' midpoint and its norm (the radius) half their distance: `(midpoint, half width)`,
/// or `None` when the circle is imaginary (it squares negative) and there are no roots.
pub fn roots(a: f64, b: f64, c: f64) -> Option<(f64, f64)> {
    let circle = gax::cga2d::Vector::<(), f64>::new(-b / a, 0.0, 1.0, c / (2.0 * a));
    (circle.norm_squared() >= 0.0).then(|| (circle.down()[0], circle.norm()))
}

/// The point of the unit sphere above `(x, y)` (the front hemisphere, `z ≥ 0`, seen along
/// `-z`), or `None` outside the unit disc: its height is half the distance between the roots of
/// `z² - (1 - x² - y²)`, the sphere's meet with the vertical line, by [`roots`].
pub fn lift(x: f64, y: f64) -> Option<Vector<(), f64>> {
    let r2 = gax::vga2d::Vector::new(x, y).norm_squared();
    let (_, z) = roots(1.0, 0.0, r2 - 1.0)?;
    Some(Vector::new(x, y, z))
}

/// The Gauss linking number of two closed polygons: the double sum, over pairs of segments, of
/// the volume their directions span with the separation of their midpoints, over the cube of
/// that separation, divided by `4π`.
pub fn linking(first: &[Vector<(), f64>], second: &[Vector<(), f64>]) -> f64 {
    let mut sum = 0.0;
    for a in first.windows(2) {
        for b in second.windows(2) {
            let separation = (a[1] + a[0] - b[1] - b[0]).gp(0.5);
            let volume = (separation ^ (a[1] - a[0]) ^ (b[1] - b[0])).dual().s();
            sum += volume / separation.norm().powi(3);
        }
    }
    sum / (4.0 * core::f64::consts::PI)
}

/// Where `below` stops holding between `lo`, where it holds, and `hi`, where it does not: the
/// bracket halved at most `steps` times, or until its ends are neighbouring numbers, and its
/// midpoint. For a function's root, `below` is whether its sign is the one at `lo`.
pub fn bisect(mut lo: f64, mut hi: f64, steps: usize, below: impl Fn(f64) -> bool) -> f64 {
    for _ in 0..steps {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if below(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}
