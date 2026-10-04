//! Measures computed with gax: the angle between two directions (in space, and signed in the
//! plane), and the Gauss linking number of two closed curves.

use gax::vga3d::{Bivector, Vector};

/// The angle between `a` and `b`, in radians: the norm of the logarithm of the rotor `b a`
/// (normalized) that turns one into the other.
pub fn angle(a: Vector<(), f64>, b: Vector<(), f64>) -> f64 {
    let log: Bivector<(), f64> = (b * a).normalized().log();
    log.norm()
}

/// The signed angle from `a` to `b` in the plane, in radians from `-π` to `π`, counterclockwise
/// positive: the product `a b` is `cos θ + sin θ e12` (normalized), and its logarithm `θ e12`.
/// Past a quarter turn (the product's scalar part negative) the angle is read from `-a`, within
/// a quarter turn of `b`, and half a turn added: the logarithm is exact there, where at half a
/// turn the rotor `-1` has no unique one.
pub fn turn(a: gax::vga2d::Vector<(), f64>, b: gax::vga2d::Vector<(), f64>) -> f64 {
    let rotor = (a * b).normalized();
    if rotor.into_inner().s() >= 0.0 {
        let log: gax::vga2d::Pseudoscalar<(), f64> = rotor.log();
        return log.e12();
    }
    let log: gax::vga2d::Pseudoscalar<(), f64> = ((-a) * b).normalized().log();
    let pi = core::f64::consts::PI;
    let angle = log.e12() + pi;
    if angle > pi { angle - 2.0 * pi } else { angle }
}

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
