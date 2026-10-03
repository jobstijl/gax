//! Measures computed with gax: the angle between two directions, and the Gauss linking number
//! of two closed curves.

use gax::vga3d::{Bivector, Vector};

/// The angle between `a` and `b`, in radians: the norm of the logarithm of the rotor `b a`
/// (normalized) that turns one into the other.
pub fn angle(a: Vector<(), f64>, b: Vector<(), f64>) -> f64 {
    let log: Bivector<(), f64> = (b * a).normalized().log();
    log.norm()
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
