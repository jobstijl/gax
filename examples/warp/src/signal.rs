//! Signals in time: an oscillation is a phasor, a point going round the unit circle, and its
//! value is the point's height.

use gax::pga2d::Point;

pub use crate::kernels::wave;

/// The unit phasor at `phase`: the unit direction at that angle from `(1, 0)`. Its height is the
/// sine, its width the cosine.
pub fn phasor(phase: f32) -> Point<(), f32> {
    Point::direction_from_angle(phase)
}

#[cfg(test)]
mod tests {
    #[test]
    #[allow(clippy::disallowed_methods)] // the reference it is checked against
    fn a_phasor_traces_a_sine() {
        for k in 0..64 {
            let p = k as f32 * 0.37;
            assert!((super::wave(p) - p.sin()).abs() < 1e-5);
        }
    }
}
