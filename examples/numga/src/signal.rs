//! Oscillations: a phasor is the unit direction turned by a rotation motor, going round the unit
//! circle, and a wave is its height. The turn is a motor, as everywhere else (as warp's
//! `signal` module).

use gax::Real;
use gax::pga2d::{Motor, Point};

/// The unit phasor at `angle`: the direction `(1, 0)` turned by `angle` about the origin. Its
/// reach across (`e20`) is the cosine of the angle, its height (`e01`) the sine.
pub fn phasor<T: Real>(angle: T) -> Point<(), T> {
    let (z, o) = (T::zero(), T::one());
    Motor::rotation(Point::xy(z, z), angle) >> Point::direction(o, z)
}

/// The tangent of `angle`: the phasor's height over its width.
pub fn tangent<T: Real>(angle: T) -> T {
    let p = phasor(angle);
    p.e01() / p.e20()
}

/// A wave in `angle`: the phasor's height.
pub fn wave<T: Real>(angle: T) -> T {
    phasor(angle).e01()
}

#[cfg(test)]
mod tests {
    #[test]
    #[allow(clippy::disallowed_methods)] // the references they are checked against
    fn phasors_trace_sines_and_turns_read_angles() {
        for k in 0..64 {
            let a = -3.0 + k as f64 * 0.093;
            assert!((super::wave(a) - a.sin()).abs() < 1e-12);
            assert!((super::phasor(a).e20() - a.cos()).abs() < 1e-12);
            let (x, y) = (a.cos(), a.sin());
            let b = gax::vga2d::Vector::new(x, y);
            let east = gax::vga2d::Vector::new(1.0, 0.0);
            assert!(
                (crate::measure::turn(east, b) - y.atan2(x)).abs() < 1e-12,
                "{a}"
            );
        }
        // Half a turn exactly, both ways round.
        let east = gax::vga2d::Vector::new(1.0, 0.0);
        let west = gax::vga2d::Vector::new(-1.0, 0.0);
        let pi = core::f64::consts::PI;
        assert!((crate::measure::turn(east, west).abs() - pi).abs() < 1e-12);
        assert!((super::tangent(0.4f64) - 0.4f64.tan()).abs() < 1e-12);
        // The roots of t² - 5t + 6 (2 and 3), and none of t² + 1.
        let (mid, half) = crate::measure::roots(1.0, -2.5, 6.0).expect("two roots");
        assert!((mid - 2.5).abs() < 1e-12 && (half - 0.5).abs() < 1e-12);
        assert!(crate::measure::roots(1.0, 0.0, 1.0).is_none());
        let p = crate::measure::lift(0.3, 0.4).expect("inside the disc");
        assert!((p.e3() - 0.75f64.sqrt()).abs() < 1e-12);
    }
}
