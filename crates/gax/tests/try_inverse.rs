//! The checked inverse of values, `try_inverse`: `None` where `x ~x` is zero next to `‖x‖²`
//! or its norm is lost to cancellation, the plain inverse elsewhere.

#[cfg(feature = "pga3d")]
mod pga3d {
    use gax::Extensor;
    use gax::pga3d::{Line, Plane, Point};

    #[test]
    fn planes() {
        let p = Plane::<(), f64>::new(1.0, 2.0, 2.0, -3.0);
        assert_eq!(p.try_inverse(), Some(p.inverse()));
        // The plane at infinity has none, also where rounding left its normal at 1e-17.
        assert_eq!(Plane::<(), f64>::infinity().try_inverse(), None);
        let rounded = Plane::<(), f64>::new(1e-17, -3e-17, 2e-17, 1.0);
        assert!(rounded.inverse().is_finite());
        assert_eq!(rounded.try_inverse(), None);
        // A value of rounding size throughout is a scaled plane like any other: the test is
        // relative to the coefficients, as a map's is.
        assert!(
            Plane::<(), f64>::new(1e-17, 0.0, 0.0, 1e-17)
                .try_inverse()
                .is_some()
        );
        // A plane far from the origin is still invertible.
        let far = Plane::<(), f64>::new(1e-6, 0.0, 0.0, 1.0);
        let inv = far.try_inverse().expect("invertible");
        assert!(((far * inv).s() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn lines() {
        let l = Point::<(), f64>::xyz(1.0, 2.0, 3.0) & Point::xyz(-1.0, 0.5, 2.0);
        assert_eq!(l.try_inverse(), Some(l.inverse()));
        // A line at infinity (the join of two directions) has none.
        let ideal = Point::<(), f64>::direction(1.0, 0.0, 0.0) & Point::direction(0.0, 1.0, 0.0);
        assert_eq!(ideal.try_inverse(), None);
        assert_eq!(Line::<(), f64>::zero().try_inverse(), None);
    }

    #[test]
    fn nan_has_none() {
        assert_eq!(
            Plane::<(), f64>::new(f64::NAN, 0.0, 0.0, 1.0).try_inverse(),
            None
        );
    }
}

#[cfg(feature = "cga3d")]
mod cga3d {
    use gax::ApproxEq;
    use gax::cga3d::{Even, Quadvector, Scalar, Vector};

    #[test]
    fn round_points_are_null() {
        let x = Vector::<(), f64>::up(1.1, 2.3, 0.7);
        assert_eq!(x.try_inverse(), None);
        // A sphere (a vector of positive square) has one.
        let s = Vector::sphere(x, 2.0);
        assert_eq!(s.try_inverse(), Some(s.inverse()));
    }

    #[test]
    fn cancellation_in_the_norm() {
        // `E = e1 e2 e3 e₊` squares to 1, so `1 + E` is a zero divisor, `(1 + E)(1 − E) = 0`:
        // `x ~x = 2 + 2E` is far from zero, its norm `a² − E² b²` cancels.
        let (e1, e2, e3) = (
            Vector::new(1.0, 0.0, 0.0, 0.0, 0.0),
            Vector::new(0.0, 1.0, 0.0, 0.0, 0.0),
            Vector::new(0.0, 0.0, 1.0, 0.0, 0.0),
        );
        let plus = Vector::<(), f64>::new(0.0, 0.0, 0.0, 1.0, -0.5); // eo − ei/2, squares to 1
        let e: Quadvector<(), f64> = e1 ^ e2 ^ e3 ^ plus;
        let one = Even::from(Scalar::new(1.0));
        let (plus_e, minus_e) = (one + Even::from(e), one - Even::from(e));
        assert!((plus_e * minus_e).max_abs_diff(&Even::zero()) < 1e-15);
        assert_eq!(plus_e.try_inverse(), None);
        // Away from the cancellation the inverse is there.
        let x = one + Even::from(e).gp(0.5);
        let inv = x.try_inverse().expect("invertible");
        assert!((x * inv).approx_eq(&one, 1e-14));
    }
}

#[cfg(feature = "vga3d")]
mod general {
    use gax::vga3d::{Multivector, Paravector};
    use gax::{ApproxEq, Inverse};

    #[test]
    fn residual_test_of_shirokovs_inverse() {
        // `1 + e1` is a zero divisor, `(1 + e1)(1 − e1) = 0`, with `x ~x = 2 + 2 e1` no Study
        // number: the general inverse, checked by its residual.
        let null = Paravector::<(), f64>::new(1.0, 1.0, 0.0, 0.0);
        assert_eq!(null.try_inverse(), None);
        let x = Paravector::<(), f64>::new(2.0, 1.0, -0.5, 0.25);
        let inv = x.try_inverse().expect("invertible");
        assert!(inv.approx_eq(&x.inverse(), 0.0));
        let m = Multivector::<(), f64>::from(x);
        assert!(
            Inverse::try_inverse(m)
                .expect("invertible")
                .approx_eq(&inv, 1e-15)
        );
        assert_eq!(
            Inverse::try_inverse(Multivector::<(), f64>::from(null)),
            None
        );
    }
}
