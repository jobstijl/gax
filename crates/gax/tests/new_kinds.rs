//! The kinds added for numga's examples: PGA directions (ideal points), VGA3D's paravector (the
//! Pauli algebra's states), and STA's phasor with the closed-form `exp` of the pseudoscalar.

#![cfg(all(
    feature = "pga2d",
    feature = "pga3d",
    feature = "vga3d",
    feature = "sta"
))]

fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

/// `exp` of a pseudoscalar and of a phasor against their Taylor series.
#[test]
fn phasor_exp_is_the_series() {
    use gax::sta::{Bivector, Even, Phasor, Pseudoscalar};
    let series = |x: Phasor<(), f64>| {
        let (mut term, mut sum) = (Phasor::new(1.0, 0.0), Phasor::new(1.0, 0.0));
        for k in 1..40 {
            term = (term * x).gp(1.0 / f64::from(k));
            sum += term;
        }
        sum
    };
    for (a, b) in [(0.0, 0.7), (0.3, -2.1), (-1.2, 3.0)] {
        let x = Phasor::<(), f64>::new(a, b);
        assert!(close(&x.exp().c, &series(x).c, 1e-12), "{:?}", x.exp());
        if a == 0.0 {
            let i = Pseudoscalar::<(), f64>::new(b);
            assert!(close(&i.exp().c, &x.exp().c, 1e-15));
        }
    }
    // The duality rotation commutes with the even subalgebra, as `i` with everything complex.
    let r: Even<(), f64> = Bivector::<(), f64>::new(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        .exp()
        .into_inner();
    let p = Pseudoscalar::<(), f64>::new(0.9).exp();
    assert!(close(&(p * r).c, &(r * p).c, 1e-14));
}

/// A point's direction drops its weight; a direction is a point at infinity.
#[test]
fn directions_are_ideal_points() {
    use gax::pga2d;
    use gax::pga3d::{Direction, Motor, Point};
    let p = Point::<(), f64>::xyz(1.0, 2.0, 3.0);
    let d: Direction<(), f64> = p.cast::<Direction>();
    assert_eq!(d.c, [1.0, 2.0, 3.0]);
    let ideal: Point<(), f64> = d.cast::<Point>();
    assert_eq!(ideal.c, [1.0, 2.0, 3.0, 0.0]);
    // Translations leave directions alone; rotations turn them.
    let t = Motor::<(), f64>::translation(5.0, -1.0, 2.0);
    assert!(close(&(t >> ideal).c, &ideal.c, 1e-15));
    let r = Motor::<(), f64>::rotation_about(0.0, 0.0, 1.0, core::f64::consts::FRAC_PI_2);
    assert!(close(
        &(r >> ideal).cast::<Direction>().c,
        &[-2.0, 1.0, 3.0],
        1e-12
    ));
    // A map on directions only: the rotation's 3x3 part.
    let m = (r >> Direction::<(), f64>::slot().cast::<Point>()).cast::<Direction>();
    assert!(close(&m.of(d).c, &[-2.0, 1.0, 3.0], 1e-12));
    let q = pga2d::Point::<(), f64>::xy(3.0, 4.0);
    assert_eq!(q.cast::<pga2d::Direction>().c, [3.0, 4.0]);
}

/// The Pauli algebra: a state `(1 + r)/2` is idempotent exactly when its Bloch vector is unit.
#[test]
fn paravectors_are_pauli_states() {
    use gax::vga3d::{Paravector, Vector};
    let r = Vector::<(), f64>::new(0.6, 0.0, 0.8);
    let rho: Paravector<(), f64> =
        (r.cast::<Paravector>() + Paravector::new(1.0, 0.0, 0.0, 0.0)).gp(0.5);
    let sq = (rho * rho).cast::<Paravector>();
    assert!(close(&sq.c, &rho.c, 1e-15));
    let mixed: Paravector<(), f64> =
        (r.gp(0.5).cast::<Paravector>() + Paravector::new(1.0, 0.0, 0.0, 0.0)).gp(0.5);
    assert!(!close(
        &(mixed * mixed).cast::<Paravector>().c,
        &mixed.c,
        1e-3
    ));
}
