//! The exponentials of kinds that are not bivectors (ADR-019, amended 2026-11-01): a central
//! part times a rest whose square is a Study number, `exp(z + V) = exp(z) (C(V²) + S(V²) V)`.
//! Each against the power series `Σ xⁿ/n!` computed with the generated products.

#![cfg(all(
    feature = "vga2d",
    feature = "vga3d",
    feature = "sta",
    feature = "pga3d"
))]

use gax::ApproxEq;

/// `Σ xⁿ/n!` to 60 terms, in a kind closed under the product.
fn series<M>(x: M, one: M) -> M
where
    M: Copy
        + core::ops::Add<Output = M>
        + core::ops::Mul<Output = M>
        + core::ops::Mul<f64, Output = M>,
{
    let (mut sum, mut term) = (one, one);
    for n in 1..60 {
        term = (term * x) * (1.0 / f64::from(n));
        sum = sum + term;
    }
    sum
}

#[test]
fn vga3d_vectors_paravectors_and_multivectors() {
    use gax::vga3d::{Multivector, Paravector, Vector};
    let one = Multivector::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    // A vector squares to |v|² > 0: a boost, cosh |v| + sinh |v| v̂.
    let v = Vector::new(0.3, -0.4, 1.2);
    let e: Paravector = v.exp();
    let r = v.norm();
    assert!((e.s() - r.cosh()).abs() < 1e-14);
    assert!(
        e.cast::<Multivector>()
            .approx_eq(&series(v.cast::<Multivector>(), one), 1e-13)
    );
    // A paravector: e^s times the vector's.
    let p = Paravector::new(0.7, 0.3, -0.4, 1.2);
    assert!(
        p.exp()
            .cast::<Multivector>()
            .approx_eq(&series(p.cast::<Multivector>(), one), 1e-13)
    );
    // A general multivector: z = a + b e123 central, V = v + e123 w.
    let x = Multivector::new(0.2, 0.5, -0.3, 0.8, -0.6, 0.1, 0.4, -0.7);
    assert!(x.exp().approx_eq(&series(x, one), 1e-12), "{:?}", x.exp());
    // The smoke rings' generator: a vector and its dual bivector commute, so the exponential of
    // their sum is a boost times a turn.
    let s = Vector::new(0.0, 0.6, 0.8);
    let dual: gax::vga3d::Bivector = s.hodge().with_orientation();
    let (rapidity, angle) = (0.9, 0.5);
    let generator =
        s.cast::<Multivector>() * (-rapidity / 2.0) + dual.cast::<Multivector>() * (-angle / 2.0);
    let boost = (s * (-rapidity / 2.0)).exp().cast::<Multivector>();
    let turn = (dual * (-angle / 2.0))
        .exp()
        .into_inner()
        .cast::<Multivector>();
    assert!(generator.exp().approx_eq(&(boost * turn), 1e-14));
}

#[test]
fn vga2d_and_sta() {
    // VGA2D: (v + b e12)² = |v|² − b², every multivector.
    use gax::vga2d::Multivector;
    let one = Multivector::new(1.0, 0.0, 0.0, 0.0);
    for x in [
        Multivector::new(0.1, 0.9, -0.2, 0.4),
        Multivector::new(-0.3, 0.2, 0.1, 1.4),
    ] {
        assert!(x.exp().approx_eq(&series(x, one), 1e-13));
    }
    // STA: a timelike vector squares to −1, so its exponential turns (cos, sin); a spacelike
    // one boosts. The even kind: e^z exp(B) with z = a + b e1234.
    use gax::sta::{Even, Vector};
    let t = Vector::new(0.0, 0.0, 0.0, 0.8);
    let e = t.exp();
    assert!((e.s() - 0.8f64.cos()).abs() < 1e-15 && (e.e4() - 0.8f64.sin()).abs() < 1e-15);
    let m = gax::sta::Multivector::from(e);
    let one4 = gax::sta::Multivector::from(gax::sta::Scalar::new(1.0));
    assert!(m.approx_eq(&series(gax::sta::Multivector::from(t), one4), 1e-14));
    let r = Even::new(0.3, 0.2, -0.5, 0.4, 0.6, -0.1, 0.3, 0.25);
    let one_e = Even::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    assert!(r.exp().approx_eq(&series(r, one_e), 1e-13));
}

#[test]
fn pga3d_motors() {
    // A motor's exponential: e^(a + b e0123) exp(B), I² = 0 (so e^(b I) = 1 + b I).
    use gax::pga3d::Motor;
    let m = Motor::new(0.2, 0.3, -0.1, 0.4, 0.5, -0.6, 0.2, 0.7);
    let one = Motor::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    assert!(m.exp().approx_eq(&series(m, one), 1e-13));
}
