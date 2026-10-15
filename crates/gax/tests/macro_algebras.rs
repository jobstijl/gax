//! User algebras declared with `algebra!`: STAP `R(3,1,1)` and CSTA `R(4,2)`, the signatures
//! the standard algebras do not cover, PGA4D, and R(8,0) with its 4-vectors (the outermorphism's
//! size limit).

#![cfg(feature = "macros")]

use gax::ApproxEq;

gax::algebra! {
    algebra stap "Spacetime algebra with a projective (null) dimension, R(3,1,1).";
    basis ep = 0, e0 = 1, e1 = -1, e2 = -1, e3 = -1;
    kind Scalar = [1];
    versor Vector = [ep, e0, e1, e2, e3];
    kind Bivector = [ep0, ep1, ep2, ep3, e01, e02, e03, e12, e31, e23];
    versor Even = [1, ep0, ep1, ep2, ep3, e01, e02, e03, e12, e31, e23, ep012, ep031, ep023, ep123, e0123];
    kind Multivector = [1, ep, e0, e1, e2, e3, ep0, ep1, ep2, ep3, e01, e02, e03, e12, e31, e23, ep01, ep02, ep03, ep12, ep31, ep23, e012, e031, e023, e123, ep012, ep031, ep023, ep123, e0123, ep0123];
}

gax::algebra! {
    algebra csta "Conformal spacetime algebra R(4,2).";
    basis e0 = 1, e1 = -1, e2 = -1, e3 = -1, ep = 1, em = -1;
    kind Scalar = [1];
    versor Vector = [e0, e1, e2, e3, ep, em];
    kind Bivector = [e01, e02, e03, e12, e31, e23, e0p, e1p, e2p, e3p, e0m, e1m, e2m, e3m, epm];
    versor Even = [1, e01, e02, e03, e12, e31, e23, e0p, e1p, e2p, e3p, e0m, e1m, e2m, e3m, epm,
        e0123, e012p, e031p, e023p, e123p, e012m, e031m, e023m, e123m, e01pm, e02pm, e03pm, e12pm, e31pm, e23pm, e0123pm];
}

#[test]
fn stap_products_and_sandwich() {
    use stap::*;
    let v = Vector::<(), f64>::new(0.5, 2.0, 0.3, -0.2, 0.1);
    let w = Vector::<(), f64>::new(-1.0, 1.0, 0.7, 0.4, -0.5);
    // v w = v . w + v ^ w, and v . w uses the metric: e_p² = 0, e0² = 1, e_i² = -1.
    let vw: Even<(), f64> = v * w;
    let dot = 2.0 * 1.0 - 0.3 * 0.7 - (-0.2) * 0.4 - 0.1 * (-0.5);
    assert!((vw.s() - dot).abs() < 1e-12);
    // A reflection keeps the metric: (v >> w)² == v⁴ w².
    let r: Vector<(), f64> = v >> w;
    let (vv, ww, rr) = ((v * v).s(), (w * w).s(), (r * r).s());
    assert!((rr - vv * vv * ww).abs() < 1e-10);
}

#[test]
fn csta_is_six_dimensional() {
    use csta::*;
    let a = Vector::<(), f64>::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    // (e0 + ep)² = 1 + 1 = 2
    assert!(((a | a).s() - 2.0).abs() < 1e-12);
    let b: Bivector<(Vector,), f64> = a ^ Vector::slot();
    assert_eq!(b.of(a), Bivector::zero());
}

#[test]
fn stap_bivector_exp_log_round_trip() {
    use stap::*;
    // A 5D bivector: its square has a scalar and a 4-vector part, handled by the general
    // Study-number functions.
    let b = Bivector::<(), f64>::new(0.1, -0.2, 0.05, 0.3, 0.2, 0.1, -0.15, 0.4, 0.25, -0.3);
    let r = b.exp();
    let back: Bivector<(), f64> = r.log();
    assert!(back.approx_eq(&b, 1e-10), "{back:?} vs {b:?}");
    // exp(B) is a unit versor: R ~R = 1.
    let n: Even<(), f64> = r.into_inner() * r.into_inner().reverse();
    assert!((n.s() - 1.0).abs() < 1e-12);
}

#[test]
fn csta_exp_by_scaling_and_squaring() {
    use csta::*;
    // A rotation in the e12 plane (e1² = e2² = -1, so e12² = -1): exp(θ e12) = cos θ + sin θ e12.
    let th = 0.9f64;
    let b = Bivector::<(), f64>::new(
        0.0, 0.0, 0.0, th, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    );
    let r = b.exp().into_inner();
    assert!(
        (r.s() - th.cos()).abs() < 1e-12 && (r.e12() - th.sin()).abs() < 1e-12,
        "{r:?}"
    );
    // A general bivector: exp(B) exp(-B) = 1 and exp(B) is a unit versor.
    let g = Bivector::<(), f64>::new(
        0.1, -0.2, 0.3, 0.2, 0.1, -0.3, 0.05, 0.1, 0.2, -0.1, 0.3, 0.2, -0.15, 0.1, 0.25,
    );
    let (p, m) = (g.exp().into_inner(), (-g).exp().into_inner());
    let id: Even<(), f64> = p * m;
    assert!(
        (id.s() - 1.0).abs() < 1e-11 && id.c[1..].iter().all(|c| c.abs() < 1e-11),
        "{id:?}"
    );
    let n: Even<(), f64> = p * p.reverse();
    assert!((n.s() - 1.0).abs() < 1e-11 && n.c[1..].iter().all(|c| c.abs() < 1e-11));
}

gax::algebra! {
    algebra pga4d "Plane-based PGA of 4D Euclidean space, R(4,0,1).";
    basis e0 = 0, e1 = 1, e2 = 1, e3 = 1, e4 = 1;
    kind Hyperplane = [e1, e2, e3, e4, e0];
    kind Bivector = [e12, e13, e14, e23, e24, e34, e01, e02, e03, e04];
    kind Point = [e0234, e0134, e0124, e0123, e1234];
    versor Motor = [1, e12, e13, e14, e23, e24, e34, e01, e02, e03, e04,
        e1234, e0123, e0124, e0134, e0234];
}

#[test]
fn pga4d_motors_move_points_rigidly() {
    use pga4d::*;
    // A rotation in two orthogonal planes plus a translation: a 5D bivector with a 4-vector
    // part in its square, so exp and log take the general Study-number path.
    let b = Bivector::<(), f64>::new(0.3, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1, -0.2, 0.3, 0.4);
    let m = b.exp();
    let back: Bivector<(), f64> = m.log();
    assert!(back.approx_eq(&b, 1e-10), "{back:?} vs {b:?}");
    // Rigid: the weight of a point and the distance between two points are preserved.
    let p = Point::<(), f64>::new(1.0, 2.0, 3.0, 4.0, 1.0);
    let q = Point::<(), f64>::new(-1.0, 0.5, 2.0, 0.0, 1.0);
    let (p2, q2) = (m >> p, m >> q);
    assert!((p2.e1234() - 1.0).abs() < 1e-12);
    let d = |a: Point<(), f64>, b: Point<(), f64>| {
        (0..4)
            .map(|i| (a.c[i] / a.c[4] - b.c[i] / b.c[4]).powi(2))
            .sum::<f64>()
    };
    assert!((d(p, q) - d(p2, q2)).abs() < 1e-10);
}

// R(8,0) with its 4-vectors, the states of numga's Hubbard ring. The extension of a map on
// vectors to the 4-vectors, 70 × 70 minors of 24 terms, is over the generator's limit
// (`gax_gen::emit_outer::OUTER_MAX_TERMS`, ADR-034): it is not emitted, and the algebra compiles
// in seconds (with it, a build ran out of 10 GB after six minutes). The bivectors' extension,
// 1568 terms, is emitted.
gax::algebra! {
    algebra r8 "Euclidean 8D, R(8,0), with its bivectors and 4-vectors.";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = 1, e6 = 1, e7 = 1, e8 = 1;
    kind Vector = [e1, e2, e3, e4, e5, e6, e7, e8];
    kind Bivector = [e12, e13, e14, e15, e16, e17, e18, e23, e24, e25, e26, e27, e28, e34, e35,
        e36, e37, e38, e45, e46, e47, e48, e56, e57, e58, e67, e68, e78];
    kind Quadvector = [e1234, e1235, e1236, e1237, e1238, e1245, e1246, e1247, e1248, e1256, e1257,
        e1258, e1267, e1268, e1278, e1345, e1346, e1347, e1348, e1356, e1357, e1358, e1367, e1368,
        e1378, e1456, e1457, e1458, e1467, e1468, e1478, e1567, e1568, e1578, e1678, e2345, e2346,
        e2347, e2348, e2356, e2357, e2358, e2367, e2368, e2378, e2456, e2457, e2458, e2467, e2468,
        e2478, e2567, e2568, e2578, e2678, e3456, e3457, e3458, e3467, e3468, e3478, e3567, e3568,
        e3578, e3678, e4567, e4568, e4578, e4678, e5678];
}

/// `Extends::<M, B>::YES` says whether `M: Outermorphism<B>`: the inherent constant exists only
/// where that bound holds, and the trait's default stands in elsewhere.
struct Extends<M, B>(core::marker::PhantomData<(M, B)>);

trait NotExtends {
    const YES: bool = false;
}

impl<M, B> NotExtends for Extends<M, B> {}

impl<M: gax::Outermorphism<B>, B> Extends<M, B> {
    const YES: bool = true;
}

mod r8_maps {
    use super::r8::{Bivector, Multivector, Quadvector, Vector};

    /// An invertible map on R(8,0)'s vectors with small integer entries, so that every minor is
    /// exact.
    pub fn map() -> Vector<(Vector,), f64> {
        let entries = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let t = Vector::from_coeffs(core::array::from_fn(|o| {
            core::array::from_fn(|i| entries[(3 * o + 5 * i) % 7] + if o == i { 5.0 } else { 0.0 })
        }));
        assert_ne!(t.det(), 0.0);
        t
    }

    /// The basis vector `e_(i+1)`.
    pub fn e(i: usize) -> Vector<(), f64> {
        Vector::from_coeffs(core::array::from_fn(|j| f64::from(u8::from(i == j))))
    }

    /// The factors of the basis blades of `k` vectors from `e_(from+1)` on, in the order the
    /// kinds above list them (`e12, e13, …, e23, …`).
    fn blades(k: usize, from: usize) -> Vec<Vec<usize>> {
        if k == 0 {
            return vec![vec![]];
        }
        (from..8)
            .flat_map(|i| {
                blades(k - 1, i + 1).into_iter().map(move |rest| {
                    let mut factors = vec![i];
                    factors.extend(rest);
                    factors
                })
            })
            .collect()
    }

    /// The wedge of the images of the basis vectors `factors`: the image of their blade under
    /// the outermorphism.
    fn image(t: &Vector<(Vector,), f64>, factors: &[usize]) -> Multivector<(), f64> {
        factors
            .iter()
            .map(|&i| t.of(e(i)).cast::<Multivector>())
            .reduce(|w, v| w ^ v)
            .expect("at least one factor")
    }

    /// The outermorphism to bivectors, built from its images.
    pub fn on_bivectors(t: &Vector<(Vector,), f64>) -> Bivector<(Bivector,), f64> {
        let blades = blades(2, 0);
        Bivector::from_images(core::array::from_fn(|b| {
            image(t, &blades[b]).cast::<Bivector>()
        }))
    }

    /// The outermorphism to 4-vectors, built from its images: what to do over the limit.
    pub fn on_quadvectors(t: &Vector<(Vector,), f64>) -> Quadvector<(Quadvector,), f64> {
        let blades = blades(4, 0);
        Quadvector::from_images(core::array::from_fn(|b| {
            image(t, &blades[b]).cast::<Quadvector>()
        }))
    }
}

#[test]
fn r8_outermorphisms_within_the_limit_only() {
    use r8::{Bivector, Quadvector, Vector};
    const {
        assert!(Extends::<Vector<(Vector,), f64>, Bivector>::YES);
        assert!(!Extends::<Vector<(Vector,), f64>, Quadvector>::YES);
    }
    // The generated bivector extension is the one built from images (exactly: integer entries).
    let t = r8_maps::map();
    assert_eq!(t.outermorphism::<Bivector>().c, r8_maps::on_bivectors(&t).c);
    // The 4-vector extension built from images maps a wedge to the wedge of the images.
    let v = |c: [i8; 8]| Vector::<(), f64>::from_coeffs(c.map(f64::from));
    let (a, b, c, d) = (
        v([1, 2, 0, -1, 3, 0, 1, 2]),
        v([0, 1, -2, 1, 0, 2, 1, -1]),
        v([2, 0, 1, 1, -1, 0, 3, 1]),
        v([1, -1, 1, 0, 2, 1, 0, 1]),
    );
    let wedge = |[a, b, c, d]: [Vector<(), f64>; 4]| {
        (a.cast::<r8::Multivector>() ^ b ^ c ^ d).cast::<Quadvector>()
    };
    let m = r8_maps::on_quadvectors(&t);
    let image = m.of(wedge([a, b, c, d]));
    assert_eq!(image, wedge([a, b, c, d].map(|x| t.of(x))));
    assert_ne!(image, Quadvector::zero());
}
