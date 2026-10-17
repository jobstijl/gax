//! Integration over simplices (ADR-048): the models' measures agree, integrals move with the
//! simplex, and the fundamental theorem of geometric calculus holds on every simplex, exactly,
//! over the prime field.

#![cfg(all(
    feature = "vga2d",
    feature = "vga3d",
    feature = "sta",
    feature = "pga2d",
    feature = "pga3d",
    feature = "cga2d",
    feature = "cga3d"
))]

mod exact;

use exact::{Rng, cubic};
use gax::Coef;
use gax::calculus::contract;
use gax::fp::Fp;
use gax::simplex::{Rule, Simplex};

/// The fundamental theorem on one simplex, `∫_σ (∇ ⌋ dX) F = Σᵢ (−1)ⁱ ∫_{∂ᵢσ} dS F`, for a cubic
/// field `F` of kind `$K`: the measure contracted by the derivative from the left,
/// `Σ (eʲ ⌋ dX) ∂ⱼ F`, and the faces without vertex `i` oriented by the boundary map. Exact: the
/// rules have the integrands' degrees.
macro_rules! fundamental_theorem {
    ($rng:expr, $alg:ident, $n:literal, $m:literal, $K:ident) => {{
        use gax::$alg::Vector;
        fn field<T: gax::Coef>(c: &[i64], x: Vector<(), T>) -> gax::$alg::$K<(), T> {
            cubic::<_, gax::$alg::$K>(c, x)
        }
        for _ in 0..3 {
            let c = $rng.cubic();
            let vs: [Vector<(), Fp>; $n] = core::array::from_fn(|_| $rng.point());
            let s = Simplex(vs);
            let b = s.measure();
            let lhs = s.mean(Rule::degree(2), |p| {
                contract(|x| field(&c, x), p, |up, column| up.lc(b) * column).1
            });
            let rhs = (0..$n)
                .map(|i| {
                    let face: [Vector<(), Fp>; $m] =
                        core::array::from_fn(|j| vs[if j < i { j } else { j + 1 }]);
                    let term = Simplex(face).directed(Rule::degree(3), |x| field(&c, x));
                    if i.is_multiple_of(2) { term } else { -term }
                })
                .reduce(|a, b| a + b)
                .unwrap();
            assert_eq!(
                lhs,
                rhs,
                "{} {}-simplex, a {} field",
                stringify!($alg),
                $n - 1,
                stringify!($K)
            );
        }
    }};
}

#[test]
fn fundamental_theorem_vga2d() {
    let mut rng = Rng(11);
    fundamental_theorem!(rng, vga2d, 2, 1, Scalar);
    fundamental_theorem!(rng, vga2d, 2, 1, Vector);
    fundamental_theorem!(rng, vga2d, 3, 2, Scalar);
    fundamental_theorem!(rng, vga2d, 3, 2, Vector);
}

#[test]
fn fundamental_theorem_vga3d() {
    let mut rng = Rng(12);
    fundamental_theorem!(rng, vga3d, 2, 1, Vector);
    fundamental_theorem!(rng, vga3d, 3, 2, Scalar);
    fundamental_theorem!(rng, vga3d, 3, 2, Vector);
    fundamental_theorem!(rng, vga3d, 4, 3, Scalar);
    fundamental_theorem!(rng, vga3d, 4, 3, Vector);
    fundamental_theorem!(rng, vga3d, 4, 3, Bivector);
}

#[test]
fn fundamental_theorem_sta() {
    let mut rng = Rng(13);
    fundamental_theorem!(rng, sta, 3, 2, Vector);
    fundamental_theorem!(rng, sta, 4, 3, Bivector);
    fundamental_theorem!(rng, sta, 5, 4, Scalar);
    fundamental_theorem!(rng, sta, 5, 4, Bivector);
}

/// The same Euclidean simplex measured in VGA, PGA and CGA: the norms of the measures are the
/// `k`-volume in each, and a simplex of the full dimension has the same signed volume.
#[test]
fn measures_agree_across_models() {
    use gax::{cga2d, cga3d, pga2d, pga3d, vga2d, vga3d};
    let mut rng = gax_rand();
    for _ in 0..20 {
        let x: [[f64; 3]; 4] = core::array::from_fn(|_| core::array::from_fn(|_| rng()));
        let v = x.map(|p| vga3d::Vector::<(), f64>::new(p[0], p[1], p[2]));
        let p = x.map(|p| pga3d::Point::<(), f64>::xyz(p[0], p[1], p[2]));
        let q = x.map(|p| cga3d::Vector::<(), f64>::up(p[0], p[1], p[2]));
        let close = |a: f64, b: f64| (a - b).abs() < 1e-12 * (1.0 + a.abs());
        let edge = (
            Simplex([v[0], v[1]]).measure().norm(),
            Simplex([p[0], p[1]]).measure().norm(),
            Simplex([q[0], q[1]]).measure().norm(),
        );
        assert!(close(edge.0, edge.1) && close(edge.0, edge.2), "{edge:?}");
        let tri = (
            Simplex([v[0], v[1], v[2]]).measure().norm(),
            Simplex([p[0], p[1], p[2]]).measure().norm(),
            Simplex([q[0], q[1], q[2]]).measure().norm(),
        );
        assert!(close(tri.0, tri.1) && close(tri.0, tri.2), "{tri:?}");
        let tet = (
            Simplex(v).measure().c[0],
            Simplex(p).measure().s(),
            Simplex(q).measure().c[0],
        );
        assert!(close(tet.0, tet.1) && close(tet.0, tet.2), "{tet:?}");

        let v = [v[0], v[1], v[2]].map(|v| vga2d::Vector::<(), f64>::new(v.c[0], v.c[1]));
        let p = v.map(|v| pga2d::Point::<(), f64>::xy(v.c[0], v.c[1]));
        let q = v.map(|v| cga2d::Vector::<(), f64>::up(v.c[0], v.c[1]));
        let edge = (
            Simplex([v[0], v[1]]).measure().norm(),
            Simplex([p[0], p[1]]).measure().norm(),
            Simplex([q[0], q[1]]).measure().norm(),
        );
        assert!(close(edge.0, edge.1) && close(edge.0, edge.2), "{edge:?}");
        let tri = (
            Simplex(v).measure().c[0],
            Simplex(p).measure().s(),
            Simplex(q).measure().c[0],
        );
        assert!(close(tri.0, tri.1) && close(tri.0, tri.2), "{tri:?}");
    }
}

/// A uniform stream in [−1, 1].
fn gax_rand() -> impl FnMut() -> f64 {
    let mut rng = Rng(99);
    move || (rng.next() >> 8) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// Integrals move with the simplex: the measure of a simplex moved by a rigid motion is the
/// moved measure, and the mean of the position over it the moved mean. In PGA3D, with a motion
/// of rational coefficients, exactly.
#[test]
fn integrals_move_with_the_simplex() {
    use gax::pga3d::{Point, Rotor, Translator};
    let r = Rotor::<(), Fp>::new(
        Fp::from_ratio(3, 5),
        Fp::zero(),
        Fp::zero(),
        Fp::from_ratio(4, 5),
    );
    let t = Translator::<(), Fp>::new(Fp::one(), Fp::new(7), Fp::from_ratio(-3, 2), Fp::new(11));
    let m = t * r;
    let mut rng = Rng(14);
    for _ in 0..4 {
        // Points of weight one.
        let vs: [Point<(), Fp>; 4] =
            core::array::from_fn(|_| Point::new(rng.fp(), rng.fp(), rng.fp(), Fp::one()));
        let moved = vs.map(|v| m >> v);
        assert_eq!(
            Simplex([moved[0], moved[1]]).measure(),
            m >> Simplex([vs[0], vs[1]]).measure()
        );
        assert_eq!(
            Simplex([moved[0], moved[1], moved[2]]).measure(),
            m >> Simplex([vs[0], vs[1], vs[2]]).measure()
        );
        assert_eq!(Simplex(moved).measure(), Simplex(vs).measure());
        assert_eq!(
            Simplex(moved).mean(Rule::degree(1), |p| p),
            m >> Simplex(vs).mean(Rule::degree(1), |p| p)
        );
    }
}
