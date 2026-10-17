//! The laws of the multivector derivative (ADR-047), exactly, over the prime field: the fields
//! are cubic polynomials with random integer coefficients, evaluated at random points of `ℤ/p`,
//! so each law that holds at the samples holds as a polynomial identity (law L). The laws that
//! take roots are checked in `f64`.

#![cfg(all(
    feature = "vga2d",
    feature = "vga3d",
    feature = "sta",
    feature = "cga3d"
))]

use gax::fp::Fp;
use gax::{Coef, Extensor, Kind};

/// A deterministic stream of samples.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 3
    }
    /// A point of `ℤ/p` for each coefficient of `X`.
    fn point<X: Extensor<Slots = (), Coef = Fp>>(&mut self) -> X {
        X::from_coeffs(<X::Kind as Kind>::arr_from_fn(|_| Fp::new(self.next())))
    }
    /// The coefficients of a cubic field.
    fn cubic(&mut self) -> Vec<i64> {
        (0..4096)
            .map(|_| (self.next() >> 2) as i64 - (1 << 59))
            .collect()
    }
}

/// The cubic field with coefficients `c`, from `X` to kind `Y`: each output coefficient is
/// `Σ c x̂ᵢ x̂ⱼ x̂ₖ` over `i ≤ j ≤ k`, with `x̂ = (1, x…)`, every monomial of degree at most 3.
fn cubic<X, Y>(c: &[i64], x: X) -> Y::Mv<(), X::Coef>
where
    X: Extensor<Slots = ()>,
    Y: Kind,
{
    let one = X::Coef::one();
    let v: Vec<X::Coef> = core::iter::once(one)
        .chain(x.coeffs().as_ref().iter().copied())
        .collect();
    let n = v.len();
    let mut terms = c.iter();
    Extensor::from_coeffs(Y::arr_from_fn(|_| {
        let mut sum = X::Coef::zero();
        for i in 0..n {
            for j in i..n {
                for k in j..n {
                    let ck = X::Coef::from_i64(*terms.next().expect("enough coefficients"));
                    sum = sum + ck * v[i] * v[j] * v[k];
                }
            }
        }
        sum
    }))
}

macro_rules! laws {
    ($name:ident, $alg:ident, $n:expr) => {
        mod $name {
            use super::*;
            use gax::calculus::{curl, divergence, laplacian, nabla};
            use gax::$alg::{Bivector, Scalar, Vector};

            type V = Vector<(), Fp>;

            /// `∇x = n`: its divergence is the dimension, its curl zero.
            #[test]
            fn identity_field() {
                let mut rng = Rng(1);
                for _ in 0..4 {
                    let x: V = rng.point();
                    let (_, d) = nabla(|x| x, x);
                    assert_eq!(d.grade::<0>().s(), Fp::from_i64($n));
                    assert_eq!(d.grade::<2>(), Bivector::zero());
                    assert_eq!(divergence(|x| x, x).1.s(), Fp::from_i64($n));
                    assert_eq!(curl(|x| x, x).1, Bivector::zero());
                }
            }

            /// `∇(x²) = 2x`.
            #[test]
            fn gradient_of_the_square() {
                let mut rng = Rng(2);
                for _ in 0..4 {
                    let x: V = rng.point();
                    let (_, d) = nabla(|x| x.scalar_product(x), x);
                    assert_eq!(d, x + x);
                }
            }

            /// `∂_X (X ∗ A) = A`, for vectors and bivectors.
            #[test]
            fn derivative_of_the_pairing() {
                let mut rng = Rng(3);
                for _ in 0..4 {
                    let (x, a): (V, V) = (rng.point(), rng.point());
                    let (_, d) = nabla(
                        |x: Vector<(), _>| x.scalar_product(a.map_coefs(Into::into)),
                        x,
                    );
                    assert_eq!(d, a);
                    let (x, a): (Bivector<(), Fp>, Bivector<(), Fp>) = (rng.point(), rng.point());
                    let (_, d) = nabla(
                        |x: Bivector<(), _>| x.scalar_product(a.map_coefs(Into::into)),
                        x,
                    );
                    assert_eq!(d, a);
                }
            }

            /// `∇∧(∇∧f) = 0` for a scalar field and `∇·(∇·B) = 0` for a bivector field.
            #[test]
            fn second_derivatives_vanish() {
                fn grad<T: Coef>(c: &[i64], x: Vector<(), T>) -> Vector<(), T> {
                    nabla(|x| cubic::<_, Scalar>(c, x), x).1
                }
                fn div<T: Coef>(c: &[i64], x: Vector<(), T>) -> Vector<(), T> {
                    divergence(|x| cubic::<_, Bivector>(c, x), x).1
                }
                let mut rng = Rng(4);
                for _ in 0..4 {
                    let (c, x) = (rng.cubic(), rng.point::<V>());
                    assert_eq!(curl(|x| grad(&c, x), x).1, Bivector::zero());
                    assert_eq!(divergence(|x| div(&c, x), x).1, Scalar::zero());
                }
            }

            /// For a vector field, `∇F` is `∇·F + ∇∧F`.
            #[test]
            fn nabla_is_divergence_plus_curl() {
                let mut rng = Rng(5);
                for _ in 0..4 {
                    let (c, x) = (rng.cubic(), rng.point::<V>());
                    let field = |x| cubic::<_, Vector>(&c, x);
                    let (_, d) = nabla(field, x);
                    assert_eq!(d.grade::<0>(), divergence(field, x).1);
                    assert_eq!(d.grade::<2>(), curl(field, x).1);
                }
            }

            /// `∇²f = ∇·(∇f)`, and for a vector field its value matches.
            #[test]
            fn laplacian_is_divergence_of_gradient() {
                fn grad<T: Coef>(c: &[i64], x: Vector<(), T>) -> Vector<(), T> {
                    nabla(|x| cubic::<_, Scalar>(c, x), x).1
                }
                let mut rng = Rng(6);
                for _ in 0..4 {
                    let (c, x) = (rng.cubic(), rng.point::<V>());
                    let (f, lap) = laplacian(|x| cubic::<_, Scalar>(&c, x), x);
                    assert_eq!(f, cubic::<_, Scalar>(&c, x));
                    assert_eq!(lap, divergence(|x| grad(&c, x), x).1);
                }
            }
        }
    };
}

laws!(vga2d, vga2d, 2);
laws!(vga3d, vga3d, 3);
laws!(sta, sta, 4);
laws!(cga3d, cga3d, 5);

/// Covariance under a rotor: the derivative of the rotated field `y ↦ R F(~R y R) ~R` at
/// `R x ~R` is `R (∇F)(x) ~R`. The rotor `(3 + 4 e12) / 5` is unit with rational coefficients,
/// so the check is exact.
#[test]
fn nabla_is_covariant() {
    use gax::calculus::nabla;
    use gax::vga3d::{Rotor, Vector};
    let r = Rotor::<(), Fp>::new(
        Fp::from_ratio(3, 5),
        Fp::zero(),
        Fp::zero(),
        Fp::from_ratio(4, 5),
    );
    let mut rng = Rng(7);
    for _ in 0..4 {
        let (c, x) = (rng.cubic(), rng.point::<Vector<(), Fp>>());
        let field = |x| cubic::<_, Vector>(&c, x);
        let rotated = |y: Vector<(), _>| {
            let r = r.map_coefs(Into::into);
            r >> field(r.reverse() >> y)
        };
        let (_, got) = nabla(rotated, r >> x);
        let (_, want) = nabla(field, x);
        assert_eq!(got, r >> want);
    }
}

/// A plane wave in vacuum: `F = (k ∧ a) φ(k · x)` with `k` null and `a ⟂ k` satisfies Maxwell's
/// `∇F = 0`, for any profile `φ` (here a random cubic).
#[test]
fn plane_wave_satisfies_maxwell() {
    use gax::calculus::nabla;
    use gax::sta::{Scalar, Vector};
    let mut rng = Rng(8);
    let k = Vector::<(), Fp>::new(Fp::one(), Fp::one(), Fp::zero(), Fp::zero());
    let a = Vector::<(), Fp>::new(Fp::zero(), Fp::zero(), Fp::one(), Fp::zero());
    for _ in 0..4 {
        let c = rng.cubic();
        let x = rng.point::<Vector<(), Fp>>();
        let wave = |x: Vector<(), _>| {
            let (k, a) = (k.map_coefs(Into::into), a.map_coefs(Into::into));
            let phase: Scalar<(), _> = k.scalar_product(x);
            (k ^ a) * cubic::<_, Scalar>(&c, phase)
        };
        let (f, d) = nabla(wave, x);
        assert!(f.coeffs().as_ref().iter().any(|&c| c != Fp::zero()));
        assert!(d.coeffs().as_ref().iter().all(|&c| c == Fp::zero()));
    }
}

/// The field of a point charge, `x / |x|³`, has no divergence and no curl away from the charge
/// (it takes a root, so in `f64`).
#[test]
fn point_charge_is_free_away_from_the_charge() {
    use gax::Real;
    use gax::calculus::{curl, divergence};
    use gax::vga3d::Vector;
    fn coulomb<T: Real>(x: Vector<(), T>) -> Vector<(), T> {
        let r2 = x.scalar_product(x).s();
        x * (r2 * r2.sqrt()).recip()
    }
    for x in [[0.3, -1.0, 2.0], [1.0, 0.0, 0.0], [-0.2, 0.4, -0.1]] {
        let x = Vector::<(), f64>::new(x[0], x[1], x[2]);
        assert!(divergence(coulomb, x).1.s().abs() < 1e-12);
        assert!(curl(coulomb, x).1.c.iter().all(|c| c.abs() < 1e-12));
    }
}

gax::algebra! {
    algebra skew "A null pair `eu · ew = 2` beside a unit `e1`, and a null `en`: an off-diagonal metric.";
    basis eu = 0, ew = 0, e1 = 1, en = 0;
    metric eu ew = 2;
    kind Scalar = [1];
    kind Vector = [eu, ew, e1];
    kind Bivector = [euw, eu1, ew1];
    kind Null = [en, e1];
}

/// An `algebra!` declaration gets reciprocal frames from its own metric, off-diagonal ones
/// included: `e^i ∗ e_k = δ_ik` for every blade, and `∇x` is the dimension.
#[test]
fn user_algebra_frames() {
    use gax::calculus::{Reciprocal, divergence, reciprocal};
    use skew::{Bivector, Vector};
    fn check<K: Reciprocal>()
    where
        K::Mv<(), Fp>: gax::ScalarProduct<K::Mv<(), Fp>>,
        <K::Mv<(), Fp> as gax::ScalarProduct<K::Mv<(), Fp>>>::Output:
            Extensor<Coef = Fp, Slots = ()>,
    {
        for i in 0..K::N {
            let up: K::Mv<(), Fp> = reciprocal(i);
            for k in 0..K::N {
                let e = K::Mv::<(), Fp>::from_coeffs(K::arr_from_fn(|j| {
                    if j == k { Fp::one() } else { Fp::zero() }
                }));
                let pairing = gax::ScalarProduct::scalar_product(up, e).coeffs().as_ref()[0];
                assert_eq!(pairing, if i == k { Fp::one() } else { Fp::zero() });
            }
        }
    }
    check::<Vector>();
    check::<Bivector>();
    let up_u: Vector<(), Fp> = reciprocal(0);
    assert_eq!(
        up_u,
        Vector::new(Fp::zero(), Fp::from_ratio(1, 2), Fp::zero())
    );
    let x = Vector::<(), Fp>::new(Fp::new(3), Fp::new(5), Fp::new(7));
    assert_eq!(divergence(|x| x, x).1.s(), Fp::from_i64(3));
}
