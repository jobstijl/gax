//! Integration over simplices (ADR-048): quadrature rules of any degree in any dimension, and
//! the directed measure `dX` of geometric calculus.
//!
//! A `k`-simplex is `k + 1` positions. What a position is depends on the model of space: a vector
//! in VGA and STA, a normalized point in PGA, a conformal point in CGA. [`Position`] says how a
//! model forms the points of a simplex from its vertices, and [`Measure`] gives the simplex's
//! directed measure, a blade whose norm is its `k`-volume. The standard algebras implement both
//! for their positions.
//!
//! The rules are Grundmann and Möller's: of odd degree `2s + 1`, exact on every polynomial of
//! that degree, with rational nodes and weights. They run on any coefficient type, and on the
//! prime field `gax::fp::Fp` an integral of a polynomial is exact.
//!
//! ```
//! use gax::simplex::{Rule, Simplex};
//! use gax::vga3d::{Scalar, Vector};
//!
//! // The mean of x² over the triangle (0,0,0), (1,0,0), (0,1,0) is 1/6, and of x y is 1/12.
//! let o = Vector::<(), f64>::new(0.0, 0.0, 0.0);
//! let t = Simplex([o, Vector::new(1.0, 0.0, 0.0), Vector::new(0.0, 1.0, 0.0)]);
//! let xx = t.mean(Rule::degree(2), |p| Scalar::new(p.c[0] * p.c[0]));
//! let xy = t.mean(Rule::degree(2), |p| Scalar::new(p.c[0] * p.c[1]));
//! assert!((xx.s() - 1.0 / 6.0).abs() < 1e-15 && (xy.s() - 1.0 / 12.0).abs() < 1e-15);
//! // ∫ x² dA is the mean times the area, 1/2; ∫ dX x² is the directed measure e12/2 times the
//! // mean.
//! let x2 = |p: Vector<(), f64>| Scalar::new(p.c[0] * p.c[0]);
//! assert!((t.integral(Rule::degree(2), x2).s() - 1.0 / 12.0).abs() < 1e-15);
//! assert!((t.directed(Rule::degree(2), x2).c[2] - 1.0 / 12.0).abs() < 1e-15);
//! ```

use crate::coef::{Coef, Real};
use crate::ops::{Gp, Norm};
use core::ops::Add;

/// The most vertices a simplex may have: an 8-simplex, in nine dimensions.
pub const MAX_VERTICES: usize = 9;

/// A model of affine space: its positions, and the points of a simplex as affine combinations
/// of the vertices.
pub trait Position: Copy {
    /// The coefficient type.
    type Coef: Coef;
    /// The affine combination `Σ λᵢ pᵢ` of the positions, for weights with `Σ λᵢ = 1`.
    fn affine(points: &[Self], weights: &[Self::Coef]) -> Self;
}

/// The directed measure of a simplex of `N` vertices in a model: a blade whose norm is the
/// simplex's `(N − 1)`-volume and whose attitude and sign are its span and orientation.
pub trait Measure<const N: usize>: Position {
    /// The blade.
    type Blade: Copy;
    /// The directed measure of the simplex with these vertices, in their order.
    fn measure(vertices: &[Self; N]) -> Self::Blade;
}

/// A quadrature rule on simplices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// Grundmann and Möller's rule of degree `2s + 1`: exact on polynomials of that degree.
    /// Its weights alternate in sign from `s = 1`.
    GrundmannMoller {
        /// The rule's `s`.
        s: u32,
    },
    /// The vertices with weight `1 / ((k+1)(k+2))` each and the centroid with `(k+1)/(k+2)`:
    /// exact on quadratics, with positive weights (on an edge, Simpson's rule, exact on cubics).
    VerticesAndCentroid,
}

impl Rule {
    /// The highest degree a rule is offered for: rounding grows with the degree, through the
    /// alternating weights.
    pub const MAX_DEGREE: u32 = 15;

    /// The cheapest Grundmann–Möller rule exact on polynomials of degree `d`.
    ///
    /// # Panics
    /// If `d` is above [`Rule::MAX_DEGREE`].
    #[must_use]
    pub fn degree(d: u32) -> Rule {
        assert!(
            d <= Self::MAX_DEGREE,
            "no simplex rule of degree {d}: the rules go to degree {}",
            Self::MAX_DEGREE
        );
        Rule::GrundmannMoller {
            s: d.saturating_sub(1).div_ceil(2),
        }
    }

    /// The degree of polynomials the rule integrates exactly.
    #[must_use]
    pub fn exact_degree(self) -> u32 {
        match self {
            Rule::GrundmannMoller { s } => 2 * s + 1,
            Rule::VerticesAndCentroid => 2,
        }
    }

    /// Visit the rule's nodes on the `k`-simplex: `f(λ, w)` for each, with `λ` its barycentric
    /// coordinates (`k + 1` of them) and `w` its weight. The weights sum to one, so the mean of
    /// `g` is `Σ w g(λ)`.
    ///
    /// # Panics
    /// If `k + 1` is above [`MAX_VERTICES`].
    pub fn for_each_node<T: Coef>(self, k: usize, mut f: impl FnMut(&[T], T)) {
        assert!(
            k < MAX_VERTICES,
            "a simplex has at most {MAX_VERTICES} vertices"
        );
        let mut node = [T::zero(); MAX_VERTICES];
        let node = &mut node[..=k];
        match self {
            Rule::VerticesAndCentroid => {
                let n = k as i64 + 1;
                for v in 0..=k {
                    for (j, l) in node.iter_mut().enumerate() {
                        *l = if j == v { T::one() } else { T::zero() };
                    }
                    f(node, T::from_ratio(1, n * (n + 1)));
                }
                node.fill(T::from_ratio(1, n));
                f(node, T::from_ratio(n, n + 1));
            }
            Rule::GrundmannMoller { s } => {
                let (s, d) = (s as usize, 2 * s as usize + 1);
                for i in 0..=s {
                    let m = (d + k - 2 * i) as i64;
                    let w = grundmann_moller_weight::<T>(s, k, i);
                    compositions(s - i, k + 1, &mut |beta| {
                        for (l, &b) in node.iter_mut().zip(beta) {
                            *l = T::from_ratio(2 * b as i64 + 1, m);
                        }
                        f(node, w);
                    });
                }
            }
        }
    }
}

/// The weight of Grundmann and Möller's rule (1978, Theorem 4) for the mean over a `k`-simplex,
/// on its `i`-th layer of nodes: `(−1)ⁱ 2⁻²ˢ (d+k−2i)ᵈ k! / (i! (d+k−i)!)` with `d = 2s + 1`. The
/// layer's nodes are `(2β + 1)/(d+k−2i)` for `β ∈ ℕᵏ⁺¹`, `|β| = s − i`.
fn grundmann_moller_weight<T: Coef>(s: usize, k: usize, i: usize) -> T {
    let d = 2 * s + 1;
    let m = (d + k - 2 * i) as i64;
    // A product of small factors, exact in every coefficient type: mᵈ, then 1/j over the
    // factors of (d+k−i)! and of i!, then k! and 4⁻ˢ.
    let mut w = if i.is_multiple_of(2) {
        T::one()
    } else {
        -T::one()
    };
    for _ in 0..d {
        w = w * T::from_i64(m);
    }
    for j in (1..=(d + k - i)).chain(1..=i) {
        w = w * T::from_ratio(1, j as i64);
    }
    for j in 1..=k {
        w = w * T::from_i64(j as i64);
    }
    for _ in 0..s {
        w = w * T::from_ratio(1, 4);
    }
    w
}

/// Visit every `β ∈ ℕᵖᵃʳᵗˢ` with `|β| = total`, in lexicographic order (largest first part
/// first).
fn compositions(total: usize, parts: usize, f: &mut impl FnMut(&[usize])) {
    fn go(total: usize, at: usize, beta: &mut [usize], f: &mut impl FnMut(&[usize])) {
        if at + 1 == beta.len() {
            beta[at] = total;
            f(beta);
            return;
        }
        for first in (0..=total).rev() {
            beta[at] = first;
            go(total - first, at + 1, beta, f);
        }
    }
    let mut beta = [0; MAX_VERTICES];
    go(total, 0, &mut beta[..parts], f);
}

/// A simplex: `N` positions, a `(N − 1)`-simplex, oriented by their order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Simplex<P, const N: usize>(pub [P; N]);

impl<P: Position, const N: usize> Simplex<P, N> {
    /// The mean of `f` over the simplex by the rule: a value, or a map, of any kind.
    pub fn mean<F>(&self, rule: Rule, f: impl Fn(P) -> F) -> F
    where
        F: Add<Output = F> + Gp<P::Coef, Output = F>,
    {
        let mut sum: Option<F> = None;
        rule.for_each_node::<P::Coef>(N - 1, |node, w| {
            let term = f(P::affine(&self.0, node)).gp(w);
            sum = Some(match sum.take() {
                Some(s) => s + term,
                None => term,
            });
        });
        sum.expect("a rule has a node")
    }

    /// The directed measure `∫ dX`: a blade whose norm is the volume.
    pub fn measure(&self) -> P::Blade
    where
        P: Measure<N>,
    {
        P::measure(&self.0)
    }

    /// `∫ dX F`: on a flat simplex the measure is constant, so this is the measure times the
    /// mean of `F` (the measure on the left, as in `∫ dX ∇F = ∮ dS F`).
    pub fn directed<F, O>(&self, rule: Rule, f: impl Fn(P) -> F) -> O
    where
        P: Measure<N>,
        P::Blade: Gp<F, Output = O>,
        F: Add<Output = F> + Gp<P::Coef, Output = F>,
    {
        self.measure().gp(self.mean(rule, f))
    }

    /// `∫ F dV`: the mean times the volume, the norm of the measure.
    pub fn integral<F>(&self, rule: Rule, f: impl Fn(P) -> F) -> F
    where
        P: Measure<N>,
        P::Coef: Real,
        P::Blade: Norm<Coef = P::Coef>,
        F: Add<Output = F> + Gp<P::Coef, Output = F>,
    {
        self.mean(rule, f).gp(self.measure().norm())
    }
}

/// `∫ f g dV` over a simplex with homogeneous vertices `vs` (PGA's points) and signed size
/// `size`, for the dyad `dyad(x) = f(x) g(x)` of two linear forms: `size / ((k+1)(k+2))
/// (Σ dyad(vᵢ) + dyad(Σ vᵢ))`, `k + 1 = vs.len()`. It is [`Rule::VerticesAndCentroid`] with the
/// centroid's weight folded into the dyad's homogeneity (`dyad(Σ vᵢ) = (k+1)² dyad(c)`), which
/// saves the division (`gax::moments`).
#[inline]
pub fn dyads<P, F, T>(vs: &[P], size: T, dyad: impl Fn(P) -> F) -> F
where
    P: Copy + Add<Output = P>,
    F: Add<Output = F> + Gp<T, Output = F>,
    T: Real,
{
    let n = vs.len() as i64;
    let sum = vs[1..].iter().fold(vs[0], |a, &b| a + b);
    let all = vs.iter().fold(dyad(sum), |acc, &v| acc + dyad(v));
    all.gp(size * T::from_i64(n * (n + 1)).recip())
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::fp::Fp;
    use std::vec::Vec;

    fn factorial(n: usize) -> Fp {
        (1..=n).fold(Fp::one(), |a, j| a * Fp::from_i64(j as i64))
    }

    /// The mean of `λ^α` by the rule, and its exact value `k! α! / (|α| + k)!`.
    fn monomial(rule: Rule, k: usize, alpha: &[usize]) -> (Fp, Fp) {
        let mut got = Fp::zero();
        rule.for_each_node::<Fp>(k, |node, w| {
            let mono = node
                .iter()
                .zip(alpha)
                .fold(Fp::one(), |m, (&l, &e)| (0..e).fold(m, |m, _| m * l));
            got = got + w * mono;
        });
        let order: usize = alpha.iter().sum();
        let alpha_fact = alpha.iter().fold(Fp::one(), |a, &e| a * factorial(e));
        (got, factorial(k) * alpha_fact / factorial(order + k))
    }

    fn all_compositions(total: usize, parts: usize) -> Vec<Vec<usize>> {
        let mut out = Vec::new();
        compositions(total, parts, &mut |b| out.push(b.to_vec()));
        out
    }

    /// Every rule is exact on the barycentric monomials up to its degree, exactly, on the prime
    /// field; and its weights sum to one. Every monomial is checked: up to tetrahedra and
    /// 4-simplices for the rules to degree 9, on edges and triangles for the higher ones.
    #[test]
    #[cfg_attr(
        miri,
        ignore = "arithmetic only, no unsafe code: millions of operations, hours under miri"
    )]
    fn rules_are_exact_on_monomials() {
        let rules = (0..=7)
            .map(|s| Rule::GrundmannMoller { s })
            .chain([Rule::VerticesAndCentroid]);
        for rule in rules {
            let top = if rule.exact_degree() > 9 { 2 } else { 4 };
            for k in 1..=top {
                let mut total = Fp::zero();
                rule.for_each_node::<Fp>(k, |_, w| total = total + w);
                assert_eq!(total, Fp::one());
                for t in 0..=rule.exact_degree() as usize {
                    for alpha in all_compositions(t, k + 1) {
                        let (got, want) = monomial(rule, k, &alpha);
                        assert_eq!(got, want, "k = {k}, {rule:?}, α = {alpha:?}");
                    }
                }
            }
        }
    }

    /// One degree past it, a rule is wrong on some monomial: the degree claimed is the true one.
    /// (On an edge the vertices and the centroid are Simpson's rule, exact on cubics.)
    #[test]
    #[cfg_attr(
        miri,
        ignore = "arithmetic only, no unsafe code: millions of operations, hours under miri"
    )]
    fn rules_are_not_exact_beyond_their_degree() {
        let rules = (0..=7)
            .map(|s| Rule::GrundmannMoller { s })
            .chain([Rule::VerticesAndCentroid]);
        for rule in rules {
            let lowest = if rule == Rule::VerticesAndCentroid {
                2
            } else {
                1
            };
            for k in lowest..=4 {
                let t = rule.exact_degree() as usize + 1;
                let wrong = all_compositions(t, k + 1).iter().any(|alpha| {
                    let (got, want) = monomial(rule, k, alpha);
                    got != want
                });
                assert!(wrong, "k = {k}, {rule:?}");
            }
        }
    }

    /// `Rule::degree` picks the cheapest rule that is exact.
    #[test]
    fn degree_picks_the_cheapest_exact_rule() {
        let picked: Vec<u32> = (0..=15).map(|d| Rule::degree(d).exact_degree()).collect();
        assert_eq!(
            picked,
            [1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15]
        );
    }

    /// In `f64` the rules' rounding grows with their cancellation, `Σ |w|`: at most `10⁻¹²`
    /// relative on every monomial of their degree, up to degree 15 and 4-simplices (numerics.md
    /// has the table).
    #[test]
    #[cfg_attr(
        miri,
        ignore = "arithmetic only, no unsafe code: millions of operations, hours under miri"
    )]
    fn rounding_stays_within_the_documented_bound() {
        fn fact(n: usize) -> f64 {
            (1..=n).map(|j| j as f64).product()
        }
        let bounds = [1e-15, 2e-15, 3e-15, 3e-14, 1.2e-13, 3e-13, 7e-13, 2e-12];
        for (s, bound) in (0..=7).zip(bounds) {
            let rule = Rule::GrundmannMoller { s };
            for k in 1..=4 {
                for t in 0..=rule.exact_degree() as usize {
                    for alpha in all_compositions(t, k + 1) {
                        let mut got = 0.0;
                        rule.for_each_node::<f64>(k, |node, w| {
                            let mono: f64 = node
                                .iter()
                                .zip(&alpha)
                                .map(|(l, &e)| l.powi(e as i32))
                                .product();
                            got += w * mono;
                        });
                        let want =
                            fact(k) * alpha.iter().map(|&e| fact(e)).product::<f64>() / fact(t + k);
                        assert!(
                            ((got - want) / want).abs() <= bound,
                            "s = {s}, k = {k}, α = {alpha:?}"
                        );
                    }
                }
            }
        }
    }
}
