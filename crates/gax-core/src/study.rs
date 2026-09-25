//! Functions of Study numbers, for the exponential and logarithm of bivectors and versors.
//!
//! In an algebra of dimension at most 4, the square of a bivector is a Study number
//! `z = a + b I`: a scalar plus a multiple of the pseudoscalar `I`, with `I²` equal to `0`
//! (degenerate metrics, as in plane-based PGA), `+1` or `-1` (as in spacetime algebra). Such
//! numbers form a commutative two-dimensional algebra, so any analytic function extends to
//! them through its values on two *channels*:
//!
//! * `I² = +1`: two real channels `a ± b`,
//! * `I² = -1`: one complex channel `a + i b`,
//! * `I² = 0`: dual numbers, `f(a + b I) = f(a) + b f'(a) I`.
//!
//! The functions themselves ([`exp_parts`], [`log_factor`]) are written once over a
//! [`Channel`] number type and evaluated in complex arithmetic, so they cover rotations
//! (negative squares) and boosts (positive squares) alike, without branches.

use crate::coef::Real;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// A complex number.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cx<T> {
    /// Real part.
    pub re: T,
    /// Imaginary part.
    pub im: T,
}

impl<T: Real> Cx<T> {
    /// A real number.
    #[inline(always)]
    pub fn real(re: T) -> Self {
        Cx { re, im: T::zero() }
    }
}

impl<T: Real> Add for Cx<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Cx {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
}
impl<T: Real> Sub for Cx<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Cx {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }
}
impl<T: Real> Neg for Cx<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Cx {
            re: -self.re,
            im: -self.im,
        }
    }
}
impl<T: Real> Mul for Cx<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        Cx {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
}
impl<T: Real> Div for Cx<T> {
    type Output = Self;
    #[inline(always)]
    fn div(self, o: Self) -> Self {
        let d = (o.re * o.re + o.im * o.im).recip();
        Cx {
            re: (self.re * o.re + self.im * o.im) * d,
            im: (self.im * o.re - self.re * o.im) * d,
        }
    }
}

/// Numbers the channel functions are evaluated in: [`Cx`] and dual numbers over [`Cx`].
pub trait Channel<T: Real>:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Embed a real number.
    fn real(x: T) -> Self;
    /// Principal square root.
    fn sqrt(self) -> Self;
    /// Principal natural logarithm.
    fn ln(self) -> Self;
    /// Hyperbolic sine.
    fn sinh(self) -> Self;
    /// Hyperbolic cosine.
    fn cosh(self) -> Self;
    /// Squared magnitude of the value (the primal part, for dual numbers).
    fn norm2(self) -> T;
    /// Lane-wise `if self.norm2() < t { a } else { b }`.
    #[inline(always)]
    fn select_small(self, t: T, a: Self, b: Self) -> Self {
        Self::select(self.norm2(), t, a, b)
    }
    /// Lane-wise `if x < t { a } else { b }`.
    fn select(x: T, t: T, a: Self, b: Self) -> Self;
}

impl<T: Real> Channel<T> for Cx<T> {
    #[inline(always)]
    fn real(x: T) -> Self {
        Cx::real(x)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        // Principal root: re = sqrt((|z| + a) / 2), im = sign(b) sqrt((|z| - a) / 2).
        let half = T::from_f64(0.5);
        let r = (self.re * self.re + self.im * self.im).sqrt();
        let re = ((r + self.re) * half).max(T::zero()).sqrt();
        let im = ((r - self.re) * half).max(T::zero()).sqrt();
        Cx {
            re,
            im: T::select_lt(self.im, T::zero(), -im, im),
        }
    }
    #[inline(always)]
    fn ln(self) -> Self {
        let half = T::from_f64(0.5);
        Cx {
            re: (self.re * self.re + self.im * self.im).ln() * half,
            im: self.im.atan2(self.re),
        }
    }
    #[inline(always)]
    fn sinh(self) -> Self {
        let (s, c) = self.im.sin_cos();
        Cx {
            re: self.re.sinh() * c,
            im: self.re.cosh() * s,
        }
    }
    #[inline(always)]
    fn cosh(self) -> Self {
        let (s, c) = self.im.sin_cos();
        Cx {
            re: self.re.cosh() * c,
            im: self.re.sinh() * s,
        }
    }
    #[inline(always)]
    fn norm2(self) -> T {
        self.re * self.re + self.im * self.im
    }
    #[inline(always)]
    fn select(x: T, t: T, a: Self, b: Self) -> Self {
        Cx {
            re: T::select_lt(x, t, a.re, b.re),
            im: T::select_lt(x, t, a.im, b.im),
        }
    }
}

/// A dual number `p + d ε` with `ε² = 0`: forward-mode derivatives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual<N> {
    /// Primal part.
    pub p: N,
    /// Derivative part.
    pub d: N,
}

impl<N: Add<Output = N>> Add for Dual<N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Dual {
            p: self.p + o.p,
            d: self.d + o.d,
        }
    }
}
impl<N: Sub<Output = N>> Sub for Dual<N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Dual {
            p: self.p - o.p,
            d: self.d - o.d,
        }
    }
}
impl<N: Neg<Output = N>> Neg for Dual<N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Dual {
            p: -self.p,
            d: -self.d,
        }
    }
}
impl<N: Copy + Add<Output = N> + Mul<Output = N>> Mul for Dual<N> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        Dual {
            p: self.p * o.p,
            d: self.p * o.d + self.d * o.p,
        }
    }
}
impl<N: Copy + Sub<Output = N> + Mul<Output = N> + Div<Output = N>> Div for Dual<N> {
    type Output = Self;
    #[inline(always)]
    fn div(self, o: Self) -> Self {
        let q = self.p / o.p;
        Dual {
            p: q,
            d: (self.d - q * o.d) / o.p,
        }
    }
}

impl<T: Real, N: Channel<T>> Channel<T> for Dual<N> {
    #[inline(always)]
    fn real(x: T) -> Self {
        Dual {
            p: N::real(x),
            d: N::real(T::zero()),
        }
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        let r = self.p.sqrt();
        Dual {
            p: r,
            d: self.d / (r + r),
        }
    }
    #[inline(always)]
    fn ln(self) -> Self {
        Dual {
            p: self.p.ln(),
            d: self.d / self.p,
        }
    }
    #[inline(always)]
    fn sinh(self) -> Self {
        Dual {
            p: self.p.sinh(),
            d: self.d * self.p.cosh(),
        }
    }
    #[inline(always)]
    fn cosh(self) -> Self {
        Dual {
            p: self.p.cosh(),
            d: self.d * self.p.sinh(),
        }
    }
    #[inline(always)]
    fn norm2(self) -> T {
        self.p.norm2()
    }
    #[inline(always)]
    fn select(x: T, t: T, a: Self, b: Self) -> Self {
        Dual {
            p: N::select(x, t, a.p, b.p),
            d: N::select(x, t, a.d, b.d),
        }
    }
}

/// `C(x) = cosh(√x)` and `S(x) = sinh(√x)/√x`, the even and odd parts of the exponential of an
/// element whose square is `x`: `exp(B) = C(B²) + S(B²) B`.
#[inline]
pub fn exp_parts<T: Real, N: Channel<T>>(x: N) -> (N, N) {
    let one = N::real(T::one());
    let r = x.sqrt();
    let c = r.cosh();
    let s = r.sinh() / r;
    // Series near x = 0, where sinh(r)/r is 0/0 (and its derivative cancels).
    let x2 = x * x;
    let x3 = x2 * x;
    let k = |v: f64| N::real(T::from_f64(v));
    let cs = one
        + x * k(1.0 / 2.0)
        + x2 * k(1.0 / 24.0)
        + x3 * k(1.0 / 720.0)
        + x3 * x * k(1.0 / 40320.0);
    let ss = one
        + x * k(1.0 / 6.0)
        + x2 * k(1.0 / 120.0)
        + x3 * k(1.0 / 5040.0)
        + x3 * x * k(1.0 / 362_880.0);
    let t = T::from_f64(1e-6);
    (x.select_small(t, cs, c), x.select_small(t, ss, s))
}

/// The factor `H` with `log(R) = H · ⟨R⟩₂`, for a versor `R = c + P` with `c = C(B²)` its
/// scalar (and pseudoscalar) part and `u = P²`.
///
/// `H = w / sinh(w)` with `cosh w = c`, evaluated as `2 atanh(t) / (t (1 + c))`,
/// `t = √u / (1 + c) = tanh(w / 2)`, which is stable near the identity (`u → 0`) and for
/// rotations up to (but not including) a half turn.
#[inline]
pub fn log_factor<T: Real, N: Channel<T>>(c: N, u: N) -> N {
    let one = N::real(T::one());
    let half = N::real(T::from_f64(0.5));
    let s = u.sqrt();
    let opc = one + c;
    let t = s / opc;
    // atanh(t) / t, with a series near t = 0.
    let direct = (((one + t) / (one - t)).ln() * half) / t;
    let t2 = t * t;
    let k = |v: f64| N::real(T::from_f64(v));
    let series = one + t2 * k(1.0 / 3.0) + t2 * t2 * k(1.0 / 5.0) + t2 * t2 * t2 * k(1.0 / 7.0);
    let ratio = t2.select_small(T::from_f64(1e-6), series, direct);
    ratio * N::real(T::from_i64(2)) / opc
}

/// Evaluate a function of one Study number `a + b I` (with `I² = isq`) through its channels,
/// returning the Study number `f0 + f1 I`.
#[inline]
pub fn study1<T: Real>(isq: i8, a: T, b: T, f: impl Fn(Dual<Cx<T>>) -> Dual<Cx<T>>) -> (T, T) {
    let zero = T::zero();
    match isq {
        0 => {
            // Dual numbers: f(a) + b f'(a) I.
            let r = f(Dual {
                p: Cx::real(a),
                d: Cx::real(b),
            });
            (r.p.re, r.d.re)
        }
        1 => {
            let lift = |x: T| Dual {
                p: Cx::real(x),
                d: Cx::real(zero),
            };
            let (fp, fm) = (f(lift(a + b)).p.re, f(lift(a - b)).p.re);
            let half = T::from_f64(0.5);
            ((fp + fm) * half, (fp - fm) * half)
        }
        _ => {
            let r = f(Dual {
                p: Cx { re: a, im: b },
                d: Cx::real(zero),
            });
            (r.p.re, r.p.im)
        }
    }
}

/// Like [`study1`], for a function of two Study numbers with the same `I`.
#[inline]
pub fn study2<T: Real>(
    isq: i8,
    x: (T, T),
    y: (T, T),
    f: impl Fn(Dual<Cx<T>>, Dual<Cx<T>>) -> Dual<Cx<T>>,
) -> (T, T) {
    let zero = T::zero();
    match isq {
        0 => {
            // f(x0 + x1 ε, y0 + y1 ε) = f + (x1 ∂x f + y1 ∂y f) ε: one dual evaluation.
            let r = f(
                Dual {
                    p: Cx::real(x.0),
                    d: Cx::real(x.1),
                },
                Dual {
                    p: Cx::real(y.0),
                    d: Cx::real(y.1),
                },
            );
            (r.p.re, r.d.re)
        }
        1 => {
            let lift = |v: T| Dual {
                p: Cx::real(v),
                d: Cx::real(zero),
            };
            let fp = f(lift(x.0 + x.1), lift(y.0 + y.1)).p.re;
            let fm = f(lift(x.0 - x.1), lift(y.0 - y.1)).p.re;
            let half = T::from_f64(0.5);
            ((fp + fm) * half, (fp - fm) * half)
        }
        _ => {
            let lift = |v: (T, T)| Dual {
                p: Cx { re: v.0, im: v.1 },
                d: Cx::real(zero),
            };
            let r = f(lift(x), lift(y));
            (r.p.re, r.p.im)
        }
    }
}

/// `(C, S)` of `exp(B) = C(B²) + S(B²) B` for `B² = lambda + mu I`, as Study numbers
/// `(c0, c1, s0, s1)`: `exp(B) = c0 + c1 I + (s0 + s1 I) B`.
///
/// ```
/// // A rotation (B² = -θ², no I part): exp(B) = cos θ + (sin θ / θ) B.
/// let th = 0.6f64;
/// let [c0, _, s0, _] = gax_core::study::exp_coeffs(-1, -th * th, 0.0);
/// assert!((c0 - th.cos()).abs() < 1e-15 && (s0 - th.sin() / th).abs() < 1e-15);
/// ```
#[inline]
pub fn exp_coeffs<T: Real>(isq: i8, lambda: T, mu: T) -> [T; 4] {
    let (c0, c1) = study1(isq, lambda, mu, |x| exp_parts(x).0);
    let (s0, s1) = study1(isq, lambda, mu, |x| exp_parts(x).1);
    [c0, c1, s0, s1]
}

/// `(h0, h1)` with `log(R) = (h0 + h1 I) ⟨R⟩₂` for a unit versor `R` whose scalar and
/// pseudoscalar parts are `c = c0 + c1 I` and whose bivector part squares to `u0 + u1 I`.
#[inline]
pub fn log_coeffs<T: Real>(isq: i8, c: (T, T), u: (T, T)) -> [T; 2] {
    let (h0, h1) = study2(isq, c, u, log_factor);
    [h0, h1]
}

/// `(a + b I)^(-1/2)` as a Study number `[r0, r1]`.
#[inline]
pub fn rsqrt<T: Real>(isq: i8, a: T, b: T) -> [T; 2] {
    let (r0, r1) = study1(isq, a, b, |z| {
        Dual {
            p: Cx::real(T::one()),
            d: Cx::real(T::zero()),
        } / z.sqrt()
    });
    [r0, r1]
}

/// A function of a Study number `a + X` whose non-scalar part squares to the scalar `q`
/// (`X² = q`), as `(f0, f1)` with `f(a + X) = f0 + f1 X`.
///
/// This covers any direction `X`, not only a fixed blade, which is what 5D algebras need: there
/// the square of a bivector is a scalar plus a 4-vector whose square is a scalar. With
/// `w = √q` (a complex root), `f0 = (f(a+w) + f(a−w))/2` and `f1 = (f(a+w) − f(a−w))/(2w)`;
/// near `q = 0` the derivative is used instead, `f1 = f'(a)`.
#[inline]
pub fn study_q<T: Real>(a: T, q: T, f: impl Fn(Dual<Cx<T>>) -> Dual<Cx<T>>) -> (T, T) {
    let zero = T::zero();
    let w = Cx::real(q).sqrt(); // real for q > 0, imaginary for q < 0
    let lift = |z: Cx<T>| Dual {
        p: z,
        d: Cx::real(zero),
    };
    let plus = f(lift(Cx::real(a) + w)).p;
    let minus = f(lift(Cx::real(a) - w)).p;
    let half = T::from_f64(0.5);
    let f0 = (plus.re + minus.re) * half;
    let diff = (plus - minus) / (w + w);
    // Near q = 0: the derivative, from a dual evaluation.
    let d = f(Dual {
        p: Cx::real(a),
        d: Cx::real(T::one()),
    });
    // Lane-wise: where |q| is tiny the finite difference is 0/0, and the select discards it.
    let eps = T::from_f64(1e-8);
    (
        T::select_lt(q.abs(), eps, d.p.re, f0),
        T::select_lt(q.abs(), eps, d.d.re, diff.re),
    )
}

/// [`exp_coeffs`] for `B² = lambda + Q` with `Q² = q`: `exp(B) = c0 + c1 Q + (s0 + s1 Q) B`.
#[inline]
pub fn exp_coeffs_q<T: Real>(lambda: T, q: T) -> [T; 4] {
    let (c0, c1) = study_q(lambda, q, |x| exp_parts(x).0);
    let (s0, s1) = study_q(lambda, q, |x| exp_parts(x).1);
    [c0, c1, s0, s1]
}

/// `acosh(y)²`, the square of the bivector whose exponential has scalar part `y`, analytic
/// at `y = 1` (series there).
#[inline]
fn acosh_sq<T: Real, N: Channel<T>>(y: N) -> N {
    let one = N::real(T::one());
    let t = y - one;
    let direct = {
        let w = (y + (y * y - one).sqrt()).ln();
        w * w
    };
    let k = |v: f64| N::real(T::from_f64(v));
    let series = t * k(2.0) - t * t * k(1.0 / 3.0) + t * t * t * k(4.0 / 45.0);
    t.select_small(T::from_f64(1e-8), series, direct)
}

/// [`log_coeffs`] for a unit versor `R = C + P` whose scalar-plus-4-vector part is
/// `C = c0 + C4` with `C4² = qc`: `log R = h0 P + h1 C4 P`.
///
/// `B² = acosh(C)²` is found from `C` alone, and `B = S(B²)⁻¹ P`.
#[inline]
pub fn log_coeffs_q<T: Real>(c0: T, qc: T) -> [T; 2] {
    let (g0, g1) = study_q(c0, qc, acosh_sq);
    let qx = g1 * g1 * qc;
    let (s0, s1) = study_q(g0, qx, |x| exp_parts(x).1);
    // (s0 + s1 g1 C4)⁻¹ = (s0 − s1 g1 C4) / (s0² − s1² g1² qc)
    let d = (s0 * s0 - s1 * s1 * qx).recip();
    [s0 * d, -(s1 * g1) * d]
}

/// Fast path of [`exp_coeffs`] for rotations: `B² = lambda + mu I` with `lambda <= 0` and
/// `I² = 0` (or `mu = 0`), as for every bivector of plane-based PGA. With `a = √(-λ)`:
/// `C = cos a`, `S = sin a / a`, `C' = S / 2`, `S' = (S - C) / (2 a²)`, and the `I` parts are
/// `μ C'` and `μ S'`. Series are used near `a = 0`.
#[inline]
pub fn exp_coeffs_rotation<T: Real>(lambda: T, mu: T) -> [T; 4] {
    let a2 = (-lambda).max(T::zero());
    let a = a2.sqrt();
    let (sin, cos) = a.sin_cos();
    let k = |v: f64| T::from_f64(v);
    let small = k(1e-4);
    let s = T::select_lt(
        a2,
        small,
        T::one() - a2 * k(1.0 / 6.0) + a2 * a2 * k(1.0 / 120.0),
        sin / a,
    );
    let ds = T::select_lt(
        a2,
        small,
        k(1.0 / 6.0) - a2 * k(1.0 / 60.0),
        (s - cos) / (a2 + a2),
    );
    [cos, mu * s * k(0.5), s, mu * ds]
}

/// Fast path of [`log_coeffs`] for rotations: `R = c + P` with `P² = u0 + u1 I`, `u0 <= 0`
/// and `I² = 0` (or no `I` part). With `s = √(-u0)` and `θ = atan2(s, c0)`: `h0 = θ / s`, and
/// the `I` part is `c1 ∂h/∂c + u1 ∂h/∂u` with `∂h/∂c = -1/(c0² + s²)` and
/// `∂h/∂u = (θ - c0 s/(c0² + s²)) / (2 s³)`. Series are used near `s = 0`.
#[inline]
pub fn log_coeffs_rotation<T: Real>(c: (T, T), u: (T, T)) -> [T; 2] {
    let s2 = (-u.0).max(T::zero());
    let s = s2.sqrt();
    let theta = s.atan2(c.0);
    let n = c.0 * c.0 + s2;
    let k = |v: f64| T::from_f64(v);
    let small = k(1e-6);
    let h0 = T::select_lt(s2, small, (T::one() + s2 * k(1.0 / 6.0)) / c.0, theta / s);
    let g = T::select_lt(
        s2,
        small,
        k(1.0 / 3.0) + s2 * k(1.0 / 10.0),
        (theta - c.0 * s / n) / (s2 * s * k(2.0)),
    );
    [h0, -c.1 / n + u.1 * g]
}

/// Fast path of [`rsqrt`] for `I² = 0`: `(a + b I)^(-1/2) = a^(-1/2) - (b/2) a^(-3/2) I`.
#[inline]
pub fn rsqrt_nil<T: Real>(a: T, b: T) -> [T; 2] {
    let r = a.sqrt().recip();
    [r, -(b * r * r * r) * T::from_f64(0.5)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_channels() {
        // Rotation by angle 2θ: B² = -θ², exp = cos θ + (sin θ / θ) B.
        let th = 0.7f64;
        let [c0, _, s0, _] = exp_coeffs(1, -th * th, 0.0);
        assert!((c0 - th.cos()).abs() < 1e-15 && (s0 - th.sin() / th).abs() < 1e-15);
        // Boost: B² = φ².
        let ph = 0.4f64;
        let [c0, _, s0, _] = exp_coeffs(-1, ph * ph, 0.0);
        assert!((c0 - ph.cosh()).abs() < 1e-15 && (s0 - ph.sinh() / ph).abs() < 1e-15);
        // Log inverts: c = cos θ, u = P² = -sin² θ, H = θ / sin θ.
        let [h0, _] = log_coeffs(0, (th.cos(), 0.0), (-(th.sin() * th.sin()), 0.0));
        assert!((h0 - th / th.sin()).abs() < 1e-14);
        // Near the identity the series takes over.
        let [c0, _, s0, _] = exp_coeffs(0, -1e-9, 0.0);
        assert!((c0 - 1.0).abs() < 1e-9 && (s0 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fast_paths_match_channels() {
        for &(lam, mu) in &[(-0.49f64, 0.3), (-1e-7, 0.2), (-2.5, -0.7), (0.0, 0.1)] {
            let a = exp_coeffs(0, lam, mu);
            let b = exp_coeffs_rotation(lam, mu);
            for i in 0..4 {
                assert!((a[i] - b[i]).abs() < 1e-9, "exp {i}: {a:?} vs {b:?}");
            }
        }
        for &th in &[0.3f64, 1e-5, 2.5] {
            // A unit rotor-like Study pair: c = cos θ (+ c1 I), u = -sin² θ (+ u1 I) with
            // 2 c0 c1 - u1 = 0 (unit condition in the I part).
            let (c0, c1) = (th.cos(), 0.2);
            let (u0, u1) = (-(th.sin() * th.sin()), 2.0 * c0 * c1);
            let a = log_coeffs(0, (c0, c1), (u0, u1));
            let b = log_coeffs_rotation((c0, c1), (u0, u1));
            assert!(
                (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-8,
                "log θ={th}: {a:?} vs {b:?}"
            );
        }
        let a = rsqrt(0, 2.0f64, 0.6);
        let b = rsqrt_nil(2.0f64, 0.6);
        assert!((a[0] - b[0]).abs() < 1e-14 && (a[1] - b[1]).abs() < 1e-14);
    }

    #[test]
    fn general_study_functions() {
        // q < 0 behaves like a complex channel, q > 0 like two real channels, q = 0 like duals.
        for &(lam, mu, isq) in &[(-0.49f64, 0.3, -1i8), (0.3, 0.2, 1), (-0.8, 0.4, 0)] {
            let q = f64::from(isq) * mu * mu;
            let a = exp_coeffs(isq, lam, mu);
            let b = exp_coeffs_q(lam, q);
            // With X = mu I: f0 + f1 X = f0 + (f1 mu) I.
            if isq != 0 {
                assert!(
                    (a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1] * mu).abs() < 1e-12,
                    "{a:?} {b:?}"
                );
                assert!(
                    (a[2] - b[2]).abs() < 1e-12 && (a[3] - b[3] * mu).abs() < 1e-12,
                    "{a:?} {b:?}"
                );
            }
        }
        // log inverts exp: for B² = λ + Q, C(B²) = c0 + c1 Q; log gives back B = h0 P + h1 C4 P.
        // Check on a scalar case: c = cos θ, q = 0 -> h0 = θ / sin θ.
        let th = 0.7f64;
        let [h0, _] = log_coeffs_q(th.cos(), 0.0);
        assert!((h0 - th / th.sin()).abs() < 1e-8, "{h0}");
        let ph = 0.4f64;
        let [h0, _] = log_coeffs_q(ph.cosh(), 0.0);
        assert!((h0 - ph / ph.sinh()).abs() < 1e-8, "{h0}");
    }

    #[test]
    fn dual_channel_derivative() {
        // PGA-style: B² = λ + μ I with I² = 0: C = C(λ) + μ C'(λ) I, C'(λ) = sin(a)/(2a), a = √-λ.
        let a = 0.9f64;
        let mu = 0.3;
        let [c0, c1, _, _] = exp_coeffs(0, -a * a, mu);
        assert!((c0 - a.cos()).abs() < 1e-15);
        assert!((c1 - mu * a.sin() / (2.0 * a)).abs() < 1e-14);
    }
}
