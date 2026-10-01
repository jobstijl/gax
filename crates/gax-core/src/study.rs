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

mod expk;
mod log8;
pub use expk::{
    exp_reach_6d, exp_reach_8d, exp_turn_6d, exp_turn_8d, exp_weights_6d, exp_weights_8d, h_weights,
};
pub use log8::{log_coeffs_8d, log_turn_8d, log_weights_8d, q_weights_8d, turn_polynomial_8d};

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
        // Principal root, in the form without cancellation: t = sqrt((|z| + |a|) / 2) is the
        // larger part, and the other is b / (2t). (Computing the smaller part as
        // sqrt((|z| - |a|) / 2) cancels when |b| << |a|, which is exactly where the Study
        // functions take a derivative from the imaginary part: in f32 it lost all digits.)
        let (zero, one, half) = (T::zero(), T::one(), T::from_f64(0.5));
        let (a, b) = (self.re, self.im);
        let r = (a * a + b * b).sqrt();
        let t = ((r + a.abs()) * half).max(zero).sqrt();
        let safe = T::select_lt(zero, t, t, one); // z = 0: t = 0 and b = 0, so b / 2 = 0
        let other = b / (safe + safe);
        let signed = T::select_lt(b, zero, -t, t);
        Cx {
            re: T::select_lt(a, zero, other.abs(), t),
            im: T::select_lt(a, zero, signed, other),
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
#[inline(always)]
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
#[inline(always)]
pub fn log_factor<T: Real, N: Channel<T>>(c: N, u: N) -> N {
    let one = N::real(T::one());
    let half = N::real(T::from_f64(0.5));
    let s = u.sqrt();
    let opc = one + c;
    let t = s / opc;
    // atanh(t) / t. The direct form loses about ε/|t| (the log of a number near 1), so the
    // series Σ t^{2k}/(2k+1), k = 0..=7, covers |t²| < 1/100 (the next term is below 10⁻¹⁷).
    let direct = (((one + t) / (one - t)).ln() * half) / t;
    let t2 = t * t;
    let k = |v: f64| N::real(T::from_f64(v));
    let mut series = N::real(T::zero());
    for j in (0..8).rev() {
        series = series * t2 + k(1.0 / (2.0 * f64::from(j) + 1.0));
    }
    let ratio = t2.select_small(T::from_f64(1e-4), series, direct); // |t²|² < 10⁻⁴
    ratio * N::real(T::from_i64(2)) / opc
}

/// Evaluate a function of one Study number `a + b I` (with `I² = isq`) through its channels,
/// returning the Study number `f0 + f1 I`.
#[inline(always)]
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
#[inline(always)]
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
#[inline(always)]
pub fn exp_coeffs<T: Real>(isq: i8, lambda: T, mu: T) -> [T; 4] {
    let (c0, c1) = study1(isq, lambda, mu, |x| exp_parts(x).0);
    let (s0, s1) = study1(isq, lambda, mu, |x| exp_parts(x).1);
    [c0, c1, s0, s1]
}

/// `(h0, h1)` with `log(R) = (h0 + h1 I) ⟨R⟩₂` for a unit versor `R` whose scalar and
/// pseudoscalar parts are `c = c0 + c1 I` and whose bivector part squares to `u0 + u1 I`.
///
/// ```
/// use gax_core::study::{exp_coeffs, log_coeffs};
/// // A rotation R = exp(B) with B² = -θ²: its scalar part c0 = cos θ, and its bivector part
/// // P = s0 B squares to s0² B². The logarithm recovers B = h0 P.
/// let th = 0.6f64;
/// let [c0, _, s0, _] = exp_coeffs(-1, -th * th, 0.0);
/// let [h0, _] = log_coeffs(-1, (c0, 0.0), (s0 * s0 * -th * th, 0.0));
/// assert!((h0 * s0 - 1.0).abs() < 1e-12);
/// ```
#[inline(always)]
pub fn log_coeffs<T: Real>(isq: i8, c: (T, T), u: (T, T)) -> [T; 2] {
    let (h0, h1) = study2(isq, c, u, log_factor);
    [h0, h1]
}

/// `(a + b I)^(-1/2)` as a Study number `[r0, r1]`.
///
/// ```
/// use gax_core::study::rsqrt;
/// // (4 + I)^(-1/2) with I² = 0: 4^(-1/2) - (1/2) 4^(-3/2) I = 1/2 - I/16.
/// assert_eq!(rsqrt(0, 4.0f64, 1.0), [0.5, -0.0625]);
/// ```
#[inline(always)]
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
#[inline(always)]
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
#[inline(always)]
pub fn exp_coeffs_q<T: Real>(lambda: T, q: T) -> [T; 4] {
    let (c0, c1) = study_q(lambda, q, |x| exp_parts(x).0);
    let (s0, s1) = study_q(lambda, q, |x| exp_parts(x).1);
    [c0, c1, s0, s1]
}

/// `acosh(y)²`, the square of the bivector whose exponential has scalar part `y`, analytic
/// at `y = 1` (series there).
#[inline(always)]
fn acosh_sq<T: Real, N: Channel<T>>(y: N) -> N {
    let one = N::real(T::one());
    let t = y - one;
    let direct = {
        let w = (y + ((y - one) * (y + one)).sqrt()).ln();
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
#[inline(always)]
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
/// `μ C'` and `μ S'`.
///
/// `S` is evaluated directly except within `a² < 10⁻⁴` of `0`, where a series is used. `S'`
/// cancels (`S - C ≈ a²/3`), so its series, `Σ (-1)^k (k+1)/(2k+3)! a^{2k}`, covers `a² < 1/4`,
/// where the direct form loses under 30 ulps; both branches are then accurate to a few ulps at
/// the boundary (docs/numerics.md).
#[inline(always)]
pub fn exp_coeffs_rotation<T: Real>(lambda: T, mu: T) -> [T; 4] {
    // Horner in a²: coefficients (-1)^k (k+1) / (2k+3)!, k = 0..=7 (the next term is below
    // 10⁻²¹ at a² = 1/4).
    const DS: [f64; 8] = [
        1.0 / 6.0,
        -2.0 / 120.0,
        3.0 / 5040.0,
        -4.0 / 362_880.0,
        5.0 / 39_916_800.0,
        -6.0 / 6_227_020_800.0,
        7.0 / 1_307_674_368_000.0,
        -8.0 / 355_687_428_096_000.0,
    ];
    let a2 = (-lambda).max(T::zero());
    let a = a2.sqrt();
    let (sin, cos) = a.sin_cos();
    let k = |v: f64| T::from_f64(v);
    // One reciprocal serves both quotients (it is infinite at a = 0, where the series is used).
    let inv = a.recip();
    let s = T::select_lt(
        a2,
        k(1e-4),
        T::one() + a2 * (k(-1.0 / 6.0) + a2 * (k(1.0 / 120.0) + a2 * k(-1.0 / 5040.0))),
        sin * inv,
    );
    // Horner for S' in a².
    let mut series = T::zero();
    for c in DS.iter().rev() {
        series = series * a2 + k(*c);
    }
    let ds = T::select_lt(a2, k(0.25), series, (s - cos) * inv * inv * k(0.5));
    [cos, mu * s * k(0.5), s, mu * ds]
}

/// Fast path of [`log_coeffs`] for rotations: `R = c + P` with `P² = u0 + u1 I`, `u0 <= 0`
/// and `I² = 0` (or no `I` part). With `s = √(-u0)` and `θ = atan2(s, c0)`: `h0 = θ / s`, and
/// the `I` part is `c1 ∂h/∂c + u1 ∂h/∂u` with `∂h/∂c = -1/(c0² + s²)` and
/// `∂h/∂u = (θ - c0 s/(c0² + s²)) / (2 s³)`.
///
/// **Branch.** `θ ∈ [0, π]`: the rotation part of the result has half-angle `θ`, so
/// `exp(log R) = R` and a versor with `c0 < 0` gets the long way round (a rotation by more than
/// a half turn); negate `R` first for the shortest motion. Where the Euclidean part of `P`
/// vanishes and `c0 < 0` (`R = −T` for a translation `T`, including `R = −1`), no unique
/// logarithm exists and the result is `log(−R)`, the same motion, with `exp(log R) = −R`.
/// Near there the axis is ill-conditioned: it is the direction of a tiny `P`.
///
/// Series in `t² = s²/c0²` (for `c0 > 0`, or `s = 0`): `h0 = (1/c0) Σ (-1)^k t^{2k}/(2k+1)` for
/// `t² < 10⁻⁶`, and `∂h/∂u = (1/c0³) Σ (-1)^k (k+1)/(2k+3) t^{2k}` for `t² < 1/25`, where the
/// direct form cancels (`θ − c0 s/n ≈ ⅔ t³`).
#[inline(always)]
pub fn log_coeffs_rotation<T: Real>(c: (T, T), u: (T, T)) -> [T; 2] {
    let s2 = (-u.0).max(T::zero());
    let s = s2.sqrt();
    let theta = s.atan2(c.0);
    let n = c.0 * c.0 + s2;
    let k = |v: f64| T::from_f64(v);
    let (zero, one) = (T::zero(), T::one());
    let ic = c.0.recip();
    let t2 = s2 * ic * ic;
    // The series applies for c0 > 0, and at s = 0 whatever the sign of c0 (the log of −R).
    let key = T::select_lt(zero, c.0, t2, T::select_lt(zero, s2, one, zero));
    let h_series = ic * (one + t2 * (k(-1.0 / 3.0) + t2 * (k(1.0 / 5.0) + t2 * k(-1.0 / 7.0))));
    let h0 = T::select_lt(key, k(1e-6), h_series, theta / s);
    // Horner in t²: (-1)^j (j+1)/(2j+3), j = 0..=11 (the next term is below 10⁻¹⁶ at t² = 1/25).
    let mut g_series = zero;
    for j in (0..12u8).rev() {
        let (j, sign) = (f64::from(j), if j % 2 == 0 { 1.0 } else { -1.0 });
        let c = sign * (j + 1.0) / (2.0 * j + 3.0);
        g_series = g_series * t2 + k(c);
    }
    let g = T::select_lt(
        key,
        k(1.0 / 25.0),
        g_series * ic * ic * ic,
        (theta - c.0 * s / n) / (s2 * s * k(2.0)),
    );
    [h0, -c.1 / n + u.1 * g]
}

/// Fast path of [`rsqrt`] for `I² = 0`: `(a + b I)^(-1/2) = a^(-1/2) - (b/2) a^(-3/2) I`.
#[inline(always)]
pub fn rsqrt_nil<T: Real>(a: T, b: T) -> [T; 2] {
    let r = a.sqrt().recip();
    [r, -(b * r * r * r) * T::from_f64(0.5)]
}

/// The weights `[w1, w2, w3]` with `log R = w1 G1 + w2 G2 + w3 G3` for a 6D even versor, with
/// `G1 = ⟨R⟩₂`, `G2 = r0 ⟨R₄ R₂⟩₂`, `G3 = r0 ⟨R₆ R₄⟩₂` and `r0 = ⟨R⟩₀`: the interpolant of
/// [`log_coeffs_6d`] folded with `Q1 = G3 − G2 + p3 G1`, `Q2 = G2 + p1 Q1 − 2 p3 G1` and the
/// division by `r0` (docs/log6d.md).
#[inline]
pub fn log_weights_6d<T: Real>(p1: T, p2: T, p3: T, r0: T) -> [T; 3] {
    let [a0, a1, a2] = log_coeffs_6d(p1, p2, p3);
    let rinv = r0.recip();
    // a2 Q2 + a1 Q1 + a0 G1 = a2 G2 + b Q1 + (a0 - 2 p3 a2) G1, b = a2 p1 + a1.
    let b = a2 * p1 + a1;
    let w1 = (a0 - (p3 + p3) * a2 + b * p3) * rinv;
    let w2 = (a2 - b) * rinv;
    let w3 = b * rinv;
    [w1, w2, w3]
}

// ---------------------------------------------------------------------------------------------
// The logarithm of a 6D even versor in closed form (docs/log6d.md).
//
// A unit versor R = exp(B) with B = b1 + b2 + b3 (commuting simple bivectors) has
// u_j = cosh²(μ_j) (μ_j² = b_j²) as the roots of t³ − p1 t² + p2 t − p3, with p1, p2, p3 from the
// scalar parts of its grade parts' squares, and B = r0⁻¹ (α2 Q2 + α1 Q1 + α0 ⟨R⟩₂) for the
// quadratic α2 u² + α1 u + α0 that interpolates φ(u) = √u · asinh(√(u−1))/√(u−1) at the roots.
// The interpolant is computed without the individual roots where they are close: φ's Taylor
// series at their mean, reduced modulo the cubic.

/// Terms of the Taylor series used for the interpolants: the regimes keep the ratio of the
/// nodes' spread to the distance to φ's singularity (u = 0) below 1/4, so 26 terms reach
/// `4⁻²⁶ ≈ 2·10⁻¹⁶`.
const LOG6_TERMS: usize = 26;

type Series<T> = [T; LOG6_TERMS];

fn series_mul<T: Real>(a: &Series<T>, b: &Series<T>) -> Series<T> {
    let mut out = [T::zero(); LOG6_TERMS];
    for i in 0..LOG6_TERMS {
        for j in 0..LOG6_TERMS - i {
            out[i + j] = out[i + j] + a[i] * b[j];
        }
    }
    out
}

fn series_div<T: Real>(a: &Series<T>, b: &Series<T>) -> Series<T> {
    let mut q = [T::zero(); LOG6_TERMS];
    let inv = b[0].recip();
    for k in 0..LOG6_TERMS {
        let mut s = a[k];
        for j in 1..=k {
            s = s - b[j] * q[k - j];
        }
        q[k] = s * inv;
    }
    q
}

/// `(c + σ ρ s)^(1/2)` or `(c + σ ρ s)^(−1/2)` as a series in `s` (`σ = ±1`, `c > 0`, the
/// scale `ρ` at most about `c`).
fn series_sqrt<T: Real>(c: T, sigma: f64, inverse: bool, rho: T) -> Series<T> {
    let e = if inverse { -0.5 } else { 0.5 };
    let root = c.sqrt();
    let lead = if inverse { root.recip() } else { root };
    let step = T::from_f64(sigma) * rho / c;
    let mut out = [T::zero(); LOG6_TERMS];
    let (mut binom, mut power) = (T::one(), T::one());
    for (k, o) in out.iter_mut().enumerate() {
        *o = lead * binom * power;
        binom = binom * T::from_f64((e - k as f64) / (k as f64 + 1.0));
        power = power * step;
    }
    out
}

/// The Taylor series of `φ(u) = √u · F(u − 1)` at `u = m`, `F(x) = asinh(√x)/√x`, in the scaled
/// variable `s = (u − m)/ρ`. With `ρ` the distance to φ's singularity (`m` itself), its
/// coefficients stay bounded: in `u − m` they grow like `m⁻ᵏ`, which overflows `f32` for a
/// centre near a half turn.
fn phi_series<T: Real>(m: T, rho: T) -> Series<T> {
    let one = T::one();
    let x0 = m - one;
    let zero = [T::zero(); LOG6_TERMS];
    // |x0| < 1/4: F's Maclaurin series (90 terms; its radius is 1) shifted to x0 by repeated
    // Horner, the first LOG6_TERMS coefficients (skipped when no lane needs it). Nearer the
    // radius the truncation would cost the higher coefficients their accuracy.
    let quarter = T::from_f64(0.25);
    let near = if T::all_lt(quarter, x0.abs()) {
        zero
    } else {
        let mut c = [T::zero(); 90];
        let mut b = 1.0f64;
        for (n, f) in c.iter_mut().enumerate() {
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            *f = T::from_f64(sign * b / (2.0 * n as f64 + 1.0));
            b *= (2.0 * n as f64 + 1.0) / (2.0 * n as f64 + 2.0);
        }
        let x = T::select_lt(x0.abs(), quarter, x0, T::zero());
        let mut out = [T::zero(); LOG6_TERMS];
        let mut power = T::one();
        for (i, o) in out.iter_mut().enumerate() {
            for j in (i..c.len() - 1).rev() {
                c[j] = c[j] + x * c[j + 1];
            }
            *o = c[i] * power;
            power = power * rho;
        }
        out
    };
    // x0 > 0: v = √(x0 + t), asinh(v)' = v' / √(m + t), F = asinh(v) / v.
    let skip_above = T::all_lt(x0, quarter);
    let above = if skip_above {
        zero
    } else {
        let x = x0.max(T::from_f64(0.25));
        let mm = x + one;
        let v = series_sqrt(x, 1.0, false, rho);
        let r = series_sqrt(mm, 1.0, true, rho);
        let mut dv = [T::zero(); LOG6_TERMS];
        for k in 1..LOG6_TERMS {
            dv[k - 1] = v[k] * T::from_i64(k as i64);
        }
        let d = series_mul(&dv, &r);
        let mut a = [T::zero(); LOG6_TERMS];
        a[0] = (v[0] + (v[0] * v[0] + one).sqrt()).ln();
        for k in 1..LOG6_TERMS {
            a[k] = d[k - 1] * T::from_i64(k as i64).recip();
        }
        series_div(&a, &v)
    };
    // x0 < 0: w = √(−x0 − t), asin(w)' = w' / √(m + t), F = asin(w) / w.
    let skip_below = T::all_lt(-quarter, x0);
    let below = if skip_below {
        zero
    } else {
        let x = (-x0).max(T::from_f64(0.25));
        let mm = (one - x).max(T::from_f64(1e-300));
        let w = series_sqrt(x, -1.0, false, rho);
        let r = series_sqrt(mm, 1.0, true, rho);
        let mut dw = [T::zero(); LOG6_TERMS];
        for k in 1..LOG6_TERMS {
            dw[k - 1] = w[k] * T::from_i64(k as i64);
        }
        let d = series_mul(&dw, &r);
        let mut a = [T::zero(); LOG6_TERMS];
        a[0] = w[0].atan2(mm.sqrt());
        for k in 1..LOG6_TERMS {
            a[k] = d[k - 1] * T::from_i64(k as i64).recip();
        }
        series_div(&a, &w)
    };
    let f: Series<T> = core::array::from_fn(|k| {
        let far = T::select_lt(x0, T::zero(), below[k], above[k]);
        T::select_lt(x0.abs(), quarter, near[k], far)
    });
    series_mul(&series_sqrt(m, 1.0, false, rho), &f)
}

/// `φ(u)` at a real `u > 0`.
fn phi_real<T: Real>(u: T) -> T {
    let one = T::one();
    let x = u - one;
    let ax = x.abs();
    let r = ax.max(T::from_f64(1e-300)).sqrt();
    let above = (r + (ax + one).sqrt()).ln() / r;
    let below = r.atan2((one - ax).max(T::zero()).sqrt()) / r;
    // |x| < 1/20: 13 terms of the series (20^-13 < 1e-16).
    let mut series = T::zero();
    let mut c = 1.0f64;
    let mut coef = [0.0f64; 13];
    for (n, f) in coef.iter_mut().enumerate() {
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        *f = sign * c / (2.0 * n as f64 + 1.0);
        c *= (2.0 * n as f64 + 1.0) / (2.0 * n as f64 + 2.0);
    }
    for f in coef.iter().rev() {
        series = series * x + T::from_f64(*f);
    }
    let far = T::select_lt(x, T::zero(), below, above);
    u.max(T::zero()).sqrt() * T::select_lt(ax, T::from_f64(0.05), series, far)
}

/// `φ(u)` at a complex `u` (a conjugate pair of roots far apart).
fn phi_complex<T: Real>(u: Cx<T>) -> Cx<T> {
    let one = Cx::real(T::one());
    let s = (u - one).sqrt();
    let asinh = (s + (s * s + one).sqrt()).ln();
    u.sqrt() * asinh / s
}

/// The interpolant `[α0, α1, α2]` of a series at centre `c` in `s = t/ρ`, reduced modulo
/// `t³ + e2 t − e3` (three nodes with mean `c`).
fn reduce3<T: Real>(ph: &Series<T>, c: T, rho: T, e2: T, e3: T) -> [T; 3] {
    let (e2, e3) = (e2 / (rho * rho), e3 / (rho * rho * rho));
    let (mut a, mut b, mut d) = (T::zero(), T::zero(), T::one());
    let (mut sa, mut sb, mut sd) = (T::zero(), T::zero(), T::zero());
    for p in ph {
        sa = sa + *p * a;
        sb = sb + *p * b;
        sd = sd + *p * d;
        // t (a t² + b t + d) = a t³ + b t² + d t, with t³ = −e2 t + e3.
        let (na, nb, nd) = (b, d - a * e2, a * e3);
        a = na;
        b = nb;
        d = nd;
    }
    // a s² + b s + d with s = (u − c)/ρ.
    let (sa, sb) = (sa / (rho * rho), sb / rho);
    [sa * c * c - sb * c + sd, sb - (sa + sa) * c, sa]
}

/// The line `[l0, l1]` of a series at centre `c` in `s = t/ρ`, reduced modulo `t² − d2` (two
/// nodes).
fn reduce2<T: Real>(ph: &Series<T>, c: T, rho: T, d2: T) -> [T; 2] {
    let d2 = d2 / (rho * rho);
    let (mut b, mut d) = (T::zero(), T::one());
    let (mut sb, mut sd) = (T::zero(), T::zero());
    for p in ph {
        sb = sb + *p * b;
        sd = sd + *p * d;
        // t (b t + d) = b t² + d t = b d2 + d t.
        let (nb, nd) = (d, b * d2);
        b = nb;
        d = nd;
    }
    let sb = sb / rho;
    [sd - sb * c, sb]
}

/// `sign(x) |x|^(1/3)`.
fn cbrt<T: Real>(x: T) -> T {
    let a = x.abs();
    let r = (a.max(T::from_f64(1e-300)).ln() * T::from_f64(1.0 / 3.0)).exp();
    let r = T::select_lt(a, T::from_f64(1e-300), T::zero(), r);
    T::select_lt(x, T::zero(), -r, r)
}

/// The coefficients `[α0, α1, α2]` of the quadratic interpolating
/// `φ(u) = √u · asinh(√(u−1))/√(u−1)` at the roots of `t³ − p1 t² + p2 t − p3`, the three
/// invariants `cosh²(μ_j)` of a 6D even versor: `log R = r0⁻¹ (α2 Q2 + α1 Q1 + α0 ⟨R⟩₂)`
/// (docs/log6d.md). Branch free: close roots through φ's Taylor series at their mean (no
/// individual roots, so coinciding ones are exact), spread roots through an isolated real root
/// and the remaining pair.
///
/// ```
/// // A rotation in one plane by half-angle θ: roots cos²θ, 1, 1.
/// let th = 0.7f64;
/// let c2 = th.cos() * th.cos();
/// let [a0, a1, a2] = gax_core::study::log_coeffs_6d(c2 + 2.0, 2.0 * c2 + 1.0, c2);
/// // At u = cos²θ the interpolant is φ = cos θ · θ / sin θ.
/// let at = a2 * c2 * c2 + a1 * c2 + a0;
/// assert!((at - th.cos() * th / th.sin()).abs() < 1e-14);
/// ```
#[inline]
pub fn log_coeffs_6d<T: Real>(p1: T, p2: T, p3: T) -> [T; 3] {
    let c = cubic6(p1, p2, p3);
    let (m, e2, e3, bound, limit) = (c.m, c.e2, c.e3, c.bound, c.limit);

    // Close roots: the series at their mean (skipped when no lane has close roots).
    let jet = if T::all_lt(limit, bound) {
        [T::zero(); 3]
    } else {
        let rho = m.max(T::from_f64(1e-30));
        reduce3(&phi_series(m, rho), m, rho, e2, e3)
    };
    if T::all_lt(bound, limit) {
        return jet;
    }

    // Spread roots: an isolated real root r, then the pair.
    let Split6 {
        r,
        sum,
        prod,
        mid,
        d2,
        d,
    } = split6(&c, p1, p2, p3);
    let spread = add_isolated(phi_pair(mid, d2, d), phi_real(r), r, sum, prod);

    core::array::from_fn(|k| T::select_lt(bound, limit, jet[k], spread[k]))
}

/// The line `[l0, l1]` through `φ` at a pair of roots with midpoint `mid` and half-difference
/// `d` (`d2 = d²`, negative for a conjugate pair): its series at the midpoint when close, else
/// through the two values (real, or a conjugate pair). Only right of φ's branch point u = 0: a
/// conjugate pair with a negative midpoint straddles the branch cut, where the series does not
/// reach.
fn phi_pair<T: Real>(mid: T, d2: T, d: T) -> [T; 2] {
    let quarter_mid = mid.max(T::zero()) * T::from_f64(0.25);
    let close = if T::all_lt(quarter_mid, d) {
        [T::zero(); 2]
    } else {
        let rho = mid.max(T::from_f64(1e-30));
        reduce2(&phi_series(mid, rho), mid, rho, d2)
    };
    let real_far = {
        let (a, b) = (mid + d, mid - d);
        let (fa, fb) = (phi_real(a), phi_real(b));
        let slope = (fa - fb) / (a - b).max(T::from_f64(1e-300));
        [fa - slope * a, slope]
    };
    let conj_far = {
        let f = phi_complex(Cx { re: mid, im: d });
        let slope = f.im / d.max(T::from_f64(1e-300));
        [f.re - slope * mid, slope]
    };
    core::array::from_fn(|k| {
        let far = T::select_lt(d2, T::zero(), conj_far[k], real_far[k]);
        T::select_lt(d, quarter_mid, close[k], far)
    })
}

/// The cubic `t³ + e2 t − e3` of the invariants in `t = u − m` (`m = p1/3`, their mean), and its
/// regime: close roots where `bound < limit` (their spread, bounded from `e2` and `e3`, against
/// a quarter of their distance to `u = 0`).
struct Cubic6<T> {
    m: T,
    e2: T,
    e3: T,
    bound: T,
    limit: T,
}

fn cubic6<T: Real>(p1: T, p2: T, p3: T) -> Cubic6<T> {
    let three = T::from_i64(3);
    let m = p1 * three.recip();
    let e2 = p2 - (m + m) * p1 + three * m * m;
    let e3 = p3 - m * p2 + m * m * p1 - m * m * m;
    let bound = (e2.abs().sqrt()).max(cbrt(e3.abs())) * T::from_i64(2);
    let limit = m * T::from_f64(0.25);
    Cubic6 {
        m,
        e2,
        e3,
        bound,
        limit,
    }
}

/// Spread roots: the most isolated real root `r`, and the remaining pair by its sum and product,
/// midpoint `mid` and half-difference `d` (`d2 = d²`, negative for a conjugate pair).
struct Split6<T> {
    r: T,
    sum: T,
    prod: T,
    mid: T,
    d2: T,
    d: T,
}

fn split6<T: Real>(c: &Cubic6<T>, p1: T, p2: T, p3: T) -> Split6<T> {
    let three = T::from_i64(3);
    let (m, pp, qq) = (c.m, c.e2, -c.e3);
    let disc = qq * qq * T::from_f64(0.25) + pp * pp * pp * T::from_f64(1.0 / 27.0);
    // One real root (Cardano) where disc >= 0.
    let sq = disc.max(T::zero()).sqrt();
    let cardano = cbrt(-qq * T::from_f64(0.5) + sq) + cbrt(-qq * T::from_f64(0.5) - sq);
    // Three real roots (trigonometric) where disc < 0: the most isolated one.
    let rad = (-pp * three.recip()).max(T::zero()).sqrt();
    let arg = qq * T::from_f64(1.5) / pp.min(T::from_f64(-1e-300))
        * (-three / pp.min(T::from_f64(-1e-300))).sqrt();
    let arg = arg.max(-T::one()).min(T::one());
    let ang = (T::one() - arg * arg).max(T::zero()).sqrt().atan2(arg) * three.recip();
    let third = T::from_f64(2.0 * core::f64::consts::PI / 3.0);
    let s: [T; 3] = [0, 1, 2].map(|k| (rad + rad) * (ang - third * T::from_i64(k)).cos());
    let gap = |k: usize| {
        let (i, j) = ((k + 1) % 3, (k + 2) % 3);
        (s[k] - s[i]).abs().min((s[k] - s[j]).abs())
    };
    let (g0, g1, g2) = (gap(0), gap(1), gap(2));
    let best01 = T::select_lt(g0, g1, s[1], s[0]);
    let gbest01 = g0.max(g1);
    let trig = T::select_lt(gbest01, g2, s[2], best01);
    let mut r = T::select_lt(disc, T::zero(), trig, cardano) + m;
    // A Newton step on the original cubic, kept only where it lowers the residual: at a
    // multiple root the derivative vanishes, and the step would throw the root off.
    let f = |r: T| ((r - p1) * r + p2) * r - p3;
    let df = (three * r - (p1 + p1)) * r + p2;
    let df_safe = T::select_lt(df.abs(), T::from_f64(1e-300), T::one(), df);
    let next = r - f(r) / df_safe;
    r = T::select_lt(f(next).abs(), f(r).abs(), next, r);
    // The pair's sum and product, each from the side with the smaller error: forward from p1
    // (for a small r), or backward from p3 (for a large one: two planes near a half turn and
    // a third, whose tiny product p2 − r·sum would lose to p2's rounding).
    let ar = r.abs();
    let sum_f = p1 - r;
    let prod_f = p2 - r * sum_f;
    let e1f = p1.abs() + ar;
    let e2f = p2.abs() + ar * (sum_f.abs() + e1f);
    let tiny = T::from_f64(1e-30);
    let rinv = T::select_lt(ar, tiny, T::one(), r).recip();
    let prod_b = p3 * rinv;
    let sum_b = (p2 - prod_b) * rinv;
    let big = T::from_f64(1e30);
    let e2b = T::select_lt(ar, tiny, big, prod_b.abs());
    let e1b = T::select_lt(ar, tiny, big, (p2.abs() + prod_b.abs() + e2b) * rinv.abs());
    // Backward only where r is larger than the pair (not, say, a zero root rounded to 10⁻¹⁵,
    // whose quotient p3/r = 0 would claim no error).
    let larger = T::select_lt(
        sum_f.abs().max(prod_f.abs().sqrt()),
        ar,
        T::one(),
        T::zero(),
    );
    let half = T::from_f64(0.5);
    let sum = T::select_lt(e1b, e1f, T::select_lt(half, larger, sum_b, sum_f), sum_f);
    let prod = T::select_lt(e2b, e2f, T::select_lt(half, larger, prod_b, prod_f), prod_f);
    let mid = sum * T::from_f64(0.5);
    let d2 = mid * mid - prod;
    Split6 {
        r,
        sum,
        prod,
        mid,
        d2,
        d: d2.abs().sqrt(),
    }
}

/// The quadratic `[α0, α1, α2]` through the pair's line `[l0, l1]` and the value `fr` at the
/// isolated root: `L(u) + (u² − sum u + prod)(fr − L(r))/(r² − sum r + prod)`.
fn add_isolated<T: Real>(pair: [T; 2], fr: T, r: T, sum: T, prod: T) -> [T; 3] {
    let at_r = pair[1] * r + pair[0];
    let denom = (r - sum) * r + prod;
    let kk = (fr - at_r) / denom;
    [pair[0] + prod * kk, pair[1] - sum * kk, kk]
}

/// The Taylor series of `g(u) = √(u/(1−u))` at `u = m`, `0 < m < 1`, in `s = (u − m)/ρ`
/// with `ρ = min(m, 1 − m)` (its distance to the singularities): the series and `ρ`.
fn g_series<T: Real>(m: T) -> (Series<T>, T) {
    let tiny = T::from_f64(1e-30);
    let rho = m.min(T::one() - m).max(tiny);
    let series = series_mul(
        &series_sqrt(m.max(tiny), 1.0, false, rho),
        &series_sqrt((T::one() - m).max(tiny), -1.0, true, rho),
    );
    (series, rho)
}

/// Which planes of a 6D even versor to turn by a quarter turn before [`log_weights_6d`], and
/// how: `[α0, α1, α2, n]` with `Z = α2 Q2 + α1 Q1 + α0 ⟨R⟩₂` the sum of the `n` turned planes'
/// unit bivectors `b̂` (docs/log6d.md). Then `R' = R E` with `E = ∏(−b̂)`, a polynomial in `Z`
/// ([`turn_polynomial`]), and `log R = log R' + (π/2) Z`.
///
/// Near a half turn in some plane (`⟨R⟩₀ → 0`) the closed form loses `ε/⟨R⟩₀`, and it is wrong
/// where `⟨R⟩₀ < 0`. Turning a set `T` of rotation planes (invariants `uⱼ < 1`) gives
/// `⟨R'⟩₀ = ∏_T sⱼ ∏_rest cⱼ`, of magnitude `∏_T √(1−uⱼ) ∏_rest √uⱼ`, and positive when `T` has
/// one or three planes, or two and `⟨R⟩₀ > 0` (each `b̂` oriented by its weight in `⟨R⟩₂`).
/// This picks, among the sets with a positive `⟨R'⟩₀`, the one with the smallest error
/// estimate `1/⟨R'⟩₀ + Σ_T 1/|wⱼ|` (the closed form on `R'` loses `ε/⟨R'⟩₀`, and a plane's
/// direction comes from its weight `wⱼ` in `⟨R⟩₂`), including none (`n = 0`, `Z = 0`): all
/// three through the series at their mean, or unions of the isolated root, the pair (through
/// its series when close) and each of the pair (when real and apart). Sets that separate the
/// isolated root from the pair are preferred where its gap is at least 1/64 (relative). `Z` interpolates `h(u) = [u ∈ T]/|wⱼ|` at the roots, `|wⱼ| = √((1−uⱼ) ∏_{i≠j} uᵢ)`
/// being plane `j`'s weight in `⟨R⟩₂`; on a cluster through the series of `√(u/(1−u))/|⟨R⟩₀|`.
#[inline]
#[allow(clippy::too_many_lines)]
pub fn log_turn_6d<T: Real>(p1: T, p2: T, p3: T, r0: T) -> [T; 4] {
    let (zero, one, half) = (T::zero(), T::one(), T::from_f64(0.5));
    let quarter = T::from_f64(0.25);
    let tiny = T::from_f64(1e-30);
    let invalid = T::from_f64(1e30);
    let ar0 = r0.abs().max(tiny);
    let positive = |x: T| T::select_lt(zero, x, one, zero);
    let inv = |x: T| x.max(tiny).recip();
    // A candidate's cost `1/⟨R'⟩₀ + extra` where `ok` is 1 and `⟨R'⟩₀ > 0`, else `invalid`.
    let gated =
        |ok: T, v: T, extra: T| T::select_lt(zero, ok * positive(v), inv(v) + extra, invalid);
    let c = cubic6(p1, p2, p3);
    let m = c.m;
    let pos_r0 = positive(r0);
    // A cluster's `1/|w| = √(u/(1−u))/|r₀|` at its centre.
    let cluster = |u: T| (u.max(zero) / (one - u).max(tiny)).sqrt() / ar0;

    // All three as one cluster, through the series of √(u/(1−u)) at their mean.
    let prod1m = one - p1 + p2 - p3;
    let series_ok = positive(m.min(one - m) * quarter - c.bound) * positive(prod1m);
    let all_series = gated(
        series_ok,
        prod1m.max(zero).sqrt(),
        T::from_i64(3) * cluster(m),
    );

    // The isolated root r and the pair (a, b) = mid ± d.
    let sp = split6(&c, p1, p2, p3);
    let (r, sum, prod, mid, d2, d) = (sp.r, sp.sum, sp.prod, sp.mid, sp.d2, sp.d);
    let (a, b) = (mid + d, (mid - d).max(zero));
    let real = positive(d2 + T::epsilon() * T::from_i64(64) * (one + mid * mid));
    let close = positive(mid * quarter - d);
    let close_g = positive((one - mid) * quarter - d) * close;
    let gap_r = T::select_lt(
        half,
        real,
        (r - a).abs().min((r - b).abs()),
        ((r - mid) * (r - mid) + d * d).sqrt(),
    );
    let apart = T::from_f64(1.0 / 64.0);
    let sep = positive(gap_r - apart * (one + r.abs()));
    // The pair's members apart from each other (for turning one of them).
    let sep_pair = positive(d + d - apart * (one + mid.abs()));
    let ok_r = positive(one - r);
    let ok_pair = real * positive(one - a) * (one - close + close_g);
    let ok_a = real * positive(d) * positive(one - a);
    let ok_b = real * positive(d) * positive(one - b);
    let (in_r, out_r) = ((one - r).max(zero).sqrt(), r.max(zero).sqrt());
    let (in_a, out_a) = ((one - a).max(zero).sqrt(), a.max(zero).sqrt());
    let (in_b, out_b) = ((one - b).max(zero).sqrt(), b.sqrt());
    let in_pair = (one - sum + prod).max(zero).sqrt();
    let out_pair = prod.max(zero).sqrt();
    // 1/|w| of each root, and of the pair (as a cluster when close).
    let iw_r = inv(((one - r) * prod).max(zero).sqrt());
    let iw_a = inv(((one - a) * b * r).max(zero).sqrt());
    let iw_b = inv(((one - b) * a * r).max(zero).sqrt());
    let iw_pair = T::select_lt(half, close, cluster(mid) + cluster(mid), iw_a + iw_b);
    // (cost, t_r, t_a, t_b, through the series of all three), each needing r apart from the
    // pair (all but none and the series), and the pair's members apart (turning one of them).
    let unseparated = [
        (gated(pos_r0, r0, zero), zero, zero, zero, zero),
        (all_series, one, one, one, one),
    ];
    let separating = [
        (
            gated(ok_r, in_r * out_pair, iw_r),
            one,
            zero,
            zero,
            zero,
            one,
        ),
        (
            gated(ok_pair * pos_r0, out_r * in_pair, iw_pair),
            zero,
            one,
            one,
            zero,
            one,
        ),
        (
            gated(ok_r * ok_pair, in_r * in_pair, iw_r + iw_pair),
            one,
            one,
            one,
            zero,
            one,
        ),
        (
            gated(ok_a, out_r * in_a * out_b, iw_a),
            zero,
            one,
            zero,
            zero,
            sep_pair,
        ),
        (
            gated(ok_b, out_r * out_a * in_b, iw_b),
            zero,
            zero,
            one,
            zero,
            sep_pair,
        ),
        (
            gated(ok_r * ok_a * pos_r0, in_r * in_a * out_b, iw_r + iw_a),
            one,
            one,
            zero,
            zero,
            sep_pair,
        ),
        (
            gated(ok_r * ok_b * pos_r0, in_r * out_a * in_b, iw_r + iw_b),
            one,
            zero,
            one,
            zero,
            sep_pair,
        ),
    ];
    let pick = |best: (T, T, T, T, T), cand: (T, T, T, T, T)| {
        (
            best.0.min(cand.0),
            T::select_lt(cand.0, best.0, cand.1, best.1),
            T::select_lt(cand.0, best.0, cand.2, best.2),
            T::select_lt(cand.0, best.0, cand.3, best.3),
            T::select_lt(cand.0, best.0, cand.4, best.4),
        )
    };
    let base = pick(unseparated[0], unseparated[1]);
    // Preferred: separating sets only where r is apart; any, if none of those is valid.
    let strict = separating.iter().fold(base, |acc, cand| {
        let cost = T::select_lt(half, sep * cand.5, cand.0, invalid);
        pick(acc, (cost, cand.1, cand.2, cand.3, cand.4))
    });
    let loose = separating.iter().fold(base, |acc, cand| {
        pick(acc, (cand.0, cand.1, cand.2, cand.3, cand.4))
    });
    let use_strict = positive(invalid * half - strict.0);
    let choose = |s: T, l: T| T::select_lt(half, use_strict, s, l);
    let (t_r, t_a, t_b, t_all) = (
        choose(strict.1, loose.1),
        choose(strict.2, loose.2),
        choose(strict.3, loose.3),
        choose(strict.4, loose.4),
    );

    // Otherwise h at the roots: 1/|w| for the turned ones.
    let h = [
        t_r / ((one - r) * prod).max(tiny).sqrt(),
        t_a / ((one - a) * b * r).max(tiny).sqrt(),
        t_b / ((one - b) * a * r).max(tiny).sqrt(),
    ];
    let alpha = turn_alpha3(&c, &sp, close, [t_r, t_a, t_b, t_all], h, ar0);
    [alpha[0], alpha[1], alpha[2], t_r + t_a + t_b]
}

/// The quadratic interpolating `h` at three roots split as in [`split6`], for the turned
/// planes' flags `[t_r, t_a, t_b, t_all]` (1 or 0; `t_all`: all three as one cluster) and `h`
/// at the roots `[h_r, h_a, h_b]` (`1/|w|` where turned, else 0): all three through the series
/// of `√(u/(1−u))/|r₀|` at their mean, or the isolated root added to the pair's line (its series
/// at the midpoint for a turned close pair, else the chord).
fn turn_alpha3<T: Real>(
    c: &Cubic6<T>,
    sp: &Split6<T>,
    close: T,
    t: [T; 4],
    h: [T; 3],
    ar0: T,
) -> [T; 3] {
    let (zero, one, half) = (T::zero(), T::one(), T::from_f64(0.5));
    let tiny = T::from_f64(1e-30);
    let [_, t_a, t_b, t_all] = t;
    let [h_r, h_a, h_b] = h;
    let (a, b) = (sp.mid + sp.d, (sp.mid - sp.d).max(zero));
    // All three through the series at the mean.
    let series_all = if T::all_lt(t_all, half) {
        [zero; 3]
    } else {
        let (g, rho) = g_series(c.m);
        reduce3(&g, c.m, rho, c.e2, c.e3).map(|x| x / ar0)
    };
    // The pair's line: a turned close pair through the series, else the chord.
    let use_series = close * t_a * t_b * (one - t_all);
    let series = if T::all_lt(use_series, half) {
        [zero; 2]
    } else {
        let (g, rho) = g_series(sp.mid);
        reduce2(&g, sp.mid, rho, sp.d2).map(|x| x / ar0)
    };
    let slope = (h_a - h_b) / (a - b).max(tiny);
    let chord = [h_a - slope * a, slope];
    let pair = [
        T::select_lt(half, use_series, series[0], chord[0]),
        T::select_lt(half, use_series, series[1], chord[1]),
    ];
    let spread = add_isolated(pair, h_r, sp.r, sp.sum, sp.prod);
    core::array::from_fn(|k| T::select_lt(half, t_all, series_all[k], spread[k]))
}

/// `[e0, e1, e2, e3]` with `∏(−b̂) = e0 + e1 Z + e2 Z² + e3 Z³` for `Z` the sum of `n` commuting
/// orthogonal unit rotation bivectors `b̂` (`b̂² = −1`): `1`, `−Z`, `1 + Z²/2`, `−(Z³ + 7Z)/6`.
#[inline]
pub fn turn_polynomial<T: Real>(n: T) -> [T; 4] {
    let pick = |v: [f64; 4]| {
        T::select_lt(
            n,
            T::from_f64(0.5),
            T::from_f64(v[0]),
            T::select_lt(
                n,
                T::from_f64(1.5),
                T::from_f64(v[1]),
                T::select_lt(n, T::from_f64(2.5), T::from_f64(v[2]), T::from_f64(v[3])),
            ),
        )
    };
    [
        pick([1.0, 0.0, 1.0, 0.0]),
        pick([0.0, -1.0, 0.0, -7.0 / 6.0]),
        pick([0.0, 0.0, 0.5, 0.0]),
        pick([0.0, 0.0, 0.0, -1.0 / 6.0]),
    ]
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
