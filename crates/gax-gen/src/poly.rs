//! Exact multivariate polynomials with rational coefficients.
//!
//! These are the coefficient-level expressions that tracing produces and the simplifier
//! rewrites. A polynomial is kept fully expanded, as a map from monomials to rationals, so
//! two expressions are algebraically equal exactly when their polynomials are equal.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// An exact rational number with `i128` numerator and positive denominator, in lowest terms.
///
/// `Ord` is structural (numerator, then denominator): the generator sorts terms by it, and
/// the order of the generated sums depends on it. Compare values with [`Rational::lt`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Rational {
    num: i128,
    den: i128,
}

fn gcd(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

impl Rational {
    /// Zero.
    pub const ZERO: Rational = Rational { num: 0, den: 1 };
    /// One.
    pub const ONE: Rational = Rational { num: 1, den: 1 };

    /// `num / den` in lowest terms. Panics when `den == 0`.
    pub fn new(num: i128, den: i128) -> Rational {
        assert!(den != 0, "rational with zero denominator");
        assert!(num != i128::MIN && den != i128::MIN, "rational overflow");
        let g = gcd(num, den).max(1);
        let s = if den < 0 { -1 } else { 1 };
        Rational {
            num: s * num / g,
            den: s * den / g,
        }
    }

    /// An integer.
    pub fn int(n: i128) -> Rational {
        Rational { num: n, den: 1 }
    }

    /// Numerator.
    pub fn num(self) -> i128 {
        self.num
    }

    /// Denominator (positive).
    pub fn den(self) -> i128 {
        self.den
    }

    /// Whether this is zero.
    pub fn is_zero(self) -> bool {
        self.num == 0
    }

    /// Whether this is an integer.
    pub fn is_integer(self) -> bool {
        self.den == 1
    }

    /// Absolute value.
    pub fn abs(self) -> Rational {
        Rational {
            num: self.num.abs(),
            den: self.den,
        }
    }

    /// Reciprocal. Panics on zero.
    pub fn recip(self) -> Rational {
        Rational::new(self.den, self.num)
    }

    /// Whether `self < o` by value: `a/b < c/d` iff `a d < c b` (denominators are positive).
    /// When the cross products overflow, the nearest `f64`s decide.
    pub fn lt(self, o: Rational) -> bool {
        match (self.num.checked_mul(o.den), o.num.checked_mul(self.den)) {
            (Some(l), Some(r)) => l < r,
            _ => self.to_f64() < o.to_f64(),
        }
    }

    /// Nearest `f64`.
    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// Exact conversion from an `f64` that is a dyadic rational of moderate size, as produced
    /// by constants in traced code. Returns `None` for values that do not fit.
    pub fn from_f64(x: f64) -> Option<Rational> {
        if !x.is_finite() {
            return None;
        }
        let mut den: i128 = 1;
        let mut v = x;
        for _ in 0..64 {
            if v.fract() == 0.0 {
                if v.abs() > 1e30 {
                    return None;
                }
                return Some(Rational::new(v as i128, den));
            }
            v *= 2.0;
            den *= 2;
        }
        None
    }
}

fn ck(x: Option<i128>) -> i128 {
    x.expect("rational overflow: coefficients exceed i128")
}

impl Add for Rational {
    type Output = Rational;
    fn add(self, o: Rational) -> Rational {
        if self.den == o.den {
            return Rational::new(ck(self.num.checked_add(o.num)), self.den);
        }
        let n = ck(ck(self.num.checked_mul(o.den)).checked_add(ck(o.num.checked_mul(self.den))));
        Rational::new(n, ck(self.den.checked_mul(o.den)))
    }
}
impl Sub for Rational {
    type Output = Rational;
    fn sub(self, o: Rational) -> Rational {
        self + (-o)
    }
}
impl Mul for Rational {
    type Output = Rational;
    fn mul(self, o: Rational) -> Rational {
        // Cross-cancel first to keep intermediates small.
        let g1 = gcd(self.num, o.den).max(1);
        let g2 = gcd(o.num, self.den).max(1);
        Rational::new(
            ck((self.num / g1).checked_mul(o.num / g2)),
            ck((self.den / g2).checked_mul(o.den / g1)),
        )
    }
}
impl Neg for Rational {
    type Output = Rational;
    fn neg(self) -> Rational {
        Rational {
            num: -self.num,
            den: self.den,
        }
    }
}

impl fmt::Debug for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

/// A variable identifier.
pub type Var = u32;

/// A monomial: a sorted list of variables with repetition (`[a, a, b]` is `a² b`).
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Monomial(pub Vec<Var>);

impl Monomial {
    /// The constant monomial `1`.
    pub fn one() -> Monomial {
        Monomial(Vec::new())
    }

    /// A single variable.
    pub fn var(v: Var) -> Monomial {
        Monomial(vec![v])
    }

    /// Total degree.
    pub fn degree(&self) -> usize {
        self.0.len()
    }

    /// Product of two monomials.
    pub fn mul(&self, o: &Monomial) -> Monomial {
        let mut v = Vec::with_capacity(self.0.len() + o.0.len());
        let (mut i, mut j) = (0, 0);
        while i < self.0.len() && j < o.0.len() {
            if self.0[i] <= o.0[j] {
                v.push(self.0[i]);
                i += 1;
            } else {
                v.push(o.0[j]);
                j += 1;
            }
        }
        v.extend_from_slice(&self.0[i..]);
        v.extend_from_slice(&o.0[j..]);
        Monomial(v)
    }

    /// Exponent of a variable.
    pub fn power(&self, v: Var) -> usize {
        self.0.iter().filter(|&&x| x == v).count()
    }

    /// Divide by another monomial if it divides this one.
    pub fn div(&self, o: &Monomial) -> Option<Monomial> {
        let mut out = Vec::with_capacity(self.0.len());
        let mut j = 0;
        for &x in &self.0 {
            if j < o.0.len() && o.0[j] == x {
                j += 1;
            } else if j < o.0.len() && o.0[j] < x {
                return None;
            } else {
                out.push(x);
            }
        }
        (j == o.0.len()).then_some(Monomial(out))
    }

    /// Whether the monomial contains the variable.
    pub fn contains(&self, v: Var) -> bool {
        self.0.binary_search(&v).is_ok()
    }
}

impl fmt::Debug for Monomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("1");
        }
        let parts: Vec<String> = self.0.iter().map(|v| format!("x{v}")).collect();
        f.write_str(&parts.join("*"))
    }
}

/// An expanded polynomial: monomials with nonzero rational coefficients.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Poly(pub BTreeMap<Monomial, Rational>);

impl Poly {
    /// The zero polynomial.
    pub fn zero() -> Poly {
        Poly(BTreeMap::new())
    }

    /// A constant.
    pub fn constant(c: Rational) -> Poly {
        let mut p = Poly::zero();
        if !c.is_zero() {
            p.0.insert(Monomial::one(), c);
        }
        p
    }

    /// A single variable.
    pub fn var(v: Var) -> Poly {
        Poly(BTreeMap::from([(Monomial::var(v), Rational::ONE)]))
    }

    /// A single term `c * m`.
    pub fn term(m: Monomial, c: Rational) -> Poly {
        let mut p = Poly::zero();
        p.add_term(m, c);
        p
    }

    /// Whether this is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    /// The value if this polynomial is a constant.
    pub fn as_constant(&self) -> Option<Rational> {
        match self.0.len() {
            0 => Some(Rational::ZERO),
            1 => self.0.get(&Monomial::one()).copied(),
            _ => None,
        }
    }

    /// Add `c * m` in place.
    pub fn add_term(&mut self, m: Monomial, c: Rational) {
        if c.is_zero() {
            return;
        }
        match self.0.entry(m) {
            Entry::Vacant(v) => {
                v.insert(c);
            }
            Entry::Occupied(mut o) => {
                let sum = *o.get() + c;
                if sum.is_zero() {
                    o.remove();
                } else {
                    *o.get_mut() = sum;
                }
            }
        }
    }

    /// Multiply by a rational.
    pub fn scale(&self, c: Rational) -> Poly {
        if c.is_zero() {
            return Poly::zero();
        }
        Poly(self.0.iter().map(|(m, v)| (m.clone(), *v * c)).collect())
    }

    /// Total degree (0 for constants and zero).
    pub fn degree(&self) -> usize {
        self.0.keys().map(Monomial::degree).max().unwrap_or(0)
    }

    /// The variables occurring in the polynomial, sorted.
    pub fn vars(&self) -> Vec<Var> {
        let mut v: Vec<Var> = self.0.keys().flat_map(|m| m.0.iter().copied()).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Number of terms.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether there are no terms.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Evaluate with `f64` values for the variables.
    pub fn eval(&self, value: &impl Fn(Var) -> f64) -> f64 {
        self.0
            .iter()
            .map(|(m, c)| c.to_f64() * m.0.iter().map(|&v| value(v)).product::<f64>())
            .sum()
    }
}

impl Add for &Poly {
    type Output = Poly;
    fn add(self, o: &Poly) -> Poly {
        let mut out = self.clone();
        for (m, &c) in &o.0 {
            out.add_term(m.clone(), c);
        }
        out
    }
}

impl Sub for &Poly {
    type Output = Poly;
    fn sub(self, o: &Poly) -> Poly {
        let mut out = self.clone();
        for (m, &c) in &o.0 {
            out.add_term(m.clone(), -c);
        }
        out
    }
}

impl Mul for &Poly {
    type Output = Poly;
    fn mul(self, o: &Poly) -> Poly {
        let mut out = Poly::zero();
        for (ma, &ca) in &self.0 {
            for (mb, &cb) in &o.0 {
                out.add_term(ma.mul(mb), ca * cb);
            }
        }
        out
    }
}

impl Neg for &Poly {
    type Output = Poly;
    fn neg(self) -> Poly {
        self.scale(-Rational::ONE)
    }
}

impl fmt::Debug for Poly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("0");
        }
        let parts: Vec<String> = self.0.iter().map(|(m, c)| format!("{c}*{m:?}")).collect();
        f.write_str(&parts.join(" + "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_arithmetic() {
        let h = Rational::new(1, 2);
        assert_eq!(h + h, Rational::ONE);
        assert_eq!(Rational::new(2, -4), Rational::new(-1, 2));
        assert_eq!(Rational::from_f64(0.375), Some(Rational::new(3, 8)));
        assert_eq!(Rational::from_f64(0.1), None.or(Rational::from_f64(0.1)));
    }

    #[test]
    fn polynomial_identity() {
        let (a, b) = (Poly::var(0), Poly::var(1));
        let lhs = &(&a + &b) * &(&a - &b);
        let rhs = &(&a * &a) - &(&b * &b);
        assert_eq!(lhs, rhs);
        let zero = &lhs - &rhs;
        assert!(zero.is_zero());
    }

    #[test]
    fn monomial_division() {
        let m = Monomial(vec![0, 0, 1, 2]);
        assert_eq!(m.div(&Monomial(vec![0, 2])), Some(Monomial(vec![0, 1])));
        assert_eq!(m.div(&Monomial(vec![1, 1])), None);
    }
}
