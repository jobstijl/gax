//! Algebra descriptions and exact blade products.
//!
//! An algebra is a real vector space with a symmetric integer bilinear form (the metric),
//! which may be degenerate (PGA) or non-diagonal (the null basis `eo`, `ei` of CGA).
//! Basis blades are the wedge products of basis vectors in increasing index order and are
//! identified by a bit mask. All products of basis blades have integer coefficients in this
//! basis, whatever the metric, so every table the generator builds is exact.

use std::collections::BTreeMap;
use std::fmt;

/// A sparse multivector with exact integer coefficients, keyed by blade mask.
pub type Sparse = BTreeMap<u32, i64>;

/// Maximum supported dimension of the generating vector space.
pub const MAX_DIM: usize = 10;

/// Errors produced while parsing or validating an algebra description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlgebraError(pub String);

impl fmt::Display for AlgebraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AlgebraError {}

fn err<T>(msg: impl Into<String>) -> Result<T, AlgebraError> {
    Err(AlgebraError(msg.into()))
}

/// The generating vector space of a geometric algebra: named basis vectors and their metric.
///
/// Basis vector names have the form `e` followed by one character (`e0`, `e1`, `eo`, `ei`),
/// so a blade is written as `e` followed by the characters of its factors, in any order:
/// `e032` is `e0 ^ e3 ^ e2`.
#[derive(Clone, PartialEq, Eq)]
pub struct Algebra {
    names: Vec<char>,
    metric: Vec<Vec<i64>>,
    /// Cached product table: `table[a * 2^n + b]` is the product of blades `a` and `b`.
    table: Vec<Vec<(u32, i64)>>,
}

impl fmt::Debug for Algebra {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Algebra")
            .field("names", &self.names)
            .field("metric", &self.metric)
            .finish_non_exhaustive()
    }
}

impl Algebra {
    /// An algebra with a diagonal metric. `basis` lists the one-character suffixes of the
    /// basis vector names, and `squares` the square of each (`1`, `-1` or `0`).
    ///
    /// ```
    /// # use gax_gen::algebra::Algebra;
    /// let pga2d = Algebra::diagonal("012", &[0, 1, 1]).unwrap();
    /// assert_eq!(pga2d.dim(), 3);
    /// ```
    pub fn diagonal(basis: &str, squares: &[i64]) -> Result<Self, AlgebraError> {
        let n = basis.chars().count();
        if n != squares.len() {
            return err("basis and squares must have the same length");
        }
        let mut metric = vec![vec![0; n]; n];
        for (i, &s) in squares.iter().enumerate() {
            metric[i][i] = s;
        }
        Self::new(basis, metric)
    }

    /// An algebra with a general symmetric integer metric (`metric[i][j]` is `e_i . e_j`).
    pub fn new(basis: &str, metric: Vec<Vec<i64>>) -> Result<Self, AlgebraError> {
        let names: Vec<char> = basis.chars().collect();
        let n = names.len();
        if n > MAX_DIM {
            return err(format!("dimension {n} exceeds the supported maximum {MAX_DIM}"));
        }
        if metric.len() != n || metric.iter().any(|row| row.len() != n) {
            return err("metric must be an n x n matrix");
        }
        for i in 0..n {
            for j in 0..n {
                if metric[i][j] != metric[j][i] {
                    return err("metric must be symmetric");
                }
            }
        }
        for (i, c) in names.iter().enumerate() {
            if !c.is_ascii_alphanumeric() {
                return err(format!("basis name suffix {c:?} must be ASCII alphanumeric"));
            }
            if names[..i].contains(c) {
                return err(format!("duplicate basis name e{c}"));
            }
        }
        let mut alg = Algebra { names, metric, table: Vec::new() };
        alg.table = alg.build_table();
        Ok(alg)
    }

    /// Dimension of the generating vector space.
    pub fn dim(&self) -> usize {
        self.names.len()
    }

    /// Number of basis blades, `2^dim`.
    pub fn blade_count(&self) -> usize {
        1 << self.dim()
    }

    /// The metric matrix.
    pub fn metric(&self) -> &[Vec<i64>] {
        &self.metric
    }

    /// Whether the metric is diagonal.
    pub fn is_diagonal(&self) -> bool {
        (0..self.dim()).all(|i| (0..self.dim()).all(|j| i == j || self.metric[i][j] == 0))
    }

    /// Mask of the pseudoscalar.
    pub fn pseudoscalar(&self) -> u32 {
        (1u32 << self.dim()) - 1
    }

    /// The one-character suffixes of the basis vector names.
    pub fn basis_suffixes(&self) -> &[char] {
        &self.names
    }

    /// Canonical name of a blade: `1` for the scalar, else `e` and its factors in index order.
    pub fn blade_name(&self, mask: u32) -> String {
        if mask == 0 {
            return "1".into();
        }
        let mut s = String::from("e");
        for (i, c) in self.names.iter().enumerate() {
            if mask & (1 << i) != 0 {
                s.push(*c);
            }
        }
        s
    }

    /// Parse a blade written as `1` or `e` followed by factor suffixes in any order.
    /// Returns the canonical mask and the sign relating the written order to the canonical one:
    /// `e032 = -e023`, so `parse_blade("e032") == (mask(e023), -1)`.
    pub fn parse_blade(&self, s: &str) -> Result<(u32, i64), AlgebraError> {
        if s == "1" || s == "s" {
            return Ok((0, 1));
        }
        let Some(rest) = s.strip_prefix('e') else {
            return err(format!("blade {s:?} must be `1` or start with `e`"));
        };
        let mut indices = Vec::new();
        for c in rest.chars() {
            let Some(i) = self.names.iter().position(|&n| n == c) else {
                return err(format!("blade {s:?}: unknown basis vector e{c}"));
            };
            if indices.contains(&i) {
                return err(format!("blade {s:?}: repeated factor e{c}"));
            }
            indices.push(i);
        }
        if indices.is_empty() {
            return err(format!("blade {s:?} has no factors"));
        }
        // Sign of the permutation that sorts the indices (count inversions).
        let mut inversions = 0;
        for a in 0..indices.len() {
            for b in a + 1..indices.len() {
                if indices[a] > indices[b] {
                    inversions += 1;
                }
            }
        }
        let mask = indices.iter().fold(0u32, |m, &i| m | (1 << i));
        Ok((mask, if inversions % 2 == 0 { 1 } else { -1 }))
    }

    /// Grade of a blade.
    pub fn grade(mask: u32) -> u32 {
        mask.count_ones()
    }

    /// Sign of reordering `e_a ^ e_b` (both in canonical order) into canonical order, or 0
    /// when they share a factor. Metric free.
    pub fn wedge_sign(a: u32, b: u32) -> i64 {
        if a & b != 0 {
            return 0;
        }
        // Count pairs (i in a, j in b) with i > j: each needs one swap.
        let mut swaps = 0;
        let mut x = a >> 1;
        while x != 0 {
            swaps += (x & b).count_ones();
            x >>= 1;
        }
        if swaps % 2 == 0 { 1 } else { -1 }
    }

    /// Product of basis vector `i` with basis blade `b`: `e_i b = e_i ⌋ b + e_i ^ b`.
    fn vector_times_blade(&self, i: usize, b: u32) -> Sparse {
        let mut out = Sparse::new();
        let bit = 1u32 << i;
        let s = Self::wedge_sign(bit, b);
        if s != 0 {
            *out.entry(b | bit).or_default() += s;
        }
        // Left contraction: e_i ⌋ (b1 ^ ... ^ bk) = Σ_j (-1)^(j) g(i, b_j) (b without b_j).
        let mut position = 0;
        for j in 0..self.dim() {
            if b & (1 << j) == 0 {
                continue;
            }
            let g = self.metric[i][j];
            if g != 0 {
                let sign = if position % 2 == 0 { 1 } else { -1 };
                *out.entry(b & !(1 << j)).or_default() += sign * g;
            }
            position += 1;
        }
        out.retain(|_, v| *v != 0);
        out
    }

    /// Product of a vector (as index) with a sparse multivector.
    fn vector_times(&self, i: usize, x: &Sparse) -> Sparse {
        let mut out = Sparse::new();
        for (&b, &c) in x {
            for (m, v) in self.vector_times_blade(i, b) {
                *out.entry(m).or_default() += c * v;
            }
        }
        out.retain(|_, v| *v != 0);
        out
    }

    /// Left contraction of vector `i` onto a sparse multivector.
    fn vector_lc(&self, i: usize, x: &Sparse) -> Sparse {
        let mut out = Sparse::new();
        for (&b, &c) in x {
            let mut position = 0;
            for j in 0..self.dim() {
                if b & (1 << j) == 0 {
                    continue;
                }
                let g = self.metric[i][j];
                if g != 0 {
                    let sign = if position % 2 == 0 { 1 } else { -1 };
                    *out.entry(b & !(1 << j)).or_default() += sign * g * c;
                }
                position += 1;
            }
        }
        out.retain(|_, v| *v != 0);
        out
    }

    /// Geometric product of two basis blades, computed by the Chevalley recursion
    /// `(e_i ^ A') X = e_i (A' X) - (e_i ⌋ A') X`, valid for any symmetric metric.
    fn blade_product_uncached(&self, a: u32, b: u32) -> Sparse {
        if a == 0 {
            return Sparse::from([(b, 1)]);
        }
        let i = a.trailing_zeros() as usize;
        let rest = a & !(1 << i);
        // e_a = e_i ^ e_rest  (i is the lowest factor, so no reordering sign)
        let rest_times_b = self.blade_product_uncached(rest, b);
        let mut out = self.vector_times(i, &rest_times_b);
        let contraction = self.vector_lc(i, &Sparse::from([(rest, 1)]));
        for (&m, &c) in &contraction {
            for (mm, v) in self.blade_product_uncached(m, b) {
                *out.entry(mm).or_default() -= c * v;
            }
        }
        out.retain(|_, v| *v != 0);
        out
    }

    fn build_table(&self) -> Vec<Vec<(u32, i64)>> {
        let n = self.blade_count();
        let mut table = Vec::with_capacity(n * n);
        for a in 0..n as u32 {
            for b in 0..n as u32 {
                table.push(self.blade_product_uncached(a, b).into_iter().collect());
            }
        }
        table
    }

    /// Geometric product of two basis blades as a list of `(blade, coefficient)`.
    pub fn blade_product(&self, a: u32, b: u32) -> &[(u32, i64)] {
        &self.table[(a as usize) * self.blade_count() + b as usize]
    }

    /// Geometric product of two sparse multivectors.
    pub fn product(&self, x: &Sparse, y: &Sparse) -> Sparse {
        let mut out = Sparse::new();
        for (&a, &ca) in x {
            for (&b, &cb) in y {
                for &(m, v) in self.blade_product(a, b) {
                    *out.entry(m).or_default() += ca * cb * v;
                }
            }
        }
        out.retain(|_, v| *v != 0);
        out
    }

    /// Sign of the reverse on a blade of the given grade: `(-1)^(k(k-1)/2)`.
    pub fn reverse_sign(grade: u32) -> i64 {
        if (grade / 2) % 2 == 0 { 1 } else { -1 }
    }

    /// Sign of the grade involution on a blade of the given grade: `(-1)^k`.
    pub fn involute_sign(grade: u32) -> i64 {
        if grade % 2 == 0 { 1 } else { -1 }
    }

    /// Sign of the Clifford conjugate on a blade of the given grade.
    pub fn conjugate_sign(grade: u32) -> i64 {
        Self::reverse_sign(grade) * Self::involute_sign(grade)
    }

    /// Right complement of a blade: the blade `c` with `e_a ^ c = I` (metric free).
    pub fn right_complement(&self, a: u32) -> (u32, i64) {
        let c = self.pseudoscalar() & !a;
        (c, Self::wedge_sign(a, c))
    }

    /// Left complement of a blade: the blade `c` with `c ^ e_a = I` (metric free).
    pub fn left_complement(&self, a: u32) -> (u32, i64) {
        let c = self.pseudoscalar() & !a;
        (c, Self::wedge_sign(c, a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pga3d() -> Algebra {
        Algebra::diagonal("0123", &[0, 1, 1, 1]).unwrap()
    }

    #[test]
    fn parse_blade_orders() {
        let a = pga3d();
        let (m, s) = a.parse_blade("e032").unwrap();
        assert_eq!(a.blade_name(m), "e023");
        assert_eq!(s, -1);
        assert_eq!(a.parse_blade("e31").unwrap().1, -1);
        assert_eq!(a.parse_blade("e0123").unwrap(), (0b1111, 1));
        assert!(a.parse_blade("e00").is_err());
        assert!(a.parse_blade("e4").is_err());
    }

    #[test]
    fn diagonal_products() {
        let a = pga3d();
        let e = |s: &str| a.parse_blade(s).unwrap().0;
        assert_eq!(a.blade_product(e("e1"), e("e1")), &[(0, 1)]);
        assert!(a.blade_product(e("e0"), e("e0")).is_empty());
        assert_eq!(a.blade_product(e("e2"), e("e1")), &[(e("e12"), -1)]);
        assert_eq!(a.blade_product(e("e12"), e("e12")), &[(0, -1)]);
    }

    #[test]
    fn null_basis_products() {
        // CGA-style null pair: eo.ei = -1, eo^2 = ei^2 = 0.
        let a = Algebra::new("oi", vec![vec![0, -1], vec![-1, 0]]).unwrap();
        let (o, i) = (1u32, 2u32);
        // eo ei = eo.ei + eo^ei = -1 + eoi
        assert_eq!(a.blade_product(o, i), &[(0, -1), (3, 1)]);
        // ei eo = -1 - eoi
        assert_eq!(a.blade_product(i, o), &[(0, -1), (3, -1)]);
        // (eo^ei)^2 = 1
        assert_eq!(a.blade_product(3, 3), &[(0, 1)]);
    }
}
