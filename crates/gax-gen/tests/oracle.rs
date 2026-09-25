//! An independent, deliberately naive oracle for blade products.
//!
//! The generator computes products in the wedge basis by the Chevalley recursion. This
//! oracle takes a different route: it diagonalizes the metric by a rational congruence
//! `G = P D Pᵀ`, multiplies in the orthogonal basis with the textbook bitmask rule
//! (sign from counting swaps, one metric factor per shared vector), and converts back with
//! the outermorphism of `P⁻¹` (determinants of minors). The two agree on every blade pair
//! of every algebra we ship.

use gax_gen::algebra::Algebra;
use proptest::prelude::*;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Q(i128, i128);

fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}
impl Q {
    fn new(n: i128, d: i128) -> Q {
        assert!(d != 0);
        let g = gcd(n, d).max(1);
        let s = if d < 0 { -1 } else { 1 };
        Q(s * n / g, s * d / g)
    }
    fn int(n: i128) -> Q {
        Q(n, 1)
    }
    fn zero() -> Q {
        Q(0, 1)
    }
    fn is_zero(self) -> bool {
        self.0 == 0
    }
    fn add(self, o: Q) -> Q {
        Q::new(self.0 * o.1 + o.0 * self.1, self.1 * o.1)
    }
    fn sub(self, o: Q) -> Q {
        self.add(Q(-o.0, o.1))
    }
    fn mul(self, o: Q) -> Q {
        Q::new(self.0 * o.0, self.1 * o.1)
    }
    fn div(self, o: Q) -> Q {
        Q::new(self.0 * o.1, self.1 * o.0)
    }
}

type Mat = Vec<Vec<Q>>;

/// Congruence diagonalization: returns (P, d) with G = P diag(d) Pᵀ, i.e. e_i = Σ_k P[i][k] f_k.
fn diagonalize(g: &[Vec<i64>]) -> (Mat, Vec<Q>) {
    let n = g.len();
    // Work on the Gram matrix of the current basis f, expressed in e: f_k = Σ_i C[k][i] e_i.
    let mut c: Mat = (0..n)
        .map(|k| (0..n).map(|i| Q::int(i128::from(k == i))).collect())
        .collect();
    let gram = |c: &Mat, a: usize, b: usize| {
        let mut s = Q::zero();
        for i in 0..n {
            for j in 0..n {
                s = s.add(c[a][i].mul(c[b][j]).mul(Q::int(i128::from(g[i][j]))));
            }
        }
        s
    };
    for k in 0..n {
        // Ensure a nonzero pivot if any later vector has a nonzero pairing.
        if gram(&c, k, k).is_zero() {
            if let Some(j) = (k + 1..n).find(|&j| !gram(&c, k, j).is_zero()) {
                // f_k <- f_k + f_j (or f_k - f_j if that is still null)
                let mut plus = c[k].clone();
                for i in 0..n {
                    plus[i] = plus[i].add(c[j][i]);
                }
                let old = std::mem::replace(&mut c[k], plus);
                if gram(&c, k, k).is_zero() {
                    let mut minus = old.clone();
                    for i in 0..n {
                        minus[i] = minus[i].sub(c[j][i]);
                    }
                    c[k] = minus;
                }
            }
        }
        let pivot = gram(&c, k, k);
        if pivot.is_zero() {
            continue;
        }
        for j in k + 1..n {
            let f = gram(&c, k, j).div(pivot);
            for i in 0..n {
                let v = c[j][i].sub(f.mul(c[k][i]));
                c[j][i] = v;
            }
        }
    }
    let d: Vec<Q> = (0..n).map(|k| gram(&c, k, k)).collect();
    for a in 0..n {
        for b in 0..n {
            if a != b {
                assert!(gram(&c, a, b).is_zero(), "diagonalization failed");
            }
        }
    }
    // P = C⁻¹ (e in terms of f), by Gauss-Jordan.
    let mut m = c.clone();
    let mut inv: Mat = (0..n)
        .map(|k| (0..n).map(|i| Q::int(i128::from(k == i))).collect())
        .collect();
    for col in 0..n {
        let p = (col..n)
            .find(|&r| !m[r][col].is_zero())
            .expect("singular basis change");
        m.swap(col, p);
        inv.swap(col, p);
        let pv = m[col][col];
        for i in 0..n {
            m[col][i] = m[col][i].div(pv);
            inv[col][i] = inv[col][i].div(pv);
        }
        for r in 0..n {
            if r != col && !m[r][col].is_zero() {
                let f = m[r][col];
                for i in 0..n {
                    m[r][i] = m[r][i].sub(f.mul(m[col][i]));
                    inv[r][i] = inv[r][i].sub(f.mul(inv[col][i]));
                }
            }
        }
    }
    // f = C e  =>  e = C⁻¹ f, and inv holds C⁻¹ with rows indexed like C: e_i = Σ_k inv?  Check:
    // C is k x i (f_k = Σ_i C[k][i] e_i). Then e = C⁻¹ f means e_i = Σ_k (C⁻¹)[i][k] f_k.
    (inv, d)
}

fn det(m: &Mat) -> Q {
    let n = m.len();
    if n == 0 {
        return Q::int(1);
    }
    let mut total = Q::zero();
    for col in 0..n {
        let minor: Mat = (1..n)
            .map(|r| (0..n).filter(|&c| c != col).map(|c| m[r][c]).collect())
            .collect();
        let term = m[0][col].mul(det(&minor));
        total = if col % 2 == 0 {
            total.add(term)
        } else {
            total.sub(term)
        };
    }
    total
}

fn bits(mask: u32) -> Vec<usize> {
    (0..32).filter(|i| mask & (1 << i) != 0).collect()
}

/// Outermorphism of a basis change: blade e_A (wedge of rows A of M) in the other basis.
fn outermorphism(m: &Mat, a: u32, n: usize) -> BTreeMap<u32, Q> {
    let rows = bits(a);
    let mut out = BTreeMap::new();
    for k in 0..(1u32 << n) {
        if k.count_ones() as usize != rows.len() {
            continue;
        }
        let cols = bits(k);
        let minor: Mat = rows
            .iter()
            .map(|&r| cols.iter().map(|&c| m[r][c]).collect())
            .collect();
        let d = det(&minor);
        if !d.is_zero() {
            out.insert(k, d);
        }
    }
    out
}

/// Textbook orthogonal-basis product: sign from swaps, metric factor per shared vector.
fn diag_blade_product(d: &[Q], a: u32, b: u32) -> (u32, Q) {
    let mut swaps = 0;
    for i in bits(a) {
        for j in bits(b) {
            if i > j {
                swaps += 1;
            }
        }
    }
    let mut c = Q::int(if swaps % 2 == 0 { 1 } else { -1 });
    for i in bits(a & b) {
        c = c.mul(d[i]);
    }
    (a ^ b, c)
}

fn oracle_product(alg: &Algebra, a: u32, b: u32) -> BTreeMap<u32, i64> {
    let n = alg.dim();
    let (p, d) = diagonalize(alg.metric());
    // inverse change: f in terms of e
    let mut pinv: Mat;
    {
        // invert p again (Gauss-Jordan) to express f in e
        let mut m = p.clone();
        pinv = (0..n)
            .map(|k| (0..n).map(|i| Q::int(i128::from(k == i))).collect())
            .collect();
        for col in 0..n {
            let pr = (col..n).find(|&r| !m[r][col].is_zero()).unwrap();
            m.swap(col, pr);
            pinv.swap(col, pr);
            let pv = m[col][col];
            for i in 0..n {
                m[col][i] = m[col][i].div(pv);
                pinv[col][i] = pinv[col][i].div(pv);
            }
            for r in 0..n {
                if r != col && !m[r][col].is_zero() {
                    let f = m[r][col];
                    for i in 0..n {
                        m[r][i] = m[r][i].sub(f.mul(m[col][i]));
                        pinv[r][i] = pinv[r][i].sub(f.mul(pinv[col][i]));
                    }
                }
            }
        }
    }
    let fa = outermorphism(&p, a, n);
    let fb = outermorphism(&p, b, n);
    let mut prod: BTreeMap<u32, Q> = BTreeMap::new();
    for (&x, &cx) in &fa {
        for (&y, &cy) in &fb {
            let (m, c) = diag_blade_product(&d, x, y);
            let e = prod.entry(m).or_insert(Q::zero());
            *e = e.add(cx.mul(cy).mul(c));
        }
    }
    // back to the e basis: f_K = outermorphism of pinv
    let mut out: BTreeMap<u32, Q> = BTreeMap::new();
    for (&k, &ck) in &prod {
        for (m, cm) in outermorphism(&pinv, k, n) {
            let e = out.entry(m).or_insert(Q::zero());
            *e = e.add(ck.mul(cm));
        }
    }
    out.into_iter()
        .filter(|(_, v)| !v.is_zero())
        .map(|(k, v)| {
            assert_eq!(v.1, 1, "non-integer product coefficient");
            (k, i64::try_from(v.0).unwrap())
        })
        .collect()
}

fn algebras() -> Vec<(&'static str, Algebra)> {
    let cga = |extra: &str| {
        // e1..en Euclidean, then eo, ei with eo.ei = -1
        let n = extra.len() + 2;
        let mut g = vec![vec![0; n]; n];
        for i in 0..extra.len() {
            g[i][i] = 1;
        }
        g[n - 2][n - 1] = -1;
        g[n - 1][n - 2] = -1;
        Algebra::new(&format!("{extra}oi"), g).unwrap()
    };
    vec![
        ("pga2d", Algebra::diagonal("012", &[0, 1, 1]).unwrap()),
        ("pga3d", Algebra::diagonal("0123", &[0, 1, 1, 1]).unwrap()),
        ("vga3d", Algebra::diagonal("123", &[1, 1, 1]).unwrap()),
        ("sta", Algebra::diagonal("0123", &[1, -1, -1, -1]).unwrap()),
        (
            "stap",
            Algebra::diagonal("p0123", &[0, 1, -1, -1, -1]).unwrap(),
        ),
        ("cga2d", cga("12")),
        ("cga3d", cga("123")),
        (
            "csta",
            Algebra::diagonal("0123pm", &[1, -1, -1, -1, 1, -1]).unwrap(),
        ),
        (
            "mixed",
            Algebra::new("abc", vec![vec![1, 2, 0], vec![2, -1, 1], vec![0, 1, 0]]).unwrap(),
        ),
    ]
}

#[test]
fn generator_matches_oracle_on_every_blade_pair() {
    for (name, alg) in algebras() {
        let n = alg.blade_count() as u32;
        for a in 0..n {
            for b in 0..n {
                let got: BTreeMap<u32, i64> = alg.blade_product(a, b).iter().copied().collect();
                let want = oracle_product(&alg, a, b);
                assert_eq!(
                    got,
                    want,
                    "{name}: {} * {}",
                    alg.blade_name(a),
                    alg.blade_name(b)
                );
            }
        }
    }
}

fn arb_sparse(n_blades: u32) -> impl Strategy<Value = BTreeMap<u32, i64>> {
    prop::collection::btree_map(0..n_blades, -3i64..=3, 0..6)
}

proptest! {
    #[test]
    fn product_is_associative_in_every_algebra(seed in 0usize..9, x in arb_sparse(64), y in arb_sparse(64), z in arb_sparse(64)) {
        let (_, alg) = &algebras()[seed];
        let n = alg.blade_count() as u32;
        let clip = |m: &BTreeMap<u32, i64>| m.iter().filter(|(k, _)| **k < n).map(|(k, v)| (*k, *v)).collect::<BTreeMap<_, _>>();
        let (x, y, z) = (clip(&x), clip(&y), clip(&z));
        let left = alg.product(&alg.product(&x, &y), &z);
        let right = alg.product(&x, &alg.product(&y, &z));
        prop_assert_eq!(left, right);
    }
}
