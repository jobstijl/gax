//! Test harness: a deliberately naive dense multivector oracle and random inputs.
//!
//! The oracle stores all `2^n` blade coefficients in `f64` and multiplies every pair of
//! blades through the exact blade tables of `gax-gen` (which are themselves checked against
//! an independent rational oracle in `gax-gen/tests/oracle.rs`).
#![allow(dead_code)]

use gax::{Extensor, Kind, Slots};
use gax_gen::algebra::Algebra;
use gax_gen::spec::AlgebraSpec;
use gax_gen::table::{BinOp, UnOp, blade_binop, blade_unop};

/// A small deterministic generator (xorshift), so failures reproduce.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

pub struct Oracle {
    pub alg: Algebra,
}

pub type Dense = Vec<f64>;

impl Oracle {
    pub fn from_spec(src: &str) -> Oracle {
        Oracle {
            alg: AlgebraSpec::parse(src).expect("spec").algebra,
        }
    }

    /// Dense coefficients of a value of any generated kind.
    pub fn dense<M: Extensor<Slots = (), Coef = f64>>(&self, m: &M) -> Dense {
        let mut d = vec![0.0; self.alg.blade_count()];
        for (b, c) in <M::Kind as Kind>::BLADES.iter().zip(m.coeffs().as_ref()) {
            let (mask, sign) = self.alg.parse_blade(b).unwrap();
            d[mask as usize] += sign as f64 * c;
        }
        d
    }

    pub fn binop(&self, op: BinOp, a: &Dense, b: &Dense) -> Dense {
        let mut out = vec![0.0; a.len()];
        for (i, &x) in a.iter().enumerate() {
            for (j, &y) in b.iter().enumerate() {
                if x == 0.0 || y == 0.0 {
                    continue;
                }
                for (m, c) in blade_binop(&self.alg, op, i as u32, j as u32) {
                    out[m as usize] += c as f64 * x * y;
                }
            }
        }
        out
    }

    pub fn unop(&self, op: UnOp, a: &Dense) -> Dense {
        let mut out = vec![0.0; a.len()];
        for (i, &x) in a.iter().enumerate() {
            let (m, s) = blade_unop(&self.alg, op, i as u32);
            out[m as usize] += s as f64 * x;
        }
        out
    }
}

/// A random value of a generated kind.
pub fn random<M: Extensor<Slots = (), Coef = f64>>(rng: &mut Rng) -> M {
    M::from_coeffs(<M::Kind as Kind>::arr_from_fn(|_| rng.next_f64()))
}

/// A random map or form of a generated kind with arbitrary slots.
pub fn random_map<M: Extensor<Coef = f64>>(rng: &mut Rng) -> M {
    M::from_coeffs(<M::Kind as Kind>::arr_from_fn(|_| {
        <M::Slots as Slots>::from_flat(&mut |_| rng.next_f64(), 0)
    }))
}

/// Flattened coefficients of any extensor.
pub fn flat<M: Extensor<Coef = f64>>(m: &M) -> Vec<f64> {
    let mut v = Vec::new();
    for col in m.coeffs().as_ref() {
        for k in 0..<M::Slots as Slots>::SIZE {
            v.push(<M::Slots as Slots>::get_flat(col, k));
        }
    }
    v
}

#[track_caller]
pub fn assert_close(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    let scale = a.iter().chain(b).fold(1.0f64, |m, x| m.max(x.abs()));
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        // Inputs are in [-1, 1]; products of a few such numbers accumulate rounding of order
        // (number of terms) * eps * scale. 1e-12 relative is far above that and far below any
        // real error (which would be O(1)).
        assert!(
            (x - y).abs() <= 1e-12 * scale,
            "{what}: coefficient {i}: {x} vs {y}\n{a:?}\n{b:?}"
        );
    }
}

impl Oracle {
    /// Scale a versor so that `v ~v = 1`, when its norm `v ~v` is a Study number `a + b I`
    /// with `I² = 0` (plane-based PGA) or a plain scalar. `None` otherwise or when `a <= 0`.
    pub fn normalize(&self, v: &Dense) -> Option<Dense> {
        let n = self.binop(BinOp::Gp, v, &self.unop(UnOp::Reverse, v));
        let ps = self.alg.pseudoscalar() as usize;
        let degenerate_ps = self.alg.blade_product(ps as u32, ps as u32).is_empty();
        let a = n[0];
        if a <= 1e-3 {
            return None;
        }
        let b = if degenerate_ps { n[ps] } else { 0.0 };
        if n.iter()
            .enumerate()
            .any(|(i, c)| i != 0 && !(degenerate_ps && i == ps) && c.abs() > 1e-12)
        {
            return None;
        }
        // (a + b I)^(-1/2) = a^(-1/2) - b/2 a^(-3/2) I, since I² = 0.
        let mut s = vec![0.0; n.len()];
        s[0] = a.powf(-0.5);
        s[ps] += -0.5 * b * a.powf(-1.5);
        let out = self.binop(BinOp::Gp, v, &s);
        let check = self.binop(BinOp::Gp, &out, &self.unop(UnOp::Reverse, &out));
        let ok = check
            .iter()
            .enumerate()
            .all(|(i, c)| (c - if i == 0 { 1.0 } else { 0.0 }).abs() < 1e-9);
        ok.then_some(out)
    }

    /// A value of kind `M` from dense coefficients; panics if the support does not fit.
    pub fn from_dense<M: Extensor<Slots = (), Coef = f64>>(&self, d: &Dense) -> M {
        let mut used = vec![false; d.len()];
        let c = <M::Kind as Kind>::arr_from_fn(|i| {
            let (mask, sign) = self.alg.parse_blade(<M::Kind as Kind>::BLADES[i]).unwrap();
            used[mask as usize] = true;
            sign as f64 * d[mask as usize]
        });
        for (i, x) in d.iter().enumerate() {
            assert!(
                used[i] || x.abs() < 1e-12,
                "from_dense: blade {i} outside the kind"
            );
        }
        M::from_coeffs(c)
    }
}

impl Oracle {
    /// A random unit versor of kind `V`: a normalized random element of `V` if that works,
    /// else a normalized product of random vectors that lies in `V`.
    pub fn random_unit<V: Extensor<Slots = (), Coef = f64>>(&self, rng: &mut Rng) -> Option<V> {
        let fits = |d: &Dense| {
            let blades: Vec<u32> = <V::Kind as Kind>::BLADES
                .iter()
                .map(|b| self.alg.parse_blade(b).unwrap().0)
                .collect();
            d.iter()
                .enumerate()
                .all(|(i, c)| c.abs() < 1e-12 || blades.contains(&(i as u32)))
        };
        let v: V = random(rng);
        if let Some(n) = self.normalize(&self.dense(&v)) {
            return Some(self.from_dense(&n));
        }
        for factors in 1..=4 {
            let mut d = vec![0.0; self.alg.blade_count()];
            d[0] = 1.0;
            for _ in 0..factors {
                let mut vec = vec![0.0; self.alg.blade_count()];
                for i in 0..self.alg.dim() {
                    vec[1 << i] = rng.next_f64();
                }
                d = self.binop(BinOp::Gp, &d, &vec);
            }
            if let Some(n) = self.normalize(&d).filter(|n| fits(n)) {
                return Some(self.from_dense(&n));
            }
        }
        None
    }
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

/// `x * inverse(x) == 1`, relative to `|x| |inverse(x)|`.
pub fn inverse<X, R>(o: &Oracle, rng: &mut Rng, f: impl Fn(X) -> R)
where
    X: Extensor<Slots = (), Coef = f64>,
    R: Extensor<Slots = (), Coef = f64>,
{
    for _ in 0..8 {
        let x: X = random(rng);
        let (dx, di) = (o.dense(&x), o.dense(&f(x)));
        let p = o.binop(BinOp::Gp, &dx, &di);
        let scale = max_abs(&dx) * max_abs(&di) * 1e-12;
        for (i, c) in p.iter().enumerate() {
            let want = if i == 0 { 1.0 } else { 0.0 };
            assert!(
                (c - want).abs() <= scale.max(1e-12),
                "x x⁻¹ = 1 for {}: {p:?}",
                std::any::type_name::<X>()
            );
        }
    }
}

/// `normalized(x) ~normalized(x) == ±1`.
pub fn normalized<X>(o: &Oracle, rng: &mut Rng, f: impl Fn(X) -> X)
where
    X: Extensor<Slots = (), Coef = f64>,
{
    for _ in 0..8 {
        let x: X = random(rng);
        let d = o.dense(&f(x));
        let n = o.binop(BinOp::Gp, &d, &o.unop(UnOp::Reverse, &d));
        assert!(
            (n[0].abs() - 1.0).abs() < 1e-10,
            "normalized scalar for {}: {n:?}",
            std::any::type_name::<X>()
        );
        assert!(
            n[1..].iter().all(|c| c.abs() < 1e-10),
            "normalized: other parts vanish for {}: {n:?}",
            std::any::type_name::<X>()
        );
    }
}

/// `exp(B)` is a unit versor, `log(exp(B)) == B`, and `exp(log(R)) == R`.
pub fn exp_log<B, R>(
    o: &Oracle,
    rng: &mut Rng,
    exp: impl Fn(B) -> gax::Unit<R>,
    log: impl Fn(gax::Unit<R>) -> B,
) where
    B: Extensor<Slots = (), Coef = f64> + gax::Gp<f64, Output = B>,
    R: Extensor<Slots = (), Coef = f64>,
{
    for _ in 0..8 {
        let b: B = random::<B>(rng).gp(0.5);
        let r = exp(b);
        let d = o.dense(&r.into_inner());
        let n = o.binop(BinOp::Gp, &d, &o.unop(UnOp::Reverse, &d));
        assert!(
            (n[0] - 1.0).abs() < 1e-12 && n[1..].iter().all(|c| c.abs() < 1e-12),
            "exp is unit: {n:?}"
        );
        assert_close(&o.dense(&log(r)), &o.dense(&b), "log(exp(B)) == B");
        assert_close(&o.dense(&exp(log(r)).into_inner()), &d, "exp(log(R)) == R");
    }
}

/// `sqrt(R)² == R` for unit versors `R = exp(B)`.
pub fn sqrt<R>(
    o: &Oracle,
    rng: &mut Rng,
    sqrt: impl Fn(gax::Unit<R>) -> R,
    square: impl Fn(gax::Unit<R>) -> R,
) where
    R: Extensor<Slots = (), Coef = f64>,
{
    for _ in 0..8 {
        let Some(r) = o.random_unit::<R>(rng) else {
            continue;
        };
        // R and -R are the same transformation; the principal root needs the representative
        // with a non-negative scalar part (it is undefined at a scalar part of exactly -1).
        let d = o.dense(&r);
        let r: R = if d[0] < 0.0 {
            o.from_dense(&d.iter().map(|c| -c).collect::<Vec<_>>())
        } else {
            r
        };
        let u = gax::Unit::new_unchecked(r);
        let s = gax::Unit::new_unchecked(sqrt(u));
        let back = square(s);
        let (db, dr) = (o.dense(&back), o.dense(&r));
        // The principal root of R and of -R differ; both square to R.
        assert_close(&db, &dr, "sqrt(R)² == R");
    }
}
