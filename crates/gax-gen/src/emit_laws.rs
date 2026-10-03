//! The law suite of an algebra: which laws apply to which of its generated types, and the
//! scale factors of the versor laws, derived here with the generator's own symbolic algebra.
//!
//! The emitted test file is one `law_suite!` invocation (see `gax/tests/law_suite`), which
//! proves each law on symbolic coefficients against the generated library code. The factors
//! computed here are the expected values, so the library's kernels are checked against an
//! independent derivation from the tables.

use crate::emit::Stats;
use crate::groebner::{groebner, normal_form};
use crate::poly::{Poly, Rational, Var};
use crate::spec::{AlgebraSpec, KindSpec};
use crate::symbolic::{self, SymMv};
use crate::table::{BinOp, UnOp, blade_binop};
use std::collections::BTreeSet;
use std::fmt::Write as _;

/// Kinds with at most this many coefficients take part in the laws (maps over them stay
/// small enough for exact symbolic arithmetic).
const LAW_KIND: usize = 8;
/// Outermorphism laws are proved for maps on kinds with at most this many coefficients.
const OUTER_BASE: usize = 5;
/// Versors with at most this many coefficients take part in the versor laws.
const LAW_VERSOR: usize = 8;

/// A factor `sign · N^power`, where `N` is the scalar part of `v ~v`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Factor {
    /// `±1`.
    pub sign: i8,
    /// The power of `N`.
    pub power: u32,
}

impl Factor {
    fn describe(self) -> String {
        let n = match self.power {
            0 => "1".to_string(),
            1 => "‖m‖²".to_string(),
            k => format!("‖m‖^{}", 2 * k),
        };
        if self.sign < 0 { format!("−{n}") } else { n }
    }
}

/// One row of the equivariance table.
#[derive(Clone, Debug)]
pub struct Equivariance {
    /// The versor kind.
    pub versor: String,
    /// The product.
    pub op: BinOp,
    /// Operands and result.
    pub kinds: (String, String, String),
    /// The factor, if one of `±N^k` (k ≤ 4) makes the law hold modulo the versor conditions.
    pub factor: Option<Factor>,
}

fn trait_of(op: BinOp) -> (&'static str, &'static str, &'static str) {
    op.names()
}

fn unop_trait(op: UnOp) -> (&'static str, &'static str) {
    (op.trait_name(), op.name())
}

fn single_grade(k: &KindSpec) -> bool {
    k.layout.grades().len() == 1
}

fn sandwich(spec: &AlgebraSpec, v: &SymMv, x: &SymMv) -> SymMv {
    let alg = &spec.algebra;
    symbolic::binop(
        alg,
        BinOp::Gp,
        &symbolic::binop(alg, BinOp::Gp, v, x),
        &symbolic::unop(alg, UnOp::Reverse, v),
    )
}

/// A versor's norm `N` and the Gröbner basis of its conditions.
type Conditions = (Poly, Vec<Poly>);

/// A product: the operation, the operand kinds and the output kind.
type Product = (BinOp, String, String, String);

/// `N`, the scalar part of `v ~v`, and a Gröbner basis of the other parts (the conditions
/// under which `v` is a versor).
fn versor_conditions(spec: &AlgebraSpec, v: &SymMv) -> Option<Conditions> {
    let alg = &spec.algebra;
    let vv = symbolic::binop(alg, BinOp::Gp, v, &symbolic::unop(alg, UnOp::Reverse, v));
    let n = vv.get(&0).cloned().unwrap_or_else(Poly::zero);
    let gens: Vec<Poly> = vv
        .iter()
        .filter(|(m, p)| **m != 0 && !p.is_zero())
        .map(|(_, p)| p.clone())
        .collect();
    let basis = groebner(&gens, 4000)?;
    Some((n, basis))
}

fn equal_mod(a: &SymMv, b: &SymMv, basis: &[Poly]) -> bool {
    let keys: BTreeSet<u32> = a.keys().chain(b.keys()).copied().collect();
    keys.into_iter().all(|m| {
        let d = &a.get(&m).cloned().unwrap_or_else(Poly::zero)
            - &b.get(&m).cloned().unwrap_or_else(Poly::zero);
        normal_form(&d, basis).is_zero()
    })
}

/// The `±N^k` that makes `lhs == factor · rhs` hold modulo the basis, if any (k ≤ 4).
fn find_factor(lhs: &SymMv, rhs: &SymMv, n: &Poly, basis: &[Poly]) -> Option<Factor> {
    let mut power = Poly::constant(Rational::ONE);
    for k in 0..=4 {
        for sign in [1i8, -1] {
            let f = power.scale(Rational::int(i128::from(sign)));
            if equal_mod(lhs, &symbolic::scale_mv(rhs, &f), basis) {
                return Some(Factor { sign, power: k });
            }
        }
        power = &power * n;
    }
    None
}

/// Whether `op` pairs `a` and `b` into a scalar through a constant signed permutation matrix.
fn signed_permutation_pairing(spec: &AlgebraSpec, op: BinOp, a: &KindSpec, b: &KindSpec) -> bool {
    let alg = &spec.algebra;
    if a.layout.len() != b.layout.len() {
        return false;
    }
    let n = a.layout.len();
    let mut m = vec![vec![0i64; n]; n];
    for (i, &(ma, _)) in a.layout.blades.iter().enumerate() {
        for (j, &(mb, _)) in b.layout.blades.iter().enumerate() {
            for (mask, c) in blade_binop(alg, op, ma, mb) {
                if mask != 0 {
                    return false;
                }
                m[i][j] += c;
            }
        }
    }
    let ok = |line: &mut dyn Iterator<Item = i64>| {
        let nz: Vec<i64> = line.filter(|c| *c != 0).collect();
        nz.len() == 1 && nz[0].abs() == 1
    };
    (0..n).all(|i| ok(&mut m[i].iter().copied())) && (0..n).all(|j| ok(&mut m.iter().map(|r| r[j])))
}

/// The determinant of `x -> b x ~b` on kind `x` for a basis blade `b` of the versor kind with
/// `b ~b = 1`: the value the determinant of every unit versor of that kind's parity takes.
fn unit_det_sign(spec: &AlgebraSpec, v: &KindSpec, x: &KindSpec) -> Option<i64> {
    let alg = &spec.algebra;
    let b = v.layout.blades.iter().map(|(m, _)| *m).find(|&m| {
        let rev = crate::table::blade_unop(alg, UnOp::Reverse, m);
        let sq = blade_binop(alg, BinOp::Gp, m, rev.0);
        sq.len() == 1 && sq.get(&0).is_some_and(|c| c * rev.1 == 1)
    })?;
    let bs = symbolic::from_coeffs(
        &crate::table::Layout {
            blades: vec![(b, 1)],
        },
        &[Poly::constant(Rational::ONE)],
    );
    let n = x.layout.len();
    let mut m = vec![vec![Rational::ZERO; n]; n];
    for i in 0..n {
        let mut e = vec![Poly::zero(); n];
        e[i] = Poly::constant(Rational::ONE);
        let img = sandwich(spec, &bs, &symbolic::from_coeffs(&x.layout, &e));
        let c = symbolic::to_coeffs(&x.layout, &img)?;
        for (o, p) in c.iter().enumerate() {
            m[o][i] = p.as_constant().unwrap_or(Rational::ZERO);
        }
    }
    // Gaussian elimination with exact rationals.
    let mut det = Rational::ONE;
    for k in 0..n {
        let p = (k..n).find(|&r| !m[r][k].is_zero())?;
        if p != k {
            m.swap(p, k);
            det = -det;
        }
        det = det * m[k][k];
        let pivot = m[k].clone();
        for row in m.iter_mut().skip(k + 1) {
            let f = row[k] * pivot[k].recip();
            for (x, p) in row.iter_mut().zip(&pivot).skip(k) {
                *x = *x - *p * f;
            }
        }
    }
    Some(if det == Rational::ONE {
        1
    } else if det == -Rational::ONE {
        -1
    } else {
        return None;
    })
}

/// The law suite of an algebra, and its equivariance table.
pub struct Laws {
    /// The body of the `law_suite!` invocation.
    pub invocation: String,
    /// The equivariance rows (for the documentation table).
    pub equivariance: Vec<Equivariance>,
    /// Conjugation factors: `(versor, map kind, factor)`.
    pub conjugation: Vec<(String, String, Option<Factor>)>,
}

/// Derive the laws of an algebra from its spec and what the emitter generated.
#[allow(clippy::too_many_lines)]
pub fn laws(spec: &AlgebraSpec, stats: &Stats) -> Laws {
    let alg = &spec.algebra;
    let kind = |name: &str| spec.kinds.iter().find(|k| k.name == name).expect("kind");
    let law_kinds: Vec<&KindSpec> = spec
        .kinds
        .iter()
        .filter(|k| k.layout.len() <= LAW_KIND)
        .collect();
    let is_law = |name: &str| law_kinds.iter().any(|k| k.name == name);
    // The slot kinds of maps: the first single-grade kinds with 2 to 6 coefficients.
    let slots: Vec<&KindSpec> = spec
        .kinds
        .iter()
        .filter(|k| single_grade(k) && (2..=6).contains(&k.layout.len()))
        .collect();
    let slot = slots.first().expect("a slot kind");
    // A second, different slot kind (some laws need two), preferably single grade.
    let other = slots.get(1).copied().unwrap_or_else(|| {
        spec.kinds
            .iter()
            .find(|k| k.name != slot.name && (2..=6).contains(&k.layout.len()))
            .expect("a second slot kind")
    });
    let scalar = spec.scalar_kind();

    let products: Vec<&(BinOp, String, String, String)> = stats
        .products
        .iter()
        .filter(|(_, a, b, o)| is_law(a) && is_law(b) && is_law(o))
        .collect();
    let unary: Vec<&(UnOp, String, String)> = stats
        .unary
        .iter()
        .filter(|(_, a, o)| is_law(a) && is_law(o))
        .collect();
    let keeps = |v: &str, x: &str, unit: bool| {
        stats
            .sandwiches
            .iter()
            .any(|(sv, sx, su, so)| sv == v && sx == x && *su == unit && so == x)
    };
    let versors: Vec<&KindSpec> = spec
        .kinds
        .iter()
        .filter(|k| k.versor && k.layout.len() <= LAW_VERSOR)
        .collect();
    let gp_out = |a: &str, b: &str| {
        stats
            .products
            .iter()
            .find(|(op, x, y, _)| *op == BinOp::Gp && x == a && y == b)
            .map(|p| p.3.clone())
    };

    let mut sandwiches = Vec::new();
    let mut actions = Vec::new();
    for v in &versors {
        for x in &law_kinds {
            if keeps(&v.name, &x.name, false) {
                sandwiches.push((
                    v.name.clone(),
                    x.name.clone(),
                    keeps(&v.name, &x.name, true),
                ));
                if gp_out(&v.name, &v.name).as_deref() == Some(v.name.as_str()) {
                    actions.push((v.name.clone(), x.name.clone()));
                }
            }
        }
    }

    // Equivariance of the single-grade products under every versor, with the factor. Each
    // versor's conditions once, then every (versor, product) pair on its own core.
    let conditions: Vec<Option<Conditions>> = crate::par::map(&versors, |v| {
        versor_conditions(spec, &symbolic::variables(&kind(&v.name).layout, 0))
    });
    let jobs: Vec<(usize, &Product)> = versors
        .iter()
        .enumerate()
        .filter(|(k, _)| conditions[*k].is_some())
        .flat_map(|(k, v)| {
            products
                .iter()
                .filter(move |(_, a, b, o)| {
                    single_grade(kind(a))
                        && single_grade(kind(b))
                        && keeps(&v.name, a, false)
                        && keeps(&v.name, b, false)
                        && keeps(&v.name, o, false)
                })
                .map(move |p| (k, *p))
        })
        .collect();
    let equivariance: Vec<Equivariance> = crate::par::map(&jobs, |&(k, (op, a, b, o))| {
        let v = versors[k];
        let (n, basis) = conditions[k].as_ref().expect("filtered");
        let vk = kind(&v.name);
        let vs = symbolic::variables(&vk.layout, 0);
        let nv = vk.layout.len() as Var;
        let (ak, bk) = (kind(a), kind(b));
        let av = symbolic::variables(&ak.layout, nv);
        let bv = symbolic::variables(&bk.layout, nv + ak.layout.len() as Var);
        let lhs = symbolic::binop(
            alg,
            *op,
            &sandwich(spec, &vs, &av),
            &sandwich(spec, &vs, &bv),
        );
        let rhs = sandwich(spec, &vs, &symbolic::binop(alg, *op, &av, &bv));
        Equivariance {
            versor: v.name.clone(),
            op: *op,
            kinds: (a.clone(), b.clone(), o.clone()),
            factor: find_factor(&lhs, &rhs, n, basis),
        }
    });

    // Conjugation of maps on the slot kind: conj_m(f).of(m >> x) against m >> f.of(x).
    let conj_versors: Vec<(usize, &&KindSpec)> = versors
        .iter()
        .enumerate()
        .filter(|(k, v)| keeps(&v.name, &slot.name, false) && conditions[*k].is_some())
        .collect();
    let conjugation: Vec<(String, String, Option<Factor>)> =
        crate::par::map(&conj_versors, |&(k, v)| {
            let (n, basis) = conditions[k].as_ref().expect("filtered");
            let vk = kind(&v.name);
            let vs = symbolic::variables(&vk.layout, 0);
            let nv = vk.layout.len() as Var;
            let nx = slot.layout.len();
            // f as a matrix of variables, x as variables.
            let fvar = |o: usize, i: usize| Poly::var(nv + (o * nx + i) as Var);
            let xs = symbolic::variables(&slot.layout, nv + (nx * nx) as Var);
            let apply = |x: &SymMv| -> SymMv {
                let c = symbolic::to_coeffs(&slot.layout, x).expect("slot kind");
                let out: Vec<Poly> = (0..nx)
                    .map(|o| (0..nx).fold(Poly::zero(), |acc, i| &acc + &(&fvar(o, i) * &c[i])))
                    .collect();
                symbolic::from_coeffs(&slot.layout, &out)
            };
            // conj_m(f)(y) = m f(~m y m) ~m, so conj_m(f)(m x ~m) = m f(~m m x ~m m) ~m.
            let rev = symbolic::unop(alg, UnOp::Reverse, &vs);
            let lhs = sandwich(
                spec,
                &vs,
                &apply(&sandwich(spec, &rev, &sandwich(spec, &vs, &xs))),
            );
            let rhs = sandwich(spec, &vs, &apply(&xs));
            (
                v.name.clone(),
                slot.name.clone(),
                find_factor(&lhs, &rhs, n, basis),
            )
        });

    // Pairings that are constant signed permutations, for the adjoint laws.
    let mut pairings = Vec::new();
    for op in [BinOp::Vee, BinOp::Dot, BinOp::Wedge, BinOp::Scalar] {
        for a in &law_kinds {
            for b in &law_kinds {
                if single_grade(a)
                    && single_grade(b)
                    && a.layout.len() > 1
                    && stats.products.iter().any(|(p, x, y, o)| {
                        *p == op && *x == a.name && *y == b.name && *o == scalar.name
                    })
                    && signed_permutation_pairing(spec, op, a, b)
                {
                    pairings.push((op, a.name.clone(), b.name.clone()));
                }
            }
        }
    }

    // Outermorphisms of maps on kinds with at most OUTER_BASE coefficients: the determinant of
    // a product of two free 6x6 maps (degree 12 in 72 variables) is out of reach.
    let outer: Vec<&(String, String, bool)> = stats
        .outermorphisms
        .iter()
        .filter(|(v, b, _)| is_law(v) && is_law(b) && kind(v).layout.len() <= OUTER_BASE)
        .collect();
    let mut unit_det = Vec::new();
    for (base, top, is_top) in &outer {
        if !is_top {
            continue;
        }
        for v in &versors {
            if keeps(&v.name, base, true)
                && let Some(d) = unit_det_sign(spec, v, kind(base))
            {
                unit_det.push((v.name.clone(), base.clone(), top.clone(), d > 0));
            }
        }
    }

    // The invocation.
    let mut s = String::new();
    let list = |s: &mut String, name: &str, items: Vec<String>| {
        let _ = writeln!(s, "    {name}: [");
        for it in items {
            let _ = writeln!(s, "        {it},");
        }
        let _ = writeln!(s, "    ],");
    };
    let _ = writeln!(
        s,
        "    scalar: {},\n    slot: {},\n    other: {},",
        scalar.name, slot.name, other.name
    );
    list(
        &mut s,
        "products",
        products
            .iter()
            .map(|(op, a, b, _)| {
                let (t, m, _) = trait_of(*op);
                format!("({t}, {m}, {a}, {b})")
            })
            .collect(),
    );
    list(
        &mut s,
        "unary",
        unary
            .iter()
            .map(|(op, a, _)| {
                let (t, m) = unop_trait(*op);
                format!("({t}, {m}, {a})")
            })
            .collect(),
    );
    list(
        &mut s,
        "sandwiches",
        sandwiches
            .iter()
            .filter(|(_, _, unit)| *unit)
            .map(|(v, x, _)| format!("({v}, {x})"))
            .collect(),
    );
    list(
        &mut s,
        "plain_sandwiches",
        sandwiches
            .iter()
            .filter(|(_, _, unit)| !*unit)
            .map(|(v, x, _)| format!("({v}, {x})"))
            .collect(),
    );
    list(
        &mut s,
        "actions",
        actions.iter().map(|(v, x)| format!("({v}, {x})")).collect(),
    );
    list(
        &mut s,
        "equivariant",
        equivariance
            .iter()
            .filter_map(|e| {
                let f = e.factor?;
                let (t, m, _) = trait_of(e.op);
                Some(format!(
                    "({}, {t}, {m}, {}, {}, {}, {})",
                    e.versor, e.kinds.0, e.kinds.1, f.sign, f.power
                ))
            })
            .collect(),
    );
    list(
        &mut s,
        "unit_equivariant",
        equivariance
            .iter()
            .filter(|e| {
                let (a, b, o) = &e.kinds;
                keeps(&e.versor, a, true) && keeps(&e.versor, b, true) && keeps(&e.versor, o, true)
            })
            .filter_map(|e| {
                let f = e.factor?;
                let (t, m, _) = trait_of(e.op);
                Some(format!(
                    "({}, {t}, {m}, {}, {}, {})",
                    e.versor, e.kinds.0, e.kinds.1, f.sign
                ))
            })
            .collect(),
    );
    // Negative controls: the law without its factor must fail (one per versor).
    let mut seen = BTreeSet::new();
    list(
        &mut s,
        "violations",
        equivariance
            .iter()
            .filter(|e| e.factor.is_some_and(|f| f.power > 0) && seen.insert(e.versor.clone()))
            .map(|e| {
                let (t, m, _) = trait_of(e.op);
                format!("({}, {t}, {m}, {}, {})", e.versor, e.kinds.0, e.kinds.1)
            })
            .collect(),
    );
    list(
        &mut s,
        "conjugations",
        conjugation
            .iter()
            .filter_map(|(v, x, f)| {
                let f = (*f)?;
                actions
                    .iter()
                    .any(|(av, ax)| av == v && ax == x)
                    .then(|| format!("({v}, {x}, {}, {})", f.sign, f.power))
            })
            .collect(),
    );
    list(
        &mut s,
        "outer",
        outer
            .iter()
            .map(|(v, b, _)| format!("({v}, {b})"))
            .collect(),
    );
    list(
        &mut s,
        "top",
        outer
            .iter()
            .filter(|(_, _, t)| *t)
            .map(|(v, b, _)| format!("({v}, {b})"))
            .collect(),
    );
    list(
        &mut s,
        "unit_det",
        unit_det
            .iter()
            .map(|(v, x, top, even)| format!("({v}, {x}, {top}, {even})"))
            .collect(),
    );
    list(
        &mut s,
        "pairings",
        pairings
            .iter()
            .map(|(op, a, b)| {
                let (t, m, _) = trait_of(*op);
                format!("({t}, {m}, {a}, {b})")
            })
            .collect(),
    );
    // Division: `(a / v) v = a` for each invertible versor kind `v` and law kind `a` whose
    // quotient's product comes back to a kind holding `a`.
    let product_kind = |x: &str, y: &str| {
        stats
            .products
            .iter()
            .find(|(op, p, q, _)| *op == BinOp::Gp && p == x && q == y)
            .map(|p| p.3.clone())
    };
    let mut divisions = Vec::new();
    for (vk, meta) in spec.kinds.iter().zip(&stats.values) {
        if !(vk.versor
            && vk.layout.len() <= LAW_VERSOR
            && meta.inverse.as_deref() == Some(&vk.name))
        {
            continue;
        }
        for a in &law_kinds {
            let Some(quotient) = product_kind(&a.name, &vk.name) else {
                continue;
            };
            let Some(back) = product_kind(&quotient, &vk.name) else {
                continue;
            };
            let holds = kind(&back)
                .layout
                .blades
                .iter()
                .map(|b| b.0)
                .collect::<BTreeSet<u32>>();
            if a.layout.blades.iter().all(|b| holds.contains(&b.0)) {
                divisions.push((a.name.clone(), vk.name.clone(), back));
            }
        }
    }
    list(
        &mut s,
        "divisions",
        divisions
            .iter()
            .map(|(a, v, back)| format!("({a}, {v}, {back})"))
            .collect(),
    );
    // Versors with more than LAW_VERSOR coefficients (law L): their free symbolic laws are out
    // of reach, so they are checked on exact unit versors sampled as products of rational unit
    // vectors. Only for versor kinds that are a whole parity (the product of vectors lands in
    // them), with a vector kind, a full multivector kind, and a basis vector of square +1.
    let n = alg.dim();
    let parity_blades = |odd: bool| -> BTreeSet<u32> {
        (0..alg.blade_count() as u32)
            .filter(|m| m.count_ones() % 2 == u32::from(odd))
            .collect()
    };
    let full = spec
        .kinds
        .iter()
        .find(|k| k.layout.len() == alg.blade_count());
    let vector = spec
        .kinds
        .iter()
        .find(|k| k.layout.len() == n && k.layout.blades.iter().all(|(m, _)| m.is_power_of_two()));
    let u0 = vector.and_then(|vk| {
        vk.layout
            .blades
            .iter()
            .enumerate()
            .find_map(|(pos, &(m, sign))| {
                let i = m.trailing_zeros() as usize;
                let row = &alg.metric()[i];
                (row[i] == 1 && (0..n).all(|j| j == i || row[j] == 0)).then_some((pos, sign))
            })
    });
    let mut sampled = Vec::new();
    let mut sampled_unit = Vec::new();
    let mut sampled_actions = Vec::new();
    let mut sampled_equivariant = Vec::new();
    if let (Some(full), Some(vector), Some((pos, sign))) = (full, vector, u0) {
        for v in spec
            .kinds
            .iter()
            .filter(|k| k.versor && k.layout.len() > LAW_VERSOR)
        {
            let blades: BTreeSet<u32> = v.layout.blades.iter().map(|(m, _)| *m).collect();
            let odd = if blades == parity_blades(false) {
                false
            } else if blades == parity_blades(true) {
                true
            } else {
                continue;
            };
            let count = if odd { 3 } else { 4 };
            let head = format!(
                "{}, {}, {}, {count}, {pos}, {sign}",
                v.name, vector.name, full.name
            );
            // A plain sandwich of a big even element leaves the passenger's kind (other grades
            // appear); the law compares it with the projection of `(v x) ~v` whatever its kind.
            // The unit kernels keep it, and carry the action and equivariance laws.
            let plain = |x: &str| {
                stats
                    .sandwiches
                    .iter()
                    .any(|(sv, sx, su, _)| sv == &v.name && sx == x && !*su)
            };
            for x in &law_kinds {
                if plain(&x.name) {
                    sampled.push(format!("({head}, {})", x.name));
                }
                if keeps(&v.name, &x.name, true) {
                    sampled_unit.push(format!("({head}, {})", x.name));
                    if gp_out(&v.name, &v.name).as_deref() == Some(v.name.as_str()) {
                        sampled_actions.push(format!("({head}, {})", x.name));
                    }
                }
            }
            for (op, a, b, o) in &products {
                if single_grade(kind(a))
                    && single_grade(kind(b))
                    && keeps(&v.name, a, true)
                    && keeps(&v.name, b, true)
                    && keeps(&v.name, o, true)
                {
                    let (t, m, _) = trait_of(*op);
                    sampled_equivariant.push(format!("({head}, {t}, {m}, {a}, {b})"));
                }
            }
        }
    }
    list(&mut s, "sampled", sampled);
    list(&mut s, "sampled_unit", sampled_unit);
    list(&mut s, "sampled_actions", sampled_actions);
    list(&mut s, "sampled_equivariant", sampled_equivariant);
    // M: the kind tables (gax_core::cast) for every pair of kinds sharing a blade, the strict
    // inclusions, and every grade part.
    let (mut casts, mut sub_kinds, mut grades) = (Vec::new(), Vec::new(), Vec::new());
    for a in &spec.kinds {
        for b in &spec.kinds {
            let shared = a
                .layout
                .blades
                .iter()
                .filter(|(m, _)| b.layout.position(*m).is_some())
                .count();
            if shared == 0 {
                continue;
            }
            casts.push(format!("({}, {})", a.name, b.name));
            if shared == a.layout.len() && a.name != b.name {
                sub_kinds.push(format!("({}, {})", a.name, b.name));
            }
        }
        let by_grade: BTreeSet<u32> = a
            .layout
            .blades
            .iter()
            .map(|(m, _)| m.count_ones())
            .collect();
        for g in by_grade {
            let support: BTreeSet<u32> = a
                .layout
                .blades
                .iter()
                .map(|(m, _)| *m)
                .filter(|m| m.count_ones() == g)
                .collect();
            let out = spec
                .kind_for_support(&support)
                .expect("the kind itself holds it");
            grades.push(format!("({}, {g}, {})", a.name, out.name));
        }
    }
    // N: the general inverse (Shirokov's), with its output kind.
    let inverses: Vec<String> = spec
        .kinds
        .iter()
        .zip(&stats.values)
        .filter(|(_, m)| m.inverse_general)
        .map(|(k, m)| format!("({}, {})", k.name, m.inverse.as_deref().expect("emitted")))
        .collect();
    list(&mut s, "casts", casts);
    list(&mut s, "sub_kinds", sub_kinds);
    list(&mut s, "grades", grades);
    list(&mut s, "inverses", inverses);
    Laws {
        invocation: s,
        equivariance,
        conjugation,
    }
}

/// The test file for an algebra: `prelude` makes the algebra's kinds visible (a `use`, or an
/// `algebra!` declaration and a `use`).
pub fn emit_law_tests(spec: &AlgebraSpec, laws: &Laws, cfg: &str, prelude: &str) -> String {
    format!(
        "// @generated by gax-regen from the `{name}` spec. Do not edit by hand.\n\
         //! The algebraic laws of `{name}`, proved exactly on symbolic coefficients\n\
         //! (docs/laws.md). The lists below are what the generator emitted for this algebra.\n\n\
         {cfg}\n\n\
         mod law_suite;\n\n\
         {prelude}\n\n\
         law_suite::law_suite! {{\n{}}}\n",
        laws.invocation,
        name = spec.name,
    )
}

/// The documentation table of an algebra's equivariance and conjugation factors: one row per
/// versor and product where the factor is the same for every pair of single-grade kinds, and
/// the exceptions listed.
pub fn factor_table(spec: &AlgebraSpec, laws: &Laws) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "### `{}`\n", spec.name);
    if laws.equivariance.is_empty() && laws.conjugation.is_empty() {
        let _ = writeln!(
            s,
            "No versor with at most {LAW_VERSOR} coefficients: the versor laws are not derived here.\n"
        );
        return s;
    }
    let _ = writeln!(s, "| versor `m` | law | factor `f` |\n|---|---|---|");
    let describe = |f: Option<Factor>| {
        f.map_or_else(
            || "none of `±‖m‖^2k` (k ≤ 4)".to_string(),
            |f| format!("`{}`", f.describe()),
        )
    };
    let mut groups: Vec<((String, BinOp), Vec<&Equivariance>)> = Vec::new();
    for e in &laws.equivariance {
        let key = (e.versor.clone(), e.op);
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, v)) => v.push(e),
            None => groups.push((key, vec![e])),
        }
    }
    for ((versor, op), rows) in &groups {
        let (_, _, sym) = trait_of(*op);
        let law = format!("`(m >> a) {sym} (m >> b) == f · (m >> (a {sym} b))`");
        // The most common factor, and the pairs that differ from it.
        let mut counts: Vec<(Option<Factor>, usize)> = Vec::new();
        for r in rows {
            match counts.iter_mut().find(|(f, _)| *f == r.factor) {
                Some((_, c)) => *c += 1,
                None => counts.push((r.factor, 1)),
            }
        }
        counts.sort_by_key(|c| std::cmp::Reverse(c.1));
        let common = counts[0].0;
        let _ = writeln!(
            s,
            "| `{versor}` | {law}, all pairs of kinds{} | {} |",
            if counts.len() > 1 {
                " except below"
            } else {
                ""
            },
            describe(common)
        );
        for r in rows.iter().filter(|r| r.factor != common) {
            let (a, b, _) = &r.kinds;
            let _ = writeln!(
                s,
                "| `{versor}` | the same, `a: {a}`, `b: {b}` | {} |",
                describe(r.factor)
            );
        }
    }
    for (v, x, f) in &laws.conjugation {
        let _ = writeln!(
            s,
            "| `{v}` | `conj_m(g).of(m >> x) == f · (m >> g.of(x))`, `g: {x}<({x},)>` | {} |",
            describe(*f)
        );
    }
    s.push('\n');
    s
}
