//! Rust code emission for an algebra.
//!
//! The output is one module of ordinary Rust: a struct per kind, the product traits of
//! `gax-core` implemented per pair of kinds (tier 1: straight-line, generic over slots), and
//! fused sandwich kernels for the versor kinds (tier 2: simplified symbolically).

use crate::cse;
use crate::poly::{Poly, Rational, Var};
use crate::slp::{Operand, Program, render};
use crate::spec::{AlgebraSpec, KindSpec};
use crate::symbolic::{self, SymMv};
use crate::table::{BinOp, UnOp, binop_support, binop_table, unop_support, unop_table};
use std::collections::BTreeSet;
use std::fmt::Write as _;

/// Emission settings.
#[derive(Clone, Debug)]
pub struct Config {
    /// Path of the `gax-core` crate as seen from the generated module, e.g. `::gax::core`.
    pub core: String,
    /// Emit the batch kernels (`gax::batch`), prefixed by this attribute (e.g.
    /// `#[cfg(feature = "batch")]`, or empty for unconditional); `None` omits them.
    pub batch: Option<String>,
    /// Emit the `check-units` assertions in the `Unit` kernels, prefixed by this attribute
    /// (e.g. `#[cfg(feature = "check-units")]`, or empty for unconditional); `None` omits them.
    pub check_units: Option<String>,
    /// Emit the GPU layout types (`{Kind}Gpu`, conversions to `GpuMat`, `bytemuck::Pod`),
    /// prefixed by this attribute (e.g. `#[cfg(feature = "bytemuck")]`); `None` omits them.
    pub gpu: Option<String>,
}

/// Summary statistics of an emitted algebra.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    /// Number of generated binary product impls.
    pub binary_impls: usize,
    /// Number of generated sandwich kernels.
    pub sandwich_impls: usize,
    /// Op counts of each sandwich kernel: `(versor, target, unit, value path, building the
    /// matrix of the map path)`.
    pub sandwich_costs: Vec<(String, String, bool, crate::slp::Cost, crate::slp::Cost)>,
    /// Every generated binary impl: `(op, left, right)`.
    pub binary: Vec<(BinOp, String, String)>,
    /// Every generated binary impl with its output kind: `(op, left, right, output)`.
    pub products: Vec<(BinOp, String, String, String)>,
    /// Every generated unary impl: `(op, kind, output)`.
    pub unary: Vec<(UnOp, String, String)>,
    /// Every sandwich kernel: `(versor, target, unit, output)`.
    pub sandwiches: Vec<(String, String, bool, String)>,
    /// Every outermorphism: `(base kind, extended kind, whether the extended kind is the top)`.
    pub outermorphisms: Vec<(String, String, bool)>,
    /// The value methods emitted per kind.
    pub values: Vec<crate::emit_values::ValueMethods>,
    /// The sandwich kernels (value and matrix paths) in language-neutral form, for the WGSL
    /// modules (see [`crate::kernel`]).
    pub kernels: Vec<crate::kernel::Kernel>,
}

/// The largest kind (in coefficients) that the WGSL modules have kernels for: every kind of the
/// standard algebras (CSTA's multivector has 64). Unused functions cost nothing at run time,
/// since WESL linking strips them (docs/shaders.md).
pub const WGSL_MAX: usize = 64;

/// Products with more terms than this are loops over a static table of their terms rather than
/// unrolled (ADR-034). Every product of the standard algebras is below (CSTA's largest has
/// 64 x 64); from 7D on, the full kinds' are not.
pub const UNROLL_MAX: usize = 4096;

/// Unrolled products and plain sandwich kernels with more terms than this are `#[inline]`
/// rather than `#[inline(always)]` (among the standard algebras only CSTA's multivector
/// product).
pub const INLINE_MAX: usize = 1024;

/// `static` tables of `(output, left, right, coefficient)` terms, grouped by output, as the
/// statements declaring them: `{name}` holds `(left, right, coefficient)`, indices as `u16`
/// and coefficients as `i8` where they fit (else `i64`), and output `o`'s terms are
/// `{name}[{name}_START[o]..{name}_START[o + 1]]`.
fn term_table(name: &str, terms: &[(usize, usize, usize, i64)], outputs: usize) -> String {
    let mut terms = terms.to_vec();
    terms.sort_unstable();
    let small = terms.iter().all(|t| i8::try_from(t.3).is_ok());
    let ty = if small { "i8" } else { "i64" };
    let mut start = vec![0usize; outputs + 1];
    for t in &terms {
        start[t.0 + 1] += 1;
    }
    for o in 0..outputs {
        start[o + 1] += start[o];
    }
    let mut s = format!(
        "        /// Where each output's terms start in `{name}` (and the last one ends).\n        static {name}_START: [u32; {}] = [",
        outputs + 1
    );
    for (k, x) in start.iter().enumerate() {
        if k % 16 == 0 {
            s.push_str("\n            ");
        } else {
            s.push(' ');
        }
        let _ = write!(s, "{x},");
    }
    let _ = write!(
        s,
        "\n        ];\n        /// `(left, right, coefficient)` of each term, by output.\n        static {name}: [(u16, u16, {ty}); {}] = [",
        terms.len()
    );
    for (k, (_, i, j, c)) in terms.iter().enumerate() {
        if k % 8 == 0 {
            s.push_str("\n            ");
        } else {
            s.push(' ');
        }
        let _ = write!(s, "({i}, {j}, {c}),");
    }
    s.push_str("\n        ];\n");
    s
}

/// A loop over the tables of [`term_table`]: `out[o] = Σ k · left[i] ⊗ right[j]`, each output
/// accumulated in a register; `mul` combines a left and a right coefficient and `scale` a
/// product and a coefficient `T`.
fn term_loop(name: &str, out: &str, zero: &str, product: &str, scale: &str) -> String {
    format!(
        "        for (o, out) in {out}.iter_mut().enumerate() {{\n            let mut acc = {zero};\n            for &(i, j, k) in &{name}[{name}_START[o] as usize..{name}_START[o + 1] as usize] {{\n                let (i, j) = (usize::from(i), usize::from(j));\n                let k = T::from_i64(i64::from(k));\n                acc = acc + {};\n            }}\n            *out = acc;\n        }}\n",
        scale.replace("{p}", product)
    )
}

/// Versors with more coefficients than this (the even kinds from 7D on) get plain sandwich
/// kernels: the two products `(v x) ~v` straight from the tables, without symbolic
/// simplification, and the map computed from the kernel's columns (ADR-034). Symbolic
/// simplification grows with the cube of the versor's size, and runs out of memory there.
pub const FUSED_MAX: usize = 32;

/// A plain sandwich is generated when its kernel has at most this many terms (products in
/// its two stages); beyond that (a 9D even versor on another, say), the sandwich is left to
/// `v * x * v.reverse()`.
pub const PLAIN_MAX_TERMS: usize = 1 << 16;

/// A sandwich is generated when its map has at most this many entries (output coefficients
/// times passenger coefficients): the map is a kind of its own, written out entry by entry.
/// Every standard algebra is far below (CSTA's largest map is 64 x 64); an 8D or 9D
/// multivector is not.
pub const MAP_MAX: usize = 1 << 14;

/// Emit the module source for an algebra.
#[allow(clippy::too_many_lines)]
pub fn emit(spec: &AlgebraSpec, cfg: &Config) -> (String, Stats) {
    let mut e = Emitter {
        spec,
        cfg,
        out: String::new(),
        stats: Stats::default(),
        helpers: BTreeSet::new(),
    };
    e.header();
    let phase = std::time::Instant::now();
    let values = crate::par::map(&spec.kinds, |k| {
        crate::par::timed(
            || format!("{} values of {}", spec.name, k.name),
            || crate::emit_values::value_methods(spec, k),
        )
    });
    for (k, (methods, meta)) in spec.kinds.iter().zip(values) {
        e.kind(k);
        e.out.push_str(&methods);
        value_traits(&k.name, &meta, &mut e.out);
        e.stats.values.push(meta);
    }
    for a in &spec.kinds {
        for b in &spec.kinds {
            let eq = if a.name == b.name { "True" } else { "False" };
            let _ = writeln!(
                e.out,
                "impl gx::KindEq<{}> for {} {{\n    type Out = gx::{eq};\n}}\n",
                b.name, a.name
            );
        }
    }
    kind_tables(spec, &mut e.out);
    crate::par::report(&format!("{} values", spec.name), phase);
    let phase = std::time::Instant::now();
    for op in BinOp::ALL {
        for a in &spec.kinds {
            for b in &spec.kinds {
                e.binary(op, a, b);
            }
        }
    }
    crate::par::report(&format!("{} products", spec.name), phase);
    let phase = std::time::Instant::now();
    e.divisions();
    e.embeddings();
    for k in &spec.kinds {
        for op in UnOp::ALL {
            e.unary(op, k);
        }
    }
    crate::par::report(
        &format!("{} divisions, embeddings, unary", spec.name),
        phase,
    );
    let phase = std::time::Instant::now();
    let jobs: Vec<(&KindSpec, &KindSpec, bool)> = spec
        .kinds
        .iter()
        .filter(|k| k.versor)
        .flat_map(|v| {
            spec.kinds
                .iter()
                .flat_map(move |x| [(v, x, false), (v, x, true)])
        })
        .collect();
    // Heaviest first: the cost grows with both sizes, and the unit kernels simplify further.
    let weight = |&(v, x, unit): &(&KindSpec, &KindSpec, bool)| {
        v.layout.len() * v.layout.len() * x.layout.len() * (1 + usize::from(unit))
    };
    // The sandwiches and the outermorphisms are independent: derived side by side.
    let (maths, (outer, pairs)) = crate::par::join(
        || {
            crate::par::map_heaviest_first(&jobs, weight, |&(v, x, unit)| {
                crate::par::timed(
                    || {
                        format!(
                            "{} sandwich {}{} >> {}",
                            spec.name,
                            if unit { "unit " } else { "" },
                            v.name,
                            x.name
                        )
                    },
                    || {
                        let out = sandwich_out(spec, v, x, unit)?;
                        if out.layout.len() * x.layout.len() > MAP_MAX {
                            None
                        } else if v.layout.len() > FUSED_MAX {
                            sandwich_plain(spec, v, x, unit)
                        } else {
                            sandwich_math(spec, v, x, unit)
                        }
                    },
                )
            })
        },
        || crate::emit_outer::outermorphisms(spec),
    );
    for (&(v, x, unit), math) in jobs.iter().zip(maths) {
        if let Some(math) = math {
            e.sandwich(v, x, unit, math);
        }
    }
    crate::par::report(
        &format!("{} sandwiches and outermorphisms", spec.name),
        phase,
    );
    e.out.push_str(&outer);
    e.stats.outermorphisms = pairs;
    for (alias, kind) in &spec.aliases {
        let _ = writeln!(
            e.out,
            "/// Alias of [`{kind}`].\npub type {alias}<S = (), T = f32> = {kind}<S, T>;\n"
        );
    }
    if let Some(gate) = &cfg.batch {
        let names = spec
            .kinds
            .iter()
            .map(|k| (&k.name, &k.name))
            .chain(spec.aliases.iter().map(|(a, k)| (a, k)));
        for (name, kind) in names {
            let _ = writeln!(
                e.out,
                "{gate}\n/// [`{kind}`] values in struct-of-arrays form, for the batch kernels.\npub type {name}Soa<E = f32> = gx::batch::Soa<{kind}, E>;\n"
            );
        }
    }
    if let Some(gate) = &cfg.gpu {
        e.gpu_types(gate);
    }
    (qualify(&e.out), e.stats)
}

/// Names of `gax-core` items that generated code refers to; a kind must not use them.
pub const RESERVED: &[&str] = &[
    "Slots",
    "Coef",
    "Real",
    "SlotArr",
    "Cat",
    "Kind",
    "Extensor",
    "Of",
    "Unit",
    "Elem",
    "SquareMap",
    "Endomorphism",
    "Form",
    "Pairing",
    "Prepare",
    "Prepared",
    "TraceFirst",
    "Outermorphism",
    "SplitLast",
    "Gp",
    "Wedge",
    "Vee",
    "Dot",
    "Lc",
    "Rc",
    "ScalarProduct",
    "Commutator",
    "Anticommutator",
    "Transform",
    "TransformInv",
    "Reverse",
    "Involute",
    "Conjugate",
    "Dual",
    "Undual",
    "DivBy",
    "Widen",
];

/// Qualify every bare reference to a core item with the private `gx` alias.
fn qualify(src: &str) -> String {
    let mut out = String::with_capacity(src.len() + src.len() / 8);
    let bytes = src.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_uppercase()
            && (i == 0
                || !(bytes[i - 1].is_ascii_alphanumeric()
                    || bytes[i - 1] == b'_'
                    || bytes[i - 1] == b':'
                    || bytes[i - 1] == b'"'))
        {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let word = &src[start..i];
            // Associated items named like core items (`type Kind = ...`, `Coef = T` in a
            // binding) stay bare.
            let rest = src[i..].trim_start();
            let before = out.trim_end();
            let binding = rest.starts_with('=')
                && !rest.starts_with("==")
                && (before.ends_with('<') || before.ends_with(','));
            if RESERVED.contains(&word) && !out.ends_with("type ") && !binding {
                out.push_str("gx::");
            }
            out.push_str(word);
        } else {
            out.push(c as char);
            i += 1;
        }
    }
    out
}

/// The symbolic work of one sandwich kernel `vk >> xk`: its output kind, the fused value
/// program, and the map path's matrix. Pure, so the kernels are computed in parallel.
struct SandwichMath {
    out: KindSpec,
    relations: Vec<Poly>,
    direct: Program,
    map: MapPath,
    /// The map's structurally nonzero entries, `(output, passenger)`.
    entries: Vec<(usize, usize)>,
    /// A plain sandwich's two products as term tables (see [`sandwich_plain`]).
    plain: Option<Plain>,
}

/// The two products of a plain sandwich, `m = v x` and `out = m ~v`, as tables of
/// `(output, left, right, coefficient)` terms, the size of `m`, and the blades where
/// `v ~v − 1` is not identically zero (for `check-units`).
struct Plain {
    first: Vec<(usize, usize, usize, i64)>,
    second: Vec<(usize, usize, usize, i64)>,
    mid: usize,
    norm: Vec<u32>,
}

/// How a sandwich's map (its matrix in the passenger) is computed.
enum MapPath {
    /// A straight-line program in the versor, one output per entry.
    Program(Program),
    /// At run time: the value kernel applied to each basis element of the passenger.
    Columns,
}

/// The kind of `v >> x` from the tables alone: every blade of `(v x) ~v`, only the passenger's
/// grades for a `Unit` versor. It contains the kind the symbolic path finds (modulo the unit
/// relations), and is it for plain sandwiches.
fn sandwich_out(spec: &AlgebraSpec, vk: &KindSpec, xk: &KindSpec, unit: bool) -> Option<KindSpec> {
    let alg = &spec.algebra;
    let mid = crate::table::Layout {
        blades: binop_support(alg, BinOp::Gp, &vk.layout, &xk.layout)
            .into_iter()
            .map(|m| (m, 1))
            .collect(),
    };
    let grades = xk.layout.grades();
    let support: BTreeSet<u32> = binop_support(alg, BinOp::Gp, &mid, &vk.layout)
        .into_iter()
        .filter(|m| !unit || grades.contains(&m.count_ones()))
        .collect();
    spec.kind_for_support(&support).cloned()
}

/// A plain sandwich (see [`FUSED_MAX`]): `(v x) ~v` as two products from the exact tables,
/// every output blade of the result's kind computed. For a `Unit` versor the result's kind
/// holds the passenger's grades, which a versor's sandwich keeps; it is homogeneous of
/// degree 2 in the versor, so drift scales it uniformly (ADR-020), with nothing to repair.
fn sandwich_plain(
    spec: &AlgebraSpec,
    vk: &KindSpec,
    xk: &KindSpec,
    unit: bool,
) -> Option<SandwichMath> {
    use crate::table::Layout;
    let alg = &spec.algebra;
    let nv = vk.layout.len();
    let out = sandwich_out(spec, vk, xk, unit)?;
    let relations = if unit {
        symbolic::unit_relations(alg, &symbolic::variables(&vk.layout, 0))
    } else {
        Vec::new()
    };
    if unit && relations.is_empty() {
        return None;
    }
    let full = |s: BTreeSet<u32>| Layout {
        blades: s.into_iter().map(|m| (m, 1)).collect(),
    };
    let mid = full(binop_support(alg, BinOp::Gp, &vk.layout, &xk.layout));
    let res = full(binop_support(alg, BinOp::Gp, &mid, &vk.layout));
    let first = binop_table(alg, BinOp::Gp, &vk.layout, &xk.layout, &mid);
    let rev: Vec<i64> = vk
        .layout
        .blades
        .iter()
        .map(|&(m, _)| if m.count_ones() % 4 >= 2 { -1 } else { 1 })
        .collect();
    let second: Vec<_> = binop_table(alg, BinOp::Gp, &mid, &vk.layout, &res)
        .into_iter()
        .filter_map(|t| {
            let (o, so) = out.layout.position(res.blades[t.o].0)?;
            Some((o, t.i, t.j, t.coef * rev[t.j] * so))
        })
        .collect();
    // The entries of the map: output `o` depends on passenger `i` through the middle blades.
    let mut through: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); mid.len()];
    for t in &first {
        through[t.o].insert(t.j);
    }
    let mut entries = BTreeSet::new();
    for &(o, m, _, _) in &second {
        for &i in &through[m] {
            entries.insert((o, i));
        }
    }
    if first.len() + second.len() > PLAIN_MAX_TERMS {
        return None;
    }
    let norm: Vec<u32> = {
        let v = symbolic::variables(&vk.layout, 0);
        let n = symbolic::binop(alg, BinOp::Gp, &v, &symbolic::unop(alg, UnOp::Reverse, &v));
        n.into_iter()
            .filter(|(m, p)| {
                if *m == 0 {
                    p != &Poly::constant(Rational::ONE)
                } else {
                    !p.is_zero()
                }
            })
            .map(|(m, _)| m)
            .collect()
    };
    let plain = Plain {
        first: first.iter().map(|t| (t.o, t.i, t.j, t.coef)).collect(),
        second: second.iter().map(|&(o, m, j, c)| (o, m, j, c)).collect(),
        mid: mid.len(),
        norm,
    };
    let mut b = cse::Builder::default();
    let var = |k: usize| Operand::Var(k as Var);
    let int = |c: i64| Rational::new(i128::from(c), 1);
    let mut sums: Vec<Vec<(Rational, Operand)>> = vec![Vec::new(); mid.len()];
    for t in &first {
        let p = b.mul(var(t.i), var(nv + t.j));
        sums[t.o].push((int(t.coef), p));
    }
    let middle: Vec<Operand> = sums.iter().map(|s| cse::emit_sum(&mut b, s)).collect();
    let mut sums: Vec<Vec<(Rational, Operand)>> = vec![Vec::new(); out.layout.len()];
    for &(o, m, j, c) in &second {
        let p = b.mul(middle[m], var(j));
        sums[o].push((int(c), p));
    }
    let outputs: Vec<Operand> = sums.iter().map(|s| cse::emit_sum(&mut b, s)).collect();
    let mut direct = b.prog;
    direct.outputs = outputs;
    direct.compact();
    Some(SandwichMath {
        out,
        relations,
        direct,
        map: MapPath::Columns,
        entries: entries.into_iter().collect(),
        plain: Some(plain),
    })
}

#[allow(clippy::too_many_lines)]
fn sandwich_math(
    spec: &AlgebraSpec,
    vk: &KindSpec,
    xk: &KindSpec,
    unit: bool,
) -> Option<SandwichMath> {
    let alg = &spec.algebra;
    let nv = vk.layout.len() as Var;
    let v = symbolic::variables(&vk.layout, 0);
    let x = symbolic::variables(&xk.layout, nv);
    let vx = symbolic::binop(alg, BinOp::Gp, &v, &x);
    let rev = symbolic::unop(alg, UnOp::Reverse, &v);
    let res: SymMv = symbolic::binop(alg, BinOp::Gp, &vx, &rev);
    let relations = if unit {
        symbolic::unit_relations(alg, &v)
    } else {
        Vec::new()
    };
    if unit && relations.is_empty() {
        return None;
    }
    // The output kind comes from the support modulo the relations (a unit versor's
    // sandwich keeps the passenger's grades); the kernels are compiled from the unreduced
    // polynomials, and the portfolio decides whether reducing them pays.
    let mut reduced = res.clone();
    for p in reduced.values_mut() {
        *p = cse::reduce_by_relations(p, &relations);
    }
    reduced.retain(|_, p| !p.is_zero());
    let support = symbolic::support(&reduced);
    if support.is_empty() {
        return None;
    }
    let out = spec
        .kind_for_support(&support)
        .expect("a full kind exists")
        .clone();
    let projected: SymMv = res
        .into_iter()
        .filter(|(m, _)| out.layout.position(*m).is_some())
        .collect();
    let coeffs = symbolic::to_coeffs(&out.layout, &projected).expect("support fits");
    let passengers: BTreeSet<Var> = (nv..nv + xk.layout.len() as Var).collect();

    // Value path: the whole formula, simplified jointly. For a `Unit` versor the simplified
    // formulas are made homogeneous again in the versor (ADR-020, drift), so that a drifted
    // versor (`v ~v = (1 + δ)²`) scales results uniformly instead of distorting them.
    let norm = symbolic::binop(alg, BinOp::Gp, &v, &rev)
        .get(&0)
        .cloned()
        .unwrap_or_else(Poly::zero);
    // Map path: the matrix M[o][i] = d out[o] / d x[i], a polynomial in v.
    let mut entries: Vec<(usize, usize, Poly)> = Vec::new();
    for (o, p) in coeffs.iter().enumerate() {
        for i in 0..xk.layout.len() {
            let xi = nv + i as Var;
            let mut c = Poly::zero();
            for (m, &k) in &p.0 {
                assert!(m.power(xi) <= 1, "sandwich must be linear in x");
                if m.contains(xi) {
                    let rest = m.div(&crate::poly::Monomial::var(xi)).expect("contains");
                    c.add_term(rest, k);
                }
            }
            if !c.is_zero() {
                entries.push((o, i, c));
            }
        }
    }
    // Entries that are constant under the unit condition (a unit motor's weight row is
    // exactly 1) are used as constants: they cost nothing and need no storage.
    let matrix_polys: Vec<Poly> = entries
        .iter()
        .map(|e| {
            let r = cse::reduce_by_relations(&e.2, &relations);
            if r.as_constant().is_some() {
                r
            } else {
                e.2.clone()
            }
        })
        .collect();
    // The value path and the map path are independent: compiled side by side.
    let (direct, matrix) = crate::par::join(
        || {
            let direct = crate::par::timed(
                || format!("{} {}>>{} value path", spec.name, vk.name, xk.name),
                || cse::compile_best(&coeffs, &passengers, &relations),
            );
            let direct = if unit {
                let h = homogeneous(&direct, nv, &norm, &passengers);
                assert!(
                    check_equal(&h, &coeffs, &relations) && is_homogeneous(&h, nv, 2),
                    "the drift-tolerant {}>>{} kernel must equal the sandwich and be homogeneous",
                    vk.name,
                    xk.name
                );
                h
            } else {
                direct
            };
            debug_assert!(check_equal(&direct, &coeffs, &relations));
            direct
        },
        || {
            let matrix = crate::par::timed(
                || format!("{} {}>>{} map path", spec.name, vk.name, xk.name),
                || cse::compile_best(&matrix_polys, &BTreeSet::new(), &relations),
            );
            if unit {
                let h = homogeneous(&matrix, nv, &norm, &BTreeSet::new());
                assert!(
                    check_equal(&h, &matrix_polys, &relations) && is_homogeneous(&h, nv, 2),
                    "the drift-tolerant {}>>{} matrix must equal the sandwich's and be homogeneous",
                    vk.name,
                    xk.name
                );
                h
            } else {
                matrix
            }
        },
    );

    Some(SandwichMath {
        out,
        relations,
        direct,
        map: MapPath::Program(matrix),
        entries: entries.into_iter().map(|e| (e.0, e.1)).collect(),
        plain: None,
    })
}

struct Emitter<'a> {
    spec: &'a AlgebraSpec,
    cfg: &'a Config,
    out: String,
    stats: Stats,
    /// The plain sandwich kernels emitted so far, by name.
    helpers: BTreeSet<String>,
}

/// Rust identifier for a blade accessor (`1` becomes `s`).
fn blade_ident(b: &str) -> String {
    if b == "1" { "s".into() } else { b.to_string() }
}

impl Emitter<'_> {
    fn w(&mut self, s: &str) {
        self.out.push_str(s);
    }

    fn header(&mut self) {
        let core = &self.cfg.core;
        let spec = self.spec;
        let alg = &spec.algebra;
        let mut basis = String::new();
        for (i, c) in alg.basis_suffixes().iter().enumerate() {
            let row: Vec<String> = alg.metric()[i].iter().map(ToString::to_string).collect();
            let _ = writeln!(basis, "//   e{c}: [{}]", row.join(", "));
        }
        let _ = write!(
            self.out,
            "// @generated by gax-gen from the `{}` declaration. Do not edit by hand.\n\
             // Metric (rows e_i . e_j):\n{basis}\n\
             use {core} as gx;\n\n",
            spec.name
        );
    }

    #[allow(clippy::too_many_lines)]
    fn kind(&mut self, k: &KindSpec) {
        let name = &k.name;
        let n = k.layout.len();
        let core = self.cfg.core.clone();
        let scalar = self.spec.scalar_kind().name.clone();
        let blades_list: Vec<String> = k.blades.iter().map(|b| format!("{b:?}")).collect();
        let doc = if k.doc.is_empty() {
            format!("A `{name}` of the `{}` algebra.", self.spec.name)
        } else {
            k.doc.clone()
        };
        let blade_doc = k.blades.join(", ");
        // Spelled out rather than `core::array::from_fn`: batch kernels need these inlined
        // (see `gax::batch`), and the core helpers are not always.
        let each = |f: &dyn Fn(usize) -> String| (0..n).map(f).collect::<Vec<_>>().join(", ");
        let arr_ff = each(&|i| format!("f({i})"));
        let arr_fm = each(&|i| format!("f(&a[{i}])"));
        let arr_fz = each(&|i| format!("f(&a[{i}], &b[{i}])"));
        let s = format!(
            r#"#[doc = {doc:?}]
///
/// Blades, in coefficient order: `[{blade_doc}]`.
///
/// `S` lists the open slots: `{name}` (that is, `{name}<()>`) is a value, `{name}<(A,)>` a
/// linear map from `A`, `{name}<(A, B)>` a bilinear map. `T` is the coefficient type.
#[repr(C)]
pub struct {name}<S: Slots = (), T: Coef = f32> {{
    /// Coefficients, output first: `c[i]` is the slot array of blade `i`.
    pub c: [S::Arr<T>; {n}],
}}

impl<S: Slots, T: Coef> Clone for {name}<S, T> {{
    #[inline(always)]
    fn clone(&self) -> Self {{
        *self
    }}
}}

impl<S: Slots, T: Coef> Copy for {name}<S, T> {{}}

impl<S: Slots, T: Coef> PartialEq for {name}<S, T> {{
    fn eq(&self, other: &Self) -> bool {{
        self.c == other.c
    }}
}}

impl<S: Slots, T: Coef> core::fmt::Debug for {name}<S, T> {{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {{
        let mut d = f.debug_struct("{name}");
        for (b, c) in <{name} as Kind>::BLADES.iter().zip(self.c.iter()) {{
            d.field(b, c);
        }}
        d.finish()
    }}
}}

impl Kind for {name} {{
    const N: usize = {n};
    const NAME: &'static str = "{name}";
    const MODULE: &'static str = module_path!();
    const BLADES: &'static [&'static str] = &[{blades}];
    type Arr<X: gx::Elem> = [X; {n}];
    type Mv<S: Slots, T: Coef> = {name}<S, T>;
    type Scalar = {scalar};
    #[inline(always)]
    fn arr_from_fn<X: gx::Elem>(mut f: impl FnMut(usize) -> X) -> [X; {n}] {{
        [{arr_ff}]
    }}
    #[inline(always)]
    fn arr_map<X: gx::Elem, Y: gx::Elem>(a: &[X; {n}], mut f: impl FnMut(&X) -> Y) -> [Y; {n}] {{
        [{arr_fm}]
    }}
    #[inline(always)]
    fn arr_zip<X: gx::Elem, Y: gx::Elem, Z: gx::Elem>(
        a: &[X; {n}],
        b: &[Y; {n}],
        mut f: impl FnMut(&X, &Y) -> Z,
    ) -> [Z; {n}] {{
        [{arr_fz}]
    }}
}}

impl<S: Slots, T: Coef> Extensor for {name}<S, T> {{
    type Kind = {name};
    type Slots = S;
    type Coef = T;
    #[inline(always)]
    fn from_coeffs(c: [S::Arr<T>; {n}]) -> Self {{
        {name} {{ c }}
    }}
    #[inline(always)]
    fn coeffs(&self) -> &[S::Arr<T>; {n}] {{
        &self.c
    }}
}}

impl<S: Slots, T: Coef> {name}<S, T> {{
    /// Construct from output-first coefficients.
    #[inline(always)]
    pub const fn from_coeffs(c: [S::Arr<T>; {n}]) -> Self {{
        {name} {{ c }}
    }}

    /// All coefficients zero.
    #[inline(always)]
    pub fn zero() -> Self {{
        {name} {{ c: [S::from_flat(&mut |_| T::zero(), 0); {n}] }}
    }}

    /// Fill the first open slot with a value, or compose a map into it.
    #[inline(always)]
    pub fn of<X>(self, x: X) -> <Self as Of<X>>::Output
    where
        Self: Of<X>,
    {{
        Of::of(self, x)
    }}

    /// This value or map as a `K`: the blades they share kept, `K`'s other blades zero (a
    /// projection, an embedding, or both; on maps and forms, of the output).
    #[inline(always)]
    pub fn cast<K: gx::Kind>(self) -> K::Mv<S, T>
    where
        {name}: gx::Cast<K>,
    {{
        gx::cast::cast::<Self, K>(&self)
    }}

    /// The grade-`G` part, as the declared kind that holds it (on maps and forms, of the
    /// output).
    #[inline(always)]
    pub fn grade<const G: usize>(self) -> <<{name} as gx::GradePart<G>>::Out as gx::Kind>::Mv<S, T>
    where
        {name}: gx::GradePart<G>,
    {{
        gx::cast::grade::<Self, G>(&self)
    }}

    /// Least squares: the least-norm `x` of the first slot's kind minimizing
    /// `‖self.of(x) − rhs‖` (coefficient norms). For a one-slot map `rhs` may have slots, which
    /// `x` keeps; for more slots `rhs` has exactly the remaining ones.
    #[inline]
    pub fn lstsq<X>(self, rhs: X) -> <Self as gx::LeastSquares<X>>::Solution
    where
        Self: gx::LeastSquares<X>,
    {{
        gx::LeastSquares::lstsq(self, rhs)
    }}

    /// [`Self::lstsq`] with singular values below `rcond` times the largest treated as zero.
    #[inline]
    pub fn lstsq_with<X>(self, rhs: X, rcond: T) -> <Self as gx::LeastSquares<X>>::Solution
    where
        Self: gx::LeastSquares<X, Coef = T>,
    {{
        gx::LeastSquares::lstsq_with(self, rhs, rcond)
    }}

    /// Move open slot `I` to the front, so that `.of(x)` fills it: `m.at::<1>().of(x)`.
    #[inline(always)]
    pub fn at<const I: usize>(self) -> {name}<<S as gx::MoveToFront<I>>::Moved, T>
    where
        S: gx::MoveToFront<I>,
    {{
        {name} {{ c: self.c.map(|col| <S as gx::MoveToFront<I>>::move_arr(&col)) }}
    }}

    /// Contract the output with open slot `I` (which must be of kind `{name}`): a trace with no
    /// metric, leaving a scalar with the other slots (numga's `trace(slot)`).
    #[inline(always)]
    pub fn trace_at<const I: usize>(self) -> <{name}<<S as gx::MoveToFront<I>>::Moved, T> as gx::TraceFirst>::Output
    where
        S: gx::MoveToFront<I>,
        {name}<<S as gx::MoveToFront<I>>::Moved, T>: gx::TraceFirst,
    {{
        gx::TraceFirst::trace_first(self.at::<I>())
    }}

    /// The outermorphism of this map on vectors (or antivectors) to the kind `B`:
    /// `m.outermorphism::<Line>().of(a ^ b) == m.of(a) ^ m.of(b)` (with `&` for antivectors).
    #[inline(always)]
    pub fn outermorphism<B>(self) -> <Self as gx::Outermorphism<B>>::Output
    where
        Self: gx::Outermorphism<B>,
    {{
        gx::Outermorphism::outermorphism(self)
    }}

    /// Fill every open slot of `x`'s kind with the value `x`; the other slots stay open.
    #[inline(always)]
    pub fn fill<X>(self, x: X) -> {name}<<S as gx::FillList<X::Kind>>::Out, T>
    where
        X: Extensor<Slots = (), Coef = T>,
        S: gx::FillList<X::Kind>,
    {{
        let xc = x.coeffs();
        {name} {{ c: self.c.map(|col| <S as gx::FillList<X::Kind>>::fill(&col, xc)) }}
    }}
"#,
            blades = blades_list.join(", "),
        );
        self.w(&s);
        for (tr, m) in [
            ("Gp", "gp"),
            ("Wedge", "wedge"),
            ("Vee", "vee"),
            ("Dot", "dot"),
            ("Lc", "lc"),
            ("Rc", "rc"),
            ("ScalarProduct", "scalar_product"),
            ("Commutator", "commutator"),
            ("Anticommutator", "anticommutator"),
            ("Transform", "transform"),
            ("TransformInv", "transform_inv"),
        ] {
            let _ = write!(
                self.out,
                "\n    /// See [`{tr}`].\n    #[inline(always)]\n    pub fn {m}<R>(self, rhs: R) -> <Self as {tr}<R>>::Output\n    where\n        Self: {tr}<R>,\n    {{\n        {tr}::{m}(self, rhs)\n    }}\n"
            );
        }
        for (tr, m) in [
            ("Reverse", "reverse"),
            ("Involute", "involute"),
            ("Conjugate", "conjugate"),
            ("Dual", "dual"),
            ("Undual", "undual"),
        ] {
            let _ = write!(
                self.out,
                "\n    /// See [`{tr}`].\n    #[inline(always)]\n    pub fn {m}(self) -> <Self as {tr}>::Output {{\n        {tr}::{m}(self)\n    }}\n"
            );
        }
        self.w("}\n\n");

        // Value constructors and accessors.
        let params: Vec<String> = k
            .blades
            .iter()
            .map(|b| format!("{}: T", blade_ident(b)))
            .collect();
        let args: Vec<String> = k.blades.iter().map(|b| blade_ident(b)).collect();
        let _ = write!(
            self.out,
            "impl<T: Coef> {name}<(), T> {{\n    /// A value from its coefficients, in blade order.\n    #[inline(always)]\n    #[allow(clippy::too_many_arguments)]\n    pub const fn new({}) -> Self {{\n        {name} {{ c: [{}] }}\n    }}\n\n    /// Prepare this versor's action on kind `X` for applying it to many objects (see
    /// [`gx::Prepared`]).
    #[inline(always)]
    pub fn prepare<X>(self) -> <Self as gx::Prepare<X>>::Output
    where
        Self: gx::Prepare<X>,
    {{
        gx::Prepare::prepare(self)
    }}

    /// The identity map on `{name}`: a `{name}` with one open `{name}` slot.\n    #[inline(always)]\n    pub fn slot() -> {name}<({name},), T> {{\n        {name} {{ c: core::array::from_fn(|i| core::array::from_fn(|j| if i == j {{ T::one() }} else {{ T::zero() }})) }}\n    }}\n",
            params.join(", "),
            args.join(", ")
        );
        for (i, b) in k.blades.iter().enumerate() {
            let _ = write!(
                self.out,
                "\n    /// Coefficient of `{b}`.\n    #[inline(always)]\n    pub fn {}(&self) -> T {{\n        self.c[{i}]\n    }}\n",
                blade_ident(b)
            );
        }
        self.w("}\n\n");

        // Building a map from the images of its input's basis blades.
        let _ = write!(
            self.out,
            r"impl<A: Kind, T: Coef> {name}<(A,), T> {{
    /// The linear map that sends each basis blade of `A`, in `A`'s layout order, to the given
    /// `{name}`. `A` may be a kind of another algebra (a projection from PGA3D points to PGA2D
    /// points is a `pga2d::Point<(pga3d::Point,)>`). The coefficients of a map are stored
    /// output first: `from_coeffs` takes rows, `c[o][i]` the coefficient `o` of the image of
    /// the input blade `i`.
    #[inline]
    pub fn from_images(images: A::Arr<{name}<(), T>>) -> Self {{
        let images = images.as_ref();
        {name} {{ c: core::array::from_fn(|o| A::arr_from_fn(|i| images[i].c[o])) }}
    }}
}}

"
        );

        // Methods of maps and forms (see `gax_core::extensor`).
        let _ = write!(
            self.out,
            r"impl<A: Kind, T: Real> {name}<(A,), T> {{
    /// The inverse map, `A <- {name}`.
    #[inline]
    pub fn inverse(self) -> <Self as SquareMap>::Inverse
    where
        Self: SquareMap<Coef = T, Kind = {name}, Input = A>,
    {{
        SquareMap::inverse(self)
    }}

    /// The determinant of the map's coefficient matrix.
    #[inline]
    pub fn det(self) -> T
    where
        Self: SquareMap<Coef = T, Kind = {name}, Input = A>,
    {{
        SquareMap::det(self)
    }}

    /// Solve `self.of(x) == rhs` for `x`; a right-hand side with slots keeps them.
    #[inline]
    pub fn solve<X>(self, rhs: X) -> <A as Kind>::Mv<X::Slots, T>
    where
        Self: SquareMap<Coef = T, Kind = {name}, Input = A>,
        X: Extensor<Kind = {name}, Coef = T>,
    {{
        SquareMap::solve(self, rhs)
    }}

    /// Singular value decomposition: `(u, sigma, v)` with `self.of(v[i]) == sigma[i] * u[i]`.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn svd(self) -> (<{name} as Kind>::Arr<{name}<(), T>>, <{name} as Kind>::Arr<T>, <A as Kind>::Arr<<A as Kind>::Mv<(), T>>)
    where
        Self: SquareMap<Coef = T, Kind = {name}, Input = A>,
    {{
        SquareMap::svd(self)
    }}

    /// The Moore–Penrose pseudo-inverse, `A <- {name}`, of a map of any shape: it sends `b` to
    /// the least-norm least-squares solution of `self.of(x) ≈ b`.
    #[inline]
    pub fn pinv(self) -> A::Mv<({name},), T>
    where
        Self: gx::PseudoInverse<Coef = T, Output = A::Mv<({name},), T>>,
    {{
        gx::PseudoInverse::pinv(self)
    }}

    /// [`Self::pinv`] with singular values below `rcond` times the largest treated as zero.
    #[inline]
    pub fn pinv_with(self, rcond: T) -> A::Mv<({name},), T>
    where
        Self: gx::PseudoInverse<Coef = T, Output = A::Mv<({name},), T>>,
    {{
        gx::PseudoInverse::pinv_with(self, rcond)
    }}

    /// The trace of a map from `{name}` to itself.
    #[inline]
    pub fn trace(self) -> T
    where
        Self: Endomorphism<Coef = T, Kind = {name}, Input = {name}>,
    {{
        Endomorphism::trace(self)
    }}
}}

impl<A: Kind, B: Kind, T: Coef> {name}<(A, B), T> {{
    /// Exchange the two slots.
    #[inline]
    pub fn swap(self) -> {name}<(B, A), T> {{
        self.at::<1>()
    }}
}}

impl<A: Kind, B: Kind, T: Real> {name}<(A, B), T> {{
    /// Solve `self(x, ·) == rhs(l, ·)` for `x`; leading slots of `rhs` become slots of `x`.
    #[inline]
    pub fn solve<R>(self, rhs: R) -> <A as Kind>::Mv<<R::Slots as SplitLast>::Init, T>
    where
        Self: Pairing<Coef = T, Kind = {name}, First = A, Second = B>,
        R: Extensor<Kind = {name}, Coef = T>,
        R::Slots: SplitLast<Last = B>,
    {{
        Pairing::solve(self, rhs)
    }}
}}

impl<A: Kind, T: Real> {name}<(A, A), T> {{
    /// Generalized symmetric eigenproblem against a positive definite metric form. Returns the
    /// eigenvalues (ascending) and the eigenvectors, as values of the slot kind.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn eigh_with(self, metric: Self) -> (<A as Kind>::Arr<T>, <A as Kind>::Arr<<A as Kind>::Mv<(), T>>)
    where
        Self: Form<Coef = T, Kind = {name}, Slot = A>,
    {{
        Form::eigh_with(self, metric)
    }}

    /// Symmetric eigenproblem in the coefficient basis (identity metric).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn eigh(self) -> (<A as Kind>::Arr<T>, <A as Kind>::Arr<<A as Kind>::Mv<(), T>>)
    where
        Self: Form<Coef = T, Kind = {name}, Slot = A>,
    {{
        Form::eigh(self)
    }}
}}

"
        );

        // Arithmetic.
        let arr = |op: &str| {
            format!(
                "{name} {{ c: core::array::from_fn(|i| (SlotArr::<S, T>(self.c[i]) {op} SlotArr::<S, T>(rhs.c[i])).0) }}"
            )
        };
        let _ = write!(
            self.out,
            r"impl<S: Slots, T: Coef> core::ops::Add for {name}<S, T> {{
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {{
        {add}
    }}
}}

impl<S: Slots, T: Coef> core::ops::Sub for {name}<S, T> {{
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {{
        {sub}
    }}
}}

impl<S: Slots, T: Coef> core::ops::AddAssign for {name}<S, T> {{
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {{
        *self = *self + rhs;
    }}
}}

impl<S: Slots, T: Coef> core::ops::SubAssign for {name}<S, T> {{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {{
        *self = *self - rhs;
    }}
}}

impl<S: Slots, T: Coef> core::ops::Neg for {name}<S, T> {{
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {{
        {name} {{ c: self.c.map(|x| (-SlotArr::<S, T>(x)).0) }}
    }}
}}

impl<S: Slots, T: Coef> Gp<T> for {name}<S, T> {{
    type Output = Self;
    #[inline(always)]
    fn gp(self, rhs: T) -> Self {{
        {name} {{ c: self.c.map(|x| SlotArr::<S, T>(x).scale(rhs).0) }}
    }}
}}

impl<S: Slots, T: Real> DivBy<T> for {name}<S, T> {{
    type Output = Self;
    #[inline(always)]
    fn div_by(self, rhs: T) -> Self {{
        let r = rhs.recip();
        {name} {{ c: self.c.map(|x| SlotArr::<S, T>(x).scale(r).0) }}
    }}
}}

impl<S: Slots, T: Coef, R> core::ops::Div<R> for {name}<S, T>
where
    Self: DivBy<R>,
{{
    type Output = <Self as DivBy<R>>::Output;
    #[inline(always)]
    fn div(self, rhs: R) -> Self::Output {{
        DivBy::div_by(self, rhs)
    }}
}}

impl<S: Slots> core::ops::Mul<{name}<S, f32>> for f32 {{
    type Output = {name}<S, f32>;
    #[inline(always)]
    fn mul(self, rhs: {name}<S, f32>) -> {name}<S, f32> {{
        rhs.gp(self)
    }}
}}

impl<S: Slots> core::ops::Mul<{name}<S, f64>> for f64 {{
    type Output = {name}<S, f64>;
    #[inline(always)]
    fn mul(self, rhs: {name}<S, f64>) -> {name}<S, f64> {{
        rhs.gp(self)
    }}
}}

",
            add = arr("+"),
            sub = arr("-"),
        );
        for (std_tr, std_m, tr, m) in [
            ("Mul", "mul", "Gp", "gp"),
            ("BitXor", "bitxor", "Wedge", "wedge"),
            ("BitAnd", "bitand", "Vee", "vee"),
            ("BitOr", "bitor", "Dot", "dot"),
            ("Shr", "shr", "Transform", "transform"),
            ("Shl", "shl", "TransformInv", "transform_inv"),
        ] {
            let _ = write!(
                self.out,
                "impl<S: Slots, T: Coef, R> core::ops::{std_tr}<R> for {name}<S, T>\nwhere\n    Self: {tr}<R>,\n{{\n    type Output = <Self as {tr}<R>>::Output;\n    #[inline(always)]\n    fn {std_m}(self, rhs: R) -> Self::Output {{\n        {tr}::{m}(self, rhs)\n    }}\n}}\n\n"
            );
        }
        let _ = core;
    }

    /// `a / b = a b⁻¹` for every value `b` with a closed-form inverse whose product with `a`
    /// exists (after the binary products, whose list it reads).
    fn divisions(&mut self) {
        let mut body = String::new();
        for (b, meta) in self.spec.kinds.iter().zip(&self.stats.values) {
            let Some(inv) = &meta.inverse else { continue };
            for a in &self.spec.kinds {
                let exists = self
                    .stats
                    .products
                    .iter()
                    .any(|(op, x, y, _)| *op == BinOp::Gp && *x == a.name && y == inv);
                if !exists {
                    continue;
                }
                let (an, bn) = (&a.name, &b.name);
                let _ = write!(
                    body,
                    "impl<S: Slots, T: Real> DivBy<{bn}<(), T>> for {an}<S, T> {{\n    type Output = <Self as Gp<{inv}<(), T>>>::Output;\n    #[inline(always)]\n    fn div_by(self, rhs: {bn}<(), T>) -> Self::Output {{\n        Gp::gp(self, rhs.inverse())\n    }}\n}}\n\n"
                );
                // A unit versor's inverse is its reverse: no arithmetic.
                if b.versor && *inv == b.name {
                    let _ = write!(
                        body,
                        "impl<S: Slots, T: Coef> DivBy<Unit<{bn}<(), T>>> for {an}<S, T> {{\n    type Output = <Self as Gp<{bn}<(), T>>>::Output;\n    /// `self ~rhs`: a unit versor's inverse is its reverse.\n    #[inline(always)]\n    fn div_by(self, rhs: Unit<{bn}<(), T>>) -> Self::Output {{\n        Gp::gp(self, rhs.into_inner().reverse())\n    }}\n}}\n\n"
                    );
                }
            }
        }
        self.w(&body);
    }

    /// `From` for every kind whose blades are among another's: the same multivector as the
    /// larger kind (a rotor as a motor), with the orientations of the target's blades.
    fn embeddings(&mut self) {
        let mut body = String::new();
        for a in &self.spec.kinds {
            for b in &self.spec.kinds {
                if a.name == b.name {
                    continue;
                }
                let place: Option<Vec<Option<(usize, bool)>>> = {
                    let within = a
                        .layout
                        .blades
                        .iter()
                        .all(|(m, _)| b.layout.blades.iter().any(|(n, _)| n == m));
                    within.then(|| {
                        b.layout
                            .blades
                            .iter()
                            .map(|(n, sb)| {
                                a.layout
                                    .blades
                                    .iter()
                                    .position(|(m, _)| m == n)
                                    .map(|j| (j, a.layout.blades[j].1 != *sb))
                            })
                            .collect()
                    })
                };
                let Some(place) = place else { continue };
                let exprs: Vec<String> = place
                    .iter()
                    .map(|p| match p {
                        None => "<S as Slots>::from_flat(&mut |_| T::zero(), 0)".to_string(),
                        Some((j, false)) => format!("x.c[{j}]"),
                        Some((j, true)) => format!("(-SlotArr::<S, T>(x.c[{j}])).0"),
                    })
                    .collect();
                let (an, bn) = (&a.name, &b.name);
                let _ = write!(
                    body,
                    "impl<S: Slots, T: Coef> From<{an}<S, T>> for {bn}<S, T> {{\n    /// The same multivector as a [`{bn}`].\n    #[inline(always)]\n    fn from(x: {an}<S, T>) -> Self {{\n        {bn} {{ c: [{}] }}\n    }}\n}}\n\nimpl<T: Coef> gx::Widen<{bn}<(), T>> for {an}<(), T> {{}}\n\n",
                    exprs.join(", ")
                );
            }
        }
        self.w(&body);
    }

    fn binary(&mut self, op: BinOp, a: &KindSpec, b: &KindSpec) {
        let alg = &self.spec.algebra;
        let support = binop_support(alg, op, &a.layout, &b.layout);
        if support.is_empty() {
            return;
        }
        let out = self
            .spec
            .kind_for_support(&support)
            .expect("a full kind exists")
            .clone();
        let table = binop_table(alg, op, &a.layout, &b.layout, &out.layout);
        let (tr, m) = trait_of(op);
        let (an, bn, on) = (&a.name, &b.name, &out.name);
        let mut body = String::new();
        if table.len() > UNROLL_MAX {
            // A loop over a static table of the terms, by output: unrolled, one such product
            // is megabytes of code, and a debug build gives each of its temporaries a stack
            // slot (ADR-034).
            let terms: Vec<_> = table.iter().map(|t| (t.o, t.i, t.j, t.coef)).collect();
            let nout = out.layout.len();
            let _ = write!(
                body,
                "impl<S1: Slots, S2: Slots, T: Coef> {tr}<{bn}<S2, T>> for {an}<S1, T> {{\n    type Output = {on}<Cat<S1, S2>, T>;\n    #[inline]\n    fn {m}(self, rhs: {bn}<S2, T>) -> {on}<Cat<S1, S2>, T> {{\n{}        let a = self.c.map(SlotArr::<S1, T>);\n        let b = rhs.c.map(SlotArr::<S2, T>);\n        let zero = SlotArr::<Cat<S1, S2>, T>(<Cat<S1, S2> as Slots>::from_flat(&mut |_| T::zero(), 0));\n        let mut c = [zero; {nout}];\n{}        {on} {{ c: c.map(|x| x.0) }}\n    }}\n}}\n\n",
                term_table("TERMS", &terms, nout),
                term_loop("TERMS", "c", "zero", "a[i] * b[j]", "({p}).scale(k)"),
            );
            self.w(&body);
            self.stats.binary_impls += 1;
            self.stats.binary.push((op, a.name.clone(), b.name.clone()));
            self.stats
                .products
                .push((op, a.name.clone(), b.name.clone(), out.name.clone()));
            return;
        }
        let mut exprs = Vec::new();
        for o in 0..out.layout.len() {
            let terms: Vec<_> = table.iter().filter(|t| t.o == o).collect();
            if terms.is_empty() {
                exprs.push("<Cat<S1, S2> as Slots>::from_flat(&mut |_| T::zero(), 0)".to_string());
                continue;
            }
            let items: Vec<(bool, String)> = terms
                .iter()
                .map(|t| {
                    let prod = format!("p({}, {})", t.i, t.j);
                    let mag = t.coef.abs();
                    let term = if mag == 1 {
                        prod
                    } else {
                        format!("{prod}.scale(T::from_i64({mag}))")
                    };
                    (t.coef < 0, term)
                })
                .collect();
            let e = balanced_sum(&items);
            exprs.push(format!("({e}).0"));
        }
        // The product of coefficients as one closure: its operator (and the slot list of its
        // result, `Cat<S1, S2>`) is resolved once per impl instead of once per term, which
        // takes a third off type checking. `move`: it captures the two small arrays by copy,
        // which is cheaper to borrow-check than borrowing them.
        let ops = "        let p = move |i: usize, j: usize| a[i] * b[j];\n";
        // A large product is not forced inline: in a debug build every inlined copy keeps its
        // own stack slots (an exponential's loop holds several).
        let inline = if table.len() > INLINE_MAX {
            "#[inline]"
        } else {
            "#[inline(always)]"
        };
        let _ = write!(
            body,
            "impl<S1: Slots, S2: Slots, T: Coef> {tr}<{bn}<S2, T>> for {an}<S1, T> {{\n    type Output = {on}<Cat<S1, S2>, T>;\n    {inline}\n    fn {m}(self, rhs: {bn}<S2, T>) -> {on}<Cat<S1, S2>, T> {{\n        let a = self.c.map(SlotArr::<S1, T>);\n        let b = rhs.c.map(SlotArr::<S2, T>);\n{ops}        {on} {{\n            c: [\n"
        );
        for e in exprs {
            let _ = writeln!(body, "                {e},");
        }
        body.push_str("            ],\n        }\n    }\n}\n\n");
        self.w(&body);
        self.stats.binary_impls += 1;
        self.stats.binary.push((op, a.name.clone(), b.name.clone()));
        self.stats
            .products
            .push((op, a.name.clone(), b.name.clone(), out.name.clone()));
    }

    fn unary(&mut self, op: UnOp, k: &KindSpec) {
        let alg = &self.spec.algebra;
        let support = unop_support(alg, op, &k.layout);
        let out = self
            .spec
            .kind_for_support(&support)
            .expect("a full kind exists")
            .clone();
        let table = unop_table(alg, op, &k.layout, &out.layout);
        self.stats
            .unary
            .push((op, k.name.clone(), out.name.clone()));
        let (tr, m) = unop_trait(op);
        let (kn, on) = (&k.name, &out.name);
        let mut exprs = Vec::new();
        for o in 0..out.layout.len() {
            match table.iter().find(|t| t.o == o) {
                None => exprs.push("S::from_flat(&mut |_| T::zero(), 0)".to_string()),
                Some(t) if t.coef == 1 => exprs.push(format!("a[{}]", t.i)),
                Some(t) if t.coef == -1 => exprs.push(format!("(-SlotArr::<S, T>(a[{}])).0", t.i)),
                Some(t) => exprs.push(format!(
                    "SlotArr::<S, T>(a[{}]).scale(T::from_i64({})).0",
                    t.i, t.coef
                )),
            }
        }
        let _ = write!(
            self.out,
            "impl<S: Slots, T: Coef> {tr} for {kn}<S, T> {{\n    type Output = {on}<S, T>;\n    #[inline]\n    fn {m}(self) -> {on}<S, T> {{\n        let a = self.c;\n        {on} {{ c: [{}] }}\n    }}\n}}\n\n",
            exprs.join(", ")
        );
    }

    /// Fused `v x ~v` with `v` a value of versor kind `vk` and `x` of kind `xk` with any slots.
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::too_many_lines)]
    fn sandwich(&mut self, vk: &KindSpec, xk: &KindSpec, unit: bool, math: SandwichMath) {
        let SandwichMath {
            out,
            relations,
            direct,
            map,
            entries,
            plain,
        } = math;
        let nv = vk.layout.len() as Var;
        let (vn, xn, on) = (&vk.name, &xk.name, &out.name);
        self.record_sandwich(vk, xk, &out, unit, &direct, &map, &entries);
        // check-units: the parts of v ~v - 1, handed to Coef::check_unit.
        let check = match (&self.cfg.check_units, unit, &plain) {
            // A plain sandwich's versor is large: v ~v by the product (a table loop).
            (Some(gate), true, Some(plain)) => {
                let alg = &self.spec.algebra;
                let nk = self
                    .spec
                    .kind_for_support(&binop_support(alg, BinOp::Gp, &vk.layout, &vk.layout))
                    .expect("a full kind exists");
                let parts: Vec<String> = plain
                    .norm
                    .iter()
                    .map(|&m| {
                        let (pos, sign) = nk.layout.position(m).expect("in the product");
                        let neg = if sign < 0 { "-" } else { "" };
                        let one = if m == 0 { " - T::one()" } else { "" };
                        format!("{neg}n[{pos}]{one}")
                    })
                    .collect();
                format!(
                    "        {gate}\n        {{\n            let w = {vn}::<(), T>::from_coeffs(v);\n            let n = (w * w.reverse()).c;\n            T::check_unit(&[{}]);\n        }}\n",
                    parts.join(", ")
                )
            }
            (Some(gate), true, _) => {
                let prog = cse::compile_best(&relations, &BTreeSet::new(), &[]);
                let vname = |var: Var| format!("v[{var}]");
                let mut lets = String::new();
                prog.emit_lets(&vname, "u", &mut lets);
                let parts: Vec<String> = prog
                    .outputs
                    .iter()
                    .map(|o| render(o, &vname, "u"))
                    .collect();
                format!(
                    "        {gate}\n        {{\n{lets}            T::check_unit(&[{}]);\n        }}\n",
                    parts.join(", ")
                )
            }
            _ => String::new(),
        };
        let self_ty = if unit {
            format!("Unit<{vn}<(), T>>")
        } else {
            format!("{vn}<(), T>")
        };
        let vexpr = if unit {
            "self.into_inner().c"
        } else {
            "self.c"
        };
        let nxv = xk.layout.len();
        let nout = out.layout.len();
        let var_name = move |var: Var| {
            if var < nv {
                format!("v[{var}]")
            } else {
                format!("xv[{}]", var - nv)
            }
        };
        let mut s = String::new();
        let mut lets = String::new();
        direct.emit_lets(&var_name, "t", &mut lets);
        let outs: Vec<String> = direct
            .outputs
            .iter()
            .map(|o| render(o, &var_name, "t"))
            .collect();
        // A plain sandwich's kernel is a function of its own: the map applies it to each basis
        // element of the passenger.
        let helper = format!(
            "sandwich_{}_{}_{}",
            crate::kernel::snake(vn),
            crate::kernel::snake(xn),
            crate::kernel::snake(on)
        );
        let value = match map {
            MapPath::Program(_) => format!(
                "{lets}            return {on} {{ c: gx::slots::from_values::<S, T, {nout}>([{}]) }};\n",
                outs.join(", ")
            ),
            MapPath::Columns => {
                // The same for a `Unit` versor and a plain one when their results agree.
                if self.helpers.insert(helper.clone()) {
                    let doc = format!(
                        "/// `v x ~v` on coefficient arrays, for `{vn}` v and `{xn}` x, as `{on}`."
                    );
                    match &plain {
                        Some(t) if t.first.len() + t.second.len() > UNROLL_MAX => {
                            let _ = write!(
                                s,
                                "{doc}\n#[inline]\nfn {helper}<T: Coef>(v: &[T; {nv}], xv: &[T; {nxv}]) -> [T; {nout}] {{\n{}{}        let mut m = [T::zero(); {}];\n{}        let mut out = [T::zero(); {nout}];\n{}        out\n}}\n\n",
                                term_table("FIRST", &t.first, t.mid),
                                term_table("SECOND", &t.second, nout),
                                t.mid,
                                term_loop("FIRST", "m", "T::zero()", "v[i] * xv[j]", "{p} * k"),
                                term_loop("SECOND", "out", "T::zero()", "m[i] * v[j]", "{p} * k"),
                            );
                        }
                        _ => {
                            let inline = match &plain {
                                Some(t) if t.first.len() + t.second.len() > INLINE_MAX => {
                                    "#[inline]"
                                }
                                _ => "#[inline(always)]",
                            };
                            let _ = write!(
                                s,
                                "{doc}\n{inline}\nfn {helper}<T: Coef>(v: &[T; {nv}], xv: &[T; {nxv}]) -> [T; {nout}] {{\n{lets}    [{}]\n}}\n\n",
                                outs.join(", ")
                            );
                        }
                    }
                }
                format!(
                    "            return {on} {{ c: gx::slots::from_values::<S, T, {nout}>({helper}(&v, &xv)) }};\n"
                )
            }
        };
        let _ = write!(
            s,
            "impl<S: Slots, T: Coef> Transform<{xn}<S, T>> for {self_ty} {{\n    type Output = {on}<S, T>;\n    #[inline(always)]\n    fn transform(self, x: {xn}<S, T>) -> {on}<S, T> {{\n        let v = {vexpr};\n{check}        if let Some(xv) = gx::slots::values::<S, T, {nxv}>(&x.c) {{\n{value}        }}\n"
        );
        // The map's entries: exact constants, or values computed from `v` by `map_lets`.
        let mat_name = move |var: Var| format!("v[{var}]");
        let mut map_lets = String::new();
        let vals: Vec<Result<Rational, String>> = match &map {
            MapPath::Program(matrix) => {
                matrix.emit_lets(&mat_name, "m", &mut map_lets);
                matrix
                    .outputs
                    .iter()
                    .map(|o| match o {
                        Operand::Const(c) => Ok(*c),
                        other => Err(render(other, &mat_name, "m")),
                    })
                    .collect()
            }
            MapPath::Columns => {
                let _ = write!(
                    map_lets,
                    "        let mut cols = [[T::zero(); {nout}]; {nxv}];\n        let mut e = [T::zero(); {nxv}];\n        for (i, col) in cols.iter_mut().enumerate() {{\n            e[i] = T::one();\n            *col = {helper}(&v, &e);\n            e[i] = T::zero();\n        }}\n"
                );
                entries
                    .iter()
                    .map(|&(o, i)| Err(format!("cols[{i}][{o}]")))
                    .collect()
            }
        };
        let mut stored: Vec<&str> = Vec::new();
        for v in &vals {
            if let Err(e) = v
                && !stored.contains(&e.as_str())
            {
                stored.push(e);
            }
        }
        let stored_at = |e: &str| stored.iter().position(|s| *s == e).expect("stored");
        s.push_str(&map_lets);
        let _ = writeln!(s, "        let x = x.c.map(SlotArr::<S, T>);");
        let sum = |term: &dyn Fn(usize, &Result<Rational, String>) -> String| -> Vec<String> {
            (0..nout)
                .map(|o| {
                    let mut e = String::new();
                    for (k, &(_, i)) in entries.iter().enumerate().filter(|(_, e)| e.0 == o) {
                        let t = match &vals[k] {
                            Ok(c) if *c == Rational::ONE => format!("x[{i}]"),
                            Ok(c) if *c == -Rational::ONE => format!("-x[{i}]"),
                            v => term(i, v),
                        };
                        if e.is_empty() {
                            e = t;
                        } else if let Some(stripped) = t.strip_prefix('-') {
                            e = format!("{e} - {stripped}");
                        } else {
                            e = format!("{e} + {t}");
                        }
                    }
                    if e.is_empty() {
                        "S::from_flat(&mut |_| T::zero(), 0)".to_string()
                    } else {
                        format!("({e}).0")
                    }
                })
                .collect()
        };
        let constant = |c: Rational| render(&Operand::Const(c), &mat_name, "m");
        let cols = sum(&|i, v| match v {
            Ok(c) => format!("x[{i}].scale({})", constant(*c)),
            Err(e) => format!("x[{i}].scale({e})"),
        });
        let _ = write!(
            s,
            "        {on} {{ c: [{}] }}\n    }}\n}}\n\n",
            cols.join(", ")
        );
        // Inverse transport ~v x v = (~v) x ~(~v).
        let rev_self = if unit {
            "Unit::new_unchecked(self.into_inner().reverse())"
        } else {
            "self.reverse()"
        };
        let _ = write!(
            s,
            "impl<S: Slots, T: Coef> TransformInv<{xn}<S, T>> for {self_ty} {{\n    type Output = {on}<S, T>;\n    #[inline(always)]\n    fn transform_inv(self, x: {xn}<S, T>) -> {on}<S, T> {{\n        Transform::transform({rev_self}, x)\n    }}\n}}\n\n"
        );
        // Prepared action: the non-constant matrix entries, applied sparsely.
        let nst = stored.len();
        let vkind = if unit {
            format!("gx::Unit<{vn}>")
        } else {
            vn.clone()
        };
        let prep_ty = format!("gx::Prepared<{vkind}, {xn}, T, {nst}>");
        let _ = write!(
            s,
            "impl<T: Coef> gx::Prepare<{xn}> for {self_ty} {{\n    type Output = {prep_ty};\n    #[inline]\n    fn prepare(self) -> {prep_ty} {{\n        let v = {vexpr};\n{check}{map_lets}        gx::Prepared::from_entries([{}])\n    }}\n}}\n\n",
            stored.join(", ")
        );
        let cols = sum(&|i, v| match v {
            Ok(c) => format!("x[{i}].scale({})", constant(*c)),
            Err(e) => format!("x[{i}].scale(m[{}])", stored_at(e)),
        });
        let _ = write!(
            s,
            "impl<S: Slots, T: Coef> Transform<{xn}<S, T>> for {prep_ty} {{\n    type Output = {on}<S, T>;\n    #[inline(always)]\n    fn transform(self, x: {xn}<S, T>) -> {on}<S, T> {{\n        let m = self.m;\n        let x = x.c.map(SlotArr::<S, T>);\n        {on} {{ c: [{}] }}\n    }}\n}}\n\n",
            cols.join(", ")
        );
        // The dense map, written entry by entry (no multiplication by an identity's zeros).
        let mut rows = Vec::new();
        for o in 0..nout {
            let row: Vec<String> = (0..nxv)
                .map(
                    |i| match entries.iter().position(|e| e.0 == o && e.1 == i) {
                        None => "T::zero()".to_string(),
                        Some(k) => match &vals[k] {
                            Ok(c) => constant(*c),
                            Err(e) => format!("m[{}]", stored_at(e)),
                        },
                    },
                )
                .collect();
            rows.push(format!("[{}]", row.join(", ")));
        }
        let _ = write!(
            s,
            "impl<T: Coef> From<{prep_ty}> for {on}<({xn},), T> {{\n    /// The dense map of the prepared action.\n    #[inline]\n    fn from(p: {prep_ty}) -> Self {{\n        let m = p.m;\n        {on} {{ c: [{}] }}\n    }}\n}}\n\n",
            rows.join(", ")
        );
        if let Some(gate) = &self.cfg.batch {
            let mapped = (0..nst)
                .map(|i| format!("f(m[{i}])"))
                .collect::<Vec<_>>()
                .join(", ");
            let fparam = if nst == 0 { "_f" } else { "mut f" };
            let (wrap, unwrap, cert) = if unit {
                ("gx::Unit::new_unchecked(v)", "v.into_inner()", "Certified")
            } else {
                ("v", "v", "Plain")
            };
            let _ = write!(
                s,
                "{gate}\nimpl gx::batch::SandwichKernel<{xn}, gx::batch::{cert}> for {vn} {{\n    type Y = {on};\n    type Versor<T: gx::Coef> = {self_ty};\n    type Prepared<T: gx::Coef> = {prep_ty};\n    #[inline(always)]\n    fn wrap<T: gx::Coef>(v: {vn}<(), T>) -> {self_ty} {{\n        {wrap}\n    }}\n    #[inline(always)]\n    fn unwrap<T: gx::Coef>(v: {self_ty}) -> {vn}<(), T> {{\n        {unwrap}\n    }}\n    #[inline(always)]\n    fn prepare<T: gx::Coef>(v: {self_ty}) -> {prep_ty} {{\n        gx::Prepare::<{xn}>::prepare(v)\n    }}\n    #[inline(always)]\n    fn map_prepared<T: gx::Coef, W: gx::Coef>(p: {prep_ty}, {fparam}: impl FnMut(T) -> W) -> gx::Prepared<{vkind}, {xn}, W, {nst}> {{\n        let m = p.m;\n        gx::Prepared::from_entries([{mapped}])\n    }}\n    #[inline(always)]\n    fn apply_prepared<T: gx::Coef>(p: {prep_ty}, x: {xn}<(), T>) -> {on}<(), T> {{\n        gx::Transform::transform(p, x)\n    }}\n    #[inline(always)]\n    fn apply<T: gx::Coef>(v: {self_ty}, x: {xn}<(), T>) -> {on}<(), T> {{\n        gx::Transform::transform(v, x)\n    }}\n}}\n\n"
            );
        }
        self.w(&s);
        self.stats.sandwich_impls += 1;
        self.stats.sandwich_costs.push((
            vn.clone(),
            xn.clone(),
            unit,
            direct.cost(),
            match &map {
                MapPath::Program(matrix) => matrix.cost(),
                MapPath::Columns => {
                    let c = direct.cost();
                    crate::slp::Cost {
                        muls: c.muls * nxv,
                        adds: c.adds * nxv,
                        divs: c.divs * nxv,
                        negs: c.negs * nxv,
                        calls: c.calls * nxv,
                    }
                }
            },
        ));
        self.stats
            .sandwiches
            .push((vn.clone(), xn.clone(), unit, on.clone()));
    }
}

/// The program made homogeneous in the versor variables
/// `0..nv`: each monomial of lower degree than the highest is multiplied by a power of `norm`,
/// the scalar part of `v ~v`, which is 1 modulo the unit relations. The result is equal to the
/// program modulo the relations, and a uniform drift of the versor scales it uniformly.
impl Emitter<'_> {
    /// `{Kind}Gpu` per kind: the layout of the WGSL modules, `ceil(N/4)` `vec4<f32>`, with
    /// conversions, `bytemuck::Pod`, and compile-time layout assertions; and `GpuMat`
    /// conversions for maps between kinds of 3 or 4 coefficients (ADR-028).
    #[allow(clippy::too_many_lines)]
    fn gpu_types(&mut self, gate: &str) {
        let spec = self.spec;
        // The WGSL modules take kinds up to `WGSL_MAX` coefficients; larger ones (from 8D on)
        // have no GPU layout either.
        for k in spec.kinds.iter().filter(|k| k.layout.len() <= WGSL_MAX) {
            let (name, n) = (&k.name, k.layout.len());
            let m = n.div_ceil(4);
            let _ = write!(
                self.out,
                "{gate}
/// [`{name}`] in the GPU layout of the `gax::wgsl` modules: its {n} coefficients in blade
/// order, four per `vec4<f32>` field, zero-padded (the WGSL struct `{name}`).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct {name}Gpu {{
    /// The coefficients, `c[i / 4][i % 4]` for coefficient `i`.
    pub c: [[f32; 4]; {m}],
}}

// The WGSL layout of `struct {name} {{ c0: vec4<f32>, ... }}`: size 16 per field, align 16.
{gate}
const _: () = {{
    assert!(core::mem::size_of::<{name}Gpu>() == {size});
    assert!(core::mem::align_of::<{name}Gpu>() == 16);
    assert!(core::mem::offset_of!({name}Gpu, c) == 0);
}};

// SAFETY: `repr(C, align(16))` over `[[f32; 4]; {m}]` (size a multiple of 16): no padding
// bytes, and every bit pattern is a valid `f32`.
{gate}
unsafe impl gx::bytemuck::Zeroable for {name}Gpu {{}}
// SAFETY: as above.
{gate}
unsafe impl gx::bytemuck::Pod for {name}Gpu {{}}

{gate}
impl From<{name}<(), f32>> for {name}Gpu {{
    #[inline]
    fn from(x: {name}<(), f32>) -> Self {{
        let mut c = [[0.0; 4]; {m}];
        for (i, v) in x.c.iter().enumerate() {{
            c[i / 4][i % 4] = *v;
        }}
        {name}Gpu {{ c }}
    }}
}}

{gate}
impl From<gx::Unit<{name}<(), f32>>> for {name}Gpu {{
    #[inline]
    fn from(x: gx::Unit<{name}<(), f32>>) -> Self {{
        x.into_inner().into()
    }}
}}

{gate}
impl From<{name}Gpu> for {name}<(), f32> {{
    #[inline]
    fn from(g: {name}Gpu) -> Self {{
        {name}::from_coeffs(core::array::from_fn(|i| g.c[i / 4][i % 4]))
    }}
}}

/// [`{name}`] in the GPU layout of the `gax::wgsl` `f16` modules (`gax::{alg}_f16`): its {n}
/// coefficients as IEEE binary16 bit patterns, four per `vec4<f16>` field, zero-padded.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct {name}Gpu16 {{
    /// The coefficients' `f16` bits, `c[i / 4][i % 4]` for coefficient `i`.
    pub c: [[u16; 4]; {m}],
}}

// The WGSL layout of `struct {name} {{ c0: vec4<f16>, ... }}`: size 8 per field, align 8.
{gate}
const _: () = {{
    assert!(core::mem::size_of::<{name}Gpu16>() == {size16});
    assert!(core::mem::align_of::<{name}Gpu16>() == 8);
}};

// SAFETY: `repr(C, align(8))` over `[[u16; 4]; {m}]` (size a multiple of 8): no padding bytes,
// and every bit pattern is a valid `u16`.
{gate}
unsafe impl gx::bytemuck::Zeroable for {name}Gpu16 {{}}
// SAFETY: as above.
{gate}
unsafe impl gx::bytemuck::Pod for {name}Gpu16 {{}}

{gate}
impl From<{name}<(), f32>> for {name}Gpu16 {{
    /// Each coefficient rounded to the nearest `f16`.
    #[inline]
    fn from(x: {name}<(), f32>) -> Self {{
        let mut c = [[0; 4]; {m}];
        for (i, v) in x.c.iter().enumerate() {{
            c[i / 4][i % 4] = gx::gpu::f16_bits(*v);
        }}
        {name}Gpu16 {{ c }}
    }}
}}

{gate}
impl From<{name}Gpu16> for {name}<(), f32> {{
    #[inline]
    fn from(g: {name}Gpu16) -> Self {{
        {name}::from_coeffs(core::array::from_fn(|i| gx::gpu::f16_to_f32(g.c[i / 4][i % 4])))
    }}
}}

",
                size = 16 * m,
                size16 = 8 * m,
                alg = spec.name
            );
        }
        let small = |kind: &str| {
            spec.kinds
                .iter()
                .any(|k| k.name == kind && k.layout.len() <= WGSL_MAX)
        };
        for (alias, kind) in spec.aliases.iter().filter(|(_, k)| small(k)) {
            let _ = writeln!(
                self.out,
                "{gate}
/// Alias of [`{kind}Gpu`].
pub type {alias}Gpu = {kind}Gpu;

{gate}
/// Alias of [`{kind}Gpu16`].
pub type {alias}Gpu16 = {kind}Gpu16;
"
            );
        }
        let rows: Vec<String> = spec
            .kinds
            .iter()
            .filter(|k| k.layout.len() <= WGSL_MAX)
            .map(|k| {
                let t = format!("{}Gpu", k.name);
                format!(
                    "    (\"{}\", core::mem::size_of::<{t}>(), core::mem::align_of::<{t}>(), core::mem::offset_of!({t}, c), core::mem::size_of::<[f32; 4]>()),",
                    k.name
                )
            })
            .collect();
        let _ = writeln!(
            self.out,
            "{gate}\n/// The Rust layout of each `{{Kind}}Gpu`: `(kind, size, align, offset of c, stride of c)`,\n/// to check against a shader compiler's layout of the WGSL structs.\npub const GPU_LAYOUTS: &[(&str, usize, usize, usize, usize)] = &[\n{}\n];\n",
            rows.join("\n")
        );
        let rows16: Vec<String> = spec
            .kinds
            .iter()
            .filter(|k| k.layout.len() <= WGSL_MAX)
            .map(|k| {
                let t = format!("{}Gpu16", k.name);
                format!(
                    "    (\"{}\", core::mem::size_of::<{t}>(), core::mem::align_of::<{t}>(), core::mem::offset_of!({t}, c), core::mem::size_of::<[u16; 4]>()),",
                    k.name
                )
            })
            .collect();
        let _ = writeln!(
            self.out,
            "{gate}\n/// [`GPU_LAYOUTS`] for the `{{Kind}}Gpu16` types of the `f16` modules.\npub const GPU_LAYOUTS_F16: &[(&str, usize, usize, usize, usize)] = &[\n{}\n];\n",
            rows16.join("\n")
        );
        let small: Vec<&KindSpec> = spec
            .kinds
            .iter()
            .filter(|k| (3..=4).contains(&k.layout.len()))
            .collect();
        for x in &small {
            for y in &small {
                let (xn, yn, c) = (&x.name, &y.name, y.layout.len());
                let _ = write!(
                    self.out,
                    "{gate}
/// The map as a WGSL `mat{c}x{r}<f32>` (columns are inputs; transposes the output-first layout).
impl From<{xn}<({yn},), f32>> for gx::GpuMat<{c}> {{
    #[inline]
    fn from(m: {xn}<({yn},), f32>) -> Self {{
        let mut cols = [[0.0; 4]; {c}];
        for (o, row) in m.c.iter().enumerate() {{
            for (i, col) in cols.iter_mut().enumerate() {{
                col[o] = <({yn},) as gx::Slots>::get_flat(row, i);
            }}
        }}
        gx::GpuMat {{ cols }}
    }}
}}

{gate}
impl From<gx::GpuMat<{c}>> for {xn}<({yn},), f32> {{
    #[inline]
    fn from(g: gx::GpuMat<{c}>) -> Self {{
        {xn}::from_coeffs(core::array::from_fn(|o| {{
            <({yn},) as gx::Slots>::from_flat(&mut |i| g.cols[i][o], 0)
        }}))
    }}
}}

",
                    r = x.layout.len()
                );
            }
        }
    }

    /// Record the value and matrix paths of a sandwich for the WGSL modules.
    #[allow(clippy::too_many_arguments)]
    fn record_sandwich(
        &mut self,
        vk: &KindSpec,
        xk: &KindSpec,
        out: &KindSpec,
        unit: bool,
        direct: &Program,
        map: &MapPath,
        entries: &[(usize, usize)],
    ) {
        use crate::kernel::{Kernel, Source, Step, Ty, snake};
        let (nv, nx, nout) = (vk.layout.len(), xk.layout.len(), out.layout.len());
        if nv.max(nx).max(nout) > WGSL_MAX {
            return;
        }
        let u = if unit { "unit_" } else { "" };
        let unit_doc = if unit { "a unit " } else { "" };
        let arg = |param, index| Source::Arg { param, index };
        let vars: Vec<(Var, Source)> = (0..nv + nx)
            .map(|i| {
                let v = i as Var;
                (v, if i < nv { arg(0, i) } else { arg(1, i - nv) })
            })
            .collect();
        let (vs, xs) = (snake(&vk.name), snake(&xk.name));
        self.stats.kernels.push(Kernel {
            name: format!("{u}{vs}_sandwich_{xs}"),
            doc: format!(
                "`v x ~v` for {unit_doc}`{}` v and `{}` x (`v >> x`).",
                vk.name, xk.name
            ),
            params: vec![
                ("v".into(), Ty::Kind(vk.name.clone())),
                ("x".into(), Ty::Kind(xk.name.clone())),
            ],
            result: Ty::Kind(out.name.clone()),
            steps: vec![Step::Lets {
                prog: direct.clone(),
                prefix: "t".into(),
                vars,
            }],
            entries: None,
        });
        if let MapPath::Program(matrix) = map
            && (3..=4).contains(&nx)
            && (3..=4).contains(&nout)
        {
            self.stats.kernels.push(Kernel {
                name: format!("{u}{vs}_matrix_{xs}"),
                doc: format!(
                    "The matrix of `x -> v x ~v` on `{}` for {unit_doc}`{}` v (`v >> {}::slot()`); apply it as `m * x`.",
                    xk.name, vk.name, xk.name
                ),
                params: vec![("v".into(), Ty::Kind(vk.name.clone()))],
                result: Ty::Mat {
                    cols: nx,
                    rows: nout,
                },
                steps: vec![Step::Lets {
                    prog: matrix.clone(),
                    prefix: "m".into(),
                    vars: (0..nv).map(|i| (i as Var, arg(0, i))).collect(),
                }],
                entries: Some(entries.to_vec()),
            });
        }
    }
}

fn homogeneous(prog: &Program, nv: Var, norm: &Poly, passengers: &BTreeSet<Var>) -> Program {
    let recompiled = homogeneous_recompiled(prog, nv, norm, passengers);
    match cse::repair_degree(prog, nv, norm, 2) {
        Some(p) if p.cost().weight() <= recompiled.cost().weight() => p,
        _ => recompiled,
    }
}

/// The program's polynomials made homogeneous and compiled afresh.
fn homogeneous_recompiled(
    prog: &Program,
    nv: Var,
    norm: &Poly,
    passengers: &BTreeSet<Var>,
) -> Program {
    let polys = prog.to_polys();
    let degree = |m: &crate::poly::Monomial| m.0.iter().filter(|&&v| v < nv).count();
    // A sandwich is quadratic in the versor; reduction only lowers degrees.
    let top = polys
        .iter()
        .flat_map(|p| p.0.keys().map(degree))
        .max()
        .unwrap_or(0)
        .max(2);
    let mut out = Vec::with_capacity(polys.len());
    for p in &polys {
        let mut h = Poly::zero();
        for (m, &c) in &p.0 {
            let gap = top - degree(m);
            if gap % 2 == 1 {
                // Not reachable by powers of the (quadratic) norm: keep the program as it was.
                return prog.clone();
            }
            let mut term = Poly::term(m.clone(), c);
            for _ in 0..gap / 2 {
                term = &term * norm;
            }
            h = &h + &term;
        }
        out.push(h);
    }
    cse::compile_best(&out, passengers, &[])
}

/// Whether every output is a homogeneous polynomial of the given degree in the variables below
/// `nv` (or zero).
fn is_homogeneous(prog: &Program, nv: Var, degree: usize) -> bool {
    prog.to_polys().iter().all(|p| {
        p.0.keys()
            .all(|m| m.0.iter().filter(|&&v| v < nv).count() == degree)
    })
}

fn check_equal(prog: &Program, want: &[Poly], relations: &[Poly]) -> bool {
    let got = prog.to_polys();
    got.iter().zip(want).all(|(g, w)| {
        let d = g - w;
        d.is_zero() || cse::reduce_by_relations(&d, relations).is_zero()
    })
}

fn trait_of(op: BinOp) -> (&'static str, &'static str) {
    match op {
        BinOp::Gp => ("Gp", "gp"),
        BinOp::Wedge => ("Wedge", "wedge"),
        BinOp::Vee => ("Vee", "vee"),
        BinOp::Lc => ("Lc", "lc"),
        BinOp::Rc => ("Rc", "rc"),
        BinOp::Dot => ("Dot", "dot"),
        BinOp::Scalar => ("ScalarProduct", "scalar_product"),
        BinOp::Commutator => ("Commutator", "commutator"),
        BinOp::Anticommutator => ("Anticommutator", "anticommutator"),
    }
}

fn unop_trait(op: UnOp) -> (&'static str, &'static str) {
    match op {
        UnOp::Reverse => ("Reverse", "reverse"),
        UnOp::Involute => ("Involute", "involute"),
        UnOp::Conjugate => ("Conjugate", "conjugate"),
        UnOp::Dual => ("Dual", "dual"),
        UnOp::Undual => ("Undual", "undual"),
    }
}

/// Emit an integration test that checks every generated kernel against the dense oracle in
/// `gax/tests/common`. `module` is the path of the generated module (e.g. `gax::pga2d`).
#[allow(clippy::too_many_lines)]
pub fn emit_tests(spec: &AlgebraSpec, stats: &Stats, module: &str, spec_path: &str) -> String {
    let mut t = String::new();
    let _ = write!(
        t,
        "// @generated by gax-gen from the `{name}` declaration. Do not edit by hand.\n\
         //! Every generated kernel of `{name}`, checked against the dense oracle.\n\
         #![cfg_attr(rustfmt, rustfmt::skip)]\n\
         #![cfg(feature = \"{name}\")]\n\
         #![allow(clippy::pedantic, clippy::too_many_lines)]\n\n\
         mod common;\n\
         use common::{{assert_close, random, Oracle, Rng}};\n\
         use gax::{{Extensor, Unit}};\n\
         use gax_gen::table::{{BinOp, UnOp}};\n\
         use {module}::*;\n\n\
         const SPEC: &str = include_str!(\"{spec_path}\");\n\n\
         fn bin<A, B, R>(o: &Oracle, rng: &mut Rng, op: BinOp, f: impl Fn(A, B) -> R)\n\
         where\n    A: Extensor<Slots = (), Coef = f64>,\n    B: Extensor<Slots = (), Coef = f64>,\n    R: Extensor<Slots = (), Coef = f64>,\n{{\n\
         \x20   for _ in 0..8 {{\n\
         \x20       let (a, b): (A, B) = (random(rng), random(rng));\n\
         \x20       let want = o.binop(op, &o.dense(&a), &o.dense(&b));\n\
         \x20       assert_close(&o.dense(&f(a, b)), &want, &format!(\"{{op:?}} {{}} {{}}\", std::any::type_name::<A>(), std::any::type_name::<B>()));\n\
         \x20   }}\n}}\n\n\
         fn un<A, R>(o: &Oracle, rng: &mut Rng, op: UnOp, f: impl Fn(A) -> R)\n\
         where\n    A: Extensor<Slots = (), Coef = f64>,\n    R: Extensor<Slots = (), Coef = f64>,\n{{\n\
         \x20   for _ in 0..8 {{\n\
         \x20       let a: A = random(rng);\n\
         \x20       assert_close(&o.dense(&f(a)), &o.unop(op, &o.dense(&a)), &format!(\"{{op:?}} {{}}\", std::any::type_name::<A>()));\n\
         \x20   }}\n}}\n\n\
         /// `v x ~v` against the oracle; with `unit`, `v` is first normalized to `v ~v = 1`.\n\
         fn sandwich<V, X, R>(o: &Oracle, rng: &mut Rng, unit: bool, f: impl Fn(V, X) -> R, prepared: impl Fn(V, X) -> R)\n\
         where\n    V: Extensor<Slots = (), Coef = f64>,\n    X: Extensor<Slots = (), Coef = f64>,\n    R: Extensor<Slots = (), Coef = f64>,\n{{\n\
         \x20   let mut tested = 0;\n\
         \x20   for _ in 0..16 {{\n\
         \x20       let (mut v, x): (V, X) = (random(rng), random(rng));\n\
         \x20       if unit {{\n\
         \x20           let Some(u) = o.random_unit(rng) else {{ continue }};\n\
         \x20           v = u;\n\
         \x20       }}\n\
         \x20       tested += 1;\n\
         \x20       let (dv, dx) = (o.dense(&v), o.dense(&x));\n\
         \x20       let want = o.binop(BinOp::Gp, &o.binop(BinOp::Gp, &dv, &dx), &o.unop(UnOp::Reverse, &dv));\n\
         \x20       assert_close(&o.dense(&f(v, x)), &want, &format!(\"sandwich {{}} {{}}\", std::any::type_name::<V>(), std::any::type_name::<X>()));\n\
         \x20       assert_close(&o.dense(&prepared(v, x)), &want, &format!(\"prepared {{}} {{}}\", std::any::type_name::<V>(), std::any::type_name::<X>()));\n\
         \x20   }}\n\
         \x20   assert!(tested > 0, \"no sample of {{}} could be normalized\", std::any::type_name::<V>());\n}}\n\n",
        name = spec.name
    );
    for op in BinOp::ALL {
        let (_, m) = trait_of(op);
        let _ = write!(
            t,
            "#[test]\nfn {m}() {{\n    let o = Oracle::from_spec(SPEC);\n    let mut rng = Rng::new({});\n",
            op as u64 + 1
        );
        for (bop, a, b) in &stats.binary {
            if *bop == op {
                let _ = writeln!(
                    t,
                    "    bin::<{a}<(), f64>, {b}<(), f64>, _>(&o, &mut rng, BinOp::{op:?}, |a, b| a.{m}(b));"
                );
            }
        }
        t.push_str("}\n\n");
    }
    for op in UnOp::ALL {
        let (_, m) = unop_trait(op);
        let _ = write!(
            t,
            "#[test]\nfn {m}() {{\n    let o = Oracle::from_spec(SPEC);\n    let mut rng = Rng::new({});\n",
            op as u64 + 20
        );
        for k in &spec.kinds {
            let _ = writeln!(
                t,
                "    un::<{}<(), f64>, _>(&o, &mut rng, UnOp::{op:?}, |a| a.{m}());",
                k.name
            );
        }
        t.push_str("}\n\n");
    }
    t.push_str("#[test]\nfn value_methods() {\n    let o = Oracle::from_spec(SPEC);\n    let mut rng = Rng::new(77);\n");
    for v in &stats.values {
        let k = &v.kind;
        if v.inverse.is_some() {
            let _ = writeln!(
                t,
                "    common::inverse::<{k}<(), f64>, _>(&o, &mut rng, |x| x.inverse());"
            );
        }
        if v.normalized {
            let _ = writeln!(
                t,
                "    common::normalized::<{k}<(), f64>>(&o, &mut rng, |x| x.normalized().into_inner());"
            );
        }
        if let Some(on) = &v.exp {
            let has_log = stats
                .values
                .iter()
                .any(|w| &w.kind == on && w.log.as_deref() == Some(k.as_str()));
            if has_log {
                let _ = writeln!(
                    t,
                    "    common::exp_log::<{k}<(), f64>, {on}<(), f64>>(&o, &mut rng, |b| b.exp(), |r| r.log());"
                );
            }
        }
        if v.sqrt && v.log.is_some() {
            let _ = writeln!(
                t,
                "    common::sqrt::<{k}<(), f64>>(&o, &mut rng, |r| r.sqrt().into_inner(), |r| r.into_inner() * r.into_inner());"
            );
        }
    }
    t.push_str("}\n\n");
    t.push_str("#[test]\nfn sandwiches() {\n    let o = Oracle::from_spec(SPEC);\n    let mut rng = Rng::new(99);\n");
    for (v, x, unit, ..) in &stats.sandwich_costs {
        if *unit {
            let _ = writeln!(
                t,
                "    sandwich::<{v}<(), f64>, {x}<(), f64>, _>(&o, &mut rng, true, |v, x| Unit::new_unchecked(v) >> x, |v, x| Unit::new_unchecked(v).prepare::<{x}>() >> x);"
            );
        } else {
            let _ = writeln!(
                t,
                "    sandwich::<{v}<(), f64>, {x}<(), f64>, _>(&o, &mut rng, false, |v, x| v >> x, |v, x| v.prepare::<{x}>() >> x);"
            );
        }
    }
    t.push_str("}\n");
    t
}

/// A signed sum as a balanced expression tree: `log2(n)` dependent additions. Floating-point
/// sums cannot be reassociated by the compiler, so the shape written here is the shape that
/// runs, and a chain would make single calls latency bound.
fn balanced_sum(items: &[(bool, String)]) -> String {
    fn go(items: &[(bool, String)]) -> (bool, String) {
        match items {
            [] => (false, "T::zero()".into()),
            [x] => x.clone(),
            _ => {
                let (l, r) = items.split_at(items.len().div_ceil(2));
                let ((ln, lv), (rn, rv)) = (go(l), go(r));
                match (ln, rn) {
                    (false, false) => (false, format!("({lv} + {rv})")),
                    (false, true) => (false, format!("({lv} - {rv})")),
                    (true, false) => (false, format!("({rv} - {lv})")),
                    (true, true) => (true, format!("({lv} + {rv})")),
                }
            }
        }
    }
    // Put positive terms first so a leading negation is rarely needed.
    let mut sorted = items.to_vec();
    sorted.sort_by_key(|x| x.0);
    let (neg, e) = go(&sorted);
    if neg { format!("-{e}") } else { e }
}

/// The blades of `a` listed by `keep` (in `a`'s order) as `(i, j, flip)` entries of `b`'s
/// layout, or `None` where a kept blade is not one of `b`'s.
fn shared_blades(
    a: &KindSpec,
    b: &KindSpec,
    keep: impl Fn(u32) -> bool,
) -> Vec<(usize, usize, bool)> {
    a.layout
        .blades
        .iter()
        .enumerate()
        .filter(|(_, (m, _))| keep(*m))
        .filter_map(|(i, &(m, sa))| b.layout.position(m).map(|(j, sb)| (i, j, sa != sb)))
        .collect()
}

fn render_shared(shared: &[(usize, usize, bool)]) -> String {
    shared
        .iter()
        .map(|(i, j, f)| format!("({i}, {j}, {f})"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `Cast` for every pair of kinds sharing a blade, `SubKind` where one's blades all lie in
/// the other's, and `GradePart` for every grade of every kind (`gax_core::cast`).
fn kind_tables(spec: &AlgebraSpec, out: &mut String) {
    for a in &spec.kinds {
        for b in &spec.kinds {
            let shared = shared_blades(a, b, |_| true);
            if shared.is_empty() {
                continue;
            }
            let (an, bn) = (&a.name, &b.name);
            let _ = writeln!(
                out,
                "impl gx::Cast<{bn}> for {an} {{\n    const SHARED: &'static [(usize, usize, bool)] = &[{}];\n}}\n",
                render_shared(&shared)
            );
            if shared.len() == a.layout.len() {
                let _ = writeln!(out, "impl gx::SubKind<{bn}> for {an} {{}}\n");
            }
        }
        let grades: BTreeSet<u32> = a
            .layout
            .blades
            .iter()
            .map(|(m, _)| m.count_ones())
            .collect();
        for g in grades {
            let support: BTreeSet<u32> = a
                .layout
                .blades
                .iter()
                .map(|(m, _)| *m)
                .filter(|m| m.count_ones() == g)
                .collect();
            let o = spec
                .kind_for_support(&support)
                .expect("the kind itself holds it");
            let shared = shared_blades(a, o, |m| m.count_ones() == g);
            let _ = writeln!(
                out,
                "impl gx::GradePart<{g}> for {} {{\n    type Out = {};\n    const SHARED: &'static [(usize, usize, bool)] = &[{}];\n}}\n",
                a.name,
                o.name,
                render_shared(&shared)
            );
        }
    }
}

/// The value methods of a kind as trait impls (`gax_core::ops::{Exp, Inverse, Norm, Normalize,
/// Sqrt}`), forwarding to the inherent methods, so generic code can call them.
fn value_traits(name: &str, meta: &crate::emit_values::ValueMethods, out: &mut String) {
    if let Some(on) = &meta.exp {
        let _ = writeln!(
            out,
            "impl<T: gx::Real> gx::Exp for {name}<(), T> {{\n    type Output = gx::Unit<{on}<(), T>>;\n    #[inline(always)]\n    fn exp(self) -> Self::Output {{\n        Self::exp(self)\n    }}\n}}\n"
        );
    }
    if let Some(on) = &meta.inverse {
        let _ = writeln!(
            out,
            "impl<T: gx::Real> gx::Inverse for {name}<(), T> {{\n    type Output = {on}<(), T>;\n    #[inline(always)]\n    fn inverse(self) -> Self::Output {{\n        Self::inverse(self)\n    }}\n}}\n"
        );
    }
    if meta.norm {
        let _ = writeln!(
            out,
            "impl<T: gx::Real> gx::Norm for {name}<(), T> {{\n    #[inline(always)]\n    fn norm_squared(self) -> T {{\n        Self::norm_squared(self)\n    }}\n    #[inline(always)]\n    fn norm(self) -> T {{\n        Self::norm(self)\n    }}\n}}\n"
        );
    }
    if meta.normalized {
        let _ = writeln!(
            out,
            "impl<T: gx::Real> gx::Normalize for {name}<(), T> {{\n    #[inline(always)]\n    fn normalized(self) -> gx::Unit<Self> {{\n        Self::normalized(self)\n    }}\n}}\n"
        );
    }
    if meta.sqrt {
        let _ = writeln!(
            out,
            "impl<T: gx::Real> gx::Sqrt for {name}<(), T> {{\n    type Output = gx::Unit<Self>;\n    #[inline(always)]\n    fn sqrt(self) -> Self::Output {{\n        Self::sqrt(self)\n    }}\n}}\n"
        );
    }
}
