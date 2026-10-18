//! Rust code emission for an algebra.
//!
//! The output is one module of ordinary Rust: a struct per kind, the product traits of
//! `gax-core` implemented per pair of kinds (tier 1: straight-line, generic over slots), and
//! fused sandwich kernels for the versor kinds (tier 2: simplified symbolically).

mod gpu;
mod kinds;
mod oracle_tests;
mod products;
mod sandwich;

pub use kinds::reciprocal_frame;
pub use oracle_tests::emit_tests;

use crate::spec::{AlgebraSpec, KindSpec};
use crate::table::{BinOp, UnOp};
use kinds::{kind_eqs, kind_tables, value_traits};
use sandwich::{sandwich_math, sandwich_out, sandwich_plain};
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
    /// Every generated binary impl with its output kind: `(op, left, right, output)`.
    pub products: Vec<(BinOp, String, String, String)>,
    /// Every generated unary impl: `(op, kind, output)`.
    pub unary: Vec<(UnOp, String, String)>,
    /// Every sandwich kernel.
    pub sandwiches: Vec<SandwichStat>,
    /// Every outermorphism: `(base kind, extended kind, whether the extended kind is the top)`.
    pub outermorphisms: Vec<(String, String, bool)>,
    /// The value methods emitted per kind.
    pub values: Vec<crate::emit_values::ValueMethods>,
    /// The sandwich kernels (value and matrix paths) in language-neutral form, for the WGSL
    /// modules (see [`crate::kernel`]).
    pub kernels: Vec<crate::kernel::Kernel>,
}

/// A generated sandwich kernel `v >> x`.
#[derive(Clone, Debug)]
pub struct SandwichStat {
    /// The versor's kind.
    pub versor: String,
    /// The passenger's kind.
    pub passenger: String,
    /// Whether the versor is a `Unit`.
    pub unit: bool,
    /// The result's kind.
    pub output: String,
    /// The op count of the value path, `v >> x`.
    pub value_cost: crate::slp::Cost,
    /// The op count of building the matrix of the map path, `v.prepare::<X>()`.
    pub map_cost: crate::slp::Cost,
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
    e.part("kinds");
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
    kind_eqs(spec, &mut e.out);
    kind_tables(spec, &mut e.out);
    crate::par::report(&format!("{} values", spec.name), phase);
    let phase = std::time::Instant::now();
    for op in BinOp::ALL {
        e.part(&format!("products_{}", format!("{op:?}").to_lowercase()));
        for a in &spec.kinds {
            for b in &spec.kinds {
                e.binary(op, a, b);
            }
        }
    }
    crate::par::report(&format!("{} products", spec.name), phase);
    let phase = std::time::Instant::now();
    e.part("conversions");
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
    let mut versor = "";
    for (&(v, x, unit), math) in jobs.iter().zip(maths) {
        if v.name != versor {
            versor = &v.name;
            e.part(&format!("sandwiches_{}", crate::kernel::snake(versor)));
        }
        if let Some(math) = math {
            e.sandwich(v, x, unit, math);
        }
    }
    crate::par::report(
        &format!("{} sandwiches and outermorphisms", spec.name),
        phase,
    );
    e.part("outermorphisms");
    e.out.push_str(&outer);
    e.stats.outermorphisms = pairs;
    e.part("layouts");
    for a in &spec.aliases {
        let (o, which) = a.orientation_name().map_or_else(Default::default, |t| {
            (format!(", gx::{t}"), format!(", {}", t.to_lowercase()))
        });
        let _ = writeln!(
            e.out,
            "/// Alias of [`{kind}`]{which}.\npub type {alias}<S = (), T = f64> = {kind}<S, T{o}>;\n",
            kind = a.kind,
            alias = a.name
        );
    }
    if let Some(gate) = &cfg.batch {
        let names = spec
            .kinds
            .iter()
            .map(|k| (k.name.clone(), k.name.clone(), k.name.clone()))
            .chain(spec.aliases.iter().map(|a| {
                let k = &a.kind;
                let marker = a
                    .orientation_name()
                    .map_or_else(|| k.clone(), |t| format!("{k}<(), f64, gx::{t}>"));
                (a.name.clone(), k.clone(), marker)
            }));
        for (name, kind, marker) in names {
            let _ = writeln!(
                e.out,
                "{gate}\n/// [`{kind}`] values in struct-of-arrays form, for the batch kernels.\npub type {name}Soa<E = f32> = gx::batch::Soa<{marker}, E>;\n"
            );
        }
    }
    if let Some(gate) = &cfg.gpu {
        e.gpu_types(gate);
    }
    e.epilogue();
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
    "Orientation",
    "Extrinsic",
    "Intrinsic",
    "Reorient",
    "Reciprocal",
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

struct Emitter<'a> {
    spec: &'a AlgebraSpec,
    cfg: &'a Config,
    out: String,
    stats: Stats,
    /// The plain sandwich kernels emitted so far, by name.
    helpers: BTreeSet<String>,
}

/// Rust identifier for a blade accessor (`1` becomes `s`).
pub(crate) fn blade_ident(b: &str) -> String {
    if b == "1" { "s".into() } else { b.to_string() }
}

/// The head of an impl for every slot list `S`, coefficient type `T` and orientation `O`, of a
/// trait for `{Kind}<S, T, O>` (or of its inherent methods).
const IMPL_ALL: &str = "impl<S: Slots, T: Coef, O: gx::Orientation>";

/// The head of an impl for every orientation `O`, mostly on a kind's marker `{Kind}<(), f64, O>`.
const IMPL_O: &str = "impl<O: gx::Orientation>";

/// The zero slot array of the slot list `slots` (a type, `S` or `<S as Slots>`).
pub(crate) fn zero_slots(slots: &str) -> String {
    format!("{slots}::from_flat(&mut |_| T::zero(), 0)")
}

/// The terms `(negative, term)` as one chain, `a + b - c`, with a leading negative term
/// negated, `-a + b`; empty for no terms.
pub(crate) fn signed_chain(terms: impl IntoIterator<Item = (bool, String)>) -> String {
    terms
        .into_iter()
        .fold(String::new(), |e, (neg, t)| match (e.is_empty(), neg) {
            (true, false) => t,
            (true, true) => format!("-{t}"),
            (false, false) => format!("{e} + {t}"),
            (false, true) => format!("{e} - {t}"),
        })
}

/// The struct literal of kind `name` with the coefficient array `c` (an expression).
pub(crate) fn kind_literal(name: &str, c: &str) -> String {
    format!("{name} {{ o: core::marker::PhantomData, c: {c} }}")
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
        for (c, row) in alg.basis_suffixes().iter().zip(alg.metric()) {
            let row: Vec<String> = row.iter().map(ToString::to_string).collect();
            let _ = writeln!(basis, "//   e{c}: [{}]", row.join(", "));
        }
        let _ = write!(
            self.out,
            "// @generated by gax-gen from the `{}` declaration. Do not edit by hand.\n\
             // Metric (rows e_i . e_j):\n{basis}\n\
             use {core} as gx;\n\n",
            spec.name
        );
        self.names();
    }

    /// The kinds by name (ADR-052): the module root names each in its default orientation,
    /// [`intrinsic`] and [`extrinsic`] in theirs, all aliases of the structs in `oriented`,
    /// which the rest of the module defines (closed by [`Self::epilogue`]).
    fn names(&mut self) {
        let spec = self.spec;
        let mut root = String::from("pub use oriented::*;\n\n");
        let mut fixed = [String::new(), String::new()];
        for k in &spec.kinds {
            let name = &k.name;
            let default = crate::spec::orientation_name(k.intrinsic);
            let doc = if k.doc.is_empty() {
                format!("A `{name}` of the `{}` algebra.", spec.name)
            } else {
                k.doc.clone()
            };
            let _ = write!(
                root,
                "#[doc = {doc:?}]\n///\n/// In its default orientation, {}: [`oriented::{name}`] is the kind for any orientation,\n/// [`intrinsic::{name}`] and [`extrinsic::{name}`] name either one.\npub type {name}<S = (), T = f64> = oriented::{name}<S, T, gx::{default}>;\n\n",
                default.to_lowercase()
            );
            for (o, text) in ["Intrinsic", "Extrinsic"].into_iter().zip(&mut fixed) {
                let _ = write!(
                    text,
                    "    /// [`oriented::{name}`], {}.\n    pub type {name}<S = (), T = f64> = oriented::{name}<S, T, gx::{o}>;\n",
                    o.to_lowercase()
                );
            }
        }
        self.out.push_str(&root);
        for (o, text) in ["intrinsic", "extrinsic"].into_iter().zip(&fixed) {
            let _ = write!(
                self.out,
                "/// Every kind of the algebra, {o} (ADR-051).\npub mod {o} {{\n    use super::{{gx, oriented}};\n\n{text}}}\n\n"
            );
        }
        self.out.push_str(
            "/// The kinds as structs generic over their orientation `O` (ADR-051), with every\n/// operation; the module root and [`intrinsic`] and [`extrinsic`] name them with `O` fixed.\npub mod oriented {\nuse super::gx;\n",
        );
    }

    /// The end of `pub mod oriented`, after the parts.
    fn epilogue(&mut self) {
        let _ = write!(self.out, "\n{}\n}}\n", crate::split::EPILOGUE);
    }

    /// Start a part of the module (see [`crate::split`]): a marker comment, which the
    /// regeneration tool cuts at and `algebra!` ignores.
    fn part(&mut self, name: &str) {
        let _ = write!(self.out, "\n{}{name}\n", crate::split::MARKER);
    }
}
