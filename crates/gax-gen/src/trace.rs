//! Build-time tracing: run generic user code on symbolic coefficients and emit fused kernels.
//!
//! ```ignore
//! // build.rs
//! #[path = "src/kernels.rs"]
//! mod kernels;
//! use gax::pga3d::{Motor, Point};
//! use gax::trace::{Sym, Tracer};
//! use gax::Unit;
//!
//! fn main() {
//!     let mut t = Tracer::new();
//!     t.kernel("rotate_fused", |m: Unit<Motor<(), Sym>>, p: Point<(), Sym>| kernels::rotate(m, p));
//!     t.write_out_dir("fused.rs");
//!     println!("cargo:rerun-if-changed=src/kernels.rs");
//! }
//!
//! // src/lib.rs
//! mod kernels;
//! include!(concat!(env!("OUT_DIR"), "/fused.rs"));
//! ```
//!
//! Each traced kernel becomes a function generic over the coefficient type `T: Real`, with
//! the same argument and result types as the closure. The symbolic outputs are simplified:
//! type conditions of `Unit` arguments, common subexpressions, square-sum completion,
//! constants folded, and shared reciprocals and roots computed once. Every simplification is
//! verified exactly (modulo the conditions) before code is emitted.

use crate::cse::{self, Stage, StageOp};
use crate::poly::{Poly, Var};
use crate::slp::{Cost, Operand, render};
pub use crate::sym::Sym;
use crate::sym::{ConstKind, NodeOp, VarDef};
use gax_core::{Coef, Traceable};
use std::collections::HashMap;
use std::fmt::Write as _;

impl Traceable for Sym {
    type Coef = Sym;
    const LEN: usize = 1;
    fn from_fn(f: &mut dyn FnMut(usize) -> Sym) -> Sym {
        f(0)
    }
    fn for_each(&self, f: &mut dyn FnMut(Sym)) {
        f(*self);
    }
    fn write_type(w: &mut dyn std::fmt::Write, coef: &str) -> std::fmt::Result {
        w.write_str(coef)
    }
    fn write_coeffs(w: &mut dyn std::fmt::Write, name: &str) -> std::fmt::Result {
        write!(w, "[{name}]")
    }
    fn write_construct(w: &mut dyn std::fmt::Write, parts: &[&str]) -> std::fmt::Result {
        w.write_str(parts[0])
    }
}

fn result_type<R: Traceable>(coef: &str) -> String {
    let mut s = String::new();
    R::write_type(&mut s, coef).expect("write to string");
    s
}

fn construct<R: Traceable>(outs: &[String]) -> String {
    let parts: Vec<&str> = outs.iter().map(String::as_str).collect();
    let mut s = String::new();
    R::write_construct(&mut s, &parts).expect("write to string");
    s
}

/// A closure that can be traced: its arguments are [`Traceable`] with `Sym` coefficients.
pub trait TraceFn<Args> {
    /// Build the arguments, run the closure, and return what the tracer needs.
    fn run(&self) -> Traced;
}

/// Builds the Rust expression of a kernel's result from its output expressions.
type Constructor = Box<dyn Fn(&[String]) -> String>;

/// How to write a traceable type's code at any coefficient type: its type, the expression of
/// a value's coefficient array, and a constructor from coefficient expressions.
#[derive(Clone)]
struct Shape {
    len: usize,
    /// The kind's path and whether it is a `Unit`, for values of one kind.
    kind: Option<(String, bool)>,
    ty: fn(&str) -> String,
    coeffs: fn(&str) -> String,
    construct: fn(&[String]) -> String,
}

impl Shape {
    fn of<A: Traceable>() -> Shape {
        let mut path = String::new();
        let kind = A::write_kind(&mut path).map(|unit| (path, unit));
        Shape {
            len: A::LEN,
            kind,
            ty: result_type::<A>,
            coeffs: |name| {
                let mut s = String::new();
                A::write_coeffs(&mut s, name).expect("write to string");
                s
            },
            construct: construct::<A>,
        }
    }
}

/// The outcome of running a closure on symbolic arguments.
pub struct Traced {
    /// `(argument type, coefficient-array expression template)` per argument.
    args: Vec<(String, usize, String)>,
    /// Conditions satisfied by the arguments.
    conditions: Vec<Poly>,
    /// Output coefficients.
    outputs: Vec<Sym>,
    /// Result type.
    result_type: String,
    /// Constructor of the result from output expressions.
    construct: Constructor,
    /// The shapes of the arguments and of the result, for the batch form.
    shapes: (Vec<Shape>, Shape),
}

fn arg_info<A: Traceable<Coef = Sym>>(k: usize) -> (A, (String, usize, String), Vec<Poly>) {
    let a = A::from_fn(&mut |i| Sym::input(k, i));
    let mut ty = String::new();
    A::write_type(&mut ty, "T").expect("write");
    let mut coeffs = String::new();
    A::write_coeffs(&mut coeffs, &format!("a{k}")).expect("write");
    let mut conds = Vec::new();
    a.for_each_condition(&mut |c| {
        let p = c.poly();
        if !p.is_zero() {
            conds.push(p);
        }
    });
    (a, (ty, A::LEN, coeffs), conds)
}

macro_rules! trace_fn {
    ($($A:ident $a:ident $k:tt),+) => {
        impl<F, R, $($A),+> TraceFn<($($A,)+)> for F
        where
            F: Fn($($A),+) -> R,
            R: Traceable<Coef = Sym>,
            $($A: Traceable<Coef = Sym>),+
        {
            fn run(&self) -> Traced {
                let mut args = Vec::new();
                let mut conditions = Vec::new();
                let mut shapes = Vec::new();
                $(
                    let ($a, info, conds) = arg_info::<$A>($k);
                    args.push(info);
                    conditions.extend(conds);
                    shapes.push(Shape::of::<$A>());
                )+
                let result = self($($a),+);
                Traced {
                    args,
                    conditions,
                    outputs: { let mut v = Vec::new(); result.for_each(&mut |c| v.push(c)); v },
                    result_type: result_type::<R>("T"),
                    construct: Box::new(|outs| construct::<R>(outs)),
                    shapes: (shapes, Shape::of::<R>()),
                }
            }
        }
    };
}

trace_fn!(A0 a0 0);
trace_fn!(A0 a0 0, A1 a1 1);
trace_fn!(A0 a0 0, A1 a1 1, A2 a2 2);
trace_fn!(A0 a0 0, A1 a1 1, A2 a2 2, A3 a3 3);
trace_fn!(A0 a0 0, A1 a1 1, A2 a2 2, A3 a3 3, A4 a4 4);
trace_fn!(A0 a0 0, A1 a1 1, A2 a2 2, A3 a3 3, A4 a4 4, A5 a5 5);
trace_fn!(A0 a0 0, A1 a1 1, A2 a2 2, A3 a3 3, A4 a4 4, A5 a5 5, A6 a6 6);
trace_fn!(A0 a0 0, A1 a1 1, A2 a2 2, A3 a3 3, A4 a4 4, A5 a5 5, A6 a6 6, A7 a7 7);

/// Strategies whose polynomials exceed this many terms are skipped.
const MAX_TERMS: usize = 20_000;

/// Report on one traced kernel.
#[derive(Clone, Debug)]
pub struct KernelReport {
    /// Kernel name.
    pub name: String,
    /// Cost of the emitted code.
    pub cost: Cost,
    /// Cost of the generic code as it runs: every operation, constants not folded.
    pub naive: Cost,
}

/// Collects traced kernels and emits their source.
#[derive(Default)]
pub struct Tracer {
    source: String,
    reports: Vec<KernelReport>,
    batch: bool,
}

impl Tracer {
    /// A tracer with no kernels.
    pub fn new() -> Tracer {
        Tracer::default()
    }

    /// Also emit a batch form `{name}_batch` of every kernel (it needs `gax`'s `batch`
    /// feature in the crate that includes the kernels): the kernel applied elementwise to
    /// slices of arguments, on the SIMD lanes of the level `gax::batch` detects.
    pub fn batch(&mut self, on: bool) -> &mut Tracer {
        self.batch = on;
        self
    }

    /// Trace `f` and add a fused kernel named `name`.
    ///
    /// # Panics
    /// If `f` branches on a coefficient, or if the simplifier fails its own exactness check
    /// (a bug).
    #[allow(clippy::needless_pass_by_value, clippy::too_many_lines)] // a closure literal is the natural argument
    pub fn kernel<Args, F: TraceFn<Args>>(&mut self, name: &str, f: F) -> &mut Tracer {
        // Trace under several expansion limits and keep the cheapest verified program. A
        // limit of 0 records the computation as written (each product and sum a node, with
        // constants folded and common subexpressions shared), so the result is never worse
        // than the generic code; no limit expands everything into polynomials, which exposes
        // cancellations and type conditions across steps.
        let mut best: Option<(crate::slp::Program, Traced, Vec<VarDef>)> = None;
        // A strategy whose polynomials grow too large is skipped, and one that overflows the
        // exact rational arithmetic (which panics rather than wrap) is dropped: the limit-0
        // trace, the computation as written, always succeeds.
        let mut failure: Option<String> = None;
        let quiet = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        for limit in [Some(0), Some(1), Some(8), Some(32), Some(128), None] {
            let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                Sym::reset_with_limit(limit);
                let traced = f.run();
                let outputs: Vec<Poly> = traced.outputs.iter().map(|s| s.poly()).collect();
                let stages = stages_of_arena();
                let size: usize = outputs.iter().map(Poly::len).sum::<usize>()
                    + stages
                        .iter()
                        .flat_map(|st| st.args.iter())
                        .map(Poly::len)
                        .sum::<usize>();
                if limit.is_some_and(|l| l > 0) && size > MAX_TERMS {
                    return None;
                }
                let mut relations = traced.conditions.clone();
                relations.extend(Sym::atom_relations());
                let (prog, _env) = cse::compile_staged_best(&outputs, &stages, &relations);
                let defs: Vec<VarDef> = (0..Sym::var_count() as Var).map(Sym::var_def).collect();
                Some((prog, traced, defs))
            }));
            if let Err(e) = &attempt {
                failure = e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()));
            }
            if let Ok(Some((prog, traced, defs))) = attempt
                && best
                    .as_ref()
                    .is_none_or(|(p, ..)| prog.cost().weight() < p.cost().weight())
            {
                best = Some((prog, traced, defs));
            }
        }
        std::panic::set_hook(quiet);
        let Some((prog, traced, defs)) = best else {
            panic!(
                "gax::trace: could not trace `{name}`: {}",
                failure.unwrap_or_else(|| "no strategy succeeded".into())
            );
        };
        // The cost of the generic code as it runs: every operation, nothing folded.
        let naive = {
            Sym::reset_runtime_model();
            let run = f.run();
            let outputs: Vec<Poly> = run.outputs.iter().map(|s| s.poly()).collect();
            let stages = stages_of_arena();
            cse::compile_staged(&outputs, &stages, &[], cse::Reduction::None, false)
                .0
                .cost()
        };
        let var_name = |v: Var| match defs[v as usize].clone() {
            VarDef::Input { arg, index } => format!("a{arg}[{index}]"),
            VarDef::Constant(ConstKind::Float(x)) => format!("T::from_f64({x:?})"),
            VarDef::Constant(ConstKind::Epsilon) => "T::epsilon()".to_string(),
            VarDef::Atom { .. }
            | VarDef::Atan2 { .. }
            | VarDef::Select { .. }
            | VarDef::Node { .. } => {
                unreachable!("atoms and nodes are bound to temporaries")
            }
        };
        let params: Vec<String> = traced
            .args
            .iter()
            .enumerate()
            .map(|(k, (ty, _, _))| format!("a{k}: {ty}"))
            .collect();
        let cost = prog.cost();
        let s = &mut self.source;
        let _ = writeln!(
            s,
            "/// Fused kernel `{name}`, traced at build time by `gax::trace`."
        );
        let _ = writeln!(
            s,
            "///\n/// Cost: {cost} (generic code at run time: {naive})."
        );
        let _ = writeln!(
            s,
            "#[inline(always)]\n#[allow(clippy::all, clippy::pedantic, unused_variables, unused_parens, non_snake_case)]"
        );
        let _ = writeln!(
            s,
            "pub fn {name}<T: ::gax::Real>({}) -> {} {{",
            params.join(", "),
            traced.result_type
        );
        for (k, (_, _, coeffs)) in traced.args.iter().enumerate() {
            let _ = writeln!(s, "    let a{k} = {coeffs};");
        }
        let mut body = String::new();
        prog.emit_lets(&var_name, "t", &mut body);
        for line in body.lines() {
            let _ = writeln!(s, "{}", line.replacen("        ", "    ", 1));
        }
        let outs: Vec<String> = prog
            .outputs
            .iter()
            .map(|o: &Operand| render(o, &var_name, "t"))
            .collect();
        let _ = writeln!(s, "    {}\n}}\n", (traced.construct)(&outs));
        if self.batch {
            batch_form(s, name, &traced.shapes.0, &traced.shapes.1);
            if traced.shapes.1.kind.is_some() {
                batch_form_soa(s, name, &traced.shapes.0, &traced.shapes.1);
            }
        }
        self.reports.push(KernelReport {
            name: name.to_string(),
            cost,
            naive,
        });
        self
    }

    /// The emitted source of all kernels so far.
    pub fn source(&self) -> String {
        format!(
            "// @generated by gax::trace. Do not edit by hand.\n\n{}",
            self.source
        )
    }

    /// Reports on the kernels traced so far.
    pub fn reports(&self) -> &[KernelReport] {
        &self.reports
    }

    /// Write the source to `$OUT_DIR/{file}` (for use from `build.rs`), and print a cargo
    /// warning-free summary line per kernel.
    ///
    /// # Panics
    /// If `OUT_DIR` is not set or the file cannot be written.
    pub fn write_out_dir(&self, file: &str) {
        let dir = std::env::var_os("OUT_DIR").expect("OUT_DIR is set when running from build.rs");
        let path = std::path::Path::new(&dir).join(file);
        std::fs::write(&path, self.source()).expect("write traced kernels");
    }
}

/// The non-polynomial steps and opaque nodes of the current trace, in creation order.
fn stages_of_arena() -> Vec<Stage> {
    let mut stages = Vec::new();
    for v in 0..Sym::var_count() as Var {
        let (op, args) = match Sym::var_def(v) {
            VarDef::Atom { func, arg } => (StageOp::Call(func), vec![arg.poly()]),
            VarDef::Atan2 { y, x } => (StageOp::Atan2, vec![y.poly(), x.poly()]),
            VarDef::Select { a, b, x, y } => (
                StageOp::Select,
                vec![a.poly(), b.poly(), x.poly(), y.poly()],
            ),
            VarDef::Node { op, a, b } => (
                match op {
                    NodeOp::Add => StageOp::Add,
                    NodeOp::Sub => StageOp::Sub,
                    NodeOp::Mul => StageOp::Mul,
                },
                vec![a.poly(), b.poly()],
            ),
            VarDef::Input { .. } | VarDef::Constant(_) => continue,
        };
        stages.push(Stage { var: v, op, args });
    }
    stages
}

/// Map from variables to their definition, for inspection in tests.
pub fn variables() -> HashMap<Var, VarDef> {
    (0..Sym::var_count() as Var)
        .map(|v| (v, Sym::var_def(v)))
        .collect()
}

/// Create a symbolic value of any traceable type, with fresh inputs for argument `arg`.
pub fn symbolic<A: Traceable<Coef = Sym>>(arg: usize) -> A {
    A::from_fn(&mut |i| Sym::input(arg, i))
}

/// Zero, as a `Sym`.
pub fn zero() -> Sym {
    Sym::zero()
}

/// Emit `{name}_batch`: the kernel on slices, gathering lanes, running the fused function on
/// them and scattering the results.
fn batch_form(s: &mut String, name: &str, args: &[Shape], result: &Shape) {
    let n = args.len();
    let slices: Vec<String> = (0..n)
        .map(|k| format!("a{k}: &'a [{}]", (args[k].ty)("E")))
        .collect();
    let fields: Vec<String> = (0..n)
        .map(|k| format!("&'a [{}]", (args[k].ty)("E")))
        .collect();
    let out_e = (result.ty)("E");
    let names: Vec<String> = (0..n).map(|k| format!("a{k}")).collect();
    let _ = writeln!(
        s,
        "/// Batch form of [`{name}`]: `out[i] = {name}(a0[i], ...)` for every `i`, on the SIMD\n\
         /// lanes of the level `gax::batch` detects. An argument slice of length 1 is broadcast.\n\
         ///\n/// # Panics\n/// If an argument slice has neither length 1 nor `out.len()`.\n\
         #[inline]\n#[allow(clippy::all, clippy::pedantic, unused_variables, unused_parens, non_snake_case, non_camel_case_types)]\n\
         pub fn {name}_batch<'a, E: ::gax::batch::LaneElem>({}, out: &'a mut [{out_e}]) {{",
        slices.join(", ")
    );
    let _ = writeln!(
        s,
        "    struct K<'a, E: ::gax::batch::LaneElem>({}, &'a mut [{out_e}]);",
        fields.join(", ")
    );
    let _ = writeln!(
        s,
        "    impl<'a, E: ::gax::batch::LaneElem> ::gax::batch::Kernel<E> for K<'a, E> {{\n        type Output = ();\n        #[inline(always)]\n        fn run<L: ::gax::batch::Batch<Elem = E>>(self) {{\n            let K({}, out) = self;\n            let n = out.len();",
        names.join(", ")
    );
    for (k, a) in args.iter().enumerate() {
        let parts: Vec<String> = (0..a.len)
            .map(|j| format!("L::splat(({})[{j}])", (a.coeffs)(&format!("a{k}[0]"))))
            .collect();
        let _ = writeln!(
            s,
            "            let b{k} = a{k}.len() == 1;\n            let s{k}: {} = if b{k} {{ {} }} else {{ {} }};",
            (a.ty)("L"),
            (a.construct)(&parts),
            (a.construct)(&vec!["L::zero()".to_string(); a.len])
        );
    }
    let _ = writeln!(
        s,
        "            let mut i = 0;\n            while i < n {{\n                let m = (n - i).min(L::LANES);"
    );
    for (k, a) in args.iter().enumerate() {
        let parts: Vec<String> = (0..a.len)
            .map(|j| {
                format!(
                    "::gax::batch::column::<L>(m, #[inline(always)] |l| ({})[{j}])",
                    (a.coeffs)(&format!("a{k}[i + l]"))
                )
            })
            .collect();
        let _ = writeln!(
            s,
            "                let x{k}: {} = if b{k} {{ s{k} }} else {{ {} }};",
            (a.ty)("L"),
            (a.construct)(&parts)
        );
    }
    let xs: Vec<String> = (0..n).map(|k| format!("x{k}")).collect();
    let _ = writeln!(
        s,
        "                let y: {} = {name}::<L>({});\n                let yc = {};",
        (result.ty)("L"),
        xs.join(", "),
        (result.coeffs)("y")
    );
    let cols: Vec<String> = (0..result.len)
        .map(|j| format!("::gax::batch::to_array(yc[{j}])"))
        .collect();
    let parts: Vec<String> = (0..result.len).map(|j| format!("cols[{j}][l]")).collect();
    let _ = writeln!(
        s,
        "                let cols = [{}];\n                for l in 0..m {{\n                    out[i + l] = {};\n                }}\n                i += L::LANES;\n            }}\n        }}\n    }}",
        cols.join(", "),
        (result.construct)(&parts)
    );
    let checks: Vec<String> = (0..n)
        .map(|k| format!("a{k}.len() == 1 || a{k}.len() == out.len()"))
        .collect();
    let _ = writeln!(
        s,
        "    assert!({}, \"{name}_batch: argument lengths\");\n    ::gax::batch::run(K({}, out));\n}}\n",
        checks.join(" && "),
        names.join(", ")
    );
}

/// Emit `{name}_batch_soa`: the kernel with arguments of one kind in [`Soa`] storage (loaded
/// without transposes) and the others in slices, writing a `Soa` of the result's kind.
///
/// [`Soa`]: gax_core::batch::Soa
#[allow(clippy::too_many_lines)]
fn batch_form_soa(s: &mut String, name: &str, args: &[Shape], result: &Shape) {
    let (rkind, runit) = result.kind.clone().expect("a kind result");
    let n = args.len();
    let arg_ty = |k: usize| match &args[k].kind {
        Some((path, _)) => format!("::gax::batch::Soa<{path}, E>"),
        None => format!("[{}]", (args[k].ty)("E")),
    };
    let params: Vec<String> = (0..n).map(|k| format!("a{k}: &'a {}", arg_ty(k))).collect();
    let fields: Vec<String> = (0..n).map(|k| format!("&'a {}", arg_ty(k))).collect();
    let names: Vec<String> = (0..n).map(|k| format!("a{k}")).collect();
    let out_ty = format!("::gax::batch::Soa<{rkind}, E>");
    let _ = writeln!(
        s,
        "/// Batch form of [`{name}`] on struct-of-arrays storage: arguments of one kind come as\n\
         /// `Soa`, others as slices, and `out` (resized to the common length) is a `Soa`. An argument\n\
         /// of length 1 is broadcast.\n///\n/// # Panics\n/// If argument lengths other than 1 differ.\n\
         #[inline]\n#[allow(clippy::all, clippy::pedantic, unused_variables, unused_parens, non_snake_case, non_camel_case_types)]\n\
         pub fn {name}_batch_soa<'a, E: ::gax::batch::LaneElem>({}, out: &'a mut {out_ty}) {{",
        params.join(", ")
    );
    let _ = writeln!(
        s,
        "    struct K<'a, E: ::gax::batch::LaneElem>({}, &'a mut {out_ty});",
        fields.join(", ")
    );
    let _ = writeln!(
        s,
        "    impl<'a, E: ::gax::batch::LaneElem> ::gax::batch::Kernel<E> for K<'a, E> {{\n        type Output = ();\n        #[inline(always)]\n        fn run<L: ::gax::batch::Batch<Elem = E>>(self) {{\n            let K({}, out) = self;\n            let n = out.len();",
        names.join(", ")
    );
    for (k, a) in args.iter().enumerate() {
        let (splat, zero) = if let Some((path, unit)) = &a.kind {
            let wrap = |e: String| {
                if *unit {
                    format!("::gax::Unit::new_unchecked({e})")
                } else {
                    e
                }
            };
            (
                wrap(format!("::gax::batch::splat::<{path}, L>(a{k}.get(0))")),
                wrap(format!("<{path} as ::gax::Kind>::Mv::<(), L>::zero()")),
            )
        } else {
            let parts: Vec<String> = (0..a.len)
                .map(|j| format!("L::splat(({})[{j}])", (a.coeffs)(&format!("a{k}[0]"))))
                .collect();
            (
                (a.construct)(&parts),
                (a.construct)(&vec!["L::zero()".to_string(); a.len]),
            )
        };
        let _ = writeln!(
            s,
            "            let b{k} = a{k}.len() == 1;\n            let s{k}: {} = if b{k} {{ {splat} }} else {{ {zero} }};",
            (a.ty)("L")
        );
    }
    // Block by block, so that the loads and stores sit at offsets the compiler knows.
    let _ = writeln!(
        s,
        "            let blocks = n.div_ceil(::gax::batch::BLOCK);\n            for blk in 0..blocks {{"
    );
    for (k, a) in args.iter().enumerate() {
        if a.kind.is_some() {
            let _ = writeln!(
                s,
                "                let blk{k}: &[E] = if b{k} {{ &[] }} else {{ a{k}.block(blk) }};"
            );
        }
    }
    let _ = writeln!(
        s,
        "                let blko = out.block_mut(blk);\n                let mut j = 0;\n                while j < ::gax::batch::BLOCK {{\n                    let i = blk * ::gax::batch::BLOCK + j;\n                    if i >= n {{\n                        break;\n                    }}\n                    let m = (n - i).min(L::LANES);"
    );
    for (k, a) in args.iter().enumerate() {
        let load = if let Some((path, unit)) = &a.kind {
            let l = format!("::gax::batch::load_block::<{path}, L>(blk{k}, j)");
            if *unit {
                format!("::gax::Unit::new_unchecked({l})")
            } else {
                l
            }
        } else {
            let parts: Vec<String> = (0..a.len)
                .map(|j| {
                    format!(
                        "::gax::batch::column::<L>(m, #[inline(always)] |l| ({})[{j}])",
                        (a.coeffs)(&format!("a{k}[i + l]"))
                    )
                })
                .collect();
            (a.construct)(&parts)
        };
        let _ = writeln!(
            s,
            "                    let x{k}: {} = if b{k} {{ s{k} }} else {{ {load} }};",
            (a.ty)("L")
        );
    }
    let xs: Vec<String> = (0..n).map(|k| format!("x{k}")).collect();
    let store = if runit { "y.into_inner()" } else { "y" };
    let _ = writeln!(
        s,
        "                    let y: {} = {name}::<L>({});\n                    ::gax::batch::store_block::<{rkind}, L>(blko, j, &{store});\n                    j += L::LANES;\n                }}\n            }}\n        }}\n    }}",
        (result.ty)("L"),
        xs.join(", ")
    );
    let lens: Vec<String> = names.iter().map(|a| format!("{a}.len()")).collect();
    let _ = writeln!(
        s,
        "    let lens = [{}];\n    let n = lens.iter().copied().filter(|&l| l != 1).max().unwrap_or(1);\n    assert!(lens.iter().all(|&l| l == 1 || l == n), \"{name}_batch_soa: argument lengths\");\n    out.resize(n);\n    ::gax::batch::run(K({}, out));\n}}\n",
        lens.join(", "),
        names.join(", ")
    );
}
