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
                $(
                    let ($a, info, conds) = arg_info::<$A>($k);
                    args.push(info);
                    conditions.extend(conds);
                )+
                let result = self($($a),+);
                Traced {
                    args,
                    conditions,
                    outputs: { let mut v = Vec::new(); result.for_each(&mut |c| v.push(c)); v },
                    result_type: result_type::<R>("T"),
                    construct: Box::new(|outs| construct::<R>(outs)),
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
}

impl Tracer {
    /// A tracer with no kernels.
    pub fn new() -> Tracer {
        Tracer::default()
    }

    /// Trace `f` and add a fused kernel named `name`.
    ///
    /// # Panics
    /// If `f` branches on a coefficient, or if the simplifier fails its own exactness check
    /// (a bug).
    #[allow(clippy::needless_pass_by_value)] // a closure literal is the natural argument
    pub fn kernel<Args, F: TraceFn<Args>>(&mut self, name: &str, f: F) -> &mut Tracer {
        // Trace under several expansion limits and keep the cheapest verified program. A
        // limit of 0 records the computation as written (each product and sum a node, with
        // constants folded and common subexpressions shared), so the result is never worse
        // than the generic code; no limit expands everything into polynomials, which exposes
        // cancellations and type conditions across steps.
        let mut best: Option<(crate::slp::Program, Traced, Vec<VarDef>)> = None;
        for limit in [Some(0), Some(1), Some(8), Some(32), Some(128), None] {
            Sym::reset_with_limit(limit);
            let traced = f.run();
            let outputs: Vec<Poly> = traced.outputs.iter().map(|s| s.poly()).collect();
            let stages = stages_of_arena();
            let mut relations = traced.conditions.clone();
            relations.extend(Sym::atom_relations());
            let (prog, _env) = cse::compile_staged_best(&outputs, &stages, &relations);
            let defs: Vec<VarDef> = (0..Sym::var_count() as Var).map(Sym::var_def).collect();
            if best
                .as_ref()
                .is_none_or(|(p, ..)| prog.cost().weight() < p.cost().weight())
            {
                best = Some((prog, traced, defs));
            }
        }
        let (prog, traced, defs) = best.expect("at least one limit");
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
            VarDef::Atom { .. } | VarDef::Atan2 { .. } | VarDef::Node { .. } => {
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
            "#[inline]\n#[allow(clippy::all, clippy::pedantic, unused_variables, unused_parens, non_snake_case)]"
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
