//! Straight-line programs: the simplifier's output and the emitter's input.
//!
//! A program is a list of instructions over input variables, earlier temporaries and exact
//! constants. It can be re-expanded into polynomials, which is how every simplification is
//! verified to be exact.

use crate::poly::{Poly, Rational, Var};
use std::fmt::Write as _;

/// An instruction operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Operand {
    /// An input variable.
    Var(Var),
    /// The result of instruction `k`.
    Temp(usize),
    /// An exact constant.
    Const(Rational),
}

/// Elementary functions of one argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Func {
    /// `1 / x`
    Recip,
    /// `sqrt(x)`
    Sqrt,
    /// `sin(x)`
    Sin,
    /// `cos(x)`
    Cos,
    /// `sinh(x)`
    Sinh,
    /// `cosh(x)`
    Cosh,
    /// `ln(x)`
    Ln,
    /// `|x|`
    Abs,
    /// `e^x`
    Exp,
}

impl Func {
    /// The WGSL expression of the function applied to the expression `a`.
    pub fn wgsl(self, a: &str) -> String {
        match self {
            Func::Recip => format!("(1.0 / {a})"),
            Func::Ln => format!("log({a})"),
            f => format!("{}({a})", f.method()),
        }
    }

    /// Name of the `Real` method.
    pub fn method(self) -> &'static str {
        match self {
            Func::Recip => "recip",
            Func::Sqrt => "sqrt",
            Func::Sin => "sin",
            Func::Cos => "cos",
            Func::Sinh => "sinh",
            Func::Cosh => "cosh",
            Func::Ln => "ln",
            Func::Abs => "abs",
            Func::Exp => "exp",
        }
    }
}

/// One instruction; its result is `Operand::Temp(index)`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Instr {
    /// `a + b`
    Add(Operand, Operand),
    /// `a - b`
    Sub(Operand, Operand),
    /// `a * b`
    Mul(Operand, Operand),
    /// `-a`
    Neg(Operand),
    /// `a / b`
    Div(Operand, Operand),
    /// `atan2(a, b)`
    Atan2(Operand, Operand),
    /// `if a < b { x } else { y }`
    Select(Operand, Operand, Operand, Operand),
    /// `f(a)`
    Call(Func, Operand),
}

/// Operation counts of a program.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Cost {
    /// Multiplications, including multiplications by constants other than ±1.
    pub muls: usize,
    /// Additions and subtractions.
    pub adds: usize,
    /// Divisions and reciprocals.
    pub divs: usize,
    /// Negations.
    pub negs: usize,
    /// Other function calls.
    pub calls: usize,
}

impl Cost {
    /// Scalar cost used to compare programs: `mul = add = 1`, `div = 4`, `call = 8`.
    pub fn weight(&self) -> f64 {
        self.muls as f64 * 1.0001
            + self.adds as f64
            + 4.0 * self.divs as f64
            + 8.0 * self.calls as f64
            + 0.01 * self.negs as f64
    }
}

impl std::fmt::Display for Cost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} mul, {} add", self.muls, self.adds)?;
        if self.divs > 0 {
            write!(f, ", {} div", self.divs)?;
        }
        if self.calls > 0 {
            write!(f, ", {} call", self.calls)?;
        }
        Ok(())
    }
}

/// A straight-line program with a list of outputs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Program {
    /// Instructions in evaluation order.
    pub instrs: Vec<Instr>,
    /// Output operands.
    pub outputs: Vec<Operand>,
}

impl Program {
    /// Append an instruction and return its operand.
    pub fn push(&mut self, i: Instr) -> Operand {
        self.instrs.push(i);
        Operand::Temp(self.instrs.len() - 1)
    }

    /// Operation counts of the instructions that are actually used by the outputs.
    pub fn cost(&self) -> Cost {
        let live = self.live();
        let mut c = Cost::default();
        for (k, i) in self.instrs.iter().enumerate() {
            if !live[k] {
                continue;
            }
            match i {
                Instr::Add(..) | Instr::Sub(..) => c.adds += 1,
                Instr::Mul(a, b) => {
                    let unit =
                        |o: &Operand| matches!(o, Operand::Const(r) if r.abs() == Rational::ONE);
                    if !unit(a) && !unit(b) {
                        c.muls += 1;
                    } else {
                        c.negs += 1;
                    }
                }
                Instr::Neg(_) => c.negs += 1,
                Instr::Div(..) | Instr::Call(Func::Recip, _) => c.divs += 1,
                Instr::Call(..) | Instr::Atan2(..) | Instr::Select(..) => c.calls += 1,
            }
        }
        c
    }

    /// Which instructions contribute to the outputs.
    pub fn live(&self) -> Vec<bool> {
        let mut live = vec![false; self.instrs.len()];
        let mut stack: Vec<usize> = self
            .outputs
            .iter()
            .filter_map(|o| {
                if let Operand::Temp(k) = o {
                    Some(*k)
                } else {
                    None
                }
            })
            .collect();
        while let Some(k) = stack.pop() {
            if live[k] {
                continue;
            }
            live[k] = true;
            for o in operands(&self.instrs[k]) {
                if let Operand::Temp(j) = o {
                    stack.push(j);
                }
            }
        }
        live
    }

    /// Remove instructions not needed by any output, and renumber.
    pub fn compact(&mut self) {
        let live = self.live();
        let mut map = vec![usize::MAX; self.instrs.len()];
        let mut out = Vec::new();
        for (k, i) in self.instrs.iter().enumerate() {
            if live[k] {
                map[k] = out.len();
                out.push(remap(i, &map));
            }
        }
        self.instrs = out;
        for o in &mut self.outputs {
            if let Operand::Temp(k) = o {
                *k = map[*k];
            }
        }
    }

    /// Expand the outputs into polynomials. Panics on non-polynomial instructions.
    pub fn to_polys(&self) -> Vec<Poly> {
        self.to_polys_with(&|_| None)
    }

    /// Expand the outputs into polynomials; `leaf(k)` may name the value of instruction `k`
    /// (a non-polynomial instruction must be named).
    pub fn to_polys_with(&self, leaf: &impl Fn(usize) -> Option<Poly>) -> Vec<Poly> {
        let mut vals: Vec<Poly> = Vec::with_capacity(self.instrs.len());
        let get = |o: &Operand, vals: &Vec<Poly>| match o {
            Operand::Var(v) => Poly::var(*v),
            Operand::Temp(k) => vals[*k].clone(),
            Operand::Const(c) => Poly::constant(*c),
        };
        for (k, i) in self.instrs.iter().enumerate() {
            if let Some(p) = leaf(k) {
                vals.push(p);
                continue;
            }
            let p = match i {
                Instr::Add(a, b) => &get(a, &vals) + &get(b, &vals),
                Instr::Sub(a, b) => &get(a, &vals) - &get(b, &vals),
                Instr::Mul(a, b) => &get(a, &vals) * &get(b, &vals),
                Instr::Neg(a) => -&get(a, &vals),
                Instr::Div(a, b) => {
                    let d = get(b, &vals)
                        .as_constant()
                        .expect("division by a non-constant in to_polys");
                    get(a, &vals).scale(d.recip())
                }
                Instr::Atan2(..) | Instr::Call(..) | Instr::Select(..) => {
                    panic!("to_polys: non-polynomial instruction")
                }
            };
            vals.push(p);
        }
        self.outputs.iter().map(|o| get(o, &vals)).collect()
    }

    /// Evaluate in `f64`.
    pub fn eval(&self, var: &impl Fn(Var) -> f64) -> Vec<f64> {
        let mut vals: Vec<f64> = Vec::with_capacity(self.instrs.len());
        let get = |o: &Operand, vals: &Vec<f64>| match o {
            Operand::Var(v) => var(*v),
            Operand::Temp(k) => vals[*k],
            Operand::Const(c) => c.to_f64(),
        };
        for i in &self.instrs {
            let x = match i {
                Instr::Add(a, b) => get(a, &vals) + get(b, &vals),
                Instr::Sub(a, b) => get(a, &vals) - get(b, &vals),
                Instr::Mul(a, b) => get(a, &vals) * get(b, &vals),
                Instr::Neg(a) => -get(a, &vals),
                Instr::Div(a, b) => get(a, &vals) / get(b, &vals),
                Instr::Atan2(a, b) => get(a, &vals).atan2(get(b, &vals)),
                Instr::Select(a, b, x, y) => {
                    if get(a, &vals) < get(b, &vals) {
                        get(x, &vals)
                    } else {
                        get(y, &vals)
                    }
                }
                Instr::Call(f, a) => {
                    let x = get(a, &vals);
                    match f {
                        Func::Recip => 1.0 / x,
                        Func::Sqrt => x.sqrt(),
                        Func::Sin => x.sin(),
                        Func::Cos => x.cos(),
                        Func::Sinh => x.sinh(),
                        Func::Cosh => x.cosh(),
                        Func::Ln => x.ln(),
                        Func::Abs => x.abs(),
                        Func::Exp => x.exp(),
                    }
                }
            };
            vals.push(x);
        }
        self.outputs.iter().map(|o| get(o, &vals)).collect()
    }

    /// Evaluate in `f32`, rounding every operation (no fused multiply-adds): the program as
    /// scalar code runs it on a target without FMA.
    pub fn eval_f32(&self, var: &impl Fn(Var) -> f32) -> Vec<f32> {
        let mut vals: Vec<f32> = Vec::with_capacity(self.instrs.len());
        let get = |o: &Operand, vals: &Vec<f32>| match o {
            Operand::Var(v) => var(*v),
            Operand::Temp(k) => vals[*k],
            Operand::Const(c) => c.to_f64() as f32,
        };
        for i in &self.instrs {
            let x = match i {
                Instr::Add(a, b) => get(a, &vals) + get(b, &vals),
                Instr::Sub(a, b) => get(a, &vals) - get(b, &vals),
                Instr::Mul(a, b) => get(a, &vals) * get(b, &vals),
                Instr::Neg(a) => -get(a, &vals),
                Instr::Div(a, b) => get(a, &vals) / get(b, &vals),
                Instr::Atan2(a, b) => get(a, &vals).atan2(get(b, &vals)),
                Instr::Select(a, b, x, y) => {
                    if get(a, &vals) < get(b, &vals) {
                        get(x, &vals)
                    } else {
                        get(y, &vals)
                    }
                }
                Instr::Call(f, a) => {
                    let x = get(a, &vals);
                    match f {
                        Func::Recip => 1.0 / x,
                        Func::Sqrt => x.sqrt(),
                        Func::Sin => x.sin(),
                        Func::Cos => x.cos(),
                        Func::Sinh => x.sinh(),
                        Func::Cosh => x.cosh(),
                        Func::Ln => x.ln(),
                        Func::Abs => x.abs(),
                        Func::Exp => x.exp(),
                    }
                }
            };
            vals.push(x);
        }
        self.outputs.iter().map(|o| get(o, &vals)).collect()
    }

    /// A first-order bound on the forward error of evaluating the program in floating point
    /// with unit roundoff `u` (`2⁻²⁴` for `f32`), per output: `|computed − exact| ≤ bound`,
    /// given bounds `mag` on the magnitudes of the (exact) inputs.
    ///
    /// Standard running error analysis: every operation contributes `u` times a bound on its
    /// result, and errors propagate through `+`, `−`, `×` and division by constants; a
    /// constant that is not a dyadic rational contributes its own rounding. Fusing a product
    /// into a sum (`mul_add`) only removes roundings, so the bound holds for fused code too.
    /// Non-polynomial steps give an infinite bound.
    pub fn error_bound(&self, mag: &impl Fn(Var) -> f64, u: f64) -> Vec<f64> {
        let mut vals: Vec<(f64, f64)> = Vec::with_capacity(self.instrs.len());
        let get = |o: &Operand, vals: &Vec<(f64, f64)>| match o {
            Operand::Var(v) => (mag(*v).abs(), 0.0),
            Operand::Temp(k) => vals[*k],
            Operand::Const(c) => {
                let x = c.to_f64().abs();
                let dyadic = c.den().count_ones() == 1 && c.num().unsigned_abs() < (1 << 24);
                (x, if dyadic { 0.0 } else { u * x })
            }
        };
        for i in &self.instrs {
            let r = match i {
                Instr::Add(a, b) | Instr::Sub(a, b) => {
                    let ((ma, ea), (mb, eb)) = (get(a, &vals), get(b, &vals));
                    let m = ma + mb;
                    (m, ea + eb + u * m)
                }
                Instr::Mul(a, b) => {
                    let ((ma, ea), (mb, eb)) = (get(a, &vals), get(b, &vals));
                    let m = ma * mb;
                    (m, ma * eb + mb * ea + ea * eb + u * m)
                }
                Instr::Neg(a) => get(a, &vals),
                Instr::Div(a, b) => match b {
                    Operand::Const(c) if !c.is_zero() => {
                        let (ma, ea) = get(a, &vals);
                        let d = c.to_f64().abs();
                        (ma / d, ea / d + u * ma / d)
                    }
                    _ => (f64::INFINITY, f64::INFINITY),
                },
                Instr::Atan2(..) | Instr::Select(..) | Instr::Call(..) => {
                    (f64::INFINITY, f64::INFINITY)
                }
            };
            vals.push(r);
        }
        self.outputs.iter().map(|o| get(o, &vals).1).collect()
    }

    /// The largest [`error_bound`](Self::error_bound) over the outputs for inputs of
    /// magnitude at most 1, in units of `u`: the score the simplifier uses to compare
    /// candidates.
    pub fn error_score(&self) -> f64 {
        self.error_bound(&|_| 1.0, 1.0)
            .into_iter()
            .fold(0.0, f64::max)
    }

    /// Emit the instructions as Rust `let` statements over a coefficient type `T`.
    ///
    /// `var` renders an input variable, and temporaries are named `{prefix}{k}`.
    pub fn emit_lets(&self, var: &impl Fn(Var) -> String, prefix: &str, out: &mut String) {
        self.emit_lets_to(Target::Rust, var, prefix, out);
    }

    /// The outputs as expressions, after [`Program::emit_lets`] with the same `var` and
    /// `prefix`.
    pub fn render_outputs(&self, var: &impl Fn(Var) -> String, prefix: &str) -> Vec<String> {
        self.outputs
            .iter()
            .map(|o| render(o, var, prefix))
            .collect()
    }

    /// Emit the instructions as `let` statements in the language `target`.
    ///
    /// Single-use products feeding a sum are fused into a multiply-add: `a.mul_add(b, c)` in
    /// Rust, `fma(a, b, c)` in WGSL (unless the target disables it). Rust statements are
    /// indented for a method body (8 spaces), WGSL statements for a function body (4).
    #[allow(clippy::too_many_lines)]
    pub fn emit_lets_to(
        &self,
        target: Target,
        var: &impl Fn(Var) -> String,
        prefix: &str,
        out: &mut String,
    ) {
        let live = self.live();
        // Uses of each temporary (outputs count), to fuse single-use products into their sum.
        let mut uses = vec![0usize; self.instrs.len()];
        for (k, i) in self.instrs.iter().enumerate() {
            if live[k] {
                for o in operands(i) {
                    if let Operand::Temp(j) = o {
                        uses[j] += 1;
                    }
                }
            }
        }
        for o in &self.outputs {
            if let Operand::Temp(j) = o {
                uses[*j] += 1;
            }
        }
        let single_mul = |o: &Operand| match o {
            Operand::Temp(j) if uses[*j] == 1 => match &self.instrs[*j] {
                Instr::Mul(x, y)
                    if !matches!(x, Operand::Const(_)) || !matches!(y, Operand::Const(_)) =>
                {
                    Some((*j, *x, *y))
                }
                _ => None,
            },
            _ => None,
        };
        // Decide the fusions first, so the fused products are not emitted.
        let mut fused = vec![false; self.instrs.len()];
        let mut plan: Vec<Option<String>> = vec![None; self.instrs.len()];
        let r = |o: &Operand| render_to(target, o, var, prefix);
        let fuse = match target {
            Target::Rust => true,
            Target::RustPlain => false,
            Target::Wgsl { fma } => fma,
        };
        // `x * y + z`, with the negations `nx` (of `x`) and `nz` (of `z`).
        let mul_add = |x: &Operand, y: &Operand, z: &Operand, nx: bool, nz: bool| {
            let n = |neg: bool, e: String| if neg { format!("-{e}") } else { e };
            match target {
                Target::Rust | Target::RustPlain => {
                    let x = if nx { format!("(-{})", r(x)) } else { r(x) };
                    format!("{x}.mul_add({}, {})", r(y), n(nz, r(z)))
                }
                Target::Wgsl { .. } => {
                    format!("fma({}, {}, {})", n(nx, r(x)), r(y), n(nz, r(z)))
                }
            }
        };
        for (k, i) in self.instrs.iter().enumerate() {
            if !live[k] || !fuse {
                continue;
            }
            let fma = match i {
                Instr::Add(a, b) => single_mul(a)
                    .map(|(j, x, y)| (j, mul_add(&x, &y, b, false, false)))
                    .or_else(|| {
                        single_mul(b).map(|(j, x, y)| (j, mul_add(&x, &y, a, false, false)))
                    }),
                Instr::Sub(a, b) => single_mul(b)
                    .map(|(j, x, y)| (j, mul_add(&x, &y, a, true, false)))
                    .or_else(|| {
                        single_mul(a).map(|(j, x, y)| (j, mul_add(&x, &y, b, false, true)))
                    }),
                _ => None,
            };
            if let Some((j, e)) = fma
                && !fused[j]
            {
                fused[j] = true;
                plan[k] = Some(e);
            }
        }
        let mut stmts: Vec<(usize, String)> = Vec::new();
        for (k, i) in self.instrs.iter().enumerate() {
            if !live[k] || fused[k] {
                continue;
            }
            let rhs = match (&plan[k], i) {
                (Some(e), _) => e.clone(),
                (None, Instr::Add(a, b)) => format!("{} + {}", r(a), r(b)),
                (None, Instr::Sub(a, b)) => format!("{} - {}", r(a), r(b)),
                (None, Instr::Mul(a, b)) => format!("{} * {}", r(a), r(b)),
                (None, Instr::Neg(a)) => format!("-{}", r(a)),
                (None, Instr::Div(a, b)) => format!("{} / {}", r(a), r(b)),
                (None, Instr::Atan2(a, b)) => match target {
                    Target::Rust | Target::RustPlain => format!("{}.atan2({})", r(a), r(b)),
                    Target::Wgsl { .. } => format!("atan2({}, {})", r(a), r(b)),
                },
                (None, Instr::Select(a, b, x, y)) => match target {
                    Target::Rust | Target::RustPlain => {
                        format!("T::select_lt({}, {}, {}, {})", r(a), r(b), r(x), r(y))
                    }
                    // WGSL's select takes the false value first.
                    Target::Wgsl { .. } => {
                        format!("select({}, {}, {} < {})", r(y), r(x), r(a), r(b))
                    }
                },
                (None, Instr::Call(f, a)) => match target {
                    Target::Rust | Target::RustPlain => format!("{}.{}()", r(a), f.method()),
                    Target::Wgsl { .. } => f.wgsl(&r(a)),
                },
            };
            stmts.push((k, rhs));
        }
        if matches!(target, Target::Wgsl { .. }) {
            for (k, rhs) in &stmts {
                let _ = writeln!(out, "    let {prefix}{k} = {rhs};");
            }
            return;
        }
        if stmts.len() <= LETS_MAX {
            for (k, rhs) in &stmts {
                let _ = writeln!(out, "        let {prefix}{k} = {rhs};");
            }
            return;
        }
        // Long programs in blocks of at most `LETS_MAX` statements, each handing the values
        // later ones use to the rest as one tuple: every `let` opens a scope that lasts to the
        // end of the function, and rustc's debug info recurses once per level (thousands of
        // levels overflow its stack).
        let refs = |k: usize| -> Vec<usize> {
            let mut r = Vec::new();
            for o in operands(&self.instrs[k]) {
                if let Operand::Temp(j) = o {
                    if fused[j] {
                        for p in operands(&self.instrs[j]) {
                            if let Operand::Temp(q) = p {
                                r.push(q);
                            }
                        }
                    } else {
                        r.push(j);
                    }
                }
            }
            r
        };
        // The last statement that reads each temporary (outputs read it after all).
        let mut last_use = vec![0usize; self.instrs.len()];
        for (at, (k, _)) in stmts.iter().enumerate() {
            for j in refs(*k) {
                last_use[j] = last_use[j].max(at);
            }
        }
        for o in &self.outputs {
            if let Operand::Temp(j) = o {
                last_use[*j] = usize::MAX;
            }
        }
        let chunks: Vec<&[(usize, String)]> = stmts.chunks(LETS_MAX).collect();
        let last = chunks.len() - 1;
        for (c, chunk) in chunks.iter().enumerate() {
            let end = (c + 1) * LETS_MAX;
            let names: Vec<String> = chunk
                .iter()
                .filter(|(k, _)| last_use[*k] >= end)
                .map(|(k, _)| format!("{prefix}{k}"))
                .collect();
            if c == last || names.is_empty() {
                for (k, rhs) in *chunk {
                    let _ = writeln!(out, "        let {prefix}{k} = {rhs};");
                }
                continue;
            }
            let tuple = format!("({},)", names.join(", "));
            let _ = writeln!(out, "        let {tuple} = {{");
            for (k, rhs) in *chunk {
                let _ = writeln!(out, "            let {prefix}{k} = {rhs};");
            }
            let _ = writeln!(out, "            {tuple}\n        }};");
        }
    }
}

/// The most `let` statements a Rust program is written with in one block (see
/// [`Program::emit_lets_to`]).
pub const LETS_MAX: usize = 512;

/// The language of emitted code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// Rust, generic over a coefficient type `T: Real`, with single-use products feeding a sum
    /// fused into `mul_add` (a hardware fused multiply-add where the target has one).
    Rust,
    /// The same Rust without `mul_add`: plain products and sums, which LLVM's SLP vectorizer
    /// may pack into SIMD more readily for a kernel called once, depending on the compiler
    /// version (performance.md).
    RustPlain,
    /// WGSL, in `f32`. With `fma`, single-use products feeding a sum become `fma(a, b, c)`;
    /// without it they stay `a * b + c`.
    Wgsl {
        /// Fuse multiply-adds.
        fma: bool,
    },
}

/// Render an operand as an expression in the language `target`.
pub fn render_to(
    target: Target,
    o: &Operand,
    var: &impl Fn(Var) -> String,
    prefix: &str,
) -> String {
    match (target, o) {
        (Target::Wgsl { .. }, Operand::Const(c)) => wgsl_const(*c),
        _ => render(o, var, prefix),
    }
}

/// An exact constant as a WGSL abstract-float expression, so that the shader compiler rounds
/// it to `f32` once: `2.0`, `(-2.0)`, `(1.0 / 3.0)`.
pub fn wgsl_const(c: Rational) -> String {
    let lit = |n: i128| format!("{n}.0");
    if c.is_integer() {
        if c.num() < 0 {
            format!("({})", lit(c.num()))
        } else {
            lit(c.num())
        }
    } else {
        format!("({} / {})", lit(c.num()), lit(c.den()))
    }
}

/// Render an operand as a Rust expression.
pub fn render(o: &Operand, var: &impl Fn(Var) -> String, prefix: &str) -> String {
    match o {
        Operand::Var(v) => var(*v),
        Operand::Temp(k) => format!("{prefix}{k}"),
        Operand::Const(c) => {
            if c.is_integer() {
                format!("T::from_i64({})", c.num())
            } else {
                match (i64::try_from(c.num()), i64::try_from(c.den())) {
                    (Ok(n), Ok(d)) => format!("T::from_ratio({n}, {d})"),
                    _ => format!("T::from_f64({}.0 / {}.0)", c.num(), c.den()),
                }
            }
        }
    }
}

fn operands(i: &Instr) -> Vec<Operand> {
    match i {
        Instr::Add(a, b)
        | Instr::Sub(a, b)
        | Instr::Mul(a, b)
        | Instr::Div(a, b)
        | Instr::Atan2(a, b) => {
            vec![*a, *b]
        }
        Instr::Neg(a) | Instr::Call(_, a) => vec![*a],
        Instr::Select(a, b, x, y) => vec![*a, *b, *x, *y],
    }
}

fn remap(i: &Instr, map: &[usize]) -> Instr {
    let m = |o: &Operand| match o {
        Operand::Temp(k) => Operand::Temp(map[*k]),
        other => *other,
    };
    match i {
        Instr::Add(a, b) => Instr::Add(m(a), m(b)),
        Instr::Sub(a, b) => Instr::Sub(m(a), m(b)),
        Instr::Mul(a, b) => Instr::Mul(m(a), m(b)),
        Instr::Neg(a) => Instr::Neg(m(a)),
        Instr::Div(a, b) => Instr::Div(m(a), m(b)),
        Instr::Atan2(a, b) => Instr::Atan2(m(a), m(b)),
        Instr::Select(a, b, x, y) => Instr::Select(m(a), m(b), m(x), m(y)),
        Instr::Call(f, a) => Instr::Call(*f, m(a)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(k: Var) -> Operand {
        Operand::Var(k)
    }
    fn c(n: i128, d: i128) -> Operand {
        Operand::Const(Rational::new(n, d))
    }

    /// One of every instruction, the fusions both ways, and the constant forms.
    fn sample() -> Program {
        let mut p = Program::default();
        let t0 = p.push(Instr::Mul(v(0), v(1)));
        let t1 = p.push(Instr::Add(t0, v(2))); // fused: x0 x1 + x2
        let t2 = p.push(Instr::Mul(v(1), v(2)));
        let t3 = p.push(Instr::Sub(t1, t2)); // fused: -(x1) x2 + t1
        let t4 = p.push(Instr::Neg(t3));
        let t5 = p.push(Instr::Div(t4, c(1, 3)));
        let t6 = p.push(Instr::Atan2(t5, v(0)));
        let t7 = p.push(Instr::Select(v(0), v(1), t6, c(-2, 1)));
        let t8 = p.push(Instr::Call(Func::Recip, t7));
        let t9 = p.push(Instr::Call(Func::Ln, t8));
        let t10 = p.push(Instr::Call(Func::Sqrt, t9));
        let t11 = p.push(Instr::Mul(t10, c(-5, 7)));
        p.outputs = vec![t11, t4];
        p
    }

    fn lets(target: Target) -> String {
        let mut s = String::new();
        sample().emit_lets_to(target, &|k| format!("x{k}"), "t", &mut s);
        s
    }

    #[test]
    fn wgsl_golden() {
        assert_eq!(
            lets(Target::Wgsl { fma: true }),
            "    let t1 = fma(x0, x1, x2);
    let t3 = fma(-x1, x2, t1);
    let t4 = -t3;
    let t5 = t4 / (1.0 / 3.0);
    let t6 = atan2(t5, x0);
    let t7 = select((-2.0), t6, x0 < x1);
    let t8 = (1.0 / t7);
    let t9 = log(t8);
    let t10 = sqrt(t9);
    let t11 = t10 * (-5.0 / 7.0);
"
        );
    }

    #[test]
    fn wgsl_golden_without_fma() {
        assert_eq!(
            lets(Target::Wgsl { fma: false }),
            "    let t0 = x0 * x1;
    let t1 = t0 + x2;
    let t2 = x1 * x2;
    let t3 = t1 - t2;
    let t4 = -t3;
    let t5 = t4 / (1.0 / 3.0);
    let t6 = atan2(t5, x0);
    let t7 = select((-2.0), t6, x0 < x1);
    let t8 = (1.0 / t7);
    let t9 = log(t8);
    let t10 = sqrt(t9);
    let t11 = t10 * (-5.0 / 7.0);
"
        );
    }

    /// The Rust form is what `emit_lets` has always written.
    #[test]
    fn rust_golden() {
        assert_eq!(
            lets(Target::Rust),
            "        let t1 = x0.mul_add(x1, x2);
        let t3 = (-x1).mul_add(x2, t1);
        let t4 = -t3;
        let t5 = t4 / T::from_ratio(1, 3);
        let t6 = t5.atan2(x0);
        let t7 = T::select_lt(x0, x1, t6, T::from_i64(-2));
        let t8 = t7.recip();
        let t9 = t8.ln();
        let t10 = t9.sqrt();
        let t11 = t10 * T::from_ratio(-5, 7);
"
        );
    }

    #[test]
    fn wgsl_constants() {
        assert_eq!(wgsl_const(Rational::new(3, 1)), "3.0");
        assert_eq!(wgsl_const(Rational::new(-3, 1)), "(-3.0)");
        assert_eq!(wgsl_const(Rational::new(-1, 2)), "(-1.0 / 2.0)");
        assert_eq!(wgsl_const(Rational::new(0, 1)), "0.0");
    }
}
