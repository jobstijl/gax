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
}

impl Func {
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
                Instr::Call(..) | Instr::Atan2(..) => c.calls += 1,
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

    /// Remove dead instructions and renumber.
    pub fn compact(&mut self) {
        self.compact_keep_live_of(self.outputs.len());
    }

    /// Remove instructions not needed by any output, renumbering all outputs. (The
    /// argument is accepted for call-site clarity; every listed output is kept alive.)
    pub fn compact_keep_live_of(&mut self, _primary: usize) {
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
                Instr::Atan2(..) | Instr::Call(..) => {
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
                    }
                }
            };
            vals.push(x);
        }
        self.outputs.iter().map(|o| get(o, &vals)).collect()
    }

    /// Emit the instructions as Rust `let` statements over a coefficient type `T`.
    ///
    /// `var` renders an input variable, and temporaries are named `{prefix}{k}`.
    pub fn emit_lets(&self, var: &impl Fn(Var) -> String, prefix: &str, out: &mut String) {
        let live = self.live();
        for (k, i) in self.instrs.iter().enumerate() {
            if !live[k] {
                continue;
            }
            let r = |o: &Operand| render(o, var, prefix);
            let rhs = match i {
                Instr::Add(a, b) => format!("{} + {}", r(a), r(b)),
                Instr::Sub(a, b) => format!("{} - {}", r(a), r(b)),
                Instr::Mul(a, b) => format!("{} * {}", r(a), r(b)),
                Instr::Neg(a) => format!("-{}", r(a)),
                Instr::Div(a, b) => format!("{} / {}", r(a), r(b)),
                Instr::Atan2(a, b) => format!("{}.atan2({})", r(a), r(b)),
                Instr::Call(f, a) => format!("{}.{}()", r(a), f.method()),
            };
            let _ = writeln!(out, "        let {prefix}{k} = {rhs};");
        }
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
                format!("T::from_f64({}.0 / {}.0)", c.num(), c.den())
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
        Instr::Call(f, a) => Instr::Call(*f, m(a)),
    }
}
