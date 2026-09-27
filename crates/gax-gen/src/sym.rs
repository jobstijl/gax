//! `Sym`: the symbolic coefficient type used by build-time tracing.
//!
//! A `Sym` is a `Copy` handle into a thread-local arena of expanded polynomials with exact
//! rational coefficients. The arena is hash-consed, so equal polynomials share one handle and
//! `==` on `Sym` is exact algebraic equality. Operations that are not polynomial (`1/x`,
//! `sqrt`, `sin`, ...) create *atoms*: fresh variables that carry their definition, again
//! hash-consed, so the reciprocal of the same denominator is computed once in the emitted
//! code (reciprocal hoisting at the source).
//!
//! Tracing runs ordinary generic code with `T = Sym`. Code that branches on coefficient
//! values cannot be traced: [`Real::select_lt`] on `Sym` panics.

use crate::poly::{Monomial, Poly, Rational, Var};
use crate::slp::Func;
use gax_core::{Coef, Real};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A symbolic coefficient: a handle to a polynomial in the thread-local arena.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sym(u32);

/// What a variable stands for.
#[derive(Clone, Debug, PartialEq)]
pub enum VarDef {
    /// Coefficient `index` of kernel argument `arg`.
    Input {
        /// Argument position.
        arg: usize,
        /// Coefficient index within the argument.
        index: usize,
    },
    /// `func(arg)`.
    Atom {
        /// The function.
        func: Func,
        /// Its argument.
        arg: Sym,
    },
    /// `atan2(y, x)`.
    Atan2 {
        /// `y`
        y: Sym,
        /// `x`
        x: Sym,
    },
    /// `if a < b { x } else { y }`, lane by lane.
    Select {
        /// Compared value.
        a: Sym,
        /// Compared against.
        b: Sym,
        /// Result when `a < b`.
        x: Sym,
        /// Result otherwise.
        y: Sym,
    },
    /// A constant that is not an exact dyadic rational, such as `0.1` or `T::epsilon()`.
    Constant(ConstKind),
    /// An operation whose expanded result exceeded the expansion limit, kept as a node.
    Node {
        /// The operation.
        op: NodeOp,
        /// Left operand.
        a: Sym,
        /// Right operand.
        b: Sym,
    },
}

/// Operation of an opaque [`VarDef::Node`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeOp {
    /// `a + b`
    Add,
    /// `a - b`
    Sub,
    /// `a * b`
    Mul,
}

/// A constant that is emitted by name.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConstKind {
    /// A floating-point literal.
    Float(f64),
    /// The machine epsilon of the coefficient type.
    Epsilon,
}

#[derive(Default)]
struct Arena {
    polys: Vec<Poly>,
    index: HashMap<Poly, u32>,
    vars: Vec<VarDef>,
    atom_index: HashMap<(Func, Sym), Var>,
    atan2_index: HashMap<(Sym, Sym), Var>,
    select_index: HashMap<(Sym, Sym, Sym, Sym), Var>,
    /// Relations `p = 0` implied by atom definitions (`s² - x`, `r x - 1`).
    relations: Vec<Poly>,
    node_index: HashMap<(NodeOp, Sym, Sym), Var>,
    /// Results with more terms than this become opaque nodes (`None`: never).
    limit: Option<usize>,
    /// Keep constants opaque (as the running program sees them) instead of folding them.
    opaque_constants: bool,
}

thread_local! {
    static ARENA: RefCell<Arena> = RefCell::new(Arena::default());
}

impl Arena {
    fn intern(&mut self, p: Poly) -> Sym {
        if let Some(&h) = self.index.get(&p) {
            return Sym(h);
        }
        let h = self.polys.len() as u32;
        self.polys.push(p.clone());
        self.index.insert(p, h);
        Sym(h)
    }

    fn new_var(&mut self, def: VarDef) -> Var {
        self.vars.push(def);
        (self.vars.len() - 1) as Var
    }

    /// Replace `s²` by the radicand for sqrt atoms and `r·x` by 1 for reciprocal atoms of a
    /// single variable, keeping polynomials in a reduced form.
    fn normalize(&self, p: Poly) -> Poly {
        let mut p = p;
        loop {
            let mut changed = false;
            let mut out = Poly::zero();
            for (m, c) in &p.0 {
                let mut m2 = m.clone();
                let mut factor = Poly::constant(*c);
                'scan: loop {
                    for &v in &m2.0 {
                        match self.vars.get(v as usize) {
                            Some(VarDef::Atom {
                                func: Func::Sqrt,
                                arg,
                            }) if m2.power(v) >= 2 => {
                                m2 = m2.div(&Monomial(vec![v, v])).expect("power checked");
                                factor = &factor * &self.polys[arg.0 as usize];
                                changed = true;
                                continue 'scan;
                            }
                            Some(VarDef::Atom {
                                func: Func::Recip,
                                arg,
                            }) => {
                                let a = &self.polys[arg.0 as usize];
                                if a.len() == 1 {
                                    let (am, ac) = a.0.iter().next().expect("one term");
                                    if let Some(rest) =
                                        m2.div(&Monomial::var(v)).and_then(|r| r.div(am))
                                    {
                                        m2 = rest;
                                        factor = factor.scale(ac.recip());
                                        changed = true;
                                        continue 'scan;
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    break;
                }
                for (fm, fc) in &factor.0 {
                    out.add_term(fm.mul(&m2), *fc);
                }
            }
            p = out;
            if !changed {
                return p;
            }
        }
    }
}

fn with<R>(f: impl FnOnce(&mut Arena) -> R) -> R {
    ARENA.with(|a| f(&mut a.borrow_mut()))
}

impl Sym {
    /// The polynomial behind this handle.
    pub fn poly(self) -> Poly {
        with(|a| a.polys[self.0 as usize].clone())
    }

    /// Intern a polynomial.
    pub fn from_poly(p: Poly) -> Sym {
        with(|a| {
            let p = a.normalize(p);
            a.intern(p)
        })
    }

    /// A fresh input variable for coefficient `index` of argument `arg`.
    pub fn input(arg: usize, index: usize) -> Sym {
        let v = with(|a| a.new_var(VarDef::Input { arg, index }));
        Sym::from_poly(Poly::var(v))
    }

    /// The definition of a variable.
    pub fn var_def(v: Var) -> VarDef {
        with(|a| a.vars[v as usize].clone())
    }

    /// Number of variables created so far.
    pub fn var_count() -> usize {
        with(|a| a.vars.len())
    }

    /// Relations implied by the atoms created so far.
    pub fn atom_relations() -> Vec<Poly> {
        with(|a| a.relations.clone())
    }

    /// Clear the arena. Handles created before are invalidated.
    pub fn reset() {
        with(|a| *a = Arena::default());
    }

    /// Clear the arena and set the expansion limit: a sum or product whose expanded
    /// polynomial has more than `limit` terms is kept as an opaque node instead. `None`
    /// expands everything.
    pub fn reset_with_limit(limit: Option<usize>) {
        with(|a| {
            *a = Arena::default();
            a.limit = limit;
        });
    }

    /// Clear the arena for a trace that models the program as it runs: every operation is a
    /// node and constants stay opaque, so nothing folds (IEEE semantics forbid folding
    /// `0 * x` and `x + 0`). Used to report the cost of the unfused code.
    pub fn reset_runtime_model() {
        with(|a| {
            *a = Arena::default();
            a.limit = Some(0);
            a.opaque_constants = true;
        });
    }

    fn runtime_constant(value: f64) -> Option<Sym> {
        let opaque = with(|a| a.opaque_constants);
        opaque.then(|| Sym::constant_atom(ConstKind::Float(value)))
    }

    /// The exact value, if this is a constant.
    pub fn as_constant(self) -> Option<Rational> {
        self.poly().as_constant()
    }

    /// A constant from its value: exact when it is a small dyadic rational (`0.5`, `3`),
    /// otherwise a named float constant. Large exact dyadics (such as the nearest `f64` to
    /// `π/4`) would overflow the exact arithmetic once multiplied a few times.
    fn value(f: f64) -> Sym {
        match Rational::from_f64(f) {
            Some(r) if r.num().abs() < 1 << 40 && r.den() < 1 << 40 => {
                Sym::from_poly(Poly::constant(r))
            }
            _ => Sym::constant_atom(ConstKind::Float(f)),
        }
    }

    /// The value, if this depends on no input: an exact constant, or a polynomial in named
    /// float constants (folded calls). Used to fold calls and selects at trace time.
    pub fn const_value(self) -> Option<f64> {
        if let Some(c) = self.as_constant() {
            return Some(c.to_f64());
        }
        let p = self.poly();
        with(|a| {
            let mut sum = 0.0;
            for (m, c) in &p.0 {
                let mut t = c.to_f64();
                for &v in &m.0 {
                    match a.vars[v as usize] {
                        VarDef::Constant(ConstKind::Float(f)) => t *= f,
                        _ => return None,
                    }
                }
                sum += t;
            }
            Some(sum)
        })
    }

    fn atom(func: Func, arg: Sym) -> Sym {
        // A call on a constant is evaluated now, not in every run of the kernel: `|c|`
        // exactly, the others in `f64` (at least as accurate as the kernel's own arithmetic).
        // The runtime model keeps the calls, as the generic code makes them.
        if !with(|a| a.opaque_constants)
            && let Some(x) = arg.const_value()
        {
            if let (Func::Abs, Some(c)) = (func, arg.as_constant()) {
                return Sym::from_poly(Poly::constant(c.abs()));
            }
            let folded = match func {
                Func::Abs => Some(x.abs()),
                Func::Sqrt if x >= 0.0 => Some(x.sqrt()),
                Func::Sin => Some(x.sin()),
                Func::Cos => Some(x.cos()),
                Func::Sinh => Some(x.sinh()),
                Func::Cosh => Some(x.cosh()),
                Func::Ln if x > 0.0 => Some(x.ln()),
                Func::Recip if x != 0.0 => Some(1.0 / x),
                _ => None,
            };
            if let Some(f) = folded {
                return Sym::value(f);
            }
        }
        let v = with(|a| {
            if let Some(&v) = a.atom_index.get(&(func, arg)) {
                return v;
            }
            let v = a.new_var(VarDef::Atom { func, arg });
            a.atom_index.insert((func, arg), v);
            let x = a.polys[arg.0 as usize].clone();
            match func {
                Func::Sqrt => {
                    let mut r = Poly::term(Monomial(vec![v, v]), Rational::ONE);
                    r = &r - &x;
                    a.relations.push(r);
                }
                Func::Recip => {
                    let r = &(&Poly::var(v) * &x) - &Poly::constant(Rational::ONE);
                    a.relations.push(r);
                }
                _ => {}
            }
            v
        });
        Sym::from_poly(Poly::var(v))
    }

    fn constant_atom(c: ConstKind) -> Sym {
        let v = with(|a| {
            // Constants are few; a linear search keeps floats out of hash keys.
            let found = a.vars.iter().position(|d| *d == VarDef::Constant(c));
            match found {
                Some(i) => i as Var,
                None => a.new_var(VarDef::Constant(c)),
            }
        });
        Sym::from_poly(Poly::var(v))
    }

    fn binary(self, o: Sym, op: NodeOp, f: impl FnOnce(&Poly, &Poly) -> Poly) -> Sym {
        enum R {
            Poly(Poly),
            Node(Var),
        }
        let r = with(|a| {
            let x = &a.polys[self.0 as usize];
            let y = &a.polys[o.0 as usize];
            // Scaling by a constant never grows a polynomial; keep it expanded.
            let trivial = x.as_constant().is_some() || y.as_constant().is_some();
            let p = f(x, y);
            let over = |limit: usize| a.opaque_constants || p.len() > limit;
            match a.limit {
                Some(limit) if !trivial && over(limit) => {
                    let key = (op, self, o);
                    if let Some(&v) = a.node_index.get(&key) {
                        return R::Node(v);
                    }
                    let v = a.new_var(VarDef::Node { op, a: self, b: o });
                    a.node_index.insert(key, v);
                    R::Node(v)
                }
                _ => R::Poly(p),
            }
        });
        match r {
            R::Poly(p) => {
                let s = Sym::from_poly(p);
                // Arithmetic on named constants only is done now, not at run time.
                if s.as_constant().is_none()
                    && !with(|a| a.opaque_constants)
                    && let Some(f) = s.const_value()
                {
                    return Sym::value(f);
                }
                s
            }
            R::Node(v) => Sym::from_poly(Poly::var(v)),
        }
    }
}

impl fmt::Debug for Sym {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sym({:?})", self.poly())
    }
}

impl Add for Sym {
    type Output = Sym;
    fn add(self, o: Sym) -> Sym {
        self.binary(o, NodeOp::Add, |x, y| x + y)
    }
}

impl Sub for Sym {
    type Output = Sym;
    fn sub(self, o: Sym) -> Sym {
        self.binary(o, NodeOp::Sub, |x, y| x - y)
    }
}

impl Mul for Sym {
    type Output = Sym;
    fn mul(self, o: Sym) -> Sym {
        self.binary(o, NodeOp::Mul, |x, y| x * y)
    }
}

impl Neg for Sym {
    type Output = Sym;
    fn neg(self) -> Sym {
        Sym::from_poly(-&self.poly())
    }
}

impl Div for Sym {
    type Output = Sym;
    fn div(self, o: Sym) -> Sym {
        match o.as_constant() {
            Some(c) if !c.is_zero() => Sym::from_poly(self.poly().scale(c.recip())),
            _ => self * o.recip(),
        }
    }
}

impl Coef for Sym {
    fn zero() -> Sym {
        Sym::runtime_constant(0.0).unwrap_or_else(|| Sym::from_poly(Poly::zero()))
    }
    fn one() -> Sym {
        Sym::runtime_constant(1.0).unwrap_or_else(|| Sym::from_poly(Poly::constant(Rational::ONE)))
    }
    fn from_i64(i: i64) -> Sym {
        Sym::runtime_constant(i as f64)
            .unwrap_or_else(|| Sym::from_poly(Poly::constant(Rational::int(i128::from(i)))))
    }
    fn from_ratio(num: i64, den: i64) -> Sym {
        if Sym::runtime_constant(num as f64 / den as f64).is_some() {
            return Sym::from_f64(num as f64 / den as f64);
        }
        Sym::from_poly(Poly::constant(Rational::new(
            i128::from(num),
            i128::from(den),
        )))
    }
    fn from_f64(f: f64) -> Sym {
        if let Some(c) = Sym::runtime_constant(f) {
            return c;
        }
        Sym::value(f)
    }
}

impl Real for Sym {
    fn sqrt(self) -> Sym {
        Sym::atom(Func::Sqrt, self)
    }
    fn recip(self) -> Sym {
        match self.as_constant() {
            Some(c) if !c.is_zero() => Sym::from_poly(Poly::constant(c.recip())),
            _ => Sym::atom(Func::Recip, self),
        }
    }
    fn abs(self) -> Sym {
        Sym::atom(Func::Abs, self)
    }
    fn sin(self) -> Sym {
        Sym::atom(Func::Sin, self)
    }
    fn cos(self) -> Sym {
        Sym::atom(Func::Cos, self)
    }
    fn sinh(self) -> Sym {
        Sym::atom(Func::Sinh, self)
    }
    fn cosh(self) -> Sym {
        Sym::atom(Func::Cosh, self)
    }
    fn atan2(self, x: Sym) -> Sym {
        let v = with(|a| {
            if let Some(&v) = a.atan2_index.get(&(self, x)) {
                return v;
            }
            let v = a.new_var(VarDef::Atan2 { y: self, x });
            a.atan2_index.insert((self, x), v);
            v
        });
        Sym::from_poly(Poly::var(v))
    }
    fn ln(self) -> Sym {
        Sym::atom(Func::Ln, self)
    }
    /// A select is data flow, not a branch: it becomes a `T::select_lt` in the kernel (and
    /// folds when both compared values are constants).
    fn select_lt(a: Sym, b: Sym, x: Sym, y: Sym) -> Sym {
        if x == y {
            return x;
        }
        if let (Some(p), Some(q)) = (a.poly().as_constant(), b.poly().as_constant()) {
            // By value (the derived order is structural: it would say 1/2 < 1/3).
            return if p.lt(q) { x } else { y };
        }
        if !with(|a| a.opaque_constants)
            && let (Some(p), Some(q)) = (a.const_value(), b.const_value())
        {
            return if p < q { x } else { y };
        }
        let v = with(|arena| {
            if let Some(&v) = arena.select_index.get(&(a, b, x, y)) {
                return v;
            }
            let v = arena.new_var(VarDef::Select { a, b, x, y });
            arena.select_index.insert((a, b, x, y), v);
            v
        });
        Sym::from_poly(Poly::var(v))
    }
    fn all_lt(_: Sym, _: Sym) -> bool {
        panic!(
            "gax tracing: a traced kernel tested convergence (all_lt); iterative solvers cannot be traced"
        )
    }
    fn epsilon() -> Sym {
        Sym::constant_atom(ConstKind::Epsilon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_consing_and_identities() {
        Sym::reset();
        let a = Sym::input(0, 0);
        let b = Sym::input(0, 1);
        assert_eq!((a + b) * (a - b), a * a - b * b);
        let s = (a * a + b * b).sqrt();
        assert_eq!(s * s, a * a + b * b);
        assert_eq!(a / a, Sym::one());
        let r1 = (a + b).recip();
        let r2 = Sym::one() / (b + a);
        assert_eq!(r1, r2, "the same denominator gives the same atom");
        assert_eq!(Sym::from_f64(0.5) * Sym::from_i64(2), Sym::one());
    }

    /// Calls on constants fold at trace time: a motor built from constant angles inside a
    /// kernel costs nothing at run time.
    #[test]
    #[allow(clippy::float_cmp)] // the same f64 operations: exact
    fn constant_calls_fold() {
        Sym::reset();
        let two = Sym::from_i64(2);
        assert_eq!(Sym::from_i64(4).sqrt(), two);
        let r2 = two.sqrt().const_value().unwrap();
        assert_eq!(r2, 2f64.sqrt());
        // Calls chain through named constants: sin(sqrt(2)) is a constant too.
        assert_eq!(two.sqrt().sin().const_value(), Some(r2.sin()));
        assert_eq!((-two).abs(), two);
        assert_eq!(Sym::from_i64(0).cos(), Sym::one());
        // Inexact results are named constants: the same value, the same symbol.
        let a = Sym::from_f64(0.3);
        assert_eq!(a.sin(), Sym::from_f64(0.3).sin());
        let x = Sym::input(0, 0);
        // Constants compare by value (once, (num, den) compared lexicographically: 1/2 < 1/3).
        let (half, third) = (Sym::from_ratio(1, 2), Sym::from_ratio(1, 3));
        assert_eq!(Sym::select_lt(half, third, x, Sym::zero()), Sym::zero());
        assert_eq!(Sym::select_lt(third, half, x, Sym::zero()), x);
        // Not constants: still atoms.
        assert!(x.sqrt().as_constant().is_none());
    }
}
