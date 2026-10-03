//! From polynomials to straight-line programs: common subexpression elimination.
//!
//! The pipeline, applied to a set of output polynomials:
//!
//! 1. **Relation reduction** (optional): add multiples of known identities (type conditions
//!    such as `a0² + a1² + a2² + a3² - 1 = 0`) while that removes terms. This is square-sum
//!    completion: `a0² + a1² - a2² - a3²` becomes `1 - 2a2² - 2a3²` for a unit rotor.
//! 2. **Passenger factoring** (optional): group each output by the monomials of the
//!    designated linear variables, `out = Σ x_i C_i(v)`, and compile the coefficient
//!    polynomials `C_i` first. This is the matrix-then-apply form of a sandwich.
//! 3. **Product extraction**: repeatedly name the pair of factors that occurs in the most
//!    terms across all outputs.
//! 4. **Sum extraction**: repeatedly name the pair of terms (up to a common factor) that
//!    occurs in the most outputs.
//! 5. **Emission**: each output is summed grouping equal coefficient magnitudes, so a common
//!    factor costs one multiplication.
//!
//! Instructions are hash-consed, so identical subexpressions are computed once. Every
//! result can be verified with [`Program::to_polys`].

use crate::poly::{Monomial, Poly, Rational, Var};
use crate::slp::{Instr, Operand, Program};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Options for [`compile`].
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// Variables in which the outputs are linear, to factor out first.
    pub passengers: BTreeSet<Var>,
    /// Polynomial identities (`p = 0`) that may be used to simplify.
    pub relations: Vec<Poly>,
    /// Extract shared kernels (sums shared across outputs under different co-kernels).
    pub kernels: bool,
}

/// Build programs with hash-consed instructions.
#[derive(Default)]
pub struct Builder {
    /// The program under construction.
    pub prog: Program,
    memo: HashMap<Instr, Operand>,
}

impl Builder {
    /// Emit an instruction, reusing an identical earlier one.
    pub fn emit(&mut self, i: Instr) -> Operand {
        let i = canonical(i);
        if let Some(&o) = self.memo.get(&i) {
            return o;
        }
        let o = self.prog.push(i.clone());
        self.memo.insert(i, o);
        o
    }

    /// `a * b`, folding constants.
    pub fn mul(&mut self, a: Operand, b: Operand) -> Operand {
        match (a, b) {
            (Operand::Const(x), Operand::Const(y)) => Operand::Const(x * y),
            (Operand::Const(x), o) | (o, Operand::Const(x)) if x == Rational::ONE => o,
            (Operand::Const(x), o) | (o, Operand::Const(x)) if x == -Rational::ONE => {
                self.emit(Instr::Neg(o))
            }
            _ => self.emit(Instr::Mul(a, b)),
        }
    }

    /// Product of several operands, left to right.
    pub fn product(&mut self, atoms: &[Operand]) -> Operand {
        let mut it = atoms.iter();
        let Some(&first) = it.next() else {
            return Operand::Const(Rational::ONE);
        };
        it.fold(first, |acc, &a| self.mul(acc, a))
    }
}

fn canonical(i: Instr) -> Instr {
    match i {
        Instr::Add(a, b) if b < a => Instr::Add(b, a),
        Instr::Mul(a, b) if b < a => Instr::Mul(b, a),
        other => other,
    }
}

#[derive(Clone, Debug)]
struct Term {
    coef: Rational,
    atoms: Vec<Operand>,
}

/// Compile a set of polynomials into a program whose outputs equal them (modulo the
/// relations, when relations are given).
pub fn compile(polys: &[Poly], opts: &Options) -> Program {
    let polys: Vec<Poly> = if opts.relations.is_empty() {
        polys.to_vec()
    } else {
        polys
            .iter()
            .map(|p| reduce_by_relations(p, &opts.relations))
            .collect()
    };
    let mut b = Builder::default();
    let outs = if opts.passengers.is_empty() {
        compile_polys(&mut b, &polys, opts.kernels)
    } else {
        compile_factored(&mut b, &polys, &opts.passengers, opts.kernels)
    };
    b.prog.outputs = outs;
    b.prog.compact();
    b.prog
}

/// Compile with several strategies and keep the cheapest program.
pub fn compile_best(polys: &[Poly], passengers: &BTreeSet<Var>, relations: &[Poly]) -> Program {
    // The strategies are independent: compiled in parallel, chosen in this order.
    let mut strategies = Vec::new();
    for use_rel in [false, true] {
        if use_rel && relations.is_empty() {
            continue;
        }
        let rel = if use_rel {
            relations.to_vec()
        } else {
            Vec::new()
        };
        for kernels in [false, true] {
            strategies.push(Options {
                passengers: BTreeSet::new(),
                relations: rel.clone(),
                kernels,
            });
            if !passengers.is_empty() {
                strategies.push(Options {
                    passengers: passengers.clone(),
                    relations: rel.clone(),
                    kernels,
                });
            }
        }
    }
    // Threads pay off only for large programs; most kernels compile in microseconds.
    let terms: usize = polys.iter().map(Poly::len).sum();
    let candidates = if terms > 2000 {
        crate::par::map(&strategies, |o| compile(polys, o))
    } else {
        strategies.iter().map(|o| compile(polys, o)).collect()
    };
    choose(candidates)
}

/// The cheapest candidate, unless it saves at most one operation over another whose error
/// bound is less than half its own: then the more accurate one (see `Program::error_score`).
fn choose(candidates: Vec<Program>) -> Program {
    let scored: Vec<(f64, f64, Program)> = candidates
        .into_iter()
        .map(|p| (p.cost().weight(), p.error_score(), p))
        .collect();
    let best = scored
        .iter()
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .expect("at least one candidate");
    let (w, e) = (best.0, best.1);
    let accurate = scored
        .iter()
        .filter(|c| c.0 <= w + 1.01 && c.1 * 2.0 < e)
        .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.total_cmp(&b.0)));
    match accurate {
        Some(c) => c.2.clone(),
        None => best.2.clone(),
    }
}

/// Compile polynomials, optionally extracting shared kernels first.
fn compile_polys(b: &mut Builder, polys: &[Poly], kernels: bool) -> Vec<Operand> {
    compile_polys_env(b, polys, kernels, &HashMap::new())
}

/// Compile polynomials whose variables may be bound to earlier results by `env`.
pub fn compile_polys_env(
    b: &mut Builder,
    polys: &[Poly],
    kernels: bool,
    env: &HashMap<Var, Operand>,
) -> Vec<Operand> {
    let mut polys = polys.to_vec();
    let mut env = env.clone();
    if kernels {
        let first_free = polys
            .iter()
            .flat_map(Poly::vars)
            .chain(env.keys().copied())
            .max()
            .map_or(0, |v| v + 1);
        let defs = extract_kernels(&mut polys, first_free);
        for (t, def) in defs {
            let op = compile_terms(b, vec![to_terms_env(&def, &env)])[0];
            env.insert(t, op);
        }
    }
    let terms = polys.iter().map(|p| to_terms_env(p, &env)).collect();
    compile_terms(b, terms)
}

/// How to use the relations when compiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    /// Compile the polynomials as given.
    None,
    /// Add multiples of the relations while that removes terms (square-sum completion).
    Greedy,
    /// Replace every polynomial by its normal form modulo a Gröbner basis of the relations.
    NormalForm,
}

/// A non-polynomial step of a staged computation: `var = op(args)`.
#[derive(Clone, Debug)]
pub struct Stage {
    /// The variable the result is bound to.
    pub var: Var,
    /// The operation.
    pub op: StageOp,
    /// Arguments, as polynomials in inputs and earlier stage variables.
    pub args: Vec<Poly>,
}

/// Operation of a [`Stage`].
#[derive(Clone, Copy, Debug)]
pub enum StageOp {
    /// A function of one argument.
    Call(crate::slp::Func),
    /// `atan2(args[0], args[1])`.
    Atan2,
    /// `if args[0] < args[1] { args[2] } else { args[3] }`.
    Select,
    /// `args[0] + args[1]`
    Add,
    /// `args[0] - args[1]`
    Sub,
    /// `args[0] * args[1]`
    Mul,
}

/// Compile outputs that depend on staged non-polynomial values.
///
/// Returns the program and, for each stage variable, the temporary holding its value.
pub fn compile_staged(
    outputs: &[Poly],
    stages: &[Stage],
    relations: &[Poly],
    reduction: Reduction,
    kernels: bool,
) -> (Program, HashMap<Var, Operand>) {
    let basis = match reduction {
        Reduction::NormalForm => crate::groebner::groebner(relations, 2000),
        _ => None,
    };
    compile_staged_with(
        outputs,
        stages,
        relations,
        basis.as_deref(),
        reduction,
        kernels,
    )
}

/// [`compile_staged`] with the Gröbner basis of the relations already computed (it is only
/// used by `Reduction::NormalForm`).
fn compile_staged_with(
    outputs: &[Poly],
    stages: &[Stage],
    relations: &[Poly],
    basis: Option<&[Poly]>,
    reduction: Reduction,
    kernels: bool,
) -> (Program, HashMap<Var, Operand>) {
    let reduce = |p: &Poly| match reduction {
        Reduction::None => p.clone(),
        Reduction::Greedy => reduce_by_relations(p, relations),
        Reduction::NormalForm => {
            basis.map_or_else(|| p.clone(), |g| crate::groebner::normal_form(p, g))
        }
    };
    let mut b = Builder::default();
    let mut env: HashMap<Var, Operand> = HashMap::new();
    let staged: std::collections::HashSet<Var> = stages.iter().map(|st| st.var).collect();
    for st in stages {
        // A stage's arguments may only use what is defined before it. Reducing them modulo
        // the relations could bring in the stage's own variable (`sqrt x` reduced with
        // `s² = x` becomes `sqrt(s s)`) or a later one; keep such an argument as it is.
        let defined = |p: &Poly| {
            p.0.keys()
                .flat_map(|m| m.0.iter())
                .all(|v| !staged.contains(v) || env.contains_key(v))
        };
        let args: Vec<Poly> = st
            .args
            .iter()
            .map(|p| {
                let r = reduce(p);
                if defined(&r) { r } else { p.clone() }
            })
            .collect();
        let ops = compile_polys_env(&mut b, &args, kernels, &env);
        let r = match st.op {
            StageOp::Call(f) => b.emit(Instr::Call(f, ops[0])),
            StageOp::Atan2 => b.emit(Instr::Atan2(ops[0], ops[1])),
            StageOp::Select => b.emit(Instr::Select(ops[0], ops[1], ops[2], ops[3])),
            StageOp::Add => b.emit(Instr::Add(ops[0], ops[1])),
            StageOp::Sub => b.emit(Instr::Sub(ops[0], ops[1])),
            StageOp::Mul => b.mul(ops[0], ops[1]),
        };
        env.insert(st.var, r);
    }
    let outs: Vec<Poly> = outputs.iter().map(reduce).collect();
    b.prog.outputs = compile_polys_env(&mut b, &outs, kernels, &env);
    // Keep the stage temporaries addressable through compaction by listing them as extra
    // outputs, then strip them again.
    let n = b.prog.outputs.len();
    let mut stage_vars: Vec<Var> = env.keys().copied().collect();
    stage_vars.sort_unstable();
    for v in &stage_vars {
        b.prog.outputs.push(env[v]);
    }
    let mut prog = b.prog;
    prog.compact();
    let extra = prog.outputs.split_off(n);
    let env = stage_vars.into_iter().zip(extra).collect();
    (prog, env)
}

/// The Gröbner work budget for a traced kernel's relations (see
/// `groebner::groebner_within`): a tenth of a second at most, per strategy.
const TRACE_BASIS_BUDGET: usize = 1_000_000;

/// Above this many terms (outputs and stage arguments), [`compile_staged_best`] skips the plain
/// compile.
const PLAIN_MAX_TERMS: usize = 1200;

/// Compile staged outputs with every strategy and keep the cheapest program that is verified
/// exact: its outputs equal the given polynomials modulo the relations.
pub fn compile_staged_best(
    outputs: &[Poly],
    stages: &[Stage],
    relations: &[Poly],
) -> (Program, HashMap<Var, Operand>) {
    let profile = std::env::var_os("GAX_GEN_PROFILE").is_some();
    let clock = std::time::Instant::now();
    // Inconsistent relations (the ideal holds 1, so every expression would reduce to zero)
    // are no basis at all: a reciprocal of something that is zero only under the other
    // relations can bring them in.
    let basis = crate::groebner::groebner_within(relations, 2000, TRACE_BASIS_BUDGET).filter(|g| {
        !g.iter()
            .any(|p| p.as_constant().is_some_and(|c| !c.is_zero()))
    });
    if profile {
        eprintln!(
            "        basis: {:.2} s ({:?} polynomials)",
            clock.elapsed().as_secs_f64(),
            basis.as_ref().map(Vec::len)
        );
    }
    let mut best: Option<(Program, HashMap<Var, Operand>)> = None;
    // Without a basis the relations cannot be checked, so only the strategy that does not use
    // them runs, and it is checked as an exact identity: nothing goes out unverified.
    let reductions: &[Reduction] = if relations.is_empty() || basis.is_none() {
        &[Reduction::None]
    } else {
        &[Reduction::None, Reduction::Greedy, Reduction::NormalForm]
    };
    let check: &[Poly] = basis.as_deref().unwrap_or(&[]);
    // The plain compile, greedy sharing over the expanded terms, grows like the cube of their
    // number (0.3 s at 700 terms, 105 s at 3600, warp's tone mapper); above `PLAIN_MAX_TERMS`
    // only the compile that extracts polynomial kernels first runs (6 s at 3600).
    let terms: usize = outputs.iter().map(Poly::len).sum::<usize>()
        + stages
            .iter()
            .flat_map(|st| &st.args)
            .map(Poly::len)
            .sum::<usize>();
    let modes: &[bool] = if terms > PLAIN_MAX_TERMS {
        &[true]
    } else {
        &[false, true]
    };
    for &reduction in reductions {
        for &kernels in modes {
            let clock = std::time::Instant::now();
            let (prog, env) = compile_staged_with(
                outputs,
                stages,
                relations,
                basis.as_deref(),
                reduction,
                kernels,
            );
            let compiled = clock.elapsed().as_secs_f64();
            assert!(
                verify_staged(&prog, &env, outputs, check),
                "simplifier produced a program that differs from its input ({reduction:?}, kernels {kernels})"
            );
            if profile {
                eprintln!(
                    "        {reduction:?}, kernels {kernels}: compiled {compiled:.2} s, verified {:.2} s",
                    clock.elapsed().as_secs_f64() - compiled
                );
            }
            let better = best
                .as_ref()
                .is_none_or(|(b, _)| prog.cost().weight() < b.cost().weight());
            if better {
                best = Some((prog, env));
            }
        }
    }
    best.expect("at least one strategy")
}

/// Check `prog`'s outputs against `want` modulo the ideal with Gröbner basis `basis`, reading
/// stage temporaries as their variables.
pub fn verify_staged(
    prog: &Program,
    env: &HashMap<Var, Operand>,
    want: &[Poly],
    basis: &[Poly],
) -> bool {
    let got = prog.to_polys_with(&|k| {
        env.iter()
            .find(|(_, o)| **o == Operand::Temp(k))
            .map(|(v, _)| Poly::var(*v))
    });
    got.iter()
        .zip(want)
        .all(|(g, w)| crate::groebner::normal_form(&(g - w), basis).is_zero())
}

fn to_terms_env(p: &Poly, env: &HashMap<Var, Operand>) -> Vec<Term> {
    p.0.iter()
        .map(|(m, &c)| {
            let mut atoms: Vec<Operand> =
                m.0.iter()
                    .map(|v| env.get(v).copied().unwrap_or(Operand::Var(*v)))
                    .collect();
            atoms.sort();
            Term { coef: c, atoms }
        })
        .collect()
}

/// One place where a candidate kernel occurs: `scale * v * kernel` is a part of output `k`.
#[derive(Clone, Debug)]
struct Occurrence {
    output: usize,
    cokernel: Var,
    scale: Rational,
    /// The original monomials of the output covered by this occurrence.
    monomials: Vec<Monomial>,
}

/// Multiplications and additions to evaluate a polynomial term by term.
fn naive_cost(p: &Poly) -> usize {
    let mults: usize =
        p.0.iter()
            .map(|(m, c)| m.degree().saturating_sub(1) + usize::from(c.abs() != Rational::ONE))
            .sum();
    mults + p.len().saturating_sub(1)
}

/// Brayton–McMullen style kernel extraction.
///
/// For each output and each variable `v`, the terms divisible by `v` form a residual `R`
/// (`P = v R + rest`). Every subset of `R` with at least two terms (up to a rational factor) is
/// a candidate kernel. A kernel occurring in several places, with disjoint original terms, is
/// named once and each occurrence becomes `scale * v * t`. The candidate with the largest
/// estimated saving is taken, and the search repeats. The result is exact: the outputs are
/// rewritten in terms of fresh variables whose definitions are returned in order.
fn extract_kernels(polys: &mut [Poly], first_free: Var) -> Vec<(Var, Poly)> {
    let mut defs = Vec::new();
    let mut next = first_free;
    for _round in 0..64 {
        let mut candidates: HashMap<Poly, Vec<Occurrence>> = HashMap::new();
        for (k, p) in polys.iter().enumerate() {
            for v in p.vars() {
                let vm = Monomial::var(v);
                let residual: Vec<(Monomial, Rational, Monomial)> =
                    p.0.iter()
                        .filter_map(|(m, &c)| m.div(&vm).map(|q| (q, c, m.clone())))
                        .collect();
                let n = residual.len();
                if n < 2 {
                    continue;
                }
                let mut subsets: Vec<Vec<usize>> = Vec::new();
                if n <= 7 {
                    for mask in 1u32..(1 << n) {
                        if mask.count_ones() >= 2 {
                            subsets.push((0..n).filter(|i| mask & (1 << i) != 0).collect());
                        }
                    }
                } else {
                    subsets.push((0..n).collect());
                    for skip in 0..n {
                        subsets.push((0..n).filter(|&i| i != skip).collect());
                    }
                }
                for sub in subsets {
                    let lead = residual[sub[0]].1;
                    let mut kernel = Poly::zero();
                    for &i in &sub {
                        kernel.add_term(residual[i].0.clone(), residual[i].1 * lead.recip());
                    }
                    // A kernel that is a single monomial times a constant is a product, not a sum.
                    if kernel.len() < 2 {
                        continue;
                    }
                    candidates.entry(kernel).or_default().push(Occurrence {
                        output: k,
                        cokernel: v,
                        scale: lead,
                        monomials: sub.iter().map(|&i| residual[i].2.clone()).collect(),
                    });
                }
            }
        }
        // Score: pick non-overlapping occurrences greedily, estimate the saving.
        let mut best: Option<(i64, Poly, Vec<Occurrence>)> = None;
        let mut keys: Vec<&Poly> = candidates.keys().collect();
        keys.sort();
        for kernel in keys {
            let occs = &candidates[kernel];
            if occs.len() < 2 {
                continue;
            }
            let mut used: BTreeSet<(usize, Monomial)> = BTreeSet::new();
            let mut kept = Vec::new();
            for o in occs {
                if o.monomials
                    .iter()
                    .all(|m| !used.contains(&(o.output, m.clone())))
                {
                    for m in &o.monomials {
                        used.insert((o.output, m.clone()));
                    }
                    kept.push(o.clone());
                }
            }
            if kept.len() < 2 {
                continue;
            }
            let kernel_cost = naive_cost(kernel) as i64;
            let per_occurrence: i64 =
                kernel.0.keys().map(|m| m.degree() as i64 + 1).sum::<i64>() - 1;
            // Before: each covered term costs its multiplications plus one addition. After: one
            // multiplication by the named kernel.
            let saving = kept.len() as i64 * (per_occurrence - 1) - kernel_cost;
            if saving > 0
                && best
                    .as_ref()
                    .is_none_or(|(s, k, _)| saving > *s || (saving == *s && kernel < k))
            {
                best = Some((saving, kernel.clone(), kept));
            }
        }
        let Some((_, kernel, occs)) = best else { break };
        let t = next;
        next += 1;
        for o in &occs {
            let p = &mut polys[o.output];
            let vm = Monomial::var(o.cokernel);
            for (m, &c) in &kernel.0 {
                p.add_term(m.mul(&vm), -(c * o.scale));
            }
            p.add_term(vm.mul(&Monomial::var(t)), o.scale);
        }
        defs.push((t, kernel));
    }
    defs
}

fn compile_factored(
    b: &mut Builder,
    polys: &[Poly],
    passengers: &BTreeSet<Var>,
    kernels: bool,
) -> Vec<Operand> {
    // out = Σ_g m_g * C_g, with m_g a passenger monomial and C_g a polynomial in the rest.
    let mut groups: Vec<Vec<(Monomial, Poly)>> = Vec::new();
    let mut coefs: Vec<Poly> = Vec::new();
    let mut coef_index: HashMap<Poly, usize> = HashMap::new();
    let mut layout: Vec<Vec<(Monomial, usize, Rational)>> = Vec::new();
    for p in polys {
        let mut by_passenger: BTreeMap<Monomial, Poly> = BTreeMap::new();
        for (m, &c) in &p.0 {
            let (pm, rest): (Vec<Var>, Vec<Var>) = m.0.iter().partition(|v| passengers.contains(v));
            by_passenger
                .entry(Monomial(pm))
                .or_default()
                .add_term(Monomial(rest), c);
        }
        let mut row = Vec::new();
        for (pm, c) in &by_passenger {
            // Normalize the coefficient polynomial up to a rational factor so that C and -C
            // (or 2C) share one computation.
            let lead = *c.0.values().next().expect("nonzero");
            let norm = c.scale(lead.recip());
            let idx = *coef_index.entry(norm.clone()).or_insert_with(|| {
                coefs.push(norm);
                coefs.len() - 1
            });
            row.push((pm.clone(), idx, lead));
        }
        groups.push(by_passenger.into_iter().collect());
        layout.push(row);
    }
    let coef_ops = compile_polys(b, &coefs, kernels);
    let terms: Vec<Vec<Term>> = layout
        .iter()
        .map(|row| {
            row.iter()
                .map(|(pm, idx, lead)| {
                    let mut atoms: Vec<Operand> = pm.0.iter().map(|&v| Operand::Var(v)).collect();
                    match coef_ops[*idx] {
                        Operand::Const(k) => {
                            return Term {
                                coef: *lead * k,
                                atoms,
                            };
                        }
                        o => atoms.push(o),
                    }
                    atoms.sort();
                    Term { coef: *lead, atoms }
                })
                .collect()
        })
        .collect();
    compile_terms(b, terms)
}

fn compile_terms(b: &mut Builder, mut outs: Vec<Vec<Term>>) -> Vec<Operand> {
    extract_products(b, &mut outs);
    // Collapse each term to a single operand.
    let mut sums: Vec<Vec<(Rational, Operand)>> = outs
        .into_iter()
        .map(|terms| {
            let mut acc: BTreeMap<Operand, Rational> = BTreeMap::new();
            for t in terms {
                let o = b.product(&t.atoms);
                let (o, k) = match o {
                    Operand::Const(k) => (Operand::Const(Rational::ONE), k),
                    o => (o, Rational::ONE),
                };
                let e = acc.entry(o).or_insert(Rational::ZERO);
                *e = *e + t.coef * k;
            }
            acc.into_iter()
                .filter(|(_, c)| !c.is_zero())
                .map(|(o, c)| (c, o))
                .collect()
        })
        .collect();
    extract_sums(b, &mut sums);
    sums.iter().map(|s| emit_sum(b, s)).collect()
}

/// Repeatedly name the most frequent pair of factors.
fn extract_products(b: &mut Builder, outs: &mut [Vec<Term>]) {
    loop {
        let mut count: HashMap<(Operand, Operand), usize> = HashMap::new();
        for t in outs.iter().flatten() {
            let mut seen = BTreeSet::new();
            for i in 0..t.atoms.len() {
                for j in i + 1..t.atoms.len() {
                    let pair = (t.atoms[i], t.atoms[j]);
                    if seen.insert(pair) {
                        *count.entry(pair).or_default() += 1;
                    }
                }
            }
        }
        let Some((&pair, &n)) = count.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0))) else {
            break;
        };
        if n < 2 {
            break;
        }
        let t = b.mul(pair.0, pair.1);
        for term in outs.iter_mut().flatten() {
            if let Some(i) = term.atoms.iter().position(|&a| a == pair.0) {
                let mut rest = term.atoms.clone();
                rest.remove(i);
                if let Some(j) = rest.iter().position(|&a| a == pair.1) {
                    rest.remove(j);
                    rest.push(t);
                    rest.sort();
                    term.atoms = rest;
                }
            }
        }
    }
}

/// Repeatedly name the pair of terms `p + r q` that occurs in the most outputs.
fn extract_sums(b: &mut Builder, sums: &mut [Vec<(Rational, Operand)>]) {
    loop {
        let mut count: HashMap<(Operand, Operand, Rational), usize> = HashMap::new();
        for s in sums.iter() {
            for i in 0..s.len() {
                for j in i + 1..s.len() {
                    let (ci, pi) = s[i];
                    let (cj, pj) = s[j];
                    if matches!(pi, Operand::Const(_)) || matches!(pj, Operand::Const(_)) {
                        continue;
                    }
                    *count.entry((pi, pj, cj * ci.recip())).or_default() += 1;
                }
            }
        }
        let Some((&(p, q, r), &n)) = count.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
        else {
            break;
        };
        if n < 2 {
            break;
        }
        let t = if r == Rational::ONE {
            b.emit(Instr::Add(p, q))
        } else if r == -Rational::ONE {
            b.emit(Instr::Sub(p, q))
        } else {
            let rq = b.mul(Operand::Const(r), q);
            b.emit(Instr::Add(p, rq))
        };
        for s in sums.iter_mut() {
            let ip = s.iter().position(|&(_, o)| o == p);
            let iq = s.iter().position(|&(_, o)| o == q);
            if let (Some(ip), Some(iq)) = (ip, iq) {
                let (cp, cq) = (s[ip].0, s[iq].0);
                if cq == cp * r {
                    let (lo, hi) = if ip < iq { (ip, iq) } else { (iq, ip) };
                    s.remove(hi);
                    s.remove(lo);
                    s.push((cp, t));
                }
            }
        }
    }
}

/// Emit `Σ c_k p_k`, grouping equal coefficient magnitudes.
pub fn emit_sum(b: &mut Builder, s: &[(Rational, Operand)]) -> Operand {
    if s.is_empty() {
        return Operand::Const(Rational::ZERO);
    }
    let mut groups: BTreeMap<Rational, Vec<(bool, Operand)>> = BTreeMap::new();
    for &(c, o) in s {
        if let Operand::Const(k) = o {
            groups
                .entry(Rational::ONE)
                .or_default()
                .push((false, Operand::Const(c * k)));
        } else {
            groups
                .entry(c.abs())
                .or_default()
                .push((c < Rational::ZERO, o));
        }
    }
    let mut parts: Vec<(bool, Operand)> = Vec::new();
    for (mag, items) in groups {
        let (neg, sum) = signed_sum(b, &items);
        if mag == Rational::ONE {
            parts.push((neg, sum));
        } else {
            let scaled = b.mul(Operand::Const(mag), sum);
            parts.push((neg, scaled));
        }
    }
    let (neg, total) = signed_sum(b, &parts);
    if neg {
        b.emit(Instr::Neg(total))
    } else {
        total
    }
}

/// Sum of `±p` terms, returned as `(negated, value)` so a final negation can be deferred.
fn signed_sum(b: &mut Builder, items: &[(bool, Operand)]) -> (bool, Operand) {
    let mut items: Vec<(bool, Operand)> = items.to_vec();
    // Constants: fold together.
    let mut constant = Rational::ZERO;
    items.retain(|&(neg, o)| {
        if let Operand::Const(k) = o {
            constant = constant + if neg { -k } else { k };
            false
        } else {
            true
        }
    });
    if !constant.is_zero() {
        items.push((constant < Rational::ZERO, Operand::Const(constant.abs())));
    }
    // Start from a positive term when there is one.
    let all_negative = items.iter().all(|x| x.0);
    if all_negative {
        for x in &mut items {
            x.0 = false;
        }
    }
    items.sort_by_key(|x| x.0);
    // A balanced tree of additions: log2(n) dependent steps instead of n - 1, which is what
    // limits the latency of a single kernel call (floating-point sums cannot be reassociated
    // by the compiler).
    let (neg, acc) = balanced(b, &items);
    (all_negative ^ neg, acc)
}

/// Sum `±x_i` as a balanced tree; returns `(negated, value)`.
fn balanced(b: &mut Builder, items: &[(bool, Operand)]) -> (bool, Operand) {
    match items {
        [] => (false, Operand::Const(Rational::ZERO)),
        [x] => *x,
        _ => {
            let (l, r) = items.split_at(items.len().div_ceil(2));
            let (ln, lv) = balanced(b, l);
            let (rn, rv) = balanced(b, r);
            match (ln, rn) {
                (false, false) => (false, b.emit(Instr::Add(lv, rv))),
                (false, true) => (false, b.emit(Instr::Sub(lv, rv))),
                (true, false) => (false, b.emit(Instr::Sub(rv, lv))),
                (true, true) => (true, b.emit(Instr::Add(lv, rv))),
            }
        }
    }
}

/// Add multiples of the relations (`r = 0`) to `p` while that reduces the number of terms.
pub fn reduce_by_relations(p: &Poly, relations: &[Poly]) -> Poly {
    let mut p = p.clone();
    loop {
        let mut best: Option<Poly> = None;
        for r in relations {
            // Candidate multipliers: monomial * rational chosen to cancel a term of p.
            for (rm, &rc) in &r.0 {
                for (pm, &pc) in &p.0 {
                    let Some(q) = pm.div(rm) else { continue };
                    let k = -(pc * rc.recip());
                    let mut shift = Poly::zero();
                    for (m2, &c2) in &r.0 {
                        shift.add_term(m2.mul(&q), c2 * k);
                    }
                    let cand = &p + &shift;
                    let better = best
                        .as_ref()
                        .map_or(cand.len() < p.len(), |b| cand.len() < b.len());
                    if better {
                        best = Some(cand);
                    }
                }
            }
        }
        match best {
            Some(b) => p = b,
            None => return p,
        }
    }
}

/// Make a program homogeneous of degree `target` in the variables below `nv` (a versor's
/// coefficients) by multiplying lower-degree operands of sums, and outputs, by powers of
/// `norm` (a quadratic form in those variables that equals 1 modulo the relations the program
/// was simplified with). The program's structure is kept, so this costs a few multiplications
/// where the simplifier substituted 1 for `norm`. `None` if the program has non-polynomial
/// steps or a degree gap that is odd.
pub fn repair_degree(prog: &Program, nv: Var, norm: &Poly, target: usize) -> Option<Program> {
    struct State {
        b: Builder,
        norm: Option<Operand>,
    }
    fn norm_op(st: &mut State, norm: &Poly) -> Operand {
        if let Some(n) = st.norm {
            return n;
        }
        let n = compile_polys_env(
            &mut st.b,
            std::slice::from_ref(norm),
            false,
            &HashMap::new(),
        )[0];
        st.norm = Some(n);
        n
    }
    fn raise(st: &mut State, norm: &Poly, (o, d): (Operand, usize), to: usize) -> Option<Operand> {
        if d > to || (to - d) % 2 == 1 {
            return None;
        }
        if matches!(o, Operand::Const(c) if c.is_zero()) {
            return Some(o);
        }
        let mut o = o;
        for _ in 0..(to - d) / 2 {
            let n = norm_op(st, norm);
            o = st.b.mul(o, n);
        }
        Some(o)
    }
    let mut st = State {
        b: Builder::default(),
        norm: None,
    };
    let mut temps: Vec<(Operand, usize)> = Vec::with_capacity(prog.instrs.len());
    let get = |o: &Operand, temps: &[(Operand, usize)]| match *o {
        Operand::Var(v) => (Operand::Var(v), usize::from(v < nv)),
        Operand::Temp(k) => temps[k],
        Operand::Const(c) => (Operand::Const(c), 0),
    };
    for i in &prog.instrs {
        let r = match i {
            Instr::Add(a, c) | Instr::Sub(a, c) => {
                let (x, y) = (get(a, &temps), get(c, &temps));
                let d = x.1.max(y.1);
                // A zero constant has every degree.
                let d = if matches!(x.0, Operand::Const(z) if z.is_zero()) {
                    y.1
                } else if matches!(y.0, Operand::Const(z) if z.is_zero()) {
                    x.1
                } else {
                    d
                };
                let (x, y) = (raise(&mut st, norm, x, d)?, raise(&mut st, norm, y, d)?);
                let op = if matches!(i, Instr::Add(..)) {
                    st.b.emit(Instr::Add(x, y))
                } else {
                    st.b.emit(Instr::Sub(x, y))
                };
                (op, d)
            }
            Instr::Mul(a, c) => {
                let (x, y) = (get(a, &temps), get(c, &temps));
                (st.b.mul(x.0, y.0), x.1 + y.1)
            }
            Instr::Neg(a) => {
                let x = get(a, &temps);
                (st.b.emit(Instr::Neg(x.0)), x.1)
            }
            Instr::Div(a, c) => {
                let (x, y) = (get(a, &temps), get(c, &temps));
                if !matches!(y.0, Operand::Const(_)) {
                    return None;
                }
                (st.b.emit(Instr::Div(x.0, y.0)), x.1)
            }
            Instr::Atan2(..) | Instr::Select(..) | Instr::Call(..) => return None,
        };
        temps.push(r);
    }
    let outs: Vec<(Operand, usize)> = prog.outputs.iter().map(|o| get(o, &temps)).collect();
    let mut outputs = Vec::with_capacity(outs.len());
    for o in outs {
        outputs.push(raise(&mut st, norm, o, target)?);
    }
    let mut p = st.b.prog;
    p.outputs = outputs;
    p.compact();
    Some(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(i: Var) -> Poly {
        Poly::var(i)
    }

    #[test]
    fn compiles_exactly() {
        let (a, b, c, d) = (v(0), v(1), v(2), v(3));
        let p1 = &(&a * &b) - &(&c * &d);
        let p2 =
            &(&(&a * &b) + &(&c * &d)).scale(Rational::int(2)) + &Poly::constant(Rational::int(3));
        let polys = vec![
            p1,
            p2.clone(),
            p2,
            Poly::zero(),
            Poly::constant(Rational::new(1, 2)),
        ];
        for opts in [
            Options::default(),
            Options {
                passengers: [0].into(),
                relations: vec![],
                kernels: false,
            },
            Options {
                passengers: BTreeSet::new(),
                relations: vec![],
                kernels: true,
            },
            Options {
                passengers: [0].into(),
                relations: vec![],
                kernels: true,
            },
        ] {
            let prog = compile(&polys, &opts);
            assert_eq!(prog.to_polys(), polys);
        }
    }

    #[test]
    fn square_sum_completion() {
        // unit quaternion: a²+b²+c²+d² = 1
        let (a, b, c, d) = (v(0), v(1), v(2), v(3));
        let rel = &(&(&(&a * &a) + &(&b * &b)) + &(&(&c * &c) + &(&d * &d)))
            - &Poly::constant(Rational::ONE);
        let diag = &(&(&a * &a) + &(&b * &b)) - &(&(&c * &c) + &(&d * &d));
        let r = reduce_by_relations(&diag, std::slice::from_ref(&rel));
        assert_eq!(r.len(), 3);
    }
}
