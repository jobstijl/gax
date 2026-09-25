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

/// Compile a set of polynomials into a program whose outputs equal them.
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
        let terms = polys.iter().map(to_terms).collect::<Vec<_>>();
        compile_terms(&mut b, terms)
    } else {
        compile_factored(&mut b, &polys, &opts.passengers)
    };
    b.prog.outputs = outs;
    b.prog.compact();
    b.prog
}

/// Compile with several strategies and keep the cheapest program.
pub fn compile_best(polys: &[Poly], passengers: &BTreeSet<Var>, relations: &[Poly]) -> Program {
    let mut candidates = Vec::new();
    for use_rel in [false, true] {
        if use_rel && relations.is_empty() {
            continue;
        }
        let rel = if use_rel {
            relations.to_vec()
        } else {
            Vec::new()
        };
        candidates.push(compile(
            polys,
            &Options {
                passengers: BTreeSet::new(),
                relations: rel.clone(),
            },
        ));
        if !passengers.is_empty() {
            candidates.push(compile(
                polys,
                &Options {
                    passengers: passengers.clone(),
                    relations: rel,
                },
            ));
        }
    }
    candidates
        .into_iter()
        .min_by(|a, b| a.cost().weight().total_cmp(&b.cost().weight()))
        .expect("at least one candidate")
}

fn to_terms(p: &Poly) -> Vec<Term> {
    p.0.iter()
        .map(|(m, &c)| Term {
            coef: c,
            atoms: m.0.iter().map(|&v| Operand::Var(v)).collect(),
        })
        .collect()
}

fn compile_factored(b: &mut Builder, polys: &[Poly], passengers: &BTreeSet<Var>) -> Vec<Operand> {
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
    let coef_terms: Vec<Vec<Term>> = coefs.iter().map(to_terms).collect();
    let coef_ops = compile_terms(b, coef_terms);
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
fn emit_sum(b: &mut Builder, s: &[(Rational, Operand)]) -> Operand {
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
    let mut acc = items[0].1;
    for &(neg, o) in &items[1..] {
        acc = if neg {
            b.emit(Instr::Sub(acc, o))
        } else {
            b.emit(Instr::Add(acc, o))
        };
    }
    (all_negative, acc)
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
