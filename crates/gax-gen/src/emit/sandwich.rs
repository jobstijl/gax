//! The sandwich kernels `v >> x` of the versor kinds: fused and simplified symbolically, or
//! plain (two products from the tables) for large versors.

use super::{
    Emitter, IMPL_ALL, IMPL_O, INLINE_MAX, PLAIN_MAX_TERMS, SandwichStat, UNROLL_MAX, kind_literal,
    signed_chain, term_loop, term_table, zero_slots,
};
use crate::cse;
use crate::poly::{Poly, Rational, Var};
use crate::slp::{Operand, Program, render};
use crate::spec::{AlgebraSpec, KindSpec};
use crate::symbolic::{self, SymMv};
use crate::table::{BinOp, UnOp, binop_support, binop_table};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

/// The symbolic work of one sandwich kernel `vk >> xk`: its output kind, the fused value
/// program, and the map path's matrix. Pure, so the kernels are computed in parallel.
pub(super) struct SandwichMath {
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
pub(super) fn sandwich_out(
    spec: &AlgebraSpec,
    vk: &KindSpec,
    xk: &KindSpec,
    unit: bool,
) -> Option<KindSpec> {
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
pub(super) fn sandwich_plain(
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
    let entries: BTreeSet<(usize, usize)> = second
        .iter()
        .flat_map(|&(o, m, _, _)| through[m].iter().map(move |&i| (o, i)))
        .collect();
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
pub(super) fn sandwich_math(
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
    // A unit versor's sandwich keeps the passenger's grades, as the plain sandwiches assume
    // (ADR-020, ADR-034). The reduction above finds most other parts zero; some vanish only on the
    // relations' variety, not in their ideal: in 5D the pseudoscalar part of an even versor's
    // sandwich of a vector is in the radical of `u ~u = 1`, never in the ideal itself, and a
    // 5D `Unit<Even> >> Vector` was typed `Odd`. Every even element of a 5D algebra with
    // `u ~u = 1` is a versor (the Lipschitz group, Lounesto), and `tests/unit_versors.rs`
    // checks it on elements projected onto the relations rather than built as versors. From 6D
    // on that fails (`cos θ + I sin θ` turns vectors into grade 5), so there only kinds without
    // the pseudoscalar qualify, each one checked by that test (CSTA's motor).
    let full = alg.pseudoscalar();
    if unit && (alg.dim() <= 5 || vk.layout.position(full).is_none()) {
        let grades = xk.layout.grades();
        reduced.retain(|m, _| grades.contains(&m.count_ones()));
    }
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

/// What the impls of one sandwich kernel `v >> x` share.
struct Sandwich<'a> {
    vn: &'a str,
    xn: &'a str,
    on: &'a str,
    unit: bool,
    /// The versor's, the passenger's and the result's numbers of coefficients.
    nv: usize,
    nxv: usize,
    nout: usize,
    /// The versor's type: `{vn}<(), T>`, or its `Unit`.
    self_ty: String,
    /// The versor's type in the transforms: any orientation (it appears twice, so its own
    /// orientation does not matter).
    self_ty_any: String,
    /// The versor's kind in `gx::Prepared`: `{vn}`, or `gx::Unit<{vn}>`.
    vkind: String,
    /// The versor's coefficients, from `self`.
    vexpr: &'static str,
    /// For an odd versor, which blades of the result are odd (see [`Sandwich::result`]).
    odd_mask: Option<String>,
}

impl<'a> Sandwich<'a> {
    fn new(vk: &'a KindSpec, xk: &'a KindSpec, out: &'a KindSpec, unit: bool) -> Self {
        let vn = vk.name.as_str();
        // An odd versor reflects: its result takes a sign per blade by the passenger's
        // orientation (ADR-051, `gx::orientation::reflect`). Even versors keep the sandwich.
        let parity = |m: u32| m.count_ones() % 2 == 1;
        let odd = vk.layout.blades.iter().all(|&(m, _)| parity(m));
        assert!(
            odd || vk.layout.blades.iter().all(|&(m, _)| !parity(m)),
            "versor kind {vn} mixes even and odd blades"
        );
        let odd_mask = odd.then(|| {
            out.layout
                .blades
                .iter()
                .map(|&(m, _)| if parity(m) { "true" } else { "false" })
                .collect::<Vec<_>>()
                .join(", ")
        });
        let versor = |o: &str| {
            if unit {
                format!("Unit<{vn}<(), T{o}>>")
            } else {
                format!("{vn}<(), T{o}>")
            }
        };
        Sandwich {
            vn,
            xn: &xk.name,
            on: &out.name,
            unit,
            nv: vk.layout.len(),
            nxv: xk.layout.len(),
            nout: out.layout.len(),
            self_ty: versor(""),
            self_ty_any: versor(", OV"),
            vkind: if unit {
                format!("gx::Unit<{vn}>")
            } else {
                vn.to_string()
            },
            vexpr: if unit {
                "self.into_inner().c"
            } else {
                "self.c"
            },
            odd_mask,
        }
    }

    /// The result from its coefficient array `c` over the slots `slots`, with an odd versor's
    /// signs.
    fn result(&self, slots: &str, c: &str) -> String {
        let c = match &self.odd_mask {
            Some(mask) => format!(
                "gx::orientation::reflect::<O, {slots}, T, {}>({c}, [{mask}])",
                self.nout
            ),
            None => c.to_string(),
        };
        kind_literal(self.on, &c)
    }
}

/// The entries of a sandwich's map: per structurally nonzero entry `(output, passenger)`, an
/// exact constant or an expression in `v` after the lines `lets`.
struct MapEntries<'a> {
    entries: &'a [(usize, usize)],
    lets: String,
    vals: Vec<Result<Rational, String>>,
    /// The entries of each output, as indices into `entries`.
    by_output: Vec<Vec<usize>>,
    /// The distinct expressions, in order: what a prepared action stores.
    stored: Vec<String>,
    /// The position of each expression in `stored`.
    stored_at: BTreeMap<String, usize>,
}

impl<'a> MapEntries<'a> {
    fn new(sw: &Sandwich, map: &MapPath, entries: &'a [(usize, usize)], helper: &str) -> Self {
        let mut lets = String::new();
        let vals: Vec<Result<Rational, String>> = match map {
            MapPath::Program(matrix) => {
                let var = |var: Var| format!("v[{var}]");
                matrix.emit_lets(&var, "m", &mut lets);
                matrix
                    .outputs
                    .iter()
                    .map(|o| match o {
                        Operand::Const(c) => Ok(*c),
                        other => Err(render(other, &var, "m")),
                    })
                    .collect()
            }
            MapPath::Columns => {
                let (nout, nxv) = (sw.nout, sw.nxv);
                let _ = write!(
                    lets,
                    "        let mut cols = [[T::zero(); {nout}]; {nxv}];\n        let mut e = [T::zero(); {nxv}];\n        for (i, col) in cols.iter_mut().enumerate() {{\n            e[i] = T::one();\n            *col = {helper}(&v, &e);\n            e[i] = T::zero();\n        }}\n"
                );
                entries
                    .iter()
                    .map(|&(o, i)| Err(format!("cols[{i}][{o}]")))
                    .collect()
            }
        };
        let mut by_output = vec![Vec::new(); sw.nout];
        for (k, &(o, _)) in entries.iter().enumerate() {
            by_output[o].push(k);
        }
        let mut stored = Vec::new();
        let mut stored_at = BTreeMap::new();
        for e in vals.iter().filter_map(|v| v.as_ref().err()) {
            if !stored_at.contains_key(e) {
                stored_at.insert(e.clone(), stored.len());
                stored.push(e.clone());
            }
        }
        MapEntries {
            entries,
            lets,
            vals,
            by_output,
            stored,
            stored_at,
        }
    }

    /// An exact constant, as in the map's expressions.
    fn constant(c: Rational) -> String {
        render(&Operand::Const(c), &|_| String::new(), "m")
    }

    /// The map applied to the slot arrays `x`, per output: `Σ entry · x[i]`, the entries `±1`
    /// as signs, the others by `scale`.
    fn apply(&self, scale: impl Fn(&Result<Rational, String>) -> String) -> Vec<String> {
        self.by_output
            .iter()
            .map(|ks| {
                let e = signed_chain(ks.iter().map(|&k| {
                    let i = self.entries[k].1;
                    match &self.vals[k] {
                        Ok(c) if *c == Rational::ONE => (false, format!("x[{i}]")),
                        Ok(c) if *c == -Rational::ONE => (true, format!("x[{i}]")),
                        v => (false, format!("x[{i}].scale({})", scale(v))),
                    }
                }));
                if e.is_empty() {
                    zero_slots("S")
                } else {
                    format!("({e}).0")
                }
            })
            .collect()
    }

    /// The prepared action's stored entry for a value, or the constant.
    fn prepared(&self, v: &Result<Rational, String>) -> String {
        match v {
            Ok(c) => Self::constant(*c),
            Err(e) => format!("m[{}]", self.stored_at[e]),
        }
    }

    /// The dense map, `nout` rows of `nxv` entries (no multiplication by an identity's zeros).
    fn dense(&self, nxv: usize) -> Vec<String> {
        let mut rows = vec![vec!["T::zero()".to_string(); nxv]; self.by_output.len()];
        for (k, &(o, i)) in self.entries.iter().enumerate().rev() {
            rows[o][i] = self.prepared(&self.vals[k]);
        }
        rows.iter()
            .map(|row| format!("[{}]", row.join(", ")))
            .collect()
    }
}

impl Emitter<'_> {
    /// Fused `v x ~v` with `v` a value of versor kind `vk` and `x` of kind `xk` with any slots.
    pub(super) fn sandwich(
        &mut self,
        vk: &KindSpec,
        xk: &KindSpec,
        unit: bool,
        math: SandwichMath,
    ) {
        let SandwichMath {
            out,
            relations,
            direct,
            map,
            entries,
            plain,
        } = math;
        self.record_sandwich(vk, xk, &out, unit, &direct, &map, &entries);
        let sw = Sandwich::new(vk, xk, &out, unit);
        let check = self.check_units(vk, unit, &relations, plain.as_ref());
        // A plain sandwich's kernel is a function of its own: the map applies it to each basis
        // element of the passenger.
        let helper = format!(
            "sandwich_{}_{}_{}",
            crate::kernel::snake(sw.vn),
            crate::kernel::snake(sw.xn),
            crate::kernel::snake(sw.on)
        );
        let mut s = String::new();
        let value = self.sandwich_value(&sw, &direct, &map, plain.as_ref(), &helper, &mut s);
        let m = MapEntries::new(&sw, &map, &entries, &helper);
        sandwich_transforms(&sw, &check, &value, &m, &mut s);
        let prep_ty = sandwich_prepared(&sw, &check, &m, &mut s);
        if let Some(gate) = &self.cfg.batch {
            sandwich_batch(&sw, gate, &prep_ty, m.stored.len(), &mut s);
        }
        self.w(&s);
        let value_cost = direct.cost();
        let nxv = sw.nxv;
        let map_cost = match &map {
            MapPath::Program(matrix) => matrix.cost(),
            // The value kernel once per basis element of the passenger.
            MapPath::Columns => crate::slp::Cost {
                muls: value_cost.muls * nxv,
                adds: value_cost.adds * nxv,
                divs: value_cost.divs * nxv,
                negs: value_cost.negs * nxv,
                calls: value_cost.calls * nxv,
            },
        };
        self.stats.sandwiches.push(SandwichStat {
            versor: vk.name.clone(),
            passenger: xk.name.clone(),
            unit,
            output: out.name.clone(),
            value_cost,
            map_cost,
        });
    }

    /// check-units: the parts of `v ~v - 1`, handed to `Coef::check_unit` (empty without the
    /// gate, or for a plain versor).
    fn check_units(
        &self,
        vk: &KindSpec,
        unit: bool,
        relations: &[Poly],
        plain: Option<&Plain>,
    ) -> String {
        let (Some(gate), true) = (&self.cfg.check_units, unit) else {
            return String::new();
        };
        let vn = &vk.name;
        // A plain sandwich's versor is large: v ~v by the product (a table loop).
        if let Some(plain) = plain {
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
        } else {
            let prog = cse::compile_plain(relations);
            let (lets, parts) = prog.lets_and_outputs(&|var: Var| format!("v[{var}]"), "u");
            format!(
                "        {gate}\n        {{\n{lets}            T::check_unit(&[{}]);\n        }}\n",
                parts.join(", ")
            )
        }
    }

    /// The value path of `transform` for a value passenger (`xv`): the fused program, or a call
    /// of the plain kernel `helper`, whose function is written to `s` the first time.
    fn sandwich_value(
        &mut self,
        sw: &Sandwich,
        direct: &Program,
        map: &MapPath,
        plain: Option<&Plain>,
        helper: &str,
        s: &mut String,
    ) -> String {
        let (vn, xn, on) = (sw.vn, sw.xn, sw.on);
        let (nv, nxv, nout) = (sw.nv, sw.nxv, sw.nout);
        let var_name = move |var: Var| {
            let var = var as usize;
            if var < nv {
                format!("v[{var}]")
            } else {
                format!("xv[{}]", var - nv)
            }
        };
        let (lets, outs) = direct.lets_and_outputs(&var_name, "t");
        let values = match map {
            MapPath::Program(_) => {
                let r = sw.result(
                    "S",
                    &format!(
                        "gx::slots::from_values::<S, T, {nout}>([{}])",
                        outs.join(", ")
                    ),
                );
                return format!("{lets}            return {r};\n");
            }
            MapPath::Columns => {
                format!("gx::slots::from_values::<S, T, {nout}>({helper}(&v, &xv))")
            }
        };
        // The same for a `Unit` versor and a plain one when their results agree.
        if self.helpers.insert(helper.to_string()) {
            let doc = format!(
                "/// `v x ~v` on coefficient arrays, for `{vn}` v and `{xn}` x, as `{on}`."
            );
            let terms = plain.map_or(0, |t| t.first.len() + t.second.len());
            match plain {
                Some(t) if terms > UNROLL_MAX => {
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
                    let inline = if terms > INLINE_MAX {
                        "#[inline]"
                    } else {
                        "#[inline(always)]"
                    };
                    let _ = write!(
                        s,
                        "{doc}\n{inline}\nfn {helper}<T: Coef>(v: &[T; {nv}], xv: &[T; {nxv}]) -> [T; {nout}] {{\n{lets}    [{}]\n}}\n\n",
                        outs.join(", ")
                    );
                }
            }
        }
        format!("            return {};\n", sw.result("S", &values))
    }
}

/// `Transform` of a sandwich (the value path for values, the map's entries for maps) and
/// `TransformInv`.
fn sandwich_transforms(sw: &Sandwich, check: &str, value: &str, m: &MapEntries, s: &mut String) {
    let (xn, on, self_ty_any, vexpr, nxv) = (sw.xn, sw.on, &sw.self_ty_any, sw.vexpr, sw.nxv);
    let _ = write!(
        s,
        "impl<S: Slots, T: Coef, O: gx::Orientation, OV: gx::Orientation> Transform<{xn}<S, T, O>> for {self_ty_any} {{\n    type Output = {on}<S, T, O>;\n    #[inline(always)]\n    fn transform(self, x: {xn}<S, T, O>) -> {on}<S, T, O> {{\n        let v = {vexpr};\n{check}        if let Some(xv) = gx::slots::values::<S, T, {nxv}>(&x.c) {{\n{value}        }}\n"
    );
    s.push_str(&m.lets);
    let _ = writeln!(s, "        let x = x.c.map(SlotArr::<S, T>);");
    let cols = m.apply(|v| match v {
        Ok(c) => MapEntries::constant(*c),
        Err(e) => e.clone(),
    });
    let _ = write!(
        s,
        "        {}\n    }}\n}}\n\n",
        sw.result("S", &format!("[{}]", cols.join(", ")))
    );
    // Inverse transport ~v x v = (~v) x ~(~v).
    let rev_self = if sw.unit {
        "Unit::new_unchecked(self.into_inner().reverse())"
    } else {
        "self.reverse()"
    };
    let _ = write!(
        s,
        "impl<S: Slots, T: Coef, O: gx::Orientation, OV: gx::Orientation> TransformInv<{xn}<S, T, O>> for {self_ty_any} {{\n    type Output = {on}<S, T, O>;\n    #[inline(always)]\n    fn transform_inv(self, x: {xn}<S, T, O>) -> {on}<S, T, O> {{\n        Transform::transform({rev_self}, x)\n    }}\n}}\n\n"
    );
}

/// The prepared action of a sandwich, its non-constant map entries applied sparsely, and its
/// dense map; returns the prepared type.
fn sandwich_prepared(sw: &Sandwich, check: &str, m: &MapEntries, s: &mut String) -> String {
    let (xn, on, self_ty_any, vexpr, vkind) = (sw.xn, sw.on, &sw.self_ty_any, sw.vexpr, &sw.vkind);
    let nst = m.stored.len();
    let prep_ty = format!("gx::Prepared<{vkind}, {xn}<(), f32, O>, T, {nst}>");
    let _ = write!(
        s,
        "impl<T: Coef, O: gx::Orientation, OV: gx::Orientation> gx::Prepare<{xn}<(), f32, O>> for {self_ty_any} {{\n    type Output = {prep_ty};\n    #[inline]\n    fn prepare(self) -> {prep_ty} {{\n        let v = {vexpr};\n{check}{}        gx::Prepared::from_entries([{}])\n    }}\n}}\n\n",
        m.lets,
        m.stored.join(", ")
    );
    let cols = m.apply(|v| m.prepared(v));
    let _ = write!(
        s,
        "{IMPL_ALL} Transform<{xn}<S, T, O>> for {prep_ty} {{\n    type Output = {on}<S, T, O>;\n    #[inline(always)]\n    fn transform(self, x: {xn}<S, T, O>) -> {on}<S, T, O> {{\n        let m = self.m;\n        let x = x.c.map(SlotArr::<S, T>);\n        {}\n    }}\n}}\n\n",
        sw.result("S", &format!("[{}]", cols.join(", ")))
    );
    let _ = write!(
        s,
        "impl<T: Coef, O: gx::Orientation> From<{prep_ty}> for {on}<({xn}<(), f32, O>,), T, O> {{\n    /// The dense map of the prepared action.\n    #[inline]\n    fn from(p: {prep_ty}) -> Self {{\n        let m = p.m;\n        {}\n    }}\n}}\n\n",
        sw.result(
            &format!("({xn}<(), f32, O>,)"),
            &format!("[{}]", m.dense(sw.nxv).join(", "))
        )
    );
    prep_ty
}

/// `gx::batch::SandwichKernel` of a sandwich, behind the batch gate.
fn sandwich_batch(sw: &Sandwich, gate: &str, prep_ty: &str, nst: usize, s: &mut String) {
    let (vn, xn, on, self_ty, vkind) = (sw.vn, sw.xn, sw.on, &sw.self_ty, &sw.vkind);
    let mapped = (0..nst)
        .map(|i| format!("f(m[{i}])"))
        .collect::<Vec<_>>()
        .join(", ");
    let fparam = if nst == 0 { "_f" } else { "mut f" };
    let (wrap, unwrap, cert) = if sw.unit {
        ("gx::Unit::new_unchecked(v)", "v.into_inner()", "Certified")
    } else {
        ("v", "v", "Plain")
    };
    let _ = write!(
        s,
        "{gate}\n{IMPL_O} gx::batch::SandwichKernel<{xn}<(), f32, O>, gx::batch::{cert}> for {vn} {{\n    type Y = {on}<(), f32, O>;\n    type Versor<T: gx::Coef> = {self_ty};\n    type Prepared<T: gx::Coef> = {prep_ty};\n    #[inline(always)]\n    fn wrap<T: gx::Coef>(v: {vn}<(), T>) -> {self_ty} {{\n        {wrap}\n    }}\n    #[inline(always)]\n    fn unwrap<T: gx::Coef>(v: {self_ty}) -> {vn}<(), T> {{\n        {unwrap}\n    }}\n    #[inline(always)]\n    fn prepare<T: gx::Coef>(v: {self_ty}) -> {prep_ty} {{\n        gx::Prepare::<{xn}<(), f32, O>>::prepare(v)\n    }}\n    #[inline(always)]\n    fn map_prepared<T: gx::Coef, W: gx::Coef>(p: {prep_ty}, {fparam}: impl FnMut(T) -> W) -> gx::Prepared<{vkind}, {xn}<(), f32, O>, W, {nst}> {{\n        let m = p.m;\n        gx::Prepared::from_entries([{mapped}])\n    }}\n    #[inline(always)]\n    fn apply_prepared<T: gx::Coef>(p: {prep_ty}, x: {xn}<(), T, O>) -> {on}<(), T, O> {{\n        gx::Transform::transform(p, x)\n    }}\n    #[inline(always)]\n    fn apply<T: gx::Coef>(v: {self_ty}, x: {xn}<(), T, O>) -> {on}<(), T, O> {{\n        gx::Transform::transform(v, x)\n    }}\n}}\n\n"
    );
}

impl Emitter<'_> {
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
        if ![vk, xk, out].iter().all(|k| k.fits_wgsl()) {
            return;
        }
        let u = if unit { "unit_" } else { "" };
        let unit_doc = if unit { "a unit " } else { "" };
        let arg = |param, index| Source::Arg { param, index };
        let vars: BTreeMap<Var, Source> = (0..nv + nx)
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

/// The program made homogeneous in the versor variables
/// `0..nv`: each monomial of lower degree than the highest is multiplied by a power of `norm`,
/// the scalar part of `v ~v`, which is 1 modulo the unit relations. The result is equal to the
/// program modulo the relations, and a uniform drift of the versor scales it uniformly.
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
