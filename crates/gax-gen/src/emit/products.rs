//! The products of pairs of kinds, the unary operations, and division.

use super::{
    Emitter, IMPL_ALL, INLINE_MAX, UNROLL_MAX, kind_literal, term_loop, term_table, zero_slots,
};
use crate::spec::KindSpec;
use crate::table::{BinOp, UnOp, binop_support, binop_table, unop_support, unop_table};
use std::fmt::Write as _;

impl Emitter<'_> {
    /// `a / b = a b⁻¹` for every value `b` with a closed-form inverse whose product with `a`
    /// exists (after the binary products, whose list it reads).
    pub(super) fn divisions(&mut self) {
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
                    "impl<S: Slots, T: Real, OA: gx::Orientation, OB: gx::Orientation> DivBy<{bn}<(), T, OB>> for {an}<S, T, OA> {{\n    type Output = <Self as Gp<{inv}<(), T, OB>>>::Output;\n    #[inline(always)]\n    fn div_by(self, rhs: {bn}<(), T, OB>) -> Self::Output {{\n        Gp::gp(self, rhs.inverse())\n    }}\n}}\n\n"
                );
                // A unit versor's inverse is its reverse: no arithmetic.
                if b.versor && *inv == b.name {
                    let _ = write!(
                        body,
                        "impl<S: Slots, T: Coef, OA: gx::Orientation, OB: gx::Orientation> DivBy<Unit<{bn}<(), T, OB>>> for {an}<S, T, OA> {{\n    type Output = <Self as Gp<{bn}<(), T, OB>>>::Output;\n    /// `self ~rhs`: a unit versor's inverse is its reverse.\n    #[inline(always)]\n    fn div_by(self, rhs: Unit<{bn}<(), T, OB>>) -> Self::Output {{\n        Gp::gp(self, rhs.into_inner().reverse())\n    }}\n}}\n\n"
                    );
                }
            }
        }
        self.w(&body);
    }

    pub(super) fn binary(&mut self, op: BinOp, a: &KindSpec, b: &KindSpec) {
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
        let (tr, m, _) = op.names();
        let (an, bn, on) = (&a.name, &b.name, &out.name);
        // The result's orientation (ADR-051): the product of the factors' signs, flipped once
        // more for the regressive product, the outer product of the duals dualized.
        let oo = if op == BinOp::Vee {
            "gx::orientation::Joined<OA, OB>"
        } else {
            "gx::orientation::Times<OA, OB>"
        };
        let mut body = String::new();
        if table.len() > UNROLL_MAX {
            // A loop over a static table of the terms, by output: unrolled, one such product
            // is megabytes of code, and a debug build gives each of its temporaries a stack
            // slot (ADR-034).
            let terms: Vec<_> = table.iter().map(|t| (t.o, t.i, t.j, t.coef)).collect();
            let nout = out.layout.len();
            let _ = write!(
                body,
                "impl<S1: Slots, S2: Slots, T: Coef, OA: gx::Orientation, OB: gx::Orientation> {tr}<{bn}<S2, T, OB>> for {an}<S1, T, OA> {{\n    type Output = {on}<Cat<S1, S2>, T, {oo}>;\n    #[inline]\n    fn {m}(self, rhs: {bn}<S2, T, OB>) -> {on}<Cat<S1, S2>, T, {oo}> {{\n{}        let a = self.c.map(SlotArr::<S1, T>);\n        let b = rhs.c.map(SlotArr::<S2, T>);\n        let zero = SlotArr::<Cat<S1, S2>, T>(<Cat<S1, S2> as Slots>::from_flat(&mut |_| T::zero(), 0));\n        let mut c = [zero; {nout}];\n{}        {}\n    }}\n}}\n\n",
                term_table("TERMS", &terms, nout),
                term_loop("TERMS", "c", "zero", "a[i] * b[j]", "({p}).scale(k)"),
                kind_literal(on, "c.map(|x| x.0)"),
            );
            self.w(&body);
            self.stats
                .products
                .push((op, a.name.clone(), b.name.clone(), out.name.clone()));
            return;
        }
        let exprs: Vec<String> = (0..out.layout.len())
            .map(|o| {
                let items: Vec<(bool, String)> = table
                    .iter()
                    .filter(|t| t.o == o)
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
                if items.is_empty() {
                    zero_slots("<Cat<S1, S2> as Slots>")
                } else {
                    format!("({}).0", balanced_sum(&items))
                }
            })
            .collect();
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
        // Maps whose slots make the unrolled product large run it once per pair of slot entries
        // instead (`gx::__private::by_entries`); the condition is decided per monomorphization, so
        // values and small maps compile only the unrolled code.
        let by_entries = format!(
            "        if const {{ <S1 as Slots>::SIZE * <S2 as Slots>::SIZE > 1 && {} * <S1 as Slots>::SIZE * <S2 as Slots>::SIZE > gx::slots::SLOT_UNROLL_MAX }} {{\n            return gx::__private::by_entries(&self, &rhs, |x, y| {tr}::{m}(x, y));\n        }}\n",
            table.len()
        );
        let _ = write!(
            body,
            "impl<S1: Slots, S2: Slots, T: Coef, OA: gx::Orientation, OB: gx::Orientation> {tr}<{bn}<S2, T, OB>> for {an}<S1, T, OA> {{\n    type Output = {on}<Cat<S1, S2>, T, {oo}>;\n    {inline}\n    fn {m}(self, rhs: {bn}<S2, T, OB>) -> {on}<Cat<S1, S2>, T, {oo}> {{\n{by_entries}        let a = self.c.map(SlotArr::<S1, T>);\n        let b = rhs.c.map(SlotArr::<S2, T>);\n{ops}        {on} {{\n            o: core::marker::PhantomData,\n            c: [\n"
        );
        for e in exprs {
            let _ = writeln!(body, "                {e},");
        }
        body.push_str("            ],\n        }\n    }\n}\n\n");
        self.w(&body);
        self.stats
            .products
            .push((op, a.name.clone(), b.name.clone(), out.name.clone()));
    }

    pub(super) fn unary(&mut self, op: UnOp, k: &KindSpec) {
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
        let (tr, m) = (op.trait_name(), op.name());
        let (kn, on) = (&k.name, &out.name);
        // The dual and its inverse flip the orientation; the involutions keep it (ADR-051).
        let oo = if matches!(op, UnOp::Dual | UnOp::Undual) {
            "gx::orientation::Flip<O>"
        } else {
            "O"
        };
        let exprs: Vec<String> = (0..out.layout.len())
            .map(|o| match table.iter().find(|t| t.o == o) {
                None => zero_slots("S"),
                Some(t) if t.coef == 1 => format!("a[{}]", t.i),
                Some(t) if t.coef == -1 => format!("(-SlotArr::<S, T>(a[{}])).0", t.i),
                Some(t) => format!(
                    "SlotArr::<S, T>(a[{}]).scale(T::from_i64({})).0",
                    t.i, t.coef
                ),
            })
            .collect();
        let _ = write!(
            self.out,
            "{IMPL_ALL} {tr} for {kn}<S, T, O> {{\n    type Output = {on}<S, T, {oo}>;\n    #[inline]\n    fn {m}(self) -> {on}<S, T, {oo}> {{\n        let a = self.c;\n        {}\n    }}\n}}\n\n",
            kind_literal(on, &format!("[{}]", exprs.join(", ")))
        );
    }
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
