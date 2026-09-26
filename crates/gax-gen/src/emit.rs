//! Rust code emission for an algebra.
//!
//! The output is one module of ordinary Rust: a struct per kind, the product traits of
//! `gax-core` implemented per pair of kinds (tier 1: straight-line, generic over slots), and
//! fused sandwich kernels for the versor kinds (tier 2: simplified symbolically).

use crate::cse;
use crate::poly::{Poly, Rational, Var};
use crate::slp::{Operand, Program, render};
use crate::spec::{AlgebraSpec, KindSpec};
use crate::symbolic::{self, SymMv};
use crate::table::{BinOp, UnOp, binop_support, binop_table, unop_support, unop_table};
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
}

/// Summary statistics of an emitted algebra.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    /// Number of generated binary product impls.
    pub binary_impls: usize,
    /// Number of generated sandwich kernels.
    pub sandwich_impls: usize,
    /// Op counts of the value path of each sandwich kernel: `(versor, target, unit, cost)`.
    pub sandwich_costs: Vec<(String, String, bool, crate::slp::Cost)>,
    /// Every generated binary impl: `(op, left, right)`.
    pub binary: Vec<(BinOp, String, String)>,
    /// The value methods emitted per kind.
    pub values: Vec<crate::emit_values::ValueMethods>,
}

/// Emit the module source for an algebra.
pub fn emit(spec: &AlgebraSpec, cfg: &Config) -> (String, Stats) {
    let mut e = Emitter {
        spec,
        cfg,
        out: String::new(),
        stats: Stats::default(),
    };
    e.header();
    for k in &spec.kinds {
        e.kind(k);
        let (methods, meta) = crate::emit_values::value_methods(spec, k);
        e.out.push_str(&methods);
        e.stats.values.push(meta);
    }
    for a in &spec.kinds {
        for b in &spec.kinds {
            let eq = if a.name == b.name { "True" } else { "False" };
            let _ = writeln!(
                e.out,
                "impl gx::KindEq<{}> for {} {{\n    type Out = gx::{eq};\n}}\n",
                b.name, a.name
            );
        }
    }
    for op in BinOp::ALL {
        for a in &spec.kinds {
            for b in &spec.kinds {
                e.binary(op, a, b);
            }
        }
    }
    for k in &spec.kinds {
        for op in UnOp::ALL {
            e.unary(op, k);
        }
    }
    for v in spec.kinds.iter().filter(|k| k.versor) {
        for x in &spec.kinds {
            e.sandwich(v, x, false);
            e.sandwich(v, x, true);
        }
    }
    e.out.push_str(&crate::emit_outer::outermorphisms(spec));
    for (alias, kind) in &spec.aliases {
        let _ = writeln!(
            e.out,
            "/// Alias of [`{kind}`].\npub type {alias}<S = (), T = f32> = {kind}<S, T>;\n"
        );
    }
    if let Some(gate) = &cfg.batch {
        let names = spec
            .kinds
            .iter()
            .map(|k| (&k.name, &k.name))
            .chain(spec.aliases.iter().map(|(a, k)| (a, k)));
        for (name, kind) in names {
            let _ = writeln!(
                e.out,
                "{gate}\n/// [`{kind}`] values in struct-of-arrays form, for the batch kernels.\npub type {name}Soa<E = f32> = gx::batch::Soa<{kind}, E>;\n"
            );
        }
    }
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
}

/// Rust identifier for a blade accessor (`1` becomes `s`).
fn blade_ident(b: &str) -> String {
    if b == "1" { "s".into() } else { b.to_string() }
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
        for (i, c) in alg.basis_suffixes().iter().enumerate() {
            let row: Vec<String> = alg.metric()[i].iter().map(ToString::to_string).collect();
            let _ = writeln!(basis, "//   e{c}: [{}]", row.join(", "));
        }
        let _ = write!(
            self.out,
            "// @generated by gax-gen from the `{}` declaration. Do not edit by hand.\n\
             // Metric (rows e_i . e_j):\n{basis}\n\
             use {core} as gx;\n\n",
            spec.name
        );
    }

    #[allow(clippy::too_many_lines)]
    fn kind(&mut self, k: &KindSpec) {
        let name = &k.name;
        let n = k.layout.len();
        let core = self.cfg.core.clone();
        let scalar = self.spec.scalar_kind().name.clone();
        let blades_list: Vec<String> = k.blades.iter().map(|b| format!("{b:?}")).collect();
        let doc = if k.doc.is_empty() {
            format!("A `{name}` of the `{}` algebra.", self.spec.name)
        } else {
            k.doc.clone()
        };
        let blade_doc = k.blades.join(", ");
        // Spelled out rather than `core::array::from_fn`: batch kernels need these inlined
        // (see `gax::batch`), and the core helpers are not always.
        let each = |f: &dyn Fn(usize) -> String| (0..n).map(f).collect::<Vec<_>>().join(", ");
        let arr_ff = each(&|i| format!("f({i})"));
        let arr_fm = each(&|i| format!("f(&a[{i}])"));
        let arr_fz = each(&|i| format!("f(&a[{i}], &b[{i}])"));
        let s = format!(
            r#"#[doc = {doc:?}]
///
/// Blades, in coefficient order: `[{blade_doc}]`.
///
/// `S` lists the open slots: `{name}` (that is, `{name}<()>`) is a value, `{name}<(A,)>` a
/// linear map from `A`, `{name}<(A, B)>` a bilinear map. `T` is the coefficient type.
#[repr(C)]
pub struct {name}<S: Slots = (), T: Coef = f32> {{
    /// Coefficients, output first: `c[i]` is the slot array of blade `i`.
    pub c: [S::Arr<T>; {n}],
}}

impl<S: Slots, T: Coef> Clone for {name}<S, T> {{
    #[inline(always)]
    fn clone(&self) -> Self {{
        *self
    }}
}}

impl<S: Slots, T: Coef> Copy for {name}<S, T> {{}}

impl<S: Slots, T: Coef> PartialEq for {name}<S, T> {{
    fn eq(&self, other: &Self) -> bool {{
        self.c == other.c
    }}
}}

impl<S: Slots, T: Coef> core::fmt::Debug for {name}<S, T> {{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {{
        let mut d = f.debug_struct("{name}");
        for (b, c) in <{name} as Kind>::BLADES.iter().zip(self.c.iter()) {{
            d.field(b, c);
        }}
        d.finish()
    }}
}}

impl Kind for {name} {{
    const N: usize = {n};
    const NAME: &'static str = "{name}";
    const MODULE: &'static str = module_path!();
    const BLADES: &'static [&'static str] = &[{blades}];
    type Arr<X: gx::Elem> = [X; {n}];
    type Mv<S: Slots, T: Coef> = {name}<S, T>;
    type Scalar = {scalar};
    #[inline(always)]
    fn arr_from_fn<X: gx::Elem>(mut f: impl FnMut(usize) -> X) -> [X; {n}] {{
        [{arr_ff}]
    }}
    #[inline(always)]
    fn arr_map<X: gx::Elem, Y: gx::Elem>(a: &[X; {n}], mut f: impl FnMut(&X) -> Y) -> [Y; {n}] {{
        [{arr_fm}]
    }}
    #[inline(always)]
    fn arr_zip<X: gx::Elem, Y: gx::Elem, Z: gx::Elem>(
        a: &[X; {n}],
        b: &[Y; {n}],
        mut f: impl FnMut(&X, &Y) -> Z,
    ) -> [Z; {n}] {{
        [{arr_fz}]
    }}
}}

impl<S: Slots, T: Coef> Extensor for {name}<S, T> {{
    type Kind = {name};
    type Slots = S;
    type Coef = T;
    #[inline(always)]
    fn from_coeffs(c: [S::Arr<T>; {n}]) -> Self {{
        {name} {{ c }}
    }}
    #[inline(always)]
    fn coeffs(&self) -> &[S::Arr<T>; {n}] {{
        &self.c
    }}
}}

impl<S: Slots, T: Coef> {name}<S, T> {{
    /// Construct from output-first coefficients.
    #[inline(always)]
    pub const fn from_coeffs(c: [S::Arr<T>; {n}]) -> Self {{
        {name} {{ c }}
    }}

    /// All coefficients zero.
    #[inline(always)]
    pub fn zero() -> Self {{
        {name} {{ c: [S::from_flat(&mut |_| T::zero(), 0); {n}] }}
    }}

    /// Fill the first open slot with a value, or compose a map into it.
    #[inline(always)]
    pub fn of<X>(self, x: X) -> <Self as Of<X>>::Output
    where
        Self: Of<X>,
    {{
        Of::of(self, x)
    }}

    /// Move open slot `I` to the front, so that `.of(x)` fills it: `m.at::<1>().of(x)`.
    #[inline(always)]
    pub fn at<const I: usize>(self) -> {name}<<S as gx::MoveToFront<I>>::Moved, T>
    where
        S: gx::MoveToFront<I>,
    {{
        {name} {{ c: self.c.map(|col| <S as gx::MoveToFront<I>>::move_arr(&col)) }}
    }}

    /// Contract the output with open slot `I` (which must be of kind `{name}`): a trace with no
    /// metric, leaving a scalar with the other slots (numga's `trace(slot)`).
    #[inline(always)]
    pub fn trace_at<const I: usize>(self) -> <{name}<<S as gx::MoveToFront<I>>::Moved, T> as gx::TraceFirst>::Output
    where
        S: gx::MoveToFront<I>,
        {name}<<S as gx::MoveToFront<I>>::Moved, T>: gx::TraceFirst,
    {{
        gx::TraceFirst::trace_first(self.at::<I>())
    }}

    /// The outermorphism of this map on vectors (or antivectors) to the kind `B`:
    /// `m.outermorphism::<Line>().of(a ^ b) == m.of(a) ^ m.of(b)` (with `&` for antivectors).
    #[inline(always)]
    pub fn outermorphism<B>(self) -> <Self as gx::Outermorphism<B>>::Output
    where
        Self: gx::Outermorphism<B>,
    {{
        gx::Outermorphism::outermorphism(self)
    }}

    /// Fill every open slot of `x`'s kind with the value `x`; the other slots stay open.
    #[inline(always)]
    pub fn fill<X>(self, x: X) -> {name}<<S as gx::FillList<X::Kind>>::Out, T>
    where
        X: Extensor<Slots = (), Coef = T>,
        S: gx::FillList<X::Kind>,
    {{
        let xc = x.coeffs();
        {name} {{ c: self.c.map(|col| <S as gx::FillList<X::Kind>>::fill(&col, xc)) }}
    }}
"#,
            blades = blades_list.join(", "),
        );
        self.w(&s);
        for (tr, m) in [
            ("Gp", "gp"),
            ("Wedge", "wedge"),
            ("Vee", "vee"),
            ("Dot", "dot"),
            ("Lc", "lc"),
            ("Rc", "rc"),
            ("ScalarProduct", "scalar_product"),
            ("Commutator", "commutator"),
            ("Anticommutator", "anticommutator"),
            ("Transform", "transform"),
            ("TransformInv", "transform_inv"),
        ] {
            let _ = write!(
                self.out,
                "\n    /// See [`{tr}`].\n    #[inline(always)]\n    pub fn {m}<R>(self, rhs: R) -> <Self as {tr}<R>>::Output\n    where\n        Self: {tr}<R>,\n    {{\n        {tr}::{m}(self, rhs)\n    }}\n"
            );
        }
        for (tr, m) in [
            ("Reverse", "reverse"),
            ("Involute", "involute"),
            ("Conjugate", "conjugate"),
            ("Dual", "dual"),
            ("Undual", "undual"),
        ] {
            let _ = write!(
                self.out,
                "\n    /// See [`{tr}`].\n    #[inline(always)]\n    pub fn {m}(self) -> <Self as {tr}>::Output {{\n        {tr}::{m}(self)\n    }}\n"
            );
        }
        self.w("}\n\n");

        // Value constructors and accessors.
        let params: Vec<String> = k
            .blades
            .iter()
            .map(|b| format!("{}: T", blade_ident(b)))
            .collect();
        let args: Vec<String> = k.blades.iter().map(|b| blade_ident(b)).collect();
        let _ = write!(
            self.out,
            "impl<T: Coef> {name}<(), T> {{\n    /// A value from its coefficients, in blade order.\n    #[inline(always)]\n    #[allow(clippy::too_many_arguments)]\n    pub const fn new({}) -> Self {{\n        {name} {{ c: [{}] }}\n    }}\n\n    /// Prepare this versor's action on kind `X` for applying it to many objects (see
    /// [`gx::Prepared`]).
    #[inline(always)]
    pub fn prepare<X>(self) -> <Self as gx::Prepare<X>>::Output
    where
        Self: gx::Prepare<X>,
    {{
        gx::Prepare::prepare(self)
    }}

    /// The identity map on `{name}`: a `{name}` with one open `{name}` slot.\n    #[inline(always)]\n    pub fn slot() -> {name}<({name},), T> {{\n        {name} {{ c: core::array::from_fn(|i| core::array::from_fn(|j| if i == j {{ T::one() }} else {{ T::zero() }})) }}\n    }}\n",
            params.join(", "),
            args.join(", ")
        );
        for (i, b) in k.blades.iter().enumerate() {
            let _ = write!(
                self.out,
                "\n    /// Coefficient of `{b}`.\n    #[inline(always)]\n    pub fn {}(&self) -> T {{\n        self.c[{i}]\n    }}\n",
                blade_ident(b)
            );
        }
        self.w("}\n\n");

        // Methods of maps and forms (see `gax_core::extensor`).
        let _ = write!(
            self.out,
            r"impl<A: Kind, T: Real> {name}<(A,), T> {{
    /// The inverse map, `A <- {name}`.
    #[inline]
    pub fn inverse(self) -> <Self as SquareMap>::Inverse
    where
        Self: SquareMap<Coef = T, Kind = {name}, Input = A>,
    {{
        SquareMap::inverse(self)
    }}

    /// The determinant of the map's coefficient matrix.
    #[inline]
    pub fn det(self) -> T
    where
        Self: SquareMap<Coef = T, Kind = {name}, Input = A>,
    {{
        SquareMap::det(self)
    }}

    /// Solve `self.of(x) == rhs` for `x`; a right-hand side with slots keeps them.
    #[inline]
    pub fn solve<X>(self, rhs: X) -> <A as Kind>::Mv<X::Slots, T>
    where
        Self: SquareMap<Coef = T, Kind = {name}, Input = A>,
        X: Extensor<Kind = {name}, Coef = T>,
    {{
        SquareMap::solve(self, rhs)
    }}

    /// Singular value decomposition: `(u, sigma, v)` with `self.of(v[i]) == sigma[i] * u[i]`.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn svd(self) -> (<{name} as Kind>::Arr<{name}<(), T>>, <{name} as Kind>::Arr<T>, <A as Kind>::Arr<<A as Kind>::Mv<(), T>>)
    where
        Self: SquareMap<Coef = T, Kind = {name}, Input = A>,
    {{
        SquareMap::svd(self)
    }}

    /// The trace of a map from `{name}` to itself.
    #[inline]
    pub fn trace(self) -> T
    where
        Self: Endomorphism<Coef = T, Kind = {name}, Input = {name}>,
    {{
        Endomorphism::trace(self)
    }}
}}

impl<A: Kind, B: Kind, T: Coef> {name}<(A, B), T> {{
    /// Exchange the two slots.
    #[inline]
    pub fn swap(self) -> {name}<(B, A), T> {{
        self.at::<1>()
    }}
}}

impl<A: Kind, B: Kind, T: Real> {name}<(A, B), T> {{
    /// Solve `self(x, ·) == rhs(l, ·)` for `x`; leading slots of `rhs` become slots of `x`.
    #[inline]
    pub fn solve<R>(self, rhs: R) -> <A as Kind>::Mv<<R::Slots as SplitLast>::Init, T>
    where
        Self: Pairing<Coef = T, Kind = {name}, First = A, Second = B>,
        R: Extensor<Kind = {name}, Coef = T>,
        R::Slots: SplitLast<Last = B>,
    {{
        Pairing::solve(self, rhs)
    }}
}}

impl<A: Kind, T: Real> {name}<(A, A), T> {{
    /// Generalized symmetric eigenproblem against a positive definite metric form. Returns the
    /// eigenvalues (ascending) and the eigenvectors, as values of the slot kind.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn eigh_with(self, metric: Self) -> (<A as Kind>::Arr<T>, <A as Kind>::Arr<<A as Kind>::Mv<(), T>>)
    where
        Self: Form<Coef = T, Kind = {name}, Slot = A>,
    {{
        Form::eigh_with(self, metric)
    }}

    /// Symmetric eigenproblem in the coefficient basis (identity metric).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn eigh(self) -> (<A as Kind>::Arr<T>, <A as Kind>::Arr<<A as Kind>::Mv<(), T>>)
    where
        Self: Form<Coef = T, Kind = {name}, Slot = A>,
    {{
        Form::eigh(self)
    }}
}}

"
        );

        // Arithmetic.
        let arr = |op: &str| {
            format!(
                "{name} {{ c: core::array::from_fn(|i| (SlotArr::<S, T>(self.c[i]) {op} SlotArr::<S, T>(rhs.c[i])).0) }}"
            )
        };
        let _ = write!(
            self.out,
            r"impl<S: Slots, T: Coef> core::ops::Add for {name}<S, T> {{
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {{
        {add}
    }}
}}

impl<S: Slots, T: Coef> core::ops::Sub for {name}<S, T> {{
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {{
        {sub}
    }}
}}

impl<S: Slots, T: Coef> core::ops::AddAssign for {name}<S, T> {{
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {{
        *self = *self + rhs;
    }}
}}

impl<S: Slots, T: Coef> core::ops::SubAssign for {name}<S, T> {{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {{
        *self = *self - rhs;
    }}
}}

impl<S: Slots, T: Coef> core::ops::Neg for {name}<S, T> {{
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {{
        {name} {{ c: self.c.map(|x| (-SlotArr::<S, T>(x)).0) }}
    }}
}}

impl<S: Slots, T: Coef> Gp<T> for {name}<S, T> {{
    type Output = Self;
    #[inline(always)]
    fn gp(self, rhs: T) -> Self {{
        {name} {{ c: self.c.map(|x| SlotArr::<S, T>(x).scale(rhs).0) }}
    }}
}}

impl<S: Slots, T: Real> core::ops::Div<T> for {name}<S, T> {{
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: T) -> Self {{
        let r = rhs.recip();
        {name} {{ c: self.c.map(|x| SlotArr::<S, T>(x).scale(r).0) }}
    }}
}}

impl<S: Slots> core::ops::Mul<{name}<S, f32>> for f32 {{
    type Output = {name}<S, f32>;
    #[inline(always)]
    fn mul(self, rhs: {name}<S, f32>) -> {name}<S, f32> {{
        rhs.gp(self)
    }}
}}

impl<S: Slots> core::ops::Mul<{name}<S, f64>> for f64 {{
    type Output = {name}<S, f64>;
    #[inline(always)]
    fn mul(self, rhs: {name}<S, f64>) -> {name}<S, f64> {{
        rhs.gp(self)
    }}
}}

",
            add = arr("+"),
            sub = arr("-"),
        );
        for (std_tr, std_m, tr, m) in [
            ("Mul", "mul", "Gp", "gp"),
            ("BitXor", "bitxor", "Wedge", "wedge"),
            ("BitAnd", "bitand", "Vee", "vee"),
            ("BitOr", "bitor", "Dot", "dot"),
            ("Shr", "shr", "Transform", "transform"),
            ("Shl", "shl", "TransformInv", "transform_inv"),
        ] {
            let _ = write!(
                self.out,
                "impl<S: Slots, T: Coef, R> core::ops::{std_tr}<R> for {name}<S, T>\nwhere\n    Self: {tr}<R>,\n{{\n    type Output = <Self as {tr}<R>>::Output;\n    #[inline(always)]\n    fn {std_m}(self, rhs: R) -> Self::Output {{\n        {tr}::{m}(self, rhs)\n    }}\n}}\n\n"
            );
        }
        let _ = core;
    }

    fn binary(&mut self, op: BinOp, a: &KindSpec, b: &KindSpec) {
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
        let (tr, m) = trait_of(op);
        let (an, bn, on) = (&a.name, &b.name, &out.name);
        let mut body = String::new();
        let mut exprs = Vec::new();
        for o in 0..out.layout.len() {
            let terms: Vec<_> = table.iter().filter(|t| t.o == o).collect();
            if terms.is_empty() {
                exprs.push("<Cat<S1, S2> as Slots>::from_flat(&mut |_| T::zero(), 0)".to_string());
                continue;
            }
            let items: Vec<(bool, String)> = terms
                .iter()
                .map(|t| {
                    let prod = format!("a[{}] * b[{}]", t.i, t.j);
                    let mag = t.coef.abs();
                    let term = if mag == 1 {
                        prod
                    } else {
                        format!("({prod}).scale(T::from_i64({mag}))")
                    };
                    (t.coef < 0, term)
                })
                .collect();
            let e = balanced_sum(&items);
            exprs.push(format!("({e}).0"));
        }
        let _ = write!(
            body,
            "impl<S1: Slots, S2: Slots, T: Coef> {tr}<{bn}<S2, T>> for {an}<S1, T> {{\n    type Output = {on}<Cat<S1, S2>, T>;\n    #[inline(always)]\n    fn {m}(self, rhs: {bn}<S2, T>) -> {on}<Cat<S1, S2>, T> {{\n        let a = self.c.map(SlotArr::<S1, T>);\n        let b = rhs.c.map(SlotArr::<S2, T>);\n        {on} {{\n            c: [\n"
        );
        for e in exprs {
            let _ = writeln!(body, "                {e},");
        }
        body.push_str("            ],\n        }\n    }\n}\n\n");
        self.w(&body);
        self.stats.binary_impls += 1;
        self.stats.binary.push((op, a.name.clone(), b.name.clone()));
    }

    fn unary(&mut self, op: UnOp, k: &KindSpec) {
        let alg = &self.spec.algebra;
        let support = unop_support(alg, op, &k.layout);
        let out = self
            .spec
            .kind_for_support(&support)
            .expect("a full kind exists")
            .clone();
        let table = unop_table(alg, op, &k.layout, &out.layout);
        let (tr, m) = unop_trait(op);
        let (kn, on) = (&k.name, &out.name);
        let mut exprs = Vec::new();
        for o in 0..out.layout.len() {
            match table.iter().find(|t| t.o == o) {
                None => exprs.push("S::from_flat(&mut |_| T::zero(), 0)".to_string()),
                Some(t) if t.coef == 1 => exprs.push(format!("a[{}]", t.i)),
                Some(t) if t.coef == -1 => exprs.push(format!("(-SlotArr::<S, T>(a[{}])).0", t.i)),
                Some(t) => exprs.push(format!(
                    "SlotArr::<S, T>(a[{}]).scale(T::from_i64({})).0",
                    t.i, t.coef
                )),
            }
        }
        let _ = write!(
            self.out,
            "impl<S: Slots, T: Coef> {tr} for {kn}<S, T> {{\n    type Output = {on}<S, T>;\n    #[inline]\n    fn {m}(self) -> {on}<S, T> {{\n        let a = self.c;\n        {on} {{ c: [{}] }}\n    }}\n}}\n\n",
            exprs.join(", ")
        );
    }

    /// Fused `v x ~v` with `v` a value of versor kind `vk` and `x` of kind `xk` with any slots.
    #[allow(clippy::too_many_lines)]
    fn sandwich(&mut self, vk: &KindSpec, xk: &KindSpec, unit: bool) {
        let alg = &self.spec.algebra;
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
            return;
        }
        // The output kind comes from the support modulo the relations (a unit versor's
        // sandwich keeps the passenger's grades); the kernels are compiled from the unreduced
        // polynomials, and the portfolio decides whether reducing them pays.
        let mut reduced = res.clone();
        for p in reduced.values_mut() {
            *p = cse::reduce_by_relations(p, &relations);
        }
        reduced.retain(|_, p| !p.is_zero());
        let support = symbolic::support(&reduced);
        if support.is_empty() {
            return;
        }
        let out = self
            .spec
            .kind_for_support(&support)
            .expect("a full kind exists")
            .clone();
        let projected: SymMv = res
            .into_iter()
            .filter(|(m, _)| out.layout.position(*m).is_some())
            .collect();
        let coeffs = symbolic::to_coeffs(&out.layout, &projected).expect("support fits");
        let passengers: BTreeSet<Var> = (nv..nv + xk.layout.len() as Var).collect();

        // Value path: the whole formula, simplified jointly.
        let direct = cse::compile_best(&coeffs, &passengers, &relations);
        debug_assert!(check_equal(&direct, &coeffs, &relations));

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
        let matrix = cse::compile_best(&matrix_polys, &BTreeSet::new(), &relations);

        let (vn, xn, on) = (&vk.name, &xk.name, &out.name);
        let self_ty = if unit {
            format!("Unit<{vn}<(), T>>")
        } else {
            format!("{vn}<(), T>")
        };
        let vexpr = if unit {
            "self.into_inner().c"
        } else {
            "self.c"
        };
        let nxv = xk.layout.len();
        let nout = out.layout.len();
        let var_name = move |var: Var| {
            if var < nv {
                format!("v[{var}]")
            } else {
                format!("xv[{}]", var - nv)
            }
        };
        let mut s = String::new();
        let _ = write!(
            s,
            "impl<S: Slots, T: Coef> Transform<{xn}<S, T>> for {self_ty} {{\n    type Output = {on}<S, T>;\n    #[inline(always)]\n    fn transform(self, x: {xn}<S, T>) -> {on}<S, T> {{\n        let v = {vexpr};\n        if let Some(xv) = gx::slots::values::<S, T, {nxv}>(&x.c) {{\n"
        );
        direct.emit_lets(&var_name, "t", &mut s);
        let outs: Vec<String> = direct
            .outputs
            .iter()
            .map(|o| render(o, &var_name, "t"))
            .collect();
        let _ = write!(
            s,
            "            return {on} {{ c: gx::slots::from_values::<S, T, {nout}>([{}]) }};\n        }}\n",
            outs.join(", ")
        );
        let mat_name = move |var: Var| format!("v[{var}]");
        matrix.emit_lets(&mat_name, "m", &mut s);
        let _ = writeln!(s, "        let x = x.c.map(SlotArr::<S, T>);");
        let mut cols = Vec::new();
        for o in 0..nout {
            let mut e = String::new();
            for (k, (_, i, _)) in entries.iter().enumerate().filter(|(_, e)| e.0 == o) {
                let coef = matrix.outputs[k];
                let term = match coef {
                    Operand::Const(c) if c == Rational::ONE => format!("x[{i}]"),
                    Operand::Const(c) if c == -Rational::ONE => format!("-x[{i}]"),
                    other => format!("x[{i}].scale({})", render(&other, &mat_name, "m")),
                };
                if e.is_empty() {
                    e = term;
                } else if let Some(stripped) = term.strip_prefix('-') {
                    e = format!("{e} - {stripped}");
                } else {
                    e = format!("{e} + {term}");
                }
            }
            if e.is_empty() {
                cols.push("S::from_flat(&mut |_| T::zero(), 0)".to_string());
            } else {
                cols.push(format!("({e}).0"));
            }
        }
        let _ = write!(
            s,
            "        {on} {{ c: [{}] }}\n    }}\n}}\n\n",
            cols.join(", ")
        );
        // Inverse transport ~v x v = (~v) x ~(~v).
        let rev_self = if unit {
            "Unit::new_unchecked(self.into_inner().reverse())"
        } else {
            "self.reverse()"
        };
        let _ = write!(
            s,
            "impl<S: Slots, T: Coef> TransformInv<{xn}<S, T>> for {self_ty} {{\n    type Output = {on}<S, T>;\n    #[inline(always)]\n    fn transform_inv(self, x: {xn}<S, T>) -> {on}<S, T> {{\n        Transform::transform({rev_self}, x)\n    }}\n}}\n\n"
        );
        // Prepared action: the non-constant matrix entries, applied sparsely.
        let mut stored: Vec<Operand> = Vec::new();
        for (k, _) in entries.iter().enumerate() {
            let op = matrix.outputs[k];
            if !matches!(op, Operand::Const(_)) && !stored.contains(&op) {
                stored.push(op);
            }
        }
        let nst = stored.len();
        let vkind = if unit {
            format!("gx::Unit<{vn}>")
        } else {
            vn.clone()
        };
        let prep_ty = format!("gx::Prepared<{vkind}, {xn}, T, {nst}>");
        let mut lets = String::new();
        matrix.emit_lets(&mat_name, "m", &mut lets);
        let vals: Vec<String> = stored.iter().map(|o| render(o, &mat_name, "m")).collect();
        let _ = write!(
            s,
            "impl<T: Coef> gx::Prepare<{xn}> for {self_ty} {{\n    type Output = {prep_ty};\n    #[inline]\n    fn prepare(self) -> {prep_ty} {{\n        let v = {vexpr};\n{lets}        gx::Prepared::from_entries([{}])\n    }}\n}}\n\n",
            vals.join(", ")
        );
        let mut cols = Vec::new();
        for o in 0..nout {
            let mut e = String::new();
            for (k, (_, i, _)) in entries.iter().enumerate().filter(|(_, e)| e.0 == o) {
                let term = match matrix.outputs[k] {
                    Operand::Const(c) if c == Rational::ONE => format!("x[{i}]"),
                    Operand::Const(c) if c == -Rational::ONE => format!("-x[{i}]"),
                    Operand::Const(c) => format!(
                        "x[{i}].scale({})",
                        render(&Operand::Const(c), &mat_name, "m")
                    ),
                    other => {
                        let idx = stored.iter().position(|s| *s == other).expect("stored");
                        format!("x[{i}].scale(m[{idx}])")
                    }
                };
                if e.is_empty() {
                    e = term;
                } else if let Some(stripped) = term.strip_prefix('-') {
                    e = format!("{e} - {stripped}");
                } else {
                    e = format!("{e} + {term}");
                }
            }
            cols.push(if e.is_empty() {
                "S::from_flat(&mut |_| T::zero(), 0)".to_string()
            } else {
                format!("({e}).0")
            });
        }
        let _ = write!(
            s,
            "impl<S: Slots, T: Coef> Transform<{xn}<S, T>> for {prep_ty} {{\n    type Output = {on}<S, T>;\n    #[inline(always)]\n    fn transform(self, x: {xn}<S, T>) -> {on}<S, T> {{\n        let m = self.m;\n        let x = x.c.map(SlotArr::<S, T>);\n        {on} {{ c: [{}] }}\n    }}\n}}\n\n",
            cols.join(", ")
        );
        // The dense map, written entry by entry (no multiplication by an identity's zeros).
        let nx = xk.layout.len();
        let mut rows = Vec::new();
        for o in 0..nout {
            let row: Vec<String> = (0..nx)
                .map(
                    |i| match entries.iter().position(|e| e.0 == o && e.1 == i) {
                        None => "T::zero()".to_string(),
                        Some(k) => match matrix.outputs[k] {
                            Operand::Const(c) => render(&Operand::Const(c), &mat_name, "m"),
                            other => format!(
                                "m[{}]",
                                stored.iter().position(|s| *s == other).expect("stored")
                            ),
                        },
                    },
                )
                .collect();
            rows.push(format!("[{}]", row.join(", ")));
        }
        let _ = write!(
            s,
            "impl<T: Coef> From<{prep_ty}> for {on}<({xn},), T> {{\n    /// The dense map of the prepared action.\n    #[inline]\n    fn from(p: {prep_ty}) -> Self {{\n        let m = p.m;\n        {on} {{ c: [{}] }}\n    }}\n}}\n\n",
            rows.join(", ")
        );
        if let Some(gate) = &self.cfg.batch {
            let mapped = (0..nst)
                .map(|i| format!("f(m[{i}])"))
                .collect::<Vec<_>>()
                .join(", ");
            let fparam = if nst == 0 { "_f" } else { "mut f" };
            let (wrap, unwrap, cert) = if unit {
                ("gx::Unit::new_unchecked(v)", "v.into_inner()", "Certified")
            } else {
                ("v", "v", "Plain")
            };
            let _ = write!(
                s,
                "{gate}\nimpl gx::batch::SandwichKernel<{xn}, gx::batch::{cert}> for {vn} {{\n    type Y = {on};\n    type Versor<T: gx::Coef> = {self_ty};\n    type Prepared<T: gx::Coef> = {prep_ty};\n    #[inline(always)]\n    fn wrap<T: gx::Coef>(v: {vn}<(), T>) -> {self_ty} {{\n        {wrap}\n    }}\n    #[inline(always)]\n    fn unwrap<T: gx::Coef>(v: {self_ty}) -> {vn}<(), T> {{\n        {unwrap}\n    }}\n    #[inline(always)]\n    fn prepare<T: gx::Coef>(v: {self_ty}) -> {prep_ty} {{\n        gx::Prepare::<{xn}>::prepare(v)\n    }}\n    #[inline(always)]\n    fn map_prepared<T: gx::Coef, W: gx::Coef>(p: {prep_ty}, {fparam}: impl FnMut(T) -> W) -> gx::Prepared<{vkind}, {xn}, W, {nst}> {{\n        let m = p.m;\n        gx::Prepared::from_entries([{mapped}])\n    }}\n    #[inline(always)]\n    fn apply_prepared<T: gx::Coef>(p: {prep_ty}, x: {xn}<(), T>) -> {on}<(), T> {{\n        gx::Transform::transform(p, x)\n    }}\n    #[inline(always)]\n    fn apply<T: gx::Coef>(v: {self_ty}, x: {xn}<(), T>) -> {on}<(), T> {{\n        gx::Transform::transform(v, x)\n    }}\n}}\n\n"
            );
        }
        self.w(&s);
        self.stats.sandwich_impls += 1;
        self.stats
            .sandwich_costs
            .push((vn.clone(), xn.clone(), unit, direct.cost()));
    }
}

fn check_equal(prog: &Program, want: &[Poly], relations: &[Poly]) -> bool {
    let got = prog.to_polys();
    got.iter().zip(want).all(|(g, w)| {
        let d = g - w;
        d.is_zero() || cse::reduce_by_relations(&d, relations).is_zero()
    })
}

fn trait_of(op: BinOp) -> (&'static str, &'static str) {
    match op {
        BinOp::Gp => ("Gp", "gp"),
        BinOp::Wedge => ("Wedge", "wedge"),
        BinOp::Vee => ("Vee", "vee"),
        BinOp::Lc => ("Lc", "lc"),
        BinOp::Rc => ("Rc", "rc"),
        BinOp::Dot => ("Dot", "dot"),
        BinOp::Scalar => ("ScalarProduct", "scalar_product"),
        BinOp::Commutator => ("Commutator", "commutator"),
        BinOp::Anticommutator => ("Anticommutator", "anticommutator"),
    }
}

fn unop_trait(op: UnOp) -> (&'static str, &'static str) {
    match op {
        UnOp::Reverse => ("Reverse", "reverse"),
        UnOp::Involute => ("Involute", "involute"),
        UnOp::Conjugate => ("Conjugate", "conjugate"),
        UnOp::Dual => ("Dual", "dual"),
        UnOp::Undual => ("Undual", "undual"),
    }
}

/// Emit an integration test that checks every generated kernel against the dense oracle in
/// `gax/tests/common`. `module` is the path of the generated module (e.g. `gax::pga2d`).
#[allow(clippy::too_many_lines)]
pub fn emit_tests(spec: &AlgebraSpec, stats: &Stats, module: &str, spec_path: &str) -> String {
    let mut t = String::new();
    let _ = write!(
        t,
        "// @generated by gax-gen from the `{name}` declaration. Do not edit by hand.\n\
         //! Every generated kernel of `{name}`, checked against the dense oracle.\n\
         #![cfg_attr(rustfmt, rustfmt::skip)]\n\
         #![cfg(feature = \"{name}\")]\n\
         #![allow(clippy::pedantic, clippy::too_many_lines)]\n\n\
         mod common;\n\
         use common::{{assert_close, random, Oracle, Rng}};\n\
         use gax::{{Extensor, Unit}};\n\
         use gax_gen::table::{{BinOp, UnOp}};\n\
         use {module}::*;\n\n\
         const SPEC: &str = include_str!(\"{spec_path}\");\n\n\
         fn bin<A, B, R>(o: &Oracle, rng: &mut Rng, op: BinOp, f: impl Fn(A, B) -> R)\n\
         where\n    A: Extensor<Slots = (), Coef = f64>,\n    B: Extensor<Slots = (), Coef = f64>,\n    R: Extensor<Slots = (), Coef = f64>,\n{{\n\
         \x20   for _ in 0..8 {{\n\
         \x20       let (a, b): (A, B) = (random(rng), random(rng));\n\
         \x20       let want = o.binop(op, &o.dense(&a), &o.dense(&b));\n\
         \x20       assert_close(&o.dense(&f(a, b)), &want, &format!(\"{{op:?}} {{}} {{}}\", std::any::type_name::<A>(), std::any::type_name::<B>()));\n\
         \x20   }}\n}}\n\n\
         fn un<A, R>(o: &Oracle, rng: &mut Rng, op: UnOp, f: impl Fn(A) -> R)\n\
         where\n    A: Extensor<Slots = (), Coef = f64>,\n    R: Extensor<Slots = (), Coef = f64>,\n{{\n\
         \x20   for _ in 0..8 {{\n\
         \x20       let a: A = random(rng);\n\
         \x20       assert_close(&o.dense(&f(a)), &o.unop(op, &o.dense(&a)), &format!(\"{{op:?}} {{}}\", std::any::type_name::<A>()));\n\
         \x20   }}\n}}\n\n\
         /// `v x ~v` against the oracle; with `unit`, `v` is first normalized to `v ~v = 1`.\n\
         fn sandwich<V, X, R>(o: &Oracle, rng: &mut Rng, unit: bool, f: impl Fn(V, X) -> R, prepared: impl Fn(V, X) -> R)\n\
         where\n    V: Extensor<Slots = (), Coef = f64>,\n    X: Extensor<Slots = (), Coef = f64>,\n    R: Extensor<Slots = (), Coef = f64>,\n{{\n\
         \x20   let mut tested = 0;\n\
         \x20   for _ in 0..16 {{\n\
         \x20       let (mut v, x): (V, X) = (random(rng), random(rng));\n\
         \x20       if unit {{\n\
         \x20           let Some(u) = o.random_unit(rng) else {{ continue }};\n\
         \x20           v = u;\n\
         \x20       }}\n\
         \x20       tested += 1;\n\
         \x20       let (dv, dx) = (o.dense(&v), o.dense(&x));\n\
         \x20       let want = o.binop(BinOp::Gp, &o.binop(BinOp::Gp, &dv, &dx), &o.unop(UnOp::Reverse, &dv));\n\
         \x20       assert_close(&o.dense(&f(v, x)), &want, &format!(\"sandwich {{}} {{}}\", std::any::type_name::<V>(), std::any::type_name::<X>()));\n\
         \x20       assert_close(&o.dense(&prepared(v, x)), &want, &format!(\"prepared {{}} {{}}\", std::any::type_name::<V>(), std::any::type_name::<X>()));\n\
         \x20   }}\n\
         \x20   assert!(tested > 0, \"no sample of {{}} could be normalized\", std::any::type_name::<V>());\n}}\n\n",
        name = spec.name
    );
    for op in BinOp::ALL {
        let (_, m) = trait_of(op);
        let _ = write!(
            t,
            "#[test]\nfn {m}() {{\n    let o = Oracle::from_spec(SPEC);\n    let mut rng = Rng::new({});\n",
            op as u64 + 1
        );
        for (bop, a, b) in &stats.binary {
            if *bop == op {
                let _ = writeln!(
                    t,
                    "    bin::<{a}<(), f64>, {b}<(), f64>, _>(&o, &mut rng, BinOp::{op:?}, |a, b| a.{m}(b));"
                );
            }
        }
        t.push_str("}\n\n");
    }
    for op in UnOp::ALL {
        let (_, m) = unop_trait(op);
        let _ = write!(
            t,
            "#[test]\nfn {m}() {{\n    let o = Oracle::from_spec(SPEC);\n    let mut rng = Rng::new({});\n",
            op as u64 + 20
        );
        for k in &spec.kinds {
            let _ = writeln!(
                t,
                "    un::<{}<(), f64>, _>(&o, &mut rng, UnOp::{op:?}, |a| a.{m}());",
                k.name
            );
        }
        t.push_str("}\n\n");
    }
    t.push_str("#[test]\nfn value_methods() {\n    let o = Oracle::from_spec(SPEC);\n    let mut rng = Rng::new(77);\n");
    for v in &stats.values {
        let k = &v.kind;
        if v.inverse.is_some() {
            let _ = writeln!(
                t,
                "    common::inverse::<{k}<(), f64>, _>(&o, &mut rng, |x| x.inverse());"
            );
        }
        if v.normalized {
            let _ = writeln!(
                t,
                "    common::normalized::<{k}<(), f64>>(&o, &mut rng, |x| x.normalized().into_inner());"
            );
        }
        if let Some(on) = &v.exp {
            let has_log = stats
                .values
                .iter()
                .any(|w| &w.kind == on && w.log.as_deref() == Some(k.as_str()));
            if has_log {
                let _ = writeln!(
                    t,
                    "    common::exp_log::<{k}<(), f64>, {on}<(), f64>>(&o, &mut rng, |b| b.exp(), |r| r.log());"
                );
            }
        }
        if v.sqrt && v.log.is_some() {
            let _ = writeln!(
                t,
                "    common::sqrt::<{k}<(), f64>>(&o, &mut rng, |r| r.sqrt().into_inner(), |r| r.into_inner() * r.into_inner());"
            );
        }
    }
    t.push_str("}\n\n");
    t.push_str("#[test]\nfn sandwiches() {\n    let o = Oracle::from_spec(SPEC);\n    let mut rng = Rng::new(99);\n");
    for (v, x, unit, _) in &stats.sandwich_costs {
        if *unit {
            let _ = writeln!(
                t,
                "    sandwich::<{v}<(), f64>, {x}<(), f64>, _>(&o, &mut rng, true, |v, x| Unit::new_unchecked(v) >> x, |v, x| Unit::new_unchecked(v).prepare::<{x}>() >> x);"
            );
        } else {
            let _ = writeln!(
                t,
                "    sandwich::<{v}<(), f64>, {x}<(), f64>, _>(&o, &mut rng, false, |v, x| v >> x, |v, x| v.prepare::<{x}>() >> x);"
            );
        }
    }
    t.push_str("}\n");
    t
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
