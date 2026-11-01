//! The kinds: a struct per kind with its methods, arithmetic and conversions, and the tables of
//! `gax_core::cast` and `gax_core::extensor` that relate them.

use super::{Emitter, IMPL_ALL, IMPL_O, blade_ident, kind_literal, zero_slots};
use crate::algebra::Algebra;
use crate::poly::Rational;
use crate::spec::{AlgebraSpec, KindSpec};
use crate::table::{BinOp, binop_support};
use std::collections::BTreeSet;
use std::fmt::Write as _;

impl Emitter<'_> {
    /// A kind: its struct and basic traits, the methods of its values and maps, its value
    /// constructors, the methods of its maps and forms, and its arithmetic.
    pub(super) fn kind(&mut self, k: &KindSpec) {
        self.kind_type(k);
        self.kind_methods(k);
        self.kind_values(k);
        self.kind_maps(k);
        self.kind_arithmetic(k);
    }

    /// The struct of a kind, with `Clone`, `Copy`, `PartialEq`, `Debug`, `Kind` and `Extensor`.
    fn kind_type(&mut self, k: &KindSpec) {
        let name = &k.name;
        let n = k.layout.len();
        let scalar = self.spec.scalar_kind().name.clone();
        let blades_list: Vec<String> = k.blades.iter().map(|b| format!("{b:?}")).collect();
        let doc = if k.doc.is_empty() {
            format!("A `{name}` of the `{}` algebra.", self.spec.name)
        } else {
            k.doc.clone()
        };
        let blade_doc = k.blades.join(", ");
        let default_intrinsic = k.intrinsic;
        let orient = crate::spec::orientation_name(k.intrinsic);
        let orient_doc = orient.to_lowercase();
        // Spelled out rather than `core::array::from_fn`: batch kernels need these inlined
        // (see `gax::batch`), and the core helpers are not always.
        let each = |f: &dyn Fn(usize) -> String| (0..n).map(f).collect::<Vec<_>>().join(", ");
        let arr_ff = each(&|i| format!("f({i})"));
        let arr_fm = each(&|i| format!("f(&a[{i}])"));
        let arr_fz = each(&|i| format!("f(&a[{i}], &b[{i}])"));
        let (ty, marker) = (format!("{name}<S, T, O>"), format!("{name}<(), f64, O>"));
        let s = format!(
            r#"#[doc = {doc:?}]
///
/// Blades, in coefficient order: `[{blade_doc}]`.
///
/// `S` lists the open slots: `{name}` (that is, `{name}<()>`) is a value, `{name}<(A,)>` a
/// linear map from `A`, `{name}<(A, B)>` a bilinear map. `T` is the coefficient type. `O` is
/// the orientation of the output, {orient_doc} by default (ADR-051).
#[repr(C)]
pub struct {name}<S: Slots = (), T: Coef = f64, O: gx::Orientation = gx::{orient}> {{
    /// Coefficients, output first: `c[i]` is the slot array of blade `i`.
    pub c: [S::Arr<T>; {n}],
    o: core::marker::PhantomData<O>,
}}

{IMPL_ALL} Clone for {ty} {{
    #[inline(always)]
    fn clone(&self) -> Self {{
        *self
    }}
}}

{IMPL_ALL} Copy for {ty} {{}}

{IMPL_ALL} PartialEq for {ty} {{
    fn eq(&self, other: &Self) -> bool {{
        self.c == other.c
    }}
}}

{IMPL_ALL} core::fmt::Debug for {ty} {{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {{
        let mut d = f.debug_struct(if O::SIGN < 0 {{ "intrinsic {name}" }} else {{ "{name}" }});
        for (b, c) in <{name} as Kind>::BLADES.iter().zip(self.c.iter()) {{
            d.field(b, c);
        }}
        d.finish()
    }}
}}

// SAFETY: `Arr<X>` is `[X; {n}]`, the layout `gx::FlatKind` promises.
unsafe {IMPL_O} gx::FlatKind for {marker} {{}}

{IMPL_O} Kind for {marker} {{
    const N: usize = {n};
    const NAME: &'static str = "{name}";
    const MODULE: &'static str = module_path!();
    const BLADES: &'static [&'static str] = &[{blades}];
    type Arr<X: gx::Elem> = [X; {n}];
    type Mv<S: Slots, T: Coef> = {name}<S, T, O>;
    type Scalar = {scalar};
    type Orientation = O;
    const DEFAULT_INTRINSIC: bool = {default_intrinsic};
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

{IMPL_ALL} Extensor for {ty} {{
    type Kind = {name}<(), f64, O>;
    type Slots = S;
    type Coef = T;
    #[inline(always)]
    fn from_coeffs(c: [S::Arr<T>; {n}]) -> Self {{
        {name} {{ o: core::marker::PhantomData, c }}
    }}
    #[inline(always)]
    fn coeffs(&self) -> &[S::Arr<T>; {n}] {{
        &self.c
    }}
}}

"#,
            blades = blades_list.join(", "),
        );
        self.w(&s);
    }

    /// The methods of every value and map of a kind (any slots, coefficients and orientation):
    /// slot filling, casts, least squares, the products the kind has and the unary operations as
    /// methods, the Hodge dual; and `Reorient`.
    #[allow(clippy::too_many_lines)]
    fn kind_methods(&mut self, k: &KindSpec) {
        let name = &k.name;
        let n = k.layout.len();
        let ty = format!("{name}<S, T, O>");
        let lit = |c: &str| kind_literal(name, c);
        let same = lit("self.c");
        let s = format!(
            r"{IMPL_ALL} {ty} {{
    /// Construct from output-first coefficients.
    #[inline(always)]
    pub const fn from_coeffs(c: [S::Arr<T>; {n}]) -> Self {{
        {name} {{ o: core::marker::PhantomData, c }}
    }}

    /// All coefficients zero.
    #[inline(always)]
    pub fn zero() -> Self {{
        {zero}
    }}

    /// Every coefficient mapped by `f`, kind and slots kept: to `Complex`, `Dual` or `f32`
    /// (see [`Extensor::map_coefs`]).
    #[inline]
    pub fn map_coefs<U: Coef>(&self, f: impl FnMut(T) -> U) -> {name}<S, U, O> {{
        gx::Extensor::map_coefs(self, f)
    }}

    /// Fill the first open slot with a value, or compose a map into it.
    #[inline(always)]
    pub fn of<X>(self, x: X) -> <Self as Of<X>>::Output
    where
        Self: Of<X>,
    {{
        Of::of(self, x)
    }}

    /// This value or map as a `K` of its own orientation (`K` names only the kind, so turbofish
    /// it: `x.cast::<Line>()`): the blades they share kept, `K`'s other blades zero (a
    /// projection, an embedding, or both; on maps and forms, of the output).
    #[inline(always)]
    pub fn cast<K>(self) -> <<K as gx::__private::Reorient<O>>::Out as gx::Kind>::Mv<S, T>
    where
        K: gx::__private::Reorient<O>,
        <K as gx::__private::Reorient<O>>::Out: gx::Kind,
        {name}<(), f64, O>: gx::Cast<<K as gx::__private::Reorient<O>>::Out>,
    {{
        gx::cast::cast::<Self, <K as gx::__private::Reorient<O>>::Out>(&self)
    }}

    /// The same coefficients with orientation `O2` (ADR-051): for coefficients from outside
    /// (a file, a GPU buffer, a raw `new`) whose orientation the type does not know.
    #[inline(always)]
    pub fn with_orientation<O2: gx::Orientation>(self) -> {name}<S, T, O2> {{
        {same}
    }}

    /// This value or map in an algebra holding this one's generators by name (numga's
    /// `from_subalgebra`, ADR-057): an extrinsic value keeps its blades, an intrinsic one its
    /// complement, so joins and pairings carry over. The result type names the target kind
    /// (`let p: pga3d::Point = q.include();`); see [`gx::FromSubalgebra`].
    #[inline(always)]
    pub fn include<B>(self) -> B
    where
        B: gx::FromSubalgebra<Self>,
    {{
        gx::FromSubalgebra::from_subalgebra(self)
    }}

    /// The value or map `a` of a subalgebra, as this kind (numga's `from_subalgebra`,
    /// ADR-057; see [`include`](Self::include) and [`gx::FromSubalgebra`]).
    #[inline(always)]
    pub fn from_subalgebra<A>(a: A) -> Self
    where
        Self: gx::FromSubalgebra<A>,
    {{
        gx::FromSubalgebra::from_subalgebra(a)
    }}

    /// The grade-`G` part, as the declared kind that holds it (on maps and forms, of the
    /// output).
    #[inline(always)]
    pub fn grade<const G: usize>(self) -> <<{name}<(), f64, O> as gx::GradePart<G>>::Out as gx::Kind>::Mv<S, T>
    where
        {name}<(), f64, O>: gx::GradePart<G>,
    {{
        gx::cast::grade::<Self, G>(&self)
    }}

    /// Least squares: the least-norm `x` of the first slot's kind minimizing
    /// `‖self.of(x) − rhs‖` (coefficient norms). For a one-slot map `rhs` may have slots, which
    /// `x` keeps; for more slots `rhs` has exactly the remaining ones.
    #[inline]
    pub fn lstsq<X>(self, rhs: X) -> <Self as gx::LeastSquares<X>>::Solution
    where
        Self: gx::LeastSquares<X>,
    {{
        gx::LeastSquares::lstsq(self, rhs)
    }}

    /// [`Self::lstsq`] with singular values below `rcond` times the largest treated as zero.
    #[inline]
    pub fn lstsq_with<X>(self, rhs: X, rcond: T) -> <Self as gx::LeastSquares<X>>::Solution
    where
        Self: gx::LeastSquares<X, Coef = T>,
    {{
        gx::LeastSquares::lstsq_with(self, rhs, rcond)
    }}

    /// Least squares for an unknown in the first two slots: the least-norm `x: A<(B,)>` with
    /// `self.of_dyad(x) ≈ rhs`, `rhs` over the remaining slots ([`gx::LeastSquaresDyad`]).
    #[inline]
    pub fn lstsq_dyad<X>(self, rhs: X) -> <Self as gx::OfDyad>::Dyad
    where
        Self: gx::LeastSquaresDyad<X>,
    {{
        gx::LeastSquaresDyad::lstsq_dyad(self, rhs)
    }}

    /// [`Self::lstsq_dyad`] with singular values below `rcond` times the largest treated as zero.
    #[inline]
    pub fn lstsq_dyad_with<X>(self, rhs: X, rcond: T) -> <Self as gx::OfDyad>::Dyad
    where
        Self: gx::LeastSquaresDyad<X, Coef = T>,
    {{
        gx::LeastSquaresDyad::lstsq_dyad_with(self, rhs, rcond)
    }}

    /// A two-slot value `x: A<(B,)>` bound into the first two slots: `Σᵢⱼ xᵢⱼ self(aᵢ, bⱼ)`.
    #[inline]
    pub fn of_dyad(self, x: <Self as gx::OfDyad>::Dyad) -> <Self as gx::OfDyad>::Image
    where
        Self: gx::OfDyad,
    {{
        gx::OfDyad::of_dyad(self, x)
    }}

    /// Move open slot `I` to the front, so that `.of(x)` fills it: `m.at::<1>().of(x)`.
    #[inline(always)]
    pub fn at<const I: usize>(self) -> {name}<<S as gx::__private::MoveToFront<I>>::Moved, T, O>
    where
        S: gx::__private::MoveToFront<I>,
    {{
        {moved}
    }}

    /// Contract the output with open slot `I` (which must be of kind `{name}`): a trace with no
    /// metric, leaving a scalar with the other slots (numga's `trace(slot)`).
    #[inline(always)]
    pub fn trace_at<const I: usize>(self) -> <{name}<<S as gx::__private::MoveToFront<I>>::Moved, T, O> as gx::TraceFirst>::Output
    where
        S: gx::__private::MoveToFront<I>,
        {name}<<S as gx::__private::MoveToFront<I>>::Moved, T, O>: gx::TraceFirst,
    {{
        gx::TraceFirst::trace_first(self.at::<I>())
    }}

    /// Pair the first two open slots, of complementary kinds, through the regressive product
    /// and sum (numga's `trace(1, 2)`): no metric, leaving `{name}` with the other slots
    /// (see [`gx::TraceInputs`]).
    #[inline(always)]
    pub fn trace_inputs(self) -> <Self as gx::TraceInputs>::Output
    where
        Self: gx::TraceInputs,
    {{
        gx::TraceInputs::trace_inputs(self)
    }}

    /// Pair the first two open slots, of one kind, through the metric `⟨a ~b⟩₀` and sum
    /// (numga's `contract`), leaving `{name}` with the other slots (see [`gx::Contract`]).
    #[inline(always)]
    pub fn contract(self) -> <Self as gx::Contract>::Output
    where
        Self: gx::Contract,
    {{
        gx::Contract::contract(self)
    }}

    /// Fill every open slot of `x`'s kind with the value `x`; the other slots stay open.
    #[inline(always)]
    pub fn fill<X>(self, x: X) -> {name}<<S as gx::__private::FillList<X::Kind>>::Out, T, O>
    where
        X: Extensor<Slots = (), Coef = T>,
        S: gx::__private::FillList<X::Kind>,
    {{
        let xc = x.coeffs();
        {filled}
    }}
",
            zero = lit(&format!("[{}; {n}]", zero_slots("S"))),
            moved = lit("self.c.map(|col| <S as gx::__private::MoveToFront<I>>::move_arr(&col))"),
            filled =
                lit("self.c.map(|col| <S as gx::__private::FillList<X::Kind>>::fill(&col, xc))"),
        );
        self.w(&s);

        // A product method where the kind has the product with some kind (the geometric
        // product always, with a coefficient).
        let alg = &self.spec.algebra;
        let has = |op: BinOp| {
            op == BinOp::Gp
                || k.operand
                    && self.spec.kinds.iter().any(|b| {
                        b.operand && !binop_support(alg, op, &k.layout, &b.layout).is_empty()
                    })
        };
        for op in [
            BinOp::Gp,
            BinOp::Wedge,
            BinOp::Vee,
            BinOp::Dot,
            BinOp::Lc,
            BinOp::Rc,
            BinOp::Scalar,
            BinOp::Commutator,
            BinOp::Anticommutator,
        ]
        .into_iter()
        .filter(|&op| has(op))
        {
            let (tr, m, _) = op.names();
            let _ = write!(self.out, "\n{}", binary_method(tr, m));
        }
        // numga's `dot`, the scalar product with the reverse: no tables of its own.
        if has(BinOp::Scalar) {
            self.w("\n    /// The metric on every grade, `⟨x ~b⟩₀`: numga's `dot`, the bilinear form of\n    /// `norm_squared` (`x.metric(x)` is `x ~x`'s scalar part), positive on every blade of a\n    /// Euclidean signature and zero between grades. Not [`dot`](Self::dot), the inner\n    /// product `|`, which is `⟨x b⟩₀` on vectors and negative on Euclidean bivectors. On blades\n    /// of one grade it is `x & b.hodge()` (docs/laws.md, law S).\n    #[inline(always)]\n    pub fn metric<B: Reverse>(self, b: B) -> <Self as ScalarProduct<<B as Reverse>::Output>>::Output\n    where\n        Self: ScalarProduct<<B as Reverse>::Output>,\n    {\n        ScalarProduct::scalar_product(self, Reverse::reverse(b))\n    }\n");
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
        // The metric (Hodge) dual and its inverse, as linear maps on the blades (no pseudoscalar
        // kind needed): wherever the image is not empty, and the inverse in non-degenerate
        // algebras, with a refusal that says why in degenerate ones.
        if let Some((on, exprs)) = hodge_map(self.spec, k, false) {
            let _ = write!(
                self.out,
                "\n    /// The Hodge dual `~x I`, with the metric (`I` the product of the basis vectors in\n    /// order): numga's `dual`. [`dual`](Self::dual) is the metric-free complement; the two\n    /// differ in sign on blades containing basis vectors of negative square (STA, R(4,1)), and\n    /// `hodge` vanishes on blades containing a null basis vector (`e0` of PGA), where `dual`\n    /// does not. Its inverse is [`hodge_inverse`](Self::hodge_inverse), in non-degenerate\n    /// algebras.\n    /// Like `dual`, it flips the orientation (ADR-051): `I` here is the intrinsic volume\n    /// form, the complement against the orientation of space.\n    #[inline]\n    pub fn hodge(self) -> {on}<S, T, gx::orientation::Flip<O>> {{\n        let a = self.c;\n        {}\n    }}\n",
                kind_literal(&on, &format!("[{}]", exprs.join(", ")))
            );
        }
        if self.spec.algebra.signature().2 > 0 {
            let _ = write!(
                self.out,
                "\n    /// Not callable: the Hodge dual of a degenerate algebra vanishes on blades containing a\n    /// null basis vector, so it has no inverse. This method only makes the compile error say\n    /// so.\n    #[doc(hidden)]\n    #[inline(always)]\n    pub fn hodge_inverse(self) -> Self\n    where\n        Self: gx::__private::HodgeNeedsMetric,\n    {{\n        gx::__private::HodgeNeedsMetric::hodge_inverse(self)\n    }}\n"
            );
        } else if let Some((on, exprs)) = hodge_map(self.spec, k, true) {
            let _ = write!(
                self.out,
                "\n    /// The inverse of the Hodge dual, `~(x I⁻¹)`: numga's `dual_inverse`, with\n    /// `x.hodge().hodge_inverse() == x` and `x.hodge_inverse().hodge() == x`. On a blade of\n    /// grade `g` in `n` dimensions it is `(−1)^(g(n−g)) / (I ~I)` times `x.hodge()`: the same in\n    /// Euclidean spaces of odd dimension, `-x.hodge()` on the vectors of the plane and on the\n    /// even grades of STA. Flips the orientation, as `hodge` does (ADR-051).\n    #[inline]\n    pub fn hodge_inverse(self) -> {on}<S, T, gx::orientation::Flip<O>> {{\n        let a = self.c;\n        {}\n    }}\n",
                kind_literal(&on, &format!("[{}]", exprs.join(", ")))
            );
        }
        self.w("}\n\n");

        // A value, map or kind of another orientation, the coefficients kept (ADR-051).
        let _ = write!(
            self.out,
            "impl<S: Slots, T: Coef, O: gx::Orientation, O2: gx::Orientation> gx::__private::Reorient<O2> for {name}<S, T, O> {{\n    type Out = {name}<S, T, O2>;\n    #[inline(always)]\n    fn reorient(self) -> {name}<S, T, O2> {{\n        {same}\n    }}\n}}\n\n"
        );
    }

    /// The value constructors (`new`, `slot`, per orientation), `gx::Slot`, a versor's
    /// `transform`, `transform_inv` and `prepare`, and the coefficient accessors of a kind.
    fn kind_values(&mut self, k: &KindSpec) {
        let name = &k.name;
        let lit = |c: &str| kind_literal(name, c);
        let params: Vec<String> = k
            .blades
            .iter()
            .map(|b| format!("{}: T", blade_ident(b)))
            .collect();
        let args: Vec<String> = k.blades.iter().map(|b| blade_ident(b)).collect();
        let identity = lit(
            "core::array::from_fn(|i| core::array::from_fn(|j| if i == j { T::one() } else { T::zero() }))",
        );
        // The constructors, once per orientation (ADR-052): a path through an alias
        // (`Line`, `intrinsic::Line`, `Forque`) fixes the orientation, so `new` and `slot` need
        // no annotation; generic code takes slots from `gx::Slot`.
        for o in ["Extrinsic", "Intrinsic"] {
            let _ = write!(
                self.out,
                "impl<T: Coef> {name}<(), T, gx::{o}> {{\n    /// A value from its coefficients, in blade order.\n    #[inline(always)]\n    #[allow(clippy::too_many_arguments)]\n    pub const fn new({params}) -> Self {{\n        {value}\n    }}\n\n    /// The identity map on `{name}`: a `{name}` with one open `{name}` slot, both of this\n    /// orientation.\n    #[inline(always)]\n    pub fn slot() -> {name}<({name}<(), f64, gx::{o}>,), T, gx::{o}> {{\n        {identity}\n    }}\n}}\n\n",
                params = params.join(", "),
                value = lit(&format!("[{}]", args.join(", "))),
            );
        }
        let _ = write!(
            self.out,
            "impl<T: Coef, O: gx::Orientation> gx::Slot for {name}<(), T, O> {{\n    type Map = {name}<({name}<(), f64, O>,), T, O>;\n    #[inline(always)]\n    fn slot() -> Self::Map {{\n        {identity}\n    }}\n}}\n\nimpl<T: Coef, O: gx::Orientation> {name}<(), T, O> {{\n"
        );
        // A versor's value transforms and prepares, where its sandwiches were generated.
        if self.transforms.contains(name) {
            let _ = write!(
                self.out,
                "{}\n{}\n    /// This versor's action on kind `X`, prepared for applying it to many objects (see\n    /// [`gx::Prepared`]).\n    #[inline(always)]\n    pub fn prepare<X>(self) -> <Self as gx::Prepare<X>>::Output\n    where\n        Self: gx::Prepare<X>,\n    {{\n        gx::Prepare::prepare(self)\n    }}\n\n",
                binary_method("Transform", "transform"),
                binary_method("TransformInv", "transform_inv")
            );
        }
        let accessors: Vec<String> = k
            .blades
            .iter()
            .enumerate()
            .map(|(i, b)| {
                format!(
                    "    /// Coefficient of `{b}`.\n    #[inline(always)]\n    pub fn {}(&self) -> T {{\n        self.c[{i}]\n    }}\n",
                    blade_ident(b)
                )
            })
            .collect();
        self.w(&accessors.join("\n"));
        self.w("}\n\n");
    }

    /// The methods of a kind's maps and forms: building them from images, filling both slots,
    /// a base kind's outermorphisms, inverses, determinants, decompositions and solves, and for
    /// the scalar kind those of forms and pairings (see `gax_core::extensor`).
    #[allow(clippy::too_many_lines)]
    fn kind_maps(&mut self, k: &KindSpec) {
        let name = &k.name;
        let lit = |c: &str| kind_literal(name, c);
        // Building a map from the images of its input's basis blades.
        let _ = write!(
            self.out,
            r"impl<A: Kind, T: Coef, O: gx::Orientation> {name}<(A,), T, O> {{
    /// The linear map that sends each basis blade of `A`, in `A`'s layout order, to the given
    /// `{name}`. `A` may be a kind of another algebra (a projection from PGA3D points to PGA2D
    /// points is a `pga2d::Point<(pga3d::Point,)>`). The coefficients of a map are stored
    /// output first: `from_coeffs` takes rows, `c[o][i]` the coefficient `o` of the image of
    /// the input blade `i`.
    #[inline]
    pub fn from_images(images: A::Arr<{name}<(), T, O>>) -> Self {{
        let images = images.as_ref();
        {from_images}
    }}
}}

",
            from_images = lit("core::array::from_fn(|o| A::arr_from_fn(|i| images[i].c[o]))"),
        );

        // Both slots of a two-slot extensor filled with maps.
        let _ = write!(
            self.out,
            r"impl<A: Kind, B: Kind, T: Coef, O: gx::Orientation> {name}<(A, B), T, O> {{
    /// Fill the first slot with the map `p: A <- (X,)` and the second with `q: B <- (Y,)`:
    /// `{name} <- (X, Y)` (see [`gx::OfBoth`]).
    #[inline]
    pub fn of_both<P, Q>(self, p: P, q: Q) -> <Self as gx::OfBoth<P, Q>>::Output
    where
        Self: gx::OfBoth<P, Q>,
    {{
        gx::OfBoth::of_both(self, p, q)
    }}
}}

"
        );

        // The transposes through the regressive product (a kind with a declared complement)
        // and through the scalar product (a kind with a reciprocal frame).
        let complement = complement_pairs(self.spec, k).is_some();
        let reciprocal = reciprocal_frame(self.spec, k).is_some();
        if complement || reciprocal {
            let _ = writeln!(
                self.out,
                "impl<A: Kind, T: Coef, O: gx::Orientation> {name}<(A,), T, O> {{"
            );
            if complement {
                let _ = write!(
                    self.out,
                    r"    /// The adjugate, the transpose through the regressive product: the map between
    /// the complement kinds with `(m.adjugate().of(c) & a)₀ == (c & m.of(a))₀`. No metric,
    /// and defined for singular maps (see [`gx::Adjugate`]).
    #[inline]
    pub fn adjugate(self) -> <Self as gx::Adjugate>::Output
    where
        Self: gx::Adjugate,
    {{
        gx::Adjugate::adjugate(self)
    }}
"
                );
            }
            if reciprocal {
                let _ = write!(
                    self.out,
                    r"    /// The adjoint, the transpose through the scalar product: the map `A <- {name}`
    /// with `m.adjoint().of(b).scalar_product(a) == b.scalar_product(m.of(a))`; for kinds on
    /// which the scalar product is non-degenerate (see [`gx::Adjoint`]).
    #[inline]
    pub fn adjoint(self) -> <Self as gx::Adjoint>::Output
    where
        Self: gx::Adjoint,
    {{
        gx::Adjoint::adjoint(self)
    }}
"
                );
            }
            self.w("}\n\n");
        }

        // The extensions of a map on vectors (or antivectors) to the other grades.
        if self.outer_bases.contains(name) {
            let _ = write!(
                self.out,
                r"impl<A: Kind, T: Coef, O: gx::Orientation> {name}<(A,), T, O> {{
    /// The outermorphism of this map on vectors (or antivectors) to the kind `B` that its
    /// inputs' products span: `m.outermorphism::<Line>().of(a ^ b) == m.of(a) ^ m.of(b)` (with
    /// `&` for antivectors). Each side multiplies by its own product, so a correlation from
    /// points to planes carries joins to meets, `c.outermorphism::<Line>().of(p & q) ==
    /// c.of(p) ^ c.of(q)` (see [`gx::Outermorphism`]).
    #[inline(always)]
    pub fn outermorphism<B>(self) -> <Self as gx::Outermorphism<B>>::Output
    where
        Self: gx::Outermorphism<B>,
    {{
        gx::Outermorphism::outermorphism(self)
    }}
}}

"
            );
        }

        // Methods of maps and forms (see `gax_core::extensor`).
        let _ = write!(
            self.out,
            r"impl<A: Kind, T: Real, O: gx::Orientation> {name}<(A,), T, O> {{
    /// The inverse map, `A <- {name}`.
    #[inline]
    pub fn inverse(self) -> <A as Kind>::Mv<({name}<(), f64, O>,), T>
    where
        Self: SquareMap<Coef = T, Kind = {name}<(), f64, O>, Input = A, Inverse = <A as Kind>::Mv<({name}<(), f64, O>,), T>>,
    {{
        SquareMap::inverse(self)
    }}

    /// [`inverse`](Self::inverse), or `None` where the map is singular or the inverse is not
    /// finite (see [`SquareMap::try_inverse`]).
    #[inline]
    pub fn try_inverse(self) -> Option<<A as Kind>::Mv<({name}<(), f64, O>,), T>>
    where
        Self: SquareMap<Coef = T, Kind = {name}<(), f64, O>, Input = A, Inverse = <A as Kind>::Mv<({name}<(), f64, O>,), T>>,
    {{
        SquareMap::try_inverse(self)
    }}

    /// The determinant of the map's coefficient matrix.
    #[inline]
    pub fn det(self) -> T
    where
        Self: SquareMap<Coef = T, Kind = {name}<(), f64, O>, Input = A>,
    {{
        SquareMap::det(self)
    }}

    /// The singular values, descending, of a map of any shape (see [`gx::SingularValues`]).
    #[inline]
    pub fn svdvals(self) -> <A as Kind>::Arr<T>
    where
        Self: gx::SingularValues<Coef = T, Input = A>,
    {{
        gx::SingularValues::svdvals(self)
    }}

    /// The singular value decomposition of a map of any shape: values, right and left singular
    /// vectors, `self.of(inputs[i]) == values[i] * outputs[i]` (see [`gx::Svd`]).
    #[inline]
    pub fn svd(self) -> gx::Svd<A, {name}<(), f64, O>, T>
    where
        Self: gx::SingularValues<Coef = T, Input = A, Kind = {name}<(), f64, O>>,
    {{
        gx::SingularValues::svd(self)
    }}

    /// The eigenvalues (ascending) and eigenvectors of the coefficient matrix taken as
    /// symmetric (see [`SquareMap::eigh`]).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn eigh(self) -> (<{name}<(), f64, O> as Kind>::Arr<T>, <{name}<(), f64, O> as Kind>::Arr<<A as Kind>::Mv<(), T>>)
    where
        Self: SquareMap<Coef = T, Kind = {name}<(), f64, O>, Input = A>,
    {{
        SquareMap::eigh(self)
    }}

    /// For a correlation, a map of a kind to its complement: the eigenproblem
    /// `self(x) = λ metric(x)` against a definite correlation, through the regressive product
    /// (see [`gx::Correlation::eigh_in`]).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn eigh_in(self, metric: Self) -> (<A as Kind>::Arr<T>, <A as Kind>::Arr<<A as Kind>::Mv<(), T>>)
    where
        Self: gx::Correlation<Coef = T, Input = A>,
    {{
        gx::Correlation::eigh_in(self, metric)
    }}

    /// The eigenvalues, complex in general, sorted (see [`SquareMap::eigvals`]).
    #[inline]
    pub fn eigvals(self) -> <{name}<(), f64, O> as Kind>::Arr<gx::Complex<T>>
    where
        Self: SquareMap<Coef = T, Kind = {name}<(), f64, O>, Input = A>,
    {{
        SquareMap::eigvals(self)
    }}

    /// The eigenvalues and an eigenvector of each, a complex `A` (see [`SquareMap::eig`]).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn eig(self) -> (<{name}<(), f64, O> as Kind>::Arr<gx::Complex<T>>, <{name}<(), f64, O> as Kind>::Arr<<A as Kind>::Mv<(), gx::Complex<T>>>)
    where
        Self: SquareMap<Coef = T, Kind = {name}<(), f64, O>, Input = A>,
    {{
        SquareMap::eig(self)
    }}

    /// Solve `self.of(x) == rhs` for `x`; a right-hand side with slots keeps them (a two-slot
    /// pairing's `solve` solves against its second slot instead).
    #[inline]
    pub fn solve<X>(self, rhs: X) -> <A as Kind>::Mv<X::Slots, T>
    where
        Self: SquareMap<Coef = T, Kind = {name}<(), f64, O>, Input = A>,
        X: Extensor<Kind = {name}<(), f64, O>, Coef = T>,
    {{
        SquareMap::solve(self, rhs)
    }}

    /// [`solve`](Self::solve), or `None` where the map is singular or the solution is not
    /// finite (see [`SquareMap::try_solve`]).
    #[inline]
    pub fn try_solve<X>(self, rhs: X) -> Option<<A as Kind>::Mv<X::Slots, T>>
    where
        Self: SquareMap<Coef = T, Kind = {name}<(), f64, O>, Input = A>,
        X: Extensor<Kind = {name}<(), f64, O>, Coef = T>,
    {{
        SquareMap::try_solve(self, rhs)
    }}

    /// The Moore–Penrose pseudo-inverse, `A <- {name}`, of a map of any shape: it sends `b` to
    /// the least-norm least-squares solution of `self.of(x) ≈ b`.
    #[inline]
    pub fn pinv(self) -> A::Mv<({name}<(), f64, O>,), T>
    where
        Self: gx::PseudoInverse<Coef = T, Output = A::Mv<({name}<(), f64, O>,), T>>,
    {{
        gx::PseudoInverse::pinv(self)
    }}

    /// [`Self::pinv`] with singular values below `rcond` times the largest treated as zero.
    #[inline]
    pub fn pinv_with(self, rcond: T) -> A::Mv<({name}<(), f64, O>,), T>
    where
        Self: gx::PseudoInverse<Coef = T, Output = A::Mv<({name}<(), f64, O>,), T>>,
    {{
        gx::PseudoInverse::pinv_with(self, rcond)
    }}

    /// The trace of a map from `{name}` to itself.
    #[inline]
    pub fn trace(self) -> T
    where
        Self: Endomorphism<Coef = T, Kind = {name}<(), f64, O>, Input = {name}<(), f64, O>>,
    {{
        Endomorphism::trace(self)
    }}
}}

impl<A: Kind, B: Kind, T: Coef, O: gx::Orientation> {name}<(A, B), T, O> {{
    /// Exchange the two slots.
    #[inline]
    pub fn swap(self) -> {name}<(B, A), T, O> {{
        self.at::<1>()
    }}
}}

"
        );

        // The methods of forms and pairings, `{name} <- (A, B)`: for the scalar kind.
        if name == &self.spec.scalar_kind().name {
            let _ = write!(
                self.out,
                r"impl<A: Kind, B: Kind, T: Coef, O: gx::Orientation> {name}<(A, B), T, O> {{
    /// This form `{name} <- (A, B)` as the map `A <- (B,)` with the same coefficients, for the
    /// methods of maps (`svd`, `pinv`, `svdvals`, `det`).
    #[inline]
    pub fn as_map(self) -> <A as Kind>::Mv<(B,), T> {{
        gx::FormAsMap::as_map(self)
    }}
}}

impl<A: Kind, B: Kind, T: Real, O: gx::Orientation> {name}<(A, B), T, O> {{
    /// Solve the pairing `self(x, ·) == rhs(l, ·)` for `x`; leading slots of `rhs` become slots
    /// of `x` (see [`Pairing::solve`]; a map's `solve` inverts the map itself).
    #[inline]
    pub fn solve<R>(self, rhs: R) -> <A as Kind>::Mv<<R::Slots as SplitLast>::Init, T>
    where
        Self: Pairing<Coef = T, Kind = {name}<(), f64, O>, First = A, Second = B>,
        R: Extensor<Kind = {name}<(), f64, O>, Coef = T>,
        R::Slots: SplitLast<Last = B>,
    {{
        Pairing::solve(self, rhs)
    }}

    /// [`solve`](Self::solve), or `None` where the pairing is singular or the solution is not
    /// finite (see [`Pairing::try_solve`]).
    #[inline]
    pub fn try_solve<R>(self, rhs: R) -> Option<<A as Kind>::Mv<<R::Slots as SplitLast>::Init, T>>
    where
        Self: Pairing<Coef = T, Kind = {name}<(), f64, O>, First = A, Second = B>,
        R: Extensor<Kind = {name}<(), f64, O>, Coef = T>,
        R::Slots: SplitLast<Last = B>,
    {{
        Pairing::try_solve(self, rhs)
    }}
}}

impl<A: Kind, T: Real, O: gx::Orientation> {name}<(A, A), T, O> {{
    /// Generalized symmetric eigenproblem against a positive definite metric form. Returns the
    /// eigenvalues (ascending) and the eigenvectors, as values of the slot kind. A form names
    /// its metric because its eigenvalues against the identity depend on the basis; where the
    /// coefficient basis is orthonormal, pass [`identity`](Self::identity) (see [`Form::eigh`]).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn eigh(self, metric: Self) -> (<A as Kind>::Arr<T>, <A as Kind>::Arr<<A as Kind>::Mv<(), T>>)
    where
        Self: Form<Coef = T, Kind = {name}<(), f64, O>, Slot = A>,
    {{
        Form::eigh(self, metric)
    }}

    /// [`eigh`](Self::eigh) against a positive semidefinite metric: infinite
    /// eigenvalues (`+∞`, last) for the directions it does not measure (see
    /// [`Form::eigh_semidefinite`]).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn eigh_semidefinite(self, metric: Self) -> (<A as Kind>::Arr<T>, <A as Kind>::Arr<<A as Kind>::Mv<(), T>>)
    where
        Self: Form<Coef = T, Kind = {name}<(), f64, O>, Slot = A>,
    {{
        Form::eigh_semidefinite(self, metric)
    }}

    /// Least squares of a symmetric form in a metric: of the `x` with `self(x, ·) == rhs`, the
    /// one smallest in `metric(x, x)`, its modes of curvature at most `rcond` times the largest
    /// dropped (numga's `form.lstsq(rhs, metric, rcond=…)`; see [`Form::lstsq_in`]).
    #[inline]
    pub fn lstsq_in(self, rhs: <{name}<(), f64, O> as Kind>::Mv<(A,), T>, metric: Self, rcond: T) -> <A as Kind>::Mv<(), T>
    where
        Self: Form<Coef = T, Kind = {name}<(), f64, O>, Slot = A>,
    {{
        Form::lstsq_in(self, rhs, metric, rcond)
    }}

    /// The identity form in the coefficient basis, `x, y ↦ Σᵢ xᵢ yᵢ`: the Euclidean metric
    /// where the layout is orthonormal, so `form.eigh(Scalar::identity())` is the coefficient
    /// matrix's own eigenproblem (see [`Form::identity`]).
    #[inline]
    #[must_use]
    pub fn identity() -> Self
    where
        Self: Form<Coef = T, Kind = {name}<(), f64, O>, Slot = A>,
    {{
        Form::identity()
    }}
}}

"
            );
        }
    }

    /// A kind's arithmetic: sums, negation, scaling and division by scalars, and the operators
    /// for the products.
    #[allow(clippy::too_many_lines)]
    fn kind_arithmetic(&mut self, k: &KindSpec) {
        let name = &k.name;
        let lit = |c: &str| kind_literal(name, c);
        let ty = format!("{name}<S, T, O>");
        // Arithmetic.
        let arr = |op: &str| {
            lit(&format!(
                "core::array::from_fn(|i| (SlotArr::<S, T>(self.c[i]) {op} SlotArr::<S, T>(rhs.c[i])).0)"
            ))
        };
        let _ = write!(
            self.out,
            r"{IMPL_ALL} core::ops::Add for {ty} {{
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {{
        {add}
    }}
}}

{IMPL_ALL} core::ops::Sub for {ty} {{
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {{
        {sub}
    }}
}}

{IMPL_ALL} core::iter::Sum for {ty} {{
    /// The sum of values or of maps; zero for none.
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {{
        iter.fold(Self::zero(), |a, b| a + b)
    }}
}}

{IMPL_ALL} core::ops::AddAssign for {ty} {{
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {{
        *self = *self + rhs;
    }}
}}

{IMPL_ALL} core::ops::SubAssign for {ty} {{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {{
        *self = *self - rhs;
    }}
}}

{IMPL_ALL} core::ops::Neg for {ty} {{
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {{
        {neg}
    }}
}}

{IMPL_ALL} Gp<T> for {ty} {{
    type Output = Self;
    #[inline(always)]
    fn gp(self, rhs: T) -> Self {{
        {scaled}
    }}
}}

impl<S: Slots, T: Real, O: gx::Orientation> DivBy<T> for {name}<S, T, O> {{
    type Output = Self;
    #[inline(always)]
    fn div_by(self, rhs: T) -> Self {{
        let r = rhs.recip();
        {divided}
    }}
}}

impl<S: Slots, T: Coef, O: gx::Orientation, R> core::ops::Div<R> for {name}<S, T, O>
where
    Self: DivBy<R>,
{{
    type Output = <Self as DivBy<R>>::Output;
    #[inline(always)]
    fn div(self, rhs: R) -> Self::Output {{
        DivBy::div_by(self, rhs)
    }}
}}

impl<S: Slots, O: gx::Orientation> core::ops::Mul<{name}<S, f32, O>> for f32 {{
    type Output = {name}<S, f32, O>;
    #[inline(always)]
    fn mul(self, rhs: {name}<S, f32, O>) -> {name}<S, f32, O> {{
        rhs.gp(self)
    }}
}}

impl<S: Slots, O: gx::Orientation> core::ops::Mul<{name}<S, f64, O>> for f64 {{
    type Output = {name}<S, f64, O>;
    #[inline(always)]
    fn mul(self, rhs: {name}<S, f64, O>) -> {name}<S, f64, O> {{
        rhs.gp(self)
    }}
}}

",
            add = arr("+"),
            sub = arr("-"),
            neg = lit("self.c.map(|x| (-SlotArr::<S, T>(x)).0)"),
            scaled = lit("self.c.map(|x| SlotArr::<S, T>(x).scale(rhs).0)"),
            divided = lit("self.c.map(|x| SlotArr::<S, T>(x).scale(r).0)"),
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
                "impl<S: Slots, T: Coef, O: gx::Orientation, R> core::ops::{std_tr}<R> for {name}<S, T, O>\nwhere\n    Self: {tr}<R>,\n{{\n    type Output = <Self as {tr}<R>>::Output;\n    #[inline(always)]\n    fn {std_m}(self, rhs: R) -> Self::Output {{\n        {tr}::{m}(self, rhs)\n    }}\n}}\n\n"
            );
        }
    }

    /// `From` for every kind whose blades are among another's: the same multivector as the
    /// larger kind (a rotor as a motor), with the orientations of the target's blades.
    pub(super) fn embeddings(&mut self) {
        let mut body = String::new();
        for a in &self.spec.kinds {
            for b in &self.spec.kinds {
                if a.name == b.name {
                    continue;
                }
                if a.layout.embed_in(&b.layout).is_none() {
                    continue;
                }
                let exprs: Vec<String> = b
                    .layout
                    .positions_in(&a.layout)
                    .iter()
                    .map(|p| match p {
                        None => zero_slots("<S as Slots>"),
                        Some((j, false)) => format!("x.c[{j}]"),
                        Some((j, true)) => format!("(-SlotArr::<S, T>(x.c[{j}])).0"),
                    })
                    .collect();
                let (an, bn) = (&a.name, &b.name);
                let _ = write!(
                    body,
                    "{IMPL_ALL} From<{an}<S, T, O>> for {bn}<S, T, O> {{\n    /// The same multivector as a [`{bn}`], of the same orientation.\n    #[inline(always)]\n    fn from(x: {an}<S, T, O>) -> Self {{\n        {}\n    }}\n}}\n\nimpl<T: Coef, O: gx::Orientation> gx::Widen<{bn}<(), T, O>> for {an}<(), T, O> {{}}\n\n",
                    kind_literal(bn, &format!("[{}]", exprs.join(", ")))
                );
                // A part and a kind holding it add into the kind (`Point + Direction`), in
                // either order, as do their differences (ADR-054, amended).
                if !a.result && b.result {
                    for (tr, m, op, what) in [
                        ("Add", "add", "+", "sum"),
                        ("Sub", "sub", "-", "difference"),
                    ] {
                        let _ = write!(
                            body,
                            "{IMPL_ALL} core::ops::{tr}<{an}<S, T, O>> for {bn}<S, T, O> {{\n    type Output = {bn}<S, T, O>;\n    /// The part `rhs` as a [`{bn}`], then the {what} of two `{bn}`s.\n    #[inline(always)]\n    fn {m}(self, rhs: {an}<S, T, O>) -> {bn}<S, T, O> {{\n        self {op} {bn}::from(rhs)\n    }}\n}}\n\n{IMPL_ALL} core::ops::{tr}<{bn}<S, T, O>> for {an}<S, T, O> {{\n    type Output = {bn}<S, T, O>;\n    /// `self` as a [`{bn}`], then the {what} of two `{bn}`s.\n    #[inline(always)]\n    fn {m}(self, rhs: {bn}<S, T, O>) -> {bn}<S, T, O> {{\n        {bn}::from(self) {op} rhs\n    }}\n}}\n\n{IMPL_ALL} core::ops::{tr}Assign<{an}<S, T, O>> for {bn}<S, T, O> {{\n    #[inline(always)]\n    fn {m}_assign(&mut self, rhs: {an}<S, T, O>) {{\n        *self = *self {op} {bn}::from(rhs);\n    }}\n}}\n\n"
                        );
                    }
                }
            }
        }
        self.w(&body);
    }
}

/// The method `m` forwarding to the binary trait `tr` (`gp`, `transform`), indented for an impl
/// block.
fn binary_method(tr: &str, m: &str) -> String {
    format!(
        "    /// See [`{tr}`].\n    #[inline(always)]\n    pub fn {m}<R>(self, rhs: R) -> <Self as {tr}<R>>::Output\n    where\n        Self: {tr}<R>,\n    {{\n        {tr}::{m}(self, rhs)\n    }}\n"
    )
}

/// The blades of `a` listed by `keep` (in `a`'s order) as `(i, j, flip)` entries of `b`'s
/// layout, or `None` where a kept blade is not one of `b`'s.
fn shared_blades(
    a: &KindSpec,
    b: &KindSpec,
    keep: impl Fn(u32) -> bool,
) -> Vec<(usize, usize, bool)> {
    a.layout
        .positions_in(&b.layout)
        .into_iter()
        .enumerate()
        .filter(|&(i, _)| keep(a.layout.blades[i].0))
        .filter_map(|(i, p)| p.map(|(j, flip)| (i, j, flip)))
        .collect()
}

fn render_shared(shared: &[(usize, usize, bool)]) -> String {
    shared
        .iter()
        .map(|(i, j, f)| format!("({i}, {j}, {f})"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The reciprocal frame of a kind (ADR-047): row `i` holds `e^i` in the kind's layout, the
/// `i`-th row of the inverse of the Gram matrix `G_ij = e_i ∗ e_j` of its oriented blades.
/// `None` when the Gram matrix is singular (a null direction, such as PGA's `e0`).
pub fn reciprocal_frame(spec: &AlgebraSpec, kind: &KindSpec) -> Option<Vec<Vec<Rational>>> {
    let blades = &kind.layout.blades;
    let n = blades.len();
    let scalar = |a: u32, b: u32| scalar_product_of_blades(spec, a, b);
    // [G | I], reduced to [I | G⁻¹] by Gauss–Jordan elimination in exact rationals.
    let mut rows: Vec<Vec<Rational>> = blades
        .iter()
        .enumerate()
        .map(|(i, &(mi, si))| {
            let gram = blades
                .iter()
                .map(|&(mj, sj)| Rational::int(i128::from(si * sj * scalar(mi, mj))));
            let unit = (0..n).map(|j| {
                if i == j {
                    Rational::ONE
                } else {
                    Rational::ZERO
                }
            });
            gram.chain(unit).collect()
        })
        .collect();
    for col in 0..n {
        let pivot = (col..n).find(|&r| !rows[r][col].is_zero())?;
        rows.swap(col, pivot);
        let scale = rows[col][col].recip();
        for x in &mut rows[col] {
            *x = *x * scale;
        }
        let pivot_row = rows[col].clone();
        for (r, row) in rows.iter_mut().enumerate() {
            let factor = row[col];
            if r != col && !factor.is_zero() {
                for (x, &p) in row.iter_mut().zip(&pivot_row) {
                    *x = *x - p * factor;
                }
            }
        }
    }
    Some(rows.into_iter().map(|row| row[n..].to_vec()).collect())
}

/// `impl Reciprocal` for a kind with a reciprocal frame: its rows as sparse
/// `(index, numerator, denominator)` terms.
fn reciprocal_impl(spec: &AlgebraSpec, kind: &KindSpec, out: &mut String) {
    let Some(frame) = reciprocal_frame(spec, kind) else {
        return;
    };
    let small = |x: i128| i64::try_from(x).expect("a reciprocal frame entry fits in i64");
    let rows: Vec<String> = frame
        .iter()
        .map(|row| {
            let terms: Vec<String> = row
                .iter()
                .enumerate()
                .filter(|(_, x)| !x.is_zero())
                .map(|(k, x)| format!("({k}, {}, {})", small(x.num()), small(x.den())))
                .collect();
            format!("&[{}]", terms.join(", "))
        })
        .collect();
    let gram: Vec<String> = kind
        .layout
        .blades
        .iter()
        .map(|&(mi, si)| {
            let terms: Vec<String> = kind
                .layout
                .blades
                .iter()
                .enumerate()
                .map(|(k, &(mk, sk))| (k, si * sk * scalar_product_of_blades(spec, mi, mk)))
                .filter(|&(_, g)| g != 0)
                .map(|(k, g)| format!("({k}, {g})"))
                .collect();
            format!("&[{}]", terms.join(", "))
        })
        .collect();
    let _ = writeln!(
        out,
        "#[diagnostic::do_not_recommend]\n{IMPL_O} gx::Reciprocal for {}<(), f64, O> {{\n    const FRAME: &'static [&'static [(usize, i64, i64)]] = &[{}];\n    const GRAM: &'static [&'static [(usize, i64)]] = &[{}];\n}}\n",
        kind.name,
        rows.join(", "),
        gram.join(", ")
    );
}

/// `⟨e_a e_b⟩₀` of two canonical basis blades.
fn scalar_product_of_blades(spec: &AlgebraSpec, a: u32, b: u32) -> i64 {
    spec.algebra
        .blade_product(a, b)
        .iter()
        .filter(|(m, _)| *m == 0)
        .map(|(_, v)| *v)
        .sum()
}

/// The declared kind holding exactly the complements of `kind`'s blades, and per blade of
/// `kind` the position of its complement there and whether their pairing
/// `⟨complement ∨ blade⟩₀` (complement on the left, layout orientations included) is `−1`.
pub fn complement_pairs<'a>(
    spec: &'a AlgebraSpec,
    kind: &KindSpec,
) -> Option<(&'a KindSpec, Vec<(usize, bool)>)> {
    let alg = &spec.algebra;
    let full = alg.pseudoscalar();
    let support: BTreeSet<u32> = kind.layout.blades.iter().map(|(m, _)| full ^ m).collect();
    let c = spec.kinds.iter().find(|c| c.layout.support() == support)?;
    let pairs = kind
        .layout
        .blades
        .iter()
        .map(|&(m, s)| {
            let (j, t) = c.layout.position(full ^ m).expect("the complement's blade");
            let v = crate::table::blade_binop(alg, BinOp::Vee, full ^ m, m)
                .get(&0)
                .copied()
                .unwrap_or(0);
            let sign = s * t * v;
            assert!(
                sign == 1 || sign == -1,
                "a blade pairs with its complement to ±1"
            );
            (j, sign < 0)
        })
        .collect();
    Some((c, pairs))
}

/// `KindEq` for every pair of kinds: whether they are the same type.
pub(super) fn kind_eqs(spec: &AlgebraSpec, out: &mut String) {
    for a in &spec.kinds {
        for b in &spec.kinds {
            // The same kind is the same blades and the same orientation (ADR-051).
            let eq = if a.name == b.name {
                "gx::orientation::Same<OA, OB>"
            } else {
                "gx::__private::False"
            };
            let _ = writeln!(
                out,
                "impl<OA: gx::Orientation, OB: gx::Orientation> gx::__private::KindEq<{}<(), f64, OB>> for {}<(), f64, OA> {{\n    type Out = {eq};\n}}\n",
                b.name, a.name
            );
        }
    }
}

/// `Cast` for every pair of kinds sharing a blade, `SubKind` where one's blades all lie in
/// the other's, `GradePart` for every grade of every kind, and `Complement` for every kind
/// whose complement is declared (`gax_core::cast`); and `Reciprocal` for every kind with a
/// reciprocal frame.
pub(super) fn kind_tables(spec: &AlgebraSpec, out: &mut String) {
    for a in &spec.kinds {
        reciprocal_impl(spec, a, out);
        if let Some((c, pairs)) = complement_pairs(spec, a) {
            let pairs: Vec<String> = pairs.iter().map(|(j, f)| format!("({j}, {f})")).collect();
            let _ = writeln!(
                out,
                "{IMPL_O} gx::Complement for {}<(), f64, O> {{\n    type Out = {}<(), f64, gx::orientation::Flip<O>>;\n    const PAIRS: &'static [(usize, bool)] = &[{}];\n}}\n",
                a.name,
                c.name,
                pairs.join(", ")
            );
        }
        for b in &spec.kinds {
            let shared = shared_blades(a, b, |_| true);
            if shared.is_empty() {
                continue;
            }
            let (an, bn) = (&a.name, &b.name);
            let _ = writeln!(
                out,
                "{IMPL_O} gx::Cast<{bn}<(), f64, O>> for {an}<(), f64, O> {{\n    const SHARED: &'static [(usize, usize, bool)] = &[{}];\n}}\n",
                render_shared(&shared)
            );
            if shared.len() == a.layout.len() {
                let _ = writeln!(
                    out,
                    "{IMPL_O} gx::SubKind<{bn}<(), f64, O>> for {an}<(), f64, O> {{}}\n"
                );
            }
        }
        let grades: BTreeSet<u32> = a
            .layout
            .blades
            .iter()
            .map(|(m, _)| m.count_ones())
            .collect();
        for g in grades {
            let support: BTreeSet<u32> = a
                .layout
                .blades
                .iter()
                .map(|(m, _)| *m)
                .filter(|m| m.count_ones() == g)
                .collect();
            let o = spec
                .result_kind(&support, &[a])
                .expect("the kind itself holds it");
            let shared = shared_blades(a, o, |m| m.count_ones() == g);
            let _ = writeln!(
                out,
                "{IMPL_O} gx::GradePart<{g}> for {}<(), f64, O> {{\n    type Out = {}<(), f64, O>;\n    const SHARED: &'static [(usize, usize, bool)] = &[{}];\n}}\n",
                a.name,
                o.name,
                render_shared(&shared)
            );
        }
    }
}

/// The value methods of a kind as trait impls (`gax_core::ops::{Exp, Inverse, Norm, Normalize,
/// Sqrt}`), forwarding to the inherent methods, so generic code can call them.
pub(super) fn value_traits(
    spec: &AlgebraSpec,
    name: &str,
    meta: &crate::emit_values::ValueMethods,
    out: &mut String,
) {
    if let Some(on) = &meta.exp {
        // The exponential's kind in the extrinsic orientation, named where it is not the
        // kind's default (a kind declared intrinsic, ADR-051).
        let oe = if spec.kinds.iter().any(|k| &k.name == on && k.intrinsic) {
            ", gx::Extrinsic"
        } else {
            ""
        };
        let _ = writeln!(
            out,
            "impl<T: gx::Real> gx::Exp for {name}<(), T, gx::Extrinsic> {{\n    type Output = gx::Unit<{on}<(), T{oe}>>;\n    #[inline(always)]\n    fn exp(self) -> Self::Output {{\n        Self::exp(self)\n    }}\n}}\n"
        );
    }
    if let Some(on) = &meta.inverse {
        let _ = writeln!(
            out,
            "impl<T: gx::Real, O: gx::Orientation> gx::Inverse for {name}<(), T, O> {{\n    type Output = {on}<(), T, O>;\n    #[inline(always)]\n    fn inverse(self) -> Self::Output {{\n        Self::inverse(self)\n    }}\n    #[inline(always)]\n    fn try_inverse(self) -> Option<Self::Output> {{\n        Self::try_inverse(self)\n    }}\n}}\n"
        );
    }
    if meta.norm {
        let _ = writeln!(
            out,
            "impl<T: gx::Real, O: gx::Orientation> gx::Norm for {name}<(), T, O> {{\n    #[inline(always)]\n    fn norm_squared(self) -> T {{\n        Self::norm_squared(self)\n    }}\n    #[inline(always)]\n    fn norm(self) -> T {{\n        Self::norm(self)\n    }}\n}}\n"
        );
    }
    if meta.normalized {
        let _ = writeln!(
            out,
            "impl<T: gx::Real, O: gx::Orientation> gx::Normalize for {name}<(), T, O> {{\n    #[inline(always)]\n    fn unit(self) -> gx::Unit<Self> {{\n        Self::unit(self)\n    }}\n}}\n"
        );
    }
    if meta.sqrt_scalar {
        let _ = writeln!(
            out,
            "impl<T: gx::Real> gx::Sqrt for {name}<(), T, gx::Extrinsic> {{\n    type Output = Self;\n    #[inline(always)]\n    fn sqrt(self) -> Self::Output {{\n        Self::sqrt(self)\n    }}\n}}\n"
        );
    }
    if meta.sqrt {
        let _ = writeln!(
            out,
            "impl<T: gx::Real> gx::Sqrt for {name}<(), T, gx::Extrinsic> {{\n    type Output = gx::Unit<Self>;\n    #[inline(always)]\n    fn sqrt(self) -> Self::Output {{\n        Self::sqrt(self)\n    }}\n}}\n"
        );
    }
}

/// The Hodge dual `x ↦ ~x I` of kind `k` (or its inverse `x ↦ ~(x I⁻¹)`) as a linear map on
/// blades: its output kind, the smallest holding the image, and each output coefficient's
/// expression in the input's `a`. `None` where the image is empty (a kind of null blades), or,
/// for the inverse, in a degenerate algebra.
pub(crate) fn hodge_map(
    spec: &AlgebraSpec,
    k: &KindSpec,
    inverse: bool,
) -> Option<(String, Vec<String>)> {
    let alg = &spec.algebra;
    let full = alg.pseudoscalar();
    let n = alg.dim() as u32;
    // `I I`, a scalar (`± det` of the metric): the inverse divides by it.
    let ii: i64 = alg
        .blade_product(full, full)
        .iter()
        .filter(|(m, _)| *m == 0)
        .map(|(_, c)| *c)
        .sum();
    if inverse && ii == 0 {
        return None;
    }
    // Per input blade, the image's blades and integer coefficients, the input's stored sign
    // applied (the inverse then divides by `I I`).
    let mut image: Vec<Vec<(u32, i64)>> = Vec::new();
    for &(m, sign) in &k.layout.blades {
        let terms: Vec<(u32, i64)> = alg
            .blade_product(m, full)
            .iter()
            .map(|&(o, c)| {
                // `~b I` reverses `b` (grade g); `~(b I)` reverses the result (grade n − g).
                let g = if inverse {
                    n - m.count_ones()
                } else {
                    m.count_ones()
                };
                (o, c * Algebra::reverse_sign(g) * sign)
            })
            .filter(|&(_, c)| c != 0)
            .collect();
        image.push(terms);
    }
    let support: BTreeSet<u32> = image.iter().flatten().map(|&(o, _)| o).collect();
    if support.is_empty() {
        return None;
    }
    let out = spec.result_kind(&support, &[k])?;
    let den = if inverse { ii } else { 1 };
    let exprs = out
        .layout
        .blades
        .iter()
        .map(|&(o, osign)| {
            let terms: Vec<(usize, Rational)> = image
                .iter()
                .enumerate()
                .flat_map(|(i, t)| {
                    t.iter().filter(move |(m, _)| *m == o).map(move |&(_, c)| {
                        (i, Rational::new(i128::from(c * osign), i128::from(den)))
                    })
                })
                .collect();
            match terms.as_slice() {
                [] => zero_slots("S"),
                [(i, c)] if *c == Rational::ONE => format!("a[{i}]"),
                [(i, c)] if *c == Rational::int(-1) => format!("(-SlotArr::<S, T>(a[{i}])).0"),
                _ => {
                    let sum = terms
                        .iter()
                        .map(|(i, c)| {
                            let x = format!("SlotArr::<S, T>(a[{i}])");
                            if *c == Rational::ONE {
                                x
                            } else if *c == Rational::int(-1) {
                                format!("(-{x})")
                            } else if c.is_integer() {
                                format!("{x}.scale(T::from_i64({}))", c.num())
                            } else {
                                format!("{x}.scale(T::from_ratio({}, {}))", c.num(), c.den())
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" + ");
                    format!("({sum}).0")
                }
            }
        })
        .collect();
    Some((out.name.clone(), exprs))
}

#[cfg(test)]
mod tests {
    use super::reciprocal_frame;
    use crate::poly::Rational;
    use crate::spec::AlgebraSpec;

    fn frames(src: &str) -> Vec<(String, Option<Vec<Vec<Rational>>>)> {
        let spec = AlgebraSpec::parse(src).expect("a valid spec");
        spec.kinds
            .iter()
            .map(|k| (k.name.clone(), reciprocal_frame(&spec, k)))
            .collect()
    }

    /// In PGA every kind with `e0` in a blade has a singular Gram matrix; the rotors and the
    /// parts through the origin (`Spin`, `Tilt`, `OriginPlane`) do not.
    #[test]
    fn pga_frames_exist_only_without_the_null_direction() {
        for (name, frame) in frames(include_str!("../../../gax/specs/pga3d.gax")) {
            assert_eq!(
                frame.is_some(),
                matches!(
                    name.as_str(),
                    "Scalar" | "Rotor" | "Spin" | "Tilt" | "OriginPlane"
                ),
                "{name}"
            );
        }
    }

    /// CGA's null pair `eo · e∞ = −1`: the reciprocal of `eo` is `−e∞`, and of `e∞` is `−eo`.
    #[test]
    fn cga_null_pair_is_its_own_reciprocal_negated() {
        let all = frames(include_str!("../../../gax/specs/cga3d.gax"));
        let (_, vector) = all
            .iter()
            .find(|(n, _)| n == "Vector")
            .expect("a vector kind");
        let vector = vector.as_ref().expect("a frame");
        let (o, i) = (3, 4); // [e1, e2, e3, eo, ei]
        let minus = -Rational::ONE;
        assert_eq!(vector[o][i], minus);
        assert_eq!(vector[i][o], minus);
        assert!(vector[o][o].is_zero() && vector[i][i].is_zero());
    }
}
