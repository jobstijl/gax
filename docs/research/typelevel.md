# Type-level slot lists in stable Rust: findings

Toolchain: `rustc 1.98.1 (48a229cea 2026-09-01)` stable, plus `rustc 1.100.0-nightly (2026-09-24)` for next-solver and feature checks.
All experiments are in `scratchpad/typelevel/` (`eNN_*.rs`, `q3/`, `q4/`). Build any of them with
`rustc --edition 2024 --crate-type=lib FILE.rs`. The ones with `fn main` build with `rustc --edition 2024 FILE.rs`.

---

## TL;DR: recommended design (verified on stable 1.98, also passes under `-Znext-solver`)

Split concatenation into a **helper trait with a GAT**. Then make the right-identity law a **supertrait
associated-type equality** on `Slots`:

```rust
pub trait HasCat { type Cat<R: Slots>: Slots; }             // GAT: concat with any R
pub trait Slots: HasCat<Cat<Nil> = Self> + Copy + 'static {  // law: S ++ [] == S
    type Arr<X: Copy>: Copy;                                 // nested-array storage
    fn map<X: Copy, Y: Copy>(a: &Self::Arr<X>, f: &mut impl FnMut(&X) -> Y) -> Self::Arr<Y>;
    fn outer<R: Slots, A: Copy, B: Copy, C: Copy>(
        a: &Self::Arr<A>, b: &R::Arr<B>, f: &mut impl FnMut(&A, &B) -> C,
    ) -> <Self::Cat<R> as Slots>::Arr<C>;
}
pub type Cat<A, B> = <A as HasCat>::Cat<B>;

impl HasCat for Nil { type Cat<R: Slots> = R; }                                     // [] ++ R = R (by impl)
impl<H: SlotType, T: Slots> HasCat for Cons<H, T> { type Cat<R: Slots> = Cons<H, T::Cat<R>>; }
impl Slots for Nil { type Arr<X: Copy> = X; /* ... */ }
impl<H: SlotType, T: Slots> Slots for Cons<H, T> { type Arr<X: Copy> = H::Arr<T::Arr<X>>; /* ... */ }

impl<S1: Slots, S2: Slots> BitAnd<Point<S2>> for Plane<S1> { type Output = Line<Cat<S1, S2>>; /*...*/ }
impl<S1: Slots, S2: Slots> BitXor<Plane<S2>> for Line<S1>  { type Output = Point<Cat<S1, S2>>; /*...*/ }
impl<S1: Slots, S2: Slots> Mul<Point<S2>>   for Point<S1>  { type Output = Motor<Cat<S1, S2>>; /*...*/ }

// USER CODE: only `S: Slots` needed. All of these compile:
pub fn shadow<S: Slots>(p: Point<S>) -> Point<S> { (LIGHT & p) ^ GROUND }
pub fn prod<S1: Slots, S2: Slots>(a: Point<S1>, b: Point<S2>) -> Motor<Cat<S1, S2>> { a * b }
pub fn twice<S: Slots>(p: Point<S>) -> Point<S> { shadow(shadow(p)) }
```

Why it works:
* `LIGHT & p` gives `Line<<Nil as HasCat>::Cat<S>>`. The concrete `Nil` impl normalizes that to `Line<S>`.
* `... ^ GROUND` gives `Point<<S as HasCat>::Cat<Nil>>`. `S: Slots` elaborates the supertrait `HasCat<Cat<Nil> = S>`,
  so this normalizes to `Point<S>` through the param-env.
* `a * b` for two generic lists needs no extra bound, because `HasCat::Cat<R>` is a GAT that exists for every `R: Slots`.
  Contrast this with a `Concat<R>` trait parameter, which needs `S1: Concat<S2>` in every user signature.
* The `Cons` impl of `Slots` must prove `Cons<H, T::Cat<Nil>> == Cons<H, T>`. It can, from `T: Slots`, so the
  law is *checked* inductively by the compiler when the impls are written. It is not merely assumed.
* There is **no cycle**. A supertrait on `HasCat` that mentions `Slots` would cycle (see below). Here `HasCat`
  mentions `Slots` only in a GAT *item bound*, and item bounds are not part of supertrait elaboration.

Full working version with nested-array storage and a generic outer product: `e11_storage.rs` (runs, prints results).
Tuple surface syntax `Point<(PointT, PointT)>` also works: `e14_tuples.rs`.

Remaining limitation: **associativity** `Cat<Cat<A,B>,C> == Cat<A,Cat<B,C>>` is not known for generic A, B, C. It only
matters when a user *annotates* a generic signature with a different bracketing than the expression produces.
The workaround is a zero-cost value-level witness method (`e13_reassoc.rs`). The law cannot be put in the trait,
because it needs a `for<C: Slots>` type binder. That is the nightly `non_lifetime_binders` feature, which is
incomplete and broke even the basic design when tried.

---

## 1. variadics_please and frunk

### variadics_please (Bevy): `ref/variadics_please`, v2.0.0, `MIT OR Apache-2.0`, MSRV 1.85, deps `quote` + `unsynn`
* Proc macros `all_tuples!(mac, START, END, P, p)`, `all_tuples_enumerated!` (passes the index too) and
  `all_tuples_with_size!` (passes the arity `$N`). Each one calls your `macro_rules!` once per arity with
  `(P0, p0), (P1, p1), ...`. It is only a code generator, so no type-level machinery is involved.
* `#[doc(fake_variadic)]` collapses the impls in rustdoc (needs nightly `rustdoc_internals`, so it is gated to docs.rs).
* Relevance: the library could use it to generate the `Slots` impls for tuples `()`, `(A,)`, ..., `(A..L)`.
  `e14_tuples.rs` does exactly that with a hand-written `macro_rules!`, which is enough for small N.
  Tuples need an extra `Prepend<H>` GAT and a witness method (details in section 2.6). The longest tuple has no
  `Prepend`, so it needs a sentinel type (`Overflow`) plus a post-monomorphization compile error (`e15`).

### frunk: `ref/frunk`, v0.5.0, `MIT` (Copyright 2016 Lloyd Chan)
* `HCons<H, T>` / `HNil`, the `HList![A, B]` type macro and the `hlist![a, b]` value macro. Concatenation is
  `impl<RHS: HList> Add<RHS> for HNil { type Output = RHS }` and `impl<H,T: Add<RHS>,RHS> Add<RHS> for HCons<H,T> { type Output = HCons<H, T::Output> }`
  (`core/src/hlist.rs:822-840`). That is exactly the baseline `Concat`, and it has the same generic-code problem.
  frunk has no `Cat<HNil> = Self` law.
* **Inferred-index trick** (`Selector<S, I>`, `Plucker<T, I>`, `Sculptor`, `hlist.rs:858-980`, indices in
  `core/src/indices.rs`: `Here`, `There<T>`):
  ```rust
  impl<T, Tail> Selector<T, Here> for HCons<T, Tail> { ... }                     // found at head
  impl<H, Tail, T, I> Selector<T, There<I>> for HCons<H, Tail> where Tail: Selector<T, I> { ... }
  ```
  Without `I`, the two impls `Sel<T> for Cons<T, Tail>` and `Sel<T> for Cons<H, Tail>` overlap when `H = T`.
  Verified, `e18_selector.rs --cfg overlap`:
  `error[E0119]: conflicting implementations of trait 'Sel<_>' for type 'Cons<_, _>'`.
  The extra type parameter `I` makes the impl *headers* differ (`Here` vs `There<_>`), so coherence accepts them.
  The caller leaves `I` as an inference variable (`fn pick<L: Selector<T, I>, T, I>`). Trait selection finds the
  unique impl chain, which fixes `I` (e.g. `There<There<Here>>`). If the target type occurs twice, inference
  is ambiguous: `error[E0283]: type annotations needed` (`--cfg ambig`).
  Limitation for us: this only resolves on **concrete** lists. In generic `S: Slots` code you would have to add
  `S: Selector<Twist, I>` bounds with an extra generic `I`. It is useful for concrete "contract the Twist slot"
  helpers, but not for the concat problem.

---

## 2. The key problem (right identity of Concat) in detail

### 2.1 Baseline: `trait Concat<R: Slots>: Slots { type Out: Slots; }` with HList recursion (`e01_baseline.rs`)
`LIGHT & p` type-checks (the `Nil` impl is generic in R). `... ^ GROUND` fails:
```
error[E0369]: no implementation for `Line<S> ^ Plane<Nil>`
help: consider further restricting type parameter `S` with trait `Concat`
29 | pub fn f<S: Slots + Concat<Nil>>(p: Point<S>) -> Point<S> {
```
Adding `S: Concat<Nil>` gives (`e02_bound_only.rs`):
```
error[E0308]: mismatched types
   expected `Point<S>`, found `Point<<S as Concat<Nil>>::Out>`
help: consider further restricting this bound
23 | pub fn f<S: Slots + Concat<Nil, Out = S>>(p: Point<S>) -> Point<S> {
```
With `S: Slots + Concat<Nil, Out = S>` it compiles, but then every user function carries that bound.
The next solver (`-Znext-solver=globally`, nightly) gives the same two errors, because the law is not derivable
by any solver.

### 2.2 (a) Supertrait equality `trait Slots: Concat<Nil, Out = Self>`
**If `Concat` itself has `Slots` as a supertrait or a param bound, this cycles** (`e03a_super.rs`):
```
error[E0391]: cycle detected when computing the super predicates of `Slots`
5 | pub trait Slots: Concat<Nil, Out = Self> {}
note: ...which requires computing the super predicates of `Concat`...
6 | pub trait Concat<R: Slots>: Slots { type Out: Slots; }
  = note: ...which again requires computing the super predicates of `Slots`, completing the cycle
error[E0391]: cycle detected when computing the implied predicates of `Slots`
```
**If `Concat` has no bounds mentioning `Slots` (`trait Concat<R> { type Out; }`), it compiles** (`e03b_super.rs`, `e04_super_full.rs`):
```rust
pub trait Concat<R> { type Out; }
impl<R> Concat<R> for Nil { type Out = R; }
impl<H, T: Concat<R>, R> Concat<R> for Cons<H, T> { type Out = Cons<H, T::Out>; }
pub trait Slots: Concat<Nil, Out = Self> {}
impl Slots for Nil {}
impl<H, T: Slots> Slots for Cons<H, T> {}   // compiler proves Cons<H, <T as Concat<Nil>>::Out> == Cons<H,T> from T: Slots
pub fn f<S: Slots>(p: Point<S>) -> Point<S> { (LIGHT & p) ^ GROUND }   // OK
```
Concrete normalization also works: `fn concrete(p: Point<Cons<Twist,Nil>>) -> Point<Cons<Twist,Nil>> { (LIGHT & p) ^ GROUND }`
and `p * p : Motor<Cons<Twist, Cons<Twist, Nil>>>` both compile.
**But** a product of two generic lists needs the extra bound (`e04b.rs`):
```
error[E0277]: the trait bound `S1: Concat<S2>` is not satisfied
37 | pub fn g<S1: Slots, S2: Slots>(a: Point<S1>, b: Point<S2>) -> Motor<<S1 as Concat<S2>>::Out> {
```
You cannot add `S1: Concat<S2>` for all `S2` as a supertrait, because that needs a type-level `for<S2>`. Hence 2.4.

### 2.3 (b) Extra "value on the right" impls / four-case split
Adding `impl<S> BitXor<Plane<Nil>> for Line<S>` next to the general impl (`e07_overlap_value_right.rs`):
```
error[E0119]: conflicting implementations of trait `BitXor<Plane<Nil>>` for type `Line<_>`
12 | impl<S1: Concat<S2>, S2> BitXor<Plane<S2>> for Line<S1> {
16 | impl<S> BitXor<Plane<Nil>> for Line<S> {
```
Four disjoint impls (Nil×Nil, Nil×Cons, Cons×Nil, Cons×Cons) are coherent, but generic user code fails
(`e08_four_cases.rs`), because no single impl covers a generic `S`:
```
error[E0369]: no implementation for `Line<S> ^ Plane<Nil>`
note: an implementation of `BitXor<Plane<Nil>>` might be missing for `Line<S>`
help: consider introducing a `where` clause ...
```
General rule: **for generic `S: Slots` to "know" anything, the knowledge must come from (i) a blanket impl over
all `S: Slots`, (ii) a supertrait or where-clause of `Slots`, or (iii) an associated item of `Slots`.** Case-split impls
(Nil vs Cons) are only usable once `S` is concrete. The same applies to marker dispatch (section 3).

### 2.4 Self-referential where-clause on the GAT-bearing trait: rejected (`e05_selfwhere.rs`)
```rust
pub trait Slots where Self: Slots<Cat<Nil> = Self> { type Cat<R: Slots>: Slots; }
```
```
error[E0391]: cycle detected when computing the super predicates of `Slots`
5 | pub trait Slots where Self: Slots<Cat<Nil> = Self> { type Cat<R: Slots>: Slots; }
  = note: ...which immediately requires computing the super predicates of `Slots` again
error[E0391]: cycle detected when computing the implied predicates of `Slots`
```
`where Self: X` on a trait is the same thing as a supertrait, and a trait cannot be its own supertrait.

### 2.5 THE WORKING DESIGN: GAT on a helper trait plus supertrait equality (`e06_gat_helper.rs`, `e11_storage.rs`)
```rust
pub trait HasCat { type Cat<R: Slots>: Slots; }
pub trait Slots: HasCat<Cat<Nil> = Self> {}
impl HasCat for Nil { type Cat<R: Slots> = R; }
impl Slots for Nil {}
impl<H, T: Slots> HasCat for Cons<H, T> { type Cat<R: Slots> = Cons<H, T::Cat<R>>; }
impl<H, T: Slots> Slots for Cons<H, T> {}
pub type Cat<A, B> = <A as HasCat>::Cat<B>;
```
Compiles on stable 1.98, and on nightly with `-Znext-solver=globally` (no errors). Verified user functions, all with
only `Slots` bounds:
```rust
pub fn f<S: Slots>(p: Point<S>) -> Point<S> { (LIGHT & p) ^ GROUND }
pub fn g<S1: Slots, S2: Slots>(a: Point<S1>, b: Point<S2>) -> Motor<Cat<S1, S2>> { a * b }
pub fn h<S: Slots>(p: Point<S>) -> Point<S> { f(f(p)) }
pub fn k<S1: Slots, S2: Slots>(..) { let p: Point<Cat<S1,S2>> = ..; f(p) * b; (LIGHT & p) ^ GROUND; }  // nested projections OK
pub fn concrete(p: Point<Cons<Twist,Nil>>) -> Point<Cons<Twist,Nil>> { f(p) }         // normalizes
pub fn concrete2(p: Point<T1>) -> Motor<Cons<Twist, Cons<Twist, Nil>>> { p * p }       // normalizes
pub fn concrete4(p: Point<Nil>) -> Motor<T1> { p * Point::<T1>(..) }
```
The two designs compared:

| formulation | stable 1.98 | `f<S: Slots>` (`(L & p) ^ G`) | `g<S1,S2: Slots>` (`a * b`) |
|---|---|---|---|
| `Concat<R: Slots>: Slots` + nothing | compiles | **E0369 / E0308** | needs `S1: Concat<S2>` |
| `Slots: Concat<Nil,Out=Self>`, `Concat<R: Slots>: Slots` | **E0391 cycle** | n/a | n/a |
| `Slots: Concat<Nil,Out=Self>`, `Concat<R>` unbounded (design A) | compiles | **OK** | needs `S1: Concat<S2>` (E0277 otherwise) |
| `trait Slots where Self: Slots<Cat<Nil>=Self> { type Cat<R> }` | **E0391 cycle** | n/a | n/a |
| four disjoint impls (Nil/Cons × Nil/Cons) | compiles | **E0369** | **E0369** |
| `HasCat { type Cat<R: Slots>: Slots }` + `Slots: HasCat<Cat<Nil>=Self>` (design B) | compiles | **OK** | **OK** |

### 2.6 Storage via GATs, and the generic outer product (`e11_storage.rs`, runs)
* `SlotType` (PointT, PlaneT, TwistT...) owns `type Arr<X: Copy>: Copy = [X; 4]` with a **literal** length.
  `Cons<H,T>::Arr<X> = H::Arr<T::Arr<X>>` nests. This avoids `[X; H::DIM]`, which is illegal in generic
  context on stable (section 3).
* `Slots::outer::<R>` returns `<Self::Cat<R> as Slots>::Arr<C>`. For `Nil` that normalizes to `R::Arr<C>` (the body is `R::map`).
  For `Cons<H,T>` it normalizes to `H::Arr<<T::Cat<R> as Slots>::Arr<C>>`, and the body is `H::map(a, |ta| T::outer::<R,..>(ta, b, f))`.
  This works *because* `Cons::Cat<R>` is structurally `Cons<H, T::Cat<R>>` even when `T::Cat<R>` is opaque.
* Multivector structs: `pub struct Point<S: Slots> { pub c: S::Arr<[f32; 4]> }`, with manual or derived
  `Clone`/`Copy`. `#[derive(Clone, Copy)]` works in generic code because `Slots: Copy` and `Arr<X>: Copy` are implied
  (`e16_derive.rs`). `#[derive(Debug, PartialEq)]` adds `S: Debug` and `S::Arr<[f32;4]>: Debug` bounds, which
  generic code cannot prove:
  ```
  error[E0277]: `S` doesn't implement `Debug`
  error[E0277]: `<S as Slots>::Arr<[f32; 4]>` doesn't implement `Debug`
  ```
  A GAT cannot say "`Arr<X>: Debug` if `X: Debug`". Either bound the GAT with a fixed coefficient trait
  (`type Arr<X: Coeff>: Coeff`) or write `Debug` by hand via a flat view.
* Program output, including concrete normalization of storage (`let arr: [[f32;4];4] = shadow(m).c;` and
  `let arr2: [[[f32;8];4];4] = prod(m,m).c;`):
  ```
  value: [1.0, 2.0, 3.0, 5.0]
  map:   [[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 2.0], [2.0, 2.0, 2.0, 3.0], [3.0, 3.0, 3.0, 4.0]]
  bilinear[3][2] = [6.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0]
  SIZE = 24
  ```
* Flat view (`e17_flat.rs`, runs: `len=24 sum=276`): nested arrays are contiguous, so
  `slice::from_raw_parts(a as *const S::Arr<X> as *const X, S::SIZE)` guarded by
  `const { assert!(size_of::<S::Arr<X>>() == S::SIZE * size_of::<X>()) }`. Inline `const {}` blocks that reference generics are
  evaluated **post-monomorphization**, which is allowed on stable. `S::SIZE` works as a value, not as an array length.
* Closures are passed as `&mut impl FnMut` so the recursion reuses one closure type per level.

### 2.7 Associativity (`e12_assoc.rs`, `e13_reassoc.rs`)
```rust
pub fn left <A,B,C: Slots>(a,b,c) -> Mv<Cat<Cat<A, B>, C>> { (a * b) * c }  // OK
pub fn right<A,B,C: Slots>(a,b,c) -> Mv<Cat<A, Cat<B, C>>> { (a * b) * c }  // error:
```
```
error[E0308]: mismatched types
   = note: expected struct `Mv<<A as HasCat>::Cat<<B as HasCat>::Cat<C>>>`
              found struct `Mv<<<A as HasCat>::Cat<B> as HasCat>::Cat<C>>`
```
A value-level witness on `Slots` compiles and is an identity copy at runtime:
```rust
fn reassoc<B: Slots, C: Slots, X: Copy>(a: &<Cat<Cat<Self, B>, C> as Slots>::Arr<X>) -> <Cat<Self, Cat<B, C>> as Slots>::Arr<X>;
// Nil:          { *a }                                    (both sides normalize to <Cat<B,C>>::Arr<X>)
// Cons<H,T>:    { H::map(a, |t| T::reassoc::<B, C, X>(t)) }
pub fn right<A,B,C: Slots>(a,b,c) -> Mv<Cat<A, Cat<B, C>>> { reassoc::<A,B,C>((a * b) * c) }   // OK
```
Encoding associativity in the trait needs `for<C: Slots>` type binders (nightly `non_lifetime_binders`). I tried it
on nightly 1.100: even the base design then fails with E0277/E0271 (`q4/nlb.rs`). Not viable.

### 2.8 Tuples instead of Cons (`e14_tuples.rs`, runs; `e15_tuples_postmono.rs`)
Works with `()` as the empty list and `Point<(PointT,)>`, `Motor<(PointT, PointT)>` as user-visible types:
```rust
pub trait Slots: HasCat<Cat<()> = Self> + Copy + 'static {
    type Prepend<H: SlotType>: Slots;
    type Arr<X: Copy>: Copy;
    fn prepend_arr<H: SlotType, X: Copy>(a: H::Arr<Self::Arr<X>>) -> <Self::Prepend<H> as Slots>::Arr<X>; // witness
    ...
}
impl<T0: SlotType, T1: SlotType> HasCat for (T0, T1) { type Cat<R: Slots> = <<(T1,) as HasCat>::Cat<R> as Slots>::Prepend<T0>; }
impl<T0: SlotType, T1: SlotType> Slots for (T0, T1) { type Prepend<H0: SlotType> = (H0, T0, T1); type Arr<X: Copy> = T0::Arr<<(T1,) as Slots>::Arr<X>>; ... }
```
The difference from Cons: `Cat<R>` = `Prepend<H>` applied to the *opaque* `Tail::Cat<R>`, so its `Arr` does not normalize
structurally. The fix is the `prepend_arr` witness method (an identity for every concrete tuple). The longest tuple
cannot `Prepend`, so it maps to a sentinel `Overflow: Slots`. Its witness contains `const { panic!(..) }`, which gives a
**compile-time error only when instantiated**:
```
error[E0080]: evaluation panicked: slot list too long (inline const, generic)
   evaluation of `<(PointT, PointT, PointT, PointT) as Slots>::prepend_arr::<PointT, [f32; 8]>::{constant#0}` failed here
```
`SlotType` needs a `Copy` supertrait, because tuples are only `Copy` if their elements are. Without it:
`error[E0277]: the trait bound 'H: Copy' is not satisfied in '(H,)'`.
**Recommendation:** use Cons internally (simpler, no length cap, no witness), and give users a type macro
`slots![Twist, Twist]` (like frunk's `HList![]`) or type aliases. Tuples are feasible if the syntax matters.

---

## 3. Stable "specialization" workarounds

* **Impls disjoint only by an associated-type value are NOT accepted** (`e09_assoc_disjoint.rs`), on stable and on the
  nightly next solver alike:
  ```rust
  impl<S: Slots<IsEmpty = True>>  Describe for Point<S> { .. }
  impl<S: Slots<IsEmpty = False>> Describe for Point<S> { .. }
  ```
  ```
  error[E0119]: conflicting implementations of trait `Describe` for type `Point<_>`
  ```
  (This is lang issue rust-lang/rust#20400: coherence does not use projection values.)
* **The standard workaround is a helper trait with the marker as a type parameter** (`e10_helper_dispatch.rs`):
  `impl<S> DescribeImpl<True> for Point<S>`, `impl<S> DescribeImpl<False> for Point<S>`, plus
  `impl<S: Slots> Describe for Point<S> where Point<S>: DescribeImpl<S::IsEmpty>`. Coherent, and it works for concrete S.
  **But generic `S: Slots` cannot use it**:
  ```
  error[E0277]: the trait bound `Point<S>: Describe` is not satisfied
  help: the trait `DescribeImpl<<S as Slots>::IsEmpty>` is not implemented for `Point<S>`
  ```
  Conclusion: generic-visible case analysis must live in **methods/assoc items of `Slots`** implemented once for `Nil`
  and once for `Cons` (as `map`/`outer`/`reassoc` do). That is "specialization by recursion", and it works on stable.
* **TypeId**: `TypeId::of::<T>()` is a stable `const fn` in 1.98 (`q3/d2_typeid_const.rs` compiles). Comparing
  TypeIds in a `const fn` is not stable:
  ```
  error[E0658]: cannot call conditionally-const operator in constant functions
  error: `PartialEq` is not yet stable as a const trait
  ```
  At runtime `if TypeId::of::<T>() == TypeId::of::<f32>()` folds completely at `-O`. The LLVM IR for `pick::<f64>` is
  `ret i32 2`, and for `pick::<f32>` it is `ret i32 1` (`q3/d3_fold.rs`). It needs `T: 'static`.
  `core::any::type_name` is **not** const-stable: `error: 'std::any::type_name' is not yet stable as a const fn`.
* **dtolnay autoref specialization** (`q3/g_autoref.rs`, runs):
  ```
  concrete f32: f32 fast path
  concrete f64: generic fallback
  generic::<f32>: generic fallback     <- inside a generic fn the method is resolved before monomorphization
  ```
  It is only useful in macros applied to concrete types.
* **const fn in traits / const traits**: `trait Zero { const fn zero() -> Self; }` fails with
  `error[E0379]: functions in traits cannot be declared const`. `const trait Zero {}` fails with
  `error[E0658]: const trait impls are experimental` (issue #143874). Note that core's own source already declares
  `pub const trait BitXor` (seen in an error note), but that is unstable for users.
* **Array lengths from generics** (`q3/`):
  * `[f32; D::N]` with `D: Dim` (associated const): `error: generic parameters may not be used in const operations` /
    `note: type parameters may not be used in const expressions`
  * `[f32; N * M]` with const params: `help: const parameters may only be used as standalone arguments here, i.e. 'N'`
  * `type Arr<X> = [X; K]` in `impl<const K: usize>` is **OK** (bare param). `[X; K * K]` gives the same error.
  * Associated consts are fine as **values** and inside inline `const {}` blocks (post-mono), e.g. `S::SIZE` in `e17`.
  * Consequence: every slot type needs a literal-length `Arr` GAT (a macro per slot type). Build nesting via GAT
    composition, never by length arithmetic.
* **GAT notes**: `type Arr<X: Copy>: Copy` works, and the `Copy` item bound propagates to generic code (no `for<'a>` needed,
  because there are no lifetimes). GAT equality in supertraits (`HasCat<Cat<Nil> = Self>`) works. Conditional bounds
  ("`Arr<X>: Debug` if `X: Debug`") are not expressible, so fix a coefficient trait instead. Associated type bounds on GATs in
  where-clauses work: `S: Slots<Arr<f32>: Debug>` compiles (`q4/atb.rs`).

## 4. What is (and is not) available by 1.98 / September 2026

| feature | status on 1.98 stable (verified) |
|---|---|
| GATs (1.65) | stable; basis of the design |
| associated type bounds `T: Trait<Assoc: Bound>` (1.79) | stable, incl. on GATs (`q4/atb.rs`) |
| inline `const {}` (1.79), incl. generic, post-mono | stable (`e15`, `e17`) |
| trait upcasting `&dyn B -> &dyn A` (1.86) | stable (`q4/upcast.rs`) |
| RPITIT `fn f() -> impl Trait` in traits (1.75) | stable (`q4/rpitit.rs`) |
| `TypeId::of` as const fn | stable; `==` in const not stable |
| ATPIT `type It = impl Iterator` in impls | **unstable**: `E0658: 'impl Trait' in associated types is unstable` (#63063) |
| TAIT | **unstable** (#63063) |
| const traits / `const fn` in traits | **unstable** (#143874 / E0379) |
| `[T; S::N]`, `[T; N*M]` in generic context | **error** on stable |
| `min_generic_const_args` (nightly 1.100) | plain `D::N` still rejected. Nightly asks for `#![feature(generic_const_args)]` and a "`type const` item", and the syntax is in flux (my guesses `type const N` / `#[type_const]` did not parse). Per the 2026 project goal it is a prototype, and no stabilization date has been announced. |
| `generic_const_exprs` | still incomplete, nightly only. Nightly 1.100 emits "`feature(generic_const_exprs)` is not supported with the next-generation trait solver" and falls back to the old solver. |
| next-generation trait solver | default **on nightly since 2026-08-22** (blog 2026-08-21). The target is stable 1.100 (Dec 2026), fallback 1.101. Not usable on stable 1.98 (`-Znext-solver` is nightly-only). |
| `non_lifetime_binders` (`for<T>`) | nightly, incomplete; broke the design in a test |

**Does the next solver help the key problem?** No. Design B compiles identically under `-Znext-solver=globally`. The
baseline (`e01`, `e02`) fails identically, because right identity for an abstract `S` is a theorem about all impls, and no
solver derives that. Coherence with assoc-type markers is also still rejected (E0119). Its benefits are elsewhere:
higher-ranked projections, TAIT/ATPIT unblocking, and compile time.

## Sources
* [Enabling the next-generation trait solver on nightly (Rust Blog, 2026-08-21)](https://blog.rust-lang.org/2026/08/21/enabling-next-solver-on-nightly/)
* [Project goal 2026: Stabilize the next-generation trait solver](https://rust-lang.github.io/rust-project-goals/2026/next-solver.html)
* [Rust's New Trait Solver Is Live on Nightly (byteiota, gives the 1.100/1.101 target)](https://byteiota.com/rust-next-gen-trait-solver-nightly-2026/)
* [Project goal 2026: Full Const Generics](https://goals.rust-lang.org/2026/const-generics.html)
* [Project goals update, April 2026](https://blog.rust-lang.org/2026/05/18/project-goals-2026-04/)
* Local clones: `ref/variadics_please` (README, `src/lib.rs`), `ref/frunk` (`core/src/hlist.rs`, `core/src/indices.rs`, `core/src/macros.rs`)

## Experiment index (`scratchpad/typelevel/`)
| file | result |
|---|---|
| e01_baseline.rs | E0369 (the key problem) |
| e02_bound_only.rs | E0308 with `Concat<Nil>`; OK with `Concat<Nil, Out = S>` |
| e03a_super.rs | E0391 cycle (Concat: Slots) |
| e03b_super.rs / e04_super_full.rs | design A compiles; user fn OK |
| e04b.rs | design A: E0277 `S1: Concat<S2>` for 2 generic lists |
| e05_selfwhere.rs | E0391 cycle (`where Self: Slots<Cat<Nil>=Self>`) |
| e06_gat_helper.rs | **design B compiles; all user fns OK** (stable + next solver) |
| e07_overlap_value_right.rs | E0119 |
| e08_four_cases.rs | coherent, but generic use E0369 |
| e09_assoc_disjoint.rs | E0119 (stable and next solver) |
| e10_helper_dispatch.rs | concrete OK; generic E0277 |
| e11_storage.rs | **design B + nested GAT storage + outer product, runs** |
| e12_assoc.rs / e13_reassoc.rs | associativity fails / witness fix compiles |
| e14_tuples.rs / e15_tuples_postmono.rs | tuple lists run; post-mono overflow error |
| e16_derive.rs | derive Copy OK generically; Debug needs bounds |
| e17_flat.rs | flat slice view with inline-const size check, runs |
| e18_selector.rs | frunk index inference, overlap and ambiguity demos |
| slots_prelude.rs | the design-B core, reusable via `include!` |
| q3/*, q4/* | const-generics / TypeId / const-trait / autoref / feature-status probes |
