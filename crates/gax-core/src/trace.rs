//! What build-time tracing needs to know about kernel arguments and results.
//!
//! The tracer itself (the symbolic coefficient type and the code emitter) lives in `gax-gen`
//! and is exposed as `gax::trace`. This trait is implemented here, for every value of a
//! generated kind and for [`Unit`] values, so that `gax-gen` can stay generic.

use crate::coef::Coef;
use crate::kind::{Extensor, Kind};
use crate::ops::{Gp, Reverse};
use crate::unit::Unit;
use core::fmt::{self, Write};

/// A kernel argument or result that tracing can construct symbolically and name in code.
///
/// The `write_*` methods write Rust source to a formatter.
///
/// # Errors
/// They propagate the formatter's errors.
pub trait Traceable: Sized {
    /// The coefficient type.
    type Coef: Coef;
    /// Number of coefficients.
    const LEN: usize;
    /// Build from coefficients.
    fn from_fn(f: &mut dyn FnMut(usize) -> Self::Coef) -> Self;
    /// Visit the coefficients in order.
    fn for_each(&self, f: &mut dyn FnMut(Self::Coef));
    /// Write the Rust type, with `coef` as the coefficient type.
    ///
    /// # Errors
    /// Propagates the formatter's errors.
    fn write_type(w: &mut dyn Write, coef: &str) -> fmt::Result;
    /// Write an expression for the coefficient array `[T; LEN]` of the value named `name`.
    ///
    /// # Errors
    /// Propagates the formatter's errors.
    fn write_coeffs(w: &mut dyn Write, name: &str) -> fmt::Result;
    /// Write an expression constructing the value from `LEN` coefficient expressions.
    ///
    /// # Errors
    /// Propagates the formatter's errors.
    fn write_construct(w: &mut dyn Write, parts: &[&str]) -> fmt::Result;
    /// Visit the polynomial conditions (each must vanish) that the value is certified to
    /// satisfy, such as the coefficients of `x ~x - 1` for a unit versor.
    fn for_each_condition(&self, _f: &mut dyn FnMut(Self::Coef)) {}
}

/// The path of a kind in code outside its own crate: `::gax::pga3d::Point` for the standard
/// algebras, `crate::path::Point` for algebras declared in the user's crate.
///
/// # Errors
/// Propagates the formatter's errors.
pub fn write_kind_path<K: Kind>(w: &mut dyn Write) -> fmt::Result {
    let module = K::MODULE;
    let (first, rest) = module.split_once("::").unwrap_or((module, ""));
    if first == "gax" {
        w.write_str("::gax")?;
    } else {
        w.write_str("crate")?;
    }
    if !rest.is_empty() {
        write!(w, "::{rest}")?;
    }
    write!(w, "::{}", K::NAME)
}

fn write_array(w: &mut dyn Write, parts: &[&str]) -> fmt::Result {
    w.write_str("[")?;
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            w.write_str(", ")?;
        }
        w.write_str(p)?;
    }
    w.write_str("]")
}

impl<M: Extensor<Slots = ()>> Traceable for M {
    type Coef = M::Coef;
    const LEN: usize = <M::Kind as Kind>::N;
    fn from_fn(f: &mut dyn FnMut(usize) -> M::Coef) -> M {
        M::from_coeffs(<M::Kind as Kind>::arr_from_fn(f))
    }
    fn for_each(&self, f: &mut dyn FnMut(M::Coef)) {
        for &c in self.coeffs().as_ref() {
            f(c);
        }
    }
    fn write_type(w: &mut dyn Write, coef: &str) -> fmt::Result {
        write_kind_path::<M::Kind>(w)?;
        write!(w, "<(), {coef}>")
    }
    fn write_coeffs(w: &mut dyn Write, name: &str) -> fmt::Result {
        write!(w, "{name}.c")
    }
    fn write_construct(w: &mut dyn Write, parts: &[&str]) -> fmt::Result {
        write_kind_path::<M::Kind>(w)?;
        w.write_str("::from_coeffs(")?;
        write_array(w, parts)?;
        w.write_str(")")
    }
}

impl<M> Traceable for Unit<M>
where
    M: Extensor<Slots = ()> + Reverse<Output = M> + Gp<M>,
    <M as Gp<M>>::Output: Extensor<Slots = (), Coef = M::Coef>,
{
    type Coef = M::Coef;
    const LEN: usize = <M::Kind as Kind>::N;
    fn from_fn(f: &mut dyn FnMut(usize) -> M::Coef) -> Self {
        Unit::new_unchecked(M::from_coeffs(<M::Kind as Kind>::arr_from_fn(f)))
    }
    fn for_each(&self, f: &mut dyn FnMut(M::Coef)) {
        for &c in self.coeffs().as_ref() {
            f(c);
        }
    }
    fn write_type(w: &mut dyn Write, coef: &str) -> fmt::Result {
        w.write_str("::gax::Unit<")?;
        write_kind_path::<M::Kind>(w)?;
        write!(w, "<(), {coef}>>")
    }
    fn write_coeffs(w: &mut dyn Write, name: &str) -> fmt::Result {
        write!(w, "{name}.into_inner().c")
    }
    fn write_construct(w: &mut dyn Write, parts: &[&str]) -> fmt::Result {
        w.write_str("::gax::Unit::new_unchecked(")?;
        write_kind_path::<M::Kind>(w)?;
        w.write_str("::from_coeffs(")?;
        write_array(w, parts)?;
        w.write_str("))")
    }
    fn for_each_condition(&self, f: &mut dyn FnMut(M::Coef)) {
        let m = **self;
        let norm = m.gp(m.reverse());
        let blades = <<<M as Gp<M>>::Output as Extensor>::Kind as Kind>::BLADES;
        for (b, &c) in blades.iter().zip(norm.coeffs().as_ref()) {
            f(if *b == "1" { c - M::Coef::one() } else { c });
        }
    }
}

macro_rules! tuple_traceable {
    ($($A:ident $i:tt),+) => {
        /// Several results of one kernel.
        impl<C: Coef, $($A: Traceable<Coef = C>),+> Traceable for ($($A,)+) {
            type Coef = C;
            const LEN: usize = 0 $(+ $A::LEN)+;
            #[allow(unused_assignments)]
            fn from_fn(f: &mut dyn FnMut(usize) -> C) -> Self {
                let mut start = 0;
                ($({
                    let s = start;
                    start += $A::LEN;
                    $A::from_fn(&mut |i| f(s + i))
                },)+)
            }
            fn for_each(&self, f: &mut dyn FnMut(C)) {
                $( self.$i.for_each(f); )+
            }
            fn write_type(w: &mut dyn Write, coef: &str) -> fmt::Result {
                w.write_str("(")?;
                $( $A::write_type(w, coef)?; w.write_str(", ")?; )+
                w.write_str(")")
            }
            fn write_coeffs(_w: &mut dyn Write, _name: &str) -> fmt::Result {
                unimplemented!("tuples are results, not kernel arguments")
            }
            fn write_construct(w: &mut dyn Write, parts: &[&str]) -> fmt::Result {
                let mut start = 0;
                w.write_str("(")?;
                $(
                    $A::write_construct(w, &parts[start..start + $A::LEN])?;
                    w.write_str(", ")?;
                    start += $A::LEN;
                )+
                let _ = start;
                w.write_str(")")
            }
            fn for_each_condition(&self, f: &mut dyn FnMut(C)) {
                $( self.$i.for_each_condition(f); )+
            }
        }
    };
}

tuple_traceable!(A0 0, A1 1);
tuple_traceable!(A0 0, A1 1, A2 2);
tuple_traceable!(A0 0, A1 1, A2 2, A3 3);
