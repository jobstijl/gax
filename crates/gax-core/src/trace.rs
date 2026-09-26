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
    /// For a value of one kind (or a `Unit` of one): write the kind's path and return whether
    /// it is a `Unit`. `None` for everything else (scalars, tuples, arrays).
    fn write_kind(_w: &mut dyn Write) -> Option<bool> {
        None
    }
    /// WGSL: write the type. A kind of a standard algebra is a WESL path (`gax::pga3d::Point`),
    /// a scalar is `f32`, an array `array<T, N>`, and a tuple a struct named `hint`, declared
    /// in `decls`. Types without a WGSL form (kinds of `algebra!` algebras) return an error.
    ///
    /// # Errors
    /// For types without a WGSL form, and the formatter's errors.
    fn write_wgsl_type(_w: &mut dyn Write, _decls: &mut dyn Write, _hint: &str) -> fmt::Result {
        Err(fmt::Error)
    }
    /// WGSL: write the expression of coefficient `i` of the value named `name`.
    ///
    /// # Errors
    /// For types without a WGSL form, and the formatter's errors.
    fn write_wgsl_coeff(_w: &mut dyn Write, _name: &str, _i: usize) -> fmt::Result {
        Err(fmt::Error)
    }
    /// WGSL: write an expression constructing the value from `LEN` coefficient expressions
    /// (`hint` as for [`Traceable::write_wgsl_type`]).
    ///
    /// # Errors
    /// For types without a WGSL form, and the formatter's errors.
    fn write_wgsl_construct(_w: &mut dyn Write, _hint: &str, _parts: &[&str]) -> fmt::Result {
        Err(fmt::Error)
    }
}

/// The WESL path of a kind of a standard algebra, `gax::pga3d::Point`; an error for other
/// algebras (which have no generated WGSL module).
fn write_wgsl_kind<K: Kind>(w: &mut dyn Write) -> fmt::Result {
    let module = K::MODULE;
    let (first, rest) = module.split_once("::").unwrap_or((module, ""));
    if first != "gax" || rest.is_empty() {
        return Err(fmt::Error);
    }
    write!(w, "gax::{rest}::{}", K::NAME)
}

fn write_wgsl_kind_coeff(w: &mut dyn Write, name: &str, i: usize) -> fmt::Result {
    write!(w, "{name}.c{}.{}", i / 4, ["x", "y", "z", "w"][i % 4])
}

fn write_wgsl_kind_construct<K: Kind>(w: &mut dyn Write, parts: &[&str]) -> fmt::Result {
    write_wgsl_kind::<K>(w)?;
    w.write_str("(")?;
    for (f, chunk) in parts.chunks(4).enumerate() {
        if f > 0 {
            w.write_str(", ")?;
        }
        w.write_str("vec4<f32>(")?;
        for j in 0..4 {
            if j > 0 {
                w.write_str(", ")?;
            }
            w.write_str(chunk.get(j).copied().unwrap_or("0.0"))?;
        }
        w.write_str(")")?;
    }
    w.write_str(")")
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
    fn write_kind(w: &mut dyn Write) -> Option<bool> {
        write_kind_path::<M::Kind>(w).ok()?;
        Some(false)
    }
    fn write_wgsl_type(w: &mut dyn Write, _decls: &mut dyn Write, _hint: &str) -> fmt::Result {
        write_wgsl_kind::<M::Kind>(w)
    }
    fn write_wgsl_coeff(w: &mut dyn Write, name: &str, i: usize) -> fmt::Result {
        write_wgsl_kind_coeff(w, name, i)
    }
    fn write_wgsl_construct(w: &mut dyn Write, _hint: &str, parts: &[&str]) -> fmt::Result {
        write_wgsl_kind_construct::<M::Kind>(w, parts)
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
    fn write_kind(w: &mut dyn Write) -> Option<bool> {
        write_kind_path::<M::Kind>(w).ok()?;
        Some(true)
    }
    fn write_wgsl_type(w: &mut dyn Write, _decls: &mut dyn Write, _hint: &str) -> fmt::Result {
        write_wgsl_kind::<M::Kind>(w)
    }
    fn write_wgsl_coeff(w: &mut dyn Write, name: &str, i: usize) -> fmt::Result {
        write_wgsl_kind_coeff(w, name, i)
    }
    fn write_wgsl_construct(w: &mut dyn Write, _hint: &str, parts: &[&str]) -> fmt::Result {
        write_wgsl_kind_construct::<M::Kind>(w, parts)
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
            fn write_coeffs(w: &mut dyn Write, name: &str) -> fmt::Result {
                // `{ let t = x; let c = ({ let v = t.0; coeffs of v }, ...); [c.0[0], ...] }`
                write!(w, "{{ let t = {name}; let c = (")?;
                $(
                    write!(w, "{{ let v = t.{}; ", $i)?;
                    $A::write_coeffs(w, "v")?;
                    w.write_str(" }, ")?;
                )+
                w.write_str("); [")?;
                $( for j in 0..$A::LEN { write!(w, "c.{}[{j}], ", $i)?; } )+
                w.write_str("] }")
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
            fn write_wgsl_type(w: &mut dyn Write, decls: &mut dyn Write, hint: &str) -> fmt::Result {
                // Field types first (they may declare structs of their own).
                let mut fields = [$({ let _ = $i; FieldBuf::new() },)+];
                $( $A::write_wgsl_type(&mut fields[$i], decls, &FieldBuf::hint(hint, $i))?; )+
                write!(decls, "struct {hint} {{\n")?;
                for (k, t) in fields.iter().enumerate() {
                    write!(decls, "    f{k}: {},\n", t.as_str())?;
                }
                decls.write_str("}\n\n")?;
                w.write_str(hint)
            }
            fn write_wgsl_coeff(w: &mut dyn Write, name: &str, i: usize) -> fmt::Result {
                let mut start = 0;
                $(
                    if i >= start && i < start + $A::LEN {
                        return $A::write_wgsl_coeff(w, &FieldBuf::field(name, $i), i - start);
                    }
                    start += $A::LEN;
                )+
                let _ = start;
                Err(fmt::Error)
            }
            fn write_wgsl_construct(w: &mut dyn Write, hint: &str, parts: &[&str]) -> fmt::Result {
                let mut start = 0;
                write!(w, "{hint}(")?;
                $(
                    if $i > 0 {
                        w.write_str(", ")?;
                    }
                    $A::write_wgsl_construct(w, &FieldBuf::hint(hint, $i), &parts[start..start + $A::LEN])?;
                    start += $A::LEN;
                )+
                let _ = start;
                w.write_str(")")
            }
        }
    };
}

/// A small fixed-capacity string for generated WGSL names (no allocation in `no_std`).
struct FieldBuf {
    buf: [u8; 128],
    len: usize,
}

impl FieldBuf {
    fn new() -> FieldBuf {
        FieldBuf {
            buf: [0; 128],
            len: 0,
        }
    }
    fn as_str(&self) -> &str {
        core::str::from_utf8(&self.buf[..self.len]).unwrap_or("")
    }
    /// `{hint}_{k}`.
    fn hint(hint: &str, k: usize) -> FieldBuf {
        let mut b = FieldBuf::new();
        let _ = write!(b, "{hint}_{k}");
        b
    }
    /// `{name}.f{k}`.
    fn field(name: &str, k: usize) -> FieldBuf {
        let mut b = FieldBuf::new();
        let _ = write!(b, "{name}.f{k}");
        b
    }
}

impl core::ops::Deref for FieldBuf {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl Write for FieldBuf {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let end = self.len + s.len();
        if end > self.buf.len() {
            return Err(fmt::Error);
        }
        self.buf[self.len..end].copy_from_slice(s.as_bytes());
        self.len = end;
        Ok(())
    }
}

tuple_traceable!(A0 0, A1 1);
tuple_traceable!(A0 0, A1 1, A2 2);
tuple_traceable!(A0 0, A1 1, A2 2, A3 3);

/// Arrays of traceable values, such as `[T; 3]` of scalars.
impl<A: Traceable, const N: usize> Traceable for [A; N] {
    type Coef = A::Coef;
    const LEN: usize = A::LEN * N;
    fn from_fn(f: &mut dyn FnMut(usize) -> A::Coef) -> Self {
        let mut k = 0;
        core::array::from_fn(|_| {
            let base = k;
            k += A::LEN;
            A::from_fn(&mut |i| f(base + i))
        })
    }
    fn for_each(&self, f: &mut dyn FnMut(A::Coef)) {
        for a in self {
            a.for_each(f);
        }
    }
    fn write_type(w: &mut dyn Write, coef: &str) -> fmt::Result {
        w.write_str("[")?;
        A::write_type(w, coef)?;
        write!(w, "; {N}]")
    }
    fn write_coeffs(w: &mut dyn Write, name: &str) -> fmt::Result {
        // `{ let t = x; let c = [{ let v = t[0]; coeffs of v }, ...]; [c[0][0], ...] }`
        write!(w, "{{ let t = {name}; let c = [")?;
        for i in 0..N {
            write!(w, "{{ let v = t[{i}]; ")?;
            A::write_coeffs(w, "v")?;
            w.write_str(" }, ")?;
        }
        w.write_str("]; [")?;
        for i in 0..N {
            for j in 0..A::LEN {
                write!(w, "c[{i}][{j}], ")?;
            }
        }
        w.write_str("] }")
    }
    fn write_construct(w: &mut dyn Write, parts: &[&str]) -> fmt::Result {
        w.write_str("[")?;
        for i in 0..N {
            if i > 0 {
                w.write_str(", ")?;
            }
            A::write_construct(w, &parts[i * A::LEN..(i + 1) * A::LEN])?;
        }
        w.write_str("]")
    }
    fn for_each_condition(&self, f: &mut dyn FnMut(A::Coef)) {
        for a in self {
            a.for_each_condition(f);
        }
    }
    fn write_wgsl_type(w: &mut dyn Write, decls: &mut dyn Write, hint: &str) -> fmt::Result {
        w.write_str("array<")?;
        A::write_wgsl_type(w, decls, &FieldBuf::hint(hint, 0))?;
        write!(w, ", {N}>")
    }
    fn write_wgsl_coeff(w: &mut dyn Write, name: &str, i: usize) -> fmt::Result {
        let mut elem = FieldBuf::new();
        write!(elem, "{name}[{}]", i / A::LEN)?;
        A::write_wgsl_coeff(w, &elem, i % A::LEN)
    }
    fn write_wgsl_construct(w: &mut dyn Write, hint: &str, parts: &[&str]) -> fmt::Result {
        let inner = FieldBuf::hint(hint, 0);
        w.write_str("array(")?;
        for i in 0..N {
            if i > 0 {
                w.write_str(", ")?;
            }
            A::write_wgsl_construct(w, &inner, &parts[i * A::LEN..(i + 1) * A::LEN])?;
        }
        w.write_str(")")
    }
}
