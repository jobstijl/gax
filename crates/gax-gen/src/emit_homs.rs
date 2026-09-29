//! Algebra homomorphisms between the standard algebras, as `From` impls.
//!
//! A homomorphism is declared by the image of each basis vector of the source algebra: a
//! combination of target blades. When the images keep the metric (`φ(eᵢ)·φ(eⱼ) = eᵢ·eⱼ`), the
//! map extends to the whole algebra and keeps every product: `φ(a b) = φ(a) φ(b)`, so motors go
//! to motors and `φ(m >> x) = φ(m) >> φ(x)`. The generator checks the metric, derives the image
//! of every blade, and emits `From<src::A<S, T>> for tgt::B<S, T>` for every kind `A`, with `B`
//! the smallest target kind that holds the image. The impls are generic over the slots, so a
//! map converts too, and `tgt::B::from(src::A::slot())` is the homomorphism itself as a map.

use crate::poly::{Poly, Rational};
use crate::spec::{AlgebraSpec, KindSpec};
use crate::symbolic::{self, SymMv};
use crate::table::{BinOp, UnOp};
use std::fmt::Write as _;

/// A homomorphism between two standard algebras.
pub struct Hom {
    /// Source algebra (module name).
    pub from: &'static str,
    /// Target algebra.
    pub to: &'static str,
    /// The image of each source basis vector: `(source vector, [(coefficient, target blade)])`.
    pub images: &'static [(&'static str, &'static [(i64, &'static str)])],
    /// What it is, for the docs.
    pub doc: &'static str,
}

/// The homomorphisms shipped with `gax`: each one natural, and injective.
pub const HOMS: &[Hom] = &[
    Hom {
        from: "vga2d",
        to: "vga3d",
        images: &[("e1", &[(1, "e1")]), ("e2", &[(1, "e2")])],
        doc: "the plane as the `xy` plane of space",
    },
    Hom {
        from: "vga2d",
        to: "pga2d",
        images: &[("e1", &[(1, "e1")]), ("e2", &[(1, "e2")])],
        doc: "vectors as lines through the origin, rotors as rotations about it",
    },
    Hom {
        from: "vga3d",
        to: "pga3d",
        images: &[
            ("e1", &[(1, "e1")]),
            ("e2", &[(1, "e2")]),
            ("e3", &[(1, "e3")]),
        ],
        doc: "vectors as planes through the origin, rotors as rotations about it",
    },
    Hom {
        from: "pga2d",
        to: "pga3d",
        images: &[
            ("e0", &[(1, "e0")]),
            ("e1", &[(1, "e1")]),
            ("e2", &[(1, "e2")]),
        ],
        doc: "the plane as the `xy` plane of space: lines as vertical planes, points as \
              vertical lines, motors as motions about vertical axes",
    },
    Hom {
        from: "pga3d",
        to: "stap",
        images: &[
            ("e0", &[(1, "e0")]),
            ("e1", &[(1, "e1")]),
            ("e2", &[(1, "e2")]),
            ("e3", &[(1, "e3")]),
        ],
        doc: "space in spacetime: planes as hyperplanes containing the time axis, motors as \
              motions at every time",
    },
    Hom {
        from: "vga2d",
        to: "cga2d",
        images: &[("e1", &[(1, "e1")]), ("e2", &[(1, "e2")])],
        doc: "the Euclidean subalgebra",
    },
    Hom {
        from: "vga3d",
        to: "cga3d",
        images: &[
            ("e1", &[(1, "e1")]),
            ("e2", &[(1, "e2")]),
            ("e3", &[(1, "e3")]),
        ],
        doc: "the Euclidean subalgebra",
    },
    Hom {
        from: "pga2d",
        to: "cga2d",
        images: &[
            ("e0", &[(-1, "ei")]),
            ("e1", &[(1, "e1")]),
            ("e2", &[(1, "e2")]),
        ],
        doc: "plane-based PGA in CGA (`e0` to `-ei`): a line `ax + by + c = 0` to the dual \
              line whose points `X` have `X·l = 0`, motors to motors",
    },
    Hom {
        from: "pga3d",
        to: "cga3d",
        images: &[
            ("e0", &[(-1, "ei")]),
            ("e1", &[(1, "e1")]),
            ("e2", &[(1, "e2")]),
            ("e3", &[(1, "e3")]),
        ],
        doc: "plane-based PGA in CGA (`e0` to `-ei`): a plane `ax + by + cz + d = 0` to the \
              dual plane whose points `X` have `X·π = 0`, motors to motors",
    },
    Hom {
        from: "cga2d",
        to: "cga3d",
        images: &[
            ("e1", &[(1, "e1")]),
            ("e2", &[(1, "e2")]),
            ("eo", &[(1, "eo")]),
            ("ei", &[(1, "ei")]),
        ],
        doc: "the plane as the `xy` plane of space",
    },
    Hom {
        from: "cga3d",
        to: "csta",
        images: &[
            ("e1", &[(1, "e1")]),
            ("e2", &[(1, "e2")]),
            ("e3", &[(1, "e3")]),
            ("eo", &[(1, "eo")]),
            ("ei", &[(1, "ei")]),
        ],
        doc: "conformal space in conformal spacetime, at time zero",
    },
    Hom {
        from: "vga3d",
        to: "sta",
        images: &[
            ("e1", &[(1, "e10")]),
            ("e2", &[(1, "e20")]),
            ("e3", &[(1, "e30")]),
        ],
        doc: "the spacetime split for the observer `e0`: vectors as relative vectors \
              `eₖ e0` (bivectors), and space's algebra as the even subalgebra",
    },
];

/// Emit the `From` impls of one homomorphism, gated on both algebras' features.
///
/// The impls, and each source kind with the target kind of its image.
///
/// # Errors
/// If an image names an unknown blade, or the images do not keep the metric.
pub fn emit_hom(
    src: &AlgebraSpec,
    tgt: &AlgebraSpec,
    hom: &Hom,
) -> Result<(String, Vec<(String, String)>), String> {
    let (sa, ta) = (&src.algebra, &tgt.algebra);
    let n = sa.dim();
    // The image of each basis vector.
    let mut images: Vec<SymMv> = vec![SymMv::new(); n];
    for (name, terms) in hom.images {
        let (m, s) = sa.parse_blade(name).map_err(|e| e.0)?;
        if m.count_ones() != 1 {
            return Err(format!("{}: {name} is not a basis vector", hom.from));
        }
        let i = m.trailing_zeros() as usize;
        let mut img = SymMv::new();
        for &(c, blade) in *terms {
            let (tm, ts) = ta.parse_blade(blade).map_err(|e| e.0)?;
            let e = img.entry(tm).or_default();
            *e = &*e + &Poly::constant(Rational::int(i128::from(c * ts * s)));
        }
        img.retain(|_, p| !p.is_zero());
        images[i] = img;
    }
    // The metric: φ(eᵢ) φ(eⱼ) + φ(eⱼ) φ(eᵢ) = 2 eᵢ·eⱼ.
    for i in 0..n {
        for j in 0..n {
            let a = symbolic::binop(ta, BinOp::Gp, &images[i], &images[j]);
            let b = symbolic::binop(ta, BinOp::Gp, &images[j], &images[i]);
            let sym = symbolic::add(&a, &b);
            let want = symbolic::scalar(Rational::int(i128::from(2 * sa.metric()[i][j])));
            if sym != want {
                return Err(format!(
                    "{} -> {}: the images of e{} and e{} do not keep the metric",
                    hom.from,
                    hom.to,
                    sa.basis_suffixes()[i],
                    sa.basis_suffixes()[j]
                ));
            }
        }
    }
    // The image of a blade: the product of its factors' images. Blades are wedge products; for
    // images that are vectors the wedge of the images is the image of the wedge. Otherwise the
    // source is orthogonal (checked), where a blade is also the geometric product of its
    // factors.
    let vectors = images
        .iter()
        .all(|img| img.keys().all(|m| m.is_power_of_two()));
    let orthogonal = (0..n).all(|i| (0..n).all(|j| i == j || sa.metric()[i][j] == 0));
    if !vectors && !orthogonal {
        return Err(format!(
            "{} -> {}: vectors must go to vectors unless the source basis is orthogonal",
            hom.from, hom.to
        ));
    }
    let op = if vectors { BinOp::Wedge } else { BinOp::Gp };
    let blade = |mask: u32| -> SymMv {
        (0..n)
            .filter(|i| mask & (1 << i) != 0)
            .fold(symbolic::scalar(Rational::ONE), |acc, i| {
                symbolic::binop(ta, op, &acc, &images[i])
            })
    };

    let mut out = String::new();
    let _ = writeln!(
        out,
        "/// The homomorphism from `{}` to `{}`: {}.",
        hom.from, hom.to, hom.doc
    );
    let _ = writeln!(
        out,
        "#[cfg(all(feature = \"{}\", feature = \"{}\"))]\nmod {}_to_{} {{\n    use crate::{{Coef, SlotArr, Slots}};\n",
        hom.from, hom.to, hom.from, hom.to
    );
    // Proved here, exactly: the image of the product of two symbolic values of every pair of
    // source kinds is the product of their images (the metric check makes it so; this checks
    // the blade images too).
    for a in &src.kinds {
        for b in &src.kinds {
            let va = symbolic::variables(&a.layout, 0);
            let vb = symbolic::variables(&b.layout, a.layout.len() as crate::poly::Var);
            let image = |mv: &SymMv| -> SymMv {
                mv.iter().fold(SymMv::new(), |acc, (&m, p)| {
                    let img: SymMv = blade(m).into_iter().map(|(t, c)| (t, &c * p)).collect();
                    symbolic::add(&acc, &img)
                })
            };
            let lhs = image(&symbolic::binop(sa, BinOp::Gp, &va, &vb));
            let rhs = symbolic::binop(ta, BinOp::Gp, &image(&va), &image(&vb));
            if lhs != rhs {
                return Err(format!(
                    "{} -> {}: the image of {} * {} is not the product of the images",
                    hom.from, hom.to, a.name, b.name
                ));
            }
        }
    }
    let mut kinds = Vec::new();
    for a in &src.kinds {
        if let Some((code, b)) = kind_impl(src, tgt, a, &blade) {
            out.push_str(&code);
            kinds.push((a.name.clone(), b));
        }
    }
    out.push_str("}\n\n");
    Ok((out, kinds))
}

/// The `From` impl for source kind `a`, if its image is not zero.
fn kind_impl(
    src: &AlgebraSpec,
    tgt: &AlgebraSpec,
    a: &KindSpec,
    blade: &impl Fn(u32) -> SymMv,
) -> Option<(String, String)> {
    // The image of each of the kind's coefficients (with the layout's orientation).
    let columns: Vec<SymMv> = a
        .layout
        .blades
        .iter()
        .map(|&(m, s)| {
            blade(m)
                .into_iter()
                .map(|(tm, c)| (tm, c.scale(Rational::int(i128::from(s)))))
                .collect()
        })
        .collect();
    let support: std::collections::BTreeSet<u32> =
        columns.iter().flat_map(|c| c.keys().copied()).collect();
    if support.is_empty() {
        return None;
    }
    let b = tgt.kind_for_support(&support)?;
    // Each target coefficient as a combination of the source coefficients.
    let exprs: Vec<String> = b
        .layout
        .blades
        .iter()
        .map(|&(tm, ts)| {
            let terms: Vec<(i128, usize)> = columns
                .iter()
                .enumerate()
                .filter_map(|(k, col)| {
                    let c = col.get(&tm)?.as_constant()?;
                    assert_eq!(c.den(), 1, "integer images");
                    Some((c.num() * i128::from(ts), k))
                })
                .collect();
            combination(&terms)
        })
        .collect();
    let (sm, tm) = (src.name.as_str(), tgt.name.as_str());
    // The reverse commutes with the map on this kind (`φ(~a) = ~φ(a)`), so it keeps
    // `x ~x = 1`: then `Unit::widen` may carry a unit versor across.
    let va = symbolic::variables(&a.layout, 0);
    let image = |mv: &SymMv| -> SymMv {
        mv.iter().fold(SymMv::new(), |acc, (&m, p)| {
            let img: SymMv = blade(m).into_iter().map(|(t, c)| (t, &c * p)).collect();
            symbolic::add(&acc, &img)
        })
    };
    let keeps_unit = image(&symbolic::unop(&src.algebra, UnOp::Reverse, &va))
        == symbolic::unop(&tgt.algebra, UnOp::Reverse, &image(&va));
    let (an, bn) = (&a.name, &b.name);
    let widen = if keeps_unit {
        format!(
            "    impl<T: Coef> crate::Widen<crate::{tm}::{bn}<(), T>> for crate::{sm}::{an}<(), T> {{}}\n\n"
        )
    } else {
        String::new()
    };
    let code = format!(
        "    impl<S: Slots, T: Coef> From<crate::{sm}::{an}<S, T>> for crate::{tm}::{bn}<S, T> {{\n        /// The image under the homomorphism from `{sm}` to `{tm}`.\n        #[inline(always)]\n        fn from(x: crate::{sm}::{an}<S, T>) -> Self {{\n            let a = x.c.map(SlotArr::<S, T>);\n            crate::{tm}::{bn}::from_coeffs([{}])\n        }}\n    }}\n\n",
        exprs.join(", ")
    ) + &widen;
    Some((code, bn.clone()))
}

/// `Σ c_k a[k]` as slot-array arithmetic (a copy, a negation, or a sum; zero when empty).
fn combination(terms: &[(i128, usize)]) -> String {
    if terms.is_empty() {
        return "<S as Slots>::from_flat(&mut |_| T::zero(), 0)".into();
    }
    let term = |&(c, k): &(i128, usize)| match c.abs() {
        1 => format!("a[{k}]"),
        m => format!("a[{k}].scale(T::from_i64({m}))"),
    };
    let mut e = String::new();
    for (i, t) in terms.iter().enumerate() {
        let s = term(t);
        e = match (i, t.0 < 0) {
            (0, false) => s,
            (0, true) => format!("-{s}"),
            (_, false) => format!("{e} + {s}"),
            (_, true) => format!("{e} - {s}"),
        };
    }
    format!("({e}).0")
}

/// A test of the homomorphism property, `φ(a b) = φ(a) φ(b)`, for every pair of source kinds
/// whose product exists, compared as target multivectors, on random values.
pub fn emit_hom_test(
    src: &AlgebraSpec,
    tgt: &AlgebraSpec,
    hom: &Hom,
    kinds: &[(String, String)],
    products: &[(String, String, String)],
) -> String {
    let image = |k: &str| kinds.iter().find(|(a, _)| a == k).map(|(_, b)| b.as_str());
    let (sm, tm) = (src.name.as_str(), tgt.name.as_str());
    // One function per pair: a single function holding every pair's values would overflow
    // the stack in a debug build of the larger algebras.
    let mut fns = String::new();
    let mut calls = String::new();
    for (k, (a, b, c)) in products.iter().enumerate() {
        let (Some(ia), Some(ib)) = (image(a), image(b)) else {
            continue;
        };
        // A zero image of the product (a kind that maps to nothing) is compared as zero.
        let lhs = match image(c) {
            Some(ic) => format!("{tm}::Multivector::from({tm}::{ic}::from(a * b))"),
            None => format!("{tm}::Multivector::zero()"),
        };
        let _ = writeln!(
            fns,
            "    #[inline(never)]\n    fn p{k}(rng: &mut Rng) {{\n        let (a, b): ({sm}::{a}<(), f64>, {sm}::{b}<(), f64>) = (rng.value(), rng.value());\n        let rhs = {tm}::Multivector::from({tm}::{ia}::from(a) * {tm}::{ib}::from(b));\n        assert!(close(&{lhs}, &rhs), \"{a} * {b}\");\n    }}"
        );
        let _ = writeln!(calls, "        p{k}(&mut rng);");
    }
    format!(
        "/// {doc}\n#[cfg(all(feature = \"{sm}\", feature = \"{tm}\"))]\n#[test]\nfn {sm}_to_{tm}() {{\n    use gax::{{{sm}, {tm}}};\n{fns}    let mut rng = Rng(0x{seed:x});\n    for _ in 0..4 {{\n{calls}    }}\n}}\n\n",
        doc = hom.doc,
        seed = sm.len() * 7919 + tm.len() * 104_729 + 1,
    )
}

/// The head of the generated test file: a small generator of values and a tolerance.
pub const TEST_HEAD: &str = "//! @generated by gax-regen. Do not edit by hand.
//!
//! The homomorphisms between the standard algebras keep products: `φ(a b) = φ(a) φ(b)` for
//! every pair of kinds, compared as multivectors of the target on random values.

#![cfg_attr(rustfmt, rustfmt::skip)]
// A multivector converts to itself when a kind's image is the whole algebra.
// The helpers go unused when no pair of algebras here is enabled.
#![allow(dead_code, clippy::pedantic, clippy::too_many_lines, clippy::useless_conversion)]

use gax::Extensor;

struct Rng(u64);
impl Rng {
    fn value<M: Extensor<Slots = (), Coef = f64>>(&mut self) -> M {
        M::from_coeffs(<M::Kind as gax::Kind>::arr_from_fn(|_| {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        }))
    }
}

fn close<M: Extensor<Slots = (), Coef = f64>>(a: &M, b: &M) -> bool {
    let (a, b) = (a.coeffs().as_ref(), b.coeffs().as_ref());
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-12 * (1.0 + x.abs().max(y.abs())))
}

";
