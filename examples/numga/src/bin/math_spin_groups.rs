//! numga's `math/spin_groups`: the spin groups of every signature up to six dimensions.
//!
//! Choosing p directions that square to plus one and q that square to minus one picks out the
//! signature (p, q): the bivectors of the chosen directions generate its rotors, the spin
//! group, and their even products are its spinors. Three maps with open slots tell the groups
//! apart.
//!
//! The invariant form of the rotors' Lie algebra, the trace of the double commutator with two
//! bivectors left open, is the algebra's own inner product on those bivectors, times
//! `2 (n - 2)`: the geometric product carries it already.
//!
//! Sandwiching the bivectors with the product of the positive directions is an involution: it
//! negates exactly the planes that mix a positive with a negative direction. The planes it fixes
//! generate rotations, the planes it negates generate boosts; a group is compact when every
//! plane is a rotation.
//!
//! In an even number of dimensions the pseudoscalar commutes with every spinor. When it squares
//! to plus one it splits the spinors into two halves; when it squares to minus one it acts on
//! them as the complex unit (its eigenvalues, from gax's `eigvals` of the map, are then
//! ±i). In four Euclidean dimensions each generator is the sum of two
//! commuting halves, each turning every plane through one angle; their orbits on the
//! three-sphere are the fibres of the Hopf fibration and of its mirror image.
//!
//! numga does all this inside one algebra, Cl(6,3), with nine directions. A gax `algebra!` of
//! nine dimensions takes minutes and gigabytes to compile, so this port declares each signature
//! as an algebra of its own, twelve of three to six dimensions with only the kinds it needs
//! (vectors, bivectors, even and odd), and one macro runs the same code on each. The maps on
//! bivectors and spinors are built from their images of the basis blades (`from_images`)
//! rather than by products with open slots: in six dimensions those take minutes each to
//! compile.
//!
//! The animation: the table of signatures, and below it the points of the three-sphere carried
//! along the two isoclinic flows (circles filling nested tori, linked in opposite senses) and
//! along a rotation turning xy twice while zw turns three times (trefoil knots), seen through
//! the stereographic projection.

use gax::ApproxEq;

use gax_numga_examples::scene3::panel3;
use gax_numga_examples::signal::wave;
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Light, Marker, ORIGIN3, Point2, Rect, backdrop, caption,
    colormap, palette, run,
};
use std::sync::OnceLock;

gax::algebra! {
    algebra cl30 "Cl(3,0): 3 directions squaring to plus one, 0 to minus one.";
    basis e1 = 1, e2 = 1, e3 = 1;
    kind Scalar = [1];
    kind Vector = [e1, e2, e3];
    kind Bivector = [e12, e13, e23];
    kind Even = [1, e12, e13, e23];
    kind Odd = [e1, e2, e3, e123];
}

gax::algebra! {
    algebra cl21 "Cl(2,1): 2 directions squaring to plus one, 1 to minus one.";
    basis e1 = 1, e2 = 1, et = -1;
    kind Scalar = [1];
    kind Vector = [e1, e2, et];
    kind Bivector = [e12, e1t, e2t];
    kind Even = [1, e12, e1t, e2t];
    kind Odd = [e1, e2, et, e12t];
}

gax::algebra! {
    algebra cl40 "Cl(4,0): 4 directions squaring to plus one, 0 to minus one.";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1;
    kind Scalar = [1];
    kind Vector = [e1, e2, e3, e4];
    kind Bivector = [e12, e13, e14, e23, e24, e34];
    kind Even = [1, e12, e13, e14, e23, e24, e34, e1234];
    kind Odd = [e1, e2, e3, e4, e123, e124, e134, e234];
}

gax::algebra! {
    algebra cl31 "Cl(3,1): 3 directions squaring to plus one, 1 to minus one.";
    basis e1 = 1, e2 = 1, e3 = 1, et = -1;
    kind Scalar = [1];
    kind Vector = [e1, e2, e3, et];
    kind Bivector = [e12, e13, e1t, e23, e2t, e3t];
    kind Even = [1, e12, e13, e1t, e23, e2t, e3t, e123t];
    kind Odd = [e1, e2, e3, et, e123, e12t, e13t, e23t];
}

gax::algebra! {
    algebra cl22 "Cl(2,2): 2 directions squaring to plus one, 2 to minus one.";
    basis e1 = 1, e2 = 1, et = -1, es = -1;
    kind Scalar = [1];
    kind Vector = [e1, e2, et, es];
    kind Bivector = [e12, e1t, e1s, e2t, e2s, ets];
    kind Even = [1, e12, e1t, e1s, e2t, e2s, ets, e12ts];
    kind Odd = [e1, e2, et, es, e12t, e12s, e1ts, e2ts];
}

gax::algebra! {
    algebra cl50 "Cl(5,0): 5 directions squaring to plus one, 0 to minus one.";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = 1;
    kind Scalar = [1];
    kind Vector = [e1, e2, e3, e4, e5];
    kind Bivector = [e12, e13, e14, e15, e23, e24, e25, e34, e35, e45];
    kind Even = [1, e12, e13, e14, e15, e23, e24, e25, e34, e35, e45, e1234, e1235, e1245, e1345, e2345];
    kind Odd = [e1, e2, e3, e4, e5, e123, e124, e125, e134, e135, e145, e234, e235, e245, e345, e12345];
}

gax::algebra! {
    algebra cl41 "Cl(4,1): 4 directions squaring to plus one, 1 to minus one.";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, et = -1;
    kind Scalar = [1];
    kind Vector = [e1, e2, e3, e4, et];
    kind Bivector = [e12, e13, e14, e1t, e23, e24, e2t, e34, e3t, e4t];
    kind Even = [1, e12, e13, e14, e1t, e23, e24, e2t, e34, e3t, e4t, e1234, e123t, e124t, e134t, e234t];
    kind Odd = [e1, e2, e3, e4, et, e123, e124, e12t, e134, e13t, e14t, e234, e23t, e24t, e34t, e1234t];
}

gax::algebra! {
    algebra cl32 "Cl(3,2): 3 directions squaring to plus one, 2 to minus one.";
    basis e1 = 1, e2 = 1, e3 = 1, et = -1, es = -1;
    kind Scalar = [1];
    kind Vector = [e1, e2, e3, et, es];
    kind Bivector = [e12, e13, e1t, e1s, e23, e2t, e2s, e3t, e3s, ets];
    kind Even = [1, e12, e13, e1t, e1s, e23, e2t, e2s, e3t, e3s, ets, e123t, e123s, e12ts, e13ts, e23ts];
    kind Odd = [e1, e2, e3, et, es, e123, e12t, e12s, e13t, e13s, e1ts, e23t, e23s, e2ts, e3ts, e123ts];
}

gax::algebra! {
    algebra cl60 "Cl(6,0): 6 directions squaring to plus one, 0 to minus one.";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = 1, e6 = 1;
    kind Scalar = [1];
    kind Vector = [e1, e2, e3, e4, e5, e6];
    kind Bivector = [e12, e13, e14, e15, e16, e23, e24, e25, e26, e34, e35, e36, e45, e46, e56];
    kind Even = [1, e12, e13, e14, e15, e16, e23, e24, e25, e26, e34, e35, e36, e45, e46, e56, e1234, e1235, e1236, e1245, e1246, e1256, e1345, e1346, e1356, e1456, e2345, e2346, e2356, e2456, e3456, e123456];
    kind Odd = [e1, e2, e3, e4, e5, e6, e123, e124, e125, e126, e134, e135, e136, e145, e146, e156, e234, e235, e236, e245, e246, e256, e345, e346, e356, e456, e12345, e12346, e12356, e12456, e13456, e23456];
}

gax::algebra! {
    algebra cl51 "Cl(5,1): 5 directions squaring to plus one, 1 to minus one.";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = 1, et = -1;
    kind Scalar = [1];
    kind Vector = [e1, e2, e3, e4, e5, et];
    kind Bivector = [e12, e13, e14, e15, e1t, e23, e24, e25, e2t, e34, e35, e3t, e45, e4t, e5t];
    kind Even = [1, e12, e13, e14, e15, e1t, e23, e24, e25, e2t, e34, e35, e3t, e45, e4t, e5t, e1234, e1235, e123t, e1245, e124t, e125t, e1345, e134t, e135t, e145t, e2345, e234t, e235t, e245t, e345t, e12345t];
    kind Odd = [e1, e2, e3, e4, e5, et, e123, e124, e125, e12t, e134, e135, e13t, e145, e14t, e15t, e234, e235, e23t, e245, e24t, e25t, e345, e34t, e35t, e45t, e12345, e1234t, e1235t, e1245t, e1345t, e2345t];
}

gax::algebra! {
    algebra cl42 "Cl(4,2): 4 directions squaring to plus one, 2 to minus one.";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, et = -1, es = -1;
    kind Scalar = [1];
    kind Vector = [e1, e2, e3, e4, et, es];
    kind Bivector = [e12, e13, e14, e1t, e1s, e23, e24, e2t, e2s, e34, e3t, e3s, e4t, e4s, ets];
    kind Even = [1, e12, e13, e14, e1t, e1s, e23, e24, e2t, e2s, e34, e3t, e3s, e4t, e4s, ets, e1234, e123t, e123s, e124t, e124s, e12ts, e134t, e134s, e13ts, e14ts, e234t, e234s, e23ts, e24ts, e34ts, e1234ts];
    kind Odd = [e1, e2, e3, e4, et, es, e123, e124, e12t, e12s, e134, e13t, e13s, e14t, e14s, e1ts, e234, e23t, e23s, e24t, e24s, e2ts, e34t, e34s, e3ts, e4ts, e1234t, e1234s, e123ts, e124ts, e134ts, e234ts];
}

gax::algebra! {
    algebra cl33 "Cl(3,3): 3 directions squaring to plus one, 3 to minus one.";
    basis e1 = 1, e2 = 1, e3 = 1, et = -1, es = -1, er = -1;
    kind Scalar = [1];
    kind Vector = [e1, e2, e3, et, es, er];
    kind Bivector = [e12, e13, e1t, e1s, e1r, e23, e2t, e2s, e2r, e3t, e3s, e3r, ets, etr, esr];
    kind Even = [1, e12, e13, e1t, e1s, e1r, e23, e2t, e2s, e2r, e3t, e3s, e3r, ets, etr, esr, e123t, e123s, e123r, e12ts, e12tr, e12sr, e13ts, e13tr, e13sr, e1tsr, e23ts, e23tr, e23sr, e2tsr, e3tsr, e123tsr];
    kind Odd = [e1, e2, e3, et, es, er, e123, e12t, e12s, e12r, e13t, e13s, e13r, e1ts, e1tr, e1sr, e23t, e23s, e23r, e2ts, e2tr, e2sr, e3ts, e3tr, e3sr, etsr, e123ts, e123tr, e123sr, e12tsr, e13tsr, e23tsr];
}

/// The coefficients of basis element `i` of a kind with `N` of them.
fn basis<const N: usize>(i: usize) -> [f64; N] {
    core::array::from_fn(|j| if j == i { 1.0 } else { 0.0 })
}

/// The groups: one function per question, run on each signature's algebra by a macro, since a
/// function generic over the algebra would have to restate every product in its bounds.
mod spin {
    use super::*;
    use gax::Complex;
    use gax_numga_examples::rng::{Draw, rng};

    /// One signature: its numbers, and the maps they come from.
    #[derive(Clone, Debug)]
    pub struct Row {
        pub p: usize,
        pub q: usize,
        /// The number of planes, the bivectors' dimension.
        pub planes: usize,
        /// Whether the invariant form is exactly `2 (n - 2)` times the inner product.
        pub form_is_inner: bool,
        /// The invariant form over the inner product, on a random bivector.
        pub factor: f64,
        /// The eigenvalues of the involution on the bivectors.
        pub signs: Vec<Complex<f64>>,
        /// The square of the pseudoscalar.
        pub square: f64,
        /// In even dimensions, the eigenvalues of the pseudoscalar acting on the spinors.
        pub action: Option<Vec<Complex<f64>>>,
    }

    impl Row {
        /// The involution's fixed planes (rotations) and negated planes (boosts).
        pub fn counts(&self) -> (usize, usize) {
            let plus = self.signs.iter().filter(|s| s.re > 0.0).count();
            (plus, self.signs.len() - plus)
        }

        /// What the pseudoscalar does to the spinors.
        pub fn spinors(&self) -> &'static str {
            match &self.action {
                None => "ONE PIECE",
                Some(values) if values.iter().all(|v| v.im.abs() < 1e-9) => "TWO HALVES",
                Some(_) => "COMPLEX",
            }
        }
    }

    /// Basis vector `i` of an algebra.
    macro_rules! axis {
        ($m:ident, $i:expr) => {
            $m::Vector::<(), f64>::from_coeffs(basis($i))
        };
    }

    /// The product of basis vectors `0..k` of an algebra, as an even and an odd part (one of
    /// them zero): pairs of vectors multiply to even values, and a last vector left over makes
    /// the product odd.
    macro_rules! blade_product {
        ($m:ident, $k:expr) => {{
            use $m::{Even, Odd, Scalar};
            let k: usize = $k;
            let mut even: Even<(), f64> = Scalar::<(), f64>::from_coeffs([1.0]).cast::<Even>();
            for i in 0..k / 2 {
                even = even * (axis!($m, 2 * i) * axis!($m, 2 * i + 1));
            }
            let odd: Odd<(), f64> = if k % 2 == 1 {
                even * axis!($m, k - 1)
            } else {
                Odd::zero()
            };
            if k % 2 == 1 {
                (Even::zero(), odd)
            } else {
                (even, odd)
            }
        }};
    }

    /// The involution on the bivectors: the sandwich with the product of the positive
    /// directions, `P B ~P`, as a map built from its images of the basis bivectors (the map
    /// with the bivector open, `P * Bivector::slot() * ~P`, is the same map; built by products
    /// with open slots, six-dimensional maps take minutes to compile).
    macro_rules! involution {
        ($m:ident, $p:expr) => {{
            use $m::Bivector;
            let (even, odd) = blade_product!($m, $p);
            Bivector::<(Bivector,), f64>::from_images(core::array::from_fn(|j| {
                let b = Bivector::<(), f64>::from_coeffs(basis(j));
                ((even * b) * even.reverse()).cast::<Bivector>()
                    + ((odd * b) * odd.reverse()).cast::<Bivector>()
            }))
        }};
    }
    #[cfg(test)]
    pub(crate) use {axis, blade_product, involution};

    /// The pseudoscalar of an even number of directions acting on the spinors by the geometric
    /// product from the left, as a map built from its images of the basis spinors.
    macro_rules! action {
        ($m:ident, $n:expr) => {{
            use $m::Even;
            let (pseudoscalar, _) = blade_product!($m, $n);
            Even::<(Even,), f64>::from_images(core::array::from_fn(|j| {
                pseudoscalar * Even::<(), f64>::from_coeffs(basis(j))
            }))
        }};
    }
    #[cfg(test)]
    pub(crate) use action;

    /// The row of a signature.
    macro_rules! row {
        ($m:ident, $p:expr, $q:expr, $rng:expr) => {{
            use $m::Bivector;
            let n: usize = $p + $q;
            // The invariant form on basis bivectors: the trace of `ad_a ad_b`, each `ad` the
            // commutator with the other bivector open. (The form with both bivectors open, the
            // double commutator with three open slots traced, takes too long to compile in
            // six dimensions.)
            let planes: Vec<Bivector<(), f64>> = (0..n * (n - 1) / 2)
                .map(|i| Bivector::from_coeffs(basis(i)))
                .collect();
            let ad: Vec<Bivector<(Bivector,), f64>> = planes
                .iter()
                .map(|a| {
                    Bivector::<(Bivector,), f64>::from_images(core::array::from_fn(|j| {
                        a.commutator(planes[j])
                    }))
                })
                .collect();
            let form: Vec<Vec<f64>> = ad
                .iter()
                .map(|a| ad.iter().map(|b| a.of(*b).trace()).collect())
                .collect();
            let inner: Vec<Vec<f64>> = planes
                .iter()
                .map(|a| planes.iter().map(|b| (*a | *b).s()).collect())
                .collect();
            let sample: Vec<f64> = (0..planes.len()).map(|_| $rng.normal()).collect();
            let pair = |m: &Vec<Vec<f64>>| -> f64 {
                (0..sample.len())
                    .map(|i| {
                        (0..sample.len())
                            .map(|j| sample[i] * m[i][j] * sample[j])
                            .sum::<f64>()
                    })
                    .sum()
            };
            let factor = pair(&form) / pair(&inner);
            let scale = 2.0 * (n as f64 - 2.0);
            let form_is_inner = form
                .iter()
                .flatten()
                .zip(inner.iter().flatten())
                .all(|(f, i)| *f == scale * *i);
            let signs = involution!($m, $p).eigvals().to_vec();
            let (even, odd) = blade_product!($m, n);
            let square = (even * even).s() + (odd * odd).s();
            let action = (n % 2 == 0).then(|| action!($m, n).eigvals().to_vec());
            Row {
                p: $p,
                q: $q,
                planes: n * (n - 1) / 2,
                form_is_inner,
                factor,
                signs,
                square,
                action,
            }
        }};
    }

    /// The signatures, in numga's order.
    pub fn zoo(seed: u64) -> Vec<Row> {
        let mut rng = rng(seed);
        vec![
            row!(cl30, 3, 0, rng),
            row!(cl21, 2, 1, rng),
            row!(cl40, 4, 0, rng),
            row!(cl31, 3, 1, rng),
            row!(cl22, 2, 2, rng),
            row!(cl50, 5, 0, rng),
            row!(cl41, 4, 1, rng),
            row!(cl32, 3, 2, rng),
            row!(cl60, 6, 0, rng),
            row!(cl51, 5, 1, rng),
            row!(cl42, 4, 2, rng),
            row!(cl33, 3, 3, rng),
        ]
    }

    /// The table, computed once.
    pub fn table() -> &'static [Row] {
        static TABLE: OnceLock<Vec<Row>> = OnceLock::new();
        TABLE.get_or_init(|| zoo(0))
    }

    /// Four Euclidean dimensions: rotors, the isoclinic split and the flows.
    pub mod four {
        use super::*;
        pub use cl40::{Bivector, Even, Vector};

        pub type B = Bivector<(), f64>;
        pub type V = Vector<(), f64>;
        pub type E = Even<(), f64>;
        /// A point of the space the three-sphere is projected into.
        pub type Space = gax::vga3d::Vector<(), f64>;

        /// The bivector with one plane: `e_ij` by its basis indices.
        pub fn plane(i: usize, j: usize) -> B {
            let names = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
            B::from_coeffs(names.map(|n| if n == (i, j) { 1.0 } else { 0.0 }))
        }

        /// The basis vector `e_i`.
        pub fn axis(i: usize) -> V {
            V::from_coeffs(basis(i))
        }

        /// The pseudoscalar, as an even multivector.
        pub fn pseudoscalar() -> E {
            blade_product!(cl40, 4).0
        }

        /// The scalar one, as an even multivector.
        pub fn one() -> E {
            cl40::Scalar::<(), f64>::from_coeffs([1.0]).cast::<Even>()
        }

        /// A four-dimensional generator as the sum of two commuting generators,
        /// `(1 + I) / 2` and `(1 - I) / 2` times it: bivectors again, one turning each plane
        /// along with its dual plane, the other against it. Their exponentials multiply to the
        /// generator's rotor.
        pub fn isoclinic(generator: B) -> (B, B) {
            let i = pseudoscalar();
            let plus = (one() + i).gp(0.5);
            let minus = (one() - i).gp(0.5);
            (
                (plus * generator).cast::<Bivector>(),
                (minus * generator).cast::<Bivector>(),
            )
        }

        /// The sandwich of a vector with a rotor.
        pub fn turn(r: E, v: V) -> V {
            ((r * v) * r.reverse()).cast::<Vector>()
        }

        /// The sandwich with a rotor, as a map on vectors.
        pub fn turn_map(r: E) -> Vector<(Vector,), f64> {
            Vector::<(Vector,), f64>::from_images(core::array::from_fn(|j| turn(r, axis(j))))
        }

        /// A unit vector of the four directions, projected from `-e4` into the space of the
        /// first three: `(v - e4 (v | e4)) / (1 + v | e4)`, as a vector of VGA3D.
        pub fn stereographic(v: V) -> Space {
            let w = axis(3);
            let h = (v | w).s();
            let s = (v - w.gp(h)).gp(1.0 / (1.0 + h));
            Space::new(s.c[0], s.c[1], s.c[2])
        }

        /// A random rotor of the four Euclidean directions and its two isoclinic factors, with
        /// the halves of its generator.
        pub fn split(seed: u64) -> (E, E, E, B, B) {
            let mut rng = rng(seed);
            let generator = B::from_coeffs(core::array::from_fn(|_| rng.normal()));
            let (along, against) = isoclinic(generator);
            (
                generator.exp().into_inner(),
                along.exp().into_inner(),
                against.exp().into_inner(),
                along,
                against,
            )
        }

        /// The orbits of points of the three-sphere under the two isoclinic flows, turning xy
        /// together with zw one way and the other, and under a rotation turning xy twice while
        /// zw turns three times (the two flows combined at different rates,
        /// `5 against - along`), from one start at each height; all projected into space.
        pub struct Flows {
            pub left: Vec<Vec<Space>>,
            pub right: Vec<Vec<Space>>,
            pub knotted: Vec<Vec<Space>>,
            /// Height and azimuth index of each orbit of the flows.
            pub index: Vec<(usize, usize)>,
            pub along: B,
            pub against: B,
        }

        pub fn flows(azimuths: usize, heights: &[f64], count: usize) -> Flows {
            let (along, against) = isoclinic(plane(0, 1));
            let half = |b: B, a: f64| b.gp(a).exp().into_inner();
            // x turned up toward z by each height, then about z by each azimuth.
            let start = |h: f64, phi: f64| {
                turn(
                    half(plane(0, 1), -phi / 2.0) * half(plane(0, 2), -h / 2.0),
                    axis(0),
                )
            };
            let orbit = |generator: B, s: V| -> Vec<Space> {
                // One full turn of every plane: the sandwich turns by twice the rotor's angle.
                (0..=count)
                    .map(|k| {
                        let angle = core::f64::consts::PI * k as f64 / count as f64;
                        stereographic(turn(half(generator, 2.0 * angle), s))
                    })
                    .collect()
            };
            let mut flows = Flows {
                left: Vec::new(),
                right: Vec::new(),
                knotted: Vec::new(),
                index: Vec::new(),
                along,
                against,
            };
            let combined = plane(0, 1).gp(2.0) + plane(2, 3).gp(3.0);
            for (i, h) in heights.iter().enumerate() {
                for k in 0..azimuths {
                    let phi = core::f64::consts::TAU * k as f64 / azimuths as f64;
                    let s = start(*h, phi);
                    flows.left.push(orbit(along, s));
                    flows.right.push(orbit(against, s));
                    flows.index.push((i, k));
                }
                flows.knotted.push(orbit(combined, start(*h, 0.0)));
            }
            flows
        }

        /// The flows of the animation, computed once.
        pub fn cached() -> &'static Flows {
            static FLOWS: OnceLock<Flows> = OnceLock::new();
            FLOWS.get_or_init(|| flows(12, &[0.25, 0.6, 1.0], 240))
        }
    }
}

use spin::*;

const SECONDS: f32 = 8.0;

/// A colour per orbit: the start's position around the xy plane as the hue, whitened a little,
/// fainter the higher it starts.
fn orbit_colour(height: usize, heights: usize, azimuth: usize, azimuths: usize) -> Light {
    let hue = colormap::hsv(azimuth as f32 / azimuths as f32);
    let soft = hue.whitened(0.2);
    soft.faded(0.95 - 0.4 * height as f32 / (heights.max(2) - 1) as f32)
}

/// The point of space a projected vector reaches from the origin.
fn at(v: four::Space) -> gax::pga3d::Point<(), f64> {
    gax::pga3d::Point::xyz(0.0, 0.0, 0.0) + gax::pga3d::Point::direction(v.e1(), v.e2(), v.e3())
}

/// The checks of the four-dimensional scenes, live: the isoclinic split of a random rotor, the
/// combined rotation as the two flows, and the linking of two orbits of each flow.
struct Checks {
    split: f64,
    angles: [f64; 2],
    combined: f64,
    links: [f64; 2],
}

fn checks() -> &'static Checks {
    static CHECKS: OnceLock<Checks> = OnceLock::new();
    CHECKS.get_or_init(|| {
        use four::*;
        let (rotor, left, right, _, _) = split(1);
        // The spread of the turning angles of each factor's eigenvalues.
        let angles = [left, right].map(|factor| {
            let a: Vec<f64> = turn_map(factor)
                .eigvals()
                .iter()
                .map(|z| z.arg().abs())
                .collect();
            let lo = a.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = a.iter().copied().fold(0.0, f64::max);
            hi - lo
        });
        let f = cached();
        let combined = plane(0, 1).gp(2.0) + plane(2, 3).gp(3.0);
        let last = f.left.len() - 12 + 4;
        Checks {
            split: (left * right).max_abs_diff(&rotor),
            angles,
            combined: combined.max_abs_diff(&(f.against.gp(5.0) - f.along)),
            links: [
                gax_numga_examples::measure::linking(&f.left[0], &f.left[last]),
                gax_numga_examples::measure::linking(&f.right[0], &f.right[last]),
            ],
        }
    })
}

/// The table of algebras' header: its width sets the table's.
const HEADER: &str = "(P,Q) PLANES ROTATIONS BOOSTS FORM/INNER I*I SPINORS";

/// The table of algebras, its header's baseline starting at `at`, one row under another.
fn draw_table(c: &mut Canvas, at: Point2, size: f32) {
    c.text(HEADER, at, size, palette::ink(), Align::Left);
    let down = Point2::direction(0.0, size * 1.4);
    for (k, row) in table().iter().enumerate() {
        let (rotations, boosts) = row.counts();
        let line = format!(
            "({},{}) {:>6} {:>9} {:>6} {:>10.1} {:>+3.0} {}",
            row.p,
            row.q,
            row.planes,
            rotations,
            boosts,
            row.factor,
            row.square,
            row.spinors()
        );
        let colour = if boosts == 0 {
            palette::sky()
        } else {
            palette::grid()
        };
        let colour = colour.mix_light(palette::ink(), 0.35);
        let row_at = at + down.gp((k + 1) as f32);
        c.text(&line, row_at, size, colour, Align::Left);
    }
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let (up, down) = (Point2::direction(0.0, -1.0), Point2::direction(0.0, 1.0));
    let phase = t / SECONDS;
    // Lengths in pixels at 960 by 540, scaled with the canvas.
    let unit = c.unit();
    let size = 8.7 * unit;
    // The table at the right, under the caption: as wide as its header and the three letters
    // by which its last column's entries outrun the header's.
    let table_width = gax_numga_examples::font::width(HEADER, size) + 3.0 * size;
    draw_table(c, Point2::xy(0.985 * w - table_width, 0.14 * h), size);

    let flows = four::cached();
    let families = [&flows.left, &flows.right, &flows.knotted];
    let extent = families
        .iter()
        .flat_map(|f| f.iter().flatten())
        .flat_map(|p| p.c)
        .fold(0.0f64, |m, x| m.max(x.abs()));
    let length = flows.left[0].len() - 1;
    let step = ((phase * length as f32) as usize).min(length - 1);
    let titles = [
        "ALONG: XY WITH ZW",
        "AGAINST: XY AGAINST ZW",
        "XY TWICE, ZW THREE TIMES",
    ];
    // The three flows side by side below the table.
    let below = Rect::new(0.0, (0.42 * h).floor(), w, h);
    for (k, family) in families.iter().enumerate() {
        let rect = below.column(k, 3);
        let cam = Camera::orbit(
            rect.width() as usize,
            rect.height() as usize,
            ORIGIN3,
            (extent * 4.2) as f32,
            (-55.0f32).to_radians() + 0.5 * wave(phase * core::f32::consts::TAU),
            24.0f32.to_radians(),
            Lens::Perspective(0.5),
        );
        let knots = k == 2;
        panel3(c, rect, cam, |scene| {
            for (o, orbit) in family.iter().enumerate() {
                let (height, azimuth) = if knots { (o, 0) } else { flows.index[o] };
                let colour = orbit_colour(height, 3, azimuth, 12);
                let pts: Vec<_> = orbit.iter().map(|v| at(*v)).collect();
                let width = if knots { 1.2 } else { 0.6 };
                scene.polyline(&pts, width, colour.faded(0.8));
                // The points carried along: one per orbit, six riding along each knot.
                let riders = if knots { 6 } else { 1 };
                for r in 0..riders {
                    let now = (step + r * length / riders) % length;
                    scene.dot(pts[now], Marker::Dot, 7.0, colour);
                }
            }
        });
        let top_middle = rect.top_middle();
        let title = top_middle + down.gp(16.0 * unit);
        c.text(titles[k], title, 11.0 * unit, palette::ink(), Align::Center);
        if k < 2 {
            let bottom_middle = rect.bottom_middle();
            let text = format!("LINKING NUMBER {:+.2}", checks().links[k]);
            let note = bottom_middle + up.gp(12.0 * unit);
            c.text(&text, note, 10.0 * unit, palette::grid(), Align::Center);
        }
    }
    let exact = table().iter().all(|row| row.form_is_inner);
    let ch = checks();
    let lines = [
        if exact {
            "FORM = 2(N-2) INNER PRODUCT EXACTLY, EVERY ROW".to_string()
        } else {
            "FORM AND INNER PRODUCT DIFFER".to_string()
        },
        format!("SPIN(4) ROTOR = LEFT * RIGHT FACTOR: {:.0E}", ch.split),
        format!(
            "EACH FACTOR TURNS ALL PLANES ALIKE: {:.0E}",
            ch.angles[0].max(ch.angles[1])
        ),
        format!("2 XY + 3 ZW = 5 AGAINST - ALONG: {:.0E}", ch.combined),
    ];
    let first = Point2::xy(0.015 * w, 0.2 * h);
    for (k, line) in lines.iter().enumerate() {
        let line_at = first + down.gp(k as f32 * size * 1.6);
        c.text(line, line_at, size, palette::grid(), Align::Left);
    }
    caption(
        c,
        "SPIN GROUPS OF EVERY SIGNATURE UP TO 6D",
        "TWELVE ALGEBRAS CL(P,Q); BELOW: THE ISOCLINIC FLOWS OF SPIN(4)",
    );
}

fn main() {
    run(Anim::new("spin groups", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::spin::four::*;
    use super::*;
    use gax::Complex;

    /// numga's first test: in the signature (3, 1) the involution fixes the planes of rotations
    /// and negates the planes of boosts, and the pseudoscalar acts on the spinors as the complex
    /// unit.
    #[test]
    fn rotations_and_boosts_of_the_signature_3_1() {
        use super::cl31::Bivector;
        let involution = involution!(cl31, 3);
        // e12 is a rotation, e1t a boost (the third blade in the layout).
        let xy: Bivector<(), f64> = Bivector::from_coeffs([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let xt: Bivector<(), f64> = Bivector::from_coeffs([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        assert!(involution.of(xy).max_abs_diff(&xy) < 1e-12);
        assert!(involution.of(xt).max_abs_diff(&-xt) < 1e-12);
        for v in action!(cl31, 4).eigvals() {
            assert!((v.im.abs() - 1.0).abs() < 1e-12, "{v:?}");
        }
    }

    /// numga's `zoo` and `centres` checks: the invariant form is `2 (n - 2)` times the inner
    /// product, exactly; the involution fixes `p (p - 1) / 2 + q (q - 1) / 2` planes and negates
    /// `p q`; the pseudoscalar's eigenvalues on the spinors square to its square.
    #[test]
    fn the_table_passes_its_checks() {
        let rows = zoo(0);
        assert_eq!(rows.len(), 12);
        for row in &rows {
            let (p, q) = (row.p, row.q);
            assert!(row.form_is_inner, "({p}, {q})");
            let n = (p + q) as f64;
            assert!(
                (row.factor - 2.0 * (n - 2.0)).abs() < 1e-12,
                "{}",
                row.factor
            );
            assert_eq!(row.counts(), ((p * p - p) / 2 + (q * q - q) / 2, p * q));
            if let Some(values) = &row.action {
                for v in values {
                    assert!((*v * *v - Complex::real(row.square)).abs() < 1e-12, "{v:?}");
                }
            }
        }
        // The table renders a header and a line per signature.
        let text_rows = 1 + rows.len();
        assert_eq!(text_rows, 13);
    }

    /// numga's `split` checks: the halves commute, the factors multiply to the rotor, and each
    /// factor turns all its planes through one angle.
    #[test]
    fn a_rotor_splits_into_isoclinic_factors() {
        let (rotor, left, right, along, against) = split(1);
        assert!(along.commutator(against).max_abs_diff(&B::zero()) < 1e-12);
        assert!((left * right).max_abs_diff(&rotor) < 1e-8);
        for factor in [left, right] {
            let angles: Vec<f64> = turn_map(factor)
                .eigvals()
                .iter()
                .map(|z| z.arg().abs())
                .collect();
            for a in &angles {
                assert!((a - angles[0]).abs() < 1e-9, "{angles:?}");
            }
        }
    }

    /// numga's `flows` checks, at the test's sizes: turning xy twice while zw turns three times
    /// is the flow against run five times over and the flow along run once backward; every
    /// orbit closes, and two orbits of one flow link once, with opposite signs for the flows.
    #[test]
    fn the_flows_close_and_link() {
        let azimuths = 6;
        let f = flows(azimuths, &[0.3, 0.9], 120);
        let combined = plane(0, 1).gp(2.0) + plane(2, 3).gp(3.0);
        assert!(combined.max_abs_diff(&(f.against.gp(5.0) - f.along)) < 1e-12);
        for orbit in f.left.iter().chain(&f.knotted) {
            assert!(orbit[0].max_abs_diff(&orbit[orbit.len() - 1]) < 1e-5);
        }
        let last = f.left.len() - azimuths + azimuths / 3;
        let forward = gax_numga_examples::measure::linking(&f.left[0], &f.left[last]);
        let backward = gax_numga_examples::measure::linking(&f.right[0], &f.right[last]);
        assert!((forward.abs() - 1.0).abs() < 1e-2, "{forward}");
        assert!((backward.abs() - 1.0).abs() < 1e-2, "{backward}");
        assert!(forward.signum() == -backward.signum());
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
