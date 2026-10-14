//! numga's `math/extensor_representations`: the smallest faithful real representations of the
//! nondegenerate Clifford algebras through dimension six, as extensors.
//!
//! A multivector acts on a state by multiplication from the left. The states need not fill the
//! algebra: blades that square to plus one and commute give commuting projectors
//! `(1 + blade) / 2`, each halving the space, and their product `P` selects a left ideal, the
//! states `x P`, closed under every left multiplication. Products of the chosen blades divide
//! the blades into cosets; one representative blade per coset gives independent state
//! coordinates (numga's `spinor_layout`, ported here and checked against the declared state
//! kinds). A state embeds as `s P`, and reads back out as its representatives' coefficients
//! times the number of the projector's copies. A multivector's representation embeds a state,
//! multiplies it from the left and reads it back out: a map `State <- State`, linear in the
//! multivector. Where the algebra splits into two simple blocks (odd dimension, pseudoscalar
//! squaring to one), the choice keeps both, so neither disappears.
//!
//! The representations preserve products, `ρ(a b) = ρ(a) ρ(b)`, send one to the identity, and
//! send the blades to orthogonal signed permutations: they lose nothing. The readout is their
//! adjoint under the normalized trace pairing: the coefficient of a blade is the trace of the
//! map composed with the blade's representation, over the state's dimension, solved against
//! the scalar product (which supplies the blades' inverses). On any map of the states it is the
//! orthogonal projection onto the representations.
//!
//! numga does each signature in one runtime algebra; gax declares an algebra's kinds at compile
//! time, so each signature is its own algebra (`algebras.rs`), and one macro stamps out the code
//! for each. Maps are built from their images of the basis states (`from_images`) rather than
//! by products with open slots, which keeps six-dimensional algebras quick to compile. numga's
//! checks run on exact rationals; these run on floats, which represent every number involved
//! (signs, and powers of two) exactly, so the checks are still equalities.
//!
//! The animation steps through the signatures in the table of real state sizes, showing the
//! chosen projector blades and the generators' representations as signed permutations.

mod algebras;

use gax_numga_examples::{
    Align, Anim, Canvas, Light, Point2, Rect, backdrop, caption, palette, run,
};
use std::sync::OnceLock;

/// The blades of `n` directions as masks, in gax's order for the full multivector: by grade,
/// then by mask.
pub fn canonical_masks(n: usize) -> Vec<u32> {
    let mut masks: Vec<u32> = (0..1u32 << n).collect();
    masks.sort_by_key(|m| (m.count_ones(), *m));
    masks
}

/// numga's `spinor_layout`: commuting blades that square to plus one, taken in mask order while
/// they enlarge their span (skipping, where the algebra splits, a blade whose product with the
/// pseudoscalar is in the span, which would fix the central volume's sign and drop one block),
/// and the least blade of each coset of their span, sorted as the declared kinds are. `square`
/// gives a blade's square, `commute` whether two blades commute.
pub fn layout(
    n: usize,
    square: impl Fn(u32) -> f64,
    commute: impl Fn(u32, u32) -> bool,
) -> (Vec<u32>, Vec<u32>) {
    let pseudoscalar = (1u32 << n) - 1;
    let split = n % 2 == 1 && square(pseudoscalar) == 1.0;
    let (mut involutions, mut span): (Vec<u32>, Vec<u32>) = (vec![], vec![0]);
    for blade in 0..1u32 << n {
        let fits = !span.contains(&blade)
            && square(blade) == 1.0
            && involutions.iter().all(|o| commute(blade, *o))
            && !(split && span.contains(&(blade ^ pseudoscalar)));
        if fits {
            involutions.push(blade);
            span = span.iter().flat_map(|s| [*s, *s ^ blade]).collect();
        }
    }
    let mut representatives: Vec<u32> = (0..1u32 << n)
        .map(|blade| {
            span.iter()
                .map(|s| blade ^ s)
                .min()
                .expect("the span holds one")
        })
        .collect();
    representatives.sort_by_key(|m| (m.count_ones(), *m));
    representatives.dedup();
    (involutions, representatives)
}

/// The largest dimension in the table.
const LARGEST: usize = 6;

/// A blade's name, `1` or `e` and its directions.
fn blade_name(mask: u32) -> String {
    if mask == 0 {
        return "1".into();
    }
    let directions: String = (0..LARGEST)
        .filter(|i| mask >> i & 1 == 1)
        .map(|i| char::from(b'1' + i as u8))
        .collect();
    format!("E{directions}")
}

/// One signature's representation, for drawing: its signature, its number of blades and of
/// state components, the projector's blades, and each direction's representation as rows of
/// numbers.
pub struct Row {
    pub p: usize,
    pub q: usize,
    pub blades: usize,
    pub state: usize,
    pub involutions: Vec<u32>,
    pub generators: Vec<Vec<Vec<f64>>>,
}

/// The representation in one algebra: its full kind and its state kind (the full multivector
/// where the states fill it, the scalars for Cl(0,0)).
macro_rules! signature {
    ($name:ident = $algebra:ident, $p:expr, $q:expr, $full:ident, $state:ident) => {
        pub mod $name {
            use super::algebras::$algebra::{Scalar, $full as Full, $state as State};
            use super::canonical_masks;

            pub type F = Full<(), f64>;
            pub type S = State<(), f64>;
            /// A map on the states.
            pub type Rep = State<(State,), f64>;
            pub const P: usize = $p;
            pub const Q: usize = $q;
            pub const N: usize = $p + $q;

            /// The blade of a mask, with coefficient one.
            pub fn blade(mask: u32) -> F {
                let index = canonical_masks(N)
                    .iter()
                    .position(|m| *m == mask)
                    .expect("a blade");
                Full::from_coeffs(core::array::from_fn(|i| if i == index { 1.0 } else { 0.0 }))
            }

            /// The geometric product, kept out of line: in six dimensions it is four thousand
            /// terms, which inlining would compile again at every call.
            #[inline(never)]
            pub fn product(a: F, b: F) -> F {
                a * b
            }

            /// The basis state `j`.
            pub fn state(j: usize) -> S {
                State::from_coeffs(core::array::from_fn(|i| if i == j { 1.0 } else { 0.0 }))
            }

            /// The number of state components.
            pub fn dimension() -> usize {
                S::zero().c.len()
            }

            /// numga's layout in this algebra, from its blades' products.
            pub fn layout() -> (Vec<u32>, Vec<u32>) {
                let square = |m: u32| product(blade(m), blade(m)).s();
                let commute =
                    |a: u32, b: u32| product(blade(a), blade(b)) == product(blade(b), blade(a));
                super::layout(N, square, commute)
            }

            /// The blades of the declared state kind, as masks.
            pub fn state_masks() -> Vec<u32> {
                let masks = canonical_masks(N);
                (0..dimension())
                    .map(|j| {
                        let embedded = state(j).cast::<Full>();
                        masks[embedded.c.iter().position(|c| *c != 0.0).expect("a blade")]
                    })
                    .collect()
            }

            /// The representations: the projector's blades, and each blade's representation.
            pub struct Representations {
                pub involutions: Vec<u32>,
                pub blades: Vec<Rep>,
            }

            /// The representations: embed a state into the ideal, `s P`; multiply from the
            /// left; read the state back out, its coefficients times the projector's copies.
            pub fn representations() -> Representations {
                let (involutions, _) = layout();
                let one = blade(0);
                let projector = involutions
                    .iter()
                    .fold(one, |p, m| product(p, (one + blade(*m)).gp(0.5)));
                let copies = (1u32 << involutions.len()) as f64;
                let embed = |s: S| product(s.cast::<Full>(), projector);
                let readout = |x: F| x.cast::<State>().gp(copies);
                let blades = canonical_masks(N)
                    .into_iter()
                    .map(|m| {
                        Rep::from_images(core::array::from_fn(|j| {
                            readout(product(blade(m), embed(state(j))))
                        }))
                    })
                    .collect();
                Representations {
                    involutions,
                    blades,
                }
            }

            impl Representations {
                /// A multivector's representation: its blades' representations, weighted.
                pub fn of(&self, x: F) -> Rep {
                    self.blades
                        .iter()
                        .zip(x.c)
                        .fold(Rep::zero(), |sum, (r, c)| sum + r.gp(c))
                }

                /// The multivector a map on the states reads out as: the form taking a
                /// multivector to the trace of its representation composed with the map, over
                /// the state's dimension, solved against the scalar product.
                pub fn readout(&self, map: Rep) -> F {
                    let d = dimension() as f64;
                    let form: Scalar<(Full,), f64> =
                        Scalar::<(Full,), f64>::from_images(core::array::from_fn(|i| {
                            Scalar::new(self.blades[i].of(map).trace() / d)
                        }));
                    let pairing = Full::<(), f64>::slot().scalar_product(Full::<(), f64>::slot());
                    pairing.solve(form)
                }
            }

            /// The row of this signature, for drawing.
            pub fn row() -> super::Row {
                let reps = representations();
                let generators = (0..N)
                    .map(|i| {
                        let r = reps.of(blade(1 << i));
                        (0..dimension())
                            .map(|row| {
                                (0..dimension())
                                    .map(|col| r.of(state(col)).c[row])
                                    .collect()
                            })
                            .collect()
                    })
                    .collect();
                super::Row {
                    p: P,
                    q: Q,
                    blades: 1 << N,
                    state: dimension(),
                    involutions: reps.involutions,
                    generators,
                }
            }
        }
    };
}

/// Every signature, with the real state size numga's test expects.
macro_rules! signatures {
    ($(($name:ident = $algebra:ident, $p:expr, $q:expr, $full:ident, $state:ident, $size:expr)),* $(,)?) => {
        $(signature!($name = $algebra, $p, $q, $full, $state);)*

        /// The rows of the table, by dimension, then from all plus to all minus.
        pub fn rows() -> Vec<Row> {
            vec![$($name::row()),*]
        }

        #[cfg(test)]
        mod every_signature {
            use super::*;

            /// numga's checks on each signature: the port of the layout chooses the declared
            /// states, of numga's size; the representations multiply as the algebra does, send
            /// one to the identity, and send the blades to orthogonal signed permutations
            /// (`C Cᵀ = d I` over their coefficients: none of the algebra is lost, including
            /// either split block).
            #[test]
            fn smallest_faithful_real_representation() {
                $(check!($name, $size);)*
            }
        }
    };
}

/// numga's checks on one signature's representations.
#[cfg(test)]
macro_rules! check {
    ($name:ident, $size:expr) => {{
        use $name::*;
        let (_, representatives) = layout();
        assert_eq!(representatives, state_masks(), "Cl({P},{Q})");
        assert_eq!(dimension(), $size, "Cl({P},{Q})");
        let reps = representations();
        let masks = canonical_masks(N);
        for (i, a) in masks.iter().enumerate() {
            for (j, b) in masks.iter().enumerate() {
                let composed = reps.blades[i].of(reps.blades[j]);
                assert_eq!(
                    composed,
                    reps.of(product(blade(*a), blade(*b))),
                    "Cl({P},{Q}) {a} {b}"
                );
                let frobenius: f64 = (0..dimension())
                    .map(|k| {
                        let (x, y) = (reps.blades[i].of(state(k)), reps.blades[j].of(state(k)));
                        x.c.iter().zip(y.c).map(|(u, v)| u * v).sum::<f64>()
                    })
                    .sum();
                let expected = if i == j { dimension() as f64 } else { 0.0 };
                assert_eq!(frobenius, expected, "Cl({P},{Q}) {a} {b}");
            }
        }
        assert_eq!(
            reps.of(blade(0)),
            Rep::from_images(core::array::from_fn(state))
        );
    }};
}

signatures! {
    (cl00 = cl10, 0, 0, Scalar, Scalar, 1),
    (cl10 = cl10, 1, 0, Multivector, Multivector, 2),
    (cl01 = cl01, 0, 1, Multivector, Multivector, 2),
    (cl20 = cl20, 2, 0, Multivector, Spinor, 2),
    (cl11 = cl11, 1, 1, Multivector, Spinor, 2),
    (cl02 = cl02, 0, 2, Multivector, Multivector, 4),
    (cl30 = cl30, 3, 0, Multivector, Spinor, 4),
    (cl21 = cl21, 2, 1, Multivector, Spinor, 4),
    (cl12 = cl12, 1, 2, Multivector, Spinor, 4),
    (cl03 = cl03, 0, 3, Multivector, Multivector, 8),
    (cl40 = cl40, 4, 0, Multivector, Spinor, 8),
    (cl31 = cl31, 3, 1, Multivector, Spinor, 4),
    (cl22 = cl22, 2, 2, Multivector, Spinor, 4),
    (cl13 = cl13, 1, 3, Multivector, Spinor, 8),
    (cl04 = cl04, 0, 4, Multivector, Spinor, 8),
    (cl50 = cl50, 5, 0, Multivector, Spinor, 16),
    (cl41 = cl41, 4, 1, Multivector, Spinor, 8),
    (cl32 = cl32, 3, 2, Multivector, Spinor, 8),
    (cl23 = cl23, 2, 3, Multivector, Spinor, 8),
    (cl14 = cl14, 1, 4, Multivector, Spinor, 16),
    (cl05 = cl05, 0, 5, Multivector, Spinor, 8),
    (cl60 = cl60, 6, 0, Multivector, Spinor, 16),
    (cl51 = cl51, 5, 1, Multivector, Spinor, 16),
    (cl42 = cl42, 4, 2, Multivector, Spinor, 8),
    (cl33 = cl33, 3, 3, Multivector, Spinor, 8),
    (cl24 = cl24, 2, 4, Multivector, Spinor, 16),
    (cl15 = cl15, 1, 5, Multivector, Spinor, 16),
    (cl06 = cl06, 0, 6, Multivector, Spinor, 8),
}

const SECONDS_PER_ROW: f32 = 0.9;

/// The rows, computed once.
fn table() -> &'static [Row] {
    static TABLE: OnceLock<Vec<Row>> = OnceLock::new();
    TABLE.get_or_init(rows)
}

/// A filled square, `side` wide, its top left corner at `corner`.
fn square(c: &mut Canvas, corner: Point2, side: f32, colour: Light) {
    let (across, down) = (Point2::direction(side, 0.0), Point2::direction(0.0, side));
    c.fill(
        &[
            corner,
            corner + across,
            corner + across + down,
            corner + down,
        ],
        colour,
        1.0,
    );
}

/// A cell's colour by its state size: brighter the more components.
fn size_colour(state: usize) -> Light {
    let level = (state as f32).log2() / 4.0;
    palette::grid().mix_light(palette::sky(), level)
}

/// The table of signatures in `rect`: a row per dimension, from all plus on the left to all
/// minus on the right, each cell its real state size; the current one lit.
fn table_panel(c: &mut Canvas, rect: Rect, current: usize) {
    // A row per dimension, from zero to the largest.
    let rows = (LARGEST + 1) as f32;
    let cell = (rect.height() / (rows + 1.0)).min(rect.width() / rows);
    for (k, row) in table().iter().enumerate() {
        let n = row.p + row.q;
        let left = rect.lo
            + Point2::direction(
                cell * (0.5 * (rows - 1.0 - n as f32) + row.q as f32),
                cell * (n as f32 + 0.6),
            );
        let fill = if k == current {
            palette::orange()
        } else {
            size_colour(row.state)
        };
        square(c, left, 0.92 * cell, fill.faded(0.5));
        let middle = left + Point2::direction(0.46 * cell, 0.64 * cell);
        c.text(
            &row.state.to_string(),
            middle,
            0.36 * cell,
            palette::ink(),
            Align::Center,
        );
    }
    let note = "REAL STATES OF CL(P,Q), P + Q = 0 TO 6";
    c.text(
        note,
        rect.lo + Point2::direction(0.0, 0.2 * cell),
        10.0,
        palette::ink().faded(0.7),
        Align::Left,
    );
}

/// The current signature's projector and its directions' representations, signed permutations
/// drawn as grids: plus one orange, minus one blue.
fn generators_panel(c: &mut Canvas, rect: Rect, row: &Row) {
    let blades: Vec<String> = row.involutions.iter().map(|m| blade_name(*m)).collect();
    let projector = if blades.is_empty() {
        "NONE".to_string()
    } else {
        blades.join(", ")
    };
    let head = format!(
        "CL({},{}): {} BLADES ON {} REAL STATES",
        row.p, row.q, row.blades, row.state
    );
    c.text(&head, rect.lo, 13.0, palette::ink(), Align::Left);
    let sub = format!("PROJECTOR BLADES: {projector}");
    c.text(
        &sub,
        rect.lo + Point2::direction(0.0, 20.0),
        11.0,
        palette::orange(),
        Align::Left,
    );
    let grid = (rect.width() / 3.2).min(rect.height() / 2.6);
    for (i, matrix) in row.generators.iter().enumerate() {
        let corner = rect.lo
            + Point2::direction(
                grid * 1.08 * (i % 3) as f32,
                44.0 + grid * 1.15 * (i / 3) as f32,
            );
        let step = grid / row.state as f32;
        for (r, entries) in matrix.iter().enumerate() {
            for (k, value) in entries.iter().enumerate() {
                let colour = if *value > 0.0 {
                    palette::orange()
                } else if *value < 0.0 {
                    palette::sky()
                } else {
                    palette::grid()
                };
                let at = corner + Point2::direction(step * k as f32, step * r as f32);
                let strength = if *value == 0.0 { 0.25 } else { 0.8 };
                square(c, at, 0.9 * step, colour.faded(strength));
            }
        }
        let label = format!("E{}", i + 1);
        c.text(
            &label,
            corner + Point2::direction(0.0, grid + 10.0),
            10.0,
            palette::ink().faded(0.7),
            Align::Left,
        );
    }
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let current = ((t / SECONDS_PER_ROW) as usize).min(table().len() - 1);
    table_panel(c, screen.part(0.03, 0.14, 0.45, 0.98), current);
    generators_panel(c, screen.part(0.5, 0.19, 0.98, 0.98), &table()[current]);
    caption(
        c,
        "EXTENSOR REPRESENTATIONS OF CL(P,Q)",
        "A PROJECTOR SELECTS A LEFT IDEAL; LEFT MULTIPLICATION ON IT IS A MAP STATE <- STATE",
    );
}

fn main() {
    run(
        Anim::new(
            "extensor_representations",
            SECONDS_PER_ROW * table().len() as f32,
        )
        .size(960, 540),
        draw,
    );
}

#[cfg(test)]
mod tests {
    /// Pairing a blade's representation with the representations of every multivector, by the
    /// normalized trace, recovers the blade, both components of a split algebra included; and a
    /// map on the states that is no representation (keeping only the scalar state component)
    /// reads out to its projection: the remainder pairs to zero with every representation, and
    /// projecting again changes nothing. numga's six signatures.
    macro_rules! recovers {
        ($name:ident, $state:ident) => {{
            use super::algebras::$name::{Scalar, $state as State};
            use super::$name::*;
            let reps = representations();
            for (i, rep) in reps.blades.iter().enumerate() {
                assert_eq!(reps.readout(*rep), blade(super::canonical_masks(N)[i]));
            }
            let one = Scalar::<(), f64>::new(1.0);
            let arbitrary: Rep = (one * one.scalar_product(State::slot())).cast::<State>();
            let projected = reps.of(reps.readout(arbitrary));
            let residual = arbitrary - projected;
            for rep in &reps.blades {
                assert_eq!(rep.of(residual).trace(), 0.0);
            }
            assert_eq!(reps.of(reps.readout(projected)), projected);
        }};
    }

    #[test]
    fn trace_recovers_every_blade_and_projects_arbitrary_maps() {
        recovers!(cl20, Spinor);
        recovers!(cl30, Spinor);
        recovers!(cl31, Spinor);
        recovers!(cl03, Multivector);
        recovers!(cl40, Spinor);
        recovers!(cl50, Spinor);
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 15.0);
    }
}
