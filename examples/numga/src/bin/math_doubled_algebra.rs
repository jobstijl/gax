//! numga's `math/doubled_algebra`: the wedge, the contraction and every linear map on the
//! multivectors of the plane, as products in the split algebra Cl(2,2).
//!
//! Double the plane's directions `x`, `y` with partners `u`, `v` that square to minus one, and the
//! null vectors `(x + u) / 2` and `(x - u) / 2` (and likewise for `y`) add a basis vector to a state
//! as a factor and remove it: acting from the left on the left ideal of the empty-state projector,
//! the product of the removers and creators, they are the wedge and the left contraction with
//! the vector. Two insertions or two removals anticommute to zero, and an insertion and a removal
//! anticommute to the inner product. A creator times a remover is a dyad on vectors and extends
//! to blades by the product rule, a derivation; summed over both directions it counts the grade.
//! The state is four coefficients (the scalar, `x`, `y` and `xy`), and the sixteen dimensions of
//! Cl(2,2) are all sixteen maps on it, composition being the geometric product: any map encodes
//! as a multivector, read out by the trace pairing solved against the scalar product (the
//! readout of `math_extensor_representations`). The same holds in Cl(3,3) and Cl(4,4), on eight
//! and sixteen states.
//!
//! The exponential of the mixed product of the creator of `x` and the remover of `y` shears `y`
//! towards `x`, and carries the scalar, both vectors and their oriented area along: the area is
//! kept.
//!
//! The animation: left a square and its grid sheared back and forth by the exponential, with the
//! images of the two basis vectors; right the maps on the four states, as matrices: the creators,
//! the removers and the shear.

use gax_numga_examples::signal::phasor;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, ORIGIN2, Point2, Rect, backdrop, caption, label_under,
    palette, run,
};

gax::algebra! {
    algebra cl22 "Cl(2,2): numga's `x+y+u-v-`, the plane's directions and their partners.";
    basis e1 = 1, e2 = 1, e3 = -1, e4 = -1;
    kind Scalar = [1];
    kind Vector = [e1, e2, e3, e4];
    kind Bivector = [e12, e13, e14, e23, e24, e34];
    versor Even = [1, e12, e13, e14, e23, e24, e34, e1234];
    kind State = [1, e1, e2, e12];
}

gax::algebra! {
    algebra cl33 "Cl(3,3): numga's `x+y+z+u-v-w-`.";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = -1, e5 = -1, e6 = -1;
    kind Scalar = [1];
    kind State = [1, e1, e2, e3, e12, e13, e23, e123];
}

gax::algebra! {
    algebra cl44 "Cl(4,4): numga's `x+y+z+t+u-v-w-s-`.";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = -1, e6 = -1, e7 = -1, e8 = -1;
    kind Scalar = [1];
    kind State = [1, e1, e2, e3, e4, e12, e13, e14, e23, e24, e34, e123, e124, e134, e234, e1234];
}

/// The blades of `n` directions as masks, in gax's order for the full multivector: by grade,
/// then by mask.
pub fn canonical_masks(n: usize) -> Vec<u32> {
    let mut masks: Vec<u32> = (0..1u32 << n).collect();
    masks.sort_by_key(|m| (m.count_ones(), *m));
    masks
}

/// The doubled algebra of `n` directions: creators and removers, the empty-state projector,
/// the representations of every multivector on the states, and the readout of any map.
macro_rules! doubled {
    ($name:ident = $algebra:ident, $n:expr) => {
        pub mod $name {
            use super::canonical_masks;
            use super::$algebra::{Multivector as Full, Scalar, State};

            pub type F = Full;
            pub type S = State;
            /// A map on the states.
            pub type Rep = State<(State,)>;
            /// The plane's (or space's) directions; the algebra has twice as many.
            pub const N: usize = $n;

            /// The blade of a mask, with coefficient one.
            pub fn blade(mask: u32) -> F {
                let index = canonical_masks(2 * N)
                    .iter()
                    .position(|m| *m == mask)
                    .expect("a blade");
                Full::from_coeffs(core::array::from_fn(|i| if i == index { 1.0 } else { 0.0 }))
            }

            /// The geometric product, kept out of line: in eight dimensions it is sixty-five
            /// thousand terms, which inlining would compile again at every call.
            #[inline(never)]
            pub fn product(a: F, b: F) -> F {
                a * b
            }

            /// The basis state `j`.
            pub fn state(j: usize) -> S {
                State::from_coeffs(core::array::from_fn(|i| if i == j { 1.0 } else { 0.0 }))
            }

            /// The number of state components, `2^N`.
            pub fn dimension() -> usize {
                1 << N
            }

            /// The null vector adding direction `i` to a state: `(e_i + f_i) / 2`.
            pub fn creator(i: usize) -> F {
                (blade(1 << i) + blade(1 << (i + N))).gp(0.5)
            }

            /// The null vector removing direction `i` from a state: `(e_i - f_i) / 2`.
            pub fn remover(i: usize) -> F {
                (blade(1 << i) - blade(1 << (i + N))).gp(0.5)
            }

            /// The empty-state projector: removing any direction from it gives zero. Its factors
            /// commute.
            pub fn projector() -> F {
                (0..N).fold(blade(0), |p, i| product(p, product(remover(i), creator(i))))
            }

            /// A state embedded in the left ideal, `s P`.
            pub fn embed(s: S) -> F {
                product(s.cast::<Full>(), projector())
            }

            /// The state an element of the ideal reads out as: its independent coefficients,
            /// times the projector's factor of one half per direction.
            pub fn read(x: F) -> S {
                x.cast::<State>().gp(dimension() as f64)
            }

            /// The representations of the blades: embed a state, multiply from the left, read
            /// it back out.
            pub struct Representations {
                pub blades: Vec<Rep>,
            }

            impl Representations {
                pub fn new() -> Representations {
                    let p = projector();
                    let blades = canonical_masks(2 * N)
                        .into_iter()
                        .map(|m| {
                            Rep::from_images(core::array::from_fn(|j| {
                                read(product(blade(m), product(state(j).cast::<Full>(), p)))
                            }))
                        })
                        .collect();
                    Representations { blades }
                }

                /// A multivector's representation: its blades' representations, weighted.
                pub fn of(&self, x: F) -> Rep {
                    self.blades
                        .iter()
                        .zip(x.c)
                        .fold(Rep::zero(), |sum, (r, c)| sum + r.gp(c))
                }

                /// The multivector whose left action is a given map on the states: the form
                /// taking a multivector to the trace of its representation composed with the
                /// map, over the states' dimension, solved against the scalar product.
                pub fn readout(&self, map: Rep) -> F {
                    let d = dimension() as f64;
                    let form = Scalar::<(Full,)>::from_images(core::array::from_fn(|i| {
                        Scalar::new(self.blades[i].of(map).trace() / d)
                    }));
                    Full::slot().scalar_product(Full::slot()).solve(form)
                }
            }

            impl Default for Representations {
                fn default() -> Self {
                    Representations::new()
                }
            }
        }
    };
}

doubled!(plane = cl22, 2);
doubled!(space = cl33, 3);
doubled!(spacetime = cl44, 4);

/// The plane's own pieces: its vectors as states, the shear.
mod shear {
    use super::cl22::{Bivector, Multivector};
    use super::plane::{self, F, Representations, S};

    /// A planar vector as a state.
    pub fn vector(x: f64, y: f64) -> S {
        S::new(0.0, x, y, 0.0)
    }

    /// Removing `y` and then adding `x` sends `y` to `x`: the generator of the shear.
    pub fn generator() -> F {
        plane::product(plane::creator(0), plane::remover(1))
    }

    /// The shear by `amount`: the exponential of the generator, which is a bivector (and squares
    /// to zero), as a multivector.
    pub fn shear(amount: f64) -> F {
        let g = generator().cast::<Bivector>().gp(amount);
        g.exp().into_inner().cast::<Multivector>()
    }

    /// The shear's map on the states.
    pub fn motion(reps: &Representations, amount: f64) -> super::plane::Rep {
        reps.of(shear(amount))
    }
}

use plane::Representations;

/// The loop's seconds: the shear back and forth.
const SECONDS: f32 = 6.0;
/// numga's amount, the largest of the sweep; the grid's half width and lines.
const AMOUNT: f64 = 0.8;
const EXTENT: f64 = 1.5;
const LINES: usize = 9;

/// The colours: the reference, the sheared grid, the square, the basis vectors.
const REFERENCE: Light = palette::INK;
const GRID: Light = palette::BLUE;
const SQUARE: Light = palette::ORANGE;
const BASIS: [Light; 2] = [palette::GREEN, palette::PURPLE];

/// The amount of shear at `t`, `AMOUNT` times the sine of the loop's phase.
fn amount(t: f32) -> f64 {
    AMOUNT * phasor(core::f64::consts::TAU * f64::from(t / SECONDS)).e01()
}

/// A state's vector part as a point of the drawing.
fn place(s: plane::S) -> Point2 {
    Point2::xy(s.c[1] as f32, s.c[2] as f32)
}

/// The square and its grid, the reference faint and dashed, the sheared image bright, and the
/// images of the two basis vectors.
fn shear_panel(c: &mut Canvas, reps: &Representations, rect: Rect, t: f32) {
    let motion = shear::motion(reps, amount(t));
    let moved = |x: f64, y: f64| place(motion.of(shear::vector(x, y)));
    let room = (rect.width() / rect.height()).max(1.0);
    let ax = Axes::equal(
        rect,
        ORIGIN2,
        (EXTENT * (1.0 + AMOUNT) * 0.62) as f32 / room * 1.6,
    );
    let offsets: Vec<f64> = (0..LINES)
        .map(|k| -EXTENT + 2.0 * EXTENT * k as f64 / (LINES - 1) as f64)
        .collect();
    for (shape, colour, width) in [(false, REFERENCE.faded(0.25), 0.6), (true, GRID, 0.9)] {
        let at = |x: f64, y: f64| {
            if shape {
                moved(x, y)
            } else {
                Point2::xy(x as f32, y as f32)
            }
        };
        for o in &offsets {
            ax.line(c, at(-EXTENT, *o), at(EXTENT, *o), width, colour);
            ax.line(c, at(*o, -EXTENT), at(*o, EXTENT), width, colour);
        }
    }
    let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
    let square: Vec<Point2> = corners.iter().map(|(x, y)| moved(*x, *y)).collect();
    ax.fill(c, &square, SQUARE, 0.25);
    let mut reference: Vec<Point2> = corners
        .iter()
        .map(|(x, y)| Point2::xy(*x as f32, *y as f32))
        .collect();
    reference.push(reference[0]);
    ax.dashed(c, &reference, 1.0, 4.0, REFERENCE.faded(0.6));
    let mut outline = square.clone();
    outline.push(square[0]);
    ax.polyline(c, &outline, 1.6, SQUARE);
    for (k, colour) in BASIS.iter().enumerate() {
        let tip = if k == 0 {
            moved(1.0, 0.0)
        } else {
            moved(0.0, 1.0)
        };
        ax.arrow(c, ORIGIN2, tip, 2.0, 11.0, *colour);
    }
    ax.scatter(
        c,
        &[ORIGIN2],
        gax_numga_examples::Marker::Dot,
        6.0,
        palette::INK,
    );
    let label = format!(
        "Y SHEARED TOWARDS X: EXP({:+.2} C_X R_Y); THE AREA KEPT",
        amount(t)
    );
    label_under(c, rect, &label, 24.0, 0.9);
}

/// A map on the four states as a matrix of cells, orange positive, blue negative, by size.
fn matrix(c: &mut Canvas, corner: Point2, side: f32, map: plane::Rep, title: &str) {
    let step = side / 4.0;
    for column in 0..4 {
        let image = map.of(plane::state(column));
        for (row, value) in image.c.iter().enumerate() {
            let colour = if *value > 0.0 {
                palette::ORANGE
            } else if *value < 0.0 {
                palette::SKY
            } else {
                palette::GRID
            };
            let strength = if *value == 0.0 {
                0.25
            } else {
                0.3 + 0.6 * value.abs().min(1.0) as f32
            };
            let at = corner + Point2::direction(step * column as f32, step * row as f32);
            let (across, down) = (
                Point2::direction(0.9 * step, 0.0),
                Point2::direction(0.0, 0.9 * step),
            );
            c.fill(
                &[at, at + across, at + across + down, at + down],
                colour.faded(strength),
                1.0,
            );
        }
    }
    let at = corner + Point2::direction(0.5 * side, side + 14.0);
    c.text(title, at, 11.0, palette::INK, Align::Center);
}

/// The maps on the states, `1, x, y, xy` down and across: the creators insert, the removers
/// contract, and the shear is one entry off the identity.
fn maps_panel(c: &mut Canvas, reps: &Representations, rect: Rect, t: f32) {
    let side = (rect.width() / 3.6).min(rect.height() / 2.8);
    let maps = [
        (reps.of(plane::creator(0)), "C_X: X ^"),
        (reps.of(plane::creator(1)), "C_Y: Y ^"),
        (reps.of(plane::remover(0)), "R_X: X _|"),
        (reps.of(plane::remover(1)), "R_Y: Y _|"),
        (shear::motion(reps, amount(t)), "THE SHEAR"),
        (
            reps.of(plane::product(plane::creator(0), plane::remover(1))),
            "C_X R_Y",
        ),
    ];
    for (k, (map, title)) in maps.into_iter().enumerate() {
        let corner =
            rect.lo + Point2::direction(side * 1.25 * (k % 3) as f32, side * 1.4 * (k / 3) as f32);
        matrix(c, corner, side, map, title);
    }
    let note = "STATES 1, X, Y, XY: COLUMNS TO ROWS";
    let at = rect.lo + Point2::direction(0.0, -12.0);
    c.text(note, at, 9.0, palette::INK.faded(0.7), Align::Left);
}

fn draw(c: &mut Canvas, t: f32) {
    thread_local! {
        static REPS: Representations = Representations::new();
    }
    backdrop(c);
    let screen = c.rect();
    REPS.with(|reps| {
        shear_panel(c, reps, screen.part(0.02, 0.13, 0.55, 0.94), t);
        maps_panel(c, reps, screen.part(0.6, 0.24, 0.98, 0.95), t);
    });
    caption(
        c,
        "THE DOUBLED ALGEBRA: EVERY MAP OF THE PLANE'S MULTIVECTORS IN CL(2,2)",
        "NULL VECTORS INSERT AND REMOVE DIRECTIONS; THEIR PRODUCTS ARE ALL SIXTEEN MAPS ON FOUR STATES",
    );
}

fn main() {
    run(Anim::new("doubled_algebra", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::cl22::{Multivector, State, Vector};
    use super::plane::{self, Rep, Representations, creator, embed, product, projector, remover};
    use super::shear;
    use gax::ApproxEq;
    use gax_numga_examples::rng::{Draw, rng};

    /// The plane's direction `k` (`x` or `y`) as a vector of Cl(2,2).
    fn direction(k: usize) -> Vector {
        Vector::from_coeffs(core::array::from_fn(|i| if i == k { 1.0 } else { 0.0 }))
    }

    /// A map taking states to states through the full algebra: `f` of a state, cast back.
    fn through(f: impl Fn(Multivector) -> Multivector) -> Rep {
        Rep::from_images(core::array::from_fn(|j| {
            f(plane::state(j).cast::<Multivector>()).cast::<State>()
        }))
    }

    /// numga's first test: the projector is idempotent and the removers annihilate it; the
    /// readout undoes the embedding; the creators are the wedge, the removers the left
    /// contraction and the vectors the geometric product, from the left; repeated insertion or
    /// removal anticommutes to zero, and a mixed pair to the inner product. Exact.
    #[test]
    fn null_generators_insert_remove_and_recover_the_familiar_product() {
        let reps = Representations::new();
        let p = projector();
        assert_eq!(product(p, p), p);
        for i in 0..2 {
            assert_eq!(product(remover(i), p), Multivector::zero());
        }
        for j in 0..4 {
            assert_eq!(plane::read(embed(plane::state(j))), plane::state(j));
        }
        let insertions = [0, 1].map(|i| reps.of(creator(i)));
        let removals = [0, 1].map(|i| reps.of(remover(i)));
        for i in 0..2 {
            let v = direction(i);
            assert_eq!(insertions[i], through(|s| (v ^ s).cast::<Multivector>()));
            assert_eq!(removals[i], through(|s| v.lc(s).cast::<Multivector>()));
            let whole = reps.of(v.cast::<Multivector>());
            assert_eq!(whole, through(|s| (v * s).cast::<Multivector>()));
        }
        for i in 0..2 {
            for j in 0..2 {
                let twice = insertions[i].of(insertions[j]) + insertions[j].of(insertions[i]);
                assert_eq!(twice, Rep::zero());
                let twice = removals[i].of(removals[j]) + removals[j].of(removals[i]);
                assert_eq!(twice, Rep::zero());
                let mixed = removals[i].of(insertions[j]) + insertions[j].of(removals[i]);
                let inner = direction(i).scalar_product(direction(j)).s();
                assert_eq!(mixed, State::slot().gp(inner));
            }
        }
    }

    /// numga's second test: the dyads, a creator times a remover, are `x ^ (y ⌋ ·)`, and extend
    /// to the area by the product rule; together the diagonal ones count the grade. Exact.
    #[test]
    fn dyads_extend_to_blades_by_the_product_rule() {
        let reps = Representations::new();
        for i in 0..2 {
            for j in 0..2 {
                let dyad = reps.of(product(creator(i), remover(j)));
                let (a, b) = (direction(i), direction(j));
                let expected = through(|s| (a ^ b.lc(s)).cast::<Multivector>());
                assert_eq!(dyad, expected);
                // On the basis: D(a ^ b) = D(a) ^ b + a ^ D(b).
                let as_state = |v: Vector| v.cast::<Multivector>().cast::<State>();
                let full = |s: State| s.cast::<Multivector>();
                for (p, q) in [(0, 1), (1, 0), (0, 0)] {
                    let (u, w) = (direction(p), direction(q));
                    let extension = dyad.of((u ^ w).cast::<Multivector>().cast::<State>());
                    let rule = (full(dyad.of(as_state(u))) ^ w.cast::<Multivector>())
                        + (u.cast::<Multivector>() ^ full(dyad.of(as_state(w))));
                    assert_eq!(full(extension), rule.cast::<State>().cast::<Multivector>());
                }
            }
        }
        let count = reps.of(product(creator(0), remover(0)) + product(creator(1), remover(1)));
        let state = State::new(3.0, 1.0, -2.0, 4.0);
        assert_eq!(count.of(state), State::new(0.0, 1.0, -2.0, 8.0));
    }

    /// numga's third test: the exponential of the shear's generator moves `y` by the amount
    /// along `x` and keeps `x`, carries the area as the wedge of the moved vectors, and keeps
    /// the area and the scalar.
    #[test]
    fn exponentiating_a_shear_moves_vectors_and_their_wedge_together() {
        let reps = Representations::new();
        for amount in [-1.25, 0.0, 0.75] {
            let motion = shear::motion(&reps, amount);
            let (x, y) = (shear::vector(1.0, 0.0), shear::vector(0.0, 1.0));
            assert!(motion.of(x).approx_eq(&x, 1e-12));
            assert!(motion.of(y).approx_eq(&shear::vector(amount, 1.0), 1e-12));
            let area = State::new(0.0, 0.0, 0.0, 1.0);
            let wedge = motion.of(x).cast::<Multivector>() ^ motion.of(y).cast::<Multivector>();
            assert!(motion.of(area).approx_eq(&wedge.cast::<State>(), 1e-12));
            assert!(motion.of(area).approx_eq(&area, 1e-12));
            let one = State::new(1.0, 0.0, 0.0, 0.0);
            assert!(motion.of(one).approx_eq(&one, 1e-12));
        }
    }

    /// numga's fourth test: an element of mixed grade exchanges the scalar and the area, and its
    /// square's representation is the representation composed with itself. Exact.
    #[test]
    fn mixed_grade_exchange_and_product_are_map_composition() {
        let reps = Representations::new();
        let exchange = product(creator(0), creator(1)) + product(remover(1), remover(0));
        let representation = reps.of(exchange);
        let state = State::new(2.0, 3.0, -1.0, 5.0);
        assert_eq!(representation.of(state), State::new(5.0, 0.0, 0.0, 2.0));
        let twice = representation.of(representation.of(state));
        assert_eq!(twice, State::new(2.0, 0.0, 0.0, 5.0));
        assert_eq!(
            reps.of(product(exchange, exchange)),
            representation.of(representation)
        );
    }

    /// numga's fifth test: arbitrary maps encode as multivectors, which compose by
    /// multiplication. numga's maps are drawn from numpy's stream; these from a seeded one.
    #[test]
    fn arbitrary_maps_encode_as_multivectors_and_compose_by_multiplication() {
        let reps = Representations::new();
        let mut r = rng(0);
        let mut random = || {
            Rep::from_coeffs(core::array::from_fn(|_| {
                core::array::from_fn(|_| r.normal())
            }))
        };
        let (first, second) = (random(), random());
        let (a, b) = (reps.readout(first), reps.readout(second));
        assert!(reps.of(a).approx_eq(&first, 1e-12));
        assert!(reps.of(b).approx_eq(&second, 1e-12));
        assert!(reps.of(product(a, b)).approx_eq(&first.of(second), 1e-12));
    }

    /// numga's parametrized test: in Cl(2,2), Cl(3,3) and Cl(4,4) the readout recovers an
    /// arbitrary map on the states, exercising the whole algebra.
    #[test]
    fn split_algebra_recovers_arbitrary_maps_through_eight_dimensions() {
        macro_rules! recovers {
            ($name:ident) => {{
                use super::$name::{Rep, Representations};
                let reps = Representations::new();
                let mut r = rng(0);
                let arbitrary = Rep::from_coeffs(core::array::from_fn(|_| {
                    core::array::from_fn(|_| r.normal())
                }));
                let recovered = reps.readout(arbitrary);
                assert!(reps.of(recovered).approx_eq(&arbitrary, 1e-12));
            }};
        }
        recovers!(plane);
        recovers!(space);
        // Cl(4,4)'s scalar-product pairing is a 256 x 256 form, half a megabyte held on the
        // stack, with its factorization: more than a test thread's default stack.
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(|| recovers!(spacetime))
            .expect("a thread")
            .join()
            .expect("Cl(4,4) recovers");
    }

    /// The scenario's check: the shear keeps the oriented area.
    #[test]
    fn the_shear_keeps_the_area() {
        let motion = shear::motion(&Representations::new(), super::AMOUNT);
        let area = State::new(0.0, 0.0, 0.0, 1.0);
        assert!(motion.of(area).approx_eq(&area, 1e-12));
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
    }
}
