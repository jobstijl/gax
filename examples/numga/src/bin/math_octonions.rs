//! numga's `math/octonions`: octonion multiplication, and the rotations that preserve it.
//!
//! The scalar and the seven vectors of Cl(0,7) carry the eight octonion components (numga's
//! units `a` to `g` are `e1` to `e7` here). A fixed trivector, the sum of seven oriented
//! triples of units, supplies their multiplication: the Clifford product of two octonions,
//! times one less the trivector, cast back to the octonions. The product is a bilinear map,
//! `Octonion <- (Octonion, Octonion)`, built once with both factors open. It is not
//! associative: `(a b) d - a (b d) = 2 g`; it is alternative, and lengths multiply. The seven
//! triples are the lines of the Fano plane, each a quaternion subalgebra (`a b = c` for
//! `abc`).
//!
//! Clifford multiplication and the grade projection commute with every rotation, so a rotation
//! preserves the octonion product exactly when it fixes the trivector. Its generators are the
//! bivectors that commute with the trivector: the null space of the map `Trivector <- Bivector`
//! taking a bivector to its commutator with the trivector, 14 of the 21, the exceptional Lie
//! algebra of G₂. They obey the product rule, close under commutators, and exponentiate to
//! rotors that are automorphisms.
//!
//! The same octonions are every multivector of space, Cl(3,0): the even part a quaternion and
//! the odd part the pseudoscalar times a second one, multiplied by doubling the quaternions.
//! With one direction squaring to minus one, Cl(2,1), the doubling gives the split octonions,
//! where `1 + z` has length zero.
//!
//! The animation: the Fano plane, its triples lit in turn, and the failure of a rotation to
//! preserve the product as it turns: an ordinary rotation in the `ab` plane, and its part in
//! G₂, which keeps it at round-off.

use gax_numga_examples::points;
use gax_numga_examples::signal::phasor;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, Point2, Rect, backdrop, caption, palette, run,
};

gax::algebra! {
    algebra cl07 "Cl(0,7): seven directions squaring to minus one, the home of the octonions.";
    basis e1 = -1, e2 = -1, e3 = -1, e4 = -1, e5 = -1, e6 = -1, e7 = -1;
    kind Scalar = [1];
    kind Vector = [e1, e2, e3, e4, e5, e6, e7];
    kind Octonion = [1, e1, e2, e3, e4, e5, e6, e7];
    kind Bivector = [e12, e13, e14, e15, e16, e17, e23, e24, e25, e26, e27, e34, e35, e36, e37, e45, e46, e47, e56, e57, e67];
    kind Trivector = [e123, e124, e125, e126, e127, e134, e135, e136, e137, e145, e146, e147, e156, e157, e167, e234, e235, e236, e237, e245, e246, e247, e256, e257, e267, e345, e346, e347, e356, e357, e367, e456, e457, e467, e567];
    kind Even = [1, e12, e13, e14, e15, e16, e17, e23, e24, e25, e26, e27, e34, e35, e36, e37, e45, e46, e47, e56, e57, e67, e1234, e1235, e1236, e1237, e1245, e1246, e1247, e1256, e1257, e1267, e1345, e1346, e1347, e1356, e1357, e1367, e1456, e1457, e1467, e1567, e2345, e2346, e2347, e2356, e2357, e2367, e2456, e2457, e2467, e2567, e3456, e3457, e3467, e3567, e4567, e123456, e123457, e123467, e123567, e124567, e134567, e234567];
}

gax::algebra! {
    algebra cl21 "Cl(2,1): numga's `x+y+z-`, whose multivectors double into the split octonions.";
    basis e1 = 1, e2 = 1, e3 = -1;
    kind Scalar = [1];
    kind Vector = [e1, e2, e3];
    versor Rotor = [1, e23, e31, e12];
    kind Odd = [e1, e2, e3, e123];
    kind Multivector = [1, e1, e2, e3, e23, e31, e12, e123];
}

mod octonions {
    use super::cl07::{Bivector, Even, Multivector, Octonion, Scalar, Trivector, Vector};
    use std::sync::OnceLock;

    pub type O = Octonion<(), f64>;
    pub type V = Vector<(), f64>;
    pub type B = Bivector<(), f64>;
    pub type T = Trivector<(), f64>;
    pub type R = Even<(), f64>;
    /// The octonion product, with both factors open.
    pub type Product = Octonion<(Octonion, Octonion), f64>;

    /// The units' names, numga's: one, and `a` to `g` for `e1` to `e7`.
    pub const NAMES: [&str; 8] = ["1", "A", "B", "C", "D", "E", "F", "G"];
    /// The seven oriented triples, numga's cycles, each in the order its product runs: `a b = c`
    /// for `abc`.
    pub const TRIPLES: [[usize; 3]; 7] = [
        [1, 2, 3],
        [1, 4, 5],
        [1, 7, 6],
        [2, 4, 6],
        [2, 5, 7],
        [3, 4, 7],
        [3, 6, 5],
    ];

    /// The unit `k`: one, or the direction `e_k`.
    pub fn unit(k: usize) -> O {
        Octonion::from_coeffs(core::array::from_fn(|i| if i == k { 1.0 } else { 0.0 }))
    }

    /// The direction `e_k` as a vector of the algebra.
    pub fn direction(k: usize) -> V {
        Vector::from_coeffs(core::array::from_fn(|i| if i + 1 == k { 1.0 } else { 0.0 }))
    }

    /// The fixed trivector: the sum of the triples' wedges.
    pub fn orientation() -> T {
        TRIPLES.iter().fold(Trivector::zero(), |sum, [a, b, c]| {
            sum + (direction(*a) ^ direction(*b) ^ direction(*c))
        })
    }

    /// The product, built once: the Clifford product of the two factors times one less the
    /// orientation, projected back onto the octonions (the projection is part of the product).
    pub fn product() -> &'static Product {
        static PRODUCT: OnceLock<Product> = OnceLock::new();
        PRODUCT.get_or_init(|| {
            let one_less = Multivector::from(Scalar::new(1.0)) - Multivector::from(orientation());
            ((Octonion::slot() * Octonion::slot()) * one_less).cast::<Octonion>()
        })
    }

    /// Two octonions multiplied.
    pub fn multiply(first: O, second: O) -> O {
        product().of(first).of(second)
    }

    /// The positive squared length: the value paired with its grade involution.
    pub fn norm_squared(value: O) -> f64 {
        value.involute().scalar_product(value).s()
    }

    /// The difference between the two bracketings of three factors.
    pub fn associator(first: O, second: O, third: O) -> O {
        multiply(multiply(first, second), third) - multiply(first, multiply(second, third))
    }

    /// The infinitesimal rotations preserving the product: the bivectors whose commutator with
    /// the orientation vanishes, the right singular vectors of that map with singular values
    /// below the tolerance.
    pub fn symmetry_generators(tolerance: f64) -> Vec<B> {
        let constraint: Trivector<(Bivector,), f64> = Bivector::slot().commutator(orientation());
        let (sigma, right, _) = constraint.svd_thin();
        sigma
            .iter()
            .zip(right)
            .filter(|(s, _)| **s < tolerance)
            .map(|(_, v)| v)
            .collect()
    }

    /// Rotate a product, or multiply the rotated factors: their difference.
    pub fn multiplication_defect(rotor: R, first: O, second: O) -> O {
        turned(rotor, multiply(first, second))
            - multiply(turned(rotor, first), turned(rotor, second))
    }

    /// An octonion turned by a rotor: its sandwich `R x ~R`, which a rotor keeps among the
    /// octonions (the sandwich's other grades vanish).
    pub fn turned(rotor: R, value: O) -> O {
        (rotor * value * rotor.reverse()).cast::<Octonion>()
    }

    /// The rotor turning by `angle` about a bivector, `exp(B angle / 2)` (numga's).
    pub fn rotor(generator: B, angle: f64) -> R {
        generator.gp(angle / 2.0).exp().into_inner()
    }

    /// The part of a bivector in the space the (orthonormal) generators span.
    pub fn project(seed: B, generators: &[B]) -> B {
        generators.iter().fold(Bivector::zero(), |sum, g| {
            sum + g.gp(g.reverse().scalar_product(seed).s())
        })
    }

    /// The plane of the first two units, numga's seed of an ordinary rotation.
    pub fn seed() -> B {
        direction(1) ^ direction(2)
    }
}

/// The octonion product of two multivectors of space, by doubling the quaternions: the even
/// parts multiply as quaternions, and the odd parts, a second quaternion apiece, enter
/// reversed and in swapped order. Stamped out for Cl(3,0) and Cl(2,1).
macro_rules! doubling {
    ($name:ident, $algebra:path) => {
        pub mod $name {
            use $algebra::{Multivector, Odd, Rotor};
            pub type M = Multivector<(), f64>;

            /// `a c - ~d b + d a + b ~c`, with `a`, `c` the even and `b`, `d` the odd parts.
            pub fn doubled(first: M, second: M) -> M {
                let (a, b) = (first.cast::<Rotor>(), first.cast::<Odd>());
                let (c, d) = (second.cast::<Rotor>(), second.cast::<Odd>());
                (a * c - d.reverse() * b).cast::<Multivector>()
                    + (d * a + b * c.reverse()).cast::<Multivector>()
            }

            /// The scalar part of a multivector times its reverse.
            pub fn length_squared(x: M) -> f64 {
                (x * x.reverse()).s()
            }
        }
    };
}

doubling!(space, gax::vga3d);
doubling!(split, super::cl21);

use octonions::*;

const SECONDS: f32 = 14.0;
/// The seconds each triple stays lit: twice round the seven in the loop.
const PER_TRIPLE: f32 = 1.0;
/// numga's tolerance on the generators' singular values.
const TOLERANCE: f64 = 1e-10;
/// The decades of the squared defect drawn, from well below round-off to above its largest.
const DECADES: [f32; 2] = [-34.0, 8.0];

/// The corners of the Fano triangle (`b`, `d`, `g`), its sides' midpoints (`f`, `c`, `e`) and
/// its centre (`a`), so that every triple lies on a side, a median, or the circle through the
/// midpoints; indexed by unit.
fn places() -> [Point2; 8] {
    let corner = |degrees: f32| {
        let p = phasor(degrees.to_radians());
        Point2::xy(p.e20(), p.e01())
    };
    let (b, d, g) = (corner(90.0), corner(210.0), corner(330.0));
    let mid = |p: Point2, q: Point2| (p + q).gp(0.5);
    let origin = Point2::xy(0.0, 0.0);
    [origin, origin, b, mid(d, g), d, mid(b, g), mid(b, d), g]
}

/// An arrowhead a fraction of the way from `from` to `to`, pointing along.
fn head(ax: &Axes, c: &mut Canvas, from: Point2, to: Point2, colour: Light) {
    let (place, step) = (from + (to - from).gp(0.62), (to - from).gp(0.02));
    ax.arrow(c, place - step, place + step, 1.0, 10.0, colour);
}

/// Whether a triple lies on the circle through the midpoints, rather than on a line.
fn on_circle(triple: [usize; 3]) -> bool {
    let radius = |k: usize| (places()[k] & Point2::xy(0.0, 0.0)).norm();
    triple.iter().all(|k| (radius(*k) - 0.5).abs() < 1e-4)
}

/// One triple: its line (a side or a median, between its two points farthest apart) or arcs
/// of the circle, with arrowheads on the steps of its cycle that do not pass the third point.
fn draw_triple(ax: &Axes, c: &mut Canvas, triple: [usize; 3], colour: Light) {
    let at = triple.map(|k| places()[k]);
    let centre = Point2::xy(0.0, 0.0);
    if on_circle(triple) {
        for k in 0..3 {
            // Along the shorter arc: towards the middle of the chord pushed out to the circle.
            let (from, middle) = (at[k], (at[k] + at[(k + 1) % 3]).gp(0.5));
            let out = (middle - centre).gp(0.5 / (middle & centre).norm());
            head(ax, c, from, centre + out, colour);
        }
        return;
    }
    let length = |(i, j): (usize, usize)| (at[i] & at[j]).norm();
    let steps = [(0, 1), (1, 2), (2, 0)];
    let ends = steps
        .into_iter()
        .max_by(|x, y| length(*x).total_cmp(&length(*y)));
    let (first, last) = ends.expect("three steps");
    ax.line(c, at[first], at[last], 1.6, colour);
    for (i, j) in steps.into_iter().filter(|s| Some(*s) != ends) {
        head(ax, c, at[i], at[j], colour);
    }
}

/// The Fano plane in `rect`, the triple `lit` brighter.
fn fano_panel(c: &mut Canvas, rect: Rect, lit: usize) {
    let ax = Axes::equal(rect, Point2::xy(0.0, 0.1), 1.05);
    let circle = points::circle(Point2::xy(0.0, 0.0), 0.5, 96);
    let dim = palette::sky().faded(0.45);
    let circle_lit = on_circle(TRIPLES[lit]);
    ax.polyline(
        c,
        &circle,
        1.6,
        if circle_lit { palette::orange() } else { dim },
    );
    for (k, triple) in TRIPLES.iter().enumerate() {
        draw_triple(
            &ax,
            c,
            *triple,
            if k == lit { palette::orange() } else { dim },
        );
    }
    for (k, place) in places().iter().enumerate().skip(1) {
        ax.scatter(
            c,
            &[*place],
            Marker::Dot,
            22.0,
            palette::bottom().whitened(0.05),
        );
        ax.scatter(c, &[*place], Marker::Ring, 22.0, palette::ink());
        let label = *place + Point2::direction(0.0, -0.035);
        ax.text(c, label, NAMES[k], 12.0, palette::ink(), Align::Center);
    }
    let [a, b, cc] = TRIPLES[lit].map(|k| NAMES[k]);
    let rule = format!("{a} {b} = {cc},  {b} {cc} = {a},  {cc} {a} = {b}");
    c.text(
        &rule,
        rect.bottom_left() + Point2::direction(0.0, -6.0),
        13.0,
        palette::orange(),
        Align::Left,
    );
}

/// The decade of the squared defect of a rotation by `angle` about `generator`, on the
/// product of `a` with `d`, at least the lowest drawn.
fn defect(generator: octonions::B, angle: f64) -> f64 {
    let d = multiplication_defect(rotor(generator, angle), unit(1), unit(4));
    norm_squared(d)
        .max(10f64.powf(f64::from(DECADES[0])))
        .log10()
}

/// The defects of an ordinary rotation and of its part in G₂ over a turn, at `t`.
fn defect_panel(c: &mut Canvas, rect: Rect, t: f32) {
    let generators = symmetry_generators(TOLERANCE);
    let ordinary = seed();
    let symmetry = project(ordinary, &generators);
    let ax = Axes::new(rect, [0.0, 1.0], DECADES);
    ax.frame(c, "LOG10 |R(A D) - R(A) R(D)|^2", "TURN", "");
    let angle = |s: f64| core::f64::consts::TAU * s;
    for (generator, colour) in [(ordinary, palette::red()), (symmetry, palette::green())] {
        let curve: Vec<Point2> = (0..=120)
            .map(|k| {
                let s = k as f64 / 120.0;
                Point2::xy(s as f32, defect(generator, angle(s)) as f32)
            })
            .collect();
        ax.polyline(c, &curve, 1.6, colour);
        let now = f64::from(t / SECONDS);
        let marker = Point2::xy(now as f32, defect(generator, angle(now)) as f32);
        ax.scatter(c, &[marker], Marker::Dot, 8.0, colour);
    }
    ax.legend(
        c,
        &[
            ("ROTATION IN THE AB PLANE", palette::red()),
            ("ITS PART IN G2", palette::green()),
        ],
    );
    let count = format!("{} OF 21 GENERATORS PRESERVE THE PRODUCT", generators.len());
    // Below the axis's label.
    let below = rect.bottom_left() + Point2::direction(-50.0, 56.0);
    c.text(&count, below, 11.0, palette::ink(), Align::Left);
    let skew = associator(unit(1), unit(2), unit(4));
    let terms: Vec<String> = (0..8)
        .filter(|k| skew.c[*k] != 0.0)
        .map(|k| format!("{} {}", skew.c[k], NAMES[k]))
        .collect();
    let rule = format!("NOT ASSOCIATIVE: (A B) D - A (B D) = {}", terms.join(" + "));
    let further = below + Point2::direction(0.0, 22.0);
    c.text(&rule, further, 11.0, palette::ink(), Align::Left);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let lit = ((t / PER_TRIPLE) as usize) % TRIPLES.len();
    fano_panel(c, screen.part(0.02, 0.14, 0.48, 0.9), lit);
    defect_panel(c, screen.part(0.56, 0.18, 0.97, 0.78), t);
    caption(
        c,
        "OCTONIONS: A PRODUCT FROM A TRIVECTOR OF CL(0,7)",
        "THE FANO PLANE'S SEVEN TRIPLES, AND THE 14 ROTATIONS THAT KEEP THE PRODUCT: G2",
    );
}

fn main() {
    run(Anim::new("octonions", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::octonions::*;
    use super::{space, split};
    use gax::ApproxEq;
    use gax_numga_examples::rng::{Draw, rng};

    /// The commutator of a bivector with an octonion, an octonion again (it turns the
    /// directions and leaves the scalar).
    fn derive(generator: B, value: O) -> O {
        generator.commutator(value).cast::<super::cl07::Octonion>()
    }

    /// The failure of an infinitesimal rotation to obey the product rule.
    fn derivation_defect(generator: B, first: O, second: O) -> O {
        derive(generator, multiply(first, second))
            - multiply(derive(generator, first), second)
            - multiply(first, derive(generator, second))
    }

    /// A random octonion, normal coefficients.
    fn random(r: &mut impl Draw) -> O {
        super::cl07::Octonion::from_coeffs(core::array::from_fn(|_| r.normal()))
    }

    /// Lengths multiply; the product is alternative (`(x x) z = x (x z)`, `(x y) y = x (y y)`)
    /// and not associative: `(a b) d - a (b d) = 2 g`, exactly. numga's draws are twelve
    /// triples of normal octonions; these are drawn from a seeded stream.
    #[test]
    fn norm_composition_alternativity_and_nonassociativity() {
        let mut r = rng(0);
        for _ in 0..12 {
            let (x, y, z) = (random(&mut r), random(&mut r), random(&mut r));
            let error = norm_squared(multiply(x, y)) - norm_squared(x) * norm_squared(y);
            assert!(error.abs() < 1e-10, "{error}");
            assert!(associator(x, x, z).approx_eq(&O::zero(), 1e-10));
            assert!(associator(x, y, y).approx_eq(&O::zero(), 1e-10));
        }
        assert_eq!(associator(unit(1), unit(2), unit(4)), unit(7).gp(2.0));
    }

    /// Left multiplications by the directions satisfy every Clifford relation: `L_i L_j +
    /// L_j L_i = 2 (e_i · e_j)`, exactly.
    #[test]
    fn left_multiplications_satisfy_all_clifford_relations() {
        let action = |k: usize| product().of(unit(k));
        for i in 1..8 {
            for j in 1..8 {
                let both = action(i).of(action(j)) + action(j).of(action(i));
                let square = (unit(i).scalar_product(unit(j))).s();
                let identity = super::cl07::Octonion::<(), f64>::slot();
                assert_eq!(both, identity.gp(2.0 * square), "{i} {j}");
            }
        }
    }

    /// Fourteen generators obey the product rule, close under commutators (their brackets
    /// still commute with the orientation), and exponentiate to rotors that preserve the
    /// product, on every pair of units.
    #[test]
    fn fourteen_generators_obey_the_product_rule_and_exponentiate_to_automorphisms() {
        let generators = symmetry_generators(super::TOLERANCE);
        assert_eq!(generators.len(), 14);
        for g in &generators {
            let turn = rotor(*g, 0.8);
            for i in 0..8 {
                for j in 0..8 {
                    let (x, y) = (unit(i), unit(j));
                    assert!(derivation_defect(*g, x, y).approx_eq(&O::zero(), 1e-12));
                    assert!(multiplication_defect(turn, x, y).approx_eq(&O::zero(), 1e-12));
                }
            }
            for h in &generators {
                let bracket = g.commutator(*h).commutator(orientation());
                assert!(bracket.approx_eq(&T::zero(), 1e-12));
            }
        }
    }

    /// The Fano plane's triples run in the order of their products: for `abc`, `a b = c`.
    /// (numga reads the triples off its trivectors; here they define the trivectors, so the
    /// check is that the product follows them.)
    #[test]
    fn the_fano_triples_run_in_the_order_of_their_products() {
        for [a, b, c] in TRIPLES {
            assert_eq!(multiply(unit(a), unit(b)), unit(c));
        }
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
    }

    /// The scene's checks (the associator of `a`, `b`, `d` is checked above): left
    /// multiplication by `a` squares to minus one; a rotation in G₂ preserves the product where
    /// the ordinary rotation it came from does not.
    #[test]
    fn a_squares_to_minus_one_and_only_the_g2_rotation_preserves_the_product() {
        let left = product().of(unit(1));
        let identity = super::cl07::Octonion::<(), f64>::slot();
        assert!(left.of(left).approx_eq(&identity.gp(-1.0), 1e-10));
        let generators = symmetry_generators(super::TOLERANCE);
        let symmetry = project(seed(), &generators);
        let (a, d) = (unit(1), unit(4));
        assert!(multiplication_defect(rotor(symmetry, 0.8), a, d).approx_eq(&O::zero(), 1e-10));
        assert!(norm_squared(multiplication_defect(rotor(seed(), 0.8), a, d)) > 0.1);
    }

    /// Doubling the quaternions of space gives the octonions: images of `a`, `b`, `d` and their
    /// products square to minus one and anticommute, and the triples not used to build them
    /// hold by themselves. Lengths multiply in both signatures; in the split one, `1 + z` has
    /// length zero and its conjugate `-1 + z` multiplies it to zero.
    #[test]
    fn doubling_in_the_algebra_of_space_gives_the_same_octonions_and_the_split_ones() {
        use gax::vga3d::Multivector;
        let blade = |k: usize| {
            Multivector::<(), f64>::from_coeffs(core::array::from_fn(|i| {
                if i == k { 1.0 } else { 0.0 }
            }))
        };
        let (a, b, d) = (blade(1), blade(2), blade(3));
        let (c, e, f) = (
            space::doubled(a, b),
            space::doubled(a, d),
            space::doubled(b, d),
        );
        let g = space::doubled(c, d);
        let images = [a, b, c, d, e, f, g];
        for (i, x) in images.iter().enumerate() {
            assert_eq!(space::doubled(*x, *x), blade(0).gp(-1.0));
            for (j, y) in images.iter().enumerate().filter(|(j, _)| *j != i) {
                assert_eq!(
                    space::doubled(*x, *y) + space::doubled(*y, *x),
                    Multivector::zero(),
                    "{i} {j}"
                );
            }
        }
        assert_eq!(space::doubled(a, g), f);
        assert_eq!(space::doubled(b, e), g);
        assert_eq!(space::doubled(c, f), e);
        let mut r = rng(0);
        for _ in 0..12 {
            let (x, y) = (
                Multivector::from_coeffs(core::array::from_fn(|_| r.normal())),
                Multivector::from_coeffs(core::array::from_fn(|_| r.normal())),
            );
            let error = space::length_squared(space::doubled(x, y))
                - space::length_squared(x) * space::length_squared(y);
            assert!(error.abs() < 1e-10);
            let (x, y) = (
                super::cl21::Multivector::from_coeffs(x.c),
                super::cl21::Multivector::from_coeffs(y.c),
            );
            let error = split::length_squared(split::doubled(x, y))
                - split::length_squared(x) * split::length_squared(y);
            assert!(error.abs() < 1e-10);
        }
        let null = super::cl21::Multivector::<(), f64>::new(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let conjugate = super::cl21::Multivector::new(-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(split::length_squared(null), 0.0);
        assert_eq!(
            split::doubled(null, conjugate),
            super::cl21::Multivector::zero()
        );
    }
}
