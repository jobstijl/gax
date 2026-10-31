//! numga's `math/triality`: Spin(8) triality, three rotations of eight-dimensional space linked
//! by the octonion product.
//!
//! The octonions are the scalar and the seven vectors of Cl(0,7), multiplied as in
//! `math_octonions`: the Clifford product times one less the Fano trivector, cast back. Vectors
//! and both kinds of spinors of Spin(8) each have eight components, and the octonion product
//! links them: rotating a product agrees with multiplying its two factors, each turned by a
//! companion rotation of its own, `R(x y) = L(x) R'(y)`, though none of the three preserves the
//! product alone. For a turn in the plane of one and a unit vector `u` the three are `x ↦ u x u`,
//! `x ↦ u x` and `x ↦ x u` with `u` the half-angle exponential (the Moufang identity); after one
//! whole turn the product's rotation is back to the identity while both companions are minus the
//! identity, and after two all three are back.
//!
//! A generator of rotations of the eight components is a vector (turning the scalar into the
//! vectors, by the anticommutator) plus a bivector (turning the vectors, by the commutator): 28
//! of them. Each splits into left multiplication by a vector, right multiplication by a vector,
//! and a derivation, which obeys the product rule; minus the trace of a composition pairs skew
//! maps, so one block solve over the two sides (`gax::sites`) finds the split, and the two
//! multiplication parts, redistributed, give each generator its two companions. The squared
//! offsets form a positive form on the generators whose eigenvectors are a frame of all 28;
//! its 14 zero modes are the generators all three actions share, the derivations of the
//! octonions, the exceptional Lie algebra G₂.
//!
//! numga's example has no render; the animation draws the finite actions turning in the plane
//! of the scalar and a direction: their matrices, their traces over two turns (the companions
//! return only after the second), and the spectrum of the offsets' form, fourteen zeros.

use gax::ApproxEq;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, Point2, Rect, backdrop, caption, palette, run_with,
};

gax::algebra! {
    algebra cl07 "Cl(0,7): numga's `a-b-c-d-e-f-g-`, the octonions and the generators of their rotations.";
    basis e1 = -1, e2 = -1, e3 = -1, e4 = -1, e5 = -1, e6 = -1, e7 = -1;
    kind Scalar = [1];
    kind Vector = [e1, e2, e3, e4, e5, e6, e7];
    kind Octonion = [1, e1, e2, e3, e4, e5, e6, e7];
    kind Bivector = [e12, e13, e14, e15, e16, e17, e23, e24, e25, e26, e27, e34, e35, e36, e37, e45, e46, e47, e56, e57, e67];
    kind Generator = [e1, e2, e3, e4, e5, e6, e7, e12, e13, e14, e15, e16, e17, e23, e24, e25, e26, e27, e34, e35, e36, e37, e45, e46, e47, e56, e57, e67];
    kind Trivector = [e123, e124, e125, e126, e127, e134, e135, e136, e137, e145, e146, e147, e156, e157, e167, e234, e235, e236, e237, e245, e246, e247, e256, e257, e267, e345, e346, e347, e356, e357, e367, e456, e457, e467, e567];
}

/// The octonions and the three linked actions of Spin(8).
mod triality {
    use super::cl07::{Bivector, Generator, Multivector, Octonion, Scalar, Trivector, Vector};
    use gax::sites::{SiteField, Sites};
    use gax_numga_examples::signal::phasor;
    use std::sync::OnceLock;

    pub type O = Octonion;
    pub type G = Generator;
    /// A map on the octonions: one action of a generator, or a finite rotation.
    pub type Action = Octonion<(Octonion,)>;
    /// The octonion product, with both factors open.
    pub type Product = Octonion<(Octonion, Octonion)>;
    /// A vector for each generator: one of its multiplication offsets.
    pub type Offset = Vector<(Generator,)>;
    /// Both offsets of each generator, the left on site zero and the right on site one.
    pub type Offsets = SiteField<Vector, 2, (Generator,), f64>;
    /// A form on the generators.
    pub type Form = Scalar<(Generator, Generator)>;

    /// The seven oriented triples, numga's cycles, each in the order its product runs.
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

    /// The direction `e_k`, from one.
    pub fn direction(k: usize) -> Vector {
        Vector::from_coeffs(core::array::from_fn(|i| if i + 1 == k { 1.0 } else { 0.0 }))
    }

    /// The basis generator `k` of 28: the seven vectors, then the 21 bivectors.
    pub fn generator(k: usize) -> G {
        Generator::from_coeffs(core::array::from_fn(|i| if i == k { 1.0 } else { 0.0 }))
    }

    /// The Fano trivector: the sum of the triples' wedges.
    pub fn fano() -> Trivector {
        TRIPLES.iter().fold(Trivector::zero(), |sum, [a, b, c]| {
            sum + (direction(*a) ^ direction(*b) ^ direction(*c))
        })
    }

    /// The product, built once: the Clifford product times one less the Fano trivector, cast
    /// back to the octonions.
    pub fn product() -> &'static Product {
        static PRODUCT: OnceLock<Product> = OnceLock::new();
        PRODUCT.get_or_init(|| {
            let one_less = Multivector::from(Scalar::new(1.0)) - Multivector::from(fano());
            ((Octonion::slot() * Octonion::slot()) * one_less).cast::<Octonion>()
        })
    }

    /// Two octonions multiplied.
    pub fn multiply(first: O, second: O) -> O {
        product().of(first).of(second)
    }

    /// Multiplication by `a` on the left, as a map.
    pub fn left_multiply(a: O) -> Action {
        product().of(a)
    }

    /// Multiplication by `b` on the right, as a map.
    pub fn right_multiply(b: O) -> Action {
        Action::from_images(core::array::from_fn(|j| multiply(unit(j), b)))
    }

    /// A map composed after a multiplication of its factors: `map(x y)`, a map of two.
    pub fn after_product(map: Action) -> Product {
        map.of(*product())
    }

    /// The product of two maps' images: `left(x) right(y)`, a map of two.
    pub fn product_of(left: Action, right: Action) -> Product {
        product().of_both(left, right)
    }

    /// A vector as an octonion.
    pub fn octonion(v: Vector) -> O {
        v.cast::<Octonion>()
    }

    /// A generator's action on the eight components: its bivector part turns the vectors by the
    /// commutator, its vector part turns the scalar into the vectors by the anticommutator.
    pub fn turn(g: G) -> Action {
        let (b, v) = (g.cast::<Bivector>(), g.cast::<Vector>());
        b.commutator(Octonion::slot()).cast::<Octonion>()
            + v.anticommutator(Octonion::slot()).cast::<Octonion>()
    }

    /// The left and right multiplications by the direction `e_a` (side zero and one).
    fn multiplication(side: usize, a: usize) -> Action {
        let v = octonion(direction(a));
        if side == 0 {
            left_multiply(v)
        } else {
            right_multiply(v)
        }
    }

    /// The positive pairing of two skew maps: minus the trace of their composition.
    fn pairing(first: Action, second: Action) -> f64 {
        -first.of(second).trace()
    }

    /// The vectors whose left and right multiplications make up a generator's skew action,
    /// apart from its derivation: the block solve of the pairing between the two sides'
    /// multiplications against the pairing of the generators' actions with them.
    pub fn multiplication_components() -> (Offset, Offset) {
        let block = |s: usize, t: usize| {
            Vector::<(Vector,)>::from_images(core::array::from_fn(|b| {
                Vector::from_coeffs(core::array::from_fn(|a| {
                    pairing(multiplication(s, a + 1), multiplication(t, b + 1))
                }))
            }))
        };
        let form: SiteField<Vector, 2, (Sites<Vector, 2>,), f64> = SiteField::from_blocks(block);
        let overlap = |t: usize| {
            Offset::from_images(core::array::from_fn(|k| {
                Vector::from_coeffs(core::array::from_fn(|b| {
                    pairing(turn(generator(k)), multiplication(t, b + 1))
                }))
            }))
        };
        let offsets: Offsets = form.solve(Offsets::from_sites([overlap(0), overlap(1)]));
        (offsets.site(0), offsets.site(1))
    }

    /// The positive form a vector part makes on the generators: the scalar part of a value times
    /// its Clifford conjugate.
    pub fn conjugate_product(map: Offset) -> Form {
        map.scalar_product(map.conjugate())
    }

    /// A complete frame of rotation generators, ordered by their squared multiplication offsets,
    /// orthonormal in the Clifford conjugate's positive pairing.
    pub fn action_modes(left: Offset, right: Offset) -> ([f64; 28], [G; 28]) {
        let constraint = conjugate_product(left) + conjugate_product(right);
        let metric = Generator::slot()
            .conjugate()
            .scalar_product(Generator::slot());
        constraint.eigh(metric)
    }

    /// The product action and its two factor actions, obeying the product rule: the derivation
    /// in all three, the multiplication parts redistributed.
    pub fn linked_actions(g: G, left: Offset, right: Offset) -> [Action; 3] {
        let output = turn(g);
        let (l, r) = (octonion(left.of(g)), octonion(right.of(g)));
        let derivation = derivation(g, left, right);
        [
            output,
            derivation + left_multiply(l) + right_multiply(l - r),
            derivation + left_multiply(r - l) + right_multiply(r),
        ]
    }

    /// A generator's derivation: its action less its two multiplication parts.
    pub fn derivation(g: G, left: Offset, right: Offset) -> Action {
        let (l, r) = (octonion(left.of(g)), octonion(right.of(g)));
        turn(g) - left_multiply(l) - right_multiply(r)
    }

    /// `R(x y) - L(x) y - x L'(y)`: the failure of three actions to obey the product rule.
    pub fn product_rule_defect(output: Action, left: Action, right: Action) -> Product {
        after_product(output)
            - product_of(left, Octonion::slot())
            - product_of(Octonion::slot(), right)
    }

    /// The half-angle exponential of a unit vector `u` (which squares to minus one):
    /// `cos(angle / 2) + u sin(angle / 2)`, written out (gax has no exponential of a vector).
    pub fn half_turn(u: Vector, angle: f64) -> O {
        let p = phasor(angle / 2.0);
        unit(0).gp(p.e20()) + octonion(u).gp(p.e01())
    }

    /// A plane rotation by `angle` and its two companions, for a unit vector `u`: `x ↦ u x u`,
    /// `x ↦ u x` and `x ↦ x u`.
    pub fn finite_actions(u: Vector, angle: f64) -> [Action; 3] {
        let half = half_turn(u, angle);
        let left = left_multiply(half);
        let right = right_multiply(half);
        [left.of(right), left, right]
    }

    /// The generators and their penalties, the triality's whole content, computed once.
    pub struct Triality {
        pub penalties: [f64; 28],
        pub generators: [G; 28],
        pub left: Offset,
        pub right: Offset,
    }

    impl Triality {
        pub fn new() -> Triality {
            let (left, right) = multiplication_components();
            let (penalties, generators) = action_modes(left, right);
            Triality {
                penalties,
                generators,
                left,
                right,
            }
        }
    }

    impl Default for Triality {
        fn default() -> Self {
            Triality::new()
        }
    }
}

use triality::{Action, Triality};

/// The loop's seconds: two whole turns.
const SECONDS: f32 = 12.0;
/// numga's tolerance on the penalties.
const TOLERANCE: f64 = 1e-10;
/// The actions' names and colours.
const ACTIONS: [(&str, Light); 3] = [
    ("PRODUCT: U X U", palette::ORANGE),
    ("FIRST FACTOR: U X", palette::SKY),
    ("SECOND FACTOR: X U", palette::GREEN),
];

/// What the frames draw: the generators' penalties, and the frame of the turn.
struct Scene {
    penalties: [f64; 28],
    /// The largest coefficient of any generator's failure to obey the product rule.
    defect: f64,
}

impl Scene {
    fn new() -> Scene {
        let t = Triality::new();
        let defect = t
            .generators
            .iter()
            .map(|g| {
                let [output, left, right] = triality::linked_actions(*g, t.left, t.right);
                let d = triality::product_rule_defect(output, left, right);
                d.max_abs_diff(&triality::Product::zero())
            })
            .fold(0.0, f64::max);
        Scene {
            penalties: t.penalties,
            defect,
        }
    }
}

/// The finite actions at `angle`, about the first direction.
fn actions(angle: f64) -> [Action; 3] {
    triality::finite_actions(triality::direction(1), angle)
}

/// A map on the octonions as a matrix of cells, `1, a, ..., g` down and across, orange
/// positive and blue negative by size.
fn matrix(c: &mut Canvas, rect: Rect, map: Action, title: &str, colour: Light) {
    let side = rect.width().min(rect.height() - 16.0);
    let step = side / 8.0;
    let corner = rect.lo + Point2::direction(0.5 * (rect.width() - side), 0.0);
    for column in 0..8 {
        let image = map.of(triality::unit(column));
        for (row, value) in image.c.iter().enumerate() {
            let (tint, strength) = if value.abs() < 1e-12 {
                (palette::GRID, 0.25)
            } else if *value > 0.0 {
                (palette::ORANGE, 0.2 + 0.7 * value.abs().min(1.0) as f32)
            } else {
                (palette::SKY, 0.2 + 0.7 * value.abs().min(1.0) as f32)
            };
            let at = corner + Point2::direction(step * column as f32, step * row as f32);
            let across = Point2::direction(0.9 * step, 0.0);
            let down = Point2::direction(0.0, 0.9 * step);
            c.fill(
                &[at, at + across, at + across + down, at + down],
                tint.faded(strength),
                1.0,
            );
        }
    }
    let at = corner + Point2::direction(0.5 * side, side + 12.0);
    c.text(title, at, 10.0, colour, Align::Center);
}

/// The three actions' traces over the eighth of their dimension, over two whole turns: the
/// product's back to one after the first, the companions' only after the second.
fn traces_panel(c: &mut Canvas, rect: Rect, angle: f64) {
    let ax = Axes::new(rect, [0.0, 2.0], [-1.15, 1.15]);
    ax.frame(c, "", "TURNS", "TRACE / 8");
    let turns = |k: usize| 2.0 * k as f64 / 160.0;
    for (index, (_, colour)) in ACTIONS.iter().enumerate() {
        let curve: Vec<Point2> = (0..=160)
            .map(|k| {
                let a = actions(core::f64::consts::TAU * turns(k))[index];
                Point2::xy(turns(k) as f32, (a.trace() / 8.0) as f32)
            })
            .collect();
        ax.polyline(c, &curve, 1.6, *colour);
        let now = actions(angle)[index].trace() / 8.0;
        let at = Point2::xy((angle / core::f64::consts::TAU) as f32, now as f32);
        ax.scatter(c, &[at], Marker::Dot, 8.0, *colour);
    }
}

/// The penalties of the 28 generators, the squared multiplication offsets, as bars: fourteen
/// zero, the generators of G₂, and fourteen positive.
fn penalties_panel(c: &mut Canvas, scene: &Scene, rect: Rect) {
    let top = scene.penalties.iter().fold(0.0f64, |m, p| m.max(*p)) * 1.1;
    let ax = Axes::new(rect, [0.0, 28.0], [0.0, top as f32]);
    ax.frame(c, "", "GENERATOR", "SQUARED OFFSETS");
    for (k, penalty) in scene.penalties.iter().enumerate() {
        let colour = if *penalty < TOLERANCE {
            palette::GREEN
        } else {
            palette::PURPLE
        };
        let (x0, x1) = (k as f32 + 0.15, k as f32 + 0.85);
        let height = (*penalty as f32).max(top as f32 * 0.01);
        let bar = [
            Point2::xy(x0, 0.0),
            Point2::xy(x1, 0.0),
            Point2::xy(x1, height),
            Point2::xy(x0, height),
        ];
        ax.fill(c, &bar, colour, 0.6);
    }
    let common = scene.penalties.iter().filter(|p| **p < TOLERANCE).count();
    let note = format!("{common} OF 28 COMMON: G2");
    let at = rect.bottom_middle() + Point2::direction(0.0, 52.0);
    c.text(&note, at, 11.0, palette::GREEN, Align::Center);
    let rule = format!("PRODUCT RULE TO {:.0E}", scene.defect.max(1e-17));
    let at = rect.bottom_middle() + Point2::direction(0.0, 70.0);
    c.text(&rule, at, 9.0, palette::INK.faded(0.7), Align::Center);
}

fn draw(c: &mut Canvas, scene: &Scene, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let angle = 2.0 * core::f64::consts::TAU * f64::from(t / SECONDS);
    let row = screen.part(0.03, 0.15, 0.62, 0.52);
    for (k, (action, (title, colour))) in actions(angle).into_iter().zip(ACTIONS).enumerate() {
        let cell = row
            .part(k as f32 / 3.0, 0.0, (k + 1) as f32 / 3.0, 1.0)
            .inset(8.0, 0.0, 8.0, 0.0);
        matrix(c, cell, action, title, colour);
    }
    traces_panel(c, screen.part(0.09, 0.62, 0.6, 0.88), angle);
    penalties_panel(c, scene, screen.part(0.71, 0.2, 0.97, 0.78));
    caption(
        c,
        "TRIALITY: THREE ROTATIONS OF EIGHT DIMENSIONS LINKED BY THE OCTONION PRODUCT",
        "ROTATE THE PRODUCT = MULTIPLY THE FACTORS TURNED BY THEIR COMPANIONS; COMMON TO ALL THREE: G2",
    );
}

fn main() {
    run_with(
        Anim::new("triality", SECONDS).size(960, 540),
        Scene::new,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::cl07::{Bivector, Generator, Octonion, Trivector, Vector};
    use super::triality::{
        Action, G, Offset, Product, Triality, after_product, derivation, direction, fano,
        finite_actions, generator, linked_actions, product_of, product_rule_defect, unit,
    };
    use gax::sites::SiteField;
    use gax::{ApproxEq, SingularValues};

    const TOLERANCE: f64 = 1e-11;

    /// The rank of a linear family of maps on the octonions, `generator -> map`, to numga's
    /// tolerance on the singular values: the family as one map from the generators to eight
    /// sites of octonions, the images of the units.
    fn rank(family: impl Fn(G) -> Action) -> usize {
        let images: SiteField<Octonion, 8, (Generator,), f64> =
            SiteField::from_sites(core::array::from_fn(|j| {
                Octonion::<(Generator,)>::from_images(core::array::from_fn(|k| {
                    family(generator(k)).of(unit(j))
                }))
            }));
        images.svdvals().iter().filter(|s| **s > TOLERANCE).count()
    }

    /// The rank of a map from generators to vectors.
    fn offset_rank(map: Offset) -> usize {
        map.svdvals().iter().filter(|s| **s > TOLERANCE).count()
    }

    /// A map skew in the coefficients, numga's `norm(action, x) + norm(x, action) = 0` with
    /// `norm` the involute's scalar product (the octonions' positive pairing).
    fn is_skew(map: Action) -> bool {
        (0..8).all(|i| {
            (0..8).all(|j| {
                let a = map.of(unit(j)).c[i];
                let b = map.of(unit(i)).c[j];
                (a + b).abs() < TOLERANCE
            })
        })
    }

    /// numga's first test: 28 orthonormal generators; each of the three action families has full
    /// rank and is skew; and every generator's three actions obey the product rule.
    #[test]
    fn every_rotation_has_two_skew_companions_preserving_the_product() {
        let t = Triality::new();
        assert_eq!(t.generators.len(), 28);
        for (i, a) in t.generators.iter().enumerate() {
            for (j, b) in t.generators.iter().enumerate() {
                let dot: f64 = a.c.iter().zip(b.c).map(|(x, y)| x * y).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < TOLERANCE);
            }
        }
        for side in 0..3 {
            assert_eq!(rank(|g| linked_actions(g, t.left, t.right)[side]), 28);
        }
        for g in t.generators {
            let [output, left, right] = linked_actions(g, t.left, t.right);
            for action in [output, left, right] {
                assert!(is_skew(action));
            }
            let defect = product_rule_defect(output, left, right);
            assert!(defect.approx_eq(&Product::zero(), TOLERANCE));
        }
    }

    /// numga's second test: the left and the right offsets have rank seven each and fourteen
    /// together; the derivations have rank fourteen, obey the product rule and kill the scalar.
    #[test]
    fn multiplication_components_leave_fourteen_derivations() {
        let t = Triality::new();
        assert_eq!(offset_rank(t.left), 7);
        assert_eq!(offset_rank(t.right), 7);
        let both: SiteField<Vector, 2, (Generator,), f64> =
            SiteField::from_sites([t.left, t.right]);
        assert_eq!(
            both.svdvals().iter().filter(|s| **s > TOLERANCE).count(),
            14
        );
        assert_eq!(rank(|g| derivation(g, t.left, t.right)), 14);
        for g in t.generators {
            let d = derivation(g, t.left, t.right);
            assert!(product_rule_defect(d, d, d).approx_eq(&Product::zero(), TOLERANCE));
            assert!(d.of(unit(0)).approx_eq(&Octonion::zero(), TOLERANCE));
        }
    }

    /// numga's third test: fourteen generators have zero penalty and fourteen do not; on the
    /// fourteen all three actions are one derivation, with no vector part, commuting with the
    /// Fano trivector.
    #[test]
    fn common_actions_are_exactly_the_fourteen_automorphism_generators() {
        let t = Triality::new();
        let common: Vec<G> = t
            .penalties
            .iter()
            .zip(t.generators)
            .filter(|(p, _)| **p < TOLERANCE)
            .map(|(_, g)| g)
            .collect();
        assert_eq!(common.len(), 14);
        assert_eq!(t.penalties.iter().filter(|p| **p > TOLERANCE).count(), 14);
        for g in common {
            let [output, left, right] = linked_actions(g, t.left, t.right);
            assert!(output.approx_eq(&left, TOLERANCE));
            assert!(output.approx_eq(&right, TOLERANCE));
            let defect = product_rule_defect(output, output, output);
            assert!(defect.approx_eq(&Product::zero(), TOLERANCE));
            assert!(g.cast::<Vector>().approx_eq(&Vector::zero(), TOLERANCE));
            let bracket = g.cast::<Bivector>().commutator(fano());
            assert!(bracket.approx_eq(&Trivector::zero(), TOLERANCE));
        }
    }

    /// numga's fourth test: the finite actions about a unit direction preserve the full product
    /// and the length; all three are the identity at the start and after two turns, and after
    /// one turn the product's is the identity while both companions are minus it.
    #[test]
    fn finite_actions_preserve_the_full_product_and_return_after_two_turns() {
        use core::f64::consts::{PI, TAU};
        let d = |k: usize| direction(k);
        let u = (d(1) + d(2).gp(2.0) - d(3).gp(3.0) + d(4).gp(5.0)).normalized();
        let identity = Octonion::slot();
        let angles = [0.0, 0.37, 1.1, PI, TAU, 2.0 * TAU];
        let all: Vec<[Action; 3]> = angles.iter().map(|a| finite_actions(u, *a)).collect();
        for [output, left, right] in &all {
            let defect = after_product(*output) - product_of(*left, *right);
            assert!(defect.approx_eq(&Product::zero(), TOLERANCE));
            for action in [output, left, right] {
                // Orthogonal in the octonions' positive pairing: its columns orthonormal.
                for i in 0..8 {
                    for j in 0..8 {
                        let dot: f64 = action
                            .of(unit(i))
                            .c
                            .iter()
                            .zip(action.of(unit(j)).c)
                            .map(|(x, y)| x * y)
                            .sum();
                        let expected = if i == j { 1.0 } else { 0.0 };
                        assert!((dot - expected).abs() < TOLERANCE);
                    }
                }
            }
        }
        for action in all[0].iter().chain(&all[5]) {
            assert!(action.approx_eq(&identity, TOLERANCE));
        }
        let [output, left, right] = all[4];
        assert!(output.approx_eq(&identity, TOLERANCE));
        assert!(left.approx_eq(&identity.gp(-1.0), TOLERANCE));
        assert!(right.approx_eq(&identity.gp(-1.0), TOLERANCE));
    }

    /// The scenario's checks: every generator's actions obey the product rule (above), 28
    /// generators of which fourteen are common; and a frame draws.
    #[test]
    fn twenty_eight_generators_fourteen_common_and_a_frame_draws() {
        let scene = super::Scene::new();
        assert_eq!(
            scene
                .penalties
                .iter()
                .filter(|p| **p < super::TOLERANCE)
                .count(),
            14
        );
        gax_numga_examples::app::assert_draws_with(super::draw, &scene, 1.0);
    }
}
