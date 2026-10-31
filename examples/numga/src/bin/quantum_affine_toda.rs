//! numga's `quantum/affine_toda`: affine E8 Toda theory, whose masses and three-particle
//! couplings are E8's root geometry.
//!
//! The Toda field takes values in the eight-dimensional space of E8's roots, in R(8,0) (shared
//! with `math_e8`): the field's values are antivectors, intrinsic, and the roots, mirrors,
//! vectors, read them by the metric-free pairing `root & field`. The potential is a sum over
//! the eight simple roots and the negative highest root of exponentials of those readings,
//! weighted by the marks (the highest root's coefficients on the simple roots, and one), which
//! make zero field an equilibrium. Its second and third derivatives at the vacuum are forms on
//! the field, sums of products of the same readings: the curvature `Σ m (r & ·)(r & ·)` and the
//! cubic form `Σ m (r & ·)(r & ·)(r & ·)`, each built once with its slots open.
//!
//! The curvature's eigenvalues against the field's metric are the eight squared masses, and they
//! stand in the ratios of the radii of the eight Coxeter orbits seen in the Coxeter plane (the
//! lightest two in the golden ratio). Three particles couple exactly when one root from each
//! of their orbits closes a triangle: 56 of the 120 triples do, and the coupling is the area of
//! that triangle in the Coxeter plane, times `4/√30` in units of the lightest mass.
//!
//! The animation turns with the Coxeter element's slowest plane: left the eight mass rings with
//! the roots on them, right numga's three cases, the root triangles of the triples 112 and 123
//! closing (they couple) and that of 235 left open (it does not).

#[path = "../shared/math_e8.rs"]
mod e8;

use gax_numga_examples::{
    Align, Anim, Axes, Canvas, ORIGIN2, Point2, Rect, backdrop, caption, label_under, palette,
    points, run_with,
};

/// Affine E8 Toda theory's vacuum: its curvature and cubic form, masses and root triangles.
mod toda {
    use super::e8::{self, Antivector, Bivector, RootSystem, Scalar, Vector};

    /// The field's values: antivectors, intrinsic.
    pub type Field = Antivector;
    /// A form on two fields: the curvature of the potential at the vacuum.
    pub type MassForm = Scalar<(Antivector, Antivector)>;
    /// A form on three fields: the potential's third derivative at the vacuum.
    pub type CubicForm = Scalar<(Antivector, Antivector, Antivector)>;

    /// The highest root's coefficients on the simple roots, then the weight of its negative.
    pub const MARKS: [f64; 9] = [2.0, 3.0, 4.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    /// numga's three cases: the triples of species drawn (from zero, by mass).
    pub const CASES: [[usize; 3]; 3] = [[0, 0, 1], [0, 1, 2], [1, 2, 4]];

    /// The potential energy density over the vacuum's: the marked sum of each root's
    /// exponentiated reading less one, both overall scales one.
    #[cfg_attr(not(test), allow(dead_code))] // numga's check of the derivatives
    pub fn potential(field: Field, interaction_roots: &[Vector; 9]) -> f64 {
        interaction_roots
            .iter()
            .zip(MARKS)
            .map(|(r, m)| m * ((*r & field).s().exp() - 1.0))
            .sum()
    }

    /// The vacuum's spectrum and the roots ordered by the masses they match.
    pub struct Spectrum {
        /// The simple roots and the negative highest root.
        #[cfg_attr(not(test), allow(dead_code))] // numga's checks of the vacuum
        pub interaction_roots: [Vector; 9],
        #[cfg_attr(not(test), allow(dead_code))] // numga's checks of the masses
        pub curvature: MassForm,
        pub cubic: CubicForm,
        /// The eight masses, ascending.
        pub masses: [f64; 8],
        /// The particles: the curvature's eigenmodes, unit in the field's metric.
        pub modes: [Field; 8],
        /// The roots, `[step][species]`: the Coxeter orbits ordered by their radius in the
        /// Coxeter plane.
        pub roots: Vec<Vec<Vector>>,
        /// The Coxeter plane's frame.
        pub plane_x: Vector,
        pub plane_y: Vector,
        /// The bivector whose exponential is one Coxeter step.
        pub generator: Bivector,
    }

    impl Spectrum {
        /// The interaction roots give the vacuum's curvature and cubic form; the curvature's
        /// eigenmodes are the particles, and the Coxeter orbits are ordered by the masses they
        /// match.
        pub fn new() -> Spectrum {
            let system = RootSystem::new();
            let simple = e8::simple_roots();
            // The negative highest root closes the weighted root sum, making zero field an
            // equilibrium.
            let affine = simple
                .iter()
                .zip(MARKS)
                .fold(Vector::zero(), |sum, (r, m)| sum - r.gp(m));
            let interaction_roots: [Vector; 9] =
                core::array::from_fn(|k| if k < 8 { simple[k] } else { affine });

            // Each exponential adds the product of its root's readings to the derivatives.
            let mut curvature = MassForm::zero();
            let mut cubic = CubicForm::zero();
            for (r, m) in interaction_roots.iter().zip(MARKS) {
                let reading = *r & Field::slot();
                curvature += (reading * reading).gp(m);
                cubic += (reading * reading * reading).gp(m);
            }
            // With a canonical kinetic term, the curvature's eigenvalues against the field's
            // metric are the squared masses. The grade-seven blades square to minus one, so the
            // positive metric is minus the inner product.
            let metric = (Field::slot() | Field::slot()).gp(-1.0);
            let (squared, modes) = curvature.eigh(metric);
            #[allow(clippy::disallowed_methods)] // a mass from its square, not a length
            let masses = squared.map(f64::sqrt);

            // The smallest-turning plane gives eight radii in the same ratios as the masses.
            let radius = |o: usize| (system.roots[0][o] | system.planes[0]).norm();
            let mut order: Vec<usize> = (0..8).collect();
            order.sort_by(|a, b| radius(*a).total_cmp(&radius(*b)));
            let roots = system
                .roots
                .iter()
                .map(|step| order.iter().map(|o| step[*o]).collect())
                .collect();
            Spectrum {
                interaction_roots,
                curvature,
                cubic,
                masses,
                modes,
                roots,
                plane_x: system.plane_x[0],
                plane_y: system.plane_y[0],
                generator: system.generator(),
            }
        }

        /// The Coxeter plane, oriented, unit.
        pub fn coxeter_plane(&self) -> Bivector {
            self.plane_x ^ self.plane_y
        }

        /// Each species' radius in the Coxeter plane: the size of a root's part in it.
        pub fn radii(&self) -> [f64; 8] {
            core::array::from_fn(|s| (self.roots[0][s] | self.coxeter_plane()).norm())
        }

        /// The cubic coupling of three species, the cubic form on their modes.
        pub fn coupling(&self, [a, b, c]: [usize; 3]) -> f64 {
            let m = &self.modes;
            self.cubic.of(m[a]).of(m[b]).of(m[c]).s()
        }

        /// The Coxeter step's rotor taken `steps` times (any real number).
        pub fn turn(&self, steps: f64) -> e8::Rotor {
            self.generator.gp(steps).exp()
        }
    }

    impl Default for Spectrum {
        fn default() -> Self {
            Spectrum::new()
        }
    }

    /// Each triple of species has a closest-closing root triangle, one root from each
    /// species' ring, and a squared closure that measures how far it fails to close. A common
    /// Coxeter turn changes no closure, so the first root stays at step zero.
    pub fn root_triangle(roots: &[Vec<Vector>], [a, b, c]: [usize; 3]) -> ([Vector; 3], f64) {
        let first = roots[0][a];
        let mut best = ([first, first, first], f64::INFINITY);
        for second in roots.iter().map(|step| step[b]) {
            for third in roots.iter().map(|step| step[c]) {
                let closure = e8::length_squared(first + second + third);
                if closure < best.1 {
                    best = ([first, second, third], closure);
                }
            }
        }
        best
    }

    /// Every unordered triple of species, its closest root triangle and closure, its cubic
    /// coupling, and the area of that triangle in the Coxeter plane.
    #[cfg_attr(not(test), allow(dead_code))] // the drawing counts; numga's checks read the rest
    pub struct Interactions {
        pub species: Vec<[usize; 3]>,
        pub triangles: Vec<[Vector; 3]>,
        pub closure: Vec<f64>,
        pub couplings: Vec<f64>,
        pub areas: Vec<f64>,
    }

    impl Interactions {
        pub fn new(spectrum: &Spectrum) -> Interactions {
            let species: Vec<[usize; 3]> = (0..8)
                .flat_map(|a| (a..8).flat_map(move |b| (b..8).map(move |c| [a, b, c])))
                .collect();
            let (triangles, closure): (Vec<_>, _) = species
                .iter()
                .map(|s| root_triangle(&spectrum.roots, *s))
                .unzip();
            let couplings = species.iter().map(|s| spectrum.coupling(*s)).collect();
            // Triangle areas in the Coxeter plane are measured by its unit oriented bivector.
            let plane = spectrum.coxeter_plane();
            let area = |t: &[Vector; 3]| (t[0] ^ t[1]).scalar_product(plane).s().abs() / 2.0;
            let areas = triangles.iter().map(area).collect();
            Interactions {
                species,
                triangles,
                closure,
                couplings,
                areas,
            }
        }
    }
}

use e8::Vector;
use e8::look::orbit_colour;
use toda::{CASES, Spectrum};

/// The loop's seconds.
const SECONDS: f32 = 12.0;
/// The Coxeter steps the pictures turn by over the loop: a tenth of a turn in the Coxeter
/// plane, which changes no closure.
const STEPS: f64 = 3.0;

/// What the frames draw: the spectrum, numga's cases' triangles and closures.
struct Scene {
    spectrum: Spectrum,
    cases: Vec<([Vector; 3], f64, f64)>,
    /// The nearest pairs of roots, by index into `[step][species]` flattened.
    edges: Vec<(usize, usize)>,
    /// How many of the 120 triples couple.
    couplings: usize,
    /// The half height of each case's view: its path and rings, with room.
    case_reach: Vec<f64>,
}

impl Scene {
    fn new() -> Scene {
        let spectrum = Spectrum::new();
        let cases = CASES
            .iter()
            .map(|s| {
                let (triangle, closure) = toda::root_triangle(&spectrum.roots, *s);
                (triangle, closure, spectrum.coupling(*s))
            })
            .collect::<Vec<_>>();
        let flat: Vec<Vector> = spectrum.roots.iter().flatten().copied().collect();
        let edges = e8::nearest_pairs(&flat);
        let interactions = toda::Interactions::new(&spectrum);
        let couplings = interactions.closure.iter().filter(|c| **c < 1e-10).count();
        // numga's limit, per case: the farthest corner of its path or ring, with a sixth more.
        let radii = spectrum.radii();
        let plane = spectrum.coxeter_plane();
        let case_reach = CASES
            .iter()
            .zip(&cases)
            .map(|(species, (t, _, _))| {
                let corners = [t[0], t[0] + t[1], t[0] + t[1] + t[2]];
                let paths = corners.into_iter().map(|v| (v | plane).norm());
                let rings = species.iter().map(|k| radii[*k]);
                paths.chain(rings).fold(0.0, f64::max) * 1.16
            })
            .collect();
        Scene {
            spectrum,
            cases,
            edges,
            couplings,
            case_reach,
        }
    }

    /// A root's place in the Coxeter plane after the picture's turn.
    fn place(&self, turn: e8::Rotor, v: Vector) -> Point2 {
        let s = &self.spectrum;
        e8::look::coordinates(turn >> v, s.plane_x, s.plane_y)
    }

    /// The largest radius of a ring.
    fn reach(&self) -> f64 {
        self.spectrum.radii().into_iter().fold(0.0, f64::max)
    }
}

/// The eight mass rings in the Coxeter plane: each species' roots on its circle, nearest
/// neighbours joined faintly, and each ring's mass over the lightest.
fn rings_panel(c: &mut Canvas, scene: &Scene, rect: Rect, turn: e8::Rotor) {
    let s = &scene.spectrum;
    let room = (rect.height() / rect.width()).max(1.0);
    let ax = Axes::equal(rect, ORIGIN2, (scene.reach() * 1.08) as f32 * room);
    let radii = s.radii();
    for (k, radius) in radii.iter().enumerate() {
        let ring = points::circle(ORIGIN2, *radius as f32, 120);
        ax.polyline(c, &ring, 0.7, orbit_colour(k).faded(0.3));
    }
    let flat: Vec<Point2> = s
        .roots
        .iter()
        .flatten()
        .map(|r| scene.place(turn, *r))
        .collect();
    for &(i, j) in &scene.edges {
        let edge = orbit_colour(i % 8).mix_light(orbit_colour(j % 8), 0.5);
        ax.line(c, flat[i], flat[j], 0.5, edge.faded(0.05));
    }
    for step in &s.roots {
        for (k, root) in step.iter().enumerate() {
            let px = ax.px(scene.place(turn, *root));
            c.disk(px, 3.0, palette::BOTTOM.faded(4.0));
            c.disk(px, 2.1, orbit_colour(k));
        }
    }
    let size = 10.0;
    let legend = c.rect().part(0.015, 0.2, 0.15, 0.9).lo;
    for (k, mass) in s.masses.iter().enumerate() {
        let line = format!("M{} {:.3}", k + 1, mass / s.masses[0]);
        let at = legend + Point2::direction(0.0, 1.8 * size * k as f32);
        c.text(&line, at, size, orbit_colour(k), Align::Left);
    }
    let couple = format!("{} OF 120 TRIPLES COUPLE", scene.couplings);
    let at = legend + Point2::direction(0.0, 1.8 * size * 9.0);
    c.text(&couple, at, size * 0.8, palette::INK, Align::Left);
    label_under(
        c,
        rect,
        "MASSES OVER THE LIGHTEST = RING RADII OVER THE SMALLEST",
        24.0,
        0.9,
    );
}

/// One case: the rings of its species, and its three roots joined head to tail in their
/// species' colours from a white start; a gap that does not close dashed.
fn case_panel(c: &mut Canvas, scene: &Scene, rect: Rect, case: usize, turn: e8::Rotor) {
    let s = &scene.spectrum;
    let species = CASES[case];
    let (triangle, closure, coupling) = scene.cases[case];
    let room = (rect.height() / rect.width()).max(1.0);
    let ax = Axes::equal(rect, ORIGIN2, scene.case_reach[case] as f32 * room);
    let radii = s.radii();
    for (k, radius) in radii.iter().enumerate() {
        if species.contains(&k) {
            let ring = points::circle(ORIGIN2, *radius as f32, 120);
            ax.polyline(c, &ring, 0.6, orbit_colour(k).faded(0.25));
            for step in &s.roots {
                let px = ax.px(scene.place(turn, step[k]));
                c.disk(px, 1.3, orbit_colour(k).faded(0.45));
            }
        }
    }
    let mut corner = Vector::zero();
    let start = scene.place(turn, corner);
    for (root, k) in triangle.iter().zip(species) {
        let next = corner + *root;
        let (from, to) = (scene.place(turn, corner), scene.place(turn, next));
        ax.arrow(c, from, to, 2.0, 11.0, orbit_colour(k));
        corner = next;
    }
    // A closed path leaves no gap; an open one is dashed back to the start.
    let end = scene.place(turn, corner);
    if closure > 1e-10 {
        ax.dashed(c, &[start, end], 1.3, 4.0, palette::INK);
    }
    c.disk(ax.px(start), 3.0, palette::INK);
    let name: String = species.iter().map(|k| format!("{}", k + 1)).collect();
    let verdict = if closure < 1e-10 {
        format!("{name}: CLOSES, COUPLING {:.3}", coupling.abs())
    } else {
        format!("{name}: OPEN, NO COUPLING")
    };
    label_under(c, rect, &verdict, 14.0, 0.95);
}

fn draw(c: &mut Canvas, scene: &Scene, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let turn = scene.spectrum.turn(STEPS * f64::from(t / SECONDS));
    rings_panel(c, scene, screen.part(0.13, 0.13, 0.52, 0.96), turn);
    let right = screen.part(0.52, 0.14, 0.99, 0.96);
    for case in 0..3 {
        let cell = if case < 2 {
            right.part(0.5 * case as f32, 0.0, 0.5 * (case + 1) as f32, 0.5)
        } else {
            right.part(0.25, 0.5, 0.75, 1.0)
        };
        case_panel(c, scene, cell.inset(4.0, 4.0, 4.0, 16.0), case, turn);
    }
    caption(
        c,
        "AFFINE E8 TODA: MASSES AND COUPLINGS FROM ROOTS (R(8,0))",
        "THE VACUUM'S CURVATURE GIVES THE MASSES, ITS CUBIC FORM THE COUPLINGS: ROOT TRIANGLES IN THE COXETER PLANE",
    );
}

fn main() {
    run_with(
        Anim::new("affine_toda", SECONDS).size(960, 540),
        Scene::new,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::e8::{self, Vector};
    use super::toda::{self, Field, Interactions, Spectrum};
    use gax::ApproxEq;

    const TOLERANCE: f64 = 1e-10;

    /// numga's first test: the marked interaction roots sum to zero and have squared length
    /// two; the potential vanishes at the vacuum; the masses are positive and the modes
    /// diagonalize the curvature with the squared masses on its diagonal; and finite
    /// differences of the potential along a mix of modes match the curvature and cubic forms.
    #[test]
    fn potential_vacuum_and_derivatives() {
        let s = Spectrum::new();
        let m = &s.modes;
        let direction = m[0] + m[1].gp(0.6) - m[2].gp(0.4);
        let step = 0.005;
        let energies = [-2.0, -1.0, 1.0, 2.0]
            .map(|a| toda::potential(direction.gp(a * step), &s.interaction_roots));
        let second = (energies[2] + energies[1]) / (step * step);
        let third = (energies[3] - 2.0 * energies[2] + 2.0 * energies[1] - energies[0])
            / (2.0 * step * step * step);

        let sum = s
            .interaction_roots
            .iter()
            .zip(toda::MARKS)
            .fold(Vector::zero(), |sum, (r, m)| sum + r.gp(m));
        assert!(sum.approx_eq(&Vector::zero(), TOLERANCE));
        for r in &s.interaction_roots {
            assert!((e8::length_squared(*r) - 2.0).abs() < TOLERANCE);
        }
        assert!(toda::potential(Field::zero(), &s.interaction_roots).abs() < TOLERANCE);
        assert!(s.masses.iter().all(|m| *m > 0.0));
        for i in 0..8 {
            for j in 0..8 {
                let value = s.curvature.of(m[i]).of(m[j]).s();
                let expected = if i == j {
                    s.masses[i] * s.masses[i]
                } else {
                    0.0
                };
                assert!((value - expected).abs() < TOLERANCE, "{i} {j}");
            }
        }
        let curvature = s.curvature.of(direction).of(direction).s();
        let cubic = s.cubic.of(direction).of(direction).of(direction).s();
        assert!((second - curvature).abs() < 1e-5, "{second} {curvature}");
        assert!((third - cubic).abs() < 1e-5, "{third} {cubic}");
    }

    /// numga's second test: the masses stand in the ratios of the rings' radii; of the 120
    /// triples 56 close a root triangle, exactly those that couple, their roots summing to
    /// zero, and each coupling is `4/√30` times the triangle's area in the Coxeter plane, in
    /// units of the lightest mass.
    #[test]
    fn all_cubic_couplings_match_root_triangles() {
        let s = Spectrum::new();
        let i = Interactions::new(&s);
        let radii = s.radii();
        let scale = s.masses[0] / radii[0];
        #[allow(clippy::disallowed_methods)] // the reference it is checked against
        let factor = 4.0 / 30f64.sqrt();

        for k in 0..8 {
            assert!((radii[k] / radii[0] - s.masses[k] / s.masses[0]).abs() < TOLERANCE);
        }
        assert_eq!(i.species.len(), 120);
        let allowed: Vec<bool> = i.closure.iter().map(|c| *c < TOLERANCE).collect();
        assert_eq!(allowed.iter().filter(|a| **a).count(), 56);
        for (k, closes) in allowed.iter().enumerate() {
            assert_eq!(
                *closes,
                i.couplings[k].abs() > TOLERANCE,
                "{:?}",
                i.species[k]
            );
            if *closes {
                let [a, b, c] = i.triangles[k];
                assert!((a + b + c).approx_eq(&Vector::zero(), TOLERANCE));
                let expected = factor * i.areas[k] * scale * scale;
                assert!((i.couplings[k].abs() - expected).abs() < TOLERANCE);
            }
        }
    }

    /// The scenario's checks: the radii's ratios are the masses', and numga's first two
    /// triples close and couple while the third does neither.
    #[test]
    fn the_cases_close_and_couple_or_neither() {
        let scene = super::Scene::new();
        let s = &scene.spectrum;
        let radii = s.radii();
        for k in 0..8 {
            assert!((radii[k] / radii[0] - s.masses[k] / s.masses[0]).abs() < 1e-10);
        }
        let closes = scene.cases.iter().map(|(_, closure, _)| *closure < 1e-10);
        let couples = scene
            .cases
            .iter()
            .map(|(_, _, coupling)| coupling.abs() > 1e-10);
        assert_eq!(closes.collect::<Vec<_>>(), [true, true, false]);
        assert_eq!(couples.collect::<Vec<_>>(), [true, true, false]);
    }

    #[test]
    fn a_frame_draws() {
        let scene = super::Scene::new();
        gax_numga_examples::app::assert_draws_with(super::draw, &scene, 1.0);
    }
}
