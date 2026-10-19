//! numga's `surfaces/koebe`: Koebe polyhedra, every triangulation of the sphere as a polyhedron
//! whose edges all touch the unit sphere, in the algebra `R(3,1)` of the sphere (declared here
//! with `gax::algebra!`).
//!
//! The algebra has the unit sphere as its metric: `x`, `y` and `z` square to one and `w` to minus
//! one. A point `x` of the sphere is the null vector `x + w`, and a circle on the sphere, where
//! the plane `n · x = s` cuts it, is the vector `n + s w`, scaled to square to one: a point lies
//! on the circle exactly when its inner product with it is zero. Two circles touch from outside
//! exactly when their inner product is minus one, and then their sum is the null vector of the
//! point where they touch. Read as a point, a circle's vector is the apex of the cone that
//! touches the sphere along it, so circles touching along every edge of a triangulation are the
//! vertices of a polyhedron whose edges all touch the sphere. The circle through the three
//! touching points of a face is orthogonal to its vertices' circles, the dual of their wedge
//! with the metric (`hodge`, numga's `dual`; gax's metric-free `dual` differs from it in the sign
//! of `w`, which squares to minus one here, and is not orthogonal to them), and its apex is a
//! vertex of the dual polyhedron, which touches the sphere at the same points.
//!
//! The equations fix the circles up to the maps that keep the sphere, the rotors of the algebra.
//! They are solved by Gauss-Newton steps of least norm, each a sparse least-squares solve of the
//! equations' Jacobian (a field map of linear forms, `c | Vector::slot()`), where numga calls
//! scipy's `least_squares`. The centred circles have their touching points balanced about the
//! centre: the boost that carries the centre seen from which they balance onto `w`, found by
//! Newton's method on gax's derivatives (numga: scipy's BFGS).
//!
//! The animation carries the centred polyhedron through boosts whose direction turns about the
//! `z` axis: left the polyhedron (blue) and its dual (orange) about the sphere, right their
//! circles on the sphere, the near side of each.

use gax_numga_examples::shared::surfaces_hull;

gax::algebra! {
    algebra sphere "The algebra of the unit sphere, R(3,1): `x`, `y` and `z` square to one, `w` to minus one.";
    basis ex = 1, ey = 1, ez = 1, ew = -1;
    kind Scalar = [1];
    versor Vector = [ex, ey, ez, ew];
    kind Bivector = [eyz, ezx, exy, exw, eyw, ezw];
    kind Trivector = [eyzw, ezxw, exyw, exyz];
    kind Pseudoscalar = [exyzw];
    versor Rotor = [1, eyz, ezx, exy, exw, eyw, ezw, exyzw];
}

use gax::field::Field;
use gax_numga_examples::label_under;
use gax_numga_examples::mesh::{Mesh, Vector3};
use gax_numga_examples::plot::panel_row;
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::signal::phasor;
use gax_numga_examples::{Anim, Canvas, Light, Marker, Point3, Rect, Scene3, backdrop, caption};
use gax_numga_examples::{palette, run};
use koebe::{Circle, apexes, face_circles, spatial, touching};
use std::sync::OnceLock;
use surfaces_hull::{beside, camera, drawn, facing, ghost, near_side};

mod koebe {
    use super::sphere::{Rotor, Scalar, Vector, intrinsic, oriented};
    use gax::field::{Field, SparseMap};
    use gax::{Orientation, vga3d};
    use gax_numga_examples::mesh::Vector3;
    use gax_numga_examples::signal::phasor;

    /// A circle on the sphere, a point of the sphere (null), or a point of space at unit weight
    /// along `w`: a vector of the algebra.
    pub type Circle = Vector;
    /// A face's circle, orthogonal to its vertices' circles: the dual of their wedge, so
    /// intrinsic where theirs are extrinsic.
    pub type FaceCircle = intrinsic::Vector;
    /// A boost or a turn: a rotor of the algebra.
    pub type Boost = Rotor;
    /// A linear form on circles, `c | Vector::slot()`: a row of the Jacobian.
    pub type CircleForm = Scalar<(Vector,)>;

    /// The unit timelike `w`, the centre of the sphere.
    pub const W: Circle = Vector::new(0.0, 0.0, 0.0, 1.0);

    /// A direction of space as a vector of the algebra.
    pub fn lifted(d: Vector3) -> Circle {
        Vector::new(d.e1(), d.e2(), d.e3(), 0.0)
    }

    /// The spatial part of a point at unit weight along `w`: where it lies in space.
    pub fn spatial<O: Orientation>(c: oriented::Vector<(), f64, O>) -> Vector3 {
        Vector3::new(c.ex(), c.ey(), c.ez()) * (1.0 / c.ew())
    }

    /// The circle on the sphere around a unit direction at an angular radius: the plane at the
    /// cosine along the direction, `direction + cos(radius) w`, scaled to square to one.
    pub fn cap(direction: Vector3, radius: f64) -> Circle {
        let turn = phasor(radius);
        (lifted(direction) + W * turn.e20()) * (1.0 / turn.e01())
    }

    /// The null vector of the point where the circles at the ends of each edge touch: their
    /// sum.
    pub fn touching(circles: &Field<Circle>, edges: &[[usize; 2]]) -> Field<Circle> {
        let sums = SparseMap::from_columns(edges, circles.len(), |_, _| Scalar::new(1.0));
        &sums * circles
    }

    /// The circle of each face, through the points where its vertices' circles touch:
    /// orthogonal to all three, the Hodge dual of their wedge.
    pub fn face_circles(circles: &Field<Circle>, faces: &[[usize; 3]]) -> Field<FaceCircle> {
        let circle =
            |&[a, b, c]: &[usize; 3]| (circles.site(a) ^ circles.site(b) ^ circles.site(c)).hodge();
        faces.iter().map(circle).collect()
    }

    /// The apex of the cone touching the sphere along each circle, as a point at unit weight
    /// along `w`.
    pub fn apexes<O: Orientation>(
        circles: &Field<oriented::Vector<(), f64, O>>,
    ) -> Field<oriented::Vector<(), f64, O>> {
        circles.map(|c| c * (-1.0 / (c | W).s()))
    }

    /// The rotor carrying a unit timelike centre onto `w`: a boost along the centre's spatial
    /// part.
    pub fn centring(centre: Circle) -> Boost {
        (Rotor::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) - W * centre).normalized()
    }

    /// The boost along a spatial unit direction by a rapidity: points of the sphere crowd
    /// towards the direction, circles there shrink and circles opposite grow.
    pub fn boost(direction: Vector3, rapidity: f64) -> Boost {
        ((lifted(direction) ^ W) * (rapidity / 2.0))
            .exp()
            .into_inner()
    }

    /// The pairs of circles the equations of a Koebe polyhedron hold, with the inner product
    /// each asks for: every circle with itself, one (it squares to one), then the circles at the
    /// ends of every edge, minus one (they touch from outside).
    struct Equations {
        pairs: Vec<[usize; 2]>,
        products: Vec<f64>,
    }

    impl Equations {
        fn new(count: usize, edges: &[[usize; 2]]) -> Equations {
            let unit = (0..count).map(|v| ([v, v], 1.0));
            let touch = edges.iter().map(|&edge| (edge, -1.0));
            let (pairs, products) = unit.chain(touch).unzip();
            Equations { pairs, products }
        }

        /// How far each equation is from holding: the pair's inner product less the one asked.
        fn residuals(&self, circles: &Field<Circle>) -> Field<Scalar> {
            let residual = |(&[a, b], &product): (&[usize; 2], &f64)| {
                (circles.site(a) | circles.site(b)) - Scalar::new(product)
            };
            self.pairs
                .iter()
                .zip(&self.products)
                .map(residual)
                .collect()
        }

        /// The equations' Jacobian: the first-order change of each pair's inner product under a
        /// change of the circles, `c_b | dc_a + c_a | dc_b`, a field map of linear forms (a circle
        /// paired with itself couples to itself twice, `2 c | dc`).
        fn jacobian(&self, circles: &Field<Circle>) -> SparseMap<CircleForm> {
            SparseMap::from_columns(&self.pairs, circles.len(), |r, end| {
                circles.site(self.pairs[r][1 - end]) | Circle::slot()
            })
        }
    }

    /// Gauss-Newton steps at most, and the residual at which they stop.
    const STEPS: usize = 40;
    const TOLERANCE: f64 = 1e-14;

    /// The circles near the start that square to one and touch from outside along every edge:
    /// Gauss-Newton steps, each the least-norm solution of the linearized equations, which
    /// leaves the free rotors of the algebra alone.
    pub fn koebe(start: Field<Circle>, edges: &[[usize; 2]]) -> Field<Circle> {
        let equations = Equations::new(start.len(), edges);
        let mut circles = start;
        for _ in 0..STEPS {
            let residuals = equations.residuals(&circles);
            if residuals.iter().all(|r| r.s().abs() < TOLERANCE) {
                break;
            }
            circles = &circles + &equations.jacobian(&circles).lstsq(&-&residuals);
        }
        circles
    }

    /// How far null points are from balancing about the point `s + w` of the unit ball (Klein's
    /// model of the hyperboloid): `Σ ln(-x̂ · p)` with `x̂` that point scaled to a unit timelike
    /// vector, least where the points balance about `x̂`. Generic over the coefficients, for
    /// gax's derivatives.
    fn imbalance<T: gax::Real>(points: &[Circle], s: vga3d::Vector<(), T>) -> T {
        let x = Vector::new(s.e1(), s.e2(), s.e3(), T::one());
        let pairings = points
            .iter()
            .map(|&p| (-(x | p.map_coefs(T::from_f64)).s()).ln());
        let scale = (-x.scalar_product(x).s()).ln() * T::from_f64(points.len() as f64 / 2.0);
        pairings.fold(T::zero(), |sum, l| sum + l) - scale
    }

    /// Newton steps at most, and the step at which they stop.
    const NEWTON_STEPS: usize = 40;
    const NEWTON_TOLERANCE: f64 = 1e-15;

    /// The unit timelike centre seen from which null points of the sphere balance: the least
    /// imbalance, by Newton's method, each step halved while it does not lower the imbalance or
    /// leaves the ball.
    pub fn centre(points: &Field<Circle>) -> Circle {
        let points = points.sites();
        let mut s = vga3d::Vector::zero();
        for _ in 0..NEWTON_STEPS {
            let (value, gradient, curvature) = gax::jet::hessian(|s| imbalance(points, s), s);
            let newton = -curvature.solve(gradient);
            let mut length = 1.0;
            let worse = |step: vga3d::Vector| {
                (s + step).norm_squared() >= 1.0 || imbalance(points, s + step) > value
            };
            while worse(newton * length) {
                length /= 2.0;
            }
            let step = newton * length;
            s += step;
            if step.norm() < NEWTON_TOLERANCE {
                break;
            }
        }
        Vector::new(s.e1(), s.e2(), s.e3(), 1.0).normalized()
    }
}

mod scenes {
    use super::koebe::*;
    use super::surfaces_hull::{POINTS, directions, uneven_faces};
    use gax::field::Field;
    use gax_numga_examples::mesh::{Mesh, Vector3};
    use gax_numga_examples::signal::{phasor, wave};

    /// The boosts the centred polyhedron is carried through: their largest rapidity, and the
    /// tilt towards the `z` axis of the direction they turn about it through.
    pub const RAPIDITY: f64 = 0.6;
    pub const TILT: f64 = 0.4;

    /// The circles of the Koebe polyhedron of the uneven triangulation, from caps around the
    /// directions half as wide as the mean angle to their neighbours, centred: the mesh of the
    /// circles.
    pub fn centred() -> Mesh<Circle> {
        let directions = directions(POINTS);
        let faces = uneven_faces(&directions);
        let mesh = Mesh::new(directions.clone(), faces.clone());
        let mut angles = vec![0.0; POINTS];
        for &[a, b] in &mesh.edges {
            let between = directions.site(a).angle_between(directions.site(b));
            angles[a] += between;
            angles[b] += between;
        }
        let valences = mesh.valences();
        let radius = |v: usize| angles[v] / valences[v] as f64 / 2.0;
        let caps = Field::from_fn(POINTS, |v| cap(directions.site(v), radius(v)));
        let circles = koebe(caps, &mesh.edges);
        let centring = centring(centre(&touching(&circles, &mesh.edges)));
        Mesh::new(circles.map(|c| centring >> c), faces)
    }

    /// The boost at a phase of one turn: its direction turns once about the `z` axis, tilted
    /// towards it, with a rapidity rising and falling twice per turn.
    pub fn boost_at(phase: f64) -> (Boost, f64) {
        let around = phasor(phase);
        let direction = Vector3::new(around.e20(), around.e01(), TILT).normalized();
        let rapidity = RAPIDITY * wave(phase).powi(2);
        (boost(direction, rapidity), rapidity)
    }

    /// The centred circles carried by the boost at a phase.
    pub fn boosted(circles: &Field<Circle>, phase: f64) -> Field<Circle> {
        let (rotor, _) = boost_at(phase);
        circles.map(|c| rotor >> c)
    }
}

use scenes::*;

/// The centred polyhedron, and the faces either side of each of its edges, the edges of the
/// dual polyhedron; computed once.
struct Shown {
    mesh: Mesh<Circle>,
    dual_edges: Vec<[usize; 2]>,
}

fn shown() -> &'static Shown {
    static SHOWN: OnceLock<Shown> = OnceLock::new();
    SHOWN.get_or_init(|| {
        let mesh = centred();
        let dual_edges = beside(&mesh);
        Shown { mesh, dual_edges }
    })
}

/// One turn of the boosts: numga's 48 frames of 80 ms.
const SECONDS: f32 = 3.84;
/// The points drawn around each circle on the sphere.
const CIRCLE_SAMPLES: usize = 64;
/// The widths of the polyhedron's edges, of its dual's, and of the circles; and the size of the
/// touching points.
const EDGE_WIDTH: f32 = 1.6;
const DUAL_WIDTH: f32 = 1.3;
const CIRCLE_WIDTH: f32 = 1.2;
const TOUCH_SIZE: f32 = 3.0;

fn primal() -> Light {
    palette::SKY
}
fn dual() -> Light {
    palette::ORANGE
}

/// The polyhedron of the circles' apexes and its dual, the polyhedron of the face circles'
/// apexes, every edge of both touching the sphere where the circles of its ends touch; only the
/// edges whose touching point faces the viewer.
fn polyhedra(c: &mut Canvas, rect: Rect, circles: &Field<Circle>) {
    let s = shown();
    let cam = camera(rect);
    ghost(c, &cam, rect);
    let toward = facing(&cam);
    let touch = touching(circles, &s.mesh.edges);
    let near: Vec<bool> = touch
        .iter()
        .map(|t| (spatial(t) | toward).s() > 0.0)
        .collect();
    let corners = apexes(circles);
    let duals = apexes(&face_circles(circles, &s.mesh.faces));
    let (corners, duals) = (corners.map(spatial), duals.map(spatial));
    let at = |points: &Field<Vector3>, v: usize| drawn(points.site(v));
    panel3(c, rect, cam, |scene: &mut Scene3| {
        for (e, &[a, b]) in s.mesh.edges.iter().enumerate().filter(|(e, _)| near[*e]) {
            scene.seg(at(&corners, a), at(&corners, b), EDGE_WIDTH, primal());
            let [f, g] = s.dual_edges[e];
            scene.seg(at(&duals, f), at(&duals, g), DUAL_WIDTH, dual());
            let touching_point = drawn(spatial(touch.site(e)));
            scene.dot(touching_point, Marker::Dot, TOUCH_SIZE, palette::INK);
        }
    });
}

/// Points around a circle on the unit sphere: the plane `n · x = s` of the circle `n + s w`
/// (squaring to one) cuts the sphere at the height `s / |n|` along `n`, in a circle of radius
/// `1 / |n|`.
fn circle_points<O: gax::Orientation>(
    circle: sphere::oriented::Vector<(), f64, O>,
) -> Vec<Vector3> {
    let n = Vector3::new(circle.ex(), circle.ey(), circle.ez());
    let (length, axis) = (n.norm(), n.normalized());
    let centre = axis * (circle.ew() / length);
    let radius = circle.norm() / length;
    // Two directions across the axis, from whichever of x and y is further from it.
    let away = if axis.e1().abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let across = ((axis ^ away) | axis).normalized();
    // The third direction, across both: their plane times the unit volume.
    let other = (across ^ axis) * gax::vga3d::Pseudoscalar::new(1.0);
    (0..=CIRCLE_SAMPLES)
        .map(|k| {
            let turn = phasor(core::f64::consts::TAU * k as f64 / CIRCLE_SAMPLES as f64);
            centre + (across * turn.e20() + other * turn.e01()) * radius
        })
        .collect()
}

/// The vertex circles and the face circles on the sphere, their near side.
fn pattern(c: &mut Canvas, rect: Rect, circles: &Field<Circle>) {
    let s = shown();
    let cam = camera(rect);
    ghost(c, &cam, rect);
    let toward = facing(&cam);
    let duals = face_circles(circles, &s.mesh.faces);
    panel3(c, rect, cam, |scene: &mut Scene3| {
        let primals = circles.iter().map(|c| (circle_points(c), primal()));
        let rings = primals.chain(duals.iter().map(|c| (circle_points(c), dual())));
        for (ring, light) in rings {
            let stroke: Vec<Point3> = ring.into_iter().map(|p| near_side(p, toward)).collect();
            scene.polyline(&stroke, CIRCLE_WIDTH, light);
        }
    });
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let s = shown();
    let phase = core::f64::consts::TAU * f64::from(t / SECONDS);
    let circles = boosted(&s.mesh.vertices, phase);
    let [left, right] = panel_row(c, 6.0);
    polyhedra(c, left, &circles);
    pattern(c, right, &circles);
    let (_, rapidity) = boost_at(phase);
    label_under(
        c,
        left,
        &format!("THE POLYHEDRON AND ITS DUAL, BOOSTED BY RAPIDITY {rapidity:.2}"),
        30.0,
        0.9,
    );
    label_under(
        c,
        right,
        "THEIR CIRCLES, TOUCHING ALONG EVERY EDGE",
        30.0,
        0.9,
    );
    caption(
        c,
        "KOEBE POLYHEDRA: EDGES THAT ALL TOUCH THE SPHERE",
        "CIRCLES ARE VECTORS OF R(3,1), TOUCHING AT INNER PRODUCT -1; THE ALGEBRA'S BOOSTS KEEP THEM TOUCHING",
    );
}

fn main() {
    run(Anim::new("koebe", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::koebe::*;
    use super::scenes::*;
    use super::surfaces_hull::{POINTS, beside};
    use gax::field::Field;
    use gax_numga_examples::mesh::Vector3;

    /// numga's frames of one turn of the boosts.
    const FRAMES: usize = 48;

    /// The distance from the centre to the line of each edge between apexes: the area of the
    /// parallelogram of its ends over its length.
    fn edge_distances<O: gax::Orientation>(
        apexes: &Field<super::sphere::oriented::Vector<(), f64, O>>,
        edges: &[[usize; 2]],
    ) -> Vec<f64> {
        let at = |v: usize| spatial(apexes.site(v));
        edges
            .iter()
            .map(|&[a, b]| (at(a) ^ at(b)).norm() / (at(b) - at(a)).norm())
            .collect()
    }

    fn all_near(values: impl IntoIterator<Item = f64>, target: f64, tolerance: f64) -> bool {
        values.into_iter().all(|v| (v - target).abs() < tolerance)
    }

    #[test]
    fn circles_touch_along_every_edge_and_both_polyhedra_touch_the_sphere_under_every_boost() {
        let mesh = centred();
        let dual_edges = beside(&mesh);
        for frame in 0..FRAMES {
            let circles = boosted(
                &mesh.vertices,
                core::f64::consts::TAU * frame as f64 / FRAMES as f64,
            );
            assert!(all_near(
                circles.iter().map(|c| c.scalar_product(c).s()),
                1.0,
                1e-10
            ));
            let pairs = mesh
                .edges
                .iter()
                .map(|&[a, b]| (circles.site(a) | circles.site(b)).s());
            assert!(all_near(pairs, -1.0, 1e-10));
            assert!(all_near(
                edge_distances(&apexes(&circles), &mesh.edges),
                1.0,
                1e-10
            ));
            let duals = face_circles(&circles, &mesh.faces);
            assert!(all_near(
                edge_distances(&apexes(&duals), &dual_edges),
                1.0,
                1e-10
            ));
            // numga's further check: every face circle is orthogonal to its vertices' circles.
            for (f, corners) in mesh.faces.iter().enumerate() {
                let dual = duals.site(f);
                let orthogonal = corners.map(|v| (dual | circles.site(v)).s().abs() / dual.norm());
                assert!(orthogonal.iter().all(|&o| o < 1e-10));
            }
        }
    }

    #[test]
    fn the_centred_circles_balance_and_the_figure_draws() {
        let mesh = centred();
        let touch = touching(&mesh.vertices, &mesh.edges);
        let mean = touch
            .iter()
            .map(spatial)
            .fold(Vector3::zero(), |sum, p| sum + p);
        assert!((mean * (1.0 / touch.len() as f64)).norm() < 1e-4);
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
    }

    /// The hull the triangulation comes from uses every point, and is closed: two faces at each
    /// edge, and as many faces as Euler's formula gives a sphere.
    #[test]
    fn every_point_is_a_vertex_of_the_triangulation() {
        let mesh = centred();
        assert_eq!(mesh.faces.len(), 2 * POINTS - 4);
        assert_eq!(beside(&mesh).len(), mesh.edges.len());
    }
}
