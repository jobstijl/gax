//! numga's `quadrics/spherical_quadrics`: a spherical conic in `Cl(3)` (gax's `vga3d`), where the
//! planes through the origin are the vectors and the points of the sphere their poles, the
//! bivectors. A polarity `C` maps each point to its polar plane, and the conic is where a point
//! lies on its own polar, `P ∨ C(P) = 0`: a quadratic cone through the origin, cut by the sphere.
//! The curve is a spherical ellipse, the sum of its geodesic distances to two foci constant. Read
//! through planes instead, the same curve is the envelope of its tangent great circles, the
//! planes with `π ∨ Q(π) = 0` for the inverse `Q`, and the product of the sines of the foci's
//! distances to those circles is constant. The level sets of `P ∨ C(P)` are Poinsot's polhodes.
//! The animation turns the three views about the sphere while a point runs along the oval with
//! its two geodesics to the foci and its tangent great circle.

use gax::vga3d::{Bivector, Vector};
use gax_colour::Light;
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Marker, ORIGIN3, Point2, backdrop, caption, colormap,
    contour, palette, run,
};

mod conic {
    use super::*;

    /// A point of the sphere: the pole of a plane, on `e23`, `e31`, `e12`, the poles of the
    /// planes x, y and z. It is the dual of the vector of its coordinates, which `undual`
    /// recovers.
    pub type P = Bivector<(), f64>;
    /// A plane through the origin: a great circle.
    pub type Plane = Vector<(), f64>;
    /// A polarity: a point to its polar plane.
    pub type Polarity = Vector<(Bivector,), f64>;
    /// A dual polarity: a plane to its pole.
    pub type DualPolarity = Bivector<(Vector,), f64>;
    pub type R = gax::Unit<gax::vga3d::Rotor<(), f64>>;

    /// The basis planes x, y and z.
    pub fn planes() -> [Plane; 3] {
        [
            Vector::new(1.0, 0.0, 0.0),
            Vector::new(0.0, 1.0, 0.0),
            Vector::new(0.0, 0.0, 1.0),
        ]
    }

    /// The polarity with the given eigenvalues on the basis planes: each plane paired with an
    /// open point, weighted.
    pub fn polarity(eigenvalues: [f64; 3]) -> Polarity {
        let p = Bivector::slot();
        let [a, b, c] = planes();
        (a & p) * a.gp(eigenvalues[0])
            + (b & p) * b.gp(eigenvalues[1])
            + (c & p) * c.gp(eigenvalues[2])
    }

    /// The point at longitude `phi` and polar angle `theta`: the pole `e12` turned toward x by
    /// the polar angle, then about z by the longitude.
    pub fn sphere(phi: f64, theta: f64) -> P {
        let turn =
            Bivector::new(0.0, 0.0, -phi / 2.0).exp() * Bivector::new(0.0, -theta / 2.0, 0.0).exp();
        turn >> Bivector::new(0.0, 0.0, 1.0)
    }

    /// The semi-axes of the cone's ellipse: the cone of a diagonal polarity with eigenvalues
    /// `l1 > l2 > 0 > l3` meets the sphere above the ellipse `x²/x0² + y²/y0² = 1` of the
    /// xy-plane.
    fn semi_axes([l1, l2, l3]: [f64; 3]) -> (f64, f64) {
        ((-l3 / (l1 - l3)).sqrt(), (-l3 / (l2 - l3)).sqrt())
    }

    /// The quadratic cone `P ∨ C(P) = 0` of a diagonal polarity, as the point at `radius` and
    /// parameter `t`.
    pub fn cone(eigenvalues: [f64; 3], radius: f64, t: f64) -> P {
        let (x0, y0) = semi_axes(eigenvalues);
        let (x, y) = (x0 * t.cos(), y0 * t.sin());
        Bivector::new(x, y, (1.0 - x * x - y * y).sqrt()).gp(radius)
    }

    /// The turn from unit point `start` to unit point `end`, as the logarithm of the rotor
    /// `end / start`: that rotor turns `start` through twice their angle, so the logarithm's norm
    /// is the angle and half of it the turn.
    fn turn(start: P, end: P) -> Bivector<(), f64> {
        (end / start).normalized().log()
    }

    /// The great-circle arc from `start` to `end`, at fraction `t` of its length: a fraction of
    /// the turn between them.
    pub fn geodesic(start: P, end: P, t: f64) -> P {
        turn(start, end).gp(t / 2.0).exp() >> start
    }

    /// The great circle of a plane, swept from a point on it by the rotation about the plane's
    /// normal, its dual.
    pub fn great_circle(plane: Plane, through: P, angle: f64) -> P {
        plane.dual().gp(angle / 2.0).exp() >> through
    }

    /// The spherical distance between unit points: the angle of the turn between them.
    pub fn distance(a: P, b: P) -> f64 {
        turn(a, b).norm()
    }

    /// The conic in the world: its forms, its points, its foci and its tangent great circles.
    pub struct Conic {
        pub eigenvalues: [f64; 3],
        pub rotor: R,
        pub c: Polarity,
        pub q: DualPolarity,
        pub curve: Vec<P>,
        pub foci: [P; 2],
        pub theta_a: f64,
        pub tangents: Vec<Plane>,
    }

    /// In principal axes both the polarity `C` and the dual polarity `Q` are diagonal, and one
    /// rotor moves either by the same sandwich. Points along the oval at `n` parameters, its
    /// foci on the major axis, and its semi-major arc.
    pub fn spherical_conic(n: usize) -> Conic {
        let eigenvalues = [1.0, 0.4, -0.8];
        let rotor = Bivector::new(0.15, 0.0, 0.25).exp();
        let c = rotor >> polarity(eigenvalues).of(rotor << Bivector::slot());
        let q = rotor >> polarity(eigenvalues).solve(rotor << Vector::slot());
        // The semi-axes as arcs, the semi-minor along x and the semi-major along y, and the focal
        // arc along the major axis.
        let (x0, y0) = semi_axes(eigenvalues);
        let (theta_b, theta_a) = (x0.asin(), y0.asin());
        let theta_c = (theta_a.cos() / theta_b.cos()).acos();
        let curve: Vec<P> = (0..n)
            .map(|k| {
                rotor
                    >> cone(
                        eigenvalues,
                        1.0,
                        core::f64::consts::TAU * k as f64 / (n - 1) as f64,
                    )
            })
            .collect();
        let foci =
            [1.0, -1.0].map(|s| rotor >> Bivector::new(0.0, s * theta_c.sin(), theta_c.cos()));
        // The polar plane of a point of the oval is its tangent great circle.
        let tangents = curve
            .iter()
            .map(|p| c.of(*p).normalized().into_inner())
            .collect();
        Conic {
            eigenvalues,
            rotor,
            c,
            q,
            curve,
            foci,
            theta_a,
            tangents,
        }
    }

    impl Conic {
        /// The potential `P ∨ C(P)` at a point.
        pub fn potential(&self, p: P) -> f64 {
            (p & self.c.of(p)).s()
        }

        /// The dual focal product at a tangent plane: the sines of the foci's distances to the
        /// great circle, multiplied.
        pub fn dual_product(&self, tangent: Plane) -> f64 {
            (self.foci[0] & tangent).s().abs() * (self.foci[1] & tangent).s().abs()
        }
    }
}

use conic::*;

const SECONDS: f32 = 12.0;

/// Where a point of the sphere is drawn: the origin plus the vector whose dual it is.
fn drawn(p: P) -> gax::pga3d::Point<(), f64> {
    let v = p.undual();
    gax::pga3d::Point::xyz(0.0, 0.0, 0.0) + gax::pga3d::Point::direction(v.e1(), v.e2(), v.e3())
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let conic = spherical_conic(200);
    let tau = core::f32::consts::TAU;
    let azimuth = 0.6 + tau * t / SECONDS;
    // The panels, side by side under the caption.
    let panels = screen.inset(0.0, screen.height() * 0.12, 0.0, 0.0);
    let titles = [
        "POINTS: D(P,F1) + D(P,F2) CONSTANT",
        "PLANES: ENVELOPE OF GREAT CIRCLES",
        "THE CONE AND THE POLHODES",
    ];
    // The point running along the oval.
    let k = ((t / SECONDS * 2.0).fract() * 199.0) as usize;
    let sample = conic.curve[k];
    let gold = Light::from_srgb(1.0, 0.8, 0.1, 1.7);
    let oval: Vec<_> = conic.curve.iter().map(|p| drawn(*p)).collect();
    // The sphere over the unit square of longitude and polar angle, and the potential over it,
    // for its colours and its level lines.
    let grid = |u: f32, v: f32| {
        sphere(
            f64::from(u) * std::f64::consts::TAU,
            f64::from(v) * std::f64::consts::PI,
        )
    };
    let pot = |u: f32, v: f32| conic.potential(grid(u, v));
    let on_sphere = |u: f32, v: f32| drawn(grid(u, v));
    // The same at a point of the unit square, where the level lines are found.
    let at = |uv: Point2| {
        let [u, v] = uv.to_euclidean();
        grid(u, v)
    };
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for i in 0..=60 {
        for j in 0..=30 {
            let v = pot(i as f32 / 60.0, j as f32 / 30.0);
            (lo, hi) = (lo.min(v), hi.max(v));
        }
    }
    for (i, title) in titles.iter().enumerate() {
        let rect = panels.column(i, 3);
        let cam = Camera::orbit(
            rect.width() as usize,
            rect.height() as usize,
            ORIGIN3,
            6.0,
            azimuth,
            0.45,
            Lens::Parallel(1.45),
        );
        panel3(c, rect, cam, |s| match i {
            0 => {
                // The potential on the sphere, the oval, its antipodal loop, the foci, and the
                // geodesics from the running point to both foci.
                s.surface(
                    on_sphere,
                    36,
                    18,
                    |u, v| (colormap::coolwarm(((pot(u, v) - lo) / (hi - lo)) as f32)).faded(0.45),
                    0.45,
                    None,
                );
                s.polyline(&oval, 3.0, gold);
                let anti: Vec<_> = conic.curve.iter().map(|p| drawn(-*p)).collect();
                s.polyline(&anti, 1.5, gold.faded(0.6));
                for (f, colour) in conic.foci.iter().zip([palette::green(), palette::sky()]) {
                    s.dot(drawn(*f), Marker::Dot, 9.0, palette::red());
                    let arc: Vec<_> = (0..=30)
                        .map(|j| drawn(geodesic(*f, sample, j as f64 / 30.0)))
                        .collect();
                    s.polyline(&arc, 2.2, colour);
                }
                s.dot(drawn(sample), Marker::Ring, 11.0, palette::ink());
            }
            1 => {
                // The envelope: sixteen tangent great circles, the running one bright with its
                // normal.
                s.sphere_wire(ORIGIN3, 1.0, 16, palette::grid().faded(0.25));
                s.polyline(&oval, 3.0, gold);
                // The great circle of a tangent plane, through its point of the oval.
                let circle = |idx: usize| -> Vec<_> {
                    (0..=90)
                        .map(|a| {
                            let angle = tau as f64 * a as f64 / 90.0;
                            drawn(great_circle(conic.tangents[idx], conic.curve[idx], angle))
                        })
                        .collect()
                };
                for j in 0..16 {
                    let colour = colormap::inferno(0.2 + 0.7 * j as f32 / 15.0);
                    s.polyline(&circle(j * 199 / 15), 1.0, colour.faded(0.45));
                }
                s.polyline(&circle(k), 2.4, palette::ink());
                // The tangent plane's normal: the plane itself, a vector.
                s.arrow(
                    drawn(sample),
                    conic.tangents[k].gp(0.4),
                    1.6,
                    7.0,
                    palette::ink(),
                );
                // The dual conic: the normals of the planes with `π ∨ Q(π) = 0`, a level line
                // of the dual form over the sphere of unit normals (each the plane whose pole is
                // that point of the sphere).
                let dual = |uv: Point2| {
                    let plane = at(uv).undual();
                    (plane & conic.q.of(plane)).s() as f32
                };
                for [a, b] in contour::of_fn(dual, [0.0, 1.0], [0.0, 1.0], 120, 0.0) {
                    s.seg(drawn(at(a)), drawn(at(b)), 2.0, palette::purple());
                }
                s.dot(drawn(sample), Marker::Ring, 11.0, palette::ink());
            }
            _ => {
                // The cone through the oval, and the polhodes: level lines of the potential.
                s.sphere_wire(ORIGIN3, 1.0, 16, palette::grid().faded(0.25));
                let ev = conic.eigenvalues;
                let rotor = conic.rotor;
                s.surface(
                    |u, v| {
                        let r = 0.1 + 1.15 * f64::from(u);
                        drawn(rotor >> cone(ev, r, f64::from(v) * std::f64::consts::TAU))
                    },
                    12,
                    40,
                    |_, _| Light::from_srgb(0.94, 0.9, 0.55, 0.4),
                    0.3,
                    None,
                );
                s.polyline(&oval, 3.0, gold);
                let salmon = Light::from_srgb(0.98, 0.5, 0.45, 1.6);
                for (j, fraction) in [0.7, 0.4667, 0.2333].iter().enumerate() {
                    for (level, colour) in [
                        (lo * fraction, salmon),
                        (hi * (0.7 - 0.2333 * j as f64), palette::sky()),
                    ] {
                        let potential = |uv| conic.potential(at(uv)) as f32;
                        let segs =
                            contour::of_fn(potential, [0.0, 1.0], [0.0, 1.0], 90, level as f32);
                        for [a, b] in segs {
                            s.seg(drawn(at(a)), drawn(at(b)), 1.4, colour.faded(0.85));
                        }
                    }
                }
            }
        });
        // Sized to fit the panel's width, in its top left corner.
        let size = (rect.width() / 35.0).clamp(7.0, 13.0);
        let corner = rect.lo + Point2::direction(size, size * 1.6);
        c.text(title, corner, size, palette::ink(), Align::Left);
    }
    let d = distance(sample, conic.foci[0]) + distance(sample, conic.foci[1]);
    caption(
        c,
        "SPHERICAL QUADRICS: A CONE CUT BY THE SPHERE",
        &format!(
            "P V C(P) = 0: D1 + D2 = {d:.4}, 2 THETA_A = {:.4}, SIN PRODUCT {:.4}",
            2.0 * conic.theta_a,
            conic.dual_product(conic.tangents[k])
        ),
    );
}

fn main() {
    run(
        Anim::new("spherical quadrics", SECONDS).size(960, 540),
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::conic::*;
    use gax::ApproxEq;
    use gax::vga3d::{Bivector, Vector};

    /// A point of the sphere is the dual of the vector of its coordinates.
    #[test]
    fn points_are_duals_of_their_coordinates() {
        let p = Bivector::new(0.36, 0.48, 0.8);
        assert_eq!(p.undual(), Vector::new(0.36, 0.48, 0.8));
        assert_eq!(Vector::new(0.36, 0.48, 0.8).dual(), p);
    }

    /// The polarity maps each basis point to its polar basis plane, scaled by its eigenvalue.
    #[test]
    fn polarity_is_diagonal_on_the_basis() {
        let eigenvalues = [1.0, 0.4, -0.8];
        let c = polarity(eigenvalues);
        for i in 0..3 {
            let mut e = [0.0; 3];
            e[i] = 1.0;
            for (j, plane) in planes().iter().enumerate() {
                let g = (c.of(Bivector::<(), f64>::from_coeffs(e)) | *plane).s();
                let want = if i == j { eigenvalues[i] } else { 0.0 };
                assert!((g - want).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn geodesics_and_great_circles_stay_on_the_sphere() {
        let (a, b) = (Bivector::new(1.0, 0.0, 0.0), Bivector::new(0.0, 0.6, 0.8));
        for k in 0..7 {
            let p = geodesic(a, b, k as f64 / 6.0);
            assert!((-(p | p).s() - 1.0).abs() < 1e-12);
        }
        assert!((-(geodesic(a, b, 1.0) | b).s() - 1.0).abs() < 1e-12);
        let plane = planes()[2];
        for k in 0..9 {
            let p = great_circle(plane, a, core::f64::consts::TAU * k as f64 / 8.0);
            assert!((p & plane).s().abs() < 1e-12);
        }
    }

    /// The longitude-latitude grid, built by rotors as numga's, is the usual parametrization.
    #[test]
    fn the_sphere_grid_is_spherical_coordinates() {
        for (phi, theta) in [(0.3f64, 0.7f64), (2.0, 1.2), (4.0, 2.9)] {
            let want = Vector::new(
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            );
            assert!(sphere(phi, theta).undual().max_abs_diff(&want) < 1e-14);
        }
    }

    /// The scenario's checks: the oval lies on its cone, the focal sum is twice the semi-major
    /// arc, the tangent planes lie on the dual conic, and the dual focal product is constant.
    /// The foci agree with numga's.
    #[test]
    fn the_spherical_conic_and_its_focal_properties() {
        let conic = spherical_conic(200);
        let product = conic.dual_product(conic.tangents[0]);
        for (p, t) in conic.curve.iter().zip(&conic.tangents) {
            assert!(conic.potential(*p).abs() < 1e-13);
            let sum = distance(*p, conic.foci[0]) + distance(*p, conic.foci[1]);
            assert!((sum - 2.0 * conic.theta_a).abs() < 1e-11);
            assert!((*t & conic.q.of(*t)).s().abs() < 1e-12);
            assert!((conic.dual_product(*t) - product).abs() < 1e-14);
        }
        assert!((conic.theta_a - 0.9553166181245093).abs() < 1e-14);
        assert!((product - 0.26666666666523).abs() < 1e-11);
        let numga = [
            Vector::new(0.35507764519436, 0.747383015023582, 0.561550082122781),
            Vector::new(-0.242143096584461, -0.308517190674206, 0.919882527190074),
        ];
        for (f, want) in conic.foci.iter().zip(numga) {
            assert!(f.undual().max_abs_diff(&want) < 1e-10);
        }
        // The rotor is numga's: it carries the pole e12 to the same place. (numga's exponential
        // is good to a few 1e-12, so these compare to 1e-10.)
        let moved = conic.rotor >> Bivector::new(0.0, 0.0, 1.0);
        let numga = Vector::new(0.072898937662932, 0.283286671487454, 0.956260637399548);
        assert!(moved.undual().max_abs_diff(&numga) < 1e-10);
    }

    #[test]
    fn a_frame_draws() {
        let mut draw = super::draw;
        let c = gax_numga_examples::app::frame(
            &gax_numga_examples::Anim::new("t", 1.0).size(480, 270),
            0.5,
            &mut draw,
        );
        assert!(c.mean().luma() > 0.0);
    }
}
