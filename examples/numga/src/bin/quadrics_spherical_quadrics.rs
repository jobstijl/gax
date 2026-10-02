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
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Marker, Scene3, backdrop, canvas, caption, colormap,
    contour, palette, run,
};

mod conic {
    use super::*;

    /// A point of the sphere: the pole of a plane, on `e23`, `e31`, `e12`.
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

    /// The point at coordinates `(x, y, z)`, on the basis points `e23`, `e31` and `e12`, the
    /// poles of the planes x, y and z.
    pub fn point([x, y, z]: [f64; 3]) -> P {
        Bivector::new(x, y, z)
    }

    /// The coordinates of a point.
    pub fn xyz(p: P) -> [f64; 3] {
        p.c
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
        point([x, y, (1.0 - x * x - y * y).sqrt()]).gp(radius)
    }

    /// The great-circle arc from `start` to `end`, at fraction `t` of its length.
    pub fn geodesic(start: P, end: P, t: f64) -> P {
        let angle = (-(start | end).s()).clamp(-1.0, 1.0).acos();
        (start.gp(((1.0 - t) * angle).sin()) + end.gp((t * angle).sin())).gp(1.0 / angle.sin())
    }

    /// The great circle of a plane, swept from a point on it by the rotation about the plane's
    /// normal, its dual.
    pub fn great_circle(plane: Plane, through: P, angle: f64) -> P {
        plane.dual().gp(angle / 2.0).exp() >> through
    }

    /// The spherical distance between unit points.
    pub fn distance(a: P, b: P) -> f64 {
        (-(a | b).s()).clamp(-1.0, 1.0).acos()
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
        let foci = [1.0, -1.0].map(|s| rotor >> point([0.0, s * theta_c.sin(), theta_c.cos()]));
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

fn f3(p: P) -> [f32; 3] {
    xyz(p).map(|v| v as f32)
}

const SECONDS: f32 = 12.0;

/// A panel's own canvas, with the backdrop the whole canvas has there.
fn panel(c: &Canvas, rect: [usize; 4]) -> Canvas {
    let mut p = Canvas::new(rect[2] - rect[0], rect[3] - rect[1]);
    let h = c.height.max(2) as f32 - 1.0;
    let at = |y: usize| canvas::mix(palette::top(), palette::bottom(), y as f32 / h);
    p.backdrop(at(rect[1]), at(rect[3] - 1));
    p
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.width, c.height);
    let conic = spherical_conic(200);
    let tau = core::f32::consts::TAU;
    let azimuth = 0.6 + tau * t / SECONDS;
    let top = (h as f32 * 0.12) as usize;
    let titles = [
        "POINTS: D(P,F1) + D(P,F2) CONSTANT",
        "PLANES: ENVELOPE OF GREAT CIRCLES",
        "THE CONE AND THE POLHODES",
    ];
    // The point running along the oval.
    let k = ((t / SECONDS * 2.0).fract() * 199.0) as usize;
    let sample = conic.curve[k];
    let gold = canvas::srgb(1.0, 0.8, 0.1);
    let oval: Vec<[f32; 3]> = conic.curve.iter().map(|p| f3(*p)).collect();
    // The potential over the sphere, for its colours and its level lines.
    let pot = |u: f32, v: f32| {
        conic.potential(sphere(
            f64::from(u) * std::f64::consts::TAU,
            f64::from(v) * std::f64::consts::PI,
        ))
    };
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for i in 0..=60 {
        for j in 0..=30 {
            let v = pot(i as f32 / 60.0, j as f32 / 30.0);
            (lo, hi) = (lo.min(v), hi.max(v));
        }
    }
    let on_sphere = |u: f32, v: f32| {
        f3(sphere(
            f64::from(u) * std::f64::consts::TAU,
            f64::from(v) * std::f64::consts::PI,
        ))
    };
    for (i, title) in titles.iter().enumerate() {
        let rect = [i * w / 3, top, (i + 1) * w / 3, h];
        let mut p = panel(c, rect);
        let cam = Camera::orbit(
            p.width,
            p.height,
            [0.0, 0.0, 0.0],
            6.0,
            azimuth,
            0.45,
            Lens::Parallel(1.45),
        );
        let mut s = Scene3::new(cam);
        match i {
            0 => {
                // The potential on the sphere, the oval, its antipodal loop, the foci, and the
                // geodesics from the running point to both foci.
                s.surface(
                    on_sphere,
                    36,
                    18,
                    |u, v| colormap::coolwarm(((pot(u, v) - lo) / (hi - lo)) as f32),
                    0.45,
                    None,
                );
                s.polyline(&oval, 3.0, gold, 1.0);
                let anti: Vec<[f32; 3]> = oval.iter().map(|p| p.map(|v| -v)).collect();
                s.polyline(&anti, 1.5, gold, 0.6);
                for (f, colour) in conic.foci.iter().zip([palette::green(), palette::sky()]) {
                    s.dot(f3(*f), Marker::Dot, 9.0, palette::red());
                    let arc: Vec<[f32; 3]> = (0..=30)
                        .map(|j| f3(geodesic(*f, sample, j as f64 / 30.0)))
                        .collect();
                    s.polyline(&arc, 2.2, colour, 1.0);
                }
                s.dot(f3(sample), Marker::Ring, 11.0, palette::ink());
            }
            1 => {
                // The envelope: sixteen tangent great circles, the running one bright with its
                // normal.
                s.sphere_wire([0.0; 3], 1.0, 16, palette::grid(), 0.25);
                s.polyline(&oval, 3.0, gold, 1.0);
                for j in 0..16 {
                    let idx = j * 199 / 15;
                    let colour = colormap::inferno(0.2 + 0.7 * j as f32 / 15.0);
                    let circle: Vec<[f32; 3]> = (0..=90)
                        .map(|a| {
                            f3(great_circle(
                                conic.tangents[idx],
                                conic.curve[idx],
                                tau as f64 * a as f64 / 90.0,
                            ))
                        })
                        .collect();
                    s.polyline(&circle, 1.0, colour, 0.45);
                }
                let circle: Vec<[f32; 3]> = (0..=90)
                    .map(|a| {
                        f3(great_circle(
                            conic.tangents[k],
                            sample,
                            tau as f64 * a as f64 / 90.0,
                        ))
                    })
                    .collect();
                s.polyline(&circle, 2.4, palette::ink(), 1.0);
                let normal = conic.tangents[k].c.map(|v| v as f32 * 0.4);
                s.arrow(f3(sample), normal, 1.6, 7.0, palette::ink());
                // The dual conic: the normals of the planes with `π ∨ Q(π) = 0`, a level line
                // of the dual form over the sphere of unit normals.
                let dual = |u: f32, v: f32| {
                    let n = xyz(sphere(
                        f64::from(u) * std::f64::consts::TAU,
                        f64::from(v) * std::f64::consts::PI,
                    ));
                    let plane = Vector::new(n[0], n[1], n[2]);
                    (plane & conic.q.of(plane)).s() as f32
                };
                for [a, b] in contour::of_fn(dual, [0.0, 1.0], [0.0, 1.0], 120, 0.0) {
                    s.seg(
                        on_sphere(a[0], a[1]),
                        on_sphere(b[0], b[1]),
                        2.0,
                        palette::purple(),
                        1.0,
                    );
                }
                s.dot(f3(sample), Marker::Ring, 11.0, palette::ink());
            }
            _ => {
                // The cone through the oval, and the polhodes: level lines of the potential.
                s.sphere_wire([0.0; 3], 1.0, 16, palette::grid(), 0.25);
                let ev = conic.eigenvalues;
                let rotor = conic.rotor;
                s.surface(
                    |u, v| {
                        let r = 0.1 + 1.15 * f64::from(u);
                        f3(rotor >> cone(ev, r, f64::from(v) * std::f64::consts::TAU))
                    },
                    12,
                    40,
                    |_, _| canvas::srgb(0.94, 0.9, 0.55),
                    0.3,
                    None,
                );
                s.polyline(&oval, 3.0, gold, 1.0);
                for (j, fraction) in [0.7, 0.4667, 0.2333].iter().enumerate() {
                    for (level, colour) in [
                        (lo * fraction, canvas::srgb(0.98, 0.5, 0.45)),
                        (hi * (0.7 - 0.2333 * j as f64), palette::sky()),
                    ] {
                        for [a, b] in contour::of_fn(
                            |u, v| pot(u, v) as f32,
                            [0.0, 1.0],
                            [0.0, 1.0],
                            90,
                            level as f32,
                        ) {
                            s.seg(
                                on_sphere(a[0], a[1]),
                                on_sphere(b[0], b[1]),
                                1.4,
                                colour,
                                0.85,
                            );
                        }
                    }
                }
            }
        }
        s.draw(&mut p);
        let size = (h as f32 / 42.0).clamp(8.0, 13.0);
        p.text(title, size, size * 1.6, size, palette::ink(), Align::Left);
        c.blit(&p, rect[0], rect[1]);
    }
    let d = distance(sample, conic.foci[0]) + distance(sample, conic.foci[1]);
    caption(
        c,
        "SPHERICAL QUADRICS: A CONE CUT BY THE SPHERE",
        &format!(
            "CL(3): P V C(P) = 0.  D1 + D2 = {d:.4} = 2 THETA_A = {:.4}.  SIN-PRODUCT = {:.4}",
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
    use gax::vga3d::Bivector;

    /// The polarity maps each basis point to its polar basis plane, scaled by its eigenvalue.
    #[test]
    fn polarity_is_diagonal_on_the_basis() {
        let eigenvalues = [1.0, 0.4, -0.8];
        let c = polarity(eigenvalues);
        for i in 0..3 {
            let mut e = [0.0; 3];
            e[i] = 1.0;
            for (j, plane) in planes().iter().enumerate() {
                let g = (c.of(point(e)) | *plane).s();
                let want = if i == j { eigenvalues[i] } else { 0.0 };
                assert!((g - want).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn geodesics_and_great_circles_stay_on_the_sphere() {
        let (a, b) = (point([1.0, 0.0, 0.0]), point([0.0, 0.6, 0.8]));
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
        for (phi, theta) in [(0.3, 0.7), (2.0, 1.2), (4.0, 2.9)] {
            let [x, y, z] = xyz(sphere(phi, theta));
            let want = [
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            ];
            assert!(
                (x - want[0]).abs() < 1e-14
                    && (y - want[1]).abs() < 1e-14
                    && (z - want[2]).abs() < 1e-14
            );
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
            [0.35507764519436, 0.747383015023582, 0.561550082122781],
            [-0.242143096584461, -0.308517190674206, 0.919882527190074],
        ];
        for (f, want) in conic.foci.iter().zip(numga) {
            for (a, b) in xyz(*f).iter().zip(want) {
                assert!((a - b).abs() < 1e-10);
            }
        }
        // The rotor is numga's: it carries the pole e12 to the same place. (numga's exponential
        // is good to a few 1e-12, so these compare to 1e-10.)
        let moved = conic.rotor >> Bivector::new(0.0, 0.0, 1.0);
        for (a, b) in
            xyz(moved)
                .iter()
                .zip([0.072898937662932, 0.283286671487454, 0.956260637399548])
        {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn a_frame_draws() {
        let mut draw = super::draw;
        let c = gax_numga_examples::app::frame(
            &gax_numga_examples::Anim::new("t", 1.0).size(480, 270),
            0.5,
            &mut draw,
        );
        assert!(c.mean()[0] > 0.0);
    }
}
