//! numga's `geometry/surface_curvature`: the curvature of quadric surfaces in PGA3D, the second
//! fundamental form against the first. A quadric is a polarity, a map `Plane <- Point`, and its
//! surface is where a point lies on its own polar plane, `p & Q(p) = 0`; that polar plane is the
//! tangent plane. Read on directions (ideal points) the same form is the Hessian, and the
//! principal curvatures are its eigenvalues on the tangent plane against the Euclidean metric.
//! The lines of curvature of a central quadric are where it meets its confocal quadrics, and the
//! members of that family through a point are the eigenvalues of one form, the dual quadric read
//! on directions less the point's own dyad. So every pixel of a ray-cast image finds its
//! Gaussian curvature (colour: red positive, blue negative) and its place in the net of
//! curvature lines with two small symmetric eigensolves. The animation turns an ellipsoid and a
//! hyperboloid of one sheet under the view.
//!
//! Directions are the slot kind of the forms. PGA3D has no kind for the ideal points alone, so
//! the slot is VGA3D's vector, embedded by the homomorphism `vga3d::Vector -> pga3d::Plane` (a
//! vector as the plane through the origin normal to it) and dualized to the ideal point. On it
//! the metric is definite, so `eigh_with` applies as in numga's `eigvalsh(metric)`.

use gax::pga3d::{Plane, Point, Scalar};
use gax::vga3d::Vector;
use gax_numga_examples::canvas::{Rgb, mix, scale, srgb};
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, backdrop, caption, colormap, palette, run,
};
use std::sync::OnceLock;

mod curvature {
    use super::*;

    pub type P = Point<(), f64>;
    pub type Pl = Plane<(), f64>;
    /// A polarity: each point to its polar plane.
    pub type Quadric = Plane<(Point,), f64>;
    /// A form on directions.
    pub type Form = Scalar<(Vector, Vector), f64>;

    /// The plane at infinity.
    pub fn infinity() -> Pl {
        Plane::new(0.0, 0.0, 0.0, 1.0)
    }

    /// Directions as ideal points: a vector's plane through the origin, dualized.
    pub fn direction() -> Point<(Vector,), f64> {
        Plane::from(Vector::<(), f64>::slot()).dual()
    }

    /// The metric on directions: the inner product of the planes they are normal to.
    pub fn metric() -> Form {
        let (a, b) = (direction().dual(), direction().dual());
        a | b
    }

    /// A central quadric with one weight per axis, a sum of plane dyads less the plane at
    /// infinity's: `Σ wᵢ Aᵢ (Aᵢ & X) - e0 (e0 & X)`.
    pub fn quadric(weights: [f64; 3]) -> Quadric {
        let x = Point::slot();
        let axes = [
            Plane::new(1.0, 0.0, 0.0, 0.0),
            Plane::new(0.0, 1.0, 0.0, 0.0),
            Plane::new(0.0, 0.0, 1.0, 0.0),
        ];
        let w = infinity();
        axes.iter()
            .zip(weights)
            .fold(Plane::zero(), |q, (a, k)| q + (*a * (*a & x)).gp(k))
            - w * (w & x)
    }

    /// Of three eigenvalues on directions, the two that are not the one nearest zero, in order.
    pub fn pair(values: [f64; 3]) -> [f64; 2] {
        let mut v = values;
        v.sort_by(|a, b| a.abs().total_cmp(&b.abs()));
        let (a, b) = (v[1], v[2]);
        [a.min(b), a.max(b)]
    }

    /// A quadric with what every pixel needs prepared: its form on points, and for the confocal
    /// family its centre and its shape on directions.
    #[derive(Clone, Copy, Debug)]
    pub struct Surface {
        pub q: Quadric,
        /// The form `Point & Q(Point)`.
        pub form: Scalar<(Point, Point), f64>,
        pub centre: P,
        /// The poles of the planes through the centre normal to each direction, as the planes
        /// those poles are normal to.
        pub shape: Plane<(Vector,), f64>,
        pub metric: Form,
    }

    impl Surface {
        pub fn new(q: Quadric) -> Surface {
            let dual = q.inverse();
            let w = infinity();
            // The pole of the plane at infinity is the centre.
            let pole = dual.of(w);
            let centre = pole.gp(1.0 / (w & pole).s());
            let d = direction().dual();
            let through_centre = d - w * (d & centre);
            Surface {
                q,
                form: Point::slot() & q,
                centre,
                shape: dual.of(through_centre).dual(),
                metric: metric(),
            }
        }

        /// Where a ray first meets the surface, and the discriminant, negative where it misses.
        /// The form bound to the ray in both slots is a quadratic in the distance along it.
        pub fn hit(&self, origin: P, heading: P) -> (P, f64) {
            let f = |x: P, y: P| self.form.of(x).of(y).s();
            let (a, b, c) = (f(heading, heading), f(heading, origin), f(origin, origin));
            let disc = b * b - a * c;
            let root = disc.max(0.0).sqrt();
            let t = [(-b - root) / a, (-b + root) / a]
                .into_iter()
                .filter(|t| *t > 0.0)
                .fold(f64::INFINITY, f64::min);
            let t = if t.is_finite() { t } else { 0.0 };
            ((origin + heading.gp(t)).unitized(), disc)
        }

        /// The two principal curvatures at a point of the surface, in order; convex surfaces
        /// curve positively. The form `Point & Q(Point)` with the projector onto the tangent
        /// plane in both slots is the second fundamental form, up to the length of the gradient;
        /// its eigenvalues against the metric are the principal curvatures and the normal's 0.
        pub fn principal(&self, p: P) -> [f64; 2] {
            let tangent = self.q.of(p);
            // The normal: the tangent plane's dual, less its weight.
            let n = tangent.dual();
            let normal = Point::new(n.c[0], n.c[1], n.c[2], 0.0);
            // The projector of directions onto the tangent plane.
            let d = direction();
            let project = d - normal * (tangent & d).gp(1.0 / (tangent & normal).s());
            let length = (normal.dual() | normal.dual()).s().sqrt();
            let second = self
                .form
                .of(project)
                .at::<1>()
                .of(project)
                .gp(-1.0 / length);
            pair(second.eigh_with(self.metric).0)
        }

        /// The parameters of the confocal quadrics through a point: the eigenvalues of the
        /// shape less the point's own dyad. One is the surface itself; the level sets of the
        /// other two are its lines of curvature.
        pub fn confocal(&self, p: P) -> [f64; 2] {
            let d = direction();
            // The point's offset from the centre, as the plane it is normal to.
            let position = (p.unitized() - self.centre).dual();
            let through_point = d & (self.shape - position * (position & d));
            pair(through_point.eigh_with(self.metric).0)
        }
    }

    pub const SEMI_AXES: [f64; 3] = [3.0, 2.0, 1.2];

    /// The ellipsoid with semi-axes 3, 2 and 1.2.
    pub fn ellipsoid() -> Quadric {
        quadric(SEMI_AXES.map(|a| 1.0 / (a * a)))
    }

    /// The hyperboloid of one sheet `x² / 1.6² + y² - z² = 1`.
    pub fn hyperboloid() -> Quadric {
        quadric([1.0 / (1.6 * 1.6), 1.0, -1.0])
    }
}

use curvature::*;

/// What a pixel sees: the light on the surface, the principal curvatures and the confocal
/// parameters.
#[derive(Clone, Copy)]
struct Seen {
    light: f32,
    k: [f64; 2],
    t: [f64; 2],
}

/// A ray-cast view of a surface, `w` x `h` pixels, cut at `|z| < height`.
fn render(surface: &Surface, cam: &Camera, w: usize, h: usize, height: f64) -> Vec<Option<Seen>> {
    let rows: Vec<usize> = (0..h).collect();
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get());
    let per = h.div_ceil(threads).max(1);
    let f64p =
        |p: Point<(), f32>| Point::new(p.c[0].into(), p.c[1].into(), p.c[2].into(), p.c[3].into());
    let mut out = vec![None; w * h];
    std::thread::scope(|s| {
        for (chunk, part) in rows.chunks(per).zip(out.chunks_mut(per * w)) {
            s.spawn(move || {
                for (i, &y) in chunk.iter().enumerate() {
                    for x in 0..w {
                        let (o, d) = cam.ray([x as f32 + 0.5, y as f32 + 0.5]);
                        let (o, d) = (f64p(o), f64p(d));
                        let (p, disc) = surface.hit(o, d);
                        if disc < 0.0 || p.to_euclidean()[2].abs() >= height {
                            continue;
                        }
                        // Lit from behind the viewer's shoulder, both sides alike.
                        let n = surface.q.of(p);
                        let n = [n.e1(), n.e2(), n.e3()];
                        let back = [-d.e032(), -d.e013(), -d.e021()];
                        let lamp = [
                            back[0] + back[1] * 0.4,
                            back[1] - back[0] * 0.4,
                            back[2] + 0.6,
                        ];
                        let dot = n[0] * lamp[0] + n[1] * lamp[1] + n[2] * lamp[2];
                        let len = (n.iter().map(|v| v * v).sum::<f64>()
                            * lamp.iter().map(|v| v * v).sum::<f64>())
                        .sqrt();
                        part[i * w + x] = Some(Seen {
                            light: (0.4 + 0.6 * (dot / len).abs()) as f32,
                            k: surface.principal(p),
                            t: surface.confocal(p),
                        });
                    }
                }
            });
        }
    });
    out
}

/// The range of each confocal parameter over a surface, from a dense sample of its points
/// (fixed, so that the level lines hold still while the view turns).
fn ranges(surface: &Surface, sample: impl Fn(f64, f64) -> P) -> [[f64; 2]; 2] {
    let mut r = [[f64::MAX, f64::MIN]; 2];
    for i in 0..=60 {
        for j in 0..120 {
            let t = surface.confocal(sample(i as f64 / 60.0, j as f64 / 120.0));
            for f in 0..2 {
                r[f] = [r[f][0].min(t[f]), r[f][1].max(t[f])];
            }
        }
    }
    r
}

struct Scene {
    surfaces: [Surface; 2],
    ranges: [[[f64; 2]; 2]; 2],
}

fn scene() -> &'static Scene {
    static SCENE: OnceLock<Scene> = OnceLock::new();
    SCENE.get_or_init(|| {
        let (e, h) = (Surface::new(ellipsoid()), Surface::new(hyperboloid()));
        let tau = core::f64::consts::TAU;
        let [a, b, c] = SEMI_AXES;
        let on_ellipsoid = |u: f64, v: f64| {
            let (lat, lon) = ((u - 0.5) * tau / 2.0, v * tau);
            Point::xyz(
                a * lat.cos() * lon.cos(),
                b * lat.cos() * lon.sin(),
                c * lat.sin(),
            )
        };
        let on_hyperboloid = |u: f64, v: f64| {
            let s = (2.0 * u - 1.0) * 2f64.asinh();
            Point::xyz(
                1.6 * s.cosh() * (v * tau).cos(),
                s.cosh() * (v * tau).sin(),
                s.sinh(),
            )
        };
        Scene {
            ranges: [ranges(&e, on_ellipsoid), ranges(&h, on_hyperboloid)],
            surfaces: [e, h],
        }
    })
}

const LEVELS: f64 = 14.0;

/// The colour of a pixel: Gaussian curvature, lit, darkened on the level lines of the two
/// confocal parameters (a fixed width in pixels, from the parameter's rate across pixels).
fn shade(
    seen: &[Option<Seen>],
    w: usize,
    x: usize,
    y: usize,
    ranges: &[[f64; 2]; 2],
) -> Option<Rgb> {
    let s = seen[y * w + x]?;
    let gauss = (s.k[0] * s.k[1] / 1.2).clamp(-1.0, 1.0);
    let mut rgb = scale(colormap::rdbu(0.5 + 0.5 * gauss as f32), s.light);
    let families = [srgb(0.10, 0.10, 0.10), srgb(0.55, 0.08, 0.08)];
    for f in 0..2 {
        let spacing = (ranges[f][1] - ranges[f][0]) / LEVELS;
        let at = |x: usize, y: usize| seen[y * w + x].map(|s| s.t[f] / spacing);
        let v = s.t[f] / spacing;
        // Central differences, as numpy.gradient; no line where a neighbour is off the surface.
        let grad = |a: Option<f64>, b: Option<f64>| Some((b? - a?) * 0.5);
        let (Some(dx), Some(dy)) = (
            grad(at(x.saturating_sub(1), y), at((x + 1).min(w - 1), y)),
            grad(
                at(x, y.saturating_sub(1)),
                at(x, (y + 1).min(seen.len() / w - 1)),
            ),
        ) else {
            continue;
        };
        let rate = (dx * dx + dy * dy).sqrt() + 1e-9;
        let cover = (1.0 - ((v - v.round()).abs() / rate - 0.5 * 1.2)).clamp(0.0, 1.0) as f32;
        rgb = mix(rgb, families[f], cover);
    }
    Some(rgb)
}

const SECONDS: f32 = 16.0;

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let scene = scene();
    let top = (c.height as f32 * 0.13) as usize;
    let (pw, ph) = (c.width / 2, c.height - top);
    let turn = core::f32::consts::TAU * t / SECONDS;
    let views = [(25f32, 3.4f32, f64::INFINITY), (18.0, 3.6, 2.0)];
    for (i, (elevation, extent, height)) in views.into_iter().enumerate() {
        let cam = Camera::orbit(
            pw,
            ph,
            [0.0; 3],
            10.0 * extent,
            (-60f32).to_radians() + turn,
            elevation.to_radians(),
            Lens::Parallel(extent),
        );
        let seen = render(&scene.surfaces[i], &cam, pw, ph, height);
        let x0 = i * pw;
        c.clip([x0 as f32, top as f32, (x0 + pw) as f32, c.height as f32]);
        let ranges = scene.ranges[i];
        c.shade(1, |x, y| {
            let (px, py) = (x as usize - x0, y as usize - top);
            shade(&seen, pw, px, py, &ranges)
        });
        c.unclip();
        let label = ["ELLIPSOID", "HYPERBOLOID OF ONE SHEET"][i];
        let size = (c.height as f32 / 30.0).clamp(7.0, 12.0);
        c.text(
            label,
            (x0 + pw / 2) as f32,
            c.height as f32 - size * 0.8,
            size,
            palette::ink(),
            Align::Center,
        );
    }
    caption(
        c,
        "CURVATURE OF QUADRICS: GAUSSIAN CURVATURE AND LINES OF CURVATURE",
        "PER PIXEL: TWO EIGENPROBLEMS ON DIRECTIONS (PGA3D)",
    );
}

fn main() {
    run(
        Anim::new("surface curvature", SECONDS)
            .size(480, 270)
            .scale(2),
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::curvature::*;
    use gax::pga3d::{Line, Motor, Point};

    /// The ellipsoid's four umbilics, where it curves equally in every direction: in the plane
    /// of its longest and shortest axes.
    fn umbilics() -> [P; 4] {
        let [a, b, c] = SEMI_AXES;
        let x = a * ((a * a - b * b) / (a * a - c * c)).sqrt();
        let z = c * ((b * b - c * c) / (a * a - c * c)).sqrt();
        [
            Point::xyz(x, 0.0, z),
            Point::xyz(-x, 0.0, z),
            Point::xyz(x, 0.0, -z),
            Point::xyz(-x, 0.0, -z),
        ]
    }

    /// The quadric conjugated by a motor: the same surface, moved.
    fn placed(q: Quadric, m: gax::Unit<Motor<(), f64>>) -> Quadric {
        m >> q.of(m << Point::slot())
    }

    fn rel(a: f64, b: f64, rtol: f64) -> bool {
        (a - b).abs() <= rtol * b.abs().max(1e-300)
    }

    const TOWARD: [[f64; 3]; 3] = [[10.0, 6.0, 4.0], [-7.0, 9.0, 3.0], [2.0, -8.0, -9.0]];

    /// Where rays toward the centre from a few directions meet the surface.
    fn surface_points(s: &Surface) -> Vec<P> {
        TOWARD
            .iter()
            .map(|t| {
                let n = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
                let heading = Point::direction(-t[0] / n, -t[1] / n, -t[2] / n);
                s.hit(Point::xyz(t[0], t[1], t[2]), heading).0
            })
            .collect()
    }

    /// numga's scenario checks: at the tip of the long axis the principal curvatures are
    /// `a / b²` and `a / c²`; at the umbilics the two curvatures agree, and so do the two
    /// confocal parameters (the net of lines closes up there).
    #[test]
    fn tip_and_umbilics() {
        let s = Surface::new(ellipsoid());
        let [a, b, c] = SEMI_AXES;
        let tip = s.principal(Point::xyz(a, 0.0, 0.0));
        assert!(
            rel(tip[0], a / (b * b), 1e-8) && rel(tip[1], a / (c * c), 1e-8),
            "{tip:?}"
        );
        for u in umbilics() {
            let k = s.principal(u);
            assert!(rel(k[0], k[1], 1e-8), "{k:?}");
            let t = s.confocal(u);
            assert!(rel(t[0], t[1], 1e-6), "{t:?}");
        }
    }

    /// `K = 1 / (a b c)² / (x²/a⁴ + y²/b⁴ + z²/c⁴)²` on the ellipsoid with semi-axes a, b, c.
    #[test]
    fn gaussian_curvature_matches_the_ellipsoid_formula() {
        let s = Surface::new(ellipsoid());
        let [a, b, c] = SEMI_AXES;
        for p in surface_points(&s) {
            let k = s.principal(p);
            let [x, y, z] = p.to_euclidean();
            let sum = x * x / a.powi(4) + y * y / b.powi(4) + z * z / c.powi(4);
            let expected = 1.0 / (a * b * c).powi(2) / (sum * sum);
            assert!(
                rel(k[0] * k[1], expected, 1e-6),
                "{} {expected}",
                k[0] * k[1]
            );
        }
    }

    /// Moving the quadric and its points by one motor changes neither the curvatures nor the
    /// confocal parameters: nothing refers to an origin.
    #[test]
    fn curvature_and_confocal_parameters_do_not_depend_on_placement() {
        let s = Surface::new(ellipsoid());
        let points = surface_points(&s);
        let m = Line::new(0.15, 0.0, 0.2, 0.0, 0.0, 0.0).exp()
            * Line::new(0.0, 0.0, 0.0, -0.75, 0.0, 0.35).exp();
        let moved = Surface::new(placed(ellipsoid(), m));
        for p in points {
            let q = m >> p;
            let (k, kq) = (s.principal(p), moved.principal(q));
            let (t, tq) = (s.confocal(p), moved.confocal(q));
            for i in 0..2 {
                assert!(rel(kq[i], k[i], 1e-6), "{k:?} {kq:?}");
                assert!(rel(tq[i], t[i], 1e-6), "{t:?} {tq:?}");
            }
        }
    }

    /// The hit lies on the surface and on the ray, and a ray that passes by misses.
    #[test]
    fn rays_hit_the_surface() {
        let s = Surface::new(hyperboloid());
        let (p, disc) = s.hit(Point::xyz(10.0, 0.3, 0.2), Point::direction(-1.0, 0.0, 0.0));
        assert!(disc > 0.0);
        assert!(s.form.of(p).of(p).s().abs() < 1e-12);
        let [x, y, z] = p.to_euclidean();
        // On x² / 1.6² + y² - z² = 1 at y = 0.3, z = 0.2.
        let expected = 1.6 * (1.0f64 + 0.2 * 0.2 - 0.3 * 0.3).sqrt();
        assert!((x - expected).abs() < 1e-12 && (y - 0.3).abs() < 1e-12 && (z - 0.2).abs() < 1e-12);
        let e = Surface::new(ellipsoid());
        assert!(
            e.hit(Point::xyz(10.0, 5.0, 0.0), Point::direction(-1.0, 0.0, 0.0))
                .1
                < 0.0
        );
    }

    #[test]
    fn a_frame_draws() {
        let mut draw = super::draw;
        let c = gax_numga_examples::app::frame(
            &gax_numga_examples::Anim::new("t", 1.0).size(160, 90),
            0.5,
            &mut draw,
        );
        assert!(c.mean()[0] > 0.0);
    }
}
