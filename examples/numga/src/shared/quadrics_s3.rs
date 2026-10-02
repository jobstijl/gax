//! Quadrics on the 3-sphere, shared by numga's `quadrics/s3_raytracer` and the S³ scenes of
//! `quadrics/elliptic_physics`: the algebra `Cl(4)` and a renderer of quadrics by projection.
//!
//! In `Cl(4)` the hyperplanes through the origin of `R⁴` are the vectors, the great spheres of
//! S³; a point of S³ is the pole of a plane, a trivector (numga's antivector), and a rotation of
//! `R⁴` is a rotor. A body is a dual quadric `Q` (a plane to its pole) placed by a rotor, and its
//! primal form `C = Q⁻¹` maps a point to its polar plane; `P ∨ C(P)` is negative inside.
//!
//! Seen from an eye, a body's outline on the image sphere a quarter turn ahead is the body
//! projected from the eye: pulled into the eye's frame that is a conic in the pixel chart
//! `(1, u, v)`, and a pixel is inside the outline where the conic's form is negative. With the
//! eye's projected polar plane, the conic gives a depth proportional to the cotangent of the angle
//! travelled along the ray's great circle; the largest depth selects the body per pixel, and only
//! that hit is reconstructed and shaded with its polar plane.
//!
//! Included by each binary that needs it with `#[path = "../shared/quadrics_s3.rs"] mod s3;`, so
//! not every binary uses every item.
#![allow(dead_code)]

gax::algebra! {
    algebra cl4 "The algebra of the 3-sphere, R(4,0,0), in numga's blade orientations for S³.";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1;
    kind Scalar = [1];
    versor Vector = [e1, e2, e3, e4];
    kind Bivector = [e23, e31, e12, e14, e24, e34];
    versor Trivector = [e234, e314, e124, e321];
    kind ScreenPoint = [e234, e314, e124];
    kind Pseudoscalar = [e1234];
    versor Rotor = [1, e23, e31, e12, e14, e24, e34, e1234];
}

pub use cl4::{Bivector, Rotor, ScreenPoint, Trivector, Vector};

/// A point of S³: a trivector, on numga's `yzw`, `zxw`, `xyw`, `zyx`.
pub type Point = Trivector<(), f64>;
/// A plane through the origin of `R⁴`: a great sphere of S³.
pub type Plane = Vector<(), f64>;
/// A rotation of `R⁴`, the rigid motions of S³.
pub type Motor = gax::Unit<Rotor<(), f64>>;
/// A rate of rotation: a bivector.
pub type Rate = Bivector<(), f64>;
/// The primal quadric: a point to its polar plane, negative inside.
pub type Quadric = Vector<(Trivector,), f64>;
/// The dual quadric: a plane to its pole.
pub type DualQuadric = Trivector<(Vector,), f64>;
/// A pixel: a point of the image sphere a quarter turn ahead of the eye, `(1, u, v)`.
pub type Pixel = ScreenPoint<(), f64>;
/// A body's outline in the pixel chart: a quadratic form on pixels.
pub type ScreenConic = cl4::Scalar<(ScreenPoint, ScreenPoint), f64>;
/// The eye's polar plane in the pixel chart: a linear form on pixels.
pub type ScreenPolar = cl4::Scalar<(ScreenPoint,), f64>;

/// The plane `w = 0`... as a vector: `e4`.
pub fn w() -> Plane {
    Vector::new(0.0, 0.0, 0.0, 1.0)
}

/// The representative of a point with positive unit weight on `w`; a unit point is a versor.
pub fn unit(p: Point) -> Point {
    p.gp(1.0 / (w() & p).s()).normalized().into_inner()
}

/// The origin of S³, `zyx`, where every body is centred in its own frame and the eye sits.
pub fn origin() -> Point {
    unit(Trivector::new(0.0, 0.0, 0.0, 1.0))
}

/// The point a quarter turn from the origin in the direction `(x, y, z)`.
pub fn direction([x, y, z]: [f64; 3]) -> Point {
    Trivector::new(x, y, z, 0.0).normalized().into_inner()
}

/// The rate `a xw + b yw + c zw`: a motion of the origin along `(a, b, c)`.
pub fn along([a, b, c]: [f64; 3]) -> Rate {
    Bivector::new(0.0, 0.0, 0.0, a, b, c)
}

/// The rate `a yz + b zx + c xy`: a turn about the origin.
pub fn turning([a, b, c]: [f64; 3]) -> Rate {
    Bivector::new(a, b, c, 0.0, 0.0, 0.0)
}

/// `exp(rate / 2)`: the rotor that moves by `rate` (a rotation by its angle).
pub fn motion(rate: Rate) -> Motor {
    rate.gp(0.5).exp()
}

/// A quadric moved by a rotor: its input pulled back, its output pushed forward.
pub fn moved(m: Motor, c: Quadric) -> Quadric {
    m >> c.of(m << Trivector::slot())
}

/// A dual quadric moved by a rotor.
pub fn moved_dual(m: Motor, q: DualQuadric) -> DualQuadric {
    m >> q.of(m << Vector::slot())
}

/// The ellipsoid about the origin with the given half-widths along the axes, each the tangent of
/// an angular half-width: the dual quadric pairing each axis direction with itself, weighted by
/// its squared half-width, and the origin with itself, weighted by `-1`, which closes it.
pub fn ellipsoid(half_widths: [f64; 3]) -> DualQuadric {
    let plane = Vector::slot();
    let mut q = (origin() & plane) * origin().gp(-1.0);
    for (k, h) in half_widths.iter().enumerate() {
        let mut d = [0.0; 3];
        d[k] = 1.0;
        let axis = direction(d);
        q += (axis & plane) * axis.gp(h * h);
    }
    q
}

/// A pixel of a pinhole looking along `+x` from the origin with horizontal field of view `fov`:
/// `u` from -1 (left) to 1 (right), `v` up, both scaled by `tan(fov / 2)`.
pub fn pixel(fov: f64, u: f64, v: f64) -> Pixel {
    let k = (fov / 2.0).tan();
    ScreenPoint::new(1.0, k * u, k * v)
}

/// The cone of rays from the eye tangent to a body: the eye's polar plane squared, less the form
/// scaled by the eye's own value. Non-negative on the directions whose ray meets the body.
pub fn outline(eye_frame: Motor, surface: Quadric) -> Quadric {
    let eye = eye_frame >> origin();
    let polar = surface.of(eye);
    (Trivector::slot() & polar) * polar - surface.gp((eye & polar).s())
}

/// Whether a ray direction lies inside a body's outline cone.
pub fn inside(cone: Quadric, ray: Pixel) -> bool {
    let r: Point = ray.cast::<Trivector>();
    (r & cone.of(r)).s() >= 0.0
}

/// A body projected from the eye: its screen conic and the eye's polar linear form. The eye
/// must be off the surface.
pub fn project(eye_frame: Motor, surface: Quadric) -> (ScreenConic, ScreenPolar) {
    let eye = eye_frame >> origin();
    let screen: Trivector<(ScreenPoint,), f64> =
        eye_frame >> ScreenPoint::slot().cast::<Trivector>();
    let normalized = surface.gp(1.0 / (eye & surface.of(eye)).s());
    let polar = screen & normalized.of(eye);
    ((screen & normalized.of(screen)) - polar * polar, polar)
}

/// The depth of the first hit along a pixel's ray, proportional to the cotangent of the angle:
/// larger is nearer; NaN is a miss.
pub fn reproject(conic: ScreenConic, polar: ScreenPolar, pixel: Pixel) -> f64 {
    -polar.of(pixel).s() + (-conic.of(pixel).of(pixel).s()).sqrt()
}

/// The point of S³ reached first along a pixel's ray at a depth, sign and all.
pub fn hit(eye_frame: Motor, depth: f64, pixel: Pixel) -> Point {
    let ray: Point = eye_frame >> pixel.cast::<Trivector>();
    ((eye_frame >> origin()).gp(depth) + ray)
        .normalized()
        .into_inner()
}

/// A scene to trace: the bodies' primal forms in the world, their colours (display values in
/// `[0, 1]`), the light, and the field of view.
#[derive(Clone)]
pub struct View {
    pub eye: Motor,
    pub surfaces: Vec<Quadric>,
    pub colors: Vec<[f64; 3]>,
    pub light: Point,
    pub fov: f64,
}

/// A frame ready to trace: every body projected once.
pub struct Tracer<'a> {
    view: &'a View,
    projected: Vec<(ScreenConic, ScreenPolar)>,
    /// The light's polar plane of each body, and the light's value.
    light_polars: Vec<(Plane, f64)>,
}

impl View {
    /// Project every body for this frame.
    pub fn tracer(&self) -> Tracer<'_> {
        let projected = self
            .surfaces
            .iter()
            .map(|s| project(self.eye, *s))
            .collect();
        let light_polars = self
            .surfaces
            .iter()
            .map(|s| {
                let polar = s.of(self.light);
                (polar, (self.light & polar).s())
            })
            .collect();
        Tracer {
            view: self,
            projected,
            light_polars,
        }
    }
}

impl Tracer<'_> {
    /// The nearest body along a pixel's ray and its depth.
    pub fn nearest(&self, pixel: Pixel) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for (body, (conic, polar)) in self.projected.iter().enumerate() {
            let depth = reproject(*conic, *polar, pixel);
            // A miss is NaN, and NaN compares false.
            if depth > best.map_or(f64::NEG_INFINITY, |b| b.1) {
                best = Some((body, depth));
            }
        }
        best
    }

    /// Whether the short arc from a hit to the light enters any body (negative inside), tested
    /// without roots.
    pub fn shadowed(&self, hit: Point) -> bool {
        let light = self.view.light;
        let hit = hit + (light - hit).gp(1e-6);
        self.view
            .surfaces
            .iter()
            .zip(&self.light_polars)
            .any(|(s, (polar, l))| {
                let h = (hit & s.of(hit)).s();
                let m = (hit & *polar).s();
                h < 0.0 || *l < 0.0 || (m < 0.0 && m * m > h * l)
            })
    }

    /// The colour of a pixel (display values), or `None` for the background. Lighting is done
    /// on S³ itself: the light is one point, and the hit is the point reached first, kept with
    /// its own sign, since the antipode of a hit is a different point facing the other way. The
    /// one great circle out of the light through the hit reaches it along an arc; the surface is
    /// lit where that arc arrives from outside, which the pairing of the polar plane with the
    /// light decides, with the falloff `1 / sin²(arc)` of a point source.
    pub fn color(&self, pixel: Pixel) -> Option<[f64; 3]> {
        let (body, depth) = self.nearest(pixel)?;
        let hit = hit(self.view.eye, depth, pixel);
        let light = self.view.light;
        let polar = self.view.surfaces[body].of(hit);
        let arc = (hit.dot(light).s() / light.dot(light).s())
            .clamp(-1.0, 1.0)
            .acos();
        let sine = arc.sin();
        let cosine = (polar & light).s() / (polar.norm() * sine);
        let mut lambert = if cosine < 0.0 {
            -cosine / (sine * sine)
        } else {
            0.0
        };
        if !lambert.is_finite() || self.shadowed(hit) {
            lambert = 0.0;
        }
        let k = 0.15 + 0.85 * lambert.clamp(0.0, 1.0);
        Some(self.view.colors[body].map(|c| (c * k).clamp(0.0, 1.0)))
    }
}

/// The pixel of a canvas point `(x, y)` on a `width` by `height` image, numga's chart: `u` across
/// from -1 to 1, `v` down from `height / width` to `-height / width`.
pub fn chart(fov: f64, x: f64, y: f64, width: f64, height: f64) -> Pixel {
    let u = -1.0 + 2.0 * x / width;
    let v = (1.0 - 2.0 * y / height) * height / width;
    pixel(fov, u, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The regressive product and the rotor exponential have numga's signs in this layout.
    #[test]
    fn conventions_match_numga() {
        let p = |c: [f64; 4]| Trivector::<(), f64>::from_coeffs(c);
        let x = Vector::<(), f64>::new(1.0, 0.0, 0.0, 0.0);
        assert_eq!((w() & p([0.0, 0.0, 0.0, 1.0])).s(), 1.0);
        assert_eq!((p([0.0, 0.0, 0.0, 1.0]) & w()).s(), -1.0);
        assert_eq!((x & p([1.0, 0.0, 0.0, 0.0])).s(), 1.0);
        assert_eq!((p([1.0, 0.0, 0.0, 0.0]) & x).s(), -1.0);
        let yzw = p([1.0, 0.0, 0.0, 0.0]);
        assert_eq!((yzw | yzw).s(), -1.0);
        // exp(0.3 xw) carries the origin 0.6 toward +x.
        let moved = Bivector::<(), f64>::new(0.0, 0.0, 0.0, 0.3, 0.0, 0.0).exp() >> origin();
        assert!(
            (moved.c[0] - 0.6f64.sin()).abs() < 1e-14 && (moved.c[3] - 0.6f64.cos()).abs() < 1e-14
        );
    }

    /// Depths from the screen conic reconstruct points on the ellipsoid; misses are NaN.
    #[test]
    fn reprojected_hits_lie_on_the_ellipsoid() {
        let q = ellipsoid([0.2, 0.2, 0.2]);
        let surface = moved_dual(motion(along([0.8, 0.0, 0.0])), q).inverse();
        let identity = motion(along([0.0; 3]));
        let (conic, polar) = project(identity, surface);
        let (mut hits, mut total) = (0, 0);
        for r in 0..30 {
            for c in 0..40 {
                let px = chart(
                    60f64.to_radians(),
                    c as f64 + 0.5,
                    r as f64 + 0.5,
                    40.0,
                    30.0,
                );
                let depth = reproject(conic, polar, px);
                total += 1;
                if depth.is_nan() {
                    continue;
                }
                hits += 1;
                let p = hit(identity, depth, px);
                assert!((p & surface.of(p)).s().abs() < 1e-8);
            }
        }
        assert!(hits > 0 && hits < total);
    }
}
