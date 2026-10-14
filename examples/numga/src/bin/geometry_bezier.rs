//! numga's `geometry/bezier`: rational quadratic Bézier curves as curves of vectors, and the
//! cone of their conic, in VGA3D (after Harquin, Breuils, Biri and Nozick, *Projective
//! Geometric Algebra powers up Bézier Curves*, WSCG 2026).
//!
//! A rational Bézier curve sums its control points, each weighted by its weight and its
//! Bernstein polynomial. With the weight as a third coordinate, the point `(x, y)` at weight `w`
//! is the vector `w (x e1 + y e2 + e3)`, and the sum is a plain Bézier curve of vectors, with
//! nothing divided. A point of the plane is the line through the origin along its vector, read
//! where it meets the unit sphere: the projective plane as a sphere, every point on it twice,
//! its equator the line at infinity. Negating both end weights traces the rest of the same
//! conic, the complement.
//!
//! The rays of a quadratic curve sweep a cone, the zero set of a quadratic form on vectors
//! built from the control triangle: the normal of the plane through the origin and the two ends
//! (`(start ^ end).dual()`), paired with itself, against the normals of the two planes through
//! an end and its tangent, in the proportion `first * last` to `4 * inner²` of the weights. A
//! form is a map `Vector <- Vector`, its value on a vector the inner product with the image.
//! It sees the end weights only through their product, so curve and complement lie on the same
//! cone. Both normals of a pair flip together under a change of the dual's sign, so gax's
//! metric-free dual gives numga's form.
//!
//! The animation sweeps the middle weight from 0.3 to 2 and back while the view turns once
//! around: the cone's two antipodal loops on the sphere miss the equator for an ellipse, touch
//! it for a parabola and cross it for a hyperbola. On the right, the same curves in the plane of
//! unit weight, where each vector is the point it stands for.

use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Light, Marker, ORIGIN3, Point2, Rect, Scene3,
    backdrop, caption, contour, palette, reach3, run, scene3::panel3, signal::phasor,
};

mod bezier {
    use gax::vga3d::Vector;

    pub type V = Vector<(), f64>;
    /// A quadratic form on vectors, as the map from a vector to the vector it pairs with.
    pub type Conic = Vector<(Vector,), f64>;
    /// The weights of a quadratic curve's three control points.
    pub type Weights = [f64; 3];

    /// The point `(x, y)` at unit weight: the vector `x e1 + y e2 + e3`.
    pub fn point(x: f64, y: f64) -> V {
        Vector::new(x, y, 1.0)
    }

    /// numga's control triangle.
    pub fn controls() -> [V; 3] {
        [point(-1.0, 0.0), point(0.0, 1.5), point(1.0, 0.0)]
    }

    /// The three Bernstein polynomials of degree two at `t`.
    pub fn bernstein(t: f64) -> [f64; 3] {
        [(1.0 - t) * (1.0 - t), 2.0 * t * (1.0 - t), t * t]
    }

    /// The rational quadratic curve at `t`: each control weighted by its weight and its
    /// Bernstein polynomial, summed.
    pub fn blend(controls: &[V; 3], weights: Weights, t: f64) -> V {
        let basis = bernstein(t);
        (0..3).fold(V::zero(), |sum, k| {
            sum + controls[k].gp(weights[k] * basis[k])
        })
    }

    /// The curve at `samples + 1` parameters from 0 to 1.
    pub fn curve(controls: &[V; 3], weights: Weights, samples: usize) -> Vec<V> {
        (0..=samples)
            .map(|k| blend(controls, weights, k as f64 / samples as f64))
            .collect()
    }

    /// The weights that trace the rest of the same conic: both end weights negated, which
    /// turns the tangents at the ends around.
    pub fn complementary([first, inner, last]: Weights) -> Weights {
        [-first, inner, -last]
    }

    /// The normals of two planes through the origin as one form: on a vector, the product of
    /// its inner products with the two.
    pub fn pair(first: V, second: V) -> Conic {
        let v = Vector::slot();
        (first * (second | v) + second * (first | v)).gp(0.5)
    }

    /// The form whose zero set is the cone a quadratic curve's rays sweep: the plane through
    /// the origin and the ends against the two through the origin and an end's tangent, in the
    /// proportion the weights set.
    pub fn conic(controls: &[V; 3], [first, inner, last]: Weights) -> Conic {
        let [start, middle, end] = *controls;
        let chord = (start ^ end).dual();
        let (before, after) = ((start ^ middle).dual(), (middle ^ end).dual());
        pair(chord, chord).gp(first * last) - pair(before, after).gp(4.0 * inner * inner)
    }

    /// The form's value on a vector: zero on the cone.
    pub fn on(shape: Conic, v: V) -> f64 {
        (v | shape.of(v)).s()
    }

    /// The middle weight swept from 0.3 out to 2 and back at the phase `cos_phase` gives the
    /// cosine of: a cosine of its logarithm, so the sweep loops smoothly.
    pub fn middle_weight(cos_phase: f64) -> f64 {
        let (low, high) = (0.3f64, 2.0f64);
        let mean = 0.5 * (low.ln() + high.ln());
        (mean - 0.5 * (high / low).ln() * cos_phase).exp()
    }

    /// The weights with the given middle weight, the ends at one.
    pub fn weights(middle: f64) -> Weights {
        [1.0, middle, 1.0]
    }
}

use bezier::*;

const SECONDS: f32 = 9.0;
/// Samples along each curve (numga's).
const SAMPLES: usize = 400;
/// Every how many samples a ray of the fan is drawn.
const FAN_STRIDE: usize = 8;
/// The samples per side of the longitude-latitude grid the cone's loops are found on.
const SPHERE_GRID: usize = 160;
/// numga's view: its elevation and starting azimuth, in degrees.
const ELEVATION: f32 = 18.0;
const AZIMUTH: f32 = -62.0;

fn curve_colour() -> Light {
    palette::sky()
}

fn complement_colour() -> Light {
    palette::orange()
}

fn control_colour() -> Light {
    palette::ink().faded(0.45)
}

/// The middle weight at time `t`.
fn middle_at(t: f32) -> f64 {
    let phase = core::f64::consts::TAU * f64::from(t / SECONDS);
    middle_weight(phasor(phase).e20())
}

/// The unit vector at a longitude and latitude, in radians.
fn on_sphere(longitude: f64, latitude: f64) -> V {
    let (around, up) = (phasor(longitude), phasor(latitude));
    let across = up.e20();
    gax::vga3d::Vector::new(across * around.e20(), across * around.e01(), up.e01())
}

/// Where the conic's form vanishes on the unit sphere, as segments: its level zero over a
/// longitude-latitude grid, numga's construction of the cone's loops.
fn cone_loops(shape: Conic) -> Vec<[V; 2]> {
    let (pi, half) = (core::f32::consts::PI, core::f32::consts::FRAC_PI_2);
    let level = |p: Point2| {
        let [lon, lat] = p.to_euclidean().map(f64::from);
        on(shape, on_sphere(lon, lat)) as f32
    };
    contour::of_fn(level, [-pi, pi], [-half, half], SPHERE_GRID, 0.0)
        .into_iter()
        .map(|segment| {
            segment.map(|p| {
                let [lon, lat] = p.to_euclidean().map(f64::from);
                on_sphere(lon, lat)
            })
        })
        .collect()
}

/// A curve as its sum is, the fan of rays from the origin to it, and where its rays pierce the
/// unit sphere.
fn draw_rays(s: &mut Scene3, points: &[V], colour: Light) {
    let sum: Vec<_> = points.iter().map(|v| reach3(*v)).collect();
    s.polyline(&sum, 1.2, colour.faded(0.7));
    for k in (0..points.len() - FAN_STRIDE).step_by(FAN_STRIDE) {
        let (a, b) = (sum[k], sum[k + FAN_STRIDE]);
        s.tri(ORIGIN3, a, b, colour.faded(0.5), 0.12);
    }
    let pierced: Vec<_> = points
        .iter()
        .map(|v| reach3(v.normalized().into_inner()))
        .collect();
    s.polyline(&pierced, 3.0, colour);
}

/// The weighted control points of a curve, each on its ray out to the sphere at least.
fn draw_controls(s: &mut Scene3, controls: &[V; 3], weights: Weights) {
    let weighted: Vec<_> = (0..3).map(|k| controls[k].gp(weights[k])).collect();
    let corners: Vec<_> = weighted.iter().map(|v| reach3(*v)).collect();
    s.polyline(&corners, 1.0, control_colour());
    for (v, corner) in weighted.iter().zip(&corners) {
        let out = v.gp(v.norm().recip().max(1.0));
        s.seg(ORIGIN3, reach3(out), 0.7, control_colour());
        s.dot(*corner, Marker::Dot, 5.0, control_colour());
    }
}

/// Homogeneous space in `rect`, seen from turning once around over the loop: the sphere and
/// its equator, the cone's loops, the controls, and both curves with their rays.
fn space_panel(c: &mut Canvas, rect: Rect, t: f32) {
    let weights = weights(middle_at(t));
    let controls = controls();
    let shape = conic(&controls, weights);
    let azimuth = (AZIMUTH + 360.0 * t / SECONDS).to_radians();
    let cam = Camera::orbit(
        rect,
        ORIGIN3,
        12.0,
        azimuth,
        ELEVATION.to_radians(),
        Lens::Perspective(0.33),
    );
    panel3(c, rect, cam, |s| {
        s.sphere_wire(ORIGIN3, 1.0, 12, palette::grid().faded(0.6));
        let equator: Vec<_> = (0..=96)
            .map(|k| reach3(on_sphere(core::f64::consts::TAU * k as f64 / 96.0, 0.0)))
            .collect();
        s.polyline(&equator, 1.4, palette::ink().faded(0.6));
        let loops: Vec<_> = cone_loops(shape)
            .into_iter()
            .map(|seg| seg.map(reach3))
            .collect();
        s.stroke(&loops, 1.0, palette::ink().faded(0.3));
        for side in [weights, complementary(weights)] {
            draw_controls(s, &controls, side);
        }
        draw_rays(s, &curve(&controls, weights, SAMPLES), curve_colour());
        let rest = curve(&controls, complementary(weights), SAMPLES);
        draw_rays(s, &rest, complement_colour());
        s.dot(ORIGIN3, Marker::Dot, 5.0, palette::ink());
    });
    let note = "THE UNIT SPHERE IS THE PROJECTIVE PLANE, ITS EQUATOR THE LINE AT INFINITY";
    let at = rect.bottom_left() + Point2::direction(12.0, -14.0);
    c.text(note, at, 9.0, palette::ink().faded(0.6), Align::Left);
}

/// A vector of space as the point of the plane it stands for: `x e1 + y e2 + w e3` is the
/// PGA2D point `x e20 + y e01 + w e12`, at `(x / w, y / w)`.
fn projective(v: V) -> gax::pga2d::Point<(), f64> {
    gax::pga2d::Point::new(v.e1(), v.e2(), v.e3())
}

/// A curve in the plane of unit weight, broken where its weight changes sign: there it runs
/// through the line at infinity, from one branch of a hyperbola to the other.
fn draw_chart_curve(ax: &Axes, c: &mut Canvas, points: &[V], colour: Light) {
    let mut run = Vec::new();
    for pair in points.windows(2) {
        run.push(projective(pair[0]));
        if pair[0].e3() * pair[1].e3() <= 0.0 {
            ax.polyline(c, &run, 2.4, colour);
            run.clear();
        }
    }
    run.push(projective(points[points.len() - 1]));
    ax.polyline(c, &run, 2.4, colour);
}

/// The kind of conic the middle weight makes, with the ends at one.
fn kind(middle: f64) -> &'static str {
    const PARABOLA: f64 = 0.02;
    if (middle - 1.0).abs() < PARABOLA {
        "PARABOLA"
    } else if middle < 1.0 {
        "ELLIPSE"
    } else {
        "HYPERBOLA"
    }
}

/// The plane of unit weight in `rect`: both curves, the control triangle, and the weight.
fn chart_panel(c: &mut Canvas, rect: Rect, t: f32) {
    let middle = middle_at(t);
    let weights = weights(middle);
    let controls = controls();
    let ax = Axes::equal(rect, Point2::xy(0.0, 0.6), 1.9);
    ax.clip(c);
    ax.polyline(c, &controls.map(projective), 1.0, control_colour());
    draw_chart_curve(&ax, c, &curve(&controls, weights, SAMPLES), curve_colour());
    let rest = curve(&controls, complementary(weights), SAMPLES);
    draw_chart_curve(&ax, c, &rest, complement_colour());
    ax.scatter(
        c,
        &controls.map(projective),
        Marker::Dot,
        7.0,
        palette::ink(),
    );
    c.unclip();
    let label = format!("MIDDLE WEIGHT {middle:.2}: {}", kind(middle));
    let at = rect.bottom_left() + Point2::direction(4.0, -8.0);
    c.text(&label, at, 11.0, palette::ink(), Align::Left);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    space_panel(c, screen.part(0.0, 0.06, 0.6, 1.0), t);
    chart_panel(c, screen.part(0.62, 0.18, 0.98, 0.94), t);
    caption(
        c,
        "BEZIER: THE CONE OF A RATIONAL QUADRATIC CURVE",
        "THE WEIGHT AS A THIRD COORDINATE: CURVE AND COMPLEMENT ON ONE CONE (VGA3D)",
    );
}

fn main() {
    run(Anim::new("bezier", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::bezier::*;
    use gax_numga_examples::signal::phasor;

    /// numga's check: at each of the 48 middle weights of the sweep, every sample of the curve
    /// and of its complement lies on the cone of the conic.
    #[test]
    fn both_arcs_lie_on_the_cone_of_their_conic() {
        let frames = 48;
        let controls = controls();
        for k in 0..frames {
            let phase = core::f64::consts::TAU * k as f64 / frames as f64;
            let weights = weights(middle_weight(phasor(phase).e20()));
            let shape = conic(&controls, weights);
            let arcs =
                [weights, complementary(weights)].map(|w| curve(&controls, w, super::SAMPLES));
            for v in arcs.iter().flatten() {
                assert!(on(shape, *v).abs() < 1e-10, "{k}: {}", on(shape, *v));
            }
        }
    }

    /// The sweep runs from 0.3 to 2 and back.
    #[test]
    fn the_middle_weight_sweeps_out_and_back() {
        assert!((middle_weight(1.0) - 0.3).abs() < 1e-12);
        assert!((middle_weight(-1.0) - 2.0).abs() < 1e-12);
    }

    /// numga's test also draws two frames of its animation.
    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
        gax_numga_examples::app::assert_draws(super::draw, 4.5);
    }
}
