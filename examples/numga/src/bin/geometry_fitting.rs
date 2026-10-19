//! numga's `geometry/fitting`: least-squares fits of PGA3D primitives, one pattern for all. Join
//! each sample with the unknown left open (a point with a plane is a scalar, with a line a plane,
//! with a point a line): that is the incidence error, linear in the unknown. Its squared norm,
//! summed, is a form on the unknown, and the fit is its least mode against the unknown's own
//! unit form, the scalar part of `X ~X`. That form measures a plane's normal, a line's direction
//! and a point's weight, and leaves the position free, so least squares fixes the position. The
//! roles swap freely: a point fitted to a bundle of lines is their point of closest approach.
//! The animation feeds the samples in, a few more each frame, and refits: the fitted point,
//! line, plane and meeting point (red) settle onto the truth (blue) as the turning view shows.
//!
//! The unit forms are singular: numga's general eigensolver sends the modes of the free
//! coefficients to infinity, and so does gax's `eigh_semidefinite`, which lists the finite
//! modes first.

use gax::pga3d::{Line, Motor, Plane, Point, intrinsic};
use gax::{Form, Kind, Reverse, ScalarProduct};

use gax_numga_examples::rng::{Draw, Rng, rng};
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Light, Marker, Point2, Rect, Scene3, backdrop, caption,
    palette, run,
};

mod fitting {
    use super::*;

    /// A line through points, oriented along itself (intrinsic), as a join of points is: the
    /// sampled rays and the line fitted to points.
    pub type L = intrinsic::Line;
    pub type M = gax::Unit<Motor>;

    /// The squared norm of `r`, the scalar part of `r ~r`: a form on the unknown when `r` is
    /// linear in it.
    pub fn squared<R>(r: R) -> <R::Output as ScalarProduct<R>>::Output
    where
        R: Reverse + Copy,
        R::Output: ScalarProduct<R>,
    {
        r.reverse().scalar_product(r)
    }

    /// The least finite mode of the pencil `(misfit, unit)`, `unit` semidefinite.
    pub fn least_mode<F: Form<Coef = f64>>(misfit: F, unit: F) -> <F::Slot as Kind>::Mv<(), f64> {
        misfit.eigh_semidefinite(unit).1.as_ref()[0]
    }

    /// A point fitted to points: the misfit sums the squared lines joining each sample to the
    /// unknown point.
    pub fn point_to_points(samples: &[Point]) -> Point {
        let x = Point::slot();
        let misfit = samples.iter().map(|s| squared(s.unitized() & x)).sum();
        least_mode(misfit, squared(x))
    }

    /// A line fitted to points: the squared planes joining each sample to the unknown line. The
    /// unit form leaves out the condition to be a line (`L ^ L = 0`); the fitted moment is the
    /// centroid's about the fitted direction, which meets it.
    pub fn line_to_points(samples: &[Point]) -> L {
        let x = L::slot();
        let misfit = samples.iter().map(|s| squared(s.unitized() & x)).sum();
        least_mode(misfit, squared(x))
    }

    /// A plane fitted to points: the squared signed distances.
    pub fn plane_to_points(samples: &[Point]) -> Plane {
        let x = Plane::slot();
        let misfit = samples.iter().map(|s| squared(s.unitized() & x)).sum();
        least_mode(misfit, squared(x))
    }

    /// A point fitted to lines: the squared planes joining each line to the unknown point.
    pub fn point_to_lines(samples: &[L]) -> Point {
        let x = Point::slot();
        let misfit = samples.iter().map(|l| squared(l.normalized() & x)).sum();
        least_mode(misfit, squared(x))
    }

    /// The shared pose of every scene.
    pub fn pose() -> M {
        Line::new(0.0, 0.0, 0.0, -0.4, 0.3, -0.6).exp()
            * Line::new(0.3, 0.0, 0.0, 0.0, 0.0, 0.0).exp()
            * Line::new(0.0, 0.0, 0.5, 0.0, 0.0, 0.0).exp()
    }

    pub fn cloud(n: usize, spread: f64, rng: &mut Rng) -> Vec<Point> {
        (0..n)
            .map(|_| {
                Point::xyz(
                    spread * rng.normal(),
                    spread * rng.normal(),
                    spread * rng.normal(),
                )
            })
            .collect()
    }

    /// Points along the y axis.
    pub fn segment(n: usize, half_length: f64) -> Vec<Point> {
        (0..n)
            .map(|i| {
                Point::xyz(
                    0.0,
                    half_length * (2.0 * i as f64 / (n - 1) as f64 - 1.0),
                    0.0,
                )
            })
            .collect()
    }

    /// Points on a square of the plane `z = 0`.
    pub fn patch(n: usize, half_width: f64, rng: &mut Rng) -> Vec<Point> {
        let mut across = || rng.range(-half_width, half_width);
        (0..n)
            .map(|_| Point::xyz(across(), across(), 0.0))
            .collect()
    }

    /// Move each point by an independent Gaussian translation.
    pub fn jitter(points: &[Point], sigma: f64, rng: &mut Rng) -> Vec<Point> {
        points
            .iter()
            .map(|p| {
                Motor::translation(Point::direction(
                    sigma * rng.normal(),
                    sigma * rng.normal(),
                    sigma * rng.normal(),
                )) >> *p
            })
            .collect()
    }

    /// Lines with unit directions, passing near the origin.
    pub fn bundle(n: usize, spread: f64, rng: &mut Rng) -> Vec<L> {
        (0..n)
            .map(|_| {
                let d: gax::vga3d::Vector = rng.direction();
                let foot = Point::xyz(
                    spread * rng.normal(),
                    spread * rng.normal(),
                    spread * rng.normal(),
                );
                foot & Point::direction(d.e1(), d.e2(), d.e3())
            })
            .collect()
    }

    /// The planes `y = ±half_length`, which cut a segment out of a line along y.
    pub fn end_planes(half_length: f64) -> [Plane; 2] {
        [
            Plane::new(0.0, 1.0, 0.0, -half_length),
            Plane::new(0.0, 1.0, 0.0, half_length),
        ]
    }

    /// The four edge lines of a square of the plane `z = 0`: meets of planes.
    pub fn patch_edges(half_width: f64) -> [Line; 4] {
        let s = [1.0, 1.0, -1.0, -1.0];
        let t = [1.0, -1.0, -1.0, 1.0];
        core::array::from_fn(|i| {
            Plane::new(1.0, 0.0, 0.0, -s[i] * half_width)
                ^ Plane::new(0.0, 1.0, 0.0, -t[i] * half_width)
        })
    }

    /// The samples of the four scenes, posed and jittered.
    pub struct Samples {
        pub cloud: Vec<Point>,
        pub segment: Vec<Point>,
        pub patch: Vec<Point>,
        pub bundle: Vec<L>,
    }

    pub fn samples() -> Samples {
        let pose = pose();
        let posed = |v: Vec<Point>| -> Vec<Point> { v.iter().map(|p| pose >> *p).collect() };
        let mut rng = rng(0x0f17_0001);
        let cloud = posed(cloud(200, 0.5, &mut rng));
        let cloud = jitter(&cloud, 0.05, &mut rng);
        let segment = jitter(&posed(segment(120, 2.0)), 0.05, &mut rng);
        let patch = posed(patch(200, 2.0, &mut rng));
        let patch = jitter(&patch, 0.05, &mut rng);
        let bundle = bundle(30, 0.05, &mut rng)
            .iter()
            .map(|l| pose >> *l)
            .collect();
        Samples {
            cloud,
            segment,
            patch,
            bundle,
        }
    }

    /// A line's direction as a unit ideal point (its sign does not matter to the callers: a
    /// segment drawn both ways, a projector `d dᵀ`).
    pub fn direction(l: L) -> Point {
        let d = l.direction();
        d.gp(1.0 / d.ideal_norm())
    }
}

use fitting::*;

const SECONDS: f32 = 8.0;

/// How much of the samples the frame at phase `s` (0 to 1) has seen: growing, then all.
fn seen(n: usize, s: f32, least: usize) -> usize {
    let grow = (s / 0.7).min(1.0);
    least + ((n - least) as f32 * grow * grow) as usize
}

/// One panel: a framed tile seen by `cam`, its title and sample count.
fn tile(c: &mut Canvas, cam: Camera, title: &str, n: usize, draw: impl FnOnce(&mut Scene3)) {
    backdrop(c);
    // The frame through the centres of the edge pixels.
    let panel = c.rect();
    let edge = panel.inset(0.5, 0.5, 0.5, 0.5);
    let corners = [edge.lo, edge.top_right(), edge.hi, edge.bottom_left()];
    c.polyline(&corners, 1.0, palette::GRID.faded(0.4), true);
    let mut s = Scene3::new(cam);
    draw(&mut s);
    s.draw(c);
    let size = (panel.height() / 22.0).clamp(8.0, 12.0);
    let left = panel.lo + Point2::direction(10.0, size * 1.6);
    c.text(title, left, size, palette::INK, Align::Left);
    let right = panel.top_right() + Point2::direction(-10.0, size * 1.6);
    let count = format!("{n} SAMPLES");
    c.text(&count, right, size * 0.85, palette::GRID, Align::Right);
}

fn samples_dots(s: &mut Scene3, pts: &[Point]) {
    for p in pts {
        s.dot(
            *p,
            Marker::Dot,
            3.5,
            palette::GRID.mix_light(palette::INK, 0.45),
        );
    }
}

/// The truth's colour and the fit's.
const TRUTH: Light = palette::SKY;
const FIT: Light = palette::RED;

/// The point nearest the cloud's samples seen at `phase`, against the pose's centre.
fn draw_point_fit(c: &mut Canvas, cam: Camera, data: &Samples, phase: f32, centre: Point) {
    let n = seen(data.cloud.len(), phase, 4);
    let fitted = point_to_points(&data.cloud[..n]);
    tile(c, cam, "POINT: MIN |P V X|", n, |s| {
        samples_dots(s, &data.cloud[..n]);
        s.dot(centre, Marker::Ring, 20.0, TRUTH);
        s.dot(fitted, Marker::Dot, 9.0, FIT);
    });
}

/// The line nearest the segment's samples seen at `phase`, against the pose's line.
fn draw_line_fit(c: &mut Canvas, cam: Camera, data: &Samples, phase: f32, pose: M) {
    let n = seen(data.segment.len(), phase, 3);
    // Feed the segment in a scrambled order, so that the early fits see all of it.
    let pts: Vec<Point> = (0..n).map(|i| data.segment[(i * 37) % 120]).collect();
    let fitted = line_to_points(&pts);
    let ends = end_planes(2.5).map(|e| pose >> e);
    let true_line = pose >> (Point::origin() & Point::xyz(0.0, 1.0, 0.0));
    tile(c, cam, "LINE: MIN |P V L|", n, |s| {
        samples_dots(s, &pts);
        s.seg(
            true_line ^ ends[0],
            true_line ^ ends[1],
            2.5,
            TRUTH.faded(0.8),
        );
        s.seg(fitted ^ ends[0], fitted ^ ends[1], 2.0, FIT);
    });
}

/// The plane nearest the patch's samples seen at `phase`, against the pose's plane.
fn draw_plane_fit(c: &mut Canvas, cam: Camera, data: &Samples, phase: f32, pose: M) {
    let n = seen(data.patch.len(), phase, 4);
    let fitted = plane_to_points(&data.patch[..n]);
    let edges = patch_edges(2.0).map(|e| pose >> e);
    let true_plane = pose >> Plane::new(0.0, 0.0, 1.0, 0.0);
    tile(c, cam, "PLANE: MIN |P V PI|", n, |s| {
        samples_dots(s, &data.patch[..n]);
        let quad = |p: Plane| edges.map(|e| p ^ e);
        let (tq, fq) = (quad(true_plane), quad(fitted));
        let mut tl = tq.to_vec();
        tl.push(tq[0]);
        s.polyline(&tl, 2.5, TRUTH.faded(0.8));
        s.quad(fq[0], fq[1], fq[2], fq[3], FIT, 0.18);
        let mut fl = fq.to_vec();
        fl.push(fq[0]);
        s.polyline(&fl, 2.0, FIT);
    });
}

/// The point nearest the bundle's lines seen at `phase`, against the pose's centre.
fn draw_point_of_lines_fit(c: &mut Canvas, cam: Camera, data: &Samples, phase: f32, centre: Point) {
    let n = seen(data.bundle.len(), phase, 3);
    let fitted = point_to_lines(&data.bundle[..n]).unitized();
    tile(c, cam, "POINT OF LINES: MIN |L V X|", n, |s| {
        for l in &data.bundle[..n] {
            // The stretch of each line about its point nearest the fit: the
            // meet of the line with the plane through the fit orthogonal to it.
            let d = direction(*l);
            let foot = (*l ^ (*l | fitted)).unitized();
            let (a, b) = (foot + d.gp(-2.0), foot + d.gp(2.0));
            s.seg(a, b, 0.8, palette::GRID.faded(0.7));
        }
        s.dot(centre, Marker::Ring, 20.0, TRUTH);
        s.dot(fitted, Marker::Dot, 9.0, FIT);
    });
}

fn draw(c: &mut Canvas, t: f32) {
    let phase = t / SECONDS;
    let az = 0.5 + core::f32::consts::TAU * phase;
    let data = samples();
    let pose = pose();
    let centre = pose >> Point::origin();
    let screen = c.rect();
    let header = 64.0;
    let (w, h) = (screen.width() / 2.0, (screen.height() - header) / 2.0);
    let camera = |view: Rect, distance: f32| {
        Camera::orbit(view, centre, distance, az, 0.45, Lens::Perspective(0.6))
    };
    for k in 0..4 {
        let at = Point2::xy((k % 2) as f32 * w, header + (k / 2) as f32 * h);
        let mut sub = c.sub(Rect::new(0.0, 0.0, w, h));
        let view = sub.rect();
        match k {
            0 => draw_point_fit(&mut sub, camera(view, 4.0), &data, phase, centre),
            1 => draw_line_fit(&mut sub, camera(view, 9.0), &data, phase, pose),
            2 => draw_plane_fit(&mut sub, camera(view, 10.0), &data, phase, pose),
            _ => draw_point_of_lines_fit(&mut sub, camera(view, 7.0), &data, phase, centre),
        }
        c.blit(&sub, at);
    }
    caption(
        c,
        "FITTING: ONE EIGENPROBLEM FOR ALL FLATS",
        "JOINS (V) WITH THE SAMPLES: TRUTH BLUE, FIT RED (PGA3D)",
    );
}

fn main() {
    run(Anim::new("fitting", SECONDS).size(960, 600), draw);
}

#[cfg(test)]
mod tests {
    use super::fitting::*;
    use gax::pga3d::{Line, Plane, Point, Scalar};
    use gax::{ApproxEq, vga3d};
    use gax_numga_examples::rng::rng;

    /// Whether two values agree as projective elements, up to scale and sign.
    fn same_element<const N: usize>(a: [f64; N], b: [f64; N], tol: f64) -> bool {
        let max = |v: [f64; N]| v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let (a, b) = (a.map(|x| x / max(a)), b.map(|x| x / max(b)));
        a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= tol)
            || a.iter().zip(&b).all(|(x, y)| (x + y).abs() <= tol)
    }

    fn rank<const N: usize>(eigenvalues: [f64; N]) -> usize {
        eigenvalues.iter().filter(|v| v.abs() > 1e-12).count()
    }

    /// The unit forms measure a point's weight, a line's direction and a plane's normal: they
    /// are degenerate on the coefficients least squares leaves free.
    #[test]
    fn unit_forms_are_degenerate_on_the_free_coefficients() {
        let ranks = [
            rank(squared(Point::slot()).eigh(Scalar::identity()).0),
            rank(squared(Line::slot()).eigh(Scalar::identity()).0),
            rank(squared(Plane::slot()).eigh(Scalar::identity()).0),
        ];
        assert_eq!(ranks, [1, 3, 3]);
    }

    #[test]
    fn exact_samples_recover_line_and_plane_exactly() {
        let pose = pose();
        let seg: Vec<Point> = segment(50, 2.0).iter().map(|p| pose >> *p).collect();
        let truth = pose >> (Point::origin() & Point::xyz(0.0, 1.0, 0.0));
        assert!(same_element(line_to_points(&seg).c, truth.c, 1e-10));
        let pat: Vec<Point> = patch(50, 2.0, &mut rng(1))
            .iter()
            .map(|p| pose >> *p)
            .collect();
        let plane = pose >> Plane::new(0.0, 0.0, 1.0, 0.0);
        assert!(same_element(plane_to_points(&pat).c, plane.c, 1e-10));
    }

    #[test]
    fn point_fit_is_the_centroid() {
        let mut draws = rng(2);
        let pose = pose();
        let cloud: Vec<Point> = cloud(300, 0.5, &mut draws)
            .iter()
            .map(|p| pose >> *p)
            .collect();
        let points = jitter(&cloud, 0.05, &mut draws);
        let centroid = point_to_points(&points).unitized();
        assert!(centroid.approx_eq(&mean(&points), 1e-10));
    }

    /// The mean of points: their sum, unitized.
    fn mean(points: &[Point]) -> Point {
        points
            .iter()
            .map(|p| p.unitized())
            .sum::<Point>()
            .unitized()
    }

    /// The principal axes of points about their centroid, by the symmetric eigenproblem of the
    /// scatter form on vectors (ascending).
    fn principal_axes(points: &[Point]) -> [vga3d::Vector; 3] {
        let centre = mean(points);
        let v = vga3d::Vector::slot();
        let scatter: vga3d::Scalar<(vga3d::Vector, vga3d::Vector)> = points
            .iter()
            .map(|p| {
                let d = p.unitized() - centre;
                let d = vga3d::Vector::new(d.e032(), d.e013(), d.e021());
                (d | v) * (d | v)
            })
            .sum();
        scatter.eigh(vga3d::Scalar::identity()).1
    }

    /// The cosine of the angle between a direction and a vector: the inner product of the
    /// planes through the origin orthogonal to them, normalized.
    fn cosine(d: Point, v: vga3d::Vector) -> f64 {
        let (a, b) = (Plane::orthogonal_to(d), Plane::from(v));
        (a | b).s() / (a.norm() * b.norm())
    }

    #[test]
    fn line_fit_matches_principal_axis_and_is_a_line() {
        let pose = pose();
        let seg: Vec<Point> = segment(200, 2.0).iter().map(|p| pose >> *p).collect();
        let points = jitter(&seg, 0.05, &mut rng(3));
        let line = line_to_points(&points);
        let scale = line.c.iter().fold(0.0f64, |m, x| m.max(x.abs())).powi(2);
        assert!((line ^ line).c.iter().all(|v| (v / scale).abs() < 1e-13));
        let axis = principal_axes(&points)[2];
        assert!((cosine(direction(line), axis).abs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn plane_fit_matches_the_normal_through_the_centroid() {
        let mut draws = rng(4);
        let pose = pose();
        let pat: Vec<Point> = patch(300, 2.0, &mut draws)
            .iter()
            .map(|p| pose >> *p)
            .collect();
        let points = jitter(&pat, 0.05, &mut draws);
        let plane = plane_to_points(&points);
        let normal = principal_axes(&points)[0];
        let along = (plane | Plane::from(normal)).s() / plane.norm();
        assert!((along.abs() - 1.0).abs() < 1e-6);
        let centroid = point_to_points(&points);
        assert!((centroid.unitized() & plane).s().abs() < 1e-10);
    }

    #[test]
    fn point_fitted_to_a_bundle_is_the_point_of_closest_approach() {
        let mut draws = rng(5);
        let pose = pose();
        let exact: Vec<L> = bundle(20, 0.0, &mut draws)
            .iter()
            .map(|l| pose >> *l)
            .collect();
        assert!(same_element(
            point_to_lines(&exact).c,
            (pose >> Point::origin()).c,
            1e-10
        ));
        let rays: Vec<L> = bundle(40, 0.05, &mut draws)
            .iter()
            .map(|l| pose >> *l)
            .collect();
        let meet = point_to_lines(&rays).to_euclidean();
        // The classical closed form: Σ (I - d dᵀ) x = Σ (I - d dᵀ) p, with p on each line.
        let mut a = [[0.0; 3]; 3];
        let mut b = [0.0; 3];
        for l in &rays {
            let d = direction(*l);
            // A point on the line: its meet with the plane through the origin normal to it.
            let p = (*l ^ Plane::orthogonal_to(d)).to_euclidean();
            let d = [d.e032(), d.e013(), d.e021()];
            for i in 0..3 {
                for j in 0..3 {
                    let proj = f64::from(u8::from(i == j)) - d[i] * d[j];
                    a[i][j] += proj;
                    b[i] += proj * p[j];
                }
            }
        }
        let map = gax::vga3d::Vector::<(gax::vga3d::Vector,)>::from_coeffs(a);
        let x = map.solve(gax::vga3d::Vector::new(b[0], b[1], b[2]));
        for (m, c) in meet.iter().zip(x.c) {
            assert!((m - c).abs() < 1e-8);
        }
    }

    #[test]
    fn a_frame_draws() {
        let mut draw = super::draw;
        let c = gax_numga_examples::app::frame(
            &gax_numga_examples::Anim::new("t", 1.0).size(320, 240),
            0.5,
            &mut draw,
        );
        assert!(c.mean().luma() > 0.0);
    }
}
