//! A 3D scene drawn by the painter's algorithm: triangles, segments and dots projected by a
//! [`Camera`] and drawn far to near, so translucent surfaces and wireframes overlap as they
//! should (without intersecting triangles split). Surfaces from a parametrization, spheres,
//! arrows and axes are built from these.

use crate::canvas::{Canvas, Px, Rgb, scale};
use crate::plot::{Marker, arrow, mark};
use crate::points::{Dir3, Point3, Pos3, finite};
use crate::view::Camera;
use gax::pga3d::{Motor, Plane, Point};

enum Prim {
    Tri([Px; 3], Rgb, f32),
    Seg(Px, Px, f32, Rgb, f32),
    Dot(Px, Marker, f32, Rgb),
    Arrow(Px, Px, f32, f32, Rgb),
}

/// Primitives collected for one frame.
pub struct Scene3 {
    /// The camera.
    pub cam: Camera,
    prims: Vec<(f32, Prim)>,
    /// The direction towards the light (for [`Scene3::lit`]), unit.
    pub light: Point3,
}

/// The direction `d` at unit length.
fn unit(d: Point<(), f32>) -> Point<(), f32> {
    d.gp(d.ideal_norm().max(1e-12).recip())
}

impl Scene3 {
    /// An empty scene seen by `cam`, lit from above and behind the viewer's left shoulder.
    pub fn new(cam: Camera) -> Scene3 {
        // The direction to the eye, turned a little to the left and up.
        let to_eye = cam.eye().unitized() - Point::xyz(0.0, 0.0, 0.0);
        let towards = unit(to_eye) + Point::direction(-0.3, 0.2, 0.9);
        Scene3 {
            cam,
            prims: Vec::new(),
            light: unit(towards),
        }
    }

    fn depth(&self, p: Point3) -> f32 {
        self.cam.depth(p)
    }

    /// `color` shaded by how squarely the triangle `a b c` faces the light (two-sided): the
    /// inner product of its plane `a & b & c` with the plane facing the light, the cosine
    /// between their normals once divided by the face's norm.
    pub fn lit(&self, a: impl Pos3, b: impl Pos3, c: impl Pos3, color: Rgb) -> Rgb {
        let face = a.point3() & b.point3() & c.point3();
        let facing = Plane::orthogonal_to(self.light);
        let k = 0.35 + 0.65 * ((face | facing).s() / face.norm().max(1e-12)).abs();
        scale(color, k)
    }

    /// A triangle.
    pub fn tri(&mut self, a: impl Pos3, b: impl Pos3, c: impl Pos3, color: Rgb, alpha: f32) {
        let (a, b, c) = (a.point3(), b.point3(), c.point3());
        if let (Some(pa), Some(pb), Some(pc)) = (self.cam.px(a), self.cam.px(b), self.cam.px(c)) {
            let d = (self.depth(a) + self.depth(b) + self.depth(c)) / 3.0;
            self.prims.push((d, Prim::Tri([pa, pb, pc], color, alpha)));
        }
    }

    /// A quadrilateral `a b c d` (two triangles).
    pub fn quad(
        &mut self,
        a: impl Pos3,
        b: impl Pos3,
        c: impl Pos3,
        d: impl Pos3,
        color: Rgb,
        alpha: f32,
    ) {
        let (a, b, c, d) = (a.point3(), b.point3(), c.point3(), d.point3());
        self.tri(a, b, c, color, alpha);
        self.tri(a, c, d, color, alpha);
    }

    /// A segment `width` pixels wide.
    pub fn seg(&mut self, a: impl Pos3, b: impl Pos3, width: f32, color: Rgb, alpha: f32) {
        let (a, b) = (a.point3(), b.point3());
        if let (Some(pa), Some(pb)) = (self.cam.px(a), self.cam.px(b)) {
            let d = (self.depth(a) + self.depth(b)) * 0.5;
            self.prims.push((d, Prim::Seg(pa, pb, width, color, alpha)));
        }
    }

    /// Segments through `pts`; non-finite points break the line.
    pub fn polyline(&mut self, pts: &[impl Pos3], width: f32, color: Rgb, alpha: f32) {
        let pts: Vec<Point3> = pts.iter().map(|p| p.point3()).collect();
        for w in pts.windows(2) {
            if finite(w[0]) && finite(w[1]) {
                self.seg(w[0], w[1], width, color, alpha);
            }
        }
    }

    /// A marker of `size` pixels.
    pub fn dot(&mut self, p: impl Pos3, marker: Marker, size: f32, color: Rgb) {
        let p = p.point3();
        if let Some(q) = self.cam.px(p) {
            self.prims
                .push((self.depth(p) - 1e-3, Prim::Dot(q, marker, size, color)));
        }
    }

    /// An arrow from `p` along `v`.
    pub fn arrow(&mut self, p: impl Pos3, v: impl Dir3, width: f32, head: f32, color: Rgb) {
        let p = p.point3().unitized();
        let q = p + v.dir3();
        if let (Some(a), Some(b)) = (self.cam.px(p), self.cam.px(q)) {
            let d = (self.depth(p) + self.depth(q)) * 0.5;
            self.prims.push((d, Prim::Arrow(a, b, width, head, color)));
        }
    }

    /// A surface `f(u, v)` over `[0, 1]²` in `nu` x `nv` quads, coloured by `color(u, v)` and lit,
    /// with optional edges.
    #[allow(clippy::too_many_arguments)]
    pub fn surface<P: Pos3>(
        &mut self,
        f: impl Fn(f32, f32) -> P,
        nu: usize,
        nv: usize,
        color: impl Fn(f32, f32) -> Rgb,
        alpha: f32,
        edges: Option<(Rgb, f32)>,
    ) {
        let p = |i: usize, j: usize| f(i as f32 / nu as f32, j as f32 / nv as f32).point3();
        for i in 0..nu {
            for j in 0..nv {
                let (a, b, c, d) = (p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1));
                if ![a, b, c, d].into_iter().all(finite) {
                    continue;
                }
                let col = self.lit(
                    a,
                    b,
                    c,
                    color((i as f32 + 0.5) / nu as f32, (j as f32 + 0.5) / nv as f32),
                );
                self.quad(a, b, c, d, col, alpha);
                if let Some((ec, w)) = edges {
                    self.seg(a, b, w, ec, 0.6);
                    self.seg(a, d, w, ec, 0.6);
                }
            }
        }
    }

    /// A wireframe sphere: `n` meridians and `n / 2` parallels.
    pub fn sphere_wire(&mut self, centre: impl Pos3, r: f32, n: usize, color: Rgb, alpha: f32) {
        let centre = centre.point3().unitized();
        // The direction `r` along x, raised by the latitude (about -y) and turned by the
        // longitude (about z), from the centre.
        let pt = |lon: f32, lat: f32| {
            let turn = Motor::rotation_about(0.0, 0.0, 1.0, lon)
                * Motor::rotation_about(0.0, -1.0, 0.0, lat);
            centre + (turn >> Point::direction(r, 0.0, 0.0))
        };
        let tau = core::f32::consts::TAU;
        let steps = 48;
        for k in 0..n {
            let lon = tau * k as f32 / n as f32;
            let line: Vec<Point3> = (0..=steps)
                .map(|s| pt(lon, -tau / 4.0 + tau / 2.0 * s as f32 / steps as f32))
                .collect();
            self.polyline(&line, 1.0, color, alpha);
        }
        for k in 1..n / 2 {
            let lat = -tau / 4.0 + tau / 2.0 * k as f32 / (n / 2) as f32;
            let line: Vec<Point3> = (0..=steps)
                .map(|s| pt(tau * s as f32 / steps as f32, lat))
                .collect();
            self.polyline(&line, 1.0, color, alpha);
        }
    }

    /// Axes from `origin`, `len` long, in red, green and blue.
    pub fn axes(&mut self, origin: impl Pos3, len: f32) {
        let origin = origin.point3();
        let axes = [
            (Point::direction(len, 0.0, 0.0), crate::palette::red()),
            (Point::direction(0.0, len, 0.0), crate::palette::green()),
            (Point::direction(0.0, 0.0, len), crate::palette::blue()),
        ];
        for (v, col) in axes {
            self.arrow(origin, v, 1.5, 8.0, col);
        }
    }

    /// Draw everything, far to near.
    pub fn draw(mut self, c: &mut Canvas) {
        self.prims.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (_, p) in self.prims {
            match p {
                Prim::Tri(t, col, a) => c.fill(&t, col, a),
                Prim::Seg(a, b, w, col, al) => c.line(a, b, w, col, al),
                Prim::Dot(q, m, s, col) => mark(c, q, m, s, col, 1.0),
                Prim::Arrow(a, b, w, h, col) => arrow(c, a, b, w, h, col, 1.0),
            }
        }
    }
}

/// A 3D panel: `cam` drawing into the canvas rectangle `rect` (clipped to it), with what `fill`
/// adds to the scene. Returns the camera as placed, for drawing more over the panel.
pub fn panel3(
    c: &mut Canvas,
    rect: [f32; 4],
    cam: Camera,
    fill: impl FnOnce(&mut Scene3),
) -> Camera {
    let cam = cam.viewport(rect);
    c.clip(rect);
    let mut scene = Scene3::new(cam);
    fill(&mut scene);
    scene.draw(c);
    c.unclip();
    cam
}
