//! A 3D scene drawn by the painter's algorithm: triangles, segments and dots projected by a
//! [`Camera`] and drawn far to near, so translucent surfaces and wireframes overlap as they
//! should (without intersecting triangles split). Surfaces from a parametrization, spheres,
//! arrows and axes are built from these.

use crate::canvas::{Canvas, Px, Rgb, scale};
use crate::coords::{Dir3, Pos3};
use crate::plot::{Marker, arrow, mark};
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
    pub light: [f32; 3],
}

fn point(p: [f32; 3]) -> Point<(), f32> {
    Point::xyz(p[0], p[1], p[2])
}

/// The direction `d` at unit length.
fn unit(d: Point<(), f32>) -> Point<(), f32> {
    d.gp(d.ideal_norm().max(1e-12).recip())
}

impl Scene3 {
    /// An empty scene seen by `cam`, lit from above and behind the viewer's left shoulder.
    pub fn new(cam: Camera) -> Scene3 {
        // The direction to the eye, turned a little to the left and up.
        let [x, y, z] = cam.eye();
        let towards = unit(Point::direction(x, y, z)) + Point::direction(-0.3, 0.2, 0.9);
        let l = unit(towards);
        Scene3 {
            cam,
            prims: Vec::new(),
            light: [l.e032(), l.e013(), l.e021()],
        }
    }

    fn depth(&self, p: [f32; 3]) -> f32 {
        self.cam.local(p)[2]
    }

    /// `color` shaded by how squarely the triangle `a b c` faces the light (two-sided): the
    /// inner product of its plane `a & b & c` with the plane facing the light, the cosine
    /// between their normals once divided by the face's norm.
    pub fn lit(&self, a: impl Pos3, b: impl Pos3, c: impl Pos3, color: Rgb) -> Rgb {
        let a = a.xyz();
        let b = b.xyz();
        let c = c.xyz();
        let face = point(a) & point(b) & point(c);
        let [x, y, z] = self.light;
        let facing = Plane::orthogonal_to(Point::direction(x, y, z));
        let k = 0.35 + 0.65 * ((face | facing).s() / face.norm().max(1e-12)).abs();
        scale(color, k)
    }

    /// A triangle.
    pub fn tri(&mut self, a: impl Pos3, b: impl Pos3, c: impl Pos3, color: Rgb, alpha: f32) {
        let a = a.xyz();
        let b = b.xyz();
        let c = c.xyz();
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
        let a = a.xyz();
        let b = b.xyz();
        let c = c.xyz();
        let d = d.xyz();
        self.tri(a, b, c, color, alpha);
        self.tri(a, c, d, color, alpha);
    }

    /// A segment `width` pixels wide.
    pub fn seg(&mut self, a: impl Pos3, b: impl Pos3, width: f32, color: Rgb, alpha: f32) {
        let a = a.xyz();
        let b = b.xyz();
        if let (Some(pa), Some(pb)) = (self.cam.px(a), self.cam.px(b)) {
            let d = (self.depth(a) + self.depth(b)) * 0.5;
            self.prims.push((d, Prim::Seg(pa, pb, width, color, alpha)));
        }
    }

    /// Segments through `pts`; non-finite points break the line.
    pub fn polyline(&mut self, pts: &[impl Pos3], width: f32, color: Rgb, alpha: f32) {
        let pts: Vec<[f32; 3]> = pts.iter().map(|p| p.xyz()).collect();
        for w in pts.windows(2) {
            if w[0].iter().chain(&w[1]).all(|v| v.is_finite()) {
                self.seg(w[0], w[1], width, color, alpha);
            }
        }
    }

    /// A marker of `size` pixels.
    pub fn dot(&mut self, p: impl Pos3, marker: Marker, size: f32, color: Rgb) {
        let p = p.xyz();
        if let Some(q) = self.cam.px(p) {
            self.prims
                .push((self.depth(p) - 1e-3, Prim::Dot(q, marker, size, color)));
        }
    }

    /// An arrow from `p` along `v`.
    pub fn arrow(&mut self, p: impl Pos3, v: impl Dir3, width: f32, head: f32, color: Rgb) {
        let p = p.xyz();
        let v = v.dxyz();
        let q = [p[0] + v[0], p[1] + v[1], p[2] + v[2]];
        if let (Some(a), Some(b)) = (self.cam.px(p), self.cam.px(q)) {
            let d = (self.depth(p) + self.depth(q)) * 0.5;
            self.prims.push((d, Prim::Arrow(a, b, width, head, color)));
        }
    }

    /// A surface `f(u, v)` over `[0, 1]²` in `nu` x `nv` quads, coloured by `color(u, v)` and lit,
    /// with optional edges.
    #[allow(clippy::too_many_arguments)]
    pub fn surface(
        &mut self,
        f: impl Fn(f32, f32) -> [f32; 3],
        nu: usize,
        nv: usize,
        color: impl Fn(f32, f32) -> Rgb,
        alpha: f32,
        edges: Option<(Rgb, f32)>,
    ) {
        let p = |i: usize, j: usize| f(i as f32 / nu as f32, j as f32 / nv as f32);
        for i in 0..nu {
            for j in 0..nv {
                let (a, b, c, d) = (p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1));
                if [a, b, c, d].iter().flatten().any(|v| !v.is_finite()) {
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
        let centre = centre.xyz();
        // The point `r` along x, raised by the latitude (about -y) and turned by the longitude
        // (about z), then moved to the centre.
        let to_centre = Motor::translation(centre[0], centre[1], centre[2]);
        let pt = |lon: f32, lat: f32| {
            let turn = Motor::rotation_about(0.0, 0.0, 1.0, lon)
                * Motor::rotation_about(0.0, -1.0, 0.0, lat);
            ((to_centre * turn) >> Point::xyz(r, 0.0, 0.0)).to_euclidean()
        };
        let tau = core::f32::consts::TAU;
        let steps = 48;
        for k in 0..n {
            let lon = tau * k as f32 / n as f32;
            let line: Vec<[f32; 3]> = (0..=steps)
                .map(|s| pt(lon, -tau / 4.0 + tau / 2.0 * s as f32 / steps as f32))
                .collect();
            self.polyline(&line, 1.0, color, alpha);
        }
        for k in 1..n / 2 {
            let lat = -tau / 4.0 + tau / 2.0 * k as f32 / (n / 2) as f32;
            let line: Vec<[f32; 3]> = (0..=steps)
                .map(|s| pt(tau * s as f32 / steps as f32, lat))
                .collect();
            self.polyline(&line, 1.0, color, alpha);
        }
    }

    /// Axes from `origin`, `len` long, in red, green and blue.
    pub fn axes(&mut self, origin: impl Pos3, len: f32) {
        let origin = origin.xyz();
        let cols = [
            crate::palette::red(),
            crate::palette::green(),
            crate::palette::blue(),
        ];
        for (i, col) in cols.into_iter().enumerate() {
            let mut v = [0.0; 3];
            v[i] = len;
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
