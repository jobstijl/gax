//! Drawing on triangle meshes of the plane, shared by the planar surfaces ports (Whitney forms,
//! Dirac–Kähler fields, Regge calculus): a field linear on each face shaded pixel by pixel, its
//! level lines, the wireframe and the rim. Each includes it with `#[path]` and uses part of it.
//!
//! A [`Flat`] mesh holds its vertices as drawing points and a grid of buckets over its faces,
//! so that each pixel finds the face under it and its barycentric weights there: the joins of
//! the pixel's point with each side over the join of the corners.

// Each port uses part of the shared module.
#![allow(dead_code)]

use gax_numga_examples::{Axes, Canvas, Light, Point2};

/// Buckets along each axis of the grid over the mesh.
const BUCKETS: usize = 48;

/// A triangle mesh of the plane, to draw on.
pub struct Flat {
    /// The vertices.
    pub points: Vec<Point2>,
    /// The faces, counter-clockwise.
    pub faces: Vec<[usize; 3]>,
    /// The box around the vertices, and the faces overlapping each bucket of the grid over it.
    low: [f32; 2],
    size: [f32; 2],
    buckets: Vec<Vec<usize>>,
}

impl Flat {
    /// The mesh of these vertices (`[x, y]`) and faces.
    pub fn new(coordinates: &[[f64; 2]], faces: &[[usize; 3]]) -> Flat {
        let points: Vec<Point2> = coordinates
            .iter()
            .map(|&[x, y]| Point2::xy(x as f32, y as f32))
            .collect();
        let place = |p: Point2| p.to_euclidean();
        let (mut low, mut high) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
        for &p in &points {
            let q = place(p);
            for k in 0..2 {
                low[k] = low[k].min(q[k]);
                high[k] = high[k].max(q[k]);
            }
        }
        let size = [0, 1].map(|k| (high[k] - low[k]).max(1e-9) / BUCKETS as f32);
        let mut buckets = vec![Vec::new(); BUCKETS * BUCKETS];
        let cell = |v: f32, k: usize| {
            (((v - low[k]) / size[k]).floor().max(0.0) as usize).min(BUCKETS - 1)
        };
        for (f, face) in faces.iter().enumerate() {
            let corners = face.map(|v| place(points[v]));
            let span = |k: usize| {
                let along = corners.map(|c| c[k]);
                let lo = along.iter().copied().fold(f32::INFINITY, f32::min);
                let hi = along.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                cell(lo, k)..=cell(hi, k)
            };
            for j in span(1) {
                for i in span(0) {
                    buckets[j * BUCKETS + i].push(f);
                }
            }
        }
        Flat {
            points,
            faces: faces.to_vec(),
            low,
            size,
            buckets,
        }
    }

    /// The face under a point and the point's barycentric weights in it, if any.
    pub fn locate(&self, q: Point2) -> Option<(usize, [f32; 3])> {
        let [x, y] = q.to_euclidean();
        let i = ((x - self.low[0]) / self.size[0]).floor();
        let j = ((y - self.low[1]) / self.size[1]).floor();
        if !(0.0..BUCKETS as f32).contains(&i) || !(0.0..BUCKETS as f32).contains(&j) {
            return None;
        }
        let bucket = &self.buckets[j as usize * BUCKETS + i as usize];
        bucket.iter().find_map(|&f| {
            let [a, b, c] = self.faces[f].map(|v| self.points[v]);
            let area = (a & b & c).s();
            let weights = [(q & b & c).s(), (a & q & c).s(), (a & b & q).s()].map(|w| w / area);
            weights.iter().all(|&w| w >= -1e-5).then_some((f, weights))
        })
    }

    /// The mesh shaded on the axes: each pixel (`samples` along each axis) in the light
    /// `colour` gives for the face under it and the weights there.
    pub fn shade(
        &self,
        c: &mut Canvas,
        ax: &Axes,
        samples: usize,
        colour: impl Fn(usize, [f32; 3]) -> Option<Light> + Sync,
    ) {
        ax.image(c, samples, |q| {
            let (f, w) = self.locate(q)?;
            colour(f, w)
        });
    }

    /// A field on the vertices, linear on each face, shaded by `colour` of its value.
    pub fn shade_field(
        &self,
        c: &mut Canvas,
        ax: &Axes,
        values: &[f64],
        colour: impl Fn(f64) -> Light + Sync,
    ) {
        self.shade(c, ax, 2, |f, w| {
            let face = self.faces[f];
            let v: f64 = (0..3).map(|k| f64::from(w[k]) * values[face[k]]).sum();
            Some(colour(v))
        });
    }

    /// The level line `value = level` of a field linear on each face: on each face it crosses,
    /// the segment between the points of its sides where the field takes the level.
    pub fn level(&self, values: &[f64], level: f64) -> Vec<[Point2; 2]> {
        let mut out = Vec::new();
        for face in &self.faces {
            let mut ends = Vec::new();
            for k in 0..3 {
                let (a, b) = (face[k], face[(k + 1) % 3]);
                let (va, vb) = (values[a] - level, values[b] - level);
                if (va < 0.0) != (vb < 0.0) {
                    let t = (va / (va - vb)) as f32;
                    let (pa, pb) = (self.points[a], self.points[b]);
                    ends.push(pa + (pb - pa).gp(t));
                }
            }
            if let [p, q] = ends[..] {
                out.push([p, q]);
            }
        }
        out
    }

    /// The edges, each once.
    pub fn wire(&self) -> Vec<[Point2; 2]> {
        let mut edges: Vec<[usize; 2]> = self
            .faces
            .iter()
            .flat_map(|f| [0, 1, 2].map(|k| sorted([f[k], f[(k + 1) % 3]])))
            .collect();
        edges.sort_unstable();
        edges.dedup();
        edges
            .into_iter()
            .map(|[a, b]| [self.points[a], self.points[b]])
            .collect()
    }

    /// The edges of a single face: the rim.
    pub fn rim(&self) -> Vec<[Point2; 2]> {
        let mut edges: Vec<[usize; 2]> = self
            .faces
            .iter()
            .flat_map(|f| [0, 1, 2].map(|k| sorted([f[k], f[(k + 1) % 3]])))
            .collect();
        edges.sort_unstable();
        let mut out = Vec::new();
        let mut k = 0;
        while k < edges.len() {
            let run = edges[k..].iter().take_while(|e| **e == edges[k]).count();
            if run == 1 {
                out.push([self.points[edges[k][0]], self.points[edges[k][1]]]);
            }
            k += run;
        }
        out
    }

    /// Each face's centroid.
    pub fn centroid(&self, f: usize) -> Point2 {
        let [a, b, c] = self.faces[f].map(|v| self.points[v]);
        (a + b + c).unitized()
    }
}

/// An edge with its lower vertex first.
fn sorted([a, b]: [usize; 2]) -> [usize; 2] {
    [a.min(b), a.max(b)]
}

/// An arrow of a field: from `at`, along `d` (data units), as a shaft and a head.
pub fn arrow(c: &mut Canvas, ax: &Axes, at: Point2, d: [f64; 2], width: f32, head: f32, l: Light) {
    let tip = at + Point2::direction(d[0] as f32, d[1] as f32);
    ax.arrow(c, at, tip, width, head, l);
}
