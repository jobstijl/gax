// Used by the GPU agreement tests; a device without compute would use it too.
#![cfg_attr(not(test), allow(dead_code))]

//! The lattice on the CPU: the same traced `grid_node` and `source_force` kernels as the
//! compute shader, run by their SIMD batch forms (`gax::batch`). It is the fallback for a
//! device without compute, and the oracle the GPU is tested against.

use super::{GridSpec, Node};
use gax::pga2d::Point;

type P = Point<(), f32>;

/// The lattice's springs: stiffness, anchor stiffness and damping (the GPU uses the same).
pub const SPRING: f32 = 95.0;
/// Stiffness of the pull back to the rest position.
pub const ANCHOR: f32 = 5.0;
/// Velocity damping per second.
pub const DAMPING: f32 = 4.5;

/// The lattice state.
pub struct CpuGrid {
    spec: GridSpec,
    /// Positions.
    pub p: Vec<P>,
    /// Velocities.
    pub v: Vec<P>,
    rest: Vec<P>,
    n: Vec<[P; 4]>,
    f: Vec<P>,
    tmp: Vec<P>,
    out: Vec<(P, P)>,
}

impl CpuGrid {
    /// A lattice at rest.
    pub fn new(spec: GridSpec) -> CpuGrid {
        let rest: Vec<P> = (0..spec.cols * spec.rows)
            .map(|k| {
                let (i, j) = (k % spec.cols, k / spec.cols);
                Point::xy(
                    spec.origin[0] + i as f32 * spec.spacing,
                    spec.origin[1] + j as f32 * spec.spacing,
                )
            })
            .collect();
        let n = rest.len();
        CpuGrid {
            spec,
            p: rest.clone(),
            v: vec![Point::direction(0.0, 0.0); n],
            rest,
            n: vec![[Point::direction(0.0, 0.0); 4]; n],
            f: vec![Point::direction(0.0, 0.0); n],
            tmp: vec![Point::direction(0.0, 0.0); n],
            out: vec![(Point::direction(0.0, 0.0), Point::direction(0.0, 0.0)); n],
        }
    }

    /// Set the state from GPU nodes.
    pub fn set(&mut self, nodes: &[Node]) {
        for (k, node) in nodes.iter().enumerate() {
            self.p[k] = node.p.into();
            self.v[k] = node.v.into();
        }
    }

    /// One step of `dt` with `sources` (`[x, y, strength, radius²]`).
    pub fn step(&mut self, sources: &[[f32; 4]], dt: f32) {
        let (w, h) = (self.spec.cols as usize, self.spec.rows as usize);
        // External forces, source by source in the GPU's order.
        self.f.fill(Point::direction(0.0, 0.0));
        for s in sources {
            crate::source_force_batch(
                &self.p,
                &[Point::xy(s[0], s[1])],
                &[[s[2], s[3]]],
                &mut self.tmp,
            );
            for (f, a) in self.f.iter_mut().zip(&self.tmp) {
                *f += *a;
            }
        }
        for k in 0..w * h {
            let (i, j) = (k % w, k / w);
            let at = |x: usize| self.p[x];
            self.n[k] = if i == 0 || j == 0 || i == w - 1 || j == h - 1 {
                [self.p[k]; 4]
            } else {
                [at(k - 1), at(k + 1), at(k - w), at(k + w)]
            };
        }
        let k = [SPRING, self.spec.spacing, ANCHOR, DAMPING, dt];
        crate::grid_node_batch(
            &self.p,
            &self.v,
            &self.rest,
            &self.n,
            &self.f,
            &[k],
            &mut self.out,
        );
        for idx in 0..w * h {
            let (i, j) = (idx % w, idx / w);
            if i == 0 || j == 0 || i == w - 1 || j == h - 1 {
                // The border is pinned.
                self.p[idx] = self.rest[idx];
                self.v[idx] = Point::direction(0.0, 0.0);
            } else {
                (self.p[idx], self.v[idx]) = self.out[idx];
            }
        }
    }
}
