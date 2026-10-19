//! The lattice on the CPU, for the tests: the same traced `grid_node`, `lattice_rest` and
//! `add_source_force` kernels as the compute shader, run by their SIMD batch forms
//! (`gax::batch`) on the same definitions (`lattice.rs`). It is the oracle the GPU is tested
//! against.

use super::gpu_types::Node;
use super::lattice::{GridSpec, SPRINGS, Source};
use gax::pga2d::Point;

type P = Point<(), f32>;

/// The lattice state.
pub struct CpuGrid {
    spec: GridSpec,
    /// Positions.
    pub p: Vec<P>,
    /// Velocities.
    pub v: Vec<P>,
    rest: Vec<P>,
    /// Each node's four neighbours (itself, on the border).
    n: Vec<[P; 4]>,
    /// The external forces, summed source by source (through `tmp`).
    f: Vec<P>,
    tmp: Vec<P>,
    out: Vec<(P, P)>,
}

impl CpuGrid {
    /// A lattice at rest.
    pub fn new(spec: GridSpec) -> CpuGrid {
        let rest: Vec<P> = (0..spec.nodes()).map(|k| spec.rest(k)).collect();
        let n = rest.len();
        let zero = Point::zero();
        CpuGrid {
            spec,
            p: rest.clone(),
            v: vec![zero; n],
            rest,
            n: vec![[zero; 4]; n],
            f: vec![zero; n],
            tmp: vec![zero; n],
            out: vec![(zero, zero); n],
        }
    }

    /// Set the state from GPU nodes.
    pub fn set(&mut self, nodes: &[Node]) {
        for ((p, v), node) in self.p.iter_mut().zip(&mut self.v).zip(nodes) {
            (*p, *v) = (node.p.into(), node.v.into());
        }
    }

    /// One step of `dt` with `sources`.
    pub fn step(&mut self, sources: &[Source], dt: f32) {
        // External forces, source by source in the GPU's order.
        self.f.fill(Point::zero());
        for s in sources {
            crate::add_source_force_batch(&self.f, &self.p, &[s.at], &[s.k()], &mut self.tmp);
            core::mem::swap(&mut self.f, &mut self.tmp);
        }
        let (w, p, spec) = (self.spec.cols as usize, &self.p, &self.spec);
        for (k, n) in self.n.iter_mut().enumerate() {
            *n = if spec.is_border(k) {
                [p[k]; 4]
            } else {
                [p[k - 1], p[k + 1], p[k - w], p[k + w]]
            };
        }
        let k = [SPRINGS.k(dt)];
        crate::grid_node_batch(
            &self.p,
            &self.v,
            &self.rest,
            &self.n,
            &self.f,
            &k,
            &mut self.out,
        );
        for (k, &next) in self.out.iter().enumerate() {
            // The border is pinned.
            (self.p[k], self.v[k]) = if self.spec.is_border(k) {
                (self.rest[k], Point::zero())
            } else {
                next
            };
        }
    }
}
