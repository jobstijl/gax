//! The Plane's lattice of warped space: its shape ([`GridSpec`]), its springs
//! ([`LatticeParams`]), the sources that pull it ([`Source`], a frame's [`GridSteps`]), and
//! the lattice on the GPU ([`GpuLattice`]), stepped in a compute shader by the traced
//! `grid_node` kernel and drawn as glowing lines. The tests' CPU twin (`cpu_grid.rs`) runs the
//! same kernels on the same definitions.

use super::factory::{Factory, Target};
use super::gpu_types::{GridLook, GridUniform, Node, SourceGpu, source_array};
use crate::light::Light;
use gax::pga2d::Point;

/// Sources (wells and blasts) per lattice step.
pub const MAX_SOURCES: usize = 64;
/// Lattice steps per frame (one per simulation tick); the ticks of a frame past these are
/// skipped by the lattice and the particles, which fall behind a fast-forwarded replay.
pub const MAX_GRID_STEPS: usize = 8;

/// The lattice's springs. Every node is pulled towards each neighbour by `spring` times their
/// offset (springs of rest length zero, `kernels::grid_node`) and towards its rest position by
/// `anchor` times its offset; `damping` takes that fraction of its velocity per second.
#[derive(Clone, Copy, Debug)]
pub struct LatticeParams {
    /// The springs' stiffness.
    pub spring: f32,
    /// The anchor's stiffness.
    pub anchor: f32,
    /// Velocity damping per second.
    pub damping: f32,
}

impl LatticeParams {
    /// `grid_node`'s parameters for a step of `dt`.
    pub fn k(self, dt: f32) -> [f32; 4] {
        [self.spring, self.anchor, self.damping, dt]
    }
}

/// The Plane's springs, on the GPU and in the CPU twin.
pub const SPRINGS: LatticeParams = LatticeParams {
    spring: 95.0,
    anchor: 5.0,
    damping: 4.5,
};

/// The lattice's shape: `cols` by `rows` nodes, `spacing` apart, node `k` in column
/// `k % cols` and row `k / cols`. The nodes on the border are pinned.
#[derive(Clone, Copy, Debug)]
pub struct GridSpec {
    /// Nodes per row.
    pub cols: u32,
    /// Rows.
    pub rows: u32,
    /// The lower-left node's rest position.
    pub origin: Point<(), f32>,
    /// Distance between nodes.
    pub spacing: f32,
}

impl GridSpec {
    /// How many nodes there are.
    pub fn nodes(self) -> usize {
        (self.cols * self.rows) as usize
    }

    /// How many edges join neighbours: along the rows, then along the columns.
    pub fn edges(self) -> u32 {
        (self.cols - 1) * self.rows + self.cols * (self.rows - 1)
    }

    /// The column and row of node `k`.
    fn cell(self, k: usize) -> (u32, u32) {
        let k = k as u32;
        (k % self.cols, k / self.cols)
    }

    /// Where node `k` rests (the traced `lattice_rest`, as on the GPU).
    pub fn rest(self, k: usize) -> Point<(), f32> {
        let (i, j) = self.cell(k);
        crate::lattice_rest(self.origin, [i as f32, j as f32, self.spacing])
    }

    /// Whether node `k` is on the border (pinned).
    pub fn is_border(self, k: usize) -> bool {
        let (i, j) = self.cell(k);
        i == 0 || j == 0 || i == self.cols - 1 || j == self.rows - 1
    }

    /// Every node at rest.
    fn at_rest(self) -> Vec<Node> {
        (0..self.nodes())
            .map(|k| Node {
                p: self.rest(k).into(),
                v: Point::direction(0.0, 0.0).into(),
            })
            .collect()
    }
}

/// A point that pulls the lattice or the particles in (a positive pull: a well) or pushes them
/// away (a negative one: a blast): the parameters of the traced `source_force` kernel. The
/// Tunnel's lattice has its own sources, which push for a positive `push`.
#[derive(Clone, Copy, Debug)]
pub struct Source {
    /// Where it is.
    pub at: Point<(), f32>,
    /// How hard it pulls (negative: how hard it pushes).
    pub pull: f32,
    /// The square of its radius, below which the force fades.
    pub radius2: f32,
}

impl Source {
    /// A source at `at`.
    pub fn new(at: Point<(), f32>, pull: f32, radius2: f32) -> Source {
        Source { at, pull, radius2 }
    }

    /// `source_force`'s parameters, `[pull, radius²]`.
    pub fn k(self) -> [f32; 2] {
        [self.pull, self.radius2]
    }

    /// A slot no source fills: no force (and none of the 0/0 a zero radius gives at its point).
    pub(super) fn none() -> Source {
        Source::new(Point::xy(0.0, 0.0), 0.0, 1.0)
    }
}

/// A frame's lattice steps, one per simulation tick, each with its sources: one flat list of
/// sources, and where each step's end.
#[derive(Debug, Default)]
pub struct GridSteps {
    sources: Vec<Source>,
    ends: Vec<usize>,
}

impl GridSteps {
    /// No steps.
    pub fn clear(&mut self) {
        self.sources.clear();
        self.ends.clear();
    }

    /// Add a step with `sources`, of which it keeps the first [`MAX_SOURCES`] (so the ones that
    /// matter most come first). A frame that has [`MAX_GRID_STEPS`] already skips it.
    pub fn push(&mut self, sources: impl IntoIterator<Item = Source>) {
        if self.ends.len() < MAX_GRID_STEPS {
            self.sources.extend(sources.into_iter().take(MAX_SOURCES));
            self.ends.push(self.sources.len());
        }
    }

    /// Each step's sources.
    pub fn iter(&self) -> impl Iterator<Item = &[Source]> {
        let starts = core::iter::once(0).chain(self.ends.iter().copied());
        starts.zip(&self.ends).map(|(a, &b)| &self.sources[a..b])
    }
}

/// The lattice on the GPU: nodes ping-ponged between two buffers, one compute dispatch per
/// step (each step with its own sources buffer), and a draw of every edge as a line.
pub(super) struct GpuLattice {
    spec: GridSpec,
    nodes: [wgpu::Buffer; 2],
    uniform: wgpu::Buffer,
    sources: Vec<wgpu::Buffer>,
    step: wgpu::ComputePipeline,
    /// Per step, per parity: the step reading `nodes[parity]` into the other.
    step_groups: Vec<[wgpu::BindGroup; 2]>,
    look: wgpu::Buffer,
    draw: wgpu::RenderPipeline,
    draw_groups: [wgpu::BindGroup; 2],
    /// The buffer holding the current state.
    parity: usize,
    /// Steps uploaded for the next [`GpuLattice::encode`].
    steps: usize,
}

impl GpuLattice {
    /// A lattice of shape `spec` at rest, drawn through the camera uniform `camera`.
    pub fn new(f: Factory, spec: GridSpec, camera: &wgpu::Buffer) -> GpuLattice {
        let rest = spec.at_rest();
        let nodes = [0, 1].map(|_| f.storage("grid nodes", bytemuck::cast_slice(&rest)));
        let uniform = f.uniform("grid", size_of::<GridUniform>());
        let step = f.compute("grid_step", &f.module("grid_step", super::wgsl::GRID_STEP));
        let sources: Vec<wgpu::Buffer> = (0..MAX_GRID_STEPS)
            .map(|_| f.uniform("grid sources", size_of::<SourceGpu>() * MAX_SOURCES))
            .collect();
        let layout = step.get_bind_group_layout(0);
        let step_groups = sources
            .iter()
            .map(|s| {
                [0, 1].map(|from| f.group(&layout, &[&uniform, s, &nodes[from], &nodes[1 - from]]))
            })
            .collect();
        let look = f.uniform("grid look", size_of::<GridLook>());
        let module = f.module("grid_draw", super::wgsl::GRID_DRAW);
        let draw = f.quads("grid_draw", &module, "fs_main", Target::Hdr, None);
        let layout = draw.get_bind_group_layout(0);
        let draw_groups = [0, 1].map(|k| f.group(&layout, &[camera, &look, &nodes[k]]));
        GpuLattice {
            spec,
            nodes,
            uniform,
            sources,
            step,
            step_groups,
            look,
            draw,
            draw_groups,
            parity: 0,
            steps: 0,
        }
    }

    /// Write a frame's steps (each step `dt` with its sources) and the lattice's light.
    pub fn upload(&mut self, q: &wgpu::Queue, steps: &GridSteps, light: Light, dt: f32) {
        let g = self.spec;
        for (buf, sources) in self.sources.iter().zip(steps.iter()) {
            let gpu = source_array::<MAX_SOURCES>(sources);
            q.write_buffer(buf, 0, bytemuck::cast_slice(&gpu));
        }
        self.steps = steps.ends.len();
        // Every step loops over the most sources any step has; unused slots pull with no force.
        let most = steps.iter().map(<[Source]>::len).max().unwrap_or(0);
        let uniform = GridUniform {
            dims: [g.cols, g.rows, most as u32, 0],
            k: SPRINGS.k(dt),
            origin: g.origin.into(),
            spacing: [g.spacing, 0.0, 0.0, 0.0],
        };
        q.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        let look = GridLook {
            dims: [g.cols, g.rows, 0, 0],
            color: light.into(),
            style: [0.014, 0.1, 0.12, g.spacing],
        };
        q.write_buffer(&self.look, 0, bytemuck::bytes_of(&look));
    }

    /// Run the uploaded steps, in order (the state ends in the other buffer after an odd
    /// number), and return how many.
    pub fn encode(&mut self, pass: &mut wgpu::ComputePass) -> usize {
        let steps = core::mem::take(&mut self.steps);
        pass.set_pipeline(&self.step);
        for groups in &self.step_groups[..steps] {
            pass.set_bind_group(0, &groups[self.parity], &[]);
            pass.dispatch_workgroups((self.spec.nodes() as u32).div_ceil(64), 1, 1);
            self.parity = 1 - self.parity;
        }
        steps
    }

    /// Draw every edge of the current state.
    pub fn draw(&self, pass: &mut wgpu::RenderPass) {
        pass.set_pipeline(&self.draw);
        pass.set_bind_group(0, &self.draw_groups[self.parity], &[]);
        pass.draw(0..4, 0..self.spec.edges());
    }

    /// Put every node back at rest.
    pub fn reset(&self, q: &wgpu::Queue) {
        q.write_buffer(
            &self.nodes[self.parity],
            0,
            bytemuck::cast_slice(&self.spec.at_rest()),
        );
    }

    /// The buffer holding the current state (for the tests).
    #[cfg(test)]
    pub fn current(&self) -> &wgpu::Buffer {
        &self.nodes[self.parity]
    }
}
