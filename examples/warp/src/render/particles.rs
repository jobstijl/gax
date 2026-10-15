//! The particles: a ring of [`MAX_PARTICLES`] on the GPU, stepped in a compute shader by the
//! traced `particle_step` (with the wells' pull, `source_force`) and drawn as streaks. New
//! particles overwrite the oldest. The CPU knows each spawned batch's lifetime, so only the
//! live window of the ring is stepped and drawn; a dead particle inside it draws nothing.

use super::factory::{Factory, Target};
use super::gpu_types::{Particle, SourceGpu, StepUniform, source_array};
use super::lattice::Source;
use std::collections::VecDeque;
use std::ops::Range;

/// Particles in the ring.
pub const MAX_PARTICLES: usize = 1 << 18;
/// Wells that pull particles.
pub const MAX_WELLS: usize = 16;

/// Particles written into the ring together.
struct Batch {
    /// Where it ends, counting every particle ever spawned.
    end: u64,
    /// The step by which all of it is dead.
    dead_at: u64,
}

/// The particles on the GPU.
pub(super) struct GpuParticles {
    buffer: wgpu::Buffer,
    uniform: wgpu::Buffer,
    wells: wgpu::Buffer,
    step: wgpu::ComputePipeline,
    step_group: wgpu::BindGroup,
    draw: wgpu::RenderPipeline,
    draw_group: wgpu::BindGroup,
    /// Particles ever spawned: the next one goes to `spawned % MAX_PARTICLES`.
    spawned: u64,
    /// Where the oldest batch that may be alive begins (counting as `spawned`).
    first: u64,
    /// The batches that may be alive, oldest first.
    batches: VecDeque<Batch>,
    /// Steps run.
    steps: u64,
}

impl GpuParticles {
    /// An empty ring, drawn through the camera uniform `camera`.
    pub fn new(f: Factory, camera: &wgpu::Buffer) -> GpuParticles {
        let buffer = f.empty_storage("particles", MAX_PARTICLES * size_of::<Particle>());
        let uniform = f.uniform("particle step", size_of::<StepUniform>());
        let wells = f.uniform("particle wells", size_of::<SourceGpu>() * MAX_WELLS);
        let module = f.module("particles_step", super::wgsl::PARTICLES_STEP);
        let step = f.compute("particles_step", &module);
        let step_group = f.group(&step.get_bind_group_layout(0), &[&uniform, &wells, &buffer]);
        let module = f.module("particles_draw", super::wgsl::PARTICLES_DRAW);
        let draw = f.quads("particles_draw", &module, "fs_main", Target::Hdr, None);
        let draw_group = f.group(&draw.get_bind_group_layout(0), &[camera, &buffer]);
        GpuParticles {
            buffer,
            uniform,
            wells,
            step,
            step_group,
            draw,
            draw_group,
            spawned: 0,
            first: 0,
            batches: VecDeque::new(),
            steps: 0,
        }
    }

    /// Write `ps` into the ring after the newest (the last [`MAX_PARTICLES`] of them, in one or
    /// two writes as the ring wraps); they are all dead within their lifetimes of steps of `dt`.
    pub fn spawn(&mut self, q: &wgpu::Queue, ps: &[Particle], dt: f32) {
        let ps = &ps[ps.len().saturating_sub(MAX_PARTICLES)..];
        if ps.is_empty() {
            return;
        }
        let head = (self.spawned % MAX_PARTICLES as u64) as usize;
        let (before_end, after_wrap) = ps.split_at((MAX_PARTICLES - head).min(ps.len()));
        let at = |i: usize| (i * size_of::<Particle>()) as u64;
        q.write_buffer(&self.buffer, at(head), bytemuck::cast_slice(before_end));
        if !after_wrap.is_empty() {
            q.write_buffer(&self.buffer, 0, bytemuck::cast_slice(after_wrap));
        }
        // The last to die; a step more, for the GPU's rounding of the ages.
        let life = ps.iter().map(|p| p.life[1] - p.life[0]).fold(0.0, f32::max);
        self.spawned += ps.len() as u64;
        self.batches.push_back(Batch {
            end: self.spawned,
            dead_at: self.steps + (life / dt).ceil() as u64 + 1,
        });
    }

    /// The particles that may be alive, counting as `spawned` (index `i` is in the ring at
    /// `i % MAX_PARTICLES`).
    pub fn live(&self) -> Range<u64> {
        let overwritten = self.spawned.saturating_sub(MAX_PARTICLES as u64);
        self.first.max(overwritten)..self.spawned
    }

    /// Write the wells' pull and the step's parameters (a step of `dt`) over the live window.
    pub fn upload(&self, q: &wgpu::Queue, wells: &[Source], dt: f32) {
        q.write_buffer(
            &self.wells,
            0,
            bytemuck::cast_slice(&source_array::<MAX_WELLS>(wells)),
        );
        let live = self.live();
        let first = (live.start % MAX_PARTICLES as u64) as u32;
        let uniform = StepUniform {
            dims: [
                (live.end - live.start) as u32,
                wells.len().min(MAX_WELLS) as u32,
                first,
                0,
            ],
            k: [dt, 1.0, 0.0, 0.0],
        };
        q.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
    }

    /// `steps` steps of the live window; then the batches dead by now leave it.
    pub fn encode(&mut self, pass: &mut wgpu::ComputePass, steps: usize) {
        let live = self.live();
        if !live.is_empty() {
            pass.set_pipeline(&self.step);
            pass.set_bind_group(0, &self.step_group, &[]);
            for _ in 0..steps {
                pass.dispatch_workgroups(((live.end - live.start) as u32).div_ceil(256), 1, 1);
            }
        }
        self.steps += steps as u64;
        let overwritten = self.spawned.saturating_sub(MAX_PARTICLES as u64);
        while let Some(b) = self.batches.front()
            && (b.dead_at <= self.steps || b.end <= overwritten)
        {
            self.first = b.end;
            self.batches.pop_front();
        }
    }

    /// Draw the live window.
    pub fn draw(&self, pass: &mut wgpu::RenderPass) {
        let live = self.live();
        if live.is_empty() {
            return;
        }
        let first = (live.start % MAX_PARTICLES as u64) as u32;
        pass.set_pipeline(&self.draw);
        pass.set_bind_group(0, &self.draw_group, &[]);
        pass.draw(0..4, first..first + (live.end - live.start) as u32);
    }

    /// No particles: every one spawned so far is forgotten.
    pub fn reset(&mut self) {
        self.batches.clear();
        self.first = self.spawned;
    }

    /// The ring (for the tests).
    #[cfg(test)]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}
