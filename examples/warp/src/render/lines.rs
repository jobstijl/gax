//! The line renderer: segments of shapes placed by unit motors, drawn as quads with a glowing
//! core (`shaders/lines.wesl`), into the HDR scene for the world and straight onto the output
//! for the HUD (never bloomed, tonemapped or fringed with the scene).

use super::factory::{Factory, Target};
use super::gpu_types::{LINE_ATTRIBUTES, LineInstance};

/// The line pipelines and their instances.
pub(super) struct GpuLines {
    world: wgpu::RenderPipeline,
    overlay: wgpu::RenderPipeline,
    world_group: wgpu::BindGroup,
    hud_group: wgpu::BindGroup,
    /// The world's instances, then the HUD's.
    instances: wgpu::Buffer,
    counts: [u32; 2],
}

impl GpuLines {
    /// The world's lines seen through the camera uniform `camera`, and the HUD's through
    /// `hud`, onto an output of `format`.
    pub fn new(
        f: Factory,
        camera: &wgpu::Buffer,
        hud: &wgpu::Buffer,
        format: wgpu::TextureFormat,
    ) -> GpuLines {
        let module = f.module("lines", super::wgsl::LINES);
        let instances = || {
            Some(wgpu::VertexBufferLayout {
                array_stride: size_of::<LineInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &LINE_ATTRIBUTES,
            })
        };
        let world = f.quads("lines", &module, "fs_main", Target::Hdr, instances());
        let overlay = f.quads(
            "lines overlay",
            &module,
            "fs_overlay",
            Target::Display(format),
            instances(),
        );
        GpuLines {
            world_group: f.group(&world.get_bind_group_layout(0), &[camera]),
            hud_group: f.group(&overlay.get_bind_group_layout(0), &[hud]),
            world,
            overlay,
            instances: instance_buffer(f, 1 << 20),
            counts: [0, 0],
        }
    }

    /// Write the world's and the HUD's segments, growing the buffer when they do not fit.
    pub fn upload(
        &mut self,
        f: Factory,
        q: &wgpu::Queue,
        world: &[LineInstance],
        hud: &[LineInstance],
    ) {
        let bytes = (size_of_val(world) + size_of_val(hud)) as u64;
        if bytes > self.instances.size() {
            self.instances = instance_buffer(f, bytes.next_power_of_two());
        }
        if !world.is_empty() {
            q.write_buffer(&self.instances, 0, bytemuck::cast_slice(world));
        }
        if !hud.is_empty() {
            q.write_buffer(
                &self.instances,
                size_of_val(world) as u64,
                bytemuck::cast_slice(hud),
            );
        }
        self.counts = [world.len() as u32, hud.len() as u32];
    }

    /// Draw the world's segments into the scene.
    pub fn draw_world(&self, pass: &mut wgpu::RenderPass) {
        let [world, _] = self.counts;
        if world > 0 {
            pass.set_pipeline(&self.world);
            pass.set_vertex_buffer(0, self.instances.slice(..));
            pass.set_bind_group(0, &self.world_group, &[]);
            pass.draw(0..4, 0..world);
        }
    }

    /// Whether there is a HUD to draw.
    pub fn has_hud(&self) -> bool {
        self.counts[1] > 0
    }

    /// Draw the HUD's segments onto the output.
    pub fn draw_hud(&self, pass: &mut wgpu::RenderPass) {
        let [world, hud] = self.counts;
        pass.set_pipeline(&self.overlay);
        pass.set_vertex_buffer(0, self.instances.slice(..));
        pass.set_bind_group(0, &self.hud_group, &[]);
        pass.draw(0..4, world..world + hud);
    }
}

/// A buffer of `size` bytes of line instances.
fn instance_buffer(f: Factory, size: u64) -> wgpu::Buffer {
    f.0.create_buffer(&wgpu::BufferDescriptor {
        label: Some("line instances"),
        size,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
