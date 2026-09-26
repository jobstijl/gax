//! The renderer: instanced shapes placed by PGA2D motors, and particles stepped by the traced
//! kernel in a compute pass. Plain wgpu; the shaders are linked at build time (`build.rs`).
//!
//! It is structured to grow: one `Gfx` owns the device-level pieces (pipelines, the particle
//! pool), and `draw` records a frame into any colour target (a window or an offscreen texture).

use bytemuck::{Pod, Zeroable};
use gax::pga2d::{MotorGpu, PointGpu};
use wgpu::util::DeviceExt;

/// The WGSL shaders, linked by `build.rs` against `gax::wgsl`'s modules.
pub mod wgsl {
    /// Instanced shapes placed by motors.
    pub const SHAPES: &str = include_str!(concat!(env!("OUT_DIR"), "/shapes.wgsl"));
    /// Instanced shapes placed by matrices (for the benchmark).
    pub const SHAPES_MATRIX: &str = include_str!(concat!(env!("OUT_DIR"), "/shapes_matrix.wgsl"));
    /// The particle step (the traced kernel).
    pub const PARTICLES_STEP: &str = include_str!(concat!(env!("OUT_DIR"), "/particles_step.wgsl"));
    /// Particles as quads.
    pub const PARTICLES_DRAW: &str = include_str!(concat!(env!("OUT_DIR"), "/particles_draw.wgsl"));
}

/// Particles in the pool.
pub const MAX_PARTICLES: usize = 1 << 20;

/// A shape instance: a unit motor (the WGSL `Motor`, one `vec4`) and a colour.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Instance {
    /// Where the shape is.
    pub motor: MotorGpu,
    /// Its colour.
    pub color: [f32; 4],
}

/// A shape instance for the matrix path: `m >> Point::slot()` as a WGSL `mat3x3<f32>`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MatrixInstance {
    /// The motor's matrix on points.
    pub matrix: gax::GpuMat<3>,
    /// Its colour.
    pub color: [f32; 4],
}

/// A particle, as the WGSL `Particle`: a motor, a constant rate (a PGA2D bivector, the kind
/// `Point`), and `[age, lifetime, hue, 0]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Particle {
    /// Where it is.
    pub motor: MotorGpu,
    /// How it moves: `motor ← motor exp(dt rate)` per step.
    pub rate: PointGpu,
    /// `[age, lifetime, hue, 0]`.
    pub life: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StepParams {
    dt: f32,
    count: u32,
    pad: [u32; 2],
}

/// Device-level state.
pub struct Gfx {
    /// The device.
    pub device: wgpu::Device,
    /// Its queue.
    pub queue: wgpu::Queue,
    /// The colour format frames are drawn in.
    pub format: wgpu::TextureFormat,
    view: wgpu::Buffer,
    corners: wgpu::Buffer,
    shapes: wgpu::RenderPipeline,
    shapes_matrix: wgpu::RenderPipeline,
    view_group: wgpu::BindGroup,
    view_group_matrix: wgpu::BindGroup,
    /// The particle pool (`MAX_PARTICLES`), stepped and drawn in place.
    pub particles: wgpu::Buffer,
    step_params: wgpu::Buffer,
    step: wgpu::ComputePipeline,
    step_group: wgpu::BindGroup,
    draw_particles: wgpu::RenderPipeline,
    particle_group: wgpu::BindGroup,
    instances: Option<(wgpu::Buffer, usize)>,
}

/// A triangle pointing along the local x axis: a ship.
const SHIP: [[f32; 2]; 3] = [[0.35, 0.0], [-0.25, 0.2], [-0.25, -0.2]];

impl Gfx {
    /// Build the pipelines for colour format `format`.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, format: wgpu::TextureFormat) -> Gfx {
        let d = &device;
        let module = |label, src: &str| {
            d.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let view = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let corners = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ship corners"),
            contents: bytemuck::cast_slice(&SHIP),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let target = [Some(wgpu::ColorTargetState {
            format,
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::OVER,
            }),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let corner_layout = wgpu::VertexBufferLayout {
            array_stride: 8,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x2],
        };
        let pipeline = |label,
                        m: &wgpu::ShaderModule,
                        buffers: &[Option<wgpu::VertexBufferLayout>],
                        topology| {
            d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: None,
                vertex: wgpu::VertexState {
                    module: m,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers,
                },
                primitive: wgpu::PrimitiveState {
                    topology,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: m,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &target,
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let shapes_module = module("shapes", wgsl::SHAPES);
        let shapes = pipeline(
            "shapes",
            &shapes_module,
            &[
                Some(corner_layout.clone()),
                Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<Instance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    // The motor is one vec4 (one location), the colour another.
                    attributes: &wgpu::vertex_attr_array![1 => Float32x4, 2 => Float32x4],
                }),
            ],
            wgpu::PrimitiveTopology::TriangleList,
        );
        let matrix_module = module("shapes_matrix", wgsl::SHAPES_MATRIX);
        let shapes_matrix = pipeline(
            "shapes_matrix",
            &matrix_module,
            &[
                Some(corner_layout),
                Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<MatrixInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![1 => Float32x4, 2 => Float32x4, 3 => Float32x4, 4 => Float32x4],
                }),
            ],
            wgpu::PrimitiveTopology::TriangleList,
        );
        let draw_module = module("particles_draw", wgsl::PARTICLES_DRAW);
        let draw_particles = pipeline(
            "particles_draw",
            &draw_module,
            &[],
            wgpu::PrimitiveTopology::TriangleStrip,
        );
        let step_module = module("particles_step", wgsl::PARTICLES_STEP);
        let step = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("particles_step"),
            layout: None,
            module: &step_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let particles = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles"),
            size: (MAX_PARTICLES * size_of::<Particle>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let step_params = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("step"),
            size: size_of::<StepParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = |layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindGroupEntry]| {
            d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries,
            })
        };
        let view_entry = wgpu::BindGroupEntry {
            binding: 0,
            resource: view.as_entire_binding(),
        };
        let view_group = group(
            &shapes.get_bind_group_layout(0),
            std::slice::from_ref(&view_entry),
        );
        let view_group_matrix = group(
            &shapes_matrix.get_bind_group_layout(0),
            std::slice::from_ref(&view_entry),
        );
        let step_group = group(
            &step.get_bind_group_layout(0),
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: step_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particles.as_entire_binding(),
                },
            ],
        );
        let particle_group = group(
            &draw_particles.get_bind_group_layout(0),
            &[
                view_entry.clone(),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particles.as_entire_binding(),
                },
            ],
        );
        Gfx {
            device,
            queue,
            format,
            view,
            corners,
            shapes,
            shapes_matrix,
            view_group,
            view_group_matrix,
            particles,
            step_params,
            step,
            step_group,
            draw_particles,
            particle_group,
            instances: None,
        }
    }

    /// Set the view: world units per half-width of the target, for a target of `aspect`
    /// (width / height).
    pub fn set_view(&self, half_width: f32, aspect: f32) {
        let s = [1.0 / half_width, aspect / half_width, 0.0, 0.0];
        self.queue
            .write_buffer(&self.view, 0, bytemuck::bytes_of(&s));
    }

    /// Upload shape instances (growing the buffer as needed).
    pub fn set_instances<T: Pod>(&mut self, data: &[T]) {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let fits = self
            .instances
            .as_ref()
            .is_some_and(|(b, _)| b.size() >= bytes.len() as u64);
        if !fits {
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instances"),
                size: (bytes.len() as u64).max(64).next_power_of_two(),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instances = Some((buf, 0));
        }
        let (buf, n) = self.instances.as_mut().expect("instances");
        self.queue.write_buffer(buf, 0, bytes);
        *n = data.len();
    }

    /// Write particles into the pool at index `at`.
    pub fn write_particles(&self, at: usize, ps: &[Particle]) {
        self.queue.write_buffer(
            &self.particles,
            (at * size_of::<Particle>()) as u64,
            bytemuck::cast_slice(ps),
        );
    }

    /// Record one particle step of `dt` for the first `count` particles.
    pub fn step_particles(&self, enc: &mut wgpu::CommandEncoder, dt: f32, count: usize) {
        let p = StepParams {
            dt,
            count: count as u32,
            pad: [0; 2],
        };
        self.queue
            .write_buffer(&self.step_params, 0, bytemuck::bytes_of(&p));
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.step);
        pass.set_bind_group(0, &self.step_group, &[]);
        pass.dispatch_workgroups((count as u32).div_ceil(256), 1, 1);
    }

    /// Record a frame into `target`: the shape instances (as motors, or as matrices with
    /// `matrix`), then `particles` particles.
    pub fn draw(
        &self,
        enc: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        matrix: bool,
        particles: usize,
    ) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("frame"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.01,
                        g: 0.01,
                        b: 0.02,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        if let Some((buf, n)) = &self.instances {
            let (pipeline, group) = if matrix {
                (&self.shapes_matrix, &self.view_group_matrix)
            } else {
                (&self.shapes, &self.view_group)
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.set_vertex_buffer(0, self.corners.slice(..));
            pass.set_vertex_buffer(1, buf.slice(..));
            pass.draw(0..3, 0..*n as u32);
        }
        if particles > 0 {
            pass.set_pipeline(&self.draw_particles);
            pass.set_bind_group(0, &self.particle_group, &[]);
            pass.draw(0..4, 0..particles as u32);
        }
    }

    /// Read `n` particles starting at `at` back from the pool (blocking; for checks).
    pub fn read_particles(&self, at: usize, n: usize) -> Vec<Particle> {
        let bytes = (n * size_of::<Particle>()) as u64;
        let read = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(
            &self.particles,
            (at * size_of::<Particle>()) as u64,
            &read,
            0,
            bytes,
        );
        self.queue.submit([enc.finish()]);
        let slice = read.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map the particles"));
        self.wait();
        let out = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
        read.unmap();
        out
    }

    /// Block until the GPU is idle.
    pub fn wait(&self) {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the GPU finishes");
    }
}

/// A device on the default adapter (for a window's surface when `surface` is given).
pub fn device(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface>,
) -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: surface,
        ..Default::default()
    }))
    .expect("a GPU adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("gax-wgpu-example"),
        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
        ..Default::default()
    }))
    .expect("a device");
    (adapter, device, queue)
}
