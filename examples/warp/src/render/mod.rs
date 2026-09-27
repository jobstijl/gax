//! The game's own renderer, on plain wgpu (no Bevy rendering).
//!
//! One HDR target (`rgba16float`) takes, additively: a parallax starfield, the warped-space
//! lattice (stepped in a compute shader by a traced gax kernel), particles (likewise), the
//! shapes (line segments placed by unit motors in the vertex shader) and the HUD. Bloom is a
//! mip chain; the composite tonemaps with AgX into the window's sRGB surface. GPU buffers
//! hold gax's Pod types (`MotorGpu`, `PointGpu`, `GpuMat`), whose layouts gax checks at
//! compile time.

pub mod cpu_grid;
pub mod font;
pub mod scene;
pub mod tunnel;

use bytemuck::{Pod, Zeroable};
use gax::pga2d::{MotorGpu, PointGpu};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use wgpu::util::DeviceExt;

/// The WGSL shaders, linked at build time against `gax::wgsl`'s modules.
mod wgsl {
    pub const LINES: &str = include_str!(concat!(env!("OUT_DIR"), "/lines.wgsl"));
    pub const GRID_STEP: &str = include_str!(concat!(env!("OUT_DIR"), "/grid_step.wgsl"));
    pub const GRID_DRAW: &str = include_str!(concat!(env!("OUT_DIR"), "/grid_draw.wgsl"));
    pub const PARTICLES_STEP: &str = include_str!(concat!(env!("OUT_DIR"), "/particles_step.wgsl"));
    pub const PARTICLES_DRAW: &str = include_str!(concat!(env!("OUT_DIR"), "/particles_draw.wgsl"));
    pub const POST: &str = include_str!(concat!(env!("OUT_DIR"), "/post.wgsl"));
    pub const STARS: &str = include_str!(concat!(env!("OUT_DIR"), "/stars.wgsl"));
}

/// The HDR scene format.
pub const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Particles in the pool.
pub const MAX_PARTICLES: usize = 1 << 18;
/// Grid sources (blasts and wells) per step.
pub const MAX_SOURCES: usize = 64;
/// Wells that pull particles.
pub const MAX_WELLS: usize = 16;
const BLOOM_MIPS: usize = 6;

/// A line segment: endpoints in the local frame of `motor` (`[ax, ay, bx, by]`), an HDR colour
/// (rgb, intensity), a style (half width, glow radius, glow strength, 0) in world units.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LineInstance {
    /// Endpoints.
    pub ab: [f32; 4],
    /// Colour and intensity.
    pub color: [f32; 4],
    /// Half width, glow radius, glow strength, unused.
    pub style: [f32; 4],
    /// Where the segment's frame is: a unit PGA2D motor, the WGSL `Motor`.
    pub motor: MotorGpu,
}

/// The camera uniform: the WGSL `Camera`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    /// World `(x, y, 1)` to clip space: gax's map `proj.of(view)` as a WGSL `mat3x3`.
    pub view_proj: gax::GpuMat<3>,
    /// Width, height (pixels), world units per pixel, time.
    pub viewport: [f32; 4],
}

/// A lattice node: the WGSL `Node`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Node {
    /// Position (a finite PGA2D point).
    pub p: PointGpu,
    /// Velocity (a direction).
    pub v: PointGpu,
}

/// A particle: the WGSL `Particle`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Particle {
    /// Position.
    pub p: PointGpu,
    /// Velocity.
    pub v: PointGpu,
    /// Colour and intensity.
    pub color: [f32; 4],
    /// Age, lifetime, drag, streak length (seconds of velocity).
    pub life: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GridUniform {
    dims: [u32; 4],
    k: [f32; 4],
    frame: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GridLook {
    dims: [u32; 4],
    color: [f32; 4],
    style: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StepUniform {
    dims: [u32; 4],
    k: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct PostUniform {
    texel: [f32; 4],
    k: [f32; 4],
    look: [f32; 4],
    shock: [f32; 4],
}

/// The lattice's shape.
#[derive(Clone, Copy, Debug)]
pub struct GridSpec {
    /// Nodes per row and column.
    pub cols: u32,
    /// Rows.
    pub rows: u32,
    /// Lower-left corner.
    pub origin: [f32; 2],
    /// Distance between nodes.
    pub spacing: f32,
}

/// Post-processing settings.
#[derive(Clone, Copy, Debug)]
pub struct PostSettings {
    /// Bloom strength.
    pub bloom: f32,
    /// Exposure.
    pub exposure: f32,
    /// Vignette.
    pub vignette: f32,
    /// Film grain.
    pub grain: f32,
    /// Chromatic aberration at the edges.
    pub aberration: f32,
    /// Saturation of the tonemapper's look.
    pub saturation: f32,
    /// A shock ripple: `(uv centre, radius, strength)`.
    pub shock: Option<([f32; 2], f32, f32)>,
}

/// Everything a frame draws.
pub struct Frame<'a> {
    /// The world camera.
    pub camera: CameraUniform,
    /// The HUD camera.
    pub hud: CameraUniform,
    /// Camera centre in the world (for the starfield's parallax).
    pub centre: [f32; 2],
    /// World-space segments (shapes, bullets, border).
    pub world: &'a [LineInstance],
    /// HUD segments.
    pub hud_lines: &'a [LineInstance],
    /// Grid steps to run this frame (one per simulation tick), each with its sources
    /// (`[x, y, strength, radius²]`).
    pub grid_steps: &'a [Vec<[f32; 4]>],
    /// Wells pulling particles (`[x, y, strength, radius²]`).
    pub wells: &'a [[f32; 4]],
    /// Particles to add before stepping.
    pub spawn: &'a [Particle],
    /// Grid colour (rgb, intensity).
    pub grid_color: [f32; 4],
    /// Post-processing.
    pub post: PostSettings,
    /// The Plane's layers (stars, lattice, particles); the Tunnel draws everything as lines.
    pub plane: bool,
}

/// GPU time per pass, in milliseconds (when the adapter has timestamp queries).
#[derive(Clone, Copy, Debug, Default)]
pub struct Timings {
    /// Grid and particle compute.
    pub compute: f32,
    /// The HDR scene.
    pub scene: f32,
    /// Bloom.
    pub bloom: f32,
    /// Tonemapping and composite.
    pub composite: f32,
}

struct Timer {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    ready: Arc<AtomicBool>,
    pending: bool,
    period: f32,
    last: Timings,
}

struct Targets {
    hdr: wgpu::TextureView,
    mips: Vec<(wgpu::TextureView, [u32; 2])>,
    down: Vec<wgpu::BindGroup>,
    up: Vec<wgpu::BindGroup>,
    composite: wgpu::BindGroup,
    size: [u32; 2],
}

/// The renderer.
pub struct Renderer {
    /// The device.
    pub device: wgpu::Device,
    /// Its queue.
    pub queue: wgpu::Queue,
    /// The lattice.
    pub grid: GridSpec,
    sampler: wgpu::Sampler,
    cam_world: wgpu::Buffer,
    cam_hud: wgpu::Buffer,
    lines: wgpu::RenderPipeline,
    lines_world: wgpu::BindGroup,
    lines_hud: wgpu::BindGroup,
    instances: wgpu::Buffer,
    grid_nodes: [wgpu::Buffer; 2],
    grid_uniform: wgpu::Buffer,
    grid_sources: Vec<wgpu::Buffer>,
    grid_step: wgpu::ComputePipeline,
    grid_step_groups: Vec<[wgpu::BindGroup; 2]>,
    grid_look: wgpu::Buffer,
    grid_draw: wgpu::RenderPipeline,
    grid_draw_groups: [wgpu::BindGroup; 2],
    grid_parity: usize,
    particles: wgpu::Buffer,
    particle_step_uniform: wgpu::Buffer,
    particle_wells: wgpu::Buffer,
    particle_step: wgpu::ComputePipeline,
    particle_step_group: wgpu::BindGroup,
    particle_draw: wgpu::RenderPipeline,
    particle_draw_group: wgpu::BindGroup,
    particle_head: usize,
    particle_count: usize,
    stars: wgpu::RenderPipeline,
    stars_uniform: wgpu::Buffer,
    stars_group: wgpu::BindGroup,
    post_down: wgpu::RenderPipeline,
    post_up: wgpu::RenderPipeline,
    post_composite: wgpu::RenderPipeline,
    post_uniforms: Vec<wgpu::Buffer>,
    targets: Targets,
    timer: Option<Timer>,
    time: f32,
}

fn additive(format: wgpu::TextureFormat) -> [Option<wgpu::ColorTargetState>; 1] {
    [Some(wgpu::ColorTargetState {
        format,
        blend: Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        }),
        write_mask: wgpu::ColorWrites::ALL,
    })]
}

impl Renderer {
    /// The features the renderer uses when the adapter has them.
    pub fn wanted_features(adapter: &wgpu::Adapter) -> wgpu::Features {
        adapter.features() & wgpu::Features::TIMESTAMP_QUERY
    }

    /// Build the renderer for output `format` and `size`.
    #[allow(clippy::too_many_lines)]
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        format: wgpu::TextureFormat,
        size: [u32; 2],
        grid: GridSpec,
    ) -> Renderer {
        let d = &device;
        let module = |label: &str, src: &str| {
            d.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let uniform = |label: &str, size: usize| {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let group = |layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindGroupEntry]| {
            d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries,
            })
        };
        fn entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buf.as_entire_binding(),
            }
        }
        let render = |label: &str,
                      m: &wgpu::ShaderModule,
                      vs: &str,
                      fs: &str,
                      buffers: &[Option<wgpu::VertexBufferLayout>],
                      topology,
                      target: wgpu::TextureFormat,
                      blend: bool| {
            let targets = if blend {
                additive(target)
            } else {
                [Some(wgpu::ColorTargetState {
                    format: target,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })]
            };
            d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: None,
                vertex: wgpu::VertexState {
                    module: m,
                    entry_point: Some(vs),
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
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let compute = |label: &str, m: &wgpu::ShaderModule| {
            d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: m,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let strip = wgpu::PrimitiveTopology::TriangleStrip;

        // Lines.
        let cam_world = uniform("camera", size_of::<CameraUniform>());
        let cam_hud = uniform("hud camera", size_of::<CameraUniform>());
        let lines_module = module("lines", wgsl::LINES);
        let lines = render(
            "lines",
            &lines_module,
            "vs_main",
            "fs_main",
            &[Some(wgpu::VertexBufferLayout {
                array_stride: size_of::<LineInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4, 3 => Float32x4],
            })],
            strip,
            HDR,
            true,
        );
        let lines_world = group(&lines.get_bind_group_layout(0), &[entry(0, &cam_world)]);
        let lines_hud = group(&lines.get_bind_group_layout(0), &[entry(0, &cam_hud)]);
        let instances = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("line instances"),
            size: 1 << 20,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // The lattice, ping-ponged between two node buffers.
        let n = (grid.cols * grid.rows) as usize;
        let rest: Vec<Node> = (0..n)
            .map(|k| {
                let (i, j) = (k as u32 % grid.cols, k as u32 / grid.cols);
                let p = gax::pga2d::Point::xy(
                    grid.origin[0] + i as f32 * grid.spacing,
                    grid.origin[1] + j as f32 * grid.spacing,
                );
                Node {
                    p: p.into(),
                    v: gax::pga2d::Point::direction(0.0, 0.0).into(),
                }
            })
            .collect();
        let grid_nodes = [0, 1].map(|_| {
            d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("grid nodes"),
                contents: bytemuck::cast_slice(&rest),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
        });
        let grid_uniform = uniform("grid", size_of::<GridUniform>());
        let step_module = module("grid_step", wgsl::GRID_STEP);
        let grid_step = compute("grid_step", &step_module);
        // One sources buffer per step in a frame (up to 8 steps), so each dispatch sees its own.
        let grid_sources: Vec<wgpu::Buffer> = (0..8)
            .map(|_| uniform("grid sources", 16 * MAX_SOURCES))
            .collect();
        let grid_step_groups = grid_sources
            .iter()
            .map(|s| {
                [0, 1].map(|from| {
                    group(
                        &grid_step.get_bind_group_layout(0),
                        &[
                            entry(0, &grid_uniform),
                            entry(1, s),
                            entry(2, &grid_nodes[from]),
                            entry(3, &grid_nodes[1 - from]),
                        ],
                    )
                })
            })
            .collect();
        let grid_look = uniform("grid look", size_of::<GridLook>());
        let grid_draw_module = module("grid_draw", wgsl::GRID_DRAW);
        let grid_draw = render(
            "grid_draw",
            &grid_draw_module,
            "vs_main",
            "fs_main",
            &[],
            strip,
            HDR,
            true,
        );
        let grid_draw_groups = [0, 1].map(|k| {
            group(
                &grid_draw.get_bind_group_layout(0),
                &[
                    entry(0, &cam_world),
                    entry(1, &grid_look),
                    entry(2, &grid_nodes[k]),
                ],
            )
        });

        // Particles.
        let particles = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles"),
            size: (MAX_PARTICLES * size_of::<Particle>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let particle_step_uniform = uniform("particle step", size_of::<StepUniform>());
        let particle_wells = uniform("particle wells", 16 * MAX_WELLS);
        let pstep_module = module("particles_step", wgsl::PARTICLES_STEP);
        let particle_step = compute("particles_step", &pstep_module);
        let particle_step_group = group(
            &particle_step.get_bind_group_layout(0),
            &[
                entry(0, &particle_step_uniform),
                entry(1, &particle_wells),
                entry(2, &particles),
            ],
        );
        let pdraw_module = module("particles_draw", wgsl::PARTICLES_DRAW);
        let particle_draw = render(
            "particles_draw",
            &pdraw_module,
            "vs_main",
            "fs_main",
            &[],
            strip,
            HDR,
            true,
        );
        let particle_draw_group = group(
            &particle_draw.get_bind_group_layout(0),
            &[entry(0, &cam_world), entry(1, &particles)],
        );

        // Stars.
        let stars_module = module("stars", wgsl::STARS);
        let stars = render(
            "stars",
            &stars_module,
            "vs_main",
            "fs_main",
            &[],
            wgpu::PrimitiveTopology::TriangleList,
            HDR,
            true,
        );
        let stars_uniform = uniform("stars", 16);
        let stars_group = group(
            &stars.get_bind_group_layout(0),
            &[entry(0, &cam_world), entry(1, &stars_uniform)],
        );

        // Post.
        let post_module = module("post", wgsl::POST);
        let tri = wgpu::PrimitiveTopology::TriangleList;
        let post_down = render(
            "bloom down",
            &post_module,
            "vs_full",
            "fs_down",
            &[],
            tri,
            HDR,
            false,
        );
        let post_up = render(
            "bloom up",
            &post_module,
            "vs_full",
            "fs_up",
            &[],
            tri,
            HDR,
            true,
        );
        let post_composite = render(
            "composite",
            &post_module,
            "vs_full",
            "fs_composite",
            &[],
            tri,
            format,
            false,
        );
        let post_uniforms: Vec<wgpu::Buffer> = (0..2 * BLOOM_MIPS + 1)
            .map(|_| uniform("post", size_of::<PostUniform>()))
            .collect();
        let sampler = d.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear clamp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let timer = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| Timer {
                set: d.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 8,
                }),
                resolve: d.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("timestamp resolve"),
                    size: 64,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                read: d.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("timestamp read"),
                    size: 64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                ready: Arc::new(AtomicBool::new(false)),
                pending: false,
                period: queue.get_timestamp_period(),
                last: Timings::default(),
            });
        let targets = Renderer::targets(
            &device,
            &sampler,
            &post_down,
            &post_up,
            &post_composite,
            &post_uniforms,
            size,
        );
        let r = Renderer {
            device,
            queue,
            grid,
            sampler,
            cam_world,
            cam_hud,
            lines,
            lines_world,
            lines_hud,
            instances,
            grid_nodes,
            grid_uniform,
            grid_sources,
            grid_step,
            grid_step_groups,
            grid_look,
            grid_draw,
            grid_draw_groups,
            grid_parity: 0,
            particles,
            particle_step_uniform,
            particle_wells,
            particle_step,
            particle_step_group,
            particle_draw,
            particle_draw_group,
            particle_head: 0,
            particle_count: 0,
            stars,
            stars_uniform,
            stars_group,
            post_down,
            post_up,
            post_composite,
            post_uniforms,
            targets,
            timer,
            time: 0.0,
        };
        r.write_post_uniforms(&PostSettings {
            bloom: 0.0,
            exposure: 1.0,
            vignette: 0.0,
            grain: 0.0,
            aberration: 0.0,
            saturation: 1.0,
            shock: None,
        });
        r
    }

    fn targets(
        d: &wgpu::Device,
        sampler: &wgpu::Sampler,
        down: &wgpu::RenderPipeline,
        up: &wgpu::RenderPipeline,
        composite: &wgpu::RenderPipeline,
        uniforms: &[wgpu::Buffer],
        size: [u32; 2],
    ) -> Targets {
        let tex = |label: &str, [w, h]: [u32; 2]| {
            d.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w.max(1),
                    height: h.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HDR,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let hdr = tex("hdr", size);
        let mut mips = Vec::new();
        let mut s = size;
        for _ in 0..BLOOM_MIPS {
            s = [(s[0] / 2).max(1), (s[1] / 2).max(1)];
            mips.push((tex("bloom", s), s));
        }
        let bind = |p: &wgpu::RenderPipeline,
                    src: &wgpu::TextureView,
                    u: &wgpu::Buffer,
                    extra: Option<&wgpu::TextureView>| {
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: u.as_entire_binding(),
                },
            ];
            if let Some(b) = extra {
                entries.push(wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(b),
                });
            }
            d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &p.get_bind_group_layout(0),
                entries: &entries,
            })
        };
        // Down k reads the level above (the HDR scene for k = 0) into mip k.
        let down_groups = (0..BLOOM_MIPS)
            .map(|k| {
                let src = if k == 0 { &hdr } else { &mips[k - 1].0 };
                bind(down, src, &uniforms[k], None)
            })
            .collect();
        // Up k reads mip k + 1 into mip k.
        let up_groups = (0..BLOOM_MIPS - 1)
            .map(|k| bind(up, &mips[k + 1].0, &uniforms[BLOOM_MIPS + k], None))
            .collect();
        let composite_group = bind(composite, &hdr, &uniforms[2 * BLOOM_MIPS], Some(&mips[0].0));
        Targets {
            hdr,
            mips,
            down: down_groups,
            up: up_groups,
            composite: composite_group,
            size,
        }
    }

    /// Resize the targets to the output's new size.
    pub fn resize(&mut self, size: [u32; 2]) {
        if size == self.targets.size || size[0] == 0 || size[1] == 0 {
            return;
        }
        self.targets = Renderer::targets(
            &self.device,
            &self.sampler,
            &self.post_down,
            &self.post_up,
            &self.post_composite,
            &self.post_uniforms,
            size,
        );
    }

    fn write_post_uniforms(&self, p: &PostSettings) {
        let q = &self.queue;
        let size = self.targets.size;
        let texel = |[w, h]: [u32; 2]| [1.0 / w.max(1) as f32, 1.0 / h.max(1) as f32];
        for k in 0..BLOOM_MIPS {
            let src = if k == 0 {
                size
            } else {
                self.targets.mips[k - 1].1
            };
            let t = texel(src);
            let u = PostUniform {
                texel: [t[0], t[1], 1.0, 0.0],
                k: [if k == 0 { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
                ..Default::default()
            };
            q.write_buffer(&self.post_uniforms[k], 0, bytemuck::bytes_of(&u));
        }
        for k in 0..BLOOM_MIPS - 1 {
            let t = texel(self.targets.mips[k + 1].1);
            let u = PostUniform {
                texel: [t[0], t[1], 1.0, 0.0],
                ..Default::default()
            };
            q.write_buffer(
                &self.post_uniforms[BLOOM_MIPS + k],
                0,
                bytemuck::bytes_of(&u),
            );
        }
        let (centre, radius, strength) = p.shock.unwrap_or(([0.5, 0.5], 0.0, 0.0));
        let u = PostUniform {
            texel: [0.0; 4],
            k: [0.0, p.bloom, p.exposure, self.time],
            look: [p.vignette, p.grain, p.aberration, p.saturation],
            shock: [centre[0], centre[1], radius, strength],
        };
        q.write_buffer(
            &self.post_uniforms[2 * BLOOM_MIPS],
            0,
            bytemuck::bytes_of(&u),
        );
    }

    fn upload_instances(&mut self, world: &[LineInstance], hud: &[LineInstance]) -> (u32, u32) {
        let bytes = size_of_val(world) + size_of_val(hud);
        if bytes as u64 > self.instances.size() {
            self.instances = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("line instances"),
                size: (bytes as u64).next_power_of_two(),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !world.is_empty() {
            self.queue
                .write_buffer(&self.instances, 0, bytemuck::cast_slice(world));
        }
        if !hud.is_empty() {
            self.queue.write_buffer(
                &self.instances,
                size_of_val(world) as u64,
                bytemuck::cast_slice(hud),
            );
        }
        (world.len() as u32, hud.len() as u32)
    }

    fn spawn_particles(&mut self, ps: &[Particle]) {
        let mut ps = ps;
        if ps.len() > MAX_PARTICLES {
            ps = &ps[ps.len() - MAX_PARTICLES..];
        }
        let first = (MAX_PARTICLES - self.particle_head).min(ps.len());
        let at = |i: usize| (i * size_of::<Particle>()) as u64;
        if first > 0 {
            self.queue.write_buffer(
                &self.particles,
                at(self.particle_head),
                bytemuck::cast_slice(&ps[..first]),
            );
        }
        if ps.len() > first {
            self.queue
                .write_buffer(&self.particles, 0, bytemuck::cast_slice(&ps[first..]));
        }
        self.particle_head = (self.particle_head + ps.len()) % MAX_PARTICLES;
        self.particle_count = (self.particle_count + ps.len()).min(MAX_PARTICLES);
    }

    /// Timings of an earlier frame (updated every few frames).
    pub fn timings(&self) -> Option<Timings> {
        self.timer.as_ref().map(|t| t.last)
    }

    fn read_timer(&mut self) {
        let Some(t) = self.timer.as_mut() else { return };
        if t.pending && t.ready.swap(false, Ordering::Acquire) {
            {
                let view = t
                    .read
                    .slice(..)
                    .get_mapped_range()
                    .expect("mapped timestamps");
                let ts: &[u64] = bytemuck::cast_slice(&view);
                let ms = |a: usize, b: usize| {
                    (ts[b].wrapping_sub(ts[a]) as f64 * f64::from(t.period) / 1e6) as f32
                };
                t.last = Timings {
                    compute: ms(0, 1),
                    scene: ms(2, 3),
                    bloom: ms(4, 5),
                    composite: ms(6, 7),
                };
            }
            t.read.unmap();
            t.pending = false;
        }
    }

    fn ts(&self, begin: u32, end: u32) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.timer
            .as_ref()
            .map(|t| wgpu::RenderPassTimestampWrites {
                query_set: &t.set,
                beginning_of_pass_write_index: (begin != u32::MAX).then_some(begin),
                end_of_pass_write_index: (end != u32::MAX).then_some(end),
            })
    }

    /// Draw a frame into `target` (the size given at creation or last resize).
    #[allow(clippy::too_many_lines)]
    pub fn render(&mut self, target: &wgpu::TextureView, f: &Frame) {
        self.read_timer();
        self.time = f.camera.viewport[3];
        let q = &self.queue;
        q.write_buffer(&self.cam_world, 0, bytemuck::bytes_of(&f.camera));
        q.write_buffer(&self.cam_hud, 0, bytemuck::bytes_of(&f.hud));
        q.write_buffer(
            &self.stars_uniform,
            0,
            bytemuck::bytes_of(&[f.centre[0], f.centre[1], 0.0f32, 0.0]),
        );
        let g = self.grid;
        q.write_buffer(
            &self.grid_look,
            0,
            bytemuck::bytes_of(&GridLook {
                dims: [g.cols, g.rows, 0, 0],
                color: f.grid_color,
                style: [0.014, 0.1, 0.12, g.spacing],
            }),
        );
        let mut wells = [[0.0f32; 4]; MAX_WELLS];
        let nw = f.wells.len().min(MAX_WELLS);
        wells[..nw].copy_from_slice(&f.wells[..nw]);
        q.write_buffer(&self.particle_wells, 0, bytemuck::cast_slice(&wells));
        q.write_buffer(
            &self.particle_step_uniform,
            0,
            bytemuck::bytes_of(&StepUniform {
                dims: [MAX_PARTICLES as u32, nw as u32, 0, 0],
                k: [crate::sim::DT, 1.0, 0.0, 0.0],
            }),
        );
        let steps = f.grid_steps.len().min(self.grid_sources.len());
        for (s, sources) in f.grid_steps.iter().take(steps).enumerate() {
            let mut buf = [[0.0f32; 4]; MAX_SOURCES];
            let n = sources.len().min(MAX_SOURCES);
            buf[..n].copy_from_slice(&sources[..n]);
            q.write_buffer(&self.grid_sources[s], 0, bytemuck::cast_slice(&buf));
        }
        // Every step loops over the most sources any step has; unused slots have strength 0.
        let max_sources = f
            .grid_steps
            .iter()
            .take(steps)
            .map(Vec::len)
            .max()
            .unwrap_or(0)
            .min(MAX_SOURCES) as u32;
        q.write_buffer(
            &self.grid_uniform,
            0,
            bytemuck::bytes_of(&GridUniform {
                dims: [g.cols, g.rows, max_sources, 0],
                k: [
                    cpu_grid::SPRING,
                    g.spacing,
                    cpu_grid::ANCHOR,
                    cpu_grid::DAMPING,
                ],
                frame: [g.origin[0], g.origin[1], g.spacing, crate::sim::DT],
            }),
        );
        self.spawn_particles(f.spawn);
        self.write_post_uniforms(&f.post);
        let (nworld, nhud) = self.upload_instances(f.world, f.hud_lines);

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute"),
                timestamp_writes: self
                    .timer
                    .as_ref()
                    .map(|t| wgpu::ComputePassTimestampWrites {
                        query_set: &t.set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }),
            });
            pass.set_pipeline(&self.grid_step);
            for s in 0..steps {
                pass.set_bind_group(0, &self.grid_step_groups[s][self.grid_parity], &[]);
                pass.dispatch_workgroups((g.cols * g.rows).div_ceil(64), 1, 1);
                self.grid_parity = 1 - self.grid_parity;
            }
            if self.particle_count > 0 {
                pass.set_pipeline(&self.particle_step);
                pass.set_bind_group(0, &self.particle_step_group, &[]);
                // One particle step per simulation tick, like the lattice.
                for _ in 0..steps {
                    pass.dispatch_workgroups((self.particle_count as u32).div_ceil(256), 1, 1);
                }
            }
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.hdr,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                timestamp_writes: self.ts(2, 3),
                ..Default::default()
            });
            // `WARP_SKIP=grid,particles,lines,stars` leaves passes out (for looking at one alone).
            let skip = std::env::var("WARP_SKIP").unwrap_or_default();
            let on = |name: &str| {
                !skip.split(',').any(|s| s == name)
                    && (f.plane || !matches!(name, "stars" | "grid" | "particles"))
            };
            if on("stars") {
                pass.set_pipeline(&self.stars);
                pass.set_bind_group(0, &self.stars_group, &[]);
                pass.draw(0..3, 0..1);
            }
            let edges = (g.cols - 1) * g.rows + g.cols * (g.rows - 1);
            if on("grid") {
                pass.set_pipeline(&self.grid_draw);
                pass.set_bind_group(0, &self.grid_draw_groups[self.grid_parity], &[]);
                pass.draw(0..4, 0..edges);
            }
            if self.particle_count > 0 && on("particles") {
                pass.set_pipeline(&self.particle_draw);
                pass.set_bind_group(0, &self.particle_draw_group, &[]);
                pass.draw(0..4, 0..self.particle_count as u32);
            }
            pass.set_pipeline(&self.lines);
            pass.set_vertex_buffer(0, self.instances.slice(..));
            if nworld > 0 && on("lines") {
                pass.set_bind_group(0, &self.lines_world, &[]);
                pass.draw(0..4, 0..nworld);
            }
            if nhud > 0 {
                pass.set_bind_group(0, &self.lines_hud, &[]);
                pass.draw(0..4, nworld..nworld + nhud);
            }
        }
        // Bloom: down the chain, then back up, adding.
        for k in 0..BLOOM_MIPS {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom down"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.mips[k].0,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                timestamp_writes: if k == 0 { self.ts(4, u32::MAX) } else { None },
                ..Default::default()
            });
            pass.set_pipeline(&self.post_down);
            pass.set_bind_group(0, &self.targets.down[k], &[]);
            pass.draw(0..3, 0..1);
        }
        for k in (0..BLOOM_MIPS - 1).rev() {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom up"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.mips[k].0,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                timestamp_writes: if k == 0 { self.ts(u32::MAX, 5) } else { None },
                ..Default::default()
            });
            pass.set_pipeline(&self.post_up);
            pass.set_bind_group(0, &self.targets.up[k], &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                timestamp_writes: self.ts(6, 7),
                ..Default::default()
            });
            pass.set_pipeline(&self.post_composite);
            pass.set_bind_group(0, &self.targets.composite, &[]);
            pass.draw(0..3, 0..1);
        }
        let mut map_timer = false;
        if let Some(t) = self.timer.as_mut()
            && !t.pending
        {
            enc.resolve_query_set(&t.set, 0..8, &t.resolve, 0);
            enc.copy_buffer_to_buffer(&t.resolve, 0, &t.read, 0, 64);
            map_timer = true;
        }
        self.queue.submit([enc.finish()]);
        if map_timer && let Some(t) = self.timer.as_mut() {
            let ready = t.ready.clone();
            t.read.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                if r.is_ok() {
                    ready.store(true, Ordering::Release);
                }
            });
            t.pending = true;
        }
        let _ = self.device.poll(wgpu::PollType::Poll);
    }

    /// Read the lattice back (blocking; for tests and checks).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn read_grid(&self) -> Vec<Node> {
        let n = (self.grid.cols * self.grid.rows) as usize;
        self.read(&self.grid_nodes[self.grid_parity], n * size_of::<Node>())
    }

    /// Read the particle pool back (blocking; for tests).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn read_particles(&self, n: usize) -> Vec<Particle> {
        self.read(
            &self.particles,
            n.min(MAX_PARTICLES) * size_of::<Particle>(),
        )
    }

    /// Replace the lattice's state (for tests).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn write_grid(&mut self, nodes: &[Node]) {
        self.queue.write_buffer(
            &self.grid_nodes[self.grid_parity],
            0,
            bytemuck::cast_slice(nodes),
        );
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn read<T: Pod>(&self, buf: &wgpu::Buffer, bytes: usize) -> Vec<T> {
        let read = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(buf, 0, &read, 0, bytes as u64);
        self.queue.submit([enc.finish()]);
        let slice = read.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU finishes");
        let out = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
        read.unmap();
        out
    }
}
