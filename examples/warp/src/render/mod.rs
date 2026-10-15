//! The game's own renderer, on plain wgpu (no Bevy rendering).
//!
//! One HDR target (`rgba16float`) takes, additively: a parallax starfield, the warped-space
//! lattice (stepped in a compute shader by a traced gax kernel, `lattice.rs`), particles
//! (likewise, `particles.rs`) and the shapes (line segments placed by unit motors in the vertex
//! shader, `lines.rs`). Bloom is a mip chain; the composite tonemaps with AgX into the output
//! (`post.rs`), and the HUD's lines go onto the output last. GPU buffers hold gax's Pod types
//! (`gpu_types.rs`), mirrored by the shaders' structs.

#[cfg(test)]
pub mod cpu_grid;
mod factory;
pub mod font;
mod gpu_types;
mod lattice;
mod lines;
mod particles;
mod post;
pub mod scene;
#[cfg(test)]
mod tests;
mod timer;
pub mod tunnel;

pub use gpu_types::{CameraUniform, LineInstance, Particle};
pub use lattice::{GridSpec, GridSteps, MAX_SOURCES, Source};
pub use post::{PostSettings, Shock};
pub use timer::Timings;

use factory::{Blend, Factory, Target};
use gax::pga2d::{Point, PointGpu};
use lattice::GpuLattice;
use lines::GpuLines;
use particles::GpuParticles;
use post::Post;
use timer::{Span, Timer};

/// The WGSL shaders, linked at build time against `gax::wgsl`'s modules (`build.rs`).
mod wgsl {
    include!(concat!(env!("OUT_DIR"), "/shaders.rs"));
}

/// The HDR scene format.
pub const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Everything a frame draws.
pub struct Frame<'a> {
    /// The world camera.
    pub camera: CameraUniform,
    /// World-space segments (shapes, bullets, border).
    pub world: &'a [LineInstance],
    /// HUD segments, in HUD units (`scene::hud_camera`).
    pub hud_lines: &'a [LineInstance],
    /// Post-processing.
    pub post: PostSettings,
    /// The Plane's own layers; the Tunnel draws everything as lines.
    pub plane: Option<PlaneLayers<'a>>,
}

/// The Plane's layers: the starfield, the lattice and the particles.
pub struct PlaneLayers<'a> {
    /// The camera's centre in the world (for the starfield's parallax).
    pub centre: Point<(), f32>,
    /// The lattice's steps to run this frame (one per simulation tick), each with its sources.
    pub grid_steps: &'a GridSteps,
    /// The lattice's light.
    pub grid_light: crate::light::Light,
    /// Wells pulling particles.
    pub wells: &'a [Source],
    /// Particles to add before stepping.
    pub spawn: &'a [Particle],
}

/// The layers `WARP_SKIP=grid,particles,lines,stars` leaves out (for looking at one alone).
#[derive(Default)]
struct Skip {
    stars: bool,
    grid: bool,
    particles: bool,
    lines: bool,
}

impl Skip {
    /// The layers named in `WARP_SKIP`.
    fn from_env() -> Skip {
        let names = std::env::var("WARP_SKIP").unwrap_or_default();
        let named = |layer: &str| names.split(',').any(|s| s == layer);
        Skip {
            stars: named("stars"),
            grid: named("grid"),
            particles: named("particles"),
            lines: named("lines"),
        }
    }
}

/// The parallax starfield behind the Plane (`shaders/stars.wesl`).
struct Stars {
    pipeline: wgpu::RenderPipeline,
    /// The camera's centre.
    centre: wgpu::Buffer,
    group: wgpu::BindGroup,
}

impl Stars {
    fn new(f: Factory, camera: &wgpu::Buffer) -> Stars {
        let module = f.module("stars", wgsl::STARS);
        let pipeline = f.full_screen(
            "stars",
            &module,
            ["vs_main", "fs_main"],
            Target::Hdr,
            Blend::Add,
        );
        let centre = f.uniform("stars", size_of::<PointGpu>());
        let group = f.group(&pipeline.get_bind_group_layout(0), &[camera, &centre]);
        Stars {
            pipeline,
            centre,
            group,
        }
    }

    fn draw(&self, pass: &mut wgpu::RenderPass) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// The renderer.
pub struct Renderer {
    /// The device.
    pub device: wgpu::Device,
    /// Its queue.
    pub queue: wgpu::Queue,
    cam_world: wgpu::Buffer,
    cam_hud: wgpu::Buffer,
    lines: GpuLines,
    lattice: GpuLattice,
    particles: GpuParticles,
    stars: Stars,
    post: Post,
    timer: Option<Timer>,
    skip: Skip,
    /// Whether frames without the Plane's layers (the Tunnel's) came since its last: its
    /// lattice and particles then start afresh.
    plane_idle: bool,
}

impl Renderer {
    /// The features the renderer uses when the adapter has them.
    pub fn wanted_features(adapter: &wgpu::Adapter) -> wgpu::Features {
        adapter.features() & wgpu::Features::TIMESTAMP_QUERY
    }

    /// Build the renderer for output `format` and `size`, with the lattice `grid`. The format
    /// is sRGB: the shaders write linear light, and the output encodes it (a surface without
    /// an sRGB format is viewed through its sRGB twin, see `gfx.rs`).
    ///
    /// # Panics
    /// If `format` is not sRGB.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        format: wgpu::TextureFormat,
        size: [u32; 2],
        grid: GridSpec,
    ) -> Renderer {
        assert!(format.is_srgb(), "the output format {format:?} is not sRGB");
        let f = Factory(&device);
        let cam_world = f.uniform("camera", size_of::<CameraUniform>());
        let cam_hud = f.uniform("hud camera", size_of::<CameraUniform>());
        Renderer {
            lines: GpuLines::new(f, &cam_world, &cam_hud, format),
            lattice: GpuLattice::new(f, grid, &cam_world),
            particles: GpuParticles::new(f, &cam_world),
            stars: Stars::new(f, &cam_world),
            post: Post::new(f, &queue, format, size),
            timer: Timer::new(&device, &queue),
            skip: Skip::from_env(),
            plane_idle: false,
            cam_world,
            cam_hud,
            device,
            queue,
        }
    }

    /// Resize the targets to the output's new size.
    pub fn resize(&mut self, size: [u32; 2]) {
        if size != self.post.size() && size[0] > 0 && size[1] > 0 {
            self.post.resize(Factory(&self.device), &self.queue, size);
        }
    }

    /// Timings of an earlier frame (updated every few frames).
    pub fn timings(&self) -> Option<Timings> {
        self.timer.as_ref().map(Timer::last)
    }

    /// Draw a frame into `target` (the size given at creation or last resize).
    pub fn render(&mut self, target: &wgpu::TextureView, f: &Frame) {
        if let Some(t) = self.timer.as_mut() {
            t.poll();
        }
        self.upload(f);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute"),
                timestamp_writes: Timer::compute(self.timer.as_ref(), Span::Compute),
            });
            if f.plane.is_some() {
                let steps = self.lattice.encode(&mut pass);
                self.particles.encode(&mut pass, steps);
            }
        }
        self.encode_scene(&mut enc, f.plane.is_some());
        self.post.encode(&mut enc, target, self.timer.as_ref());
        if self.lines.has_hud() {
            let mut pass = begin(&mut enc, "hud", target, wgpu::LoadOp::Load, None);
            self.lines.draw_hud(&mut pass);
        }
        let map_timer = self.timer.as_ref().is_some_and(|t| t.resolve(&mut enc));
        self.queue.submit([enc.finish()]);
        if map_timer && let Some(t) = self.timer.as_mut() {
            t.map();
        }
        let _ = self.device.poll(wgpu::PollType::Poll);
    }

    /// Write what frame `f` draws: the cameras, the Plane's layers, the post-processing and the
    /// lines.
    fn upload(&mut self, f: &Frame) {
        let q = &self.queue;
        let time = f.camera.viewport[3];
        q.write_buffer(&self.cam_world, 0, bytemuck::bytes_of(&f.camera));
        let hud = scene::hud_camera(self.post.size(), time);
        q.write_buffer(&self.cam_hud, 0, bytemuck::bytes_of(&hud));
        match &f.plane {
            Some(p) => self.upload_plane(p),
            None => self.plane_idle = true,
        }
        self.post.write(&self.queue, &f.post, time);
        let factory = Factory(&self.device);
        self.lines
            .upload(factory, &self.queue, f.world, f.hud_lines);
    }

    /// Write the Plane's layers: the starfield's centre, the lattice's steps, and the
    /// particles' spawn and wells (all stepped by the simulation's tick).
    fn upload_plane(&mut self, p: &PlaneLayers) {
        let (q, dt) = (&self.queue, crate::sim::DT);
        if self.plane_idle {
            self.lattice.reset(q);
            self.particles.reset();
            self.plane_idle = false;
        }
        q.write_buffer(
            &self.stars.centre,
            0,
            bytemuck::bytes_of(&PointGpu::from(p.centre)),
        );
        self.lattice.upload(q, p.grid_steps, p.grid_light, dt);
        self.particles.spawn(q, p.spawn, dt);
        self.particles.upload(q, p.wells, dt);
    }

    /// The HDR scene: the Plane's layers (`plane`), then the world's lines.
    fn encode_scene(&self, enc: &mut wgpu::CommandEncoder, plane: bool) {
        let ts = Timer::render(self.timer.as_ref(), Some(Span::Scene), Some(Span::Scene));
        let clear = wgpu::LoadOp::Clear(wgpu::Color::BLACK);
        let mut pass = begin(enc, "scene", self.post.hdr(), clear, ts);
        if plane && !self.skip.stars {
            self.stars.draw(&mut pass);
        }
        if plane && !self.skip.grid {
            self.lattice.draw(&mut pass);
        }
        if plane && !self.skip.particles {
            self.particles.draw(&mut pass);
        }
        if !self.skip.lines {
            self.lines.draw_world(&mut pass);
        }
    }
}

/// A render pass into `view`, starting from `load`.
fn begin<'e>(
    enc: &'e mut wgpu::CommandEncoder,
    label: &str,
    view: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites>,
) -> wgpu::RenderPass<'e> {
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        })],
        timestamp_writes,
        ..Default::default()
    })
}
