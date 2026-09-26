//! PGA2D motors on the GPU with plain wgpu.
//!
//! * Ships are instanced triangles placed by unit motors in the vertex shader (16 bytes per
//!   instance, the `gax::wgsl` layout), moved on the CPU.
//! * Explosions are particles stepped in a compute shader by `particle_step`, a kernel
//!   written once in Rust (`src/kernels.rs`) and traced at build time into a fused Rust
//!   function and a WGSL function: the CPU and the GPU run the same verified program.
//! * The shaders import gax's WGSL modules and are linked by the `wesl` crate in `build.rs`.
//!
//! `cargo run --release` opens a window. `-- --check` runs headless and compares a few
//! particles on the GPU with their CPU twins (in debug builds the window does too, every
//! second). `-- --bench` measures motor against matrix instancing, and the GPU particle step
//! against the CPU batch kernel.

mod bench;
mod gfx;
mod kernels;
mod scene;

// The traced kernel: `particle_step`, its batch forms, and `FUSED_WESL`.
include!(concat!(env!("OUT_DIR"), "/fused.rs"));

use gfx::Gfx;
use scene::{HALF_WIDTH, Scene};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const DT: f32 = 1.0 / 60.0;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--bench") {
        bench::run(&mut headless());
    } else if args.iter().any(|a| a == "--check") {
        check();
    } else {
        let event_loop = EventLoop::new().expect("an event loop");
        let mut app = App { state: None };
        event_loop.run_app(&mut app).expect("the event loop");
    }
}

/// A `Gfx` without a window, drawing into `Rgba8Unorm` textures.
fn headless() -> Gfx {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let (adapter, device, queue) = gfx::device(&instance, None);
    let info = adapter.get_info();
    println!(
        "GPU: {} ({:?}, {:?})",
        info.name, info.device_type, info.backend
    );
    Gfx::new(device, queue, wgpu::TextureFormat::Rgba8Unorm)
}

/// An offscreen colour target.
fn target(gfx: &Gfx, width: u32, height: u32) -> wgpu::TextureView {
    gfx.device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: gfx.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

/// Run the scene headless and compare GPU particles with their CPU twins.
fn check() {
    // The traced kernel equals the code it was traced from.
    let mut rng = scene::Rng::new(9);
    for _ in 0..1000 {
        let m = gax::pga2d::Motor::<(), f32>::from_coeffs(core::array::from_fn(|_| {
            rng.range(-1.0, 1.0)
        }));
        let r = scene::rate(rng.range(-3.0, 3.0), rng.range(-6.0, 6.0));
        let (a, b) = (particle_step(m, r, DT), kernels::particle_step(m, r, DT));
        for (x, y) in a.c.iter().zip(&b.c) {
            assert!(
                (x - y).abs() < 1e-5 * (1.0 + y.abs()),
                "fused {a:?} vs generic {b:?}"
            );
        }
    }
    let mut gfx = headless();
    let view = target(&gfx, 512, 288);
    gfx.set_view(HALF_WIDTH, 512.0 / 288.0);
    let mut scene = Scene::new(64, 20_000);
    for frame in 1..=300 {
        let mut enc = gfx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        scene.update(&mut gfx, &mut enc, DT);
        gfx.draw(&mut enc, &view, false, scene.particles());
        gfx.queue.submit([enc.finish()]);
        if frame % 60 == 0 {
            let worst = scene.check(&gfx);
            println!(
                "frame {frame}: {} particles, GPU vs CPU motors differ by at most {worst:.1e}",
                scene.particles()
            );
        }
    }
    assert!(scene.compared > 10, "compared {} probes", scene.compared);
    assert!(
        scene.worst < 1e-4,
        "GPU and CPU particles differ by {}",
        scene.worst
    );
    println!(
        "ok: {} probe comparisons, worst difference {:.1e}",
        scene.compared, scene.worst
    );
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gfx: Gfx,
    scene: Scene,
    frames: u64,
}

struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes().with_title("gax: PGA2D motors on the GPU"),
                )
                .expect("a window"),
        );
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(
                Box::new(event_loop.owned_display_handle()),
            ));
        let surface = instance.create_surface(window.clone()).expect("a surface");
        let (adapter, device, queue) = gfx::device(&instance, Some(&surface));
        let size = window.inner_size();
        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("a surface configuration");
        surface.configure(&device, &config);
        let gfx = Gfx::new(device, queue, config.format);
        self.state = Some(State {
            window,
            surface,
            config,
            gfx,
            scene: Scene::new(96, 25_000),
            frames: 0,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(s) = self.state.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                s.config.width = size.width.max(1);
                s.config.height = size.height.max(1);
                s.surface.configure(&s.gfx.device, &s.config);
            }
            WindowEvent::RedrawRequested => {
                let frame = match s.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(f)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                    _ => {
                        s.surface.configure(&s.gfx.device, &s.config);
                        s.window.request_redraw();
                        return;
                    }
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                s.gfx
                    .set_view(HALF_WIDTH, s.config.width as f32 / s.config.height as f32);
                let mut enc = s
                    .gfx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                s.scene.update(&mut s.gfx, &mut enc, DT);
                s.gfx.draw(&mut enc, &view, false, s.scene.particles());
                s.gfx.queue.submit([enc.finish()]);
                s.gfx.queue.present(frame);
                s.frames += 1;
                // In debug builds, compare the GPU particles with their CPU twins every second.
                if cfg!(debug_assertions) && s.frames % 60 == 0 {
                    let worst = s.scene.check(&s.gfx);
                    assert!(worst < 1e-4, "GPU and CPU particles differ by {worst}");
                }
                s.window.request_redraw();
            }
            _ => {}
        }
    }
}
