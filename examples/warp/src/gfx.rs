//! The GPU side of the app: the window's surface and the renderer, the headless device, and
//! building a frame from the game.

use crate::fx::{self, Fx};
use crate::game::{Game, Run, Screen};
use crate::hud::hud;
use crate::render::font::Align;
use crate::render::scene::{self, palette};
use crate::render::{self, GridSpec, Renderer};
use crate::sim::ARENA;
use bevy::prelude::*;
use bevy::window::{PrimaryWindow, RawHandleWrapper};

/// The renderer and the window's surface (main thread only).
pub struct Gfx {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    /// The sRGB format the frames are drawn in: the surface's, or its sRGB twin (a view of
    /// the same texels that encodes on write).
    view_format: wgpu::TextureFormat,
    renderer: Renderer,
}

/// The lattice: 129 x 73 nodes over the arena (128 x 72 cells).
pub fn grid_spec() -> GridSpec {
    GridSpec {
        cols: 129,
        rows: 73,
        origin: gax::pga2d::Point::xy(-ARENA[0], -ARENA[1]),
        spacing: 2.0 * ARENA[0] / 128.0,
    }
}

/// Build the frame and draw it (an exclusive system: the surface lives on the main thread).
pub fn draw(world: &mut World) {
    let Ok((handle, size)) = world
        .query_filtered::<(&RawHandleWrapper, &Window), With<PrimaryWindow>>()
        .single(world)
        .map(|(h, w)| {
            (
                h.clone(),
                [w.physical_width().max(1), w.physical_height().max(1)],
            )
        })
    else {
        return;
    };
    if world.get_non_send::<Gfx>().is_none() {
        let gfx = create_gfx(&handle, size);
        world.insert_non_send(gfx);
    }
    world.resource_scope(|world, mut game: Mut<Game>| {
        let mut gfx = world.non_send_mut::<Gfx>();
        draw_frame(&mut game, &mut gfx, size);
    });
}

/// Draw a frame of `size` pixels onto the surface, reconfigured first if the size changed
/// (or the surface was lost).
fn draw_frame(game: &mut Game, gfx: &mut Gfx, size: [u32; 2]) {
    if gfx.config.width != size[0] || gfx.config.height != size[1] {
        gfx.config.width = size[0];
        gfx.config.height = size[1];
        gfx.surface.configure(&gfx.renderer.device, &gfx.config);
        gfx.renderer.resize(size);
    }
    let frame = match gfx.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
        _ => {
            gfx.surface.configure(&gfx.renderer.device, &gfx.config);
            return;
        }
    };
    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
        format: Some(gfx.view_format),
        ..Default::default()
    });
    render_game(game, &mut gfx.renderer, &view, size);
    gfx.renderer.queue.present(frame);
}

/// The renderer on the window's surface: a high-performance adapter, vsync, and an sRGB
/// format (the surface's own, or a view of a plain one through its sRGB twin).
fn create_gfx(handle: &RawHandleWrapper, size: [u32; 2]) -> Gfx {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    // SAFETY: the handles come from Bevy's primary window, which outlives the surface (the app
    // owns both), and this runs on the main thread (an exclusive system).
    let surface = unsafe {
        instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: Some(handle.get_display_handle()),
            raw_window_handle: handle.get_window_handle(),
        })
    }
    .expect("a surface for the window");
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        ..Default::default()
    }))
    .expect("a GPU adapter");
    let info = adapter.get_info();
    info!(
        "GPU: {} ({:?}, {:?})",
        info.name, info.device_type, info.backend
    );
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("warp"),
        required_features: Renderer::wanted_features(&adapter),
        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
        ..Default::default()
    }))
    .expect("a device");
    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(wgpu::TextureFormat::is_srgb)
        .unwrap_or(caps.formats[0]);
    let view_format = format.add_srgb_suffix();
    let mut config = surface
        .get_default_config(&adapter, size[0], size[1])
        .expect("a surface configuration");
    config.format = format;
    if view_format != format {
        config.view_formats = vec![view_format];
    }
    // Vsync: smooth, and no power burnt on frames nobody sees.
    config.present_mode = wgpu::PresentMode::AutoVsync;
    config.desired_maximum_frame_latency = 2;
    surface.configure(&device, &config);
    let renderer = Renderer::new(device, queue, view_format, size, grid_spec());
    Gfx {
        surface,
        config,
        view_format,
        renderer,
    }
}

/// A device without a window, with validation, failing loudly on any GPU error.
pub fn headless_device() -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    try_headless_device().expect("a GPU adapter")
}

/// [`headless_device`], or `None` without an adapter.
pub fn try_headless_device() -> Option<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    desc.flags |= wgpu::InstanceFlags::VALIDATION | wgpu::InstanceFlags::DEBUG;
    let instance = wgpu::Instance::new(desc);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("warp headless"),
        required_features: Renderer::wanted_features(&adapter),
        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
        ..Default::default()
    }))
    .ok()?;
    device.on_uncaptured_error(std::sync::Arc::new(|e| panic!("wgpu error: {e}")));
    Some((adapter, device, queue))
}

/// Build a frame from the game and draw it into `view`, of `size` pixels.
pub fn render_game(
    game: &mut Game,
    renderer: &mut Renderer,
    view: &wgpu::TextureView,
    size: [u32; 2],
) {
    let timings = renderer.timings();
    let frame = match game.run {
        Run::Plane(_) => plane_frame(game, size),
        Run::Tunnel(_) => tunnel_frame(game, size),
    };
    hud(game, size, timings);
    let plane = frame.plane.as_ref().map(|p| render::PlaneLayers {
        centre: game.cam >> crate::geom::ORIGIN,
        grid_steps: &game.fx.grid_steps,
        grid_light: p.grid_light,
        wells: &p.wells,
        spawn: &game.fx.spawn,
    });
    renderer.render(
        view,
        &render::Frame {
            camera: frame.camera,
            world: &game.world_lines,
            hud_lines: &game.hud_lines,
            post: frame.post,
            plane,
        },
    );
}

/// A frame's camera and post settings, and the Plane's layers.
struct Layers {
    camera: render::CameraUniform,
    post: render::PostSettings,
    plane: Option<PlaneLayers>,
}

/// The Plane's layers that are not the effects' own: the lattice's light and the wells that
/// pull the particles.
struct PlaneLayers {
    grid_light: crate::light::Light,
    wells: Vec<render::Source>,
}

/// The Plane's frame: the world's lines with the wreckage and popups, the menus' backdrop,
/// and the lattice warming with the intensity.
fn plane_frame(game: &mut Game, size: [u32; 2]) -> Layers {
    let Run::Plane(run) = &game.run else {
        unreachable!("a Plane run");
    };
    let w = &run.world;
    let out = &mut game.world_lines;
    scene::world_lines(w, game.alpha, game.time, out);
    for p in &game.fx.wreckage {
        let c = p.color.faded(p.life / fx::WRECK_LIFE);
        scene::outline(out, &p.tri, 1.0, c, scene::THIN, p.body.pose);
    }
    for p in &game.fx.popups {
        // Rising and fading.
        let c = p.color.faded((1.0 - p.age / fx::POPUP_LIFE).max(0.0));
        let at = p.pos + gax::pga2d::Point::direction(0.0, 0.8 + p.age * 1.5);
        scene::text(out, &p.text, at, 0.9, c, Align::Center);
    }
    // Menus over the world: the world steps back.
    let backdrop = matches!(
        game.screen,
        Screen::Settings | Screen::Scores | Screen::Paused | Screen::Initials
    );
    if backdrop {
        for l in out.iter_mut() {
            l.recede(0.2);
        }
    }
    let intensity = if game.screen == Screen::Playing {
        w.director.intensity
    } else {
        0.3
    };
    let dim = if backdrop { 0.5 } else { 1.0 };
    // The lattice warms towards violet as the intensity rises: its hue turned about OkLab's
    // lightness axis.
    let grid_light = (palette::GRID.perceptually(|c| c.rotate_hue(0.5 * intensity)))
        .faded((0.85 + 0.5 * intensity) * dim);
    let wells = Fx::particle_wells(w);
    let post = game
        .fx
        .post(&scene::view_map(game.cam, game.half_height, size));
    Layers {
        camera: scene::camera(game.cam, game.half_height, size, game.time),
        post,
        plane: Some(PlaneLayers { grid_light, wells }),
    }
}

/// The Tunnel's frame: everything projected into the HUD's units, so the HUD's camera draws
/// it all.
fn tunnel_frame(game: &mut Game, size: [u32; 2]) -> Layers {
    let Run::Tunnel(run) = &mut game.run else {
        unreachable!("a Tunnel run");
    };
    run.view
        .draw(&run.world, game.alpha, game.time, &mut game.world_lines);
    if game.screen == Screen::Paused {
        for l in &mut game.world_lines {
            l.recede(0.2);
        }
    }
    Layers {
        camera: scene::hud_camera(size, game.time),
        post: run.view.post(&run.world, size),
        plane: None,
    }
}
