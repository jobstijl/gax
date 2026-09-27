//! warp: a neon twin-stick shooter on warping space, and a real-load test of gax.
//!
//! Bevy is plumbing only (app, ECS, windows, input, gamepads, time); everything visual is the
//! game's own wgpu renderer with WESL shaders on gax's generated modules, and the audio is the
//! game's own synthesizer on Firewheel through bevy_seedling. The simulation (`sim`) is a pure
//! core; all geometry is gax.

mod audio;
mod fx;
mod headless;
mod input;
#[cfg(test)]
mod kernels;
mod render;
mod sim;

// The traced kernels (`grid_node`, `source_force`, `particle_step`), their batch forms, and
// their WGSL (`FUSED_WESL`).
include!(concat!(env!("OUT_DIR"), "/fused.rs"));

use bevy::prelude::*;
use bevy::window::{PrimaryWindow, RawHandleWrapper, WindowMode, WindowResized};
use fx::Fx;
use render::font::Align;
use render::scene::{self, palette};
use render::{GridSpec, LineInstance, Renderer};
use sim::{ARENA, DT, Phase, World as Sim};
use std::time::Instant;

/// Where the game is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Screen {
    Title,
    Playing,
    Paused,
    Over,
}

/// The game: the simulation, effects, and what the HUD shows.
#[derive(Resource)]
struct Game {
    sim: Sim,
    fx: Fx,
    screen: Screen,
    acc: f32,
    alpha: f32,
    input: sim::Input,
    device: input::Device,
    cam: sim::body::Pose,
    half_height: f32,
    time: f32,
    debug: bool,
    best: u64,
    sim_ms: f32,
    fps: f32,
    over_timer: f32,
    /// A bot plays (the `--smoke` run).
    bot: bool,
    world_lines: Vec<LineInstance>,
    hud_lines: Vec<LineInstance>,
}

impl Game {
    fn new() -> Game {
        let mut g = Game {
            sim: Sim::new(1),
            fx: Fx::new(),
            screen: Screen::Title,
            acc: 0.0,
            alpha: 0.0,
            input: sim::Input::default(),
            device: input::Device::Mouse,
            cam: sim::body::pose_at(0.0, 0.0, 0.0),
            half_height: 20.0,
            time: 0.0,
            debug: false,
            best: 0,
            sim_ms: 0.0,
            fps: 60.0,
            over_timer: 0.0,
            bot: false,
            world_lines: Vec::new(),
            hud_lines: Vec::new(),
        };
        g.attract();
        g
    }

    /// The attract world behind the title: enemies wandering, a singularity breathing.
    fn attract(&mut self) {
        self.sim = Sim::new(0xa77ac7);
        self.sim.phase = Phase::Over;
        self.sim.director.enabled = false;
        for k in 0..14 {
            let a = k as f32 * 0.45;
            let kind = if k % 3 == 0 {
                sim::Kind::Chaser
            } else {
                sim::Kind::Drifter
            };
            self.sim
                .spawn(kind, [a.cos() * (10.0 + k as f32), a.sin() * 9.0]);
        }
        self.sim.spawn(sim::Kind::Singularity, [12.0, -4.0]);
    }

    fn start(&mut self) {
        let seed = u64::from(std::process::id()) ^ (self.time.to_bits() as u64) << 20 ^ 0x5eed;
        self.sim = Sim::new(seed);
        self.screen = Screen::Playing;
        self.acc = 0.0;
    }
}

/// The renderer and the window's surface (main thread only).
struct Gfx {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--shot") {
        headless::shots(&args[i + 1..]);
        return;
    }
    if let Some(i) = args.iter().position(|a| a == "--music") {
        audio::offline::run(&args[i + 1..]);
        return;
    }
    let known = ["--smoke", "--shot", "--music"];
    if let Some(bad) = args
        .iter()
        .skip(1)
        .find(|a| a.starts_with("--") && !known.contains(&a.as_str()))
    {
        eprintln!("warp: unknown option {bad} (known: {})", known.join(", "));
        std::process::exit(2);
    }
    let smoke = args.iter().any(|a| a == "--smoke");
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "warp".into(),
            resolution: (1600u32, 900u32).into(),
            ..default()
        }),
        ..default()
    }));
    if std::env::var_os("WARP_NO_AUDIO").is_some() {
        // No audio device: the same game, silent.
        app.insert_resource(audio::Sound::silent());
    } else {
        app.add_plugins(audio::AudioPlugin);
    }
    app.insert_resource(Game::new())
        .add_systems(Update, (update, draw).chain());
    if smoke {
        // `--smoke`: play a few seconds with the bot, then quit (a startup test).
        app.insert_resource(Smoke { frames: 0 })
            .add_systems(Update, smoke_test.before(update));
    }
    app.run();
}

/// Frames left in a `--smoke` run.
#[derive(Resource)]
struct Smoke {
    frames: u32,
}

fn smoke_test(mut smoke: ResMut<Smoke>, mut game: ResMut<Game>, mut exit: MessageWriter<AppExit>) {
    smoke.frames += 1;
    if smoke.frames == 30 {
        game.start();
    }
    game.bot = smoke.frames > 30;
    if game.time > 8.0 {
        info!(
            "smoke test: {} frames, score {}, drift {:.1e}",
            smoke.frames, game.sim.score, game.sim.worst_drift
        );
        exit.write(AppExit::Success);
    }
}

/// Input, the fixed-step simulation, effects and audio events.
#[allow(clippy::too_many_arguments)]
fn update(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    pads: Query<&bevy::input::gamepad::Gamepad>,
    mut game: ResMut<Game>,
    mut sound: ResMut<audio::Sound>,
    mut exit: MessageWriter<AppExit>,
) {
    let dt = time.delta_secs().min(0.1);
    let g = &mut *game;
    g.time += dt;
    g.fps = g.fps * 0.95 + 0.05 / dt.max(1e-4);
    let mut window = windows.single_mut().ok();
    let (mut input, menu) = input::read(
        &keys,
        &mouse,
        window.as_deref(),
        &pads,
        g.cam,
        g.half_height,
        g.sim.ship.body.xy(),
        &mut g.device,
    );
    if g.bot {
        input = headless::bot_input(g, g.time);
    }
    if menu.debug {
        g.debug = !g.debug;
    }
    if menu.fullscreen
        && let Some(w) = window.as_mut()
    {
        w.mode = match w.mode {
            WindowMode::Windowed => {
                WindowMode::BorderlessFullscreen(bevy::window::MonitorSelection::Current)
            }
            _ => WindowMode::Windowed,
        };
    }
    let aspect = window
        .as_ref()
        .map_or(16.0 / 9.0, |w| w.width() / w.height().max(1.0));
    if advance(g, input, menu, dt, aspect, &mut sound) {
        exit.write(AppExit::Success);
    }
}

/// One frame of game logic, without Bevy: menus, the fixed-step simulation, effects and sound.
/// Returns whether the player asked to quit.
fn advance(
    g: &mut Game,
    input: sim::Input,
    menu: input::Menu,
    dt: f32,
    aspect: f32,
    sound: &mut audio::Sound,
) -> bool {
    let mut quit = false;
    match g.screen {
        Screen::Title if menu.start => g.start(),
        Screen::Title if menu.back => quit = true,
        Screen::Playing if menu.back => g.screen = Screen::Paused,
        Screen::Paused if menu.start || menu.back => g.screen = Screen::Playing,
        Screen::Over => {
            g.over_timer += dt;
            if g.over_timer > 1.5 && (menu.start || menu.back) {
                g.over_timer = 0.0;
                g.screen = Screen::Title;
                g.attract();
            }
        }
        _ => {}
    }
    // Bomb is an edge: keep it until a tick consumes it.
    g.input = sim::Input {
        bomb: g.input.bomb || input.bomb,
        ..input
    };
    g.fx.grid_steps.clear();
    g.fx.spawn.clear();
    let started = Instant::now();
    if g.screen != Screen::Paused {
        g.acc += dt;
        while g.acc >= DT {
            g.acc -= DT;
            if g.sim.hitstop > 0.0 {
                // Hit-stop: the world holds its breath; effects go on.
                g.sim.hitstop = (g.sim.hitstop - DT).max(0.0);
            } else {
                let tick_input = if g.screen == Screen::Playing {
                    g.input
                } else {
                    sim::Input::default()
                };
                g.sim.tick(&tick_input);
                g.input.bomb = false;
                g.fx.on_events(&g.sim.events);
                sound.on_events(&g.sim.events, &g.sim, g.cam);
                if g.sim.events.contains(&sim::Event::GameOver) {
                    g.best = g.best.max(g.sim.score);
                    g.screen = Screen::Over;
                    g.over_timer = 0.0;
                }
            }
            if g.fx.grid_steps.len() < 8 {
                g.fx.grid_tick(&g.sim, DT);
            }
        }
        g.alpha = g.acc / DT;
    }
    g.sim_ms = g.sim_ms * 0.9 + 0.1 * started.elapsed().as_secs_f32() * 1e3;
    sound.update(&g.sim, g.screen == Screen::Playing, g.cam);
    // A view about 28 units tall (wider screens see more), moving over the arena.
    g.half_height = 14.0_f32.max(24.0 / aspect);
    g.fx.half_height = g.half_height;
    g.fx.aspect = aspect;
    g.cam = g.fx.camera(&g.sim, dt);
    quit
}

/// Build the frame and draw it (an exclusive system: the surface lives on the main thread).
fn draw(world: &mut World) {
    let mut resized = false;
    {
        let mut reader = world.resource_mut::<Messages<WindowResized>>();
        if !reader.is_empty() {
            resized = true;
            reader.clear();
        }
    }
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
        draw_frame(&mut game, &mut gfx, size, resized);
    });
}

fn draw_frame(game: &mut Game, gfx: &mut Gfx, size: [u32; 2], resized: bool) {
    if resized || gfx.config.width != size[0] || gfx.config.height != size[1] {
        gfx.config.width = size[0];
        gfx.config.height = size[1];
        let Gfx {
            surface,
            config,
            renderer,
        } = gfx;
        surface.configure(&renderer.device, config);
        renderer.resize(size);
    }
    let frame = match gfx.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
        _ => {
            let Gfx {
                surface,
                config,
                renderer,
            } = gfx;
            surface.configure(&renderer.device, config);
            return;
        }
    };
    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    render_game(game, &mut gfx.renderer, &view, size);
    gfx.renderer.queue.present(frame);
}

/// Build a frame from the game and draw it into `view`.
fn render_game(game: &mut Game, renderer: &mut Renderer, view: &wgpu::TextureView, size: [u32; 2]) {
    let camera = scene::camera(game.cam, game.half_height, size, game.time);
    let hud_cam = scene::camera(sim::body::pose_at(0.0, 0.0, 0.0), 18.0, size, game.time);
    scene::world_lines(&game.sim, game.alpha, game.time, &mut game.world_lines);
    for (p, text, age, color) in &game.fx.popups {
        // Rising and fading.
        let fade = (1.0 - age / 1.1).max(0.0);
        let c = [color[0], color[1], color[2], color[3] * fade];
        scene::text(
            &mut game.world_lines,
            text,
            p[0],
            p[1] + 0.8 + age * 1.5,
            0.9,
            c,
            Align::Center,
        );
    }
    hud(game, size, renderer.timings());
    let wells = Fx::particle_wells(&game.sim);
    let post = game.fx.post(&camera);
    let intensity = if game.screen == Screen::Playing {
        game.sim.director.intensity
    } else {
        0.3
    };
    let mut grid_color = palette::GRID;
    grid_color[3] *= 0.85 + 0.5 * intensity;
    let f = render::Frame {
        camera,
        hud: hud_cam,
        centre: (game.cam >> gax::pga2d::Point::xy(0.0, 0.0)).to_euclidean(),
        world: &game.world_lines,
        hud_lines: &game.hud_lines,
        grid_steps: &game.fx.grid_steps,
        wells: &wells,
        spawn: &game.fx.spawn,
        grid_color,
        post,
    };
    renderer.render(view, &f);
}

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
    // Vsync: smooth, and no power burnt on frames nobody sees.
    let mode = wgpu::PresentMode::AutoVsync;
    let mut config = surface
        .get_default_config(&adapter, size[0], size[1])
        .expect("a surface configuration");
    config.format = format;
    config.present_mode = mode;
    config.desired_maximum_frame_latency = 2;
    surface.configure(&device, &config);
    let renderer = Renderer::new(device, queue, format, size, grid_spec());
    Gfx {
        surface,
        config,
        renderer,
    }
}

/// The lattice: 129 x 73 nodes over the arena (128 x 72 cells).
pub fn grid_spec() -> GridSpec {
    GridSpec {
        cols: 129,
        rows: 73,
        origin: [-ARENA[0], -ARENA[1]],
        spacing: 2.0 * ARENA[0] / 128.0,
    }
}

/// The HUD, in HUD units (the screen is 36 units tall, centred).
fn hud(g: &mut Game, size: [u32; 2], timings: Option<render::Timings>) {
    let out = &mut g.hud_lines;
    out.clear();
    let aspect = size[0] as f32 / size[1] as f32;
    let (left, right, top) = (-18.0 * aspect + 2.2, 18.0 * aspect - 2.2, 18.0 - 2.6);
    let hud = palette::HUD;
    let dim = [hud[0], hud[1], hud[2], hud[3] * 0.45];
    let blink = (g.time * 3.0).sin() > -0.3;
    match g.screen {
        Screen::Title => {
            scene::text(
                out,
                "WARP",
                0.0,
                3.0,
                6.5,
                [0.55, 0.7, 1.0, 1.7],
                Align::Center,
            );
            scene::text(
                out,
                "A NEON SHOOTER ON WARPED SPACE",
                0.0,
                -0.6,
                1.0,
                dim,
                Align::Center,
            );
            if blink {
                let msg = if g.device == input::Device::Pad {
                    "PRESS START"
                } else {
                    "PRESS ENTER"
                };
                scene::text(out, msg, 0.0, -5.5, 1.4, hud, Align::Center);
            }
            scene::text(
                out,
                "MOVE WASD / LEFT STICK    AIM + FIRE MOUSE / RIGHT STICK    BOMB SPACE / TRIGGER",
                0.0,
                -12.0,
                0.7,
                dim,
                Align::Center,
            );
            if g.best > 0 {
                scene::text(
                    out,
                    &format!("BEST {}", scene::grouped(g.best)),
                    0.0,
                    -14.0,
                    0.9,
                    dim,
                    Align::Center,
                );
            }
        }
        Screen::Playing | Screen::Paused | Screen::Over => {
            let s = &g.sim;
            scene::text(
                out,
                &scene::grouped(s.score),
                left,
                top - 0.2,
                1.3,
                hud,
                Align::Left,
            );
            scene::text(
                out,
                &format!("X{}", s.mult),
                left,
                top - 2.2,
                0.9,
                palette::SHARD,
                Align::Left,
            );
            // Lives as small ships, bombs as rings.
            for k in 0..s.lives.saturating_sub(1).min(8) {
                let m = sim::body::pose_at(
                    right - 0.5 - k as f32 * 1.4,
                    top + 0.3,
                    core::f32::consts::FRAC_PI_2,
                );
                scene::draw_ship(out, m, 0.0, g.time, 0.8);
            }
            for k in 0..s.bombs.min(8) {
                let x = right - 0.5 - k as f32 * 1.3;
                let m = sim::body::pose_at(x, top - 1.9, 0.0);
                for i in 0..12 {
                    let (a, b) = (
                        i as f32 * core::f32::consts::FRAC_PI_6,
                        (i + 1) as f32 * core::f32::consts::FRAC_PI_6,
                    );
                    out.push(scene::seg(
                        [
                            0.45 * a.cos(),
                            0.45 * a.sin(),
                            0.45 * b.cos(),
                            0.45 * b.sin(),
                        ],
                        [0.55, 0.8, 1.0, 2.0],
                        [0.04, 0.2, 0.2, 0.0],
                        m,
                    ));
                }
            }
            if let Phase::Dead(_) = s.phase {
                scene::text(
                    out,
                    "SHIP LOST",
                    0.0,
                    1.0,
                    2.0,
                    palette::SHIP,
                    Align::Center,
                );
            }
            if g.screen == Screen::Paused {
                scene::text(out, "PAUSED", 0.0, 1.0, 3.0, hud, Align::Center);
                scene::text(
                    out,
                    "ENTER / START TO GO ON    ESC / BACK TO GO ON",
                    0.0,
                    -2.5,
                    0.8,
                    dim,
                    Align::Center,
                );
            }
            if g.screen == Screen::Over {
                scene::text(
                    out,
                    "GAME OVER",
                    0.0,
                    3.0,
                    3.2,
                    [1.0, 0.35, 0.5, 3.0],
                    Align::Center,
                );
                scene::text(
                    out,
                    &format!("SCORE {}", scene::grouped(s.score)),
                    0.0,
                    -0.8,
                    1.4,
                    hud,
                    Align::Center,
                );
                if s.score >= g.best && s.score > 0 {
                    scene::text(
                        out,
                        "NEW BEST",
                        0.0,
                        -3.2,
                        1.0,
                        palette::SHARD,
                        Align::Center,
                    );
                }
                if g.over_timer > 1.5 && blink {
                    scene::text(out, "PRESS ENTER", 0.0, -6.5, 1.0, dim, Align::Center);
                }
            }
        }
    }
    if g.debug {
        let s = &g.sim;
        let mut lines = vec![
            format!("FPS {:.0}  SIM {:.2} MS", g.fps, g.sim_ms),
            format!(
                "ENEMIES {}  BULLETS {}  SHARDS {}",
                s.enemies.len(),
                s.bullets.len(),
                s.shards.len()
            ),
            format!(
                "INTENSITY {:.2}  DRIFT {:.1E}",
                s.director.intensity, s.worst_drift
            ),
        ];
        if let Some(t) = timings {
            lines.push(format!(
                "GPU COMPUTE {:.2} SCENE {:.2} BLOOM {:.2} POST {:.2}",
                t.compute, t.scene, t.bloom, t.composite
            ));
        }
        for (i, l) in lines.iter().enumerate() {
            scene::text(
                out,
                l,
                left,
                -16.5 + 1.1 * (lines.len() - 1 - i) as f32,
                0.6,
                dim,
                Align::Left,
            );
        }
    }
}
