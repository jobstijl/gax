//! warp: a neon twin-stick shooter on warping space, and a real-load test of gax.
//!
//! Bevy is plumbing only (app, ECS, windows, input, gamepads, time); everything visual is the
//! game's own wgpu renderer with WESL shaders on gax's generated modules, and the audio is the
//! game's own synthesizer on Firewheel through bevy_seedling. The simulations (`sim`, the
//! Plane, and `tunnel`) are pure cores; all geometry is gax.
//!
//! The app: the command line (`cli`), the game between frames (`game`), a frame of its logic
//! (`frame`, with the menus in `menus`), and drawing it (`gfx`, with the HUD in `hud`).

mod audio;
mod bot;
mod cli;
mod frame;
mod fx;
mod game;
mod geom;
mod gfx;
mod hud;
mod input;
// The kernels run traced (below); the tests use the generic forms to check them.
#[cfg_attr(not(test), allow(dead_code))]
mod kernels;
mod light;
mod menus;
mod png;
mod render;
mod shots;
mod signal;
mod sim;
mod store;
mod tunnel;

// The traced kernels (`grid_node`, `source_force`, `particle_step`), their batch forms, and
// their WGSL (`FUSED_WESL`).
include!(concat!(env!("OUT_DIR"), "/fused.rs"));

use bevy::prelude::*;
use bevy::window::{MonitorSelection, PrimaryWindow, WindowMode};
use cli::Command;
use game::{Game, Run};
use menus::Quit;
use std::path::Path;
use store::Store;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let command = cli::parse(&args).unwrap_or_else(|e| {
        eprintln!("warp: {e}");
        std::process::exit(2);
    });
    match command {
        Command::Shots(rest) => shots::shots(&rest),
        Command::Music(rest) => audio::offline::run(&rest),
        Command::Verify(path) => match cli::verify(Path::new(&path)) {
            Ok(line) => println!("{line}"),
            Err((e, cli::Failure::File)) => {
                eprintln!("warp: {e}");
                std::process::exit(2);
            }
            Err((e, cli::Failure::Diverged)) => {
                println!("{e}");
                std::process::exit(1);
            }
        },
        Command::Play { smoke, replay } => play(smoke, replay.as_deref()),
    }
}

/// The game in a window, watching `replay` first if given; with `smoke`, a bot plays a few
/// seconds and nothing is kept.
fn play(smoke: bool, replay: Option<&str>) {
    let replay = replay.map(|path| {
        cli::load_any(Path::new(path)).unwrap_or_else(|e| {
            eprintln!("warp: {e}");
            std::process::exit(2);
        })
    });
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
        app.insert_resource(audio::Sound::new());
    } else {
        app.add_plugins(audio::AudioPlugin);
    }
    // The smoke run keeps nothing: no settings, scores or replays are read or written.
    let mut game = Game::new(if smoke {
        Store::new(None)
    } else {
        Store::open()
    });
    if let Some(r) = replay {
        game.watch(r);
    }
    app.insert_resource(game)
        .add_systems(Update, (update, gfx::draw).chain());
    if smoke {
        app.insert_resource(Smoke { frames: 0 })
            .add_systems(Update, smoke_test.before(update));
    }
    app.run();
}

/// Frames into a `--smoke` run.
#[derive(Resource)]
struct Smoke {
    frames: u32,
}

/// `--smoke`: the bot starts a run after half a second, plays, and quits after 8 s (a startup
/// test).
fn smoke_test(mut smoke: ResMut<Smoke>, mut game: ResMut<Game>, mut exit: MessageWriter<AppExit>) {
    smoke.frames += 1;
    if smoke.frames == 30 {
        game.start(game::Mode::Plane);
    }
    game.bot = smoke.frames > 30;
    if game.time > 8.0 {
        let drift = game.run.plane().map_or(0.0, |r| r.world.worst_drift);
        info!(
            "smoke test: {} frames, score {}, drift {drift:.1e}",
            smoke.frames,
            game.run.ledger().score,
        );
        exit.write(AppExit::Success);
    }
}

/// Input, the window's mode, and a frame of the game.
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
    let ship = match &g.run {
        Run::Plane(r) => r.world.ship.body.pos(),
        Run::Tunnel(_) => geom::ORIGIN,
    };
    let view = input::View {
        camera: g.cam,
        half_height: g.half_height,
        ship,
    };
    let (mut input, menu, flight) = input::read(
        &keys,
        &mouse,
        window.as_deref(),
        &pads,
        view,
        &mut g.device,
        &mut g.latch,
    );
    g.flight = flight;
    if g.bot {
        (input, g.flight) = bot::input(&g.run, g.time);
    }
    if menu.debug {
        g.debug = !g.debug;
    }
    if menu.fullscreen {
        g.settings.fullscreen = !g.settings.fullscreen;
        g.store.save_settings(&g.settings);
    }
    if let Some(w) = window.as_mut() {
        let full = !matches!(w.mode, WindowMode::Windowed);
        if full != g.settings.fullscreen {
            w.mode = if g.settings.fullscreen {
                WindowMode::BorderlessFullscreen(MonitorSelection::Current)
            } else {
                WindowMode::Windowed
            };
        }
    }
    let aspect = window
        .as_ref()
        .map_or(16.0 / 9.0, |w| w.width() / w.height().max(1.0));
    if frame::advance(g, input, menu, dt, aspect, &mut sound) == Quit::Yes {
        exit.write(AppExit::Success);
    }
}
