//! warp: a neon twin-stick shooter on warping space, and a real-load test of gax.
//!
//! Bevy is plumbing only (app, ECS, windows, input, gamepads, time); everything visual is the
//! game's own wgpu renderer with WESL shaders on gax's generated modules, and the audio is the
//! game's own synthesizer on Firewheel through bevy_seedling. The simulation (`sim`) is a pure
//! core; all geometry is gax.

mod audio;
mod fx;
mod geom;
mod headless;
mod input;
// The kernels run traced (below); the game uses the generic forms only for `colour_map`, the
// tests for all of them.
#[cfg_attr(not(test), allow(dead_code))]
mod kernels;
mod light;
mod render;
mod signal;
mod sim;
mod store;
mod tunnel;

// The traced kernels (`grid_node`, `source_force`, `particle_step`), their batch forms, and
// their WGSL (`FUSED_WESL`).
include!(concat!(env!("OUT_DIR"), "/fused.rs"));

use bevy::prelude::*;
use bevy::window::{PrimaryWindow, RawHandleWrapper, WindowMode, WindowResized};
use fx::Fx;
use render::font::Align;
use render::scene::{self, palette};
use render::{GridSpec, LineInstance, Renderer};
use sim::replay::{Packed, Replay};
use sim::{ARENA, DT, Phase, World as Sim};
use std::time::Instant;
use store::{Scores, Settings, Store};
use tunnel::replay::Packed as TunnelPacked;

/// Where the game is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Screen {
    Title,
    Settings,
    Scores,
    Playing,
    Paused,
    /// Entering initials for the high-score table.
    Initials,
    Over,
    /// Watching a replay (or a demo, in the attract mode).
    Watch,
}

/// Seconds on the title before the attract mode starts a demo.
const ATTRACT_AFTER: f32 = 20.0;
/// A demo plays this long at most.
const DEMO_LENGTH: f32 = 75.0;

const TITLE_ITEMS: [&str; 5] = ["PLANE", "TUNNEL", "HIGH SCORES", "SETTINGS", "QUIT"];

/// Which game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Plane,
    Tunnel,
}

/// A Tunnel run: the simulation, its view, the input waiting for the next tick, and its
/// recording.
struct TunnelRun {
    world: tunnel::World,
    view: render::tunnel::View,
    input: tunnel::Input,
    rec: Replay<TunnelPacked>,
}
const PAUSE_ITEMS: [&str; 2] = ["RESUME", "END RUN"];

/// A recording of either game.
enum Watched {
    Plane(Replay),
    Tunnel(Replay<TunnelPacked>),
}

impl Watched {
    fn seconds(&self) -> f32 {
        match self {
            Watched::Plane(r) => r.seconds(),
            Watched::Tunnel(r) => r.seconds(),
        }
    }
}

/// A replay being watched.
struct Watch {
    replay: Watched,
    next: usize,
    /// Set when the playback ended or diverged.
    end: Option<String>,
    /// Ticks per tick (fast forward).
    speed: u32,
    /// Recorded on another build (it may diverge).
    other_build: bool,
    /// Part of the attract mode: any input goes back to the title.
    demo: bool,
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
    latch: input::Latch,
    cam: sim::body::Pose,
    half_height: f32,
    time: f32,
    debug: bool,
    best: u64,
    /// The Tunnel's best.
    tunnel_best: u64,
    sim_ms: f32,
    fps: f32,
    over_timer: f32,
    /// A bot plays (the `--smoke` run).
    bot: bool,
    store: Store,
    settings: Settings,
    scores: Scores,
    tunnel_scores: Scores,
    /// Which table the scores screen shows.
    table: Mode,
    /// Seconds on the title without input (the attract mode's clock).
    idle: f32,
    /// A demo is playing (the attract mode); which game the next one plays.
    demo: bool,
    demo_turn: Mode,
    /// The menu cursor.
    sel: usize,
    /// The run being recorded.
    rec: Replay,
    watch: Option<Watch>,
    /// Initials being entered, and the cursor in them.
    initials: [u8; 3],
    cursor: usize,
    /// The table row to highlight (the run just entered).
    highlight: Option<usize>,
    mode: Mode,
    tunnel: Option<Box<TunnelRun>>,
    /// The Tunnel's extra controls this frame.
    flight: input::Flight,
    world_lines: Vec<LineInstance>,
    hud_lines: Vec<LineInstance>,
}

impl Game {
    fn new(store: Store) -> Game {
        let settings = store.settings();
        let scores = store.scores::<Packed>();
        let tunnel_scores = store.scores::<TunnelPacked>();
        let mut g = Game {
            sim: Sim::new(1),
            fx: Fx::new(),
            screen: Screen::Title,
            acc: 0.0,
            alpha: 0.0,
            input: sim::Input::default(),
            device: input::Device::Mouse,
            latch: input::Latch::default(),
            cam: sim::body::pose_at(0.0, 0.0, 0.0),
            half_height: 20.0,
            time: 0.0,
            debug: false,
            best: scores.entries.first().map_or(0, |e| e.score),
            tunnel_best: tunnel_scores.entries.first().map_or(0, |e| e.score),
            sim_ms: 0.0,
            fps: 60.0,
            over_timer: 0.0,
            bot: false,
            store,
            settings,
            scores,
            tunnel_scores,
            table: Mode::Plane,
            idle: 0.0,
            demo: false,
            demo_turn: Mode::Tunnel,
            sel: 0,
            rec: Replay::new(1),
            watch: None,
            initials: *b"AAA",
            cursor: 0,
            highlight: None,
            mode: Mode::Plane,
            tunnel: None,
            flight: input::Flight::default(),
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
            let kind = match k % 5 {
                0 => sim::Kind::Chaser,
                2 => sim::Kind::Evader,
                4 if k > 8 => sim::Kind::Splitter,
                _ => sim::Kind::Drifter,
            };
            self.sim.spawn(kind, attract_spot(a, 10.0 + k as f32, 9.0));
        }
        self.sim.spawn(sim::Kind::Singularity, sim::at(12.0, -4.0));
    }

    /// Start a run.
    fn start(&mut self) {
        let seed = u64::from(std::process::id()) ^ (self.time.to_bits() as u64) << 20 ^ 0x5eed;
        self.start_seeded(seed);
    }

    /// Start a Tunnel run.
    fn start_tunnel(&mut self) {
        let seed = u64::from(std::process::id()) ^ (self.time.to_bits() as u64) << 20 ^ 0x70e1;
        self.start_tunnel_seeded(seed);
    }

    /// Start a Tunnel run from `seed`.
    fn start_tunnel_seeded(&mut self, seed: u64) {
        self.tunnel = Some(Box::new(self.tunnel_run(seed)));
        self.mode = Mode::Tunnel;
        self.screen = Screen::Playing;
        self.acc = 0.0;
        self.highlight = None;
    }

    /// A fresh Tunnel run from `seed`, its view set up by the settings.
    fn tunnel_run(&self, seed: u64) -> TunnelRun {
        let mut view = render::tunnel::View::new(f32::from(self.settings.fov));
        view.follow_roll = self.settings.camera_roll;
        TunnelRun {
            world: tunnel::World::new(seed),
            view,
            input: tunnel::Input::default(),
            rec: Replay::new(seed),
        }
    }

    /// Back to the title (from either mode).
    fn back_to_title(&mut self) {
        self.mode = Mode::Plane;
        self.tunnel = None;
        self.watch = None;
        self.demo = false;
        self.bot = false;
        self.idle = 0.0;
        self.screen = Screen::Title;
        self.attract();
    }

    /// The attract mode: the best run of one game or the other, replayed; or, with no table
    /// yet, the bot playing.
    fn start_demo(&mut self) {
        let mode = self.demo_turn;
        self.demo_turn = if mode == Mode::Plane {
            Mode::Tunnel
        } else {
            Mode::Plane
        };
        self.demo = true;
        self.idle = 0.0;
        let best = |s: &Scores| s.entries.first().map(|e| e.replay.clone());
        match mode {
            Mode::Plane => {
                if let Some(r) = best(&self.scores).and_then(|f| self.store.replay::<Packed>(&f)) {
                    self.watch_plane(r);
                } else {
                    self.start_seeded(0xde70);
                    self.bot = true;
                }
            }
            Mode::Tunnel => {
                if let Some(r) =
                    best(&self.tunnel_scores).and_then(|f| self.store.replay::<TunnelPacked>(&f))
                {
                    self.watch_tunnel(r);
                } else {
                    self.start_tunnel_seeded(0xde71);
                    self.bot = true;
                }
            }
        }
        if let Some(w) = self.watch.as_mut() {
            w.demo = true;
        }
    }

    /// The Tunnel run is over (lost, or ended from the pause menu): keep its replay, and ask
    /// for initials if it makes the Tunnel's table.
    fn end_tunnel_run(&mut self) {
        let Some(run) = self.tunnel.as_mut() else {
            return;
        };
        run.world.phase = Phase::Over;
        run.rec.score = run.world.score;
        let score = run.world.score;
        self.tunnel_best = self.tunnel_best.max(score);
        self.over_timer = 0.0;
        if self.demo {
            self.back_to_title();
        } else if !self.bot && self.tunnel_scores.rank(score).is_some() {
            self.screen = Screen::Initials;
            self.cursor = 0;
        } else {
            self.store.finish(&run.rec, None);
            self.screen = Screen::Over;
        }
    }

    /// Start a run from `seed`.
    fn start_seeded(&mut self, seed: u64) {
        self.mode = Mode::Plane;
        self.sim = Sim::new(seed);
        self.rec = Replay::new(seed);
        self.screen = Screen::Playing;
        self.acc = 0.0;
        self.highlight = None;
    }

    /// The run is over (lost, or ended from the pause menu): keep its replay, and ask for
    /// initials if it makes the table.
    fn end_run(&mut self) {
        self.rec.score = self.sim.score;
        self.best = self.best.max(self.sim.score);
        self.sim.phase = Phase::Over;
        self.over_timer = 0.0;
        if self.demo {
            self.back_to_title();
        } else if !self.bot && self.scores.rank(self.sim.score).is_some() {
            self.screen = Screen::Initials;
            self.cursor = 0;
        } else {
            self.store.finish(&self.rec, None);
            self.screen = Screen::Over;
        }
    }

    /// Watch a Plane replay.
    fn watch_plane(&mut self, replay: Replay) {
        self.mode = Mode::Plane;
        self.tunnel = None;
        self.sim = Sim::new(replay.seed);
        self.watch = Some(Watch {
            end: None,
            other_build: replay.build != sim::replay::build_id(),
            replay: Watched::Plane(replay),
            next: 0,
            speed: 1,
            demo: false,
        });
        self.screen = Screen::Watch;
        self.acc = 0.0;
    }

    /// Watch a Tunnel replay.
    fn watch_tunnel(&mut self, replay: Replay<TunnelPacked>) {
        self.tunnel = Some(Box::new(self.tunnel_run(replay.seed)));
        self.mode = Mode::Tunnel;
        self.watch = Some(Watch {
            end: None,
            other_build: replay.build != sim::replay::build_id(),
            replay: Watched::Tunnel(replay),
            next: 0,
            speed: 1,
            demo: false,
        });
        self.screen = Screen::Watch;
        self.acc = 0.0;
    }

    /// Apply the settings to effects, colours and sound.
    fn apply_settings(&mut self, sound: &mut audio::Sound) {
        let s = &self.settings;
        self.fx.shake_scale = f32::from(s.shake) / 4.0;
        self.fx.flash_scale = if s.reduced_flashes { 0.3 } else { 1.0 };
        scene::set_scheme(s.scheme);
        if s.reduced_motion {
            self.fx.shake_scale = 0.0;
        }
        self.fx.reduced_motion = s.reduced_motion;
        if let Some(run) = self.tunnel.as_mut() {
            run.view.shake_scale = if s.reduced_motion {
                0.0
            } else {
                f32::from(s.shake) / 4.0
            };
            run.view.flash_scale = if s.reduced_flashes { 0.3 } else { 1.0 };
            run.view.focal = render::tunnel::View::focal_for(f32::from(s.fov));
            run.view.follow_roll = s.camera_roll;
            run.view.assist = s.aim_reach();
            run.view.lensing = !s.reduced_motion;
        }
        sound.gains = s.gains();
        sound.on_beat = s.on_beat;
    }
}

/// A spot on an ellipse around the title: the unit heading at `angle`, stretched to the
/// half axes `rx` and `ry`.
fn attract_spot(angle: f32, rx: f32, ry: f32) -> sim::P {
    let h = sim::body::heading(angle, 1.0);
    sim::at(h.e20() * rx, h.e01() * ry)
}

/// Move a menu cursor over `n` items.
fn nav(sel: usize, n: usize, menu: &input::Menu) -> usize {
    let n = n.max(1);
    if menu.up {
        (sel + n - 1) % n
    } else if menu.down {
        (sel + 1) % n
    } else {
        sel.min(n - 1)
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
    if let Some(i) = args.iter().position(|a| a == "--verify") {
        // Play a replay headless and check every state hash.
        let Some(path) = args.get(i + 1) else {
            eprintln!("warp: --verify FILE");
            std::process::exit(2);
        };
        let r = load_any(path);
        let (build, seconds, hashes) = match &r {
            Watched::Plane(r) => (r.build.clone(), r.seconds(), r.hashes.len()),
            Watched::Tunnel(r) => (r.build.clone(), r.seconds(), r.hashes.len()),
        };
        if build != sim::replay::build_id() {
            eprintln!(
                "note: recorded on build {build}, this is {}",
                sim::replay::build_id()
            );
        }
        let result = match &r {
            Watched::Plane(r) => r.verify().map(|w| w.score),
            Watched::Tunnel(r) => r.verify().map(|w| w.score),
        };
        match result {
            Ok(score) => println!("ok: {seconds:.0} s, score {score}, {hashes} hashes matched"),
            Err(d) => {
                println!("diverged: {d:?}");
                std::process::exit(1);
            }
        }
        return;
    }
    let watch = match args.iter().position(|a| a == "--replay") {
        Some(i) => {
            let Some(path) = args.get(i + 1) else {
                eprintln!("warp: --replay FILE");
                std::process::exit(2);
            };
            Some(load_any(path))
        }
        None => None,
    };
    let known = ["--smoke", "--shot", "--music", "--replay", "--verify"];
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
    // The smoke run keeps nothing: no settings, scores or replays are read or written.
    let mut game = Game::new(if smoke {
        Store::new(None)
    } else {
        Store::open()
    });
    let fullscreen = game.settings.fullscreen;
    match watch {
        Some(Watched::Plane(r)) => game.watch_plane(r),
        Some(Watched::Tunnel(r)) => game.watch_tunnel(r),
        None => {}
    }
    app.insert_resource(game)
        .add_systems(Update, (update, draw).chain());
    if fullscreen {
        app.add_systems(Startup, |mut w: Query<&mut Window, With<PrimaryWindow>>| {
            if let Ok(mut w) = w.single_mut() {
                w.mode = WindowMode::BorderlessFullscreen(bevy::window::MonitorSelection::Current);
            }
        });
    }
    if smoke {
        // `--smoke`: play a few seconds with the bot, then quit (a startup test).
        app.insert_resource(Smoke { frames: 0 })
            .add_systems(Update, smoke_test.before(update));
    }
    app.run();
}

/// A replay file of either game (told apart by its magic), or exit.
fn load_any(path: &str) -> Watched {
    let path = std::path::Path::new(path);
    if let Ok(r) = store::load_replay::<Packed>(path) {
        return Watched::Plane(r);
    }
    match store::load_replay::<TunnelPacked>(path) {
        Ok(r) => Watched::Tunnel(r),
        Err(e) => {
            eprintln!("warp: {e}");
            std::process::exit(2);
        }
    }
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
    let (mut input, menu, flight) = input::read(
        &keys,
        &mouse,
        window.as_deref(),
        &pads,
        g.cam,
        g.half_height,
        g.sim.ship.body.pos(),
        &mut g.device,
        &mut g.latch,
    );
    g.flight = flight;
    if g.bot {
        if g.mode == Mode::Tunnel && g.tunnel.is_some() {
            let (i, f) = headless::tunnel_bot(g, g.time);
            input = i;
            g.flight = f;
        } else {
            input = headless::bot_input(g, g.time);
        }
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
                WindowMode::BorderlessFullscreen(bevy::window::MonitorSelection::Current)
            } else {
                WindowMode::Windowed
            };
        }
    }
    let aspect = window
        .as_ref()
        .map_or(16.0 / 9.0, |w| w.width() / w.height().max(1.0));
    if advance(g, input, menu, dt, aspect, &mut sound) {
        exit.write(AppExit::Success);
    }
}

/// One simulation tick: the player's input (recorded), a replay's, or none.
fn tick(g: &mut Game) {
    match g.screen {
        Screen::Playing => {
            // The simulation sees the quantized input, exactly what the replay stores.
            let p = Packed::pack(&g.input);
            g.rec.record(p);
            g.sim.tick(&p.unpack());
            g.rec.after(&g.sim);
            g.input.bomb = false;
        }
        Screen::Watch => {
            let w = g.watch.as_mut().expect("watching");
            let Watched::Plane(replay) = &w.replay else {
                return;
            };
            match replay.inputs.get(w.next) {
                Some(p) if w.end.is_none() => {
                    w.next += 1;
                    g.sim.tick(&p.unpack());
                    if let Err(d) = replay.check(&g.sim) {
                        let t = match d {
                            sim::replay::Desync::Hash(t) => t,
                            sim::replay::Desync::Score { .. } => g.sim.tick,
                        };
                        w.end = Some(format!("DIVERGED AT {}", clock(t as f32 * DT)));
                        w.speed = 1;
                    }
                }
                _ => {
                    if w.end.is_none() {
                        // The run's last tick: its score must be the recorded one.
                        w.end = Some(match replay.final_score(g.sim.score) {
                            Ok(()) => "END OF REPLAY".into(),
                            Err(_) => "DIVERGED AT THE END".into(),
                        });
                        w.speed = 1;
                    }
                    g.sim.tick(&sim::Input::default());
                }
            }
        }
        _ => g.sim.tick(&sim::Input::default()),
    }
}

/// `m:ss`.
fn clock(seconds: f32) -> String {
    let s = seconds.max(0.0) as u32;
    format!("{}:{:02}", s / 60, s % 60)
}

/// Menus: moving between screens, settings, initials.
fn menu_step(g: &mut Game, menu: input::Menu, dt: f32, quit: &mut bool) {
    let any = menu.start || menu.back || menu.up || menu.down || menu.left || menu.right;
    if g.demo {
        // The attract mode: any input, the end of the demo, or its time, and back to the title.
        g.idle += dt;
        let ended = g.watch.as_ref().is_some_and(|w| w.end.is_some())
            || matches!(g.screen, Screen::Over | Screen::Title);
        if any || ended || g.idle > DEMO_LENGTH {
            g.back_to_title();
        }
        return;
    }
    match g.screen {
        Screen::Title => {
            g.idle = if any { 0.0 } else { g.idle + dt };
            if g.idle > ATTRACT_AFTER {
                g.start_demo();
                return;
            }
            g.sel = nav(g.sel, TITLE_ITEMS.len(), &menu);
            if menu.start {
                match g.sel {
                    0 => g.start(),
                    1 => g.start_tunnel(),
                    2 => {
                        g.screen = Screen::Scores;
                        g.sel = 0;
                    }
                    3 => {
                        g.screen = Screen::Settings;
                        g.sel = 0;
                    }
                    _ => *quit = true,
                }
            } else if menu.back {
                *quit = true;
            }
        }
        Screen::Settings => {
            let rows = Settings::ROWS.len() + 1; // and BACK
            g.sel = nav(g.sel, rows, &menu);
            let mut by = i32::from(menu.right) - i32::from(menu.left);
            if by == 0 && menu.start && Settings::toggles(g.sel) {
                by = 1;
            }
            if by != 0 && g.sel < Settings::ROWS.len() {
                g.settings.adjust(g.sel, by);
                g.store.save_settings(&g.settings);
            }
            if menu.back || (menu.start && g.sel == rows - 1) {
                g.screen = Screen::Title;
                g.sel = 3;
            }
        }
        Screen::Scores => {
            if menu.left || menu.right {
                g.table = if g.table == Mode::Plane {
                    Mode::Tunnel
                } else {
                    Mode::Plane
                };
                g.sel = 0;
                g.highlight = None;
            }
            let table = if g.table == Mode::Plane {
                &g.scores
            } else {
                &g.tunnel_scores
            };
            g.sel = nav(g.sel, table.entries.len(), &menu);
            let file = table.entries.get(g.sel).map(|e| e.replay.clone());
            if menu.start
                && let Some(f) = file
            {
                match g.table {
                    Mode::Plane => {
                        if let Some(r) = g.store.replay::<Packed>(&f) {
                            g.watch_plane(r);
                        }
                    }
                    Mode::Tunnel => {
                        if let Some(r) = g.store.replay::<TunnelPacked>(&f) {
                            g.watch_tunnel(r);
                        }
                    }
                }
            } else if menu.back {
                g.screen = Screen::Title;
                g.sel = 2;
            }
        }
        Screen::Playing if menu.back => {
            g.screen = Screen::Paused;
            g.sel = 0;
        }
        Screen::Paused => {
            g.sel = nav(g.sel, PAUSE_ITEMS.len(), &menu);
            if menu.back || (menu.start && g.sel == 0) {
                g.screen = Screen::Playing;
            } else if menu.start {
                if g.mode == Mode::Tunnel {
                    g.end_tunnel_run();
                } else {
                    g.end_run();
                }
            }
        }
        Screen::Initials => {
            // Typing letters, or choosing them with up and down (the keyboard's WASD type).
            if let Some(c) = menu.letter {
                if g.cursor < 3 {
                    g.initials[g.cursor] = c as u8;
                    g.cursor += 1;
                }
            } else if g.cursor < 3 && (menu.up || menu.down) {
                let l = &mut g.initials[g.cursor];
                let k = (*l - b'A') as i32 + if menu.up { 1 } else { -1 };
                *l = b'A' + k.rem_euclid(26) as u8;
            } else if menu.right {
                g.cursor = (g.cursor + 1).min(3);
            } else if menu.left {
                g.cursor = g.cursor.saturating_sub(1);
            }
            if menu.erase {
                g.cursor = g.cursor.saturating_sub(1);
            }
            if menu.start {
                if g.cursor < 2 && menu.letter.is_none() && g.device == input::Device::Pad {
                    // On a pad, A confirms a letter and moves on.
                    g.cursor += 1;
                } else {
                    let name = String::from_utf8_lossy(&g.initials).into_owned();
                    g.table = g.mode;
                    match (g.mode, g.tunnel.take()) {
                        (Mode::Tunnel, Some(run)) => {
                            g.highlight = g.store.finish(&run.rec, Some(&name));
                            g.tunnel_scores = g.store.scores::<TunnelPacked>();
                        }
                        _ => {
                            g.highlight = g.store.finish(&g.rec, Some(&name));
                            g.scores = g.store.scores::<Packed>();
                            g.best = g
                                .scores
                                .entries
                                .first()
                                .map_or(g.best, |e| e.score.max(g.best));
                        }
                    }
                    g.mode = Mode::Plane;
                    g.sel = g.highlight.unwrap_or(0);
                    g.screen = Screen::Scores;
                    g.attract();
                }
            }
        }
        Screen::Over => {
            g.over_timer += dt;
            if g.over_timer > 1.5 && (menu.start || menu.back) {
                g.over_timer = 0.0;
                g.sel = if g.mode == Mode::Tunnel { 1 } else { 0 };
                g.back_to_title();
            }
        }
        Screen::Watch => {
            let ended = g.watch.as_ref().is_some_and(|w| w.end.is_some());
            if menu.back || (ended && menu.start) {
                g.watch = None;
                g.table = g.mode;
                g.mode = Mode::Plane;
                g.tunnel = None;
                g.screen = Screen::Scores;
                g.attract();
            } else if (menu.right || menu.left)
                && let Some(w) = g.watch.as_mut()
                && w.end.is_none()
            {
                w.speed = if menu.right {
                    (w.speed * 2).min(8)
                } else {
                    (w.speed / 2).max(1)
                };
            }
        }
        _ => {}
    }
}

/// One frame of the Tunnel: controls through the camera, the fixed-step simulation (recorded,
/// or played back from a replay), its view, and sound.
fn advance_tunnel(g: &mut Game, input: sim::Input, dt: f32, sound: &mut audio::Sound) {
    use gax::pga2d::Point as Point2;
    let flight = g.flight;
    let run = g.tunnel.as_mut().expect("a tunnel run");
    if g.screen == Screen::Watch {
        // The replay aims: show where.
        run.view.follow(&run.world, run.input.aim);
    } else {
        // Screen directions become directions across the tunnel, through the camera.
        let movement = run.view.across(&run.world, input.movement);
        // The reticle: the cursor, or the right stick pushing it out from the ship on screen.
        let reticle = match (flight.stick, flight.cursor) {
            (Some(stick), _) => run.view.ship_on_screen(&run.world) + stick * 9.0,
            (None, Some(cursor)) => cursor,
            (None, None) => run.view.ship_on_screen(&run.world) + Point2::direction(0.0, 4.0),
        };
        let aim = run.view.aim(&run.world, reticle);
        // Edges wait for the tick that consumes them.
        run.input = tunnel::Input {
            movement,
            aim,
            fire: input.fire,
            bomb: run.input.bomb || flight.bomb,
            roll: if run.input.roll != 0 {
                run.input.roll
            } else {
                flight.roll
            },
            throttle: flight.throttle,
        };
    }
    let mut over = false;
    if g.screen != Screen::Paused {
        let speed = g.watch.as_ref().map_or(1, |w| w.speed);
        g.acc += dt * speed as f32;
        while g.acc >= DT {
            g.acc -= DT;
            match g.screen {
                Screen::Playing => {
                    // The world sees the packed input, exactly what the replay keeps.
                    let i = run.rec.take(&run.input, &run.world);
                    run.world.tick(&i);
                    run.rec.after(&run.world);
                    run.input.bomb = false;
                    run.input.roll = 0;
                }
                Screen::Watch => {
                    let w = g.watch.as_mut().expect("watching");
                    let Watched::Tunnel(replay) = &w.replay else {
                        break;
                    };
                    match replay.inputs.get(w.next) {
                        Some(p) if w.end.is_none() => {
                            w.next += 1;
                            run.input = p.unpack(run.world.ship.s());
                            run.world.tick(&run.input);
                            if let Err(d) = replay.check(&run.world) {
                                let t = match d {
                                    sim::replay::Desync::Hash(t) => t,
                                    sim::replay::Desync::Score { .. } => run.world.tick,
                                };
                                w.end = Some(format!("DIVERGED AT {}", clock(t as f32 * DT)));
                                w.speed = 1;
                            }
                        }
                        _ => {
                            // The run's last tick: its score must be the recorded one. The
                            // world then holds still (flying on, it would pass gates and
                            // score without the player).
                            if w.end.is_none() {
                                w.end = Some(match replay.final_score(run.world.score) {
                                    Ok(()) => "END OF REPLAY".into(),
                                    Err(_) => "DIVERGED AT THE END".into(),
                                });
                                w.speed = 1;
                            }
                            g.acc = 0.0;
                            break;
                        }
                    }
                }
                _ => run.world.tick(&tunnel::Input::default()),
            }
            run.view.on_events(&run.world.events);
            let heard = tunnel_sounds(&run.world, &run.view);
            let origin = sim::at(0.0, 0.0);
            sound.play(&heard, run.world.mult, origin, sim::body::identity());
            if g.screen == Screen::Playing && run.world.events.contains(&tunnel::Event::GameOver) {
                over = true;
                break;
            }
        }
        g.alpha = g.acc / DT;
    }
    run.view.update(&run.world, g.alpha, dt);
    let alive = run.world.phase == Phase::Playing;
    let intensity = run.world.intensity * if alive { 1.0 } else { 0.5 };
    sound.steer(
        run.world.seed,
        audio::music::TUNNEL_TEMPO,
        intensity,
        run.world.darkness(),
        run.world.mult,
        matches!(g.screen, Screen::Playing | Screen::Watch),
    );
    if over {
        g.end_tunnel_run();
    }
}

/// The Tunnel's events as the sound hears them: each position taken into the camera's frame
/// (`cam << p`, the listener), where across is the pan and the depth the distance.
fn tunnel_sounds(w: &tunnel::World, view: &render::tunnel::View) -> Vec<sim::Event> {
    use tunnel::Event as T;
    let at = |p: tunnel::P| view.listen(w.track.place(p));
    w.events
        .iter()
        .filter_map(|e| {
            Some(match *e {
                T::Fire { pos } => sim::Event::Fire {
                    pos: at(pos),
                    dir: sim::dir(0.0, 1.0),
                },
                T::Hit { pos, foe } => sim::Event::Hit {
                    pos: at(pos),
                    kind: foe.kind(),
                },
                T::Kill { pos, foe, points } => sim::Event::Kill {
                    pos: at(pos),
                    kind: foe.kind(),
                    size: match foe {
                        tunnel::Foe::Turret | tunnel::Foe::Serpent => 2.0,
                        tunnel::Foe::Singularity => 3.0,
                        _ => 1.0,
                    },
                    scored: points > 0,
                    points,
                    wreck: None,
                },
                T::Bolt { pos } => sim::Event::Deflect {
                    pos: at(pos),
                    kind: sim::Kind::Warden,
                },
                T::Wall { pos } => sim::Event::Wall { pos: at(pos) },
                T::Warn { pos, foe } => sim::Event::Warn {
                    pos: at(pos),
                    kind: foe.kind(),
                },
                T::Pickup { pos, mult } => sim::Event::Pickup { pos: at(pos), mult },
                T::Death { pos } => sim::Event::Death { pos: at(pos) },
                T::Bomb { pos } => sim::Event::Bomb { pos: at(pos) },
                T::Respawn => sim::Event::Respawn,
                T::Extra { life } => sim::Event::Extra { life },
                T::GameOver => sim::Event::GameOver,
                T::Dive { pos } => sim::Event::Warn {
                    pos: at(pos),
                    kind: tunnel::Foe::Drone.kind(),
                },
                T::Gate { pos, chain, .. } => sim::Event::Pickup {
                    pos: at(pos),
                    mult: chain * 2,
                },
                T::GateMiss { pos } => sim::Event::Wall { pos: at(pos) },
                T::Absorb { pos, mass } => sim::Event::Absorb { pos: at(pos), mass },
                T::Burst { pos } => sim::Event::Burst { pos: at(pos) },
                T::Slingshot { .. } => sim::Event::Extra { life: false },
                T::Roll { .. } => return None,
            })
        })
        .collect()
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
    menu_step(g, menu, dt, &mut quit);
    g.apply_settings(sound);
    if g.mode == Mode::Tunnel && g.tunnel.is_some() {
        advance_tunnel(g, input, dt, sound);
        return quit;
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
        let speed = g.watch.as_ref().map_or(1, |w| w.speed);
        g.acc += dt * speed as f32;
        while g.acc >= DT {
            g.acc -= DT;
            if g.sim.hitstop > 0.0 {
                // Hit-stop: the world holds its breath; effects go on.
                g.sim.hitstop = (g.sim.hitstop - DT * speed as f32).max(0.0);
            } else {
                tick(g);
                g.fx.on_events(&g.sim.events);
                sound.on_events(&g.sim.events, &g.sim, g.cam);
                if g.screen == Screen::Playing && g.sim.events.contains(&sim::Event::GameOver) {
                    g.end_run();
                }
            }
            if g.fx.grid_steps.len() < 8 {
                g.fx.grid_tick(&g.sim, DT);
            }
        }
        g.alpha = g.acc / DT;
    }
    g.sim_ms = g.sim_ms * 0.9 + 0.1 * started.elapsed().as_secs_f32() * 1e3;
    let run = matches!(g.screen, Screen::Playing | Screen::Watch);
    sound.update(&g.sim, run, g.cam);
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
    if game.mode == Mode::Tunnel
        && let Some(run) = game.tunnel.as_mut()
    {
        // Everything is projected by hand into screen units: the HUD's camera draws it all.
        let hud_cam = scene::camera(sim::body::pose_at(0.0, 0.0, 0.0), 18.0, size, game.time);
        run.view
            .draw(&run.world, game.alpha, game.time, &mut game.world_lines);
        if game.screen == Screen::Paused {
            for l in &mut game.world_lines {
                l.recede(0.2);
            }
        }
        let post = run
            .view
            .post(&run.world, size[0] as f32 / size[1].max(1) as f32);
        hud(game, size, renderer.timings());
        let f = render::Frame {
            camera: hud_cam,
            hud: hud_cam,
            centre: crate::geom::ORIGIN,
            world: &game.world_lines,
            hud_lines: &game.hud_lines,
            grid_steps: &[],
            wells: &[],
            spawn: &[],
            grid_color: light::DARK,
            post,
            plane: false,
        };
        renderer.render(view, &f);
        return;
    }
    let camera = scene::camera(game.cam, game.half_height, size, game.time);
    let hud_cam = scene::camera(sim::body::pose_at(0.0, 0.0, 0.0), 18.0, size, game.time);
    scene::world_lines(&game.sim, game.alpha, game.time, &mut game.world_lines);
    for p in &game.fx.wreckage {
        let c = light::fade(p.color, p.life / fx::WRECK_LIFE);
        scene::outline(
            &mut game.world_lines,
            &p.tri,
            1.0,
            c,
            scene::THIN,
            p.body.pose,
        );
    }
    for (p, text, age, color) in &game.fx.popups {
        // Rising and fading.
        let c = light::fade(*color, (1.0 - age / 1.1).max(0.0));
        let at = *p + gax::pga2d::Point::direction(0.0, 0.8 + age * 1.5);
        scene::text(&mut game.world_lines, text, at, 0.9, c, Align::Center);
    }
    // Menus over the world: the world steps back.
    let backdrop = matches!(
        game.screen,
        Screen::Settings | Screen::Scores | Screen::Paused | Screen::Initials
    );
    if backdrop {
        for l in &mut game.world_lines {
            l.recede(0.2);
        }
    }
    hud(game, size, renderer.timings());
    let wells = Fx::particle_wells(&game.sim);
    let post = game
        .fx
        .post(&scene::view_map(game.cam, game.half_height, size));
    let intensity = if game.screen == Screen::Playing {
        game.sim.director.intensity
    } else {
        0.3
    };
    let dim = if backdrop { 0.5 } else { 1.0 };
    // The lattice warms towards violet as the intensity rises: its hue turned about OkLab's
    // lightness axis.
    let grid_color = light::fade(
        light::hue_shift(palette::GRID, 0.5 * intensity),
        (0.85 + 0.5 * intensity) * dim,
    );
    let f = render::Frame {
        camera,
        hud: hud_cam,
        centre: game.cam >> crate::geom::ORIGIN,
        world: &game.world_lines,
        hud_lines: &game.hud_lines,
        grid_steps: &game.fx.grid_steps,
        wells: &wells,
        spawn: &game.fx.spawn,
        grid_color,
        post,
        plane: true,
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
        origin: gax::pga2d::Point::xy(-ARENA[0], -ARENA[1]),
        spacing: 2.0 * ARENA[0] / 128.0,
    }
}

/// The HUD, in HUD units (the screen is 36 units tall, centred).
fn hud(g: &mut Game, size: [u32; 2], timings: Option<render::Timings>) {
    let stats = Stats::of(g);
    // The lines are built apart from the game, which the screens read, then put back.
    let mut out = std::mem::take(&mut g.hud_lines);
    out.clear();
    let aspect = size[0] as f32 / size[1] as f32;
    let (left, right, top) = (-18.0 * aspect + 2.2, 18.0 * aspect - 2.2, 18.0 - 2.6);
    let hud = palette::HUD;
    let dim = light::fade(hud, 0.45);
    let blink = signal::wave(g.time * 3.0) > -0.3;
    // Menu rows: the selected one bright and pulsing, the others faint.
    let hot = light::fade(hud, 1.1 + 0.2 * signal::wave(g.time * 5.0));
    let faint = light::fade(hud, 0.3);
    let st = HudStyle {
        left,
        right,
        top,
        hud,
        dim,
        blink,
        hot,
        faint,
    };
    match g.screen {
        Screen::Title => hud_title(g, &mut out, st),
        Screen::Settings => hud_settings(g, &mut out, st),
        Screen::Scores => hud_scores(g, &mut out, st),
        Screen::Playing | Screen::Paused | Screen::Over | Screen::Initials | Screen::Watch => {
            hud_play(g, &stats, &mut out, st)
        }
    }
    // The HUD is drawn over the finished image (never bloomed): a tight glow keeps its text
    // sharp.
    for l in out.iter_mut() {
        l.style[1] = (l.style[1] * 0.45).max(0.06);
        l.style[2] *= 0.6;
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
                &mut out,
                l,
                scene::pt(left, -16.5 + 1.1 * (lines.len() - 1 - i) as f32),
                0.6,
                dim,
                Align::Left,
            );
        }
    }
    g.hud_lines = out;
}

/// The HUD's layout and colours, shared by its screens.
#[derive(Clone, Copy)]
struct HudStyle {
    left: f32,
    right: f32,
    top: f32,
    hud: light::Light,
    dim: light::Light,
    blink: bool,
    hot: light::Light,
    faint: light::Light,
}

/// The HUD of the title screen.
fn hud_title(g: &Game, out: &mut Vec<LineInstance>, st: HudStyle) {
    let HudStyle { dim, .. } = st;
    scene::text(
        out,
        "WARP",
        scene::pt(0.0, 7.0),
        6.5,
        light::light(0.55, 0.7, 1.0, 1.7),
        Align::Center,
    );
    scene::text(
        out,
        "A NEON SHOOTER ON WARPED SPACE",
        scene::pt(0.0, 3.4),
        1.0,
        dim,
        Align::Center,
    );
    menu_items(out, &TITLE_ITEMS, g.sel, -0.8, g.time);
    let controls = if g.sel == 1 {
        "MOVE WASD / L-STICK   AIM MOUSE / R-STICK   ROLL Q E / BUMPERS   BOOST SHIFT / RT   BRAKE CTRL / LT"
    } else {
        "MOVE WASD / LEFT STICK    AIM + FIRE MOUSE / RIGHT STICK    BOMB SPACE / TRIGGER"
    };
    scene::text(
        out,
        controls,
        scene::pt(0.0, -13.0),
        0.7,
        dim,
        Align::Center,
    );
    let best = match (g.best, g.tunnel_best) {
        (0, 0) => String::new(),
        (p, 0) => format!("BEST {}", scene::grouped(p)),
        (0, t) => format!("TUNNEL BEST {}", scene::grouped(t)),
        (p, t) => format!("BEST {}    TUNNEL {}", scene::grouped(p), scene::grouped(t)),
    };
    scene::text(out, &best, scene::pt(0.0, -15.0), 0.9, dim, Align::Center);
}

/// The HUD of the settings screen.
fn hud_settings(g: &Game, out: &mut Vec<LineInstance>, st: HudStyle) {
    let HudStyle {
        hud,
        dim,
        hot,
        faint,
        ..
    } = st;
    scene::text(
        out,
        "SETTINGS",
        scene::pt(0.0, 13.0),
        3.0,
        hud,
        Align::Center,
    );
    let (lx, rx) = (-15.0, 15.0);
    let row_y = |row: usize| 8.5 - row as f32 * 1.75;
    for (row, label) in Settings::ROWS.iter().enumerate() {
        let y = row_y(row);
        let on = row == g.sel;
        let c = if on { hot } else { faint };
        if on {
            scene::text(out, ">", scene::pt(lx - 2.0, y), 1.1, hud, Align::Left);
        }
        scene::text(out, label, scene::pt(lx, y), 1.1, c, Align::Left);
        match g.settings.value(row) {
            Ok(v) => {
                let v = if on { format!("< {v} >") } else { v };
                scene::text(out, &v, scene::pt(rx, y), 1.1, c, Align::Right);
            }
            Err(level) => {
                // Ten bars, lit up to the level.
                for k in 0..10 {
                    let x = rx - 9.5 * 0.9 + k as f32 * 0.9;
                    let lit = k < level;
                    let bc = light::fade(c, if lit { 1.2 } else { 0.25 });
                    out.push(scene::seg(
                        scene::pt(x, y + 0.15),
                        scene::pt(x, y + 0.35 + 0.1 * k as f32),
                        bc,
                        [0.09, 0.3, 0.2, 0.0],
                        sim::body::identity(),
                    ));
                }
            }
        }
    }
    let back = Settings::ROWS.len();
    let y = row_y(back) - 0.5;
    let c = if g.sel == back { hot } else { faint };
    if g.sel == back {
        scene::text(out, ">", scene::pt(lx - 2.0, y), 1.1, hud, Align::Left);
    }
    scene::text(out, "BACK", scene::pt(lx, y), 1.1, c, Align::Left);
    scene::text(
        out,
        "LEFT / RIGHT TO CHANGE    ESC / B TO GO BACK",
        scene::pt(0.0, -16.2),
        0.7,
        dim,
        Align::Center,
    );
}

/// The HUD of the high scores.
fn hud_scores(g: &Game, out: &mut Vec<LineInstance>, st: HudStyle) {
    let HudStyle {
        hud,
        dim,
        hot,
        faint,
        ..
    } = st;
    scene::text(
        out,
        "HIGH SCORES",
        scene::pt(0.0, 12.0),
        3.0,
        hud,
        Align::Center,
    );
    let (table, name) = if g.table == Mode::Plane {
        (&g.scores, "< PLANE >")
    } else {
        (&g.tunnel_scores, "< TUNNEL >")
    };
    scene::text(out, name, scene::pt(0.0, 9.0), 1.2, hot, Align::Center);
    if table.entries.is_empty() {
        scene::text(
            out,
            "NO RUNS YET",
            scene::pt(0.0, 2.0),
            1.4,
            dim,
            Align::Center,
        );
    }
    for (r, e) in table.entries.iter().enumerate() {
        let y = 6.0 - r as f32 * 1.75;
        let new = g.highlight == Some(r);
        let c = if new {
            scene::shard()
        } else if r == g.sel {
            hot
        } else {
            faint
        };
        if r == g.sel {
            scene::text(out, ">", scene::pt(-15.0, y), 1.0, c, Align::Left);
        }
        scene::text(
            out,
            &format!("{:>2}", r + 1),
            scene::pt(-13.0, y),
            1.0,
            c,
            Align::Left,
        );
        scene::text(out, &e.name, scene::pt(-9.0, y), 1.0, c, Align::Left);
        scene::text(
            out,
            &scene::grouped(e.score),
            scene::pt(7.0, y),
            1.0,
            c,
            Align::Right,
        );
        scene::text(
            out,
            &clock(e.seconds as f32),
            scene::pt(14.0, y),
            1.0,
            c,
            Align::Right,
        );
    }
    scene::text(
        out,
        "ENTER / A TO WATCH    LEFT / RIGHT FOR THE OTHER GAME    ESC / B TO GO BACK",
        scene::pt(0.0, -14.5),
        0.7,
        dim,
        Align::Center,
    );
}

/// The HUD of play: the score, lives and bombs, and the pause, game-over and initials overlays.
fn hud_play(g: &Game, stats: &Stats, out: &mut Vec<LineInstance>, st: HudStyle) {
    let HudStyle {
        left,
        right,
        top,
        hud,
        dim,
        blink,
        ..
    } = st;
    let s = &stats;
    scene::text(
        out,
        &scene::grouped(s.score),
        scene::pt(left, top - 0.2),
        1.3,
        hud,
        Align::Left,
    );
    scene::text(
        out,
        &format!("X{}", s.mult),
        scene::pt(left, top - 2.2),
        0.9,
        scene::shard(),
        Align::Left,
    );
    if s.chain > 0 {
        scene::text(
            out,
            &format!("GATES X{}", s.chain),
            scene::pt(left, top - 5.2),
            0.7,
            render::tunnel::GATE_LIGHT,
            Align::Left,
        );
    }
    if let Some(speed) = s.speed {
        // The speed bonus: boost for more points; pale gold turning to hot orange (a
        // perceptual gradient).
        let c = light::blend(
            light::light(1.0, 0.95, 0.55, 1.2),
            light::light(1.0, 0.45, 0.15, 2.4),
            ((speed - 1.0) / 0.45).clamp(0.0, 1.0),
        );
        scene::text(
            out,
            &format!("SPEED X{speed:.1}"),
            scene::pt(left, top - 3.8),
            0.7,
            c,
            Align::Left,
        );
    }
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
        let ring = light::light(0.55, 0.8, 1.0, 2.0);
        scene::circle(out, 0.45, 12, 0.0, ring, [0.04, 0.2, 0.2, 0.0], m);
    }
    if let Phase::Dead(_) = s.phase {
        scene::text(
            out,
            "SHIP LOST",
            scene::pt(0.0, 1.0),
            2.0,
            palette::SHIP,
            Align::Center,
        );
    }
    if g.screen == Screen::Paused {
        scene::text(out, "PAUSED", scene::pt(0.0, 4.0), 3.0, hud, Align::Center);
        menu_items(out, &PAUSE_ITEMS, g.sel, -0.5, g.time);
    }
    if g.screen == Screen::Initials {
        scene::text(
            out,
            "A NEW HIGH SCORE",
            scene::pt(0.0, 6.0),
            2.2,
            scene::shard(),
            Align::Center,
        );
        scene::text(
            out,
            &scene::grouped(s.score),
            scene::pt(0.0, 2.5),
            1.6,
            hud,
            Align::Center,
        );
        for k in 0..3 {
            let x = (k as f32 - 1.0) * 3.2;
            let on = k == g.cursor;
            let c = if on { hud } else { dim };
            let l = (g.initials[k] as char).to_string();
            scene::text(out, &l, scene::pt(x, -2.5), 2.4, c, Align::Center);
            if on && blink {
                scene::text(out, "_", scene::pt(x, -2.9), 2.4, hud, Align::Center);
            }
        }
        let msg = if g.cursor >= 3 {
            "ENTER TO KEEP"
        } else {
            "TYPE OR UP / DOWN    ENTER TO KEEP"
        };
        scene::text(out, msg, scene::pt(0.0, -6.5), 0.8, dim, Align::Center);
    }
    if g.demo {
        scene::text(
            out,
            "DEMO",
            scene::pt(0.0, top - 0.2),
            1.2,
            hud,
            Align::Center,
        );
        if blink {
            scene::text(
                out,
                "PRESS ENTER",
                scene::pt(0.0, -15.5),
                1.0,
                dim,
                Align::Center,
            );
        }
    } else if g.screen == Screen::Watch
        && let Some(w) = &g.watch
    {
        let t = clock(w.next as f32 * DT);
        let total = clock(w.replay.seconds());
        scene::text(
            out,
            &format!("REPLAY {t} / {total}"),
            scene::pt(0.0, top - 0.2),
            0.9,
            dim,
            Align::Center,
        );
        if w.speed > 1 {
            scene::text(
                out,
                &format!("X{} SPEED", w.speed),
                scene::pt(0.0, top - 1.8),
                0.7,
                dim,
                Align::Center,
            );
        }
        if w.other_build {
            scene::text(
                out,
                "RECORDED ON ANOTHER BUILD",
                scene::pt(0.0, -15.0),
                0.7,
                dim,
                Align::Center,
            );
        }
        if let Some(end) = &w.end {
            scene::text(out, end, scene::pt(0.0, 2.0), 2.0, hud, Align::Center);
            if blink {
                scene::text(
                    out,
                    "PRESS ENTER",
                    scene::pt(0.0, -1.5),
                    0.9,
                    dim,
                    Align::Center,
                );
            }
        } else {
            scene::text(
                out,
                "LEFT / RIGHT SPEED    ESC TO STOP",
                scene::pt(0.0, -16.5),
                0.6,
                dim,
                Align::Center,
            );
        }
    }
    if g.screen == Screen::Over {
        scene::text(
            out,
            "GAME OVER",
            scene::pt(0.0, 3.0),
            3.2,
            light::light(1.0, 0.35, 0.5, 3.0),
            Align::Center,
        );
        scene::text(
            out,
            &format!("SCORE {}", scene::grouped(s.score)),
            scene::pt(0.0, -0.8),
            1.4,
            hud,
            Align::Center,
        );
        if s.score >= g.best && s.score > 0 {
            scene::text(
                out,
                "NEW BEST",
                scene::pt(0.0, -3.2),
                1.0,
                scene::shard(),
                Align::Center,
            );
        }
        if g.over_timer > 1.5 && blink {
            scene::text(
                out,
                "PRESS ENTER",
                scene::pt(0.0, -6.5),
                1.0,
                dim,
                Align::Center,
            );
        }
    }
}

/// What the in-game HUD shows, from either mode.
struct Stats {
    score: u64,
    mult: u32,
    lives: u32,
    bombs: u32,
    phase: Phase,
    /// The Tunnel's speed bonus.
    speed: Option<f32>,
    /// The Tunnel's gate chain.
    chain: u32,
}

impl Stats {
    fn of(g: &Game) -> Stats {
        match (g.mode, &g.tunnel) {
            (Mode::Tunnel, Some(run)) => {
                let w = &run.world;
                Stats {
                    score: w.score,
                    mult: w.mult,
                    lives: w.lives,
                    bombs: w.bombs,
                    phase: w.phase,
                    speed: Some(w.speed_bonus()),
                    chain: w.chain,
                }
            }
            _ => Stats {
                score: g.sim.score,
                mult: g.sim.mult,
                lives: g.sim.lives,
                bombs: g.sim.bombs,
                phase: g.sim.phase,
                speed: None,
                chain: 0,
            },
        }
    }
}

/// A vertical menu, centred, with the selected item marked.
fn menu_items(out: &mut Vec<LineInstance>, items: &[&str], sel: usize, y0: f32, time: f32) {
    let hud = palette::HUD;
    let dim = light::fade(hud, 0.45);
    for (k, item) in items.iter().enumerate() {
        let y = y0 - k as f32 * 2.2;
        if k == sel {
            let c = light::fade(hud, 1.0 + 0.25 * signal::wave(time * 5.0));
            scene::text(
                out,
                &format!("> {item} <"),
                scene::pt(0.0, y),
                1.4,
                c,
                Align::Center,
            );
        } else {
            scene::text(out, item, scene::pt(0.0, y), 1.2, dim, Align::Center);
        }
    }
}
