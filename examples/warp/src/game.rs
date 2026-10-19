//! The game between frames: which screen it is on, the run (played, watched, or the attract
//! world behind the menus), the tables and the settings, and the moves between them.

use crate::audio::{self, music};
use crate::fx::Fx;
use crate::input;
use crate::render::LineInstance;
use crate::render::tunnel::{View, ViewSettings};
use crate::sim::body::Pose;
use crate::sim::replay::{Packed, Record, Replay};
use crate::sim::{self, Ledger, Phase};
use crate::store::{Scores, Settings, Store};
use crate::tunnel::{self, replay::Packed as TunnelPacked};
use bevy::prelude::Resource;
use gax::pga2d::Point;
use std::ops::{Index, IndexMut};

/// Where the game is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Screen {
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
pub const ATTRACT_AFTER: f32 = 20.0;
/// A demo plays this long at most.
pub const DEMO_LENGTH: f32 = 75.0;

/// Which game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Plane,
    Tunnel,
}

impl Mode {
    /// The other game.
    pub fn other(self) -> Mode {
        match self {
            Mode::Plane => Mode::Tunnel,
            Mode::Tunnel => Mode::Plane,
        }
    }

    /// Its name on the menus.
    pub fn name(self) -> &'static str {
        match self {
            Mode::Plane => "PLANE",
            Mode::Tunnel => "TUNNEL",
        }
    }

    /// Its music's tempo.
    fn tempo(self) -> f32 {
        match self {
            Mode::Plane => music::TEMPO,
            Mode::Tunnel => music::TUNNEL_TEMPO,
        }
    }

    /// Mixed into its runs' seeds.
    fn salt(self) -> u64 {
        match self {
            Mode::Plane => 0x5eed,
            Mode::Tunnel => 0x70e1,
        }
    }

    /// The seed of the bot's demo, when the table has no run to show.
    fn demo_seed(self) -> u64 {
        match self {
            Mode::Plane => 0xde70,
            Mode::Tunnel => 0xde71,
        }
    }

    /// Its high-score table in `store`.
    fn scores(self, store: &Store) -> Scores {
        match self {
            Mode::Plane => store.scores::<Packed>(),
            Mode::Tunnel => store.scores::<TunnelPacked>(),
        }
    }

    /// The replay `file` of its table in `store`.
    pub fn replay(self, store: &Store, file: &str) -> Option<AnyReplay> {
        match self {
            Mode::Plane => store.replay(file).map(AnyReplay::Plane),
            Mode::Tunnel => store.replay(file).map(AnyReplay::Tunnel),
        }
    }
}

/// One of a thing for each game.
#[derive(Clone, Debug, Default)]
pub struct PerMode<T> {
    pub plane: T,
    pub tunnel: T,
}

impl<T> PerMode<T> {
    /// Each game's, from `f`.
    pub fn new(mut f: impl FnMut(Mode) -> T) -> PerMode<T> {
        PerMode {
            plane: f(Mode::Plane),
            tunnel: f(Mode::Tunnel),
        }
    }
}

impl<T> Index<Mode> for PerMode<T> {
    type Output = T;
    fn index(&self, m: Mode) -> &T {
        match m {
            Mode::Plane => &self.plane,
            Mode::Tunnel => &self.tunnel,
        }
    }
}

impl<T> IndexMut<Mode> for PerMode<T> {
    fn index_mut(&mut self, m: Mode) -> &mut T {
        match m {
            Mode::Plane => &mut self.plane,
            Mode::Tunnel => &mut self.tunnel,
        }
    }
}

/// A game's high scores, and its best score this session (the table's first, or better).
#[derive(Clone, Debug, Default)]
pub struct Table {
    pub scores: Scores,
    pub best: u64,
}

impl Table {
    /// The table of `scores`.
    fn new(scores: Scores) -> Table {
        let best = scores.entries.first().map_or(0, |e| e.score);
        Table { scores, best }
    }
}

/// A replay of either game.
pub enum AnyReplay {
    Plane(Replay),
    Tunnel(Replay<TunnelPacked>),
}

impl AnyReplay {
    /// Which game.
    pub fn mode(&self) -> Mode {
        match self {
            AnyReplay::Plane(_) => Mode::Plane,
            AnyReplay::Tunnel(_) => Mode::Tunnel,
        }
    }
}

/// How a replay is being watched.
#[derive(Clone, Debug)]
pub struct Watch {
    /// The next tick's index in the replay.
    pub next: usize,
    /// The replay's length.
    pub seconds: f32,
    /// Set when the playback ended or diverged: what to say.
    pub end: Option<String>,
    /// Ticks per tick (fast forward).
    pub speed: u32,
    /// Recorded on another build (it may diverge).
    pub other_build: bool,
}

impl Watch {
    /// The playback ended (or diverged): say `why`, at normal speed.
    pub fn stop(&mut self, why: String) {
        self.end = Some(why);
        self.speed = 1;
    }
}

/// A replay played back.
pub struct Playback<I: Record> {
    pub replay: Replay<I>,
    pub watch: Watch,
}

/// Where a run's inputs go, or come from.
pub enum Tape<I: Record> {
    /// The player's, recorded.
    Record(Replay<I>),
    /// A replay's.
    Play(Playback<I>),
}

impl<I: Record> Tape<I> {
    /// A replay to play back.
    fn play(replay: Replay<I>) -> Tape<I> {
        let watch = Watch {
            next: 0,
            seconds: replay.seconds(),
            end: None,
            speed: 1,
            other_build: !replay.is_this_build(),
        };
        Tape::Play(Playback { replay, watch })
    }

    /// The watch of a replay being played back.
    fn watch(&self) -> Option<&Watch> {
        match self {
            Tape::Play(p) => Some(&p.watch),
            Tape::Record(_) => None,
        }
    }

    fn watch_mut(&mut self) -> Option<&mut Watch> {
        match self {
            Tape::Play(p) => Some(&mut p.watch),
            Tape::Record(_) => None,
        }
    }

    /// The recording ends with `score`.
    fn close(&mut self, score: u64) {
        if let Tape::Record(rec) = self {
            rec.score = score;
        }
    }

    /// Keep the recording in `store` (as an entry named `name`, if any); its rank.
    fn keep(&self, store: &Store, name: Option<&str>) -> Option<usize> {
        match self {
            Tape::Record(rec) => store.finish(rec, name),
            Tape::Play(_) => None,
        }
    }
}

/// A Plane run: its world, the input waiting for the next tick, and its tape.
pub struct PlaneRun {
    pub world: sim::World,
    pub input: sim::Input,
    pub tape: Tape<Packed>,
}

impl PlaneRun {
    /// A run from `seed`, recorded.
    fn new(seed: u64) -> PlaneRun {
        PlaneRun::taped(seed, Tape::Record(Replay::new(seed)))
    }

    fn taped(seed: u64, tape: Tape<Packed>) -> PlaneRun {
        PlaneRun {
            world: sim::World::new(seed),
            input: sim::Input::default(),
            tape,
        }
    }

    /// The attract world behind the menus: enemies wandering, a singularity breathing.
    fn attract() -> PlaneRun {
        let mut run = PlaneRun::new(0xa77ac7);
        let w = &mut run.world;
        w.phase = Phase::Over;
        w.director.enabled = false;
        for k in 0..14 {
            let kind = match k % 5 {
                0 => sim::Kind::Chaser,
                2 => sim::Kind::Evader,
                4 if k > 8 => sim::Kind::Splitter,
                _ => sim::Kind::Drifter,
            };
            w.spawn(kind, attract_spot(k as f32 * 0.45, 10.0 + k as f32, 9.0));
        }
        w.spawn(sim::Kind::Singularity, sim::at(12.0, -4.0));
        run
    }
}

/// A spot on an ellipse around the title: the unit direction at `angle`, stretched to the
/// half axes `rx` and `ry`.
fn attract_spot(angle: f32, rx: f32, ry: f32) -> sim::P {
    let h = Point::direction_from_angle(angle);
    sim::at(h.e20() * rx, h.e01() * ry)
}

/// A Tunnel run: its world and view, the input waiting for the next tick, its tape, and the
/// sound's events of the last tick.
pub struct TunnelRun {
    pub world: tunnel::World,
    pub view: View,
    pub input: tunnel::Input,
    pub tape: Tape<TunnelPacked>,
    pub heard: Vec<sim::Event>,
}

impl TunnelRun {
    /// A run from `seed` on `tape`, its view set up by the settings.
    fn new(seed: u64, tape: Tape<TunnelPacked>, settings: &Settings) -> TunnelRun {
        let mut view = View::new(f32::from(settings.fov));
        view.settings = view_settings(settings);
        TunnelRun {
            world: tunnel::World::new(seed),
            view,
            input: tunnel::Input::default(),
            tape,
            heard: Vec::new(),
        }
    }
}

/// The Tunnel view's settings from the player's.
fn view_settings(s: &Settings) -> ViewSettings {
    ViewSettings {
        focal: View::focal_for(f32::from(s.fov)),
        follow_roll: s.camera_roll,
        shake_scale: if s.reduced_motion {
            0.0
        } else {
            f32::from(s.shake) / 4.0
        },
        flash_scale: if s.reduced_flashes { 0.3 } else { 1.0 },
        assist: s.aim_reach(),
        lensing: !s.reduced_motion,
        scheme: s.scheme,
    }
}

/// The run on screen: played, watched, or the attract world behind the menus.
pub enum Run {
    Plane(Box<PlaneRun>),
    Tunnel(Box<TunnelRun>),
}

impl Run {
    /// Which game.
    pub fn mode(&self) -> Mode {
        match self {
            Run::Plane(_) => Mode::Plane,
            Run::Tunnel(_) => Mode::Tunnel,
        }
    }

    /// Its score, multiplier, lives and bombs.
    pub fn ledger(&self) -> &Ledger {
        match self {
            Run::Plane(r) => &r.world.ledger,
            Run::Tunnel(r) => &r.world.ledger,
        }
    }

    /// Where the run is.
    pub fn phase(&self) -> Phase {
        match self {
            Run::Plane(r) => r.world.phase,
            Run::Tunnel(r) => r.world.phase,
        }
    }

    /// The watch, when a replay is played back.
    pub fn watch(&self) -> Option<&Watch> {
        match self {
            Run::Plane(r) => r.tape.watch(),
            Run::Tunnel(r) => r.tape.watch(),
        }
    }

    /// The watch, to change.
    pub fn watch_mut(&mut self) -> Option<&mut Watch> {
        match self {
            Run::Plane(r) => r.tape.watch_mut(),
            Run::Tunnel(r) => r.tape.watch_mut(),
        }
    }

    /// The Plane run (`None` in the Tunnel).
    pub fn plane(&mut self) -> Option<&mut PlaneRun> {
        match self {
            Run::Plane(r) => Some(r),
            Run::Tunnel(_) => None,
        }
    }

    /// The Tunnel run (`None` in the Plane).
    pub fn tunnel(&self) -> Option<&TunnelRun> {
        match self {
            Run::Plane(_) => None,
            Run::Tunnel(r) => Some(r),
        }
    }

    /// The run is over: its world ends, and its recording keeps the score. Returns the score.
    fn end(&mut self) -> u64 {
        let score = self.ledger().score;
        match self {
            Run::Plane(r) => {
                r.world.end();
                r.tape.close(score);
            }
            Run::Tunnel(r) => {
                r.world.phase = Phase::Over;
                r.tape.close(score);
            }
        }
        score
    }

    /// Keep the recording in `store` (as an entry named `name`, if any); its rank.
    fn keep(&self, store: &Store, name: Option<&str>) -> Option<usize> {
        match self {
            Run::Plane(r) => r.tape.keep(store, name),
            Run::Tunnel(r) => r.tape.keep(store, name),
        }
    }
}

/// The game: the run, effects, the tables and settings, and what the menus and the HUD show.
#[derive(Resource)]
pub struct Game {
    pub run: Run,
    pub fx: Fx,
    pub screen: Screen,
    /// Simulation time not yet ticked, and how far between ticks a frame is drawn.
    pub acc: f32,
    pub alpha: f32,
    /// Seconds of hit-stop left: the world holds its breath (ticks skipped, effects go on).
    /// Not simulation state, so replays are unaffected.
    pub hitstop: f32,
    pub device: input::Device,
    pub latch: input::Latch,
    /// The Plane's camera, and the half height of its view.
    pub cam: Pose,
    pub half_height: f32,
    pub time: f32,
    /// The debug overlay (F3), and what it shows.
    pub debug: bool,
    pub sim_ms: f32,
    pub fps: f32,
    pub over_timer: f32,
    /// A bot plays (the `--smoke` run, and the attract mode without a table).
    pub bot: bool,
    pub store: Store,
    pub settings: Settings,
    pub tables: PerMode<Table>,
    /// Which table the scores screen shows.
    pub table: Mode,
    /// Seconds on the title without input (the attract mode's clock).
    pub idle: f32,
    /// A demo is playing (the attract mode); which game the next one plays.
    pub demo: bool,
    pub demo_turn: Mode,
    /// The menu cursor.
    pub sel: usize,
    /// Initials being entered, and the cursor in them.
    pub initials: [u8; 3],
    pub cursor: usize,
    /// The table row to highlight (the run just entered).
    pub highlight: Option<usize>,
    /// The Tunnel's extra controls this frame.
    pub flight: input::Flight,
    /// A run's music to start: its seed and tempo (taken by the next frame).
    pub music: Option<(u64, f32)>,
    pub world_lines: Vec<LineInstance>,
    pub hud_lines: Vec<LineInstance>,
}

impl Game {
    /// The game on the title, with what `store` keeps.
    pub fn new(store: Store) -> Game {
        let settings = store.settings();
        let tables = PerMode::new(|m| Table::new(m.scores(&store)));
        Game {
            run: Run::Plane(Box::new(PlaneRun::attract())),
            fx: Fx::new(),
            screen: Screen::Title,
            acc: 0.0,
            alpha: 0.0,
            hitstop: 0.0,
            device: input::Device::Mouse,
            latch: input::Latch::default(),
            cam: sim::body::Pose::identity(),
            half_height: 20.0,
            time: 0.0,
            debug: false,
            sim_ms: 0.0,
            fps: 60.0,
            over_timer: 0.0,
            bot: false,
            store,
            settings,
            tables,
            table: Mode::Plane,
            idle: 0.0,
            demo: false,
            demo_turn: Mode::Tunnel,
            sel: 0,
            initials: *b"AAA",
            cursor: 0,
            highlight: None,
            flight: input::Flight::default(),
            music: None,
            world_lines: Vec::new(),
            hud_lines: Vec::new(),
        }
    }

    /// The attract world behind the menus.
    pub fn attract(&mut self) {
        self.run = Run::Plane(Box::new(PlaneRun::attract()));
    }

    /// Start a run of `mode`.
    pub fn start(&mut self, mode: Mode) {
        let time = u64::from(self.time.to_bits());
        let seed = u64::from(std::process::id()) ^ time << 20 ^ mode.salt();
        self.start_seeded(mode, seed);
    }

    /// Start a run of `mode` from `seed`.
    pub fn start_seeded(&mut self, mode: Mode, seed: u64) {
        self.run = match mode {
            Mode::Plane => Run::Plane(Box::new(PlaneRun::new(seed))),
            Mode::Tunnel => {
                let tape = Tape::Record(Replay::new(seed));
                Run::Tunnel(Box::new(TunnelRun::new(seed, tape, &self.settings)))
            }
        };
        self.screen = Screen::Playing;
        self.acc = 0.0;
        self.highlight = None;
        self.music = Some((seed, mode.tempo()));
    }

    /// Watch a replay.
    pub fn watch(&mut self, replay: AnyReplay) {
        let mode = replay.mode();
        let seed = match &replay {
            AnyReplay::Plane(r) => r.seed,
            AnyReplay::Tunnel(r) => r.seed,
        };
        self.run = match replay {
            AnyReplay::Plane(r) => Run::Plane(Box::new(PlaneRun::taped(seed, Tape::play(r)))),
            AnyReplay::Tunnel(r) => Run::Tunnel(Box::new(TunnelRun::new(
                seed,
                Tape::play(r),
                &self.settings,
            ))),
        };
        self.screen = Screen::Watch;
        self.acc = 0.0;
        self.music = Some((seed, mode.tempo()));
    }

    /// Back to the title (from either game).
    pub fn back_to_title(&mut self) {
        self.demo = false;
        self.bot = false;
        self.idle = 0.0;
        self.screen = Screen::Title;
        self.attract();
    }

    /// Done watching a replay: back to its table.
    pub fn stop_watching(&mut self) {
        self.table = self.run.mode();
        self.screen = Screen::Scores;
        self.attract();
    }

    /// The attract mode: the best run of one game or the other, replayed; or, with no table
    /// yet, the bot playing.
    pub fn start_demo(&mut self) {
        let mode = self.demo_turn;
        self.demo_turn = mode.other();
        self.demo = true;
        self.idle = 0.0;
        let best = self.tables[mode].scores.entries.first();
        match best.and_then(|e| mode.replay(&self.store, &e.replay)) {
            Some(replay) => self.watch(replay),
            None => {
                self.start_seeded(mode, mode.demo_seed());
                self.bot = true;
            }
        }
    }

    /// The run is over (lost, or ended from the pause menu): its replay is kept, and the
    /// player asked for initials if it makes its game's table. Only the player's runs count
    /// (not a demo's or the bot's).
    pub fn end(&mut self) {
        let mode = self.run.mode();
        let score = self.run.end();
        self.over_timer = 0.0;
        if self.demo {
            self.back_to_title();
            return;
        }
        let table = &mut self.tables[mode];
        let ranked = !self.bot && table.scores.rank(score).is_some();
        if !self.bot {
            table.best = table.best.max(score);
        }
        if ranked {
            self.screen = Screen::Initials;
            self.cursor = 0;
        } else {
            self.run.keep(&self.store, None);
            self.screen = Screen::Over;
        }
    }

    /// Enter the run just over on its table as `name`, and show the table.
    pub fn enter(&mut self, name: &str) {
        let mode = self.run.mode();
        self.highlight = self.run.keep(&self.store, Some(name));
        self.tables[mode].scores = mode.scores(&self.store);
        self.table = mode;
        self.sel = self.highlight.unwrap_or(0);
        self.screen = Screen::Scores;
        self.attract();
    }

    /// Apply the settings to effects, colours and sound.
    pub fn apply_settings(&mut self, sound: &mut audio::Sound) {
        let s = &self.settings;
        self.fx.shake_scale = if s.reduced_motion {
            0.0
        } else {
            f32::from(s.shake) / 4.0
        };
        self.fx.flash_scale = if s.reduced_flashes { 0.3 } else { 1.0 };
        self.fx.reduced_motion = s.reduced_motion;
        self.fx.scheme = s.scheme;
        if let Run::Tunnel(run) = &mut self.run {
            run.view.settings = view_settings(s);
        }
        sound.gains = s.gains();
        sound.on_beat = s.on_beat;
    }
}
