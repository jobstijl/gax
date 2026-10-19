//! One frame of game logic, without Bevy: the menus, the fixed-step simulation of either game
//! (recorded, or played back from a replay), effects and sound.

use crate::audio::{self, Sound};
use crate::game::{Game, Playback, Run, Screen, Tape, TunnelRun};
use crate::input::{Flight, Menu};
use crate::menus::{self, Quit};
use crate::sim::replay::{Desync, Packed, Record, Replay};
use crate::sim::{self, DT, Phase};
use crate::tunnel::{self, replay::Packed as TunnelPacked};
use gax::pga2d::Point;
use std::time::Instant;

/// One frame: the menus, then the run's ticks, effects and sound. `input` is the Plane's
/// controls (the Tunnel's extra ones are `g.flight`), `aspect` the window's width over its
/// height.
pub fn advance(
    g: &mut Game,
    input: sim::Input,
    menu: Menu,
    dt: f32,
    aspect: f32,
    sound: &mut Sound,
) -> Quit {
    let quit = menus::step(g, menu, dt);
    g.apply_settings(sound);
    if let Some((seed, tempo)) = g.music.take() {
        sound.begin_run(seed, tempo);
    }
    let started = Instant::now();
    let over = match g.run {
        Run::Plane(_) => advance_plane(g, input, dt, aspect, sound),
        Run::Tunnel(_) => advance_tunnel(g, input, dt, sound),
    };
    g.sim_ms = g.sim_ms * 0.9 + 0.1 * started.elapsed().as_secs_f32() * 1e3;
    if over {
        g.end();
    }
    quit
}

/// A frame of the Plane: its ticks, effects, sound and camera. Returns whether the run was
/// lost.
fn advance_plane(g: &mut Game, input: sim::Input, dt: f32, aspect: f32, sound: &mut Sound) -> bool {
    let Run::Plane(run) = &mut g.run else {
        return false;
    };
    // In play, the bomb is an edge, kept until a tick consumes it; out of play, nothing waits.
    run.input = if g.screen == Screen::Playing {
        sim::Input {
            bomb: run.input.bomb || input.bomb,
            ..input
        }
    } else {
        sim::Input::default()
    };
    g.fx.grid_steps.clear();
    g.fx.spawn.clear();
    let mut over = false;
    if g.screen != Screen::Paused {
        g.acc += dt * speed(&run.tape) as f32;
        while g.acc >= DT && !over {
            g.acc -= DT;
            if g.hitstop > 0.0 {
                // Hit-stop: the world holds its breath; effects go on.
                g.hitstop = (g.hitstop - DT).max(0.0);
            } else if tick(&mut run.world, &mut run.input, &mut run.tape, g.screen) {
                let w = &run.world;
                g.hitstop = hitstop(&w.events);
                g.fx.on_events(&w.events);
                sound.on_events(&w.events, w, g.cam);
                over = g.screen == Screen::Playing && w.events.contains(&sim::Event::GameOver);
            }
            g.fx.grid_tick(&run.world, DT);
        }
        g.alpha = g.acc / DT;
    }
    let w = &run.world;
    let (intensity, darkness) = (w.director.intensity, w.darkness());
    steer_music(sound, g.screen, intensity, w.phase, darkness, w.ledger.mult);
    // A view about 28 units tall (wider screens see more), moving over the arena.
    g.half_height = 14.0_f32.max(24.0 / aspect);
    g.fx.half_height = g.half_height;
    g.fx.aspect = aspect;
    g.fx.frame(dt);
    g.cam = g.fx.camera(&run.world, dt);
    over
}

/// A frame of the Tunnel: controls through the camera, its ticks, its view, and sound.
/// Returns whether the run was lost.
fn advance_tunnel(g: &mut Game, input: sim::Input, dt: f32, sound: &mut Sound) -> bool {
    let Run::Tunnel(run) = &mut g.run else {
        return false;
    };
    match g.screen {
        // The replay aims: show where.
        Screen::Watch => run.view.follow(&run.world, run.input.aim),
        Screen::Playing => steer(run, input, g.flight),
        _ => {}
    }
    let mut over = false;
    if g.screen != Screen::Paused {
        g.acc += dt * speed(&run.tape) as f32;
        while g.acc >= DT && !over {
            g.acc -= DT;
            if !tick(&mut run.world, &mut run.input, &mut run.tape, g.screen) {
                g.acc = 0.0;
                break;
            }
            run.view.on_events(&run.world.events);
            audio::tunnel::sounds(&run.world, &run.view, &mut run.heard);
            let listener = sim::body::Pose::identity();
            sound.play(
                &run.heard,
                run.world.ledger.mult,
                crate::geom::ORIGIN,
                listener,
            );
            let lost = run.world.events.contains(&tunnel::Event::GameOver);
            over = g.screen == Screen::Playing && lost;
        }
        g.alpha = g.acc / DT;
    }
    run.view.update(&run.world, g.alpha, dt);
    let w = &run.world;
    steer_music(
        sound,
        g.screen,
        w.intensity,
        w.phase,
        w.darkness(),
        w.ledger.mult,
    );
    over
}

/// Steer the music from a run: its intensity (halved while the ship is gone), how near the
/// ship is to a singularity, its multiplier, and whether it is being played.
fn steer_music(
    sound: &mut Sound,
    screen: Screen,
    intensity: f32,
    phase: Phase,
    darkness: f32,
    mult: u32,
) {
    let presence = if phase == Phase::Playing { 1.0 } else { 0.5 };
    let playing = matches!(screen, Screen::Playing | Screen::Watch);
    sound.steer(intensity * presence, darkness, mult, playing);
}

/// The player's Tunnel controls for the next tick, through the camera: screen directions
/// become directions across the tunnel, and the reticle (the cursor, or the right stick
/// pushing it out from the ship on screen) an aim point. Edges wait for the tick that
/// consumes them.
fn steer(run: &mut TunnelRun, input: sim::Input, flight: Flight) {
    let (w, view) = (&run.world, &mut run.view);
    let movement = view.across(w, input.movement);
    let reticle = match (flight.stick, flight.cursor) {
        (Some(stick), _) => view.ship_on_screen(w) + stick * 9.0,
        (None, Some(cursor)) => cursor,
        (None, None) => view.ship_on_screen(w) + Point::direction(0.0, 4.0),
    };
    let aim = view.aim(w, reticle);
    let waiting = run.input;
    run.input = tunnel::Input {
        movement,
        aim,
        fire: input.fire,
        bomb: waiting.bomb || flight.bomb,
        roll: if waiting.roll != 0 {
            waiting.roll
        } else {
            flight.roll
        },
        throttle: flight.throttle,
    };
}

/// The hit-stop a tick's events call for: a death's, or a singularity shot down.
fn hitstop(events: &[sim::Event]) -> f32 {
    events.iter().fold(0.0, |stop, e| match e {
        sim::Event::Death { .. } => 0.15,
        sim::Event::Kill {
            kind: sim::Kind::Singularity,
            scored: true,
            ..
        } => stop.max(0.07),
        _ => stop,
    })
}

/// Ticks per frame tick: a watched replay's speed, or one.
fn speed<I: Record>(tape: &Tape<I>) -> u32 {
    match tape {
        Tape::Play(p) => p.watch.speed,
        Tape::Record(_) => 1,
    }
}

/// A game's world as its tape sees it: inputs recorded and played back through the game's
/// replay record, and the hashes kept and checked.
pub trait Simulation {
    type Record: Record;
    type Input: Default;
    /// Record `input` on `rec`; what the world must see (the input as the record keeps it).
    fn take(&self, rec: &mut Replay<Self::Record>, input: &Self::Input) -> Self::Input;
    /// The input a record stands for at this tick.
    fn unpack(&self, p: Self::Record) -> Self::Input;
    /// One tick.
    fn step(&mut self, input: &Self::Input);
    /// The input's edges, consumed by a tick.
    fn clear_edges(input: &mut Self::Input);
    /// After a recorded tick: the replay keeps the hash and the score.
    fn after(&self, rec: &mut Replay<Self::Record>);
    /// The world played back from `replay` against its recorded hash, if one falls on this
    /// tick.
    fn check(&self, replay: &Replay<Self::Record>) -> Result<(), Desync>;
    /// Ticks since the start.
    fn ticks(&self) -> u64;
    fn score(&self) -> u64;
}

impl Simulation for sim::World {
    type Record = Packed;
    type Input = sim::Input;
    fn take(&self, rec: &mut Replay<Packed>, input: &sim::Input) -> sim::Input {
        let p = Packed::pack(input);
        rec.record(p);
        p.unpack()
    }
    fn unpack(&self, p: Packed) -> sim::Input {
        p.unpack()
    }
    fn step(&mut self, input: &sim::Input) {
        self.tick(input);
    }
    fn clear_edges(input: &mut sim::Input) {
        input.bomb = false;
    }
    fn after(&self, rec: &mut Replay<Packed>) {
        rec.after(self);
    }
    fn check(&self, replay: &Replay<Packed>) -> Result<(), Desync> {
        replay.check(self)
    }
    fn ticks(&self) -> u64 {
        self.tick
    }
    fn score(&self) -> u64 {
        self.ledger.score
    }
}

impl Simulation for tunnel::World {
    type Record = TunnelPacked;
    type Input = tunnel::Input;
    fn take(&self, rec: &mut Replay<TunnelPacked>, input: &tunnel::Input) -> tunnel::Input {
        rec.take(input, self)
    }
    fn unpack(&self, p: TunnelPacked) -> tunnel::Input {
        p.unpack(self.ship.s())
    }
    fn step(&mut self, input: &tunnel::Input) {
        self.tick(input);
    }
    fn clear_edges(input: &mut tunnel::Input) {
        input.bomb = false;
        input.roll = 0;
    }
    fn after(&self, rec: &mut Replay<TunnelPacked>) {
        rec.after(self);
    }
    fn check(&self, replay: &Replay<TunnelPacked>) -> Result<(), Desync> {
        replay.check(self)
    }
    fn ticks(&self) -> u64 {
        self.tick
    }
    fn score(&self) -> u64 {
        self.ledger.score
    }
}

/// One tick: in play, the player's `input`, recorded; while watching, the replay's; otherwise
/// none. Returns whether the world moved (a replay that has ended holds still: flying on, a
/// Tunnel run would pass gates and score without the player, and a Plane ship could die).
fn tick<W: Simulation>(
    w: &mut W,
    input: &mut W::Input,
    tape: &mut Tape<W::Record>,
    screen: Screen,
) -> bool {
    match tape {
        Tape::Record(rec) if screen == Screen::Playing => {
            record(w, input, rec);
            true
        }
        Tape::Record(_) => {
            w.step(&W::Input::default());
            true
        }
        Tape::Play(p) => play(w, input, p),
    }
}

/// A recorded tick: the world sees the packed input, exactly what the replay keeps.
fn record<W: Simulation>(w: &mut W, input: &mut W::Input, rec: &mut Replay<W::Record>) {
    let seen = w.take(rec, input);
    w.step(&seen);
    w.after(rec);
    W::clear_edges(input);
}

/// What a tick of playback did.
#[derive(Clone, Debug, PartialEq)]
pub enum Played {
    /// The world ticked, as recorded.
    Ticked,
    /// The world ticked, and its hash differs from the recorded one: what to say.
    Diverged(String),
    /// The replay has no more ticks.
    Ended,
}

/// Play the replay's next tick, its input into `input`.
pub fn play_back<W: Simulation>(
    w: &mut W,
    input: &mut W::Input,
    p: &mut Playback<W::Record>,
) -> Played {
    let Some(&record) = p.replay.inputs.get(p.watch.next) else {
        return Played::Ended;
    };
    p.watch.next += 1;
    *input = w.unpack(record);
    w.step(input);
    match w.check(&p.replay) {
        Ok(()) => Played::Ticked,
        Err(d) => {
            let tick = match d {
                Desync::Hash(t) => t,
                Desync::Score { .. } => w.ticks(),
            };
            Played::Diverged(format!("DIVERGED AT {}", clock(tick as f32 * DT)))
        }
    }
}

/// A watched tick, until the playback ends or diverges; then the world holds still. At the
/// end the score must be the recorded one. Returns whether the world moved.
fn play<W: Simulation>(w: &mut W, input: &mut W::Input, p: &mut Playback<W::Record>) -> bool {
    if p.watch.end.is_some() {
        return false;
    }
    match play_back(w, input, p) {
        Played::Ticked => true,
        Played::Diverged(why) => {
            p.watch.stop(why);
            true
        }
        Played::Ended => {
            let why = match p.replay.final_score(w.score()) {
                Ok(()) => "END OF REPLAY",
                Err(_) => "DIVERGED AT THE END",
            };
            p.watch.stop(why.into());
            false
        }
    }
}

/// `m:ss`.
pub fn clock(seconds: f32) -> String {
    let s = seconds.max(0.0) as u32;
    format!("{}:{:02}", s / 60, s % 60)
}

#[cfg(test)]
mod tests;
