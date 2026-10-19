//! The HUD, in HUD units (the screen is `2 HUD_HALF_HEIGHT` units tall, centred): the menus,
//! the score, lives and bombs in play, the overlays, and the debug overlay.

use crate::frame::clock;
use crate::game::{Game, Run, Screen};
use crate::light::{self, Light};
use crate::menus::{PauseItem, TitleItem};
use crate::render::font::Align;
use crate::render::scene::{self, HUD_HALF_HEIGHT, palette};
use crate::render::{self, LineInstance};
use crate::signal;
use crate::sim::{self, Phase};
use crate::store::{Row, Value};

/// The HUD of the game's screen into `g.hud_lines`, for a target of `size` pixels; `timings`
/// are the GPU's, for the debug overlay.
pub fn hud(g: &mut Game, size: [u32; 2], timings: Option<render::Timings>) {
    // The lines are built apart from the game, which the screens read, then put back.
    let mut out = std::mem::take(&mut g.hud_lines);
    out.clear();
    let mut h = Hud {
        out: &mut out,
        st: HudStyle::new(size, g.time),
    };
    match g.screen {
        Screen::Title => title(g, &mut h),
        Screen::Settings => settings(g, &mut h),
        Screen::Scores => scores(g, &mut h),
        Screen::Playing | Screen::Paused | Screen::Over | Screen::Initials | Screen::Watch => {
            play(g, &mut h)
        }
    }
    // The HUD is drawn over the finished image (never bloomed): a tight glow keeps its text
    // sharp.
    for l in h.out.iter_mut() {
        l.style[1] = (l.style[1] * 0.45).max(0.06);
        l.style[2] *= 0.6;
    }
    if g.debug {
        debug(g, timings, &mut h);
    }
    g.hud_lines = out;
}

/// The HUD's layout and colours, shared by its screens.
#[derive(Clone, Copy)]
struct HudStyle {
    /// The margins: left, right and top.
    left: f32,
    right: f32,
    top: f32,
    hud: Light,
    dim: Light,
    /// On and off, for prompts.
    blink: bool,
    /// Menu rows: the selected one bright and pulsing, the others faint.
    hot: Light,
    faint: Light,
}

impl HudStyle {
    /// The style on a target of `size` pixels at `time`.
    fn new(size: [u32; 2], time: f32) -> HudStyle {
        let aspect = size[0] as f32 / size[1].max(1) as f32;
        let half_width = HUD_HALF_HEIGHT * aspect;
        let hud = palette::HUD;
        HudStyle {
            left: -half_width + 2.2,
            right: half_width - 2.2,
            top: HUD_HALF_HEIGHT - 2.6,
            hud,
            dim: hud.faded(0.45),
            blink: signal::wave(time * 3.0) > -0.3,
            hot: hud.faded(1.1 + 0.2 * signal::wave(time * 5.0)),
            faint: hud.faded(0.3),
        }
    }
}

/// Lines of the HUD being built, and its style.
struct Hud<'a> {
    out: &'a mut Vec<LineInstance>,
    st: HudStyle,
}

impl Hud<'_> {
    /// Text of `size` in `light`, centred on the baseline at height `y`.
    fn center(&mut self, s: &str, y: f32, size: f32, light: Light) {
        self.at(s, 0.0, y, size, light, Align::Center);
    }

    /// Text starting at `x`.
    fn left(&mut self, s: &str, x: f32, y: f32, size: f32, light: Light) {
        self.at(s, x, y, size, light, Align::Left);
    }

    /// Text ending at `x`.
    fn right(&mut self, s: &str, x: f32, y: f32, size: f32, light: Light) {
        self.at(s, x, y, size, light, Align::Right);
    }

    fn at(&mut self, s: &str, x: f32, y: f32, size: f32, light: Light, align: Align) {
        scene::text(self.out, s, sim::at(x, y), size, light, align);
    }

    /// A vertical menu of `items` from height `y0`, centred, the selected one marked.
    fn menu(&mut self, items: impl IntoIterator<Item = &'static str>, sel: usize, y0: f32) {
        for (k, item) in items.into_iter().enumerate() {
            let y = y0 - k as f32 * 2.2;
            if k == sel {
                self.center(&format!("> {item} <"), y, 1.4, self.st.hot);
            } else {
                self.center(item, y, 1.2, self.st.dim);
            }
        }
    }
}

/// The title screen.
fn title(g: &Game, h: &mut Hud) {
    let dim = h.st.dim;
    h.center("WARP", 7.0, 6.5, light::light(0.55, 0.7, 1.0, 1.7));
    h.center("A NEON SHOOTER ON WARPED SPACE", 3.4, 1.0, dim);
    h.menu(TitleItem::ALL.map(TitleItem::label), g.sel, -0.8);
    let controls = if TitleItem::ALL[g.sel] == TitleItem::Tunnel {
        "MOVE WASD / L-STICK   AIM MOUSE / R-STICK   ROLL Q E / BUMPERS   BOOST SHIFT / RT   BRAKE CTRL / LT"
    } else {
        "MOVE WASD / LEFT STICK    AIM + FIRE MOUSE / RIGHT STICK    BOMB SPACE / TRIGGER"
    };
    h.center(controls, -13.0, 0.7, dim);
    let grouped = scene::grouped;
    let best = match (g.tables.plane.best, g.tables.tunnel.best) {
        (0, 0) => String::new(),
        (p, 0) => format!("BEST {}", grouped(p)),
        (0, t) => format!("TUNNEL BEST {}", grouped(t)),
        (p, t) => format!("BEST {}    TUNNEL {}", grouped(p), grouped(t)),
    };
    h.center(&best, -15.0, 0.9, dim);
}

/// The settings screen: a row per setting, then BACK.
fn settings(g: &Game, h: &mut Hud) {
    let HudStyle {
        hud,
        dim,
        hot,
        faint,
        ..
    } = h.st;
    h.center("SETTINGS", 13.0, 3.0, hud);
    let (lx, rx) = (-15.0, 15.0);
    let row_y = |row: usize| 8.5 - row as f32 * 1.75;
    for (k, &row) in Row::ALL.iter().enumerate() {
        let y = row_y(k);
        let on = k == g.sel;
        let c = if on { hot } else { faint };
        if on {
            h.left(">", lx - 2.0, y, 1.1, hud);
        }
        h.left(row.label(), lx, y, 1.1, c);
        match g.settings.value(row) {
            Value::Text(v) if on => h.right(&format!("< {v} >"), rx, y, 1.1, c),
            Value::Text(v) => h.right(&v, rx, y, 1.1, c),
            Value::Level(level) => bars(h, rx, y, level, c),
        }
    }
    let back = Row::ALL.len();
    let y = row_y(back) - 0.5;
    let on = g.sel == back;
    if on {
        h.left(">", lx - 2.0, y, 1.1, hud);
    }
    h.left("BACK", lx, y, 1.1, if on { hot } else { faint });
    h.center(
        "LEFT / RIGHT TO CHANGE    ESC / B TO GO BACK",
        -16.2,
        0.7,
        dim,
    );
}

/// A volume: ten bars ending at `right`, lit up to `level`.
fn bars(h: &mut Hud, right: f32, y: f32, level: u8, c: Light) {
    h.out.extend((0..10u8).map(|k| {
        let x = right - 9.5 * 0.9 + f32::from(k) * 0.9;
        let bc = c.faded(if k < level { 1.2 } else { 0.25 });
        scene::seg(
            sim::at(x, y + 0.15),
            sim::at(x, y + 0.35 + 0.1 * f32::from(k)),
            bc,
            scene::Stroke::new(0.09, 0.3, 0.2),
            sim::body::Pose::identity(),
        )
    }));
}

/// The high scores of the game the screen shows.
fn scores(g: &Game, h: &mut Hud) {
    let HudStyle {
        hud,
        dim,
        hot,
        faint,
        ..
    } = h.st;
    h.center("HIGH SCORES", 12.0, 3.0, hud);
    h.center(&format!("< {} >", g.table.name()), 9.0, 1.2, hot);
    let entries = &g.tables[g.table].scores.entries;
    if entries.is_empty() {
        h.center("NO RUNS YET", 2.0, 1.4, dim);
    }
    for (r, e) in entries.iter().enumerate() {
        let y = 6.0 - r as f32 * 1.75;
        let c = if g.highlight == Some(r) {
            g.settings.scheme.shard()
        } else if r == g.sel {
            hot
        } else {
            faint
        };
        if r == g.sel {
            h.left(">", -15.0, y, 1.0, c);
        }
        h.left(&format!("{:>2}", r + 1), -13.0, y, 1.0, c);
        h.left(&e.name, -9.0, y, 1.0, c);
        h.right(&scene::grouped(e.score), 7.0, y, 1.0, c);
        h.right(&clock(e.seconds as f32), 14.0, y, 1.0, c);
    }
    let help = "ENTER / A TO WATCH    LEFT / RIGHT FOR THE OTHER GAME    ESC / B TO GO BACK";
    h.center(help, -14.5, 0.7, dim);
}

/// What the in-game HUD shows, from either game.
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
    fn of(run: &Run) -> Stats {
        let l = run.ledger();
        let (speed, chain) = match run {
            Run::Tunnel(r) => (Some(r.world.speed_bonus()), r.world.chain),
            Run::Plane(_) => (None, 0),
        };
        Stats {
            score: l.score,
            mult: l.mult,
            lives: l.lives,
            bombs: l.bombs,
            phase: run.phase(),
            speed,
            chain,
        }
    }
}

/// The HUD of play: the score, multiplier, lives and bombs, and the overlays of the screens
/// over it.
fn play(g: &Game, h: &mut Hud) {
    let s = Stats::of(&g.run);
    counters(g, &s, h);
    if let Phase::Dead(_) = s.phase {
        h.center("SHIP LOST", 1.0, 2.0, palette::SHIP);
    }
    match g.screen {
        Screen::Paused => {
            h.center("PAUSED", 4.0, 3.0, h.st.hud);
            h.menu(PauseItem::ALL.map(PauseItem::label), g.sel, -0.5);
        }
        Screen::Initials => initials(g, &s, h),
        Screen::Over => over(g, &s, h),
        _ => {}
    }
    if g.demo {
        h.center("DEMO", h.st.top - 0.2, 1.2, h.st.hud);
        if h.st.blink {
            h.center("PRESS ENTER", -15.5, 1.0, h.st.dim);
        }
    } else if g.screen == Screen::Watch {
        watching(g, h);
    }
}

/// The score and multiplier at the top left (and the Tunnel's gates and speed), the lives as
/// small ships and the bombs as rings at the top right.
fn counters(g: &Game, s: &Stats, h: &mut Hud) {
    let HudStyle {
        left,
        right,
        top,
        hud,
        ..
    } = h.st;
    h.left(&scene::grouped(s.score), left, top - 0.2, 1.3, hud);
    h.left(
        &format!("X{}", s.mult),
        left,
        top - 2.2,
        0.9,
        g.settings.scheme.shard(),
    );
    if s.chain > 0 {
        let gates = format!("GATES X{}", s.chain);
        h.left(&gates, left, top - 5.2, 0.7, render::tunnel::GATE_LIGHT);
    }
    if let Some(speed) = s.speed {
        // The speed bonus: boost for more points; pale gold turning to hot orange (a
        // perceptual gradient).
        let c = (light::light(1.0, 0.95, 0.55, 1.2)).blend(
            light::light(1.0, 0.45, 0.15, 2.4),
            ((speed - 1.0) / 0.45).clamp(0.0, 1.0),
        );
        h.left(&format!("SPEED X{speed:.1}"), left, top - 3.8, 0.7, c);
    }
    for k in 0..s.lives.saturating_sub(1).min(8) {
        let x = right - 0.5 - k as f32 * 1.4;
        let m = sim::body::pose_at(sim::at(x, top + 0.3), core::f32::consts::FRAC_PI_2);
        scene::draw_ship(h.out, m, 0.0, g.time, 0.8);
    }
    for k in 0..s.bombs.min(8) {
        let x = right - 0.5 - k as f32 * 1.3;
        let m = sim::body::pose_at(sim::at(x, top - 1.9), 0.0);
        let ring = light::light(0.55, 0.8, 1.0, 2.0);
        scene::circle(
            h.out,
            0.45,
            12,
            0.0,
            ring,
            scene::Stroke::new(0.04, 0.2, 0.2),
            m,
        );
    }
}

/// Entering initials for a new high score.
fn initials(g: &Game, s: &Stats, h: &mut Hud) {
    let HudStyle {
        hud, dim, blink, ..
    } = h.st;
    h.center("A NEW HIGH SCORE", 6.0, 2.2, g.settings.scheme.shard());
    h.center(&scene::grouped(s.score), 2.5, 1.6, hud);
    for (k, &letter) in g.initials.iter().enumerate() {
        let x = (k as f32 - 1.0) * 3.2;
        let on = k == g.cursor;
        h.at(
            &char::from(letter).to_string(),
            x,
            -2.5,
            2.4,
            if on { hud } else { dim },
            Align::Center,
        );
        if on && blink {
            h.at("_", x, -2.9, 2.4, hud, Align::Center);
        }
    }
    let msg = if g.cursor >= 3 {
        "ENTER TO KEEP"
    } else {
        "TYPE OR UP / DOWN    ENTER TO KEEP"
    };
    h.center(msg, -6.5, 0.8, dim);
}

/// Game over: the score, and whether it is the game's new best.
fn over(g: &Game, s: &Stats, h: &mut Hud) {
    h.center("GAME OVER", 3.0, 3.2, light::light(1.0, 0.35, 0.5, 3.0));
    let score = format!("SCORE {}", scene::grouped(s.score));
    h.center(&score, -0.8, 1.4, h.st.hud);
    if s.score >= g.tables[g.run.mode()].best && s.score > 0 {
        h.center("NEW BEST", -3.2, 1.0, g.settings.scheme.shard());
    }
    if g.over_timer > 1.5 && h.st.blink {
        h.center("PRESS ENTER", -6.5, 1.0, h.st.dim);
    }
}

/// Watching a replay: where in it, how fast, and how it ended.
fn watching(g: &Game, h: &mut Hud) {
    let Some(w) = g.run.watch() else {
        return;
    };
    let HudStyle {
        top,
        hud,
        dim,
        blink,
        ..
    } = h.st;
    let at = format!(
        "REPLAY {} / {}",
        clock(w.next as f32 * sim::DT),
        clock(w.seconds)
    );
    h.center(&at, top - 0.2, 0.9, dim);
    if w.speed > 1 {
        h.center(&format!("X{} SPEED", w.speed), top - 1.8, 0.7, dim);
    }
    if w.other_build {
        h.center("RECORDED ON ANOTHER BUILD", -15.0, 0.7, dim);
    }
    if let Some(end) = &w.end {
        h.center(end, 2.0, 2.0, hud);
        if blink {
            h.center("PRESS ENTER", -1.5, 0.9, dim);
        }
    } else {
        h.center("LEFT / RIGHT SPEED    ESC TO STOP", -16.5, 0.6, dim);
    }
}

/// The debug overlay (F3): the frame rate, the simulation's time and state, and the GPU's
/// time per pass.
fn debug(g: &Game, timings: Option<render::Timings>, h: &mut Hud) {
    let mut lines = vec![format!("FPS {:.0}  SIM {:.2} MS", g.fps, g.sim_ms)];
    match &g.run {
        Run::Plane(r) => {
            let w = &r.world;
            lines.push(format!(
                "ENEMIES {}  BULLETS {}  SHARDS {}",
                w.enemies.len(),
                w.bullets.len(),
                w.shards.len()
            ));
            lines.push(format!(
                "INTENSITY {:.2}  DRIFT {:.1E}",
                w.director.intensity, w.worst_drift
            ));
        }
        Run::Tunnel(r) => {
            let w = &r.world;
            lines.push(format!(
                "ENEMIES {}  SHOTS {}  BOLTS {}  SHARDS {}",
                w.enemies.len(),
                w.shots.len(),
                w.bolts.len(),
                w.shards.len()
            ));
            lines.push(format!("INTENSITY {:.2}  S {:.0}", w.intensity, w.ship.s()));
        }
    }
    if let Some(t) = timings {
        lines.push(format!(
            "GPU COMPUTE {:.2} SCENE {:.2} BLOOM {:.2} POST {:.2}",
            t.compute, t.scene, t.bloom, t.composite
        ));
    }
    let n = lines.len();
    for (i, l) in lines.iter().enumerate() {
        let y = -16.5 + 1.1 * (n - 1 - i) as f32;
        h.left(l, h.st.left, y, 0.6, h.st.dim);
    }
}
