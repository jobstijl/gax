//! The menus: what the menu buttons do on each screen, and the moves between screens.

use crate::game::{ATTRACT_AFTER, DEMO_LENGTH, Game, Mode, Screen};
use crate::input::{Device, Menu};
use crate::store::Row;

/// Whether the player asked to quit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quit {
    No,
    Yes,
}

/// The title's menu.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TitleItem {
    Plane,
    Tunnel,
    Scores,
    Settings,
    Quit,
}

impl TitleItem {
    /// The items, in order.
    pub const ALL: [TitleItem; 5] = [
        TitleItem::Plane,
        TitleItem::Tunnel,
        TitleItem::Scores,
        TitleItem::Settings,
        TitleItem::Quit,
    ];

    /// What the menu shows.
    pub fn label(self) -> &'static str {
        match self {
            TitleItem::Plane => "PLANE",
            TitleItem::Tunnel => "TUNNEL",
            TitleItem::Scores => "HIGH SCORES",
            TitleItem::Settings => "SETTINGS",
            TitleItem::Quit => "QUIT",
        }
    }

    /// Its place in the menu.
    pub fn index(self) -> usize {
        self as usize
    }

    /// The item starting a run of `mode`.
    pub fn of(mode: Mode) -> TitleItem {
        match mode {
            Mode::Plane => TitleItem::Plane,
            Mode::Tunnel => TitleItem::Tunnel,
        }
    }
}

/// The pause menu.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PauseItem {
    Resume,
    EndRun,
}

impl PauseItem {
    /// The items, in order.
    pub const ALL: [PauseItem; 2] = [PauseItem::Resume, PauseItem::EndRun];

    /// What the menu shows.
    pub fn label(self) -> &'static str {
        match self {
            PauseItem::Resume => "RESUME",
            PauseItem::EndRun => "END RUN",
        }
    }
}

/// The menu buttons of a frame, on whatever screen the game is on.
pub fn step(g: &mut Game, menu: Menu, dt: f32) -> Quit {
    if g.demo {
        demo(g, &menu, dt);
        return Quit::No;
    }
    match g.screen {
        Screen::Title => return title(g, &menu, dt),
        Screen::Settings => settings(g, &menu),
        Screen::Scores => scores(g, &menu),
        Screen::Playing => playing(g, &menu),
        Screen::Paused => paused(g, &menu),
        Screen::Initials => initials(g, &menu),
        Screen::Over => over(g, &menu, dt),
        Screen::Watch => watch(g, &menu),
    }
    Quit::No
}

/// Any button at all.
fn any(menu: &Menu) -> bool {
    menu.start || menu.back || menu.up || menu.down || menu.left || menu.right
}

/// Move a menu cursor over `n` items.
fn nav(sel: usize, n: usize, menu: &Menu) -> usize {
    let n = n.max(1);
    if menu.up {
        (sel + n - 1) % n
    } else if menu.down {
        (sel + 1) % n
    } else {
        sel.min(n - 1)
    }
}

/// The attract mode: any button, the end of the demo, or its time, and back to the title.
fn demo(g: &mut Game, menu: &Menu, dt: f32) {
    g.idle += dt;
    let ended = g.run.watch().is_some_and(|w| w.end.is_some())
        || matches!(g.screen, Screen::Over | Screen::Title);
    if any(menu) || ended || g.idle > DEMO_LENGTH {
        g.back_to_title();
    }
}

/// The title: a game, the tables, the settings, or quitting (back moves to QUIT; only
/// confirming it quits). Left alone, the attract mode starts.
fn title(g: &mut Game, menu: &Menu, dt: f32) -> Quit {
    g.idle = if any(menu) { 0.0 } else { g.idle + dt };
    if g.idle > ATTRACT_AFTER {
        g.start_demo();
        return Quit::No;
    }
    g.sel = nav(g.sel, TitleItem::ALL.len(), menu);
    if menu.back {
        g.sel = TitleItem::Quit.index();
    } else if menu.start {
        match TitleItem::ALL[g.sel] {
            TitleItem::Plane => g.start(Mode::Plane),
            TitleItem::Tunnel => g.start(Mode::Tunnel),
            TitleItem::Scores => {
                g.screen = Screen::Scores;
                g.sel = 0;
            }
            TitleItem::Settings => {
                g.screen = Screen::Settings;
                g.sel = 0;
            }
            TitleItem::Quit => return Quit::Yes,
        }
    }
    Quit::No
}

/// The settings: a row per setting, then BACK. Left and right change a row (Enter steps a
/// toggle or a cycle too); every change is saved.
fn settings(g: &mut Game, menu: &Menu) {
    let back = Row::ALL.len();
    g.sel = nav(g.sel, back + 1, menu);
    if let Some(&row) = Row::ALL.get(g.sel) {
        let mut by = i32::from(menu.right) - i32::from(menu.left);
        if by == 0 && menu.start && row.kind().steps_on_enter() {
            by = 1;
        }
        if by != 0 {
            g.settings.adjust(row, by);
            g.store.save_settings(&g.settings);
        }
    }
    if menu.back || (menu.start && g.sel == back) {
        g.screen = Screen::Title;
        g.sel = TitleItem::Settings.index();
    }
}

/// The high scores: left and right switch between the games' tables, Enter watches a run.
fn scores(g: &mut Game, menu: &Menu) {
    if menu.left || menu.right {
        g.table = g.table.other();
        g.sel = 0;
        g.highlight = None;
    }
    let entries = &g.tables[g.table].scores.entries;
    g.sel = nav(g.sel, entries.len(), menu);
    let file = entries.get(g.sel).map(|e| e.replay.clone());
    if menu.start {
        if let Some(replay) = file.and_then(|f| g.table.replay(&g.store, &f)) {
            g.watch(replay);
        }
    } else if menu.back {
        g.screen = Screen::Title;
        g.sel = TitleItem::Scores.index();
    }
}

/// In play, back pauses.
fn playing(g: &mut Game, menu: &Menu) {
    if menu.back {
        g.screen = Screen::Paused;
        g.sel = 0;
    }
}

/// The pause menu: resume (or back), or end the run.
fn paused(g: &mut Game, menu: &Menu) {
    g.sel = nav(g.sel, PauseItem::ALL.len(), menu);
    let item = PauseItem::ALL[g.sel];
    if menu.back || (menu.start && item == PauseItem::Resume) {
        g.screen = Screen::Playing;
    } else if menu.start {
        g.end();
    }
}

/// Initials: typed, or chosen with up and down (the keyboard's WASD type) and moved through
/// with left and right; on a pad, A confirms a letter and moves on. Enter keeps the run.
fn initials(g: &mut Game, menu: &Menu) {
    if let Some(c) = menu.letter {
        if g.cursor < 3 {
            g.initials[g.cursor] = c as u8;
            g.cursor += 1;
        }
    } else if g.cursor < 3 && (menu.up || menu.down) {
        let l = &mut g.initials[g.cursor];
        let k = i32::from(*l - b'A') + if menu.up { 1 } else { -1 };
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
        if g.cursor < 2 && menu.letter.is_none() && g.device == Device::Pad {
            g.cursor += 1;
        } else {
            let name = String::from_utf8_lossy(&g.initials).into_owned();
            g.enter(&name);
        }
    }
}

/// Game over: after a moment, Enter or back returns to the title, on the game just played.
fn over(g: &mut Game, menu: &Menu, dt: f32) {
    g.over_timer += dt;
    if g.over_timer > 1.5 && (menu.start || menu.back) {
        g.over_timer = 0.0;
        g.sel = TitleItem::of(g.run.mode()).index();
        g.back_to_title();
    }
}

/// Watching: back (or Enter at the end) returns to the table; left and right change the
/// speed.
fn watch(g: &mut Game, menu: &Menu) {
    let ended = g.run.watch().is_some_and(|w| w.end.is_some());
    if menu.back || (ended && menu.start) {
        g.stop_watching();
    } else if (menu.right || menu.left)
        && let Some(w) = g.run.watch_mut()
        && w.end.is_none()
    {
        w.speed = if menu.right {
            (w.speed * 2).min(8)
        } else {
            (w.speed / 2).max(1)
        };
    }
}
