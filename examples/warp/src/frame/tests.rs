//! The game loop without a window: runs, menus and replays through `advance`.

use super::*;
use crate::game::{Mode, PerMode, Table};
use crate::menus::TitleItem;
use crate::store::{Row, Store};
use std::path::PathBuf;

const ASPECT: f32 = 16.0 / 9.0;

/// A game and its sound, driven a frame at a time.
struct Rig {
    g: Game,
    sound: Sound,
    /// The store's directory, removed at the end.
    dir: Option<PathBuf>,
}

impl Rig {
    /// A game that keeps nothing.
    fn new() -> Rig {
        Rig {
            g: Game::new(Store::new(None)),
            sound: Sound::new(),
            dir: None,
        }
    }

    /// A game keeping its tables and replays in a fresh directory named for `test`.
    fn stored(test: &str) -> Rig {
        let dir = std::env::temp_dir().join(format!("warp-test-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Rig {
            g: Game::new(Store::new(Some(dir.clone()))),
            sound: Sound::new(),
            dir: Some(dir),
        }
    }

    /// A frame of `dt` with the Plane's `input` and the menu buttons `menu`.
    fn frame(&mut self, input: sim::Input, menu: Menu, dt: f32) -> Quit {
        advance(&mut self.g, input, menu, dt, ASPECT, &mut self.sound)
    }

    /// A 16 ms frame with the menu buttons `menu` and nothing else.
    fn press(&mut self, menu: Menu) -> Quit {
        self.frame(sim::Input::default(), menu, 0.016)
    }

    /// `n` quiet frames.
    fn wait(&mut self, n: usize) {
        for _ in 0..n {
            self.press(Menu::default());
        }
    }

    /// The bot plays until `seconds` pass or the run ends, at uneven frames of 7 to 20 ms.
    fn bot_plays(&mut self, seconds: f32) {
        let (mut t, mut frame) = (0.0, 0u32);
        while self.g.screen == Screen::Playing && t < seconds {
            let dt = 0.007 + 0.013 * ((frame * 7919) % 97) as f32 / 97.0;
            frame += 1;
            t += dt;
            let (input, flight) = crate::bot::input(&self.g.run, t);
            self.g.flight = flight;
            self.frame(input, Menu::default(), dt);
        }
    }

    /// Pause and choose END RUN.
    fn end_run(&mut self) {
        self.press(back());
        assert_eq!(self.g.screen, Screen::Paused);
        self.press(down());
        self.press(start());
    }

    /// Type `name` and keep it.
    fn initials(&mut self, name: &str) {
        assert_eq!(self.g.screen, Screen::Initials);
        for c in name.chars() {
            self.press(Menu {
                letter: Some(c),
                ..Menu::default()
            });
        }
        self.press(start());
    }

    /// Watch fast until the replay ends; what it says then.
    fn watch_to_the_end(&mut self) -> String {
        assert_eq!(self.g.screen, Screen::Watch);
        self.g.run.watch_mut().unwrap().speed = 8;
        while self.g.run.watch().unwrap().end.is_none() {
            self.frame(sim::Input::default(), Menu::default(), 0.05);
        }
        self.g.run.watch().unwrap().end.clone().unwrap()
    }

    /// The Plane run's world.
    fn plane(&mut self) -> &mut sim::World {
        &mut self.g.run.plane().expect("a Plane run").world
    }
}

impl Drop for Rig {
    fn drop(&mut self) {
        if let Some(d) = &self.dir {
            let _ = std::fs::remove_dir_all(d);
        }
    }
}

fn start() -> Menu {
    Menu {
        start: true,
        ..Menu::default()
    }
}

fn back() -> Menu {
    Menu {
        back: true,
        ..Menu::default()
    }
}

fn up() -> Menu {
    Menu {
        up: true,
        ..Menu::default()
    }
}

fn down() -> Menu {
    Menu {
        down: true,
        ..Menu::default()
    }
}

fn left() -> Menu {
    Menu {
        left: true,
        ..Menu::default()
    }
}

fn right() -> Menu {
    Menu {
        right: true,
        ..Menu::default()
    }
}

/// The whole Plane loop: a bot plays at uneven frame rates (with hit-stops and a death or
/// two), ends the run from the pause menu, enters initials, and then watches its own replay,
/// which must reach its end without diverging, and then holds still.
#[test]
fn a_played_run_goes_on_the_table_and_replays_exactly() {
    let mut r = Rig::stored("flow");
    r.g.start_seeded(Mode::Plane, 99);
    r.bot_plays(70.0);
    if r.g.screen == Screen::Playing {
        r.end_run();
    }
    let score = r.g.run.ledger().score;
    assert!(score > 0);
    r.initials("WRP");
    assert_eq!((r.g.screen, r.g.highlight), (Screen::Scores, Some(0)));
    let e = r.g.tables.plane.scores.entries[0].clone();
    assert_eq!((e.name.as_str(), e.score), ("WRP", score));
    assert_eq!(r.g.tables.plane.best, score);
    r.press(start());
    assert_eq!(r.watch_to_the_end(), "END OF REPLAY");
    assert_eq!(r.g.run.ledger().score, score);
    // The world holds still after the end (it used to tick on without input).
    let ticks = r.plane().tick;
    r.wait(30);
    assert_eq!(r.plane().tick, ticks);
}

/// The Tunnel through the loop: chosen from the title, flown by the bot at uneven frame
/// rates, ended from the pause menu, entered on the Tunnel's table, and watched back, which
/// must reach the end with the same score; then back to the table.
#[test]
fn a_tunnel_run_through_the_loop() {
    let mut r = Rig::stored("tunnel");
    r.press(down());
    r.press(start());
    assert_eq!(
        (r.g.run.mode(), r.g.screen),
        (Mode::Tunnel, Screen::Playing)
    );
    r.bot_plays(25.0);
    let w = &r.g.run.tunnel().unwrap().world;
    let score = w.ledger.score;
    assert!(score > 0 && w.ship.s() > 300.0, "{score} at {}", w.ship.s());
    if r.g.screen == Screen::Playing {
        r.end_run();
    }
    r.initials("TUN");
    assert_eq!((r.g.screen, r.g.table), (Screen::Scores, Mode::Tunnel));
    assert_eq!(r.g.tables.tunnel.scores.entries[0].score, score);
    assert_eq!(
        r.g.tables.plane.scores.entries,
        [],
        "the Plane's table is separate"
    );
    assert_eq!((r.g.tables.tunnel.best, r.g.tables.plane.best), (score, 0));
    r.press(start());
    assert_eq!(r.g.run.mode(), Mode::Tunnel);
    assert_eq!(r.watch_to_the_end(), "END OF REPLAY");
    assert_eq!(r.g.run.ledger().score, score);
    r.press(back());
    assert_eq!((r.g.run.mode(), r.g.screen), (Mode::Plane, Screen::Scores));
}

/// The attract mode: idle on the title, a demo starts (the bot, with no table yet), any
/// input ends it; the next demo plays the other game. Demos never set a best.
#[test]
fn the_attract_mode_plays_demos() {
    let mut r = Rig::new();
    let mut modes = Vec::new();
    for _ in 0..2 {
        let mut t = 0.0;
        while !r.g.demo && t < 30.0 {
            t += 0.05;
            r.frame(sim::Input::default(), Menu::default(), 0.05);
        }
        assert!(r.g.demo, "no demo after {t} s");
        modes.push(r.g.run.mode());
        for _ in 0..500 {
            r.g.time += 0.02;
            let (input, flight) = crate::bot::input(&r.g.run, r.g.time);
            r.g.flight = flight;
            r.frame(input, Menu::default(), 0.02);
        }
        assert!(
            r.g.demo && r.g.run.ledger().score > 0,
            "the demo does not play"
        );
        r.press(start());
        assert_eq!(r.g.screen, Screen::Title);
        assert!(!r.g.demo);
    }
    assert_eq!(modes, [Mode::Tunnel, Mode::Plane]);
}

/// A demo that ends (or is lost) leaves the bests alone; a run of the player's sets its own
/// game's best only.
#[test]
fn only_the_players_runs_set_a_best() {
    let mut r = Rig::new();
    r.g.tables = PerMode::<Table>::default();
    r.g.demo_turn = Mode::Plane;
    r.g.start_demo();
    assert!(r.g.demo && r.g.bot);
    r.wait(10);
    r.g.run.plane().unwrap().world.ledger.score = 5_000;
    r.g.end();
    assert_eq!(r.g.screen, Screen::Title);
    assert_eq!((r.g.tables.plane.best, r.g.tables.tunnel.best), (0, 0));
    // The player's Tunnel run: its best, not the Plane's.
    r.g.tables.plane.best = 10_000;
    r.g.start(Mode::Tunnel);
    r.wait(10);
    if let Run::Tunnel(run) = &mut r.g.run {
        run.world.ledger.score = 5_000;
    }
    r.end_run();
    assert_eq!(r.g.tables.tunnel.best, 5_000);
    assert_eq!(r.g.tables.plane.best, 10_000);
}

/// A bomb pressed on the title, or while paused, is not kept for the run: it used to go off
/// on the first tick of the next run, or on resuming. Likewise the Tunnel's bomb and roll.
#[test]
fn bombs_wait_only_in_play() {
    let mut r = Rig::new();
    let bomb = sim::Input {
        bomb: true,
        ..sim::Input::default()
    };
    r.frame(bomb, Menu::default(), 0.016);
    r.press(start());
    r.wait(30);
    assert_eq!(r.g.run.ledger().bombs, 3, "a bomb from the title");
    r.press(back());
    r.frame(bomb, Menu::default(), 0.016);
    r.press(back());
    r.wait(30);
    assert_eq!(r.g.screen, Screen::Playing);
    assert_eq!(r.g.run.ledger().bombs, 3, "a bomb from the pause menu");
    // And one in play goes off.
    r.frame(bomb, Menu::default(), 0.016);
    r.wait(2);
    assert_eq!(r.g.run.ledger().bombs, 2);

    r.g.start(Mode::Tunnel);
    r.wait(30);
    r.press(back());
    r.g.flight = Flight {
        bomb: true,
        roll: 1,
        ..Flight::default()
    };
    r.press(Menu::default());
    r.g.flight = Flight::default();
    r.press(back());
    r.wait(30);
    let w = &r.g.run.tunnel().unwrap().world;
    assert_eq!(w.ledger.bombs, 3, "a Tunnel bomb from the pause menu");
    assert!(w.ship.roll.is_none(), "a roll from the pause menu");
}

/// On the title, back moves to QUIT, and only confirming it quits; backing out of the other
/// screens comes back to the title, not out of the game.
#[test]
fn the_title_quits_only_on_quit() {
    let mut r = Rig::new();
    assert_eq!(r.press(back()), Quit::No);
    assert_eq!(TitleItem::ALL[r.g.sel], TitleItem::Quit);
    assert_eq!(r.press(back()), Quit::No);
    r.press(up());
    r.press(start());
    assert_eq!(r.g.screen, Screen::Settings);
    assert_eq!(r.press(back()), Quit::No);
    assert_eq!(
        (r.g.screen, TitleItem::ALL[r.g.sel]),
        (Screen::Title, TitleItem::Settings)
    );
    assert_eq!(r.press(back()), Quit::No);
    assert_eq!(r.g.screen, Screen::Title);
    assert_eq!(r.press(start()), Quit::Yes);
}

/// Settings: up and down move between the rows and BACK, left and right change a row, Enter
/// flips a toggle, and BACK returns to the title on SETTINGS.
#[test]
fn settings_are_navigated_and_changed() {
    let mut r = Rig::new();
    r.g.sel = TitleItem::Settings.index();
    r.press(start());
    assert_eq!((r.g.screen, r.g.sel), (Screen::Settings, 0));
    r.press(left());
    assert_eq!(r.g.settings.shake, 3);
    r.press(down());
    r.press(start());
    assert!(r.g.settings.reduced_flashes);
    let master = Row::ALL
        .iter()
        .position(|&x| x == Row::MasterVolume)
        .unwrap();
    while r.g.sel != master {
        r.press(down());
    }
    r.press(start());
    assert_eq!(r.g.settings.master, 8, "Enter does not step a volume");
    r.press(right());
    assert_eq!(r.g.settings.master, 9);
    // Up from the first row wraps to BACK.
    while r.g.sel != 0 {
        r.press(up());
    }
    r.press(up());
    assert_eq!(r.g.sel, Row::ALL.len());
    r.press(start());
    assert_eq!(
        (r.g.screen, TitleItem::ALL[r.g.sel]),
        (Screen::Title, TitleItem::Settings)
    );
}

/// The scores screen: left and right switch between the games' tables.
#[test]
fn scores_switch_tables() {
    let mut r = Rig::new();
    r.g.sel = TitleItem::Scores.index();
    r.press(start());
    assert_eq!((r.g.screen, r.g.table), (Screen::Scores, Mode::Plane));
    r.g.highlight = Some(1);
    r.press(right());
    assert_eq!((r.g.table, r.g.highlight), (Mode::Tunnel, None));
    r.press(left());
    assert_eq!(r.g.table, Mode::Plane);
    r.press(back());
    assert_eq!(
        (r.g.screen, TitleItem::ALL[r.g.sel]),
        (Screen::Title, TitleItem::Scores)
    );
}

/// Initials on a pad: up and down choose a letter, A confirms it and moves on, and A on the
/// last keeps the run.
#[test]
fn initials_on_a_pad() {
    let mut r = Rig::stored("pad");
    r.g.start(Mode::Plane);
    r.wait(10);
    r.plane().ledger.score = 1234;
    r.end_run();
    assert_eq!(r.g.screen, Screen::Initials);
    r.g.device = crate::input::Device::Pad;
    r.press(up());
    r.press(start());
    r.press(down());
    r.press(start());
    assert_eq!((r.g.initials, r.g.cursor), (*b"BZA", 2));
    r.press(start());
    assert_eq!(r.g.screen, Screen::Scores);
    assert_eq!(r.g.tables.plane.scores.entries[0].name, "BZA");
}

/// Watching: left and right halve and double the speed (1 to 8); a hit-stop in fast forward
/// lasts its time in the world, not a speed squared less.
#[test]
fn watching_changes_speed() {
    let mut r = Rig::stored("watch");
    r.g.start_seeded(Mode::Plane, 5);
    r.bot_plays(4.0);
    // A score to enter (set after the last hash, it does not show in the part watched).
    r.plane().ledger.score += 1;
    r.end_run();
    r.initials("SPD");
    r.press(start());
    assert_eq!(r.g.screen, Screen::Watch);
    for want in [2, 4, 8, 8] {
        r.press(right());
        assert_eq!(r.g.run.watch().unwrap().speed, want);
    }
    r.press(left());
    assert_eq!(r.g.run.watch().unwrap().speed, 4);
    // At 4 ticks a tick, a 16 ms frame is 64 ms of the world: seven whole ticks, all of them
    // in the hit-stop, which has seven ticks less left.
    r.g.hitstop = 0.15;
    r.g.acc = 0.0;
    r.press(Menu::default());
    let left_over = 0.15 - 7.0 * sim::DT;
    assert!((r.g.hitstop - left_over).abs() < 1e-4, "{}", r.g.hitstop);
}

/// END RUN from the pause menu ends a run of either game (without a score: game over).
#[test]
fn the_pause_menu_ends_either_game() {
    for mode in [Mode::Plane, Mode::Tunnel] {
        let mut r = Rig::new();
        r.g.start(mode);
        r.wait(10);
        r.end_run();
        assert_eq!((r.g.screen, r.g.run.phase()), (Screen::Over, Phase::Over));
        assert_eq!(r.g.run.mode(), mode);
        r.wait(100);
        r.press(start());
        assert_eq!(r.g.screen, Screen::Title);
        assert_eq!(TitleItem::ALL[r.g.sel], TitleItem::of(mode));
    }
}
