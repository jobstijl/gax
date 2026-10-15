//! What the game keeps between sessions: settings, the high-score table, and the replays of
//! the runs on it. Plain text files (and the replays' own format) in the user's data
//! directory, or `$WARP_DATA`.

use crate::sim::replay::{Record, Replay};
use std::path::{Path, PathBuf};

/// The number of entries in the high-score table.
pub const TABLE: usize = 10;

/// The data directory: `$WARP_DATA`, or the platform's per-user data directory.
pub fn data_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("WARP_DATA") {
        return Some(PathBuf::from(d));
    }
    let home = std::env::var_os("HOME").map(PathBuf::from);
    if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(|a| PathBuf::from(a).join("warp"))
    } else if cfg!(target_os = "macos") {
        home.map(|h| h.join("Library/Application Support/warp"))
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| home.map(|h| h.join(".local/share")))
            .map(|d| d.join("warp"))
    }
}

/// The colour schemes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Scheme {
    /// The neon palette.
    #[default]
    Standard,
    /// Safe for red–green colour blindness (protan, deutan).
    RedGreen,
    /// Safe for blue–yellow colour blindness (tritan).
    BlueYellow,
}

impl Scheme {
    const ALL: [Scheme; 3] = [Scheme::Standard, Scheme::RedGreen, Scheme::BlueYellow];

    /// The name in the settings file and menu.
    pub fn name(self) -> &'static str {
        match self {
            Scheme::Standard => "STANDARD",
            Scheme::RedGreen => "RED-GREEN SAFE",
            Scheme::BlueYellow => "BLUE-YELLOW SAFE",
        }
    }

    /// The next (or previous) scheme.
    pub fn step(self, by: i32) -> Scheme {
        let i = Scheme::ALL.iter().position(|s| *s == self).unwrap_or(0) as i32;
        Scheme::ALL[(i + by).rem_euclid(Scheme::ALL.len() as i32) as usize]
    }
}

/// The player's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    /// Screen shake, in quarters (0 is off, 4 is full).
    pub shake: u8,
    /// Fewer and softer full-screen flashes.
    pub reduced_flashes: bool,
    /// Less motion: no screen shake, shock ripples or lensing.
    pub reduced_motion: bool,
    /// The colour scheme.
    pub scheme: Scheme,
    /// Volumes, `0..=10`.
    pub master: u8,
    /// Music volume.
    pub music: u8,
    /// Effects volume.
    pub sfx: u8,
    /// Shots, hits, kills and pickups on the music's beat.
    pub on_beat: bool,
    /// The Tunnel's vertical field of view, in degrees.
    pub fov: u8,
    /// The Tunnel's camera rolls with the track (off: it stays level).
    pub camera_roll: bool,
    /// How generously the Tunnel's reticle locks on: 0 off, 1 low, 2 high.
    pub aim_assist: u8,
    /// Full screen.
    pub fullscreen: bool,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            shake: 4,
            reduced_flashes: false,
            reduced_motion: false,
            scheme: Scheme::Standard,
            master: 8,
            music: 8,
            sfx: 8,
            on_beat: false,
            fov: 85,
            camera_roll: false,
            aim_assist: 1,
            fullscreen: false,
        }
    }
}

/// A row of the settings menu: a setting, its label, how it changes, and its key in the
/// settings file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Row {
    Shake,
    Flashes,
    Motion,
    Colours,
    MasterVolume,
    MusicVolume,
    EffectsVolume,
    OnBeat,
    FieldOfView,
    CameraRoll,
    AimAssist,
    FullScreen,
}

/// How a row's value changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowKind {
    /// On or off: any step flips it.
    Toggle,
    /// One of a few, wrapping round.
    Cycle,
    /// A volume, `0..=10`, shown as bars.
    Level,
    /// A number in steps, held at its ends.
    Step,
}

impl RowKind {
    /// Whether Enter steps it (as right does).
    pub fn steps_on_enter(self) -> bool {
        matches!(self, RowKind::Toggle | RowKind::Cycle)
    }
}

/// A row's value as the menu shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    /// Words or a number.
    Text(String),
    /// A volume's level, out of 10 (drawn as bars).
    Level(u8),
}

impl Row {
    /// The rows, in the menu's (and the file's) order.
    pub const ALL: [Row; 12] = [
        Row::Shake,
        Row::Flashes,
        Row::Motion,
        Row::Colours,
        Row::MasterVolume,
        Row::MusicVolume,
        Row::EffectsVolume,
        Row::OnBeat,
        Row::FieldOfView,
        Row::CameraRoll,
        Row::AimAssist,
        Row::FullScreen,
    ];

    /// What the menu shows.
    pub fn label(self) -> &'static str {
        match self {
            Row::Shake => "SCREEN SHAKE",
            Row::Flashes => "FLASHES",
            Row::Motion => "MOTION",
            Row::Colours => "COLOURS",
            Row::MasterVolume => "MASTER VOLUME",
            Row::MusicVolume => "MUSIC",
            Row::EffectsVolume => "EFFECTS",
            Row::OnBeat => "EFFECTS ON THE BEAT",
            Row::FieldOfView => "TUNNEL FIELD OF VIEW",
            Row::CameraRoll => "TUNNEL CAMERA ROLL",
            Row::AimAssist => "TUNNEL AIM ASSIST",
            Row::FullScreen => "FULL SCREEN",
        }
    }

    /// Its key in the settings file.
    pub fn key(self) -> &'static str {
        match self {
            Row::Shake => "shake",
            Row::Flashes => "reduced_flashes",
            Row::Motion => "reduced_motion",
            Row::Colours => "scheme",
            Row::MasterVolume => "master",
            Row::MusicVolume => "music",
            Row::EffectsVolume => "sfx",
            Row::OnBeat => "on_beat",
            Row::FieldOfView => "fov",
            Row::CameraRoll => "camera_roll",
            Row::AimAssist => "aim_assist",
            Row::FullScreen => "fullscreen",
        }
    }

    /// How it changes.
    pub fn kind(self) -> RowKind {
        match self {
            Row::Shake | Row::FieldOfView => RowKind::Step,
            Row::Colours | Row::AimAssist => RowKind::Cycle,
            Row::MasterVolume | Row::MusicVolume | Row::EffectsVolume => RowKind::Level,
            Row::Flashes | Row::Motion | Row::OnBeat | Row::CameraRoll | Row::FullScreen => {
                RowKind::Toggle
            }
        }
    }
}

/// The names of the aim assist's levels.
const ASSIST: [&str; 3] = ["OFF", "LOW", "HIGH"];

impl Settings {
    /// The Tunnel's lock-on margin (screen units beyond an enemy's apparent size).
    pub fn aim_reach(&self) -> f32 {
        [0.25, 1.1, 2.2][usize::from(self.aim_assist.min(2))]
    }

    /// The flag a toggle row sets.
    fn flag(&mut self, row: Row) -> Option<&mut bool> {
        match row {
            Row::Flashes => Some(&mut self.reduced_flashes),
            Row::Motion => Some(&mut self.reduced_motion),
            Row::OnBeat => Some(&mut self.on_beat),
            Row::CameraRoll => Some(&mut self.camera_roll),
            Row::FullScreen => Some(&mut self.fullscreen),
            _ => None,
        }
    }

    /// The number a step, level or cycle row sets, with its range.
    fn number(&mut self, row: Row) -> Option<(&mut u8, u8, u8)> {
        match row {
            Row::Shake => Some((&mut self.shake, 0, 4)),
            Row::MasterVolume => Some((&mut self.master, 0, 10)),
            Row::MusicVolume => Some((&mut self.music, 0, 10)),
            Row::EffectsVolume => Some((&mut self.sfx, 0, 10)),
            Row::FieldOfView => Some((&mut self.fov, 60, 110)),
            Row::AimAssist => Some((&mut self.aim_assist, 0, 2)),
            _ => None,
        }
    }

    /// A row's value as the menu shows it.
    pub fn value(&self, row: Row) -> Value {
        let on_off = |on: bool| if on { "ON" } else { "OFF" };
        let full = |reduced: bool| if reduced { "REDUCED" } else { "FULL" };
        Value::Text(match row {
            Row::Shake if self.shake == 0 => "OFF".into(),
            Row::Shake => format!("{}%", 25 * u32::from(self.shake)),
            Row::Flashes => full(self.reduced_flashes).into(),
            Row::Motion => full(self.reduced_motion).into(),
            Row::Colours => self.scheme.name().into(),
            Row::MasterVolume => return Value::Level(self.master),
            Row::MusicVolume => return Value::Level(self.music),
            Row::EffectsVolume => return Value::Level(self.sfx),
            Row::OnBeat => on_off(self.on_beat).into(),
            Row::FieldOfView => self.fov.to_string(),
            Row::CameraRoll => if self.camera_roll { "FOLLOW" } else { "LEVEL" }.into(),
            Row::AimAssist => ASSIST[usize::from(self.aim_assist.min(2))].into(),
            Row::FullScreen => on_off(self.fullscreen).into(),
        })
    }

    /// Step a row's value by `by`: a toggle flips on any step, a cycle wraps, the field of
    /// view moves 5 degrees a step, and the rest are held at their ends.
    pub fn adjust(&mut self, row: Row, by: i32) {
        if row == Row::Colours {
            self.scheme = self.scheme.step(by);
        } else if let Some(flag) = self.flag(row) {
            *flag = !*flag;
        } else if let Some((n, lo, hi)) = self.number(row) {
            let by = if row == Row::FieldOfView { 5 * by } else { by };
            let next = i32::from(*n) + by;
            *n = if row.kind() == RowKind::Cycle {
                (next - i32::from(lo)).rem_euclid(i32::from(hi - lo) + 1) + i32::from(lo)
            } else {
                next.clamp(i32::from(lo), i32::from(hi))
            } as u8;
        }
    }

    /// Linear gains of the effects and the music (volumes on a perceptual, squared, curve).
    pub fn gains(&self) -> [f32; 2] {
        let v = |x: u8| (f32::from(x) / 10.0).powi(2);
        [v(self.master) * v(self.sfx), v(self.master) * v(self.music)]
    }

    /// A row's value as the settings file keeps it.
    fn stored(mut self, row: Row) -> String {
        if row == Row::Colours {
            self.scheme.name().into()
        } else if let Some(flag) = self.flag(row) {
            flag.to_string()
        } else {
            self.number(row)
                .map_or(String::new(), |(n, ..)| n.to_string())
        }
    }

    /// Set a row from the settings file's `text`; malformed text keeps the value, a number is
    /// held in its range, and an unknown scheme is the standard one.
    fn restore(&mut self, row: Row, text: &str) {
        if row == Row::Colours {
            self.scheme = Scheme::ALL
                .into_iter()
                .find(|x| x.name() == text)
                .unwrap_or_default();
        } else if let Some(flag) = self.flag(row) {
            *flag = text == "true";
        } else if let Some((n, lo, hi)) = self.number(row) {
            *n = text.parse::<u8>().map_or(*n, |x| x.clamp(lo, hi));
        }
    }

    /// The text form: a `key = value` line per row.
    pub fn to_text(self) -> String {
        Row::ALL
            .iter()
            .map(|&row| format!("{} = {}\n", row.key(), self.stored(row)))
            .collect()
    }

    /// Read the text form; unknown or malformed lines keep their defaults.
    pub fn from_text(text: &str) -> Settings {
        let mut s = Settings::default();
        for line in text.lines() {
            let Some((k, v)) = line.split_once('=') else {
                continue;
            };
            if let Some(&row) = Row::ALL.iter().find(|r| r.key() == k.trim()) {
                s.restore(row, v.trim());
            }
        }
        s
    }
}

/// An entry of the high-score table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// The score.
    pub score: u64,
    /// Three letters.
    pub name: String,
    /// Seconds the run lasted.
    pub seconds: u32,
    /// The replay's file name in `replays/` (empty if none).
    pub replay: String,
}

/// The high-score table, best first.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Scores {
    /// The entries, at most `TABLE`.
    pub entries: Vec<Entry>,
}

impl Scores {
    /// Where `score` would land in the table, if it makes it.
    pub fn rank(&self, score: u64) -> Option<usize> {
        if score == 0 {
            return None;
        }
        let r = self.entries.iter().take_while(|e| e.score >= score).count();
        (r < TABLE).then_some(r)
    }

    /// Insert an entry; returns its rank and the entry pushed off the end, if any.
    pub fn insert(&mut self, e: Entry) -> Option<(usize, Option<Entry>)> {
        let r = self.rank(e.score)?;
        self.entries.insert(r, e);
        let dropped = (self.entries.len() > TABLE).then(|| self.entries.pop().unwrap());
        Some((r, dropped))
    }

    /// The text form: `score name seconds replay` per line.
    pub fn to_text(&self) -> String {
        self.entries
            .iter()
            .map(|e| format!("{} {} {} {}\n", e.score, e.name, e.seconds, e.replay))
            .collect()
    }

    /// Read the text form, skipping malformed lines.
    pub fn from_text(text: &str) -> Scores {
        let mut entries: Vec<Entry> = text
            .lines()
            .filter_map(|l| {
                let mut w = l.split_whitespace();
                Some(Entry {
                    score: w.next()?.parse().ok()?,
                    name: w.next()?.chars().take(3).collect(),
                    seconds: w.next()?.parse().ok()?,
                    replay: w.next().unwrap_or("").to_string(),
                })
            })
            .collect();
        entries.sort_by_key(|e| std::cmp::Reverse(e.score));
        entries.truncate(TABLE);
        Scores { entries }
    }
}

/// The store on disk (or nowhere, if there is no data directory: then nothing persists).
pub struct Store {
    dir: Option<PathBuf>,
}

impl Store {
    /// The store in `dir`.
    pub fn new(dir: Option<PathBuf>) -> Store {
        Store { dir }
    }

    /// The store in the user's data directory.
    pub fn open() -> Store {
        Store::new(data_dir())
    }

    /// The text of `name`: `None` if there is no such file (or no directory), an error if
    /// there is one that cannot be read.
    fn read(&self, name: &str) -> Result<Option<String>, String> {
        let Some(dir) = &self.dir else {
            return Ok(None);
        };
        let path = dir.join(name);
        match std::fs::read_to_string(&path) {
            Ok(text) => Ok(Some(text)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }

    /// Write `bytes` to `name` (through a temporary file, so a crash never leaves half a file).
    fn write(&self, name: &str, bytes: &[u8]) {
        let Some(dir) = &self.dir else {
            return;
        };
        let path = dir.join(name);
        let tmp = temporary(&path);
        let ok = path
            .parent()
            .is_some_and(|p| std::fs::create_dir_all(p).is_ok())
            && std::fs::write(&tmp, bytes).is_ok()
            && std::fs::rename(&tmp, &path).is_ok();
        if !ok {
            eprintln!("warp: could not write {}", path.display());
        }
    }

    /// The settings (defaults if none are saved).
    pub fn settings(&self) -> Settings {
        match self.read("settings.txt") {
            Ok(text) => text.map(|t| Settings::from_text(&t)).unwrap_or_default(),
            Err(e) => {
                eprintln!("warp: {e}; using the default settings");
                Settings::default()
            }
        }
    }

    /// Save the settings.
    pub fn save_settings(&self, s: &Settings) {
        self.write("settings.txt", s.to_text().as_bytes());
    }

    /// The high-score table of the game whose replays are `I` (empty if it cannot be read,
    /// which is logged).
    pub fn scores<I: Record>(&self) -> Scores {
        self.table::<I>().unwrap_or_else(|e| {
            eprintln!("warp: {e}");
            Scores::default()
        })
    }

    /// The high-score table of the game whose replays are `I`, or why it cannot be read.
    fn table<I: Record>(&self) -> Result<Scores, String> {
        Ok(self
            .read(I::TABLE)?
            .map(|t| Scores::from_text(&t))
            .unwrap_or_default())
    }

    /// Record a finished run: always as `replays/last.<ext>`; if it makes the table, as an
    /// entry named `name` with its own replay. Returns its rank. A table that is there but
    /// cannot be read is left alone (rewriting it would lose its entries).
    pub fn finish<I: Record>(&self, replay: &Replay<I>, name: Option<&str>) -> Option<usize> {
        let bytes = replay.encode();
        self.write(&format!("replays/last.{}", I::EXT), &bytes);
        let name = name?;
        let mut scores = self
            .table::<I>()
            .map_err(|e| eprintln!("warp: {e}; the run is not entered"))
            .ok()?;
        let file = format!("{}-{}.{}", replay.score, replay.seed, I::EXT);
        let (rank, dropped) = scores.insert(Entry {
            score: replay.score,
            name: name.to_string(),
            seconds: replay.seconds() as u32,
            replay: file.clone(),
        })?;
        self.write(&format!("replays/{file}"), &bytes);
        if let (Some(d), Some(dir)) = (dropped, &self.dir)
            && !d.replay.is_empty()
            && !scores.entries.iter().any(|e| e.replay == d.replay)
        {
            let _ = std::fs::remove_file(dir.join("replays").join(&d.replay));
        }
        self.write(I::TABLE, scores.to_text().as_bytes());
        Some(rank)
    }

    /// Load a replay by its file name in `replays/`.
    pub fn replay<I: Record>(&self, file: &str) -> Option<Replay<I>> {
        let dir = self.dir.as_ref()?;
        load_replay(&dir.join("replays").join(file)).ok()
    }
}

/// The temporary file a write to `path` goes through: `path` with `.tmp` appended (not its
/// extension replaced, which would give `last.warp` and `last.tunnel` the same one).
fn temporary(path: &Path) -> PathBuf {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    PathBuf::from(tmp)
}

/// Load a replay file.
pub fn load_replay<I: Record>(path: &Path) -> Result<Replay<I>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Replay::decode(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::replay::Packed;

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("warp-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn settings_round_trip_and_survive_garbage() {
        let s = Settings {
            shake: 1,
            reduced_flashes: true,
            reduced_motion: true,
            scheme: Scheme::BlueYellow,
            master: 3,
            music: 0,
            sfx: 10,
            on_beat: true,
            fov: 95,
            camera_roll: true,
            aim_assist: 2,
            fullscreen: true,
        };
        assert_eq!(Settings::from_text(&s.to_text()), s);
        let g = Settings::from_text("shake = 99\nnonsense\nmusic = x\nscheme = PLAID\n");
        assert_eq!(g.shake, 4);
        assert_eq!(g.music, Settings::default().music);
        assert_eq!(g.scheme, Scheme::Standard);
    }

    /// The settings file keeps its format: a `key = value` line per row, in order.
    #[test]
    fn the_settings_file_format() {
        let text = "shake = 4\nreduced_flashes = false\nreduced_motion = false\n\
            scheme = STANDARD\nmaster = 8\nmusic = 8\nsfx = 8\non_beat = false\nfov = 85\n\
            camera_roll = false\naim_assist = 1\nfullscreen = false\n";
        assert_eq!(Settings::default().to_text(), text);
    }

    /// Every row shows its value and steps: toggles flip, cycles wrap, the rest are held at
    /// their ends.
    #[test]
    fn every_row_shows_and_steps() {
        let text = |s: &str| Value::Text(s.into());
        let rows = [
            (Row::Shake, text("100%"), text("100%"), text("75%")),
            (Row::Flashes, text("FULL"), text("REDUCED"), text("REDUCED")),
            (Row::Motion, text("FULL"), text("REDUCED"), text("REDUCED")),
            (
                Row::Colours,
                text("STANDARD"),
                text("RED-GREEN SAFE"),
                text("BLUE-YELLOW SAFE"),
            ),
            (
                Row::MasterVolume,
                Value::Level(8),
                Value::Level(9),
                Value::Level(7),
            ),
            (
                Row::MusicVolume,
                Value::Level(8),
                Value::Level(9),
                Value::Level(7),
            ),
            (
                Row::EffectsVolume,
                Value::Level(8),
                Value::Level(9),
                Value::Level(7),
            ),
            (Row::OnBeat, text("OFF"), text("ON"), text("ON")),
            (Row::FieldOfView, text("85"), text("90"), text("80")),
            (
                Row::CameraRoll,
                text("LEVEL"),
                text("FOLLOW"),
                text("FOLLOW"),
            ),
            (Row::AimAssist, text("LOW"), text("HIGH"), text("OFF")),
            (Row::FullScreen, text("OFF"), text("ON"), text("ON")),
        ];
        assert_eq!(rows.clone().map(|r| r.0), Row::ALL);
        for (row, now, up, down) in rows {
            let s = Settings::default();
            assert_eq!(s.value(row), now, "{row:?}");
            let (mut a, mut b) = (s, s);
            a.adjust(row, 1);
            b.adjust(row, -1);
            assert_eq!((a.value(row), b.value(row)), (up, down), "{row:?}");
            assert_eq!(
                row.kind().steps_on_enter(),
                matches!(row.kind(), RowKind::Toggle | RowKind::Cycle)
            );
            // Every row is kept by the file.
            assert_eq!(Settings::from_text(&a.to_text()), a, "{row:?}");
        }
        // Held at the ends, and wrapping round.
        let mut s = Settings::default();
        for _ in 0..20 {
            s.adjust(Row::Shake, -1);
            s.adjust(Row::MasterVolume, 1);
            s.adjust(Row::FieldOfView, 1);
        }
        assert_eq!(s.value(Row::Shake), text("OFF"));
        assert_eq!(s.value(Row::MasterVolume), Value::Level(10));
        assert_eq!(s.value(Row::FieldOfView), text("110"));
        for _ in 0..3 {
            s.adjust(Row::AimAssist, 1);
            s.adjust(Row::Colours, 1);
        }
        assert_eq!(s.value(Row::AimAssist), text("LOW"));
        assert_eq!(s.scheme, Scheme::Standard);
    }

    /// A table that is there but cannot be read is not overwritten (its entries and replays
    /// would be lost); the run is still kept as the last one.
    #[test]
    fn an_unreadable_table_is_left_alone() {
        let dir = temp("unreadable");
        // A directory where the table's file should be: it exists, and cannot be read.
        std::fs::create_dir_all(dir.join("scores.txt")).unwrap();
        let store = Store::new(Some(dir.clone()));
        let mut r: Replay = Replay::new(1);
        r.score = 500;
        assert_eq!(store.finish(&r, Some("ABC")), None);
        assert!(dir.join("scores.txt").is_dir());
        assert!(dir.join("replays/last.warp").is_file());
        assert_eq!(store.scores::<Packed>().entries, []);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The two games' last runs go through temporary files of their own.
    #[test]
    fn temporaries_are_per_file() {
        let (a, b) = (
            Path::new("replays/last.warp"),
            Path::new("replays/last.tunnel"),
        );
        assert_ne!(temporary(a), temporary(b));
        assert_eq!(temporary(a), Path::new("replays/last.warp.tmp"));
    }

    #[test]
    fn the_table_keeps_the_best_ten_and_their_replays() {
        let dir = temp("scores");
        let store = Store::new(Some(dir.clone()));
        for k in 0..14u64 {
            let mut r: Replay = Replay::new(k);
            r.score = 1000 * (k % 7 + 1) + k;
            let rank = store.finish(&r, Some("ABC"));
            assert!(rank.is_some() || k >= 10, "{k}");
        }
        let scores = store.scores::<Packed>();
        assert_eq!(scores.entries.len(), TABLE);
        assert!(scores.entries.windows(2).all(|w| w[0].score >= w[1].score));
        // Every entry's replay is on disk, and nothing else but `last`.
        let files: Vec<_> = std::fs::read_dir(dir.join("replays"))
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(files.len(), TABLE + 1, "{files:?}");
        for e in &scores.entries {
            assert_eq!(store.replay::<Packed>(&e.replay).unwrap().score, e.score);
        }
        // The Tunnel keeps its own table.
        let mut t: Replay<crate::tunnel::replay::Packed> = Replay::new(5);
        t.score = 777;
        assert_eq!(store.finish(&t, Some("TUN")), Some(0));
        assert_eq!(
            store.scores::<crate::tunnel::replay::Packed>().entries[0].score,
            777
        );
        assert_eq!(store.scores::<Packed>().entries.len(), TABLE);
        assert_eq!(scores.rank(1), None);
        assert_eq!(scores.rank(1_000_000), Some(0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn nothing_persists_without_a_directory() {
        let store = Store::new(None);
        store.save_settings(&Settings::default());
        assert_eq!(store.finish(&Replay::<Packed>::new(1), Some("AAA")), None);
        assert_eq!(store.scores::<Packed>().entries, []);
    }
}
