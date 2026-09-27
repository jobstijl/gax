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

impl Settings {
    /// The rows of the settings menu (their labels), in order.
    pub const ROWS: [&'static str; 12] = [
        "SCREEN SHAKE",
        "FLASHES",
        "MOTION",
        "COLOURS",
        "MASTER VOLUME",
        "MUSIC",
        "EFFECTS",
        "EFFECTS ON THE BEAT",
        "TUNNEL FIELD OF VIEW",
        "TUNNEL CAMERA ROLL",
        "TUNNEL AIM ASSIST",
        "FULL SCREEN",
    ];

    /// Rows that toggle or cycle (Enter steps them too).
    pub fn toggles(row: usize) -> bool {
        matches!(row, 1 | 2 | 3 | 7 | 9 | 10 | 11)
    }

    /// The Tunnel's lock-on margin (screen units beyond an enemy's apparent size).
    pub fn aim_reach(&self) -> f32 {
        [0.25, 1.1, 2.2][usize::from(self.aim_assist.min(2))]
    }

    /// A row's value as the menu shows it; volumes are `(level, 10)` bars instead.
    pub fn value(&self, row: usize) -> Result<String, u8> {
        match row {
            0 => Ok(match self.shake {
                0 => "OFF".into(),
                n => format!("{}%", 25 * u32::from(n)),
            }),
            1 => Ok(if self.reduced_flashes {
                "REDUCED"
            } else {
                "FULL"
            }
            .into()),
            2 => Ok(if self.reduced_motion {
                "REDUCED"
            } else {
                "FULL"
            }
            .into()),
            3 => Ok(self.scheme.name().into()),
            4 => Err(self.master),
            5 => Err(self.music),
            6 => Err(self.sfx),
            7 => Ok(if self.on_beat { "ON" } else { "OFF" }.into()),
            8 => Ok(format!("{}", self.fov)),
            9 => Ok(if self.camera_roll { "FOLLOW" } else { "LEVEL" }.into()),
            10 => Ok(["OFF", "LOW", "HIGH"][usize::from(self.aim_assist.min(2))].into()),
            _ => Ok(if self.fullscreen { "ON" } else { "OFF" }.into()),
        }
    }

    /// Step a row's value by `by` (toggles flip on any step).
    pub fn adjust(&mut self, row: usize, by: i32) {
        let vol = |v: &mut u8| *v = (i32::from(*v) + by).clamp(0, 10) as u8;
        match row {
            0 => self.shake = (i32::from(self.shake) + by).clamp(0, 4) as u8,
            1 => self.reduced_flashes = !self.reduced_flashes,
            2 => self.reduced_motion = !self.reduced_motion,
            3 => self.scheme = self.scheme.step(by),
            4 => vol(&mut self.master),
            5 => vol(&mut self.music),
            6 => vol(&mut self.sfx),
            7 => self.on_beat = !self.on_beat,
            8 => self.fov = (i32::from(self.fov) + 5 * by).clamp(60, 110) as u8,
            9 => self.camera_roll = !self.camera_roll,
            10 => self.aim_assist = (i32::from(self.aim_assist) + by).rem_euclid(3) as u8,
            11 => self.fullscreen = !self.fullscreen,
            _ => {}
        }
    }

    /// Linear gains of the effects and the music (volumes on a perceptual, squared, curve).
    pub fn gains(&self) -> [f32; 2] {
        let v = |x: u8| (f32::from(x) / 10.0).powi(2);
        [v(self.master) * v(self.sfx), v(self.master) * v(self.music)]
    }

    /// The text form.
    pub fn to_text(self) -> String {
        format!(
            "shake = {}\nreduced_flashes = {}\nreduced_motion = {}\nscheme = {}\nmaster = {}\nmusic = {}\nsfx = {}\non_beat = {}\nfov = {}\ncamera_roll = {}\naim_assist = {}\nfullscreen = {}\n",
            self.shake,
            self.reduced_flashes,
            self.reduced_motion,
            self.scheme.name(),
            self.master,
            self.music,
            self.sfx,
            self.on_beat,
            self.fov,
            self.camera_roll,
            self.aim_assist,
            self.fullscreen
        )
    }

    /// Read the text form; unknown or malformed lines keep their defaults.
    pub fn from_text(text: &str) -> Settings {
        let mut s = Settings::default();
        for line in text.lines() {
            let Some((k, v)) = line.split_once('=') else {
                continue;
            };
            let (k, v) = (k.trim(), v.trim());
            let vol = |v: &str, d: u8| v.parse::<u8>().map_or(d, |x| x.min(10));
            match k {
                "shake" => s.shake = v.parse::<u8>().map_or(s.shake, |x| x.min(4)),
                "reduced_flashes" => s.reduced_flashes = v == "true",
                "reduced_motion" => s.reduced_motion = v == "true",
                "aim_assist" => s.aim_assist = v.parse::<u8>().map_or(s.aim_assist, |x| x.min(2)),
                "scheme" => {
                    s.scheme = Scheme::ALL
                        .into_iter()
                        .find(|x| x.name() == v)
                        .unwrap_or_default()
                }
                "master" => s.master = vol(v, s.master),
                "music" => s.music = vol(v, s.music),
                "sfx" => s.sfx = vol(v, s.sfx),
                "on_beat" => s.on_beat = v == "true",
                "fov" => s.fov = v.parse::<u8>().map_or(s.fov, |x| x.clamp(60, 110)),
                "camera_roll" => s.camera_roll = v == "true",
                "fullscreen" => s.fullscreen = v == "true",
                _ => {}
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

    fn read(&self, name: &str) -> Option<String> {
        std::fs::read_to_string(self.dir.as_ref()?.join(name)).ok()
    }

    /// Write `bytes` to `name` (through a temporary file, so a crash never leaves half a file).
    fn write(&self, name: &str, bytes: &[u8]) {
        let Some(dir) = &self.dir else {
            return;
        };
        let path = dir.join(name);
        let tmp = path.with_extension("tmp");
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
        self.read("settings.txt")
            .map(|t| Settings::from_text(&t))
            .unwrap_or_default()
    }

    /// Save the settings.
    pub fn save_settings(&self, s: &Settings) {
        self.write("settings.txt", s.to_text().as_bytes());
    }

    /// The high-score table of the game whose replays are `I`.
    pub fn scores<I: Record>(&self) -> Scores {
        self.read(I::TABLE)
            .map(|t| Scores::from_text(&t))
            .unwrap_or_default()
    }

    /// Record a finished run: always as `replays/last.<ext>`; if it makes the table, as an
    /// entry named `name` with its own replay. Returns its rank.
    pub fn finish<I: Record>(&self, replay: &Replay<I>, name: Option<&str>) -> Option<usize> {
        let bytes = replay.encode();
        self.write(&format!("replays/last.{}", I::EXT), &bytes);
        let name = name?;
        let mut scores = self.scores::<I>();
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
        assert!(store.scores::<Packed>().entries.is_empty());
    }
}
