//! The command line: what to run, replays from files, and `--verify`.

use crate::game::AnyReplay;
use crate::sim::replay::{Packed, Record, Replay};
use crate::tunnel::replay::Packed as TunnelPacked;
use std::path::Path;

/// What the command line asks for.
pub enum Command {
    /// `--shot DIR [SECONDS...]`: headless snapshots.
    Shots(Vec<String>),
    /// `--music DIR [SECONDS] [SEED]`: the music, offline.
    Music(Vec<String>),
    /// `--verify FILE`: a replay played headless, every state hash checked.
    Verify(String),
    /// The game in a window: with `--smoke`, a bot plays a few seconds; with `--replay FILE`,
    /// it starts watching that replay.
    Play { smoke: bool, replay: Option<String> },
}

const KNOWN: [&str; 5] = ["--smoke", "--shot", "--music", "--replay", "--verify"];

/// Read the arguments (the program's name first).
pub fn parse(args: &[String]) -> Result<Command, String> {
    let after = |flag: &str| {
        let i = args.iter().position(|a| a == flag)?;
        Some(args[i + 1..].to_vec())
    };
    let file = |flag: &str, rest: Vec<String>| {
        rest.into_iter()
            .next()
            .ok_or_else(|| format!("{flag} FILE"))
    };
    if let Some(rest) = after("--shot") {
        return Ok(Command::Shots(rest));
    }
    if let Some(rest) = after("--music") {
        return Ok(Command::Music(rest));
    }
    if let Some(rest) = after("--verify") {
        return Ok(Command::Verify(file("--verify", rest)?));
    }
    if let Some(bad) = args
        .iter()
        .skip(1)
        .find(|a| a.starts_with("--") && !KNOWN.contains(&a.as_str()))
    {
        return Err(format!(
            "unknown option {bad} (known: {})",
            KNOWN.join(", ")
        ));
    }
    let replay = after("--replay")
        .map(|rest| file("--replay", rest))
        .transpose()?;
    Ok(Command::Play {
        smoke: args.iter().any(|a| a == "--smoke"),
        replay,
    })
}

/// A replay file of either game, told apart by its magic.
pub fn load_any(path: &Path) -> Result<AnyReplay, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let magic = bytes.get(..8);
    let replay = if magic == Some(Packed::MAGIC) {
        Replay::decode(&bytes).map(AnyReplay::Plane)
    } else if magic == Some(TunnelPacked::MAGIC) {
        Replay::decode(&bytes).map(AnyReplay::Tunnel)
    } else {
        Err("not a replay of warp".into())
    };
    replay.map_err(|e| format!("{}: {e}", path.display()))
}

/// Play the replay at `path` headless and check every state hash: the line to print, or why
/// it failed (and whether it is the replay's fault, a divergence, rather than the file's).
pub fn verify(path: &Path) -> Result<String, (String, Failure)> {
    let replay = load_any(path).map_err(|e| (e, Failure::File))?;
    let result = match &replay {
        AnyReplay::Plane(r) => check(r, |r| r.verify().map(|w| w.ledger.score)),
        AnyReplay::Tunnel(r) => check(r, |r| r.verify().map(|w| w.ledger.score)),
    };
    result.map_err(|e| (e, Failure::Diverged))
}

/// Why `--verify` failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Failure {
    /// The file is not a readable replay.
    File,
    /// The replay diverged.
    Diverged,
}

/// Verify `r` with `verify`, which returns the final score (noting on stderr first if it was
/// recorded on another build).
fn check<I: Record>(
    r: &Replay<I>,
    verify: impl FnOnce(&Replay<I>) -> Result<u64, crate::sim::replay::Desync>,
) -> Result<String, String> {
    if !r.is_this_build() {
        eprintln!("note: recorded on build {}, this is {}", r.build, I::BUILD);
    }
    match verify(r) {
        Ok(score) => Ok(format!(
            "ok: {:.0} s, score {score}, {} hashes matched",
            r.seconds(),
            r.hashes.len()
        )),
        Err(d) => Err(format!("diverged: {d:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn options_are_read() {
        assert!(matches!(
            parse(&args("warp --replay a.warp")),
            Ok(Command::Play { smoke: false, replay: Some(f) }) if f == "a.warp"
        ));
        assert!(matches!(
            parse(&args("warp --smoke")),
            Ok(Command::Play {
                smoke: true,
                replay: None
            })
        ));
        assert!(matches!(parse(&args("warp --verify x")), Ok(Command::Verify(f)) if f == "x"));
        assert!(parse(&args("warp --verify")).is_err());
        assert!(parse(&args("warp --fast")).is_err());
    }

    /// A replay of either game loads from one read; a corrupt one says what is wrong with
    /// it, not that it is some other game's.
    #[test]
    fn replays_load_by_their_magic() {
        let dir = std::env::temp_dir().join(format!("warp-test-cli-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let plane = Replay::<Packed>::new(3).encode();
        let tunnel = Replay::<TunnelPacked>::new(4).encode();
        let write = |name: &str, bytes: &[u8]| {
            let p = dir.join(name);
            std::fs::write(&p, bytes).unwrap();
            p
        };
        let p = write("a.warp", &plane);
        assert!(matches!(load_any(&p), Ok(AnyReplay::Plane(r)) if r.seed == 3));
        let p = write("b.tunnel", &tunnel);
        assert!(matches!(load_any(&p), Ok(AnyReplay::Tunnel(r)) if r.seed == 4));
        let p = write("c.warp", &plane[..plane.len() - 3]);
        let e = load_any(&p).err().unwrap();
        assert!(e.ends_with("truncated replay"), "{e}");
        let p = write("d.txt", b"hello, world");
        assert!(
            load_any(&p)
                .err()
                .unwrap()
                .ends_with("not a replay of warp")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
