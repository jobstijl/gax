//! Regenerate the standard algebras shipped in the `gax` crate.
//!
//! Usage: `cargo run -p gax-gen --bin gax-regen [--check]` from the workspace root.
//! With `--check`, nothing is written and the exit code is nonzero when a committed file
//! differs from what the generator produces (used in CI).

use gax_gen::emit::{Config, emit, emit_tests};
use gax_gen::spec::AlgebraSpec;
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let check = std::env::args().any(|a| a == "--check");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../gax");
    let mut specs: Vec<_> = std::fs::read_dir(root.join("specs"))
        .expect("specs directory")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "gax"))
        .collect();
    specs.sort();
    let mut stale = false;
    for path in specs {
        let src = std::fs::read_to_string(&path).expect("read spec");
        let spec = match AlgebraSpec::parse(&src) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        };
        let (code, stats) = emit(
            &spec,
            &Config {
                core: "crate".into(),
            },
        );
        let tests = emit_tests(
            &spec,
            &stats,
            &format!("gax::{}", spec.name),
            &format!("../specs/{}.gax", spec.name),
        );
        let test_target = root.join("tests").join(format!("ops_{}.rs", spec.name));
        stale |= write_or_check(&test_target, &tests, check);
        let target = root.join("src/algebras").join(format!("{}.rs", spec.name));
        let current = std::fs::read_to_string(&target).unwrap_or_default();
        eprintln!(
            "{}: {} lines, {} product impls, {} sandwich kernels",
            spec.name,
            code.lines().count(),
            stats.binary_impls,
            stats.sandwich_impls
        );
        for (v, x, unit, cost) in &stats.sandwich_costs {
            eprintln!("    {}{v} >> {x}: {cost}", if *unit { "Unit " } else { "" });
        }
        if current != code {
            if check {
                eprintln!("{} is out of date", target.display());
                stale = true;
            } else {
                std::fs::write(&target, code).expect("write generated file");
            }
        }
    }
    if stale {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn write_or_check(target: &Path, content: &str, check: bool) -> bool {
    let current = std::fs::read_to_string(target).unwrap_or_default();
    if current == content {
        return false;
    }
    if check {
        eprintln!("{} is out of date", target.display());
        true
    } else {
        std::fs::write(target, content).expect("write generated file");
        false
    }
}
