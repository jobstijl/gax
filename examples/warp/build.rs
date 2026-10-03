//! Trace the kernels in `src/kernels.rs` into Rust and WGSL, then link the game's WESL shaders
//! against gax's generated modules with the `wesl` crate. The game includes plain WGSL; it
//! does not depend on `wesl` at run time.

#[path = "src/kernels.rs"]
#[allow(dead_code)]
mod kernels;

use gax::pga2d::Point;
use gax::trace::{Sym, Tracer};
use std::path::Path;
use wesl::CompileOptions;
use wesl::resolver::VirtualResolver;

/// Shader modules: `(module, is it a main module compiled to WGSL)`.
const SHADERS: [(&str, bool); 8] = [
    ("common", false),
    ("lines", true),
    ("grid_step", true),
    ("grid_draw", true),
    ("particles_step", true),
    ("particles_draw", true),
    ("post", true),
    ("stars", true),
];

type P = Point<(), Sym>;
type L = gax::pga3d::Point<(), Sym>;

fn main() {
    let mut t = Tracer::new();
    t.batch(true);
    t.wgsl(true);
    t.kernel(
        "grid_node",
        |p: P, v: P, rest: P, n: [P; 4], f: P, k: [Sym; 5]| kernels::grid_node(p, v, rest, n, f, k),
    );
    t.kernel("source_force", |p: P, s: P, k: [Sym; 2]| {
        kernels::source_force(p, s, k)
    });
    t.kernel("particle_step", |p: P, v: P, f: P, k: [Sym; 2]| {
        kernels::particle_step(p, v, f, k)
    });
    t.kernel("segment_corner", |a: P, b: P, k: [Sym; 3]| {
        kernels::segment_corner(a, b, k)
    });
    t.kernel("segment_distance", |a: P, b: P, p: P| {
        kernels::segment_distance(a, b, p)
    });
    t.kernel("edge_heat", |a: P, b: P, va: P, vb: P, k: [Sym; 3]| {
        kernels::edge_heat(a, b, va, vb, k)
    });
    t.kernel("streak_tail", |p: P, v: P, dt: Sym| {
        kernels::streak_tail(p, v, dt)
    });
    t.kernel("luma", |l: L| kernels::luma(l));
    t.kernel("agx", |l: L, s: Sym| kernels::agx(l, s));
    t.kernel("ripple", |p: P, c: P, k: [Sym; 2]| kernels::ripple(p, c, k));
    t.kernel("scale_about", |p: P, c: P, k: Sym| {
        kernels::scale_about(p, c, k)
    });
    t.kernel("distance", |a: P, b: P| kernels::distance(a, b));
    t.kernel("wave", |t: Sym| kernels::wave(t));
    t.kernel("light_mix", |a: L, b: L, t: Sym| {
        kernels::light_mix(a, b, t)
    });
    t.kernel("light_whiten", |l: L, t: Sym| kernels::light_whiten(l, t));
    t.kernel("light_fade", |l: L, k: Sym| kernels::light_fade(l, k));
    t.write_out_dir("fused.rs");
    for r in t.reports() {
        println!(
            "cargo:warning={}: {} (generic code: {})",
            r.name, r.cost, r.naive
        );
    }

    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let out = Path::new(&out);
    let mut resolver = VirtualResolver::new();
    for m in gax::wgsl::ALL {
        resolver.add_module(m.path.parse().expect("module path"), m.wesl_public().into());
    }
    let fused = std::fs::read_to_string(out.join("fused.wesl")).expect("traced WGSL");
    resolver.add_module("package::fused".parse().expect("path"), fused.into());
    for (s, _) in SHADERS {
        let src = std::fs::read_to_string(format!("shaders/{s}.wesl")).expect("shader source");
        resolver.add_module(format!("package::{s}").parse().expect("path"), src.into());
        println!("cargo:rerun-if-changed=shaders/{s}.wesl");
    }
    let options = CompileOptions {
        strip: true,
        ..CompileOptions::default()
    };
    for (s, main) in SHADERS {
        if !main {
            continue;
        }
        let res = wesl::compile(
            &format!("package::{s}").parse().expect("path"),
            &options,
            &resolver,
        )
        .unwrap_or_else(|e| panic!("shaders/{s}.wesl: {e}"));
        std::fs::write(out.join(format!("{s}.wgsl")), res.syntax.to_string()).expect("write WGSL");
    }
    println!("cargo:rerun-if-changed=src/kernels.rs");
    sim_hash();
}

/// A hash of everything a replay depends on: both games' simulations (the Plane's `sim`, the
/// Tunnel's `tunnel`), the helpers they call (`geom`, `signal`), and gax's sources. Replays
/// record it, so a replay from another build is flagged instead of silently diverging.
fn sim_hash() {
    fn walk(path: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
        if path.extension().is_some_and(|x| x == "rs") {
            files.push(path.to_path_buf());
        }
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for e in entries.flatten() {
            walk(&e.path(), files);
        }
    }
    let mut files = Vec::new();
    for path in [
        "src/sim",
        "src/tunnel",
        "src/geom.rs",
        "src/signal.rs",
        "../../crates/gax/src",
        "../../crates/gax-core/src",
    ] {
        walk(std::path::Path::new(path), &mut files);
        println!("cargo:rerun-if-changed={path}");
    }
    files.sort();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for f in files {
        for b in std::fs::read(&f).unwrap_or_default() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }
    println!("cargo:rustc-env=WARP_SIM_HASH={:08x}", h >> 32);
}
