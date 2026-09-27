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
}
