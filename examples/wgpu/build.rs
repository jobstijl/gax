//! Trace the particle kernel (Rust and WGSL), then link the shaders against gax's WGSL modules
//! with the `wesl` crate. The program includes plain WGSL; it does not depend on `wesl`.

#[path = "src/kernels.rs"]
#[allow(dead_code)]
mod kernels;

use gax::pga2d::{Motor, Point};
use gax::trace::{Sym, Tracer};
use std::path::Path;
use wesl::CompileOptions;
use wesl::resolver::VirtualResolver;

const SHADERS: [&str; 4] = [
    "shapes",
    "shapes_matrix",
    "particles_step",
    "particles_draw",
];

fn main() {
    let mut t = Tracer::new();
    t.batch(true);
    t.wgsl(true);
    t.kernel(
        "particle_step",
        |m: Motor<(), Sym>, r: Point<(), Sym>, dt: Sym| kernels::particle_step(m, r, dt),
    );
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
    // gax's modules, in the form with `public` declarations that `wesl` 0.5 imports across
    // packages; the traced kernels; the program's own shaders.
    for m in gax::wgsl::ALL {
        resolver.add_module(m.path.parse().expect("module path"), m.wesl_public().into());
    }
    let fused = std::fs::read_to_string(out.join("fused.wesl")).expect("traced WGSL");
    resolver.add_module("package::fused".parse().expect("path"), fused.into());
    for s in SHADERS {
        let src = std::fs::read_to_string(format!("shaders/{s}.wesl")).expect("shader source");
        resolver.add_module(format!("package::{s}").parse().expect("path"), src.into());
        println!("cargo:rerun-if-changed=shaders/{s}.wesl");
    }
    let options = CompileOptions {
        strip: true,
        ..CompileOptions::default()
    };
    for s in SHADERS {
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
