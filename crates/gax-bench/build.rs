//! Trace the benchmark kernels into fused versions (build-time tracing in practice).

#[path = "src/kernels.rs"]
#[allow(dead_code)]
mod kernels;

use gax::pga3d::{Line, Motor};
use gax::trace::{Sym, Tracer};

fn main() {
    let mut t = Tracer::new();
    t.batch(true);
    t.kernel(
        "rigid_step_fused",
        |m: Motor<(), Sym>,
         b: Line<(), Sym>,
         f: Line<(), Sym>,
         dt: Sym,
         mass: Sym,
         moments: [Sym; 3]| { kernels::rigid_step(m, b, f, dt, mass, moments) },
    );
    t.kernel(
        "rigid_rate_fused",
        |b: Line<(), Sym>, f: Line<(), Sym>, dt: Sym, mass: Sym, moments: [Sym; 3]| {
            kernels::rigid_rate(b, f, dt, mass, moments)
        },
    );
    t.kernel(
        "rigid_step_fixed_fused",
        |m: Motor<(), Sym>, b: Line<(), Sym>, f: Line<(), Sym>| kernels::rigid_step_fixed(m, b, f),
    );
    t.write_out_dir("fused.rs");
    for r in t.reports() {
        println!(
            "cargo:warning={}: {} (generic code: {})",
            r.name, r.cost, r.naive
        );
    }
    println!("cargo:rerun-if-changed=src/kernels.rs");
}
