//! Traces the generic kernels of `src/kernels.rs` at build time into fused straight-line code
//! (`fused.rs`), their batch forms and WGSL (`fused.wesl`), which the crate includes.

#[path = "src/kernels.rs"]
#[allow(dead_code)]
mod kernels;

use gax::Unit;
use gax::pga3d::{Line, Motor, Plane, Point};
use gax::trace::{Sym, Tracer};

fn main() {
    let mut t = Tracer::new();
    // Also emit `*_batch` forms that run the kernels on SIMD lanes.
    t.batch(true);
    // Also emit WGSL forms of the kernels (`fused.wesl`, `FUSED_WESL`).
    t.wgsl(true);
    t.kernel(
        "shadow_of_moved_fused",
        |m: Unit<Motor<(), Sym>>, l: Point<(), Sym>, g: Plane<(), Sym>, p: Point<(), Sym>| {
            kernels::shadow_of_moved(m, l, g, p)
        },
    );
    t.kernel(
        "compose_apply_fused",
        |a: Unit<Motor<(), Sym>>, b: Unit<Motor<(), Sym>>, p: Point<(), Sym>| {
            kernels::compose_apply(a, b, p)
        },
    );
    t.kernel(
        "shadow_on_floor_fused",
        |m: Unit<Motor<(), Sym>>, l: Point<(), Sym>, p: Point<(), Sym>| {
            kernels::shadow_on_floor(m, l, p)
        },
    );
    t.kernel("euclidean_fused", |p: Point<(), Sym>| kernels::euclidean(p));
    t.kernel(
        "screw_apply_fused",
        |b: Line<(), Sym>, p: Point<(), Sym>| kernels::screw_apply(b, p),
    );
    t.kernel(
        "between_lines_fused",
        |a: Line<(), Sym>, b: Line<(), Sym>| kernels::between_lines(a, b),
    );
    t.kernel(
        "between_planes_fused",
        |a: Plane<(), Sym>, b: Plane<(), Sym>| kernels::between_planes(a, b),
    );
    t.kernel(
        "between_points_fused",
        |a: Point<(), Sym>, b: Point<(), Sym>| kernels::between_points(a, b),
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
