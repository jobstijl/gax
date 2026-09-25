#[path = "src/kernels.rs"]
#[allow(dead_code)]
mod kernels;

use gax::Unit;
use gax::pga3d::{Motor, Plane, Point};
use gax::trace::{Sym, Tracer};

fn main() {
    let mut t = Tracer::new();
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
    t.write_out_dir("fused.rs");
    for r in t.reports() {
        println!(
            "cargo:warning={}: {} (generic code: {})",
            r.name, r.cost, r.naive
        );
    }
    println!("cargo:rerun-if-changed=src/kernels.rs");
}
