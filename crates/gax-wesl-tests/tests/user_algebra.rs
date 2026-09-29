//! An algebra declared with `algebra!` has WGSL modules too (`WGSL_MODULE`, `WGSL_MODULE_F16`,
//! with gax's `wgsl` feature), and a kernel traced over its kinds names them
//! `package::{algebra}::Kind`, so it links against the module registered there.

use gax::trace::{Sym, Tracer};
use naga::valid::{Capabilities, ValidationFlags, Validator};
use wesl::CompileOptions;
use wesl::resolver::VirtualResolver;

gax::algebra! {
    algebra plane "Plane-based PGA of the plane, declared in a test.";
    basis e0 = 0, e1 = 1, e2 = 1;
    kind Scalar = [1];
    versor Line = [e1, e2, e0];
    versor Point = [e20, e01, e12];
    kind Pseudoscalar = [e012];
    versor Motor = [1, e12, e20, e01];
    versor Flector = [e1, e2, e0, e012];
    kind Multivector = [1, e0, e1, e2, e01, e20, e12, e012];
}

fn validate(what: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("{what}: {}", e.emit_to_string(src)));
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{what}: {}", e.emit_to_string(src)));
}

#[test]
fn a_declared_algebra_has_wgsl_modules() {
    assert_eq!(plane::WGSL_MODULE.path, "package::plane");
    assert_eq!(plane::WGSL_MODULE_F16.path, "package::plane_f16");
    validate("plane", plane::WGSL_MODULE.source);
    validate("plane_f16", plane::WGSL_MODULE_F16.source);
    assert!(
        plane::WGSL_MODULE
            .source
            .contains("fn unit_motor_sandwich_point(v: Motor, x: Point) -> Point")
    );
}

#[test]
fn its_traced_kernels_link_against_them() {
    let mut t = Tracer::new();
    t.wgsl(true);
    t.kernel(
        "move_point",
        |m: gax::Unit<plane::Motor<(), Sym>>, p: plane::Point<(), Sym>| m >> p,
    );
    let traced = t.wgsl_source();
    assert!(traced.contains("package::plane::Point"), "{traced}");
    let mut resolver = VirtualResolver::new();
    resolver.add_module(
        "package::plane".parse().expect("path"),
        plane::WGSL_MODULE.source.into(),
    );
    resolver.add_module("package::fused".parse().expect("path"), traced.into());
    let options = CompileOptions {
        strip: false,
        ..CompileOptions::default()
    };
    let linked = wesl::compile(
        &"package::fused".parse().expect("path"),
        &options,
        &resolver,
    )
    .unwrap_or_else(|e| panic!("link: {e}"))
    .syntax
    .to_string();
    assert!(linked.contains("fn move_point"), "{linked}");
    validate("linked", &linked);
}
