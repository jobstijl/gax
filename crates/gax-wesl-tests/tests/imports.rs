//! The generated modules as WESL (docs/shaders.md, layer 2): imported across packages by a
//! user shader, stripped down to what it uses, and valid WGSL afterwards; and the traced
//! kernels' module (`FUSED_WESL`), which refers to the algebra modules by path.

use naga::valid::{Capabilities, ValidationFlags, Validator};
use wesl::CompileOptions;
use wesl::resolver::VirtualResolver;

/// A resolver with every gax module under its path (in the `public` form that `wesl` 0.5
/// needs across packages) and `extra` modules.
fn resolver(extra: &[(&str, &str)]) -> VirtualResolver<'static> {
    let mut r = VirtualResolver::new();
    for m in gax::wgsl::ALL {
        r.add_module(
            m.path.parse().expect("a module path"),
            m.wesl_public().into(),
        );
    }
    for (path, src) in extra {
        r.add_module(
            path.parse().expect("a module path"),
            (*src).to_string().into(),
        );
    }
    r
}

/// Compile `package::main` to WGSL and validate it with naga.
fn compile(r: &VirtualResolver<'static>, strip: bool) -> String {
    let options = CompileOptions {
        strip,
        ..CompileOptions::default()
    };
    let res = wesl::compile(&"package::main".parse().expect("a path"), &options, r)
        .unwrap_or_else(|e| panic!("WESL: {e}"));
    let wgsl = res.syntax.to_string();
    let module = naga::front::wgsl::parse_str(&wgsl)
        .unwrap_or_else(|e| panic!("naga: {}", e.emit_to_string(&wgsl)));
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("naga: {}", e.emit_to_string(&wgsl)));
    wgsl
}

const MAIN: &str = "
import gax::pga3d::{Motor, Point, unit_motor_sandwich_point, line_exp, Line};
import gax::pga2d;

@group(0) @binding(0) var<storage, read_write> out: array<vec4<f32>>;

@compute @workgroup_size(1)
fn main() {
    let m = line_exp(Line(vec4<f32>(0.1, 0.2, 0.3, 0.4), vec4<f32>(0.5, 0.6, 0.0, 0.0)));
    out[0] = unit_motor_sandwich_point(m, Point(vec4<f32>(1.0, 2.0, 3.0, 1.0))).c0;
    // Both algebras have a `Motor`; the module paths keep them apart.
    let m2 = pga2d::point_exp(pga2d::Point(vec4<f32>(0.1, 0.2, 0.3, 0.0)));
    out[1] = pga2d::unit_motor_sandwich_point(m2, pga2d::point_new(1.0, 2.0, 1.0)).c0;
}
";

#[test]
fn imports_across_packages_with_stripping() {
    let r = resolver(&[("package::main", MAIN)]);
    let full = compile(&r, false);
    let stripped = compile(&r, true);
    // Stripping keeps what `main` reaches: a few functions, not the whole modules.
    let fns = |s: &str| s.matches("\nfn ").count();
    println!(
        "functions: {} unstripped, {} stripped",
        fns(&full),
        fns(&stripped)
    );
    assert!(fns(&stripped) < 20, "{}", fns(&stripped));
    assert!(fns(&full) > 300);
    assert!(stripped.contains("sandwich_point"));
    assert!(!stripped.contains("mul_motor"));
}

#[test]
fn traced_kernels_compile_against_the_algebra_modules() {
    let main = "
import package::fused::compose_apply_fused;
import gax::pga3d::{Motor, Point};

@group(0) @binding(0) var<storage, read_write> out: array<vec4<f32>>;

@compute @workgroup_size(1)
fn main() {
    let a = Motor(vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.5, 0.0, 0.0, 0.0));
    out[0] = compose_apply_fused(a, a, Point(vec4<f32>(1.0, 2.0, 3.0, 1.0))).c0;
}
";
    let r = resolver(&[
        ("package::main", main),
        ("package::fused", gax_example_traced::FUSED_WESL),
    ]);
    let wgsl = compile(&r, true);
    assert!(wgsl.contains("compose_apply_fused"));
}

/// Without imports, a module is plain WGSL: it validates on its own.
#[test]
fn modules_are_plain_wgsl() {
    for m in gax::wgsl::ALL {
        let module = naga::front::wgsl::parse_str(m.source).expect("parse");
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .expect("validate");
    }
}
