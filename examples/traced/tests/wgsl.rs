//! The WGSL forms of the traced kernels link against gax's modules and validate with naga,
//! as a shader that imports them would.

use gax_example_traced::FUSED_WESL;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use wesl::CompileOptions;
use wesl::resolver::VirtualResolver;

#[test]
fn traced_wgsl_links_and_validates() {
    let mut resolver = VirtualResolver::new();
    for m in gax::wgsl::ALL {
        resolver.add_module(m.path.parse().expect("module path"), m.wesl_public().into());
    }
    resolver.add_module("package::fused".parse().expect("path"), FUSED_WESL.into());
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
    for k in [
        "between_lines_fused",
        "between_planes_fused",
        "between_points_fused",
    ] {
        assert!(linked.contains(k), "{k} is missing from the linked WGSL");
    }
    let module = naga::front::wgsl::parse_str(&linked).unwrap_or_else(|e| {
        panic!("parse: {}", e.emit_to_string(&linked));
    });
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("validate: {}", e.emit_to_string(&linked)));
}
