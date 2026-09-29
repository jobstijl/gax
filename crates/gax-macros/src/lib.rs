//! The `algebra!` macro: declare a geometric algebra of any signature `Cl(p, q, r)`, including
//! degenerate and non-diagonal metrics, and get the same typed API as the standard algebras.
//!
//! Use it through the `gax` crate, which re-exports it as `gax::algebra!`. See the `gax`
//! documentation for the declaration format.
//!
//! The macro runs the full generator (exact tables, symbolic simplification) at compile time.
//! For algebras of dimension 5 and up this takes a few seconds in an optimized build and much
//! longer unoptimized, so add this to the using crate's `Cargo.toml`:
//!
//! ```toml
//! [profile.dev.build-override]
//! opt-level = 3
//! ```

use proc_macro::TokenStream;
use std::fmt::Write as _;

/// Declare a geometric algebra. The generated items are placed in a module named after the
/// `algebra` line.
///
/// ```ignore
/// gax::algebra! {
///     algebra stap "Spacetime algebra with a projective dimension, R(3,1,1).";
///     basis ep = 0, e0 = 1, e1 = -1, e2 = -1, e3 = -1;
///     kind Scalar = [1];
///     kind Vector = [ep, e0, e1, e2, e3];
///     ...
/// }
/// ```
#[proc_macro]
pub fn algebra(input: TokenStream) -> TokenStream {
    let src = input.to_string();
    let spec = match gax_gen::spec::AlgebraSpec::parse(&src) {
        Ok(s) => s,
        Err(e) => return compile_error(&format!("gax::algebra!: {e}")),
    };
    let (mut code, stats) = gax_gen::emit::emit(
        &spec,
        &gax_gen::emit::Config {
            core: "::gax".into(),
            // Enabled by gax's `batch` feature, which turns this crate's on.
            batch: cfg!(feature = "batch").then(String::new),
            // Enabled by gax's `check-units` feature.
            check_units: cfg!(feature = "check-units").then(String::new),
            gpu: cfg!(feature = "bytemuck").then(String::new),
        },
    );
    // The algebra's WGSL modules, for shaders and for its traced kernels, which name its kinds
    // `package::{name}::Kind` (enabled by gax's `wgsl` feature).
    if cfg!(feature = "wgsl") {
        let name = &spec.name;
        for (constant, prec, path) in [
            ("WGSL_MODULE", gax_gen::kernel::Precision::F32, name.clone()),
            (
                "WGSL_MODULE_F16",
                gax_gen::kernel::Precision::F16,
                format!("{name}_f16"),
            ),
        ] {
            let source = gax_gen::emit_wgsl::module_in(&spec, &stats, true, prec);
            let _ = write!(
                code,
                "/// The WGSL module of this algebra ({}): register it under its `path`,\n/// `package::{path}`, next to shaders that import it (see gax's docs/shaders.md).\npub const {constant}: ::gax::wgsl::Module = ::gax::wgsl::Module {{ path: \"package::{path}\", source: {source:?} }};\n",
                prec.scalar()
            );
        }
    }
    let doc = if spec.doc.is_empty() {
        format!("The `{}` algebra.", spec.name)
    } else {
        spec.doc.clone()
    };
    let module = format!(
        "#[doc = {doc:?}]\n#[allow(missing_docs, unused_variables, unused_parens, clippy::all, clippy::pedantic)]\npub mod {} {{\n{code}\n}}",
        spec.name
    );
    module
        .parse()
        .unwrap_or_else(|e| compile_error(&format!("gax::algebra!: internal error: {e}")))
}

fn compile_error(msg: &str) -> TokenStream {
    format!("::core::compile_error!({msg:?});")
        .parse()
        .expect("valid tokens")
}
