//! WGSL modules of the standard algebras, as source strings (feature `wgsl`, ADR-028).
//!
//! Each module is plain WGSL and self-contained, so it is also a valid WESL module:
//!
//! * with the [`wesl`](https://crates.io/crates/wesl) crate or an engine's WESL resolver,
//!   register [`Module::source`] under [`Module::path`] (`gax::pga3d`) and import from it:
//!   `import gax::pga3d::{Motor, Point, unit_motor_sandwich_point};`. The `wesl` crate from 0.5
//!   on needs declarations marked `public` to import them across packages: use
//!   [`Module::wesl_public`] there;
//! * with plain WGSL, prepend the source to your shader.
//!
//! Each kind is a struct of `ceil(N/4)` `vec4<f32>` fields, the layout of the `{Kind}Gpu` Rust
//! types (feature `bytemuck`). The kernels are printed from the same verified programs as the
//! Rust kernels; see `docs/shaders.md` for the list and the naming scheme.
//!
//! ```
//! # #[cfg(feature = "pga3d")] {
//! let m = gax::wgsl::PGA3D;
//! assert_eq!(m.path, "gax::pga3d");
//! assert!(m.source.contains("fn unit_motor_sandwich_point(v: Motor, x: Point) -> Point"));
//! # }
//! ```

/// A generated shader module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Module {
    /// The intended WESL module path, such as `gax::pga3d`.
    pub path: &'static str,
    /// The source: plain WGSL, also valid WESL.
    pub source: &'static str,
}

impl Module {
    /// The source with `public` before every top-level declaration, for WESL resolvers that
    /// implement visibility (the `wesl` crate from 0.5 on): imports from another package need
    /// it. Older resolvers (such as `wesl` 0.4) do not parse `public`; use [`Module::source`].
    #[cfg(feature = "std")]
    pub fn wesl_public(&self) -> std::string::String {
        let mut out =
            std::string::String::with_capacity(self.source.len() + self.source.len() / 16);
        for line in self.source.lines() {
            if line.starts_with("fn ") || line.starts_with("struct ") || line.starts_with("alias ")
            {
                out.push_str("public ");
            }
            out.push_str(line);
            out.push('\n');
        }
        out
    }
}

macro_rules! modules {
    ($($feature:literal $name:ident $file:literal $doc:literal;)*) => {
        $(
            #[doc = $doc]
            #[cfg(feature = $feature)]
            pub const $name: Module = Module {
                path: concat!("gax::", $feature),
                source: include_str!(concat!("wgsl/", $file)),
            };
        )*

        /// Every module of the enabled algebras.
        pub const ALL: &[Module] = &[$(
            #[cfg(feature = $feature)]
            $name,
        )*];
    };
}

modules! {
    "pga2d" PGA2D "pga2d.wgsl" "The PGA2D module, `gax::pga2d`.";
    "pga3d" PGA3D "pga3d.wgsl" "The PGA3D module, `gax::pga3d`.";
    "vga2d" VGA2D "vga2d.wgsl" "The VGA2D module, `gax::vga2d`.";
    "vga3d" VGA3D "vga3d.wgsl" "The VGA3D module, `gax::vga3d`.";
    "sta" STA "sta.wgsl" "The STA module, `gax::sta`.";
    "cga2d" CGA2D "cga2d.wgsl" "The CGA2D module, `gax::cga2d`.";
    "cga3d" CGA3D "cga3d.wgsl" "The CGA3D module, `gax::cga3d`.";
    "stap" STAP "stap.wgsl" "The STAP module, `gax::stap`.";
    "csta" CSTA "csta.wgsl" "The CSTA module, `gax::csta`.";
}
