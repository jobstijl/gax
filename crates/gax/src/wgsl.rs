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
//! types (feature `bytemuck`). Every module also exists in `f16` (such as `PGA3D_F16`, path
//! `gax::pga3d_f16`), with the layout of the `{Kind}Gpu16` types. The kernels are printed from
//! the same verified programs as the Rust kernels; see `docs/shaders.md` for the list and the
//! naming scheme.
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
    ($($feature:literal $name:ident $name16:ident;)*) => {
        $(
            #[doc = concat!("The ", stringify!($name), " module, `gax::", $feature, "`.")]
            #[cfg(feature = $feature)]
            pub const $name: Module = Module {
                path: concat!("gax::", $feature),
                source: include_str!(concat!("wgsl/", $feature, ".wgsl")),
            };

            #[doc = concat!("The same module in `f16`, `gax::", $feature, "_f16`: the same functions on")]
            /// structs of `vec4<f16>`. It starts with `enable f16;`, so the device needs the
            /// `shader-f16` feature. Its `exp`, `log` and `normalized` compute in `f32` inside.
            #[cfg(feature = $feature)]
            pub const $name16: Module = Module {
                path: concat!("gax::", $feature, "_f16"),
                source: include_str!(concat!("wgsl/", $feature, "_f16.wgsl")),
            };
        )*

        /// Every `f32` module of the enabled algebras.
        pub const ALL: &[Module] = &[$(
            #[cfg(feature = $feature)]
            $name,
        )*];

        /// Every `f16` module of the enabled algebras (separate from [`ALL`]: they need the
        /// device's `shader-f16` feature).
        pub const ALL_F16: &[Module] = &[$(
            #[cfg(feature = $feature)]
            $name16,
        )*];
    };
}

modules! {
    "pga2d" PGA2D PGA2D_F16;
    "pga3d" PGA3D PGA3D_F16;
    "vga2d" VGA2D VGA2D_F16;
    "vga3d" VGA3D VGA3D_F16;
    "sta" STA STA_F16;
    "cga2d" CGA2D CGA2D_F16;
    "cga3d" CGA3D CGA3D_F16;
    "stap" STAP STAP_F16;
    "csta" CSTA CSTA_F16;
}
