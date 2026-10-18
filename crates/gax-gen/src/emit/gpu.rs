//! The GPU layout types of the kinds (ADR-028).

use super::{Emitter, IMPL_O};
use crate::spec::KindSpec;
use std::fmt::Write as _;

impl Emitter<'_> {
    /// `{Kind}Gpu` per kind: the layout of the WGSL modules, `ceil(N/4)` `vec4<f32>`, with
    /// conversions, `bytemuck::Pod`, and compile-time layout assertions; and `GpuMat`
    /// conversions for maps between kinds of 3 or 4 coefficients (ADR-028).
    #[allow(clippy::too_many_lines)]
    pub(super) fn gpu_types(&mut self, gate: &str) {
        let spec = self.spec;
        // The WGSL modules take kinds up to `WGSL_MAX` coefficients; larger ones (from 8D on)
        // have no GPU layout either.
        for k in spec.kinds.iter().filter(|k| k.fits_wgsl()) {
            let (name, n) = (&k.name, k.layout.len());
            let m = n.div_ceil(4);
            let _ = write!(
                self.out,
                "{gate}
/// [`{name}`] in the GPU layout of the `gax::wgsl` modules: its {n} coefficients in blade
/// order, four per `vec4<f32>` field, zero-padded (the WGSL struct `{name}`).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct {name}Gpu {{
    /// The coefficients, `c[i / 4][i % 4]` for coefficient `i`.
    pub c: [[f32; 4]; {m}],
}}

// The WGSL layout of `struct {name} {{ c0: vec4<f32>, ... }}`: size 16 per field, align 16.
{gate}
const _: () = {{
    assert!(core::mem::size_of::<{name}Gpu>() == {size});
    assert!(core::mem::align_of::<{name}Gpu>() == 16);
    assert!(core::mem::offset_of!({name}Gpu, c) == 0);
}};

// SAFETY: `repr(C, align(16))` over `[[f32; 4]; {m}]` (size a multiple of 16): no padding
// bytes, and every bit pattern is a valid `f32`.
{gate}
unsafe impl gx::bytemuck::Zeroable for {name}Gpu {{}}
// SAFETY: as above.
{gate}
unsafe impl gx::bytemuck::Pod for {name}Gpu {{}}

{gate}
{IMPL_O} From<{name}<(), f32, O>> for {name}Gpu {{
    #[inline]
    fn from(x: {name}<(), f32, O>) -> Self {{
        let mut c = [[0.0; 4]; {m}];
        for (i, v) in x.c.iter().enumerate() {{
            c[i / 4][i % 4] = *v;
        }}
        {name}Gpu {{ c }}
    }}
}}

{gate}
{IMPL_O} From<gx::Unit<{name}<(), f32, O>>> for {name}Gpu {{
    #[inline]
    fn from(x: gx::Unit<{name}<(), f32, O>>) -> Self {{
        x.into_inner().into()
    }}
}}

{gate}
{IMPL_O} From<{name}Gpu> for {name}<(), f32, O> {{
    /// The coefficients, in the orientation the receiving type asks for: the GPU layout carries
    /// none (ADR-051).
    #[inline]
    fn from(g: {name}Gpu) -> Self {{
        {name}::from_coeffs(core::array::from_fn(|i| g.c[i / 4][i % 4]))
    }}
}}

/// [`{name}`] in the GPU layout of the `gax::wgsl` `f16` modules (`gax::{alg}_f16`): its {n}
/// coefficients as IEEE binary16 bit patterns, four per `vec4<f16>` field, zero-padded.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct {name}Gpu16 {{
    /// The coefficients' `f16` bits, `c[i / 4][i % 4]` for coefficient `i`.
    pub c: [[u16; 4]; {m}],
}}

// The WGSL layout of `struct {name} {{ c0: vec4<f16>, ... }}`: size 8 per field, align 8.
{gate}
const _: () = {{
    assert!(core::mem::size_of::<{name}Gpu16>() == {size16});
    assert!(core::mem::align_of::<{name}Gpu16>() == 8);
}};

// SAFETY: `repr(C, align(8))` over `[[u16; 4]; {m}]` (size a multiple of 8): no padding bytes,
// and every bit pattern is a valid `u16`.
{gate}
unsafe impl gx::bytemuck::Zeroable for {name}Gpu16 {{}}
// SAFETY: as above.
{gate}
unsafe impl gx::bytemuck::Pod for {name}Gpu16 {{}}

{gate}
{IMPL_O} From<{name}<(), f32, O>> for {name}Gpu16 {{
    /// Each coefficient rounded to the nearest `f16`.
    #[inline]
    fn from(x: {name}<(), f32, O>) -> Self {{
        let mut c = [[0; 4]; {m}];
        for (i, v) in x.c.iter().enumerate() {{
            c[i / 4][i % 4] = gx::gpu::f16_bits(*v);
        }}
        {name}Gpu16 {{ c }}
    }}
}}

{gate}
{IMPL_O} From<{name}Gpu16> for {name}<(), f32, O> {{
    #[inline]
    fn from(g: {name}Gpu16) -> Self {{
        {name}::from_coeffs(core::array::from_fn(|i| gx::gpu::f16_to_f32(g.c[i / 4][i % 4])))
    }}
}}

",
                size = 16 * m,
                size16 = 8 * m,
                alg = spec.name
            );
        }
        let small = |kind: &str| spec.kinds.iter().any(|k| k.name == kind && k.fits_wgsl());
        for (alias, kind) in spec
            .aliases
            .iter()
            .filter(|a| small(&a.kind))
            .map(|a| (&a.name, &a.kind))
        {
            let _ = writeln!(
                self.out,
                "{gate}
/// Alias of [`{kind}Gpu`].
pub type {alias}Gpu = {kind}Gpu;

{gate}
/// Alias of [`{kind}Gpu16`].
pub type {alias}Gpu16 = {kind}Gpu16;
"
            );
        }
        let rows: Vec<String> = spec
            .kinds
            .iter()
            .filter(|k| k.fits_wgsl())
            .map(|k| {
                let t = format!("{}Gpu", k.name);
                format!(
                    "    (\"{}\", core::mem::size_of::<{t}>(), core::mem::align_of::<{t}>(), core::mem::offset_of!({t}, c), core::mem::size_of::<[f32; 4]>()),",
                    k.name
                )
            })
            .collect();
        let _ = writeln!(
            self.out,
            "{gate}\n/// The Rust layout of each `{{Kind}}Gpu`: `(kind, size, align, offset of c, stride of c)`,\n/// to check against a shader compiler's layout of the WGSL structs.\npub const GPU_LAYOUTS: &[(&str, usize, usize, usize, usize)] = &[\n{}\n];\n",
            rows.join("\n")
        );
        let rows16: Vec<String> = spec
            .kinds
            .iter()
            .filter(|k| k.fits_wgsl())
            .map(|k| {
                let t = format!("{}Gpu16", k.name);
                format!(
                    "    (\"{}\", core::mem::size_of::<{t}>(), core::mem::align_of::<{t}>(), core::mem::offset_of!({t}, c), core::mem::size_of::<[u16; 4]>()),",
                    k.name
                )
            })
            .collect();
        let _ = writeln!(
            self.out,
            "{gate}\n/// [`GPU_LAYOUTS`] for the `{{Kind}}Gpu16` types of the `f16` modules.\npub const GPU_LAYOUTS_F16: &[(&str, usize, usize, usize, usize)] = &[\n{}\n];\n",
            rows16.join("\n")
        );
        let small: Vec<&KindSpec> = spec
            .kinds
            .iter()
            .filter(|k| (3..=4).contains(&k.layout.len()))
            .collect();
        for x in &small {
            for y in &small {
                let (xn, yn, c) = (&x.name, &y.name, y.layout.len());
                let _ = write!(
                    self.out,
                    "{gate}
/// The map as a WGSL `mat{c}x{r}<f32>` (columns are inputs; transposes the output-first layout).
{IMPL_O} From<{xn}<({yn},), f32, O>> for gx::GpuMat<{c}> {{
    #[inline]
    fn from(m: {xn}<({yn},), f32, O>) -> Self {{
        let mut cols = [[0.0; 4]; {c}];
        for (o, row) in m.c.iter().enumerate() {{
            for (i, col) in cols.iter_mut().enumerate() {{
                col[o] = <({yn},) as gx::Slots>::get_flat(row, i);
            }}
        }}
        gx::GpuMat {{ cols }}
    }}
}}

{gate}
impl From<gx::GpuMat<{c}>> for {xn}<({yn},), f32> {{
    #[inline]
    fn from(g: gx::GpuMat<{c}>) -> Self {{
        {xn}::from_coeffs(core::array::from_fn(|o| {{
            <({yn},) as gx::Slots>::from_flat(&mut |i| g.cols[i][o], 0)
        }}))
    }}
}}

",
                    r = x.layout.len()
                );
            }
        }
    }
}
