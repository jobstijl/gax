//! Plain GPU memory layouts (ADR-028).
//!
//! The generated WGSL modules (`gax::wgsl`) store a kind with `N` coefficients as `ceil(N/4)`
//! `vec4<f32>`, and each algebra has a matching Rust type `{Kind}Gpu` (feature `bytemuck`).
//! Linear maps between kinds of 3 or 4 coefficients go to the GPU as WGSL matrices,
//! [`GpuMat`]. Both are plain `#[repr(C)]` data: `bytemuck::Pod` with the feature on, and no
//! runtime layout machinery. Their layouts are checked at compile time against WGSL's rules.

/// A WGSL matrix `matCxR<f32>` with `R` of 3 or 4: `C` columns of `vec4<f32>` stride (for
/// `R = 3` the fourth lane of each column is padding, kept zero).
///
/// Columns are inputs: column `i` holds the image of the input's basis coefficient `i`, so a
/// shader applies it as `m * x`. A gax map `y = X<(Y,)>` stores `c[o][i]` output first, which
/// is the transpose; the generated `From` conversions transpose.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuMat<const C: usize> {
    /// The columns.
    pub cols: [[f32; 4]; C],
}

impl<const C: usize> Default for GpuMat<C> {
    fn default() -> Self {
        GpuMat {
            cols: [[0.0; 4]; C],
        }
    }
}

const _: () = {
    assert!(core::mem::size_of::<GpuMat<3>>() == 48);
    assert!(core::mem::size_of::<GpuMat<4>>() == 64);
    assert!(core::mem::align_of::<GpuMat<4>>() == 16);
};

// SAFETY: `repr(C, align(16))` over `[[f32; 4]; C]`, whose size is a multiple of 16, so there
// are no padding bytes, and every bit pattern is a valid `f32`.
#[cfg(feature = "bytemuck")]
unsafe impl<const C: usize> bytemuck::Zeroable for GpuMat<C> {}
// SAFETY: as above; `GpuMat` is `Copy` and `'static`.
#[cfg(feature = "bytemuck")]
unsafe impl<const C: usize> bytemuck::Pod for GpuMat<C> {}
