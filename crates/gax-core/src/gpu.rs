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

/// The IEEE binary16 (`f16`) bit pattern nearest to `x` (round to nearest, ties to even), for
/// the `{Kind}Gpu16` layouts of the `f16` WGSL modules. Out of range values become infinities,
/// tiny ones subnormals or zero, and NaN stays NaN.
///
/// ```
/// use gax_core::gpu::{f16_bits, f16_to_f32};
/// assert_eq!(f16_bits(1.0), 0x3c00);
/// assert_eq!(f16_to_f32(f16_bits(0.1)), 0.099_975_586);
/// assert_eq!(f16_bits(65_520.0), 0x7c00); // rounds to infinity
/// ```
#[must_use]
pub fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        // Infinity, or NaN (kept quiet, with the top of its payload).
        let nan = if man == 0 {
            0
        } else {
            0x0200 | (man >> 13) as u16
        };
        return sign | 0x7c00 | nan;
    }
    // Round `m` to the nearest multiple of `2^shift`, ties to even, and shift it down.
    let round = |m: u64, shift: u32| -> u64 {
        if shift >= 40 {
            return 0;
        }
        let half = 1u64 << (shift - 1);
        (m + half - 1 + ((m >> shift) & 1)) >> shift
    };
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // A subnormal (or zero): the value in units of 2^-24.
        let m = u64::from(man | 0x0080_0000);
        let shift = (14 - e) as u32;
        return sign | round(m, shift) as u16;
    }
    // A normal number; rounding up may carry into the exponent, up to infinity.
    let h = ((e as u64) << 10) + round(u64::from(man), 13);
    sign | h as u16
}

/// The `f32` value of an IEEE binary16 bit pattern (exact: every `f16` is an `f32`).
#[must_use]
pub fn f16_to_f32(h: u16) -> f32 {
    let h = u32::from(h);
    let sign = (h & 0x8000) << 16;
    let exp = (h >> 10) & 0x1f;
    let man = h & 0x03ff;
    if exp == 0 {
        // Zero or a subnormal: man * 2^-24, exact in f32.
        let v = man as f32 * (1.0 / 16_777_216.0);
        return if sign == 0 { v } else { -v };
    }
    let bits = if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod f16_tests {
    use super::{f16_bits, f16_to_f32};

    #[test]
    fn every_f16_round_trips() {
        for h in 0..=u16::MAX {
            let x = f16_to_f32(h);
            if x.is_nan() {
                assert!(f16_to_f32(f16_bits(x)).is_nan());
            } else {
                assert_eq!(f16_bits(x), h, "{h:#06x} -> {x}");
            }
        }
    }

    /// The conversion picks the nearest `f16`, ties to even, for f32 values between and
    /// around every pair of neighbours.
    #[test]
    fn rounds_to_nearest_even() {
        for h in 0..0x7bffu16 {
            let (a, b) = (f16_to_f32(h), f16_to_f32(h + 1));
            let mid = a + (b - a) * 0.5;
            let even = if h % 2 == 0 { h } else { h + 1 };
            assert_eq!(f16_bits(mid), even, "tie between {h:#06x} and the next");
            let below = f32::from_bits(mid.to_bits() - 1);
            let above = f32::from_bits(mid.to_bits() + 1);
            assert_eq!(f16_bits(below), h, "below the midpoint of {h:#06x}");
            assert_eq!(f16_bits(above), h + 1, "above the midpoint of {h:#06x}");
            assert_eq!(f16_bits(-below), h | 0x8000);
        }
        // Past the largest finite f16 (65504), halfway to the next binade: infinity.
        assert_eq!(f16_bits(65_519.99), 0x7bff);
        assert_eq!(f16_bits(65_520.0), 0x7c00);
        assert_eq!(f16_bits(f32::MAX), 0x7c00);
        assert_eq!(f16_bits(1e-10), 0);
    }
}
