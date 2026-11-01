//! numga's spacetime of the plane, `STA2D = x+y+t-`: gax's `sta2d` (`x y t` are `e1 e2 e4`),
//! with numga's reading of points as antivectors (intrinsic bivectors through time) and its
//! planes `x`, `y`, `t`.

pub use gax::sta2d::*;

/// The observer's direction in time, which pairs to one with `t` (numga's `-mv.t.dual()`).
pub fn time() -> intrinsic::Bivector {
    -Vector::new(0.0, 0.0, 1.0).hodge()
}

/// The planes `x` and `t`.
#[cfg(test)] // the symmetry checks reflect in it
pub fn x_plane() -> Vector {
    Vector::new(1.0, 0.0, 0.0)
}

pub fn t_plane() -> Vector {
    Vector::new(0.0, 0.0, 1.0)
}
