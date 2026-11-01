//! numga's spacetime of the plane, `STA2D = x+y+t-`: gax's `sta2d` (`x y t` are `e1 e2 e4`),
//! with numga's reading of points as antivectors, intrinsic: the metric duals of the vectors, so
//! that the point at `(x, y)` is `x yt − y xt` and the observer's direction in time, minus the
//! dual of `t`, is `xy`. The spatial planes `x`, `y` read a point's place.

pub use gax::sta2d::*;

/// A point of the plane's spacetime, or a displacement (numga's `Point-`).
pub type Point = intrinsic::Bivector;

/// The point at `(x, y)` at time zero: the metric dual of its vector (numga's
/// `x * mv.x.dual() + y * mv.y.dual()`).
pub fn point(x: f64, y: f64) -> Point {
    Vector::new(x, y, 0.0).hodge()
}

/// The observer's direction in time, which pairs to one with `t` (numga's `-mv.t.dual()`).
pub fn time() -> Point {
    -Vector::new(0.0, 0.0, 1.0).hodge()
}

/// A point's place, read by the planes `x` and `y`.
pub fn coordinates(p: Point) -> [f64; 2] {
    [
        (Vector::new(1.0, 0.0, 0.0) & p).s(),
        (Vector::new(0.0, 1.0, 0.0) & p).s(),
    ]
}

/// The magnetic unit, the plane `xy`.
pub fn xy() -> Bivector {
    Bivector::new(1.0, 0.0, 0.0)
}
