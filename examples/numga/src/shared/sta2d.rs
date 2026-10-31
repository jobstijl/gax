//! numga's spacetime of the plane, `STA2D = x+y+t-`: two spatial directions that square to one
//! and a time that squares to minus one. gax has no standard algebra for it; it is declared here,
//! with numga's blades in numga's order (`x y t`, `xy xt yt`, `xyt`), and shared by the ports of
//! Lie advection and of piezoelectricity, which use part of it.
//!
//! Points are antivectors, intrinsic: the metric duals of the vectors, so that the point at
//! `(x, y)` is `x yt − y xt` and the observer's direction in time, minus the dual of `t`, is
//! `xy`. The spatial planes `x`, `y` read a point's place, and `t` its time.
//!
//! The parts are numga's sub-kinds that the ports type with (guide §9): `Spatial` (`x y`), the
//! planes of space, which read positions and displacements; `Temporal` (`xt yt`), the planes
//! through time, extrinsic an electric field and intrinsic a spatial point or displacement; and
//! `Magnetic` (`xy`) with its complement `Time` (`t`).

gax::algebra! {
    algebra sta2d "numga's spacetime of the plane, `x+y+t-`: space `x` and `y` square to one, time `t` to minus one.";
    basis ex = 1, ey = 1, et = -1;
    kind Scalar = [1];
    versor Vector = [ex, ey, et] "A plane of spacetime (a 1-form): a gradient, a worldline's normal; as a versor, a reflection.";
    kind Bivector = [exy, ext, eyt] "A field: `xy` magnetic, `xt yt` electric; intrinsic, an event or a displacement.";
    kind Pseudoscalar = [exyt];
    versor Even = [1, exy, ext, eyt] "A rotation of space (`xy`) or a boost (`xt`, `yt`).";
    part Spatial = [ex, ey] "A plane of space: reads positions and displacements. Its complement is `Temporal`.";
    part Temporal = [ext, eyt] "A plane through time: extrinsic, an electric field; intrinsic, a spatial point or displacement. Its complement is `Spatial`.";
    part Magnetic = [exy] "The plane of space: a magnetic field; intrinsic, the observer's direction in time. Its complement is `Time`.";
    part Time = [et] "Time's plane: reads an event's time. Its complement is `Magnetic`.";
}

pub use sta2d::*;

/// A point of the plane's spacetime, or a displacement (numga's `Point-`).
pub type Point = sta2d::intrinsic::Bivector;

/// The point at `(x, y)` at time zero: the metric dual of its vector (numga's
/// `x * mv.x.dual() + y * mv.y.dual()`).
pub fn point(x: f64, y: f64) -> Point {
    Vector::new(x, y, 0.0).hodge()
}

/// The observer's direction in time, which pairs to one with `t` (numga's `-mv.t.dual()`).
pub fn time() -> Point {
    -Vector::new(0.0, 0.0, 1.0).hodge()
}

/// The planes `x`, `y` and `t`.
pub fn x_plane() -> Vector {
    Vector::new(1.0, 0.0, 0.0)
}

pub fn y_plane() -> Vector {
    Vector::new(0.0, 1.0, 0.0)
}

pub fn t_plane() -> Vector {
    Vector::new(0.0, 0.0, 1.0)
}

/// A point's place, read by the planes `x` and `y`.
pub fn coordinates(p: Point) -> [f64; 2] {
    [(x_plane() & p).s(), (y_plane() & p).s()]
}

/// The magnetic unit, the plane `xy`.
pub fn xy() -> Bivector {
    Bivector::new(1.0, 0.0, 0.0)
}
