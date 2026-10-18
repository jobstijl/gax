//! Straightened space: the PGA3D space of the simulation, where `x` and `y` run across the
//! tunnel and `z` is the arc length `s` along it. The tunnel's axis is a line; the distance
//! from it is a join, the foot on it a meet, and turns about it are motors.

use gax::Unit;
use gax::pga3d::{Motor, Point, intrinsic};

/// A point or a direction in straightened coordinates.
pub type P = Point<(), f32>;

/// The point `(x, y)` across the tunnel at arc length `s`.
pub fn at(x: f32, y: f32, s: f32) -> P {
    Point::xyz(x, y, s)
}

/// The direction `(x, y, s)`.
pub fn dir(x: f32, y: f32, s: f32) -> P {
    Point::direction(x, y, s)
}

/// The tunnel's axis, the join of the origin and the direction `z`: an intrinsic line
/// (ADR-051).
pub fn axis() -> intrinsic::Line<(), f32> {
    at(0.0, 0.0, 0.0) & dir(0.0, 0.0, 1.0)
}

/// The arc length of a point.
pub fn arc(p: P) -> f32 {
    p.e021() / p.e123()
}

/// The foot of `p` on the axis: the meet of the axis with the plane through `p` orthogonal to
/// it (weight 1).
pub fn foot(p: P) -> P {
    crate::geom::foot(p, axis()).unitized()
}

/// The distance of `p` from the axis: the norm of their join (the axis is a unit line).
pub fn off_axis(p: P) -> f32 {
    (axis() & p).norm()
}

/// The rotation about the tunnel's axis by `angle`.
pub fn about_axis(angle: f32) -> Unit<Motor<(), f32>> {
    Motor::rotation(axis(), angle)
}

/// The point at distance `r` from the axis, at `angle` around it, at arc length `s`: `(r, 0, s)`
/// turned about the axis.
pub fn around(r: f32, angle: f32, s: f32) -> P {
    about_axis(angle) >> at(r, 0.0, s)
}

/// Does the segment `a → b` pass within `r` of `c`? The distance to the segment's line is the
/// weight of the plane `a & b & c` over the weight of the line `a & b`; whether the nearest
/// point lies between the ends is the sign of `c` against the planes through the ends
/// orthogonal to the line (`l | a`, `l | b`).
pub fn segment_hits_sphere(a: P, b: P, c: P, r: f32) -> bool {
    let dist = |p: P, q: P| (p & q).norm();
    if dist(a, c) < r || dist(b, c) < r {
        return true;
    }
    let l: intrinsic::Line<(), f32> = a & b;
    let len = l.norm();
    if len < 1e-6 {
        return false;
    }
    let between = ((l | a) & c).s() * ((l | b) & c).s() < 0.0;
    between && (l & c).norm() / len < r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tunnel::tests::close;

    #[test]
    fn segments_hit_spheres_by_the_join() {
        let a = at(0.0, 0.0, 0.0);
        let b = at(0.0, 0.0, 10.0);
        assert!(segment_hits_sphere(a, b, at(0.5, 0.0, 5.0), 0.6));
        assert!(!segment_hits_sphere(a, b, at(0.7, 0.0, 5.0), 0.6));
        // Beyond the ends: only the endpoints count.
        assert!(!segment_hits_sphere(a, b, at(0.3, 0.0, 11.0), 0.6));
        assert!(segment_hits_sphere(a, b, at(0.3, 0.0, 10.4), 0.6));
        // Diagonal.
        let c = at(3.0, 4.0, 12.0);
        assert!(segment_hits_sphere(a, c, at(1.5, 2.0, 6.0), 0.1));
    }

    #[test]
    fn the_axis_the_foot_and_turns_about_it() {
        let p = at(3.0, 4.0, 7.0);
        assert!((off_axis(p) - 5.0).abs() < 1e-5);
        assert!(close(foot(p), at(0.0, 0.0, 7.0)));
        let q = around(2.0, core::f32::consts::FRAC_PI_2, 3.0);
        assert!(close(q, at(0.0, 2.0, 3.0)), "{:?}", q.to_euclidean());
    }
}
