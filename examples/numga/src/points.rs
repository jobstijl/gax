//! The points, directions and maps the drawing code works with: PGA points of either
//! precision, drawn as `f32` PGA points, and the maps of the plane between data, the screen and
//! pixels.
//!
//! The examples hand over their own gax values through traits rather than `From` conversions:
//! both sides are gax types, so the orphan rule leaves no room for an `f64` to `f32`
//! conversion here. Positions are PGA points; a VGA vector is a direction only (to draw where it
//! points, add it to the origin).

use gax::{Coef, pga2d, pga3d, vga2d, vga3d};
use pga2d::Motor;

/// A point of the plane, as the drawing code holds it.
pub type Point2 = pga2d::Point<(), f32>;
/// A point of space, as the drawing code holds it.
pub type Point3 = pga3d::Point<(), f32>;

/// The origin of the plane.
pub const ORIGIN2: Point2 = Point2::new(0.0, 0.0, 1.0);
/// The origin of space.
pub const ORIGIN3: Point3 = Point3::new(0.0, 0.0, 0.0, 1.0);

/// A map of the plane on points, as gax holds it: a slot map from points to points.
pub type Map2 = pga2d::Point<(pga2d::Point,), f32>;

/// The stretch by `sx` across and `sy` up about the origin, as a map on points (the weight
/// kept; a negative factor mirrors).
pub fn stretch(sx: f32, sy: f32) -> Map2 {
    Map2::from_coeffs([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
}

/// The map taking the box with corners `from` onto the box with corners `to`, corner to
/// corner (each axis stretched on its own; a box's corners in the other order mirror it): a
/// translation of `from[0]` to the origin, the stretch, and a translation to `to[0]`.
pub fn box_map(from: [Point2; 2], to: [Point2; 2]) -> Map2 {
    let (a, b) = (from[1] - from[0], to[1] - to[0]);
    let s = stretch(b.e20() / a.e20(), b.e01() / a.e01());
    Motor::between(ORIGIN2, to[0]) >> s.of(Motor::between(from[0], ORIGIN2) >> Point2::slot())
}

/// The coefficient types the examples compute in.
pub trait Precision: Coef {
    /// Rounded to `f32`, the drawing precision.
    fn to_f32(self) -> f32;
}
impl Precision for f32 {
    fn to_f32(self) -> f32 {
        self
    }
}
impl Precision for f64 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

/// A point of the plane.
pub trait Pos2: Copy {
    /// As the drawing code's point.
    fn point2(self) -> Point2;
}

/// A point of space.
pub trait Pos3: Copy {
    /// As the drawing code's point.
    fn point3(self) -> Point3;
}

/// A direction (a displacement) in the plane.
pub trait Dir2: Copy {
    /// As the drawing code's ideal point.
    fn dir2(self) -> Point2;
}

/// A direction (a displacement) in space.
pub trait Dir3: Copy {
    /// As the drawing code's ideal point.
    fn dir3(self) -> Point3;
}

impl<T: Precision> Pos2 for pga2d::Point<(), T> {
    fn point2(self) -> Point2 {
        self.map_coefs(T::to_f32)
    }
}
impl<T: Precision> Pos3 for pga3d::Point<(), T> {
    fn point3(self) -> Point3 {
        self.map_coefs(T::to_f32)
    }
}

// Directions: a PGA point's weightless part (a weight, if any, is dropped), the `Direction`
// kinds, and VGA vectors.
impl<T: Precision> Dir2 for pga2d::Point<(), T> {
    fn dir2(self) -> Point2 {
        Point2::direction(self.e20().to_f32(), self.e01().to_f32())
    }
}
impl<T: Precision> Dir3 for pga3d::Point<(), T> {
    fn dir3(self) -> Point3 {
        Point3::direction(
            self.e032().to_f32(),
            self.e013().to_f32(),
            self.e021().to_f32(),
        )
    }
}
impl<T: Precision> Dir2 for pga2d::Direction<(), T> {
    fn dir2(self) -> Point2 {
        Point2::from(self.map_coefs(T::to_f32))
    }
}
impl<T: Precision> Dir3 for pga3d::Direction<(), T> {
    fn dir3(self) -> Point3 {
        Point3::from(self.map_coefs(T::to_f32))
    }
}
impl<T: Precision> Dir2 for vga2d::Vector<(), T> {
    fn dir2(self) -> Point2 {
        let [x, y] = self.c.map(T::to_f32);
        Point2::direction(x, y)
    }
}
impl<T: Precision> Dir3 for vga3d::Vector<(), T> {
    fn dir3(self) -> Point3 {
        let [x, y, z] = self.c.map(T::to_f32);
        Point3::direction(x, y, z)
    }
}

/// The point a direction reaches from the origin of the plane: where a VGA vector (or any
/// direction) points, as a position to draw.
pub fn reach2(d: impl Dir2) -> Point2 {
    ORIGIN2 + d.dir2()
}

/// The point a direction reaches from the origin of space.
pub fn reach3(d: impl Dir3) -> Point3 {
    ORIGIN3 + d.dir3()
}

/// A point of space seen from above (along `-z`): the point of the plane under it, its `z`
/// dropped. Directions stay directions.
pub fn from_above(p: impl Pos3) -> Point2 {
    let p = p.point3();
    Point2::new(p.e032(), p.e013(), p.e123())
}

/// A drawing point whose coefficients can be checked.
pub trait Finite: Copy {
    /// Whether every coefficient is finite.
    fn is_finite(self) -> bool;
}
impl Finite for Point2 {
    fn is_finite(self) -> bool {
        self.c.iter().all(|x| x.is_finite())
    }
}
impl Finite for Point3 {
    fn is_finite(self) -> bool {
        self.c.iter().all(|x| x.is_finite())
    }
}

/// Whether a sample did not blow up: every coefficient finite.
pub fn finite(p: impl Finite) -> bool {
    p.is_finite()
}
