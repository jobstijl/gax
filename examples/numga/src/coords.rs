//! What the drawing code accepts as coordinates: gax points, directions and vectors of either
//! precision, so the examples hand their values over as they are. Inside, the drawing code
//! works with `f32` PGA points; plain coordinates appear only as pixels.

use gax::{pga2d, pga3d, vga2d, vga3d};

/// A point of the plane, as an `f32` PGA2D point.
pub type Point2 = pga2d::Point<(), f32>;
/// A point of space, as an `f32` PGA3D point.
pub type Point3 = pga3d::Point<(), f32>;

/// The origin of the plane.
pub const ORIGIN2: Point2 = Point2::new(0.0, 0.0, 1.0);
/// The origin of space.
pub const ORIGIN3: Point3 = Point3::new(0.0, 0.0, 0.0, 1.0);

/// A point of the plane.
pub trait Pos2: Copy {
    /// As an `f32` PGA2D point.
    fn point2(self) -> Point2;
}

/// A point of space.
pub trait Pos3: Copy {
    /// As an `f32` PGA3D point.
    fn point3(self) -> Point3;
}

/// A direction (a displacement) in the plane.
pub trait Dir2: Copy {
    /// As an `f32` PGA2D ideal point.
    fn dir2(self) -> Point2;
}

/// A direction (a displacement) in space.
pub trait Dir3: Copy {
    /// As an `f32` PGA3D ideal point.
    fn dir3(self) -> Point3;
}

/// `x` rounded to `f32` (the drawing precision).
fn f32_of(x: f64) -> f32 {
    x as f32
}

macro_rules! convert {
    ($tr:ident $f:ident -> $o:ty: $($t:ty => |$x:ident| $e:expr),* $(,)?) => {$(
        impl $tr for $t {
            fn $f(self) -> $o {
                let $x = self;
                $e
            }
        }
    )*};
}

// Points, and the VGA vectors as points (their tips, from the origin).
convert!(Pos2 point2 -> Point2:
    Point2 => |p| p,
    pga2d::Point<(), f64> => |p| p.map_coefs(f32_of),
    vga2d::Vector<(), f32> => |v| Point2::xy(v.c[0], v.c[1]),
    vga2d::Vector<(), f64> => |v| v.map_coefs(f32_of).point2(),
);
convert!(Pos3 point3 -> Point3:
    Point3 => |p| p,
    pga3d::Point<(), f64> => |p| p.map_coefs(f32_of),
    vga3d::Vector<(), f32> => |v| Point3::xyz(v.c[0], v.c[1], v.c[2]),
    vga3d::Vector<(), f64> => |v| v.map_coefs(f32_of).point3(),
);

// Directions: a PGA point's weightless part (a weight, if any, is dropped), the `Direction`
// kinds, and VGA vectors.
convert!(Dir2 dir2 -> Point2:
    Point2 => |p| Point2::direction(p.e20(), p.e01()),
    pga2d::Point<(), f64> => |p| p.map_coefs(f32_of).dir2(),
    pga2d::Direction<(), f32> => |d| Point2::from(d),
    pga2d::Direction<(), f64> => |d| Point2::from(d.map_coefs(f32_of)),
    vga2d::Vector<(), f32> => |v| Point2::direction(v.c[0], v.c[1]),
    vga2d::Vector<(), f64> => |v| v.map_coefs(f32_of).dir2(),
);
convert!(Dir3 dir3 -> Point3:
    Point3 => |p| Point3::direction(p.e032(), p.e013(), p.e021()),
    pga3d::Point<(), f64> => |p| p.map_coefs(f32_of).dir3(),
    pga3d::Direction<(), f32> => |d| Point3::from(d),
    pga3d::Direction<(), f64> => |d| Point3::from(d.map_coefs(f32_of)),
    vga3d::Vector<(), f32> => |v| Point3::direction(v.c[0], v.c[1], v.c[2]),
    vga3d::Vector<(), f64> => |v| v.map_coefs(f32_of).dir3(),
);

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
