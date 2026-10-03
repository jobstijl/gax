//! What the drawing code accepts as coordinates: arrays of `f32` or `f64`, and gax values of
//! either precision, so the examples hand their points and directions over as they are.

use gax::{pga2d, pga3d, vga2d, vga3d};

/// A point of the plane.
pub trait Pos2: Copy {
    /// Its Euclidean coordinates.
    fn xy(self) -> [f32; 2];
}

/// A point of space.
pub trait Pos3: Copy {
    /// Its Euclidean coordinates.
    fn xyz(self) -> [f32; 3];
}

/// A direction (a displacement) in the plane.
pub trait Dir2: Copy {
    /// Its components.
    fn dxy(self) -> [f32; 2];
}

/// A direction (a displacement) in space.
pub trait Dir3: Copy {
    /// Its components.
    fn dxyz(self) -> [f32; 3];
}

/// `f64` coordinates as `f32`.
pub fn f32s<const N: usize>(a: [f64; N]) -> [f32; N] {
    a.map(|x| x as f32)
}

macro_rules! arrays {
    ($tr:ident $f:ident $n:literal) => {
        impl $tr for [f32; $n] {
            fn $f(self) -> [f32; $n] {
                self
            }
        }
        impl $tr for [f64; $n] {
            fn $f(self) -> [f32; $n] {
                f32s(self)
            }
        }
    };
}
arrays!(Pos2 xy 2);
arrays!(Pos3 xyz 3);
arrays!(Dir2 dxy 2);
arrays!(Dir3 dxyz 3);

impl Pos2 for pga2d::Point<(), f32> {
    fn xy(self) -> [f32; 2] {
        self.to_euclidean()
    }
}
impl Pos2 for pga2d::Point<(), f64> {
    fn xy(self) -> [f32; 2] {
        f32s(self.to_euclidean())
    }
}
impl Pos3 for pga3d::Point<(), f32> {
    fn xyz(self) -> [f32; 3] {
        self.to_euclidean()
    }
}
impl Pos3 for pga3d::Point<(), f64> {
    fn xyz(self) -> [f32; 3] {
        f32s(self.to_euclidean())
    }
}
impl Pos3 for vga3d::Vector<(), f64> {
    fn xyz(self) -> [f32; 3] {
        f32s(self.c)
    }
}
impl Dir2 for pga2d::Point<(), f64> {
    fn dxy(self) -> [f32; 2] {
        f32s([self.e20(), self.e01()])
    }
}
impl Dir2 for vga2d::Vector<(), f64> {
    fn dxy(self) -> [f32; 2] {
        f32s(self.c)
    }
}
impl Dir3 for pga3d::Point<(), f64> {
    fn dxyz(self) -> [f32; 3] {
        f32s([self.e032(), self.e013(), self.e021()])
    }
}
impl Dir3 for vga3d::Vector<(), f64> {
    fn dxyz(self) -> [f32; 3] {
        f32s(self.c)
    }
}

impl Pos2 for vga2d::Vector<(), f64> {
    fn xy(self) -> [f32; 2] {
        f32s(self.c)
    }
}
impl Dir2 for pga2d::Direction<(), f64> {
    fn dxy(self) -> [f32; 2] {
        f32s([self.e20(), self.e01()])
    }
}
impl Dir3 for pga3d::Direction<(), f64> {
    fn dxyz(self) -> [f32; 3] {
        f32s([self.e032(), self.e013(), self.e021()])
    }
}
