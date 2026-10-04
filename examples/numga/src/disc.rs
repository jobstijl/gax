//! A unit ball drawn as a disc of the canvas, seen along `-z` (x to the right, y up, z towards
//! the viewer): the pixels of the disc are the points of the ball's front hemisphere.

use crate::points::Point2;

/// A disc of the canvas showing the unit ball: its centre and radius in pixels.
#[derive(Clone, Copy, Debug)]
pub struct Disc {
    /// The centre.
    pub centre: Point2,
    /// The radius in pixels.
    pub radius: f32,
}

impl Disc {
    /// The disc at `centre` with `radius` pixels.
    pub fn new(centre: Point2, radius: f32) -> Disc {
        Disc {
            centre: centre.unitized(),
            radius,
        }
    }

    /// The point of the front hemisphere under pixel `q`, as a unit vector of space, or `None`
    /// outside the disc ([`crate::measure::lift`]).
    pub fn point(&self, q: Point2) -> Option<gax::vga3d::Vector<(), f64>> {
        // The pixel's offset from the centre in radii, y up.
        let off = (q.unitized() - self.centre).gp(self.radius.recip());
        crate::measure::lift(f64::from(off.e20()), -f64::from(off.e01()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hemisphere's points are unit vectors over their pixels, towards the viewer.
    #[test]
    fn pixels_lift_onto_the_front_hemisphere() {
        let disc = Disc::new(Point2::xy(100.0, 100.0), 50.0);
        for (x, y) in [(100.0, 100.0), (130.0, 80.0), (70.0, 140.0), (149.0, 100.0)] {
            let p = disc.point(Point2::xy(x, y)).expect("inside");
            assert!((p.norm() - 1.0).abs() < 1e-6, "{p:?}");
            assert!((p.e1() - f64::from((x - 100.0) / 50.0)).abs() < 1e-6);
            assert!((p.e2() + f64::from((y - 100.0) / 50.0)).abs() < 1e-6);
            assert!(p.e3() >= 0.0);
        }
        assert!(disc.point(Point2::xy(160.0, 100.0)).is_none());
    }
}
