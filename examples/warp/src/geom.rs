//! Constructions written once for both games: the Plane in PGA2D and the Tunnel and the colour
//! space in PGA3D. The kinds in between, and the result's, follow from gax's product tables.

use gax::{Dot, Wedge};

/// The foot of `x` on the flat `onto`, up to weight: the meet of `onto` with the flat through
/// `x` orthogonal to it, `(x | onto) ^ onto`. In PGA2D a point and a line give a line, then a
/// point; in PGA3D a point and a line give a plane, then a point (and a point and a plane give a
/// line, then a point). Unitize the result for the Euclidean foot.
pub fn foot<X, F: Copy>(x: X, onto: F) -> <<X as Dot<F>>::Output as Wedge<F>>::Output
where
    X: Dot<F>,
    <X as Dot<F>>::Output: Wedge<F>,
{
    x.dot(onto).wedge(onto)
}

#[cfg(test)]
mod tests {
    use super::foot;

    #[test]
    fn feet_in_2d_and_3d() {
        use gax::{pga2d, pga3d};
        // The x axis of the plane, and of space.
        let l2 = pga2d::Point::<(), f32>::xy(0.0, 0.0) & pga2d::Point::xy(1.0, 0.0);
        let f2 = foot(pga2d::Point::xy(3.0, 2.0), l2).unitized();
        assert_eq!(f2.to_euclidean(), [3.0, 0.0]);
        let l3 = pga3d::Point::<(), f32>::xyz(0.0, 0.0, 0.0) & pga3d::Point::xyz(1.0, 0.0, 0.0);
        let f3 = foot(pga3d::Point::xyz(3.0, 2.0, -1.0), l3).unitized();
        assert_eq!(f3.to_euclidean(), [3.0, 0.0, 0.0]);
        // A point onto a plane: the floor.
        let floor = pga3d::Plane::<(), f32>::new(0.0, 0.0, 1.0, 0.0);
        let f = foot(pga3d::Point::xyz(3.0, 2.0, -1.0), floor).unitized();
        assert_eq!(f.to_euclidean(), [3.0, 2.0, 0.0]);
    }
}
