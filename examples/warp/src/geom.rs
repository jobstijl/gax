//! Constructions written once for both games: the Plane in PGA2D and the Tunnel and the colour
//! space in PGA3D. The kinds in between, and the result's, follow from gax's product tables.

use gax::{Dot, Wedge, pga2d, pga3d};

/// The origin of the plane: the centre of every turn in the Plane and the HUD.
pub const ORIGIN: pga2d::Point<(), f32> = pga2d::Point::new(0.0, 0.0, 1.0);

/// The origin of space: the Tunnel's straightened origin, a camera's eye in its own frame, and
/// the centre of a model.
pub const ORIGIN3: pga3d::Point<(), f32> = pga3d::Point::new(0.0, 0.0, 0.0, 1.0);

/// A direction of either game (a point at infinity, PGA2D or PGA3D) and its length, the ideal
/// norm.
pub trait Direction: Copy + core::ops::Mul<f32, Output = Self> {
    /// The squared length.
    fn length_squared(self) -> f32;
    /// The length.
    fn length(self) -> f32;
}

impl Direction for pga2d::Point<(), f32> {
    fn length_squared(self) -> f32 {
        self.ideal_norm_squared()
    }
    fn length(self) -> f32 {
        self.ideal_norm()
    }
}

impl Direction for pga3d::Point<(), f32> {
    fn length_squared(self) -> f32 {
        self.ideal_norm_squared()
    }
    fn length(self) -> f32 {
        self.ideal_norm()
    }
}

/// `d` scaled to length `len` (a zero direction stays zero).
pub fn with_length<D: Direction>(d: D, len: f32) -> D {
    d * (len / d.length().max(1e-9))
}

/// `d` shortened to length `max` if it is longer.
pub fn clamp_length<D: Direction>(d: D, max: f32) -> D {
    let n = d.length();
    if n > max { d * (max / n) } else { d }
}

/// An inverse-square pull along the offset `d` towards its source, of strength `k`, with a
/// core softened by `soft`: `k d / (|d|² + soft)^{3/2}`.
pub fn inverse_square<D: Direction>(d: D, k: f32, soft: f32) -> D {
    d * (k / length_cubed(d.length_squared() + soft))
}

/// `a` to `b` by `t` (points of weight 1, or directions), in either game.
pub fn lerp<X>(a: X, b: X, t: f32) -> X
where
    X: Copy
        + core::ops::Add<Output = X>
        + core::ops::Sub<Output = X>
        + core::ops::Mul<f32, Output = X>,
{
    a + (b - a) * t
}

/// `n` steps of `step` from `start`, as the pairs `(x, step x)`: the chords of a circle, the
/// arms of a reticle, a spiral's turns.
pub fn orbit<M, X>(start: X, n: usize, step: M) -> impl Iterator<Item = (X, X)>
where
    M: Copy + core::ops::Shr<X, Output = X>,
    X: Copy,
{
    let mut a = start;
    (0..n).map(move |_| {
        let b = step >> a;
        (core::mem::replace(&mut a, b), b)
    })
}

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

/// The cube of a length from its square, `(r²)^{3/2}`: the denominator of an inverse-square
/// law written along the offset, `s d / |d|³`, softened by adding to the square.
#[inline]
pub fn length_cubed(squared: f32) -> f32 {
    squared * gax::Real::sqrt(squared)
}

#[cfg(test)]
mod tests {
    use super::{foot, length_cubed};

    #[test]
    fn a_length_cubed() {
        assert_eq!(length_cubed(4.0), 8.0);
        // 2.5^1.5.
        assert!((length_cubed(2.5) - 3.952_847).abs() < 1e-6);
    }

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
