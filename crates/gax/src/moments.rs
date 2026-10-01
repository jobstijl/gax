//! Mass properties of meshes and polygons from their boundary: size, centre of mass and
//! inertia, after De Keninck, Roelfs, Dorst and Eelbode, *Clean up your Mesh! Part 1: Plane and
//! simplex* (2025).
//!
//! A body's moments are one bilinear form on planes (on lines in 2D),
//! `M(P, Q) = ∫ (P & x)(Q & x) dV` over its points `x`: the second moments, which hold the first
//! and zeroth too, since a point's pairing with the plane at infinity is its weight. So the
//! volume, the centre of mass and the inertia are pairings of one extensor
//! ([`pga3d::Moments`](crate::pga3d::Moments), [`pga2d::Moments`](crate::pga2d::Moments)); the
//! moments of parts add, and a motion moves them by composing the form's slots with its action
//! on planes.
//!
//! The form is exact for any closed boundary: each boundary simplex (a triangle, an edge) spans a
//! cone with an apex `o` (any point; the origin by default), the cones' signed sizes cancel
//! outside the body, and over a simplex the integral of a product of two linear functions is
//! exact from its vertices, `V/((k+1)(k+2)) (Σᵢ f(vᵢ) g(vᵢ) + f(Σᵢ vᵢ) g(Σᵢ vᵢ))`. A mesh cut by a
//! plane (the paper's fuel tank) is closed by that plane alone: put the apex on it.

use crate::{Gp, Real};
use core::ops::Add;

/// `∫ f g dV` over a simplex with vertices `vs` and signed size `size`, for the dyad
/// `dyad(x) = f(x) g(x)`: `size / ((k+1)(k+2)) (Σ dyad(vᵢ) + dyad(Σ vᵢ))` with `k + 1 = vs.len()`.
/// Written once for every dimension: the dyad is a form on planes (3D) or lines (2D).
#[inline]
fn simplex_dyads<P, F, T>(vs: &[P], size: T, dyad: impl Fn(P) -> F) -> F
where
    P: Copy + Add<Output = P>,
    F: Add<Output = F> + Gp<T, Output = F>,
    T: Real,
{
    let n = vs.len() as i64;
    let sum = vs[1..].iter().fold(vs[0], |a, &b| a + b);
    let all = vs.iter().fold(dyad(sum), |acc, &v| acc + dyad(v));
    all.gp(size * T::from_i64(n * (n + 1)).recip())
}

#[cfg(feature = "pga3d")]
mod pga3d_moments {
    use super::simplex_dyads;
    use crate::extras::PrincipalInertia;
    use crate::pga3d::{Motor, Plane, Point, Scalar};
    use crate::{Real, Unit};

    /// The moments of a solid in 3D: the form `M(P, Q) = ∫ (P & x)(Q & x) dV` (module docs).
    ///
    /// ```
    /// use gax::pga3d::{Moments, Point};
    /// // The unit cube, as 12 outward-facing triangles.
    /// let v: Vec<Point<(), f64>> = (0..8)
    ///     .map(|i| Point::xyz(f64::from(i & 1), f64::from(i >> 1 & 1), f64::from(i >> 2 & 1)))
    ///     .collect();
    /// let faces = [[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
    ///              [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]];
    /// let m = Moments::of_mesh(&v, &faces);
    /// assert!((m.volume() - 1.0).abs() < 1e-12);
    /// assert!((m.centroid().to_euclidean()[2] - 0.5).abs() < 1e-12);
    /// let (inertia, _frame) = m.inertia(1.0); // principal moments 1/6 each
    /// # let _ = inertia;
    /// ```
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Moments<T: Real = f64> {
        /// `M(P, Q) = ∫ (P & x)(Q & x) dV`.
        pub form: Scalar<(Plane, Plane), T>,
    }

    impl<T: Real> Default for Moments<T> {
        fn default() -> Self {
            Moments {
                form: Scalar::zero(),
            }
        }
    }

    impl<T: Real> core::ops::Add for Moments<T> {
        type Output = Self;
        /// The moments of two bodies together.
        fn add(self, o: Self) -> Self {
            Moments {
                form: self.form + o.form,
            }
        }
    }

    impl<T: Real> core::ops::AddAssign for Moments<T> {
        fn add_assign(&mut self, o: Self) {
            self.form += o.form;
        }
    }

    /// `(P & x)(Q & x)`: the dyad of a point, a form on planes.
    #[inline]
    fn dyad<T: Real>(x: Point<(), T>) -> Scalar<(Plane, Plane), T> {
        (Plane::slot() & x) * (Plane::slot() & x)
    }

    /// The planes whose pairings with a point `w (x, y, z, 1)` are `w x`, `w y`, `w z` and `w`.
    #[inline]
    fn coordinate_planes<T: Real>() -> [Plane<(), T>; 4] {
        let o = Point::xyz(T::zero(), T::zero(), T::zero());
        let unit = |p: Plane<(), T>, x: Point<(), T>| p.gp((p & x).s().recip());
        let one = T::one();
        let z = T::zero();
        // `e0 & o` is the weight; `e1 & (1, 0, 0)` minus `e1 & o` is the x coordinate.
        let w = unit(Plane::new(z, z, z, one), o);
        let along = |p: Plane<(), T>, x: Point<(), T>| p.gp(((p & x).s() - (p & o).s()).recip());
        [
            along(Plane::new(one, z, z, z), Point::xyz(one, z, z)),
            along(Plane::new(z, one, z, z), Point::xyz(z, one, z)),
            along(Plane::new(z, z, one, z), Point::xyz(z, z, one)),
            w,
        ]
    }

    impl<T: Real> Moments<T> {
        /// The moments of the solid bounded by a closed, consistently oriented triangle mesh:
        /// `faces` index `vertices`, each counterclockwise seen from outside.
        pub fn of_mesh(vertices: &[Point<(), T>], faces: &[[usize; 3]]) -> Self {
            Self::of_triangles(
                faces
                    .iter()
                    .map(|f| [vertices[f[0]], vertices[f[1]], vertices[f[2]]]),
                Point::xyz(T::zero(), T::zero(), T::zero()),
            )
        }

        /// The moments of the solid bounded by `triangles` (counterclockwise seen from outside)
        /// and closed, if they are not, by a plane through `apex`: the triangles below a fuel
        /// surface, with the apex on that surface, give the fuel's moments.
        pub fn of_triangles(
            triangles: impl IntoIterator<Item = [Point<(), T>; 3]>,
            apex: Point<(), T>,
        ) -> Self {
            let mut m = Self::default();
            for t in triangles {
                m.add_triangle(t, apex);
            }
            m
        }

        /// Add the cone from `apex` to the triangle `t` (counterclockwise seen from outside).
        pub fn add_triangle(&mut self, t: [Point<(), T>; 3], apex: Point<(), T>) {
            let [a, b, c] = t.map(Point::unitized);
            let o = apex.unitized();
            // The join of the face with the apex: six times the cone's signed volume.
            let size = (a & b & c & o).s() * T::from_f64(-1.0 / 6.0);
            self.form += simplex_dyads(&[a, b, c, o], size, dyad);
        }

        /// The signed volume (positive for an outward-oriented mesh).
        pub fn volume(&self) -> T {
            let w = coordinate_planes::<T>()[3];
            self.form.of(w).of(w).s()
        }

        /// The centre of mass, a point of weight 1 (uniform density).
        pub fn centroid(&self) -> Point<(), T> {
            let [px, py, pz, w] = coordinate_planes::<T>();
            let first = self.form.of(w);
            Point::new(
                first.of(px).s(),
                first.of(py).s(),
                first.of(pz).s(),
                first.of(w).s(),
            )
            .unitized()
        }

        /// `∫ (x − c)ᵢ (x − c)ⱼ dV` about the centre of mass `c`.
        pub fn central_second_moments(&self) -> [[T; 3]; 3] {
            let planes = coordinate_planes::<T>();
            let v = self.volume();
            let c = self.centroid().to_euclidean();
            core::array::from_fn(|i| {
                core::array::from_fn(|j| {
                    self.form.of(planes[i]).of(planes[j]).s() - v * c[i] * c[j]
                })
            })
        }

        /// The moments after moving the body by `m`.
        pub fn moved(&self, m: Unit<Motor<(), T>>) -> Self {
            // ∫ over the moved body of (P & x)(Q & x) is ∫ over this one of
            // ((m << P) & x)((m << Q) & x): compose both slots with the plane map of `~m`.
            let back = m << Plane::<(), T>::slot();
            Moments {
                form: self.form.of(back).at::<1>().of(back),
            }
        }

        /// The inertia of the body at `density`, about its centre of mass: the principal
        /// moments as a [`PrincipalInertia`] in the body frame, and the motor from that frame
        /// (principal axes along x, y, z, ascending moments, centre at the origin) to the world.
        pub fn inertia(&self, density: T) -> (PrincipalInertia<T>, Unit<Motor<(), T>>) {
            let c2 = self.central_second_moments();
            let tr = c2[0][0] + c2[1][1] + c2[2][2];
            let i: [[T; 3]; 3] = core::array::from_fn(|r| {
                core::array::from_fn(|s| {
                    let d = if r == s { tr } else { T::zero() };
                    (d - c2[r][s]) * density
                })
            });
            let (mut values, mut vectors) = crate::linalg::eigh(&i, 10);
            crate::linalg::sort_pairs(&mut values, &mut vectors);
            // A right-handed frame: the third axis the cross product of the first two.
            let [a, b, _] = vectors;
            let c = [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ];
            let rotation = super::rotor_from_frame([a, b, c]);
            let [x, y, z] = self.centroid().to_euclidean();
            let frame = Motor::translation(x, y, z) * rotation;
            let mass = self.volume() * density;
            (PrincipalInertia::new(mass, values), frame)
        }
    }
}

/// The rotation that takes the x, y and z axes to the orthonormal, right-handed directions
/// `frame[0..3]`: its quaternion by Shepperd's method (from the largest of the trace and the
/// diagonal, so no division is small), then its axis and angle, as a gax rotation.
#[cfg(feature = "pga3d")]
fn rotor_from_frame<T: Real>(frame: [[T; 3]; 3]) -> crate::Unit<crate::pga3d::Motor<(), T>> {
    // r[i][k]: component i of the image of axis k.
    let r = |i: usize, k: usize| frame[k][i];
    let (one, quarter) = (T::one(), T::from_f64(0.25));
    let tr = r(0, 0) + r(1, 1) + r(2, 2);
    // The four candidates for 4 q², one per component; take the largest (branch free).
    let cand = [
        one + tr,
        one + r(0, 0) - r(1, 1) - r(2, 2),
        one - r(0, 0) + r(1, 1) - r(2, 2),
        one - r(0, 0) - r(1, 1) + r(2, 2),
    ];
    // q from the candidate k: the largest component is ½√cand[k], the others follow from the
    // off-diagonal sums and differences over 4 times it.
    let from = |k: usize| -> [T; 4] {
        let h = T::from_f64(0.5) * cand[k].max(T::zero()).sqrt();
        let q = quarter / h;
        match k {
            0 => [
                h,
                (r(2, 1) - r(1, 2)) * q,
                (r(0, 2) - r(2, 0)) * q,
                (r(1, 0) - r(0, 1)) * q,
            ],
            1 => [
                (r(2, 1) - r(1, 2)) * q,
                h,
                (r(0, 1) + r(1, 0)) * q,
                (r(0, 2) + r(2, 0)) * q,
            ],
            2 => [
                (r(0, 2) - r(2, 0)) * q,
                (r(0, 1) + r(1, 0)) * q,
                h,
                (r(1, 2) + r(2, 1)) * q,
            ],
            _ => [
                (r(1, 0) - r(0, 1)) * q,
                (r(0, 2) + r(2, 0)) * q,
                (r(1, 2) + r(2, 1)) * q,
                h,
            ],
        }
    };
    // The largest candidate is at least 1 (they sum to 4), lane by lane; the others' NaNs are
    // never selected.
    let (mut best, mut quat) = (cand[0], from(0));
    for (k, &ck) in cand.iter().enumerate().skip(1) {
        let next = from(k);
        quat = core::array::from_fn(|i| T::select_lt(best, ck, next[i], quat[i]));
        best = best.max(ck);
    }
    let [w, x, y, z] = quat;
    let n = (x * x + y * y + z * z).sqrt();
    let angle = T::from_i64(2) * n.atan2(w);
    // No rotation (to rounding): any axis, here z.
    let tiny = T::epsilon() * T::epsilon();
    let safe = T::select_lt(n, tiny, one, n);
    let (ax, ay, az) = (
        T::select_lt(n, tiny, T::zero(), x / safe),
        T::select_lt(n, tiny, T::zero(), y / safe),
        T::select_lt(n, tiny, one, z / safe),
    );
    crate::pga3d::Motor::rotation_about(ax, ay, az, angle)
}

#[cfg(feature = "pga2d")]
mod pga2d_moments {
    use super::simplex_dyads;
    use crate::pga2d::{Line, Motor, Point, Scalar};
    use crate::{Real, Unit};

    /// The moments of a region of the plane: the form `M(P, Q) = ∫ (P & x)(Q & x) dA` on lines
    /// (module docs).
    ///
    /// ```
    /// use gax::pga2d::{Moments, Point};
    /// // A 2 × 1 rectangle, counterclockwise.
    /// let p = [Point::xy(0.0, 0.0), Point::xy(2.0, 0.0), Point::xy(2.0, 1.0), Point::xy(0.0, 1.0)];
    /// let m = Moments::<f64>::of_polygon(&p);
    /// assert!((m.area() - 2.0).abs() < 1e-12);
    /// assert_eq!(m.centroid().to_euclidean(), [1.0, 0.5]);
    /// // The polar moment about the centre, (a² + b²) A / 12 at density 1.
    /// assert!((m.polar_moment(1.0) - 5.0 / 6.0).abs() < 1e-12);
    /// ```
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Moments<T: Real = f64> {
        /// `M(P, Q) = ∫ (P & x)(Q & x) dA`.
        pub form: Scalar<(Line, Line), T>,
    }

    impl<T: Real> Default for Moments<T> {
        fn default() -> Self {
            Moments {
                form: Scalar::zero(),
            }
        }
    }

    impl<T: Real> core::ops::Add for Moments<T> {
        type Output = Self;
        /// The moments of two regions together.
        fn add(self, o: Self) -> Self {
            Moments {
                form: self.form + o.form,
            }
        }
    }

    impl<T: Real> core::ops::AddAssign for Moments<T> {
        fn add_assign(&mut self, o: Self) {
            self.form += o.form;
        }
    }

    /// `(P & x)(Q & x)`: the dyad of a point, a form on lines.
    #[inline]
    fn dyad<T: Real>(x: Point<(), T>) -> Scalar<(Line, Line), T> {
        (Line::slot() & x) * (Line::slot() & x)
    }

    /// The lines whose pairings with a point `w (x, y, 1)` are `w x`, `w y` and `w`.
    #[inline]
    fn coordinate_lines<T: Real>() -> [Line<(), T>; 3] {
        let (z, one) = (T::zero(), T::one());
        let o = Point::xy(z, z);
        let w = Line::new(z, z, one);
        let w = w.gp((w & o).s().recip());
        let along = |l: Line<(), T>, x: Point<(), T>| l.gp(((l & x).s() - (l & o).s()).recip());
        [
            along(Line::new(one, z, z), Point::xy(one, z)),
            along(Line::new(z, one, z), Point::xy(z, one)),
            w,
        ]
    }

    impl<T: Real> Moments<T> {
        /// The moments of the region inside a closed polygon, its vertices counterclockwise.
        pub fn of_polygon(vertices: &[Point<(), T>]) -> Self {
            let o = Point::xy(T::zero(), T::zero());
            let mut m = Self::default();
            for (i, &a) in vertices.iter().enumerate() {
                m.add_edge([a, vertices[(i + 1) % vertices.len()]], o);
            }
            m
        }

        /// Add the triangle from `apex` to the edge `e` (counterclockwise around the region).
        pub fn add_edge(&mut self, e: [Point<(), T>; 2], apex: Point<(), T>) {
            let [a, b] = e.map(Point::unitized);
            let o = apex.unitized();
            // The join of the edge with the apex: twice the triangle's signed area.
            let size = (a & b & o).s() * T::from_f64(0.5);
            self.form += simplex_dyads(&[a, b, o], size, dyad);
        }

        /// The signed area (positive for a counterclockwise boundary).
        pub fn area(&self) -> T {
            let w = coordinate_lines::<T>()[2];
            self.form.of(w).of(w).s()
        }

        /// The centre of mass, a point of weight 1 (uniform density).
        pub fn centroid(&self) -> Point<(), T> {
            let [lx, ly, w] = coordinate_lines::<T>();
            let first = self.form.of(w);
            let a = first.of(w).s();
            Point::xy(first.of(lx).s() / a, first.of(ly).s() / a)
        }

        /// `∫ (x − c)ᵢ (x − c)ⱼ dA` about the centre of mass `c`.
        pub fn central_second_moments(&self) -> [[T; 2]; 2] {
            let lines = coordinate_lines::<T>();
            let a = self.area();
            let c = self.centroid().to_euclidean();
            core::array::from_fn(|i| {
                core::array::from_fn(|j| self.form.of(lines[i]).of(lines[j]).s() - a * c[i] * c[j])
            })
        }

        /// The moment of inertia about the centre of mass at `density`: `ρ ∫ |x − c|² dA`.
        pub fn polar_moment(&self, density: T) -> T {
            let c = self.central_second_moments();
            (c[0][0] + c[1][1]) * density
        }

        /// The moments after moving the region by `m`.
        pub fn moved(&self, m: Unit<Motor<(), T>>) -> Self {
            let back = m << Line::<(), T>::slot();
            Moments {
                form: self.form.of(back).at::<1>().of(back),
            }
        }
    }
}

#[cfg(feature = "pga2d")]
pub use pga2d_moments::Moments as Moments2;
#[cfg(feature = "pga3d")]
pub use pga3d_moments::Moments as Moments3;
