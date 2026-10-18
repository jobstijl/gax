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

#[cfg(feature = "pga3d")]
use crate::Real;
use crate::simplex::dyads as simplex_dyads;

/// What the moments in 2D and 3D share, on the form's slot kind `$P` (lines or planes): the
/// sum of two bodies, the dyad of a point and the action of motors (`m >> moments`).
macro_rules! scaffolding {
    ($P:ident, $ps:literal, $body:literal, $bodies:literal) => {
        impl<T: Real> Default for Moments<T> {
            fn default() -> Self {
                Moments {
                    form: Scalar::zero(),
                }
            }
        }

        impl<T: Real> core::ops::Add for Moments<T> {
            type Output = Self;
            #[doc = concat!("The moments of two ", $bodies, " together.")]
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

        #[doc = concat!("`(P & x)(Q & x)`: the dyad of a point, a form on ", $ps, ".")]
        #[inline]
        fn dyad<T: Real>(x: Point<(), T>) -> Scalar<($P, $P), T> {
            ($P::slot() & x) * ($P::slot() & x)
        }

        #[doc = concat!("`m >> moments`: the moments after moving the ", $body, " by `m`.")]
        impl<T: Real> crate::Transform<Moments<T>> for Unit<Motor<(), T>> {
            type Output = Moments<T>;
            #[inline]
            fn transform(self, moments: Moments<T>) -> Moments<T> {
                // ∫ over the moved body of (P & x)(Q & x) is ∫ over this one of
                // ((m << P) & x)((m << Q) & x): compose both slots with the map of `~m`.
                let back = self << $P::<(), T>::slot();
                Moments {
                    form: moments.form.of(back).at::<1>().of(back),
                }
            }
        }

        #[doc = concat!("`m << moments`: the moments after moving the ", $body, " by `~m`.")]
        impl<T: Real> crate::TransformInv<Moments<T>> for Unit<Motor<(), T>> {
            type Output = Moments<T>;
            #[inline]
            fn transform_inv(self, moments: Moments<T>) -> Moments<T> {
                self.reverse() >> moments
            }
        }
    };
}
use scaffolding;

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
    /// let m = Moments::of_faces(&v, &faces);
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

    super::scaffolding!(Plane, "planes", "body", "bodies");

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
        /// The moments of the solid bounded by a closed, consistently oriented surface: the
        /// mesh's triangles, each counterclockwise seen from outside (`gax::mesh`).
        #[cfg(feature = "std")]
        pub fn of_mesh(mesh: &crate::mesh::Mesh<Point<(), T>, 3>) -> Self {
            let v = mesh.vertices.sites();
            Self::of_triangles(
                mesh.topology()
                    .simplices(2)
                    .map(|t| [v[t[0]], v[t[1]], v[t[2]]]),
                Point::xyz(T::zero(), T::zero(), T::zero()),
            )
        }

        /// The moments of the solid bounded by a closed, consistently oriented triangle mesh
        /// given as slices: `faces` index `vertices`, each counterclockwise seen from outside.
        pub fn of_faces(vertices: &[Point<(), T>], faces: &[[usize; 3]]) -> Self {
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
            let (mut values, mut vectors) = crate::__private::linalg::eigh(&i, 10);
            crate::__private::linalg::sort_pairs(&mut values, &mut vectors);
            // A right-handed frame: the third axis the cross product of the first two.
            let [a, b, _] = vectors;
            let c = super::cross3(a, b);
            let rotation = super::rotor_from_frame([a, b, c]);
            let [x, y, z] = self.centroid().to_euclidean();
            let frame = Motor::translation(Point::direction(x, y, z)) * rotation;
            let mass = self.volume() * density;
            (PrincipalInertia::new(mass, values), frame)
        }
    }
}

#[cfg(feature = "pga3d")]
/// The cross product of two triples.
pub(crate) fn cross3<T: Real>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The rotation that takes the x, y and z axes to the orthonormal, right-handed directions
/// `frame[0..3]`: its quaternion by Shepperd's method (from the largest of the trace and the
/// diagonal, so no division is small), then its axis and angle, as a gax rotation.
#[cfg(feature = "pga3d")]
pub(crate) fn rotor_from_frame<T: Real>(
    frame: [[T; 3]; 3],
) -> crate::Unit<crate::pga3d::Motor<(), T>> {
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
    crate::pga3d::Motor::rotation_about(crate::pga3d::Point::direction(ax, ay, az), angle)
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
    /// // A motor moves the moments as it moves the region (`m >> moments`).
    /// let turn = gax::pga2d::Motor::rotation(Point::xy(0.0, 0.0), std::f64::consts::FRAC_PI_2);
    /// let [x, y] = (turn >> m).centroid().to_euclidean();
    /// assert!((x + 0.5).abs() < 1e-12 && (y - 1.0).abs() < 1e-12);
    /// ```
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Moments<T: Real = f64> {
        /// `M(P, Q) = ∫ (P & x)(Q & x) dA`.
        pub form: Scalar<(Line, Line), T>,
    }

    super::scaffolding!(Line, "lines", "region", "regions");

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
        /// The moments of the region inside a closed curve: the mesh's edges, running
        /// counterclockwise around the region (`gax::mesh`; a polygon's edges, or several
        /// loops, holes running clockwise).
        #[cfg(feature = "std")]
        pub fn of_mesh(mesh: &crate::mesh::Mesh<Point<(), T>, 2>) -> Self {
            let v = mesh.vertices.sites();
            Self::of_edges(
                mesh.topology().simplices(1).map(|e| [v[e[0]], v[e[1]]]),
                Point::xy(T::zero(), T::zero()),
            )
        }

        /// The moments of the region inside a closed polygon, its vertices counterclockwise.
        pub fn of_polygon(vertices: &[Point<(), T>]) -> Self {
            let next = vertices.iter().cycle().skip(1);
            Self::of_edges(
                vertices.iter().zip(next).map(|(&a, &b)| [a, b]),
                Point::xy(T::zero(), T::zero()),
            )
        }

        /// The moments of the region bounded by `edges` (counterclockwise around it) and closed,
        /// if they are not, by a line through `apex`: the edges below a water line, with the
        /// apex on that line, give the submerged region's moments.
        pub fn of_edges(
            edges: impl IntoIterator<Item = [Point<(), T>; 2]>,
            apex: Point<(), T>,
        ) -> Self {
            let mut m = Self::default();
            for e in edges {
                m.add_edge(e, apex);
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
    }
}

#[cfg(feature = "pga2d")]
pub use pga2d_moments::Moments as Moments2;
#[cfg(feature = "pga3d")]
pub use pga3d_moments::Moments as Moments3;
