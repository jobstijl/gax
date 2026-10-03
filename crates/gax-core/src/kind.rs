//! Kinds (typed subspaces) and the [`Extensor`] trait shared by all generated types.

use crate::coef::{Coef, Elem};
use crate::slots::Slots;
use core::fmt::Debug;

/// A typed subspace of an algebra, such as `Point` or `Motor`: an ordered list of oriented
/// basis blades.
///
/// A kind is written as its multivector type with default parameters, so `Point` is both
/// the type of a point value and the marker for an open point slot in `Line<(Point,)>`.
///
/// ```
/// use gax::pga3d::Point;
/// use gax::Kind;
/// assert_eq!(<Point as Kind>::N, 4);
/// assert_eq!(<Point as Kind>::BLADES, &["e032", "e013", "e021", "e123"]);
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a kind",
    label = "not a kind",
    note = "slot lists name kinds by their bare type name: write `Point<(Point,), f64>`, not `Point<(Point<(), f64>,), f64>`"
)]
pub trait Kind: Copy + Debug + PartialEq + 'static {
    /// Number of coefficients.
    const N: usize;
    /// Name of the kind.
    const NAME: &'static str;
    /// `module_path!()` of the module that defines the kind.
    const MODULE: &'static str;
    /// Blade names in layout order, e.g. `["e032", "e013", "e021", "e123"]`.
    const BLADES: &'static [&'static str];
    /// Coefficient array of this kind: `[X; N]`.
    type Arr<X: Elem>: Elem + AsRef<[X]> + AsMut<[X]>;
    /// The multivector type of this kind with slots `S` and coefficients `T`.
    type Mv<S: Slots, T: Coef>: Extensor<Kind = Self, Slots = S, Coef = T>;
    /// The scalar kind of the same algebra.
    type Scalar: Kind;

    /// Build an array from its index.
    fn arr_from_fn<X: Elem>(f: impl FnMut(usize) -> X) -> Self::Arr<X>;
    /// Elementwise map.
    fn arr_map<X: Elem, Y: Elem>(a: &Self::Arr<X>, f: impl FnMut(&X) -> Y) -> Self::Arr<Y>;
    /// Elementwise combination.
    fn arr_zip<X: Elem, Y: Elem, Z: Elem>(
        a: &Self::Arr<X>,
        b: &Self::Arr<Y>,
        f: impl FnMut(&X, &Y) -> Z,
    ) -> Self::Arr<Z>;
}

/// The coefficient storage of a multivector type: one slot array per output coefficient.
pub type Coeffs<M> = <<M as Extensor>::Kind as Kind>::Arr<
    <<M as Extensor>::Slots as Slots>::Arr<<M as Extensor>::Coef>,
>;

/// A multivector, map or form of some kind: the common interface of all generated types.
///
/// ```
/// use gax::pga3d::{Line, Point};
/// use gax::Extensor;
/// let m: Line<(Point,), f64> = Point::xyz(0.0, 0.0, 0.0) & Point::slot();
/// // Output first: one array over the slot per output coefficient.
/// let c: &[[f64; 4]; 6] = m.coeffs();
/// assert_eq!(Line::<(Point,), f64>::from_coeffs(*c), m);
/// ```
pub trait Extensor: Copy + Debug + PartialEq + 'static {
    /// The output kind.
    type Kind: Kind;
    /// The open slots.
    type Slots: Slots;
    /// The coefficient type.
    type Coef: Coef;

    /// Construct from output-first coefficients.
    fn from_coeffs(c: Coeffs<Self>) -> Self;
    /// The output-first coefficients.
    fn coeffs(&self) -> &Coeffs<Self>;

    /// Every coefficient mapped by `f`, kind and slots kept: a value or map of `f64` as
    /// `Complex<f64>`, as `Dual<f64, N>` (to differentiate through it), or as `f32`.
    ///
    /// ```
    /// use gax::{Complex, Extensor};
    /// use gax::pga3d::Point;
    /// let p = Point::<(), f64>::xyz(1.0, 2.0, 3.0);
    /// let z: Point<(), Complex<f64>> = p.map_coefs(Complex::real);
    /// assert_eq!(z.c[0], Complex::real(1.0));
    /// ```
    #[inline]
    fn map_coefs<U: Coef>(&self, mut f: impl FnMut(Self::Coef) -> U) -> Retype<Self, Self::Slots, U>
    where
        Self: Sized,
    {
        let c = <Self::Kind as Kind>::arr_map(self.coeffs(), |col| {
            <Self::Slots as Slots>::map(col, &mut |x| f(*x))
        });
        <Retype<Self, Self::Slots, U> as Extensor>::from_coeffs(c)
    }
}

/// Same kind, other slots and coefficients.
pub type Retype<M, S, T> = <<M as Extensor>::Kind as Kind>::Mv<S, T>;

/// `if a < b { x } else { y }`, coefficient by coefficient, for values and maps alike. It is
/// data flow, not a branch (`Real::select_lt` on every coefficient), so it vectorizes on SIMD
/// lanes and traces.
///
/// ```
/// use gax::pga3d::Point;
/// let (near, far) = (Point::<(), f64>::xyz(1.0, 0.0, 0.0), Point::xyz(9.0, 0.0, 0.0));
/// assert_eq!(gax::select_lt(0.2, 0.5, near, far), near);
/// assert_eq!(gax::select_lt(0.7, 0.5, near, far), far);
/// ```
#[inline]
pub fn select_lt<M: Extensor>(a: M::Coef, b: M::Coef, x: M, y: M) -> M
where
    M::Coef: crate::coef::Real,
{
    M::from_coeffs(<M::Kind as Kind>::arr_zip(
        x.coeffs(),
        y.coeffs(),
        |cx, cy| {
            <M::Slots as Slots>::zip(cx, cy, &mut |p, q| {
                <M::Coef as crate::coef::Real>::select_lt(a, b, *p, *q)
            })
        },
    ))
}

/// Approximate equality of values, maps and forms: every coefficient within `tol` of the
/// other's, relative to the larger of 1 and the largest coefficient of the two (so it is an
/// absolute tolerance near zero and a relative one for large values).
///
/// ```
/// use gax::ApproxEq;
/// use gax::pga3d::{Motor, Point};
/// let m = Motor::<(), f64>::rotation_about(0.0, 0.0, 1.0, std::f64::consts::FRAC_PI_2);
/// assert!((m >> Point::xyz(1.0, 0.0, 0.0)).approx_eq(&Point::xyz(0.0, 1.0, 0.0), 1e-12));
/// // A NaN coefficient is never close.
/// assert!(!Point::<(), f64>::xyz(f64::NAN, 0.0, 0.0).approx_eq(&Point::xyz(0.0, 0.0, 0.0), 1.0));
/// ```
pub trait ApproxEq {
    /// Whether every coefficient is within `tol` (relative to the magnitude, see above).
    fn approx_eq(&self, other: &Self, tol: f64) -> bool;
    /// The largest difference of two coefficients (absolute); NaN if any difference is NaN.
    fn max_abs_diff(&self, other: &Self) -> f64;
}

impl<M: Extensor> ApproxEq for M
where
    M::Coef: Into<f64>,
{
    fn approx_eq(&self, other: &Self, tol: f64) -> bool {
        let scale = flat(self)
            .chain(flat(other))
            .fold(1.0f64, |m, x| m.max(x.abs()));
        self.max_abs_diff(other) <= tol * scale
    }

    fn max_abs_diff(&self, other: &Self) -> f64 {
        // NaN propagates (`f64::max` would drop it), so a NaN coefficient is never close.
        flat(self).zip(flat(other)).fold(0.0f64, |m, (x, y)| {
            let d = (x - y).abs();
            if d.is_nan() || d > m { d } else { m }
        })
    }
}

/// Every coefficient of `m`, output first, as `f64`.
fn flat<M: Extensor>(m: &M) -> impl Iterator<Item = f64> + '_
where
    M::Coef: Into<f64>,
{
    m.coeffs().as_ref().iter().flat_map(|col| {
        (0..<M::Slots as Slots>::SIZE).map(move |k| <M::Slots as Slots>::get_flat(col, k).into())
    })
}
