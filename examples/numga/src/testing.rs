//! What the examples' tests share: closeness to an absolute tolerance, of numbers and of
//! extensors (gax's `abs_diff_eq` as a function; its `approx_eq` is relative to the
//! coefficients' size).

use gax::ApproxEq;

/// Whether every coefficient of `a` is within `tol` of `b`'s.
pub fn near<X: ApproxEq>(a: X, b: X, tol: f64) -> bool {
    a.abs_diff_eq(&b, tol)
}

/// Whether `a` is within `tol` of `b`.
pub fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}
