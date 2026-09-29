//! Random programs over PGA3D values, for testing the tracer against the generic code (the
//! `trace_programs` test and the `trace_programs` fuzz target share this file).
//!
//! A program is a list of byte triples `(op, i, j)` run on two registers each of points,
//! lines and motors. It is generic over the coefficients, so the same bytes run as `Sym` (to
//! be traced) and as `f64` (the reference). Every operation is one gax offers: sandwiches,
//! products, joins and meets, `exp`, `log`, `Motor::between`, normalization, unitizing and a
//! branch-free select.

use gax::pga3d::{Line, Motor, Point};
use gax::{Log, Real};

/// The number of operations; an op byte is taken modulo this.
pub const OPS: u8 = 13;

/// Run `ops` on the arguments; the result is `p₀ + m₀ >> p₁` of the final registers.
pub fn run<T: Real>(
    ops: &[u8],
    m: Motor<(), T>,
    p: Point<(), T>,
    q: Point<(), T>,
    l: Line<(), T>,
) -> Point<(), T> {
    let mut ps = [p, q];
    let mut ls = [l, (p & q)];
    let mut ms = [m, m.reverse()];
    for c in ops.as_chunks::<3>().0 {
        let (i, j) = (usize::from(c[1] % 2), usize::from(c[2] % 2));
        match c[0] % OPS {
            0 => ps[i] = ms[j] >> ps[i],
            1 => ls[i] = ms[j] >> ls[i],
            2 => ms[i] = ms[i] * ms[j],
            3 => ls[i] = ps[i] & ps[1 - i],
            // A plane through a point and a line, met with the other line.
            4 => ps[i] = (ps[i] & ls[j]) ^ ls[1 - j],
            5 => ms[i] = ls[j].gp(T::from_f64(0.25)).exp().into_inner(),
            6 => ms[i] = Motor::between(ps[i], ps[1 - i]).into_inner(),
            7 => ps[i] = ps[i].unitized(),
            8 => ps[i] += ps[j],
            9 => ls[i] = ls[i].gp((ps[i] | ps[1 - i]).s()),
            10 => ps[i] = gax::select_lt(ps[i].e123(), ps[1 - i].e123(), ps[i], ps[1 - i]),
            11 => ls[i] = ls[i].normalized().into_inner(),
            _ => {
                ms[i] = Log::log(ms[i].normalized())
                    .gp(T::from_f64(0.5))
                    .exp()
                    .into_inner();
            }
        }
    }
    ps[0] + (ms[0] >> ps[1])
}

/// The arguments of a program.
pub type Args<T> = (Motor<(), T>, Point<(), T>, Point<(), T>, Line<(), T>);

/// The arguments from 22 numbers.
pub fn args<T: Real>(x: &[T; 22]) -> Args<T> {
    let m = Motor::from_coeffs(core::array::from_fn(|k| x[k]));
    let p = Point::from_coeffs(core::array::from_fn(|k| x[8 + k]));
    let q = Point::from_coeffs(core::array::from_fn(|k| x[12 + k]));
    let l = Line::from_coeffs(core::array::from_fn(|k| x[16 + k]));
    (m, p, q, l)
}

/// Whether the traced result `got` agrees with the reference: within `tol` relative to the
/// result's size, when the reference is well conditioned. Ill-conditioned inputs (a select near
/// a tie, a normalization of almost nothing) are skipped: `run` at a slightly moved input must
/// stay within the same tolerance of the reference, or there is nothing to compare.
pub fn agrees(ops: &[u8], x: &[f64; 22], got: &[f64], tol: f64) -> bool {
    let want = reference(ops, x);
    let size = want.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    if want.iter().any(|v| !v.is_finite()) || size > 1e6 {
        return true;
    }
    let moved: [f64; 22] = core::array::from_fn(|k| x[k] * (1.0 + 1e-9 * (k as f64 + 1.0)));
    let near = reference(ops, &moved);
    let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(u, v)| (u - v).abs() <= tol * size);
    if !close(&near, &want) {
        return true;
    }
    close(got, &want)
}

fn reference(ops: &[u8], x: &[f64; 22]) -> [f64; 4] {
    let (m, p, q, l) = args(x);
    run(ops, m, p, q, l).c
}
