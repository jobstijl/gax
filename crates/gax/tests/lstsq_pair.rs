//! `lstsq_pair`: least squares for an unknown in a map's first two slots, and `of_pair`, the
//! map with a two-slot value bound in. Checked by the normal equations (the residual is
//! orthogonal to the image of every pair of blades), and for least norm where the map has a
//! kernel.

#![cfg(feature = "pga3d")]

use gax::pga3d::{Line, Plane, Point};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// The dyad with a single 1 at `(i, j)`.
fn unit_dyad(i: usize, j: usize) -> Point<(Point,), f64> {
    Point::from_coeffs(core::array::from_fn(|a| {
        core::array::from_fn(|b| if (a, b) == (i, j) { 1.0 } else { 0.0 })
    }))
}

/// An overdetermined map (a Line over a Plane slot: 24 equations for 16 unknowns): the
/// residual is orthogonal to every column, and an exact right-hand side is recovered.
#[test]
fn normal_equations() {
    let mut rng = Rng(0x00a1_2b3c);
    for _ in 0..20 {
        let m = Line::<(Point, Point, Plane), f64>::from_coeffs(core::array::from_fn(|_| {
            core::array::from_fn(|_| core::array::from_fn(|_| core::array::from_fn(|_| rng.next())))
        }));
        let rhs = Line::<(Plane,), f64>::from_coeffs(core::array::from_fn(|_| {
            core::array::from_fn(|_| rng.next())
        }));
        let x = m.lstsq_pair(rhs);
        let r = m.of_pair(x) - rhs;
        for i in 0..4 {
            for j in 0..4 {
                let col = m.of_pair(unit_dyad(i, j));
                let dot: f64 = col
                    .c
                    .iter()
                    .flatten()
                    .zip(r.c.iter().flatten())
                    .map(|(a, b)| a * b)
                    .sum();
                assert!(dot.abs() < 1e-11, "column ({i}, {j}): {dot}");
            }
        }
        // A right-hand side in the image comes back exactly, and with it the dyad.
        let y = Point::<(Point,), f64>::from_coeffs(core::array::from_fn(|_| {
            core::array::from_fn(|_| rng.next())
        }));
        let back = m.lstsq_pair(m.of_pair(y));
        for (a, b) in back.c.iter().flatten().zip(y.c.iter().flatten()) {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
    }
}

/// The join of two points has the symmetric dyads as its kernel (10 of 16 dimensions): the
/// least-norm solution is antisymmetric, and the default cutoff drops the kernel cleanly.
#[test]
fn least_norm_on_a_kernel() {
    let join: Line<(Point, Point), f64> = Point::slot() & Point::slot();
    let mut rng = Rng(0x0bad_cafe);
    for _ in 0..20 {
        let l = Line::<(), f64>::new(
            rng.next(),
            rng.next(),
            rng.next(),
            rng.next(),
            rng.next(),
            rng.next(),
        );
        let x = join.lstsq_pair(l);
        let back = join.of_pair(x);
        for (a, b) in back.c.iter().zip(l.c) {
            assert!((a - b).abs() < 1e-12);
        }
        for i in 0..4 {
            for j in 0..4 {
                assert!((x.c[i][j] + x.c[j][i]).abs() < 1e-12);
            }
        }
        // Least norm: x is orthogonal to the kernel, so adding a symmetric dyad (another
        // solution) only makes it longer.
        let norm = |d: &Point<(Point,), f64>| d.c.iter().flatten().map(|v| v * v).sum::<f64>();
        let mut shifted = x;
        shifted.c[0][1] += 0.1;
        shifted.c[1][0] += 0.1;
        assert!(norm(&shifted) > norm(&x));
    }
}
