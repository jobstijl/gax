//! Eigenvalues and eigenvectors of general (non-symmetric) maps, complex in general: rotations
//! (unit complex pairs), boosts (real pairs `e^{±φ}`), and random maps, where every pair is
//! checked as `A v = λ v` with complex coefficients.

#![cfg(all(feature = "vga3d", feature = "sta", feature = "pga3d"))]

use gax::Complex;

fn close(a: Complex<f64>, b: Complex<f64>, tol: f64) -> bool {
    (a - b).abs() < tol
}

#[test]
fn a_rotation_has_unit_complex_eigenvalues() {
    use gax::vga3d::{Bivector, Vector};
    let theta = 0.7;
    let r = Bivector::<(), f64>::new(0.0, 0.0, -theta / 2.0).exp();
    let m = r >> Vector::<(), f64>::slot();
    let (values, vectors) = m.eig();
    let want = [
        Complex::from_polar(1.0, -theta),
        Complex::real(1.0),
        Complex::from_polar(1.0, theta),
    ];
    let mut got = values;
    got.sort_by(|a, b| a.im.total_cmp(&b.im));
    for (g, w) in got.iter().zip(want) {
        assert!(close(*g, w, 1e-12), "{got:?}");
    }
    // A v = λ v, with the map's coefficients made complex.
    let mc = m.map_coefs(Complex::real);
    for (l, v) in values.iter().zip(vectors) {
        let lhs = mc.of(v);
        for (a, b) in lhs.c.iter().zip(v.c) {
            assert!(close(*a, *l * b, 1e-12));
        }
    }
}

#[test]
fn a_boost_has_real_eigenvalues() {
    use gax::sta::{Bivector, Vector};
    let phi = 0.9;
    // A boost along e1 (e10 squares to +1).
    let mut b = Bivector::<(), f64>::zero();
    b.c[0] = phi / 2.0;
    let m = b.exp() >> Vector::<(), f64>::slot();
    let values = m.eigvals();
    let mut re: Vec<f64> = values.iter().map(|z| z.re).collect();
    re.sort_by(f64::total_cmp);
    assert!(values.iter().all(|z| z.im.abs() < 1e-12), "{values:?}");
    for (g, w) in re.iter().zip([(-phi).exp(), 1.0, 1.0, phi.exp()]) {
        assert!((g - w).abs() < 1e-12, "{re:?}");
    }
}

#[test]
fn random_maps_satisfy_their_eigenpairs() {
    use gax::pga3d::Line;
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    for _ in 0..200 {
        let m = Line::<(Line,), f64>::from_coeffs(core::array::from_fn(|_| {
            core::array::from_fn(|_| next())
        }));
        let (values, vectors) = m.eig();
        // The eigenvalues multiply to the determinant and add up to the trace.
        let prod = values.iter().fold(Complex::real(1.0), |a, b| a * *b);
        let sum = values.iter().fold(Complex::real(0.0), |a, b| a + *b);
        assert!(
            close(prod, Complex::real(m.det()), 1e-9),
            "{prod:?} vs {}",
            m.det()
        );
        let trace: f64 = (0..6).map(|i| m.c[i][i]).sum();
        assert!(close(sum, Complex::real(trace), 1e-9));
        let mc = m.map_coefs(Complex::real);
        for (l, v) in values.iter().zip(vectors) {
            let lhs = mc.of(v);
            let scale = 1.0 + l.abs();
            for (a, b) in lhs.c.iter().zip(v.c) {
                assert!(close(*a, *l * b, 1e-8 * scale), "{l:?}");
            }
        }
    }
}

/// `eigh` on a map takes its coefficient matrix as symmetric (numga's `eigh` on a map): the
/// stretch along an axis, turned by a rotor, has its eigenvectors along the turned axes.
#[test]
fn a_symmetric_map_s_eigenpairs() {
    use gax::vga3d::{Bivector, Vector};
    let r = Bivector::<(), f64>::new(0.3, -0.2, 0.5).exp();
    let axes = [
        Vector::new(1.0, 0.0, 0.0),
        Vector::new(0.0, 1.0, 0.0),
        Vector::new(0.0, 0.0, 1.0),
    ];
    let stretch = [1.0, 2.0, 5.0];
    // Σ λ (r a)(r a | ·): a sum of maps (`Sum`).
    let m: Vector<(Vector,), f64> = axes
        .iter()
        .zip(stretch)
        .map(|(a, l)| {
            let a = r >> *a;
            a * (a | Vector::slot()).gp(l)
        })
        .sum();
    let (values, vectors) = m.eigh();
    for (g, w) in values.iter().zip(stretch) {
        assert!((g - w).abs() < 1e-12, "{values:?}");
    }
    for ((v, a), l) in vectors.iter().zip(&axes).zip(stretch) {
        let want = r >> *a;
        assert!((*v | want).s().abs() > 1.0 - 1e-12);
        let image = m.of(*v);
        for (x, y) in image.c.iter().zip(v.c) {
            assert!((x - l * y).abs() < 1e-12);
        }
    }
}
