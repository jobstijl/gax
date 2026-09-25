//! Conformal and spacetime geometric algebra.
//!
//! CGA3D (null basis `eo`, `ei`): the sphere through four points is their outer product, and
//! inversion in a sphere is a sandwich. STA: a Lorentz boost of an electromagnetic field
//! changes the electric and magnetic parts but not the invariants `E² - B²` and `E · B`.
//!
//! Run with `cargo run --example cga_sta --features cga3d,sta`.

fn cga() {
    use gax::cga3d::{Pseudoscalar, Quadvector, Vector};
    // A Euclidean point x as a conformal point: x + ½|x|² ei + eo.
    let up = |x: f64, y: f64, z: f64| {
        Vector::<(), f64>::new(x, y, z, 1.0, 0.5 * (x * x + y * y + z * z))
    };

    // The sphere through four points (direct representation), and its dual vector.
    let sphere: Quadvector<(), f64> =
        up(3.0, 1.0, 0.0) ^ up(1.0, 1.0, 0.0) ^ up(2.0, 2.0, 0.0) ^ up(2.0, 1.0, 1.0);
    let i: Pseudoscalar<(), f64> = Pseudoscalar::new(1.0);
    let dual: Vector<(), f64> = sphere * i.inverse();
    let s = dual.gp(1.0 / dual.eo()); // scale so that the eo coefficient is 1
    let radius2 = s.dot(s).s(); // for a dual sphere c + ½(c² - r²) ei + eo, s · s = r²
    println!(
        "sphere: centre ({:.3}, {:.3}, {:.3}), radius {:.3}",
        s.e1(),
        s.e2(),
        s.e3(),
        radius2.sqrt()
    );
    assert!((radius2.sqrt() - 1.0).abs() < 1e-12 && (s.e1() - 2.0).abs() < 1e-12);

    // Inversion in the unit sphere around the origin: x -> x / |x|².
    let unit = Vector::<(), f64>::new(0.0, 0.0, 0.0, 1.0, -0.5);
    let p = unit >> up(2.0, 0.0, 0.0);
    let q = p.gp(1.0 / p.eo());
    println!(
        "inversion of (2, 0, 0) in the unit sphere: ({:.3}, {:.3}, {:.3})",
        q.e1(),
        q.e2(),
        q.e3()
    );
    assert!((q.e1() - 0.5).abs() < 1e-12);
}

fn sta() {
    use gax::sta::{Bivector, Even};
    // F = E + I B: e10, e20, e30 hold the electric field, e23, e31, e12 the magnetic field.
    let f = Bivector::<(), f64>::new(1.0, 0.5, 0.0, 0.2, 0.0, 0.8);
    let invariants = |f: Bivector<(), f64>| {
        let sq: Even<(), f64> = f * f;
        (sq.s(), sq.e0123())
    };
    // A boost along x with rapidity 0.8: exp(½ φ e10).
    let boost = Bivector::<(), f64>::new(0.4, 0.0, 0.0, 0.0, 0.0, 0.0).exp();
    let g: Bivector<(), f64> = boost >> f;
    let (a, b) = (invariants(f), invariants(g));
    println!(
        "E before: ({:.3}, {:.3}, {:.3})  after: ({:.3}, {:.3}, {:.3})",
        f.e10(),
        f.e20(),
        f.e30(),
        g.e10(),
        g.e20(),
        g.e30()
    );
    println!(
        "invariants before: ({:.6}, {:.6})  after: ({:.6}, {:.6})",
        a.0, a.1, b.0, b.1
    );
    assert!((a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-12);
    // And the boost itself is recovered by the logarithm.
    let log: Bivector<(), f64> = boost.log();
    assert!((log.e10() - 0.4).abs() < 1e-12);
}

fn main() {
    cga();
    sta();
}
