//! Reflections and orientation (ADR-051). `>>` takes its sign from the passenger's orientation,
//! so a reflection commutes with every product, the join included: the cases of ganja.js issue
//! 180 and kingdon issue 154 come out right with no sign written by hand. A momentum reflected
//! keeps its kinetic energy, and a point stays on its side of a mirrored plane. (Before
//! orientation types, `>>` was the plain sandwich and both took a hand-written minus.)

/// A unit mass at the origin translating along `x`, reflected in the wall `x = 0`: the momentum
/// built from the reflected point and rate is the reflected momentum, and the kinetic energy is
/// kept.
#[cfg(feature = "pga3d")]
#[test]
fn pga3d_momentum_keeps_its_energy() {
    use gax::pga3d::{Forque, Line, Plane, Point};
    let wall = Plane::<(), f64>::new(1.0, 0.0, 0.0, 0.0);
    let p = Point::<(), f64>::xyz(0.0, 0.0, 0.0);
    let rate = Line::<(), f64>::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    // A twist is extrinsic (a turn), a momentum intrinsic (a join of a point with a velocity).
    let momentum =
        |p: Point<(), f64>, rate: Line<(), f64>| -> Forque<(), f64> { p & p.commutator(rate) };
    let energy = |rate: Line<(), f64>, m: Forque<(), f64>| 0.5 * (rate & m).s();
    let m = momentum(p, rate);
    assert_eq!(energy(rate, m), 0.5);

    let (p2, rate2) = (wall >> p, wall >> rate);
    let rebuilt = momentum(p2, rate2);
    assert_eq!(rebuilt, wall >> m);
    assert_eq!(energy(rate2, wall >> m), 0.5);
}

/// A duck at `(2, 1, 1)` above the water `z = 0`, the scene mirrored in the wall `x = 1/2`: the
/// duck stays above the water. A plane's side is extrinsic and survives the mirror; a triangle's
/// winding, the join of its corners in order, is intrinsic and reverses.
#[cfg(feature = "pga3d")]
#[test]
fn pga3d_sides_survive_and_windings_reverse() {
    use gax::pga3d::{Plane, Point};
    let wall = Plane::<(), f64>::new(1.0, 0.0, 0.0, -0.5);
    let water = Plane::<(), f64>::new(0.0, 0.0, 1.0, 0.0);
    let duck = Point::<(), f64>::xyz(2.0, 1.0, 1.0);
    let side = (water & duck).s();
    assert_eq!(side, 1.0);
    assert_eq!(wall >> duck, Point::xyz(-1.0, 1.0, 1.0));
    assert_eq!(((wall >> water) & (wall >> duck)).s(), side);

    // numga's surface normals: the mirror x = 0 keeps the plane z = 0's positive side up, and
    // turns the corners of a triangle in it the other way round.
    let mirror = Plane::<(), f64>::new(1.0, 0.0, 0.0, 0.0);
    let corners = [
        Point::<(), f64>::xyz(0.0, 0.0, 0.0),
        Point::xyz(1.0, 0.0, 0.0),
        Point::xyz(0.0, 1.0, 0.0),
    ];
    let winding = corners[0] & corners[1] & corners[2];
    assert_eq!(mirror >> water, water);
    assert_eq!(mirror >> winding, -winding);
    assert_eq!(
        (mirror >> corners[0]) & (mirror >> corners[1]) & (mirror >> corners[2]),
        mirror >> winding
    );
}

/// In PGA2D a line is a mirror and a point a position: `>>` reflects a point keeping its
/// weight and mirrors a direction, and the momentum of a reflected point is the reflected
/// momentum.
#[cfg(feature = "pga2d")]
#[test]
fn pga2d_reflections_commute_with_the_join() {
    use gax::pga2d::{Line, Point};
    let wall = Line::<(), f64>::new(1.0, 0.0, 0.0);
    let p = Point::<(), f64>::xy(1.0, 2.0);
    let v = Point::<(), f64>::direction(1.0, 0.5);
    assert_eq!(wall >> p, Point::xy(-1.0, 2.0));
    assert_eq!(wall >> v, Point::direction(-1.0, 0.5));
    assert_eq!((wall >> p) & (wall >> v), wall >> (p & v));
}

/// `m >> x.hodge() == (m >> x).hodge()` for reflections and rotations in the non-degenerate
/// algebras: the Hodge dual flips the orientation, and the reflection's sign flips with it. In
/// an orthonormal Euclidean basis the metric-free `dual()` is the Hodge dual up to sign, and
/// commutes too; in CGA's null basis it does not (it is not a metric operation).
macro_rules! hodge_commutes {
    ($name:ident, $feature:literal, $alg:ident, euclidean: $euclidean:literal, [$($x:ident),*]) => {
        #[cfg(feature = $feature)]
        #[test]
        fn $name() {
            use gax::ApproxEq;
            use gax::$alg::*;
            let mut seed = 0x5eed_u64;
            let mut next = move || {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            };
            // A unit vector with a positive square.
            let unit = |next: &mut dyn FnMut() -> f64| loop {
                let v = Vector::<(), f64>::from_coeffs(core::array::from_fn(|_| next()));
                let n = (v * v).cast::<Scalar>().s();
                if n > 0.05 {
                    break v * (1.0 / n.sqrt());
                }
            };
            for _ in 0..4 {
                let (a, b) = (unit(&mut next), unit(&mut next));
                let (reflection, rotation) = (gax::Unit::new_unchecked(a), gax::Unit::new_unchecked(a * b));
                $(
                    let x = $x::<(), f64>::from_coeffs(core::array::from_fn(|_| next()));
                    let what = stringify!($x);
                    assert!((reflection >> x.hodge()).approx_eq(&(reflection >> x).hodge(), 1e-12), "reflection, hodge: {what}");
                    assert!((rotation >> x.hodge()).approx_eq(&(rotation >> x).hodge(), 1e-12), "rotation, hodge: {what}");
                    if $euclidean {
                        assert!((reflection >> x.dual()).approx_eq(&(reflection >> x).dual(), 1e-12), "reflection, dual: {what}");
                        assert!((rotation >> x.dual()).approx_eq(&(rotation >> x).dual(), 1e-12), "rotation, dual: {what}");
                    }
                )*
            }
        }
    };
}

hodge_commutes!(vga3d_hodge_commutes, "vga3d", vga3d, euclidean: true, [Vector, Bivector]);
hodge_commutes!(cga3d_hodge_commutes, "cga3d", cga3d, euclidean: false, [Vector, Bivector, Trivector]);
hodge_commutes!(sta_hodge_commutes, "sta", sta, euclidean: false, [Vector, Bivector]);
