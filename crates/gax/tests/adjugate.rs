//! The transposes of maps through a pairing (ADR-053), beyond the symbolic laws of
//! `law_suite` (`adjugates`, `adjoints`): the classical adjugate of an outermorphism, the
//! statements of numga's docstrings, the parts' maps, and field maps.

#![cfg(all(feature = "pga3d", feature = "vga3d"))]

#[path = "support/rng.rs"]
mod rng;
use rng::{Draw, rng};

use gax::field::{Field, SparseMap};
use gax::{ApproxEq, Extensor, Kind, Slots};

/// A map's coefficients as rows, output first, whatever its orientations.
fn rows<M: Extensor<Coef = f64, Slots = (K,)>, K: Kind>(m: &M) -> Vec<Vec<f64>> {
    m.coeffs()
        .as_ref()
        .iter()
        .map(|r| (0..K::N).map(|i| <(K,) as Slots>::get_flat(r, i)).collect())
        .collect()
}

fn product(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .map(|r| {
            (0..b[0].len())
                .map(|j| r.iter().zip(b).map(|(x, row)| x * row[j]).sum())
                .collect()
        })
        .collect()
}

fn is_scaled_identity(m: &[Vec<f64>], s: f64) {
    for (i, r) in m.iter().enumerate() {
        for (j, x) in r.iter().enumerate() {
            let want = if i == j { s } else { 0.0 };
            assert!(
                (x - want).abs() < 1e-10 * (1.0 + s.abs()),
                "{m:?} is not {s} · id"
            );
        }
    }
}

fn random_map<M: Extensor<Coef = f64>>(rng: &mut impl Draw) -> M {
    M::from_coeffs(<M::Kind as Kind>::arr_from_fn(|_| {
        <M::Slots as Slots>::from_flat(&mut |_| rng.next_f64(), 0)
    }))
}

/// The adjugate of an outermorphism is the classical one: `adj(t_k) ∘ t_(n−k) = det(t) · id`,
/// the outermorphism on the complementary grade times its adjugate is the determinant.
#[test]
fn the_adjugate_of_an_outermorphism_is_the_classical_adjugate() {
    use gax::{pga3d, vga3d};
    let mut rng = rng(0x0ad1_u64);
    for _ in 0..20 {
        let t: pga3d::Plane<(pga3d::Plane,), f64> = random_map(&mut rng);
        let det = t.outermorphism::<pga3d::Pseudoscalar>().c[0][0];
        let (t2, t3) = (
            t.outermorphism::<pga3d::Line>(),
            t.outermorphism::<pga3d::Point>(),
        );
        is_scaled_identity(&product(&rows(&t3.adjugate()), &rows(&t)), det);
        is_scaled_identity(&product(&rows(&t2.adjugate()), &rows(&t2)), det);
        is_scaled_identity(&product(&rows(&t.adjugate()), &rows(&t3)), det);

        let v: vga3d::Vector<(vga3d::Vector,), f64> = random_map(&mut rng);
        let det = v.outermorphism::<vga3d::Pseudoscalar>().c[0][0];
        let v2 = v.outermorphism::<vga3d::Bivector>();
        is_scaled_identity(&product(&rows(&v2.adjugate()), &rows(&v)), det);
        is_scaled_identity(&product(&rows(&v.adjugate()), &rows(&v2)), det);
    }
}

/// numga's docstring: in PGA3D the adjugate of the adjugate of a `Point <- Twist` is its
/// negative (the pairing of a point with a plane changes sign when swapped, a line's does
/// not); in VGA3D, an odd dimension, it is the map itself.
#[test]
fn twice_is_the_map_up_to_the_swap_sign() {
    use gax::{pga3d, vga3d};
    let mut rng = rng(0x0ad2_u64);
    let t: pga3d::Point<(pga3d::Line,), f64> = random_map(&mut rng);
    assert_eq!(t.adjugate().adjugate(), -t);
    let u: pga3d::Line<(pga3d::Line,), f64> = random_map(&mut rng);
    assert_eq!(u.adjugate().adjugate(), u);
    let v: vga3d::Bivector<(vga3d::Vector,), f64> = random_map(&mut rng);
    assert_eq!(v.adjugate().adjugate(), v);
}

/// A correlation, a map into the complement of its input, has an adjugate of its own type: the
/// polarity of a sphere is symmetric, its adjugate itself up to the swap sign of a plane and a
/// point, `−1` in 4D.
#[test]
fn a_correlation_has_an_adjugate_of_its_own_type() {
    use gax::pga3d::{Plane, Point};
    // The polarity of the unit sphere: a point (x, y, z, w) to its polar plane x X + y Y + z Z − w = 0.
    let polarity = Plane::<(Point,), f64>::from_images([
        Plane::new(1.0, 0.0, 0.0, 0.0),
        Plane::new(0.0, 1.0, 0.0, 0.0),
        Plane::new(0.0, 0.0, 1.0, 0.0),
        Plane::new(0.0, 0.0, 0.0, -1.0),
    ]);
    let back: Plane<(Point,), f64> = polarity.adjugate();
    assert_eq!(back, -polarity);
}

/// The parts' maps: a stretch of directions and its adjugate on origin planes, which `solve`
/// and `det` take, and the outermorphisms onto moments and spins.
#[test]
fn the_parts_maps_transpose_and_extend() {
    use gax::pga3d::{Direction, Moment, OriginPlane, Spin};
    let mut rng = rng(0x0ad3_u64);
    let stretch: Direction<(Direction,), f64> = random_map(&mut rng);
    let across: OriginPlane<(OriginPlane,), f64> = stretch.adjugate();
    assert!((across.det() - stretch.det()).abs() < 1e-12);
    let mirror = OriginPlane::<(), f64>::new(0.3, -0.4, 1.2);
    let d = Direction::new(1.0, 2.0, -0.5);
    // Incidence carries over: pairing the adjugate's image with `d` pairs the plane with the
    // stretched direction.
    assert!((across.of(mirror) & d).approx_eq(&(mirror & stretch.of(d)), 1e-12));
    let solved = across.solve(across.of(mirror));
    assert!(solved.approx_eq(&mirror, 1e-12));
    // The extension to moments by `∨`, and its adjugate on spins.
    let couples: Moment<(Moment,), f64> = stretch.outermorphism::<Moment>();
    let (a, b) = (Direction::new(1.0, 0.0, 0.0), Direction::new(0.0, 1.0, 1.0));
    assert!(
        couples
            .of((a & b).cast::<Moment>())
            .approx_eq(&(stretch.of(a) & stretch.of(b)).cast::<Moment>(), 1e-12)
    );
    let spins: Spin<(Spin,), f64> = couples.adjugate();
    let s = Spin::new(0.2, 0.7, -1.0);
    let m = Moment::new(-0.3, 0.4, 0.9);
    assert!((spins.of(s) & m).approx_eq(&(s & couples.of(m)), 1e-12));
}

/// numga's docstring: a versor's sandwich has its inverse sandwich as its adjoint. On PGA3D's
/// spins, where the scalar product is non-degenerate, a rotation's.
#[test]
fn a_rotation_has_its_inverse_as_its_adjoint() {
    use gax::pga3d::{Line, Motor, Point, Spin};
    let r = Motor::<(), f64>::rotation_about(Point::direction(0.3, -1.0, 0.5), 0.9).into_inner();
    let turn: Spin<(Spin,), f64> = (r >> Line::slot()).cast::<Spin>().of(Spin::slot());
    let back: Spin<(Spin,), f64> = (r.reverse() >> Line::slot())
        .cast::<Spin>()
        .of(Spin::slot());
    assert!(turn.adjoint().approx_eq(&back, 1e-12));
}

/// Field maps: the couplings run back with each cell transposed, so the summed pairings agree.
#[test]
fn field_maps_transpose_cell_by_cell() {
    use gax::vga3d::{Rotor, Vector, intrinsic};
    let mut rng = rng(0x0ad4_u64);
    // A connection on a ring of four sites: each edge's rotor applied to its tail.
    let cells: Vec<(usize, usize, Rotor<(Rotor,), f64>)> = (0..4)
        .flat_map(|e| {
            let r =
                Rotor::<(), f64>::rotation(Vector::new(0.0, 0.0, 1.0), 0.3 * e as f64).into_inner();
            [
                (e, e, -Rotor::<(), f64>::slot()),
                (e, (e + 1) % 4, (r * Rotor::slot()).cast::<Rotor>()),
            ]
        })
        .collect();
    let covariant = SparseMap::from_triples(cells, (4, 4));
    let x = Field::from_fn(4, |_| random_map::<Rotor<(), f64>>(&mut rng));
    let y = Field::from_fn(4, |_| random_map::<Rotor<(), f64>>(&mut rng));
    let pair = |a: &Field<Rotor<(), f64>>, b: &Field<Rotor<(), f64>>| -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(p, q)| p.scalar_product(q).s())
            .sum()
    };
    let adjoint = covariant.adjoint();
    assert!((pair(&adjoint.of(&y), &x) - pair(&y, &covariant.of(&x))).abs() < 1e-12);
    // The connection Laplacian is symmetric under the same pairing.
    let laplacian = adjoint.compose(&covariant);
    assert!((pair(&laplacian.of(&y), &x) - pair(&y, &laplacian.of(&x))).abs() < 1e-12);
    // And the adjugate, through the regressive product, pairs rotors with the odd kind (their
    // complement, intrinsic).
    let adjugate = covariant.adjugate();
    let c = Field::from_fn(4, |_| random_map::<intrinsic::Odd<(), f64>>(&mut rng));
    let meet = |a: &Field<intrinsic::Odd<(), f64>>, b: &Field<Rotor<(), f64>>| -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(p, q)| (p & q).grade::<0>().s())
            .sum()
    };
    assert!((meet(&adjugate.of(&c), &x) - meet(&c, &covariant.of(&x))).abs() < 1e-12);
}
