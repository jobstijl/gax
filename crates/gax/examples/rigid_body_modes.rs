//! Vibration modes of a rigid body on springs (PGA2D).
//!
//! A plate made of point masses hangs on three springs. Inertia and stiffness are maps from
//! twists (velocities, the bivectors `Point`) to forques (the vectors `Line`), built as sums:
//! one term per mass and one per spring, with no origin and no parallel-axis shift. The energy
//! forms are these maps paired with an open twist, and the generalized eigenproblem between
//! them returns the modes *as twists*: each one a rotation about a point, printed as that
//! centre.
//!
//! Run with `cargo run --example rigid_body_modes`.

use gax::pga2d::{Line, Point, Scalar};

type Twist = Point<(), f64>;

fn main() {
    // The body: four point masses (x, y, mass).
    let masses = [
        (-1.0, -0.5, 1.0),
        (1.0, -0.5, 1.0),
        (1.0, 0.5, 2.0),
        (-1.0, 0.5, 2.0),
    ];
    // The springs: lines of action through an attachment point, with a direction and a stiffness.
    let springs = [
        ((-1.0, 0.5), (0.0, 1.0), 50.0),
        ((1.0, 0.5), (0.3, 1.0), 40.0),
        ((0.0, -0.5), (1.0, 0.0), 30.0),
    ];

    // Inertia: each mass x contributes m · x ∨ (x × B), a forque, for a twist B.
    let mut inertia: Line<(Point,), f64> = Line::zero();
    for (x, y, m) in masses {
        let p: Twist = Point::xy(x, y);
        inertia += (p & p.commutator(Point::slot())).gp(m);
    }

    // Stiffness: each spring along line l contributes k · l (l ∨ B).
    let mut stiffness: Line<(Point,), f64> = Line::zero();
    for ((x, y), (dx, dy), k) in springs {
        let l = (Point::xy(x, y) & Point::direction(dx, dy))
            .normalized()
            .into_inner();
        stiffness += (l * (l & Point::slot())).gp(k);
    }

    // Energy forms on twists: pair the forque with an open twist.
    let kinetic: Scalar<(Point, Point), f64> = Point::slot() & inertia;
    let potential: Scalar<(Point, Point), f64> = Point::slot() & stiffness;

    // K x = λ M x, solved as forms: the modes come out as twists.
    let (values, modes) = potential.eigh_with(kinetic);
    for (lambda, mode) in values.iter().zip(modes) {
        let frequency = lambda.max(0.0).sqrt() / (2.0 * std::f64::consts::PI);
        if mode.e12().abs() > 1e-9 {
            let [cx, cy] = mode.to_euclidean();
            println!("{frequency:.4} Hz: rotation about ({cx:+.3}, {cy:+.3})");
        } else {
            println!(
                "{frequency:.4} Hz: translation along ({:+.3}, {:+.3})",
                mode.e20(),
                mode.e01()
            );
        }
        // The defining property, K x = λ M x, checked on the linear forms K(x, ·) and M(x, ·).
        let lhs: Scalar<(Point,), f64> = potential.of(mode);
        let rhs: Scalar<(Point,), f64> = kinetic.of(mode).gp(*lambda);
        for (a, b) in lhs.c[0].iter().zip(rhs.c[0].iter()) {
            assert!((a - b).abs() < 1e-9 * (1.0 + a.abs()));
        }
    }
}
