//! Mass properties from boundaries (`gax::pga3d::Moments`, `gax::pga2d::Moments`): classical
//! results for cubes, tetrahedra, rectangles and polygons; independence from the apex;
//! additivity; equivariance under motions; a mesh cut by a plane closed by the plane alone; and
//! a sphere's mesh converging.

#![cfg(all(feature = "pga2d", feature = "pga3d"))]

use gax::{pga2d, pga3d};

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

/// The box `[0, a] × [0, b] × [0, c]`, as outward triangles.
fn cuboid(a: f64, b: f64, c: f64) -> (Vec<pga3d::Point<(), f64>>, Vec<[usize; 3]>) {
    let v = (0..8)
        .map(|i| {
            pga3d::Point::xyz(
                a * f64::from(i & 1),
                b * f64::from(i >> 1 & 1),
                c * f64::from(i >> 2 & 1),
            )
        })
        .collect();
    let f = vec![
        [0, 2, 1],
        [1, 2, 3],
        [4, 5, 6],
        [5, 7, 6],
        [0, 1, 4],
        [1, 5, 4],
        [2, 6, 3],
        [3, 6, 7],
        [0, 4, 2],
        [2, 4, 6],
        [1, 3, 5],
        [3, 7, 5],
    ];
    (v, f)
}

#[test]
fn a_box() {
    let (a, b, c) = (2.0, 3.0, 5.0);
    let (v, f) = cuboid(a, b, c);
    let m = pga3d::Moments::of_mesh(&v, &f);
    assert!(close(m.volume(), a * b * c, 1e-12));
    let [x, y, z] = m.centroid().to_euclidean();
    assert!(close(x, a / 2.0, 1e-12) && close(y, b / 2.0, 1e-12) && close(z, c / 2.0, 1e-12));
    // ∫ (x − c)² dV = V a² / 12, no products of inertia.
    let s = m.central_second_moments();
    let v0 = a * b * c;
    assert!(close(s[0][0], v0 * a * a / 12.0, 1e-12));
    assert!(close(s[1][1], v0 * b * b / 12.0, 1e-12));
    assert!(close(s[2][2], v0 * c * c / 12.0, 1e-12));
    assert!(s[0][1].abs() < 1e-12 && s[0][2].abs() < 1e-12 && s[1][2].abs() < 1e-12);
    // Inertia at density 2: mass 60, moments m (b² + c²) / 12, ... ascending.
    let (inertia, frame) = m.inertia(2.0);
    let mass = 2.0 * v0;
    assert!(close(inertia.mass, mass, 1e-12));
    let mut want = [
        mass * (b * b + c * c) / 12.0,
        mass * (a * a + c * c) / 12.0,
        mass * (a * a + b * b) / 12.0,
    ];
    want.sort_by(f64::total_cmp);
    for k in 0..3 {
        assert!(
            close(inertia.moments[k], want[k], 1e-10),
            "{:?} vs {want:?}",
            inertia.moments
        );
    }
    // The frame takes the body origin to the centre of mass.
    let o = (frame >> pga3d::Point::xyz(0.0, 0.0, 0.0)).to_euclidean();
    assert!(close(o[0], a / 2.0, 1e-12) && close(o[2], c / 2.0, 1e-12));
}

/// The tetrahedron (0, e1, e2, e3): volume 1/6, centroid 1/4, ∫ x² = 1/60, ∫ x y = 1/120.
#[test]
fn a_tetrahedron() {
    let p = |x, y, z| pga3d::Point::xyz(x, y, z);
    let v = [
        p(0.0, 0.0, 0.0),
        p(1.0, 0.0, 0.0),
        p(0.0, 1.0, 0.0),
        p(0.0, 0.0, 1.0),
    ];
    let f = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
    let m = pga3d::Moments::of_mesh(&v, &f);
    assert!(close(m.volume(), 1.0 / 6.0, 1e-12));
    for c in m.centroid().to_euclidean() {
        assert!(close(c, 0.25, 1e-12));
    }
    let s = m.central_second_moments();
    let vol = 1.0 / 6.0;
    // About the centroid: ∫ x² − V c² and ∫ x y − V c².
    assert!(close(s[0][0], 1.0 / 60.0 - vol / 16.0, 1e-12));
    assert!(close(s[0][1], 1.0 / 120.0 - vol / 16.0, 1e-12));
}

/// The apex is arbitrary for a closed mesh; the moments of parts add.
#[test]
fn apex_and_additivity() {
    let (v, f) = cuboid(1.0, 2.0, 3.0);
    let tris = || f.iter().map(|t| [v[t[0]], v[t[1]], v[t[2]]]);
    let a = pga3d::Moments::of_triangles(tris(), pga3d::Point::xyz(0.0, 0.0, 0.0));
    let b = pga3d::Moments::of_triangles(tris(), pga3d::Point::xyz(7.0, -3.0, 2.0));
    for (x, y) in a.form.c[0]
        .iter()
        .flatten()
        .zip(b.form.c[0].iter().flatten())
    {
        assert!(close(*x, *y, 1e-11));
    }
    // Two boxes side by side are one box of twice the length.
    let (w, g) = cuboid(1.0, 2.0, 3.0);
    let shifted: Vec<_> = w
        .iter()
        .map(|p| pga3d::Motor::translation(1.0, 0.0, 0.0) >> *p)
        .collect();
    let two = pga3d::Moments::of_mesh(&w, &g) + pga3d::Moments::of_mesh(&shifted, &g);
    let (u, h) = cuboid(2.0, 2.0, 3.0);
    let one = pga3d::Moments::of_mesh(&u, &h);
    for (x, y) in two.form.c[0]
        .iter()
        .flatten()
        .zip(one.form.c[0].iter().flatten())
    {
        assert!(close(*x, *y, 1e-11));
    }
}

/// Moving the body moves its moments: `moved` agrees with the moved mesh.
#[test]
fn equivariance() {
    let (v, f) = cuboid(1.0, 2.0, 3.0);
    let m = pga3d::Motor::translation(0.3, -1.0, 2.0)
        * pga3d::Motor::rotation_about(1.0, 2.0, 0.5, 0.7);
    let moved_mesh: Vec<_> = v.iter().map(|p| m >> *p).collect();
    let direct = pga3d::Moments::of_mesh(&moved_mesh, &f);
    let moved = pga3d::Moments::of_mesh(&v, &f).moved(m);
    for (x, y) in direct.form.c[0]
        .iter()
        .flatten()
        .zip(moved.form.c[0].iter().flatten())
    {
        assert!(close(*x, *y, 1e-10), "{x} vs {y}");
    }
    // The principal moments do not depend on the pose.
    let (a, _) = pga3d::Moments::of_mesh(&v, &f).inertia(1.0);
    let (b, frame) = direct.inertia(1.0);
    for k in 0..3 {
        assert!(close(a.moments[k], b.moments[k], 1e-10));
    }
    let c = (frame >> pga3d::Point::xyz(0.0, 0.0, 0.0)).to_euclidean();
    let want = (m >> pga3d::Point::xyz(0.5, 1.0, 1.5)).to_euclidean();
    for k in 0..3 {
        assert!(close(c[k], want[k], 1e-10));
    }
}

/// The paper's fuel tank: the part of a closed mesh below a plane, from the triangles below it
/// alone, closed by the plane through the apex. Here a box without its top, filled to the brim.
#[test]
fn capped_by_a_plane() {
    let (v, f) = cuboid(2.0, 3.0, 4.0);
    let below = f
        .iter()
        .filter(|t| t.iter().any(|&i| v[i].to_euclidean()[2] < 3.9))
        .map(|t| [v[t[0]], v[t[1]], v[t[2]]]);
    let m = pga3d::Moments::of_triangles(below, pga3d::Point::xyz(0.7, 0.2, 4.0));
    assert!(close(m.volume(), 24.0, 1e-12));
    assert!(close(m.centroid().to_euclidean()[2], 2.0, 1e-12));
}

/// A sphere's triangulation (a subdivided octahedron, pushed out to the unit sphere): volume
/// and moments converge to 4π/3 and 8π/15 with the subdivision.
#[test]
fn a_sphere_converges() {
    let mut tris: Vec<[[f64; 3]; 3]> = {
        let (x, y, z) = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let n = |a: [f64; 3]| a.map(|c| -c);
        vec![
            [x, y, z],
            [y, n(x), z],
            [n(x), n(y), z],
            [n(y), x, z],
            [y, x, n(z)],
            [n(x), y, n(z)],
            [n(y), n(x), n(z)],
            [x, n(y), n(z)],
        ]
    };
    let unit = |a: [f64; 3]| {
        let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        a.map(|c| c / l)
    };
    let mid = |a: [f64; 3], b: [f64; 3]| unit([a[0] + b[0], a[1] + b[1], a[2] + b[2]]);
    for _ in 0..6 {
        tris = tris
            .iter()
            .flat_map(|&[a, b, c]| {
                let (ab, bc, ca) = (mid(a, b), mid(b, c), mid(c, a));
                [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
            })
            .collect();
    }
    let p = |a: [f64; 3]| pga3d::Point::xyz(a[0], a[1], a[2]);
    let m = pga3d::Moments::of_triangles(
        tris.iter().map(|t| t.map(p)),
        pga3d::Point::xyz(0.0, 0.0, 0.0),
    );
    let ball = 4.0 * std::f64::consts::PI / 3.0;
    assert!(close(m.volume(), ball, 2e-3), "{}", m.volume());
    // ∫ x² dV = 4π/15 for the unit ball.
    assert!(close(
        m.central_second_moments()[0][0],
        4.0 * std::f64::consts::PI / 15.0,
        4e-3
    ));
}

/// 2D: the shoelace area and centroid of a polygon, and polar moments.
#[test]
fn polygons() {
    let p = |x, y| pga2d::Point::<(), f64>::xy(x, y);
    // A right triangle (0,0), (3,0), (0,4): area 6, centroid (1, 4/3), and J about the centroid
    // A (a² + b²) / 18 for legs a, b.
    let t = pga2d::Moments::of_polygon(&[p(0.0, 0.0), p(3.0, 0.0), p(0.0, 4.0)]);
    assert!(close(t.area(), 6.0, 1e-12));
    let [x, y] = t.centroid().to_euclidean();
    assert!(close(x, 1.0, 1e-12) && close(y, 4.0 / 3.0, 1e-12));
    assert!(close(t.polar_moment(1.0), 6.0 * 25.0 / 18.0, 1e-12));
    // Clockwise: negative area.
    let back = pga2d::Moments::of_polygon(&[p(0.0, 4.0), p(3.0, 0.0), p(0.0, 0.0)]);
    assert!(close(back.area(), -6.0, 1e-12));
    // A regular hexagon of circumradius 1: area 3√3/2, J = 5√3/8 at density 1.
    let hex: Vec<_> = (0..6)
        .map(|k| {
            let a = f64::from(k) * std::f64::consts::FRAC_PI_3;
            p(a.cos(), a.sin())
        })
        .collect();
    let h = pga2d::Moments::of_polygon(&hex);
    assert!(close(h.area(), 1.5 * 3f64.sqrt(), 1e-12));
    assert!(close(h.polar_moment(1.0), 5.0 * 3f64.sqrt() / 8.0, 1e-12));
    // Moved polygons: the same polar moment, the centroid moved.
    let m = pga2d::Motor::translation(2.0, -1.0) * pga2d::Motor::rotation(p(0.0, 0.0), 0.4);
    let moved: Vec<_> = hex.iter().map(|&q| m >> q).collect();
    let g = pga2d::Moments::of_polygon(&moved);
    assert!(close(g.polar_moment(1.0), h.polar_moment(1.0), 1e-12));
    let c = g.centroid().to_euclidean();
    assert!(close(c[0], 2.0, 1e-12) && close(c[1], -1.0, 1e-12));
    let mm = h.moved(m);
    for (x, y) in g.form.c[0]
        .iter()
        .flatten()
        .zip(mm.form.c[0].iter().flatten())
    {
        assert!(close(*x, *y, 1e-12));
    }
}

/// The principal frame diagonalizes the inertia: its axes are eigenvectors, for poses that
/// include half turns (each branch of the frame's rotation from its axes).
#[test]
fn the_frame_diagonalizes_the_inertia() {
    let (v, f) = cuboid(1.0, 2.0, 3.0);
    let pi = std::f64::consts::PI;
    for m in [
        pga3d::Motor::rotation_about(0.0, 0.0, 1.0, pi),
        pga3d::Motor::rotation_about(1.0, 0.0, 0.0, pi),
        pga3d::Motor::rotation_about(0.0, 1.0, 0.0, pi),
        pga3d::Motor::rotation_about(1.0, 1.0, 0.0, pi),
        pga3d::Motor::rotation_about(0.3, -0.5, 0.8, 2.0),
        pga3d::Motor::rotation_about(0.0, 0.0, 1.0, 0.0),
    ] {
        let moved: Vec<_> = v.iter().map(|p| m >> *p).collect();
        let body = pga3d::Moments::of_mesh(&moved, &f);
        let c = body.central_second_moments();
        let tr = c[0][0] + c[1][1] + c[2][2];
        let (inertia, frame) = body.inertia(1.0);
        for (k, axis) in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            .iter()
            .enumerate()
        {
            let d = (frame >> pga3d::Point::direction(axis[0], axis[1], axis[2])).c;
            let d = [d[0], d[1], d[2]];
            // I d = (tr C − C) d = λ d.
            for i in 0..3 {
                let id: f64 = (0..3)
                    .map(|j| (if i == j { tr } else { 0.0 } - c[i][j]) * d[j])
                    .sum();
                assert!(
                    close(id, inertia.moments[k] * d[i], 1e-9),
                    "{id} vs {}",
                    inertia.moments[k] * d[i]
                );
            }
        }
    }
}

/// `f32`: the same results to single precision.
#[test]
fn single_precision() {
    let p = |x: f32, y: f32, z: f32| pga3d::Point::<(), f32>::xyz(x, y, z);
    let v = [
        p(0.0, 0.0, 0.0),
        p(1.0, 0.0, 0.0),
        p(0.0, 1.0, 0.0),
        p(0.0, 0.0, 1.0),
    ];
    let f = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
    let m = pga3d::Moments::<f32>::of_mesh(&v, &f);
    assert!((m.volume() - 1.0 / 6.0).abs() < 1e-6);
    let (inertia, _) = m.inertia(1.0);
    assert!(inertia.moments.iter().all(|x| x.is_finite() && *x > 0.0));
}
