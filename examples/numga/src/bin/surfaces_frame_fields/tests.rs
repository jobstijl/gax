//! numga's tests of the frame fields, and its scenario checks. (numga's check that the math
//! imports no plotting library is the module split here: `frames.rs` draws nothing.)

use super::frames::*;
use crate::simplicial::{boundary, exterior, incidence, interior};
use gax::field::{Field, SparseMap};

/// The size of a value: the norm of its coefficients.
fn size<M: gax::Extensor<Coef = f64>>(m: M) -> f64 {
    Field::new(vec![m]).norm()
}

#[test]
fn darboux_whitney_grade_decomposition_and_maurer_cartan() {
    let (corners, rotors, points) = single_triangle(8);
    let sides = sides(corners);
    let patch = (sides[0] & sides[1]) * 0.5;
    let orientation = [1.0, -1.0, 1.0];
    for &p in &points {
        let d = triangle_darboux(corners, p, rotors);
        // The bivector and scalar parts of `2 ~R dR` are the Whitney rates and the symmetric
        // differentials.
        assert!(size(d.darboux.grade::<2>() - d.whitney_rates) < 1e-14);
        assert!(size(d.darboux.grade::<0>() - d.chord_stretch) < 1e-14);
        assert!((d.unit.norm_squared() - 1.0).abs() < 1e-14);
        // Inside, the chord cuts strictly inside the unit sphere.
        assert!(d.chord.norm_squared() < 1.0 - 1e-3);
        // Each Whitney form is the step from its corner contracted with the patch's inverse,
        // halved, signed as the triangle runs along the stored edge.
        for k in 0..3 {
            let radial = ((p - corners[k]) | patch.inverse()) * (0.5 * orientation[k]);
            assert!(size(d.whitney_forms[k] - radial) < 1e-14);
        }
        // The rates over the chord's squared size are the unit rotors' Maurer–Cartan form.
        let step = (sides[0] * 0.6 - sides[1] * 0.35) * 1e-7;
        let shifted = triangle_darboux(corners, p + step, rotors).unit;
        let finite = (d.unit.reverse() * (shifted - d.unit)).grade::<2>() * 2.0;
        assert!(size(finite - d.angular_velocity.of(step)) < 1e-12);
    }
    // At the corners the chord is the corner's unit rotor.
    for k in 0..3 {
        let d = triangle_darboux(corners, corners[k], rotors);
        assert!(size(d.chord - rotors[k]) < 1e-14);
        assert!(size(d.darboux.grade::<2>() - d.whitney_rates) < 1e-14);
        assert!(size(d.darboux.grade::<0>() - d.chord_stretch) < 1e-14);
    }
    // Constant rotors: no rate, no stretch, unit everywhere.
    let constant = [rotors[0]; 3];
    for &p in &points {
        let d = triangle_darboux(corners, p, constant);
        assert!((d.chord.norm_squared() - 1.0).abs() < 1e-14);
        let probe = point(0.3, -0.2, 0.7);
        assert!(size(d.whitney_rates.of(probe)) < 1e-14);
        assert!(size(d.chord_stretch.of(probe)) < 1e-14);
    }
}

/// The topological charge: the swept solid angles over four pi.
fn charge(s: &Skyrmion) -> f64 {
    s.solid_angles.iter().sum::<f64>() / (4.0 * core::f64::consts::PI)
}

#[test]
fn skyrmion_topological_charge_and_cotangent_trace() {
    let (mesh, vertices) = triangular_disk(1.0, 6);
    let s = skyrmion_winding(&mesh, &vertices, core::f64::consts::PI);
    let rim = exterior(&mesh, 0);
    for v in 0..vertices.len() {
        let unit =
            s.scalar_parts[v] * s.scalar_parts[v] + s.bivector_norms[v] * s.bivector_norms[v];
        assert!((unit - 1.0).abs() < 1e-14);
    }
    let z = |p: Point| coordinates(p)[2];
    assert!((s.scalar_parts[0] - 1.0).abs() < 1e-14);
    assert!((z(s.directors.site(0)) - 1.0).abs() < 1e-14);
    for &v in &rim {
        assert!(s.scalar_parts[v].abs() < 1e-13);
        assert!((z(s.directors.site(v)) + 1.0).abs() < 1e-13);
    }

    // The cotangent trace of the Maurer–Cartan form is its orthonormal trace on every face.
    let edges = &mesh.d::<Scalar>(0) * &vertices;
    let sides = boundary(&mesh, 2).read_all(&edges);
    let at = incidence(&mesh, 2, 0);
    let axes = [
        point(1.0, 0.0, 0.0),
        point(0.0, 1.0, 0.0),
        point(0.0, 0.0, 1.0),
    ];
    for f in 0..mesh.len(2) {
        let g = gradients(core::array::from_fn(|k| sides[k].site(f)));
        let r: [Rotor; 3] = core::array::from_fn(|k| s.rotors.site(at.sites[f][k]));
        let chord = (r[0] + r[1] + r[2]) * (1.0 / 3.0);
        let differential = (0..3)
            .map(|k| r[k] * (g[k] & Point::slot()))
            .reduce(|a, b| a + b)
            .unwrap();
        let rates =
            (chord.reverse() * differential * 2.0).grade::<2>() * chord.norm_squared().recip();
        let trace: f64 = axes.iter().map(|&e| rates.of(e).norm_squared()).sum();
        assert!((s.energy_density[f] - trace).abs() < 1e-13);
    }

    // The charge is one, closer on a finer disk.
    let (coarse_mesh, coarse_vertices) = triangular_disk(1.0, 4);
    let coarse = charge(&skyrmion_winding(
        &coarse_mesh,
        &coarse_vertices,
        core::f64::consts::PI,
    ));
    let fine = charge(&s);
    assert!((fine - 1.0).abs() < (coarse - 1.0).abs(), "{fine} {coarse}");
    assert!((fine - 1.0).abs() < 1.5e-2, "{fine}");

    // A whole turn reaches the opposite rotor at the rim, and its director is back at the
    // north pole.
    let whole = skyrmion_winding(&mesh, &vertices, 2.0 * core::f64::consts::PI);
    for &v in &rim {
        assert!((whole.scalar_parts[v] + 1.0).abs() < 1e-13);
        assert!((z(whole.directors.site(v)) - 1.0).abs() < 1e-13);
    }
}

#[test]
fn harmonic_rotor_frames_and_projected_relaxation() {
    let saddle = saddle_surface(1.0, 5, 0.28);
    let (mesh, vertices) = (&saddle.mesh, &saddle.vertices);
    let operators = rotor_laplacian(mesh, vertices);

    // The Whitney 0-form gradient pulled back by its adjoint is the cotangent Laplacian on the
    // curved saddle.
    let edges = &mesh.d::<Scalar>(0) * vertices;
    let sides = boundary(mesh, 2).read_all(&edges);
    let corner_gradients: [Field<Plane>; 3] = core::array::from_fn(|k| {
        Field::from_fn(mesh.len(2), |f| {
            gradients(core::array::from_fn(|j| sides[j].site(f)))[k]
        })
    });
    let gradient =
        incidence(mesh, 2, 0).assemble(|f, k| corner_gradients[k].site(f) * Scalar::slot());
    let areas = SparseMap::from_diagonal(&operators.areas.map(|a| a * Plane::slot()));
    let stiffness = gradient.adjoint().compose(&areas.compose(&gradient));
    let d0 = mesh.d::<Scalar>(0);
    let weighted = SparseMap::from_diagonal(&operators.weights);
    let scalar_laplacian = &(&d0.reverse() * &weighted) * &d0;
    let test = vertices.map(|p| {
        let [x, _, z] = coordinates(p);
        Scalar::new(x * x - 0.4 * z)
    });
    let (a, b) = (stiffness.of(&test), &scalar_laplacian * &test);
    assert!((0..vertices.len()).all(|v| (a.site(v) - b.site(v)).s().abs() < 1e-14));

    // The chords satisfy Laplace's equation; their unit rotors keep the boundary's.
    let free = interior(mesh, 0);
    for prescribed in &saddle.prescribed {
        let (chords, rotors) = harmonic_rotor_frames(mesh, vertices, &saddle.rim, prescribed);
        let residual = operators.laplacian.of(&chords);
        assert!(free.iter().all(|&v| size(residual.site(v)) < 1e-11));
        for &v in &saddle.rim {
            assert!(size(rotors.site(v) - prescribed.site(v)) < 1e-13);
        }
        assert!(
            rotors
                .iter()
                .all(|r| (r.norm_squared() - 1.0).abs() < 1e-14)
        );
    }

    // The projected relaxation keeps every rotor unit and the boundary fixed, and lowers the
    // energy at every step.
    let states = relax_rotor_frames(mesh, vertices, &saddle.start, 0.18, 12);
    assert!(states.windows(2).all(|w| w[1].1 < w[0].1));
    for (rotors, _) in &states {
        assert!(
            rotors
                .iter()
                .all(|r| (r.norm_squared() - 1.0).abs() < 1e-14)
        );
        for &v in &saddle.rim {
            assert!(size(rotors.site(v) - saddle.start.site(v)) < 1e-14);
        }
    }
}

/// numga's scenario checks on its own sizes.
#[test]
fn the_scenes_pass_numgas_checks() {
    let (corners, rotors, points) = single_triangle(TRIANGLE_SUBDIVISIONS);
    for &p in &points {
        let d = triangle_darboux(corners, p, rotors);
        assert!(size(d.darboux.grade::<2>() - d.whitney_rates) < 1e-14);
        assert!(size(d.darboux.grade::<0>() - d.chord_stretch) < 1e-14);
    }
    let (mesh, vertices) = triangular_disk(DISK_RADIUS, SKYRMION_DIVISIONS);
    let s = skyrmion_winding(&mesh, &vertices, SKYRMION_TWIST);
    assert!((charge(&s) - 1.0).abs() < 1.5e-2);
    let saddle = saddle_surface(DISK_RADIUS, SADDLE_DIVISIONS, SADDLE_HEIGHT);
    let operators = rotor_laplacian(&saddle.mesh, &saddle.vertices);
    let free = interior(&saddle.mesh, 0);
    for prescribed in &saddle.prescribed {
        let (chords, _) =
            harmonic_rotor_frames(&saddle.mesh, &saddle.vertices, &saddle.rim, prescribed);
        let residual = operators.laplacian.of(&chords);
        assert!(free.iter().all(|&v| size(residual.site(v)) < 1e-11));
    }
    let states = relax_rotor_frames(
        &saddle.mesh,
        &saddle.vertices,
        &saddle.start,
        RELAXATION_STEP,
        RELAXATION_STEPS,
    );
    assert!(states.windows(2).all(|w| w[1].1 < w[0].1));
}

#[test]
fn every_figure_draws() {
    let shown = super::shown();
    for t in [1.0, 4.5, 8.0, 12.0] {
        gax_numga_examples::app::assert_draws_with(super::draw, &shown, t);
    }
}
