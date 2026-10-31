//! numga's tests of the minimal surfaces, and its scenario checks. (numga's check that the math
//! does not import the plotting libraries is the module split here: `minimal.rs` uses no
//! drawing code.)

use super::minimal::*;

/// The largest coefficient of a list of points.
fn largest(points: impl IntoIterator<Item = Point>) -> f64 {
    let each = points.into_iter().flat_map(|p| p.c).map(f64::abs);
    each.fold(0.0, f64::max)
}

/// The largest coefficient of the difference of two lists of points.
fn apart(a: &[Point], b: &[Point]) -> f64 {
    largest(a.iter().zip(b).map(|(p, q)| *p - *q))
}

/// The points at the interior vertices.
fn interior_of<V: gax::Extensor<Slots = (), Coef = f64>>(
    mesh: &gax_numga_examples::mesh::Mesh<V>,
    points: &[Point],
) -> Vec<Point> {
    interior_and_wire(mesh)
        .0
        .iter()
        .map(|v| points[*v])
        .collect()
}

/// The quarter-turned sides are twice the area-weighted corner gradients, and summed at the
/// vertices they are the area gradient and the cotangent Laplacian's pull alike.
fn check_area_gradient(faces: &[[usize; 3]], vertices: &[Point]) -> Vec<Gradient> {
    let n = vertices.len();
    let triangles = triangles(faces, vertices);
    for t in &triangles {
        for (turned, gradient) in t.turned_sides().iter().zip(t.gradients) {
            let twice = (gradient * (2.0 * t.area)).hodge();
            assert!(largest([*turned - twice]) <= 1e-13);
        }
    }
    let laplacian = laplacian(faces, &triangles, n);
    let pull = area_pull(faces, &triangles, n);
    let weighted = triangles.iter().map(|t| t.gradients.map(|g| g * t.area));
    let gradient = at_vertices(faces, weighted, Gradient::zero(), n);
    let duals: Vec<Point> = gradient.iter().map(|g| g.hodge()).collect();
    assert!(apart(&pull, &duals) <= 1e-13);
    let field = gax::field::Field::new(vertices.to_vec());
    let pulled: gax::field::Field<Point> = (&laplacian * Point::slot()).of(&field);
    assert!(apart(&pull, pulled.sites()) <= 1e-12);
    gradient
}

#[test]
fn turned_sides_area_gradient_and_plateau_relaxation() {
    let (mesh, _, initial) = four_arch_dome(1.0, 6, 0.35, 0.55);
    let gradient = check_area_gradient(&mesh.faces, &initial);

    // A finite difference against the exact sum of triangle areas, at the first interior vertex.
    let (interior, _) = interior_and_wire(&mesh);
    let displacement = (x_point() * 0.4 - z_point() * 0.7) * 1e-7;
    let total = |shift: f64| {
        let mut moved = initial.clone();
        moved[interior[0]] += displacement * shift;
        triangles(&mesh.faces, &moved)
            .iter()
            .map(|t| t.area)
            .sum::<f64>()
    };
    // The area gradient, a form on displacements, reads the step.
    let derivative = (gradient[interior[0]] & displacement).s();
    assert!(((total(1.0) - total(-1.0)) / 2.0 - derivative).abs() <= 1e-12);

    // The damped relaxation keeps the wire and strictly decreases the area.
    let states = relax_plateau(&mesh, &initial, 0.25, 16);
    assert!(states.windows(2).all(|w| w[1].area < w[0].area));
    let (_, wire) = interior_and_wire(&mesh);
    for s in &states {
        assert!(largest(wire.iter().map(|v| s.vertices[*v] - initial[*v])) <= 1e-14);
    }
    let mean_square =
        |s: &State| s.curvature.iter().map(|c| c * c).sum::<f64>() / s.curvature.len() as f64;
    assert!(mean_square(&states[states.len() - 1]) < 1e-2 * mean_square(&states[0]));
}

#[test]
fn conjugate_minimal_surface_interior_cancellation_and_enneper() {
    let (mesh, _, enneper, exact) = enneper_disk(0.8, 10);
    let (conjugate, flux) = conjugate_minimal_surface(&mesh.faces, &enneper);
    // The sides facing each vertex sum to zero at every interior vertex.
    assert!(largest(interior_of(&mesh, &flux)) <= 1e-14);
    // The conjugate matches the exact Weierstrass-Enneper one up to a translation.
    assert!(apart(&centred(&conjugate), &centred(&exact)) <= 2e-2);
    // A second conjugate turns the differentials by a half turn, negating the surface.
    let (second, second_flux) = conjugate_minimal_surface(&mesh.faces, &exact);
    assert!(largest(interior_of(&mesh, &second_flux)) <= 1e-14);
    let negated: Vec<Point> = centred(&enneper).iter().map(|p| -*p).collect();
    assert!(apart(&centred(&second), &negated) <= 2e-2);
}

#[test]
fn bonnet_isometric_family_catenoid_to_helicoid() {
    let c = slit_catenoid(1.15, 0.85 * core::f64::consts::PI, 22, 44);
    let (helicoid, flux) = conjugate_minimal_surface(&c.mesh.faces, &c.catenoid);
    assert!(largest(interior_of(&c.mesh, &flux)) <= 1e-14);
    assert!(apart(&centred(&helicoid), &centred(&c.helicoid)) <= 2e-2);
    // Along Bonnet's circle the normal at each interior vertex keeps its direction.
    let (interior, _) = interior_and_wire(&c.mesh);
    let normals: Vec<Vec<Patch>> = (0..5)
        .map(|k| {
            let angle = core::f64::consts::FRAC_PI_2 * k as f64 / 4.0;
            let (_, n) = bonnet(&c.mesh.faces, &c.catenoid, &c.helicoid, angle);
            interior.iter().map(|v| n[*v]).collect()
        })
        .collect();
    for later in &normals[1..] {
        let each = later.iter().zip(&normals[0]).flat_map(|(a, b)| (*a - *b).c);
        assert!(each.map(f64::abs).fold(0.0, f64::max) <= 5e-4);
    }
}

/// The lifting of the domain curves samples the flat disk where it is: each sample's hats in its
/// triangle sum to one and weigh the corners to the sample.
#[test]
fn the_lifting_reproduces_the_flat_disk() {
    let (mesh, flat) = disk(1.0, 6);
    let curves = super::domain_curves(&mesh, &flat);
    let lift = super::lifting(&mesh.faces, &flat, &curves);
    for (curve, lifted) in curves.iter().zip(&lift) {
        let samples: Vec<Point> = lifted.iter().map(|l| super::lifted(l, &flat)).collect();
        assert!(apart(curve, &samples) <= 1e-12);
    }
}

/// numga's scenario checks: the relaxation's area falls at every step; the sums of the sides
/// facing each vertex cancel inside, on all three surfaces; and the recovered conjugates match
/// the exact ones up to a translation.
#[test]
fn the_scenes_relax_and_recover_their_conjugates() {
    let (mesh, _, initial) = four_arch_dome(DISK_RADIUS, DISK_DIVISIONS, ARCH_HEIGHT, DOME_HEIGHT);
    check_area_gradient(&mesh.faces, &initial);
    let states = relax_plateau(&mesh, &initial, RELAXATION_STEP, RELAXATION_STEPS);
    assert!(states.windows(2).all(|w| w[1].area < w[0].area));
    let film = &states[states.len() - 1].vertices;
    let (_, film_flux) = conjugate_minimal_surface(&mesh.faces, film);
    assert!(largest(interior_of(&mesh, &film_flux)) <= 1e-14);

    let (enneper_mesh, _, enneper, exact) = enneper_disk(ENNEPER_RADIUS, ENNEPER_DIVISIONS);
    let (conjugate, flux) = conjugate_minimal_surface(&enneper_mesh.faces, &enneper);
    assert!(largest(interior_of(&enneper_mesh, &flux)) <= 1e-14);
    assert!(apart(&centred(&conjugate), &centred(&exact)) <= 2e-2);

    let c = slit_catenoid(
        CATENOID_HEIGHT,
        CATENOID_SPAN,
        CATENOID_HEIGHTS,
        CATENOID_AZIMUTHS,
    );
    let (helicoid, flux) = conjugate_minimal_surface(&c.mesh.faces, &c.catenoid);
    assert!(largest(interior_of(&c.mesh, &flux)) <= 1e-14);
    assert!(apart(&centred(&helicoid), &centred(&c.helicoid)) <= 2e-2);
}

#[test]
fn a_frame_draws() {
    gax_numga_examples::app::assert_draws_with(super::draw, &super::scenes(), 0.5);
}
