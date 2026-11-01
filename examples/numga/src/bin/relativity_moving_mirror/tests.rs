//! numga's tests: hats, edge forms and the Lorentzian dual on a spacetime triangle; the wave
//! operator's symmetry and the causal light-cone wake; the moving mirror's Dirichlet walls,
//! stationarity, energy conservation and Doppler gain; and the scenario's checks at its size.

use super::scenes::*;
use super::spacetime::*;
use gax::field::Field;
use gax::mesh::corners as local_corners;

/// Whether a list of numbers all lie within `tolerance` of zero.
fn small(values: impl IntoIterator<Item = f64>, tolerance: f64) -> bool {
    values.into_iter().all(|v| v.abs() < tolerance)
}

#[test]
fn spacetime_barycentric_hats_edge_forms_and_lorentzian_dual() {
    let (corners, points) = single_triangle(TRIANGLE_SUBDIVISIONS);
    let (hats, gradients, _) = barycentric_hats(corners, &points);
    let (corner_hats, _, _) = barycentric_hats(corners, &corners);
    for (i, row) in corner_hats.iter().enumerate() {
        assert!(small(
            (0..3).map(|j| row[j].s() - f64::from(u8::from(i == j))),
            1e-14
        ));
    }
    assert!(small(
        hats.iter().map(|h| (h[0] + h[1] + h[2]).s() - 1.0),
        1e-14
    ));
    assert!(small((gradients[0] + gradients[1] + gradients[2]).c, 1e-14));

    // Barycentric interpolation reproduces any affine spacetime potential and its gradient.
    let true_gradient = x_plane() * 1.3 - t_plane() * 0.75;
    let value = |p: Point| 0.4 + (true_gradient & p).s();
    let corner_values = corners.map(value);
    for (p, h) in points.iter().zip(&hats) {
        let interpolated: f64 = (0..3).map(|k| corner_values[k] * h[k].s()).sum();
        assert!((interpolated - value(*p)).abs() < 1e-14);
    }
    let reconstructed = (0..3)
        .map(|k| gradients[k] * corner_values[k])
        .fold(Plane::zero(), |a, b| a + b);
    assert!(small((reconstructed - true_gradient).c, 1e-14));

    // The edge forms are Whitney's alternating formula, each edge from its tail to its head run
    // the way the triangle runs it, and read one along their own edge, nothing along the others.
    let forms = edge_whitney_forms(corners, &points);
    let ends = local_corners(2, 1);
    for ((form, h), _) in forms.iter().zip(&hats).zip(&points) {
        for e in 0..3 {
            let (t, hd) = (ends[e][0], ends[e][1]);
            // numga's `face_edge_orientation`: the facet facing corner 1 runs from 2 to 0.
            let sense = if e == 1 { -1.0 } else { 1.0 };
            let classic = (gradients[hd] * h[t] - gradients[t] * h[hd]) * sense;
            assert!(small((form[e] - classic).c, 1e-14));
        }
    }
    let sides = sides(corners);
    for fraction in [0.2, 0.5, 0.85] {
        for e in 0..3 {
            let (t, hd) = (ends[e][0], ends[e][1]);
            let on_edge = corners[t] * (1.0 - fraction) + corners[hd] * fraction;
            let readings = edge_whitney_forms(corners, &[on_edge])[0];
            for (k, form) in readings.iter().enumerate() {
                let expected = f64::from(u8::from(k == e));
                assert!(((*form & sides[e]).s() - expected).abs() < 1e-14);
            }
        }
    }

    // A null gradient pairs to nothing with its Lorentzian dual; the observer's energy is
    // positive.
    for null in [x_plane() - t_plane(), x_plane() + t_plane()] {
        assert!((null & dual().of(null)).s().abs() < 1e-14);
    }
    let plane = x_plane() * 3.0 + t_plane() * 4.0;
    assert!(((plane & dual().of(plane)).s() + 7.0).abs() < 1e-14);
    let reflected = t_plane() >> dual();
    assert!((-(plane & reflected.of(plane)).s() - 25.0).abs() < 1e-14);
    // The unit displacements pair to one with the planes that read them.
    assert!(((x_plane() & space()).s() - 1.0).abs() < 1e-15);
    assert!(((t_plane() & time()).s() - 1.0).abs() < 1e-15);
}

/// The largest coefficient of a field of numbers.
fn largest(field: &Field<Scalar>, keep: impl Fn(usize) -> bool) -> f64 {
    field
        .iter()
        .enumerate()
        .filter(|&(i, _)| keep(i))
        .map(|(_, x)| x.s().abs())
        .fold(0.0, f64::max)
}

/// The wake outside its forward light cone, against its largest value anywhere.
fn wake_outside_cone(
    columns: usize,
    rows: usize,
    duration: f64,
    source: Point,
    width: f64,
    margins: [f64; 2],
) -> f64 {
    let (mesh, vertices, _) = spacetime_strip(columns, rows, &vec![1.0; rows], duration);
    let operators = Operators::new(&mesh, &vertices);
    let wake = causal_wake(&operators, columns, rows, source, width);
    let outside = |i: usize| {
        let d = vertices.site(i) - source;
        let [distance, delay] = coordinates(d);
        distance.abs() > delay + margins[0] || delay < -margins[1]
    };
    largest(&wake, outside) / largest(&wake, |_| true)
}

#[test]
fn lorentzian_wave_operator_and_causal_light_cone_wake() {
    let (columns, rows) = (45, 75);
    let (mesh, vertices, _) = spacetime_strip(columns, rows, &vec![1.0; rows], 0.72);
    let operators = Operators::new(&mesh, &vertices);
    // The wave operator is symmetric: its adjugate is itself.
    let adjugate = operators.wave.adjugate();
    let unit =
        |i: usize| Field::from_fn(vertices.len(), |k| Scalar::new(f64::from(u8::from(k == i))));
    for i in (0..vertices.len()).step_by(97) {
        let (a, b) = (operators.wave.of(&unit(i)), adjugate.of(&unit(i)));
        assert!(small(
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).undual().s()),
            1e-12
        ));
    }
    let ratio = wake_outside_cone(columns, rows, 0.72, event(0.5, 0.14), 0.042, [0.16, 0.12]);
    assert!(ratio < 0.025, "{ratio}");
}

/// numga's test strip: a smaller cavity, its pulse marched up it.
fn mirror(columns: usize, rows: usize) -> (Operators, Wave) {
    let walls: Vec<f64> = (0..rows)
        .map(|r| mirror_wall(r as f64 * MIRROR_DURATION / (rows - 1) as f64))
        .collect();
    let (mesh, vertices, dt) = spacetime_strip(columns, rows, &walls, MIRROR_DURATION);
    let operators = Operators::new(&mesh, &vertices);
    let wave = moving_mirror_wave(
        &operators,
        columns,
        rows,
        dt,
        PULSE_CENTRE,
        PULSE_WIDTH,
        WAVELENGTH,
    );
    (operators, wave)
}

/// The largest wave rate along a row.
fn row_peak(wave: &Wave, columns: usize, row: usize) -> f64 {
    largest(&wave.rate, |i| i / columns == row)
}

#[test]
fn relativistic_moving_mirror_doppler_compression_and_energy_gain() {
    let (columns, rows) = (65, 300);
    let (operators, wave) = mirror(columns, rows);
    // The Dirichlet condition holds along both mirrors' worldlines.
    let walls = |i: usize| i.is_multiple_of(columns) || i % columns == columns - 1;
    assert!(largest(&wave.potential, walls) < 1e-14);
    // The Euler–Lagrange equation holds at every interior event.
    let interior = placing(
        &grid_sites(columns, 1..rows - 1, 1..columns - 1),
        operators.vertices.len(),
    );
    let residual = interior.adjugate().of(&operators.wave.of(&wave.potential));
    assert!(small(residual.iter().map(|a| a.undual().s()), 1e-10));
    // The energy is kept before and after the reflection, and boosted across it.
    let e = &wave.slab_energies;
    assert!((e[45] / e[15] - 1.0).abs() < 2e-2);
    assert!((e[260] / e[230] - 1.0).abs() < 2e-2);
    assert!(row_peak(&wave, columns, 240) > 1.70 * row_peak(&wave, columns, 45));
    assert!(e[240] > 2.10 * e[45]);
}

#[test]
fn the_scenarios_wake_is_causal_and_the_mirror_amplifies() {
    // The scenario's checks at its own sizes.
    let ratio = wake_outside_cone(
        WAKE_COLUMNS,
        WAKE_ROWS,
        WAKE_DURATION,
        event(SOURCE[0], SOURCE[1]),
        SOURCE_WIDTH,
        [0.14, 0.10],
    );
    assert!(ratio < 0.02, "{ratio}");
    let (_, wave) = mirror(MIRROR_COLUMNS, MIRROR_ROWS);
    let e = &wave.slab_energies;
    assert!(
        row_peak(&wave, MIRROR_COLUMNS, AFTER_ROW)
            > 1.85 * row_peak(&wave, MIRROR_COLUMNS, BEFORE_ROW)
    );
    assert!(e[AFTER_ROW] > 2.15 * e[BEFORE_ROW]);
}

#[test]
fn frames_draw() {
    let shown = super::shown();
    for t in [0.0, 1.7] {
        gax_numga_examples::app::assert_draws_with(super::draw, &shown, t);
    }
}
