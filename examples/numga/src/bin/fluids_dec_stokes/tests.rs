//! numga's tests: typed discrete derivatives and Hodge maps, the straight channel's convergence to
//! Poiseuille flow, the obstacle's flow balancing every cell, and the flow's independence of the
//! edges' directions.

use super::scenes::*;
use super::stokes::*;
use gax::field::Field;
use gax::vga2d;

fn small(values: impl IntoIterator<Item = f64>, tolerance: f64) -> bool {
    values.into_iter().all(|v| v.abs() < tolerance)
}

/// The numbers a scalar-celled map holds.
fn cells(map: &gax::field::SparseMap<ScalarMap>) -> Vec<f64> {
    map.couplings()
        .map(|(_, _, c)| c.of(Scalar::new(1.0)).s())
        .collect()
}

/// The numbers a Hodge map holds.
fn hodge_cells(map: &gax::field::SparseMap<HodgeMap>) -> Vec<f64> {
    map.couplings()
        .map(|(_, _, c)| c.of(Scalar::new(1.0)).s())
        .collect()
}

fn unsigned(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

#[test]
fn incidence_is_topological_and_hodges_supply_geometry_and_duality() {
    let channel = channel(8);
    let (mesh, vertices) = (&channel.mesh, &channel.vertices);
    let signs = unsigned(mesh.len(1));
    let operators = Operators::new(mesh, vertices, &signs);
    for d in [&operators.d0, &operators.d1] {
        assert!(cells(d).iter().all(|&c| c == 1.0 || c == -1.0));
    }
    assert!(small(cells(&operators.d1.compose(&operators.d0)), 1e-15));

    // Scaling and stretching the geometry leave every oriented boundary sum as it was.
    let scale = 1.7;
    let scaled = Operators::new(mesh, &vertices.map(|v| v * scale), &signs);
    let stretched = Operators::new(
        mesh,
        &vertices.map(|v| {
            let [x, y] = coordinates(v);
            point(1.2 * x, 0.85 * y)
        }),
        &signs,
    );
    for changed in [&scaled, &stretched] {
        assert_eq!(cells(&changed.d0), cells(&operators.d0));
        assert_eq!(cells(&changed.d1), cells(&operators.d1));
    }
    let pairs =
        |a: Vec<f64>, b: Vec<f64>, k: f64| a.iter().zip(&b).all(|(x, y)| (x - k * y).abs() < 1e-11);
    assert!(pairs(
        hodge_cells(&scaled.star0),
        hodge_cells(&operators.star0),
        scale * scale
    ));
    assert!(pairs(
        hodge_cells(&scaled.star1),
        hodge_cells(&operators.star1),
        1.0
    ));
    assert!(pairs(
        hodge_cells(&scaled.star2),
        hodge_cells(&operators.star2),
        1.0 / (scale * scale)
    ));
    assert!(
        stretched
            .areas
            .iter()
            .zip(operators.areas.iter())
            .all(|(a, b)| (a.s() - 1.02 * b.s()).abs() < 1e-11)
    );
    let differs = stretched
        .edge_ratios
        .iter()
        .zip(operators.edge_ratios.iter())
        .map(|(a, b)| (a - b).s().abs())
        .fold(0.0, f64::max);
    assert!(differs > 0.01);

    // The circumcentres are equidistant from each triangle's corners; the dual weights are
    // length ratios, all positive on this mesh.
    for f in 0..mesh.len(2) {
        let c = operators.circumcentres.site(f);
        let r = [0, 1, 2].map(|k| (operators.corners[k].site(f) - c).norm_squared());
        assert!((r[1] - r[0]).abs() < 1e-11 && (r[2] - r[0]).abs() < 1e-11);
    }
    for e in 0..mesh.len(1) {
        let ratio = operators.dual_edges.site(e).norm() / operators.edges.site(e).norm();
        assert!((operators.edge_ratios.site(e).s() - ratio).abs() < 1e-11);
        assert!(operators.edge_ratios.site(e).s() > 0.0);
    }

    // A constant flow reads positive circulation along the dual direction drawn.
    let gradient = vga2d::Vector::new(-0.3, 1.0);
    let potential = vertices.map(|v| gradient & v);
    let circulation = operators.dual_edges.map(|d| -gradient.hodge() & d);
    let read = operators.star1.of(&operators.d0.of(&potential));
    assert!(small(
        read.iter()
            .zip(circulation.iter())
            .map(|(a, b)| (a - b).s()),
        1e-11
    ));

    // The Hodge maps invert, and the inverse star turns a vertex value into its vorticity.
    let values = Field::from_fn(mesh.len(0), |i| {
        Scalar::new(gax_numga_examples::signal::wave(i as f64))
    });
    let vorticities = values.map(|v| Bivector::new(v.s()));
    let edge_values = Field::from_fn(mesh.len(1), |i| {
        Scalar::new(gax_numga_examples::signal::wave(i as f64 * 0.7))
    });
    // A patch read as a number: the same coefficient, extrinsic.
    let number = |p: Patch| p.with_orientation::<gax::Extrinsic>();
    let star0 = operators
        .star0
        .map_cells(|c| c.with_orientation::<gax::Extrinsic>());
    let back = star0.solve_direct(&operators.star0.of(&values).map(number));
    assert!(small(
        back.iter().zip(values.iter()).map(|(a, b)| (a - b).s()),
        1e-11
    ));
    let turned = operators.inverse_star0.of(&operators.star0.of(&values));
    assert!(small(
        turned
            .iter()
            .zip(vorticities.iter())
            .map(|(a, b)| (a - b).e12()),
        1e-11
    ));
    let dual_edges = operators.star1.of(&edge_values);
    let star1 = operators
        .star1
        .map_cells(|c| c.with_orientation::<gax::Extrinsic>());
    let back = star1.solve_direct(&dual_edges.map(number));
    assert!(small(
        back.iter()
            .zip(edge_values.iter())
            .map(|(a, b)| (a - b).s()),
        1e-11
    ));

    // The incidence's adjugate and the Hodge map's adjugate read the same dissipation from all
    // three sides.
    let vertex_work: f64 = vorticities
        .iter()
        .zip(operators.d0_vorticity.adjugate().of(&dual_edges).iter())
        .map(|(w, d)| (w & d).s())
        .sum();
    let dual_edge_work: f64 = operators
        .d0_vorticity
        .of(&vorticities)
        .iter()
        .zip(dual_edges.iter())
        .map(|(w, d)| (w & d).s())
        .sum();
    let primal_edge_work: f64 = operators
        .star1
        .adjugate()
        .of(&operators.d0_vorticity.of(&vorticities))
        .iter()
        .zip(edge_values.iter())
        .map(|(w, v)| (w & v).s())
        .sum();
    assert!((vertex_work - dual_edge_work).abs() < 1e-11);
    assert!((dual_edge_work - primal_edge_work).abs() < 1e-11);

    // Reflecting the mesh reflects the vorticity and leaves the viscous map as it was.
    let mirror = vga2d::Vector::new(0.0, 1.0);
    let mirrored = Operators::new(mesh, &vertices.map(|v| mirror >> v), &signs);
    let a = mirrored.curl.of(&edge_values);
    let b = operators.curl.of(&edge_values).map(|w| mirror >> w);
    assert!(small(
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).e12()),
        1e-11
    ));
    let viscous = |o: &Operators| -> Vec<f64> {
        o.star1
            .adjugate()
            .compose(&o.d0_vorticity.compose(&o.curl))
            .couplings()
            .map(|(_, _, c)| c.of(Scalar::new(1.0)).undual().s())
            .collect()
    };
    assert!(small(
        viscous(&mirrored)
            .iter()
            .zip(viscous(&operators))
            .map(|(a, b)| a - b),
        1e-11
    ));
}

/// The area-weighted root mean square of the differences.
fn rms(differences: impl Iterator<Item = f64>, weights: impl Iterator<Item = f64>) -> f64 {
    let (mut sum, mut total) = (0.0, 0.0);
    for (d, w) in differences.zip(weights) {
        sum += d * d * w;
        total += w;
    }
    #[allow(clippy::disallowed_methods)] // a root mean square, a statistic
    (sum / total).sqrt()
}

#[test]
fn straight_channel_converges_to_poiseuille_velocity_pressure_and_vorticity() {
    let mut errors = Vec::new();
    for divisions in [8, 16] {
        let channel = plain_channel(divisions);
        let signs = unsigned(channel.mesh.len(1));
        let (operators, _, flow) = flow(&channel, &signs);
        let areas = || operators.areas.iter().map(|a| a.s());
        let velocity = flow
            .velocity
            .iter()
            .zip(operators.face_centres.iter())
            .map(|(v, c)| {
                let [_, y] = coordinates(c);
                let exact = point(1.5 * (1.0 - 4.0 * y * y / (HEIGHT * HEIGHT)), 0.0);
                (v - exact).norm()
            });
        let exact_pressure: Vec<f64> = operators
            .circumcentres
            .iter()
            .map(|c| -12.0 * VISCOSITY * coordinates(c)[0] / (HEIGHT * HEIGHT))
            .collect();
        let total: f64 = areas().sum();
        let mean = exact_pressure
            .iter()
            .zip(areas())
            .map(|(p, a)| p * a)
            .sum::<f64>()
            / total;
        let pressure = flow
            .pressure
            .iter()
            .zip(&exact_pressure)
            .map(|(p, e)| p.s() - (e - mean));
        let vorticity = flow
            .vorticity
            .iter()
            .zip(channel.vertices.iter())
            .map(|(w, v)| w.e12() - 12.0 * coordinates(v)[1] / (HEIGHT * HEIGHT));
        errors.push([
            rms(velocity, areas()),
            rms(pressure, areas()),
            rms(vorticity, operators.dual_areas.iter().map(|a| a.s())),
        ]);
    }
    let (coarse, fine) = (errors[0], errors[1]);
    for k in 0..3 {
        assert!(fine[k] < 0.75 * coarse[k], "{errors:?}");
        assert!(fine[k] < [0.05, 0.08, 0.04][k], "{errors:?}");
    }
}

#[test]
fn obstacle_flow_balances_every_cell_and_preserves_boundary_flux() {
    let channel = channel(12);
    let signs = unsigned(channel.mesh.len(1));
    let (operators, prescribed, flow) = flow(&channel, &signs);
    assert!(small(
        operators.d1.of(&flow.flux).iter().map(|x| x.s()),
        1e-10
    ));
    for &e in &boundary_edges(&channel.mesh) {
        assert!((flow.flux.site(e) - prescribed.site(e)).s().abs() < 1e-11);
    }
    assert!(small(
        channel.walls.iter().map(|&e| flow.flux.site(e).s()),
        1e-11
    ));

    // Summing the faces' oriented boundaries cancels every interior edge and reads the net
    // throughflow.
    let mut outgoing = vec![0.0; channel.mesh.len(1)];
    for (_, e, cell) in operators.d1.couplings() {
        outgoing[e] += cell.of(flow.flux.site(e)).s();
    }
    let through = |edges: &[usize]| edges.iter().map(|&e| outgoing[e]).sum::<f64>();
    assert!((through(&channel.inlet) + HEIGHT).abs() < 1e-11);
    assert!((through(&channel.outlet) - HEIGHT).abs() < 1e-11);

    let (mut upstream, mut downstream) = (Vec::new(), Vec::new());
    for (c, p) in operators.face_centres.iter().zip(flow.pressure.iter()) {
        let [x, _] = coordinates(c);
        if x < 0.5 {
            upstream.push(p.s());
        } else if x > LENGTH - 0.5 {
            downstream.push(p.s());
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    assert!(mean(&upstream) > mean(&downstream));
    let enstrophy: f64 = flow
        .vorticity
        .iter()
        .zip(operators.dual_areas.iter())
        .map(|(w, a)| w.e12() * w.e12() * a.s())
        .sum();
    assert!(enstrophy > 0.0);
    assert!(
        flow.velocity
            .iter()
            .all(|v| v.c.iter().all(|c| c.is_finite()))
    );
}

#[test]
fn reversing_mesh_edges_changes_readings_and_preserves_physical_flow() {
    let channel = channel(12);
    let edges = channel.mesh.len(1);
    let (_, _, before) = flow(&channel, &unsigned(edges));
    let signs: Vec<f64> = (0..edges)
        .map(|e| if e % 3 == 0 { -1.0 } else { 1.0 })
        .collect();
    let (_, _, after) = flow(&channel, &signs);
    assert!(small(
        (0..edges).map(|e| after.flux.site(e).s() - before.flux.site(e).s() * signs[e]),
        1e-10
    ));
    assert!(small(
        after
            .pressure
            .iter()
            .zip(before.pressure.iter())
            .map(|(a, b)| (a - b).s()),
        1e-9
    ));
    assert!(small(
        after
            .vorticity
            .iter()
            .zip(before.vorticity.iter())
            .map(|(a, b)| (a - b).e12()),
        1e-9
    ));
    assert!(small(
        after
            .velocity
            .iter()
            .zip(before.velocity.iter())
            .flat_map(|(a, b)| (a - b).c),
        1e-9
    ));
}

#[test]
fn frames_draw() {
    let shown = super::shown();
    for t in [0.0, 1.9] {
        gax_numga_examples::app::assert_draws_with(super::draw, &shown, t);
    }
}
