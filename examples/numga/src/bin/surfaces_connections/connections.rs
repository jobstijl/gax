//! The math: parallel transport and its holonomy, connections on a triangle mesh with their
//! covariant derivatives and Laplacians, vector heat, trivial connections with chosen
//! singularities, and stripes along their fields. No drawing.
//!
//! Positions and tangent vectors are antivectors: intrinsic bivectors whose coefficients are the
//! coordinates (numga's `Point-`). A unit normal is the patch of a tangent plane, the join of two
//! edges, an intrinsic vector (`Vector-`); its Hodge dual is the plane tangent vectors turn in.

use crate::simplicial::{
    boundary, circulation, corners as local_corners, edges as edge_list, incidence, spanning_tree,
};
use gax::field::{Field, SparseMap};
use gax::mesh::Topology;
use gax::vga3d;

pub type Scalar<S = ()> = vga3d::Scalar<S>;
pub type Bivector<S = ()> = vga3d::Bivector<S>;
pub type Rotor<S = ()> = vga3d::Rotor<S>;
/// A position or a tangent vector (numga's `Point-`).
pub type Point<S = ()> = vga3d::intrinsic::Bivector<S>;
/// A unit normal, a tangent plane's patch (numga's `Vector-`).
pub type Normal<S = ()> = vga3d::intrinsic::Vector<S>;
/// A map on tangent vectors, the cell of a covariant derivative or a connection Laplacian.
pub type PointMap = Point<(Point,)>;

/// The point at `(x, y, z)`.
pub fn point(x: f64, y: f64, z: f64) -> Point {
    vga3d::Vector::new(x, y, z).hodge()
}

/// A point's coordinates.
pub fn coordinates(p: Point) -> [f64; 3] {
    [
        vga3d::Vector::new(1.0, 0.0, 0.0),
        vga3d::Vector::new(0.0, 1.0, 0.0),
        vga3d::Vector::new(0.0, 0.0, 1.0),
    ]
    .map(|s| (s & p).s())
}

/// numga's `a.dot(b)` on points, `⟨a ~b⟩`: their Euclidean inner product.
pub fn dot(a: Point, b: Point) -> f64 {
    a.metric(b).s()
}

/// The plane a tangent vector turns in at a normal: its Hodge dual.
pub fn plane(n: Normal) -> Bivector {
    n.hodge()
}

/// The edges facing each corner of each face, counter-clockwise.
fn sides(mesh: &Topology<3>, vertices: &Field<Point>) -> Vec<Field<Point>> {
    boundary(mesh, 2).read_all(&(&mesh.d::<Scalar>(0) * vertices))
}

/// The unit normal at each vertex: the sum of its faces' normals, each the join of two of the
/// face's edges, twice the face's area large, normalized.
pub fn vertex_normals(mesh: &Topology<3>, vertices: &Field<Point>) -> Field<Normal> {
    let s = sides(mesh, vertices);
    let corners = incidence(mesh, 2, 0);
    let mut sum = vec![Normal::new(0.0, 0.0, 0.0); mesh.len(0)];
    for f in 0..mesh.len(2) {
        let n = s[0].site(f) & s[1].site(f);
        for &v in &corners.sites[f] {
            sum[v] += n;
        }
    }
    sum.into_iter().map(|n| n.normalized()).collect()
}

/// Each vertex's area: a third of each face's at it.
pub fn vertex_areas(mesh: &Topology<3>, vertices: &Field<Point>) -> Field<Scalar> {
    let s = sides(mesh, vertices);
    let areas = s[0].zip_with(&s[1], |a, b| Scalar::new((a & b).norm() * 0.5));
    &mesh.means::<Scalar>(2).reverse() * &areas
}

/// Each edge's weight: half the cotangents of the angles facing it, each face's share weighted
/// by `face_weights`. The cotangent at a corner is one side's slice read along the other, over the
/// size of the plane they span; head to tail, the corner's own sides point the other way.
pub fn cotangent_weights(
    mesh: &Topology<3>,
    vertices: &Field<Point>,
    face_weights: &[f64],
) -> Field<Scalar> {
    let s = sides(mesh, vertices);
    let ends = local_corners(2, 1);
    let facing = incidence(mesh, 2, 1);
    let mut weights = vec![0.0; mesh.len(1)];
    for f in 0..mesh.len(2) {
        for (k, end) in ends.iter().enumerate() {
            let (after, before) = (s[end[0]].site(f), s[end[1]].site(f));
            let cot = -dot(after, before) / after.commutator(before).norm();
            weights[facing.sites[f][k]] += face_weights[f] * cot / 2.0;
        }
    }
    weights.into_iter().map(Scalar::new).collect()
}

/// The shortest turn carrying the unit normal `from` onto `to`: the square root of their product.
pub fn shortest(to: Normal, from: Normal) -> Rotor {
    (to * from).sqrt().into_inner()
}

/// A tangent vector carried along a sequence of unit normals, at each of them.
pub fn transported(normals: &[Normal], start: Point) -> Vec<Point> {
    let mut out = vec![start];
    for pair in normals.windows(2) {
        let next = shortest(pair[1], pair[0]) >> out[out.len() - 1];
        out.push(next);
    }
    out
}

/// The holonomy of transport around a closed loop of unit normals: the turn of the product of
/// the steps' rotors, each later one multiplying from the left, read from the rotor's square so
/// that whole turns read as nothing: the plane of its logarithm, read across the first tangent
/// plane, an angle in `(−π, π]`.
pub fn loop_turn(normals: &[Normal]) -> f64 {
    let n = normals.len();
    let around = (0..n)
        .map(|k| shortest(normals[(k + 1) % n], normals[k]))
        .fold(Rotor::new(1.0, 0.0, 0.0, 0.0), |acc, step| step * acc);
    ((around * around).unit().log() | plane(normals[0])).s()
}

/// The rotors from each end of each edge to its head: the surface's own transport from the tail
/// followed by the edge's extra turn in the head's tangent plane, and none at the head.
pub fn adjusted_transport(
    mesh: &Topology<3>,
    normals: &Field<Normal>,
    angles: &[f64],
) -> Vec<Rotor> {
    edge_list(mesh)
        .into_iter()
        .zip(angles)
        .map(|([tail, head], &angle)| {
            let extra = (plane(normals.site(head)) * (-angle / 2.0))
                .exp()
                .into_inner();
            extra * shortest(normals.site(head), normals.site(tail))
        })
        .collect()
}

/// A connection's covariant derivative, from tangent fields on the vertices to the edges (each
/// edge's head vector less its tail's carried to the head), and its Laplacian with the edges
/// weighted.
pub fn connection(
    mesh: &Topology<3>,
    forward: &[Rotor],
    weights: &Field<Scalar>,
) -> (SparseMap<PointMap>, SparseMap<PointMap>) {
    let ends = boundary(mesh, 1);
    let identity = Point::slot();
    let covariant = ends.assemble(|e, k| {
        if k == 0 {
            forward[e] >> identity
        } else {
            identity
        }
    });
    let weighted = SparseMap::from_diagonal(&weights.map(|w| identity * w.s()));
    let laplacian = covariant.adjoint().compose(&weighted.compose(&covariant));
    (covariant, laplacian)
}

/// A direction's part along a unit normal: the normal's plane times the direction's pairing with
/// the normal's patch, a map on directions.
pub fn along_normal(n: Normal) -> PointMap {
    plane(n) * (n & Point::slot())
}

/// The `count` smoothest tangent fields of a connection Laplacian against the vertex areas, with
/// their energies per unit area: the normal bundle lifted out of the low spectrum by
/// `stiffness`.
pub fn low_modes(
    laplacian: &SparseMap<PointMap>,
    normals: &Field<Normal>,
    areas: &Field<Scalar>,
    stiffness: f64,
    count: usize,
) -> (Vec<f64>, Vec<Field<Point>>) {
    let penalty = SparseMap::from_diagonal(&normals.map(|n| along_normal(n) * stiffness));
    let metric = SparseMap::from_diagonal(&areas.map(|a| Point::slot() * a.s()));
    (laplacian + &penalty).eigh(&metric, count)
}

/// Backward-Euler vector heat steps of `step` from `source`: the diffused field at each step and
/// its unit field.
pub fn heat_transport(
    laplacian: &SparseMap<PointMap>,
    areas: &Field<Scalar>,
    source: &Field<Point>,
    step: f64,
    steps: usize,
) -> Vec<(Field<Point>, Field<Point>)> {
    let mass = SparseMap::from_diagonal(&areas.map(|a| Point::slot() * a.s()));
    let operator = &mass + &(laplacian * step);
    let mut field = source.clone();
    let mut out = Vec::with_capacity(steps);
    for _ in 0..steps {
        field = operator.solve(&mass.of(&field));
        let unit = field.map(|p| p.normalized());
        out.push((field.clone(), unit));
    }
    out
}

/// The holonomy of each face, transported around its corners' normals: the curvature inside it.
pub fn face_curvatures(mesh: &Topology<3>, normals: &Field<Normal>) -> Vec<f64> {
    (0..mesh.len(2))
        .map(|f| {
            loop_turn(
                &mesh
                    .simplex(2, f)
                    .iter()
                    .map(|&v| normals.site(v))
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

/// A trivial connection: the smallest extra turns on the edges that make every face turn by two
/// pi times its index and every generator (a closed loop of vertices) by nothing; with the
/// circulation of edge turns around the faces and generators, and its target.
///
/// numga solves the least-squares system with `lstsq`. Here, to the `1e-12` its checks ask: the
/// circulation `C` has one dependency on a closed surface (the faces' rows sum to nothing), so
/// the target's part off it is the faces' mean, which no turns can meet; the rest is met exactly
/// by `Cᵀ y` with `C Cᵀ y` the projected target, by conjugate gradients, and `Cᵀ y` is the
/// least-norm least-squares solution.
pub fn trivial_connection(
    mesh: &Topology<3>,
    normals: &Field<Normal>,
    indices: &[f64],
    generators: &[Vec<usize>],
) -> (Vec<f64>, SparseMap<Scalar<(Scalar,)>>, Vec<f64>) {
    let faces = mesh.len(2);
    let curvatures = face_curvatures(mesh, normals);
    let mut target: Vec<f64> = (0..faces)
        .map(|f| core::f64::consts::TAU * indices[f] - curvatures[f])
        .collect();
    // Around each generator the extra turns undo the surface's own holonomy.
    for g in generators {
        let along: Vec<Normal> = g.iter().map(|&v| normals.site(v)).collect();
        target.push(-loop_turn(&along));
    }
    let circulation = &circulation::<Scalar, 3>(mesh, generators) * Scalar::slot();
    let mean = target[..faces].iter().sum::<f64>() / faces as f64;
    let met: Field<Scalar> = (0..target.len())
        .map(|r| {
            Scalar::new(if r < faces {
                target[r] - mean
            } else {
                target[r]
            })
        })
        .collect();
    let transpose = circulation.transpose();
    let normal = circulation.compose(&transpose);
    let solved = normal.cg(&met, 1e-15, 40 * target.len());
    let angles = transpose.of(&solved.x).iter().map(|s| s.s()).collect();
    (angles, circulation, target)
}

/// The circulation of edge turns minus its target: what a trivial connection leaves.
pub fn residual(
    circulation: &SparseMap<Scalar<(Scalar,)>>,
    angles: &[f64],
    target: &[f64],
) -> Vec<f64> {
    let turns: Field<Scalar> = angles.iter().map(|&a| Scalar::new(a)).collect();
    circulation
        .of(&turns)
        .iter()
        .zip(target)
        .map(|(c, t)| c.s() - t)
        .collect()
}

/// A unit tangent vector at every vertex carried out from the first by the transport of the given
/// turns along a breadth-first tree: each tree edge asks its far vertex to be its near one
/// carried across it, and the first vertex to point across its first edge. Where the turns make a
/// trivial connection whose faces turn by quarter turns, this is a cross field.
pub fn carried_out(
    mesh: &Topology<3>,
    vertices: &Field<Point>,
    normals: &Field<Normal>,
    angles: &[f64],
) -> Field<Point> {
    let (children, tree) = spanning_tree(mesh);
    let forward = adjusted_transport(mesh, normals, angles);
    let (covariant, _) = connection(
        mesh,
        &forward,
        &Field::from_fn(mesh.len(1), |_| Scalar::new(0.0)),
    );
    let rows: Vec<(usize, usize, PointMap)> = covariant.couplings().collect();
    let mut triples: Vec<(usize, usize, PointMap)> = Vec::new();
    for (&child, &edge) in children.iter().zip(&tree) {
        triples.extend(
            rows.iter()
                .filter(|r| r.0 == edge)
                .map(|&(_, c, cell)| (child, c, cell)),
        );
    }
    triples.push((0, 0, Point::slot()));
    let system = SparseMap::from_triples(triples, (mesh.len(0), mesh.len(0)));
    // The direction across the first edge in the first tangent plane: the plane through the
    // normal and the edge, as an antivector.
    let [a, b] = edge_list(mesh)[0];
    let edge = (vertices.site(b) - vertices.site(a)).undual();
    let seed = (normals.site(0) ^ edge).normalized();
    let rhs = Field::from_fn(mesh.len(0), |v| if v == 0 { seed } else { seed * 0.0 });
    system.solve_direct(&rhs)
}

/// The stripes across a cross field's two lines: the lines at every vertex, the unit stripe
/// field of each, and the target across every edge for each line at its head.
///
/// Carried across an edge, each of the tail's lines lands on one of the head's lines, along it
/// or against it. A stripe phase along a line reads backwards along the line reversed, so a
/// stripe field carried onto a line it lands against is mirrored. The two families are one
/// field over the cross's two lines at every vertex (line `a` of vertex `v` the site `2v + a`),
/// coupled across the edges where the lines trade places.
pub struct Cross {
    pub lines: [Field<Point>; 2],
    pub stripes: [Field<Point>; 2],
    pub advance: [Vec<f64>; 2],
}

#[allow(clippy::too_many_arguments)]
pub fn cross_stripes(
    mesh: &Topology<3>,
    vertices: &Field<Point>,
    normals: &Field<Normal>,
    areas: &Field<Scalar>,
    angles: &[f64],
    cross: &Field<Point>,
    regular: &[f64],
    frequency: f64,
    stiffness: f64,
) -> Cross {
    let (n, e) = (mesh.len(0), mesh.len(1));
    let forward = adjusted_transport(mesh, normals, angles);
    // The cross's second line: the first turned a quarter turn, its commutator with the plane.
    let lines = [
        cross.clone(),
        cross.zip_with(normals, |c, n| c.commutator(plane(n))),
    ];
    let edges = edge_list(mesh);
    let steps = &mesh.d::<Scalar>(0) * vertices;
    let cot = cotangent_weights(mesh, vertices, regular);
    let mut triples = Vec::new();
    let mut advance = [vec![0.0; e], vec![0.0; e]];
    for (k, &[tail, head]) in edges.iter().enumerate() {
        let r = forward[k];
        let at_tail = [lines[0].site(tail), lines[1].site(tail)];
        let ahead = [lines[0].site(head), lines[1].site(head)];
        // One where the tail's line b lands along the head's line a, minus one against it.
        let matched: [[f64; 2]; 2] =
            core::array::from_fn(|a| core::array::from_fn(|b| dot(ahead[a], r >> at_tail[b])));
        let moved = r >> Point::slot();
        for a in 0..2 {
            let behind = at_tail[0] * matched[a][0] + at_tail[1] * matched[a][1];
            advance[a][k] = dot((ahead[a] + behind) * 0.5, steps.site(k)) * frequency;
            let turn = (plane(normals.site(head)) * (-advance[a][k] / 2.0))
                .exp()
                .into_inner();
            triples.push((2 * k + a, 2 * head + a, turn.reverse() >> Point::slot()));
            // Carried, then mirrored onto the head's line where it lands against it.
            // The head's line times the carried vector's part along it, `⟨a ~(R x)⟩`.
            let reading = ahead[a] * ahead[a].scalar_product(moved.reverse());
            for (b, &m) in matched[a].iter().enumerate() {
                let landed = moved * (m * m) + reading * (m - m * m);
                triples.push((2 * k + a, 2 * tail + b, -landed));
            }
        }
    }
    let covariant = SparseMap::from_triples(triples, (2 * e, 2 * n));
    let weights = Field::from_fn(2 * e, |s| cot.site(s / 2));
    let weighted = SparseMap::from_diagonal(&weights.map(|w| Point::slot() * w.s()));
    let laplacian = covariant.adjoint().compose(&weighted.compose(&covariant));
    let doubled_normals = Field::from_fn(2 * n, |s| normals.site(s / 2));
    let doubled_areas = Field::from_fn(2 * n, |s| areas.site(s / 2));
    let (_, fields) = low_modes(&laplacian, &doubled_normals, &doubled_areas, stiffness, 1);
    let stripes =
        core::array::from_fn(|a| Field::from_fn(n, |v| fields[0].site(2 * v + a).normalized()));
    Cross {
        lines,
        stripes,
        advance,
    }
}

/// Closed latitude circles on the unit sphere at the colatitudes, over `steps` segments, as their
/// unit normals `[steps + 1][rings]`, with the unit meridian tangent vector at each start.
pub fn latitude_rings(colatitudes: &[f64], steps: usize) -> (Vec<Vec<Normal>>, Vec<Point>) {
    let up = vga3d::Bivector::new(0.0, 0.0, 1.0).hodge();
    let tilts: Vec<Rotor> = colatitudes
        .iter()
        .map(|&c| {
            (Bivector::new(0.0, 1.0, 0.0) * (-c / 2.0))
                .exp()
                .into_inner()
        })
        .collect();
    let normals = (0..=steps)
        .map(|k| {
            let longitude = core::f64::consts::TAU * k as f64 / steps as f64;
            let spin = (Bivector::new(0.0, 0.0, 1.0) * (-longitude / 2.0))
                .exp()
                .into_inner();
            tilts.iter().map(|&t| spin >> (t >> up)).collect()
        })
        .collect();
    let starts = tilts.iter().map(|&t| t >> point(1.0, 0.0, 0.0)).collect();
    (normals, starts)
}
