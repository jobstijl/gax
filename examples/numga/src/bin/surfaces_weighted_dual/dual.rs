//! The hull and its polar dual about the unit sphere, in VGA3D: the poles of the faces, which
//! side of each edge a pole lies on, the signed lengths of the dual edges, and the folds of the
//! edges and the poles' places in their faces as linear maps of the points' weights.

use super::program::collect;
use super::surfaces_hull::beside;
use gax::field::{Field, SparseMap};
use gax_numga_examples::mesh::{Mesh, Scalar3, Vector3};

/// The pole of each face with respect to the unit sphere: the point whose inner product with
/// each of the face's vertices is one, the face's edges joined to the centre, summed, over the
/// volume of its vertices.
pub fn poles(points: &Field<Vector3>, faces: &[[usize; 3]]) -> Field<Vector3> {
    let pole = |&corners: &[usize; 3]| {
        let [a, b, c] = corners.map(|v| points.site(v));
        ((a ^ b) + (b ^ c) + (c ^ a)) * (a ^ b ^ c).inverse()
    };
    faces.iter().map(pole).collect()
}

/// For each face and each of its edges, the edge from corner `k` to the next: positive when the
/// face's pole, seen from the centre, lies on the face's side of the plane through the centre
/// and that edge, as the face's third corner does; all three positive when the pole lands
/// inside the face.
pub fn sides(
    points: &Field<Vector3>,
    faces: &[[usize; 3]],
    poles: &Field<Vector3>,
) -> [Field<Scalar3>; 3] {
    [0, 1, 2].map(|k| {
        let side = |(f, corners): (usize, &[usize; 3])| {
            let corner = |j: usize| points.site(corners[(k + j) % 3]);
            let wall = corner(0) ^ corner(1);
            (wall ^ poles.site(f)) * (wall ^ corner(2)).reverse()
        };
        faces.iter().enumerate().map(side).collect()
    })
}

/// The signed length of each dual edge across its edge, seen from the centre: how much further
/// the pole of the face on one side lies to that side of the plane through the centre and the
/// edge than the pole of the face on the other side; negative where the two poles have crossed
/// over.
pub fn dual_lengths(points: &Field<Vector3>, mesh: &Mesh) -> Field<Scalar3> {
    let poles = poles(points, &mesh.faces);
    let length = |(&edge, [first, second]): (&[usize; 2], [usize; 2])| {
        let wall = points.site(edge[0]) ^ points.site(edge[1]);
        let toward = wall ^ points.site(opposite(mesh.faces[second], edge));
        let across = poles.site(second) - poles.site(first);
        (wall ^ across) * toward.reverse() * (1.0 / toward.norm())
    };
    mesh.edges.iter().zip(beside(mesh)).map(length).collect()
}

/// The folds of the edges as a linear map of the points' weights `[E, V]`, for points along the
/// directions at the distances one over their weights: how far each edge bends outward, as on a
/// convex polyhedron, or folds inward where negative. The fold is the volume of the edge's ends,
/// the third corner of the face on one side and the far corner of the other, times all four
/// weights: a sum over the four corners, each weight times the volume of the other three
/// directions, with alternating signs. Each is scaled by the volume of the first face's
/// directions, positive on a mesh turning counter-clockwise from outside.
pub fn fold_map(directions: &Field<Vector3>, mesh: &Mesh) -> SparseMap<Scalar3> {
    // Each corner's sign, and the other three corners, in order.
    const SIGNS: [f64; 4] = [-1.0, 1.0, -1.0, 1.0];
    const OTHERS: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];
    let mut couplings = Vec::new();
    for (e, (&edge, faces)) in mesh.edges.iter().zip(beside(mesh)).enumerate() {
        let [first, second] = faces.map(|f| opposite(mesh.faces[f], edge));
        let corners = [edge[0], edge[1], first, second];
        let volume_without = |k: usize| {
            let [a, b, c] = OTHERS[k].map(|j| directions.site(corners[j]));
            a ^ b ^ c
        };
        let face = volume_without(3).reverse();
        for (k, sign) in SIGNS.into_iter().enumerate() {
            couplings.push((e, corners[k], volume_without(k) * face * sign));
        }
    }
    collect(couplings, (mesh.edges.len(), directions.len()))
}

/// Every pole's signed distances inside its face as a linear map of the points' weights,
/// `[3 F, V]`, for points along the directions at the distances one over their weights: for
/// face `f` and its edge from corner `k`, row `3 f + k`, the distance of the pole from the plane
/// through the centre and that edge, positive on the face's side. The pole is the sum of the
/// corners' weights times the face's reciprocal frame, the three vectors with inner product one
/// with their own corner's direction and zero with the other two, and the planes are fixed by
/// the directions, so the distances are linear in the weights.
pub fn inside_map(directions: &Field<Vector3>, faces: &[[usize; 3]]) -> SparseMap<Scalar3> {
    let mut couplings = Vec::new();
    for (f, corners) in faces.iter().enumerate() {
        let [a, b, c] = corners.map(|v| directions.site(v));
        let volume = (a ^ b ^ c).inverse();
        let frame = [(b ^ c) * volume, (c ^ a) * volume, (a ^ b) * volume];
        let at = |k: usize| [a, b, c][k % 3];
        for k in 0..3 {
            let wall = (at(k) ^ at(k + 1)).normalized();
            let third = wall ^ at(k + 2);
            for (j, &corner) in corners.iter().enumerate() {
                let distance = (wall ^ frame[j]) * third.reverse() * (1.0 / third.norm());
                couplings.push((3 * f + k, corner, distance));
            }
        }
    }
    collect(couplings, (3 * faces.len(), directions.len()))
}

/// Points along the given unit directions at the distances one over their weights.
pub fn weighted(directions: &Field<Vector3>, weights: &Field<Scalar3>) -> Field<Vector3> {
    directions.zip_with(weights, |d, w| d * (1.0 / w.s()))
}

/// The corner of a face across from one of its edges.
pub fn opposite(face: [usize; 3], [a, b]: [usize; 2]) -> usize {
    face.into_iter()
        .find(|&v| v != a && v != b)
        .expect("a triangle's third corner")
}
