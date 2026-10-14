//! Points of a flat mesh's plane on the mesh: the face that holds a point, with the point's
//! barycentric weights there (the areas of the corners' opposite triangles, each the `e12` part
//! of a wedge, over the face's).

use gax_numga_examples::mesh::{Mesh, Vector3};

/// A point inside a face: the face, and the weights of its corners.
#[derive(Clone, Copy, Debug)]
pub struct Located {
    pub face: usize,
    pub weights: [f64; 3],
}

/// The signed area of the triangle `a b c` in the plane, twice over: positive counter-clockwise.
fn turning(a: Vector3, b: Vector3, c: Vector3) -> f64 {
    ((b - a) ^ (c - a)).e12()
}

/// The face of a flat counter-clockwise mesh that holds `point`, if any, with its weights.
pub fn locate(mesh: &Mesh, point: Vector3) -> Option<Located> {
    // A point on a shared edge belongs to both faces; the first found holds it.
    let slack = -1e-12;
    mesh.faces.iter().enumerate().find_map(|(face, &corners)| {
        let [a, b, c] = corners.map(|v| mesh.vertices.site(v));
        let whole = turning(a, b, c);
        let weights = [
            turning(point, b, c),
            turning(a, point, c),
            turning(a, b, point),
        ]
        .map(|w| w / whole);
        weights
            .iter()
            .all(|&w| w >= slack)
            .then_some(Located { face, weights })
    })
}
