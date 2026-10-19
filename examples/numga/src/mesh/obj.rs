//! Meshes read from Wavefront OBJ text: numga's `Mesh.read`.

use super::{Mesh, Vector3};
use gax::field::Field;

impl Mesh {
    /// A mesh from Wavefront OBJ text, its vertices and triangles only, centred on its bounding
    /// box and scaled so that the box's longest side is `size`. A face's corners may carry
    /// texture and normal indices (`v/t/n`); only the vertex index is read.
    ///
    /// # Panics
    /// If a vertex line has fewer than three numbers, or a face line fewer than three vertex
    /// indices.
    #[must_use]
    pub fn from_obj(text: &str, size: f64) -> Self {
        let mut places: Vec<Vector3> = Vec::new();
        let mut faces: Vec<[usize; 3]> = Vec::new();
        for line in text.lines() {
            let mut words = line.split_whitespace();
            match words.next() {
                Some("v") => {
                    let mut number = || -> f64 {
                        let word = words.next().expect("a vertex has three coordinates");
                        word.parse().expect("a coordinate is a number")
                    };
                    places.push(Vector3::new(number(), number(), number()));
                }
                Some("f") => faces.push(core::array::from_fn(|_| {
                    let corner = words.next().expect("a face has three corners");
                    let index = corner.split('/').next().unwrap_or(corner);
                    index.parse::<usize>().expect("a vertex index") - 1
                })),
                _ => {}
            }
        }
        let (low, high) = places.iter().fold(
            ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]),
            |(low, high), p| {
                let c = p.c;
                (
                    core::array::from_fn(|k| low[k].min(c[k])),
                    core::array::from_fn(|k| high[k].max(c[k])),
                )
            },
        );
        let [low, high] = [low, high].map(|[x, y, z]| Vector3::new(x, y, z));
        let middle = (low + high) * 0.5;
        let extent = (high - low).c.into_iter().fold(0.0, f64::max);
        let vertices: Field<Vector3> = places
            .iter()
            .map(|p| (*p - middle) * (size / extent))
            .collect();
        Mesh::new(vertices, faces)
    }
}
