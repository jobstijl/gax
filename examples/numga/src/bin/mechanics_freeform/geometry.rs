//! The bunny's surface and its volume quadrature (numga's `geometry.py`): the closed mesh, the
//! samples inside it by the parity of a ray's crossings, and the kernel centres covering it.

use gax::field::Field;
use gax::mesh::Topology;
use gax::vga2d;

use super::freeform::{Antivector, Vector, coordinates, point};
use super::numerics::Halton;

/// A closed surface: its triangles, wound outward, and its vertices.
pub struct Surface {
    pub faces: Vec<[usize; 3]>,
    pub topology: Topology<3>,
    pub vertices: Vec<Antivector>,
}

/// The cells across `y` and `z` of the grid that sorts the triangles for the ray tests.
const GRID: usize = 48;

/// The triangles' crossings with rays along `+x`, sorted by where they lie across the rays: a
/// grid over the `yz` plane, each cell listing the triangles whose shadow meets it.
pub struct Crossings {
    corners: Vec<[[f64; 3]; 3]>,
    low: [f64; 2],
    cell: [f64; 2],
    cells: Vec<Vec<usize>>,
}

impl Crossings {
    pub fn new(triangles: Vec<[[f64; 3]; 3]>) -> Crossings {
        let (mut low, mut high) = ([f64::MAX; 2], [f64::MIN; 2]);
        for t in &triangles {
            for c in t {
                for k in 0..2 {
                    low[k] = low[k].min(c[k + 1]);
                    high[k] = high[k].max(c[k + 1]);
                }
            }
        }
        let cell = [0, 1].map(|k| ((high[k] - low[k]) / GRID as f64).max(1e-12));
        let index = |v: f64, k: usize| (((v - low[k]) / cell[k]) as usize).min(GRID - 1);
        let mut cells = vec![Vec::new(); GRID * GRID];
        for (i, t) in triangles.iter().enumerate() {
            let span = |k: usize| {
                let values = t.map(|c| c[k + 1]);
                let lo = values.iter().copied().fold(f64::MAX, f64::min);
                let hi = values.iter().copied().fold(f64::MIN, f64::max);
                index(lo, k)..=index(hi, k)
            };
            for a in span(0) {
                for b in span(1) {
                    cells[a * GRID + b].push(i);
                }
            }
        }
        Crossings {
            corners: triangles,
            low,
            cell,
            cells,
        }
    }

    /// Whether a point lies inside: an odd number of triangles crossed by the ray from it along
    /// `+x`. Each triangle's shadow on the `yz` plane holds the ray's foot when the foot's
    /// barycentric weights, ratios of wedges in that plane, are all non-negative; the crossing
    /// lies ahead of the point when its `x` does.
    pub fn inside(&self, p: [f64; 3]) -> bool {
        let at = |c: [f64; 3]| vga2d::Vector::new(c[1], c[2]);
        let index = |k: usize| {
            let v = ((p[k + 1] - self.low[k]) / self.cell[k]).floor();
            (v >= 0.0 && v < GRID as f64).then_some(v as usize)
        };
        let (Some(a), Some(b)) = (index(0), index(1)) else {
            return false;
        };
        let foot = at(p);
        let crossed = self.cells[a * GRID + b]
            .iter()
            .filter(|&&i| {
                let [o, u, v] = self.corners[i];
                let (origin, first, second) = (at(o), at(u) - at(o), at(v) - at(o));
                let area = (first ^ second).e12();
                if area == 0.0 {
                    return false;
                }
                let offset = foot - origin;
                let s = (offset ^ second).e12() / area;
                let t = (first ^ offset).e12() / area;
                let x = o[0] + s * (u[0] - o[0]) + t * (v[0] - o[0]);
                s >= 0.0 && t >= 0.0 && s + t <= 1.0 && x > p[0]
            })
            .count();
        crossed % 2 == 1
    }
}

/// The volume a closed surface encloses: the cones its faces span from the origin, each the
/// join of its corners, summed, over six.
pub fn enclosed(surface: &Surface) -> f64 {
    surface
        .faces
        .iter()
        .map(|f| {
            let [a, b, c] = f.map(|v| surface.vertices[v]);
            (a & b & c).s()
        })
        .sum::<f64>()
        .abs()
        / 6.0
}

/// Equal-volume quadrature from low-discrepancy samples inside a closed surface: the samples
/// and each one's volume.
pub fn volume_samples(surface: &Surface, count: usize) -> (Vec<Antivector>, Vec<f64>) {
    let volume = enclosed(surface);
    let places: Vec<[f64; 3]> = surface.vertices.iter().map(|v| coordinates(*v)).collect();
    let (mut lower, mut upper) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in &places {
        for k in 0..3 {
            lower[k] = lower[k].min(p[k]);
            upper[k] = upper[k].max(p[k]);
        }
    }
    let crossings = Crossings::new(surface.faces.iter().map(|f| f.map(|v| places[v])).collect());
    let box_volume: f64 = (0..3).map(|k| upper[k] - lower[k]).product();
    let occupancy = volume / box_volume;
    // A power of two of candidates, as numga draws its Sobol points.
    let candidates = (1.2 * count as f64 / occupancy).ceil() as usize;
    let candidates = candidates.next_power_of_two();
    let mut sequence = Halton::new(0);
    let mut samples = Vec::with_capacity(count);
    // Bounded: each round adds about `1.2 count` samples.
    for _ in 0..64 {
        if samples.len() >= count {
            break;
        }
        for u in sequence.take(candidates) {
            let p: [f64; 3] = core::array::from_fn(|k| lower[k] + u[k] * (upper[k] - lower[k]));
            if crossings.inside(p) {
                samples.push(point(p[0], p[1], p[2]));
            }
        }
    }
    samples.truncate(count);
    assert_eq!(samples.len(), count, "too few samples inside the surface");
    (samples, vec![volume / count as f64; count])
}

/// Farthest-point coverage of the sampled volume, thin extremities included: from the sample
/// nearest the mean, each next one the farthest from those taken.
pub fn center_indices(points: &[Antivector], count: usize) -> Vec<usize> {
    let mean = points.iter().copied().sum::<Antivector>() * (1.0 / points.len() as f64);
    let distance = |a: Antivector, b: Antivector| (a - b).norm_squared();
    let first = (0..points.len())
        .min_by(|&a, &b| distance(points[a], mean).total_cmp(&distance(points[b], mean)))
        .expect("samples");
    let mut indices = vec![first];
    let mut nearest = vec![f64::INFINITY; points.len()];
    while indices.len() < count {
        let last = points[indices[indices.len() - 1]];
        for (n, p) in nearest.iter_mut().zip(points) {
            *n = n.min(distance(*p, last));
        }
        let farthest = (0..points.len())
            .max_by(|&a, &b| nearest[a].total_cmp(&nearest[b]))
            .expect("samples");
        indices.push(farthest);
    }
    indices
}

/// The y-up bunny, `height` tall along `y` with its lowest point on `y = 0`, then stood up (its
/// `y` turned onto `z`), wound outward: its surface, volume samples and their volumes, and
/// kernel centres.
pub fn load(
    height: f64,
    samples: usize,
    centers: usize,
) -> (Surface, Vec<Antivector>, Vec<f64>, Vec<Antivector>) {
    const BUNNY: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/bunny.obj"));
    let model = gax_numga_examples::mesh::Mesh::from_obj(BUNNY, height);
    let faces = model.closed().faces;
    let heights: Vec<f64> = model.vertices.iter().map(|v| v.e2()).collect();
    let extent = heights.iter().copied().fold(f64::MIN, f64::max)
        - heights.iter().copied().fold(f64::MAX, f64::min);
    let lift = Vector::new(0.0, height / 2.0, 0.0);
    let placed: Vec<Antivector> = model
        .vertices
        .iter()
        .map(|v| (v * (height / extent) + lift).hodge())
        .collect();
    let surface = Surface {
        topology: Topology::new(placed.len(), &faces).expect("the bunny"),
        faces,
        vertices: placed,
    };
    let (quadrature, volumes) = volume_samples(&surface, samples);
    let chosen = center_indices(&quadrature, centers);
    let upright = gax::vga3d::Bivector::new(-core::f64::consts::FRAC_PI_4, 0.0, 0.0).exp();
    let up = |points: &[Antivector]| -> Vec<Antivector> {
        points.iter().map(|p| upright >> *p).collect()
    };
    let centres: Vec<Antivector> = chosen.iter().map(|&i| quadrature[i]).collect();
    (
        Surface {
            vertices: up(&surface.vertices),
            ..surface
        },
        up(&quadrature),
        volumes,
        up(&centres),
    )
}

/// The vertices as a field.
pub fn field(points: &[Antivector]) -> Field<Antivector> {
    Field::new(points.to_vec())
}
