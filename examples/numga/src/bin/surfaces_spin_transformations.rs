//! numga's `surfaces/spin_transformations`: Keenan Crane's spin transformations
//! (<https://www.cs.cmu.edu/~kmcrane/Projects/SpinTransformations/paper.pdf>) in the geometric
//! algebra of space, VGA3D. A spin transformation changes a surface's mean curvature by a
//! prescribed amount `ρ` and keeps every angle: a quaternion at each vertex, an element of the even
//! subalgebra, turns and scales the edges around it, `~q e q`. The quaternions are the field of
//! least energy of the Dirac operator less `ρ`, and the new vertices are those whose edges best
//! match the turned ones.
//!
//! The operators are field maps over the mesh ([`gax::field::SparseMap`]) whose cells are
//! multivectors, as the paper's quaternionic matrices. The Dirac operator takes each corner of a
//! face by the edge it faces (a vector) over minus twice the face's area: the geometric
//! derivative divided by the face's plane. `ρ` times the pseudoscalar is a trivector, so the
//! operator less `ρ` has odd cells, and its energy `~A M A` even ones. With the quaternion
//! left open (`Rotor::slot()`), the energy and the vertex areas become maps on quaternion fields,
//! and one sparse generalized eigensolve finds the field of least energy. Each edge is turned
//! by the field along it, by Simpson's rule, and the vertices follow from one least-squares
//! solve with the cotangent Laplacian.
//!
//! The animation shows a sphere given a change in mean curvature that grows along `y`, and the
//! surface it becomes, checkered by where each face sat on the sphere; a cube rounded by
//! conformal curvature flow, iterated spin transformations that shave off each face's departure
//! from the mean curvature; and Dirac spheres: with `ρ` constant, the fields of zero energy are
//! the Dirac operator's eigenfields, and the surfaces they spin the sphere into keep its angles
//! (eigenvalues the integers but `-1`). The outside of each surface is orange, its inside blue.

#[path = "../shared/surfaces_sphere.rs"]
mod surfaces_sphere;
#[path = "../shared/surfaces_spin.rs"]
mod surfaces_spin;

use gax_numga_examples::mesh::Mesh;
use gax_numga_examples::{
    Anim, Camera, Canvas, Lens, Light, Rect, backdrop, caption, colormap, label_under, reach3, run,
};
use std::sync::OnceLock;
use surfaces_spin::{Hit, checker_light, render};

mod spin {
    pub use super::surfaces_sphere::icosphere;
    pub use super::surfaces_spin::cube;
    use gax::field::{Field, SparseMap};
    use gax::vga3d::{Odd, Rotor};
    use gax_numga_examples::mesh::{Mesh, Scalar3, Vector3};

    /// A quaternion: an even multivector, a turn and a scale.
    pub type Quaternion = Rotor<(), f64>;
    /// A map on quaternions: the cells of the energy and of the area metric.
    pub type QuaternionMap = Rotor<(Rotor,), f64>;

    /// The Dirac operator less `ρ` (a number on each face), `[F, V]` with odd cells: each face
    /// takes each corner by the edge it faces over minus twice its area, less `ρ` times the
    /// pseudoscalar of the mean of its corners. `ρ`'s pseudoscalar is its dual, gax's metric-free
    /// one, which agrees with numga's Hodge dual where every basis vector squares to `+1`.
    pub fn dirac_less(mesh: &Mesh, rho: &Field<Scalar3>) -> SparseMap<Odd<(), f64>> {
        let inverse_areas = mesh.triangle_areas().map(|a| a.inverse());
        let corners = mesh.at_corners(&mesh.triangle_edges());
        let dirac = &SparseMap::from_diagonal(&inverse_areas) * &corners;
        let curvature = &SparseMap::from_diagonal(&rho.map(|r| r.dual())) * &mesh.face_means();
        &dirac.map_cells(|e| (e * -0.5).cast::<Odd>())
            + &curvature.map_cells(|p| (-p).cast::<Odd>())
    }

    /// The energy of quaternion fields, `~A M A` with `A` the Dirac operator less `ρ` and `M` the
    /// face areas, as a map on them: its cells even multivectors with the quaternion left open.
    pub fn energy(mesh: &Mesh, rho: &Field<Scalar3>) -> SparseMap<QuaternionMap> {
        let a = dirac_less(mesh, rho);
        let areas = SparseMap::from_diagonal(&mesh.triangle_areas());
        &(&(&a.reverse() * &areas) * &a) * Quaternion::slot()
    }

    /// The vertex areas, as a map on quaternion fields: the metric of the eigenproblem.
    pub fn area_metric(mesh: &Mesh) -> SparseMap<QuaternionMap> {
        &SparseMap::from_diagonal(&mesh.vertex_areas()) * Quaternion::slot()
    }

    /// Each edge turned and scaled by the field along it, `~q e q`, integrated by Simpson's
    /// rule: the turns at its two ends, averaged as maps, and twice the turn of the mean
    /// quaternion at its middle.
    pub fn spun_edges(mesh: &Mesh, q: &Field<Quaternion>) -> Field<Vector3> {
        let means = mesh.edge_means();
        let ends = &means * &q.map(|q| q << Vector3::slot());
        let middles = &means * q;
        let edges = mesh.edge_vectors();
        Field::from_fn(edges.len(), |e| {
            let edge = edges.site(e);
            (ends.site(e).of(edge) + (middles.site(e) << edge) * 2.0) * (1.0 / 3.0)
        })
    }

    /// The vertices whose edges best match the given ones under the cotangent weights: the
    /// least-squares solve of the Laplacian, its translation left at zero (centred).
    pub fn matching(mesh: &Mesh, edges: &Field<Vector3>) -> Field<Vector3> {
        let d0 = mesh.d0();
        let weights = SparseMap::from_diagonal(&mesh.edge_ratio());
        let divergence = &(&d0.reverse() * &weights) * edges;
        (&mesh.laplacian() * Vector3::slot()).lstsq(&divergence)
    }

    /// The mesh after the spin transformation that changes each face's mean curvature by `ρ`:
    /// the field of least energy per unit of vertex area, divided by its area-weighted mean,
    /// spinning the edges.
    pub fn spin_transform_deform(mesh: &Mesh, rho: &Field<Scalar3>) -> Mesh {
        let (_, modes) = energy(mesh, rho).eigh(&area_metric(mesh), 1);
        let areas = mesh.vertex_areas();
        let weighted = areas.zip_with(&modes[0], |a, q| a * q).sum();
        let mean = weighted * (1.0 / areas.sum().s());
        let q = modes[0].map(|q| q / mean);
        mesh.with_vertices(matching(mesh, &spun_edges(mesh, &q)))
    }

    /// The energies of the `count` least-energy fields for the constant `ρ` of the eigenvalue,
    /// and the surfaces they spin the mesh into: the Dirac spheres. With `ρ` constant, the fields
    /// of zero energy are the eigenfields of the Dirac operator itself; on the unit sphere its
    /// eigenvalues are the integers but -1, each with multiplicity its value plus one. The
    /// fields are taken as they come, at unit size per unit of vertex area, their mean zero.
    pub fn dirac_spheres(mesh: &Mesh, eigenvalue: i32, count: usize) -> (Vec<f64>, Vec<Mesh>) {
        let rho = Field::from_fn(mesh.faces.len(), |_| Scalar3::new(f64::from(eigenvalue)));
        let (energies, fields) = energy(mesh, &rho).eigh(&area_metric(mesh), count);
        let spheres = fields
            .iter()
            .map(|q| mesh.with_vertices(matching(mesh, &spun_edges(mesh, q))))
            .collect();
        (energies, spheres)
    }

    /// The pointwise signed mean curvature on each face. The cotangent Laplacian of the
    /// positions is the area-integrated mean curvature normal, `L x = 2 A H n` at each vertex;
    /// divided by twice the vertex area it is the pointwise `H`, averaged onto the faces.
    pub fn mean_curvature(mesh: &Mesh) -> Field<Scalar3> {
        let normal_curvature = &mesh.laplacian() * &mesh.vertices;
        let integrated = normal_curvature.zip_with(&mesh.vertex_normals(), |h, n| h | n);
        let pointwise = integrated.zip_with(&mesh.vertex_areas(), |h, a| h * (0.5 / a.s()));
        &mesh.face_means() * &pointwise
    }

    /// The mesh centred and scaled to unit mean distance from its centre: the translation and
    /// scale the flow leaves open.
    pub fn recenter(mesh: &Mesh) -> Mesh {
        let centre = mesh.vertices.mean();
        let centred = mesh.vertices.map(|v| v - centre);
        let size = centred.iter().map(|v| v.norm()).sum::<f64>() / centred.len() as f64;
        mesh.with_vertices(centred.scaled(1.0 / size))
    }

    /// Conformal curvature flow by iterated spin transformations: the mesh before each step and
    /// after the last. Each step asks for `ρ = -rate (H - H̄)`, a change in mean curvature that
    /// shaves off each face's departure from the area-weighted mean, so the surface relaxes
    /// towards constant mean curvature while every step keeps the angles. A cube's corners stay
    /// cone points: a conformal map keeps the angle deficit there.
    pub fn conformal_smooth(mesh: &Mesh, iterations: usize, rate: f64) -> Vec<Mesh> {
        let mut frames = vec![recenter(mesh)];
        for _ in 0..iterations {
            let mesh = frames.last().expect("the mesh so far");
            let (h, areas) = (mean_curvature(mesh), mesh.triangle_areas());
            let mean = h.zip_with(&areas, |h, a| h * a).sum().s() / areas.sum().s();
            let rho = h.map(|h| (h - Scalar3::new(mean)) * -rate);
            frames.push(recenter(&spin_transform_deform(mesh, &rho)));
        }
        frames
    }
}

mod scenes {
    use super::spin::*;
    use gax::field::Field;
    use gax_numga_examples::mesh::{Mesh, Scalar3, Vector3};
    use gax_numga_examples::par_map;

    /// How often the sphere's icosahedron is split, and how strongly its mean curvature
    /// changes along y.
    pub const LEVELS: usize = 3;
    pub const DIPOLE: f64 = 0.5;
    /// The cube's squares per side edge, the share of each face's curvature departure each flow
    /// step removes, and the number of steps.
    pub const DIVISIONS: usize = 8;
    pub const RATE: f64 = 0.25;
    pub const STEPS: usize = 30;
    /// The finer sphere the Dirac spheres are spun from (one split coarser than numga's, to
    /// start in seconds; the checks use numga's), the eigenvalues shown, and the states shown of
    /// each.
    pub const DIRAC_LEVELS: usize = 3;
    pub const EIGENVALUES: [i32; 5] = [1, 2, 3, -2, -3];
    pub const STATES: usize = 3;

    /// A sphere, the change in mean curvature asked of each face, and the surface that has it.
    pub struct Dipole {
        pub sphere: Mesh,
        pub rho: Field<Scalar3>,
        pub deformed: Mesh,
    }

    /// The change in mean curvature on each face of a mesh, growing along y.
    pub fn dipole_rho(mesh: &Mesh, strength: f64) -> Field<Scalar3> {
        let y = Vector3::new(0.0, 1.0, 0.0);
        mesh.face_centers().map(|c| (c | y) * strength)
    }

    /// The sphere given the dipole, and the surface the spin transformation makes of it.
    pub fn dipole() -> Dipole {
        let sphere = icosphere(LEVELS);
        let rho = dipole_rho(&sphere, DIPOLE);
        let deformed = spin_transform_deform(&sphere, &rho);
        Dipole {
            sphere,
            rho,
            deformed,
        }
    }

    /// A cube under conformal curvature flow: the mesh at every step.
    pub fn rounding() -> Vec<Mesh> {
        conformal_smooth(&cube(DIVISIONS), STEPS, RATE)
    }

    /// The Dirac spheres of one eigenvalue, with the energies of their fields.
    pub struct Gallery {
        pub eigenvalue: i32,
        pub energies: Vec<f64>,
        pub spheres: Vec<Mesh>,
    }

    /// The sphere split the given number of times, and a few Dirac spheres for each of the
    /// eigenvalues.
    pub fn dirac(levels: usize) -> (Mesh, Vec<Gallery>) {
        let mesh = icosphere(levels);
        let galleries = par_map(&EIGENVALUES, |&eigenvalue| {
            let (energies, spheres) = dirac_spheres(&mesh, eigenvalue, STATES);
            Gallery {
                eigenvalue,
                energies,
                spheres,
            }
        });
        (mesh, galleries)
    }
}

use scenes::*;

/// Every scene, computed once, the three side by side.
struct Scenes {
    dipole: Dipole,
    rounding: Vec<Mesh>,
    sphere: Mesh,
    galleries: Vec<Gallery>,
}

fn scenes() -> &'static Scenes {
    static SCENES: OnceLock<Scenes> = OnceLock::new();
    SCENES.get_or_init(|| {
        std::thread::scope(|s| {
            let dipole = s.spawn(dipole);
            let rounding = s.spawn(rounding);
            let (sphere, galleries) = dirac(DIRAC_LEVELS);
            Scenes {
                dipole: dipole.join().expect("the dipole"),
                rounding: rounding.join().expect("the rounding"),
                sphere,
                galleries,
            }
        })
    })
}

/// The loop: one turn of the view, the flow played once and held, each Dirac state in turn.
const SECONDS: f32 = 12.0;
/// The share of the loop over which the flow plays.
const FLOW_SHARE: f32 = 0.75;
/// The camera's elevation, and its azimuth at the start of the loop.
const ELEVATION: f32 = 0.35;
const AZIMUTH: f32 = -0.96;
/// How far the camera stands from a surface, in units of its reach from its centre: room for
/// the label under it.
const FRAMING: f32 = 5.0;
/// The checkerboard's cubes across the reference surface.
const SQUARES: f64 = 8.0;

/// A camera on `rect` looking at the mesh's centre, far enough to fit it, at the azimuth.
fn camera(rect: Rect, mesh: &Mesh, azimuth: f32) -> Camera {
    let centre = mesh.vertices.mean();
    let reach = mesh
        .vertices
        .iter()
        .map(|v| (v - centre).norm())
        .fold(0.0, f64::max);
    let lens = Lens::Perspective(0.5);
    Camera::orbit(
        rect,
        reach3(centre),
        FRAMING * reach as f32,
        azimuth,
        ELEVATION,
        lens,
    )
}

/// The colour at a hit of a surface spun from the reference (the same faces): the square of a
/// solid checkerboard of cubes, `SQUARES` of them across the reference, that the point lies in on
/// the reference (straight squares on flat faces, circles on a sphere).
fn checker(reference: &Mesh, reach: f64, hit: &Hit) -> Light {
    let on_reference = hit.at(reference, &reference.vertices);
    let cell = |x: f64| (x / reach * SQUARES / 2.0).round_ties_even() as i64;
    let place = [on_reference.e1(), on_reference.e2(), on_reference.e3()].map(cell);
    let dark = place.iter().sum::<i64>().rem_euclid(2) == 1;
    checker_light(dark, hit.outside)
}

/// One surface, checkered by where it was on the reference, in its panel, with a label under it.
fn checkered(c: &mut Canvas, rect: Rect, mesh: &Mesh, reference: &Mesh, azimuth: f32, label: &str) {
    let reach = reference
        .vertices
        .iter()
        .map(|v| v.norm())
        .fold(0.0, f64::max);
    let cam = camera(rect, mesh, azimuth);
    render(c, &cam, rect, mesh, |hit| checker(reference, reach, hit));
    label_under(c, rect, label, 16.0, 0.8);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let s = scenes();
    let screen = c.rect();
    let phase = (t / SECONDS).rem_euclid(1.0);
    let azimuth = AZIMUTH + core::f32::consts::TAU * phase;
    let (top, split) = (screen.height() * 0.12, screen.height() * 0.62);
    let upper = Rect::new(0.0, top, screen.width(), split);
    let lower = Rect::new(0.0, split, screen.width(), screen.height());

    // The sphere coloured by the change asked of it, and the surface that has it.
    let dipole = &s.dipole;
    let limit = dipole.rho.iter().map(|r| r.s().abs()).fold(0.0, f64::max);
    let rect = upper.column(0, 3);
    let cam = camera(rect, &dipole.sphere, azimuth);
    render(c, &cam, rect, &dipole.sphere, |hit| {
        colormap::coolwarm((0.5 + 0.5 * dipole.rho.site(hit.face).s() / limit) as f32)
    });
    label_under(c, rect, "CHANGE IN MEAN CURVATURE", 16.0, 0.8);
    let (rect, label) = (upper.column(1, 3), "SPIN TRANSFORMATION");
    checkered(c, rect, &dipole.deformed, &dipole.sphere, azimuth, label);

    // The flow, played over most of the loop and held at its end.
    let steps = s.rounding.len() - 1;
    let step = ((phase / FLOW_SHARE).min(1.0) * steps as f32).round() as usize;
    let (rect, label) = (upper.column(2, 3), format!("CONFORMAL FLOW, STEP {step}"));
    checkered(c, rect, &s.rounding[step], &s.rounding[0], azimuth, &label);

    // A Dirac sphere of each eigenvalue, the states in turn.
    let state = ((phase * STATES as f32) as usize).min(STATES - 1);
    for (i, gallery) in s.galleries.iter().enumerate() {
        let rect = lower.column(i, s.galleries.len());
        let energy = gallery.energies[state];
        let label = format!("N = {}, ENERGY {energy:.3}", gallery.eigenvalue);
        checkered(c, rect, &gallery.spheres[state], &s.sphere, azimuth, &label);
    }
    let sub = format!(
        "LEAST-ENERGY FIELDS OF THE DIRAC OPERATOR LESS RHO (VGA3D EVEN CELLS); DIRAC SPHERES, STATE {} OF {STATES}",
        state + 1
    );
    let title = "SPIN TRANSFORMATIONS: CONFORMAL MAPS FROM QUATERNION FIELDS";
    caption(c, title, &sub);
}

fn main() {
    run(
        Anim::new("spin transformations", SECONDS).size(960, 540),
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::scenes::*;
    use super::spin::*;
    use gax::field::{Field, SparseMap};
    use gax::vga3d::{Odd, Rotor};
    use gax_numga_examples::mesh::{Mesh, Scalar3, Vector3};
    use std::collections::BTreeMap;

    /// numga's finer sphere for the Dirac spheres.
    const NUMGA_DIRAC_LEVELS: usize = 4;

    /// The geometric derivative of fields on the vertices, linear on each face, `[F, V]` with
    /// vector cells: each corner's coupling is the gradient of its hat function. Divided by the
    /// face's plane it is the Dirac operator.
    fn geometric_derivative(mesh: &Mesh) -> SparseMap<Vector3> {
        mesh.at_corners(&mesh.hat_gradients())
    }

    /// Every corner's cosine.
    fn cosines(mesh: &Mesh) -> Vec<f64> {
        mesh.corner_cosines()
            .iter()
            .flat_map(|c| c.iter().map(|c| c.s()))
            .collect()
    }

    /// The change in every corner's cosine from one mesh to another on the same faces.
    fn cosine_changes(from: &Mesh, to: &Mesh) -> Vec<f64> {
        cosines(from)
            .iter()
            .zip(cosines(to))
            .map(|(a, b)| (b - a).abs())
            .collect()
    }

    fn largest(xs: &[f64]) -> f64 {
        xs.iter().copied().fold(0.0, f64::max)
    }

    fn median(xs: &[f64]) -> f64 {
        let mut sorted = xs.to_vec();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        (sorted[(n - 1) / 2] + sorted[n / 2]) / 2.0
    }

    fn mean(xs: &[f64]) -> f64 {
        xs.iter().sum::<f64>() / xs.len() as f64
    }

    fn variance(xs: &[f64]) -> f64 {
        let m = mean(xs);
        mean(&xs.iter().map(|x| (x - m).powi(2)).collect::<Vec<_>>())
    }

    /// Whether the correlation of two series exceeds `least`, without its square root.
    fn correlated_over(a: &[f64], b: &[f64], least: f64) -> bool {
        let (ma, mb) = (mean(a), mean(b));
        let covariance = mean(
            &a.iter()
                .zip(b)
                .map(|(x, y)| (x - ma) * (y - mb))
                .collect::<Vec<_>>(),
        );
        covariance > 0.0 && covariance.powi(2) > least.powi(2) * variance(a) * variance(b)
    }

    fn values(field: &Field<Scalar3>) -> Vec<f64> {
        field.iter().map(|s| s.s()).collect()
    }

    /// The discretization is first order: each split of the faces roughly halves the largest
    /// change in a corner's cosine.
    #[test]
    fn conformal_error_falls_with_refinement() {
        let errors = [2, 3].map(|levels| {
            let mesh = icosphere(levels);
            let deformed = spin_transform_deform(&mesh, &dipole_rho(&mesh, 0.5));
            largest(&cosine_changes(&mesh, &deformed))
        });
        assert!(errors[1] < 0.6 * errors[0], "{errors:?}");
    }

    /// numga's `dipole` checks.
    #[test]
    fn the_dipole_keeps_angles_and_raises_curvature_where_asked() {
        let Dipole {
            sphere,
            rho,
            deformed,
        } = dipole();
        // No change in curvature leaves the sphere as it is, centred (numga's 1e-11).
        let still = spin_transform_deform(&sphere, &rho.map(|r| r * 0.0));
        let centre = sphere.vertices.mean();
        let off = still
            .vertices
            .iter()
            .zip(sphere.vertices.iter())
            .map(|(a, b)| (a - (b - centre)).norm())
            .fold(0.0, f64::max);
        assert!(off < 1e-11, "{off:e}");
        // The deformation keeps every corner's angle, up to the mesh's resolution.
        assert!(largest(&cosine_changes(&sphere, &deformed)) < 0.05);
        // Mean curvature rises where it was asked to rise.
        let risen = &mean_curvature(&recenter(&deformed)) - &mean_curvature(&sphere);
        assert!(correlated_over(&values(&risen), &values(&rho), 0.5));
    }

    /// numga's `rounding` checks.
    #[test]
    fn conformal_flow_rounds_the_cube_and_keeps_its_angles() {
        let frames = rounding();
        assert_eq!(frames.len(), STEPS + 1);
        // The spread of the vertices' distances from the centre falls fivefold.
        let radii = |mesh: &Mesh| mesh.vertices.iter().map(|v| v.norm()).collect::<Vec<_>>();
        let (first, last) = (&frames[0], &frames[STEPS]);
        assert!(variance(&radii(last)) < 0.2f64.powi(2) * variance(&radii(first)));
        // The angles hold up to the mesh's resolution: in the median, against a cosine change
        // of up to 0.12 at the creases in the first step.
        assert!(median(&cosine_changes(first, last)) < 0.05);
    }

    /// numga's `dirac` checks.
    #[test]
    fn dirac_spheres_have_integer_eigenvalues_and_keep_angles() {
        let (mesh, galleries) = dirac(NUMGA_DIRAC_LEVELS);
        for gallery in &galleries {
            // Each eigenvalue asks for a field of zero energy, met up to the mesh's resolution
            // against the spectrum's unit spacing.
            assert!(
                gallery.energies.iter().all(|e| e.abs() < 0.05),
                "{:?}",
                gallery.energies
            );
            // Every Dirac sphere keeps the angles, up to the mesh's resolution away from where
            // its field vanishes.
            for sphere in &gallery.spheres {
                assert!(median(&cosine_changes(&mesh, sphere)) < 0.1);
            }
        }
    }

    /// Each coupling's cell, those between the same sites summed.
    fn cells<C: Copy + core::ops::Add<Output = C>>(
        map: &SparseMap<C>,
    ) -> BTreeMap<(usize, usize), C> {
        let mut cells = BTreeMap::new();
        for (r, c, cell) in map.couplings() {
            cells
                .entry((r, c))
                .and_modify(|sum| *sum = *sum + cell)
                .or_insert(cell);
        }
        cells
    }

    /// Crane's operator, each corner taken by the edge it faces over minus twice the area, is
    /// the geometric derivative divided by each face's plane; and the geometric derivative of
    /// position is 2 on every face, the surface's dimension.
    #[test]
    fn the_dirac_operator_is_the_geometric_derivative_divided_by_the_face_planes() {
        let mesh = icosphere(3);
        let edges = mesh.edge_vectors();
        let facing = [0, 1, 2].map(|k| {
            Field::from_fn(mesh.faces.len(), |f| {
                edges.site(mesh.face_edges[f][k]) * mesh.face_edge_orientation[f][k]
            })
        });
        let inverse_areas = mesh.triangle_areas().map(|a| a.inverse());
        let dirac = (&SparseMap::from_diagonal(&inverse_areas) * &mesh.at_corners(&facing))
            .map_cells(|e| e * -0.5);
        let derivative = geometric_derivative(&mesh);
        let divided =
            &SparseMap::from_diagonal(&mesh.face_planes().map(|p| p.inverse())) * &derivative;
        let (divided, dirac) = (cells(&divided), cells(&dirac));
        assert_eq!(divided.len(), dirac.len());
        for (sites, cell) in &divided {
            assert!((*cell - dirac[sites].cast::<Odd>()).norm() < 1e-12);
        }
        let of_position = &derivative * &mesh.vertices;
        assert!(
            of_position
                .iter()
                .all(|d| (d - Scalar3::new(2.0).cast::<Rotor>()).norm() < 1e-12)
        );
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
    }
}
