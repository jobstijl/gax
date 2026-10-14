//! numga's `surfaces/arap`: as rigid as possible. A surface is bent by moving a few of its
//! vertices, and every neighbourhood of it is kept as close to a turn of its rest shape as it
//! can be, in the geometric algebra of space, VGA3D.
//!
//! The surface's energy is the cotangent-weighted squared mismatch, over every edge at each of
//! its two ends, between the edge as it is and the rest edge turned by a rotor of that end's own.
//! A rotor keeps lengths, so an edge's two mismatches add up to
//! `2 (edge² + rest² - 2 edge | turned)`, with `turned` the rest edge turned by the mean of its
//! two ends' turns. The energy falls in two alternating steps.
//!
//! With the vertices held, each vertex's best rotor `R` turns its rest edges onto its edges as
//! they are: it makes the most of `edge | (R >> rest)`, summed over the vertex's edges. With the
//! rotor left open in both places the sandwich holds it, `Rotor::slot() >> rest` is the rest edge
//! turned by any even multivector, and paired with the edge it is a form on two rotors, a number
//! for every two, with `form(R, R) == edge | (R >> rest)`. The sparse mean over each edge's ends
//! sums each vertex's forms as it would sum numbers, weighted by the edges, and the rotor that
//! makes the most of each sum is its top eigenvector: one eigensolve per vertex.
//!
//! With the rotors held, the best vertices match every edge to its rest edge turned by its two
//! ends' rotors. A rotor bound into the sandwich with the vector left open, `R >> Vector::slot()`,
//! is the turn itself, a map on vectors; the sparse mean averages the two turns at each edge's
//! ends as maps, and the averaged map takes the rest edge. The average of the maps does not depend
//! on the sign the eigensolver gives each rotor, as an average of rotors would: `R >> Vector` and
//! `-R >> Vector` are one map. The vertices then follow from one sparse Poisson solve with the
//! cotangent Laplacian, its right side `~d0 H turned`, as in the last step of a spin
//! transformation. The handles are held where they are moved by a stiff penalty in the same solve.
//! Laplacian editing is the same solve without the turns: every rest edge matched as it is, so
//! the surface shears and shrinks where it is bent.
//!
//! In the notation of Sorkine and Alexa the energy reads as
//! `E = Σᵢ Σ_{j ∈ N(i)} wᵢⱼ ‖(p'ᵢ - p'ⱼ) - Rᵢ (pᵢ - pⱼ)‖²`, the rotor's form as the trace of
//! `Rᵢ Sᵢ` with the covariance `Sᵢ = Σⱼ wᵢⱼ eᵢⱼ e'ᵢⱼᵀ`.
//!
//! The animation bends a bar held at one end a quarter turn up at the other, as rigid as possible
//! beside Laplacian editing, each checkered by where its faces sat at rest.

#[path = "../shared/surfaces_spin.rs"]
mod surfaces_spin;

use gax_numga_examples::mesh::Mesh;
use gax_numga_examples::{
    Anim, Camera, Canvas, Lens, Point3, Rect, backdrop, caption, label_under, run,
};
use std::sync::OnceLock;
use surfaces_spin::{checker_light, render};

mod arap {
    pub use super::surfaces_spin::cube;
    use gax::field::{Field, SparseMap};
    use gax::vga3d::{Rotor, Scalar};
    use gax_numga_examples::mesh::{Mesh, Scalar3, Vector3};

    /// A form on two rotors: a number for every two.
    pub type RotorForm = Scalar<(Rotor, Rotor), f64>;

    /// One pose's answers: the vertices as rigid as possible, the vertices by Laplacian editing,
    /// and the energy at each iteration of the first.
    pub struct Deformed {
        pub rigid: Field<Vector3>,
        pub laplacian: Field<Vector3>,
        pub energies: Vec<f64>,
    }

    /// For each pose of the handles (the targets of every vertex, of which the handles' are held),
    /// the mesh as rigid as possible after the iterations, each pose starting from the last
    /// one's, and the mesh by Laplacian editing.
    pub fn deform(
        mesh: &Mesh,
        handles: &[bool],
        poses: &[Field<Vector3>],
        stiffness: f64,
        iterations: usize,
    ) -> Vec<Deformed> {
        let (d0, means) = (mesh.d0(), mesh.edge_means());
        let weights = SparseMap::from_diagonal(&mesh.edge_ratio());
        let hold = |held: bool| Scalar3::new(if held { stiffness } else { 0.0 });
        let penalty = SparseMap::from_diagonal(&handles.iter().map(|&h| hold(h)).collect());
        let rest = &d0 * &mesh.vertices;
        // The cotangent Laplacian with the handles held, and the divergence of edges into it.
        let held = &(&mesh.laplacian() + &penalty) * Vector3::slot();
        let divergence = &d0.reverse() * &weights;

        let mut vertices = mesh.vertices.clone();
        let mut deformed = Vec::new();
        for targets in poses {
            let pull = &penalty * targets;
            let mut energies = Vec::new();
            for _ in 0..iterations {
                let edges = &d0 * &vertices;
                let rotors = best_rotors(&edges, &rest, &means, &weights);
                // Each rest edge turned by its two ends' rotors, the turns averaged as maps.
                let turns = &means * &rotors.map(|r| r >> Vector3::slot());
                let turned = turns.zip_with(&rest, |turn, r| turn.of(r));
                energies.push(energy(
                    &edges, &rest, &turned, &weights, &vertices, targets, &penalty,
                ));
                // The vertices whose edges best match the turned ones.
                vertices = held.solve(&(&(&divergence * &turned) + &pull));
            }
            // The vertices whose edges best match the rest edges as they are.
            let laplacian = held.solve(&(&(&divergence * &rest) + &pull));
            deformed.push(Deformed {
                rigid: vertices.clone(),
                laplacian,
                energies,
            });
        }
        deformed
    }

    /// Each vertex's best rotor: the top eigenvector of its edges' summed forms, each form the
    /// edge paired with the rest edge turned by an open rotor.
    fn best_rotors(
        edges: &Field<Vector3>,
        rest: &Field<Vector3>,
        means: &SparseMap<Scalar3>,
        weights: &SparseMap<Scalar3>,
    ) -> Field<Rotor<(), f64>> {
        // The rest edge turned by any rotor: the sandwich with the rotor open in both places.
        let open = Rotor::<(), f64>::slot();
        let forms: Field<RotorForm> =
            edges.zip_with(rest, |e, r| e.scalar_product((open * r) * open.reverse()));
        let summed = &means.reverse() * &(weights * &forms);
        summed.map(|form| {
            let (_, rotors) = form.as_map().eigh();
            rotors[3]
        })
    }

    /// The energy: twice each edge's weighted mismatch with its turned rest edge, and the
    /// handles' penalty.
    fn energy(
        edges: &Field<Vector3>,
        rest: &Field<Vector3>,
        turned: &Field<Vector3>,
        weights: &SparseMap<Scalar3>,
        vertices: &Field<Vector3>,
        targets: &Field<Vector3>,
        penalty: &SparseMap<Scalar3>,
    ) -> f64 {
        let mismatch = Field::from_fn(edges.len(), |e| {
            let (edge, rest, turned) = (edges.site(e), rest.site(e), turned.site(e));
            let squares = edge.norm_squared() + rest.norm_squared();
            Scalar3::new(squares) - (edge | turned) * 2.0
        });
        let misplaced = vertices.zip_with(targets, |v, t| Scalar3::new((v - t).norm_squared()));
        2.0 * (weights * &mismatch).sum().s() + (penalty * &misplaced).sum().s()
    }

    /// A closed bar along x, its square cross section 2 across and its length the given one,
    /// each side of the cube it is stretched from a grid of `divisions` by `divisions` squares.
    pub fn bar(divisions: usize, length: f64) -> Mesh {
        let cube = cube(divisions);
        let x = Vector3::new(1.0, 0.0, 0.0);
        let stretched = cube
            .vertices
            .map(|v| v + x * ((v | x).s() * (length / 2.0 - 1.0)));
        cube.with_vertices(stretched)
    }
}

mod scenes {
    use super::arap::*;
    use gax::field::Field;
    use gax::vga3d::Bivector;
    use gax_numga_examples::mesh::{Mesh, Vector3};

    /// The bar: its squares per side of the cube it is stretched from, and its length.
    pub const DIVISIONS: usize = 10;
    pub const LENGTH: f64 = 6.0;
    /// How stiffly the handles are held, how far the far end turns, and the frames over the bend
    /// with the iterations at each, each frame starting from the last.
    pub const STIFFNESS: f64 = 1e3;
    pub const BEND: f64 = core::f64::consts::FRAC_PI_2;
    pub const FRAMES: usize = 24;
    pub const ITERATIONS: usize = 6;
    /// How near a vertex is to an end of the bar to be a handle.
    const AT_END: f64 = 1e-6;

    const X: Vector3 = Vector3::new(1.0, 0.0, 0.0);
    const Z: Vector3 = Vector3::new(0.0, 0.0, 1.0);

    /// The problem: the bar at rest, its handles (the vertices at either end), and where they
    /// are moved at each frame.
    pub struct Bend {
        pub bar: Mesh,
        pub handles: Vec<bool>,
        pub poses: Vec<Field<Vector3>>,
    }

    /// The far end of the bar where an arc of its length from the near end would carry it,
    /// turned by the angle about y, curving up: the bar's end turned about the arc's centre.
    pub fn bent(vertices: &Field<Vector3>, angle: f64) -> Field<Vector3> {
        let radius = LENGTH / angle;
        let pivot = X * (-LENGTH / 2.0) + Z * radius;
        let turn = Bivector::<(), f64>::new(0.0, angle / 2.0, 0.0).exp();
        let end = pivot + (turn >> (Z * -radius));
        vertices.map(|v| end + (turn >> (v - X * (LENGTH / 2.0))))
    }

    /// The bar held at its near end and its far end bent a quarter turn up over the frames.
    pub fn bend() -> Bend {
        let bar = bar(DIVISIONS, LENGTH);
        let along: Vec<f64> = bar.vertices.iter().map(|v| (v | X).s()).collect();
        let far: Vec<bool> = along.iter().map(|&a| a > LENGTH / 2.0 - AT_END).collect();
        let handles = along
            .iter()
            .zip(&far)
            .map(|(&a, &f)| f || a < -LENGTH / 2.0 + AT_END)
            .collect();
        let pose = |angle: f64| {
            let bent = bent(&bar.vertices, angle);
            let at = |v: usize| {
                if far[v] {
                    bent.site(v)
                } else {
                    bar.vertices.site(v)
                }
            };
            Field::from_fn(bar.vertices.len(), at)
        };
        let poses = (1..=FRAMES)
            .map(|k| pose(BEND * k as f64 / FRAMES as f64))
            .collect();
        Bend {
            bar,
            handles,
            poses,
        }
    }

    /// The bend at every frame, as rigid as possible and by Laplacian editing.
    pub fn bending(bend: &Bend) -> Vec<Deformed> {
        deform(&bend.bar, &bend.handles, &bend.poses, STIFFNESS, ITERATIONS)
    }
}

use arap::Deformed;
use scenes::*;

/// The bend and its frames, computed once.
struct Scene {
    bend: Bend,
    frames: Vec<Deformed>,
}

fn scene() -> &'static Scene {
    static SCENE: OnceLock<Scene> = OnceLock::new();
    SCENE.get_or_init(|| {
        let bend = bend();
        let frames = bending(&bend);
        Scene { bend, frames }
    })
}

/// The loop: numga's 24 frames of 80 ms bending the bar, and as many unbending it.
const SECONDS: f32 = 3.84;
/// The view, its target, and how far it stands from it.
const ELEVATION: f32 = 0.31;
const AZIMUTH: f32 = -1.12;
const TARGET: [f32; 3] = [0.1, 0.0, 1.45];
const DISTANCE: f32 = 15.0;

/// The bar in one shape, checkered by the cell of the rest grid each face's centroid sat in.
fn panel(c: &mut Canvas, rect: Rect, shape: &Mesh, dark: &[bool], label: &str) {
    let [x, y, z] = TARGET;
    let target = Point3::xyz(x, y, z);
    let cam = Camera::orbit(
        rect,
        target,
        DISTANCE,
        AZIMUTH,
        ELEVATION,
        Lens::Perspective(0.5),
    );
    render(c, &cam, rect, shape, |hit| {
        checker_light(dark[hit.face], hit.outside)
    });
    label_under(c, rect, label, 16.0, 0.8);
}

/// Each face's checker parity at rest: the cell, one grid square of the bar, its centroid lies in.
fn checker(bar: &Mesh) -> Vec<bool> {
    let cell = [LENGTH, 2.0, 2.0].map(|side| side / DIVISIONS as f64);
    bar.face_centers()
        .iter()
        .map(|c| {
            let place = [c.e1(), c.e2(), c.e3()];
            let index: i64 = (0..3).map(|k| (place[k] / cell[k]).floor() as i64).sum();
            index.rem_euclid(2) == 1
        })
        .collect()
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let s = scene();
    // Bend over the first half of the loop and unbend over the second.
    let phase = (t / SECONDS).rem_euclid(1.0);
    let bend = 1.0 - (2.0 * phase - 1.0).abs();
    let frame = ((bend * FRAMES as f32).round() as usize).clamp(1, FRAMES) - 1;
    let deformed = &s.frames[frame];
    let bar = &s.bend.bar;
    let dark = checker(bar);
    let screen = c.rect();
    let below = Rect::new(0.0, screen.height() * 0.12, screen.width(), screen.height());
    let shapes = [
        (&deformed.rigid, "AS RIGID AS POSSIBLE"),
        (&deformed.laplacian, "LAPLACIAN EDITING"),
    ];
    for (i, (vertices, label)) in shapes.into_iter().enumerate() {
        let shape = bar.with_vertices(vertices.clone());
        panel(c, below.column(i, 2), &shape, &dark, label);
    }
    let degrees = (BEND * (frame + 1) as f64 / FRAMES as f64).to_degrees();
    let energy = deformed.energies.last().copied().unwrap_or_default();
    let sub = format!(
        "BENT {degrees:.0} DEG, ENERGY {energy:.4}; ROTORS FROM FORMS ON OPEN ROTORS, VERTICES FROM ONE POISSON SOLVE (VGA3D)"
    );
    caption(c, "AS RIGID AS POSSIBLE: A BAR BENT BY ITS HANDLES", &sub);
}

fn main() {
    run(Anim::new("arap", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::arap::*;
    use super::scenes::*;
    use gax::field::Field;
    use gax::vga3d::Bivector;
    use gax_numga_examples::mesh::{Mesh, Vector3};

    fn mean_cosine_change(from: &Mesh, to: &Field<Vector3>) -> f64 {
        let cosines = |mesh: &Mesh| -> Vec<f64> {
            mesh.corner_cosines()
                .iter()
                .flat_map(|c| c.iter().map(|c| c.s()))
                .collect()
        };
        let (before, after) = (cosines(from), cosines(&from.with_vertices(to.clone())));
        before
            .iter()
            .zip(&after)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / before.len() as f64
    }

    /// The same bend of the bar turned as a whole: as rigid as possible gives the turned answer.
    #[test]
    fn turning_the_bar_and_its_handles_turns_the_answer() {
        let bar = bar(4, 4.0);
        let x = Vector3::new(1.0, 0.0, 0.0);
        let along: Vec<f64> = bar.vertices.iter().map(|v| (v | x).s()).collect();
        let handles: Vec<bool> = along.iter().map(|a| a.abs() > 1.999).collect();
        let lift = Vector3::new(0.0, 0.0, 0.8);
        let pose = Field::from_fn(along.len(), |v| {
            bar.vertices.site(v)
                + if along[v] > 1.999 {
                    lift
                } else {
                    Vector3::new(0.0, 0.0, 0.0)
                }
        });
        // numga's `exp(-0.3 xy) exp(0.2 yz)`.
        let turn =
            Bivector::<(), f64>::new(0.0, 0.0, -0.3).exp() * Bivector::new(0.2, 0.0, 0.0).exp();
        let shape = &deform(&bar, &handles, core::slice::from_ref(&pose), 1e3, 4)[0].rigid;
        let turned_bar = bar.with_vertices(bar.vertices.map(|v| turn >> v));
        let turned = &deform(&turned_bar, &handles, &[pose.map(|v| turn >> v)], 1e3, 4)[0].rigid;
        assert!(
            turned
                .iter()
                .zip(shape.iter())
                .all(|(a, b)| (a - (turn >> b)).norm() < 1e-9)
        );
    }

    /// numga's `bending` checks, and a frame drawn.
    #[test]
    fn the_bending_scene_passes_its_checks_and_draws() {
        let problem = bend();
        let frames = bending(&problem);
        let Bend {
            bar,
            handles,
            poses,
        } = problem;
        let last = frames.last().expect("frames");
        // At the last frame the energy falls with every iteration, and the handles are where
        // they were moved to.
        let e = &last.energies;
        assert!(e.windows(2).all(|w| w[1] - w[0] <= 1e-9 * e[0]), "{e:?}");
        let pose = poses.last().expect("poses");
        let off = |v: usize| (last.rigid.site(v) - pose.site(v)).norm();
        assert!(
            (0..handles.len())
                .filter(|&v| handles[v])
                .all(|v| off(v) < 1e-3)
        );
        // The bar keeps its angles: its corners' cosines change a fraction as much as by
        // Laplacian editing.
        assert!(
            mean_cosine_change(&bar, &last.rigid) < 0.5 * mean_cosine_change(&bar, &last.laplacian)
        );
        // Left at rest, the bar stays as it is.
        let still = &deform(
            &bar,
            &handles,
            core::slice::from_ref(&bar.vertices),
            STIFFNESS,
            2,
        )[0]
        .rigid;
        assert!(
            still
                .iter()
                .zip(bar.vertices.iter())
                .all(|(a, b)| (a - b).norm() < 1e-9)
        );
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
    }
}
