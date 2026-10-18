//! numga's `optics/crystal_diffraction`: a crystal's atoms and its diffraction peaks carried
//! through a deformation, in the geometric algebra of space (VGA3D).
//!
//! A phase is the pairing of a wavevector with a position, `k · x`. The deformation, a map on
//! positions built with the position left open (`Vector<(Vector,)>`), moves the atoms. The
//! wavevectors move so that every phase is kept: the moved `k'` solves `k' · A(x) = k · x` for
//! every `x`, a pairing solve with no transpose (the inverse of the deformation's adjoint). Read
//! as phase planes, the wavevectors' duals, the same phase is the incidence `P ∨ x`, and the
//! planes move by the same solve against the regressive pairing, without the metric.
//!
//! A finite square layer of atoms scatters a wave with the momentum transfer `q` by the sum of
//! each atom's wave `exp(I q · x)`, its intensity the squared size of that sum over the atom
//! count's square: one at each reciprocal-lattice vector, where every atom scatters in phase.
//! VGA3D's pseudoscalar `I` squares to minus one and commutes with everything, so its scalars and
//! pseudoscalars are the complex numbers; gax's VGA3D has no kind for that pair, so the waves
//! are `gax::Complex` exponentials.
//!
//! The animation: the layer stretched along `x`, sheared and turned, and back (left), and its
//! diffraction pattern on a logarithmic scale (right): the peaks follow the moved reciprocal
//! lattice, ringed, squeezing where the atoms stretch.

use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, ORIGIN2, Point2, Rect, backdrop, caption, colormap,
    palette, run, signal::haversine,
};

mod crystal {
    use gax::Complex;
    use gax::vga3d::{Vector, intrinsic};

    pub type V = Vector<(), f64>;
    /// A phase plane: the Hodge dual of a wavevector, of the other orientation (intrinsic).
    pub type B = intrinsic::Bivector<(), f64>;
    /// A linear map on positions.
    pub type Linear = Vector<(Vector,), f64>;

    pub fn x() -> V {
        Vector::new(1.0, 0.0, 0.0)
    }

    pub fn y() -> V {
        Vector::new(0.0, 1.0, 0.0)
    }

    /// A layer of atoms, its wavevectors and their phase planes.
    #[derive(Clone, Debug)]
    pub struct Crystal {
        pub positions: Vec<V>,
        pub reciprocal: Vec<V>,
        pub phase_planes: Vec<B>,
    }

    /// The wavevector whose phase at every deformed position is the old one's at the position:
    /// the solve of `k' · A(x) = k · x` for every `x`.
    pub fn carried_wavevector(deformation: Linear, k: V) -> V {
        let pairing = Vector::slot() | deformation.of(Vector::slot());
        pairing.solve_pairing(k | Vector::slot())
    }

    /// The phase plane whose incidence with every deformed position is the old one's with the
    /// position: the same solve against the regressive pairing, `P' ∨ A(x) = P ∨ x`.
    pub fn carried_plane(deformation: Linear, plane: B) -> B {
        let incidence = B::slot() & deformation.of(Vector::slot());
        incidence.solve_pairing(plane & Vector::slot())
    }

    impl Crystal {
        /// The atoms moved, and the phases measured at them kept.
        pub fn carried(&self, deformation: Linear) -> Crystal {
            Crystal {
                positions: self.positions.iter().map(|p| deformation.of(*p)).collect(),
                reciprocal: self
                    .reciprocal
                    .iter()
                    .map(|k| carried_wavevector(deformation, *k))
                    .collect(),
                phase_planes: self
                    .phase_planes
                    .iter()
                    .map(|p| carried_plane(deformation, *p))
                    .collect(),
            }
        }

        /// The coherent scattering of equal atoms at a momentum transfer (the outgoing
        /// wavevector less the incoming one), normalized to one when every atom scatters in
        /// phase; no polarization factor is included.
        pub fn intensity(&self, transfer: V) -> f64 {
            let waves = self
                .positions
                .iter()
                .map(|p| Complex::new(0.0, (transfer | *p).s()).exp());
            let sum = waves.fold(Complex::real(0.0), |a, w| a + w);
            sum.norm_sqr() / (self.positions.len() * self.positions.len()) as f64
        }
    }

    /// Stretch along `x`, slide horizontal rows along `x`, and turn the whole layer.
    pub fn deformation(stretch: f64, shear: f64, angle: f64) -> Linear {
        let local = Vector::slot()
            + x() * (x() | Vector::slot()) * (stretch - 1.0)
            + x() * (y() | Vector::slot()) * shear;
        ((x() ^ y()) * (-angle / 2.0)).exp() >> local
    }

    /// A finite square layer of `(2 shells + 1)²` atoms and its reciprocal lattice of
    /// `(2 orders + 1)²` wavevectors, embedded in space, row by row along `y`.
    pub fn square(shells: i32, orders: i32, spacing: f64) -> Crystal {
        let lattice = |n: i32, step: f64| -> Vec<V> {
            (-n..=n)
                .flat_map(|j| {
                    (-n..=n).map(move |i| (x() * f64::from(i) + y() * f64::from(j)) * step)
                })
                .collect()
        };
        let reciprocal = lattice(orders, core::f64::consts::TAU / spacing);
        Crystal {
            positions: lattice(shells, spacing),
            phase_planes: reciprocal.iter().map(|k| k.hodge()).collect(),
            reciprocal,
        }
    }

    /// Momentum transfers on a square grid of `pixels` across, `half_width` from the middle to
    /// each side, row by row along `y`.
    pub fn transfer_grid(half_width: f64, pixels: usize) -> Vec<V> {
        let along = |i: usize| -half_width + 2.0 * half_width * i as f64 / (pixels - 1) as f64;
        (0..pixels * pixels)
            .map(|k| x() * along(k % pixels) + y() * along(k / pixels))
            .collect()
    }

    // --- the scene --------------------------------------------------------------------

    pub const SHELLS: i32 = 3;
    pub const ORDERS: i32 = 3;
    pub const SPACING: f64 = 1.0;
    pub const MAX_STRETCH: f64 = 1.7;
    pub const MAX_SHEAR: f64 = 0.6;
    pub const MAX_ANGLE: f64 = core::f64::consts::PI / 9.0;
    pub const HALF_WIDTH: f64 = 10.0;
    pub const PIXELS: usize = 161;

    /// The stretch, shear and turn a share of the way to the full ones.
    pub fn amounts(share: f64) -> [f64; 3] {
        [
            1.0 + (MAX_STRETCH - 1.0) * share,
            MAX_SHEAR * share,
            MAX_ANGLE * share,
        ]
    }

    /// The deformation a share of the way to the full stretch, shear and turn.
    pub fn deformed(share: f64) -> Linear {
        let [stretch, shear, angle] = amounts(share);
        deformation(stretch, shear, angle)
    }
}

use crystal::*;

/// One loop: deformed and back.
const SECONDS: f32 = 6.0;
/// The intensity shown at the bottom of the logarithmic colour scale.
const INTENSITY_FLOOR: f64 = 1e-4;
/// The atoms shown within this distance of the middle.
const POSITION_LIMIT: f32 = 7.8;

/// The share of the deformation at time `t`: a smooth cycle out and back.
fn share(t: f32) -> f64 {
    let turn = f64::from(t / SECONDS) * core::f64::consts::TAU;
    haversine(turn)
}

/// A vector of the layer's plane as a drawing point.
fn in_layer(v: V) -> Point2 {
    Point2::xy(v.e1() as f32, v.e2() as f32)
}

/// An intensity on the logarithmic colour scale, from the floor to one.
fn glow(intensity: f64) -> Light {
    let decades = INTENSITY_FLOOR.log10().abs();
    let t = (intensity.max(INTENSITY_FLOOR).log10() + decades) / decades;
    colormap::inferno(t as f32)
}

/// The atoms, with the axes through the middle.
fn draw_atoms(c: &mut Canvas, rect: Rect, crystal: &Crystal) {
    let ax = Axes::equal(rect, ORIGIN2, POSITION_LIMIT);
    let reach = POSITION_LIMIT * 2.0;
    let axes = [
        [Point2::xy(-reach, 0.0), Point2::xy(reach, 0.0)],
        [Point2::xy(0.0, -reach), Point2::xy(0.0, reach)],
    ];
    ax.stroke(c, &axes, 1.0, palette::grid());
    let atoms: Vec<Point2> = crystal.positions.iter().map(|p| in_layer(*p)).collect();
    ax.scatter(c, &atoms, Marker::Dot, 7.0, palette::green());
    ax.frame(c, "ATOMS", "", "");
}

/// The intensity over the transfer grid (nearest sample per pixel), with the reciprocal
/// vectors ringed.
fn draw_pattern(c: &mut Canvas, rect: Rect, crystal: &Crystal, grid: &[f64]) {
    let half = HALF_WIDTH as f32;
    let ax = Axes::equal(rect, ORIGIN2, half);
    let pixel = 2.0 * half / (PIXELS - 1) as f32;
    let index = |v: f32| ((v + half) / pixel).round().clamp(0.0, (PIXELS - 1) as f32) as usize;
    ax.image(c, 1, |p| {
        let [u, v] = p.to_euclidean();
        Some(glow(grid[index(v) * PIXELS + index(u)]))
    });
    let peaks: Vec<Point2> = crystal.reciprocal.iter().map(|k| in_layer(*k)).collect();
    ax.scatter(c, &peaks, Marker::Ring, 10.0, palette::sky().faded(0.5));
    ax.frame(c, "DIFFRACTION, LOG INTENSITY", "", "");
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let share = share(t);
    let crystal = square(SHELLS, ORDERS, SPACING).carried(deformed(share));
    let intensities: Vec<f64> = transfer_grid(HALF_WIDTH, PIXELS)
        .into_iter()
        .map(|q| crystal.intensity(q))
        .collect();
    let row = Rect::new(0.0, h * 0.17, w, h * 0.87);
    let side = row.height().min(w * 0.45);
    let square_in = |r: Rect| Rect::around(r.centre(), Point2::direction(side * 0.5, side * 0.5));
    draw_atoms(c, square_in(row.column(0, 2)), &crystal);
    draw_pattern(c, square_in(row.column(1, 2)), &crystal, &intensities);
    let size = (h / 40.0).clamp(6.0, 13.0);
    let [stretch, shear, angle] = amounts(share);
    let degrees = angle.to_degrees();
    let readout = format!("STRETCH {stretch:.2}  SHEAR {shear:.2}  TURN {degrees:.0} DEG");
    let below = Point2::xy(w * 0.5, h * 0.975);
    c.text(&readout, below, size, palette::yellow(), Align::Center);
    caption(
        c,
        "CRYSTAL DIFFRACTION: PEAKS CARRIED THROUGH A DEFORMATION",
        "VGA3D: WAVEVECTORS MOVE BY A PAIRING SOLVE, SO EVERY PHASE K . X IS KEPT",
    );
}

fn main() {
    run(
        Anim::new("crystal diffraction", SECONDS).size(960, 540),
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::crystal::*;
    use gax::ApproxEq;

    /// The phase of every wavevector at every atom.
    fn phases(c: &Crystal) -> Vec<f64> {
        c.reciprocal
            .iter()
            .flat_map(|k| c.positions.iter().map(move |p| (*k | *p).s()))
            .collect()
    }

    /// Every phase measured at every atom is kept, the phase planes stay the wavevectors'
    /// duals, every reciprocal vector is a peak, and the deformed pattern is the reference one
    /// seen through the moved transfers.
    fn assert_carried(reference: &Crystal, deformation: Linear, grid: &[V]) {
        let crystal = reference.carried(deformation);
        for (a, b) in phases(&crystal).iter().zip(phases(reference)) {
            assert!((a - b).abs() < 1e-10);
        }
        for (plane, k) in crystal.phase_planes.iter().zip(&crystal.reciprocal) {
            assert!(plane.max_abs_diff(&k.hodge()) < 1e-11);
        }
        for k in &crystal.reciprocal {
            assert!((crystal.intensity(*k) - 1.0).abs() < 1e-12);
        }
        for q in grid {
            let moved = carried_wavevector(deformation, *q);
            assert!((crystal.intensity(moved) - reference.intensity(*q)).abs() < 1e-11);
        }
    }

    #[test]
    fn deforming_atoms_and_phase_measurements_preserves_scattering() {
        let reference = square(2, 2, 1.3);
        let grid = transfer_grid(8.0, 17);
        for (stretch, shear, angle) in [(0.8, -0.3, 0.2), (1.7, 0.6, -0.4), (1.2, 0.4, 0.7)] {
            assert_carried(&reference, deformation(stretch, shear, angle), &grid);
        }
    }

    /// numga's scenario checks, for its three cases: the square lattice, stretched, and
    /// stretched, sheared and turned.
    #[test]
    fn scenes_keep_every_phase() {
        let reference = square(SHELLS, ORDERS, SPACING);
        let grid = transfer_grid(HALF_WIDTH, 9);
        let cases = [
            deformation(1.0, 0.0, 0.0),
            deformation(MAX_STRETCH, 0.0, 0.0),
            deformation(MAX_STRETCH, MAX_SHEAR, MAX_ANGLE),
        ];
        for case in cases {
            assert_carried(&reference, case, &grid);
        }
    }

    /// The phase plane's incidence with a position is the wavevector's phase there: `k^* ∨ x
    /// = k · x`.
    #[test]
    fn phase_planes_read_the_phase_by_incidence() {
        let k = x() * 0.7 - y() * 1.3 + V::new(0.0, 0.0, 0.4);
        let p = x() * -2.0 + y() * 0.5 + V::new(0.0, 0.0, 1.1);
        assert!(((k.hodge() & p).s() - (k | p).s()).abs() < 1e-14);
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
