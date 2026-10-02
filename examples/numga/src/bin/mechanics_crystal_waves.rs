//! numga's `mechanics/crystal_waves`: elastic waves in cubic crystals, their speeds, their
//! polarizations, and where they carry energy (VGA3D). A crystal's stiffness is a map with three
//! vector slots: given a face's normal and a displacement varying along a direction, it returns
//! the traction on the face. An isotropic solid has a dilation and a shear term; a cubic crystal
//! adds one in which each of its axes reads all three slots. A plane wave along a heading binds
//! the heading into the normal and gradient slots, leaving the Christoffel map on displacements,
//! whose eigenpairs are density times squared speed and polarization: a compressional and two
//! shear waves. Binding the polarization into the normal and displacement slots gives the flow of
//! the wave's energy, which in a crystal swings away from the heading; where the energy of many
//! headings converges, heat pulses focus into caustics. The animation builds the phonon-focusing
//! images heading by heading (numga bins 1.2 million; this bins up to 240 thousand), while a
//! heading sweeps the cube face of each crystal in turn, its three group velocities tracing the
//! wave surfaces.

use gax::vga3d::{Scalar, Vector};
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Marker, Rgb, backdrop, caption, colormap, palette, plot, run,
};
use std::sync::OnceLock;

mod crystal {
    use super::*;

    pub type V = Vector<(), f64>;
    /// The traction on a face from its normal, the displacement and its gradient direction.
    pub type Stiffness = Vector<(Vector, Vector, Vector), f64>;

    pub const NAMES: [&str; 3] = ["FUSED SILICA", "SILICON", "BETA-BRASS"];
    /// Cubic elastic constants in GPa and density in g/cm³, so speeds come out in km/s. Fused
    /// silica is isotropic (`c12 == c11 - 2 c44`); silicon is moderately and beta-brass strongly
    /// anisotropic: `2 c44 / (c11 - c12)` is 1, 1.6 and 8.5.
    pub const C11: [f64; 3] = [78.5, 165.7, 129.1];
    pub const C12: [f64; 3] = [16.1, 63.9, 109.7];
    pub const C44: [f64; 3] = [31.2, 79.6, 82.4];
    pub const DENSITY: [f64; 3] = [2.20, 2.33, 7.60];
    pub const MODES: [&str; 3] = ["SLOW SHEAR", "FAST SHEAR", "COMPRESSION"];

    /// The crystal's axes.
    pub fn axes() -> [V; 3] {
        [
            Vector::new(1.0, 0.0, 0.0),
            Vector::new(0.0, 1.0, 0.0),
            Vector::new(0.0, 0.0, 1.0),
        ]
    }

    /// The stiffness of a cubic crystal from its elastic constants. With normal `n`,
    /// displacement `v` and gradient `h`: the dilation `v | h` pushes along the normal, the
    /// shear pulls along `v` and `h` (`n | (v ^ h) == (n | v) h - (n | h) v` turns one into the
    /// other), and along the crystal's axes the stiffness differs from isotropic by
    /// `c11 - c12 - 2 c44`.
    pub fn stiffness(c11: f64, c12: f64, c44: f64) -> Stiffness {
        let s = Vector::slot;
        let dilation: Stiffness = s() * (s() | s());
        let shear: Stiffness = ((s() | s()) * s()).gp(2.0) - (s() | (s() ^ s()));
        let cubic: Stiffness = axes().iter().fold(Vector::zero(), |sum: Stiffness, a| {
            sum + *a * ((*a | s()) * (*a | s()) * (*a | s()))
        });
        dilation.gp(c12) + shear.gp(c44) + cubic.gp(c11 - c12 - 2.0 * c44)
    }

    /// Every material's stiffness.
    pub fn crystals() -> [Stiffness; 3] {
        core::array::from_fn(|m| stiffness(C11[m], C12[m], C44[m]))
    }

    /// The waves along a heading, slowest first: density times squared speed, and polarization.
    /// The heading bound into the normal and gradient slots leaves the Christoffel map; numga
    /// takes its eigenpairs directly, gax as those of the form `u | Γ(v)` (Euclidean vectors
    /// have the identity metric in their layout).
    pub fn waves(crystal: &Stiffness, heading: V) -> ([f64; 3], [V; 3]) {
        let christoffel = crystal.of(heading).at::<1>().of(heading);
        let form: Scalar<(Vector, Vector), f64> = Vector::slot() | christoffel;
        form.eigh()
    }

    /// The velocity of each wave's energy, its group velocity. With the polarization in the
    /// normal and displacement slots and the heading in the gradient slot, the stiffness returns
    /// the flux of the wave's energy; divided by density times phase speed it is the group
    /// velocity, whose component along the heading is the phase speed.
    pub fn energy_flow(
        crystal: &Stiffness,
        heading: V,
        polarization: &[V; 3],
        density: f64,
    ) -> [V; 3] {
        polarization.map(|p| {
            let squared = (p | crystal.of(heading).of(p).of(heading)).s();
            crystal
                .of(p)
                .of(p)
                .of(heading)
                .gp(1.0 / (squared * density).sqrt())
        })
    }

    /// The group velocities of the three waves for one heading, in material `m`, with the
    /// checks of numga's `focusing`: along its heading the group velocity has the phase speed.
    pub fn group_velocities(crystal: &Stiffness, m: usize, heading: V) -> [V; 3] {
        let (values, polarization) = waves(crystal, heading);
        let velocity = energy_flow(crystal, heading, &polarization, DENSITY[m]);
        for (v, value) in velocity.iter().zip(values) {
            let phase = (value / DENSITY[m]).sqrt();
            debug_assert!(((*v | heading).s() - phase).abs() <= 1e-8 * phase);
        }
        velocity
    }

    /// A small xorshift generator and Box-Muller normals: numga's NumPy stream cannot be
    /// reproduced, and nothing here depends on it beyond spreading headings evenly.
    pub struct Rng(u64);
    impl Rng {
        pub fn new(seed: u64) -> Rng {
            Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
        }
        pub fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        pub fn normal(&mut self) -> f64 {
            let u = self.uniform().max(1e-300);
            let v = self.uniform();
            (-2.0 * u.ln()).sqrt() * (core::f64::consts::TAU * v).cos()
        }
        /// A heading uniform on the sphere: a normalized Gaussian vector.
        pub fn heading(&mut self) -> V {
            let v: V = Vector::new(self.normal(), self.normal(), self.normal());
            v.normalized().into_inner()
        }
    }

    /// Where each wave's energy meets the cube face one unit along `z`, for `count` headings
    /// spread over the sphere: per material and wave, the points of the rays that go up.
    pub fn focusing(count: usize, seed: u64) -> [[Vec<[f32; 2]>; 3]; 3] {
        let crystals = crystals();
        let mut rng = Rng::new(seed);
        let headings: Vec<V> = (0..count).map(|_| rng.heading()).collect();
        core::array::from_fn(|m| {
            let mut out: [Vec<[f32; 2]>; 3] = Default::default();
            for h in &headings {
                for (w, v) in group_velocities(&crystals[m], m, *h).iter().enumerate() {
                    if v.e3() > 0.0 {
                        out[w].push([(v.e1() / v.e3()) as f32, (v.e2() / v.e3()) as f32]);
                    }
                }
            }
            out
        })
    }

    /// The heading at `angle` in the cube face `z = 0`: `x` turned by the angle.
    pub fn face_heading(angle: f64) -> V {
        let r = gax::vga3d::Bivector::<(), f64>::new(0.0, 0.0, -angle / 2.0).exp();
        r >> Vector::new(1.0, 0.0, 0.0)
    }

    /// Group velocities for `count` headings in the cube face: where each wave's energy is after
    /// unit time, a section through its wave surface. The waves are ordered by speed, so where
    /// the two shear speeds cross they swap polarizations: each curve follows a speed rank. With
    /// numga's checks of the textbook speeds along a cube edge and a face diagonal.
    pub fn wave_fronts(count: usize) -> [Vec<[V; 3]>; 3] {
        let crystals = crystals();
        for m in 0..3 {
            let speeds = |h: V| waves(&crystals[m], h).0.map(|v| (v / DENSITY[m]).sqrt());
            let edge = speeds(Vector::new(1.0, 0.0, 0.0));
            let expected = [C44[m], C44[m], C11[m]].map(|c| (c / DENSITY[m]).sqrt());
            for (a, b) in edge.iter().zip(expected) {
                assert!((a - b).abs() <= 1e-8 * b);
            }
            let diagonal = speeds(Vector::new(1.0, 1.0, 0.0).normalized().into_inner());
            let mut expected = [
                (C11[m] - C12[m]) / 2.0,
                C44[m],
                (C11[m] + C12[m] + 2.0 * C44[m]) / 2.0,
            ];
            expected.sort_by(f64::total_cmp);
            for (a, b) in diagonal.iter().zip(expected) {
                let b = (b / DENSITY[m]).sqrt();
                assert!((a - b).abs() <= 1e-8 * b);
            }
        }
        core::array::from_fn(|m| {
            (0..count)
                .map(|k| {
                    let h = face_heading(core::f64::consts::TAU * k as f64 / count as f64);
                    group_velocities(&crystals[m], m, h)
                })
                .collect()
        })
    }
}

use crystal::*;

/// Headings binned by the end of the loop (fewer in tests, which only check that frames draw).
const HEADINGS: usize = if cfg!(test) { 3000 } else { 240_000 };
/// Seconds per crystal of the sweep; the focusing images build over all three.
const SWEEP: f32 = 5.0;
/// Histogram bins across a panel, over `[-1.5, 1.5]`.
const BINS: usize = 110;
/// Headings in the cube face for the wave surfaces.
const FRONT: usize = 720;

struct Data {
    focus: [[Vec<[f32; 2]>; 3]; 3],
    fronts: [Vec<[V; 3]>; 3],
}

fn data() -> &'static Data {
    static D: OnceLock<Data> = OnceLock::new();
    D.get_or_init(|| Data {
        focus: focusing(HEADINGS, 7),
        fronts: wave_fronts(FRONT),
    })
}

/// A 2D histogram of `pts` over `[-1.5, 1.5]²`.
fn histogram(pts: &[[f32; 2]]) -> Vec<u32> {
    let mut h = vec![0u32; BINS * BINS];
    for p in pts {
        let i = ((p[0] + 1.5) / 3.0 * BINS as f32).floor();
        let j = ((p[1] + 1.5) / 3.0 * BINS as f32).floor();
        if (0.0..BINS as f32).contains(&i) && (0.0..BINS as f32).contains(&j) {
            h[j as usize * BINS + i as usize] += 1;
        }
    }
    h
}

fn mode_colour(w: usize) -> Rgb {
    [palette::red(), palette::sky(), palette::purple()][w]
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.width as f32, c.height as f32);
    let d = data();
    let total = 3.0 * SWEEP;
    let u = t.rem_euclid(total) / total;
    let current = ((u * 3.0) as usize).min(2);
    // The focusing images: a growing share of the headings, faster at first.
    let share = u.sqrt().max(0.02);
    let top = h * 0.17;
    let left = w * 0.11;
    let size = ((h - top - h * 0.03) / 3.0).min((w * 0.56 - left) / 3.0);
    for (m, name) in NAMES.iter().enumerate() {
        for (wv, mode) in MODES.iter().enumerate() {
            let rect = [
                left + wv as f32 * size,
                top + m as f32 * size,
                left + (wv + 1) as f32 * size,
                top + (m + 1) as f32 * size,
            ];
            let rect = plot::inset(rect, 2.0, 2.0, 2.0, 2.0);
            let ax = Axes::new(rect, [-1.5, 1.5], [-1.5, 1.5]);
            let pts = &d.focus[m][wv];
            let n = ((pts.len() as f32 * share) as usize).min(pts.len());
            let hist = histogram(&pts[..n]);
            let peak = (hist.iter().copied().max().unwrap_or(1).max(1) as f32).ln_1p();
            ax.image(c, 1, |x, y| {
                let i = ((x + 1.5) / 3.0 * BINS as f32).floor() as usize;
                let j = ((y + 1.5) / 3.0 * BINS as f32).floor() as usize;
                let v = hist[j.min(BINS - 1) * BINS + i.min(BINS - 1)] as f32;
                Some(colormap::inferno(v.ln_1p() / peak))
            });
            if m == current {
                c.polyline(
                    &[
                        [rect[0], rect[1]],
                        [rect[2], rect[1]],
                        [rect[2], rect[3]],
                        [rect[0], rect[3]],
                    ],
                    1.5,
                    palette::yellow(),
                    0.9,
                    true,
                );
            }
            if m == 0 {
                c.text(
                    mode,
                    (rect[0] + rect[2]) * 0.5,
                    rect[1] - 6.0,
                    10.0,
                    mode_colour(wv),
                    Align::Center,
                );
            }
        }
        c.text(
            name,
            left - 6.0,
            top + (m as f32 + 0.5) * size,
            10.0,
            if m == current {
                palette::yellow()
            } else {
                palette::ink()
            },
            Align::Right,
        );
    }
    c.text(
        &format!(
            "{} HEADINGS",
            ((HEADINGS as f32 * share) as usize / 1000) * 1000
        ),
        left,
        h - h * 0.005,
        10.0,
        palette::grid(),
        Align::Left,
    );

    // The wave surfaces of the current crystal in the cube face, and the sweeping heading.
    let fronts = &d.fronts[current];
    let reach = fronts
        .iter()
        .flatten()
        .map(|v| v.norm())
        .fold(0.0, f64::max) as f32;
    let rect = plot::inset([w * 0.58, 0.0, w, h], w * 0.05, h * 0.2, w * 0.03, h * 0.1);
    let ax = Axes::equal(rect, [0.0, 0.0], reach * 1.12);
    ax.frame(c, NAMES[current], "KM/S ALONG (100)", "");
    for wv in 0..3 {
        let pts: Vec<[f32; 2]> = fronts
            .iter()
            .map(|v| [v[wv].e1() as f32, v[wv].e2() as f32])
            .collect();
        ax.scatter(c, &pts, Marker::Dot, 1.6, mode_colour(wv), 0.8);
    }
    let s = (u * 3.0).fract();
    let angle = f64::from(s) * core::f64::consts::TAU;
    let heading = face_heading(angle);
    let crystals = crystals();
    let (values, _) = waves(&crystals[current], heading);
    let group = d.fronts[current][((s * FRONT as f32) as usize).min(FRONT - 1)];
    let hx = [heading.e1() as f32, heading.e2() as f32];
    ax.line(
        c,
        [0.0, 0.0],
        [hx[0] * reach * 1.1, hx[1] * reach * 1.1],
        1.0,
        palette::ink(),
        0.6,
    );
    for wv in 0..3 {
        // The phase velocity lies along the heading; the energy goes along the group velocity.
        let phase = (values[wv] / DENSITY[current]).sqrt() as f32;
        ax.scatter(
            c,
            &[[hx[0] * phase, hx[1] * phase]],
            Marker::Ring,
            8.0,
            mode_colour(wv),
            1.0,
        );
        ax.arrow(
            c,
            [0.0, 0.0],
            [group[wv].e1() as f32, group[wv].e2() as f32],
            2.0,
            9.0,
            mode_colour(wv),
        );
    }
    let legend: Vec<(&str, Rgb)> = MODES.iter().copied().zip((0..3).map(mode_colour)).collect();
    ax.legend(c, &legend);
    caption(
        c,
        "CRYSTAL WAVES: PHONON FOCUSING IN CUBIC CRYSTALS",
        "ENERGY OF EVENLY SPREAD HEADINGS ON A CUBE FACE; RIGHT: GROUP VELOCITIES (ARROWS) AND PHASE (RINGS) (VGA3D)",
    );
}

fn main() {
    run(Anim::new("crystal waves", 3.0 * SWEEP).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::crystal::*;

    /// The traction's components C11, C12 and C44 on the cube's faces: a stretch along x pulls on
    /// the x face with C11 and on the y face with C12, and a shear of the xy face pulls on it
    /// with C44.
    #[test]
    fn stiffness_reads_back_the_elastic_constants() {
        let [x, y, _] = axes();
        for (m, s) in crystals().iter().enumerate() {
            let read = [
                (x | s.of(x).of(x).of(x)).s(),
                (y | s.of(y).of(x).of(x)).s(),
                (x | s.of(y).of(x).of(y)).s(),
            ];
            for (r, c) in read.iter().zip([C11[m], C12[m], C44[m]]) {
                assert!((r - c).abs() <= 1e-10 * c, "{read:?}");
            }
        }
    }

    /// In fused silica every wave carries its energy along its heading and the two shear waves
    /// share one speed; in silicon the shear energy swings away from the heading. (numga draws
    /// 64 headings from NumPy's stream; these come from a local generator.)
    #[test]
    fn energy_follows_the_wave_only_when_isotropic() {
        let crystals = crystals();
        let mut rng = Rng::new(1);
        let headings: Vec<V> = (0..64).map(|_| rng.heading()).collect();
        let mut swing = [0.0f64; 3];
        for (m, crystal) in crystals.iter().enumerate() {
            for h in &headings {
                let (values, polarization) = waves(crystal, *h);
                let velocity = energy_flow(crystal, *h, &polarization, DENSITY[m]);
                for v in velocity {
                    let along = (v | *h).s() / v.norm();
                    swing[m] = swing[m].max(along.clamp(-1.0, 1.0).acos());
                }
                if m == 0 {
                    assert!((values[0] - values[1]).abs() <= 1e-8 * values[1]);
                }
            }
        }
        assert!(swing[0] < 1e-5 && swing[1] > 0.2, "{swing:?}");
    }

    /// The scenarios' checks: the phase speed along the heading, and the textbook speeds.
    #[test]
    fn scenario_checks() {
        let crystals = crystals();
        let mut rng = Rng::new(0);
        for _ in 0..200 {
            let h = rng.heading();
            for (m, crystal) in crystals.iter().enumerate() {
                let (values, polarization) = waves(crystal, h);
                let velocity = energy_flow(crystal, h, &polarization, DENSITY[m]);
                for (v, value) in velocity.iter().zip(values) {
                    let phase = (value / DENSITY[m]).sqrt();
                    assert!(((*v | h).s() - phase).abs() <= 1e-8 * phase);
                }
            }
        }
        let fronts = wave_fronts(64);
        assert!(fronts.iter().all(|f| f.len() == 64));
    }

    #[test]
    fn a_frame_draws() {
        let mut draw = super::draw;
        let c = gax_numga_examples::app::frame(
            &gax_numga_examples::Anim::new("t", 1.0).size(320, 180),
            0.5,
            &mut draw,
        );
        assert!(c.mean()[0] > 0.0);
    }
}
