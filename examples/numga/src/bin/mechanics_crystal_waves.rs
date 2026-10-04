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

use gax::pga2d;
use gax::vga3d::{Bivector, Scalar, Vector};

use gax_numga_examples::points::{Map2, box_map};
use gax_numga_examples::rng::{Draw, rng};
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, ORIGIN2, Point2, Pos2, Rect, backdrop, caption,
    colormap, font, palette, run,
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

    /// The phase speed of a wave from the Christoffel map's eigenvalue, density times squared
    /// speed: `sqrt(value / density)`, the law of elastic waves (as `sqrt(k / m)` is a spring's).
    #[allow(clippy::disallowed_methods)] // a law of elastic waves, the speed sqrt(c / ρ)
    pub fn phase_speed(value: f64, density: f64) -> f64 {
        (value / density).sqrt()
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
    /// velocity, whose component along the heading is the phase speed. (The polarization's
    /// pairing is density times squared phase speed.)
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
                .gp(1.0 / (density * phase_speed(squared, density)))
        })
    }

    /// The group velocities of the three waves for one heading, in material `m`, with the
    /// checks of numga's `focusing`: along its heading the group velocity has the phase speed.
    pub fn group_velocities(crystal: &Stiffness, m: usize, heading: V) -> [V; 3] {
        let (values, polarization) = waves(crystal, heading);
        let velocity = energy_flow(crystal, heading, &polarization, DENSITY[m]);
        for (v, value) in velocity.iter().zip(values) {
            let phase = phase_speed(value, DENSITY[m]);
            debug_assert!(((*v | heading).s() - phase).abs() <= 1e-8 * phase);
        }
        velocity
    }

    /// `count` headings spread uniformly over the sphere.
    pub fn headings(count: usize, seed: u64) -> Vec<V> {
        let mut r = rng(seed);
        (0..count).map(|_| r.direction()).collect()
    }

    /// Where each wave's energy meets the cube face one unit along `z`, for `count` headings
    /// spread over the sphere: per material and wave, the points of the rays that go up. A
    /// direction read as homogeneous coordinates `(x, y, z)` of the plane is the point where its
    /// ray meets `z = 1`.
    pub fn focusing(count: usize, seed: u64) -> [[Vec<Point2>; 3]; 3] {
        let crystals = crystals();
        let headings = headings(count, seed);
        core::array::from_fn(|m| {
            let mut out: [Vec<Point2>; 3] = Default::default();
            for h in &headings {
                for (w, v) in group_velocities(&crystals[m], m, *h).iter().enumerate() {
                    if v.e3() > 0.0 {
                        let hit = pga2d::Point::new(v.e1(), v.e2(), v.e3());
                        out[w].push(hit.unitized().point2());
                    }
                }
            }
            out
        })
    }

    /// The heading at `angle` in the cube face `z = 0`: `x` turned by the angle.
    pub fn face_heading(angle: f64) -> V {
        let r = Bivector::<(), f64>::new(0.0, 0.0, -angle / 2.0).exp();
        r >> Vector::new(1.0, 0.0, 0.0)
    }

    /// Group velocities for `count` headings in the cube face: where each wave's energy is after
    /// unit time, a section through its wave surface. The waves are ordered by speed, so where
    /// the two shear speeds cross they swap polarizations: each curve follows a speed rank. With
    /// numga's checks of the textbook speeds along a cube edge and a face diagonal.
    pub fn wave_fronts(count: usize) -> [Vec<[V; 3]>; 3] {
        let crystals = crystals();
        for m in 0..3 {
            let speeds = |h: V| waves(&crystals[m], h).0.map(|v| phase_speed(v, DENSITY[m]));
            let edge = speeds(Vector::new(1.0, 0.0, 0.0));
            let expected = [C44[m], C44[m], C11[m]].map(|c| phase_speed(c, DENSITY[m]));
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
                let b = phase_speed(b, DENSITY[m]);
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
    focus: [[Vec<Point2>; 3]; 3],
    fronts: [Vec<[V; 3]>; 3],
}

fn data() -> &'static Data {
    static D: OnceLock<Data> = OnceLock::new();
    D.get_or_init(|| Data {
        focus: focusing(HEADINGS, 7),
        fronts: wave_fronts(FRONT),
    })
}

/// The square `[-1.5, 1.5]²` onto the histogram's grid of `BINS` x `BINS` unit cells.
fn to_grid() -> Map2 {
    let square = [Point2::xy(-1.5, -1.5), Point2::xy(1.5, 1.5)];
    box_map(square, [ORIGIN2, Point2::xy(BINS as f32, BINS as f32)])
}

/// The column and row of the grid cell a point falls in (outside the grid too).
fn cell(grid: &Map2, p: Point2) -> [f32; 2] {
    grid.of(p).to_euclidean().map(f32::floor)
}

/// A 2D histogram of `pts` over `[-1.5, 1.5]²`.
fn histogram(pts: &[Point2]) -> Vec<u32> {
    let grid = to_grid();
    let mut h = vec![0u32; BINS * BINS];
    for p in pts {
        let [i, j] = cell(&grid, *p);
        if (0.0..BINS as f32).contains(&i) && (0.0..BINS as f32).contains(&j) {
            h[j as usize * BINS + i as usize] += 1;
        }
    }
    h
}

/// A vector in the cube face `z = 0`, as plotted: the point at its tip, from its `x` and `y`.
fn face(v: V) -> pga2d::Point<(), f64> {
    pga2d::Point::xy(v.e1(), v.e2())
}

fn mode_colour(w: usize) -> Light {
    [palette::red(), palette::sky(), palette::purple()][w]
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    // Lengths in pixels at 960 by 540, scaled with the canvas.
    let unit = c.unit();
    let d = data();
    let total = 3.0 * SWEEP;
    let u = t.rem_euclid(total) / total;
    let current = ((u * 3.0) as usize).min(2);
    // The focusing images: a growing share of the headings, faster at first.
    #[allow(clippy::disallowed_methods)] // the animation's pace, a timing curve
    let share = u.sqrt().max(0.02);
    let top = h * 0.17;
    let left = w * 0.13;
    let size = ((h - top - h * 0.03) / 3.0).min((w * 0.56 - left) / 3.0);
    // The square of crystal `m` and mode `wv`: rows of crystals, columns of modes.
    let square = |m: usize, wv: usize| {
        let lo = Point2::xy(left, top) + Point2::direction(wv as f32, m as f32).gp(size);
        Rect {
            lo,
            hi: lo + Point2::direction(size, size),
        }
    };
    let (up, right) = (Point2::direction(0.0, -1.0), Point2::direction(1.0, 0.0));
    let grid = to_grid();
    for (m, name) in NAMES.iter().enumerate() {
        for (wv, mode) in MODES.iter().enumerate() {
            let rect = square(m, wv).inset(2.0 * unit, 2.0 * unit, 2.0 * unit, 2.0 * unit);
            let ax = Axes::new(rect, [-1.5, 1.5], [-1.5, 1.5]);
            let pts = &d.focus[m][wv];
            let n = ((pts.len() as f32 * share) as usize).min(pts.len());
            let hist = histogram(&pts[..n]);
            let peak = (hist.iter().copied().max().unwrap_or(1).max(1) as f32).ln_1p();
            ax.image(c, 1, |p| {
                let [i, j] = cell(&grid, p).map(|k| k as usize);
                let v = hist[j.min(BINS - 1) * BINS + i.min(BINS - 1)] as f32;
                Some(colormap::inferno(v.ln_1p() / peak))
            });
            if m == current {
                let corners = [rect.lo, rect.top_right(), rect.hi, rect.bottom_left()];
                c.polyline(&corners, 1.5, palette::yellow().faded(0.9), true);
            }
            if m == 0 {
                let above = rect.top_middle() + up.gp(6.0 * unit);
                c.text(mode, above, 10.0 * unit, mode_colour(wv), Align::Center);
            }
        }
        // The crystal's name left of the middle of its row, a word a line where it would not
        // fit in the margin with room to spare.
        let row = square(m, 0);
        let at = row.left_middle() - right.gp(6.0 * unit);
        let tone = if m == current {
            palette::yellow()
        } else {
            palette::ink()
        };
        let words: Vec<&str> = if font::width(name, 10.0 * unit) + 16.0 * unit > left {
            name.split(' ').collect()
        } else {
            vec![name]
        };
        let middle = (words.len() as f32 - 1.0) / 2.0;
        for (i, word) in words.iter().enumerate() {
            let line = at - up.gp(13.0 * unit * (i as f32 - middle));
            c.text(word, line, 10.0 * unit, tone, Align::Right);
        }
    }
    let count = ((HEADINGS as f32 * share) as usize / 1000) * 1000;
    let at = screen.bottom_left() + Point2::direction(left, -h * 0.005);
    let text = format!("{count} HEADINGS");
    c.text(&text, at, 10.0 * unit, palette::grid(), Align::Left);

    // The wave surfaces of the current crystal in the cube face, and the sweeping heading.
    let fronts = &d.fronts[current];
    let reach = fronts
        .iter()
        .flatten()
        .map(|v| v.norm())
        .fold(0.0, f64::max) as f32;
    let rect = Rect::new(w * 0.58, 0.0, w, h).inset(w * 0.05, h * 0.2, w * 0.03, h * 0.1);
    let origin = ORIGIN2;
    let ax = Axes::equal(rect, origin, reach * 1.12);
    ax.frame(c, NAMES[current], "KM/S ALONG (100)", "");
    for wv in 0..3 {
        let pts: Vec<pga2d::Point<(), f64>> = fronts.iter().map(|v| face(v[wv])).collect();
        // The dots lie closer than their glow reaches, so their light adds up along the
        // surface: each is faint.
        ax.scatter(c, &pts, Marker::Dot, 1.6, (mode_colour(wv)).faded(0.25));
    }
    let s = (u * 3.0).fract();
    let angle = f64::from(s) * core::f64::consts::TAU;
    let heading = face_heading(angle);
    let crystals = crystals();
    let (values, _) = waves(&crystals[current], heading);
    let group = d.fronts[current][((s * FRONT as f32) as usize).min(FRONT - 1)];
    let tip = face(heading.gp(f64::from(reach) * 1.1));
    ax.line(c, origin, tip, 1.0, palette::ink().faded(0.6));
    for wv in 0..3 {
        // The phase velocity lies along the heading; the energy goes along the group velocity.
        let phase = phase_speed(values[wv], DENSITY[current]);
        let ring = face(heading.gp(phase));
        ax.scatter(c, &[ring], Marker::Ring, 8.0, mode_colour(wv));
        ax.arrow(c, origin, face(group[wv]), 2.0, 9.0, mode_colour(wv));
    }
    let legend: Vec<(&str, Light)> = MODES.iter().copied().zip((0..3).map(mode_colour)).collect();
    ax.legend(c, &legend);
    caption(
        c,
        "CRYSTAL WAVES: PHONON FOCUSING IN CUBIC CRYSTALS",
        "LEFT: ENERGY ON A CUBE FACE. RIGHT: GROUP VELOCITY, PHASE (RINGS) (VGA3D)",
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
    /// 64 headings from NumPy's stream; these come from `rand`'s.)
    #[test]
    fn energy_follows_the_wave_only_when_isotropic() {
        let crystals = crystals();
        let headings = headings(64, 1);
        let mut swing = [0.0f64; 3];
        for (m, crystal) in crystals.iter().enumerate() {
            for h in &headings {
                let (values, polarization) = waves(crystal, *h);
                let velocity = energy_flow(crystal, *h, &polarization, DENSITY[m]);
                for v in velocity {
                    swing[m] = swing[m].max(gax_numga_examples::measure::angle(*h, v));
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
        for h in headings(200, 0) {
            for (m, crystal) in crystals.iter().enumerate() {
                let (values, polarization) = waves(crystal, h);
                let velocity = energy_flow(crystal, h, &polarization, DENSITY[m]);
                for (v, value) in velocity.iter().zip(values) {
                    let phase = phase_speed(value, DENSITY[m]);
                    assert!(((*v | h).s() - phase).abs() <= 1e-8 * phase);
                }
            }
        }
        let fronts = wave_fronts(64);
        assert!(fronts.iter().all(|f| f.len() == 64));
    }

    /// The angle between headings in the cube face is the angle that turned them apart.
    #[test]
    fn angles_between_vectors() {
        for a in [0.0, 0.4, 1.9, 3.1] {
            let b = 0.25;
            let angle =
                gax_numga_examples::measure::angle(face_heading(b), face_heading(a + b).gp(2.5));
            assert!((angle - a).abs() < 1e-12, "{a} {angle}");
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
