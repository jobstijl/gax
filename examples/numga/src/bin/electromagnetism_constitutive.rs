//! numga's `electromagnetism/constitutive`: materials as constitutive maps in spacetime algebra.
//! A medium maps the field, a bivector `F`, to the excitation it causes, an antibivector
//! `G = χ(F)` (in four dimensions a bivector of the other orientation). A medium at rest responds
//! to each part of the field an observer tells apart with a dyad: the patch that reads that part,
//! times its reading `F ∨ patch` of the field, weighted by the material constant. The planes each
//! axis of space sweeps through time read the electric field and carry the permittivities; the
//! patches of the planes of space read the magnetic field and carry the inverse permeabilities.
//! Glass weights all three of each kind alike, a crystal its electric dyads apart, a ferrite its
//! magnetic ones; the metric enters only as the material map itself. Seen moving, a medium is its
//! rest map carried by the boost, and an axion response adds a multiple of the field itself.
//! Pairing the excitation with a second field gives the constitutive form, and the field and
//! excitation together give the stress-energy map, from directions to currents.
//!
//! A plane wave with wavevector `k` solves both source-free Maxwell equations, `k ∧ F = 0` and
//! `k ∧ χ(F) = 0`. Packed into one map `F ↦ k ∧ F + (k ∧ χ(F))*` from the six field
//! coefficients to the eight of the odd subalgebra, a wave exists where that map has a
//! nullspace: its smallest singular value dips to zero. Scanning the phase speed finds the
//! dispersion of each medium, scanning the heading too traces the Fresnel surfaces, and scanning
//! the speed of moving glass shows Fresnel drag, exactly Einstein's addition of speeds.
//!
//! The animation shows the travelling fields in glass (one speed: the polarization stays put)
//! and in a birefringent crystal (two speeds: it turns as the slow and fast modes slip), the
//! electric field as arrows and the magnetic field as discs in its own plane, with cursors
//! sweeping the dispersion scans, the Fresnel surfaces and the drag curve.

use gax::pga2d::Point;
use gax::sta::Bivector;

use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Light, Marker, Point2, Rect, Scene3, backdrop, caption,
    palette, run_with,
    signal::{haversine, phasor},
};

mod constitutive {
    // numga's API in full: the tests use parts the animation does not.
    #![cfg_attr(not(test), allow(dead_code))]

    use gax::Orientation;
    use gax::sta::{Bivector, Odd, Vector, intrinsic, oriented};
    use gax_numga_examples::parallel::par_map;
    use gax_numga_examples::signal::linspace;

    /// An excitation, a field's response: an antibivector (a bivector, intrinsic).
    pub type Excitation = intrinsic::Bivector;
    /// A map on fields, `Bivector <- Bivector`.
    pub type Projector = Bivector<(Bivector,)>;
    /// A constitutive map, `Antibivector <- Bivector`.
    pub type Medium = intrinsic::Bivector<(Bivector,)>;
    /// An axion coupling is handed: an intrinsic scalar (numga's antipseudoscalar).
    pub type AxionCoupling = intrinsic::Scalar;
    /// A direction in spacetime as numga writes it, an antivector (a trivector, intrinsic).
    pub type Direction = intrinsic::Trivector;
    /// The stress-energy map, from directions to the currents along them, `Antivector <- Antivector`.
    pub type StressEnergy = intrinsic::Trivector<(Direction,)>;
    /// Both Maxwell residuals of a plane wave, `Odd <- Bivector`.
    pub type WaveMap = Odd<(Bivector,)>;
    pub type Rotor = gax::Unit<gax::sta::Even>;
    /// A vector of the rest observer's space, for drawing.
    pub type Space = gax::vga3d::Vector;
    /// Waves along one heading: their speeds and the fields they carry.
    pub type Modes = ([f64; 2], [Bivector; 2]);

    /// The open field.
    pub fn open() -> Projector {
        Bivector::slot()
    }

    /// The open direction.
    pub fn open_direction() -> StressEnergy {
        Direction::slot()
    }

    pub fn x() -> Vector {
        Vector::new(1.0, 0.0, 0.0, 0.0)
    }
    pub fn y() -> Vector {
        Vector::new(0.0, 1.0, 0.0, 0.0)
    }
    pub fn z() -> Vector {
        Vector::new(0.0, 0.0, 1.0, 0.0)
    }
    pub fn t() -> Vector {
        Vector::new(0.0, 0.0, 0.0, 1.0)
    }

    /// The observer's direction in time, which pairs to one with `t`: `-t*`, the Hodge dual
    /// (numga's `dual`) carrying the minus of `t`'s square.
    pub fn time() -> Direction {
        -t().hodge()
    }

    /// The planes of space across x, y and z: `yz, zx, xy`.
    pub fn magnetic_planes() -> [Bivector; 3] {
        [
            Bivector::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Bivector::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            Bivector::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        ]
    }

    /// The planes each axis of space sweeps through time, `x* ∨ time`, which read the electric
    /// field.
    pub fn swept_planes() -> [Excitation; 3] {
        [x(), y(), z()].map(|axis| axis.hodge() & time())
    }

    /// The patches of the planes of space, which read the magnetic field.
    pub fn patches() -> [Excitation; 3] {
        magnetic_planes().map(|plane| plane.hodge())
    }

    /// A medium at rest, from its permittivity along each axis and its inverse permeability
    /// across each: the dyads of the patches that read the field, each weighted by its material
    /// constant. Time squares to minus one, so the electric dyads enter with a minus: the sign
    /// that makes the Lagrangian the electric field squared less the magnetic.
    pub fn medium(permittivities: [f64; 3], inverse_permeabilities: [f64; 3]) -> Medium {
        let dyad = |patch: Excitation| patch * (open() & patch);
        let electric: Medium = swept_planes()
            .into_iter()
            .zip(permittivities)
            .map(|(p, e)| dyad(p).gp(e))
            .sum();
        let magnetic: Medium = patches()
            .into_iter()
            .zip(inverse_permeabilities)
            .map(|(p, m)| dyad(p).gp(m))
            .sum();
        magnetic - electric
    }

    /// The electric part of a field as a unit timelike observer `u` tells it apart,
    /// `(F - u F u⁻¹) / 2`: the observer's reflection keeps magnetic planes and reverses electric
    /// ones. gax's sandwich `u F ~u` is `-u F u⁻¹` for `u² = -1`.
    pub fn electric_part(observer: Vector) -> Projector {
        (open() + (observer >> open())).gp(0.5)
    }

    /// The electromagnetic stress-energy: from a direction in spacetime, an antivector, to the
    /// current of energy and momentum along it, intrinsic like the direction. A unit timelike
    /// observer's slice of space `t`, paired with the current along its direction in time,
    /// `t ∨ current(time)`, is the energy density it measures. In matter this is the
    /// electromagnetic Minkowski current, excluding the material's own stress-energy.
    pub fn stress_energy(field: Bivector, excitation: Excitation) -> StressEnergy {
        // An electric plane pairs with its own dual to minus its square, so the field paired
        // with its excitation is minus twice the Lagrangian.
        let negative_lagrangian = (field & excitation).s() / 2.0;
        // Minus the Lagrangian along the direction, less the field read along the direction
        // wedged with the excitation: no metric enters.
        open_direction().gp(negative_lagrangian) - ((open_direction() & field) ^ excitation)
    }

    /// Both exterior Maxwell residuals, packed into separate grades: `k ∧ F` is a trivector
    /// and the dual of `k ∧ χ(F)` a vector, so they cannot cancel. gax adds values of one
    /// kind, so both are cast into the odd subalgebra.
    pub fn wave_map(k: Vector, medium: Medium) -> WaveMap {
        (k ^ open()).cast::<Odd>() + (k ^ medium).hodge().cast::<Odd>()
    }

    /// How far the nearest field is from a wave along the wavevector: the wave map's smallest
    /// singular value, zero where a wave travels.
    pub fn residual(k: Vector, medium: Medium) -> f64 {
        wave_map(k, medium).svd().values[5]
    }

    /// The field a wave carries: the wave map's last right singular vector.
    pub fn field_mode(k: Vector, medium: Medium) -> Bivector {
        wave_map(k, medium).svd().inputs[5]
    }

    /// Where the residuals along the trial speeds fall to a local minimum under the threshold:
    /// the speeds at which waves travel, as a mask over the interior speeds.
    pub fn dips(residual: &[f64], threshold: f64) -> Vec<bool> {
        (1..residual.len() - 1)
            .map(|i| {
                let m = residual[i];
                m <= residual[i - 1] && m <= residual[i + 1] && m < threshold
            })
            .collect()
    }

    /// The speeds at the dips.
    pub fn dip_speeds(speeds: &[f64], residual: &[f64], threshold: f64) -> Vec<f64> {
        dips(residual, threshold)
            .into_iter()
            .zip(&speeds[1..])
            .filter_map(|(d, s)| d.then_some(*s))
            .collect()
    }

    /// The speed at which the residual is smallest.
    pub fn deepest(speeds: &[f64], residual: &[f64]) -> f64 {
        let k = (0..residual.len())
            .min_by(|&a, &b| residual[a].total_cmp(&residual[b]))
            .expect("a scan");
        speeds[k]
    }

    /// The spatial part of a vector as the rest observer `t` reads it, a vector of space (for
    /// drawing, of either orientation).
    pub fn spatial<O: Orientation>(v: oriented::Vector<(), f64, O>) -> Space {
        Space::new(v.e1(), v.e2(), v.e3())
    }

    /// The electric and magnetic vectors a rest observer reads off a field: `F · t` and
    /// `-F* · t`, so that `E x B` points along the wave's travel.
    pub fn arrows(f: Bivector) -> (Space, Space) {
        (spatial(f | t()), spatial(-f.hodge() | t()))
    }

    // --- scenes -------------------------------------------------------------------------

    /// The media at rest, glass, a crystal and a ferrite: the permittivity along x, y and z,
    /// and the inverse permeability across each.
    pub const PERMITTIVITIES: [[f64; 3]; 3] = [[2.25; 3], [2.25, 1.5, 1.5], [2.25; 3]];
    pub const INVERSE_PERMEABILITIES: [[f64; 3]; 3] = [[1.0; 3], [1.0; 3], [1.0, 0.5, 1.0]];
    pub const NAMES: [&str; 3] = ["GLASS", "CRYSTAL", "FERRITE"];
    /// The glass's axion coupling, and the speed of the glass in motion as a fraction of the
    /// speed of light.
    pub const AXION_COUPLING: f64 = 0.4;
    pub const GLASS_SPEED: f64 = 0.35;
    /// The trial speeds of waves along z, and the residual under which a local minimum counts
    /// as a wave.
    pub const SPEED_RANGE: (f64, f64) = (0.05, 1.5);
    pub const N_SPEEDS: usize = 3001;
    pub const THRESHOLD: f64 = 4e-3;
    /// The headings through the xz plane, and the trial speeds along each.
    pub const N_HEADINGS: usize = 72;
    pub const HEADING_SPEED_RANGE: (f64, f64) = (0.4, 1.0);
    pub const N_HEADING_SPEEDS: usize = 601;
    /// Light with the glass and against it.
    pub const SENSES: [f64; 2] = [1.0, -1.0];
    /// The speeds of the glass for the drag, the trial speeds of light through it, and the
    /// samples of the analytic curves.
    pub const DRAG_RANGE: (f64, f64) = (-0.6, 0.6);
    pub const N_DRAGS: usize = 15;
    pub const DRAG_SPEED_RANGE: (f64, f64) = (0.05, 1.2);
    pub const N_DRAG_SPEEDS: usize = 1151;
    pub const N_CURVE: usize = 200;

    /// Glass, the crystal and the ferrite at rest.
    pub fn media() -> [Medium; 3] {
        core::array::from_fn(|i| medium(PERMITTIVITIES[i], INVERSE_PERMEABILITIES[i]))
    }

    /// The speeds of the two waves each medium carries along z: the electric field along x
    /// feels the permittivity along x, and its magnetic plane zx the inverse permeability
    /// across y; the electric field along y the other two.
    #[allow(clippy::disallowed_methods)] // wave speeds √(1/(εμ)): a physical law, not geometry
    pub fn expected_speeds() -> [[f64; 2]; 3] {
        let (e, m) = (PERMITTIVITIES, INVERSE_PERMEABILITIES);
        core::array::from_fn(|i| [(m[i][1] / e[i][0]).sqrt(), (m[i][0] / e[i][1]).sqrt()])
    }

    /// numga's boost, `exp((z ∧ t) atanh(β) / 2)`: to a fraction of the speed of light along
    /// -z (gax's `Even::boost(z, ζ)` is along +z), the way numga's waves along z travel.
    pub fn boost(fraction: f64) -> Rotor {
        (z() ^ t()).gp(fraction.atanh() / 2.0).exp()
    }

    /// The medium seen moving along -z (numga's "along z", see `modes`) at a fraction of the speed of light: its rest map carried
    /// by the boost.
    pub fn moving(medium: Medium, fraction: f64) -> Medium {
        let b = boost(fraction);
        b >> medium.of(b << open())
    }

    /// The speed of light through glass moving at a fraction of the speed of light, with the
    /// glass and against it: the relativistic sum of the two speeds.
    pub fn relativistic_sum(fraction: f64) -> [f64; 2] {
        let light = expected_speeds()[0][0];
        SENSES.map(|s| (light + s * fraction) / (1.0 + light * s * fraction))
    }

    /// The trial speeds along z, each medium's residual at each, and the expected speeds.
    pub fn dispersion() -> (Vec<f64>, [Vec<f64>; 3], [[f64; 2]; 3]) {
        let speeds = linspace(SPEED_RANGE.0, SPEED_RANGE.1, N_SPEEDS);
        let residuals = media().map(|m| {
            speeds
                .iter()
                .map(|s| residual(t().gp(*s) - z(), m))
                .collect()
        });
        (speeds, residuals, expected_speeds())
    }

    /// The fields the glass and the crystal carry along z, with their speeds: the glass two at
    /// one speed, the last two fields of its wave map, and the crystal one at each of its two
    /// speeds.
    ///
    /// numga's wavevectors are `s t - z`. A wave of phase `φ` solves `∇φ ∧ F = 0`, and in
    /// `x+y+z+t-` the gradient of `s T - Z` (a wave along +z) is `-(s t + z)`: the waves of
    /// `s t - z` travel along -z (their `E x B` says so too, tested), as do those of
    /// `s t - heading` along minus the heading. numga's boost `exp((z ∧ t) atanh(β) / 2)` moves
    /// the glass along -z as well, so "with the glass" holds; the drawing shows the waves
    /// travelling the way they do.
    pub fn modes() -> (Modes, Modes) {
        let [glass, crystal, _] = media();
        let [glass_speeds, crystal_speeds, _] = expected_speeds();
        let glass_fields = wave_map(t().gp(glass_speeds[0]) - z(), glass).svd().inputs;
        (
            (glass_speeds, [glass_fields[4], glass_fields[5]]),
            (
                crystal_speeds,
                crystal_speeds.map(|s| field_mode(t().gp(s) - z(), crystal)),
            ),
        )
    }

    /// The slice of space across the heading at `angle` in the xz plane: z turned towards x.
    pub fn heading(angle: f64) -> Vector {
        // The turn from z towards x is exp(-½θ zx), as x and z square to one.
        Bivector::new(0.0, -angle / 2.0, 0.0, 0.0, 0.0, 0.0).exp() >> z()
    }

    /// The headings through the xz plane, the trial speeds, and the residual at every trial
    /// speed along every heading for the glass, the crystal and the glass in motion
    /// `[media][headings][speeds]`.
    pub fn headings() -> (Vec<f64>, Vec<f64>, [Vec<Vec<f64>>; 3]) {
        let angles = linspace(0.0, core::f64::consts::TAU, N_HEADINGS);
        let speeds = linspace(
            HEADING_SPEED_RANGE.0,
            HEADING_SPEED_RANGE.1,
            N_HEADING_SPEEDS,
        );
        let [glass, crystal, _] = media();
        let cases = [glass, crystal, moving(glass, GLASS_SPEED)];
        let residuals = cases.map(|m| {
            par_map(&angles, |a| {
                let slice = heading(*a);
                speeds
                    .iter()
                    .map(|s| residual(t().gp(*s) - slice, m))
                    .collect()
            })
        });
        (angles, speeds, residuals)
    }

    /// The drag of moving glass: the speeds of the glass, and the speeds of light through it,
    /// with the glass and against it, found from the wave map `[senses][drags]`; and on a fine
    /// grid of speeds of the glass, the relativistic sum of the two speeds and Fresnel's
    /// first-order drag `[senses][curve]`.
    pub struct Drag {
        pub fractions: Vec<f64>,
        pub found: [Vec<f64>; 2],
        pub curve: Vec<f64>,
        pub relativistic: [Vec<f64>; 2],
        pub first_order: [Vec<f64>; 2],
    }

    pub fn drag() -> Drag {
        let fractions = linspace(DRAG_RANGE.0, DRAG_RANGE.1, N_DRAGS);
        let speeds = linspace(DRAG_SPEED_RANGE.0, DRAG_SPEED_RANGE.1, N_DRAG_SPEEDS);
        let [glass, _, _] = media();
        let found = SENSES.map(|sense| {
            par_map(&fractions, |f| {
                let m = moving(glass, *f);
                let scan: Vec<f64> = speeds
                    .iter()
                    .map(|s| residual(t().gp(*s) - z().gp(sense), m))
                    .collect();
                deepest(&speeds, &scan)
            })
        });
        let light = expected_speeds()[0][0];
        let curve = linspace(DRAG_RANGE.0, DRAG_RANGE.1, N_CURVE);
        let relativistic = [0, 1].map(|k| curve.iter().map(|f| relativistic_sum(*f)[k]).collect());
        // Fresnel's drag: light gains the fraction 1 - 1 / n² of the glass's speed.
        let first_order = SENSES.map(|s| {
            curve
                .iter()
                .map(|f| light + s * f * (1.0 - light * light))
                .collect()
        });
        Drag {
            fractions,
            found,
            curve,
            relativistic,
            first_order,
        }
    }
}

use constitutive::*;

const SECONDS: f32 = 8.0;
/// A wave's length along the beam, in units where ω = 1.
const Z_MAX: f64 = 4.0 * core::f64::consts::PI;

/// The scans the animation draws, computed once.
struct Scans {
    speeds: Vec<f64>,
    dispersion: [Vec<f64>; 3],
    expected: [[f64; 2]; 3],
    angles: Vec<f64>,
    /// Per medium, per heading, the speeds of its waves.
    sheets: [Vec<Vec<f64>>; 3],
    drag: Drag,
    glass: Modes,
    crystal: Modes,
}

fn scans() -> Scans {
    let (speeds, dispersion, expected) = dispersion();
    let (angles, heading_speeds, residuals) = headings();
    let sheets = residuals.map(|per_heading| {
        per_heading
            .iter()
            .map(|r| dip_speeds(&heading_speeds, r, THRESHOLD))
            .collect()
    });
    let (glass, crystal) = modes();
    Scans {
        speeds,
        dispersion,
        expected,
        angles,
        sheets,
        drag: drag(),
        glass,
        crystal,
    }
}

/// The travelling wave of two modes at distance `z` along their travel and time `tau`: each
/// mode normalized to a unit electric amplitude, summed with its phase `ω (z / v - τ)`.
fn wave_at(modes: &Modes, z: f64, tau: f64) -> Bivector {
    let (speeds, fields) = modes;
    speeds
        .iter()
        .zip(fields)
        .map(|(v, f)| {
            let (e, _) = arrows(*f);
            f.gp(phasor(z / v - tau).e20() / e.norm())
        })
        .sum()
}

/// The wave in 3D, the beam running across the screen along the waves' travel. The electric
/// field is an arrow at each station, the curve its tips trace; the magnetic field, a plane, is
/// a disc in that plane at every other station, its area growing with the field's strength,
/// with an arc round its rim for the sense it turns in.
fn draw_wave(c: &mut Canvas, rect: Rect, modes: &Modes, tau: f64, label: &str) {
    let view = rect.inset(0.0, 20.0, 0.0, 0.0);
    let length = 6.0;
    let cam = Camera::parallel(view, rect.width() * 0.125, -1.1, 0.35);
    c.clip(rect);
    let mut sc = Scene3::new(cam);
    let on_beam = |z: f64| gax::pga3d::Point::xyz(z / Z_MAX * length - length / 2.0, 0.0, 0.0);
    let scale = 0.8;
    // The waves travel along -z (see `modes`): world x runs along -z, world y and z along x
    // and -y, a proper turn, so the drawing keeps its handedness.
    let across =
        |v: Space| gax::pga3d::Point::direction(-v.e3() * scale, v.e1() * scale, -v.e2() * scale);
    sc.seg(on_beam(0.0), on_beam(Z_MAX), 1.0, palette::GRID);
    let samples = 200;
    let e_line: Vec<_> = (0..samples)
        .map(|i| {
            let z = Z_MAX * i as f64 / (samples - 1) as f64;
            on_beam(z) + across(arrows(wave_at(modes, z, tau)).0)
        })
        .collect();
    sc.polyline(&e_line, 2.0, palette::ORANGE);
    // The direction of travel, from which the arcs start.
    let beam = Space::new(0.0, 0.0, -1.0);
    for k in 0..21 {
        let z = Z_MAX * k as f64 / 20.0;
        let (e, b) = arrows(wave_at(modes, z, tau));
        sc.arrow(on_beam(z), across(e), 1.2, 5.0, palette::ORANGE);
        let strength = b.norm();
        if k % 2 == 1 || strength < 1e-9 {
            continue;
        }
        // The disc's plane is the magnetic plane, across `b`: spanned by the beam's axis (the
        // waves are transverse) and `n x beam`, turning from the first to the second
        // right-handed about `b`.
        let n = b.gp(1.0 / strength);
        let u = (beam - n.gp(beam.dot(n).s())).normalized();
        let w: Space = (n ^ u).dual().with_orientation();
        // Its area grows with the strength, as a patch's does: numga's 0.41 at unit strength,
        // on numga's beam of 4π.
        #[allow(clippy::disallowed_methods)] // a disc's radius from its area
        let radius = 0.41 * strength.sqrt() * length / Z_MAX / scale;
        let rim = |angle: f64, reach: f64| {
            let p = phasor(angle);
            on_beam(z) + across((u.gp(p.e20()) + w.gp(p.e01())).gp(radius * reach))
        };
        let n_rim = 24;
        let step = core::f64::consts::TAU / n_rim as f64;
        for i in 0..n_rim {
            let (a0, a1) = (step * i as f64, step * (i + 1) as f64);
            sc.tri(
                on_beam(z),
                rim(a0, 1.0),
                rim(a1, 1.0),
                palette::SKY.faded(0.5),
                0.35,
            );
        }
        // The arc of its sense of turning, just outside the rim, ending in an arrow.
        let (start, span) = (30f64.to_radians(), 240f64.to_radians());
        let arc: Vec<_> = (0..=16)
            .map(|i| rim(start + span * i as f64 / 16.0, 1.2))
            .collect();
        sc.polyline(&arc, 1.2, palette::SKY);
        let (tail, tip) = (arc[15], rim(start + span + 0.2, 1.2));
        sc.arrow(tail, tip - tail, 1.2, 5.0, palette::SKY);
    }
    sc.draw(c);
    c.unclip();
    // 11, and no wider than the panel.
    let size = (rect.height() / 21.0)
        .clamp(5.0, 11.0)
        .min(rect.width() / gax_numga_examples::font::width(label, 1.0));
    let top_middle = rect.top_middle();
    let at = top_middle + Point2::direction(0.0, size * 14.0 / 11.0);
    c.text(label, at, size, palette::INK, Align::Center);
}

/// How much smaller a plot's own text is drawn than in a plot 240 pixels tall (numga's
/// canvas), down to half.
fn text_scale(rect: Rect) -> f32 {
    (rect.height() / 240.0).clamp(0.5, 1.0)
}

fn draw_dispersion(c: &mut Canvas, rect: Rect, s: &Scans, cursor: f64) {
    let inset = rect.inset(40.0, 26.0, 10.0, 34.0);
    let ax = Axes::new(inset, [0.05, 1.5], [1e-4, 3.0]).log_y();
    ax.frame(c, "RESIDUAL", "SPEED ALONG Z", "");
    let k = s
        .speeds
        .iter()
        .position(|v| *v >= cursor)
        .unwrap_or(s.speeds.len() - 1);
    for (i, (scan, expected)) in s.dispersion.iter().zip(&s.expected).enumerate() {
        let col = palette::series(i);
        for v in expected {
            let line = [Point::xy(*v, 1e-4), Point::xy(*v, 3.0)];
            ax.dashed(c, &line, 1.0, 4.0, col.faded(0.6));
        }
        let pts: Vec<Point> = s
            .speeds
            .iter()
            .zip(scan)
            .map(|(v, m)| Point::xy(*v, m.max(1e-4)))
            .collect();
        ax.polyline(c, &pts, 1.5, col.faded(0.9));
        ax.scatter(c, &pts[k..=k], Marker::Dot, 6.0, col);
    }
    ax.line(
        c,
        Point::xy(cursor, 1e-4),
        Point::xy(cursor, 3.0),
        1.0,
        palette::INK.faded(0.5),
    );
    let names: Vec<(&str, Light)> = NAMES
        .iter()
        .enumerate()
        .map(|(i, n)| (*n, palette::series(i)))
        .collect();
    ax.legend(c, &names);
}

fn draw_polarizations(c: &mut Canvas, rect: Rect, s: &Scans, tau: f64) {
    let (_, fields) = s.crystal;
    let origin = Point::origin();
    let inset = rect.inset(14.0, 26.0, 10.0, 34.0);
    let ax = Axes::equal(inset, origin, 1.3);
    ax.frame(c, "CRYSTAL MODES", "X", "");
    let faint = palette::GRID.faded(0.6);
    ax.line(c, Point::xy(-1.3, 0.0), Point::xy(1.3, 0.0), 1.0, faint);
    ax.line(c, Point::xy(0.0, -1.3), Point::xy(0.0, 1.3), 1.0, faint);
    // Each mode's field oscillating in its plane at the entrance face, in phase.
    for (f, col) in fields.iter().zip([palette::RED, palette::BLUE]) {
        // The electric vector lies across the beam: its x and y, at unit length.
        let (e, _) = arrows(*f);
        let a = e.normalized();
        let along = |k: f64| origin + Point::direction(a.e1(), a.e2()).gp(k);
        ax.dashed(c, &[along(-1.0), along(1.0)], 1.0, 4.0, col.faded(0.7));
        ax.arrow(c, origin, along(1.0), 2.5, 9.0, col);
        ax.scatter(
            c,
            &[along(phasor(tau).e20())],
            Marker::Dot,
            7.0,
            palette::INK,
        );
    }
    // The names in the top left corner (the panel is narrower than it is tall).
    let corner = ax.at(0.0, 1.0);
    let slow = corner + Point2::direction(0.1, -0.2);
    let fast = corner + Point2::direction(0.1, -0.45);
    let size = 10.0 * text_scale(rect);
    ax.text(c, slow, "SLOW", size, palette::RED, Align::Left);
    ax.text(c, fast, "FAST", size, palette::BLUE, Align::Left);
}

fn draw_fresnel(c: &mut Canvas, rect: Rect, s: &Scans, angle: f64) {
    let ax = Axes::equal(
        rect.inset(14.0, 26.0, 10.0, 34.0),
        Point::xy(0.0, 0.05),
        1.1,
    );
    ax.frame(c, "FRESNEL SURFACES", "X", "");
    // Zero along +z (up), angles towards +x (right): a point is the speed times the direction
    // of travel from the origin, which is minus the heading (see `modes`), so that the moving
    // glass, moving along -z, is fast downwards.
    let origin = Point::origin();
    let place = |a: f64, v: f64| -> Point {
        let d = heading(a);
        origin + Point::direction(-d.e1(), -d.e3()).gp(v)
    };
    for r in [0.25, 0.5, 0.75, 1.0] {
        let ring: Vec<Point> = (0..=72)
            .map(|k| place(core::f64::consts::TAU * k as f64 / 72.0, r))
            .collect();
        ax.polyline(c, &ring, 1.0, palette::GRID.faded(0.6));
    }
    for (m, (name, per_angle)) in ["GLASS", "CRYSTAL", "MOVING"]
        .iter()
        .zip(&s.sheets)
        .enumerate()
    {
        let col = palette::series([0, 2, 4][m]);
        let branches = per_angle.iter().map(Vec::len).max().unwrap_or(0);
        for b in 0..branches {
            let pts: Vec<Point> = s
                .angles
                .iter()
                .zip(per_angle)
                .filter_map(|(a, v)| v.get(b).map(|v| place(*a, *v)))
                .collect();
            ax.polyline(c, &pts, 1.6, col.faded(0.9));
        }
        // Where the sweeping heading meets each sheet, from the nearest scanned angle.
        let i = ((angle / core::f64::consts::TAU * (s.angles.len() - 1) as f64).round() as usize)
            .min(s.angles.len() - 1);
        let hits: Vec<Point> = per_angle[i]
            .iter()
            .map(|v| place(s.angles[i], *v))
            .collect();
        ax.scatter(c, &hits, Marker::Dot, 7.0, col);
        ax.text(
            c,
            Point::xy(-1.0, -0.62 - 0.17 * m as f64),
            name,
            9.0 * text_scale(rect),
            col,
            Align::Left,
        );
    }
    let d = place(angle, 1.1);
    ax.line(c, origin, d, 1.0, palette::INK.faded(0.6));
}

fn draw_drag(c: &mut Canvas, rect: Rect, s: &Scans, fraction: f64) {
    let inset = rect.inset(40.0, 26.0, 10.0, 34.0);
    let ax = Axes::new(inset, [-0.6, 0.6], [0.0, 1.0]);
    ax.frame(c, "FRESNEL DRAG", "SPEED OF THE GLASS", "");
    let d = &s.drag;
    let curve = |v: &[f64]| -> Vec<Point> {
        d.curve
            .iter()
            .zip(v)
            .map(|(f, v)| Point::xy(*f, *v))
            .collect()
    };
    let samples = |v: &[f64]| -> Vec<Point> {
        d.fractions
            .iter()
            .zip(v)
            .map(|(f, v)| Point::xy(*f, *v))
            .collect()
    };
    let colours = [palette::ORANGE, palette::SKY];
    let markers = [Marker::Dot, Marker::Square];
    for k in 0..2 {
        ax.polyline(c, &curve(&d.relativistic[k]), 1.5, colours[k]);
        // Fresnel's first-order drag, dashed.
        ax.dashed(
            c,
            &curve(&d.first_order[k]),
            1.0,
            5.0,
            colours[k].faded(0.6),
        );
        ax.scatter(c, &samples(&d.found[k]), markers[k], 6.0, colours[k]);
    }
    let now = relativistic_sum(fraction).map(|v| Point::xy(fraction, v));
    let cursor = palette::INK.faded(0.5);
    ax.line(
        c,
        Point::xy(fraction, 0.0),
        Point::xy(fraction, 1.0),
        1.0,
        cursor,
    );
    ax.scatter(c, &now, Marker::Ring, 11.0, palette::INK);
    ax.legend(
        c,
        &[("WITH THE GLASS", colours[0]), ("AGAINST", colours[1])],
    );
}

fn draw(c: &mut Canvas, s: &Scans, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let at = f64::from((t / SECONDS).rem_euclid(1.0));
    let tau = core::f64::consts::TAU * 2.0 * at;
    // Below the caption (its subtitle's baseline three caption sizes down), two rows: the
    // waves above, the plots below, by fractions across.
    let top = (h / 30.0).clamp(10.0, 22.0) * 3.0 + 4.0;
    let mid = top + (h - top) * 0.5;
    let upper = |x0: f32, x1: f32| Rect::new(w * x0, top, w * x1, mid);
    let lower = |x0: f32, x1: f32| Rect::new(w * x0, mid, w * x1, h);
    draw_wave(
        c,
        upper(0.0, 0.5),
        &s.glass,
        tau,
        "GLASS: ONE SPEED, FIXED POLARIZATION",
    );
    draw_wave(
        c,
        upper(0.5, 1.0),
        &s.crystal,
        tau,
        "CRYSTAL: SLOW AND FAST MODES SLIP",
    );
    let sweep = haversine(core::f64::consts::TAU * at);
    draw_dispersion(c, lower(0.0, 0.34), s, 0.05 + 1.45 * sweep);
    draw_polarizations(c, lower(0.34, 0.5), s, tau);
    draw_fresnel(c, lower(0.5, 0.75), s, core::f64::consts::TAU * at);
    draw_drag(c, lower(0.75, 1.0), s, -0.6 + 1.2 * sweep);
    caption(
        c,
        "CONSTITUTIVE MAPS: GLASS, CRYSTALS, MOVING MEDIA",
        "PLANE WAVES WHERE THE 8X6 WAVE MAP HAS A NULLSPACE (STA)",
    );
}

fn main() {
    run_with(
        Anim::new("constitutive", SECONDS).size(960, 540),
        scans,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::constitutive::*;
    use gax::sta::{Bivector, Trivector, Vector, intrinsic};
    use gax_numga_examples::rng::{Draw, Rng, rng};
    use gax_numga_examples::signal::linspace;
    use gax_numga_examples::testing::near;

    fn trial_speeds() -> Vec<f64> {
        linspace(0.05, 1.5, 6001)
    }

    fn scan_tolerance() -> f64 {
        1.45 / 6000.0
    }

    fn isotropic(permittivity: f64, permeability: f64) -> Medium {
        medium([permittivity; 3], [1.0 / permeability; 3])
    }

    fn phase_speeds(medium: Medium, direction: Vector, speeds: &[f64]) -> Vec<f64> {
        let scan: Vec<f64> = speeds
            .iter()
            .map(|s| residual(t().gp(*s) - direction, medium))
            .collect();
        dip_speeds(speeds, &scan, 2e-3)
    }

    fn all_close(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(p, q)| (p - q).abs() <= tol)
    }

    /// A field with normal coefficients.
    fn bivector(r: &mut Rng) -> Bivector {
        Bivector::from_coeffs(core::array::from_fn(|_| r.normal()))
    }

    fn magnetic_part(observer: Vector) -> Projector {
        open() - electric_part(observer)
    }

    #[test]
    fn unit_medium_is_the_vacuum() {
        let mut rng = rng(5);
        let vacuum = isotropic(1.0, 1.0);
        for _ in 0..7 {
            let f = bivector(&mut rng);
            assert!(near(vacuum.of(f), f.hodge(), 1e-14));
        }
    }

    /// The port's own (numga dropped its projectors): the observer's electric and magnetic
    /// parts are complementary idempotents.
    #[test]
    fn observer_projectors_are_complementary_idempotents() {
        let (electric, magnetic) = (electric_part(t()), magnetic_part(t()));
        assert!(near(electric.of(electric), electric, 1e-14));
        assert!(near(magnetic.of(magnetic), magnetic, 1e-14));
        assert!(near(electric.of(magnetic), Projector::zero(), 1e-14));
        let f = Bivector::new(3.0, 0.0, 4.0, 1.0, 2.0, 5.0);
        assert!(near(electric.of(f) + magnetic.of(f), f, 1e-14));
    }

    /// The port's own, in place of numga's dropped "crystal reduces to glass": the dyads of a
    /// medium at rest are the observer's electric and magnetic parts, weighted, then
    /// Hodge-dualized, for glass and for each axis of the crystal and the ferrite.
    #[test]
    fn dyads_are_the_weighted_observer_parts_dualized() {
        let (electric, magnetic) = (electric_part(t()), magnetic_part(t()));
        let rebuilt = (electric.gp(2.25) + magnetic.gp(1.0 / 1.5)).hodge();
        assert!(near(isotropic(2.25, 1.5), rebuilt, 1e-14));
        let [_, crystal, ferrite] = media();
        let planes = [x() ^ t(), y() ^ t(), z() ^ t()];
        let electric_crystal: Projector = planes
            .iter()
            .zip(PERMITTIVITIES[1])
            .map(|(p, e)| (*p * (*p | open())).gp(e))
            .sum();
        assert!(near(crystal, (electric_crystal + magnetic).hodge(), 1e-14));
        let magnetic_ferrite: Projector = planes
            .iter()
            .zip(INVERSE_PERMEABILITIES[2])
            .map(|(p, m)| {
                let plane = p.hodge();
                (plane * (plane | open())).gp(m)
            })
            .sum();
        assert!(near(
            ferrite,
            (electric.gp(2.25) - magnetic_ferrite).hodge(),
            1e-14
        ));
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // the speeds 1/√(εμ) it is checked against
    fn isotropic_medium_speed_is_one_over_n() {
        let s = trial_speeds();
        for (eps, mu) in [(2.25, 1.0), (4.0, 1.5)] {
            assert!(all_close(
                &phase_speeds(isotropic(eps, mu), z(), &s),
                &[1.0 / (eps * mu).sqrt()],
                scan_tolerance()
            ));
        }
    }

    #[test]
    fn material_and_stress_forms_give_the_electric_and_magnetic_energy() {
        let (eps, mu) = (2.25, 1.5);
        let medium = isotropic(eps, mu);
        let mut rng = rng(318);
        let (electric, magnetic) = (electric_part(t()), magnetic_part(t()));
        let bilinear_form = open() & medium;
        // numga's `t F t⁻¹` is `-(t >> F)`.
        let energy_form = -(t() >> open()) & medium;
        let vacuum = isotropic(1.0, 1.0);
        for _ in 0..17 {
            let f = bivector(&mut rng);
            let (e, m) = (electric.of(f), magnetic.of(f));
            let e2 = e.scalar_product(e).s();
            let m2 = m.scalar_product(m).s();
            let action = (eps * e2 + m2 / mu) / 2.0;
            let energy = (eps * e2 - m2 / mu) / 2.0;
            let stress_form = Vector::slot() & stress_energy(f, medium.of(f));
            // The constitutive form pairs the field with its excitation: minus twice the
            // Lagrangian.
            assert!((bilinear_form.of(f).of(f).s() / 2.0 + action).abs() < 1e-12);
            assert!((energy_form.of(f).of(f).s() / 2.0 - energy).abs() < 1e-12);
            assert!((stress_form.of(t()).of(time()).s() - energy).abs() < 1e-12);
            // In vacuum the current along a direction is the field's sandwich of the
            // direction, halved.
            let current = stress_energy(f, vacuum.of(f));
            for d in [t(), x(), z()] {
                let expected = (f.cast::<gax::sta::Even>() >> d).hodge().gp(0.5);
                assert!(near(current.of(d.hodge()), expected, 1e-12));
            }
        }
        let flux_field = Bivector::new(0.0, 1.0, 0.0, -1.0, 0.0, 0.0); // numga's tx + zx
        let flux_form = Vector::slot() & stress_energy(flux_field, medium.of(flux_field));
        assert!((flux_form.of(z()).of(time()).s() + 1.0 / mu).abs() < 1e-12);
        assert!((flux_form.of(t()).of(z().hodge()).s() - eps).abs() < 1e-12);
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // the speeds 1/√(εμ) it is checked against
    fn crystal_is_birefringent_along_z_and_not_along_x() {
        let [_, crystal, _] = media();
        let s = trial_speeds();
        assert!(all_close(
            &phase_speeds(crystal, z(), &s),
            &[1.0 / 2.25f64.sqrt(), 1.0 / 1.5f64.sqrt()],
            scan_tolerance()
        ));
        assert!(all_close(
            &phase_speeds(crystal, x(), &s),
            &[1.0 / 1.5f64.sqrt()],
            scan_tolerance()
        ));
    }

    #[test]
    fn crystal_fields_satisfy_maxwell_and_lie_in_their_material_planes() {
        let (_, (speeds, fields)) = modes();
        let [_, crystal, _] = media();
        // The plane z sweeps through time reads the electric field along z, and the patch of
        // the plane xy the magnetic field in it: the electric fields lie along x and y, the
        // magnetic planes contain z.
        let along_z = z().hodge() & time();
        let across_z = Bivector::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0).hodge();
        for (s, f) in speeds.iter().zip(fields) {
            let k = t().gp(*s) - z();
            assert!(near(k ^ f, Trivector::zero(), 1e-12));
            assert!(near(k ^ crystal.of(f), intrinsic::Trivector::zero(), 1e-12));
            assert!((f & along_z).s().abs() < 1e-12);
            assert!((f & across_z).s().abs() < 1e-12);
        }
    }

    #[test]
    fn vacuum_wave_has_two_transverse_field_modes() {
        let vacuum = isotropic(1.0, 1.0);
        let k = t() - z();
        let svd = wave_map(k, vacuum).svd();
        assert_eq!(svd.values.iter().filter(|s| **s < 1e-12).count(), 2);
        let action_form = open() & vacuum;
        for f in &svd.inputs[4..] {
            assert!(near(k ^ *f, Trivector::zero(), 1e-12));
            assert!(near(k ^ vacuum.of(*f), intrinsic::Trivector::zero(), 1e-12));
            assert!((action_form.of(*f).of(*f).s() / 2.0).abs() < 1e-12);
            let stress_form = Vector::slot() & stress_energy(*f, vacuum.of(*f));
            assert!(stress_form.of(t()).of(time()).s() > 0.0);
        }
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // the speeds 1/√(εμ) it is checked against
    fn ferrite_lifts_permeability_through_the_dual_field() {
        let [_, _, ferrite] = media();
        assert!(all_close(
            &phase_speeds(ferrite, z(), &trial_speeds()),
            &[1.0 / 4.5f64.sqrt(), 1.0 / 2.25f64.sqrt()],
            scan_tolerance()
        ));
    }

    #[test]
    fn axion_term_is_invisible_to_bulk_waves() {
        let [glass, _, _] = media();
        let f = Bivector::new(3.0, 0.0, 0.0, -2.0, 0.0, 0.0); // numga's 2 tx + 3 yz
        let base_form = open() & glass;
        let base_current = stress_energy(f, glass.of(f));
        let s = trial_speeds();
        for alpha in [0.4, -1.3] {
            let coupling = AxionCoupling::new(alpha);
            let axion = glass + coupling * open();
            let axion_form = open() & axion;
            let change = (axion_form.of(f).of(f).s() - base_form.of(f).of(f).s()) / 2.0;
            assert!(all_close(
                &phase_speeds(axion, z(), &s),
                &phase_speeds(glass, z(), &s),
                scan_tolerance()
            ));
            assert!((change - alpha * (f & f).s() / 2.0).abs() < 1e-12);
            assert!(near(stress_energy(f, axion.of(f)), base_current, 1e-12));
        }
    }

    #[test]
    fn material_and_stress_forms_transform_with_the_field_and_observer() {
        let (eps, mu, fraction) = (2.25, 1.5, 0.35);
        let medium = isotropic(eps, mu);
        let rotation = boost(fraction);
        let observer = rotation >> t();
        let moved_medium = moving(medium, fraction);
        // The same medium built from the moving observer's electric and magnetic planes.
        let electric = electric_part(observer);
        let rebuilt = (electric.gp(eps) + (open() - electric).gp(1.0 / mu)).hodge();
        let mut rng = rng(627);
        let (form, moved_form) = (open() & medium, open() & moved_medium);
        for _ in 0..13 {
            let f = bivector(&mut rng);
            let moved = rotation >> f;
            let current = stress_energy(f, medium.of(f));
            let moved_current = stress_energy(moved, moved_medium.of(moved));
            assert!(near(moved_medium.of(f), rebuilt.of(f), 1e-7));
            assert!((moved_form.of(moved).of(moved).s() - form.of(f).of(f).s()).abs() < 1e-7);
            let carried = rotation >> current.of(rotation << open_direction());
            assert!(near(moved_current, carried, 1e-7));
            let stress_form = Vector::slot() & current;
            let moved_stress_form = Vector::slot() & moved_current;
            let moved_time = -observer.hodge();
            assert!(
                (moved_stress_form.of(observer).of(moved_time).s()
                    - stress_form.of(t()).of(time()).s())
                .abs()
                    < 1e-7
            );
        }
    }

    #[test]
    fn moving_glass_shows_exact_fresnel_drag() {
        let [glass, _, _] = media();
        let s = trial_speeds();
        for fraction in [0.0, 0.3, -0.5] {
            let [expected, _] = relativistic_sum(fraction);
            assert!(all_close(
                &phase_speeds(moving(glass, fraction), z(), &s),
                &[expected],
                scan_tolerance()
            ));
        }
    }

    /// numga's scenario checks, at numga's sizes.
    #[test]
    fn scenario_checks() {
        // Every medium carries its waves along z at the expected speeds.
        let (speeds, residuals, expected) = dispersion();
        let step = speeds[1] - speeds[0];
        for (r, e) in residuals.iter().zip(expected) {
            let mut unique = e.to_vec();
            unique.dedup_by(|a, b| a == b);
            assert!(
                all_close(&dip_speeds(&speeds, r, THRESHOLD), &unique, step),
                "{e:?}"
            );
        }
        // The crystal's waves satisfy both source-free Maxwell equations.
        let [glass, crystal, _] = media();
        let (_, (crystal_speeds, fields)) = modes();
        for (s, f) in crystal_speeds.iter().zip(fields) {
            let k = t().gp(*s) - z();
            assert!(near(k ^ f, Trivector::zero(), 1e-12));
            assert!(near(k ^ crystal.of(f), intrinsic::Trivector::zero(), 1e-12));
        }
        // The axion term changes no wave's speed.
        let axion = glass + AxionCoupling::new(AXION_COUPLING) * open();
        let scan: Vec<f64> = speeds
            .iter()
            .map(|s| residual(t().gp(*s) - z(), axion))
            .collect();
        assert!(all_close(
            &dip_speeds(&speeds, &scan, THRESHOLD),
            &[expected[0][0]],
            step
        ));
        // Light in the moving glass travels at the relativistic sum of the two speeds.
        let d = drag();
        let drag_step = (DRAG_SPEED_RANGE.1 - DRAG_SPEED_RANGE.0) / (N_DRAG_SPEEDS - 1) as f64;
        for (i, f) in d.fractions.iter().enumerate() {
            let sum = relativistic_sum(*f);
            for (found, expected) in d.found.iter().zip(sum) {
                assert!((found[i] - expected).abs() <= drag_step + 1e-12);
            }
        }
    }

    /// gax's STA has numga's blades in numga's order (ADR-055) and its regressive product
    /// agrees with numga's, so the maps read the same numbers: the three media, glass moving
    /// at 0.35, and a stress-energy map, against numga 35797de's.
    #[test]
    fn maps_read_numga_s_numbers() {
        let basis = |i: usize| -> Bivector {
            Bivector::from_coeffs(core::array::from_fn(|j| f64::from(i == j)))
        };
        let rest = |eps: [f64; 3], m: [f64; 3]| -> [[f64; 6]; 6] {
            let mut r = [[0.0; 6]; 6];
            for i in 0..3 {
                r[i][i + 3] = m[i];
                r[i + 3][i] = -eps[i];
            }
            r
        };
        let moving_glass = [
            [0.0, -0.4985754986, 0.0, 0.8254985755, 0.0, 0.0],
            [0.4985754986, 0.0, 0.0, 0.0, 0.8254985755, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [-2.4245014245, 0.0, 0.0, 0.0, 0.4985754986, 0.0],
            [0.0, -2.4245014245, 0.0, -0.4985754986, 0.0, 0.0],
            [0.0, 0.0, -2.25, 0.0, 0.0, 0.0],
        ];
        let [glass, crystal, ferrite] = media();
        let cases = [
            (glass, rest([2.25; 3], [1.0; 3])),
            (crystal, rest([2.25, 1.5, 1.5], [1.0; 3])),
            (ferrite, rest([2.25; 3], [1.0, 0.5, 1.0])),
            (moving(glass, 0.35), moving_glass),
        ];
        for (m, rows) in cases {
            for (i, row) in rows.iter().enumerate() {
                let got: [f64; 6] = m.of(basis(i)).c;
                assert!(
                    got.iter().zip(row).all(|(a, b)| (a - b).abs() < 1e-9),
                    "{got:?}"
                );
            }
        }
        let f = Bivector::new(0.3, -0.7, 1.1, 0.5, 0.2, -0.4);
        assert!(((f & glass.of(f)).s() - 0.7775).abs() < 1e-12);
        let current = stress_energy(f, glass.of(f));
        let numga = [
            [1.40125, 0.06, 0.67, 0.41],
            [-0.135, -0.74875, 0.015, -0.12],
            [-1.5075, 0.015, -0.82125, -0.95],
            [-0.9225, -0.12, -0.95, 0.16875],
        ];
        for (i, row) in numga.iter().enumerate() {
            let d: Direction =
                intrinsic::Trivector::from_coeffs(core::array::from_fn(|j| f64::from(i == j)));
            let got: [f64; 4] = current.of(d).c;
            assert!(
                got.iter().zip(row).all(|(a, b)| (a - b).abs() < 1e-12),
                "{got:?}"
            );
        }
        assert!(near(
            time(),
            -intrinsic::Trivector::new(1.0, 0.0, 0.0, 0.0),
            1e-15
        ));
    }

    /// The modes of numga's wavevectors `s t - z` travel along -z: the Poynting vector of
    /// each (E x B as the animation reads them off) points along -z, and that of a wave of
    /// `t + z`, the gradient of the phase `T - Z`, along +z.
    #[test]
    fn numga_s_waves_travel_along_minus_z() {
        let (glass, crystal) = modes();
        for f in glass.1.into_iter().chain(crystal.1) {
            // E x B along -z: the plane E ^ B turns from y towards x.
            let (e, b) = arrows(f);
            assert!((e ^ b).e12() < 0.0, "{e:?} {b:?}");
        }
        let vacuum = isotropic(1.0, 1.0);
        for f in &wave_map(t() + z(), vacuum).svd().inputs[4..] {
            let (e, b) = arrows(*f);
            assert!((e ^ b).e12() > 0.0, "{e:?} {b:?}");
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scans(), 0.5);
    }
}
