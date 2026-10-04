//! numga's `electromagnetism/constitutive`: materials as constitutive maps in spacetime algebra.
//! A medium maps the field bivector `F` to the excitation antibivector `G = χ(F)` (in four
//! dimensions antibivectors are bivectors again, reached by the complement): it weights the
//! observer's electric and magnetic planes (glass), weights them per axis (a crystal, a
//! ferrite), adds the identity (an axion term), or is conjugated by a boost (moving glass).
//! Pairing an open field with the medium gives the material's bilinear form, and the field and
//! excitation give a stress-energy current.
//!
//! A plane wave with wave covector `k` solves both source-free Maxwell equations,
//! `k ∧ F = 0` and `k ∧ χ(F) = 0`. Packed into one map `F ↦ k ∧ F + (k ∧ χ(F))*` from the six
//! field coefficients to the eight of the odd subalgebra, a wave exists where that map has a
//! nullspace: its smallest singular value dips to zero. Scanning the phase speed finds the
//! dispersion of each medium, scanning the direction too traces the Fresnel surfaces, and
//! scanning the speed of moving glass shows Fresnel drag, exactly Einstein's velocity addition.
//!
//! The animation shows the travelling fields in glass (one speed: the polarization stays put)
//! and in a birefringent crystal (two speeds: it turns as the slow and fast modes slip), with
//! cursors sweeping the dispersion scans, the Fresnel surfaces and the drag curve.

use gax::pga2d::Point;

use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Light, Marker, Point2, Rect, Scene3, backdrop, caption,
    palette, run, signal::phasor,
};

mod constitutive {
    // numga's API in full: the tests use parts the animation does not.
    #![cfg_attr(not(test), allow(dead_code))]

    use gax::Vee;
    use gax::sta::{Bivector, Odd, Trivector, Vector};

    pub type V = Vector<(), f64>;
    pub type B = Bivector<(), f64>;
    /// A constitutive map, `Antibivector <- Bivector` (antibivectors are bivectors in 4D).
    pub type Medium = Bivector<(Bivector,), f64>;
    /// A stress-energy current, `Antivector <- Vector`.
    pub type Current = Trivector<(Vector,), f64>;
    /// Both Maxwell residuals of a plane wave, `Odd <- Bivector`.
    pub type WaveMap = Odd<(Bivector,), f64>;
    pub type Rotor = gax::Unit<gax::sta::Even<(), f64>>;
    /// A vector of the rest observer's space, for drawing.
    pub type Space = gax::vga3d::Vector<(), f64>;

    /// numga's regressive product of complementary grades, `(a* ∧ b*) I⁻¹` with numga's dual
    /// (gax's `hodge`, `~x I`, which keeps the metric): the pairing behind the material and
    /// stress forms. The Hodge dual is the metric-free complement times the product of its
    /// blade's squares, so on complementary grades this is gax's regressive product `a & b`
    /// times the product of all the squares: `-(a & b)` in spacetime (checked in the tests).
    pub fn pair<X: Vee<Y>, Y>(a: X, b: Y) -> <X::Output as core::ops::Neg>::Output
    where
        X::Output: core::ops::Neg,
    {
        -a.vee(b)
    }

    /// The open field.
    pub fn open() -> Medium {
        Bivector::slot()
    }

    /// The open vector.
    pub fn open_vector() -> Vector<(Vector,), f64> {
        Vector::slot()
    }

    pub fn t() -> V {
        Vector::new(1.0, 0.0, 0.0, 0.0)
    }
    pub fn x() -> V {
        Vector::new(0.0, 1.0, 0.0, 0.0)
    }
    pub fn y() -> V {
        Vector::new(0.0, 0.0, 1.0, 0.0)
    }
    pub fn z() -> V {
        Vector::new(0.0, 0.0, 0.0, 1.0)
    }

    /// The spatial part of a vector as the rest observer `t` reads it, a vector of space.
    pub fn spatial(v: V) -> Space {
        Space::new(v.e1(), v.e2(), v.e3())
    }

    /// The electric and magnetic vectors a rest observer reads off a field: `F · t` and
    /// `F* · t`, as numga draws them.
    pub fn arrows(f: B) -> (Space, Space) {
        (spatial(f | t()), spatial(f.hodge() | t()))
    }

    /// A field from numga's blade coefficients `tx, ty, tz, yz, zx, xy` (numga's `tx = t ^ x`
    /// is gax's `-e10`).
    pub fn field(tx: f64, ty: f64, tz: f64, yz: f64, zx: f64, xy: f64) -> B {
        Bivector::new(-tx, -ty, -tz, yz, zx, xy)
    }

    /// The electric and magnetic projectors of a unit timelike observer: `(F - u F u) / 2`
    /// and the rest.
    pub fn observer_projectors(observer: V) -> (Medium, Medium) {
        let electric = (open() - (observer >> open())).gp(0.5);
        (electric, open() - electric)
    }

    /// Weight the observer's electric and magnetic planes, then take the Hodge dual (numga's
    /// `dual`; gax's `dual` is the metric-free complement, which in spacetime differs in sign
    /// on every blade with an odd number of spatial vectors, and with it vacuum `F ↦ F*` has
    /// no waves).
    pub fn isotropic_medium(eps: f64, mu: f64, observer: V) -> Medium {
        let (electric, magnetic) = observer_projectors(observer);
        (electric.gp(eps) + magnetic.gp(1.0 / mu)).hodge()
    }

    /// An anisotropic electric response: each axis's plane with the observer weighted.
    pub fn crystal_medium(eps: [f64; 3], mu: f64, observer: V) -> Medium {
        let (_, magnetic) = observer_projectors(observer);
        let electric = [x(), y(), z()]
            .iter()
            .zip(eps)
            .fold(Medium::zero(), |acc, (axis, e)| {
                let plane = *axis ^ observer;
                acc + (plane * (plane | open())).gp(e)
            });
        (electric + magnetic.gp(1.0 / mu)).hodge()
    }

    /// An anisotropic magnetic response: each axis's magnetic plane weighted by the inverse
    /// permeability.
    pub fn ferrite_medium(eps: f64, mu_inv: [f64; 3], observer: V) -> Medium {
        let (electric, _) = observer_projectors(observer);
        let magnetic = [x(), y(), z()]
            .iter()
            .zip(mu_inv)
            .fold(Medium::zero(), |acc, (axis, m)| {
                let plane = (*axis ^ observer).hodge();
                acc + (plane * (plane | open())).gp(m)
            });
        (electric.gp(eps) - magnetic).hodge()
    }

    /// Add the axion response, a multiple of the identity on field planes.
    pub fn axion_medium(base: Medium, alpha: f64) -> Medium {
        base + open().gp(alpha)
    }

    /// The boost to speed `beta` along a unit spatial `direction`.
    pub fn boost_rotor(beta: f64, direction: V) -> Rotor {
        ((direction ^ t()) * (beta.atanh() / 2.0)).exp()
    }

    /// A medium at rest, set moving: conjugated by the boost.
    pub fn boosted_medium(base: Medium, beta: f64, direction: V) -> Medium {
        let r = boost_rotor(beta, direction);
        r >> base.of(r << open())
    }

    /// The electromagnetic stress-energy current `v ↦ (v · F) ∧ G + ½ (F ∨ G) v*`. An
    /// observer's energy density is `u ∨ current(u)`.
    pub fn stress_energy(field: B, excitation: B) -> Current {
        let lagrangian = pair(field, excitation).s() / 2.0;
        ((open_vector() | field) ^ excitation) + open_vector().hodge().gp(lagrangian)
    }

    /// Both exterior Maxwell residuals, packed into separate grades: `k ∧ F` is a trivector
    /// and the dual of `k ∧ χ(F)` a vector, so they cannot cancel. gax adds values of one
    /// kind, so both are cast into the odd subalgebra.
    pub fn wave_map(k: V, medium: Medium) -> WaveMap {
        (k ^ open()).cast::<Odd>() + (k ^ medium).hodge().cast::<Odd>()
    }

    /// The singular values (descending) and right singular vectors of a wave map (8 x 6).
    pub fn svd(w: WaveMap) -> ([f64; 6], [B; 6]) {
        let (values, right, _) = w.svd_thin();
        (values, right)
    }

    /// The smallest singular value of the wave map over trial phase speeds, for waves along
    /// `direction`.
    pub fn dispersion_scan(speeds: &[f64], medium: Medium, direction: V) -> Vec<f64> {
        speeds
            .iter()
            .map(|s| svd(wave_map(t().gp(*s) + direction, medium)).0[5])
            .collect()
    }

    /// The speeds where the scan has a local minimum below `threshold`: the allowed waves.
    pub fn minimum_speeds(speeds: &[f64], scan: &[f64], threshold: f64) -> Vec<f64> {
        (1..scan.len() - 1)
            .filter(|&i| scan[i] <= scan[i - 1] && scan[i] <= scan[i + 1] && scan[i] < threshold)
            .map(|i| speeds[i])
            .collect()
    }

    /// The field in the wave map's nullspace.
    pub fn field_eigenmode(k: V, medium: Medium) -> B {
        svd(wave_map(k, medium)).1[5]
    }

    /// The speed of the deepest dip of a scan.
    pub fn argmin_speed(speeds: &[f64], scan: &[f64]) -> f64 {
        let k = (0..scan.len())
            .min_by(|&a, &b| scan[a].total_cmp(&scan[b]))
            .expect("a scan");
        speeds[k]
    }

    /// Downstream and upstream phase speeds along z in glass moving at each speed `beta`.
    pub fn fresnel_drag_velocities(
        eps: f64,
        mu: f64,
        betas: &[f64],
        speeds: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let glass = isotropic_medium(eps, mu, t());
        betas
            .iter()
            .map(|b| {
                let moving = boosted_medium(glass, *b, z());
                (
                    argmin_speed(speeds, &dispersion_scan(speeds, moving, z())),
                    argmin_speed(speeds, &dispersion_scan(speeds, moving, -z())),
                )
            })
            .unzip()
    }

    // --- scenes -------------------------------------------------------------------------

    pub const EPS_GLASS: f64 = 2.25;
    pub const MU_GLASS: f64 = 1.0;
    pub const BETA_BOOST: f64 = 0.3;
    pub const AXION_ALPHA: f64 = 0.4;

    /// Einstein's addition of collinear speeds, `(u + v) / (1 + u v)`: the phase speed of
    /// light in glass moving at `v`, for its speed `u` in the glass at rest.
    pub fn add_speeds(u: f64, v: f64) -> f64 {
        (u + v) / (1.0 + u * v)
    }

    pub fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
            .collect()
    }

    /// The crystal of the scenes: slow along x, fast along y and z.
    pub fn crystal() -> Medium {
        crystal_medium([2.25, 1.5, 1.5], 1.0, t())
    }

    /// The media of the dispersion scan along z, each with its expected speeds.
    #[allow(clippy::disallowed_methods)] // wave speeds 1/√(εμ): a physical law, not geometry
    pub fn dispersion_media() -> Vec<(&'static str, Medium, Vec<f64>)> {
        let glass = isotropic_medium(EPS_GLASS, MU_GLASS, t());
        let n = (EPS_GLASS * MU_GLASS).sqrt();
        let with_flow = add_speeds(1.0 / n, BETA_BOOST);
        let against_flow = add_speeds(1.0 / n, -BETA_BOOST);
        vec![
            ("GLASS", glass, vec![1.0 / n]),
            ("AXION", axion_medium(glass, AXION_ALPHA), vec![1.0 / n]),
            (
                "CRYSTAL",
                crystal(),
                vec![1.0 / 2.25f64.sqrt(), 1.0 / 1.5f64.sqrt()],
            ),
            (
                "FERRITE",
                ferrite_medium(2.25, [1.0, 0.5, 1.0], t()),
                vec![1.0 / 4.5f64.sqrt(), 1.0 / 2.25f64.sqrt()],
            ),
            (
                "WITH FLOW",
                boosted_medium(glass, BETA_BOOST, z()),
                vec![with_flow],
            ),
            (
                "AGAINST",
                boosted_medium(glass, BETA_BOOST, -z()),
                vec![against_flow],
            ),
        ]
    }

    /// The crystal's slow and fast waves along z: their speeds and fields.
    #[allow(clippy::disallowed_methods)] // wave speeds 1/√(εμ): a physical law, not geometry
    pub fn field_modes() -> ([f64; 2], [B; 2]) {
        let speeds = [1.0 / 2.25f64.sqrt(), 1.0 / 1.5f64.sqrt()];
        (
            speeds,
            speeds.map(|s| field_eigenmode(t().gp(s) + z(), crystal())),
        )
    }

    /// Two field modes along z in glass, sharing one speed: the two smallest singular vectors.
    #[allow(clippy::disallowed_methods)] // wave speeds 1/√(εμ): a physical law, not geometry
    pub fn glass_modes() -> ([f64; 2], [B; 2]) {
        let glass = isotropic_medium(EPS_GLASS, MU_GLASS, t());
        let speed = 1.0 / (EPS_GLASS * MU_GLASS).sqrt();
        let (_, v) = svd(wave_map(t().gp(speed) + z(), glass));
        ([speed; 2], [v[4], v[5]])
    }

    /// The direction in the xz-plane at `angle` from z towards x.
    pub fn direction(angle: f64) -> V {
        // x and z square to minus one, which turns the sense of the rotor.
        Bivector::new(0.0, 0.0, 0.0, 0.0, angle / 2.0, 0.0).exp() >> z()
    }

    /// The Fresnel scans of glass, the crystal and moving glass: per medium, per angle, the
    /// smallest singular value over the speeds.
    pub fn fresnel_surface(angles: &[f64], speeds: &[f64]) -> Vec<(&'static str, Vec<Vec<f64>>)> {
        let glass = isotropic_medium(EPS_GLASS, MU_GLASS, t());
        let moving = boosted_medium(glass, 0.35, z());
        [("GLASS", glass), ("CRYSTAL", crystal()), ("MOVING", moving)]
            .into_iter()
            .map(|(name, medium)| {
                let scans = angles
                    .iter()
                    .map(|a| dispersion_scan(speeds, medium, direction(*a)))
                    .collect();
                (name, scans)
            })
            .collect()
    }
}

use constitutive::*;

const SECONDS: f32 = 8.0;
/// A wave's length along the beam, in units where ω = 1.
const Z_MAX: f64 = 4.0 * core::f64::consts::PI;

/// The scans the animation draws, computed once, coarser than numga's figures.
struct Scans {
    speeds: Vec<f64>,
    dispersion: Vec<(&'static str, Vec<f64>, Vec<f64>)>,
    angles: Vec<f64>,
    /// Per medium, per angle, the sheet speeds.
    sheets: Vec<(&'static str, Vec<Vec<f64>>)>,
    betas: Vec<f64>,
    drag: (Vec<f64>, Vec<f64>),
    glass: ([f64; 2], [B; 2]),
    crystal: ([f64; 2], [B; 2]),
}

fn scans() -> &'static Scans {
    static S: std::sync::OnceLock<Scans> = std::sync::OnceLock::new();
    S.get_or_init(|| {
        let speeds = linspace(0.05, 1.5, 1161);
        let dispersion = dispersion_media()
            .into_iter()
            .map(|(name, medium, expected)| (name, dispersion_scan(&speeds, medium, z()), expected))
            .collect();
        let angles = linspace(0.0, core::f64::consts::TAU, 49);
        let fresnel_speeds = linspace(0.4, 1.0, 301);
        let sheets = fresnel_surface(&angles, &fresnel_speeds)
            .into_iter()
            .map(|(name, scans)| {
                let per_angle = scans
                    .iter()
                    .map(|s| minimum_speeds(&fresnel_speeds, s, 4e-3))
                    .collect();
                (name, per_angle)
            })
            .collect();
        let betas = linspace(-0.6, 0.6, 13);
        let drag = fresnel_drag_velocities(EPS_GLASS, MU_GLASS, &betas, &linspace(0.05, 1.2, 576));
        Scans {
            speeds,
            dispersion,
            angles,
            sheets,
            betas,
            drag,
            glass: glass_modes(),
            crystal: field_modes(),
        }
    })
}

/// The travelling wave of two modes at time `tau`: each mode normalized to a unit electric
/// amplitude, summed with its phase `ω (z / v - τ)`.
fn wave_at(modes: &([f64; 2], [B; 2]), z: f64, tau: f64) -> B {
    let (speeds, fields) = modes;
    speeds.iter().zip(fields).fold(B::zero(), |acc, (v, f)| {
        let (e, _) = arrows(*f);
        acc + f.gp(phasor(z / v - tau).e20() / e.norm())
    })
}

/// The wave in 3D, the beam running across the screen: world x along the beam (scaled),
/// world y and z the field's x and y.
fn draw_wave(c: &mut Canvas, rect: Rect, modes: &([f64; 2], [B; 2]), tau: f64, label: &str) {
    let unit = unit(c);
    let view = rect.inset(0.0, 20.0 * unit, 0.0, 0.0);
    let length = 6.0;
    let cam = Camera::parallel(view, rect.width() * 0.125, -1.1, 0.35);
    c.clip(rect);
    let mut sc = Scene3::new(cam);
    // The world's x runs along the beam: the point of the beam at `z`.
    let on_beam = |z: f64| gax::pga3d::Point::xyz(z / Z_MAX * length - length / 2.0, 0.0, 0.0);
    // A field's vector in the world: its x and y across the beam (the world's y and z), a little
    // of its z along it.
    let across = |v: Space| gax::pga3d::Point::direction(v.e3() * 0.1, v.e1() * 0.8, v.e2() * 0.8);
    sc.seg(on_beam(0.0), on_beam(Z_MAX), 1.0, palette::grid());
    let samples = 200;
    let mut e_line = Vec::new();
    let mut b_line = Vec::new();
    for i in 0..samples {
        let z = Z_MAX * i as f64 / (samples - 1) as f64;
        let (e, b) = arrows(wave_at(modes, z, tau));
        e_line.push(on_beam(z) + across(e));
        b_line.push(on_beam(z) + across(b));
    }
    sc.polyline(&b_line, 1.5, palette::sky().faded(0.8));
    sc.polyline(&e_line, 2.0, palette::orange());
    for k in 0..21 {
        let z = Z_MAX * k as f64 / 20.0;
        let (e, b) = arrows(wave_at(modes, z, tau));
        for (v, col) in [(e, palette::orange()), (b, palette::sky())] {
            sc.arrow(on_beam(z), across(v), 1.2, 5.0 * unit, col);
        }
    }
    sc.draw(c);
    c.unclip();
    // 11 pixels on a 960 x 540 canvas, smaller on a small one, and no wider than the panel.
    let size = (rect.height() / 21.0)
        .clamp(5.0 * unit, 11.0 * unit)
        .min(rect.width() / gax_numga_examples::font::width(label, 1.0));
    let top_middle = rect.top_middle();
    let at = top_middle + Point2::direction(0.0, size * 14.0 / 11.0);
    c.text(label, at, size, palette::ink(), Align::Center);
}

/// The canvas's height over 540: pixel sizes scale with it.
fn unit(c: &Canvas) -> f32 {
    c.unit()
}

/// How much smaller a plot's own text is drawn than in a plot 240 pixels tall (numga's
/// canvas), down to half, on a 960 x 540 canvas (`unit` 1), scaled with the canvas.
fn text_scale(rect: Rect, unit: f32) -> f32 {
    (rect.height() / 240.0).clamp(0.5 * unit, unit)
}

fn draw_dispersion(c: &mut Canvas, rect: Rect, cursor: f64) {
    let s = scans();
    let unit = unit(c);
    let inset = rect.inset(40.0 * unit, 26.0 * unit, 10.0 * unit, 34.0 * unit);
    let ax = Axes::new(inset, [0.05, 1.5], [1e-4, 3.0]).log_y();
    ax.frame(c, "SMALLEST SINGULAR VALUE", "PHASE SPEED", "");
    let k = s
        .speeds
        .iter()
        .position(|v| *v >= cursor)
        .unwrap_or(s.speeds.len() - 1);
    for (i, (_, scan, expected)) in s.dispersion.iter().enumerate() {
        let col = palette::series(i);
        for v in expected {
            let line = [Point::xy(*v, 1e-4), Point::xy(*v, 3.0)];
            ax.dashed(c, &line, 1.0, 4.0, col.faded(0.6));
        }
        let pts: Vec<Point<(), f64>> = s
            .speeds
            .iter()
            .zip(scan)
            .map(|(v, m)| Point::xy(*v, m.max(1e-4)))
            .collect();
        ax.polyline(c, &pts, if i == 1 { 1.0 } else { 1.5 }, col.faded(0.9));
        ax.scatter(c, &pts[k..=k], Marker::Dot, 6.0 * unit, col);
    }
    ax.line(
        c,
        Point::xy(cursor, 1e-4),
        Point::xy(cursor, 3.0),
        1.0,
        palette::ink().faded(0.5),
    );
    let names: Vec<(&str, Light)> = s
        .dispersion
        .iter()
        .enumerate()
        .map(|(i, (n, _, _))| (*n, palette::series(i)))
        .collect();
    ax.legend(c, &names);
}

fn draw_polarizations(c: &mut Canvas, rect: Rect, tau: f64) {
    let (_, fields) = scans().crystal;
    let origin = Point::xy(0.0, 0.0);
    let unit = unit(c);
    let inset = rect.inset(14.0 * unit, 26.0 * unit, 10.0 * unit, 34.0 * unit);
    let ax = Axes::equal(inset, origin, 1.3);
    ax.frame(c, "CRYSTAL MODES", "X", "");
    let faint = palette::grid().faded(0.6);
    ax.line(c, Point::xy(-1.3, 0.0), Point::xy(1.3, 0.0), 1.0, faint);
    ax.line(c, Point::xy(0.0, -1.3), Point::xy(0.0, 1.3), 1.0, faint);
    // Each mode's field oscillating in its plane at the entrance face, in phase.
    for (f, col) in fields.iter().zip([palette::red(), palette::blue()]) {
        // The electric vector lies across the beam: its x and y, at unit length.
        let (e, _) = arrows(*f);
        let a = e.normalized().into_inner();
        let along = |k: f64| origin + Point::direction(a.e1(), a.e2()).gp(k);
        ax.dashed(c, &[along(-1.0), along(1.0)], 1.0, 4.0, col.faded(0.7));
        ax.arrow(c, origin, along(1.0), 2.5, 9.0 * unit, col);
        ax.scatter(
            c,
            &[along(phasor(tau).e20())],
            Marker::Dot,
            7.0 * unit,
            palette::ink(),
        );
    }
    // The names in the top left corner (the panel is narrower than it is tall).
    let corner = ax.at(0.0, 1.0);
    let slow = corner + Point2::direction(0.1, -0.2);
    let fast = corner + Point2::direction(0.1, -0.45);
    let size = 10.0 * text_scale(rect, unit);
    ax.text(c, slow, "SLOW", size, palette::red(), Align::Left);
    ax.text(c, fast, "FAST", size, palette::blue(), Align::Left);
}

fn draw_fresnel(c: &mut Canvas, rect: Rect, angle: f64) {
    let s = scans();
    let unit = unit(c);
    let ax = Axes::equal(
        rect.inset(14.0 * unit, 26.0 * unit, 10.0 * unit, 34.0 * unit),
        Point::xy(0.0, 0.05),
        1.1,
    );
    ax.frame(c, "FRESNEL SURFACES", "X", "");
    // Zero along +z (up), angles towards +x (right): a point is the speed times the direction
    // from the origin.
    let origin = Point::xy(0.0, 0.0);
    let place = |a: f64, v: f64| -> Point<(), f64> {
        let d = direction(a);
        origin + Point::direction(d.e1(), d.e3()).gp(v)
    };
    for r in [0.25, 0.5, 0.75, 1.0] {
        let ring: Vec<Point<(), f64>> = (0..=72)
            .map(|k| place(core::f64::consts::TAU * k as f64 / 72.0, r))
            .collect();
        ax.polyline(c, &ring, 1.0, palette::grid().faded(0.6));
    }
    for (m, (name, per_angle)) in s.sheets.iter().enumerate() {
        let col = palette::series([0, 2, 4][m]);
        let branches = per_angle.iter().map(Vec::len).max().unwrap_or(0);
        for b in 0..branches {
            let pts: Vec<Point<(), f64>> = s
                .angles
                .iter()
                .zip(per_angle)
                .filter_map(|(a, v)| v.get(b).map(|v| place(*a, *v)))
                .collect();
            ax.polyline(c, &pts, 1.6, col.faded(0.9));
        }
        // Where the sweeping direction meets each sheet, from the nearest scanned angle.
        let i = ((angle / core::f64::consts::TAU * (s.angles.len() - 1) as f64).round() as usize)
            .min(s.angles.len() - 1);
        let hits: Vec<Point<(), f64>> = per_angle[i]
            .iter()
            .map(|v| place(s.angles[i], *v))
            .collect();
        ax.scatter(c, &hits, Marker::Dot, 7.0 * unit, col);
        ax.text(
            c,
            Point::xy(-1.0, -0.62 - 0.17 * m as f64),
            name,
            9.0 * text_scale(rect, unit),
            col,
            Align::Left,
        );
    }
    let d = place(angle, 1.1);
    ax.line(c, origin, d, 1.0, palette::ink().faded(0.6));
}

fn draw_drag(c: &mut Canvas, rect: Rect, beta: f64) {
    let s = scans();
    #[allow(clippy::disallowed_methods)] // the refractive index √(εμ): a physical law
    let n = (EPS_GLASS * MU_GLASS).sqrt();
    let unit = unit(c);
    let inset = rect.inset(40.0 * unit, 26.0 * unit, 10.0 * unit, 34.0 * unit);
    let ax = Axes::new(inset, [-0.6, 0.6], [0.0, 1.0]);
    ax.frame(c, "FRESNEL DRAG", "MEDIUM SPEED", "");
    let fine = linspace(-0.6, 0.6, 121);
    let curve = |f: &dyn Fn(f64) -> f64| -> Vec<Point<(), f64>> {
        fine.iter().map(|b| Point::xy(*b, f(*b))).collect()
    };
    let coeff = 1.0 - 1.0 / (n * n);
    let (down, up) = (palette::orange(), palette::sky());
    ax.polyline(c, &curve(&|b| add_speeds(1.0 / n, b)), 1.5, down);
    ax.polyline(c, &curve(&|b| add_speeds(1.0 / n, -b)), 1.5, up);
    // First-order Fresnel drag, dashed.
    let first_order = |b: f64| 1.0 / n + b * coeff;
    ax.dashed(c, &curve(&first_order), 1.0, 5.0, down.faded(0.6));
    ax.dashed(c, &curve(&|b| first_order(-b)), 1.0, 5.0, up.faded(0.6));
    let pts = |v: &[f64]| -> Vec<Point<(), f64>> {
        s.betas
            .iter()
            .zip(v)
            .map(|(b, v)| Point::xy(*b, *v))
            .collect()
    };
    ax.scatter(c, &pts(&s.drag.0), Marker::Dot, 6.0 * unit, down);
    ax.scatter(c, &pts(&s.drag.1), Marker::Square, 6.0 * unit, up);
    let now = [
        Point::xy(beta, add_speeds(1.0 / n, beta)),
        Point::xy(beta, add_speeds(1.0 / n, -beta)),
    ];
    let cursor = palette::ink().faded(0.5);
    ax.line(c, Point::xy(beta, 0.0), Point::xy(beta, 1.0), 1.0, cursor);
    ax.scatter(c, &now, Marker::Ring, 11.0 * unit, palette::ink());
    ax.legend(c, &[("WITH FLOW", down), ("AGAINST", up)]);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let at = f64::from((t / SECONDS).rem_euclid(1.0));
    let tau = core::f64::consts::TAU * 2.0 * at;
    let s = scans();
    // Below the caption (its subtitle's baseline three caption sizes down), two rows: the
    // waves above, the plots below, by fractions across.
    let unit = c.unit();
    let top = (h / 30.0).clamp(10.0 * unit, 22.0 * unit) * 3.0 + 4.0 * unit;
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
    let sweep = 0.5 - 0.5 * phasor(core::f64::consts::TAU * at).e20();
    draw_dispersion(c, lower(0.0, 0.34), 0.05 + 1.45 * sweep);
    draw_polarizations(c, lower(0.34, 0.5), tau);
    draw_fresnel(c, lower(0.5, 0.75), core::f64::consts::TAU * at);
    draw_drag(c, lower(0.75, 1.0), -0.6 + 1.2 * sweep);
    caption(
        c,
        "CONSTITUTIVE MAPS: GLASS, CRYSTALS, MOVING MEDIA",
        "PLANE WAVES WHERE THE 8X6 WAVE MAP HAS A NULLSPACE (STA)",
    );
}

fn main() {
    run(Anim::new("constitutive", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::constitutive::*;
    use gax::ApproxEq;
    use gax::sta::{Pseudoscalar, Trivector};
    use gax_numga_examples::rng::{Draw, Rng, rng};

    fn speeds() -> Vec<f64> {
        linspace(0.05, 1.5, 6001)
    }

    fn tolerance() -> f64 {
        1.45 / 6000.0
    }

    fn phase_speeds(medium: Medium, direction: V, speeds: &[f64]) -> Vec<f64> {
        minimum_speeds(speeds, &dispersion_scan(speeds, medium, direction), 2e-3)
    }

    fn all_close(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(p, q)| (p - q).abs() <= tol)
    }

    /// Whether `a` is within `tol` of `b`, coefficient by coefficient.
    fn near<X: ApproxEq>(a: X, b: X, tol: f64) -> bool {
        a.max_abs_diff(&b) <= tol
    }

    /// A field with coefficients uniform in `[-1, 1)`.
    fn bivector(r: &mut Rng) -> B {
        B::from_coeffs(core::array::from_fn(|_| r.range(-1.0, 1.0)))
    }

    #[test]
    fn observer_projectors_are_complementary_idempotents() {
        let (electric, magnetic) = observer_projectors(t());
        assert!(near(electric.of(electric), electric, 1e-14));
        assert!(near(magnetic.of(magnetic), magnetic, 1e-14));
        assert!(near(electric.of(magnetic), Medium::zero(), 1e-14));
        let f = field(1.0, 2.0, 5.0, 3.0, 0.0, 4.0);
        assert!(near(electric.of(f) + magnetic.of(f), f, 1e-14));
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // the speeds 1/√(εμ) it is checked against
    fn isotropic_medium_speed_is_one_over_n() {
        let s = speeds();
        for (eps, mu) in [(2.25, 1.0), (4.0, 1.5)] {
            let chi = isotropic_medium(eps, mu, t());
            assert!(all_close(
                &phase_speeds(chi, z(), &s),
                &[1.0 / (eps * mu).sqrt()],
                tolerance()
            ));
        }
    }

    #[test]
    fn material_and_stress_forms_give_the_electric_and_magnetic_energy() {
        let (eps, mu) = (2.25, 1.5);
        let medium = isotropic_medium(eps, mu, t());
        let mut rng = rng(318);
        let (electric, magnetic) = observer_projectors(t());
        let bilinear_form = pair(open(), medium);
        let energy_form = pair(-(t() >> open()), medium);
        for _ in 0..17 {
            let f = bivector(&mut rng);
            let (e, m) = (electric.of(f), magnetic.of(f));
            let e2 = e.scalar_product(e).s();
            let m2 = m.scalar_product(m).s();
            let action = (eps * e2 + m2 / mu) / 2.0;
            let energy = (eps * e2 - m2 / mu) / 2.0;
            let stress_form = pair(open_vector(), stress_energy(f, medium.of(f)));
            assert!((bilinear_form.of(f).of(f).s() / 2.0 - action).abs() < 1e-12);
            assert!((energy_form.of(f).of(f).s() / 2.0 - energy).abs() < 1e-12);
            assert!((stress_form.of(t()).of(t()).s() - energy).abs() < 1e-12);
            let vacuum = isotropic_medium(1.0, 1.0, t());
            let current = stress_energy(f, vacuum.of(f));
            let expected = (f.cast::<gax::sta::Even>() >> open_vector())
                .hodge()
                .gp(-0.5);
            assert!(near(current, expected, 1e-12));
        }
        let flux_field = field(1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let flux_form = pair(
            open_vector(),
            stress_energy(flux_field, medium.of(flux_field)),
        );
        assert!((flux_form.of(z()).of(t()).s() + 1.0 / mu).abs() < 1e-12);
        assert!((flux_form.of(t()).of(z()).s() + eps).abs() < 1e-12);
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // the speeds 1/√(εμ) it is checked against
    fn crystal_is_birefringent_and_reduces_to_glass_when_isotropic() {
        let iso = crystal_medium([2.25; 3], 1.0, t());
        let glass = isotropic_medium(2.25, 1.0, t());
        assert!(near(iso, glass, 1e-14));
        let s = speeds();
        let crystal = crystal();
        assert!(all_close(
            &phase_speeds(crystal, z(), &s),
            &[1.0 / 2.25f64.sqrt(), 1.0 / 1.5f64.sqrt()],
            tolerance()
        ));
        assert!(all_close(
            &phase_speeds(crystal, x(), &s),
            &[1.0 / 1.5f64.sqrt()],
            tolerance()
        ));
    }

    #[test]
    fn crystal_fields_satisfy_maxwell_and_lie_in_their_material_planes() {
        let crystal = crystal();
        let (speeds, fields) = field_modes();
        let (electric, magnetic) = observer_projectors(t());
        let electric_planes = [
            field(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            field(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        ];
        // numga's xz and yz.
        let magnetic_planes = [
            field(0.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            field(0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        ];
        for (s, f) in speeds.iter().zip(fields) {
            let k = t().gp(*s) + z();
            assert!(near(k ^ f, Trivector::zero(), 1e-12));
            assert!(near(k ^ crystal.of(f), Trivector::zero(), 1e-12));
            let e_part = electric_planes
                .iter()
                .fold(B::zero(), |acc, p| acc + *p * (*p | f));
            let m_part = magnetic_planes
                .iter()
                .fold(B::zero(), |acc, p| acc + *p * (*p | f));
            assert!(near(electric.of(f), e_part, 1e-12));
            assert!(near(magnetic.of(f), -m_part, 1e-12));
        }
    }

    #[test]
    fn vacuum_wave_has_two_transverse_field_modes() {
        let vacuum = isotropic_medium(1.0, 1.0, t());
        let k = t() + z();
        let (sigma, fields) = svd(wave_map(k, vacuum));
        assert_eq!(sigma.iter().filter(|s| **s < 1e-12).count(), 2);
        let action_form = pair(open(), vacuum);
        for f in &fields[4..] {
            assert!(near(k ^ *f, Trivector::zero(), 1e-12));
            assert!(near(k ^ vacuum.of(*f), Trivector::zero(), 1e-12));
            assert!((action_form.of(*f).of(*f).s() / 2.0).abs() < 1e-12);
            let stress_form = pair(open_vector(), stress_energy(*f, vacuum.of(*f)));
            assert!(stress_form.of(t()).of(t()).s() > 0.0);
        }
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // the speeds 1/√(εμ) it is checked against
    fn ferrite_lifts_permeability_through_the_dual_field() {
        let ferrite = ferrite_medium(2.25, [1.0, 0.5, 1.0], t());
        assert!(all_close(
            &phase_speeds(ferrite, z(), &speeds()),
            &[1.0 / 4.5f64.sqrt(), 1.0 / 2.25f64.sqrt()],
            tolerance()
        ));
    }

    #[test]
    fn axion_term_is_invisible_to_bulk_waves() {
        let glass = isotropic_medium(2.25, 1.0, t());
        let f = field(2.0, 0.0, 0.0, 3.0, 0.0, 0.0);
        let base_form = pair(open(), glass);
        let base_current = stress_energy(f, glass.of(f));
        let s = speeds();
        for alpha in [0.4, -1.3] {
            let axion = axion_medium(glass, alpha);
            let axion_form = pair(open(), axion);
            let change = (axion_form.of(f).of(f).s() - base_form.of(f).of(f).s()) / 2.0;
            assert_eq!(
                phase_speeds(axion, z(), &s).len(),
                phase_speeds(glass, z(), &s).len()
            );
            assert!(all_close(
                &phase_speeds(axion, z(), &s),
                &phase_speeds(glass, z(), &s),
                tolerance()
            ));
            assert!((change - alpha * pair(f, f).s() / 2.0).abs() < 1e-12);
            let current = stress_energy(f, axion.of(f));
            assert!(near(current, base_current, 1e-12));
        }
    }

    #[test]
    fn material_and_stress_forms_transform_with_the_field_and_observer() {
        let (eps, mu, beta) = (2.25, 1.5, 0.35);
        let medium = isotropic_medium(eps, mu, t());
        let rotation = boost_rotor(beta, z());
        let observer = rotation >> t();
        let moving = boosted_medium(medium, beta, z());
        let rebuilt = isotropic_medium(eps, mu, observer);
        let mut rng = rng(627);
        let (form, moved_form) = (pair(open(), medium), pair(open(), moving));
        for _ in 0..13 {
            let f = bivector(&mut rng);
            let moved = rotation >> f;
            let current = stress_energy(f, medium.of(f));
            let moved_current = stress_energy(moved, moving.of(moved));
            assert!(near(moving.of(f), rebuilt.of(f), 1e-7));
            assert!((moved_form.of(moved).of(moved).s() - form.of(f).of(f).s()).abs() < 1e-7);
            let carried = rotation >> current.of(rotation << open_vector());
            assert!(near(moved_current, carried, 1e-7));
            let stress_form = pair(open_vector(), current);
            let moved_stress_form = pair(open_vector(), moved_current);
            assert!(
                (moved_stress_form.of(observer).of(observer).s() - stress_form.of(t()).of(t()).s())
                    .abs()
                    < 1e-7
            );
        }
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // the speeds 1/√(εμ) it is checked against
    fn moving_glass_shows_exact_fresnel_drag() {
        let (eps, mu) = (2.25f64, 1.0);
        let n = (eps * mu).sqrt();
        let glass = isotropic_medium(eps, mu, t());
        let s = speeds();
        for beta in [0.0, 0.3, -0.5] {
            let moving = boosted_medium(glass, beta, z());
            assert!(all_close(
                &phase_speeds(moving, z(), &s),
                &[add_speeds(1.0 / n, beta)],
                tolerance()
            ));
        }
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // the speeds 1/√(εμ) it is checked against
    fn fresnel_surface_and_drag_scenarios() {
        let angles = linspace(0.0, core::f64::consts::TAU, 12);
        let speeds = linspace(0.4, 1.0, 601);
        let surfaces = fresnel_surface(&angles, &speeds);
        assert_eq!(surfaces.len(), 3);
        assert!(
            surfaces
                .iter()
                .all(|(_, s)| s.len() == 12 && s[0].len() == 601)
        );
        let betas = linspace(-0.6, 0.6, 5);
        let (down, up) =
            fresnel_drag_velocities(EPS_GLASS, MU_GLASS, &betas, &linspace(0.05, 1.2, 1151));
        let n = (EPS_GLASS * MU_GLASS).sqrt();
        for (b, (d, u)) in betas.iter().zip(down.iter().zip(&up)) {
            assert!((d - add_speeds(1.0 / n, *b)).abs() < 2e-3);
            assert!((u - add_speeds(1.0 / n, -b)).abs() < 2e-3);
        }
    }

    /// The modes drawn travel along +z: the Poynting vector of the vacuum modes (E x B as
    /// the animation reads them off) points along the beam.
    #[test]
    fn drawn_fields_carry_energy_along_the_beam() {
        let (_, fields) = glass_modes();
        for f in fields {
            // E x B along +z: the plane E ^ B turns from x towards y.
            let (e, b) = arrows(f);
            assert!((e ^ b).e12() > 0.0, "{e:?} {b:?}");
        }
    }

    /// numga's regressive pairing, spelled out with the Hodge dual, is minus gax's regressive
    /// product on complementary grades.
    #[test]
    fn the_pairing_is_numga_s_regressive_product() {
        let numga =
            |a: B, b: B| (a.hodge() ^ b.hodge()) * Pseudoscalar::<(), f64>::new(1.0).inverse();
        let mut rng = rng(5);
        for _ in 0..7 {
            let (a, b) = (bivector(&mut rng), bivector(&mut rng));
            assert!((pair(a, b).s() - numga(a, b).s()).abs() < 1e-12);
            let v = V::new(rng.normal(), rng.normal(), rng.normal(), rng.normal());
            let w = v ^ a;
            let numga_vw = (v.hodge() ^ w.hodge()) * Pseudoscalar::<(), f64>::new(1.0).inverse();
            assert!((pair(v, w).s() - numga_vw.s()).abs() < 1e-12);
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
