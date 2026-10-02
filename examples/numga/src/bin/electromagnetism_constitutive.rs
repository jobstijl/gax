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

use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Marker, Rgb, Scene3, backdrop, caption, palette, plot,
    run,
};

mod constitutive {
    // numga's API in full: the tests use parts the animation does not.
    #![cfg_attr(not(test), allow(dead_code))]

    use gax::sta::{Bivector, Odd, Pseudoscalar, Trivector, Vector};

    pub type V = Vector<(), f64>;
    pub type B = Bivector<(), f64>;
    /// A constitutive map, `Antibivector <- Bivector` (antibivectors are bivectors in 4D).
    pub type Medium = Bivector<(Bivector,), f64>;
    /// A stress-energy current, `Antivector <- Vector`.
    pub type Current = Trivector<(Vector,), f64>;
    /// Both Maxwell residuals of a plane wave, `Odd <- Bivector`.
    pub type WaveMap = Odd<(Bivector,), f64>;
    pub type Rotor = gax::Unit<gax::sta::Even<(), f64>>;

    type P = Pseudoscalar<(), f64>;

    /// The unit four-volume `I = txyz`.
    pub fn pseudoscalar() -> P {
        Pseudoscalar::new(1.0)
    }

    /// numga's `dual`, the right Hodge dual: each basis blade's complement, signed by the
    /// metric (the signs of its vectors' squares), which is `x ↦ ~x I`. gax's `dual()` is the
    /// metric-free complement (ADR-009); in spacetime the two differ on every blade with an
    /// odd number of spatial vectors, and with the metric-free one vacuum `F ↦ F*` has no
    /// waves. So this port spells numga's dual out.
    pub trait Hodge {
        type Output;
        fn hodge(self) -> Self::Output;
    }

    impl<X> Hodge for X
    where
        X: gax::Reverse,
        X::Output: gax::Gp<P>,
    {
        type Output = <X::Output as gax::Gp<P>>::Output;
        fn hodge(self) -> Self::Output {
            gax::Gp::gp(gax::Reverse::reverse(self), pseudoscalar())
        }
    }

    /// numga's regressive product of complementary grades, `(a* ∧ b*)` read back through the
    /// Hodge dual (`1* = I`): the metric pairing behind the material and stress forms.
    #[allow(clippy::type_complexity)]
    pub fn pair<X: Hodge, Y: Hodge>(
        a: X,
        b: Y,
    ) -> <<X::Output as gax::Wedge<Y::Output>>::Output as gax::Gp<P>>::Output
    where
        X::Output: gax::Wedge<Y::Output>,
        <X::Output as gax::Wedge<Y::Output>>::Output: gax::Gp<P>,
    {
        gax::Gp::gp(
            gax::Wedge::wedge(a.hodge(), b.hodge()),
            pseudoscalar().inverse(),
        )
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

    /// Weight the observer's electric and magnetic planes, then take the Hodge dual.
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

    /// The singular values (descending) and right singular vectors of a wave map. gax's
    /// `svd` takes square maps only, and this one is 8 x 6; gax's one-sided Jacobi
    /// (`linalg::orthogonalize`) takes columns of any length, so it runs on the map's six
    /// columns directly.
    pub fn svd(w: WaveMap) -> ([f64; 6], [B; 6]) {
        let c = w.c;
        let columns: [[f64; 8]; 6] = core::array::from_fn(|j| core::array::from_fn(|i| c[i][j]));
        let (rotated, v): ([[f64; 8]; 6], [[f64; 6]; 6]) = gax::linalg::orthogonalize(&columns, 14);
        let norm = |k: usize| rotated[k].iter().map(|a| a * a).sum::<f64>().sqrt();
        let mut order: [usize; 6] = core::array::from_fn(|i| i);
        order.sort_by(|&p, &q| norm(q).total_cmp(&norm(p)));
        (
            order.map(norm),
            order.map(|k| {
                let r = v[k];
                Bivector::new(r[0], r[1], r[2], r[3], r[4], r[5])
            }),
        )
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
    pub fn dispersion_media() -> Vec<(&'static str, Medium, Vec<f64>)> {
        let glass = isotropic_medium(EPS_GLASS, MU_GLASS, t());
        let n = (EPS_GLASS * MU_GLASS).sqrt();
        let with_flow = (1.0 / n + BETA_BOOST) / (1.0 + BETA_BOOST / n);
        let against_flow = (1.0 / n - BETA_BOOST) / (1.0 - BETA_BOOST / n);
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
    pub fn field_modes() -> ([f64; 2], [B; 2]) {
        let speeds = [1.0 / 2.25f64.sqrt(), 1.0 / 1.5f64.sqrt()];
        (
            speeds,
            speeds.map(|s| field_eigenmode(t().gp(s) + z(), crystal())),
        )
    }

    /// Two field modes along z in glass, sharing one speed: the two smallest singular vectors.
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

/// A camera whose view centre lands on pixel `centre` with `scale` pixels per unit (the
/// projection centres on half the camera's size).
fn camera(centre: [f32; 2], scale: f32, azimuth: f32, elevation: f32) -> Camera {
    Camera::orbit(
        (2.0 * centre[0]) as usize,
        (2.0 * centre[1]) as usize,
        [0.0, 0.0, 0.0],
        20.0,
        azimuth,
        elevation,
        Lens::Parallel(centre[1] / scale),
    )
}

/// The electric and magnetic vectors `[x, y, z]` a rest observer reads off a field: `F · t`
/// and `F* · t`, as numga draws them.
fn arrows(f: B) -> ([f64; 3], [f64; 3]) {
    let e = f | t();
    let b = f.hodge() | t();
    ([e.e1(), e.e2(), e.e3()], [b.e1(), b.e2(), b.e3()])
}

/// The travelling wave of two modes at time `tau`: each mode normalized to a unit electric
/// amplitude, summed with its phase `ω (z / v - τ)`.
fn wave_at(modes: &([f64; 2], [B; 2]), z: f64, tau: f64) -> B {
    let (speeds, fields) = modes;
    speeds.iter().zip(fields).fold(B::zero(), |acc, (v, f)| {
        let e = *f | t();
        let unit = f.gp(1.0 / (-e.dot(e).s()).sqrt());
        acc + unit.gp((z / v - tau).cos())
    })
}

/// The wave in 3D, the beam running across the screen: world x along the beam (scaled),
/// world y and z the field's x and y.
fn draw_wave(c: &mut Canvas, rect: [f32; 4], modes: &([f64; 2], [B; 2]), tau: f64, label: &str) {
    let centre = [(rect[0] + rect[2]) * 0.5, (rect[1] + rect[3]) * 0.5 + 10.0];
    let length = 6.0f32;
    let cam = camera(centre, (rect[2] - rect[0]) * 0.125, -1.1, 0.35);
    c.clip(rect);
    let mut sc = Scene3::new(cam);
    let at = |z: f64, v: [f64; 3]| -> [f32; 3] {
        let along = (z / Z_MAX) as f32 * length - length / 2.0;
        [
            along + v[2] as f32 * 0.1,
            v[0] as f32 * 0.8,
            v[1] as f32 * 0.8,
        ]
    };
    sc.seg(
        at(0.0, [0.0; 3]),
        at(Z_MAX, [0.0; 3]),
        1.0,
        palette::grid(),
        1.0,
    );
    let samples = 200;
    let mut e_line = Vec::new();
    let mut b_line = Vec::new();
    for i in 0..samples {
        let z = Z_MAX * i as f64 / (samples - 1) as f64;
        let (e, b) = arrows(wave_at(modes, z, tau));
        e_line.push(at(z, e));
        b_line.push(at(z, b));
    }
    sc.polyline(&b_line, 1.5, palette::sky(), 0.8);
    sc.polyline(&e_line, 2.0, palette::orange(), 1.0);
    for k in 0..21 {
        let z = Z_MAX * k as f64 / 20.0;
        let (e, b) = arrows(wave_at(modes, z, tau));
        let base = at(z, [0.0; 3]);
        for (v, col) in [(e, palette::orange()), (b, palette::sky())] {
            let tip = at(z, v);
            sc.arrow(
                base,
                [tip[0] - base[0], tip[1] - base[1], tip[2] - base[2]],
                1.2,
                5.0,
                col,
            );
        }
    }
    sc.draw(c);
    c.unclip();
    c.text(
        label,
        (rect[0] + rect[2]) * 0.5,
        rect[1] + 14.0,
        11.0,
        palette::ink(),
        Align::Center,
    );
}

fn draw_dispersion(c: &mut Canvas, rect: [f32; 4], cursor: f64) {
    let s = scans();
    let ax = Axes::new(
        plot::inset(rect, 40.0, 26.0, 10.0, 34.0),
        [0.05, 1.5],
        [1e-4, 3.0],
    )
    .log_y();
    ax.frame(c, "SMALLEST SINGULAR VALUE", "PHASE SPEED", "");
    let k = s
        .speeds
        .iter()
        .position(|v| *v >= cursor)
        .unwrap_or(s.speeds.len() - 1);
    for (i, (_, scan, expected)) in s.dispersion.iter().enumerate() {
        let col = palette::series(i);
        for v in expected {
            ax.dashed(
                c,
                &[[*v as f32, 1e-4], [*v as f32, 3.0]],
                1.0,
                4.0,
                col,
                0.6,
            );
        }
        let pts: Vec<[f32; 2]> = s
            .speeds
            .iter()
            .zip(scan)
            .map(|(v, m)| [*v as f32, m.max(1e-4) as f32])
            .collect();
        ax.polyline(c, &pts, if i == 1 { 1.0 } else { 1.5 }, col, 0.9);
        ax.scatter(c, &pts[k..=k], Marker::Dot, 6.0, col, 1.0);
    }
    ax.line(
        c,
        [cursor as f32, 1e-4],
        [cursor as f32, 3.0],
        1.0,
        palette::ink(),
        0.5,
    );
    let names: Vec<(&str, Rgb)> = s
        .dispersion
        .iter()
        .enumerate()
        .map(|(i, (n, _, _))| (*n, palette::series(i)))
        .collect();
    ax.legend(c, &names);
}

fn draw_polarizations(c: &mut Canvas, rect: [f32; 4], tau: f64) {
    let (_, fields) = scans().crystal;
    let ax = Axes::equal(plot::inset(rect, 14.0, 26.0, 10.0, 34.0), [0.0, 0.0], 1.3);
    ax.frame(c, "CRYSTAL MODES", "X", "");
    ax.line(c, [-1.3, 0.0], [1.3, 0.0], 1.0, palette::grid(), 0.6);
    ax.line(c, [0.0, -1.3], [0.0, 1.3], 1.0, palette::grid(), 0.6);
    for (k, (f, col)) in fields
        .iter()
        .zip([palette::red(), palette::blue()])
        .enumerate()
    {
        let (e, _) = arrows(*f);
        let n = (e[0] * e[0] + e[1] * e[1]).sqrt();
        let a = [(e[0] / n) as f32, (e[1] / n) as f32];
        ax.dashed(c, &[[-a[0], -a[1]], a], 1.0, 4.0, col, 0.7);
        ax.arrow(c, [0.0, 0.0], a, 2.5, 9.0, col);
        // Each mode's field oscillating in its plane at the entrance face.
        let s = (tau + k as f64 * 0.0).cos() as f32;
        ax.scatter(
            c,
            &[[a[0] * s, a[1] * s]],
            Marker::Dot,
            7.0,
            palette::ink(),
            1.0,
        );
    }
    ax.text(c, [-1.2, 1.1], "SLOW", 10.0, palette::red(), Align::Left);
    ax.text(c, [-1.2, 0.9], "FAST", 10.0, palette::blue(), Align::Left);
}

fn draw_fresnel(c: &mut Canvas, rect: [f32; 4], angle: f64) {
    let s = scans();
    let ax = Axes::equal(plot::inset(rect, 14.0, 26.0, 10.0, 34.0), [0.0, 0.05], 1.1);
    ax.frame(c, "FRESNEL SURFACES", "X", "");
    for r in [0.25f32, 0.5, 0.75, 1.0] {
        let ring: Vec<[f32; 2]> = (0..=72)
            .map(|k| {
                let a = core::f32::consts::TAU * k as f32 / 72.0;
                [r * a.sin(), r * a.cos()]
            })
            .collect();
        ax.polyline(c, &ring, 1.0, palette::grid(), 0.6);
    }
    // Zero along +z (up), angles towards +x (right): a point is the speed times the direction.
    let place = |a: f64, v: f64| -> [f32; 2] {
        let d = direction(a);
        [(v * d.e1()) as f32, (v * d.e3()) as f32]
    };
    for (m, (name, per_angle)) in s.sheets.iter().enumerate() {
        let col = palette::series([0, 2, 4][m]);
        let branches = per_angle.iter().map(Vec::len).max().unwrap_or(0);
        for b in 0..branches {
            let pts: Vec<[f32; 2]> = s
                .angles
                .iter()
                .zip(per_angle)
                .filter_map(|(a, v)| v.get(b).map(|v| place(*a, *v)))
                .collect();
            ax.polyline(c, &pts, 1.6, col, 0.9);
        }
        // Where the sweeping direction meets each sheet, from the nearest scanned angle.
        let i = ((angle / core::f64::consts::TAU * (s.angles.len() - 1) as f64).round() as usize)
            .min(s.angles.len() - 1);
        let hits: Vec<[f32; 2]> = per_angle[i]
            .iter()
            .map(|v| place(s.angles[i], *v))
            .collect();
        ax.scatter(c, &hits, Marker::Dot, 7.0, col, 1.0);
        ax.text(
            c,
            [-1.0, -0.75 - 0.13 * m as f32],
            name,
            9.0,
            col,
            Align::Left,
        );
    }
    let d = place(angle, 1.1);
    ax.line(c, [0.0, 0.0], d, 1.0, palette::ink(), 0.6);
}

fn draw_drag(c: &mut Canvas, rect: [f32; 4], beta: f64) {
    let s = scans();
    let n = (EPS_GLASS * MU_GLASS).sqrt();
    let ax = Axes::new(
        plot::inset(rect, 40.0, 26.0, 10.0, 34.0),
        [-0.6, 0.6],
        [0.0, 1.0],
    );
    ax.frame(c, "FRESNEL DRAG", "MEDIUM SPEED", "");
    let fine = linspace(-0.6, 0.6, 121);
    let curve = |f: &dyn Fn(f64) -> f64| -> Vec<[f32; 2]> {
        fine.iter().map(|b| [*b as f32, f(*b) as f32]).collect()
    };
    let coeff = 1.0 - 1.0 / (n * n);
    let (down, up) = (palette::orange(), palette::sky());
    ax.polyline(
        c,
        &curve(&|b| (1.0 / n + b) / (1.0 + b / n)),
        1.5,
        down,
        1.0,
    );
    ax.polyline(c, &curve(&|b| (1.0 / n - b) / (1.0 - b / n)), 1.5, up, 1.0);
    // First-order Fresnel drag, dashed.
    ax.dashed(c, &curve(&|b| 1.0 / n + b * coeff), 1.0, 5.0, down, 0.6);
    ax.dashed(c, &curve(&|b| 1.0 / n - b * coeff), 1.0, 5.0, up, 0.6);
    let pts = |v: &[f64]| -> Vec<[f32; 2]> {
        s.betas
            .iter()
            .zip(v)
            .map(|(b, v)| [*b as f32, *v as f32])
            .collect()
    };
    ax.scatter(c, &pts(&s.drag.0), Marker::Dot, 6.0, down, 1.0);
    ax.scatter(c, &pts(&s.drag.1), Marker::Square, 6.0, up, 1.0);
    let now = [
        [beta as f32, ((1.0 / n + beta) / (1.0 + beta / n)) as f32],
        [beta as f32, ((1.0 / n - beta) / (1.0 - beta / n)) as f32],
    ];
    ax.line(
        c,
        [beta as f32, 0.0],
        [beta as f32, 1.0],
        1.0,
        palette::ink(),
        0.5,
    );
    ax.scatter(c, &now, Marker::Ring, 11.0, palette::ink(), 1.0);
    ax.legend(c, &[("WITH FLOW", down), ("AGAINST", up)]);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.width as f32, c.height as f32);
    let at = f64::from((t / SECONDS).rem_euclid(1.0));
    let tau = core::f64::consts::TAU * 2.0 * at;
    let s = scans();
    let top = h * 0.1;
    let mid = top + (h - top) * 0.5;
    draw_wave(
        c,
        [0.0, top, w * 0.5, mid],
        &s.glass,
        tau,
        "GLASS: ONE SPEED, FIXED POLARIZATION",
    );
    draw_wave(
        c,
        [w * 0.5, top, w, mid],
        &s.crystal,
        tau,
        "CRYSTAL: SLOW AND FAST MODES, THE POLARIZATION TURNS",
    );
    let sweep = 0.5 - 0.5 * (core::f64::consts::TAU * at).cos();
    draw_dispersion(c, [0.0, mid, w * 0.34, h], 0.05 + 1.45 * sweep);
    draw_polarizations(c, [w * 0.34, mid, w * 0.5, h], tau);
    draw_fresnel(c, [w * 0.5, mid, w * 0.75, h], core::f64::consts::TAU * at);
    draw_drag(c, [w * 0.75, mid, w, h], -0.6 + 1.2 * sweep);
    caption(
        c,
        "CONSTITUTIVE MAPS: WAVES IN GLASS, CRYSTALS AND MOVING MEDIA",
        "PLANE WAVES WHERE THE 8X6 WAVE MAP F -> K^F + (K^X(F))* HAS A NULLSPACE (STA)",
    );
}

fn main() {
    run(Anim::new("constitutive", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::constitutive::*;

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

    fn near<const N: usize>(a: [f64; N], b: [f64; N], tol: f64) -> bool {
        a.iter().zip(b).all(|(p, q)| (p - q).abs() <= tol)
    }

    struct Rng(u64);
    impl Rng {
        fn unit(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        }
        fn bivector(&mut self) -> B {
            B::new(
                self.unit(),
                self.unit(),
                self.unit(),
                self.unit(),
                self.unit(),
                self.unit(),
            )
        }
    }

    #[test]
    fn observer_projectors_are_complementary_idempotents() {
        let (electric, magnetic) = observer_projectors(t());
        let zero = |m: Medium| m.c.iter().flatten().all(|v| v.abs() <= 1e-14);
        assert!(zero(electric.of(electric) - electric));
        assert!(zero(magnetic.of(magnetic) - magnetic));
        assert!(zero(electric.of(magnetic)));
        let f = field(1.0, 2.0, 5.0, 3.0, 0.0, 4.0);
        assert!(near(
            (electric.of(f) + magnetic.of(f) - f).c,
            [0.0; 6],
            1e-14
        ));
    }

    #[test]
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
        let mut rng = Rng(318);
        let (electric, magnetic) = observer_projectors(t());
        let bilinear_form = pair(open(), medium);
        let energy_form = pair(-(t() >> open()), medium);
        for _ in 0..17 {
            let f = rng.bivector();
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
            assert!(
                current
                    .c
                    .iter()
                    .flatten()
                    .zip(expected.c.iter().flatten())
                    .all(|(a, b)| (a - b).abs() < 1e-12)
            );
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
    fn crystal_is_birefringent_and_reduces_to_glass_when_isotropic() {
        let iso = crystal_medium([2.25; 3], 1.0, t());
        let glass = isotropic_medium(2.25, 1.0, t());
        assert!(
            iso.c
                .iter()
                .flatten()
                .zip(glass.c.iter().flatten())
                .all(|(a, b)| (a - b).abs() <= 1e-14)
        );
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
            assert!(near((k ^ f).c, [0.0; 4], 1e-12));
            assert!(near((k ^ crystal.of(f)).c, [0.0; 4], 1e-12));
            let e_part = electric_planes
                .iter()
                .fold(B::zero(), |acc, p| acc + *p * (*p | f));
            let m_part = magnetic_planes
                .iter()
                .fold(B::zero(), |acc, p| acc + *p * (*p | f));
            assert!(near((electric.of(f) - e_part).c, [0.0; 6], 1e-12));
            assert!(near((magnetic.of(f) + m_part).c, [0.0; 6], 1e-12));
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
            assert!(near((k ^ *f).c, [0.0; 4], 1e-12));
            assert!(near((k ^ vacuum.of(*f)).c, [0.0; 4], 1e-12));
            assert!((action_form.of(*f).of(*f).s() / 2.0).abs() < 1e-12);
            let stress_form = pair(open_vector(), stress_energy(*f, vacuum.of(*f)));
            assert!(stress_form.of(t()).of(t()).s() > 0.0);
        }
    }

    #[test]
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
            assert!(
                current
                    .c
                    .iter()
                    .flatten()
                    .zip(base_current.c.iter().flatten())
                    .all(|(a, b)| (a - b).abs() < 1e-12)
            );
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
        let mut rng = Rng(627);
        let (form, moved_form) = (pair(open(), medium), pair(open(), moving));
        for _ in 0..13 {
            let f = rng.bivector();
            let moved = rotation >> f;
            let current = stress_energy(f, medium.of(f));
            let moved_current = stress_energy(moved, moving.of(moved));
            assert!(near((moving.of(f) - rebuilt.of(f)).c, [0.0; 6], 1e-7));
            assert!((moved_form.of(moved).of(moved).s() - form.of(f).of(f).s()).abs() < 1e-7);
            let carried = rotation >> current.of(rotation << open_vector());
            assert!(
                moved_current
                    .c
                    .iter()
                    .flatten()
                    .zip(carried.c.iter().flatten())
                    .all(|(a, b)| (a - b).abs() < 1e-7)
            );
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
    fn moving_glass_shows_exact_fresnel_drag() {
        let (eps, mu) = (2.25f64, 1.0);
        let n = (eps * mu).sqrt();
        let glass = isotropic_medium(eps, mu, t());
        let s = speeds();
        for beta in [0.0, 0.3, -0.5] {
            let moving = boosted_medium(glass, beta, z());
            assert!(all_close(
                &phase_speeds(moving, z(), &s),
                &[(1.0 / n + beta) / (1.0 + beta / n)],
                tolerance()
            ));
        }
    }

    #[test]
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
            assert!((d - (1.0 / n + b) / (1.0 + b / n)).abs() < 2e-3);
            assert!((u - (1.0 / n - b) / (1.0 - b / n)).abs() < 2e-3);
        }
    }

    /// The modes drawn travel along +z: the Poynting vector of the vacuum modes (E x B as
    /// the animation reads them off) points along the beam.
    #[test]
    fn drawn_fields_carry_energy_along_the_beam() {
        let (_, fields) = glass_modes();
        for f in fields {
            let (e, b) = super::arrows(f);
            assert!(e[0] * b[1] - e[1] * b[0] > 0.0, "{e:?} {b:?}");
        }
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
