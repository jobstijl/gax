//! numga's `electromagnetism/maxwell`: stress-energy maps in spacetime algebra (STA, `t² = 1`,
//! `x² = y² = z² = -1`). A stress-energy map sends an observer `v` to the momentum flux it sees,
//! a `Vector <- Vector` extensor. The electromagnetic one is the field's sandwich,
//! `T(v) = ½ F v ~F`: traceless, and for a travelling wave a single null dyad. Dust is the rank-1
//! map `ρ u (u · v)`, and a cloud of particles sums such dyads into an ideal fluid whose spectrum
//! is the energy density and minus the pressure; a cloud of light rays is exactly traceless,
//! radiation with a third of its energy as pressure.
//!
//! The animation shows, on the left, a general field boosted back and forth along `x`: the
//! electric and magnetic vectors and the energy flux its stress-energy map gives the rest
//! observer, with the energy density over the rapidity while the invariants stay put. On the
//! right, a particle cloud speeds up from dust towards light: its velocities in space, and the
//! pressure over the energy density from the map's spectrum, against `v² / 3` and the null
//! cloud's exact third.

use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Marker, Rgb, Scene3, backdrop, caption, palette, run,
};

mod maxwell {
    // numga's API in full: the tests use parts the animation does not.
    #![cfg_attr(not(test), allow(dead_code))]

    use gax::sta::{Bivector, Even, Scalar, Vector};

    pub type V = Vector<(), f64>;
    pub type F = Bivector<(), f64>;
    /// A stress-energy map: an observer in, the momentum flux it sees out.
    pub type StressEnergy = Vector<(Vector,), f64>;
    pub type Boost = gax::Unit<Even<(), f64>>;

    /// The open vector: numga's `Vector` used as a hole in an expression.
    pub fn open() -> StressEnergy {
        Vector::slot()
    }

    /// The rest observer and the spatial axes.
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

    /// A field from numga's blade coefficients `tx, ty, tz, yz, zx, xy`. gax orders the
    /// boost planes as `e10, e20, e30`, so numga's `tx = t ^ x` is `-e10`.
    pub fn field(tx: f64, ty: f64, tz: f64, yz: f64, zx: f64, xy: f64) -> F {
        Bivector::new(-tx, -ty, -tz, yz, zx, xy)
    }

    /// The electromagnetic stress-energy map `T(v) = ½ F v ~F`. gax sandwiches with versor
    /// kinds only, so the field is cast to the even subalgebra first.
    pub fn stress_energy(f: F) -> StressEnergy {
        0.5 * (f.cast::<Even>() >> open())
    }

    /// The same map from the Lorentz force generator `v ↦ F × v`, iterated, minus the
    /// Lagrangian `⟨F F⟩₀` (squared electric minus squared magnetic field) as an isotropic
    /// pressure.
    pub fn stress_from_force(f: F) -> StressEnergy {
        f.commutator(f.commutator(open())) - (0.5 * lagrangian(f)) * open()
    }

    /// `⟨F F⟩₀`, a Lorentz invariant.
    pub fn lagrangian(f: F) -> f64 {
        (f * f).grade::<0>().s()
    }

    /// The dyad `u (u · v)`: dust moving with four-velocity `u`, per unit density.
    pub fn dyad(u: V) -> StressEnergy {
        u * (u | open())
    }

    /// Normal traction on a face with unit spatial normal `n` (`n² = -1`): `n⁻¹ · T(n)`.
    pub fn normal_stress(t_map: StressEnergy, n: V) -> f64 {
        (n.inverse() | t_map.of(n)).s()
    }

    /// The energy density a unit observer `u` sees, `u · T(u)`.
    pub fn energy(t_map: StressEnergy, u: V) -> f64 {
        (u | t_map.of(u)).s()
    }

    /// The flux a rest observer sees, less its energy part: the Poynting vector `[x, y, z]`.
    pub fn poynting(t_map: StressEnergy) -> [f64; 3] {
        let flux = t_map.of(t());
        let p = flux - t() * (t() | flux);
        [p.e1(), p.e2(), p.e3()]
    }

    /// The real spectrum of a stress-energy map, ascending. numga calls a general eigensolver
    /// (gax's `eigvals`); the symmetric pencil keeps the values exactly real. `T` is self-adjoint for the Minkowski metric `g`, so
    /// `T v = λ v` is the symmetric pencil `S(v, ·) = λ g(v, ·)` with `S(a, b) = a · T(b)`.
    /// The metric is indefinite, but a cloud's `S(v, v) = Σ m (u · v)²` is positive definite,
    /// so the pencil is solved the other way round, `g(v, ·) = λ⁻¹ S(v, ·)`.
    pub fn spectrum(t_map: StressEnergy) -> [f64; 4] {
        let s: Scalar<(Vector, Vector), f64> = open() | t_map;
        let g: Scalar<(Vector, Vector), f64> = open() | open();
        let (inverse, _) = g.eigh_with(s);
        let mut values = inverse.map(|m| 1.0 / m);
        values.sort_by(f64::total_cmp);
        values
    }

    /// The boost by rapidity `zeta` along `axis` (a unit spatial vector), `exp(axis t ζ / 2)`.
    pub fn boost(zeta: f64, axis: V) -> Boost {
        ((axis ^ t()) * (0.5 * zeta)).exp()
    }

    /// The map seen after boosting everything: `R T(~R v R) ~R`.
    pub fn boosted(r: Boost, t_map: StressEnergy) -> StressEnergy {
        r >> t_map.of(r << open())
    }

    /// The sum `Σ m u (u · v)` over a cloud of four-velocities, pairwise as NumPy sums, so
    /// that rounding grows with the logarithm of the cloud's size (the trace checks hold to
    /// `1e-14`).
    pub fn cloud_tensor(us: &[V], mass: f64) -> StressEnergy {
        if us.len() <= 8 {
            us.iter()
                .fold(StressEnergy::zero(), |acc, u| acc + dyad(*u).gp(mass))
        } else {
            let (a, b) = us.split_at(us.len() / 2);
            cloud_tensor(a, mass) + cloud_tensor(b, mass)
        }
    }

    /// The ideal fluid at rest: `(ρ + p) t (t · v) - p v`.
    pub fn fluid(energy: f64, pressure: f64) -> StressEnergy {
        dyad(t()).gp(energy + pressure) - pressure * open()
    }

    pub use gax_numga_examples::rng::Rng;

    /// `n` unit spatial directions, uniform on the sphere.
    pub fn sphere_directions(n: usize, rng: &mut Rng) -> Vec<V> {
        (0..n)
            .map(|_| {
                Vector::new(0.0, rng.normal(), rng.normal(), rng.normal())
                    .normalized()
                    .into_inner()
            })
            .collect()
    }

    /// `n` four-velocities at one speed, directions uniform on the sphere.
    pub fn isotropic_cloud(n: usize, speed: f64, rng: &mut Rng) -> Vec<V> {
        let gamma = 1.0 / (1.0 - speed * speed).sqrt();
        sphere_directions(n, rng)
            .into_iter()
            .map(|d| (t() + d.gp(speed)).gp(gamma))
            .collect()
    }

    /// `n` null rays `t + direction`.
    pub fn null_cloud(n: usize, rng: &mut Rng) -> Vec<V> {
        sphere_directions(n, rng)
            .into_iter()
            .map(|d| t() + d)
            .collect()
    }

    /// The general field of numga's commutator check, used by the animation.
    pub fn general_field() -> F {
        field(0.7, -0.4, 1.1, 0.3, -0.8, 0.5)
    }

    /// The electric and magnetic vectors a rest observer reads off a field: `t · F` and
    /// `t · F*` (the complement turns magnetic planes into electric ones).
    pub fn electric_magnetic(f: F) -> ([f64; 3], [f64; 3]) {
        let e = t() | f;
        let b = t() | f.dual();
        ([e.e1(), e.e2(), e.e3()], [b.e1(), b.e2(), b.e3()])
    }
}

use maxwell::*;

const SECONDS: f32 = 8.0;
/// The largest rapidity of the boost.
const SWING: f64 = 0.9;

/// The rapidity of the boost at animation phase `p` (radians).
fn rapidity(p: f64) -> f64 {
    SWING * p.sin()
}

/// The cloud's speed at animation phase `p`: from dust nearly to light and back.
fn cloud_speed(p: f64) -> f64 {
    0.02 + 0.94 * 0.5 * (1.0 - p.cos())
}

/// The fixed cloud directions; the animation scales them by its speed.
fn directions() -> &'static [V] {
    static DIRS: std::sync::OnceLock<Vec<V>> = std::sync::OnceLock::new();
    DIRS.get_or_init(|| sphere_directions(2000, &mut Rng(0x9e37_79b9_7f4a_7c15)))
}

/// The null cloud's pressure over its energy, from its spectrum (exactly a third).
fn null_ratio() -> f64 {
    static R: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *R.get_or_init(|| {
        let rays: Vec<V> = directions().iter().map(|d| t() + *d).collect();
        let s = spectrum(cloud_tensor(&rays, 1.0 / rays.len() as f64));
        -(s[0] + s[1] + s[2]) / 3.0 / s[3]
    })
}

/// A camera whose view centre lands on pixel `centre` with `scale` pixels per unit: the
/// projection centres on half the camera's size, so the camera gets twice the centre's offsets.
fn camera(centre: [f32; 2], scale: f32, azimuth: f32, elevation: f32) -> Camera {
    Camera::orbit(
        (2.0 * centre[0]) as usize,
        (2.0 * centre[1]) as usize,
        [0.0, 0.0, 0.0],
        10.0,
        azimuth,
        elevation,
        Lens::Parallel(centre[1] / scale),
    )
}

fn f3(v: [f64; 3]) -> [f32; 3] {
    v.map(|c| c as f32)
}

/// Grey axes with their names at the ends.
fn axes3(c: &mut Canvas, cam: &Camera, s: &mut Scene3, len: f32) {
    for (i, name) in ["X", "Y", "Z"].iter().enumerate() {
        let mut a = [0.0; 3];
        a[i] = len;
        s.seg([0.0; 3], a, 1.0, palette::grid(), 1.0);
        a[i] = len * 1.1;
        if let Some(q) = cam.px(a) {
            c.text(name, q[0], q[1] + 4.0, 10.0, palette::grid(), Align::Center);
        }
    }
}

/// Text in a colour at a pixel, left aligned.
fn note(c: &mut Canvas, s: &str, x: f32, y: f32, col: Rgb) {
    c.text(s, x, y, 11.0, col, Align::Left);
}

/// The plot rectangle in the lower part of a panel.
fn lower(rect: [f32; 4]) -> [f32; 4] {
    [
        rect[0] + 52.0,
        rect[1] + (rect[3] - rect[1]) * 0.68,
        rect[2] - 16.0,
        rect[3] - 34.0,
    ]
}

/// The general field boosted by rapidity `zeta` along x, and its stress-energy map. The map is
/// boosted by covariance; it equals the boosted field's own map (checked in the tests).
fn boosted_field(zeta: f64) -> (F, StressEnergy) {
    let f = general_field();
    let r = boost(zeta, x());
    (r >> f, boosted(r, stress_energy(f)))
}

/// The rest observer's energy density and flux size, the invariant `⟨F F⟩₀` and the map's
/// trace, for the field boosted by `zeta`.
fn readout(zeta: f64) -> [f64; 4] {
    let (g, tm) = boosted_field(zeta);
    let p = poynting(tm);
    [
        energy(tm, t()),
        (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt(),
        lagrangian(g),
        tm.trace(),
    ]
}

fn draw_field(c: &mut Canvas, rect: [f32; 4], phase: f64, spin: f32) {
    let zeta = rapidity(phase);
    let (g, tm) = boosted_field(zeta);
    let (e, b) = electric_magnetic(g);
    let s = poynting(tm);
    let centre = [
        (rect[0] + rect[2]) * 0.5,
        rect[1] + (rect[3] - rect[1]) * 0.34,
    ];
    let cam = camera(centre, 48.0, 0.6 + spin, 0.35);
    c.clip([rect[0], rect[1], rect[2], lower(rect)[1] - 12.0]);
    let mut sc = Scene3::new(cam);
    axes3(c, &cam, &mut sc, 2.2);
    // The tips' paths over the whole swing.
    let n = 96;
    let tips: Vec<[[f32; 3]; 3]> = (0..=n)
        .map(|i| {
            let (g, tm) = boosted_field(SWING * (-1.0 + 2.0 * i as f64 / n as f64));
            let (e, b) = electric_magnetic(g);
            [f3(e), f3(b), f3(poynting(tm))]
        })
        .collect();
    let cols = [palette::orange(), palette::sky(), palette::yellow()];
    for (k, col) in cols.iter().enumerate() {
        let path: Vec<[f32; 3]> = tips.iter().map(|p| p[k]).collect();
        sc.polyline(&path, 1.0, *col, 0.45);
    }
    for (v, col) in [(e, cols[0]), (b, cols[1]), (s, cols[2])] {
        sc.arrow([0.0; 3], f3(v), 2.5, 10.0, col);
    }
    sc.draw(c);
    c.unclip();
    let (x0, y0) = (rect[0] + 14.0, rect[1] + 14.0);
    note(
        c,
        &format!("BOOST ALONG X, RAPIDITY {zeta:+.2}"),
        x0,
        y0,
        palette::ink(),
    );
    note(c, "E", x0, y0 + 16.0, palette::orange());
    note(c, "B", x0 + 16.0, y0 + 16.0, palette::sky());
    note(c, "POYNTING FLUX", x0 + 32.0, y0 + 16.0, palette::yellow());

    let ax = Axes::new(lower(rect), [-SWING as f32, SWING as f32], [-1.0, 6.0]);
    ax.frame(c, "", "RAPIDITY", "");
    let at = |k: usize| -> Vec<[f32; 2]> {
        (0..=n)
            .map(|i| {
                let z = SWING * (-1.0 + 2.0 * i as f64 / n as f64);
                [z as f32, readout(z)[k] as f32]
            })
            .collect()
    };
    let cols = [
        palette::orange(),
        palette::yellow(),
        palette::green(),
        palette::purple(),
    ];
    for (k, col) in cols.iter().enumerate() {
        ax.polyline(c, &at(k), 1.8, *col, 1.0);
    }
    let now = readout(zeta);
    ax.scatter(
        c,
        &[[zeta as f32, now[0] as f32], [zeta as f32, now[1] as f32]],
        Marker::Dot,
        8.0,
        palette::ink(),
        1.0,
    );
    ax.legend(
        c,
        &[
            ("ENERGY", cols[0]),
            ("FLUX", cols[1]),
            ("<FF>", cols[2]),
            ("TRACE", cols[3]),
        ],
    );
}

fn draw_cloud(c: &mut Canvas, rect: [f32; 4], phase: f64, spin: f32) {
    let speed = cloud_speed(phase);
    let gamma = 1.0 / (1.0 - speed * speed).sqrt();
    let dirs = directions();
    let us: Vec<V> = dirs.iter().map(|d| (t() + d.gp(speed)).gp(gamma)).collect();
    let tm = cloud_tensor(&us, 1.0 / us.len() as f64);
    let s = spectrum(tm);
    let (stresses, energy) = ([s[0], s[1], s[2]], s[3]);
    let pressure = -(stresses[0] + stresses[1] + stresses[2]) / 3.0;
    let centre = [
        (rect[0] + rect[2]) * 0.5,
        rect[1] + (rect[3] - rect[1]) * 0.36,
    ];
    let cam = camera(centre, 80.0, -0.4 + spin, 0.3);
    c.clip([rect[0], rect[1], rect[2], lower(rect)[1] - 12.0]);
    let mut sc = Scene3::new(cam);
    sc.sphere_wire([0.0; 3], 1.0, 12, palette::grid(), 0.5);
    for d in dirs.iter().step_by(4) {
        let p = f3([d.e1(), d.e2(), d.e3()].map(|v| v * speed));
        sc.dot(p, Marker::Dot, 3.0, palette::sky());
    }
    sc.draw(c);
    c.unclip();
    let (x0, y0) = (rect[0] + 14.0, rect[1] + 14.0);
    note(
        c,
        &format!("SPEED {speed:.2}   TRACE {:.3}", tm.trace()),
        x0,
        y0,
        palette::ink(),
    );
    note(
        c,
        &format!("ENERGY {energy:.2}   PRESSURE {pressure:.3}"),
        x0,
        y0 + 16.0,
        palette::ink(),
    );
    note(c, "VELOCITIES IN SPACE", x0, y0 + 32.0, palette::sky());

    let ax = Axes::new(lower(rect), [0.0, 1.0], [0.0, 0.4]);
    ax.frame(c, "", "SPEED", "");
    let theory: Vec<[f32; 2]> = (0..=60)
        .map(|i| {
            let v = i as f32 / 60.0;
            [v, v * v / 3.0]
        })
        .collect();
    ax.polyline(c, &theory, 1.5, palette::green(), 1.0);
    ax.dashed(
        c,
        &[[0.0, 1.0 / 3.0], [1.0, 1.0 / 3.0]],
        1.0,
        8.0,
        palette::grid(),
        1.0,
    );
    // Each stress eigenvalue over the energy: the three nearly coincide (isotropy).
    let marks: Vec<[f32; 2]> = stresses
        .iter()
        .map(|p| [speed as f32, (-p / energy) as f32])
        .collect();
    ax.scatter(c, &marks, Marker::Ring, 9.0, palette::orange(), 1.0);
    ax.scatter(
        c,
        &[[1.0, null_ratio() as f32]],
        Marker::Star,
        12.0,
        palette::yellow(),
        1.0,
    );
    ax.legend(
        c,
        &[
            ("V*V/3", palette::green()),
            ("-STRESS/ENERGY", palette::orange()),
            ("LIGHT", palette::yellow()),
        ],
    );
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.width as f32, c.height as f32);
    let phase = f64::from(t / SECONDS) * core::f64::consts::TAU;
    let spin = 0.25 * (core::f32::consts::TAU * t / SECONDS).sin();
    let top = h * 0.12;
    draw_field(c, [0.0, top, w * 0.5, h], phase, spin);
    draw_cloud(c, [w * 0.5, top, w, h], phase, spin);
    caption(
        c,
        "MAXWELL: STRESS-ENERGY MAPS IN SPACETIME",
        "T(V) = 1/2 F V ~F OF A BOOSTED FIELD (LEFT); A CLOUD OF DUST DYADS, DUST TO RADIATION (RIGHT)",
    );
}

fn main() {
    run(Anim::new("maxwell", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::maxwell::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    fn maps_close(a: StressEnergy, b: StressEnergy, tol: f64) -> bool {
        a.c.iter()
            .flatten()
            .zip(b.c.iter().flatten())
            .all(|(p, q)| (p - q).abs() <= tol)
    }

    fn random_vector(rng: &mut Rng) -> V {
        V::new(rng.normal(), rng.normal(), rng.normal(), rng.normal())
    }

    /// numga's `main`: every check of its tutorial.
    #[test]
    fn tutorial_checks() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        // A plane wave along +z: electric field along x, magnetic along y.
        let (ex, by) = (2.0, 2.0);
        let f = field(ex, 0.0, 0.0, 0.0, by, 0.0);
        let t_em = stress_energy(f);
        let energy = (t() | t_em.of(t())).s();
        assert!(close(energy, 0.5 * (ex * ex + by * by), 1e-14));
        let p = poynting(t_em);
        assert!(close(p[0], 0.0, 1e-14) && close(p[1], 0.0, 1e-14));
        assert!(close(p[2], ex * by, 1e-14), "{p:?}");
        for n in [x(), y()] {
            assert!(close(normal_stress(t_em, n), 0.0, 1e-14));
        }
        assert!(close(
            normal_stress(t_em, z()),
            -0.5 * (ex * ex + by * by),
            1e-14
        ));
        assert!(close(t_em.trace(), 0.0, 1e-14));
        // A travelling wave is one null dyad along k = t + z.
        let k = t() + z();
        assert!(maps_close(t_em, dyad(k).gp(ex * ex), 1e-14));
        let f_any = general_field();
        assert!(maps_close(
            stress_energy(f_any),
            stress_from_force(f_any),
            1e-14
        ));

        // Dust at rest, then boosted.
        let rho = 4.0;
        let t_dust = dyad(t()).gp(rho);
        let zeta = 0.6f64.atanh();
        let b = boost(zeta, z());
        let u = b >> t();
        let (gamma, beta) = (zeta.cosh(), zeta.tanh());
        let t_moving = dyad(u).gp(rho);
        for n in [x(), y(), z()] {
            assert!(close(normal_stress(t_dust, n), 0.0, 1e-14));
        }
        assert!(close(t_dust.trace(), rho, 1e-14));
        assert!(close(
            (t() | t_moving.of(t())).s(),
            rho * gamma * gamma,
            1e-14
        ));
        assert!(close(
            normal_stress(t_moving, z()),
            -rho * gamma * gamma * beta * beta,
            1e-14
        ));
        assert!(close(t_moving.trace(), rho, 1e-14));

        // A particle cloud is an ideal fluid.
        let n = 2000;
        let speed = 0.5;
        let cloud = isotropic_cloud(n, speed, &mut rng);
        let mass = 1.0 / n as f64;
        let t_cloud = cloud_tensor(&cloud, mass);
        let s = spectrum(t_cloud);
        let (stresses, energy_cloud) = ([s[0], s[1], s[2]], s[3]);
        let pressure = -(stresses[0] + stresses[1] + stresses[2]) / 3.0;
        assert!(close(t_cloud.trace(), 1.0, 1e-14));
        let expected = energy_cloud * speed * speed / 3.0;
        assert!(close(pressure, expected, 0.05 * expected.abs()));
        // numga checks each stress to 5%, which its NumPy stream meets; the extreme eigenvalue
        // of 2000 directions strays by about 2% per standard deviation, so on another stream
        // the band is 8%, as numga's own null-cloud test allows.
        for st in stresses {
            assert!(
                close(st, -pressure, 0.08 * pressure.abs()),
                "{s:?} {pressure}"
            );
        }

        // A cloud of null rays is exactly traceless.
        let rays = null_cloud(n, &mut rng);
        let t_light = cloud_tensor(&rays, mass);
        let ls = spectrum(t_light);
        assert!(close(t_light.trace(), 0.0, 1e-11));
        assert!(close(-(ls[0] + ls[1] + ls[2]) / 3.0, ls[3] / 3.0, 1e-12));

        // The ideal fluid matches the cloud.
        let t_fluid = fluid(energy_cloud, pressure);
        assert!(close(t_fluid.trace(), energy_cloud - 3.0 * pressure, 1e-14));
        assert!(maps_close(t_fluid, t_cloud, 0.02));
        let t_radiation = fluid(rho, rho / 3.0);
        assert!(close(t_radiation.trace(), 0.0, 1e-13));

        // The Lorentz force density on a current along u.
        let rho_q = 0.5;
        let force = u.gp(rho_q).commutator(f);
        assert!(close(
            (force | x().inverse()).s(),
            rho_q * gamma * (ex - beta * by),
            1e-14
        ));

        // Covariance: the boosted map is the boosted field's map.
        assert!(maps_close(boosted(b, t_em), stress_energy(b >> f), 1e-14));
        assert!(maps_close(boosted(b, t_dust), t_moving, 1e-14));
    }

    #[test]
    fn maxwell_stress_tracelessness_and_symmetry() {
        let (ex, by) = (2.5, 2.5);
        let t_em = stress_energy(field(ex, 0.0, 0.0, 0.0, by, 0.0));
        assert!(close(t_em.trace(), 0.0, 1e-14));
        let mut rng = Rng(123);
        let (a, b) = (random_vector(&mut rng), random_vector(&mut rng));
        assert!(close((a | t_em.of(b)).s(), (b | t_em.of(a)).s(), 1e-14));
    }

    #[test]
    fn dust_extensor_invariants_and_ram_pressure() {
        let rho = 5.0;
        let t_rest = dyad(t()).gp(rho);
        for n in [x(), y(), z()] {
            assert!(close(normal_stress(t_rest, n), 0.0, 1e-14));
        }
        assert!(close(t_rest.trace(), rho, 1e-14));
        let zeta = 0.6f64.atanh();
        let u = boost(zeta, z()) >> t();
        let (gamma, beta) = (zeta.cosh(), zeta.tanh());
        let t_moving = dyad(u).gp(rho);
        assert!(close(t_moving.trace(), rho, 1e-14));
        assert!(close(
            normal_stress(t_moving, z()),
            -rho * gamma * gamma * beta * beta,
            1e-14
        ));
    }

    #[test]
    fn particle_cloud_is_an_ideal_fluid() {
        let mut rng = Rng(7);
        let speed = 0.7;
        let n = 4000;
        let u = isotropic_cloud(n, speed, &mut rng);
        let t_map = cloud_tensor(&u, 1.0 / n as f64);
        let s = spectrum(t_map);
        let pressure = -(s[0] + s[1] + s[2]) / 3.0;
        assert!(close(t_map.trace(), 1.0, 1e-14));
        let expected = s[3] * speed * speed / 3.0;
        assert!(close(pressure, expected, 0.05 * expected));
        for st in &s[..3] {
            assert!(close(*st, -pressure, 0.05 * pressure));
        }
    }

    #[test]
    fn pure_wave_is_a_null_dyad_and_the_commutator_form_holds() {
        let e = 3.0;
        let t_map = stress_energy(field(e, 0.0, 0.0, 0.0, e, 0.0));
        assert!(maps_close(t_map, dyad(t() + z()).gp(e * e), 1e-14));
        let mut rng = Rng(11);
        let c: Vec<f64> = (0..6).map(|_| rng.normal()).collect();
        let f = field(c[0], c[1], c[2], c[3], c[4], c[5]);
        assert!(maps_close(stress_energy(f), stress_from_force(f), 1e-13));
        let iterated = f.commutator(f.commutator(open()));
        assert!(close(iterated.trace(), 2.0 * lagrangian(f), 1e-13));
    }

    #[test]
    fn null_cloud_is_exactly_traceless_radiation() {
        let mut rng = Rng(12);
        let n = 3000;
        let rays = null_cloud(n, &mut rng);
        let t_map = cloud_tensor(&rays, 2.0 / n as f64);
        let s = spectrum(t_map);
        assert!(close(t_map.trace(), 0.0, 1e-11));
        assert!(close(-(s[0] + s[1] + s[2]) / 3.0, s[3] / 3.0, 1e-12));
        for st in &s[..3] {
            assert!(close(*st, -2.0 / 3.0, 0.08 * 2.0 / 3.0), "{s:?}");
        }
    }

    #[test]
    fn fluid_equation_of_state_interpolation() {
        let rho = 6.0;
        for p in [0.0, 1.5, rho / 3.0] {
            assert!(close(fluid(rho, p).trace(), rho - 3.0 * p, 1e-14));
        }
    }

    #[test]
    fn lorentz_4force_density() {
        let (ex, by) = (3.0, 2.0);
        let f = field(ex, 0.0, 0.0, 0.0, by, 0.0);
        let zeta = 0.5f64.atanh();
        let u = boost(zeta, z()) >> t();
        let (gamma, beta) = (zeta.cosh(), zeta.tanh());
        let force = u.gp(0.8).commutator(f);
        assert!(close(
            (force | x().inverse()).s(),
            0.8 * gamma * (ex - beta * by),
            1e-14
        ));
    }

    #[test]
    fn lorentz_boost_covariance() {
        let rho = 4.0;
        let t_dust = dyad(t()).gp(rho);
        // numga's `(zt * 0.4).exp()`: rapidity 0.8 along z.
        let b = boost(0.8, z());
        let u = b >> t();
        assert!(maps_close(boosted(b, t_dust), dyad(u).gp(rho), 1e-14));
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
