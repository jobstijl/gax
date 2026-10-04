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

use gax::pga2d::Point;
use gax_light::fade;
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Light, Marker, ORIGIN3, Point2, Point3, Rect, Scene3,
    backdrop, caption, palette, reach3, run,
};

mod maxwell {
    // numga's API in full: the tests use parts the animation does not.
    #![cfg_attr(not(test), allow(dead_code))]

    use gax::sta::{Bivector, Even, Scalar, Vector};
    use gax_numga_examples::rng::{Draw, Rng};

    pub type V = Vector<(), f64>;
    pub type F = Bivector<(), f64>;
    /// A vector of the rest observer's space, for drawing.
    pub type Space = gax::vga3d::Vector<(), f64>;
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

    /// The spatial part of a vector as the rest observer `t` reads it, a vector of space.
    pub fn spatial(v: V) -> Space {
        Space::new(v.e1(), v.e2(), v.e3())
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

    /// The Poynting vector: the spatial part of the flux the rest observer sees.
    pub fn poynting(t_map: StressEnergy) -> Space {
        spatial(t_map.of(t()))
    }

    /// The real spectrum of a stress-energy map, ascending. numga calls a general eigensolver
    /// (gax's `eigvals`); the symmetric pencil keeps the values exactly real. `T` is
    /// self-adjoint for the Minkowski metric `g`, so `T v = λ v` is the symmetric pencil
    /// `S(v, ·) = λ g(v, ·)` with `S(a, b) = a · T(b)`. The metric is indefinite, but a
    /// cloud's `S(v, v) = Σ m (u · v)²` is positive definite, so the pencil is solved the
    /// other way round, `g(v, ·) = λ⁻¹ S(v, ·)`.
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

    /// `n` unit spatial directions, uniform on the sphere.
    pub fn sphere_directions(n: usize, rng: &mut Rng) -> Vec<V> {
        (0..n)
            .map(|_| {
                let [a, b, c] = rng.direction();
                Vector::new(0.0, a, b, c)
            })
            .collect()
    }

    /// The four-velocities at one speed along each direction: the rest observer boosted by
    /// the rapidity `atanh(speed)`.
    pub fn moving(directions: &[V], speed: f64) -> Vec<V> {
        let zeta = speed.atanh();
        directions.iter().map(|d| boost(zeta, *d) >> t()).collect()
    }

    /// The null rays `t + direction`.
    pub fn rays(directions: &[V]) -> Vec<V> {
        directions.iter().map(|d| t() + *d).collect()
    }

    /// `n` four-velocities at one speed, directions uniform on the sphere.
    pub fn isotropic_cloud(n: usize, speed: f64, rng: &mut Rng) -> Vec<V> {
        moving(&sphere_directions(n, rng), speed)
    }

    /// `n` null rays, directions uniform on the sphere.
    pub fn null_cloud(n: usize, rng: &mut Rng) -> Vec<V> {
        rays(&sphere_directions(n, rng))
    }

    /// A cloud's pressure and energy density from its spectrum: the energy is the timelike
    /// eigenvalue, the pressure minus the mean of the three stresses.
    pub fn pressure_energy(t_map: StressEnergy) -> (f64, f64) {
        let s = spectrum(t_map);
        (-(s[0] + s[1] + s[2]) / 3.0, s[3])
    }

    /// The general field of numga's commutator check, used by the animation.
    pub fn general_field() -> F {
        field(0.7, -0.4, 1.1, 0.3, -0.8, 0.5)
    }

    /// The electric and magnetic vectors a rest observer reads off a field: `t · F` and
    /// `t · F*` (the complement turns magnetic planes into electric ones).
    pub fn electric_magnetic(f: F) -> (Space, Space) {
        (spatial(t() | f), spatial(t() | f.dual()))
    }
}

use maxwell::*;

const SECONDS: f32 = 8.0;
/// The largest rapidity of the boost.
const SWING: f64 = 0.9;
/// Rapidities sampled across the swing.
const SAMPLES: usize = 96;

/// The rapidity of the boost at animation phase `p` (radians).
fn rapidity(p: f64) -> f64 {
    SWING * p.sin()
}

/// The `i`-th of the sampled rapidities, from `-SWING` to `SWING`.
fn sampled(i: usize) -> f64 {
    SWING * (-1.0 + 2.0 * i as f64 / SAMPLES as f64)
}

/// The cloud's speed at animation phase `p`: from dust nearly to light and back.
fn cloud_speed(p: f64) -> f64 {
    0.02 + 0.94 * 0.5 * (1.0 - p.cos())
}

/// The fixed cloud directions; the animation boosts the rest observer along them.
fn directions() -> &'static [V] {
    static DIRS: std::sync::OnceLock<Vec<V>> = std::sync::OnceLock::new();
    DIRS.get_or_init(|| sphere_directions(2000, &mut gax_numga_examples::rng::rng(0x9e37)))
}

/// The null cloud's pressure over its energy, from its spectrum (exactly a third).
fn null_ratio() -> f64 {
    static R: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *R.get_or_init(|| {
        let rays = rays(directions());
        let (pressure, energy) = pressure_energy(cloud_tensor(&rays, 1.0 / rays.len() as f64));
        pressure / energy
    })
}

/// Grey axes with their names at the ends.
fn axes3(c: &mut Canvas, cam: &Camera, s: &mut Scene3, len: f32) {
    let axes = [
        Point3::direction(1.0, 0.0, 0.0),
        Point3::direction(0.0, 1.0, 0.0),
        Point3::direction(0.0, 0.0, 1.0),
    ];
    for (d, name) in axes.into_iter().zip(["X", "Y", "Z"]) {
        s.seg(ORIGIN3, ORIGIN3 + d.gp(len), 1.0, palette::grid());
        if let Some(q) = cam.px(ORIGIN3 + d.gp(len * 1.1)) {
            let below = q + Point2::direction(0.0, 4.0);
            c.text(name, below, 10.0, palette::grid(), Align::Center);
        }
    }
}

/// Text in a colour at a pixel, left aligned.
fn note(c: &mut Canvas, s: &str, at: Point2, col: Light) {
    c.text(s, at, 11.0, col, Align::Left);
}

/// A panel's parts: the 3D view in its upper two thirds, the view's clipping rectangle (a
/// margin above the plot's title), and the plot rectangle below.
fn parts(rect: Rect) -> (Rect, Rect, Rect) {
    let cut = rect.height() * 0.68;
    let view = Rect {
        lo: rect.lo,
        hi: rect.top_right() + Point2::direction(0.0, cut),
    };
    let plot = Rect {
        lo: view.bottom_left(),
        hi: rect.hi,
    };
    (
        view,
        view.inset(0.0, 0.0, 0.0, 12.0),
        plot.inset(52.0, 0.0, 16.0, 34.0),
    )
}

/// Where a panel's notes start, and the step down to the next line.
fn notes(rect: Rect) -> (Point2, Point2) {
    (
        rect.lo + Point2::direction(14.0, 14.0),
        Point2::direction(0.0, 16.0),
    )
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
    [
        energy(tm, t()),
        poynting(tm).norm(),
        lagrangian(g),
        tm.trace(),
    ]
}

fn draw_field(c: &mut Canvas, rect: Rect, phase: f64, spin: f32) {
    let zeta = rapidity(phase);
    let (g, tm) = boosted_field(zeta);
    let (e, b) = electric_magnetic(g);
    let (view, clip, plot) = parts(rect);
    let cam = Camera::parallel(view, 48.0, 0.6 + spin, 0.35);
    c.clip(clip);
    let mut sc = Scene3::new(cam);
    axes3(c, &cam, &mut sc, 2.2);
    // The tips' paths over the whole swing.
    let tips: Vec<[Point3; 3]> = (0..=SAMPLES)
        .map(|i| {
            let (g, tm) = boosted_field(sampled(i));
            let (e, b) = electric_magnetic(g);
            [e, b, poynting(tm)].map(reach3)
        })
        .collect();
    let cols = [palette::orange(), palette::sky(), palette::yellow()];
    for (k, col) in cols.iter().enumerate() {
        let path: Vec<Point3> = tips.iter().map(|p| p[k]).collect();
        sc.polyline(&path, 1.0, fade(*col, 0.45));
    }
    for (v, col) in [(e, cols[0]), (b, cols[1]), (poynting(tm), cols[2])] {
        sc.arrow(ORIGIN3, v, 2.5, 10.0, col);
    }
    sc.draw(c);
    c.unclip();
    let (at, line) = notes(rect);
    let boost = format!("BOOST ALONG X, RAPIDITY {zeta:+.2}");
    note(c, &boost, at, palette::ink());
    let across = Point2::direction(16.0, 0.0);
    note(c, "E", at + line, palette::orange());
    note(c, "B", at + line + across, palette::sky());
    let flux = at + line + across.gp(2.0);
    note(c, "POYNTING FLUX", flux, palette::yellow());

    let ax = Axes::new(plot, [-SWING as f32, SWING as f32], [-1.0, 6.0]);
    ax.frame(c, "", "RAPIDITY", "");
    let readouts: Vec<(f64, [f64; 4])> = (0..=SAMPLES)
        .map(|i| (sampled(i), readout(sampled(i))))
        .collect();
    let cols = [
        palette::orange(),
        palette::yellow(),
        palette::green(),
        palette::purple(),
    ];
    for (k, col) in cols.iter().enumerate() {
        let curve: Vec<Point<(), f64>> =
            readouts.iter().map(|(z, r)| Point::xy(*z, r[k])).collect();
        ax.polyline(c, &curve, 1.8, *col);
    }
    let now = readout(zeta);
    ax.scatter(
        c,
        &[Point::xy(zeta, now[0]), Point::xy(zeta, now[1])],
        Marker::Dot,
        8.0,
        palette::ink(),
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

fn draw_cloud(c: &mut Canvas, rect: Rect, phase: f64, spin: f32) {
    let speed = cloud_speed(phase);
    let dirs = directions();
    let us = moving(dirs, speed);
    let tm = cloud_tensor(&us, 1.0 / us.len() as f64);
    let s = spectrum(tm);
    let (pressure, energy) = pressure_energy(tm);
    let (view, clip, plot) = parts(rect);
    let cam = Camera::parallel(view, 80.0, -0.4 + spin, 0.3);
    c.clip(clip);
    let mut sc = Scene3::new(cam);
    sc.sphere_wire(ORIGIN3, 1.0, 12, fade(palette::grid(), 0.5));
    // Five hundred dots add their light: each faint, so that the cloud glows rather than burns.
    let dot = fade(palette::sky(), 0.2);
    for d in dirs.iter().step_by(4) {
        sc.dot(reach3(spatial(*d) * speed), Marker::Dot, 3.0, dot);
    }
    sc.draw(c);
    c.unclip();
    let (at, line) = notes(rect);
    let speeds = format!("SPEED {speed:.2}   TRACE {:.3}", tm.trace());
    note(c, &speeds, at, palette::ink());
    let balance = format!("ENERGY {energy:.2}   PRESSURE {pressure:.3}");
    note(c, &balance, at + line, palette::ink());
    note(c, "VELOCITIES IN SPACE", at + line.gp(2.0), palette::sky());

    let ax = Axes::new(plot, [0.0, 1.0], [0.0, 0.4]);
    ax.frame(c, "", "SPEED", "");
    let theory: Vec<Point<(), f32>> = (0..=60)
        .map(|i| {
            let v = i as f32 / 60.0;
            Point::xy(v, v * v / 3.0)
        })
        .collect();
    ax.polyline(c, &theory, 1.5, palette::green());
    ax.dashed(
        c,
        &[Point::xy(0.0, 1.0 / 3.0), Point::xy(1.0, 1.0 / 3.0)],
        1.0,
        8.0,
        palette::grid(),
    );
    // Each stress eigenvalue over the energy: the three nearly coincide (isotropy).
    let marks: Vec<Point<(), f64>> = s[..3]
        .iter()
        .map(|p| Point::xy(speed, -p / energy))
        .collect();
    ax.scatter(c, &marks, Marker::Ring, 9.0, palette::orange());
    let light = Point::xy(1.0, null_ratio());
    ax.scatter(c, &[light], Marker::Star, 12.0, palette::yellow());
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
    let screen = c.rect();
    let phase = f64::from(t / SECONDS) * core::f64::consts::TAU;
    let spin = 0.25 * (core::f32::consts::TAU * t / SECONDS).sin();
    // Two panels side by side below the caption.
    let below = screen.inset(0.0, screen.height() * 0.12, 0.0, 0.0);
    draw_field(c, below.column(0, 2), phase, spin);
    draw_cloud(c, below.column(1, 2), phase, spin);
    caption(
        c,
        "MAXWELL: STRESS-ENERGY MAPS IN SPACETIME",
        "LEFT: T(V) = F V REV(F)/2 OF A BOOSTED FIELD. RIGHT: A CLOUD OF DUST DYADS SPEEDING UP",
    );
}

fn main() {
    run(Anim::new("maxwell", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::maxwell::*;
    use gax::ApproxEq;
    use gax_numga_examples::rng::{Draw, Rng, rng};

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    fn maps_close(a: StressEnergy, b: StressEnergy, tol: f64) -> bool {
        a.max_abs_diff(&b) <= tol
    }

    fn random_vector(rng: &mut Rng) -> V {
        V::new(rng.normal(), rng.normal(), rng.normal(), rng.normal())
    }

    /// A cloud's three stresses agree with its pressure to `band`: isotropy, within the
    /// sampling noise of a finite cloud.
    fn isotropic(t_map: StressEnergy, band: f64) -> bool {
        let (pressure, _) = pressure_energy(t_map);
        spectrum(t_map)[..3]
            .iter()
            .all(|st| close(*st, -pressure, band * pressure.abs()))
    }

    /// numga's `main`: every check of its tutorial.
    #[test]
    fn tutorial_checks() {
        let mut rng = rng(0x2545_f491);
        // A plane wave along +z: electric field along x, magnetic along y.
        let (ex, by) = (2.0, 2.0);
        let f = field(ex, 0.0, 0.0, 0.0, by, 0.0);
        let t_em = stress_energy(f);
        assert!(close(energy(t_em, t()), 0.5 * (ex * ex + by * by), 1e-14));
        let p = poynting(t_em);
        assert!(
            p.max_abs_diff(&Space::new(0.0, 0.0, ex * by)) <= 1e-14,
            "{p:?}"
        );
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
        assert!(close(energy(t_moving, t()), rho * gamma * gamma, 1e-14));
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
        let (pressure, energy_cloud) = pressure_energy(t_cloud);
        assert!(close(t_cloud.trace(), 1.0, 1e-14));
        let expected = energy_cloud * speed * speed / 3.0;
        assert!(close(pressure, expected, 0.05 * expected.abs()));
        // numga checks each stress to 5%, which its NumPy stream meets; the extreme eigenvalue
        // of 2000 directions strays by about 2% per standard deviation, so on another stream
        // the band is 8%, as numga's own null-cloud test allows.
        assert!(isotropic(t_cloud, 0.08), "{:?}", spectrum(t_cloud));

        // A cloud of null rays is exactly traceless.
        let rays = null_cloud(n, &mut rng);
        let t_light = cloud_tensor(&rays, mass);
        let (pressure_light, energy_light) = pressure_energy(t_light);
        assert!(close(t_light.trace(), 0.0, 1e-11));
        assert!(close(pressure_light, energy_light / 3.0, 1e-12));

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

    /// The boost of the rest observer along a direction is the four-velocity `γ (t + β d)`.
    #[test]
    fn a_boosted_observer_moves_at_the_tanh_of_the_rapidity() {
        let mut rng = rng(3);
        for d in sphere_directions(5, &mut rng) {
            for speed in [0.1f64, 0.6, 0.95] {
                let gamma = 1.0 / (1.0 - speed * speed).sqrt();
                let want = (t() + d.gp(speed)).gp(gamma);
                assert!(moving(&[d], speed)[0].max_abs_diff(&want) <= 1e-12);
            }
        }
    }

    #[test]
    fn maxwell_stress_tracelessness_and_symmetry() {
        let (ex, by) = (2.5, 2.5);
        let t_em = stress_energy(field(ex, 0.0, 0.0, 0.0, by, 0.0));
        assert!(close(t_em.trace(), 0.0, 1e-14));
        let mut rng = rng(123);
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
        let mut rng = rng(7);
        let speed = 0.7;
        let n = 4000;
        let u = isotropic_cloud(n, speed, &mut rng);
        let t_map = cloud_tensor(&u, 1.0 / n as f64);
        let (pressure, energy) = pressure_energy(t_map);
        assert!(close(t_map.trace(), 1.0, 1e-14));
        let expected = energy * speed * speed / 3.0;
        assert!(close(pressure, expected, 0.05 * expected));
        assert!(isotropic(t_map, 0.05), "{:?}", spectrum(t_map));
    }

    #[test]
    fn pure_wave_is_a_null_dyad_and_the_commutator_form_holds() {
        let e = 3.0;
        let t_map = stress_energy(field(e, 0.0, 0.0, 0.0, e, 0.0));
        assert!(maps_close(t_map, dyad(t() + z()).gp(e * e), 1e-14));
        let mut rng = rng(11);
        let mut c = || rng.normal();
        let f = field(c(), c(), c(), c(), c(), c());
        assert!(maps_close(stress_energy(f), stress_from_force(f), 1e-13));
        let iterated = f.commutator(f.commutator(open()));
        assert!(close(iterated.trace(), 2.0 * lagrangian(f), 1e-13));
    }

    #[test]
    fn null_cloud_is_exactly_traceless_radiation() {
        let mut rng = rng(12);
        let n = 3000;
        let rays = null_cloud(n, &mut rng);
        let t_map = cloud_tensor(&rays, 2.0 / n as f64);
        let (pressure, energy) = pressure_energy(t_map);
        assert!(close(t_map.trace(), 0.0, 1e-11));
        assert!(close(pressure, energy / 3.0, 1e-12));
        // The rays carry the energy 2; their spectrum's energy differs from it by the
        // cloud's small net momentum, at second order.
        assert!(close(pressure, 2.0 / 3.0, 0.01 * 2.0 / 3.0), "{pressure}");
        assert!(isotropic(t_map, 0.08), "{:?}", spectrum(t_map));
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
