//! numga's `quadrics/cga_spherical_quadrics`: exotic quadrics on the sphere S² in the conformal
//! algebra `Cl(3,1)`, declared with `gax::algebra!`. A point of S² is a null vector, its unit
//! coordinates on x, y and z and weight one on w. A quadric is a map from vectors to planes
//! (trivectors), and a point is inside where `Q(p) ∨ p < 0`. Sums of plane dyads give conical
//! donuts, Cassini peanuts and islands, lemniscates, crescents, Dupin cyclides, pinched horns,
//! spindles, teardrops, clovers, hourglasses and parabolic bows. Two off-centre circles on S² meet
//! in the 2-blade `c1 ∧ c2`, and its exponential is a conformal rotor that carries every quadric
//! around a vortex. The animation shows the trio of numga's GIF, large, and every single shape
//! carried by the same flow.

use gax_numga_examples::{Align, Anim, Canvas, Rgb, backdrop, canvas, caption, palette, run};

gax::algebra! {
    algebra cl31 "The conformal algebra of the sphere S², R(3,1,0): x, y, z square to 1, w to -1.";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = -1;
    kind Scalar = [1];
    versor Vector = [e1, e2, e3, e4];
    kind Bivector = [e23, e31, e12, e14, e24, e34];
    versor Trivector = [e234, e314, e124, e321];
    kind Pseudoscalar = [e1234];
    versor Rotor = [1, e23, e31, e12, e14, e24, e34, e1234];
}

mod cga {
    use super::cl31::{Bivector, Trivector, Vector};

    /// A point of S², or any vector.
    pub type V = Vector<(), f64>;
    /// A plane: a trivector, the dual of a vector.
    pub type Plane = Trivector<(), f64>;
    /// A 2-blade, the generator of a flow.
    pub type B = Bivector<(), f64>;
    /// A conformal rotor.
    pub type R = gax::Unit<super::cl31::Rotor<(), f64>>;
    /// A spherical quadric: a map from vectors to planes.
    pub type Quadric = Trivector<(Vector,), f64>;

    /// The basis vectors x, y, z and w.
    pub fn x() -> V {
        Vector::new(1.0, 0.0, 0.0, 0.0)
    }
    pub fn y() -> V {
        Vector::new(0.0, 1.0, 0.0, 0.0)
    }
    pub fn z() -> V {
        Vector::new(0.0, 0.0, 1.0, 0.0)
    }
    pub fn w() -> V {
        Vector::new(0.0, 0.0, 0.0, 1.0)
    }

    /// The pseudoscalar `xyzw`.
    pub fn pseudoscalar() -> super::cl31::Pseudoscalar<(), f64> {
        super::cl31::Pseudoscalar::new(1.0)
    }

    /// The dual basis planes: the basis vectors times the pseudoscalar (`-w I` for w).
    pub fn px() -> Plane {
        (x() * pseudoscalar()).cast::<Trivector>()
    }
    pub fn py() -> Plane {
        (y() * pseudoscalar()).cast::<Trivector>()
    }
    pub fn pz() -> Plane {
        (z() * pseudoscalar()).cast::<Trivector>()
    }
    pub fn pw() -> Plane {
        (-(w() * pseudoscalar())).cast::<Trivector>()
    }

    /// Rotation generators in numga's blade names: `yz`, `zx`, `xy`, and the boost `xw`.
    pub fn yz(a: f64) -> B {
        Bivector::new(a, 0.0, 0.0, 0.0, 0.0, 0.0)
    }
    pub fn zx(a: f64) -> B {
        Bivector::new(0.0, a, 0.0, 0.0, 0.0, 0.0)
    }
    pub fn xy(a: f64) -> B {
        Bivector::new(0.0, 0.0, a, 0.0, 0.0, 0.0)
    }
    pub fn xw(a: f64) -> B {
        Bivector::new(0.0, 0.0, 0.0, a, 0.0, 0.0)
    }

    /// The null vector of the point of S² at unit coordinates, with weight one on w.
    pub fn point([px, py, pz]: [f64; 3]) -> V {
        Vector::new(px, py, pz, 1.0)
    }

    /// The dyad `a (b ∨ ·)`: a map from vectors to planes.
    pub fn dyad(a: Plane, b: Plane) -> Quadric {
        (b & Vector::slot()) * a
    }

    /// The symmetric dyad `(a (b ∨ ·) + b (a ∨ ·)) / 2`.
    pub fn sym(a: Plane, b: Plane) -> Quadric {
        (dyad(a, b) + dyad(b, a)).gp(0.5)
    }

    /// A quadric carried by a versor: the point pulled back, the plane pushed forward.
    pub fn moved(q: Quadric, versor: R) -> Quadric {
        versor >> q.of(versor << Vector::slot())
    }

    /// Whether a point is inside: `Q(p) ∨ p < 0`, and the value itself.
    pub fn potential(q: Quadric, p: V) -> f64 {
        (q.of(p) & p).s()
    }

    // --- shapes ----------------------------------------------------------------------------

    /// A donut (torus) about the pole: the band between the circles at `r_core ± r_tube`.
    pub fn spherical_donut(r_core: f64, r_tube: f64) -> Quadric {
        let c_out = (r_core + r_tube).cos();
        let c_in = (r_core - r_tube).cos();
        let z0 = (c_out + c_in) / 2.0;
        let dz = (c_in - c_out) / 2.0;
        dyad(pz(), pz()) - (dyad(pz(), pw()) + dyad(pw(), pz())).gp(z0)
            + dyad(pw(), pw()).gp(z0 * z0 - dz * dz)
    }

    /// A donut (limaçon) whose central hole meets in a sharp conical apex.
    pub fn conical_donut(a: f64, b: f64) -> Quadric {
        let line = pw() - pz() - px().gp(a);
        dyad(line, line) - (dyad(px(), px()) + dyad(py(), py())).gp(b * b)
    }

    /// The lemniscate of Bernoulli: `(w - z)² - 2 a² (x² - y²) < 0`.
    pub fn bernoulli_lemniscate(scale_a: f64) -> Quadric {
        let p_wz = pw() - pz();
        dyad(p_wz, p_wz) - (dyad(px(), px()) - dyad(py(), py())).gp(2.0 * scale_a * scale_a)
    }

    /// The pinched horn cyclide: the donut whose inner radius shrinks to a cusp at the pole.
    pub fn pinched_horn(r_outer: f64) -> Quadric {
        spherical_donut(r_outer / 2.0, r_outer / 2.0)
    }

    /// An eccentric Dupin cyclide with unequal tube width: a donut moved by a boost.
    pub fn eccentric_cyclide(r_core: f64, r_tube: f64, boost_beta: f64) -> Quadric {
        moved(spherical_donut(r_core, r_tube), xw(boost_beta / 2.0).exp())
    }

    /// A spherical Cassini oval: inside where `(p1 ∨ p)(p2 ∨ p) < c (pw ∨ p)²`, with `p1` and
    /// `p2` the planes tangent to S² at the two foci (the plane z turned toward x by the first
    /// angle and away from it by the second). Twin islands for small `c`, a figure eight near the
    /// pinch, a peanut beyond, a teardrop for unequal angles.
    pub fn spherical_cassini(alpha1: f64, alpha2: f64, c_threshold: f64) -> Quadric {
        let p1 = pw() - (zx(-alpha1 / 2.0).exp() >> pz());
        let p2 = pw() - (zx(alpha2 / 2.0).exp() >> pz());
        sym(p1, p2) - dyad(pw(), pw()).gp(c_threshold)
    }

    /// A crescent bounded by two eccentric circles.
    pub fn spherical_crescent(r_outer: f64, r_inner: f64, offset: f64) -> Quadric {
        let c_outer = pz() - pw().gp(r_outer.cos());
        let c_inner = (zx(-offset / 2.0).exp() >> pz()) - pw().gp(r_inner.cos());
        sym(c_outer, c_inner)
    }

    /// A spindle with two opposite conical poles.
    pub fn spherical_spindle(weight_z: f64, weight_xy: f64, bias: f64) -> Quadric {
        dyad(pz(), pz()).gp(weight_z)
            - dyad(px(), px()).gp(weight_xy)
            - dyad(py(), py()).gp(weight_xy)
            - dyad(pw(), pw()).gp(bias)
    }

    /// A three-lobed clover: three circle dyads 120° apart (the plane z tilted toward x, then
    /// turned about z).
    pub fn spherical_clover(tilt_angle: f64, radius: f64, bias: f64) -> Quadric {
        let mut q = -dyad(pw(), pw()).gp(bias);
        for degrees in [0.0f64, 120.0, 240.0] {
            let turn = xy(-degrees.to_radians() / 2.0).exp() * zx(-tilt_angle / 2.0).exp();
            let circle = (turn >> pz()) - pw().gp(radius.cos());
            q += dyad(circle, circle);
        }
        q
    }

    /// An hourglass: two bulbs joined by a narrow waist (the plane z turned toward y and away).
    pub fn spherical_hourglass(theta: f64, c_waist: f64) -> Quadric {
        let p1 = pw() - (yz(theta / 2.0).exp() >> pz());
        let p2 = pw() - (yz(-theta / 2.0).exp() >> pz());
        sym(p1, p2) - dyad(pw(), pw()).gp(c_waist)
    }

    /// A parabolic bow.
    pub fn spherical_parabola(weight_y: f64, linear_x: f64, bias: f64) -> Quadric {
        dyad(py(), py()).gp(weight_y)
            - (dyad(px(), pw()) + dyad(pw(), px())).gp(0.5 * linear_x)
            - dyad(pw(), pw()).gp(bias)
    }

    /// Every single shape, tilted into view, with its name and colour.
    pub fn shapes() -> Vec<(&'static str, Quadric, u32)> {
        let deg = f64::to_radians;
        let tilt = |a: f64| yz(a).exp();
        vec![
            (
                "conical donut",
                moved(conical_donut(0.55, 0.22), zx(0.20).exp()),
                0x38bdf8,
            ),
            (
                "peanut",
                moved(spherical_cassini(deg(25.0), deg(25.0), 0.012), tilt(0.30)),
                0xf59e0b,
            ),
            (
                "islands",
                moved(spherical_cassini(deg(25.0), deg(25.0), 0.006), tilt(0.30)),
                0x10b981,
            ),
            ("lemniscate", bernoulli_lemniscate(0.38), 0xa855f7),
            (
                "crescent",
                moved(
                    spherical_crescent(deg(45.0), deg(24.0), deg(16.0)),
                    tilt(0.25),
                ),
                0xfb7185,
            ),
            (
                "cyclide",
                moved(eccentric_cyclide(deg(34.0), deg(13.0), 0.65), tilt(0.30)),
                0xfbbf24,
            ),
            (
                "pinched",
                moved(pinched_horn(deg(65.0)), tilt(0.35)),
                0xf43f5e,
            ),
            (
                "spindle",
                moved(spherical_spindle(1.8, 0.8, 0.35), tilt(0.30)),
                0x06b6d4,
            ),
            (
                "teardrop",
                moved(spherical_cassini(deg(10.0), deg(35.0), 0.010), tilt(0.30)),
                0xec4899,
            ),
            (
                "clover",
                moved(spherical_clover(deg(32.0), deg(28.0), 0.35), tilt(0.20)),
                0x4ade80,
            ),
            ("hourglass", spherical_hourglass(deg(24.0), 0.010), 0xf59e0b),
            (
                "parabola",
                moved(spherical_parabola(1.0, 0.5, 0.20), tilt(0.25)),
                0xf97316,
            ),
        ]
    }

    /// The trio: the conical donut, the lemniscate and the hourglass.
    pub fn trio() -> Vec<(&'static str, Quadric, u32)> {
        shapes()
            .into_iter()
            .filter(|(name, _, _)| ["conical donut", "lemniscate", "hourglass"].contains(name))
            .collect()
    }

    // --- the flow --------------------------------------------------------------------------

    /// The 2-blade where two off-centre circles meet, normalized to square to -1: each circle
    /// is the vector of its centre with weight the cosine of its radius.
    pub fn circle_intersection_vortex(center1: V, radius1: f64, center2: V, radius2: f64) -> B {
        let c1 = center1 + w().gp(radius1.cos());
        let c2 = center2 + w().gp(radius2.cos());
        (c1 ^ c2).normalized().into_inner()
    }

    /// numga's vortex: circles of radii 48° and 52° about the pole tilted toward x and toward y.
    pub fn vortex() -> B {
        circle_intersection_vortex(
            zx(-0.35 / 2.0).exp() >> z(),
            48f64.to_radians(),
            yz(0.40 / 2.0).exp() >> z(),
            52f64.to_radians(),
        )
    }

    /// A quadric carried by `exp(generator phase / 2)`.
    pub fn flow(q: Quadric, generator: B, phase: f64) -> Quadric {
        moved(q, generator.gp(phase / 2.0).exp())
    }
}

use cga::*;

const SECONDS: f32 = 6.0;

/// The front hemisphere seen along z in the disc of `radius` pixels at `centre`: each pixel the
/// colour of the last quadric that holds it, the disc and its rim otherwise.
fn hemisphere(c: &mut Canvas, centre: [f32; 2], radius: f32, quadrics: &[(Quadric, Rgb)]) {
    let (disk, rim) = (
        canvas::srgb(0.067, 0.094, 0.153),
        canvas::srgb(0.2, 0.255, 0.333),
    );
    c.clip([
        centre[0] - radius - 1.0,
        centre[1] - radius - 1.0,
        centre[0] + radius + 1.0,
        centre[1] + radius + 1.0,
    ]);
    c.shade(2, |px, py| {
        let (u, v) = ((px - centre[0]) / radius, (centre[1] - py) / radius);
        let r2 = u * u + v * v;
        if r2 > 1.0 {
            return None;
        }
        let p = point([
            f64::from(u),
            f64::from(v),
            f64::from((1.0 - r2).max(0.0).sqrt()),
        ]);
        let hit = quadrics.iter().rev().find(|(q, _)| potential(*q, p) < 0.0);
        Some(match hit {
            Some((_, colour)) => *colour,
            None if r2 > 0.985 => rim,
            None => disk,
        })
    });
    c.unclip();
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.width as f32, c.height as f32);
    let generator = vortex();
    let phase = f64::from(t / SECONDS) * core::f64::consts::TAU;
    let carried = |q: Quadric| flow(q, generator, phase);
    // The trio, large, on the left.
    let big = (h * 0.42).min(w * 0.24);
    let trio: Vec<(Quadric, Rgb)> = trio()
        .iter()
        .map(|(_, q, col)| (carried(*q), canvas::hex(*col)))
        .collect();
    hemisphere(c, [w * 0.26, h * 0.54], big, &trio);
    // Every shape on the right, in a grid of four by three.
    let (x0, y0) = (w * 0.52, h * 0.13);
    let (cw, ch) = ((w - x0 - 8.0) / 4.0, (h - y0 - 4.0) / 3.0);
    let r = (cw.min(ch) * 0.5 - 12.0).max(4.0);
    let label = (h / 50.0).clamp(7.0, 11.0);
    for (k, (name, q, col)) in shapes().into_iter().enumerate() {
        let centre = [
            x0 + cw * ((k % 4) as f32 + 0.5),
            y0 + ch * ((k / 4) as f32 + 0.5) - label * 0.5,
        ];
        hemisphere(c, centre, r, &[(carried(q), canvas::hex(col))]);
        c.text(
            name,
            centre[0],
            centre[1] + r + label * 1.3,
            label,
            palette::grid(),
            Align::Center,
        );
    }
    caption(
        c,
        "SPHERICAL QUADRICS IN A CONFORMAL VORTEX",
        "CL(3,1): INSIDE WHERE Q(P) V P < 0; CARRIED BY EXP(C1 ^ C2 T/2)",
    );
}

fn main() {
    run(
        Anim::new("cga spherical quadrics", SECONDS)
            .size(800, 450)
            .scale(1),
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::cga::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// Pixels of the hemisphere are null vectors of weight one.
    #[test]
    fn null_cone_pixels() {
        for (u, v) in [(0.0, 0.0), (0.3, -0.5), (-0.9, 0.1), (0.6, 0.79)] {
            let p = point([u, v, (1.0f64 - u * u - v * v).sqrt()]);
            assert!((p | p).s().abs() < 1e-12);
            assert!(close((p | w()).s(), -1.0, 1e-12));
        }
    }

    /// The dual planes are the trivectors numga builds, and pairing them with a point reads off
    /// its coordinates.
    #[test]
    fn planes_pair_with_points() {
        let p = point([0.36, 0.48, 0.8]);
        let read = [px(), py(), pz(), pw()].map(|q| (q & p).s());
        let want = [0.36, 0.48, 0.8, 1.0];
        for (a, b) in read.iter().zip(want) {
            assert!(close(a.abs(), b, 1e-14), "{read:?}");
        }
    }

    /// The conical donut's apex touches the north pole.
    #[test]
    fn conical_donut_topology() {
        let apex = point([0.0, 0.0, 1.0]);
        assert!(potential(conical_donut(0.55, 0.22), apex).abs() < 1e-12);
    }

    /// The donut's potential is negative inside the ring, positive in the hole and outside.
    #[test]
    fn spherical_donut_topology() {
        let (core, tube) = (40f64.to_radians(), 10f64.to_radians());
        let donut = spherical_donut(core, tube);
        assert!(potential(donut, point([0.0, 0.0, 1.0])) > 0.0);
        assert!(potential(donut, point([core.sin(), 0.0, core.cos()])) < 0.0);
        assert!(potential(donut, point([1.0, 0.0, 0.0])) > 0.0);
    }

    /// The pinched horn cyclide touches the pole with zero potential.
    #[test]
    fn pinched_horn_cusp() {
        let horn = pinched_horn(60f64.to_radians());
        assert!(potential(horn, point([0.0, 0.0, 1.0])).abs() < 1e-12);
    }

    /// The intersection 2-blade squares to -1 and generates a conformal rotor: it keeps points
    /// null.
    #[test]
    fn circle_intersection_vortex_is_a_conformal_rotation() {
        let g = vortex();
        assert!(close((g * g).s(), -1.0, 1e-12));
        let rotor = g.gp(0.4).exp();
        let p = point([0.5, 0.5, 0.5f64.sqrt()]);
        let q = rotor >> p;
        assert!((q | q).s().abs() < 1e-12);
    }

    /// The scenario's checks: the flow keeps points of S² null, and a full turn brings every
    /// quadric back.
    #[test]
    fn the_flow_is_conformal_and_periodic() {
        let g = vortex();
        let probes = [[0.6, 0.0, 0.8], [0.0, -0.28, 0.96], [0.36, 0.48, 0.8]].map(point);
        for p in probes {
            let q = g.gp(0.4).exp() >> p;
            assert!((q | q).s().abs() < 1e-12);
        }
        for (_, q, _) in shapes() {
            let back = flow(q, g, core::f64::consts::TAU);
            for p in probes {
                assert!(close(potential(back, p), potential(q, p), 1e-6));
            }
        }
    }

    /// Every shape, and every shape carried 1.4 rad along the vortex, has numga's potential at a
    /// probe point (numga's exponential is good to about 1e-11).
    #[test]
    fn potentials_match_numga() {
        let numga = [
            (-0.028706646998050804, 0.10221357098921147),
            (-0.010824344608939964, 0.019529911452355525),
            (-0.004824344609015898, 0.02135153044134526),
            (0.06911103999999998, 0.21453368524286745),
            (0.01840346544239634, -0.0008249618318519423),
            (0.03128327680923761, 0.019369148255120497),
            (-0.039330947733828836, -0.020197366704325034),
            (1.1050160805205602, 0.13871022331579388),
            (-0.0009776636797949028, 0.024167479028599354),
            (-0.25849941891701217, 0.06542332885103488),
            (0.024332907660516143, 0.10698310328849381),
            (-0.3785787704078519, 0.044630806813971975),
        ];
        let p = point([0.36, 0.48, 0.8]);
        for ((name, q, _), (still, carried)) in shapes().into_iter().zip(numga) {
            assert!(close(potential(q, p), still, 1e-10), "{name}");
            assert!(
                close(potential(flow(q, vortex(), 1.4), p), carried, 1e-10),
                "{name}"
            );
        }
    }

    /// Every one of the thirteen shapes (twelve and the trio) covers part of the hemisphere and
    /// leaves part of it free, so its frame is not blank.
    #[test]
    fn every_shape_shows() {
        let mut table: Vec<Vec<Quadric>> = shapes().iter().map(|(_, q, _)| vec![*q]).collect();
        table.push(trio().iter().map(|(_, q, _)| *q).collect());
        assert_eq!(table.len(), 13);
        for quadrics in table {
            let (mut inside, mut outside) = (0, 0);
            for i in 0..60 {
                for j in 0..60 {
                    let (u, v) = (-1.0 + 2.0 * i as f64 / 59.0, 1.0 - 2.0 * j as f64 / 59.0);
                    if u * u + v * v > 1.0 {
                        continue;
                    }
                    let p = point([u, v, (1.0 - u * u - v * v).sqrt()]);
                    if quadrics.iter().any(|q| potential(*q, p) < 0.0) {
                        inside += 1;
                    } else {
                        outside += 1;
                    }
                }
            }
            assert!(inside > 0 && outside > 0);
        }
    }

    #[test]
    fn a_frame_draws() {
        let mut draw = super::draw;
        for t in [0.0, 1.5] {
            let c = gax_numga_examples::app::frame(
                &gax_numga_examples::Anim::new("t", 1.0).size(320, 180),
                t,
                &mut draw,
            );
            assert!(c.mean()[0] > 0.0);
        }
    }
}
