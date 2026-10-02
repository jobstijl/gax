//! numga's `quantum/graphene`: electrons in graphene in the geometric algebra of space (VGA3D).
//! An electron hops between neighbouring carbon atoms, and its state has a part on each of the
//! honeycomb's two sublattices; the hopping mixes them through a vector that depends on the
//! momentum, the pseudospin field: each bond contributes `x` turned in the plane by the phase
//! `k · b` across it, and a difference between the sublattices (as in boron nitride) adds a gap
//! along `z`. The Hamiltonian is a map on even multivectors, `psi ↦ field psi z`; its eigenvalues
//! are plus and minus the field's length, and the pseudospin of a state, `psi z ~psi`, points along
//! the field for the upper band. At the corners of the Brillouin zone the field vanishes and the
//! bands meet in cones.
//!
//! Around a loop in momentum the pseudospin direction traces a curve on the unit sphere; carrying a
//! frame along it by the smallest rotation from each direction to the next brings it back turned
//! by the solid angle the curve encloses. The holonomy is a rotor with scalar part `cos γ`, `γ`
//! the Berry phase: `π` around a gapless cone.
//!
//! The animation turns the two bands with their six cones; beside them, a loop about the valley K
//! grows and shrinks on the map of the field, the pseudospin it meets traces a curve on the sphere
//! (with a gap of 0.5 eV) while a transported frame vector turns along it, and the Berry phase of
//! loops of every radius is plotted for four gaps.

use gax_numga_examples::canvas::{mix, scale};
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Marker, Rgb, Scene3, backdrop, caption, colormap,
    palette, run,
};
use std::sync::OnceLock;

mod graphene {
    use gax::Unit;
    use gax::vga3d::{Bivector, Pseudoscalar, Rotor, Vector};

    pub type V = Vector<(), f64>;
    pub type R = Rotor<(), f64>;
    /// The Hamiltonian, a map on even multivectors.
    pub type Hamiltonian = Rotor<(Rotor,), f64>;

    /// The energy of a hop between neighbours, in eV.
    pub const HOPPING: f64 = 2.8;
    /// The slope of the cones, in eV per inverse carbon-carbon distance.
    pub const VELOCITY: f64 = 1.5 * HOPPING;

    pub fn x() -> V {
        Vector::new(1.0, 0.0, 0.0)
    }
    pub fn z() -> V {
        Vector::new(0.0, 0.0, 1.0)
    }
    fn xy() -> Bivector<(), f64> {
        Bivector::new(0.0, 0.0, 1.0)
    }

    /// The bonds from an atom to its three neighbours, in carbon-carbon distances (0.142 nm).
    pub fn bonds() -> [V; 3] {
        let h = 3f64.sqrt() / 2.0;
        [
            Vector::new(0.5, h, 0.0),
            Vector::new(0.5, -h, 0.0),
            Vector::new(-1.0, 0.0, 0.0),
        ]
    }

    /// The two inequivalent corners of the Brillouin zone, where the cones sit.
    pub fn valleys() -> [V; 2] {
        let (a, b) = (
            2.0 * core::f64::consts::PI / 3.0,
            2.0 * core::f64::consts::PI / (3.0 * 3f64.sqrt()),
        );
        [Vector::new(a, b, 0.0), Vector::new(a, -b, 0.0)]
    }

    /// The pseudospin field: minus the hopping times the sum over the bonds of `x` turned by the
    /// phase `k | b` across each bond, plus the gap along `z`.
    pub fn pseudospin(momentum: V, gap: f64) -> V {
        let turned = bonds().map(|b| {
            let phase = (momentum | b).s();
            ((xy() * phase).exp().into_inner() * x()).cast::<Vector>()
        });
        (turned[0] + turned[1] + turned[2]) * -HOPPING + z() * gap
    }

    /// The Hamiltonian as a map on even multivectors: the field on the left, `z` on the right.
    pub fn hamiltonian(field: V) -> Hamiltonian {
        field * Rotor::slot() * z()
    }

    /// The eigenvalues (ascending) and eigenstates of the Hamiltonian. Its matrix is symmetric in
    /// the coefficients of the even multivectors, whose metric `<~a b>` is the identity there, so
    /// it is paired with the open reverse into a form and that form's eigenproblem solved.
    pub fn eigh(h: Hamiltonian) -> ([f64; 4], [R; 4]) {
        let form = Rotor::slot().reverse().scalar_product(h);
        form.eigh()
    }

    /// The pseudospin direction of a state: `psi z ~psi` over `psi ~psi`.
    pub fn direction(psi: R) -> V {
        (psi >> z()) / (psi * psi.reverse()).s()
    }

    /// The rotors that carry a frame along a curve of unit directions, each step the smallest
    /// rotation from one direction to the next, `normalize(1 + d' d)`: from the first direction to
    /// each of the others. Around a closed curve the last is the holonomy.
    pub fn transport(directions: &[V]) -> Vec<Unit<R>> {
        let mut running: Option<Unit<R>> = None;
        directions
            .windows(2)
            .map(|w| {
                let step = (Rotor::new(1.0, 0.0, 0.0, 0.0) + w[1] * w[0]).normalized();
                let next = match running {
                    Some(r) => step * r,
                    None => step,
                };
                running = Some(next);
                next
            })
            .collect()
    }

    /// Unit vectors around the circle in the plane in `count` steps, the last equal to the first.
    pub fn circle(count: usize) -> Vec<V> {
        (0..=count)
            .map(|k| {
                let angle = core::f64::consts::TAU * k as f64 / count as f64;
                (xy() * (-angle / 2.0)).exp() >> x()
            })
            .collect()
    }

    /// The Berry phase of a holonomy: its rotor's angle, signed by the sense of the turn about the
    /// direction the loop starts from.
    pub fn phase(rotor: Unit<R>, start: V) -> f64 {
        let r = rotor.into_inner();
        let about = Pseudoscalar::new(1.0) * start;
        let turned = (r.cast::<Bivector>() | about).s();
        turned.atan2(r.s())
    }

    fn grid(extent: f64, count: usize) -> Vec<V> {
        let along = |i: usize| -extent + 2.0 * extent * i as f64 / (count - 1) as f64;
        (0..count * count)
            .map(|k| Vector::new(along(k % count), along(k / count), 0.0))
            .collect()
    }

    /// The pseudospin field and the eigenvalues over a square of momenta (row by row, `count`
    /// across) that holds the Brillouin zone.
    pub struct Bands {
        pub count: usize,
        pub momenta: Vec<V>,
        pub values: Vec<[f64; 4]>,
    }

    pub fn bands(extent: f64, count: usize, gap: f64) -> Bands {
        let momenta = grid(extent, count);
        let values = momenta
            .iter()
            .map(|k| eigh(hamiltonian(pseudospin(*k, gap))).0)
            .collect();
        Bands {
            count,
            momenta,
            values,
        }
    }

    /// The pseudospin of the upper band at a momentum: the direction of its last eigenstate.
    pub fn upper_pseudospin(momentum: V, gap: f64) -> V {
        direction(eigh(hamiltonian(pseudospin(momentum, gap))).1[3])
    }

    /// Per valley, the fields and the upper band's pseudospin directions.
    pub type Textures = [(Vec<V>, Vec<V>); 2];

    /// The pseudospin of the upper band on a small square of momenta about each valley: the
    /// offsets, and per valley the fields and the directions.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn textures(radius: f64, count: usize, gap: f64) -> (Vec<V>, Textures) {
        let offsets = grid(radius, count);
        let per_valley = valleys().map(|valley| {
            let field = offsets
                .iter()
                .map(|o| pseudospin(valley + *o, gap))
                .collect();
            let directions = offsets
                .iter()
                .map(|o| upper_pseudospin(valley + *o, gap))
                .collect();
            (field, directions)
        });
        (offsets, per_valley)
    }

    /// The pseudospin directions around a loop of `radius` about `valley`, in `count` steps.
    pub fn loop_directions(valley: V, radius: f64, gap: f64, count: usize) -> Vec<V> {
        circle(count)
            .iter()
            .map(|u| {
                pseudospin(valley + *u * radius, gap)
                    .normalized()
                    .into_inner()
            })
            .collect()
    }

    /// The holonomy of loops of the given radii about each valley, for each gap, with the
    /// direction each loop starts from: `[gap][valley][radius]`.
    #[allow(clippy::type_complexity)]
    pub fn berry(radii: &[f64], gaps: &[f64], count: usize) -> Vec<[Vec<(Unit<R>, V)>; 2]> {
        gaps.iter()
            .map(|gap| {
                valleys().map(|valley| {
                    radii
                        .iter()
                        .map(|r| {
                            let d = loop_directions(valley, *r, *gap, count);
                            let holonomy = *transport(&d).last().expect("a closed loop");
                            (holonomy, d[0])
                        })
                        .collect()
                })
            })
            .collect()
    }
}

use graphene::*;

/// The gaps of the Berry phase plot, and the one shown on the sphere.
const GAPS: [f64; 4] = [0.0, 0.2, 0.5, 1.0];
const SHOWN: usize = 2;
const SECONDS: f32 = 12.0;

/// What the frames share: the bands and the Berry phases, computed once.
struct Data {
    bands: Bands,
    radii: Vec<f64>,
    phases: Vec<[Vec<f64>; 2]>,
}

fn data() -> &'static Data {
    static D: OnceLock<Data> = OnceLock::new();
    D.get_or_init(|| {
        let radii: Vec<f64> = (0..30).map(|k| 0.01 + 0.59 * k as f64 / 29.0).collect();
        let phases = berry(&radii, &GAPS, 400)
            .into_iter()
            .map(|pair| pair.map(|row| row.into_iter().map(|(r, s)| phase(r, s)).collect()))
            .collect();
        Data {
            bands: bands(4.5, 31, 0.0),
            radii,
            phases,
        }
    })
}

fn v3(v: V) -> [f32; 3] {
    [v.e1() as f32, v.e2() as f32, v.e3() as f32]
}

/// A 3D panel: the scene drawn on its own canvas over the matching stretch of the backdrop, and
/// copied into `rect`.
fn panel3(c: &mut Canvas, rect: [usize; 4], cam: Camera, fill: impl FnOnce(&mut Scene3)) {
    let [x0, y0, x1, y1] = rect;
    let mut sub = Canvas::new(x1 - x0, y1 - y0);
    let rows = (c.height.max(2) - 1) as f32;
    sub.backdrop(
        mix(palette::top(), palette::bottom(), y0 as f32 / rows),
        mix(palette::top(), palette::bottom(), (y1 - 1) as f32 / rows),
    );
    let mut scene = Scene3::new(cam);
    fill(&mut scene);
    scene.draw(&mut sub);
    c.blit(&sub, x0, y0);
}

/// The two bands over the momentum plane, energies scaled down to sit beside the momenta.
fn draw_bands(s: &mut Scene3, b: &Bands) {
    let n = b.count;
    let squash = 0.3f32;
    let top = 3.0 * HOPPING as f32;
    let node = |u: f32, v: f32| {
        let (i, j) = (
            (u * (n - 1) as f32).round() as usize,
            (v * (n - 1) as f32).round() as usize,
        );
        (i.min(n - 1), j.min(n - 1))
    };
    for (band, low, high) in [
        (3usize, palette::red(), palette::yellow()),
        (0, palette::blue(), palette::sky()),
    ] {
        s.surface(
            |u, v| {
                let (i, j) = node(u, v);
                let k = b.momenta[j * n + i];
                let e = b.values[j * n + i][band] as f32;
                [k.e1() as f32, k.e2() as f32, e * squash]
            },
            n - 1,
            n - 1,
            |u, v| {
                let (i, j) = node(u, v);
                let e = b.values[j * n + i][band].abs() as f32;
                mix(high, low, e / top)
            },
            0.95,
            None,
        );
    }
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let d = data();
    let (w, h) = (c.width, c.height);
    let (wf, hf) = (w as f32, h as f32);
    let tau = core::f32::consts::TAU;
    let phase_t = t / SECONDS;
    // The loop's radius grows and shrinks; a marker runs round it six times per cycle.
    let radius = 0.01 + 0.59 * (0.5 - 0.5 * (tau * phase_t).cos());
    let around = (6.0 * phase_t).fract();
    let top = (hf * 0.13) as usize;

    // The bands, turning.
    let left = (wf * 0.42) as usize;
    let cam = Camera::orbit(
        left,
        h - top,
        [0.0, 0.0, 0.0],
        19.0,
        tau * phase_t - 1.05,
        0.42,
        Lens::Perspective(0.62),
    );
    panel3(c, [0, top, left, h], cam, |s| draw_bands(s, &d.bands));
    c.text(
        "THE TWO BANDS: CONES WHERE THEY MEET",
        wf * 0.21,
        top as f32 + 4.0,
        12.0,
        palette::ink(),
        Align::Center,
    );

    // The field over the momentum plane, with the loop about K.
    let [valley_k, _] = valleys();
    let kv = v3(valley_k);
    let map_rect = [wf * 0.46, hf * 0.16, wf * 0.72, hf * 0.56];
    let ax = Axes::equal(map_rect, [0.0, 0.0], 3.2);
    ax.image(c, 1, |x, y| {
        let f = pseudospin(momentum(x, y), 0.0);
        let len = f.norm() as f32;
        Some(scale(colormap::viridis(len / (3.0 * HOPPING as f32)), 0.75))
    });
    // The Brillouin zone: the hexagon through the six valleys.
    let corner = 4.0 * core::f32::consts::PI / (3.0 * 3f32.sqrt());
    let hexagon: Vec<[f32; 2]> = (0..=6)
        .map(|k| {
            let a = tau / 12.0 + tau * k as f32 / 6.0;
            [corner * a.cos(), corner * a.sin()]
        })
        .collect();
    ax.polyline(c, &hexagon, 1.0, palette::ink(), 0.6);
    // The upper band's pseudospin with the gap: in the plane it turns once around each corner,
    // and near the corners it tilts out of the plane (red up, blue down), oppositely in K and K'.
    let step = 0.8f32;
    for i in -4..=4 {
        for j in -4..=4 {
            let p = [i as f32 * step, j as f32 * step];
            let d = v3(upper_pseudospin(momentum(p[0], p[1]), GAPS[SHOWN]));
            let half = 0.3;
            ax.arrow(
                c,
                [p[0] - half * d[0], p[1] - half * d[1]],
                [p[0] + half * d[0], p[1] + half * d[1]],
                1.2,
                5.0,
                colormap::coolwarm(0.5 + 0.5 * d[2]),
            );
        }
    }
    let ring: Vec<[f32; 2]> = (0..=64)
        .map(|k| {
            let a = tau * k as f32 / 64.0;
            [kv[0] + radius * a.cos(), kv[1] + radius * a.sin()]
        })
        .collect();
    ax.polyline(c, &ring, 2.0, palette::orange(), 1.0);
    let a = tau * around;
    let here = [kv[0] + radius * a.cos(), kv[1] + radius * a.sin()];
    ax.scatter(c, &[here], Marker::Dot, 7.0, palette::orange(), 1.0);
    ax.text(
        c,
        [kv[0] + 0.25, kv[1] + 0.2],
        "K",
        12.0,
        palette::ink(),
        Align::Left,
    );
    ax.frame(c, "THE PSEUDOSPIN FIELD", "MOMENTUM X (1/A)", "");

    // The pseudospin met around the loop, on the sphere, with a gap; and a frame carried along.
    let count = 240;
    let gap = GAPS[SHOWN];
    let dirs = loop_directions(valley_k, f64::from(radius), gap, count);
    let rotors = transport(&dirs);
    let at = ((around * count as f32) as usize).min(count - 1);
    let start = dirs[0];
    // A frame vector perpendicular to the start: `(z ^ d) d` is z less its part along d.
    let first_frame = ((z() ^ start) * start).cast::<gax::vga3d::Vector>();
    let first_frame = first_frame / first_frame.norm();
    let carried = if at == 0 {
        first_frame
    } else {
        rotors[at - 1] >> first_frame
    };
    let current = dirs[at];
    let holonomy = *rotors.last().expect("a closed loop");
    let gamma = phase(holonomy, start) / core::f64::consts::PI;
    // A small loop sees a cone of slope VELOCITY: its phase in closed form.
    let v_r = VELOCITY * f64::from(radius);
    let cone = 1.0 - gap / (gap * gap + v_r * v_r).sqrt();
    let sphere_rect = [(wf * 0.74) as usize, top, w, (hf * 0.6) as usize];
    let cam = Camera::orbit(
        sphere_rect[2] - sphere_rect[0],
        sphere_rect[3] - sphere_rect[1],
        [0.0; 3],
        4.6,
        0.6 + 0.3 * (tau * phase_t).sin(),
        0.35,
        Lens::Perspective(0.6),
    );
    panel3(c, sphere_rect, cam, |s| {
        s.sphere_wire([0.0; 3], 1.0, 16, palette::grid(), 0.5);
        let curve: Vec<[f32; 3]> = dirs.iter().map(|v| v3(*v)).collect();
        s.polyline(&curve, 2.0, palette::orange(), 1.0);
        s.arrow([0.0; 3], v3(current), 2.5, 9.0, palette::orange());
        let tip = v3(current);
        let f0 = v3(first_frame);
        let s0 = v3(start);
        s.arrow(s0, f0.map(|v| 0.45 * v), 1.5, 6.0, palette::grid());
        s.arrow(
            tip,
            v3(carried).map(|v| 0.45 * v),
            2.0,
            7.0,
            palette::green(),
        );
    });
    c.text(
        "PSEUDOSPIN AROUND THE LOOP",
        wf * 0.87,
        top as f32 + 4.0,
        12.0,
        palette::ink(),
        Align::Center,
    );
    c.text(
        &format!("GAP {gap:.1} EV: PHASE {gamma:+.3} PI, CONE {cone:.3}"),
        wf * 0.87,
        hf * 0.6 + 2.0,
        11.0,
        palette::green(),
        Align::Center,
    );

    // The Berry phase against the loop's radius, per gap; solid K, dashed K'.
    let ax = Axes::new(
        [wf * 0.49, hf * 0.7, wf * 0.97, hf * 0.9],
        [0.0, 0.6],
        [-1.15, 1.15],
    );
    ax.frame(c, "", "LOOP RADIUS (1/A)", "BERRY PHASE (PI)");
    ax.line(c, [0.0, 0.0], [0.6, 0.0], 1.0, palette::grid(), 1.0);
    let mut legend: Vec<(String, Rgb)> = Vec::new();
    for (g, pair) in d.phases.iter().enumerate() {
        let colour = palette::series(g + 1);
        for (valley, row) in pair.iter().enumerate() {
            let pts: Vec<[f32; 2]> = d
                .radii
                .iter()
                .zip(row)
                .map(|(r, p)| [*r as f32, (*p / core::f64::consts::PI) as f32])
                .collect();
            let width = if g == SHOWN { 2.2 } else { 1.3 };
            if valley == 0 {
                ax.polyline(c, &pts, width, colour, 1.0);
            } else {
                ax.dashed(c, &pts, width, 5.0, colour, 1.0);
            }
        }
        legend.push((format!("GAP {:.1}", GAPS[g]), colour));
    }
    ax.line(
        c,
        [radius, -1.15],
        [radius, 1.15],
        1.0,
        palette::orange(),
        0.8,
    );
    ax.scatter(
        c,
        &[[radius, gamma as f32]],
        Marker::Dot,
        7.0,
        palette::series(SHOWN + 1),
        1.0,
    );
    let entries: Vec<(&str, Rgb)> = legend.iter().map(|(s, c)| (s.as_str(), *c)).collect();
    ax.legend(c, &entries);
    caption(
        c,
        "GRAPHENE: DIRAC CONES AND THE BERRY PHASE",
        "THE PSEUDOSPIN FIELD IN VGA3D; H(PSI) = FIELD PSI Z",
    );
}

/// A momentum in the plane, from drawing coordinates.
fn momentum(x: f32, y: f32) -> V {
    gax::vga3d::Vector::new(f64::from(x), f64::from(y), 0.0)
}

fn main() {
    run(Anim::new("graphene", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::graphene::*;
    use gax::vga3d::{Rotor, Vector};

    fn close(a: V, b: V, tol: f64) -> bool {
        a.c.iter().zip(b.c).all(|(x, y)| (x - y).abs() <= tol)
    }

    /// `H(H(psi)) == (field | field) psi`, so the eigenvalues are plus and minus the field's length.
    #[test]
    fn the_hamiltonian_squares_to_the_field_s_length() {
        let field = Vector::new(0.3, -1.1, 0.7);
        let psi = Rotor::new(0.4, -0.2, 1.3, 0.5);
        let h = hamiltonian(field);
        let twice = h.of(h.of(psi)) - psi * (field | field).s();
        assert!(twice.c.iter().all(|v| v.abs() < 1e-12), "{twice:?}");
    }

    /// Around the circle at polar angle `theta` the enclosed solid angle is `2 pi (1 - cos theta)`,
    /// and the holonomy's scalar part is the cosine of half of it.
    #[test]
    fn transport_around_a_circle_of_latitude_turns_by_the_enclosed_solid_angle() {
        for theta in [0.3f64, 1.0, 2.0] {
            let directions: Vec<V> = circle(400)
                .iter()
                .map(|u| *u * theta.sin() + z() * theta.cos())
                .collect();
            let holonomy = transport(&directions).last().expect("steps").into_inner();
            let want = (core::f64::consts::PI * (1.0 - theta.cos())).cos();
            assert!((holonomy.s() - want).abs() < 1e-3, "{theta}: {holonomy:?}");
        }
    }

    /// numga's `bands` checks: the energies are minus and plus the field's length, each twice;
    /// without a gap the field vanishes at the valleys.
    #[test]
    fn the_bands_are_plus_and_minus_the_field_s_length() {
        for count in [21, 121] {
            let b = bands(4.5, count, 0.0);
            for (k, v) in b.momenta.iter().zip(&b.values) {
                let f = pseudospin(*k, 0.0);
                let l = (f | f).s().sqrt();
                for (got, want) in v.iter().zip([-l, -l, l, l]) {
                    assert!((got - want).abs() < 1e-9, "{v:?} vs {l}");
                }
            }
            assert_eq!(b.momenta.len(), count * count);
        }
        for valley in valleys() {
            assert!(close(pseudospin(valley, 0.0), Vector::zero(), 1e-6));
        }
    }

    /// numga's `textures` check: the upper band's pseudospin points along the field.
    #[test]
    fn the_upper_band_s_pseudospin_points_along_the_field() {
        for count in [7, 15] {
            let (offsets, valleys) = textures(0.5, count, 0.3);
            assert_eq!(offsets.len(), count * count);
            for (field, directions) in &valleys {
                for (f, d) in field.iter().zip(directions) {
                    assert!(close(*d, f.normalized().into_inner(), 1e-10), "{d:?} {f:?}");
                }
            }
        }
    }

    /// numga's `berry` checks: without a gap every loop comes back turned by a full turn (the
    /// rotor is -1, a Berry phase of pi); with one, a small loop's phase is that of a cone,
    /// `pi (1 - gap / sqrt(gap² + (v r)²))`, in both valleys.
    #[test]
    fn the_berry_phase_is_pi_without_a_gap_and_that_of_a_cone_with_one() {
        let cases: [(Vec<f64>, Vec<f64>, usize); 2] = [
            (
                (0..5).map(|k| 0.01 + 0.39 * k as f64 / 4.0).collect(),
                vec![0.0, 0.5],
                200,
            ),
            (
                (0..30).map(|k| 0.01 + 0.59 * k as f64 / 29.0).collect(),
                vec![0.0, 0.2, 0.5, 1.0],
                400,
            ),
        ];
        for (radii, gaps, count) in cases {
            let rotors = berry(&radii, &gaps, count);
            for (gap, pair) in gaps.iter().zip(&rotors) {
                if *gap == 0.0 {
                    for row in pair {
                        for (r, _) in row {
                            assert!((r.into_inner().s() + 1.0).abs() < 1e-12);
                        }
                    }
                }
                let v_r = VELOCITY * radii[0];
                let near = core::f64::consts::PI * (1.0 - gap / (gap * gap + v_r * v_r).sqrt());
                for row in pair {
                    let got = 1.0 - row[0].0.into_inner().s();
                    let want = 1.0 - near.cos();
                    assert!((got - want).abs() <= 1e-2 * want, "{gap}: {got} vs {want}");
                }
            }
        }
    }

    /// The two valleys wind in opposite senses: with a gap their Berry phases have opposite signs.
    #[test]
    fn the_valleys_have_opposite_berry_phases() {
        let rotors = berry(&[0.2], &[0.5], 400);
        let [k, k2] = &rotors[0];
        let (a, b) = (phase(k[0].0, k[0].1), phase(k2[0].0, k2[0].1));
        assert!((a + b).abs() < 1e-9 && a.abs() > 0.1, "{a} {b}");
    }

    #[test]
    fn a_frame_draws() {
        let mut draw = super::draw;
        let anim = gax_numga_examples::Anim::new("t", super::SECONDS).size(480, 270);
        let a = gax_numga_examples::app::frame(&anim, 0.5, &mut draw);
        let b = gax_numga_examples::app::frame(&anim, 4.0, &mut draw);
        assert!(a.mean()[0] > 0.0);
        assert!(a.mean() != b.mean());
    }
}
