//! numga's `mechanics/slip_systems`: which of a crystal's slip systems a pull makes yield
//! first, and how slipping turns the crystal under the pull (VGA3D).
//!
//! Space is the crystal's own. A face-centred cubic metal (copper, aluminium) deforms by sliding
//! layers of atoms over one another: along the four densest planes, the {111} planes, in the
//! densest directions within each, the <110> directions, which are the lines where two of those
//! planes meet. With the planes as the duals of their normals, a slip direction is the meet of
//! two planes, `planes[i] & planes[j]`: each plane meets the other three in three directions,
//! twelve slip systems in all.
//!
//! A pull along an axis is the stress map `axis (axis | Vector)`: the traction on a plane with a
//! given normal points along the axis, by the normal's share of it. The part of that traction
//! along a slip direction, the resolved shear stress, is `direction | stress(normal)`: the
//! Schmid factor `(direction | axis) (normal | axis)` times the pull, at most a half. The system
//! with the largest Schmid factor slips first.
//!
//! A small slip on a system moves each point along the slip direction by its height above the
//! plane, `amount direction (normal | x)`, and slips on several systems add. Each system slips at
//! a rate rising steeply with its Schmid factor (a power law), so the leading system takes almost
//! all of it, and two systems with equal factors share it alike. Under a pull the specimen stays
//! lined up with the pull, so in the crystal's frame the axis is what moves: the part of its move
//! across it turns it, by the rotor of the bivector `axis ^ move`, towards the leading slip
//! direction, until a second system catches up at [112] and the two slip together.
//!
//! Left, the stereographic chart of pull directions (the upper half of the sphere, a direction
//! lifted from the chart by the reflection of `z` in `z + p`), each coloured by the system that
//! yields first and dimmed where its Schmid factor is small, with the slip planes' normals
//! (triangles) and the slip directions (dots). Right, the standard triangle between [001], [011]
//! and [111] close up, with the paths of eight pulls' axes growing, each step coloured by the
//! systems that slipped in it.

use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, ORIGIN2, Point2, Rect, backdrop, caption, colormap,
    from_above, palette, reach3, run,
};
use std::sync::OnceLock;

mod slip {
    use gax::vga3d::Vector;

    pub type V = Vector<(), f64>;
    /// A stress: a map on vectors, the traction on a plane for its normal.
    pub type Stress = Vector<(Vector,), f64>;
    /// One value per slip system: per plane, per slip direction in it.
    pub type Systems<T> = [[T; DIRECTIONS]; PLANES];

    /// The slip planes, and the slip directions in each.
    pub const PLANES: usize = 4;
    pub const DIRECTIONS: usize = 3;
    pub const Z: V = Vector::new(0.0, 0.0, 1.0);

    /// The slip directions of each slip plane: the lines where it meets each of the others,
    /// normalized.
    pub fn slip_directions(normals: &[V; PLANES]) -> Systems<V> {
        let planes = normals.map(|n| n.dual());
        core::array::from_fn(|i| {
            core::array::from_fn(|k| {
                let other = planes[(i + k + 1) % PLANES];
                (planes[i] & other).normalized().into_inner()
            })
        })
    }

    /// A unit pull along the axis: the traction on a plane is the axis, by the plane's normal's
    /// share of it.
    pub fn tension(axis: V) -> Stress {
        axis * (axis | Vector::slot())
    }

    /// The shear stress a pull resolves along each slip system, per unit pull: the traction on
    /// the slip plane, along the slip direction.
    pub fn schmid(axis: V, normals: &[V; PLANES], directions: &Systems<V>) -> Systems<f64> {
        let stress = tension(axis);
        core::array::from_fn(|i| {
            let traction = stress.of(normals[i]);
            directions[i].map(|d| d.dot(traction).s())
        })
    }

    /// The axis after a small slip on every system by its amount: each slip moves the axis along
    /// its direction by its height above its plane, the moves add, and the part of their sum
    /// across the axis turns it, in the plane of the axis and the move.
    pub fn slipped(
        axis: V,
        normals: &[V; PLANES],
        directions: &Systems<V>,
        amounts: &Systems<f64>,
    ) -> V {
        let mut shift = V::zero();
        for ((normal, plane), amount) in normals.iter().zip(directions).zip(amounts) {
            let height = normal.dot(axis).s();
            for (d, a) in plane.iter().zip(amount) {
                shift += *d * (a * height);
            }
        }
        (-(axis ^ shift) / 2.0).exp() >> axis
    }

    // --- scenarios

    /// The normals of the four slip planes, the {111} planes.
    pub const NORMALS: [[f64; 3]; PLANES] = [
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
    ];
    /// The axes the pulls start along, spread over the standard triangle between [001], [011]
    /// and [111] away from [001], where eight systems nearly tie.
    pub const STARTS: [[f64; 3]; 8] = [
        [0.05, 0.3, 1.0],
        [0.15, 0.32, 1.0],
        [0.03, 0.5, 1.0],
        [0.2, 0.5, 1.0],
        [0.05, 0.75, 1.0],
        [0.3, 0.6, 1.0],
        [0.12, 0.62, 1.0],
        [0.4, 0.55, 1.0],
    ];
    /// The slip of a step on the leading system; how steeply a system's slip falls off below
    /// the leading factor (so steeply that the leading system takes almost all of it, and two
    /// systems with equal factors share it alike); and the steps taken.
    pub const AMOUNT: f64 = 0.01;
    pub const STEEPNESS: i32 = 30;
    pub const STEPS: usize = 500;

    /// A unit vector from coordinates.
    pub fn unit(c: [f64; 3]) -> V {
        Vector::from_coeffs(c).normalized().into_inner()
    }

    /// The slip planes' normals.
    pub fn normals() -> [V; PLANES] {
        NORMALS.map(unit)
    }

    /// The pull direction at a point `(x, y)` of the chart, the plane across [001]: [001]
    /// reflected in the line along [001] plus the point, the stereographic lift.
    pub fn lift(x: f64, y: f64) -> V {
        (Z + Vector::new(x, y, 0.0)).normalized() >> Z
    }

    /// The point of the chart a direction projects to, its opposite (the same pull) taken on
    /// the upper side: its part across [001], over one plus its part along it.
    pub fn project(d: V) -> V {
        let upper = if d.dot(Z).s() < 0.0 { -d } else { d };
        let across = (upper ^ Z) | Z;
        across / (1.0 + upper.dot(Z).s())
    }

    /// How much each system slips in a step: the step's slip, in the sign of the Schmid factor,
    /// scaled by the factor's size over the leading one raised to the steepness, a power law of
    /// the resolved shear stress.
    pub fn power_law(factors: &Systems<f64>) -> Systems<f64> {
        let leading = factors
            .iter()
            .flatten()
            .fold(0.0, |m: f64, f| m.max(f.abs()));
        factors
            .map(|plane| plane.map(|f| AMOUNT * f.signum() * (f.abs() / leading).powi(STEEPNESS)))
    }

    /// The slip planes' normals and slip directions.
    #[derive(Clone, Copy, Debug)]
    pub struct Crystal {
        pub normals: [V; PLANES],
        pub slips: Systems<V>,
    }

    impl Crystal {
        /// The face-centred cubic crystal.
        pub fn fcc() -> Crystal {
            let normals = normals();
            Crystal {
                normals,
                slips: slip_directions(&normals),
            }
        }

        /// The Schmid factors of a pull along the axis.
        pub fn schmid(&self, axis: V) -> Systems<f64> {
            schmid(axis, &self.normals, &self.slips)
        }

        /// One pull's axis after a slip step, and how much each system slipped in it.
        pub fn step(&self, axis: V) -> (V, Systems<f64>) {
            let amounts = power_law(&self.schmid(axis));
            (slipped(axis, &self.normals, &self.slips, &amounts), amounts)
        }

        /// A pull's axis at every step (`STEPS + 1`), and how much each system slipped in each
        /// step (`STEPS`).
        pub fn path(&self, start: V) -> (Vec<V>, Vec<Systems<f64>>) {
            let (mut axes, mut amounts) = (vec![start], Vec::with_capacity(STEPS));
            for _ in 0..STEPS {
                let (axis, slipped) = self.step(axes[axes.len() - 1]);
                axes.push(axis);
                amounts.push(slipped);
            }
            (axes, amounts)
        }

        /// Every pull's path.
        pub fn paths(&self) -> Vec<(Vec<V>, Vec<Systems<f64>>)> {
            STARTS.iter().map(|s| self.path(unit(*s))).collect()
        }
    }
}

use slip::*;

/// The paths grow over five seconds and stay for one.
const SECONDS: f32 = 6.0;
const GROWING: f32 = 5.0;
/// The Schmid factor at which a region is drawn dark, and the one at which it is in full colour.
const FAINT: f64 = 0.25;
const FULL: f64 = 0.5;
/// The regions' light at those factors: dim, so that the paths stand out.
const REGION_LIGHT: [f32; 2] = [0.03, 0.18];
/// The share of a step's slip that makes a system one of the step's carriers.
const CARRIER_SHARE: f64 = 0.1;
/// The corners of the standard triangle and their names, the directions a pull's axis heads
/// for (the leading slip direction, and the corner where two systems share), and the window
/// that frames the triangle.
const CORNERS: [([f64; 3], &str); 3] = [
    ([0.0, 0.0, 1.0], "[001]"),
    ([0.0, 1.0, 1.0], "[011]"),
    ([1.0, 1.0, 1.0], "[111]"),
];
const GOALS: [([f64; 3], &str); 2] = [([1.0, 0.0, 1.0], "[101]"), ([1.0, 1.0, 2.0], "[112]")];
const WINDOW: [f32; 2] = [-0.04, 0.46];
/// Samples of each edge of the standard triangle, an arc of a great circle.
const EDGE_SAMPLES: usize = 60;

/// The crystal and the pulls' paths, computed once.
struct Scene {
    crystal: Crystal,
    paths: Vec<(Vec<V>, Vec<Systems<f64>>)>,
}

fn scene() -> &'static Scene {
    static S: OnceLock<Scene> = OnceLock::new();
    S.get_or_init(|| {
        let crystal = Crystal::fcc();
        Scene {
            crystal,
            paths: crystal.paths(),
        }
    })
}

/// A direction on the chart, as drawn.
fn chart(d: V) -> Point2 {
    from_above(reach3(project(d)))
}

/// The colour of slip system `k` (plane by plane), around the hues.
fn system_colour(k: usize) -> Light {
    colormap::hsv(k as f32 / (PLANES * DIRECTIONS) as f32)
}

/// The colour of a pull direction: the hue of the system that yields first, dimmed where its
/// Schmid factor is small.
fn shade(factors: &Systems<f64>) -> Light {
    let sizes = factors.iter().flatten().map(|f| f.abs()).enumerate();
    let (leading, factor) = sizes.max_by(|a, b| a.1.total_cmp(&b.1)).expect("systems");
    let strength = ((factor - FAINT) / (FULL - FAINT)).clamp(0.0, 1.0) as f32;
    let [faint, full] = REGION_LIGHT;
    system_colour(leading).faded(faint + (full - faint) * strength)
}

/// The chart shaded by the leading system, inside the unit disc.
fn draw_shading(c: &mut Canvas, ax: &Axes, crystal: &Crystal) {
    ax.image(c, 1, |q| {
        let [x, y] = q.to_euclidean().map(f64::from);
        ((q & ORIGIN2).norm() <= 1.0).then(|| shade(&crystal.schmid(lift(x, y))))
    });
}

/// The standard triangle's edges, arcs of great circles between its corners.
fn draw_triangle(c: &mut Canvas, ax: &Axes) {
    for (k, (start, _)) in CORNERS.iter().enumerate() {
        let end = CORNERS[(k + 1) % CORNERS.len()].0;
        let arc: Vec<Point2> = (0..=EDGE_SAMPLES)
            .map(|i| {
                let share = i as f64 / EDGE_SAMPLES as f64;
                chart(unit(core::array::from_fn(|a| {
                    start[a] * (1.0 - share) + end[a] * share
                })))
            })
            .collect();
        ax.polyline(c, &arc, 1.2, palette::ink().faded(0.6));
    }
}

/// The whole chart: each region in the colour of its first system, the slip planes' normals as
/// triangles and the slip directions as dots, and the standard triangle.
fn draw_chart(c: &mut Canvas, rect: Rect, crystal: &Crystal) {
    let ax = Axes::equal(rect, Point2::xy(0.0, 0.0), 1.04);
    draw_shading(c, &ax, crystal);
    c.ring(
        ax.px(Point2::xy(0.0, 0.0)),
        ax.scale(),
        1.2,
        palette::ink().faded(0.5),
    );
    let normals: Vec<Point2> = crystal.normals.iter().map(|n| chart(*n)).collect();
    ax.scatter(c, &normals, Marker::Triangle, 10.0, palette::ink());
    let slips: Vec<Point2> = crystal.slips.iter().flatten().map(|d| chart(*d)).collect();
    ax.scatter(c, &slips, Marker::Dot, 6.0, palette::ink());
    draw_triangle(c, &ax);
    title(c, rect, "THE SYSTEM THAT YIELDS FIRST");
}

/// A panel's title above it.
fn title(c: &mut Canvas, rect: Rect, text: &str) {
    let above = rect.top_middle() - Point2::direction(0.0, 8.0);
    c.text(text, above, 13.0, palette::ink(), Align::Center);
}

/// A step's colour: the colours of the systems that slipped in it, mixed by how much.
fn mixed(amounts: &Systems<f64>) -> Light {
    let weights: Vec<f32> = amounts.iter().flatten().map(|a| a.abs() as f32).collect();
    let total: f32 = weights.iter().sum();
    (weights.iter().enumerate()).fold(gax_colour::DARK, |light, (k, w)| {
        light + system_colour(k) * (w / total)
    })
}

/// The systems that carry a step's slip, one bit each.
fn carriers(amounts: &Systems<f64>) -> u32 {
    let total: f64 = amounts.iter().flatten().map(|a| a.abs()).sum();
    let bits = amounts.iter().flatten().enumerate();
    bits.filter(|(_, a)| a.abs() >= CARRIER_SHARE * total)
        .fold(0, |set, (k, _)| set | 1 << k)
}

/// A path's first `reached` steps as runs on the same systems, each drawn as one stroke in the
/// colour of its first step.
fn draw_path(c: &mut Canvas, ax: &Axes, axes: &[V], amounts: &[Systems<f64>], reached: usize) {
    let mut start = 0;
    while start < reached {
        let systems = carriers(&amounts[start]);
        let length = amounts[start..reached]
            .iter()
            .take_while(|a| carriers(a) == systems)
            .count();
        let run: Vec<Point2> = axes[start..=start + length]
            .iter()
            .map(|d| chart(*d))
            .collect();
        ax.polyline(c, &run, 2.0, mixed(&amounts[start]).whitened(0.3));
        start += length;
    }
    ax.scatter(c, &[chart(axes[reached])], Marker::Dot, 6.0, palette::ink());
}

/// A name beside a point of the chart, `lift` pixels above it.
fn name(c: &mut Canvas, ax: &Axes, at: Point2, text: &str, lift: f32) {
    let px = ax.px(at) - Point2::direction(0.0, lift);
    c.text(text, px, 11.0, palette::ink().faded(0.7), Align::Center);
}

/// The standard triangle close up, with each pull's path so far, each step in the colours of
/// the systems that slipped in it, and the directions the axes head for.
fn draw_paths(c: &mut Canvas, rect: Rect, s: &Scene, reached: usize) {
    let middle = (WINDOW[0] + WINDOW[1]) / 2.0;
    let ax = Axes::equal(
        rect,
        Point2::xy(middle, middle),
        (WINDOW[1] - WINDOW[0]) / 2.0,
    );
    draw_shading(c, &ax, &s.crystal);
    draw_triangle(c, &ax);
    for ((corner, label), lift) in CORNERS.iter().zip([-18.0, 10.0, 10.0]) {
        name(c, &ax, chart(unit(*corner)), label, lift);
    }
    for (goal, label) in GOALS {
        let at = chart(unit(goal));
        ax.scatter(c, &[at], Marker::Square, 7.0, palette::ink().faded(0.7));
        name(c, &ax, at, label, 10.0);
    }
    for (axes, amounts) in &s.paths {
        draw_path(c, &ax, axes, amounts, reached);
    }
    title(c, rect, "PULLS TURNING TOWARDS [112]");
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let s = scene();
    let growth = (t.rem_euclid(SECONDS) / GROWING).min(1.0);
    let reached = 1 + (growth * (STEPS - 1) as f32).round() as usize;
    let body = c
        .rect()
        .part(0.0, 0.14, 1.0, 1.0)
        .inset(20.0, 24.0, 20.0, 16.0);
    draw_chart(c, body.part(0.0, 0.0, 0.5, 1.0), &s.crystal);
    draw_paths(c, body.part(0.5, 0.0, 1.0, 1.0), s, reached);
    caption(
        c,
        "SLIP SYSTEMS: A CRYSTAL TURNING UNDER A PULL",
        "SCHMID FACTOR (DIRECTION | AXIS)(NORMAL | AXIS); THE AXIS TURNS BY EXP(AXIS ^ MOVE) (VGA3D)",
    );
}

fn main() {
    run(Anim::new("slip_systems", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::slip::*;

    /// numga's samples across the stereographic chart of pull directions, along each side.
    const GRID: usize = 1000;

    /// The magnitudes of a set of Schmid factors.
    fn sizes(factors: &Systems<f64>) -> impl Iterator<Item = f64> + '_ {
        factors.iter().flatten().map(|f| f.abs())
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // the reference 1 / √6
    fn slip_directions_lie_in_their_planes_and_schmid_factors_are_at_most_a_half() {
        let crystal = Crystal::fcc();
        for (normal, plane) in crystal.normals.iter().zip(&crystal.slips) {
            for d in plane {
                assert!(d.dot(*normal).s().abs() < 1e-12);
            }
        }
        let across = |i: usize| -1.0 + 2.0 * i as f64 / (GRID - 1) as f64;
        for row in 0..GRID {
            for column in 0..GRID {
                let factors = crystal.schmid(lift(across(column), across(row)));
                assert!(sizes(&factors).all(|f| f <= 0.5 + 1e-12));
            }
        }
        // Along [001] eight systems share the largest factor, one over the root of six.
        let along = crystal.schmid(Z);
        let largest = sizes(&along).fold(0.0, f64::max);
        assert!((largest - 1.0 / 6f64.sqrt()).abs() < 1e-8);
        assert_eq!(
            sizes(&along).filter(|f| (f - largest).abs() < 1e-8).count(),
            8
        );
    }

    /// Every pull's axis ends at [112], two systems carrying nearly all of its slip on the way.
    #[test]
    fn pulls_turn_to_the_double_slip_corner_on_two_systems() {
        let target = unit([1.0, 1.0, 2.0]);
        for (axes, amounts) in Crystal::fcc().paths() {
            let mut end = axes[STEPS].c;
            end.sort_by(f64::total_cmp);
            assert!(V::from_coeffs(end).dot(target).s() > 0.999, "{end:?}");
            let mut carried = [0.0; PLANES * DIRECTIONS];
            for step in &amounts {
                for (sum, a) in carried.iter_mut().zip(step.iter().flatten()) {
                    *sum += a.abs();
                }
            }
            carried.sort_by(f64::total_cmp);
            let total: f64 = carried.iter().sum();
            assert!(carried[PLANES * DIRECTIONS - 2..].iter().sum::<f64>() / total > 0.95);
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
        gax_numga_examples::app::assert_draws(super::draw, 5.5);
    }
}
