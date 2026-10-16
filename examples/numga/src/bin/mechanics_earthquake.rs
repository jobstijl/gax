//! numga's `mechanics/earthquake`: earthquake focal mechanisms, the moment of a slip on a fault,
//! the P waves it sends out and the beachball seismologists draw of it (VGA3D).
//!
//! Space is north, east and down (`e1`, `e2`, `e3`). A fault's orientation is one rotor, three
//! turns: by the rake within the fault plane, by the dip about north, and by the strike from north
//! towards east. It carries north onto the slip, the direction the far side moves, and up onto the
//! fault's normal; the fault dips to the right of its strike.
//!
//! A slip acts on the ground as a moment, a map on directions (`Vector <- Vector`): the slip times
//! the normal's component of its argument, `s (n · x)`, plus the mirrored couple `n (s · x)`. The
//! first term alone would turn the ground: its curl, the sum of `e_i ^ M(e_i)` over the
//! orthonormal basis (numga's `(Vector ^ M(Vector)).contract(1, 2)`, which gax has no method for),
//! is the plane of shear `n ^ s`, and the mirror cancels it. The moment is symmetric in the normal
//! and the slip: the fault and the auxiliary plane perpendicular to its slip give the same moment,
//! and no record of the waves tells them apart.
//!
//! Along a unit direction `d` the P wave's first motion is `d · M(d)`: a push away from the source
//! where it is positive, a pull where it is negative. Over the lower hemisphere, flattened by the
//! equal-area (Lambert) projection, the pushes are shaded: the beachball. Its nodal lines are the
//! fault and the auxiliary plane; the moment's eigenvectors are the pressure (P) and tension (T)
//! axes, and the two planes come back from them as `(T ± P) / √2`. The projection's square root is
//! a norm: a direction `d` lands at its horizontal part times `√2 / |down + d|`, and comes back as
//! the half turn of down about the half-way direction `(down + d) / |down + d|`, lifted from the
//! disc.
//!
//! The picture: four kinds of fault side by side (strike-slip, normal, thrust and oblique), and a
//! fault whose rake turns once around, in space with the slip of each side and the lobes of its P
//! waves (each direction drawn as far as its first motion, pushes warm and pulls cool), and its
//! beachball. The lobes are drawn over a sphere of directions turned with the fault, so that its
//! nodal planes lie along the grid lines and each cell is a push or a pull throughout.

use gax::pga2d;
use gax::vga3d::Vector;

use gax_numga_examples::plot::mark;
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Light, Marker, ORIGIN2, ORIGIN3, Point2, Rect, Scene3,
    backdrop, caption, contour, measure, palette, reach3, run,
};
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, PI, SQRT_2, TAU};
use std::sync::OnceLock;

#[path = "../shared/mechanics_vga3d.rs"]
mod space;

mod earthquake {
    pub use super::space::{Turn as Frame, V, turn};
    use super::*;

    /// A moment: a map on directions.
    pub type Moment = super::space::Map;
    /// A point of the beachball's projection.
    pub type Spot = pga2d::Point<(), f64>;

    pub const NORTH: V = Vector::new(1.0, 0.0, 0.0);
    pub const EAST: V = Vector::new(0.0, 1.0, 0.0);
    pub const DOWN: V = Vector::new(0.0, 0.0, 1.0);

    /// The faults side by side: their names, strikes, dips and rakes, in degrees.
    pub const NAMES: [&str; 4] = ["STRIKE-SLIP", "NORMAL", "THRUST", "OBLIQUE"];
    pub const STRIKES: [f64; 4] = [30.0, 30.0, 30.0, 30.0];
    pub const DIPS: [f64; 4] = [90.0, 60.0, 30.0, 50.0];
    pub const RAKES: [f64; 4] = [0.0, -90.0, 90.0, 45.0];
    /// The seismic moment of every fault.
    pub const MOMENT: f64 = 1.0;
    /// The sweep: one fault's strike and dip, in degrees; its rake turns once around.
    pub const SWEEP_STRIKE: f64 = 30.0;
    pub const SWEEP_DIP: f64 = 50.0;
    /// The grids of directions: the hemisphere's rings from straight down, and the spokes
    /// around the vertical.
    pub const RINGS: usize = 60;
    pub const SPOKES: usize = 120;

    /// The orientation of a fault, from its strike, dip and rake in radians: the turn by the rake
    /// in the horizontal plane (north towards west), then the dip about north (east towards
    /// down), then the strike from north towards east.
    pub fn fault_frame(strike: f64, dip: f64, rake: f64) -> Frame {
        turn(NORTH ^ EAST, strike) * turn(EAST ^ DOWN, dip) * turn(NORTH ^ EAST, -rake)
    }

    /// A fault's normal and slip: where its orientation carries up and north.
    pub fn normal_and_slip(frame: Frame) -> (V, V) {
        (frame >> -DOWN, frame >> NORTH)
    }

    /// The moment of a slip on a fault: the slip's couple across the fault, and the mirrored
    /// couple that keeps the ground from turning.
    pub fn double_couple(normal: V, slip: V, moment: f64) -> Moment {
        (slip * (normal | Vector::slot()) + normal * (slip | Vector::slot())) * moment
    }

    /// Along a unit direction, the P wave's first motion, positive for a push away from the
    /// source, and the S wave's, the part of the moment's response across the direction.
    pub fn radiation(moment: Moment, direction: V) -> (f64, V) {
        let response = moment.of(direction);
        let push = (direction | response).s();
        (push, response - direction * push)
    }

    /// The direction `polar` away from straight down towards north, then turned about the
    /// vertical by `azimuth` from north towards east.
    pub fn direction(polar: f64, azimuth: f64) -> V {
        (turn(NORTH ^ EAST, azimuth) * turn(DOWN ^ NORTH, polar)) >> DOWN
    }

    /// Directions on the sphere from straight down to `reach` away from it: `rings` rings of
    /// `SPOKES + 1` spokes, the last spoke closing the first.
    #[derive(Clone, Copy, Debug)]
    pub struct Grid {
        pub reach: f64,
        pub rings: usize,
    }

    /// The lower half of the sphere of directions, where seismologists draw the beachball.
    pub const HEMISPHERE: Grid = Grid {
        reach: FRAC_PI_2,
        rings: RINGS,
    };
    /// The whole sphere of directions, with the hemisphere's spacing and a ring on its equator.
    pub const SPHERE: Grid = Grid {
        reach: PI,
        rings: 2 * RINGS - 1,
    };

    impl Grid {
        /// The direction a fraction `u` of the way from straight down to the reach, and a
        /// fraction `v` of a turn around the vertical.
        pub fn at(self, u: f64, v: f64) -> V {
            direction(u * self.reach, v * TAU)
        }

        /// The grid's directions `[rings][SPOKES + 1]`.
        pub fn directions(self) -> Vec<Vec<V>> {
            let fraction = |k: usize, n: usize| k as f64 / n as f64;
            (0..self.rings)
                .map(|i| {
                    (0..=SPOKES)
                        .map(|j| self.at(fraction(i, self.rings - 1), fraction(j, SPOKES)))
                        .collect()
                })
                .collect()
        }
    }

    /// A fault: its orientation, its normal and slip, its moment with its principal axes
    /// (pressure, null, tension), and the P wave's first motion over the lower hemisphere
    /// `[RINGS][SPOKES + 1]`.
    #[derive(Clone, Debug)]
    pub struct Mechanism {
        pub frame: Frame,
        pub normal: V,
        pub slip: V,
        pub moment: Moment,
        pub axes: [V; 3],
        pub push: Vec<Vec<f64>>,
    }

    /// The fault of the given orientation and moment.
    pub fn mechanism(frame: Frame, moment: f64) -> Mechanism {
        let (normal, slip) = normal_and_slip(frame);
        let moment = double_couple(normal, slip, moment);
        let (_, axes) = moment.eigh();
        let push = HEMISPHERE
            .directions()
            .iter()
            .map(|ring| ring.iter().map(|d| radiation(moment, *d).0).collect())
            .collect();
        Mechanism {
            frame,
            normal,
            slip,
            moment,
            axes,
            push,
        }
    }

    /// The four faults side by side.
    pub fn faults() -> [Mechanism; 4] {
        core::array::from_fn(|k| {
            let [strike, dip, rake] = [STRIKES[k], DIPS[k], RAKES[k]].map(f64::to_radians);
            mechanism(fault_frame(strike, dip, rake), MOMENT)
        })
    }

    /// The sweep's fault at `rake` (radians).
    pub fn swept(rake: f64) -> Mechanism {
        let frame = fault_frame(SWEEP_STRIKE.to_radians(), SWEEP_DIP.to_radians(), rake);
        mechanism(frame, MOMENT)
    }

    /// A direction in the equal-area projection of the lower hemisphere, east across and north
    /// up: its horizontal part times `√2 / |down + d|`. A direction on the upper hemisphere is
    /// drawn as its opposite.
    pub fn lambert(d: V) -> Spot {
        let d = if d.e3() < 0.0 { -d } else { d };
        let k = SQRT_2 / (DOWN + d).norm();
        pga2d::Point::xy(d.e2() * k, d.e1() * k)
    }

    /// The direction of the lower hemisphere at a spot of the projection, or `None` outside the
    /// unit disc: the half-way direction between down and it has the spot's horizontal part over
    /// `√2`, lifted onto the unit sphere, and turns down onto the direction by half a turn about
    /// itself. The disc's rim is where the half-way direction is an eighth of a turn from down.
    pub fn unlambert(spot: Spot) -> Option<V> {
        let [east, north] = spot.to_euclidean();
        let lifted = measure::lift(east * FRAC_1_SQRT_2, north * FRAC_1_SQRT_2)?;
        let half = Vector::new(lifted.e2(), lifted.e1(), lifted.e3());
        (half.e3() >= FRAC_1_SQRT_2).then(|| half * (2.0 * (half | DOWN).s()) - DOWN)
    }
}

use earthquake::*;

/// One turn of the sweep's rake: numga's 72 frames at 80 ms.
const SECONDS: f32 = 72.0 * 0.08;
/// The half width of the fault patch drawn in space.
const PATCH: f64 = 0.55;
/// The view in space: numga's elevation and azimuth.
const ELEVATION: f32 = 24.0;
const AZIMUTH: f32 = -60.0;
/// The radius of the compass on the ground.
const COMPASS: f64 = 1.1;

fn faults_drawn() -> &'static [Mechanism; 4] {
    static F: OnceLock<[Mechanism; 4]> = OnceLock::new();
    F.get_or_init(faults)
}

/// The sweep's rake at time `t`: once around, from -180 to 180 degrees.
fn rake_at(t: f32) -> f64 {
    -PI + TAU * f64::from(t.rem_euclid(SECONDS) / SECONDS)
}

/// Pushes glow warm, pulls stay dim and cool.
fn push_light() -> Light {
    palette::orange().faded(0.5)
}

fn pull_light() -> Light {
    palette::sky().faded(0.06)
}

/// The pressure axis, where the ground is squeezed, and the tension axis, where it is pulled.
fn pressure_light() -> Light {
    palette::sky()
}

fn tension_light() -> Light {
    palette::yellow()
}

/// A direction in space for drawing: east, north and up.
fn upright(v: V) -> V {
    Vector::new(v.e2(), v.e1(), -v.e3())
}

/// The beachball of a fault in the square `rect`: the lower hemisphere in the equal-area
/// projection with the pushes shaded, the nodal lines (the level zero of the first motions over
/// the grid, its points carried through the projection), the rim, and the pressure and tension
/// axes.
fn beachball(c: &mut Canvas, rect: Rect, fault: &Mechanism) {
    let ax = Axes::equal(rect, ORIGIN2, 1.15);
    ax.image(c, 2, |q| {
        let d = unlambert(q.map_coefs(f64::from))?;
        let (push, _) = radiation(fault.moment, d);
        Some(if push > 0.0 {
            push_light()
        } else {
            pull_light()
        })
    });
    let samples: Vec<f32> = fault.push.iter().flatten().map(|p| *p as f32).collect();
    let on_grid = |q: Point2| {
        let [spoke, ring] = q.to_euclidean().map(f64::from);
        let d = HEMISPHERE.at(ring / (HEMISPHERE.rings - 1) as f64, spoke / SPOKES as f64);
        ax.px(lambert(d))
    };
    let nodal: Vec<[Point2; 2]> = contour::segments(&samples, SPOKES + 1, HEMISPHERE.rings, 0.0)
        .into_iter()
        .map(|seg| seg.map(on_grid))
        .collect();
    let width = (rect.height() / 110.0).max(1.0);
    c.stroke(&nodal, width, palette::ink());
    c.ring(ax.px(ORIGIN2), ax.scale(), width * 1.2, palette::ink());
    let size = rect.height() / 14.0;
    let axes = [
        (fault.axes[0], "P", pressure_light()),
        (fault.axes[2], "T", tension_light()),
    ];
    for (axis, label, light) in axes {
        let at = ax.px(lambert(axis));
        mark(c, at, Marker::Ring, size, palette::bottom());
        mark(c, at, Marker::Dot, size, light);
        let beside = at + Point2::direction(0.6 * size, -0.4 * size);
        c.text(label, beside, size * 0.9, light, Align::Left);
    }
    let north = ax.px(Spot::xy(0.0, 1.08));
    c.text(
        "N",
        north,
        size * 0.8,
        palette::ink().faded(0.5),
        Align::Center,
    );
}

/// The lobes of the P waves: each direction drawn as far as its first motion, over the sphere
/// of directions turned with the fault, so that its nodal planes lie along the grid lines; each
/// cell coloured by the first motion at its middle.
fn lobes(s: &mut Scene3, fault: &Mechanism) {
    let along = |u: f32, v: f32| fault.frame >> SPHERE.at(f64::from(u), f64::from(v));
    let tip = |u: f32, v: f32| {
        let d = along(u, v);
        let (push, _) = radiation(fault.moment, d);
        reach3(upright(d * push.abs()))
    };
    let shade = |u: f32, v: f32| {
        let (push, _) = radiation(fault.moment, along(u, v));
        if push > 0.0 {
            palette::orange().faded(0.8)
        } else {
            palette::sky().faded(0.35)
        }
    };
    s.surface(tip, SPHERE.rings - 1, SPOKES, shade, 0.6, None);
}

/// The fault patch, a disc across the normal, and the slip of each side across it.
fn patch(s: &mut Scene3, fault: &Mechanism) {
    let null = fault.axes[1];
    let rim: Vec<_> = (0..=64)
        .map(|k| {
            let around = turn(fault.slip ^ null, TAU * f64::from(k) / 64.0);
            reach3(upright((around >> fault.slip) * PATCH))
        })
        .collect();
    for pair in rim.windows(2) {
        s.tri(
            ORIGIN3,
            pair[0],
            pair[1],
            palette::yellow().faded(0.12),
            0.3,
        );
    }
    s.polyline(&rim, 1.5, palette::yellow().faded(0.8));
    // Each side's slip: an arrow just off the fault on that side, centred on the source.
    for side in [1.0, -1.0] {
        let base = (fault.normal * 0.15 - fault.slip * 0.45) * side;
        s.arrow(
            reach3(upright(base)),
            upright(fault.slip * (0.9 * side)),
            2.5,
            10.0,
            palette::yellow(),
        );
    }
}

/// A faint compass on the ground: a circle around the source, with a tick to the north.
fn compass(s: &mut Scene3) {
    let ring: Vec<_> = (0..=96)
        .map(|k| {
            reach3(upright(
                direction(FRAC_PI_2, TAU * f64::from(k) / 96.0) * COMPASS,
            ))
        })
        .collect();
    s.polyline(&ring, 1.0, palette::grid());
    let tick = reach3(upright(NORTH * COMPASS));
    s.arrow(tick, upright(NORTH * 0.1), 1.0, 6.0, palette::grid());
}

/// The sweep's fault in space, in `rect`: the lobes of its P waves, with the fault and its slip
/// drawn over them, as numga does.
fn fault_in_space(c: &mut Canvas, rect: Rect, fault: &Mechanism) {
    let (azimuth, elevation) = (AZIMUTH.to_radians(), ELEVATION.to_radians());
    let lens = Lens::Perspective(0.42);
    let cam = Camera::orbit(rect, ORIGIN3, 7.0, azimuth, elevation, lens);
    let cam = panel3(c, rect, cam, |s| {
        compass(s);
        lobes(s, fault);
    });
    panel3(c, rect, cam, |s| patch(s, fault));
    if let Some(at) = cam.px(reach3(upright(NORTH * (COMPASS + 0.15)))) {
        let size = rect.height() / 32.0;
        c.text("N", at, size, palette::grid().faded(4.0), Align::Center);
    }
}

/// A beachball's label under it.
fn label(c: &mut Canvas, rect: Rect, title: &str, detail: &str) {
    let size = rect.height() / 15.0;
    let under = rect.bottom_middle() + Point2::direction(0.0, size * 0.2);
    c.text(title, under, size, palette::ink(), Align::Center);
    let below = under + Point2::direction(0.0, size * 1.3);
    c.text(
        detail,
        below,
        size * 0.7,
        palette::ink().faded(0.5),
        Align::Center,
    );
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let top = h * 0.14;
    // The four faults, two by two on the left.
    let cell_w = w * 0.25;
    let cell_h = (h - top) * 0.5;
    for (k, (fault, name)) in faults_drawn().iter().zip(NAMES).enumerate() {
        let corner = Point2::xy(cell_w * (k % 2) as f32, top + cell_h * (k / 2) as f32);
        let side = cell_h * 0.72;
        let centre = corner + Point2::direction(cell_w * 0.5, cell_h * 0.42);
        let square = Rect::around(centre, Point2::direction(side * 0.5, side * 0.5));
        beachball(c, square, fault);
        let detail = format!("STRIKE {} DIP {} RAKE {}", STRIKES[k], DIPS[k], RAKES[k]);
        label(c, square, name, &detail);
    }
    // The sweep on the right: the fault in space, and its beachball in the corner.
    let rake = rake_at(t);
    let fault = swept(rake);
    let right = Rect::new(w * 0.47, top, w * 0.86, h);
    fault_in_space(c, right, &fault);
    let side = (h - top) * 0.3;
    let corner = screen.hi - Point2::direction(side * 0.65, side * 0.85);
    let square = Rect::around(corner, Point2::direction(side * 0.5, side * 0.5));
    beachball(c, square, &fault);
    let detail = format!("STRIKE {SWEEP_STRIKE} DIP {SWEEP_DIP}");
    label(
        c,
        square,
        &format!("RAKE {:+.0}", rake.to_degrees()),
        &detail,
    );
    caption(
        c,
        "EARTHQUAKE FOCAL MECHANISMS",
        "A SLIP'S MOMENT IS A MAP ON DIRECTIONS; THE P WAVE ITS FORM D . M(D) (VGA3D)",
    );
}

fn main() {
    run(Anim::new("earthquake", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::earthquake::*;
    use super::space::curl;
    use gax::vga3d::Vector;

    /// The two faults with a moment, as `(normal, slip)`, from its pressure and tension axes:
    /// each fault's normal is the other's slip, `(T ± P) / √2`.
    fn fault_planes(moment: Moment) -> [(V, V); 2] {
        let (_, [pressure, _, tension]) = moment.eigh();
        let first = (tension + pressure).normalized().into_inner();
        let second = (tension - pressure).normalized().into_inner();
        [(first, second), (second, first)]
    }

    /// The largest coefficient of a moment.
    fn largest(m: Moment) -> f64 {
        m.c.iter().flatten().fold(0.0, |a: f64, x| a.max(x.abs()))
    }

    /// numga's `test_fault_frame_matches_aki_and_richards`: the normal and slip of the fault
    /// rotor are the textbook formulas in strike, dip and rake.
    #[test]
    #[allow(clippy::disallowed_methods)] // the textbook's formulas, the reference
    fn fault_frame_matches_aki_and_richards() {
        let strikes = [0.0f64, 75.0, 210.0].map(f64::to_radians);
        let dips = [90.0f64, 35.0, 60.0].map(f64::to_radians);
        let rakes = [0.0f64, 120.0, -45.0].map(f64::to_radians);
        for k in 0..3 {
            let (s, d, r) = (strikes[k], dips[k], rakes[k]);
            let (normal, slip) = normal_and_slip(fault_frame(s, d, r));
            let expected_normal = Vector::new(-d.sin() * s.sin(), d.sin() * s.cos(), -d.cos());
            let expected_slip = Vector::new(
                r.cos() * s.cos() + r.sin() * d.cos() * s.sin(),
                r.cos() * s.sin() - r.sin() * d.cos() * s.cos(),
                -r.sin() * d.sin(),
            );
            assert!((normal - expected_normal).norm() < 1e-13, "{normal:?}");
            assert!((slip - expected_slip).norm() < 1e-13, "{slip:?}");
        }
    }

    /// numga's `test_p_wave_is_twice_the_product_of_components`: along a unit direction the P
    /// wave of a unit double couple is twice its component along the normal times its component
    /// along the slip, and the S wave is across the direction.
    #[test]
    fn p_wave_is_twice_the_product_of_components() {
        let frame = fault_frame(40f64.to_radians(), 55f64.to_radians(), 20f64.to_radians());
        let (normal, slip) = normal_and_slip(frame);
        let moment = double_couple(normal, slip, 1.0);
        for d in SPHERE.directions().into_iter().flatten() {
            let (push, shear) = radiation(moment, d);
            let expected = 2.0 * (d | normal).s() * (d | slip).s();
            assert!((push - expected).abs() < 1e-13);
            assert!((shear | d).s().abs() < 1e-13);
        }
    }

    /// The checks of numga's `mechanisms()`: a slip neither swells nor shrinks the ground and has
    /// no third axis; the single couple turns the ground and the double couple does not; the
    /// fault and the auxiliary plane give the same moment and come back from it; the P wave is
    /// strongest along the tension axis and silent along the slip.
    #[test]
    fn the_four_faults_pass_their_checks() {
        for fault in faults() {
            let m = fault.moment;
            assert!(m.trace().abs() < 1e-14 && m.det().abs() < 1e-14);
            let (values, _) = m.eigh();
            for (value, expected) in values.iter().zip([-MOMENT, 0.0, MOMENT]) {
                assert!((value - expected).abs() < 1e-13, "{values:?}");
            }
            let single = fault.slip * (fault.normal | Vector::slot());
            let shear = fault.normal ^ fault.slip;
            assert!((curl(single) - shear).norm() < 1e-14);
            assert!(curl(m).norm() < 1e-14);
            let swapped = double_couple(fault.slip, fault.normal, MOMENT);
            assert!(largest(swapped - m) < 1e-14);
            let planes = fault_planes(m);
            for (normal, slip) in planes {
                assert!(largest(double_couple(normal, slip, MOMENT) - m) < 1e-13);
            }
            let alignment = planes
                .map(|(normal, _)| (normal | fault.normal).s().abs())
                .into_iter()
                .fold(0.0, f64::max);
            assert!((alignment - 1.0).abs() < 1e-12, "{alignment}");
            for (axis, value) in fault.axes.iter().zip(values) {
                assert!((radiation(m, *axis).0 - value).abs() < 1e-13);
            }
            assert!(radiation(m, fault.slip).0.abs() < 1e-14);
        }
    }

    /// The checks of numga's `sweep()` over its 72 frames: turning the rake turns the slip
    /// within the fault (the normal stays, the slip stays in the plane), and half a turn of rake
    /// reverses the slip and flips the moment's sign.
    #[test]
    fn the_rake_sweep_passes_its_checks() {
        let frames = 72;
        let rakes = (0..frames)
            .map(|k| -core::f64::consts::PI + core::f64::consts::TAU * k as f64 / frames as f64);
        let sweep: Vec<Mechanism> = rakes.map(swept).collect();
        for fault in &sweep {
            assert!((fault.normal - sweep[0].normal).norm() < 1e-14);
            assert!((fault.slip | fault.normal).s().abs() < 1e-14);
        }
        let half = frames / 2;
        for k in 0..half {
            assert!(largest(sweep[k + half].moment + sweep[k].moment) < 1e-14);
        }
    }

    /// The projection and its inverse agree on the lower hemisphere, and agree with numga's
    /// `(east, north) / sqrt(1 + down)`.
    #[test]
    fn the_projection_comes_back() {
        for d in HEMISPHERE.directions().into_iter().flatten() {
            let spot = lambert(d);
            let [east, north] = spot.to_euclidean();
            let k = (1.0 + d.e3()).powf(-0.5);
            assert!((east - d.e2() * k).abs() < 1e-12 && (north - d.e1() * k).abs() < 1e-12);
            // Pulled in a hair from the rim, where the half-way direction is exactly at the limit.
            let shrink = 1.0 - 1e-12;
            let inside = Spot::xy(east * shrink, north * shrink);
            let back = unlambert(inside).expect("inside");
            assert!((back - d).norm() < 1e-6, "{d:?} {back:?}");
        }
        assert!(unlambert(gax::pga2d::Point::xy(0.8, 0.7)).is_none());
    }

    /// Numbers from numga: the oblique fault's normal and moment, and a first motion of the
    /// normal fault on the hemisphere's grid (ring 10, spoke 7).
    #[test]
    fn numbers_match_numga() {
        let faults = faults();
        let oblique = &faults[3];
        let normal = Vector::new(-0.3830222215594891, 0.6634139481689383, -0.6427876096865393);
        assert!((oblique.normal - normal).norm() < 1e-14);
        let first_row = [-0.643195561564002, 0.572372171411945, -0.332231512194337];
        let row = oblique.moment.of(Vector::new(1.0, 0.0, 0.0));
        assert!((row - Vector::new(first_row[0], first_row[1], first_row[2])).norm() < 1e-14);
        assert!((faults[1].push[10][7] - -0.8443177238822391).abs() < 1e-14);
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
    }
}
