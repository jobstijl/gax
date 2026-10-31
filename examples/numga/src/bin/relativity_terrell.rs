//! numga's `relativity/terrell`: the photograph of a body passing a camera at nearly the speed
//! of light, in numga's spacetime algebra `x+y+z+t-`.
//!
//! A body moving at nearly the speed of light is contracted along its motion, yet no photograph
//! shows the contraction. A photograph records the light that reaches the camera at one
//! instant, and light from the far parts of a passing body set out earlier than light from its
//! near parts, when the body was further back. The delay and the contraction together turn the
//! picture instead of squashing it: a passing cube is seen turned, showing the face that trails
//! behind it, and a sphere still shows a circular outline (Terrell and Penrose, 1959).
//!
//! Events, and the rays of light between them, are directions in spacetime, intrinsic
//! trivectors (numga's antivectors, ADR-051); a ray is null, and the directions of the camera's
//! sky are the rays arriving at it. In the body's own frame nothing moves: there the camera, at
//! `frame << event`, takes an ordinary photograph of a body at rest, each ray running straight
//! from a point of the body to the camera, and the boost `frame >> ray` carries the photograph
//! to the camera's frame. A boost is linear, so rays in one hyperplane through the camera stay
//! in one, and those rays are a circle on the sky: the boost maps circles on the sky to circles.
//! The outline of a sphere is such a circle in its own frame, so it is one in every frame, and
//! the whole photograph is the one taken at rest from where the camera is in the body's frame:
//! turned, not squashed. Seen at right angles to its motion, the body is photographed from
//! `cos θ' = β`, turned by `arcsin β`.
//!
//! The animation: a sphere above a cube passing the camera, photographed at rest, at six tenths
//! and at nine tenths of the speed of light, each time its centre is seen along the same line
//! of sight, on a stereographic map of the camera's sky, which draws circles on the sky as
//! circles. numga traces the photographs in a WGSL shader; here every pixel is traced on the
//! CPU the same way: its ray of light read in the body's frame, `frame << ray`, run straight
//! back from the camera to the first surface it meets. The cube's face ahead is red, the face
//! behind blue, the face towards the camera violet.

use gax::{vga2d, vga3d};
use gax_numga_examples::signal::frame;
use gax_numga_examples::{
    Anim, Axes, Canvas, Light, Point2, Rect, backdrop, caption, label_under, palette, run_with,
};

#[path = "../shared/spacetime.rs"]
mod spacetime;

mod terrell {
    #[cfg(test)]
    use super::spacetime::z;
    use super::spacetime::{Bivector, Direction, Even, in_space, space, time, x, y};

    /// An event or a ray of light: an intrinsic trivector.
    pub type Event = Direction;
    /// The body's frame.
    pub type Rotor = Even;

    /// The direction the body travels, the camera's line of sight to its track, and up.
    pub fn travel() -> Direction {
        space()[0]
    }
    pub fn line_of_sight() -> Direction {
        space()[1]
    }
    pub fn up() -> Direction {
        space()[2]
    }

    /// The body's frame at a speed, a fraction of the speed of light along its motion: a turn in
    /// the plane of the motion and time.
    pub fn boost(speed: f64) -> Rotor {
        let plane = (travel() & time()).hodge();
        plane.gp(speed.atanh() / 2.0).exp().into_inner()
    }

    /// The event where the body's centre sends the light the camera, at the origin, receives
    /// along a line of sight at `angle` from the body's motion, on a track at `distance`: on the
    /// camera's past light cone, the ray back along the line of sight, scaled to reach the
    /// track.
    pub fn sighting(distance: f64, angle: f64) -> Event {
        let xy = (x() ^ y()).cast::<Bivector>();
        let sight = xy.gp(-angle / 2.0).exp() >> travel();
        (sight - time()).gp(distance / (y() & sight).s())
    }

    /// Where the camera is in the body's frame when it receives the light the body's centre
    /// sent from the event `centre`: the camera's event relative to it, seen from the body,
    /// without its time.
    pub fn rest_camera(frame: Rotor, centre: Event) -> Event {
        in_space(-(frame << centre))
    }

    /// The ray from a point of a body at rest to the camera at rest beside it, in their frame:
    /// a straight line in space, taking as long as it is long.
    #[cfg(test)]
    pub fn ray(camera: Event, point: Event) -> Event {
        let offset = camera - point;
        offset + time().gp(offset.norm())
    }

    /// The outline of a sphere as the camera sees it, at `angle` around it: where the rays from
    /// the camera graze the sphere, a circle on the sphere square to the camera's direction.
    #[cfg(test)]
    pub fn outline(camera: Event, centre: Event, radius: f64, angle: f64) -> Event {
        let offset = camera - centre;
        let ratio = radius / offset.norm();
        // The grazing points ring the line from the sphere's centre to the camera. The ring is
        // built around the direction to a camera straight in front, and turned onto the
        // camera's, which is never straight behind.
        let turn = (offset.normalized() * (-line_of_sight()).inverse()).sqrt();
        #[allow(clippy::disallowed_methods)] // the grazing ring's rise √(1 - (r/d)²), numga's
        let rise = (1.0 - ratio * ratio).sqrt();
        let nearest = (-line_of_sight()).gp(ratio) + up().gp(rise);
        let zx = (z() ^ x()).cast::<Bivector>();
        let around = zx.gp(angle / 2.0).exp() >> nearest;
        centre + (turn >> around).gp(radius)
    }

    /// A ray scaled to one unit of the camera's time.
    pub fn unit_time(ray: Event) -> Event {
        ray.gp(1.0 / (super::spacetime::t() & ray).s())
    }

    // --- the scene ----------------------------------------------------------------------

    /// The distance from the camera to the track, the speeds of the cases, and the frames, the
    /// lines of sight to the centre of the pair running from three quarters of a half turn from
    /// the motion to a quarter.
    pub const TRACK_DISTANCE: f64 = 4.5;
    pub const SPEEDS: [f64; 3] = [0.0, 0.6, 0.9];
    pub const FRAMES: usize = 108;
    /// The sphere's radius and the cube's half width, and how far above and below the centre of
    /// the pair their centres are.
    pub const SPHERE_RADIUS: f64 = 1.0;
    pub const HALF_WIDTH: f64 = 1.0;
    pub const HEIGHT: f64 = 1.6;

    /// The line of sight of the `k`-th frame.
    pub fn sighting_angle(k: usize) -> f64 {
        let pi = core::f64::consts::PI;
        0.75 * pi - 0.5 * pi * k as f64 / (FRAMES - 1) as f64
    }

    /// Where the camera is in each body's frame when the centre of the pair is seen along the
    /// `k`-th frame's line of sight.
    pub fn cameras(k: usize) -> [Event; 3] {
        SPEEDS.map(|s| rest_camera(boost(s), sighting(TRACK_DISTANCE, sighting_angle(k))))
    }
}

use spacetime::{Direction, Vector, time, x, y, z};
use terrell::*;

/// numga's frame of 40 ms.
const SECONDS: f32 = FRAMES as f32 * 0.04;
/// The window of the map each lane shows, across and up.
const ACROSS: [f32; 2] = [-0.61, 0.61];
const UP: [f32; 2] = [-0.39, 0.34];
/// The widths of the cube's edges, of the sphere's grid and of its outline, in pixels; the
/// angles between the sphere's meridians and between its parallels, and how far its parallels
/// reach from its equator.
const EDGE_WIDTH: f64 = 1.1;
const GRID_WIDTH: f64 = 0.6;
const OUTLINE_WIDTH: f64 = 1.0;
const MERIDIANS: f64 = core::f64::consts::FRAC_PI_4;
const PARALLELS: f64 = core::f64::consts::PI / 6.0;
const PARALLEL_REACH: f64 = core::f64::consts::PI / 3.0;
/// Where a ray passes: far.
const FAR: f64 = 1e6;

/// The cube's faces: the slices of space they face along, outward, and their colours: the top,
/// the bottom, the face ahead, the face behind, and the faces towards and away from the camera.
fn faces() -> [(Vector, Light); 6] {
    [
        (z(), palette::YELLOW),
        (-z(), palette::INK.faded(0.5)),
        (x(), palette::RED),
        (-x(), palette::BLUE),
        (-y(), palette::PURPLE),
        (y(), palette::GREEN),
    ]
}

/// One photograph: the body's frame and where the camera is in it.
struct Photograph {
    frame: terrell::Rotor,
    camera: Direction,
    sphere: Direction,
    cube: Direction,
}

impl Photograph {
    fn new(speed: f64, camera: Direction) -> Photograph {
        Photograph {
            frame: boost(speed),
            camera,
            sphere: up().gp(HEIGHT),
            cube: up().gp(-HEIGHT),
        }
    }

    /// The direction back along the light arriving from a point of the map, in the body's
    /// frame, at unit length. The map's point on its chart (the plane through the camera
    /// square to its line of sight) gives the ray of light arriving from there: the ray from
    /// the map's pole behind the camera, reflected along the chart's point less the pole, the
    /// inverse of the stereographic projection. In the body's frame the light runs straight
    /// back from the camera to a body at rest.
    fn backward(&self, at: [f64; 2]) -> Direction {
        let behind = -line_of_sight();
        let chart = travel().gp(at[0]) + up().gp(at[1]);
        let arriving = (chart - behind).hodge() >> (time() - behind);
        let seen = self.frame << arriving;
        time() - unit_time(seen)
    }

    /// How far back along a ray from the camera its line meets the sphere first, far where it
    /// passes.
    fn sphere_hit(&self, back: Direction) -> f64 {
        let offset = self.camera - self.sphere;
        let b = offset.scalar_product(back).s();
        let discriminant =
            b * b - offset.scalar_product(offset).s() + SPHERE_RADIUS * SPHERE_RADIUS;
        if discriminant < 0.0 {
            return FAR;
        }
        #[allow(clippy::disallowed_methods)] // the ray-sphere quadratic's root, numga's shader
        let distance = -b - discriminant.sqrt();
        if distance > 0.0 { distance } else { FAR }
    }

    /// Where a ray from the camera enters the cube: how far back along it, far where it passes,
    /// and the face it enters by. Each face's pairing with the offset from the cube's centre,
    /// less half the cube's width, is linear in the distance along the ray, and the ray is
    /// inside where every one is negative.
    fn cube_hit(&self, back: Direction) -> (f64, usize) {
        let offset = self.camera - self.cube;
        let (mut entry, mut exit, mut entering) = (-FAR, FAR, 0);
        for (index, (slice, _)) in faces().iter().enumerate() {
            let b = (*slice & back).s();
            let c = (*slice & offset).s() - HALF_WIDTH;
            if b == 0.0 {
                if c > 0.0 {
                    return (FAR, 0);
                }
                continue;
            }
            let crossing = -c / b;
            if b < 0.0 && crossing > entry {
                (entry, entering) = (crossing, index);
            }
            if b > 0.0 && crossing < exit {
                exit = crossing;
            }
        }
        (
            if entry < exit && entry > 0.0 {
                entry
            } else {
                FAR
            },
            entering,
        )
    }

    /// The colour of the photograph at a point of the map, a pixel being `pixel` wide there;
    /// `None` for the sky.
    fn sample(&self, at: [f64; 2], pixel: f64) -> Option<Light> {
        let back = self.backward(at);
        // A pixel's width in space where the ray has run a distance: the angle to the ray a
        // pixel over, times the distance.
        let spread = (self.backward([at[0] + pixel, at[1]]) - back).norm();
        let to_sphere = self.sphere_hit(back);
        let (to_cube, face) = self.cube_hit(back);
        let mut colour = None;
        if to_cube < to_sphere {
            // The face the ray enters by colours it; an edge within half an edge's width of
            // another face's plane.
            let local = self.camera + back.gp(to_cube) - self.cube;
            let faces = faces();
            let gap = faces
                .iter()
                .enumerate()
                .filter(|(k, _)| *k != face)
                .map(|(_, (slice, _))| HALF_WIDTH - (*slice & local).s())
                .fold(FAR, f64::min);
            colour = Some(if gap < EDGE_WIDTH / 2.0 * spread * to_cube {
                palette::BOTTOM
            } else {
                faces[face].1.faded(0.5)
            });
        } else if to_sphere < FAR {
            let unit = (self.camera + back.gp(to_sphere) - self.sphere).gp(1.0 / SPHERE_RADIUS);
            let half_line = GRID_WIDTH / 2.0 * spread * to_sphere / SPHERE_RADIUS;
            colour = Some(if on_grid(unit, half_line) {
                palette::SKY
            } else {
                palette::SKY.faded(0.25)
            });
        }
        // The outline: where the ray passes within half a line's width of grazing the sphere.
        let offset = self.camera - self.sphere;
        let closest = -offset.scalar_product(back).s();
        let passing = (offset + back.gp(closest)).norm();
        let near = (passing - SPHERE_RADIUS).abs() < OUTLINE_WIDTH / 2.0 * spread * closest;
        if near && closest < to_cube {
            colour = Some(palette::INK);
        }
        colour
    }
}

/// Whether a point of the unit sphere lies within `half_line` of a meridian or a parallel,
/// along the sphere's surface.
fn on_grid(unit: Direction, half_line: f64) -> bool {
    let [ux, uy, uz, _] = spacetime::coordinates(unit);
    let u = vga3d::Vector::new(ux, uy, uz);
    let latitude =
        core::f64::consts::FRAC_PI_2 - u.angle_between(vga3d::Vector::new(0.0, 0.0, 1.0));
    let around = vga2d::Vector::new(ux, uy);
    let longitude = vga2d::Vector::new(1.0, 0.0).angle_to(around);
    let off_meridian =
        (longitude - (longitude / MERIDIANS).round() * MERIDIANS).abs() * around.norm();
    let nearest =
        ((latitude / PARALLELS).round() * PARALLELS).clamp(-PARALLEL_REACH, PARALLEL_REACH);
    off_meridian.min((latitude - nearest).abs()) < half_line
}

/// One lane: the photograph at a speed, from where the camera is.
fn lane(c: &mut Canvas, rect: Rect, photograph: &Photograph) {
    let ax = Axes::new(rect, ACROSS, UP);
    let pixel = f64::from(ACROSS[1] - ACROSS[0]) / f64::from(rect.width());
    ax.image(c, 1, |p| {
        let [a, b] = p.to_euclidean();
        photograph.sample([f64::from(a), f64::from(b)], pixel)
    });
    let r = ax.rect;
    c.polyline(
        &[r.lo, r.top_right(), r.hi, r.bottom_left()],
        1.0,
        palette::GRID,
        true,
    );
}

fn draw(c: &mut Canvas, _: &(), t: f32) {
    backdrop(c);
    let screen = c.rect();
    let k = frame(t / SECONDS, FRAMES);
    let cameras = cameras(k);
    let names = ["AT REST", "AT 0.6 c", "AT 0.9 c"];
    for (case, (speed, camera)) in SPEEDS.iter().zip(cameras).enumerate() {
        let column = screen.part(case as f32 / 3.0, 0.2, (case + 1) as f32 / 3.0, 0.85);
        let width = column.width() - 16.0;
        let height = width * (UP[1] - UP[0]) / (ACROSS[1] - ACROSS[0]);
        let rect = Rect::around(
            column.centre(),
            Point2::direction(width / 2.0, height / 2.0),
        );
        lane(c, rect, &Photograph::new(*speed, camera));
        label_under(c, rect.inset(0.0, 0.0, 0.0, -24.0), names[case], 10.0, 0.8);
    }
    caption(
        c,
        "TERRELL ROTATION",
        "A PASSING SPHERE AND CUBE PHOTOGRAPHED: TURNED, NOT SQUASHED; THE OUTLINE STAYS ROUND (STA)",
    );
}

fn main() {
    run_with(
        Anim::new("terrell rotation", SECONDS).size(960, 540),
        || (),
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::spacetime::{Direction, x};
    use super::terrell::*;
    use gax::ApproxEq;

    /// numga's first test: a point of the moving body is seen where its worldline in the
    /// camera's frame meets the camera's past light cone, the earlier root of a quadratic in the
    /// point's proper time; the ray from there has the direction of the ray in the body's frame,
    /// boosted.
    #[test]
    #[allow(clippy::disallowed_methods)] // the light cone's quadratic, the reference
    fn photograph_at_rest_matches_the_past_light_cone() {
        let centre = sighting(4.0, 2.2);
        let point = travel().gp(0.7) + up().gp(0.4) - line_of_sight().gp(0.3);
        for speed in [0.0, 0.6, 0.9, 0.99] {
            let frame = boost(speed);
            let start = centre + (frame >> point);
            let velocity = frame >> super::time();
            let lean = velocity.scalar_product(start).s();
            let proper = lean - (lean * lean + start.scalar_product(start).s()).sqrt();
            let direct = -(start + velocity.gp(proper));
            let boosted = frame >> ray(rest_camera(frame, centre), point);
            assert!(unit_time(direct).max_abs_diff(&unit_time(boosted)) < 1e-12);
        }
    }

    /// The join of four rays scaled to unit time.
    fn joined(p: [Direction; 4]) -> f64 {
        (p[0] & p[1] & p[2] & p[3]).s()
    }

    /// numga's second test: the rays grazing a sphere lie in one hyperplane through the camera
    /// at any speed and line of sight, so the join of any four vanishes; four rays with one
    /// from the point nearest the camera do not.
    #[test]
    fn outline_of_a_sphere_stays_a_circle() {
        let centre = up().gp(0.5) + travel().gp(0.2);
        for (speed, angle) in [(0.5, 0.6), (0.95, 2.4)] {
            let frame = boost(speed);
            let camera = rest_camera(frame, sighting(3.0, angle));
            let rays = [0.0, 1.0, 2.5, 4.0]
                .map(|a| unit_time(frame >> ray(camera, outline(camera, centre, 1.0, a))));
            assert!(joined(rays).abs() < 1e-12, "{}", joined(rays));
            let offset = camera - centre;
            let nearest = unit_time(frame >> ray(camera, centre + offset.normalized()));
            assert!(joined([rays[0], rays[1], rays[2], nearest]).abs() > 1e-6);
        }
    }

    /// numga's scenario checks: seen at right angles to its motion, the body is photographed
    /// from behind the perpendicular, by the speed's fraction of the distance; and the sphere's
    /// outline in the first frame is a circle on the sky.
    #[test]
    fn the_body_is_seen_turned_and_its_outline_round() {
        for speed in SPEEDS {
            let frame = boost(speed);
            let camera = rest_camera(
                frame,
                sighting(TRACK_DISTANCE, core::f64::consts::FRAC_PI_2),
            );
            let leaning = (x() & camera).s() / camera.norm();
            assert!((leaning + speed).abs() < 1e-12, "{leaning}");
            let camera = rest_camera(frame, sighting(TRACK_DISTANCE, sighting_angle(0)));
            let point = |k: usize| {
                let angle = core::f64::consts::TAU * k as f64 / 96.0;
                let grazing = outline(camera, up().gp(HEIGHT), SPHERE_RADIUS, angle);
                unit_time(frame >> ray(camera, grazing))
            };
            assert!(joined([point(0), point(24), point(48), point(67)]).abs() < 1e-12);
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &(), 1.0);
    }
}
