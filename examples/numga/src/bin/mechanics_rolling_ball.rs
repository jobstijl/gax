//! numga's `mechanics/rolling_ball` notebook: a ball rolled around a loop on a table (VGA3D).
//!
//! A ball that rolls without slipping on a table turns in the plane of the vertical and its
//! motion. Moving by a small step `d`, a ball of radius `r` turns by the rotor
//! `exp(-(z ^ d) / 2r)`: the outer product of the vertical with the step, the plane it rolls in,
//! scaled by the angle the step rolls it through. The ball's orientation is the running product
//! of these rotors, each later one multiplying from the left. Rolled once around a closed loop,
//! the steps do not undo each other, and the ball comes back turned: for a small loop, about the
//! vertical, by the loop's area over the square of the radius, the outer products of the steps
//! summed. The point where the ball touches the table, taken back through the orientation, draws
//! a trail on the ball, a curve on its surface.
//!
//! Left, the table from above with the loop, rolled around three times, and the ball on it, each
//! pixel showing the point of the ball beneath, taken back into the ball's own frame where its
//! pattern (two colours by the sign of the product of the coordinates, split by three great
//! circles) and the trail are fixed. Right, the ball in its own frame with the trail its contact
//! point has drawn so far.

use gax::vga3d::Bivector;
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Light, Marker, ORIGIN3, Point2, Rect, Scene3,
    backdrop, caption, from_above, measure, palette, reach3, run,
};
use std::sync::OnceLock;

#[path = "../shared/mechanics_vga3d.rs"]
mod space;

mod rolling {
    use super::space::AXES;
    pub use super::space::{B, Turn, V, turn};
    use core::f64::consts::TAU;
    use gax::vga3d::{Bivector, Rotor};

    pub const X: V = AXES[0];
    pub const Y: V = AXES[1];
    pub const Z: V = AXES[2];
    /// The plane of the table.
    pub const TABLE: B = Bivector::new(0.0, 0.0, 1.0);

    /// The ball's radius, the loop's, the times it goes around, and the steps per loop.
    pub const RADIUS: f64 = 1.0;
    pub const LOOP: f64 = 2.5;
    pub const LOOPS: usize = 3;
    pub const STEPS_PER_LOOP: usize = 720;

    /// The ball's centre around a circle of the given radius, `loops` times: the circle's points
    /// turned from x by the rotor in the table's plane.
    pub fn circle(radius: f64, loops: usize) -> Vec<V> {
        let steps = loops * STEPS_PER_LOOP;
        (0..=steps)
            .map(|k| {
                let angle = TAU * loops as f64 * k as f64 / steps as f64;
                turn(TABLE, angle) >> (X * radius)
            })
            .collect()
    }

    /// The turn of the ball rolled by a step: in the plane of the vertical and the step, by the
    /// angle it rolls through.
    pub fn roll(step: V) -> Turn {
        ((Z ^ step) * (-1.0 / (2.0 * RADIUS))).exp()
    }

    /// The ball's orientation (from its frame to the table's) after each step of a path: the
    /// running product of the steps' turns, each later one multiplying from the left.
    pub fn orientations(path: &[V]) -> Vec<Turn> {
        path.windows(2)
            .scan(Rotor::new(1.0, 0.0, 0.0, 0.0).unit(), |o, w| {
                *o = roll(w[1] - w[0]) * *o;
                Some(*o)
            })
            .collect()
    }

    /// The point of the ball touching the table, in the ball's own frame.
    pub fn contact(orientation: Turn) -> V {
        orientation << (Z * -RADIUS)
    }

    /// The area of a closed loop of points: half the sum of the outer products of successive
    /// points, a bivector in the table's plane.
    pub fn area(loop_points: &[V]) -> B {
        loop_points.windows(2).map(|w| w[0] ^ w[1]).sum::<B>() * 0.5
    }

    /// The loop's path, the orientation after each step, and the contact's trail on the ball.
    #[derive(Clone, Debug)]
    pub struct Rolling {
        pub path: Vec<V>,
        pub orientation: Vec<Turn>,
        pub contact: Vec<V>,
    }

    /// The ball rolled around the loop.
    pub fn rolling() -> Rolling {
        let path = circle(LOOP, LOOPS);
        let orientation = orientations(&path);
        let contact = orientation.iter().map(|o| contact(*o)).collect();
        Rolling {
            path,
            orientation,
            contact,
        }
    }

    /// The turn after one loop, and the loop's area.
    pub fn once(r: &Rolling) -> (Turn, B) {
        let loop_points = &r.path[..=STEPS_PER_LOOP];
        (r.orientation[STEPS_PER_LOOP - 1], area(loop_points))
    }
}

use rolling::*;

/// numga's 45 frames per loop at 50 ms, three loops.
const SECONDS: f32 = 6.75;
/// The trail's marks drawn on the ball at most, evenly spaced along it, and how near a mark a
/// point of the ball shows it, in radii.
const TRAIL_MARKS: usize = 240;
const TRAIL_WIDTH: f64 = 0.05;
/// How near a great circle of the pattern a point shows it, in radii.
const CIRCLE_WIDTH: f64 = 0.04;
/// Points around each of the pattern's great circles, in the ball's frame.
const CIRCLE_SAMPLES: usize = 96;
/// The ball's brightness where the light falls straight on it, and the share of it that lights
/// the ball everywhere.
const BALL_LIGHT: f32 = 0.3;
const AMBIENT: f64 = 0.4;
/// The table's half width as drawn, beyond the loop, in radii (numga's).
const TABLE_MARGIN: f64 = 1.3;
/// The camera on the ball's frame: its azimuth at the start (it turns once per loop), its
/// elevation and its distance.
const FRAME_AZIMUTH: f32 = -0.6;
const FRAME_ELEVATION: f32 = 0.45;
const FRAME_DISTANCE: f32 = 4.6;
/// The ball's shell, in latitudes and longitudes.
const SHELL: [usize; 2] = [18, 36];
/// The light falling on the table from the upper left.
const LIGHT: V = gax::vga3d::Vector::new(-0.4, 0.4, 0.82);

fn scene() -> &'static Rolling {
    static S: OnceLock<Rolling> = OnceLock::new();
    S.get_or_init(rolling)
}

/// A point of the table, as drawn from above.
fn table(v: V) -> Point2 {
    from_above(reach3(v))
}

/// The pattern's colour where the product of the coordinates is positive.
fn light_side() -> Light {
    palette::yellow()
}

/// The pattern's colour where it is negative.
fn dark_side() -> Light {
    palette::blue()
}

/// The colour of the great circles between them.
fn circle_light() -> Light {
    palette::bottom()
}

/// The colour of the contact's trail.
fn trail_light() -> Light {
    palette::red()
}

/// The trail so far, thinned to at most [`TRAIL_MARKS`] marks evenly along it.
fn marks(trail: &[V]) -> Vec<V> {
    let count = trail.len().min(TRAIL_MARKS);
    (0..count)
        .map(|k| trail[k * (trail.len() - 1) / (count - 1).max(1)])
        .collect()
}

/// The colour of a point of the ball, in its own frame: the pattern, darkened along the three
/// great circles where a coordinate vanishes, and the trail over it.
fn paint(body: V, marks: &[V]) -> Light {
    let [x, y, z] = body.c;
    let side = if x * y * z > 0.0 {
        light_side()
    } else {
        dark_side()
    };
    let nearest_plane = body.c.iter().fold(f64::MAX, |m, c| m.min(c.abs()));
    let on_circle = (1.0 - nearest_plane / CIRCLE_WIDTH).clamp(0.0, 1.0) as f32;
    let colour = side.mix_light(circle_light(), on_circle);
    let nearest_mark = marks
        .iter()
        .fold(f64::MAX, |m, p| m.min((body - *p).norm()));
    let on_trail = (1.0 - nearest_mark / TRAIL_WIDTH).clamp(0.0, 1.0) as f32;
    colour.mix_light(trail_light(), on_trail)
}

/// The ball seen from above at its position: each pixel the point of the ball beneath, in the
/// ball's frame, lit from the upper left.
fn draw_ball(c: &mut Canvas, ax: &Axes, centre: V, orientation: Turn, trail: &[V]) {
    let marks: Vec<V> = marks(trail).iter().map(|m| *m / RADIUS).collect();
    let light = LIGHT / LIGHT.norm();
    let middle = table(centre);
    ax.image(c, 1, |q| {
        let offset = q.unitized() - middle;
        let (u, v) = (f64::from(offset.e20()), f64::from(offset.e01()));
        let view = measure::lift(u / RADIUS, v / RADIUS)?;
        let lit = AMBIENT + (1.0 - AMBIENT) * view.dot(light).s().max(0.0);
        Some(paint(orientation << view, &marks).faded(BALL_LIGHT * lit as f32))
    });
}

/// The table from above: the loop, the centre's path so far, and the ball.
fn draw_table(c: &mut Canvas, rect: Rect, r: &Rolling, step: usize) {
    let reach = (LOOP + TABLE_MARGIN * RADIUS) as f32;
    let ax = Axes::equal(rect, Point2::xy(0.0, 0.0), reach);
    let whole: Vec<Point2> = r.path[..=STEPS_PER_LOOP]
        .iter()
        .map(|p| table(*p))
        .collect();
    ax.polyline(c, &whole, 1.0, palette::grid().faded(2.0));
    draw_ball(
        c,
        &ax,
        r.path[step + 1],
        r.orientation[step],
        &r.contact[..=step],
    );
    let corners = [rect.lo, rect.top_right(), rect.hi, rect.bottom_left()];
    c.polyline(&corners, 1.0, palette::grid(), true);
    label(c, rect, "THE TABLE FROM ABOVE");
}

/// A panel's title above it.
fn label(c: &mut Canvas, rect: Rect, title: &str) {
    let above = rect.top_middle() - Point2::direction(0.0, 8.0);
    c.text(title, above, 13.0, palette::ink(), Align::Center);
}

/// The ball in its own frame, the camera turning once per loop of the animation: a shell over
/// its three great circles, and the contact's trail with the point touching the table now.
fn draw_frame(c: &mut Canvas, rect: Rect, r: &Rolling, step: usize, t: f32) {
    let azimuth = FRAME_AZIMUTH + core::f32::consts::TAU * t / SECONDS;
    let lens = Lens::Perspective(0.62);
    let cam = Camera::orbit(
        rect,
        ORIGIN3,
        FRAME_DISTANCE,
        azimuth,
        FRAME_ELEVATION,
        lens,
    );
    panel3(c, rect, cam, |s| {
        shell(s);
        let trail: Vec<_> = r.contact[..=step].iter().map(|p| reach3(*p)).collect();
        s.polyline(&trail, 2.0, trail_light());
        s.dot(
            reach3(r.contact[step]),
            Marker::Dot,
            9.0,
            trail_light().whitened(0.4),
        );
    });
    let corners = [rect.lo, rect.top_right(), rect.hi, rect.bottom_left()];
    c.polyline(&corners, 1.0, palette::grid(), true);
    label(c, rect, "THE CONTACT'S TRAIL, IN THE BALL'S FRAME");
}

/// The ball as a faint shell, which dims what lies behind it, with the pattern's three great
/// circles.
fn shell(s: &mut Scene3) {
    let sphere = |u: f32, v: f32| {
        let (lat, lon) = (
            core::f32::consts::PI * (u - 0.5),
            core::f32::consts::TAU * v,
        );
        let turn =
            gax::pga3d::Motor::rotation_about(gax::pga3d::Point::direction(0.0, 0.0, 1.0), lon)
                * gax::pga3d::Motor::rotation_about(
                    gax::pga3d::Point::direction(0.0, -1.0, 0.0),
                    lat,
                );
        ORIGIN3 + (turn >> gax::pga3d::Point::direction(RADIUS as f32, 0.0, 0.0))
    };
    let [latitudes, longitudes] = SHELL;
    let shade = |_, _| palette::bottom().faded(4.0);
    s.surface(sphere, latitudes, longitudes, shade, 0.55, None);
    // Each circle where a coordinate vanishes: its plane, and a direction in it to start from.
    let circles = [
        (TABLE, X),
        (Bivector::new(1.0, 0.0, 0.0), Y),
        (Bivector::new(0.0, 1.0, 0.0), Z),
    ];
    for (plane, start) in circles {
        let ring: Vec<_> = (0..=CIRCLE_SAMPLES)
            .map(|k| {
                let angle = core::f64::consts::TAU * k as f64 / CIRCLE_SAMPLES as f64;
                reach3(turn(plane, angle) >> (start * RADIUS))
            })
            .collect();
        s.polyline(&ring, 1.2, palette::ink().faded(0.5));
    }
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let r = scene();
    let last = r.orientation.len() - 1;
    let step = ((t / SECONDS).rem_euclid(1.0) * last as f32).round() as usize;
    let body = c
        .rect()
        .part(0.0, 0.17, 1.0, 1.0)
        .inset(20.0, 26.0, 20.0, 20.0);
    let side = body.height();
    let left = Rect::new(
        body.lo.e20(),
        body.lo.e01(),
        body.lo.e20() + side,
        body.hi.e01(),
    );
    draw_table(c, left, r, step);
    let right = Rect::new(
        left.hi.e20() + 30.0,
        body.lo.e01(),
        body.hi.e20(),
        body.hi.e01(),
    );
    draw_frame(c, right, r, step, t);
    let (turn, _) = once(r);
    let degrees = 2.0 * turn.log().norm().to_degrees();
    let note = format!("ONE LOOP TURNS THE BALL BY {degrees:.0} DEGREES");
    let at = right.bottom_middle() - Point2::direction(0.0, 10.0);
    c.text(&note, at, 11.0, palette::ink().faded(0.6), Align::Center);
    caption(
        c,
        "ROLLING BALL: A LOOP THAT DOES NOT UNDO ITSELF",
        "EACH STEP D TURNS THE BALL BY EXP(-(Z ^ D) / 2R), THE TURNS MULTIPLIED IN ORDER (VGA3D)",
    );
}

fn main() {
    run(Anim::new("rolling_ball", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::rolling::*;

    /// The rotors keep the ball rigid, and the contact point stays on the ball.
    #[test]
    fn the_ball_stays_rigid_and_the_contact_on_it() {
        let r = rolling();
        for (o, c) in r.orientation.iter().zip(&r.contact) {
            let o = o.into_inner();
            assert!(((o * o.reverse()).s() - 1.0).abs() <= 1e-9);
            assert!((c.norm_squared() - RADIUS * RADIUS).abs() <= 1e-9);
        }
    }

    /// For a loop small against the ball, the turn after one loop is the exponential of half
    /// the loop's area over the square of the radius, about the vertical.
    #[test]
    fn a_small_loop_turns_the_ball_by_its_area() {
        let small = circle(0.05 * RADIUS, 1);
        let turned = orientations(&small)[STEPS_PER_LOOP - 1].into_inner();
        let predicted = (area(&small) / (2.0 * RADIUS * RADIUS)).exp().into_inner();
        assert!(
            (turned - predicted).c.iter().all(|x| x.abs() <= 3e-4),
            "{turned:?}"
        );
    }

    /// Regression against numga's notebook (e2765b5): the turn after one loop of the drawn
    /// loop, and its area.
    #[test]
    fn one_loop_turns_as_in_numga() {
        let (turn, loop_area) = once(&rolling());
        let expected = [0.568_736_73, -0.763_694_48, 0.0, -0.305_465_67];
        for (ours, theirs) in turn.into_inner().c.iter().zip(expected) {
            assert!((ours - theirs).abs() <= 1e-8, "{turn:?}");
        }
        assert!((loop_area.c[2] - 19.634_704_87).abs() <= 1e-8);
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
    }
}
