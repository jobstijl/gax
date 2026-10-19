//! numga's `geometry/three_point_pose`: a rigid pose recovered from three measured markers, by
//! matching a point, then a line, then a plane (after the motor reconstruction example of
//! ganja.js by Steven De Keninck).
//!
//! In PGA3D the motor that carries one element onto another, a point onto a point, a line onto
//! a line or a plane onto a plane, is the square root of their ratio (`Motor::between`). The
//! first motor translates the first marker onto its target. The second turns the line through
//! the first two markers onto the target's, about an axis through the matched first point,
//! which it keeps. The third turns the plane of all three onto the target's, about the matched
//! line, which it keeps. Each step removes the freedom the previous one left, and their product
//! is the pose. The construction is exact for congruent, ordered, noncollinear triples; noisy
//! measurements call for a fit instead.
//!
//! The animation carries the markers through each motor at a constant rate along its screw (a
//! fraction of the motor's logarithm), with the motor's axis drawn: at infinity for the
//! translation, through the first point for the turn, along the first edge for the roll. On the
//! right, the markers after each stage.

use gax::pga3d::Point;
use gax_numga_examples::signal::smoothstep;
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Light, Marker, Point2, Rect, Scene3, backdrop, caption,
    palette, points::sides, run, scene3::panel3,
};

mod pose {
    use gax::Unit;
    use gax::pga3d::{Line, Motor, Point};

    pub type M = Unit<Motor>;
    /// Three markers, in order.
    pub type Markers = [Point; 3];

    /// The three incremental motors that carry the source markers onto the congruent target
    /// markers: the translation of the first point, the turn of the first edge about the
    /// matched point, and the roll of the triangle's plane about the matched edge. Each is the
    /// shorter motor between its elements, so no intermediate element may oppose its target.
    pub fn reconstruct(source: &Markers, target: &Markers) -> [M; 3] {
        let translation = Motor::between(source[0], target[0]);
        let target_line = target[0] & target[1];
        let current_line = target[0] & (translation >> source[1]);
        let turn = Motor::between(current_line, target_line);
        let placed = turn * translation;
        let target_plane = target_line & target[2];
        let current_plane = target_line & (placed >> source[2]);
        let roll = Motor::between(current_plane, target_plane);
        [translation, turn, roll]
    }

    /// The markers moved by a motor.
    pub fn moved(motor: M, markers: &Markers) -> Markers {
        markers.map(|p| motor >> p)
    }

    /// The markers after each alignment.
    pub fn placements(source: &Markers, increments: &[M; 3]) -> [Markers; 3] {
        let mut placed = *source;
        increments.map(|m| {
            placed = moved(m, &placed);
            placed
        })
    }

    /// The markers a fraction of the way through an alignment from where the previous ones
    /// left them: a fraction of the motor's logarithm moves at a constant rate along its screw.
    pub fn partway(start: &Markers, increment: M, fraction: f64) -> Markers {
        let screw = increment.log();
        moved(screw.gp(fraction).exp(), start)
    }

    /// numga's markers: a triangle, and the same triangle moved by a screw motion (numga's
    /// `exp(0.7 xw + 0.2 yw + 0.4 zw) exp(0.6 xy + 0.2 yz)`, with `xw = -e01`).
    pub fn markers() -> (Markers, Markers) {
        let source = [
            Point::xyz(-0.8, -0.4, 0.0),
            Point::xyz(0.9, -0.3, 0.0),
            Point::xyz(-0.3, 0.8, 0.2),
        ];
        let displacement = Line::new(0.0, 0.0, 0.0, -0.7, -0.2, -0.4).exp();
        let rotation = Line::new(0.2, 0.0, 0.6, 0.0, 0.0, 0.0).exp();
        let target = moved(displacement * rotation, &source);
        (source, target)
    }
}

use pose::*;

/// The seconds per alignment, and the hold at the end of the loop.
const STAGE: f32 = 2.5;
const HOLD: f32 = 1.5;
const SECONDS: f32 = 3.0 * STAGE + HOLD;
/// The stages' names.
const STAGES: [&str; 3] = ["1  MATCH A POINT", "2  MATCH A LINE", "3  MATCH A PLANE"];

fn target_colour() -> Light {
    palette::ORANGE
}

fn marker_colour() -> Light {
    palette::SKY
}

/// The stage under way at time `t` and how far through it, eased in and out (numga's
/// smoothstep); past the last stage, the last one complete.
fn stage_at(t: f32) -> (usize, f64) {
    let stage = ((t / STAGE) as usize).min(2);
    (
        stage,
        smoothstep(f64::from((t - stage as f32 * STAGE) / STAGE)),
    )
}

/// The camera on the markers from `distance` away: numga's view from 55 degrees up at an
/// azimuth of -110 degrees, about the middle of the source and target markers.
fn camera(rect: Rect, distance: f32) -> Camera {
    let (source, target) = markers();
    let middle = source
        .iter()
        .chain(&target)
        .fold(gax::pga3d::Point::direction(0.0, 0.0, 0.0), |sum, p| {
            sum + p.unitized()
        });
    Camera::orbit(
        rect,
        middle.gp(1.0 / 6.0),
        distance,
        (-110f32).to_radians(),
        55f32.to_radians(),
        Lens::Perspective(0.42),
    )
}

/// The target markers: a dashed outline and dots.
fn draw_target(s: &mut Scene3, target: &Markers) {
    // Dashes per side, and the share of its stretch of the side each dash inks.
    const DASHES: usize = 9;
    const INKED: f64 = 0.55;
    for (a, b) in sides(target) {
        let at = |f: f64| a + (b - a).gp(f / DASHES as f64);
        for d in 0..DASHES {
            let from = d as f64;
            s.seg(at(from), at(from + INKED), 2.0, target_colour());
        }
    }
    for p in target {
        s.dot(*p, Marker::Dot, 9.0, target_colour());
    }
}

/// A placed marker triangle: filled, outlined, its corners dotted.
fn draw_markers(s: &mut Scene3, markers: &Markers, strength: f32) {
    let colour = marker_colour().faded(strength);
    let [a, b, c] = *markers;
    s.tri(a, b, c, colour, 0.35 * strength);
    s.polyline(&[a, b, c, a], 2.0, colour);
    for p in markers {
        s.dot(*p, Marker::Dot, 6.0, colour);
    }
}

/// The axis of a motor's screw, a stretch of its line about the point of it nearest to
/// `near`; nothing for a translation, whose axis lies at infinity.
fn draw_axis(s: &mut Scene3, motor: M, near: Point) {
    let screw = motor.log();
    // The axis's direction, as a displacement (its sign does not matter: the stretch is drawn
    // both ways).
    let direction = screw.direction();
    if direction.ideal_norm() < 1e-9 {
        return;
    }
    // The foot of the perpendicular from `near`: the line met with the plane through `near`
    // orthogonal to it.
    let foot = ((screw | near) ^ screw).unitized();
    let reach = direction.gp(1.6 / direction.ideal_norm());
    s.seg(foot - reach, foot + reach, 1.6, palette::YELLOW);
}

/// The animated alignment, in `rect`.
fn alignment_panel(c: &mut Canvas, rect: Rect, t: f32) {
    let (source, target) = markers();
    let increments = reconstruct(&source, &target);
    let stages = placements(&source, &increments);
    let (stage, fraction) = stage_at(t);
    let starts = [source, stages[0], stages[1]];
    let now = partway(&starts[stage], increments[stage], fraction);
    panel3(c, rect, camera(rect, 10.0), |s| {
        draw_target(s, &target);
        draw_markers(s, &source, 0.25);
        draw_axis(s, increments[stage], target[0]);
        draw_markers(s, &now, 1.0);
    });
    let at = rect.lo + Point2::direction(rect.width() * 0.04, rect.height() * 0.12);
    c.text(STAGES[stage], at, 13.0, palette::INK, Align::Left);
}

/// The markers after each stage, in panels down `rect`.
fn stages_panel(c: &mut Canvas, rect: Rect) {
    let (source, target) = markers();
    let stages = placements(&source, &reconstruct(&source, &target));
    for (k, placed) in stages.iter().enumerate() {
        let panel = rect.part(0.0, k as f32 / 3.0, 1.0, (k + 1) as f32 / 3.0);
        panel3(c, panel, camera(panel, 9.5), |s| {
            draw_target(s, &target);
            draw_markers(s, placed, 1.0);
        });
        let at = panel.lo + Point2::direction(4.0, 14.0);
        c.text(STAGES[k], at, 9.0, palette::INK.faded(0.7), Align::Left);
    }
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    alignment_panel(c, screen.part(0.0, 0.08, 0.7, 1.0), t);
    stages_panel(c, screen.part(0.7, 0.14, 1.0, 1.0));
    caption(
        c,
        "THREE-POINT POSE: A POINT, A LINE, A PLANE",
        "EACH MOTOR IS THE SQUARE ROOT OF A RATIO, AND KEEPS WHAT THE LAST ONE MATCHED (PGA3D)",
    );
}

fn main() {
    run(Anim::new("three_point_pose", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::pose::*;
    use gax::ApproxEq;
    use gax::pga3d::{Line, Point};

    /// The largest coordinate difference between two lists of points, at unit weight.
    fn gap(a: &[Point], b: &[Point]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(p, q)| p.unitized().max_abs_diff(&q.unitized()))
            .fold(0.0, f64::max)
    }

    /// numga's batch: seventeen poses; along each alignment the first point, and then the
    /// whole first edge, stay matched, and the last stage lands on the target.
    #[test]
    fn batched_marker_registration_and_intermediate_constraints() {
        let (source, _) = markers();
        let fractions: Vec<f64> = (1..=8).map(|k| k as f64 / 8.0).collect();
        for k in 0..17 {
            let angle = -0.9 + 1.8 * k as f64 / 16.0;
            // numga's `exp(xw a + zw a / 3) exp(xy a + yz a / 2)`.
            let pose = Line::new(0.0, 0.0, 0.0, -angle, 0.0, -angle / 3.0).exp()
                * Line::new(angle / 2.0, 0.0, angle, 0.0, 0.0, 0.0).exp();
            let target = moved(pose, &source);
            let increments = reconstruct(&source, &target);
            let stages = placements(&source, &increments);
            let starts = [source, stages[0], stages[1]];
            let [translating, turning, rolling]: [Vec<Markers>; 3] = core::array::from_fn(|i| {
                let path = fractions.iter();
                path.map(|f| partway(&starts[i], increments[i], *f))
                    .collect()
            });
            let end = |path: &[Markers]| *path.last().expect("eight fractions");
            assert!(gap(&end(&translating)[..1], &target[..1]) < 1e-10);
            for placed in &turning {
                assert!(gap(&placed[..1], &target[..1]) < 1e-10);
            }
            for placed in &rolling {
                assert!(gap(&placed[..2], &target[..2]) < 1e-10);
            }
            assert!(gap(&end(&rolling), &target) < 1e-10, "{k}");
        }
    }

    /// Moving the world moves the reconstruction with it.
    #[test]
    fn reconstruction_is_independent_of_the_world_frame() {
        let (source, target) = markers();
        // numga's `exp(0.8 yw) exp(-0.7 xz)`, with `yw = -e02` and `xz = -e31`.
        let frame = Line::new(0.0, 0.0, 0.0, 0.0, -0.8, 0.0).exp()
            * Line::new(0.0, 0.7, 0.0, 0.0, 0.0, 0.0).exp();
        let (moved_source, moved_target) = (moved(frame, &source), moved(frame, &target));
        let stages = placements(&source, &reconstruct(&source, &target));
        let moved_stages = placements(&moved_source, &reconstruct(&moved_source, &moved_target));
        for (placed, moved_placed) in stages.iter().zip(&moved_stages) {
            assert!(gap(moved_placed, &moved(frame, placed)) < 1e-12);
        }
    }

    /// The scene's markers after each stage, as numga places them.
    #[test]
    fn the_stages_agree_with_numga() {
        let (source, target) = markers();
        let numga_target = [
            Point::xyz(0.741322578201426, 1.003261912439876, 0.752892473932858),
            Point::xyz(1.462527508141459, -0.504522748433966, 1.07915749728618),
            Point::xyz(2.054341534820859, 0.972613548077326, 0.681886155059714),
        ];
        assert!(gap(&target, &numga_target) < 1e-14);
        let stages = placements(&source, &reconstruct(&source, &target));
        let translated = [
            Point::xyz(0.741322578201426, 1.003261912439876, 0.752892473932858),
            Point::xyz(2.441322578201426, 1.103261912439876, 0.752892473932858),
            Point::xyz(1.241322578201426, 2.203261912439876, 0.952892473932858),
        ];
        let turned = Point::xyz(1.975624477618484, 1.048175803783343, 1.205088851030055);
        assert!(gap(&stages[0], &translated) < 1e-14);
        assert!(gap(&stages[1][2..], &[turned]) < 1e-14);
        assert!(gap(&stages[2], &numga_target) < 1e-14);
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 3.0);
    }
}
