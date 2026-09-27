//! The track: a chain of screw motions. The frame at arc length `s` is
//! `M(s) = K_i exp((s - s_i) T_i)`, where each segment's twist `T_i` (a PGA3D line) is a unit
//! forward translation plus small rotations: pitch, yaw and roll. The keyframes `K_i` are the
//! running product, and between them `M(s)` is motor interpolation, which for a screw is exact.
//!
//! Gameplay lives in *straightened* coordinates `(x, y, s)`: across the tunnel and along it.
//! `M(s) >> (x, y, 0)` is the world point. Forward is the frame's `+z`.

use crate::sim::rng::Rng;
use gax::Unit;
use gax::pga3d::{Line, Motor, Point};

/// A rigid frame in 3D.
pub type Frame = Unit<Motor<(), f32>>;

/// Arc length of a segment.
pub const SEG: f32 = 10.0;

/// The largest curvature (radians per unit of arc length): a turn radius of at least 28, four
/// times the tunnel's radius.
const MAX_BEND: f32 = 0.036;
/// The largest roll rate.
const MAX_ROLL: f32 = 0.025;

/// `a exp(t log(~a b))`: the screw motion from `a` to `b`, at `t`.
pub fn interpolate(a: Frame, b: Frame, t: f32) -> Frame {
    let rel = (a.reverse() * b).log();
    (a * rel.gp(t).exp()).renormalize_fast()
}

/// The procedurally generated track, extended ahead on demand.
pub struct Track {
    rng: Rng,
    /// Keyframes from segment `first` on.
    keys: Vec<Frame>,
    first: usize,
    /// The current bend and roll rates (a bounded random walk).
    pitch: f32,
    yaw: f32,
    roll: f32,
}

impl Track {
    /// A track from a seed, starting at the origin heading along `+z`.
    pub fn new(seed: u64) -> Track {
        let mut t = Track {
            rng: Rng::new(seed ^ 0x7ac0),
            keys: vec![Motor::translation(0.0, 0.0, 0.0)],
            first: 0,
            pitch: 0.0,
            yaw: 0.0,
            roll: 0.0,
        };
        // A straight run-in.
        for _ in 0..3 {
            t.push(0.0, 0.0, 0.0);
        }
        t
    }

    /// The twist of a segment: forward at unit speed, with the given rotation rates about the
    /// frame's own axes (the body frame, so it multiplies on the right).
    pub fn twist(pitch: f32, yaw: f32, roll: f32) -> Line<(), f32> {
        let o = Point::xyz(0.0, 0.0, 0.0);
        Line::translation_twist(0.0, 0.0, 1.0)
            + Line::rotation_twist(o & Point::direction(1.0, 0.0, 0.0), pitch)
            + Line::rotation_twist(o & Point::direction(0.0, 1.0, 0.0), yaw)
            + Line::rotation_twist(o & Point::direction(0.0, 0.0, 1.0), roll)
    }

    fn push(&mut self, pitch: f32, yaw: f32, roll: f32) {
        let last = *self.keys.last().expect("keys");
        // Keep the heading near level, so that a camera without roll stays comfortable: the
        // world's up in the frame's own coordinates says how much it climbs (`uz`), and a
        // rotation about `forward × up` (in the frame) turns it back.
        let up = last << Point::direction(0.0, 1.0, 0.0);
        let (ux, uy, uz) = (up.e032(), up.e013(), up.e021());
        let across = (ux * ux + uy * uy).sqrt().max(1e-3);
        let o = Point::xyz(0.0, 0.0, 0.0);
        let level = Line::rotation_twist(o & Point::direction(-uy, ux, 0.0), -0.05 * uz / across);
        let step = (Track::twist(pitch, yaw, roll) + level).gp(SEG).exp();
        self.keys.push((last * step).renormalize_fast());
    }

    fn grow(&mut self) {
        let r = &mut self.rng;
        // A random walk on the rates, pulled back towards zero.
        self.yaw = (0.8 * self.yaw + r.range(-0.018, 0.018)).clamp(-MAX_BEND, MAX_BEND);
        self.pitch = (0.7 * self.pitch + r.range(-0.012, 0.012)).clamp(-MAX_BEND, MAX_BEND);
        self.roll = (0.85 * self.roll + r.range(-0.01, 0.01)).clamp(-MAX_ROLL, MAX_ROLL);
        self.push(self.pitch, self.yaw, self.roll);
    }

    /// Make sure the track reaches arc length `s`, and forget what lies before `behind`.
    pub fn extend(&mut self, s: f32, behind: f32) {
        while ((self.first + self.keys.len() - 1) as f32) * SEG < s + SEG {
            self.grow();
        }
        let drop = ((behind / SEG).floor().max(0.0) as usize).saturating_sub(self.first);
        let drop = drop.min(self.keys.len().saturating_sub(2));
        if drop > 0 {
            self.keys.drain(..drop);
            self.first += drop;
        }
    }

    /// The frame at arc length `s` (clamped to the generated range).
    pub fn frame(&self, s: f32) -> Frame {
        let u = (s / SEG).max(self.first as f32);
        let i = ((u.floor() as usize).max(self.first) - self.first).min(self.keys.len() - 2);
        let t = (u - (self.first + i) as f32).clamp(0.0, 1.0);
        interpolate(self.keys[i], self.keys[i + 1], t)
    }

    /// The world point at straightened coordinates `(x, y, s)`.
    pub fn point(&self, x: f32, y: f32, s: f32) -> [f32; 3] {
        (self.frame(s) >> Point::xyz(x, y, 0.0)).to_euclidean()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    /// The axis is parametrized by arc length: points `ds` apart are `ds` apart in space (for
    /// small `ds`), the frame is continuous across segment joins, and the heading stays near
    /// level.
    #[test]
    fn the_track_is_smooth_arc_length_and_level() {
        let mut t = Track::new(5);
        t.extend(2000.0, 0.0);
        let ds = 0.05;
        let mut s = 0.0;
        while s < 2000.0 {
            let a = t.point(0.0, 0.0, s);
            let b = t.point(0.0, 0.0, s + ds);
            let d = dist(a, b);
            assert!((d - ds).abs() < 2e-3, "at {s}: {d}");
            // The forward axis is the direction of travel.
            let f = t.frame(s) >> Point::direction(0.0, 0.0, 1.0);
            let dir = [(b[0] - a[0]) / d, (b[1] - a[1]) / d, (b[2] - a[2]) / d];
            let dot = f.e032() * dir[0] + f.e013() * dir[1] + f.e021() * dir[2];
            assert!(dot > 0.999, "at {s}: {dot}");
            // Never steep: the heading's vertical part stays small.
            assert!(f.e013().abs() < 0.6, "at {s}: climbing {}", f.e013());
            s += 3.7;
        }
    }

    #[test]
    fn a_body_twist_rotates_about_the_frames_own_axes() {
        // Yaw only: the path curves in the frame's x-z plane and the y axis stays up.
        let m = Track::twist(0.0, 0.1, 0.0).gp(5.0).exp();
        let up = (m >> Point::direction(0.0, 1.0, 0.0)).e013();
        assert!((up - 1.0).abs() < 1e-5);
        let p = (m >> Point::xyz(0.0, 0.0, 0.0)).to_euclidean();
        assert!(p[1].abs() < 1e-5 && p[0] > 0.1 && p[2] > 4.0, "{p:?}");
    }

    #[test]
    fn forgetting_the_past_keeps_the_future() {
        let mut t = Track::new(9);
        t.extend(500.0, 0.0);
        let at = t.point(1.0, -2.0, 420.0);
        t.extend(600.0, 400.0);
        assert!(dist(at, t.point(1.0, -2.0, 420.0)) < 1e-4);
    }
}
