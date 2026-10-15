//! Aiming through the camera, backwards: a screen point is a ray from the eye, which meets
//! the tunnel's cross-section; `Track::straighten` takes the meet back into the simulation's
//! coordinates. Also where things are seen and heard.

use super::View;
use super::proj::{CamMap, SECTION};
use crate::tunnel::{AIM_DEPTH, LOCK_AHEAD, P, World, arc, dir};
use gax::pga2d::Point as Point2;
use gax::pga3d::Point;

impl View {
    /// Watching a replay: the reticle over the replay's aim point, locked onto the enemy
    /// there.
    pub fn follow(&mut self, w: &World, aim: P) {
        if let Some(q) = self.on_screen(w, aim) {
            self.reticle = q;
        }
        self.lock = w
            .enemies
            .iter()
            .find(|e| (e.pos & aim).norm() < 0.5)
            .map(|e| e.id);
    }

    /// The reticle at a screen point: lock onto the enemy under it (the nearest on screen
    /// within reach), or aim where the ray through it meets the cross-section at
    /// `AIM_DEPTH`. Returns the aim point for the simulation.
    pub fn aim(&mut self, w: &World, reticle: Point2<(), f32>) -> P {
        self.reticle = reticle;
        let p = &self.proj;
        let s = w.ship.s();
        let lock = w
            .enemies
            .iter()
            .filter(|e| arc(e.pos) > s + LOCK_AHEAD)
            .filter_map(|e| {
                let c = p.camera(w.track.place(e.pos))?;
                // Where it is seen (a singularity's lens moves what lies behind it).
                let on_screen = p.lensed(c);
                // Within reach on screen: a margin plus the enemy's own apparent size.
                let reach = self.settings.assist + e.radius() * p.focal / p.depth(c);
                let d = (on_screen & reticle).norm();
                (d < reach).then_some((e, d))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1));
        self.lock = lock.map(|(e, _)| e.id);
        match lock {
            Some((e, _)) => e.pos,
            None => self.through(w, reticle, s + AIM_DEPTH),
        }
    }

    /// Where the ray from the eye through the screen point `at` meets the tunnel's
    /// cross-section at arc length `s`, in straightened coordinates.
    pub fn through(&self, w: &World, at: Point2<(), f32>, s: f32) -> P {
        let ray = self.proj.ray(at);
        let hit = ray ^ (w.track.frame(s) >> SECTION);
        w.track.straighten(hit.unitized(), s)
    }

    /// A direction on screen (movement, length up to 1) as a direction across the tunnel at
    /// the ship: how the ship's cross-section is met by the ray through its screen point, as
    /// that point moves along `v`.
    ///
    /// In the frame of the cross-section (`z = 0` there, `x` and `y` straightened), the ray
    /// along a camera-frame direction meets it at `hit.of(d)`: a projective map. The screen
    /// point `(x, y)` is the direction `(-x/f, y/f, 1)`, so moving it at `v` moves the hit
    /// as `(a + t b) / (w_a + t w_b)`, whose direction at the ship is `b w_a - a w_b`: the
    /// exact differential, from two applications of one map.
    pub fn across(&self, w: &World, v: Point2<(), f32>) -> P {
        let p = &self.proj;
        let Some(c) = p.camera(w.track.place(w.ship.pos)) else {
            return dir(0.0, 0.0, 0.0);
        };
        let cam = w.track.frame(w.ship.s()).reverse() * p.cam;
        let eye = cam >> Point::xyz(0.0, 0.0, 0.0);
        let hit: CamMap = (eye & (cam >> Point::slot())) ^ SECTION;
        let z = c.e021();
        let (a, b) = (
            hit.of(dir(c.e032() / z, c.e013() / z, 1.0)),
            hit.of(dir(-v.e20() / p.focal, v.e01() / p.focal, 0.0)),
        );
        let d = b * a.e123() - a * b.e123();
        let d = dir(d.e032(), d.e013(), 0.0);
        let n = d.ideal_norm();
        if n < 1e-6 {
            dir(0.0, 0.0, 0.0)
        } else {
            d * (v.ideal_norm().min(1.0) / n)
        }
    }

    /// Where a world point is for the ears: in the camera's frame (`cam << p`), to the right
    /// as `x` and ahead as `y` (the camera's `+x` is on the left).
    pub fn listen(&self, p: P) -> Point2<(), f32> {
        let c = (self.cam << p).unitized();
        Point2::xy(-c.e032(), c.e021())
    }

    /// Where a straightened point is seen on screen (lensed), if in front of the camera.
    pub fn on_screen(&self, w: &World, q: P) -> Option<Point2<(), f32>> {
        let p = &self.proj;
        p.camera(w.track.place(q)).map(|c| p.lensed(c))
    }

    /// The screen point of the ship (for the pad's reticle, which sits ahead of it).
    pub fn ship_on_screen(&self, w: &World) -> Point2<(), f32> {
        let p = &self.proj;
        p.camera(w.track.place(w.ship.pos))
            .map_or(Point2::xy(0.0, 0.0), |c| p.screen(c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::tunnel::tests::flown;
    use crate::tunnel::Foe;

    /// Input to the right on screen moves the ship to the right on screen, however the track
    /// has rolled; and the reticle's ray comes back as the point it was cast through.
    #[test]
    fn screen_directions_and_rays_map_through_the_camera() {
        let w = flown(8, 30);
        let mut v = View::new(85.0);
        v.update(&w, 0.0, 1.0);
        let d = v.across(&w, Point2::direction(1.0, 0.0));
        let p = &v.proj;
        let a = p.screen(p.camera(w.track.place(w.ship.pos)).unwrap());
        let b = p.screen(p.camera(w.track.place(w.ship.pos + d)).unwrap());
        let (a, b) = (a.to_euclidean(), b.to_euclidean());
        assert!(
            b[0] - a[0] > 0.1 && (b[1] - a[1]).abs() < 0.1 * (b[0] - a[0]),
            "{a:?} {b:?}"
        );
        // Cast through a screen point at the aim depth, and project back: the same point.
        let s = w.ship.s() + AIM_DEPTH;
        let at_screen = Point2::xy(3.0, -2.0);
        let q = v.through(&w, at_screen, s);
        assert!((arc(q) - s).abs() < 0.05, "{} vs {s}", arc(q));
        let back = p.screen(p.camera(w.track.place(q)).unwrap());
        assert!(
            (back & at_screen).norm() < 0.05,
            "{:?}",
            back.to_euclidean()
        );
    }

    /// The reticle locks onto the enemy under it, at any depth, and the aim is that enemy.
    #[test]
    fn the_reticle_locks_what_is_under_it() {
        let mut w = flown(6, 12);
        w.enemies.retain(|e| e.foe == Foe::Drone);
        let mut v = View::new(85.0);
        v.update(&w, 0.0, 1.0);
        let s = w.ship.s();
        let Some(e) = w.enemies.iter().find(|e| arc(e.pos) > s + 5.0).cloned() else {
            return;
        };
        let p = &v.proj;
        let on_screen = p.screen(p.camera(w.track.place(e.pos)).unwrap());
        let aim = v.aim(&w, on_screen + Point2::direction(0.08, -0.05));
        assert_eq!(v.lock, Some(e.id));
        assert!((aim & e.pos).norm() < 1e-4);
    }
}
