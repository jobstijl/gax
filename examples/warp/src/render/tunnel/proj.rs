//! The projection: a camera motor, a pinhole map between algebras, fog, near-plane clipping,
//! and the singularities' lensing, all on points.

use super::super::LineInstance;
use super::super::scene;
use crate::light::Light;
use crate::sim::body::identity;
use crate::tunnel::track::{Frame, Track};
use crate::tunnel::{P, arc, dir};
use gax::pga2d::Point as Point2;
use gax::pga3d::{Motor, Plane, Point};

/// How far the fog lets objects be seen.
pub(super) const FOG: f32 = 90.0;
/// The wall fades sooner: its lines crowd towards the vanishing point, where enemies come
/// from, and must not glare there.
const WALL_FOG: f32 = 58.0;
/// The near plane's distance from the eye.
pub(super) const NEAR: f32 = 0.25;
/// The camera's image plane, through its eye, facing along its view (`z = 0` in its frame).
const IMAGE_PLANE: Plane<(), f32> = Plane::new(0.0, 0.0, 1.0, 0.0);
/// The near plane, `z = NEAR` in the camera's frame.
const NEAR_PLANE: Plane<(), f32> = Plane::new(0.0, 0.0, 1.0, -NEAR);
/// A frame's cross-section of the tunnel: its plane `z = 0`.
pub(super) const SECTION: Plane<(), f32> = Plane::new(0.0, 0.0, 1.0, 0.0);

/// A map from points (straightened, or in an object's own frame) to camera-frame points.
pub(super) type CamMap = Point<(Point,), f32>;

/// Which fog a segment sees.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Fog {
    /// The wall and the dust: short, and fading right at the camera too.
    Wall,
    /// Everything else: seen from further.
    Objects,
}

/// A singularity as the camera sees it: its screen point, its depth, and its lensing
/// strength (the Einstein radius² of what lies far behind it).
#[derive(Clone, Copy, Debug)]
pub(super) struct Lens {
    pub(super) centre: Point2<(), f32>,
    pub(super) depth: f32,
    pub(super) k: f32,
}

/// A camera-frame point as it is seen: its depth, and where it lands on screen (lensed).
#[derive(Clone, Copy, Debug)]
pub(super) struct Seen {
    pub(super) depth: f32,
    pub(super) at: Point2<(), f32>,
}

/// The projection: a camera motor, a focal length, and the lenses in view.
pub(super) struct Proj {
    pub(super) cam: Frame,
    pub(super) focal: f32,
    pub(super) lenses: Vec<Lens>,
    /// The pinhole: camera-frame points to screen points, with the depth as the weight.
    pinhole: Point2<(Point,), f32>,
}

impl Proj {
    /// The projection through the camera `cam`, without lenses.
    pub(super) fn new(cam: Frame, focal: f32) -> Self {
        // The images of `x`, `y`, `z` and the weight: the camera looks along `+z`, with `x`
        // mirrored onto the screen, and the depth becomes the weight (the perspective divide).
        let pinhole = Point2::<(Point,), f32>::from_images([
            Point2::new(-focal, 0.0, 0.0),
            Point2::new(0.0, focal, 0.0),
            Point2::new(0.0, 0.0, 1.0),
            Point2::new(0.0, 0.0, 0.0),
        ]);
        Proj {
            cam,
            focal,
            lenses: Vec::new(),
            pinhole,
        }
    }

    /// The camera map of a placement: `cam << (placement >> x)`, one 4x4 built from the motor
    /// `~cam placement` and applied to every point placed by it.
    pub(super) fn to_camera(&self, placement: Frame) -> CamMap {
        (self.cam.reverse() * placement) >> Point::slot()
    }

    /// The camera map for straightened points near `q`, placed rigidly with the frame at its
    /// arc length.
    pub(super) fn at(&self, track: &Track, q: P) -> CamMap {
        self.to_camera(track.placement(arc(q)))
    }

    /// The camera map of an object at `q` (straightened) turned by `turn`: from the object's
    /// own frame, through the track's frame at `q`, into the camera.
    pub(super) fn object(&self, track: &Track, q: P, turn: Frame) -> CamMap {
        let [x, y, s] = q.to_euclidean();
        self.to_camera(track.frame(s) * Motor::translation(x, y, 0.0) * turn)
    }

    /// A world point in the camera's frame (`cam << p`), or `None` behind the near plane.
    pub(super) fn camera(&self, p: P) -> Option<P> {
        let c = self.cam << p;
        (self.depth(c) > NEAR).then_some(c)
    }

    /// The depth of a camera-frame point (of weight 1): its signed distance from the camera's
    /// image plane, `plane & point`.
    pub(super) fn depth(&self, c: P) -> f32 {
        (IMAGE_PLANE & c).s()
    }

    /// The screen point of a camera-frame point: the PGA2D point with the depth as its
    /// weight (the perspective divide is normalizing it). The camera looks along `+z` with
    /// `+y` up, so its `+x` is on the left.
    pub(super) fn screen(&self, c: P) -> Point2<(), f32> {
        self.pinhole.of(c).unitized()
    }

    /// Where a camera-frame point is seen: its screen point, lensed by every singularity in
    /// front of it. For a point mass the Einstein radius² of a source at depth `z` behind a
    /// lens at `z_l` is `k (z - z_l) / z`, and a source at `β` from the lens is seen at the
    /// outer root of `θ² - βθ - θ_E² = 0`, `θ = (β + |(β, 2θ_E)|) / 2`: a scaling about the
    /// lens by `θ / β` (the traced `scale_about`).
    pub(super) fn lensed(&self, c: P) -> Point2<(), f32> {
        let mut q = self.screen(c);
        let z = self.depth(c);
        for l in &self.lenses {
            if z <= l.depth {
                continue;
            }
            let e = gax::Real::sqrt((l.k * (z - l.depth) / z).min(100.0));
            let beta = (q & l.centre).norm().max(1e-3);
            let theta = 0.5 * (beta + Point2::direction(beta, 2.0 * e).ideal_norm());
            q = crate::scale_about(q, l.centre, theta / beta);
        }
        q
    }

    /// A camera-frame point's depth and lensed screen point.
    pub(super) fn seen(&self, c: P) -> Seen {
        Seen {
            depth: self.depth(c),
            at: self.lensed(c),
        }
    }

    /// The world ray from the eye through a screen point: the camera-frame direction whose
    /// projection it is, taken into the world, joined with the eye (so intrinsic).
    pub(super) fn ray(&self, at: Point2<(), f32>) -> gax::pga3d::intrinsic::Line<(), f32> {
        let [x, y] = at.to_euclidean();
        let d = dir(-x / self.focal, y / self.focal, 1.0);
        let eye = self.cam >> Point::xyz(0.0, 0.0, 0.0);
        eye & (self.cam >> d)
    }

    /// A segment between two seen points: width and glow by depth, faded by fog.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn segment(
        &self,
        out: &mut Vec<LineInstance>,
        a: Seen,
        b: Seen,
        color: Light,
        width: f32,
        glow: f32,
        fog: Fog,
    ) {
        let z = 0.5 * (a.depth + b.depth);
        let k = match fog {
            Fog::Wall => {
                let far = (1.0 - z / WALL_FOG).clamp(0.0, 1.0);
                // Right at the camera the wall fades too, so it never floods the screen.
                let near = ((z - 0.8) / 4.0).clamp(0.0, 1.0);
                far * far * far * near
            }
            Fog::Objects => (1.0 - (z - 45.0) / (FOG - 45.0)).clamp(0.0, 1.0),
        };
        if k <= 0.0 {
            return;
        }
        let scale = self.focal / z;
        let (wmin, wmax) = if fog == Fog::Wall {
            (0.01, 0.1)
        } else {
            (0.035, 0.3)
        };
        let style = [
            (width * scale).clamp(wmin, wmax),
            (glow * scale).clamp(0.08, 0.6),
            0.3,
            0.0,
        ];
        out.push(scene::seg(a.at, b.at, color.faded(k), style, identity()));
    }

    /// A segment between two world points, cut at the near plane by its meet with it.
    pub(super) fn line(
        &self,
        out: &mut Vec<LineInstance>,
        a: P,
        b: P,
        color: Light,
        width: f32,
        fog: Fog,
    ) {
        self.cam_line(out, self.cam << a, self.cam << b, color, width, fog);
    }

    /// A short segment from `a` to `b` (straightened), both seen through `view`, the camera
    /// map near them: streaks, whose ends are too close for the track to bend between them.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn streak(
        &self,
        out: &mut Vec<LineInstance>,
        view: &CamMap,
        a: P,
        b: P,
        color: Light,
        width: f32,
        fog: Fog,
    ) {
        self.cam_line(out, view.of(a), view.of(b), color, width, fog);
    }

    /// A segment between two camera-frame points, cut at the near plane by its meet with it.
    pub(super) fn cam_line(
        &self,
        out: &mut Vec<LineInstance>,
        ca: P,
        cb: P,
        color: Light,
        width: f32,
        fog: Fog,
    ) {
        let (fa, fb) = (self.depth(ca) > NEAR, self.depth(cb) > NEAR);
        let cut = || ((ca & cb) ^ NEAR_PLANE).unitized();
        let (ca, cb) = match (fa, fb) {
            (true, true) => (ca, cb),
            (false, false) => return,
            (true, false) => (ca, cut()),
            (false, true) => (cut(), cb),
        };
        let (a, b) = (self.seen(ca), self.seen(cb));
        self.segment(out, a, b, color, width, width * 5.0, fog);
    }

    /// A wireframe solid seen through its camera map `model` (see [`Proj::object`]), scaled
    /// by `size`: each vertex is a direction from the centre.
    pub(super) fn solid(
        &self,
        out: &mut Vec<LineInstance>,
        model: &CamMap,
        edges: &[(P, P)],
        size: f32,
        color: Light,
    ) {
        let o = Point::xyz(0.0, 0.0, 0.0);
        for &(a, b) in edges {
            let (a, b) = (model.of(o + a * size), model.of(o + b * size));
            self.cam_line(out, a, b, color, 0.05, Fog::Objects);
        }
    }

    /// A ring across the tunnel (perpendicular to the track) at `q`: chords of a radius
    /// turned about the line through `q` along the track, seen through `view`, the camera map
    /// at `q` (every point of the ring is at its arc length).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ring(
        &self,
        out: &mut Vec<LineInstance>,
        view: &CamMap,
        q: P,
        r: f32,
        color: Light,
        n: usize,
        width: f32,
    ) {
        let spin = Motor::rotation(q & dir(0.0, 0.0, 1.0), core::f32::consts::TAU / n as f32);
        let mut a = q + dir(r, 0.0, 0.0);
        for _ in 0..n {
            let b = spin >> a;
            self.cam_line(out, view.of(a), view.of(b), color, width, Fog::Objects);
            a = b;
        }
    }
}

/// The cosine of the angle between two directions: the inner product of the planes
/// orthogonal to them, over their norms.
pub(super) fn cosine(a: P, b: P) -> f32 {
    let (pa, pb) = (Plane::orthogonal_to(a), Plane::orthogonal_to(b));
    (pa | pb).s() / (pa.norm() * pb.norm()).max(1e-9)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::scene::palette;
    use crate::tunnel::at;

    #[test]
    fn a_projection_is_a_weight() {
        let p = Proj::new(Motor::translation(0.0, 0.0, 0.0), 10.0);
        let s = p
            .screen(p.camera(at(2.0, 1.0, 4.0)).unwrap())
            .to_euclidean();
        assert!((s[0] + 5.0).abs() < 1e-5 && (s[1] - 2.5).abs() < 1e-5);
        // A segment across the near plane is cut where it meets it.
        let mut out = Vec::new();
        p.line(
            &mut out,
            at(0.0, 0.0, -1.0),
            at(0.0, 1.0, 3.0),
            palette::HUD,
            0.05,
            Fog::Objects,
        );
        assert_eq!(out.len(), 1);
    }

    /// A point behind a lens at `β` from it on screen is seen at `θ` on the same ray from the
    /// lens, the outer root of `θ² - βθ - θ_E² = 0`; a point in front of the lens is not moved.
    #[test]
    fn a_singularity_lenses_what_lies_behind_it() {
        let mut p = Proj::new(Motor::translation(0.0, 0.0, 0.0), 10.0);
        let centre = Point2::xy(1.0, -0.5);
        let lens = Lens {
            centre,
            depth: 10.0,
            k: 3.0,
        };
        p.lenses.push(lens);
        for (x, y, z) in [(0.8, 0.3, 25.0), (-1.5, 2.0, 40.0), (3.0, -1.0, 12.0)] {
            let c = p.camera(at(x, y, z)).unwrap();
            let (q, seen) = (p.screen(c), p.lensed(c));
            let beta = (q & centre).norm();
            let theta = (seen & centre).norm();
            let e2 = lens.k * (z - lens.depth) / z;
            let residual = theta * theta - beta * theta - e2;
            assert!(residual.abs() < 1e-3 * (1.0 + e2), "{residual}");
            assert!(theta > beta, "pushed out");
            // On the ray from the lens through the unlensed point: the join of the three
            // points, over the line's weight, is the distance from the line.
            assert!(((centre & q) & seen).norm() / beta < 1e-3 * theta);
        }
        let c = p.camera(at(0.8, 0.3, 6.0)).unwrap();
        assert_eq!(p.lensed(c), p.screen(c));
    }
}
