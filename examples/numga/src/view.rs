//! From the world to the canvas: a 3D camera whose pose is a PGA3D motor (`Motor::look_at`),
//! with perspective or parallel projection, ending in a map of the plane onto the canvas, and
//! the ray through each canvas point for the ray-traced examples. (2D views are
//! [`Axes`](crate::Axes).)

use crate::canvas::Rect;
use crate::points::{Map2, ORIGIN2, ORIGIN3, Point2, Point3, Pos3, box_map, from_above};
use gax::Unit;
use gax::pga3d::{Motor, Plane, Point};

/// How the camera projects.
#[derive(Clone, Copy, Debug)]
pub enum Lens {
    /// Perspective, with this vertical field of view in radians.
    Perspective(f32),
    /// Parallel, with this half height of the view in world units.
    Parallel(f32),
}

/// A camera: its pose (camera frame to world; it looks along its `+z`, `+y` up) and lens.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// Camera to world.
    pub pose: Unit<Motor<(), f32>>,
    /// The projection.
    pub lens: Lens,
    /// Where on the canvas it draws.
    viewport: Rect,
}

impl Camera {
    /// A camera at `eye` looking at `target` with `+z` up in the world, drawing into the canvas
    /// rectangle `viewport`.
    pub fn looking(viewport: Rect, eye: impl Pos3, target: impl Pos3, lens: Lens) -> Camera {
        Camera {
            pose: Motor::look_at(
                eye.point3(),
                target.point3(),
                Point::direction(0.0, 0.0, 1.0),
            ),
            lens,
            viewport,
        }
    }

    /// A camera on a sphere about `target`: `distance` away, at `azimuth` about the world's `z`
    /// axis (from `+x`) and `elevation` above the horizontal, both in radians.
    pub fn orbit(
        viewport: Rect,
        target: impl Pos3,
        distance: f32,
        azimuth: f32,
        elevation: f32,
        lens: Lens,
    ) -> Camera {
        let target = target.point3().unitized();
        // The eye: `distance` along x, raised by the elevation, then turned by the azimuth
        // (rotations about the y and z axes), from the target.
        let turn = Motor::rotation_about(Point::direction(0.0, 0.0, 1.0), azimuth)
            * Motor::rotation_about(Point::direction(0.0, -1.0, 0.0), elevation);
        let eye = target + (turn >> Point::direction(distance, 0.0, 0.0));
        Camera::looking(viewport, eye, target, lens)
    }

    /// A parallel camera orbiting the origin (see [`Camera::orbit`]) that draws into the canvas
    /// rectangle `view` at `scale` canvas units per world unit.
    pub fn parallel(view: Rect, scale: f32, azimuth: f32, elevation: f32) -> Camera {
        Camera::orbit(
            view,
            ORIGIN3,
            20.0,
            azimuth,
            elevation,
            Lens::Parallel(0.5 * view.height() / scale),
        )
    }

    /// The same camera drawing into `rect` of a larger canvas (a panel): its view fills the
    /// rectangle, and [`Camera::px`] and [`Camera::ray`] work in the canvas's units. Pair it
    /// with `canvas.clip(rect)` to keep the drawing inside.
    pub fn viewport(self, rect: Rect) -> Camera {
        Camera {
            viewport: rect,
            ..self
        }
    }

    /// Canvas units per unit of the image plane: at unit depth (perspective) or per world unit
    /// (parallel).
    fn focal(&self) -> f32 {
        let half = 0.5 * self.viewport.height();
        match self.lens {
            Lens::Perspective(fov) => {
                // The edge of the view: the direction at half the field of view; its height
                // over its width is the tangent.
                let edge = Point2::direction_from_angle(fov * 0.5);
                half * edge.e20() / edge.e01()
            }
            Lens::Parallel(half_height) => half / half_height,
        }
    }

    /// The image plane onto the viewport: its origin to the viewport's centre, `focal` pixels a
    /// unit, `y` up to down, and `x` mirrored (the frame looks along `+z` with `+y` up, so its
    /// `+x` is to the left on screen).
    fn image_to_px(&self) -> Map2 {
        let (c, f) = (self.viewport.centre(), self.focal());
        box_map(
            [ORIGIN2, Point2::xy(1.0, 1.0)],
            [c, c + Point2::direction(-f, -f)],
        )
    }

    /// A world point in the camera's frame.
    pub fn local(&self, p: impl Pos3) -> Point3 {
        self.pose.reverse() >> p.point3()
    }

    /// How far in front of the camera a world point is.
    pub fn depth(&self, p: impl Pos3) -> f32 {
        self.local(p).unitized().e021()
    }

    /// Where a camera-frame point lands on the image plane, seen along `z`: for perspective, the
    /// meet of its ray from the eye with the plane `z = 1`; for parallel, the point itself.
    /// `None` behind the camera or at infinity.
    fn image(&self, local: Point3) -> Option<Point2> {
        if local.e123().abs() < 1e-12 {
            return None;
        }
        match self.lens {
            Lens::Perspective(_) if local.unitized().e021() <= 1e-3 => None,
            Lens::Perspective(_) => {
                let screen = Plane::with_normal(Point::direction(0.0, 0.0, 1.0), 1.0);
                Some(from_above((ORIGIN3 & local) ^ screen))
            }
            Lens::Parallel(_) => Some(from_above(local)),
        }
    }

    /// The pixel of a world point, or `None` behind the camera or at infinity.
    pub fn px(&self, p: impl Pos3) -> Option<Point2> {
        self.image(self.local(p))
            .map(|i| self.image_to_px().of(i).unitized())
    }

    /// The ray through pixel `q`: its origin (a point) and its direction (a unit ideal point).
    pub fn ray(&self, q: Point2) -> (Point3, Point3) {
        let [x, y] = self.image_to_px().inverse().of(q).to_euclidean();
        match self.lens {
            Lens::Perspective(_) => {
                let d = Point::direction(x, y, 1.0);
                let dir = self.pose >> d.gp(d.ideal_norm().recip());
                (self.eye(), dir)
            }
            Lens::Parallel(_) => (
                self.pose >> Point::xyz(x, y, 0.0),
                self.pose >> Point::direction(0.0, 0.0, 1.0),
            ),
        }
    }

    /// The camera's position.
    pub fn eye(&self) -> Point3 {
        self.pose >> ORIGIN3
    }
}
