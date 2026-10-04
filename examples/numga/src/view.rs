//! From world coordinates to pixels: a 2D view (a rectangle of the plane) and a 3D camera whose
//! pose is a PGA3D motor (`Motor::look_at`), with perspective or parallel projection, and the
//! ray through each pixel for the ray-traced examples.

use crate::canvas::Px;
use crate::coords::{Point2, Pos2, Pos3};
use gax::Unit;
use gax::pga3d::{Motor, Point};

/// A rectangle of the plane on the canvas: `centre` in the middle, `half_height` units from the
/// middle to the top edge, y up.
#[derive(Clone, Copy, Debug)]
pub struct View2 {
    centre: Point2,
    /// Pixels per world unit.
    pub scale: f32,
    /// Width and height in pixels.
    size: Px,
}

impl View2 {
    /// The view of a `width` x `height` canvas.
    pub fn new(width: usize, height: usize, centre: impl Pos2, half_height: f32) -> View2 {
        View2 {
            centre: centre.point2().unitized(),
            scale: height as f32 * 0.5 / half_height,
            size: [width as f32, height as f32],
        }
    }

    /// The pixel of a world point.
    pub fn px(&self, p: impl Pos2) -> Px {
        // Its displacement from the centre, in pixels (y down).
        let d = p.point2().unitized() - self.centre;
        [
            self.size[0] * 0.5 + d.e20() * self.scale,
            self.size[1] * 0.5 - d.e01() * self.scale,
        ]
    }

    /// The world point at pixel `q`.
    pub fn world(&self, q: Px) -> Point2 {
        let (dx, dy) = (q[0] - self.size[0] * 0.5, self.size[1] * 0.5 - q[1]);
        self.centre + Point2::direction(dx, dy).gp(self.scale.recip())
    }

    /// The world's half width.
    pub fn half_width(&self) -> f32 {
        self.size[0] * 0.5 / self.scale
    }
}

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
    /// The viewport's top left corner on the canvas, in pixels.
    origin: Px,
    /// Its width and height in pixels.
    size: Px,
}

impl Camera {
    /// A camera at `eye` looking at `target` with `+z` up in the world.
    pub fn looking(
        width: usize,
        height: usize,
        eye: impl Pos3,
        target: impl Pos3,
        lens: Lens,
    ) -> Camera {
        Camera {
            pose: Motor::look_at(
                eye.point3(),
                target.point3(),
                Point::direction(0.0, 0.0, 1.0),
            ),
            lens,
            origin: [0.0, 0.0],
            size: [width as f32, height as f32],
        }
    }

    /// A camera on a sphere about `target`: `distance` away, at `azimuth` about the world's `z`
    /// axis (from `+x`) and `elevation` above the horizontal, both in radians.
    pub fn orbit(
        width: usize,
        height: usize,
        target: impl Pos3,
        distance: f32,
        azimuth: f32,
        elevation: f32,
        lens: Lens,
    ) -> Camera {
        let target = target.point3().unitized();
        // The eye: `distance` along x, raised by the elevation, then turned by the azimuth
        // (rotations about the y and z axes), from the target.
        let turn = Motor::rotation_about(0.0, 0.0, 1.0, azimuth)
            * Motor::rotation_about(0.0, -1.0, 0.0, elevation);
        let eye = target + (turn >> Point::direction(distance, 0.0, 0.0));
        Camera::looking(width, height, eye, target, lens)
    }

    /// A parallel camera orbiting the origin (see [`Camera::orbit`]) that draws into the canvas
    /// rectangle `view` at `scale` pixels per world unit.
    pub fn parallel(view: [f32; 4], scale: f32, azimuth: f32, elevation: f32) -> Camera {
        let height = view[3] - view[1];
        Camera::orbit(
            (view[2] - view[0]) as usize,
            height as usize,
            Point::xyz(0.0, 0.0, 0.0),
            20.0,
            azimuth,
            elevation,
            Lens::Parallel(0.5 * height / scale),
        )
        .viewport(view)
    }

    /// The same camera drawing into the pixel rectangle `[x0, y0, x1, y1]` of a larger canvas (a
    /// panel): its view fills the rectangle, and [`Camera::px`] and [`Camera::ray`] work in the
    /// canvas's pixels. Pair it with `canvas.clip(rect)` to keep the drawing inside.
    pub fn viewport(self, rect: [f32; 4]) -> Camera {
        Camera {
            origin: [rect[0], rect[1]],
            size: [rect[2] - rect[0], rect[3] - rect[1]],
            ..self
        }
    }

    /// Pixels per unit at unit depth (perspective) or per world unit (parallel).
    fn focal(&self) -> f32 {
        match self.lens {
            Lens::Perspective(fov) => {
                // The edge of the view: the direction along x turned by half the field of view;
                // its height over its width is the tangent.
                let origin = gax::pga2d::Point::xy(0.0, 0.0);
                let edge = gax::pga2d::Motor::rotation(origin, fov * 0.5)
                    >> gax::pga2d::Point::direction(1.0, 0.0);
                self.size[1] * 0.5 * edge.e20() / edge.e01()
            }
            Lens::Parallel(half) => self.size[1] * 0.5 / half,
        }
    }

    /// A world point in the camera's frame, which looks along `+z` with `+y` up (so `+x` is to
    /// the left on screen).
    pub fn local(&self, p: impl Pos3) -> Point {
        self.pose.reverse() >> p.point3()
    }

    /// How far in front of the camera a world point is.
    pub fn depth(&self, p: impl Pos3) -> f32 {
        self.local(p).unitized().e021()
    }

    /// The pixel of a world point, or `None` behind the camera or at infinity.
    pub fn px(&self, p: impl Pos3) -> Option<Px> {
        let local = self.local(p);
        if local.e123().abs() < 1e-12 {
            return None;
        }
        let [x, y, z] = local.to_euclidean();
        // Screen x runs against the frame's x.
        let x = -x;
        let f = self.focal();
        let (sx, sy) = match self.lens {
            Lens::Perspective(_) if z <= 1e-3 => return None,
            Lens::Perspective(_) => (x / z, y / z),
            Lens::Parallel(_) => (x, y),
        };
        Some([
            self.origin[0] + self.size[0] * 0.5 + sx * f,
            self.origin[1] + self.size[1] * 0.5 - sy * f,
        ])
    }

    /// The ray through pixel `q`: its origin (a point) and its direction (a unit ideal point).
    pub fn ray(&self, q: Px) -> (Point<(), f32>, Point<(), f32>) {
        let f = self.focal();
        let (sx, sy) = (
            (q[0] - self.origin[0] - self.size[0] * 0.5) / f,
            (self.origin[1] + self.size[1] * 0.5 - q[1]) / f,
        );
        match self.lens {
            Lens::Perspective(_) => {
                let d = Point::direction(-sx, sy, 1.0);
                let dir = self.pose >> d.gp(d.ideal_norm().recip());
                (self.pose >> Point::xyz(0.0, 0.0, 0.0), dir)
            }
            Lens::Parallel(_) => (
                self.pose >> Point::xyz(-sx, sy, 0.0),
                self.pose >> Point::direction(0.0, 0.0, 1.0),
            ),
        }
    }

    /// The camera's position.
    pub fn eye(&self) -> Point<(), f32> {
        self.pose >> Point::xyz(0.0, 0.0, 0.0)
    }
}
