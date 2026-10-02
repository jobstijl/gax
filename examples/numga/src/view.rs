//! From world coordinates to pixels: a 2D view (a rectangle of the plane) and a 3D camera whose
//! pose is a PGA3D motor (`Motor::look_at`), with perspective or parallel projection, and the
//! ray through each pixel for the ray-traced examples.

use crate::canvas::Px;
use gax::Unit;
use gax::pga3d::{Motor, Point};

/// A rectangle of the plane on the canvas: `centre` in the middle, `half_height` units from the
/// middle to the top edge, y up.
#[derive(Clone, Copy, Debug)]
pub struct View2 {
    centre: [f32; 2],
    /// Pixels per world unit.
    pub scale: f32,
    size: [f32; 2],
}

impl View2 {
    /// The view of a `width` x `height` canvas.
    pub fn new(width: usize, height: usize, centre: [f32; 2], half_height: f32) -> View2 {
        View2 {
            centre,
            scale: height as f32 * 0.5 / half_height,
            size: [width as f32, height as f32],
        }
    }

    /// The pixel of world point `(x, y)`.
    pub fn px(&self, p: [f32; 2]) -> Px {
        [
            self.size[0] * 0.5 + (p[0] - self.centre[0]) * self.scale,
            self.size[1] * 0.5 - (p[1] - self.centre[1]) * self.scale,
        ]
    }

    /// The world point at pixel `q`.
    pub fn world(&self, q: Px) -> [f32; 2] {
        [
            self.centre[0] + (q[0] - self.size[0] * 0.5) / self.scale,
            self.centre[1] - (q[1] - self.size[1] * 0.5) / self.scale,
        ]
    }

    /// The pixel of a PGA2D point.
    pub fn point(&self, p: gax::pga2d::Point<(), f32>) -> Px {
        self.px(p.to_euclidean())
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
    size: [f32; 2],
}

impl Camera {
    /// A camera at `eye` looking at `target` with `+z` up in the world.
    pub fn looking(
        width: usize,
        height: usize,
        eye: [f32; 3],
        target: [f32; 3],
        lens: Lens,
    ) -> Camera {
        let p = |v: [f32; 3]| Point::xyz(v[0], v[1], v[2]);
        Camera {
            pose: Motor::look_at(p(eye), p(target), Point::direction(0.0, 0.0, 1.0)),
            lens,
            size: [width as f32, height as f32],
        }
    }

    /// A camera on a sphere about `target`: `distance` away, at `azimuth` about the world's `z`
    /// axis (from `+x`) and `elevation` above the horizontal, both in radians.
    pub fn orbit(
        width: usize,
        height: usize,
        target: [f32; 3],
        distance: f32,
        azimuth: f32,
        elevation: f32,
        lens: Lens,
    ) -> Camera {
        // The eye: the point at `distance` along x, raised by the elevation, then turned by the
        // azimuth (rotations about the y and z axes).
        let turn = Motor::rotation_about(0.0, 0.0, 1.0, azimuth)
            * Motor::rotation_about(0.0, -1.0, 0.0, elevation);
        let eye = (turn >> Point::xyz(distance, 0.0, 0.0)).to_euclidean();
        let eye = [eye[0] + target[0], eye[1] + target[1], eye[2] + target[2]];
        Camera::looking(width, height, eye, target, lens)
    }

    /// Pixels per unit at unit depth (perspective) or per world unit (parallel).
    fn focal(&self) -> f32 {
        match self.lens {
            Lens::Perspective(fov) => self.size[1] * 0.5 / (fov * 0.5).tan(),
            Lens::Parallel(half) => self.size[1] * 0.5 / half,
        }
    }

    /// A world point in the camera's frame: `[x, y, depth]` (`+x` to the right on screen).
    pub fn local(&self, p: [f32; 3]) -> [f32; 3] {
        let [x, y, z] = (self.pose.reverse() >> Point::xyz(p[0], p[1], p[2])).to_euclidean();
        // The frame looks along +z with +y up, so +x is to the left on screen.
        [-x, y, z]
    }

    /// The pixel of a world point, or `None` behind the camera.
    pub fn px(&self, p: [f32; 3]) -> Option<Px> {
        let [x, y, z] = self.local(p);
        let f = self.focal();
        let (sx, sy) = match self.lens {
            Lens::Perspective(_) if z <= 1e-3 => return None,
            Lens::Perspective(_) => (x / z, y / z),
            Lens::Parallel(_) => (x, y),
        };
        Some([self.size[0] * 0.5 + sx * f, self.size[1] * 0.5 - sy * f])
    }

    /// The pixel of a PGA3D point (`None` behind the camera or at infinity).
    pub fn point(&self, p: Point<(), f32>) -> Option<Px> {
        if p.e123().abs() < 1e-12 {
            return None;
        }
        self.px(p.to_euclidean())
    }

    /// The ray through pixel `q`: its origin (a point) and its direction (a unit ideal point).
    pub fn ray(&self, q: Px) -> (Point<(), f32>, Point<(), f32>) {
        let f = self.focal();
        let (sx, sy) = (
            (q[0] - self.size[0] * 0.5) / f,
            (self.size[1] * 0.5 - q[1]) / f,
        );
        match self.lens {
            Lens::Perspective(_) => {
                let d = (sx * sx + sy * sy + 1.0).sqrt();
                let dir = self.pose >> Point::direction(-sx / d, sy / d, 1.0 / d);
                (self.pose >> Point::xyz(0.0, 0.0, 0.0), dir)
            }
            Lens::Parallel(_) => (
                self.pose >> Point::xyz(-sx, sy, 0.0),
                self.pose >> Point::direction(0.0, 0.0, 1.0),
            ),
        }
    }

    /// The camera's position.
    pub fn eye(&self) -> [f32; 3] {
        (self.pose >> Point::xyz(0.0, 0.0, 0.0)).to_euclidean()
    }
}
