//! The mathematics of numga's `geometry/scenegraph` (its `core.py`), shared with the
//! manipulability example, which poses the same robot arm: PGA3D point maps (extensors
//! `Point <- Point`) that shape and place the arm's boxes, and the compound camera, two thin
//! lenses as maps on lines, that photographs them. Kinematics, camera pose, pupil rays, both
//! lenses, the sensor and the viewport compose into one map per body, from the canonical unit
//! box to sensor pixels.
//!
//! numga's rotors `exp(xy θ / 2)` turn by `-θ` about `z` (gax's `Motor::rotation` is right
//! handed and so is `exp(-B θ / 2)`); the port writes numga's exponentials literally, so its
//! joint angles pose the arm as numga's do.

// Each binary that includes this module uses a different part of it.
#![allow(dead_code)]

use gax::Unit;
use gax::pga3d::{Line, Motor, Plane, Point};

/// A point (a finite point, or a direction with weight 0).
pub type P = Point<(), f64>;
/// A plane.
pub type Pl = Plane<(), f64>;
/// A line.
pub type L = Line<(), f64>;
/// A rigid motion.
pub type M = Unit<Motor<(), f64>>;
/// A collineation, `Point <- Point`.
pub type PointMap = Point<(Point,), f64>;
/// An optical map of rays, `Line <- Line`.
pub type LineMap = Line<(Line,), f64>;

/// The origin.
pub fn origin() -> P {
    Point::xyz(0.0, 0.0, 0.0)
}

/// The finite point `(x, y, z)`.
pub fn point(c: [f64; 3]) -> P {
    Point::xyz(c[0], c[1], c[2])
}

/// numga's bivector blades: `xy = e12`, `zx = e31`, `yz = e23` (the lines through the origin
/// along `z`, `y` and `x`, numga's rotation generators).
pub fn xy() -> L {
    Line::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
}

/// `zx = e31`.
pub fn zx() -> L {
    Line::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
}

/// `yz = e23`.
pub fn yz() -> L {
    Line::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
}

/// numga's rotor `exp(generator angle / 2)`.
pub fn rotor(generator: L, angle: f64) -> M {
    generator.gp(angle * 0.5).exp()
}

/// The translation by `d` along `z`: numga's `exp(zw d / 2)`, with `zw = e3 e0 = -e03`.
pub fn up(d: f64) -> M {
    Motor::translation(0.0, 0.0, d)
}

/// The Euclidean coordinates of a finite point.
pub fn xyz(p: P) -> [f64; 3] {
    p.to_euclidean()
}

// --- geometry primitives ------------------------------------------------------------------

/// The 8 vertices of the canonical unit box: a unit cube centred on the origin.
pub fn canonical_unit_box() -> [P; 8] {
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ]
    .map(point)
}

/// The unit box's edges, as vertex index pairs: the bottom ring (`z = -0.5`), the top ring and
/// the pillars.
pub const BOX_EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

/// The unit box's faces: bottom (`-z`), top, front (`-y`), back, left (`-x`), right.
pub const BOX_FACES: [[usize; 4]; 6] = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [0, 3, 7, 4],
    [1, 2, 6, 5],
];

// --- affine and kinematic maps --------------------------------------------------------------

/// The anisotropic scaling `Point <- Point` along the principal axes. The coordinate planes read
/// a point's coordinates (`e1 & p` is its `x`, `e0 & p` its weight); each coordinate is sent to
/// its scaled axis direction, and the weight to the origin: a sum of dyads.
pub fn anisotropic_scale(sx: f64, sy: f64, sz: f64) -> PointMap {
    let p = Point::slot();
    let coordinate_planes: [Pl; 4] = [
        Plane::new(1.0, 0.0, 0.0, 0.0),
        Plane::new(0.0, 1.0, 0.0, 0.0),
        Plane::new(0.0, 0.0, 1.0, 0.0),
        Plane::new(0.0, 0.0, 0.0, 1.0),
    ];
    let scaled_axes = [
        Point::direction(sx, 0.0, 0.0),
        Point::direction(0.0, sy, 0.0),
        Point::direction(0.0, 0.0, sz),
        origin(),
    ];
    let mut map = scaled_axes[0] * (coordinate_planes[0] & p);
    for (axis, plane) in scaled_axes.iter().zip(&coordinate_planes).skip(1) {
        map += *axis * (*plane & p);
    }
    map
}

/// Forward kinematics of a 5-body robot arm. Each body is the canonical unit box shaped by an
/// anisotropic scaling and placed in the world by its cumulative joint motor. The joint angles
/// are base yaw, shoulder pitch, elbow pitch and wrist pitch; the result is the five
/// unit-box-to-world maps and the four joint pivots (turret, shoulder, elbow, wrist).
pub fn robot_arm(angles: [f64; 4]) -> ([PointMap; 5], [M; 4]) {
    let [base, shoulder, elbow, wrist] = angles;
    // The pedestal rests on the floor `z = 0`.
    let body_0 = up(0.125) >> anisotropic_scale(0.9, 0.9, 0.25);
    // The turret yaws about the vertical at height 0.25.
    let turret = up(0.25) * rotor(xy(), base);
    let body_1 = (turret * up(0.175)) >> anisotropic_scale(0.5, 0.5, 0.35);
    // The shoulder, 0.35 above the turret's base, pitches about its local y axis.
    let shoulder = turret * up(0.35) * rotor(zx(), shoulder);
    let body_2 = (shoulder * up(0.6)) >> anisotropic_scale(0.24, 0.24, 1.2);
    // The elbow at the tip of the upper arm, 1.2 along it.
    let elbow = shoulder * up(1.2) * rotor(zx(), elbow);
    let body_3 = (elbow * up(0.5)) >> anisotropic_scale(0.18, 0.18, 1.0);
    // The wrist at the tip of the forearm, 1.0 along it.
    let wrist = elbow * up(1.0) * rotor(zx(), wrist);
    // The gripper block.
    let body_4 = (wrist * up(0.15)) >> anisotropic_scale(0.28, 0.14, 0.3);
    (
        [body_0, body_1, body_2, body_3, body_4],
        [turret, shoulder, elbow, wrist],
    )
}

// --- the compound camera ------------------------------------------------------------------

/// A thin lens `Line <- Line` that focuses rays towards its centre: it shears a line by its
/// incidence with the centre, `L - (centre & (L ^ plane)) / f`.
pub fn thin_lens(centre: P, plane: Pl, focal: f64) -> LineMap {
    let l = Line::slot();
    l - (centre & (l ^ plane)).gp(1.0 / focal)
}

/// The front and rear lenses on the optical axis `-z`, and the rear lens plane. The front lens
/// sits at the origin; the rear lens is the same lens at home conjugated by its placement.
pub fn lens_train(focal_front: f64, focal_rear: f64, rear_gap: f64) -> (LineMap, LineMap, Pl) {
    let front_plane = Plane::new(0.0, 0.0, -1.0, 0.0);
    let front = thin_lens(origin(), front_plane, focal_front);
    let placement = up(rear_gap);
    let rear =
        placement >> thin_lens(origin(), front_plane, focal_rear).of(placement << Line::slot());
    (front, rear, placement >> front_plane)
}

/// The compound camera `Point <- Point` from world points to sensor points in the camera frame:
/// pull the world into the camera frame, join with the pupil into a ray, refract through the
/// front and the rear lens (a composition of maps), and meet the sensor.
pub fn lens_camera(pose: M, front: LineMap, rear: LineMap, pupil: P, sensor: Pl) -> PointMap {
    let incoming: Line<(Point,), f64> = (pose << Point::slot()) & pupil;
    rear.of(front.of(incoming)) ^ sensor
}

/// The camera motor at `position` that turns its `+z` axis to `target`: a pitch about `x` (a
/// quarter turn less the pitch down to the target), then the translation. The lenses and the
/// sensor lie along `-z`, behind the pupil as light travels. (numga's docstring has the optical
/// axis `-z` point at the target; its numbers, and this port's, have `+z`.)
pub fn look_at(position: [f64; 3], target: [f64; 3]) -> M {
    let translation = Motor::translation(position[0], position[1], position[2]);
    let pitch = (target[2] - position[2]).atan2(target[1] - position[1]);
    translation * rotor(yz(), core::f64::consts::FRAC_PI_2 - pitch)
}

/// The viewport `Point <- Point` from metric sensor coordinates to pixels: the
/// `sensor_width` x `sensor_height` chip onto `width` x `height` pixels, rows growing downward.
pub fn viewport(width: f64, height: f64, sensor_width: f64, sensor_height: f64) -> PointMap {
    let scale = anisotropic_scale(width / sensor_width, -height / sensor_height, 1.0);
    Motor::translation(width / 2.0, height / 2.0, 0.0) >> scale
}

/// A projected point with unit weight: the perspective division.
pub fn project(map: PointMap, p: P) -> P {
    map.of(p).unitized()
}

/// The world points where the ray from a scene point through the pupil crosses each optical
/// interface: the scene point, the pupil, the rear lens plane and the sensor.
pub fn trace_ray(
    scene: P,
    pose: M,
    front: LineMap,
    rear: LineMap,
    rear_plane: Pl,
    pupil: P,
    sensor: Pl,
) -> [P; 4] {
    let ray_in = (pose << scene) & pupil;
    let ray_mid = front.of(ray_in);
    let rear_hit = ray_mid ^ rear_plane;
    let sensor_hit = rear.of(ray_mid) ^ sensor;
    [
        scene,
        pose >> pupil,
        pose >> rear_hit.unitized(),
        pose >> sensor_hit.unitized(),
    ]
}
