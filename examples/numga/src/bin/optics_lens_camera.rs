//! numga's `optics/lens_camera`: a zoom camera with depth of field, from cones pulled through
//! non-rigid maps in PGA3D. An ideal thin lens is a projective collineation of space,
//! `Point + origin (home & Point) / f`; its action on lines, `Line - (origin & (Line ^ home)) / f`,
//! is the join of the images, so the train of two lenses composes either way. The rays a scene
//! point sends through the aperture form a cone: the pupil ball pulled back through the central
//! projection from the point onto the aperture plane. The train carries the cone to the image
//! cone, a pullback through the inverse of its point map, and the sensor cuts the image cone in
//! the point's blur conic. Nothing asks where a point focuses: the cone's vertex is wherever
//! the collineation put it. The animation swings the rear lens to zoom while the focus and the
//! aperture change; the sensor is rasterised from the 60 cones as implicit functions, with a
//! side view of the train and numga's three stills (wide, tele, and tele with a tilted sensor).

use gax_numga_examples::canvas::{mix, srgb};
use gax_numga_examples::{Align, Anim, Axes, Canvas, Marker, backdrop, caption, palette, run};

mod lens {
    use gax::Unit;
    use gax::pga3d::{Line, Motor, Plane, Point};

    pub type P = Point<(), f64>;
    pub type Pl = Plane<(), f64>;
    pub type M = Unit<Motor<(), f64>>;
    /// A collineation maps points to points.
    pub type PointMap = Point<(Point,), f64>;
    pub type LineMap = Line<(Line,), f64>;
    /// A primal quadric maps a point to its polar plane.
    pub type Quadric = Plane<(Point,), f64>;
    /// A dual quadric maps a plane to its pole.
    #[cfg(test)]
    pub type DualQuadric = Point<(Plane,), f64>;
    /// A camera maps a scene point and a pupil point to a sensor point.
    pub type Camera = Point<(Point, Point), f64>;

    /// The origin; every element is built in its home frame, centred on it in the plane `x = 0`.
    pub fn origin() -> P {
        Point::xyz(0.0, 0.0, 0.0)
    }

    /// The home plane `x = 0`.
    pub fn home() -> Pl {
        Plane::new(1.0, 0.0, 0.0, 0.0)
    }

    /// A thin lens of focal length `focal` in the home plane, as a collineation of points and as
    /// the induced map on lines.
    pub fn thin_lens(focal: f64) -> (PointMap, LineMap) {
        let p = Point::slot();
        let l = Line::slot();
        let points = p + origin() * (home() & p).gp(1.0 / focal);
        let lines = l - (origin() & (l ^ home())).gp(1.0 / focal);
        (points, lines)
    }

    /// A ball of the given radius about the origin; its section with the home plane is the
    /// aperture rim. The coordinate planes through the origin, each with its pairing (the
    /// squared distances add up), less the plane at infinity times the squared radius.
    pub fn ball(radius: f64) -> Quadric {
        let x = Point::slot();
        let infinity = Plane::new(0.0, 0.0, 0.0, 1.0);
        let mut q = (infinity * (infinity & x)).gp(-radius * radius);
        for normal in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
            let e = Plane::from_normal(normal, 0.0);
            q += e * (e & x);
        }
        q
    }

    /// The map on planes induced by a map on points, through incidence: for a plane `p` and a
    /// point `q`, `on_planes(collineation)(p) & q == p & collineation(q)`. It exists for a
    /// singular map too, such as a central projection.
    pub fn on_planes(collineation: PointMap) -> Plane<(Plane,), f64> {
        (Plane::slot() & Point::slot()).solve(Plane::slot() & collineation)
    }

    /// One exposure: the lens train's collineation, the camera map, the sensor frame, the image
    /// cone of every scene point, the element planes [front, rear, sensor], and a fan of rays
    /// from the subject through the aperture rim, as its points on successive planes; and where
    /// each point's chief ray (through the front lens's centre) meets the sensor, inside its
    /// blur conic.
    pub struct Exposure {
        pub collineation: PointMap,
        // Read by the checks; the drawing uses the chief rays it gave.
        #[cfg_attr(not(test), allow(dead_code))]
        pub cam: Camera,
        pub frame: M,
        pub cones: Vec<Quadric>,
        pub chief: Vec<P>,
        pub planes: [Pl; 3],
        pub legs: [[P; 3]; 4],
    }

    /// Place the lenses, focus the sensor, and carry each scene point's aperture cone to the
    /// sensor side.
    #[allow(clippy::too_many_arguments)]
    pub fn expose(
        scene: &[P],
        subject: P,
        focal: [f64; 2],
        placements: [M; 2],
        focus: P,
        tilt: M,
        radius: f64,
        rim: [P; 3],
    ) -> Exposure {
        // Place the lenses [front, rear], then compose their point maps and their line maps.
        let lenses = focal.map(thin_lens);
        let points: [PointMap; 2] = core::array::from_fn(|i| {
            placements[i] >> lenses[i].0.of(placements[i] << Point::slot())
        });
        let lines: [LineMap; 2] = core::array::from_fn(|i| {
            placements[i] >> lenses[i].1.of(placements[i] << Line::slot())
        });
        let [front, rear] = lines;
        let (front_plane, rear_plane) = (placements[0] >> home(), placements[1] >> home());
        let collineation = points[1].of(points[0]);
        let train = rear.of(front);
        let pupil_ball = placements[0] >> ball(radius).of(placements[0] << Point::slot());

        // The focus point's image places the sensor; the tilt turns it about that image.
        let image = collineation.of(focus).unitized();
        let frame = gax::pga3d::Motor::between(origin(), image) * tilt;
        let sensor = frame >> home();
        // The camera: the train applied to the join of two open points (scene and pupil), met
        // with the sensor. (Written `p & Point::slot()`, as `p & p` trips clippy's `eq_op`.)
        let p = Point::<(), f64>::slot();
        let cam: Camera = train.of(p & Point::slot()) ^ sensor;

        // Project through each scene point onto the pupil and pull back the pupil ball; then
        // pull back once more through the inverse lens train for the image cone.
        let back = collineation.inverse();
        let back_on_planes = on_planes(back);
        let cones = scene
            .iter()
            .map(|s| {
                let project: PointMap = (*s & Point::slot()) ^ front_plane;
                let cone = on_planes(project).of(pupil_ball.of(project));
                back_on_planes.of(cone.of(back))
            })
            .collect();

        // The chief rays, through the front lens's centre: the camera with that pupil point bound.
        let centre = placements[0] >> origin();
        let chief = scene.iter().map(|s| cam.of(*s).of(centre)).collect();
        let rays = rim.map(|r| subject & (placements[0] >> r));
        let legs = [
            [subject; 3],
            rays.map(|r| r ^ front_plane),
            rays.map(|r| front.of(r) ^ rear_plane),
            rays.map(|r| rear.of(front.of(r)) ^ sensor),
        ];
        Exposure {
            collineation,
            cam,
            frame,
            cones,
            chief,
            planes: [front_plane, rear_plane, sensor],
            legs,
        }
    }

    /// The boundary of the sensor's section of a cone, traced from a point inside it along the
    /// sensor plane: along each direction the cone's form is a quadratic, and its root is the
    /// boundary.
    pub fn section(cone: Quadric, start: P, frame: M, samples: usize) -> Vec<P> {
        (0..samples)
            .map(|k| {
                let theta = core::f64::consts::TAU * k as f64 / (samples - 1) as f64;
                // The direction `+y` turned by theta about `x`, in the sensor.
                let turn = Motor::rotation_about(1.0, 0.0, 0.0, theta);
                let across = frame >> (turn >> Point::direction(0.0, 1.0, 0.0));
                let (a, b, c) = (
                    (across & cone.of(across)).s(),
                    (across & cone.of(start)).s(),
                    (start & cone.of(start)).s(),
                );
                // A rounding-level negative at the vertex.
                let root = (b * b - a * c).max(0.0).sqrt();
                start + across.gp((root - b) / a)
            })
            .collect()
    }
}

/// The scenes: the same two-lens train on a grid of points at three depths; a setting places
/// the rear lens, picks the point to focus on, tilts the sensor and sets the aperture.
mod scenes {
    use super::lens::*;
    use gax::pga3d::{Motor, Point};

    /// Focal lengths of the front and the rear lens. The front lens sits at `x = 1`; the rear
    /// lens moves to zoom.
    pub const FOCAL: [f64; 2] = [1.0, 0.6];
    pub const FRONT_AT: f64 = 1.0;

    /// A grid of points at three depths in front of the camera, `[depth][height][width]`
    /// flattened (index `20 d + 4 h + w`).
    pub fn scene() -> Vec<P> {
        let mut pts = Vec::with_capacity(60);
        for x in [-3.2, -2.2, -1.6] {
            for h in 0..5 {
                for w in 0..4 {
                    pts.push(Point::xyz(x, -0.8 + 0.4 * h as f64, -0.5 + w as f64 / 3.0));
                }
            }
        }
        pts
    }

    /// The subject on the far layer whose rays through the aperture rim are traced.
    pub fn subject() -> P {
        Point::xyz(-3.2, 0.0, -1.0 / 6.0)
    }

    /// Expose the scene with the rear lens at `rear_at`, focused on the axis at `focus_at`, the
    /// sensor tilted by `tilt` radians, and an aperture of the given radius.
    pub fn setting(rear_at: f64, focus_at: f64, tilt: f64, radius: f64) -> Exposure {
        let placements = [FRONT_AT, rear_at].map(|x| Motor::translation(x, 0.0, 0.0));
        let focus = Point::xyz(focus_at, 0.0, 0.0);
        // The sensor tilts clockwise about the vertical through the focus point's image.
        let turn = Motor::rotation_about(0.0, 0.0, 1.0, -tilt);
        let rim = [radius, 0.0, -radius].map(|y| Point::xyz(0.0, y, 0.0));
        expose(
            &scene(),
            subject(),
            FOCAL,
            placements,
            focus,
            turn,
            radius,
            rim,
        )
    }

    /// Wide and tele at one focus and aperture, and tele with the sensor tilted 25 degrees.
    pub fn stills() -> [Exposure; 3] {
        [
            setting(1.4, -2.2, 0.0, 0.45),
            setting(1.8, -2.2, 0.0, 0.45),
            setting(1.8, -2.2, 25f64.to_radians(), 0.45),
        ]
    }

    /// The rear lens swings to zoom while the focus and the aperture change, at phase `t`
    /// (radians): the rear lens position, the focus distance, the aperture radius and the
    /// exposure.
    pub fn motion(t: f64) -> (f64, f64, f64, Exposure) {
        let rear_at = 1.6 + 0.2 * t.sin();
        let focus_at = 2.4 - 0.8 * t.cos();
        let radius = 0.3 + 0.1 * (2.0 * t).sin();
        (
            rear_at,
            focus_at,
            radius,
            setting(rear_at, -focus_at, 0.0, radius),
        )
    }
}

/// The sensor rasterised from the image cones as implicit functions.
mod raster {
    use super::lens::{Exposure, Quadric, section};
    use gax::pga2d;
    use gax::pga3d::{Plane, Point};

    /// Half extents of the sensor window in its own frame.
    pub const SENSOR: [f64; 2] = [0.225, 0.175];
    /// numga's sensor: 280 x 360 pixels, each 2 x 2 samples.
    const NUMGA_SAMPLES: f64 = 560.0 * 720.0;
    /// numga's logistic edge: two of its samples.
    const EDGE: f64 = 2.0 * SENSOR[0] / 720.0 * 2.0;

    /// A cone cut by the sensor: its form on sensor points (`(y, z)` in the sensor's frame, as
    /// PGA2D points), and the polar plane at each.
    struct Conic {
        form: gax::pga3d::Scalar<(pga2d::Point, pga2d::Point), f64>,
        polar: Plane<(pga2d::Point,), f64>,
    }

    fn conic(cone: Quadric, on_sensor: Point<(pga2d::Point,), f64>) -> Conic {
        let polar = cone.of(on_sensor);
        Conic {
            form: on_sensor & polar,
            polar,
        }
    }

    impl Conic {
        /// The first-order signed distance of a sensor point from the blur disc's edge: the
        /// cone's form over the length of its gradient, twice the polar plane's normal.
        fn distance(&self, p: pga2d::Point<(), f64>) -> f64 {
            let f = self.form.of(p).of(p).s();
            f / (2.0 * self.polar.of(p).norm())
        }

        /// The disc's coverage of a sensor point: a logistic edge two of numga's samples wide,
        /// a crude diffraction limit that spreads a focused point over a few pixels.
        fn coverage(&self, p: pga2d::Point<(), f64>) -> f64 {
            1.0 / (1.0 + (-self.distance(p) / EDGE).exp())
        }
    }

    /// The sensor point at the centre of pixel `(row, col)` of a `rows` x `cols` raster.
    fn sample(row: usize, col: usize, rows: usize, cols: usize) -> pga2d::Point<(), f64> {
        let y = -SENSOR[0] + 2.0 * SENSOR[0] * (col as f64 + 0.5) / cols as f64;
        let z = SENSOR[1] - 2.0 * SENSOR[1] * (row as f64 + 0.5) / rows as f64;
        pga2d::Point::xy(y, z)
    }

    fn threads() -> usize {
        std::thread::available_parallelism().map_or(4, |t| t.get())
    }

    /// The pixel rows and columns a blur disc can light: the bounding box of its conic, traced
    /// by [`section`] from the chief-ray hit, widened by 20 logistic edges (beyond which a
    /// pixel's coverage is below `e^-20`); the whole raster if the trace fails.
    fn reach(exposure: &Exposure, k: usize, rows: usize, cols: usize) -> [usize; 4] {
        let pitch = 2.0 * SENSOR[0] / cols as f64;
        let margin = 20.0 * EDGE / pitch + 1.0;
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for p in section(exposure.cones[k], exposure.chief[k], exposure.frame, 25) {
            let [_, y, z] = (exposure.frame << p).to_euclidean();
            // Column and row, fractional.
            let at = [
                (y + SENSOR[0]) / pitch - 0.5,
                (SENSOR[1] - z) / (2.0 * SENSOR[1] / rows as f64) - 0.5,
            ];
            for i in 0..2 {
                lo[i] = lo[i].min(at[i]);
                hi[i] = hi[i].max(at[i]);
            }
        }
        if !(lo.iter().chain(&hi).all(|v| v.is_finite())) {
            return [0, rows, 0, cols];
        }
        let clamp = |v: f64, n: usize| v.clamp(0.0, n as f64) as usize;
        [
            clamp((lo[1] - margin).floor(), rows),
            clamp((hi[1] + margin).ceil() + 1.0, rows),
            clamp((lo[0] - margin).floor(), cols),
            clamp((hi[0] + margin).ceil() + 1.0, cols),
        ]
    }

    /// The sensor image, `rows` x `cols` sRGB values row by row. Each disc deposits the same
    /// energy, so a point in focus is a bright dot and a defocused one a dim wide disc; the
    /// points of each depth layer light one colour channel (far red, middle green, near blue).
    pub fn rasterise(exposure: &Exposure, energy: f64, rows: usize, cols: usize) -> Vec<[f32; 3]> {
        // Sensor points `(y, z)` in the sensor's own frame, carried into the world.
        let on_sensor = exposure.frame
            >> Point::<(pga2d::Point,), f64>::from_images([
                Point::new(0.0, 1.0, 0.0, 0.0),
                Point::new(0.0, 0.0, 1.0, 0.0),
                Point::new(0.0, 0.0, 0.0, 1.0),
            ]);
        let conics: Vec<Conic> = exposure
            .cones
            .iter()
            .map(|c| conic(*c, on_sensor))
            .collect();
        let boxes: Vec<[usize; 4]> = (0..conics.len())
            .map(|k| reach(exposure, k, rows, cols))
            .collect();
        // Each disc's total coverage, to share its energy over the pixels it covers.
        let n = threads();
        let mut totals = vec![0.0; conics.len()];
        std::thread::scope(|s| {
            let per = conics.len().div_ceil(n).max(1);
            for ((chunk, out), boxes) in conics
                .chunks(per)
                .zip(totals.chunks_mut(per))
                .zip(boxes.chunks(per))
            {
                s.spawn(move || {
                    for ((conic, total), [r0, r1, c0, c1]) in chunk.iter().zip(out).zip(boxes) {
                        for row in *r0..*r1 {
                            for col in *c0..*c1 {
                                *total += conic.coverage(sample(row, col, rows, cols));
                            }
                        }
                    }
                });
            }
        });
        // numga's brightness, 550 per unit share at its resolution, kept at this one.
        let gain = 550.0 * (rows * cols) as f64 / NUMGA_SAMPLES * energy;
        let weights: Vec<f64> = totals.iter().map(|t| gain / t.max(1e-12)).collect();
        let mut image = vec![[0.0f32; 3]; rows * cols];
        std::thread::scope(|s| {
            let per = rows.div_ceil(n).max(1);
            for (k, out) in image.chunks_mut(per * cols).enumerate() {
                let (conics, weights, boxes) = (&conics, &weights, &boxes);
                s.spawn(move || {
                    let band = k * per..(k * per + out.len() / cols);
                    let mut acc = vec![[0.0f64; 3]; out.len()];
                    for (j, ((conic, w), [r0, r1, c0, c1])) in
                        conics.iter().zip(weights).zip(boxes).enumerate()
                    {
                        for row in (*r0).max(band.start)..(*r1).min(band.end) {
                            for col in *c0..*c1 {
                                let p = sample(row, col, rows, cols);
                                acc[(row - band.start) * cols + col][j / 20] +=
                                    conic.coverage(p) * w;
                            }
                        }
                    }
                    for (px, rgb) in out.iter_mut().zip(acc) {
                        *px = rgb.map(|v| v.clamp(0.0, 1.0) as f32);
                    }
                });
            }
        });
        image
    }
}

use gax::pga2d;
use gax::pga3d::Point;
use lens::Exposure;
use scenes::*;

const SECONDS: f32 = 8.0;
/// The live sensor in canvas pixels, rows by columns (numga's 280 x 360 at 0.9).
const LIVE: [usize; 2] = [252, 324];
/// The stills' thumbnails.
const THUMB: [usize; 2] = [63, 81];

/// Paint a raster with its top left corner at `(x, y)`.
fn paint(c: &mut Canvas, image: &[[f32; 3]], x: usize, y: usize, rows: usize, cols: usize) {
    c.clip([x as f32, y as f32, (x + cols) as f32, (y + rows) as f32]);
    c.shade(1, |px, py| {
        let (col, row) = (px as usize - x, py as usize - y);
        let [r, g, b] = image[row * cols + col];
        Some(srgb(r, g, b))
    });
    c.unclip();
}

/// The stills, rasterised once.
fn stills_images() -> &'static [Vec<[f32; 3]>; 3] {
    static STILLS: std::sync::OnceLock<[Vec<[f32; 3]>; 3]> = std::sync::OnceLock::new();
    STILLS.get_or_init(|| stills().map(|e| raster::rasterise(&e, 1.0, THUMB[0], THUMB[1])))
}

/// The side view: points of space seen along `z`, as points of the `x`-`y` plane (PGA2D).
fn side_view() -> pga2d::Point<(Point,), f64> {
    pga2d::Point::from_images([
        pga2d::Point::new(1.0, 0.0, 0.0),
        pga2d::Point::new(0.0, 1.0, 0.0),
        pga2d::Point::new(0.0, 0.0, 0.0),
        pga2d::Point::new(0.0, 0.0, 1.0),
    ])
}

/// Side view in the `x`-`y` plane: the element planes as segments, the scene layers, and the
/// ray fan leg by leg.
fn side(c: &mut Canvas, ax: &Axes, exposure: &Exposure, radius: f64) {
    let flat = side_view();
    let z = gax::pga3d::Plane::new(0.0, 0.0, 1.0, 0.0);
    let at = |h: f64| gax::pga3d::Plane::new(0.0, 1.0, 0.0, -h);
    let heights = [radius, 0.6, 0.35];
    let names = ["FRONT", "REAR", "SENSOR"];
    for ((plane, h), name) in exposure.planes.iter().zip(heights).zip(names) {
        let (top, bottom) = (*plane ^ z ^ at(h), *plane ^ z ^ at(-h));
        let colour = if name == "SENSOR" {
            palette::orange()
        } else {
            palette::sky()
        };
        let top = flat.of(top);
        ax.line(c, top, flat.of(bottom), 2.5, colour, 1.0);
        ax.text(
            c,
            pga2d::Motor::translation(0.0, 0.12) >> top,
            name,
            7.0,
            palette::grid(),
            Align::Center,
        );
    }
    let pts: Vec<_> = scene().iter().map(|p| flat.of(*p)).collect();
    let layer = [palette::red(), palette::green(), palette::blue()];
    for (k, chunk) in pts.chunks(20).enumerate() {
        ax.scatter(
            c,
            chunk,
            Marker::Dot,
            3.5,
            mix(layer[k], palette::ink(), 0.3),
            1.0,
        );
    }
    // Where the collineation puts each point: its image cone's vertex, the point's focus.
    let images: Vec<_> = scene()
        .iter()
        .map(|p| flat.of(exposure.collineation.of(*p)))
        .collect();
    for (k, chunk) in images.chunks(20).enumerate() {
        ax.scatter(
            c,
            chunk,
            Marker::Cross,
            4.0,
            mix(layer[k], palette::ink(), 0.3),
            0.9,
        );
    }
    for ray in 0..3 {
        let fan: Vec<_> = exposure.legs.iter().map(|leg| flat.of(leg[ray])).collect();
        ax.polyline(c, &fan, 1.0, palette::yellow(), 0.9);
    }
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.width as f32, c.height as f32);
    let k = (h / 360.0).min(w / 640.0);
    let phase = f64::from(t / SECONDS) * core::f64::consts::TAU;
    let (rear_at, focus_at, radius, exposure) = motion(phase);
    // The live sensor, at the left; smaller canvases get a smaller raster.
    let (rows, cols) = (
        ((LIVE[0] as f32 * k) as usize).max(8),
        ((LIVE[1] as f32 * k) as usize).max(8),
    );
    let (x0, y0) = ((12.0 * k) as usize, (58.0 * k) as usize);
    let energy = (radius / 0.45).powi(2);
    let image = raster::rasterise(&exposure, energy, rows, cols);
    paint(c, &image, x0, y0, rows, cols);
    let (fx, fy) = (x0 as f32, y0 as f32);
    c.polyline(
        &[
            [fx, fy],
            [fx + cols as f32, fy],
            [fx + cols as f32, fy + rows as f32],
            [fx, fy + rows as f32],
        ],
        1.0,
        palette::grid(),
        1.0,
        true,
    );
    let s = 9.0 * k;
    c.text(
        &format!("REAR LENS AT {rear_at:.2}, FOCUSED AT {focus_at:.2}, APERTURE {radius:.2}"),
        fx,
        fy + rows as f32 + s * 1.6,
        s * 0.85,
        palette::ink(),
        Align::Left,
    );
    // The side view, top right.
    let right = fx + cols as f32 + 16.0 * k;
    let width = w - right - 10.0 * k;
    let ax = Axes::new(
        [right, fy, right + width, fy + width * 2.4 / 6.2],
        [-3.6, 2.6],
        [-1.2, 1.2],
    );
    side(c, &ax, &exposure, radius);
    c.text(
        "SIDE VIEW: LAYERS FAR, MID, NEAR LIGHT RED, GREEN, BLUE",
        right,
        ax.rect[3] + s * 1.3,
        s * 0.7,
        palette::grid(),
        Align::Left,
    );
    // The stills, below it.
    let stills = stills_images();
    let ty = (ax.rect[3] + s * 4.0) as usize;
    let gap = ((width - 3.0 * THUMB[1] as f32) / 2.0).max(2.0);
    for (i, (img, label)) in stills.iter().zip(["WIDE", "TELE", "TILTED 25"]).enumerate() {
        let tx = (right + i as f32 * (THUMB[1] as f32 + gap)) as usize;
        if tx + THUMB[1] <= c.width && ty + THUMB[0] <= c.height {
            paint(c, img, tx, ty, THUMB[0], THUMB[1]);
        }
        c.text(
            label,
            tx as f32,
            (ty + THUMB[0]) as f32 + s * 1.2,
            s * 0.7,
            palette::ink(),
            Align::Left,
        );
    }
    c.text(
        "STILLS: APERTURE 0.45, FOCUSED AT 2.2",
        right,
        ty as f32 - s * 0.6,
        s * 0.7,
        palette::grid(),
        Align::Left,
    );
    caption(
        c,
        "LENS CAMERA: A ZOOM WITH DEPTH OF FIELD",
        "APERTURE CONES PULLED THROUGH THE LENS COLLINEATIONS, CUT BY THE SENSOR (PGA3D)",
    );
}

fn main() {
    run(
        Anim::new("lens camera", SECONDS).size(640, 360).scale(2),
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::lens::*;
    use super::scenes::*;
    use gax::ApproxEq;
    use gax::pga2d;
    use gax::pga3d::{Plane, Point};

    /// Whether a point is within `tol` of `want`, coordinate by coordinate.
    fn near(p: P, want: [f64; 3], tol: f64) -> bool {
        p.unitized()
            .max_abs_diff(&Point::xyz(want[0], want[1], want[2]))
            <= tol
    }

    /// numga's checks in `stills`: the train on lines is the join of the collineation's images;
    /// each image cone's vertex is the image of its scene point; the sensor's section of a cone
    /// lies on the conic got the short way; and a point at the focus depth images to a point.
    #[test]
    fn stills_pass_their_checks() {
        let [wide, _, tilted] = stills();
        let scene = scene();
        let Exposure {
            collineation,
            cam,
            frame,
            cones,
            ..
        } = &tilted;
        let place_front = gax::pga3d::Motor::translation(FRONT_AT, 0.0, 0.0);
        let (sensor, centre) = (*frame >> home(), place_front >> origin());
        // The train on lines is the join of the collineation's images: the lens is a
        // collineation.
        for other in [
            Point::xyz(-1.0, 0.3, 0.1),
            Point::xyz(-1.5, 0.2, -0.4),
            Point::xyz(-2.0, -0.5, 0.6),
        ] {
            let via_cam = cam.of(scene[0]).of(other);
            let joined = (collineation.of(scene[0]) & collineation.of(other)) ^ sensor;
            assert!(
                via_cam.max_abs_diff(&joined) <= 1e-12,
                "{via_cam:?} {joined:?}"
            );
        }
        // The image cone's vertex is the collineation's image of its point.
        for (s, cone) in scene.iter().zip(cones) {
            let v = collineation.of(*s);
            assert!((v & cone.of(v)).c.iter().all(|x| x.abs() < 1e-9));
        }
        // The aperture disc as a flat dual quadric, at the front lens.
        let (dy, dz) = (
            Point::direction(0.0, 1.0, 0.0),
            Point::direction(0.0, 0.0, 1.0),
        );
        let pl = Plane::slot();
        let r2 = 0.45f64 * 0.45;
        let disc = (dy * (dy & pl) + dz * (dz & pl)).gp(r2) - origin() * (origin() & pl);
        let pupil = place_front >> disc.of(place_front << Plane::slot());
        // Sensor lines (PGA2D lines in the sensor's frame) as planes, and sensor-frame points
        // as PGA2D points: numga's subspaces `y z w` and `zxw xyw zyx`.
        let lines_in = Plane::<(pga2d::Line,), f64>::from_images([
            Plane::new(0.0, 1.0, 0.0, 0.0),
            Plane::new(0.0, 0.0, 1.0, 0.0),
            Plane::new(0.0, 0.0, 0.0, 1.0),
        ]);
        let drop_x = pga2d::Point::<(Point,), f64>::from_images([
            pga2d::Point::new(0.0, 0.0, 0.0),
            pga2d::Point::new(1.0, 0.0, 0.0),
            pga2d::Point::new(0.0, 1.0, 0.0),
            pga2d::Point::new(0.0, 0.0, 1.0),
        ]);
        for index in [0, 59] {
            let s = scene[index];
            let start = (collineation.of(s) & collineation.of(centre)) ^ sensor;
            let boundary = section(cones[index], start, *frame, 48);
            // The disc pushed through the pupil-to-sensor collineation of the same point.
            let to_sensor: PointMap = cam.of(s);
            let pushed: DualQuadric = to_sensor.of(pupil.of(on_planes(to_sensor)));
            let section_dual = drop_x.of(*frame << pushed.of(*frame >> lines_in));
            for b in boundary {
                let hit = drop_x.of(*frame << b);
                let line = section_dual.solve(hit);
                assert!((hit & line).s().abs() < 1e-10, "{:?}", (hit & line));
            }
        }
        // A point at the focus depth images to a point: its section collapses onto the
        // chief-ray hit.
        let Exposure {
            collineation,
            frame,
            cones,
            ..
        } = &wide;
        let start = (collineation.of(scene[29]) & collineation.of(centre)) ^ (*frame >> home());
        for b in section(cones[29], start, *frame, 48) {
            assert!(b.unitized().max_abs_diff(&start.unitized()) <= 1e-6);
        }
    }

    /// The stills against numga: the sensor's place, the subject's image, the ray fan, a cone's
    /// vertex, the signed distance inside a blur disc, and a traced section.
    #[test]
    fn stills_agree_with_numga() {
        let [wide, tele, tilted] = stills();
        let scene = scene();
        let cases = [
            (
                &wide,
                1.782417582418,
                [1.761983471074, 0.0, 0.020661157025],
                [1.4, 0.312857142857, 0.015873015873],
                [1.782417582418, -0.017660910518, 0.020931449503],
                [1.802739726027, -0.164383561644, -0.102739726027],
            ),
            (
                &tele,
                2.113043478261,
                [2.076404494382, 0.0, 0.02808988764],
                [1.8, 0.175714285714, 0.031746031746],
                [2.113043478261, -0.023291925466, 0.027605244997],
                [2.147368421053, -0.210526315789, -0.131578947368],
            ),
            (
                &tilted,
                2.113043478261,
                [2.076404494382, 0.0, 0.02808988764],
                [1.8, 0.175714285714, 0.031746031746],
                [2.104665754426, -0.017966086742, 0.027716061449],
                [2.147368421053, -0.210526315789, -0.131578947368],
            ),
        ];
        for (e, sensor_x, image, rear, hit, vertex) in cases {
            assert!(near(e.frame >> origin(), [sensor_x, 0.0, 0.0], 1e-11));
            assert!(near(e.collineation.of(subject()), image, 1e-11));
            assert!(near(e.legs[1][0], [1.0, 0.45, 0.0], 1e-12));
            assert!(near(e.legs[2][0], rear, 1e-11));
            assert!(near(e.legs[3][0], hit, 1e-11));
            assert!(near(e.collineation.of(scene[59]), vertex, 1e-11));
        }
        // Inside the far corner's blur disc, at the chief-ray hit, the distance is positive.
        let centre = Point::xyz(FRONT_AT, 0.0, 0.0);
        let col = wide.collineation;
        let start = (col.of(scene[0]) & col.of(centre)) ^ (wide.frame >> home());
        assert!(near(
            start,
            [1.782417582418, 0.100470957614, 0.062794348509],
            1e-11
        ));
        let polar = wide.cones[0].of(start);
        let d = (start & polar).s() / (2.0 * polar.norm());
        assert!((d - 0.04291163382074708).abs() < 1e-10, "{d}");
        // numga's section from there, 8 samples.
        let want = [
            [0.082810047096, 0.062794348509],
            [0.089459560014, 0.048986492657],
            [0.104400879915, 0.04557623389],
            [0.116382888166, 0.055131566611],
            [0.116382888176, 0.070457130386],
            [0.104400879947, 0.08001246312],
            [0.089459560056, 0.076602204394],
            [0.082810047096, 0.062794348594],
        ];
        for (b, want) in section(wide.cones[0], start, wide.frame, 8)
            .iter()
            .zip(want)
        {
            assert!((b.to_euclidean()[0] - 1.782417582418).abs() < 1e-11);
            assert!(near(*b, [1.782417582418, want[0], want[1]], 1e-9), "{b:?}");
        }
    }

    /// The thin lens equation, for the point map: a point at distance `d` in front images at
    /// `d f / (d - f)` behind.
    #[test]
    fn a_thin_lens_images_by_the_lens_equation() {
        let (points, lines) = thin_lens(0.8);
        let image = points.of(Point::xyz(-2.0, 0.3, 0.0));
        let [x, _, _] = image.to_euclidean();
        assert!((x - 2.0 * 0.8 / (2.0 - 0.8)).abs() < 1e-14);
        // And on lines: the image of a join is the join of the images.
        let (a, b) = (Point::xyz(-2.0, 0.3, 0.1), Point::xyz(-1.0, -0.2, 0.4));
        let l = lines.of(a & b);
        let j = points.of(a) & points.of(b);
        assert!(l.max_abs_diff(&j) <= 1e-14, "{l:?} {j:?}");
    }

    /// The motion loops, and its frames rasterise to light on every channel.
    #[test]
    fn the_motion_loops_and_rasterises() {
        let (a, b) = (motion(0.0), motion(core::f64::consts::TAU));
        let drift = [a.0 - b.0, a.1 - b.1, a.2 - b.2];
        assert!(drift.iter().all(|d| d.abs() <= 1e-12), "{drift:?}");
        let image = super::raster::rasterise(&a.3, 1.0, 28, 36);
        let sum = image
            .iter()
            .fold([0.0f32; 3], |s, p| [s[0] + p[0], s[1] + p[1], s[2] + p[2]]);
        assert!(sum.iter().all(|v| *v > 1.0), "{sum:?}");
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
