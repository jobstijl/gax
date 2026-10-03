//! numga's `geometry/projection`: projective cameras, shadows and epipolar geometry as one
//! expression with holes. A camera is `(centre & Point) ^ screen`: join the point with the centre
//! into a ray, meet the ray with the screen. With the point slot open it is a map `Point <-
//! Point` (the projection matrix); with a direction for the centre it is the orthographic
//! shadow of a sun; with the corner bound and the light left open it is the map from light
//! positions to that corner's shadow. A line in the slot gives the camera for lines, and two
//! cameras' rays with both point slots open give the correspondence form whose zeros are
//! matching image points (the fundamental matrix). The animation moves the point light round
//! its circle (the corner's shadow runs along the trail the reopened slot drew), and turns the
//! subject in front of the stereo rig, whose epipolar lines follow.

use gax_numga_examples::canvas::{mix, srgb};
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Marker, Rgb, Scene3, backdrop, caption, palette, run,
};

mod projection {
    use gax::Unit;
    use gax::pga3d::{Line, Motor, Plane, Point, Pseudoscalar};

    pub type P = Point<(), f64>;
    pub type Pl = Plane<(), f64>;
    pub type L = Line<(), f64>;
    pub type M = Unit<Motor<(), f64>>;
    /// A camera, or a shadow: `Point <- Point`.
    pub type Camera = Point<(Point,), f64>;
    /// The camera for lines.
    pub type LineCamera = Line<(Line,), f64>;
    /// The bilinear form on two cameras' image points that vanishes on corresponding pairs.
    pub type Correspondence = Pseudoscalar<(Point, Point), f64>;

    /// Which corner pairs of [`cube`] are joined by an edge.
    pub const CUBE_EDGES: [[usize; 2]; 12] = [
        [0, 1],
        [1, 3],
        [3, 2],
        [2, 0],
        [4, 5],
        [5, 7],
        [7, 6],
        [6, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ];

    /// The finite point `(x, y, z)`.
    pub fn point(c: [f64; 3]) -> P {
        Point::xyz(c[0], c[1], c[2])
    }

    /// The ideal point in direction `(x, y, z)`.
    pub fn direction(c: [f64; 3]) -> P {
        Point::direction(c[0], c[1], c[2])
    }

    /// The origin.
    pub fn origin() -> P {
        point([0.0; 3])
    }

    /// `n` points round the unit circle in the `xy` plane, centred on the origin: `(1, 0, 0)`
    /// turned about `z`.
    pub fn circle(n: usize) -> Vec<P> {
        (0..n)
            .map(|k| {
                let t = core::f64::consts::TAU * k as f64 / n as f64;
                Motor::rotation_about(0.0, 0.0, 1.0, t) >> point([1.0, 0.0, 0.0])
            })
            .collect()
    }

    /// The eight corners of an axis-aligned cube centred on the origin, corner `4 x + 2 y + z`
    /// at the signs of `x`, `y`, `z`.
    pub fn cube(size: f64) -> [P; 8] {
        core::array::from_fn(|k| {
            let s = |bit: usize| {
                if k >> bit & 1 == 1 {
                    size / 2.0
                } else {
                    -size / 2.0
                }
            };
            point([s(2), s(1), s(0)])
        })
    }

    /// The camera (or shadow) map with centre `centre` onto `screen`: join, then meet.
    pub fn camera(centre: P, screen: Pl) -> Camera {
        (centre & Point::slot()) ^ screen
    }

    /// A body's shadows under a point light and the sun, and one corner's shadow as the light
    /// moves.
    pub struct Shadows {
        pub spot: [P; 8],
        pub sun: [P; 8],
        pub trail: Vec<P>,
    }

    /// Cast a body's shadows, then reopen the light slot to sweep one corner's shadow.
    pub fn shadows(body: &[P; 8], ground: Pl, light: P, sun: P, corner: P, path: &[P]) -> Shadows {
        // The light joined with the open point, met with the ground: whether the light is a
        // finite point or a direction at infinity, the expression is the same.
        let spotlight = camera(light, ground);
        let sunlight = camera(sun, ground);
        // Bind the corner and leave the light open: the corner's shadow is a linear map of the
        // light, and a whole path of lights binds at once.
        let shadow_of_corner: Camera = (Point::slot() & corner) ^ ground;
        Shadows {
            spot: body.map(|p| spotlight.of(p)),
            sun: body.map(|p| sunlight.of(p)),
            trail: path.iter().map(|l| shadow_of_corner.of(*l)).collect(),
        }
    }

    /// Two posed copies of one rig over a subject.
    pub struct Stereo {
        pub image_1: [P; 8],
        pub image_2: [P; 8],
        pub epipole_2: P,
        pub epipolar_lines_2: [L; 8],
        pub correspondence: Correspondence,
    }

    /// Move the rig's centre and screen into two poses and relate their images.
    pub fn stereo(subject: &[P; 8], centre: P, screen: Pl, rig_1: M, rig_2: M) -> Stereo {
        let camera = camera(centre, screen);
        // The same join-then-meet with a line in the slot is the camera for lines.
        let line_camera: LineCamera = (centre & Line::slot()) ^ screen;
        // The camera moves as a map: pull the world back into the rig's frame, push the image
        // forward.
        let camera_1 = rig_1 >> camera.of(rig_1 << Point::slot());
        let camera_2 = rig_2 >> camera.of(rig_2 << Point::slot());
        let line_camera_2 = rig_2 >> line_camera.of(rig_2 << Line::slot());
        let (centre_1, centre_2) = (rig_1 >> centre, rig_2 >> centre);
        let image_1 = subject.map(|p| camera_1.of(p));
        let image_2 = subject.map(|p| camera_2.of(p));
        // The epipole is the image of the other centre; an image point's epipolar line is the
        // image of its ray.
        let epipole_2 = camera_2.of(centre_1);
        let epipolar_lines_2 = image_1.map(|i| line_camera_2.of(centre_1 & i));
        // Two image points correspond if their rays meet, and two lines meet if their wedge
        // vanishes: with both point slots open, a bilinear form.
        let correspondence = (centre_1 & Point::slot()) ^ (centre_2 & Point::slot());
        Stereo {
            image_1,
            image_2,
            epipole_2,
            epipolar_lines_2,
            correspondence,
        }
    }

    /// A rig's screen coordinates of an image point: pulled back into the rig's frame.
    pub fn screen_coordinates(rig: M, image: P) -> [f64; 2] {
        let [x, y, _] = (rig << image).to_euclidean();
        [x, y]
    }

    /// A line on a rig's screen clipped at `x = ±half_width`: the screen coordinates of its
    /// meets with the two planes.
    pub fn screen_line(rig: M, line: L, half_width: f64) -> [[f64; 2]; 2] {
        [-half_width, half_width]
            .map(|x| screen_coordinates(rig, line ^ (rig >> Plane::new(1.0, 0.0, 0.0, -x))))
    }

    /// The scene at phase `t` (radians): at 0 it is numga's figure.
    pub struct Scene {
        pub body: [P; 8],
        pub light: P,
        pub sun: P,
        pub path: Vec<P>,
        pub cast: Shadows,
        pub rig_1: M,
        pub rig_2: M,
        pub views: Stereo,
    }

    pub fn scene(t: f64) -> Scene {
        let body = cube(1.0).map(|p| Motor::translation(0.0, 0.0, 1.5) >> p);
        let ground = Plane::new(0.0, 0.0, 1.0, 0.0);
        let sun = direction([-1.0, 0.6, -2.5]);
        // The light's path: the unit circle carried to (0, -1, 4), 64 lights along it; the
        // light moves round it, starting at numga's (1, -1, 4).
        let lift = Motor::translation(0.0, -1.0, 4.0);
        let path: Vec<P> = circle(64).into_iter().map(|p| lift >> p).collect();
        let light = lift >> (Motor::rotation_about(0.0, 0.0, 1.0, t) >> point([1.0, 0.0, 0.0]));
        // Two copies of the rig, 0.6 either side of the origin along x and turned 0.12 rad
        // about y so that they converge, looking along +z at the screen z = 1.
        let screen = Plane::new(0.0, 0.0, 1.0, -1.0);
        let rig_1 = Motor::translation(-0.6, 0.0, 0.0) * Motor::rotation_about(0.0, 1.0, 0.0, 0.12);
        let rig_2 = Motor::translation(0.6, 0.0, 0.0) * Motor::rotation_about(0.0, 1.0, 0.0, -0.12);
        // The subject, 5 ahead, turning about its vertical (the screen's y) and nodding.
        let pose = Motor::translation(0.0, 0.0, 5.0)
            * Motor::rotation_about(0.0, 1.0, 0.0, t)
            * Motor::rotation_about(1.0, 0.0, 0.0, 0.25 * t.sin());
        let subject = cube(1.6).map(|p| pose >> p);
        let cast = shadows(&body, ground, light, sun, body[7], &path);
        let views = stereo(&subject, origin(), screen, rig_1, rig_2);
        Scene {
            body,
            light,
            sun,
            path,
            cast,
            rig_1,
            rig_2,
            views,
        }
    }
}

use projection::*;

const SECONDS: f32 = 10.0;

fn sky() -> Rgb {
    srgb(0.22, 0.74, 0.97)
}

fn amber() -> Rgb {
    srgb(0.98, 0.75, 0.14)
}

fn violet() -> Rgb {
    srgb(0.66, 0.33, 0.97)
}

fn rose() -> Rgb {
    srgb(0.96, 0.25, 0.37)
}

fn f3(p: P) -> [f32; 3] {
    p.to_euclidean().map(|v| v as f32)
}

/// The shadow scene in 3D: the body, both lights, both shadows and the corner's trail.
fn shadow_scene(c: &mut Canvas, t: f32, sc: &Scene) {
    let turn = 0.25 * (core::f32::consts::TAU * t / SECONDS).sin();
    let cam = Camera::orbit(
        c.width,
        c.height,
        [0.0, 0.3, 1.6],
        15.0,
        (-60f32).to_radians() + turn,
        24f32.to_radians(),
        Lens::Perspective(0.5),
    );
    let mut s = Scene3::new(cam);
    for k in 0..=6 {
        let g = -3.0 + k as f32;
        let line = mix(palette::grid(), palette::ink(), 0.15);
        s.seg([g, -3.0, 0.0], [g, 3.0, 0.0], 1.0, line, 0.7);
        s.seg([-3.0, g, 0.0], [3.0, g, 0.0], 1.0, line, 0.7);
    }
    let body = sc.body.map(f3);
    let spot = sc.cast.spot.map(f3);
    let sun = sc.cast.sun.map(f3);
    for [a, b] in CUBE_EDGES {
        s.seg(body[a], body[b], 2.0, sky(), 1.0);
        s.seg(spot[a], spot[b], 1.5, amber(), 1.0);
        // The sun's shadow, dashed.
        for k in 0..6 {
            let (u, v) = (k as f32 / 6.0, (k as f32 + 0.55) / 6.0);
            let at = |w: f32| core::array::from_fn(|i| sun[a][i] + (sun[b][i] - sun[a][i]) * w);
            s.seg(at(u), at(v), 1.5, violet(), 1.0);
        }
    }
    // The corner's shadow as the light moves round its path: the reopened slot, bound to every
    // light of the path at once.
    let mut trail: Vec<[f32; 3]> = sc.cast.trail.iter().map(|p| f3(*p)).collect();
    trail.push(trail[0]);
    for (k, w) in trail.windows(2).enumerate() {
        if k % 2 == 0 {
            s.seg(w[0], w[1], 1.2, amber(), 0.8);
        }
    }
    let light = f3(sc.light);
    let path: Vec<[f32; 3]> = sc
        .path
        .iter()
        .chain(&sc.path[..1])
        .map(|p| f3(*p))
        .collect();
    s.polyline(&path, 1.0, mix(amber(), palette::bottom(), 0.5), 0.8);
    // The ray from the light through the corner to its shadow.
    s.seg(light, spot[7], 1.0, amber(), 0.5);
    s.dot(light, Marker::Dot, 11.0, amber());
    s.dot(spot[7], Marker::Dot, 6.0, amber());
    // The sun's direction, an arrow at (-2, 2, 4).
    let d = sc.sun.c;
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    s.arrow(
        [-2.0, 2.0, 4.0],
        [d[0] / n, d[1] / n, d[2] / n].map(|v| v as f32),
        2.0,
        9.0,
        violet(),
    );
    s.draw(c);
}

/// One rig's screen: the subject's edges, and for the second camera the epipolar lines of the
/// first camera's corners through their matches.
fn screen_panel(
    c: &mut Canvas,
    rect: [f32; 4],
    title: &str,
    rig: M,
    image: &[P; 8],
    lines: Option<&[L; 8]>,
) {
    let half = 0.5;
    let ax = Axes::equal(rect, [0.0, 0.0], half as f32);
    let xy = |p: P| screen_coordinates(rig, p).map(|v| v as f32);
    ax.clip(c);
    if let Some(lines) = lines {
        for l in lines {
            let [a, b] = screen_line(rig, *l, half).map(|p| p.map(|v| v as f32));
            ax.line(c, a, b, 1.0, rose(), 0.8);
        }
    }
    for [a, b] in CUBE_EDGES {
        ax.line(c, xy(image[a]), xy(image[b]), 2.0, sky(), 1.0);
    }
    if lines.is_some() {
        let pts: Vec<[f32; 2]> = image.iter().map(|p| xy(*p)).collect();
        ax.scatter(c, &pts, Marker::Dot, 6.0, rose(), 1.0);
    }
    c.unclip();
    ax.frame(c, title, "", "");
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.width, c.height);
    let phase = f64::from(t / SECONDS) * core::f64::consts::TAU;
    let sc = scene(phase);
    // The shadows on the left, on a canvas of their own.
    let left = (w as f32 * 0.42) as usize;
    let mut sub = Canvas::new(left, h);
    backdrop(&mut sub);
    shadow_scene(&mut sub, t, &sc);
    let (wf, hf) = (w as f32, h as f32);
    let s = (hf / 40.0).clamp(7.0, 13.0);
    sub.text(
        "SHADOWS: (LIGHT JOIN POINT) MEET GROUND",
        s,
        hf - s * 4.2,
        s,
        palette::ink(),
        Align::Left,
    );
    for (k, (label, col)) in [
        ("POINT LIGHT, ITS SHADOW", amber()),
        ("SUN, ITS SHADOW (DASHED)", violet()),
        (
            "CORNER'S SHADOW AS THE LIGHT MOVES",
            mix(amber(), palette::ink(), 0.4),
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let y = hf - s * (2.8 - 1.2 * k as f32);
        sub.line([s, y - s * 0.3], [s * 2.4, y - s * 0.3], 2.0, col, 1.0);
        sub.text(label, s * 3.0, y, s * 0.75, palette::ink(), Align::Left);
    }
    c.blit(&sub, 0, 0);
    // The two cameras on the right.
    let panel = ((wf - left as f32) / 2.0 - wf * 0.035).min(hf * 0.62);
    let top = (hf - panel) * 0.55;
    let x0 = left as f32 + wf * 0.035;
    let rect_1 = [x0, top, x0 + panel, top + panel];
    let x1 = x0 + panel + wf * 0.035;
    let rect_2 = [x1, top, x1 + panel, top + panel];
    let v = &sc.views;
    screen_panel(c, rect_1, "CAMERA 1", sc.rig_1, &v.image_1, None);
    screen_panel(
        c,
        rect_2,
        "CAMERA 2: EPIPOLAR LINES OF CAMERA 1",
        sc.rig_2,
        &v.image_2,
        Some(&v.epipolar_lines_2),
    );
    // Off the screen, to the left: where camera 2 sees camera 1's centre, and the form on the
    // matched pairs.
    let [ex, _] = screen_coordinates(sc.rig_2, v.epipole_2);
    let worst = v
        .image_1
        .iter()
        .zip(&v.image_2)
        .map(|(a, b)| v.correspondence.of(*a).of(*b).c[0].abs())
        .fold(0.0, f64::max);
    for (k, text) in [
        format!("EPIPOLE OF CAMERA 2 AT X = {ex:.2}"),
        format!("CORRESPONDENCE FORM ON MATCHES: {worst:.0E}"),
    ]
    .iter()
    .enumerate()
    {
        c.text(
            text,
            x0,
            top + panel + s * (3.4 + 1.4 * k as f32),
            s * 0.8,
            mix(palette::grid(), palette::ink(), 0.5),
            Align::Left,
        );
    }
    caption(
        c,
        "PROJECTION: ONE EXPRESSION, (CENTRE JOIN POINT) MEET SCREEN",
        "CAMERAS, SHADOWS AND EPIPOLAR GEOMETRY BY WHICH SLOT IS LEFT OPEN (PGA3D)",
    );
}

fn main() {
    run(Anim::new("projection", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::projection::*;
    use gax::pga3d::{Line, Motor, Plane, Point};

    fn screen() -> Pl {
        Plane::new(0.0, 0.0, 1.0, -1.0)
    }

    fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    /// A pinhole at the origin with screen `z = 1` images `(x, y, z)` to `(x/z, y/z, 1)`.
    #[test]
    fn pinhole_image_is_perspective_division() {
        let camera: Camera = (origin() & Point::slot()) ^ screen();
        for (p, want) in [
            ([1.0, 2.0, 4.0], [0.25, 0.5, 1.0]),
            ([-3.0, 0.5, 2.0], [-1.5, 0.25, 1.0]),
            ([0.2, -0.4, 0.5], [0.4, -0.8, 1.0]),
        ] {
            assert!(close(&camera.of(point(p)).to_euclidean(), &want, 1e-14));
        }
    }

    /// The centre has no image, screen points are fixed, and the map is projectively
    /// idempotent: a rank-three projection.
    #[test]
    fn camera_is_a_rank_three_projection() {
        let centre = point([0.3, -0.2, -1.0]);
        let camera = camera(centre, screen());
        let (_, sigma, _) = camera.svd();
        assert_eq!(sigma.iter().filter(|s| **s > 1e-9 * sigma[0]).count(), 3);
        assert!(camera.of(centre).c.iter().all(|v| v.abs() < 1e-14));
        for p in [[0.7, 0.1, 1.0], [-2.0, 3.0, 1.0]] {
            assert!(close(&camera.of(point(p)).to_euclidean(), &p, 1e-14));
        }
        // Twice is once, up to scale.
        let twice = camera.of(camera);
        let (a, b): (Vec<f64>, Vec<f64>) = (
            twice.c.iter().flatten().copied().collect(),
            camera.c.iter().flatten().copied().collect(),
        );
        let support: Vec<usize> = (0..16).filter(|&i| b[i].abs() > 1e-9).collect();
        let scale = support.iter().map(|&i| a[i] / b[i]).sum::<f64>() / support.len() as f64;
        assert!(a.iter().zip(&b).all(|(x, y)| (x - y * scale).abs() < 1e-14));
    }

    /// Binding the centre and the screen of the open ternary projector gives the camera.
    #[test]
    fn ternary_projector_binds_to_camera() {
        let projector: Point<(Point, Point, Plane), f64> =
            (Point::slot() & Point::slot()) ^ Plane::slot();
        let centre = point([0.3, -0.2, -1.0]);
        let world = point([1.0, 2.0, 4.0]);
        let camera = camera(centre, screen());
        let bound: Camera = projector.of(centre).at::<1>().of(screen());
        let (a, b): (Vec<f64>, Vec<f64>) = (
            bound.c.iter().flatten().copied().collect(),
            camera.c.iter().flatten().copied().collect(),
        );
        assert!(close(&a, &b, 1e-14));
        let all_three = projector.of(centre).of(world).of(screen());
        assert!(close(&all_three.c, &camera.of(world).c, 1e-14));
    }

    /// A centre at infinity projects along a fixed direction: the shadows of a distant sun.
    #[test]
    fn ideal_centre_is_orthographic() {
        let ground = Plane::new(0.0, 0.0, 1.0, 0.0);
        let down = camera(direction([0.0, 0.0, -1.0]), ground);
        assert!(close(
            &down.of(point([1.0, 2.0, 4.0])).to_euclidean(),
            &[1.0, 2.0, 0.0],
            1e-14
        ));
        assert!(close(
            &down.of(point([-3.0, 0.5, 2.0])).to_euclidean(),
            &[-3.0, 0.5, 0.0],
            1e-14
        ));
        let slanted = camera(direction([1.0, 0.0, -1.0]), ground);
        assert!(close(
            &slanted.of(point([0.0, 0.0, 2.0])).to_euclidean(),
            &[2.0, 0.0, 0.0],
            1e-14
        ));
    }

    /// One camera per centre (numga binds a batch at once).
    #[test]
    fn batched_centres_give_batched_cameras() {
        let world = point([1.0, 2.0, 4.0]);
        for (c, want) in [
            ([-0.5, 0.0, 0.0], [-0.125, 0.5, 1.0]),
            ([0.5, 0.0, 0.0], [0.625, 0.5, 1.0]),
        ] {
            assert!(close(
                &camera(point(c), screen()).of(world).to_euclidean(),
                &want,
                1e-14
            ));
        }
    }

    /// The translator adds its displacement, and the rotation about the join of the origin and
    /// `+y` is right handed. gax's join orients that axis opposite to numga's (ADR-009), so
    /// numga's `exp(axis π/4)` is gax's `exp(-axis π/4)`, `Motor::rotation(axis, π/2)`.
    #[test]
    fn bivector_exponentials_translate_and_rotate() {
        // numga's `xw 0.5 - yw + zw 0.25`, with `xw = e1 e0 = -e01`.
        let b = Line::new(0.0, 0.0, 0.0, -0.5, 1.0, -0.25);
        let moved = b.exp() >> point([1.0, 2.0, 3.0]);
        assert!(close(&moved.to_euclidean(), &[2.0, 0.0, 3.5], 1e-14));
        let y_axis = (origin() & direction([0.0, 1.0, 0.0]))
            .normalized()
            .into_inner();
        let turned = y_axis.gp(-core::f64::consts::FRAC_PI_4).exp() >> point([1.0, 0.0, 0.0]);
        assert!(close(&turned.to_euclidean(), &[0.0, 0.0, -1.0], 1e-8));
        let rotation = Motor::rotation(y_axis, core::f64::consts::FRAC_PI_2);
        assert!(close(
            &(rotation >> point([1.0, 0.0, 0.0])).to_euclidean(),
            &[0.0, 0.0, -1.0],
            1e-15
        ));
    }

    /// Moving the camera map by a motor equals building it from the moved centre and screen.
    #[test]
    fn moving_the_rig_equals_moving_centre_and_screen() {
        let y_axis = origin() & direction([0.0, 1.0, 0.0]);
        let motor = Motor::translation(0.4, -0.3, 2.0) * Motor::rotation(y_axis, 0.7);
        let camera = camera(origin(), screen());
        let moved = motor >> camera.of(motor << Point::slot());
        let rebuilt = ((motor >> origin()) & Point::slot()) ^ (motor >> screen());
        let (a, b): (Vec<f64>, Vec<f64>) = (
            moved.c.iter().flatten().copied().collect(),
            rebuilt.c.iter().flatten().copied().collect(),
        );
        assert!(close(&a, &b, 1e-14), "{a:?} {b:?}");
        let world = point([1.0, 2.0, 4.0]);
        let local = camera.of(motor << world).to_euclidean();
        assert!(close(
            &screen_coordinates(motor, moved.of(world)),
            &local[..2],
            1e-14
        ));
    }

    /// The image of the line through two points is the line through their images.
    #[test]
    fn line_camera_commutes_with_join() {
        let centre = point([0.3, -0.2, -1.0]);
        let camera = camera(centre, screen());
        let line_camera: LineCamera = (centre & Line::slot()) ^ screen();
        let (a, b) = (point([1.0, 0.0, 2.0]), point([0.0, 1.0, 3.0]));
        let imaged_join = line_camera.of(a & b);
        let joined_images = camera.of(a) & camera.of(b);
        let k = (0..6)
            .find(|&i| joined_images.c[i].abs() > 1e-9)
            .expect("a nonzero coefficient");
        let scale = imaged_join.c[k] / joined_images.c[k];
        assert!(close(
            &imaged_join.c,
            &joined_images.c.map(|v| v * scale),
            1e-14
        ));
    }

    /// Corresponding image points annihilate the fundamental form, which has rank 2; the
    /// epipole spans its kernel, and the epipolar lines pass through the matches and the epipole.
    #[test]
    fn fundamental_form_and_epipolar_geometry() {
        let (centre_1, centre_2) = (point([-0.6, 0.0, 0.0]), point([0.6, 0.1, -0.2]));
        let (camera_1, camera_2) = (camera(centre_1, screen()), camera(centre_2, screen()));
        let form: Correspondence = (centre_1 & Point::slot()) ^ (centre_2 & Point::slot());
        // The form's matrix, read as a map for its singular values.
        let (_, sigma, _) = Camera::from_coeffs(form.c[0]).svd();
        assert_eq!(sigma.iter().filter(|s| **s > 1e-9 * sigma[0]).count(), 2);
        let world = [[1.0, 2.0, 4.0], [-3.0, 0.5, 2.0], [0.2, -0.4, 3.0]].map(point);
        let image_1 = world.map(|p| camera_1.of(p));
        let image_2 = world.map(|p| camera_2.of(p));
        for (a, b) in image_1.iter().zip(&image_2) {
            assert!(form.of(*a).of(*b).c[0].abs() < 1e-13);
        }
        assert!(form.of(image_1[0]).of(image_2[2]).c[0].abs() > 1e-3);
        let epipole_2 = camera_2.of(centre_1);
        assert!(
            form.at::<1>()
                .of(epipole_2)
                .c
                .iter()
                .flatten()
                .all(|v| v.abs() < 1e-14)
        );
        let line_camera_2: LineCamera = (centre_2 & Line::slot()) ^ screen();
        for (i1, i2) in image_1.iter().zip(&image_2) {
            let line = line_camera_2.of(centre_1 & *i1);
            assert!((line & *i2).c.iter().all(|v| v.abs() < 1e-12));
            assert!((line & epipole_2).c.iter().all(|v| v.abs() < 1e-13));
        }
    }

    /// Reopening the light slot reproduces the corner's shadow under the bound light.
    #[test]
    fn shadow_trail_agrees_with_the_body_shadow() {
        let body = cube(1.0);
        let ground = Plane::new(0.0, 0.0, 1.0, 0.0);
        let light = point([1.0, -1.0, 4.0]);
        let sun = direction([-1.0, 0.6, -2.5]);
        let cast = shadows(&body, ground, light, sun, body[7], &[light]);
        assert!(close(&cast.trail[0].c, &cast.spot[7].c, 1e-14));
    }

    /// Corresponding images annihilate the form, and the epipolar lines meet their points; at
    /// every phase of the animation.
    #[test]
    fn stereo_correspondence_and_epipolar_lines_vanish() {
        for k in 0..8 {
            let sc = scene(core::f64::consts::TAU * k as f64 / 8.0);
            let v = &sc.views;
            for (a, b) in v.image_1.iter().zip(&v.image_2) {
                assert!(v.correspondence.of(*a).of(*b).c[0].abs() < 1e-12);
            }
            assert!(
                v.correspondence
                    .at::<1>()
                    .of(v.epipole_2)
                    .c
                    .iter()
                    .flatten()
                    .all(|x| x.abs() < 1e-12)
            );
            for (l, p) in v.epipolar_lines_2.iter().zip(&v.image_2) {
                assert!((*l & *p).c.iter().all(|x| x.abs() < 1e-12));
            }
        }
    }

    /// numga's scenario at phase 0: shadows, the two images and the epipole.
    #[test]
    fn the_scenario_agrees_with_numga() {
        let sc = scene(0.0);
        assert!(close(&sc.body[7].to_euclidean(), &[0.5, 0.5, 2.0], 1e-14));
        assert!(close(
            &sc.cast.spot[7].to_euclidean(),
            &[0.0, 2.0, 0.0],
            1e-14
        ));
        assert!(close(
            &sc.cast.sun[7].to_euclidean(),
            &[-0.3, 0.98, 0.0],
            1e-14
        ));
        assert!(close(
            &sc.cast.trail[0].to_euclidean(),
            &[0.0, 2.0, 0.0],
            1e-14
        ));
        assert!(close(
            &sc.cast.trail[16].to_euclidean(),
            &[1.0, 1.0, 0.0],
            1e-12
        ));
        let v = &sc.views;
        let want_1 = [
            [-0.169169736004, -0.192963870163],
            [-0.155709521726, -0.139510202261],
            [-0.169169736004, 0.192963870163],
        ];
        let want_2 = [
            [-0.204533171375, -0.184442574942],
            [-0.11738348487, -0.135000882728],
            [-0.204533171375, 0.184442574942],
        ];
        for k in 0..3 {
            assert!(close(
                &screen_coordinates(sc.rig_1, v.image_1[k]),
                &want_1[k],
                1e-11
            ));
            assert!(close(
                &screen_coordinates(sc.rig_2, v.image_2[k]),
                &want_2[k],
                1e-11
            ));
        }
        assert!(close(
            &screen_coordinates(sc.rig_2, v.epipole_2),
            &[-8.293294880597, 0.0],
            1e-10
        ));
        assert!(close(
            &(sc.rig_1 >> point([0.0, 0.0, 1.0])).to_euclidean(),
            &[-0.480287792711, 0.0, 0.992808635854],
            1e-11
        ));
    }

    /// The moving light's shadow of the corner runs along the trail.
    #[test]
    fn the_corner_shadow_runs_along_the_trail() {
        for k in 0..64 {
            let sc = scene(core::f64::consts::TAU * k as f64 / 64.0);
            let [x, y, _] = sc.cast.spot[7].to_euclidean();
            let [tx, ty, _] = sc.cast.trail[k].to_euclidean();
            assert!((x - tx).abs() < 1e-12 && (y - ty).abs() < 1e-12);
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
