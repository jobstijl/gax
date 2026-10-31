//! numga's `fluids/smoke_rings`: the smoke ring flow of a closed polygon and the Hashimoto
//! surface it sweeps, Hoffmann's discrete-time flow by Bäcklund transformations, a soliton on a
//! straight vortex line, and vortex rings leapfrogging by the Biot–Savart law (VGA3D).
//!
//! A thin vortex filament, the core of a smoke ring, moves along its binormal at a speed equal to
//! its curvature: the local induction approximation, the Hashimoto or smoke ring flow, under
//! which a circle flies along its axis without changing shape. On a polygon (Hoffmann, 2000) the
//! points and edges are antivectors; at a vertex the commutator of the incoming and outgoing
//! edges is the bivector of the turn, and over the lengths plus the edges' pairing it is the
//! tangent of half the turning angle, the turn's Cayley parameter; times the circulation (an
//! intrinsic scalar) it is a point, the step along the binormal. The velocity is square to both
//! edges, so every edge keeps its length, and the flow keeps the polygon's vector area, the sum
//! of each point joined with the next. The surface a curve sweeps is its Hashimoto surface.
//!
//! Kept whole, the Biot–Savart integral over straight segments gives at a point the commutator
//! of its separation from each segment's middle with the segment, over the softened cube of the
//! separation: two coaxial rings moved this way leapfrog, the rear one shrinking and passing
//! through the front one.
//!
//! Time can be discrete too, by Hoffmann's Bäcklund transformation: across an edge the next
//! offset is the current one turned about the quad's diagonal, the diagonal plus the parameter
//! being a quaternion. On the sphere of unit offsets the fold is a Lorentz transformation, a
//! boost along the edge and a turn about it, acting on the offset's null paravector; a lap of
//! them carries any start to the offset that comes round to itself, and relabelled by one
//! vertex the transformation is a time step that keeps every edge exactly of unit length and
//! moves a ring exactly rigidly. One transformation of a straight line makes Hasimoto's soliton,
//! a loop running along the line, through which the tangent turns once around.
//!
//! The animation shows numga's figures and animations together: the wavy ring's Hashimoto
//! surface, a trefoil of vortex flowing, the leapfrogging rings over the tubes they sweep (their
//! near halves cut away), the discrete-time surface, the soliton running over the surface it
//! sweeps, and the turning angles along it through time. Curves are coloured by their speed.

// The checks' quantities (heights, radii, impulses, the passes) serve numga's tests alone.
#[cfg_attr(not(test), allow(dead_code))]
mod rings;
#[cfg(test)]
mod tests;

use gax_numga_examples::scene3::{Scene3, panel3};
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Light, ORIGIN3, Point2, Point3, Rect, backdrop, caption,
    colormap, palette, reach3, run_with,
};
use rings::*;

/// The loop: numga's trefoil, 91 frames of 40 ms; the leapfrog and the soliton stretched to it.
const SECONDS: f32 = 3.64;
/// numga's view: elevation 50 degrees, the azimuth 40 degrees from looking along `+y`.
const ELEVATION: f32 = 50.0;
const AZIMUTH: f32 = -50.0;
/// The leapfrog's frames: every second step.
const FROG_STRIDE: usize = 2;

/// Everything computed once.
struct Scenes {
    surface: Vec<Vec<Point>>,
    knot: Vec<Vec<Point>>,
    knot_speeds: Vec<Vec<f64>>,
    discrete: Vec<Vec<Point>>,
    soliton: Vec<Vec<Point>>,
    soliton_speeds: Vec<Vec<f64>>,
    soliton_angles: Vec<Vec<f64>>,
    frogs: Vec<Vec<Vec<Point>>>,
    frog_speeds: Vec<Vec<Vec<f64>>>,
}

fn speeds(rate: &[Point]) -> Vec<f64> {
    rate.iter().map(|v| v.norm()).collect()
}

fn scenes() -> Scenes {
    let knot = knot();
    let knot_speeds = knot
        .iter()
        .map(|c| speeds(&velocity(c, circulation())))
        .collect();
    let soliton = soliton();
    let frogs: Vec<Vec<Vec<Point>>> = leapfrog().into_iter().step_by(FROG_STRIDE).collect();
    let frog_speeds = frogs
        .iter()
        .map(|rings| pushed(rings).iter().map(|r| speeds(r)).collect())
        .collect();
    Scenes {
        surface: wavy_surface(),
        knot,
        knot_speeds,
        discrete: discrete().into_iter().map(|[_, wavy]| wavy).collect(),
        soliton_speeds: soliton.iter().map(|c| line_speeds(c)).collect(),
        soliton_angles: soliton.iter().map(|c| turn_angles(c)).collect(),
        soliton,
        frogs,
        frog_speeds,
    }
}

/// A point drawn: where its dual direction reaches.
fn place(p: Point) -> Point3 {
    reach3(p.hodge())
}

/// The surface's light from the first time to the last.
fn age_light(share: f32) -> Light {
    palette::YELLOW
        .faded(0.35)
        .mix_light(palette::BLUE.faded(0.4), share)
}

/// A speed's light, against the fastest: viridis from rest to fast.
fn speed_light(speed: f64, fastest: f64) -> Light {
    colormap::viridis(0.3 + 0.48 * (speed / fastest) as f32)
}

/// The mean of all the points, and the largest distance of any from it.
fn frame_of<'a>(points: impl Iterator<Item = &'a Point> + Clone) -> (Point, f32) {
    let count = points.clone().count() as f64;
    let mean = points.clone().copied().sum::<Point>() * (1.0 / count);
    let reach = points
        .map(|p| (*p - mean).norm() as f32)
        .fold(0.0, f32::max);
    (mean, reach)
}

/// A panel's camera: numga's view, centred on `centre`, seeing `reach` whole.
fn camera(rect: Rect, reach: f32) -> Camera {
    let scale = 0.5 * rect.height().min(rect.width()) / reach;
    Camera::parallel(rect, scale, AZIMUTH.to_radians(), ELEVATION.to_radians())
}

/// Strips between successive curves: two triangles to each quad between an edge at one time
/// and the same edge at the next, lit, coloured from early to late, where `keep` says.
fn strips(
    s: &mut Scene3,
    curves: &[Vec<Point>],
    closed: bool,
    opacity: f32,
    light: impl Fn(usize) -> Light,
    keep: impl Fn(usize, usize) -> bool,
) {
    let n = curves[0].len();
    let edges = if closed { n } else { n - 1 };
    for (t, pair) in curves.windows(2).enumerate() {
        for k in 0..edges {
            if !keep(t, k) {
                continue;
            }
            let next = (k + 1) % n;
            let (a, b, c, d) = (
                place(pair[0][k]),
                place(pair[1][k]),
                place(pair[1][next]),
                place(pair[0][next]),
            );
            let lit = s.lit(a, b, c, light(t));
            s.quad(a, b, c, d, lit, opacity);
        }
    }
}

/// A curve as strokes, each segment coloured by the mean speed at its ends.
fn curve(s: &mut Scene3, points: &[Point], speeds: &[f64], fastest: f64, closed: bool, width: f32) {
    let n = points.len();
    let segments = if closed { n } else { n - 1 };
    for k in 0..segments {
        let next = (k + 1) % n;
        let speed = (speeds[k] + speeds[next]) / 2.0;
        s.seg(
            place(points[k]),
            place(points[next]),
            width,
            speed_light(speed, fastest),
        );
    }
}

/// Points moved by `-centre`.
fn moved(points: &[Vec<Point>], centre: Point) -> Vec<Vec<Point>> {
    points
        .iter()
        .map(|c| c.iter().map(|p| *p - centre).collect())
        .collect()
}

/// A swept surface, coloured from early to late.
fn draw_surface(c: &mut Canvas, rect: Rect, curves: &[Vec<Point>]) {
    let (centre, reach) = frame_of(curves.iter().flatten());
    let curves = moved(curves, centre);
    let last = (curves.len() - 1) as f32;
    panel3(c, rect, camera(rect, reach), |s| {
        strips(
            s,
            &curves,
            true,
            0.97,
            |t| age_light(t as f32 / last),
            |_, _| true,
        );
    });
}

/// The trefoil flowing at a frame.
fn draw_knot(c: &mut Canvas, rect: Rect, s: &Scenes, frame: usize) {
    let (centre, reach) = frame_of(s.knot.iter().flatten());
    let fastest = s.knot_speeds.iter().flatten().copied().fold(0.0, f64::max);
    let now = moved(&s.knot[frame..=frame], centre);
    panel3(c, rect, camera(rect, reach), |scene| {
        curve(scene, &now[0], &s.knot_speeds[frame], fastest, true, 2.2);
    });
}

/// The leapfrogging rings at a frame, over the far halves of the tubes they sweep.
fn draw_frogs(c: &mut Canvas, rect: Rect, s: &Scenes, frame: usize) {
    let (centre, reach) = frame_of(s.frogs.iter().flatten().flatten());
    let fastest = s
        .frog_speeds
        .iter()
        .flatten()
        .flatten()
        .copied()
        .fold(0.0, f64::max);
    let cam = camera(rect, reach);
    // The upright plane through the axis facing the viewer: the near halves are cut away.
    let toward = cam.eye().unitized() - ORIGIN3;
    let level = Point3::direction(toward.e032(), toward.e013(), 0.0);
    let cut = gax::pga3d::Plane::orthogonal_to(level);
    let facing = |p: Point| (cut & place(p)).s() < 0.0;
    let tubes = [palette::ORANGE.faded(0.25), palette::SKY.faded(0.25)];
    panel3(c, rect, cam, |scene| {
        for (r, tube) in tubes.iter().enumerate() {
            let sweep: Vec<Vec<Point>> = s
                .frogs
                .iter()
                .map(|rings| rings[r].iter().map(|p| *p - centre).collect())
                .collect();
            let far =
                |t: usize, k: usize| facing(sweep[t][k]) && facing(sweep[t + 1][(k + 1) % FROG]);
            strips(scene, &sweep, true, 0.6, |_| *tube, far);
        }
        let now = moved(&s.frogs[frame], centre);
        for (ring, speeds) in now.iter().zip(&s.frog_speeds[frame]) {
            curve(scene, ring, speeds, fastest, true, 2.4);
        }
    });
}

/// The soliton at a frame over the surface it has swept, framed on the vertices that move.
fn draw_soliton(c: &mut Canvas, rect: Rect, s: &Scenes, frame: usize) {
    let first = &s.soliton[0];
    let travel: Vec<f64> = (0..first.len())
        .map(|k| {
            s.soliton
                .iter()
                .map(|c| (c[k] - first[k]).norm())
                .fold(0.0, f64::max)
        })
        .collect();
    let most = travel.iter().copied().fold(0.0, f64::max);
    let moving: Vec<usize> = (0..first.len())
        .filter(|k| travel[*k] > 0.01 * most)
        .collect();
    let (lo, hi) = (moving[0], moving[moving.len() - 1]);
    let kept: Vec<Vec<Point>> = s.soliton.iter().map(|c| c[lo..=hi].to_vec()).collect();
    let (centre, reach) = frame_of(kept.iter().flatten());
    let kept = moved(&kept, centre);
    let fastest = s
        .soliton_speeds
        .iter()
        .flatten()
        .copied()
        .fold(0.0, f64::max);
    let last = (kept.len() - 1) as f32;
    panel3(c, rect, camera(rect, reach), |scene| {
        strips(
            scene,
            &kept[..=frame],
            false,
            0.6,
            |t| age_light(t as f32 / last),
            |_, _| true,
        );
        curve(
            scene,
            &kept[frame],
            &s.soliton_speeds[frame][lo..=hi],
            fastest,
            false,
            2.2,
        );
    });
}

/// The turning angle at every inner vertex of the soliton through time: vertices across, times
/// down, the angle as light, and the frame shown marked.
fn draw_turns(c: &mut Canvas, rect: Rect, s: &Scenes, frame: usize) {
    let (times, vertices) = (s.soliton_angles.len(), s.soliton_angles[0].len());
    let ax = Axes::new(rect, [0.0, vertices as f32], [times as f32, 0.0]);
    let largest = s
        .soliton_angles
        .iter()
        .flatten()
        .copied()
        .fold(0.0, f64::max);
    ax.image(c, 1, |q| {
        let [x, y] = q.to_euclidean();
        let (k, t) = (x as usize, y as usize);
        let angle = s.soliton_angles.get(t)?.get(k)?;
        Some(palette::BOTTOM.mix_light(palette::INK, (angle / largest) as f32))
    });
    let at = frame as f32 + 0.5;
    ax.line(
        c,
        Point2::xy(0.0, at),
        Point2::xy(vertices as f32, at),
        1.0,
        palette::ORANGE,
    );
    ax.frame(c, "", "VERTEX", "TIME");
}

fn draw(c: &mut Canvas, s: &Scenes, t: f32) {
    backdrop(c);
    let fraction = (t / SECONDS).rem_euclid(1.0);
    let at = |count: usize| ((fraction * count as f32) as usize).min(count - 1);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let body = Rect::new(0.0, h * 0.13, w, h);
    let cell = |column: usize, row: usize| {
        let top = body.part(0.0, row as f32 * 0.5, 1.0, row as f32 * 0.5 + 0.5);
        top.column(column, 3).inset(8.0, 16.0, 8.0, 4.0)
    };
    let label = |c: &mut Canvas, rect: Rect, text: &str| {
        let at = rect.top_middle() - Point2::direction(0.0, 4.0);
        c.text(text, at, 9.0, palette::INK, Align::Center);
    };
    draw_surface(c, cell(0, 0), &s.surface);
    label(c, cell(0, 0), "A WAVY RING'S HASHIMOTO SURFACE");
    draw_knot(c, cell(1, 0), s, at(s.knot.len()));
    label(c, cell(1, 0), "A TREFOIL OF VORTEX FLOWING");
    draw_frogs(c, cell(2, 0), s, at(s.frogs.len()));
    label(c, cell(2, 0), "TWO RINGS LEAPFROGGING (BIOT-SAVART)");
    draw_surface(c, cell(0, 1), &s.discrete);
    label(c, cell(0, 1), "DISCRETE TIME: BACKLUND STEPS");
    let soliton = at(s.soliton.len());
    draw_soliton(c, cell(1, 1), s, soliton);
    label(c, cell(1, 1), "A SOLITON RUNNING ALONG A LINE");
    let turns = cell(2, 1).inset(34.0, 16.0, 4.0, 26.0);
    draw_turns(c, turns, s, soliton);
    label(c, cell(2, 1), "ITS TURNS: ONCE AROUND");
    caption(
        c,
        "SMOKE RINGS: THE FLOW OF A VORTEX FILAMENT",
        "VELOCITY = 2 (S' X S) G / (|S||S'| + S.S') / MEAN EDGE: ALONG THE BINORMAL, EDGES KEPT (VGA3D)",
    );
}

fn main() {
    run_with(
        Anim::new("smoke_rings", SECONDS).size(960, 540),
        scenes,
        draw,
    );
}
