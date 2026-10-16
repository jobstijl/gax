//! numga's `mechanics/elastica`: the shapes of elastic rods loaded only at their ends, the
//! elastica of Euler and Kirchhoff, in PGA3D.
//!
//! A rod is a chain of cross-sections, each a rigid slice of the rod placed by a motor. With
//! loads only at the ends, the part of the rod beyond any cross-section is held in balance by
//! the end load and by the cross-section alone, so every cross-section carries the same load:
//! one line, fixed in space, a force along it or a couple. Seen from a cross-section,
//! `frame << load`, the load's moments about the section's own axes bend it, across and up,
//! and twist it, along the rod, each in proportion over the rod's rigidity against that turn:
//! the compliance, a map on lines. A cross-section moves to the next by a shift along the rod
//! and that turn, a line read as a motion; its exponential over a short piece is the motor
//! from one cross-section to the next, and their running product places the whole rod.
//!
//! In motion (the telephone cord), the rod is a chain of short rigid segments with mass, each
//! joined to the next by an elastic joint whose rest placement is the rod's own shape. A
//! joint's deviation from rest is a line; its energy is a quadratic form of that line, and the
//! load it puts on its two segments is the derivative of that energy, the stiffness applied to
//! the deviation through the derivative of the exponential. Each segment moves under the loads
//! of its two joints, its weight and some damping, by Euler's equations for a rigid body.
//!
//! In the notation of Kirchhoff's rod theory the cross-sections' turn per unit length is
//! `κ = B⁻¹ m`, with `B` the rigidities and `m` the moment carried, and the frames solve
//! `d/ds (R, r) = (R [κ]ₓ, R e₃)`.
//!
//! The animation: a strip curled by a couple at its end, from straight to a double ring (numga's
//! six curls faint behind); Euler's elastica between their pins on the line of thrust as the end
//! angle grows (numga's plate faint behind); Kirchhoff's four helices with a ribbon showing the
//! twist; a wire under a fixed couple with its tilt swept from square to the couple to along it,
//! from a ring to a straight twist, the camera turning; and a coiled telephone cord, hanging
//! under its own weight, its handset pulled out along the coil and let go. numga draws the cord
//! with `z` up, so its weight, along `-y`, pulls it sideways; here it hangs down.

mod rod;
mod scenes;

use gax::pga2d;
use gax::pga3d::Point;
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{
    Anim, Axes, Camera, Canvas, Lens, Marker, Rect, Scene3, backdrop, caption, colormap,
    label_under, palette, run, signal,
};
use rod::{Frame, P};
use scenes::*;
use std::f64::consts::TAU;
use std::sync::OnceLock;

/// One loop: three runs of the cord's 70 frames at numga's 40 ms; the curls, the sweep of the
/// end angle and the tilt of the uncoiling wire go out and back once.
const SECONDS: f32 = 3.0 * CORD_FRAMES as f32 * 0.04;
/// The spacing of the helices as a share of the widest, and their view's azimuth and
/// elevation in degrees (numga's).
const SPACING: f64 = 1.4;
const HELIX_VIEW: [f32; 2] = [-80.0, 12.0];
/// The uncoiling wire's elevation, the azimuths the camera turns through over the loop, and
/// the tilts sampled for its scale (numga's frames).
const UNCOIL_ELEVATION: f32 = 16.0;
const UNCOIL_AZIMUTHS: [f32; 2] = [-120.0, 60.0];
const UNCOIL_SAMPLES: usize = 90;
/// The cord's view, azimuth and elevation (numga's).
const CORD_VIEW: [f32; 2] = [-64.0, 20.0];
/// The margin around what a panel shows, as a share of it.
const MARGIN: f64 = 1.08;

/// A point of the plane of bending.
type P2 = pga2d::Point<(), f64>;

/// The box around points of space: its centre, and its half extents along `x`, `y` and `z`.
#[derive(Clone, Copy, Debug)]
struct Bounds {
    centre: P,
    half: [f64; 3],
}

fn bounds(points: impl IntoIterator<Item = P>) -> Bounds {
    let (mut low, mut high) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in points {
        for (i, v) in p.to_euclidean().into_iter().enumerate() {
            (low[i], high[i]) = (low[i].min(v), high[i].max(v));
        }
    }
    let [x, y, z] = [0, 1, 2].map(|i| (low[i] + high[i]) / 2.0);
    Bounds {
        centre: Point::xyz(x, y, z),
        half: [0, 1, 2].map(|i| (high[i] - low[i]) / 2.0),
    }
}

/// The points of a rod drawn in space: the edges of its ribbon.
fn edges(frames: &[Frame], width: f64) -> impl Iterator<Item = P> + '_ {
    frames.iter().flat_map(move |f| ribbon(*f, width))
}

/// What does not change over the loop: numga's six curls and Euler's plate in the plane of
/// bending, with the box the sweep of the elastica fills; the four helices set side by side and
/// their box; the box of the uncoiling wire over its sweep; and the cord, its weight down, at
/// each of its frames with frames drawn between its segments, and its box.
struct Scene {
    curls: Vec<Vec<P2>>,
    plate: Vec<Vec<P2>>,
    swept: Vec<P2>,
    helices: Vec<Vec<Frame>>,
    spacing: f64,
    helix_box: Bounds,
    uncoil_box: Bounds,
    cord: Vec<Vec<Frame>>,
    cord_box: Bounds,
}

fn scene() -> &'static Scene {
    static SCENE: OnceLock<Scene> = OnceLock::new();
    SCENE.get_or_init(|| {
        let helices: Vec<Vec<Frame>> = TILTS.iter().map(|tilt| coiled(*tilt, COILS)).collect();
        // The widest extent of a helix's ribbon across or up, numga's `ptp`.
        let widest = helices
            .iter()
            .flat_map(|frames| {
                let [x, y, _] = bounds(edges(frames, RIBBON)).half;
                [x, y]
            })
            .fold(0.0, f64::max);
        let spacing = SPACING * 2.0 * widest;
        let helix_box =
            bounds(helices.iter().enumerate().flat_map(|(k, frames)| {
                edges(frames, RIBBON).map(move |p| p + across(spacing, k))
            }));
        let sweep: Vec<Vec<Frame>> = (0..UNCOIL_SAMPLES)
            .map(|k| {
                coiled(
                    uncoiling_tilt(TAU * k as f64 / UNCOIL_SAMPLES as f64),
                    UNCOIL_TURNS,
                )
            })
            .collect();
        let uncoil_box = bounds(sweep.iter().flat_map(|frames| edges(frames, UNCOIL_RIBBON)));
        let cord: Vec<Vec<Frame>> = pulled()
            .iter()
            .map(|frames| rod::between(frames, DRAWN_BETWEEN))
            .collect();
        let view = weight_down();
        let cord_box = bounds(
            cord.iter()
                .flat_map(|frames| edges(frames, CORD_RIBBON).map(|p| view.of(p))),
        );
        let swept = SWEEP
            .iter()
            .flat_map(|angle| in_plane(&euler(*angle)))
            .collect();
        Scene {
            curls: CURLS
                .iter()
                .map(|turns| in_plane(&curling(*turns)))
                .collect(),
            plate: END_ANGLES
                .iter()
                .map(|angle| in_plane(&euler(*angle)))
                .collect(),
            swept,
            helices,
            spacing,
            helix_box,
            uncoil_box,
            cord,
            cord_box,
        }
    })
}

/// The shift of the `k`-th helix across, the four centred on the first's place.
fn across(spacing: f64, k: usize) -> P {
    Point::direction(spacing * (k as f64 - 1.5), 0.0, 0.0)
}

/// The plane of bending, a map on points: along the unbent rod (`z`) across, and up (`y`) up.
fn bending_plane() -> pga2d::Point<(Point,), f64> {
    let image = pga2d::Point::new;
    pga2d::Point::from_images([
        image(0.0, 0.0, 0.0),
        image(0.0, 1.0, 0.0),
        image(1.0, 0.0, 0.0),
        image(0.0, 0.0, 1.0),
    ])
}

/// The centres of a rod's cross-sections in the plane of bending.
fn in_plane(frames: &[Frame]) -> Vec<P2> {
    let plane = bending_plane();
    frames
        .iter()
        .map(|f| plane.of(*f >> rod::centre()))
        .collect()
}

/// Space with the cord's weight down: `x` across, `y` up and `z` towards the viewer, as a map
/// on points.
fn weight_down() -> Point<(Point,), f64> {
    let image = Point::new;
    Point::from_images([
        image(1.0, 0.0, 0.0, 0.0),
        image(0.0, 0.0, 1.0, 0.0),
        image(0.0, -1.0, 0.0, 0.0),
        image(0.0, 0.0, 0.0, 1.0),
    ])
}

/// The share of the loop at time `t`, and its phase: half a turn at the loop's middle.
fn share(t: f32) -> f32 {
    t.rem_euclid(SECONDS) / SECONDS
}

fn phase(t: f32) -> f64 {
    TAU * f64::from(share(t))
}

/// The height of the strip under a panel that holds its label.
const LABEL: f32 = 16.0;

/// A panel's label in the strip under it, one size for all the panels.
fn label(c: &mut Canvas, rect: Rect, text: &str) {
    let [x0, x1, y] = [rect.lo.e20(), rect.hi.e20(), rect.hi.e01()];
    label_under(c, Rect::new(x0, y, x1, y + LABEL), text, LABEL / 8.0, 0.95);
}

/// A parallel camera on the box, from an azimuth and an elevation in radians, showing the box
/// whole: its half height the larger of the box's height and its width across the panel.
fn camera(rect: Rect, b: Bounds, azimuth: f32, elevation: f32, up: f64, across: f64) -> Camera {
    let aspect = f64::from(rect.height() / rect.width());
    let half = (up.max(across * aspect) * MARGIN) as f32;
    Camera::orbit(
        rect,
        b.centre,
        10.0,
        azimuth,
        elevation,
        Lens::Parallel(half),
    )
}

/// The curled strip: numga's six curls faint, from straight, darkest, to a double ring, and
/// the strip at `turns` bright, with its clamp.
fn draw_curls(c: &mut Canvas, rect: Rect, turns: f64) {
    let curls = &scene().curls;
    let ax = Axes::fitting(rect, curls.iter().flatten().copied(), MARGIN as f32);
    let [low, high] = [-0.06, 0.7].map(|y| P2::xy(0.0, y));
    ax.polyline(c, &[low, high], 3.0, palette::grid().faded(1.5));
    for (k, curl) in curls.iter().enumerate() {
        let colour = colormap::viridis(0.05 + 0.85 * k as f32 / (curls.len() - 1) as f32);
        ax.polyline(c, curl, 1.2, colour.faded(0.3));
    }
    ax.polyline(c, &in_plane(&curling(turns)), 2.4, palette::sky());
    label(c, rect, &format!("CURLED BY A COUPLE: {turns:.2} TURNS"));
}

/// Euler's elastica: numga's plate faint, the rod leaving its pin at `degrees` bright between
/// its pins on the line of thrust, the far pin a chord of the closed form away.
fn draw_euler(c: &mut Canvas, rect: Rect, degrees: f64) {
    let sc = scene();
    let shapes = sc.plate.iter().flatten().chain(&sc.swept).copied();
    let ax = Axes::fitting(rect, shapes, MARGIN as f32);
    // The line of thrust across the panel.
    let thrust = [0.0, 1.0].map(|x| P2::xy(f64::from(ax.at(x, 0.0).to_euclidean()[0]), 0.0));
    ax.polyline(c, &thrust, 1.0, palette::grid());
    for shape in &sc.plate {
        ax.polyline(c, shape, 1.0, palette::blue().faded(0.25));
    }
    let rod = in_plane(&euler(degrees));
    ax.polyline(c, &rod, 2.4, palette::sky());
    let pins = [rod[0], P2::xy(chord(degrees.to_radians()), 0.0)];
    ax.scatter(c, &pins, Marker::Triangle, 9.0, palette::ink());
    label(
        c,
        rect,
        &format!("EULER'S ELASTICA: {degrees:.0} DEG AT THE PIN"),
    );
}

/// A rod in a 3D scene: its ribbon, lit, and its centre line; its points placed by `place`.
fn draw_rod(s: &mut Scene3, frames: &[Frame], width: f64, place: impl Fn(P) -> P) {
    let edges: Vec<[P; 2]> = frames
        .iter()
        .map(|f| ribbon(*f, width).map(&place))
        .collect();
    for pair in edges.windows(2) {
        let ([a, b], [d, e]) = (pair[0], pair[1]);
        let colour = s.lit(a, b, e, palette::sky().faded(0.4));
        s.quad(a, b, e, d, colour, 0.9);
    }
    let centres: Vec<P> = frames.iter().map(|f| place(*f >> rod::centre())).collect();
    s.polyline(&centres, 0.8, palette::blue().faded(0.5));
}

/// Kirchhoff's helices side by side, each a little more than the widest apart.
fn draw_helices(c: &mut Canvas, rect: Rect) {
    let sc = scene();
    let [azimuth, elevation] = HELIX_VIEW.map(f32::to_radians);
    let [x, _, z] = sc.helix_box.half;
    let cam = camera(rect, sc.helix_box, azimuth, elevation, z, x);
    panel3(c, rect, cam, |s| {
        for (k, frames) in sc.helices.iter().enumerate() {
            draw_rod(s, frames, RIBBON, |p| p + across(sc.spacing, k));
        }
    });
    label(c, rect, "KIRCHHOFF'S HELICES: TILTS 80 TO 20 DEG");
}

/// The wire under its fixed couple at time `t` of the sweep of its tilt, the camera turning,
/// all to one scale: a square footprint about the coil's axis, so that the turning camera sees
/// the same extent.
fn draw_uncoiling(c: &mut Canvas, rect: Rect, t: f32) {
    let tilt = uncoiling_tilt(phase(t));
    let [from, to] = UNCOIL_AZIMUTHS;
    let azimuth = (from + (to - from) * share(t)).to_radians();
    let b = scene().uncoil_box;
    let [x, y, z] = b.half;
    let cam = camera(rect, b, azimuth, UNCOIL_ELEVATION.to_radians(), z, x.max(y));
    panel3(c, rect, cam, |s| {
        draw_rod(s, &coiled(tilt, UNCOIL_TURNS), UNCOIL_RIBBON, |p| p);
    });
    label(
        c,
        rect,
        &format!("UNDER A FIXED COUPLE: TILTED {tilt:.0} DEG"),
    );
}

/// The telephone cord at time `t`: numga's frame, as a ribbon with its base and its handset,
/// from one view and to one scale.
fn draw_cord(c: &mut Canvas, rect: Rect, t: f32) {
    let sc = scene();
    let index = (t.rem_euclid(SECONDS) / 0.04) as usize % CORD_FRAMES;
    let frames = &sc.cord[index];
    let view = weight_down();
    let [azimuth, elevation] = CORD_VIEW.map(f32::to_radians);
    let [x, y, _] = sc.cord_box.half;
    let cam = camera(rect, sc.cord_box, azimuth, elevation, y, x);
    panel3(c, rect, cam, |s| {
        draw_rod(s, frames, CORD_RIBBON, |p| view.of(p));
        for end in [frames[0], frames[frames.len() - 1]] {
            s.dot(
                view.of(end >> rod::centre()),
                Marker::Dot,
                9.0,
                palette::ink(),
            );
        }
    });
    let time = index as f64 * STEPS_PER_FRAME as f64 * TIME_STEP;
    label(
        c,
        rect,
        &format!("A TELEPHONE CORD, PULLED AND LET GO: {time:.2} S"),
    );
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let (top, middle) = (h * 0.12, h * 0.56);
    let cell = |x0: f32, y0: f32, x1: f32, y1: f32| {
        Rect::new(x0, y0, x1, y1).inset(6.0, 4.0, 6.0, 4.0 + LABEL)
    };
    let sweep = signal::haversine(phase(t));
    draw_curls(c, cell(0.0, top, w * 0.32, middle), 2.0 * sweep);
    let degrees = SWEEP[0] + (SWEEP[1] - SWEEP[0]) * sweep;
    draw_euler(c, cell(w * 0.32, top, w * 0.64, middle), degrees);
    draw_helices(c, cell(w * 0.64, top, w, middle));
    draw_uncoiling(c, cell(0.0, middle, w * 0.32, h), t);
    draw_cord(c, cell(w * 0.32, middle, w, h), t);
    caption(
        c,
        "THE ELASTICA: CURLS, EULER'S BUCKLED RODS, KIRCHHOFF'S HELICES AND A TELEPHONE CORD",
        "A ROD IS A CHAIN OF MOTORS: THE END LOAD, ONE LINE, SEEN FROM EACH CROSS-SECTION TURNS IT \
         (PGA3D)",
    );
}

fn main() {
    run(Anim::new("elastica", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::rod::*;
    use super::scenes::*;
    use gax::pga3d::Line;
    use gax_numga_examples::rng::{Draw, rng};
    use gax_numga_examples::signal::phasor;

    /// Where a frame carries the centre.
    fn place(frame: Frame) -> [f64; 3] {
        (frame >> centre()).to_euclidean()
    }

    /// The tip of a rod.
    fn tip(frames: &[Frame]) -> [f64; 3] {
        place(frames[frames.len() - 1])
    }

    /// The largest difference between coordinates.
    fn apart(a: [f64; 3], b: [f64; 3]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    /// numga's `test_curled_strip_closes_on_its_clamp`, with the scenario's check of the
    /// straight strip: curled by whole turns, the strip's end comes back to its start.
    #[test]
    fn curled_strip_closes_on_its_clamp() {
        for turns in [1.0, 2.0] {
            assert!(apart(tip(&curling(turns)), [0.0; 3]) < 1e-11);
        }
        assert!(apart(tip(&curling(0.0)), [0.0, 0.0, LENGTH]) < 1e-11);
    }

    /// numga's `test_euler_elastica_reach_their_far_pins`: they reach the line of thrust again
    /// at their far ends, a chord away from the first pin; the tolerance tests the steps along
    /// the rod.
    #[test]
    fn euler_elastica_reach_their_far_pins() {
        for angle in END_ANGLES {
            let [x, y, z] = tip(&euler(angle));
            assert!(x.abs() < 1e-3 && y.abs() < 1e-3, "{angle}");
            assert!((z - chord(angle.to_radians())).abs() < 1e-3, "{angle}");
        }
    }

    /// numga's `test_helices_end_above_their_starts`: coiled through whole turns, the wire's
    /// end lies straight above its start, the length times the cosine of the tilt higher; the
    /// tolerance tests the steps along the rod.
    #[test]
    fn helices_end_above_their_starts() {
        for tilt in TILTS {
            let rise = LENGTH * phasor(tilt.to_radians()).e20();
            assert!(
                apart(tip(&coiled(tilt, COILS)), [0.0, 0.0, rise]) < 3e-3,
                "{tilt}"
            );
        }
    }

    /// numga's `test_moving_cord_keeps_its_energy`: kicked, with no weight and no damping, the
    /// cord's kinetic and elastic energy together stay put, the joint loads being the derivative
    /// of the joints' energy. The kick is `rand`'s draw, not numpy's; the energy drifts in
    /// proportion to the time step, and the tolerance tests the time steps.
    #[test]
    fn moving_cord_keeps_its_energy() {
        let cord = cord();
        let (_, mut frames) = coil(CORD_PIECES);
        let mut draw = rng(0);
        let mut rates: Vec<Line<(), f64>> = (0..CORD_PIECES)
            .map(|_| {
                let [a, b, c] = [(); 3].map(|_| 0.3 * draw.normal());
                Line::new(0.0, 0.0, 0.0, a, b, c)
            })
            .collect();
        let free = Segment {
            weight: gax::pga3d::Point::direction(0.0, 0.0, 0.0),
            damping: 0.0,
            ..cord.segment
        };
        let energy = |frames: &[Frame], rates: &[Line<(), f64>]| {
            let kinetic: f64 = rates.iter().map(|r| (free.inertia.of(*r) & *r).s()).sum();
            let elastic: f64 = frames
                .windows(2)
                .map(|pair| {
                    let d = deviation(cord.joint, pair[0], pair[1]) * cord.step.recip();
                    (cord.stiff.of(d) & d).s() * cord.step
                })
                .sum();
            (kinetic + elastic) / 2.0
        };
        let start = energy(&frames, &rates);
        for _ in 0..200 {
            let dt = TIME_STEP / 4.0;
            moving(
                &mut frames,
                &mut rates,
                cord.joint,
                cord.stiff,
                cord.step,
                &free,
                dt,
            );
        }
        let drift = energy(&frames, &rates) / start - 1.0;
        assert!(drift.abs() < 3e-2, "{drift}");
    }

    /// The scenario's check of the telephone cord: it bends and twists but hardly stretches or
    /// shears, its joints' shifts within a few hundredths of their rest, the most where the
    /// handset springs back.
    #[test]
    fn telephone_cord_hardly_stretches_or_shears() {
        let cord = cord();
        let largest = pulled()
            .iter()
            .flat_map(|frames| {
                frames.windows(2).map(|pair| {
                    let d = deviation(cord.joint, pair[0], pair[1]) * cord.step.recip();
                    [d.e01(), d.e02(), d.e03()]
                        .map(f64::abs)
                        .into_iter()
                        .fold(0.0, f64::max)
                })
            })
            .fold(0.0, f64::max);
        assert!(largest < 0.1, "{largest}");
        // numga's largest, at its frame 43, the last joint.
        assert!((largest - 0.07298043582540033).abs() < 1e-8, "{largest}");
    }

    /// Numbers from numga: the quarter curl's end, the thrusts and chords of the plate and its
    /// far ends, the helices' tips, the cord at three frames and the frames drawn between its
    /// segments.
    #[test]
    fn numbers_match_numga() {
        let quarter = tip(&curling(0.25));
        assert!(apart(quarter, [0.0, 0.6366197723675843, 0.6366197723675764]) < 1e-13);
        let thrusts = [
            10.02156537298278,
            11.367017035003174,
            13.750371636040741,
            16.56026560657328,
            21.54588073454704,
            30.648694306995356,
            47.1388487204709,
            81.72967382151525,
        ];
        let chords = [
            0.9697309073635154,
            0.741019606076327,
            0.456946581044464,
            0.2372046519155173,
            0.0001140960232985933,
            -0.222268382946877,
            -0.4028210724606214,
            -0.555851461156482,
        ];
        for ((angle, thrust_), chord_) in END_ANGLES.iter().zip(thrusts).zip(chords) {
            assert!((thrust(angle.to_radians()) - thrust_).abs() < 1e-12 * thrust_);
            assert!((chord(angle.to_radians()) - chord_).abs() < 1e-14);
        }
        let far = tip(&euler(130.7));
        assert!(apart(far, [0.0, 8.254598899348115e-06, 0.0001349140659683575]) < 1e-12);
        let helix = tip(&coiled(40.0, COILS));
        let numga = [
            -1.84034925797344e-05,
            0.0003011285820788873,
            0.766085524412596,
        ];
        assert!(apart(helix, numga) < 1e-12);
        let scene = pulled();
        let numga = [
            (
                20,
                [
                    0.0005005171632073148,
                    -0.0004735003713601751,
                    7.193035110417531e-05,
                ],
                [0.337518873800235, -0.07029931534120994, 0.07534609603640893],
            ),
            (
                35,
                [
                    0.001376549132958214,
                    -0.0005476213278828588,
                    0.0002071579514002773,
                ],
                [0.4864819784917827, -0.07169558374815727, 0.1040403870220854],
            ),
            (
                69,
                [
                    0.0001522054411915503,
                    -0.0004991424234405797,
                    7.551511646901471e-06,
                ],
                [
                    0.1514515007726656,
                    -0.04547266278666595,
                    0.04028699718421293,
                ],
            ),
        ];
        for (frame, first, middle) in numga {
            assert!(apart(place(scene[frame][1]), first) < 1e-10, "{frame}");
            assert!(apart(place(scene[frame][20]), middle) < 1e-10, "{frame}");
        }
        let drawn = between(&scene[69], DRAWN_BETWEEN);
        assert_eq!(drawn.len(), 125);
        let numga = [
            -0.0006914128001681038,
            0.009342463289897058,
            -0.02238118867912779,
        ];
        assert!(apart(place(drawn[1]), numga) < 1e-12);
        let numga = [
            0.1214224539892217,
            -0.08155056612580668,
            -0.01423173534689752,
        ];
        assert!(apart(place(drawn[50]), numga) < 1e-10);
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
    }
}
