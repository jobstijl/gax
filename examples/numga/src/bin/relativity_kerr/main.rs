//! numga's `relativity/kerr`: light around a rotating black hole, in the spacetime algebra
//! (STA). In Kerr–Schild form the inverse metric is the identity plus a null dyad,
//! `g⁻¹ = 1 + 2 H l (l · ·)`, with a scalar profile `H` and a null direction `l`, and the ray
//! Hamiltonian is its quadratic form, `½ p² + H (l · p)²`. Its momentum derivative is the
//! ray's tangent, and minus its position derivative the momentum's rate, without component
//! equations: the profile's gradient is a vector, the null direction's derivative a map with
//! the step left open (`Vector<(Vector,)>`), and the gradient of `l · v` is that map's adjoint
//! applied to `v`, a pairing solve (gax has no transpose). Momentum is a covector, represented
//! by its Minkowski dual. Units set G and c to one; the spin is a spatial bivector of
//! magnitude J/M. Ingoing Kerr–Schild coordinates cover the future horizon.
//!
//! A camera traces rays back from its pixels until each is captured or escapes. The disk takes
//! a share of each ray it crosses, by its optical depth along the ray, and the camera's screen
//! is carried along by parallel transport: its commutator with the plane the local frame turns
//! in. Each crossing glows as a blackbody at the disk's temperature (Page and Thorne) times the
//! ratio of the frequency the camera sees to the one the orbiting gas emits; the light is
//! polarized across the gas's azimuthal magnetic field, a coherency map on the screen that an
//! analyzer reads.
//!
//! The animation: on the left, equatorial rays as the spin sweeps from -0.8 M to 0.8 M and
//! back (the dark disc is the horizon); on the right, the camera's view 75° from the spin axis,
//! the disk's gas orbiting (its clumps carried along their orbits) over the lensed sky, and
//! below it the same view through a polarizer whose axis turns half a turn per loop.
//!
//! Differences from numga: gax's STA has no kind for the screen plane `x y`, so the camera's
//! screen frame and the coherency are maps on vectors that read only `x` and `y`; numga's
//! batched ray generators become one loop per ray; the colours of the disk are gax-colour's
//! Planckian locus at the visible luminance of Planck's law (numga integrates the spectrum
//! against the CIE matching functions), and the sky is a star catalogue looked up by direction
//! rather than a texture. Reference: Pelle et al., Skylight (2022), equations 24–26,
//! <https://academic.oup.com/mnras/article/515/1/1316/6631564>.
//!
//! `kerr.rs` holds the geometry and the rays, `disk.rs` the disk, `scenes.rs` numga's two
//! scenes, and `emission.rs` the colours of the disk's and the sky's light.

mod disk;
mod emission;
mod kerr;
mod scenes;

use std::sync::OnceLock;

use emission::{Sky, blackbody, blackbody_colour, emissivity, flowing_waves};
use gax::{vga2d, vga3d};
use gax_numga_examples::points;
use gax_numga_examples::signal::phasor;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, ORIGIN2, Point2, Rect, backdrop, caption, colormap, light,
    palette, par_map, run,
};
use kerr::V;
use scenes::Glint;

/// The loop, numga's 80 frames of 80 ms: the spin sweeps there and back, the polarizer turns
/// half a turn, and the disk's gas turns through [`scenes::FLOW_TIME`].
const SECONDS: f32 = 6.4;
/// The spins of the sweep traced at start-up: one per frame at 25 frames per second.
const SWEEP_FRAMES: usize = 160;
/// The camera's pixels, numga's: 150 000 rays, traced at start-up in about two seconds on
/// eight or more cores.
const CAMERA_SIZE: [usize; 2] = [480, 320];
/// A star's image is this share of a pixel wide (numga's is that share of a texel of its sky
/// texture).
const STAR_WIDTH: f64 = 0.6;
/// The temperature scale's brightness.
const BAR_LUMINANCE: f32 = 0.25;

/// A disk crossing ready to colour: the pixel it lights, the glint, and its light before the
/// texture.
struct Shine {
    pixel: usize,
    glint: Glint,
    glow: Light,
}

/// What the frames share: the sweep's equatorial paths at each spin, as points of the plane,
/// and the camera's disk crossings and the sky's light in each pixel.
struct Scene {
    sweep: Vec<Vec<Vec<Point2>>>,
    shines: Vec<Shine>,
    sky: Vec<Light>,
}

/// An event's place in the equatorial plane, for drawing.
fn on_plane(event: V) -> Point2 {
    Point2::xy(event.e1() as f32, event.e2() as f32)
}

/// The sweep's paths, at each of its spins.
fn sweep_paths() -> Vec<Vec<Vec<Point2>>> {
    let frames: Vec<usize> = (0..SWEEP_FRAMES).collect();
    par_map(&frames, |&k| {
        let paths = scenes::sweep(scenes::sweep_spin(k, SWEEP_FRAMES));
        paths
            .iter()
            .map(|path| path.iter().map(|e| on_plane(*e)).collect())
            .collect()
    })
}

/// The camera's view: each disk crossing's glint and glow, and the light each pixel's
/// escaped ray brings from the sky.
fn camera_view() -> (Vec<Shine>, Vec<Light>) {
    let resolutions = scenes::camera_resolutions(CAMERA_SIZE);
    let star_width = STAR_WIDTH * 2.0 * scenes::half_view() / CAMERA_SIZE[1] as f64;
    let sky = Sky::new();
    let skies = par_map(&resolutions, |r| {
        // A captured ray, or one the disk absorbed whole, brings no sky.
        r.sky.map_or(light(0.0, 0.0, 0.0, 1.0), |escape| {
            let d = escape.direction;
            let direction = vga3d::Vector::new(d.e1(), d.e2(), d.e3()).normalized();
            sky.light(direction.into_inner(), star_width) * escape.weight as f32
        })
    });
    let crossings: Vec<_> = resolutions
        .iter()
        .enumerate()
        .flat_map(|(pixel, r)| r.disk.iter().map(move |c| (pixel, *c)))
        .collect();
    let shines = par_map(&crossings, |(pixel, crossing)| {
        let glint = scenes::glint(crossing);
        Shine {
            pixel: *pixel,
            glow: blackbody(glint.temperature) * glint.weight as f32,
            glint,
        }
    });
    (shines, skies)
}

fn scene() -> &'static Scene {
    static SCENE: OnceLock<Scene> = OnceLock::new();
    SCENE.get_or_init(|| {
        let (shines, sky) = camera_view();
        Scene {
            sweep: sweep_paths(),
            shines,
            sky,
        }
    })
}

/// The camera's pixels at time `t`, as seen and through the polarizer: the sky, and each
/// disk crossing's glow, textured where its gas is. Through the polarizer each crossing
/// passes its transmitted power, doubled to make up for the half of unpolarized light a
/// polarizer stops.
fn camera_lights(s: &Scene, t: f32) -> (Vec<Light>, Vec<Light>) {
    let progress = f64::from(t / SECONDS);
    let turn = core::f64::consts::PI * progress;
    let analyzer = (kerr::xy() * (turn / 2.0)).exp() >> kerr::x();
    let lit = par_map(&s.shines, |shine| {
        let waves = flowing_waves(&shine.glint, progress);
        let light = shine.glow * emissivity(shine.glint.radius, waves) as f32;
        let passed = 2.0 * disk::transmitted(shine.glint.coherency, analyzer);
        (light, light * passed as f32)
    });
    let (mut seen, mut polarized) = (s.sky.clone(), s.sky.clone());
    for (shine, (light, passed)) in s.shines.iter().zip(lit) {
        seen[shine.pixel] += light;
        polarized[shine.pixel] += passed;
    }
    (seen, polarized)
}

/// The camera's pixels over `rect`, blended between pixel centres.
fn draw_view(c: &mut Canvas, rect: Rect, pixels: &[Light]) {
    let [width, height] = CAMERA_SIZE;
    let at = |i: isize, j: isize| {
        let (i, j) = (
            i.clamp(0, width as isize - 1),
            j.clamp(0, height as isize - 1),
        );
        pixels[j as usize * width + i as usize]
    };
    c.clip(rect);
    c.shade(2, |p| {
        let offset = p - rect.lo;
        let u = offset.e20() / rect.width() * width as f32 - 0.5;
        let v = offset.e01() / rect.height() * height as f32 - 0.5;
        let (i, j) = (u.floor() as isize, v.floor() as isize);
        let (fu, fv) = (u - u.floor(), v - v.floor());
        let row = |j| at(i, j) * (1.0 - fu) + at(i + 1, j) * fu;
        Some(row(j) * (1.0 - fv) + row(j + 1) * fv)
    });
    c.unclip();
    let corners = [rect.lo, rect.top_right(), rect.hi, rect.bottom_left()];
    c.polyline(&corners, 1.0, palette::grid(), true);
}

/// A camera view with its title above it.
fn titled_view(c: &mut Canvas, rect: Rect, pixels: &[Light], title: &str) {
    draw_view(c, rect, pixels);
    let at = rect.lo + Point2::direction(0.0, -7.0);
    c.text(title, at, 12.0, palette::ink(), Align::Left);
}

/// The sweep: the equatorial rays at the frame's spin, coloured across the bundle, and the
/// horizon, where a constant oblate radius cuts the equatorial plane in a circle of radius
/// `√(r₊² + a²)`.
fn draw_sweep(c: &mut Canvas, ax: &Axes, t: f32) {
    let s = scene();
    let frame = ((t / SECONDS * SWEEP_FRAMES as f32).round() as usize) % SWEEP_FRAMES;
    let spin = scenes::sweep_spin(frame, SWEEP_FRAMES);
    let paths = &s.sweep[frame];
    for (i, path) in paths.iter().enumerate() {
        let shade = 0.05 + 0.9 * i as f32 / (paths.len() - 1) as f32;
        ax.polyline(c, path, 1.0, colormap::coolwarm(shade).faded(0.9));
    }
    let hole = scenes::hole(spin);
    let outline = vga2d::Vector::new(hole.horizon(), spin).norm() as f32;
    let circle = points::circle(ORIGIN2, outline, 96);
    ax.fill(c, &circle, light(0.01, 0.012, 0.03, 1.0), 1.0);
    ax.polyline(c, &circle, 1.0, palette::grid());
    ax.frame(c, "EQUATORIAL RAYS", "X / M", "Y / M");
    let sense = if spin >= 0.0 {
        "COUNTERCLOCKWISE"
    } else {
        "CLOCKWISE"
    };
    let readout = format!("SPIN {:+.2} M, {sense}", spin / scenes::MASS);
    ax.text(
        c,
        ax.at(0.03, 0.95),
        &readout,
        12.0,
        palette::yellow(),
        Align::Left,
    );
}

/// The colours of the observed temperatures, at one brightness, with their scale.
fn draw_temperature_bar(c: &mut Canvas, rect: Rect) {
    const RANGE: [f64; 2] = [2000.0, 10_000.0];
    let kelvin = |fraction: f32| RANGE[0] + (RANGE[1] - RANGE[0]) * f64::from(fraction);
    c.clip(rect);
    c.shade(1, |p| {
        let across = (p - rect.lo).e20() / rect.width();
        Some(blackbody_colour(kelvin(across), BAR_LUMINANCE))
    });
    c.unclip();
    let size = 11.0;
    for k in [3000.0, 6000.0, 9000.0] {
        let fraction = ((k - RANGE[0]) / (RANGE[1] - RANGE[0])) as f32;
        let foot = rect.bottom_left() + Point2::direction(fraction * rect.width(), size * 1.4);
        let label = format!("{k:.0}");
        c.text(&label, foot, size, palette::ink(), Align::Center);
    }
    let title = rect.lo + Point2::direction(0.0, -size * 0.6);
    let name = "OBSERVED TEMPERATURE, K";
    c.text(name, title, size, palette::ink(), Align::Left);
}

/// The polarizer's axis on the screen, as a line across a ring: `x` right and `y` up.
fn draw_analyzer(c: &mut Canvas, centre: Point2, radius: f32, angle: f32) {
    let axis = phasor(angle);
    let across = Point2::direction(axis.e20(), -axis.e01()).gp(radius);
    c.ring(centre, radius, 1.0, palette::grid());
    c.line(centre - across, centre + across, 1.5, palette::sky());
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.rect().width(), c.rect().height());
    let (size, margin) = (12.0, 20.0);
    let top = h * 0.2;

    // The sweep in a square on the left, with room below for its tick and axis labels.
    let side = h - top - 4.5 * size - margin;
    let square = Rect::new(w * 0.06, top, w * 0.06 + side, top + side);
    draw_sweep(c, &Axes::equal(square, ORIGIN2, scenes::EXTENT as f32), t);

    // The camera's view on the right, and below it the polarized view and the colours.
    let (seen, polarized) = camera_lights(scene(), t);
    let aspect = CAMERA_SIZE[1] as f32 / CAMERA_SIZE[0] as f32;
    let view_width = (h * 0.56) / aspect;
    let left = w - margin - view_width;
    let view = Rect::new(left, top, w - margin, top + view_width * aspect);
    titled_view(c, view, &seen, "THE CAMERA, 75 DEG FROM THE SPIN AXIS");

    let below = view.hi.to_euclidean()[1] + size * 2.0;
    let small_height = h - margin * 0.5 - below;
    let small = Rect::new(
        left,
        below,
        left + small_height / aspect,
        below + small_height,
    );
    titled_view(c, small, &polarized, "THROUGH A POLARIZER");
    let glyph = small.top_right() + Point2::direction(size * 2.0, size * 1.5);
    draw_analyzer(c, glyph, size, core::f32::consts::PI * t / SECONDS);

    let bar_left = small.hi.to_euclidean()[0] + size * 4.5;
    let bar = Rect::new(bar_left, below + size * 4.0, w - margin, below + size * 5.4);
    draw_temperature_bar(c, bar);
    caption(
        c,
        "KERR: LIGHT AROUND A ROTATING BLACK HOLE",
        "NULL RAYS FROM THE GEOMETRIC DERIVATIVES OF THEIR HAMILTONIAN (STA); A THIN DISK AT ITS SHIFTED TEMPERATURE",
    );
}

fn main() {
    run(Anim::new("kerr", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
    }
}
