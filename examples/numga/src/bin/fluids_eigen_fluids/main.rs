//! numga's `fluids/eigen_fluids`: fluids made of Laplacian eigenfunctions, a surface flow that
//! evolves through its modal amplitudes (de Witt, Lessig and Fiume's *Fluid simulation using
//! Laplacian eigenfunctions*, on a curved surface).
//!
//! The surface's geometry gives its Laplace–Beltrami modes and the coefficients of their
//! interaction. Each face's corner gradient is the side facing the corner contracted with the
//! inverse patch, the join of two sides; the area-weighted material pulled back through the
//! gradient's adjugate is the stiffness, and with the lumped vertex mass it gives the modes, a
//! sparse generalized eigenproblem. A streamfunction drives the tangent flow `∇ψ ^ n`, and a
//! field's gradient joined with that flow is the rate at which the flow carries it. In mode space
//! the vorticity's equation is the Lie–Poisson bracket of the streamfunction with the vorticity,
//! whose structure constants integrate each mode against the Jacobian of two others: totally
//! antisymmetric, so the inviscid flow keeps its kinetic energy and its enstrophy, and viscosity
//! damps each mode by its eigenvalue, the fine ones first.
//!
//! Top: the Stanford bunny's 128 modes carrying twelve vortex dipoles, coloured by vorticity
//! with the streamlines drawn as the streamfunction's level lines; a sphere's 80 modes, the
//! complete spherical harmonic bands through degree eight, rolling rippled shear layers into
//! vortices; and the invariants of the inviscid run against a viscous one, with the share of the
//! enstrophy below each mode. Below: four of the bunny's modes, from a global dipole to a fine
//! standing wave.

mod eigen;
mod scenes;
#[cfg(test)]
mod tests;

#[path = "../../shared/simplicial.rs"]
mod simplicial;

use eigen::*;
use gax::field::Field;
use gax_numga_examples::scene3::{Scene3, panel3};
use gax_numga_examples::signal::frame;
use gax_numga_examples::{
    Anim, Axes, Camera, Canvas, Lens, Light, ORIGIN3, Point2, Point3, Rect, backdrop, caption,
    label_under, palette, par_map, run_with,
};
use scenes::*;

/// One loop: numga's 101 frames of 40 ms.
const SECONDS: f32 = 4.04;
/// numga's camera: its elevation and azimuth (radians), and the half height of the view.
const ELEVATION: f32 = 0.314;
const AZIMUTH: f32 = 0.838;
const HALF: f32 = 1.05;
/// The exposures: the percentiles of the field and of the streamfunction that reach full colour
/// and the contour spacing.
const FIELD_EXPOSURE: f64 = 0.955;
const STREAM_EXPOSURE: f64 = 0.985;
/// The level lines per exposure of a mode's streamfunction, and of a flow's.
const MODE_CONTOURS: f64 = 5.5;
const FLOW_CONTOURS: f64 = 6.0;

/// A flow and the frames of its run, with the scales of its first frame.
struct Played {
    flow: Flow,
    frames: Vec<Field<Scalar>>,
    field_scale: f64,
    stream_scale: f64,
}

impl Played {
    fn new(flow: Flow, frames: Vec<Field<Scalar>>) -> Self {
        let (vorticity, stream) = flow.fields(&frames[0]);
        Played {
            field_scale: percentile(&vorticity, FIELD_EXPOSURE),
            stream_scale: percentile(&stream, STREAM_EXPOSURE),
            flow,
            frames,
        }
    }
}

/// The invariants over the runs, relative to the inviscid start, and the cumulative share of the
/// enstrophy below each mode at the start and at the end of each run.
struct Conservation {
    times: Vec<f64>,
    inviscid: Vec<(f64, f64)>,
    viscous: Vec<(f64, f64)>,
    spectra: [Vec<f64>; 3],
}

struct Shown {
    bunny: Played,
    sphere: Played,
    conservation: Conservation,
}

/// The value of `|field|` below which a share `p` of the sites lie.
fn percentile(field: &Field<Scalar>, p: f64) -> f64 {
    let mut values: Vec<f64> = field.iter().map(|x| x.s().abs()).collect();
    values.sort_by(f64::total_cmp);
    let at = (p * (values.len() - 1) as f64).round() as usize;
    values[at].max(1e-12)
}

/// The cumulative enstrophy over the modes, as a share of `total`.
fn cumulative(state: &Field<Scalar>, total: f64) -> Vec<f64> {
    let mut sum = 0.0;
    state
        .iter()
        .map(|w| {
            sum += (w * w).s() / (2.0 * total);
            sum
        })
        .collect()
}

fn shown() -> Shown {
    let flows = par_map(&[true, false], |&bunny| {
        if bunny { Flow::bunny() } else { Flow::sphere() }
    });
    let [bunny, sphere]: [Flow; 2] = flows
        .try_into()
        .unwrap_or_else(|_| unreachable!("two flows"));
    // The three runs at once: the bunny inviscid and viscous, the sphere.
    let runs = par_map(&[0, 1, 2], |&k| match k {
        0 => bunny.run(0.0, DT, STEPS),
        1 => bunny.run(VISCOSITY, DT, STEPS),
        _ => sphere.run(SPHERE_VISCOSITY, SPHERE_DT, SPHERE_STEPS),
    });
    let [inviscid, viscous, rolled]: [Vec<Field<Scalar>>; 3] = runs
        .try_into()
        .unwrap_or_else(|_| unreachable!("three runs"));
    let trace = |run: &[Field<Scalar>]| -> Vec<(f64, f64)> {
        run.iter()
            .map(|w| invariants(w, &bunny.eigenvalues))
            .collect()
    };
    let (inviscid_trace, viscous_trace) = (trace(&inviscid), trace(&viscous));
    let (e0, z0) = inviscid_trace[0];
    let relative = |t: Vec<(f64, f64)>| t.into_iter().map(|(e, z)| (e / e0, z / z0)).collect();
    let conservation = Conservation {
        times: (0..=STEPS).map(|k| k as f64 * DT).collect(),
        inviscid: relative(inviscid_trace),
        viscous: relative(viscous_trace),
        spectra: [&inviscid[0], &inviscid[STEPS], &viscous[STEPS]].map(|w| cumulative(w, z0)),
    };
    let frames = inviscid[..=FLOW_STEPS]
        .iter()
        .step_by(PLAYBACK_STRIDE)
        .cloned()
        .collect();
    let sphere_frames = rolled.into_iter().step_by(SPHERE_STRIDE).collect();
    Shown {
        bunny: Played::new(bunny, frames),
        sphere: Played::new(sphere, sphere_frames),
        conservation,
    }
}

/// A point of space as drawn.
fn drawn(p: Point) -> Point3 {
    let [x, y, z] = coordinates(p).map(|c| c as f32);
    Point3::xyz(x, y, z)
}

/// The vorticity's light: orange where it turns one way, blue the other, the surface's faint
/// blue where it rests.
fn vorticity_light(level: f64) -> Light {
    let t = level.clamp(-1.0, 1.0) as f32;
    let accent = if t >= 0.0 {
        palette::ORANGE
    } else {
        palette::SKY
    };
    palette::BLUE.faded(0.1) + accent.faded(0.05 + 0.5 * t.abs().powf(0.72))
}

/// The segment of a face's level line at `level` of a field linear on it.
fn level_on(corners: [Point3; 3], values: [f64; 3], level: f64) -> Option<[Point3; 2]> {
    let mut ends = Vec::new();
    for k in 0..3 {
        let (a, b) = (k, (k + 1) % 3);
        let (va, vb) = (values[a] - level, values[b] - level);
        if (va < 0.0) != (vb < 0.0) {
            let t = (va / (va - vb)) as f32;
            ends.push(corners[a] + (corners[b] - corners[a]).gp(t));
        }
    }
    match ends[..] {
        [p, q] => Some([p, q]),
        _ => None,
    }
}

/// A surface coloured by a field, with the streamfunction's level lines: `contours` of them per
/// stream scale, a little off the surface along each face's normal so that they draw over it.
#[allow(clippy::too_many_arguments)]
fn draw_surface(
    c: &mut Canvas,
    rect: Rect,
    surface: &Surface,
    field: &Field<Scalar>,
    stream: &Field<Scalar>,
    field_scale: f64,
    stream_scale: f64,
    contours: f64,
) {
    let cam = Camera::orbit(rect, ORIGIN3, 8.0, AZIMUTH, ELEVATION, Lens::Parallel(HALF));
    panel3(c, rect, cam, |s: &mut Scene3| {
        let mut lines = Vec::new();
        for f in 0..surface.mesh.len(2) {
            let corners = surface.mesh.simplex(2, f);
            let at = [0, 1, 2].map(|k| drawn(surface.vertices.site(corners[k])));
            let field = [0, 1, 2].map(|k| field.site(corners[k]).s() / field_scale);
            let mean = (field[0] + field[1] + field[2]) / 3.0;
            let lit = s.lit(at[0], at[1], at[2], vorticity_light(mean));
            s.tri(at[0], at[1], at[2], lit, 1.0);
            let levels = [0, 1, 2].map(|k| stream.site(corners[k]).s() / stream_scale * contours);
            let low = levels.iter().copied().fold(f64::INFINITY, f64::min).ceil() as i64;
            let high = levels
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
                .floor() as i64;
            if low <= high {
                let n = surface.normals.site(f) * 0.004;
                let lift = Point3::direction(n.e1() as f32, n.e2() as f32, n.e3() as f32);
                let raised = at.map(|p| p + lift);
                for k in low..=high {
                    lines.extend(level_on(raised, levels, k as f64));
                }
            }
        }
        s.stroke(&lines, 0.6, palette::INK.faded(0.28));
    });
}

/// A flow's frame at time `t`.
fn draw_flow(c: &mut Canvas, rect: Rect, p: &Played, t: f32) {
    let state = &p.frames[frame((t / SECONDS).rem_euclid(1.0), p.frames.len())];
    let (vorticity, stream) = p.flow.fields(state);
    draw_surface(
        c,
        rect,
        &p.flow.surface,
        &vorticity,
        &stream,
        p.field_scale,
        p.stream_scale,
        FLOW_CONTOURS,
    );
}

/// The invariants over time, and the cumulative enstrophy over the modes.
fn draw_conservation(c: &mut Canvas, rect: Rect, k: &Conservation, t: f32) {
    let history = rect.part(0.0, 0.0, 1.0, 0.5).inset(34.0, 26.0, 8.0, 22.0);
    let end = *k.times.last().expect("times") as f32;
    let ax = Axes::new(history, [0.0, end], [0.0, 1.08]);
    ax.frame(c, "ENERGY AND ENSTROPHY", "TIME", "");
    let curve = |pick: &dyn Fn(&(f64, f64)) -> f64, run: &[(f64, f64)]| -> Vec<Point2> {
        k.times
            .iter()
            .zip(run)
            .map(|(&t, v)| Point2::xy(t as f32, pick(v) as f32))
            .collect()
    };
    let energy = |v: &(f64, f64)| v.0;
    let enstrophy = |v: &(f64, f64)| v.1;
    ax.polyline(c, &curve(&energy, &k.inviscid), 1.4, palette::INK);
    ax.dashed(c, &curve(&enstrophy, &k.inviscid), 1.4, 5.0, palette::INK);
    ax.polyline(c, &curve(&energy, &k.viscous), 1.6, palette::GREEN);
    ax.polyline(c, &curve(&enstrophy, &k.viscous), 1.6, palette::ORANGE);
    // The time the bunny's frame shows: the animation plays the first half of the run.
    let now = (t / SECONDS).rem_euclid(1.0) * (FLOW_STEPS as f64 * DT) as f32;
    ax.line(
        c,
        Point2::xy(now, 0.0),
        Point2::xy(now, 1.08),
        1.0,
        palette::YELLOW.faded(0.6),
    );
    ax.legend(
        c,
        &[
            ("INVISCID", palette::INK),
            ("VISCOUS ENERGY", palette::GREEN),
            ("VISCOUS ENSTROPHY", palette::ORANGE),
        ],
    );

    let spectrum = rect.part(0.0, 0.5, 1.0, 1.0).inset(34.0, 26.0, 8.0, 22.0);
    let modes = k.spectra[0].len() as f32;
    let ax = Axes::new(spectrum, [1.0, modes], [0.0, 1.08]);
    ax.frame(c, "ENSTROPHY BELOW EACH MODE", "MODE", "");
    let curve = |values: &[f64]| -> Vec<Point2> {
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| Point2::xy(i as f32 + 1.0, v as f32))
            .collect()
    };
    ax.dashed(c, &curve(&k.spectra[0]), 1.2, 4.0, palette::INK.faded(0.6));
    ax.polyline(c, &curve(&k.spectra[1]), 1.4, palette::INK);
    ax.polyline(c, &curve(&k.spectra[2]), 1.6, palette::ORANGE);
}

/// One of the bunny's modes, its own streamfunction, scaled to its own percentiles.
fn draw_mode(c: &mut Canvas, rect: Rect, flow: &Flow, k: usize) {
    let mode = &flow.modes[k];
    draw_surface(
        c,
        rect,
        &flow.surface,
        mode,
        mode,
        percentile(mode, FIELD_EXPOSURE),
        percentile(mode, STREAM_EXPOSURE),
        MODE_CONTOURS,
    );
}

fn draw(c: &mut Canvas, s: &Shown, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let top = Rect::new(0.0, h * 0.12, w, h * 0.64);
    let flows = top.part(0.0, 0.0, 0.64, 1.0);
    for (i, (played, label)) in [
        (
            &s.bunny,
            "THE BUNNY'S 128 MODES CARRY TWELVE VORTEX DIPOLES",
        ),
        (&s.sphere, "A SPHERE'S 80 ROLL SHEAR LAYERS INTO VORTICES"),
    ]
    .into_iter()
    .enumerate()
    {
        let rect = flows.column(i, 2).inset(4.0, 2.0, 4.0, 4.0);
        draw_flow(c, rect, played, t);
        label_under(c, rect, label, 20.0, 0.95);
    }
    draw_conservation(c, top.part(0.64, 0.0, 1.0, 1.0), &s.conservation, t);
    let bottom = Rect::new(0.0, h * 0.66, w, h * 0.97);
    for (i, &k) in SHOWN_MODES.iter().enumerate() {
        let rect = bottom.column(i, 4).inset(4.0, 2.0, 4.0, 4.0);
        draw_mode(c, rect.part(0.0, 0.0, 1.0, 0.88), &s.bunny.flow, k);
        label_under(c, rect, &format!("MODE {}", k + 1), 14.0, 0.5);
    }
    caption(
        c,
        "EIGEN FLUIDS: A SURFACE FLOW IN LAPLACE-BELTRAMI MODES",
        "THE GRADIENT'S ADJUGATE GIVES THE MODES; THEIR LIE-POISSON BRACKET MOVES THE VORTICITY, KEEPING ENERGY AND ENSTROPHY (VGA3D)",
    );
}

fn main() {
    run_with(
        Anim::new("eigen fluids", SECONDS).size(960, 540),
        shown,
        draw,
    );
}
