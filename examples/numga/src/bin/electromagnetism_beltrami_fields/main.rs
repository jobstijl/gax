//! numga's `electromagnetism/beltrami_fields`: Beltrami fields carry magnetic linkage, and hold
//! their energy stationary at a fixed helicity.
//!
//! Magnetostatics in spacetime, `x+y+z+t-` (`gax::sta`): a solid torus rests in one instant, its
//! points are directions of space, and the magnetic field is a plane of space that reads its flux
//! through an oriented patch by the join. A hat is constant in time, so its gradient is the patch
//! its opposite face sweeps through time over three times the prism the tetrahedron sweeps.
//! Whitney edge forms reconstruct the potential, and their curls give the field. The energy pairs
//! the field with its dual; the helicity pairs the potential with its curl through the signed
//! volume, with no metric; their adjugates carry both readings back to edge amplitudes. On the
//! edges that keep the flux through the wall at nothing, the helicity's eigenmodes against the
//! energy are the Beltrami fields, `∇ × B = λ B`: a chiral spectrum, symmetric about zero.
//!
//! Left, five Beltrami modes relax selectively at fixed helicity, the finer faster, towards the
//! Taylor state, the mode of least energy for that helicity (seen obliquely and from above, the
//! field lines traced through the smoothed field); right, the spectrum of inverse eigenvalues,
//! and the Taylor mode and an excited one.

mod beltrami;
mod lines;
mod scenes;
#[cfg(test)]
mod tests;

use beltrami::*;
use gax::vga3d::Vector;
use gax_numga_examples::scene3::{Scene3, panel3};
use gax_numga_examples::shared::simplicial;
use gax_numga_examples::signal::frame;
use gax_numga_examples::{
    Anim, Axes, Camera, Canvas, Lens, Light, Marker, ORIGIN3, Point2, Point3, Rect, backdrop,
    caption, label_under, palette, run_with,
};
use lines::{Samples, space};
use scenes::*;

/// One loop: numga's 46 frames of 40 ms.
const SECONDS: f32 = 1.84;
/// numga's views: oblique and from above (elevation, azimuth in radians), and the half height.
const OBLIQUE: [f32; 2] = [0.489, 0.628];
const ABOVE: [f32; 2] = [1.431, 0.628];
const HALF: f32 = 1.45;
/// The steps of a field line drawn as one stroke.
const CHUNK: usize = 6;

/// A field's lines, each with its shell.
type Lines = Vec<(Vec<Point3>, usize)>;

struct Shown {
    wall: Vec<[Point3; 3]>,
    relaxation: Vec<Lines>,
    energies: Vec<f64>,
    taylor: f64,
    mu: Vec<f64>,
    shown: [usize; 2],
    modes: [Lines; 2],
}

/// A direction of space as drawn.
fn drawn(v: Vector) -> Point3 {
    Point3::xyz(v.e1() as f32, v.e2() as f32, v.e3() as f32)
}

/// The field lines of a magnetic field, as drawn.
fn traced(o: &Operators, magnetic: &gax::field::Field<Magnetic>) -> Lines {
    Samples::new(&o.centres, &o.volumes, magnetic)
        .lines()
        .into_iter()
        .map(|(line, shell)| (line.into_iter().map(drawn).collect(), shell))
        .collect()
}

fn shown() -> Shown {
    let cavity = Cavity::new();
    let o = &cavity.operators;
    let (modes, eigenvalues, states) = cavity.relaxation();
    let fields = reconstruct(&states, &modes, &o.curl);
    let relaxation = gax_numga_examples::par_map(&fields, |m| traced(o, m));
    let energies: Vec<f64> = states
        .iter()
        .map(|a| a.iter().map(|x| x * x).sum::<f64>() / 2.0)
        .collect();
    let helicity: f64 = states[0]
        .iter()
        .zip(&eigenvalues)
        .map(|(x, l)| x * x / l)
        .sum();
    let shown = SHOWN.map(|k| at(k, cavity.mu.len()));
    let wall = crate::simplicial::exterior(&o.mesh, 2)
        .into_iter()
        .map(|f| {
            let corners = o.mesh.simplex(2, f);
            [0, 1, 2].map(|k| drawn(space(o.vertices.site(corners[k]))))
        })
        .collect();
    Shown {
        wall,
        relaxation,
        energies,
        taylor: helicity * eigenvalues[0] / 2.0,
        mu: cavity.mu.clone(),
        shown,
        modes: shown.map(|k| traced(o, &o.curl.of(&cavity.modes[k]))),
    }
}

/// The light of a line's segment: its shell's, warmed towards yellow as it climbs.
fn segment_light(shell: usize, climb: f32) -> Light {
    let base = [palette::ORANGE, palette::GREEN, palette::SKY][shell];
    let pitch = (0.5 + 0.45 * climb).clamp(0.0, 1.0);
    base.faded(1.0 - 0.38 * pitch) + palette::YELLOW.faded(0.38 * pitch)
}

/// The torus's wall, faint, and the field lines in it, from a view.
fn draw_torus(c: &mut Canvas, rect: Rect, s: &Shown, lines: &Lines, view: [f32; 2]) {
    let half = HALF * (rect.height() / rect.width()).max(1.0);
    let cam = Camera::orbit(rect, ORIGIN3, 8.0, view[1], view[0], Lens::Parallel(half));
    panel3(c, rect, cam, |scene: &mut Scene3| {
        for &[a, b, d] in &s.wall {
            let lit = scene.lit(a, b, d, palette::BLUE.faded(0.3));
            scene.tri(a, b, d, lit, 0.12);
        }
        for (line, shell) in lines {
            // In runs of a few steps, each lit by how steeply it climbs.
            for run in line.windows(CHUNK + 1).step_by(CHUNK) {
                let step = run[CHUNK] - run[0];
                let climb = step.e021() / step.ideal_norm().max(1e-9);
                scene.polyline(run, 2.2, segment_light(*shell, climb));
            }
            let middle = line.len() / 2;
            if middle >= 2 && middle + 2 < line.len() {
                let heading = line[middle + 2] - line[middle - 2];
                let reach = heading.gp(0.14 / heading.ideal_norm().max(1e-9));
                scene.arrow(
                    line[middle - 2],
                    reach,
                    2.2,
                    9.0,
                    segment_light(*shell, 0.0),
                );
            }
        }
    });
}

/// The spectrum of inverse eigenvalues, the shown modes marked.
fn draw_spectrum(c: &mut Canvas, rect: Rect, s: &Shown) {
    let n = s.mu.len();
    let largest = s.mu.iter().map(|m| m.abs()).fold(0.0, f64::max) as f32;
    let ax = Axes::new(
        rect,
        [-4.0, n as f32 + 3.0],
        [-1.15 * largest, 1.15 * largest],
    );
    ax.frame(c, "THE CHIRAL SPECTRUM, 1 / LAMBDA", "MODE", "");
    ax.line(
        c,
        Point2::xy(-4.0, 0.0),
        Point2::xy(n as f32 + 3.0, 0.0),
        1.0,
        palette::GRID.faded(3.0),
    );
    for (k, &m) in s.mu.iter().enumerate() {
        let l = if m < 0.0 {
            palette::SKY
        } else {
            palette::ORANGE
        };
        ax.scatter(
            c,
            &[Point2::xy(k as f32, m as f32)],
            Marker::Dot,
            3.0,
            l.faded(0.8),
        );
    }
    for (&k, l) in s.shown.iter().zip([palette::ORANGE, palette::GREEN]) {
        ax.scatter(
            c,
            &[Point2::xy(k as f32, s.mu[k] as f32)],
            Marker::Ring,
            9.0,
            l,
        );
    }
}

/// The relaxation's energy over the Taylor bound, falling towards it.
fn draw_energy(c: &mut Canvas, rect: Rect, s: &Shown, now: usize) {
    let frames = s.energies.len();
    let top = (s.energies[0] / s.taylor * 1.1) as f32;
    let ax = Axes::new(rect, [0.0, (frames - 1) as f32], [0.0, top]);
    ax.frame(c, "ENERGY OVER THE TAYLOR BOUND", "FRAME", "");
    ax.dashed(
        c,
        &[Point2::xy(0.0, 1.0), Point2::xy((frames - 1) as f32, 1.0)],
        1.0,
        4.0,
        palette::YELLOW.faded(0.7),
    );
    let curve: Vec<Point2> = s
        .energies
        .iter()
        .enumerate()
        .map(|(k, e)| Point2::xy(k as f32, (e / s.taylor) as f32))
        .collect();
    ax.polyline(c, &curve, 1.6, palette::INK);
    ax.scatter(c, &curve[now..=now], Marker::Dot, 7.0, palette::ORANGE);
}

fn draw(c: &mut Canvas, s: &Shown, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let now = frame((t / SECONDS).rem_euclid(1.0), s.relaxation.len());
    let left = Rect::new(0.0, h * 0.12, w * 0.62, h * 0.98);
    let views = [
        (OBLIQUE, "RELAXING AT FIXED HELICITY"),
        (ABOVE, "FROM ABOVE"),
    ];
    for (i, (view, label)) in views.into_iter().enumerate() {
        let rect = left.column(i, 2).inset(4.0, 2.0, 4.0, 2.0);
        let panel = rect.part(0.0, 0.0, 1.0, 0.72);
        draw_torus(c, panel, s, &s.relaxation[now], view);
        label_under(c, panel, label, 16.0, 0.8);
    }
    draw_energy(
        c,
        left.part(0.08, 0.76, 0.95, 0.98)
            .inset(30.0, 18.0, 6.0, 20.0),
        s,
        now,
    );
    let right = Rect::new(w * 0.62, h * 0.12, w, h * 0.98);
    draw_spectrum(
        c,
        right.part(0.0, 0.0, 1.0, 0.36).inset(36.0, 22.0, 8.0, 24.0),
        s,
    );
    let labels = ["THE TAYLOR STATE", "AN EXCITED MODE"];
    for (i, label) in labels.into_iter().enumerate() {
        let rect = right
            .part(0.0, 0.38, 1.0, 1.0)
            .column(i, 2)
            .inset(2.0, 2.0, 2.0, 2.0);
        draw_torus(c, rect, s, &s.modes[i], OBLIQUE);
        label_under(c, rect, label, 14.0, 0.9);
    }
    caption(
        c,
        "BELTRAMI FIELDS: MAGNETIC LINKAGE IN A SOLID TORUS",
        "WHITNEY EDGE FORMS IN SPACETIME (X+Y+Z+T-); THE HELICITY'S EIGENMODES AGAINST THE ENERGY CURL ONTO THEMSELVES",
    );
}

fn main() {
    run_with(
        Anim::new("beltrami fields", SECONDS).size(960, 540),
        shown,
        draw,
    );
}
