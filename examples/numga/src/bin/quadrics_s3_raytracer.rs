//! numga's `quadrics/s3_raytracer`: rendering ellipsoids on the 3-sphere, in `Cl(4)` declared
//! with `gax::algebra!`, by projection: each body's outline is a conic in the pixel chart, and
//! the conic with the eye's polar gives the depth of the first hit (see `shared/quadrics_s3.rs`).
//! Four identical ellipsoids sit along the line of sight at 0.7, 1.4, 2.1 and 2.6 rad. The
//! spherical signature shows in their sizes: past a quarter turn the great circles from the eye
//! reconverge on its antipode, and the farthest ellipsoid looms larger than the middle ones. The
//! eye walks toward them and back.

#[path = "../shared/quadrics_s3.rs"]
mod s3;

use gax_numga_examples::{Anim, Canvas, backdrop, canvas, caption, run};
use s3::*;

mod walk {
    use super::s3::*;

    /// The scene from an eye: four copies of one ellipsoid, angular half-widths 0.2, 0.28 and
    /// 0.15, carried to increasing angular distances along `+x`, offset sideways and up so they
    /// don't overlap, and turned a little about the line of sight.
    pub fn walk(eye: Motor) -> View {
        let shape = ellipsoid([0.2f64.tan(), 0.28f64.tan(), 0.15f64.tan()]);
        let distances = [0.7, 1.4, 2.1, 2.6];
        let sideways = [-0.4, 0.4, -0.4, 0.4];
        let upward = [-0.3, -0.3, 0.3, 0.3];
        let surfaces = (0..4)
            .map(|k| {
                let placed = motion(along([distances[k], 0.0, 0.0]))
                    * motion(along([0.0, sideways[k], 0.0]))
                    * motion(along([0.0, 0.0, upward[k]]))
                    * motion(turning([0.0, 0.0, 0.8]));
                // The dual quadric in the world; its inverse maps a point to its polar plane.
                moved_dual(placed, shape).inverse()
            })
            .collect();
        View {
            eye,
            surfaces,
            colors: vec![
                [0.9, 0.3, 0.3],
                [0.3, 0.8, 0.4],
                [0.3, 0.5, 0.95],
                [0.95, 0.8, 0.3],
            ],
            light: direction([-0.4, 0.6, 0.7]),
            fov: 80f64.to_radians(),
        }
    }

    /// The eye a distance `s` along the x geodesic; the scene stays fixed in the world.
    pub fn eye(s: f64) -> Motor {
        motion(along([s, 0.0, 0.0]))
    }
}

use walk::*;

const SECONDS: f32 = 8.0;

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    // Toward the bodies and back: the eye's distance along the geodesic from 0 to 1.2.
    let phase = f64::from(t / SECONDS) * core::f64::consts::TAU;
    let s = 0.6 - 0.6 * phase.cos();
    let view = walk(eye(s));
    let tracer = view.tracer();
    let (w, h) = (c.width as f64, c.height as f64);
    let background = canvas::srgb(0.02, 0.02, 0.02);
    c.shade(2, |x, y| {
        let rgb = tracer
            .color(chart(view.fov, f64::from(x), f64::from(y), w, h))
            .map(|[r, g, b]| canvas::srgb(r as f32, g as f32, b as f32));
        Some(rgb.unwrap_or(background))
    });
    caption(
        c,
        "ELLIPSOIDS ON THE 3-SPHERE",
        &format!("EQUAL BODIES AT 0.7, 1.4, 2.1, 2.6 RAD, EYE AT {s:.2}"),
    );
}

fn main() {
    run(
        Anim::new("s3 raytracer", SECONDS).size(480, 360).scale(2),
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::s3::*;
    use super::walk::*;

    /// The scenario's checks. A ray is inside a body's outline cone exactly when its great circle
    /// `origin + pixel λ` meets the body, when the quadratic `a λ² + 2 b λ + c` has real roots
    /// (compared away from the outline itself); the screen conic is `-disc / c²`; and reprojected
    /// hits lie on their surfaces.
    #[test]
    fn outlines_and_depths_agree_with_the_rays() {
        let identity = eye(0.0);
        let scene = walk(identity);
        let o = origin();
        let (rows, cols) = (90, 120);
        for (body, surface) in scene.surfaces.iter().enumerate() {
            let cone = outline(identity, *surface);
            let (conic, polar) = project(identity, *surface);
            let k_eye = surface.of(o);
            let mut discs = Vec::new();
            for r in 0..rows {
                for col in 0..cols {
                    let px = chart(scene.fov, col as f64, r as f64, cols as f64, rows as f64);
                    let ray: Point = px.cast::<Trivector>();
                    let k_dir = surface.of(ray);
                    let (a, b, c) = ((ray & k_dir).s(), (ray & k_eye).s(), (o & k_eye).s());
                    let disc = b * b - a * c;
                    discs.push((px, disc, c));
                }
            }
            let largest = discs.iter().map(|d| d.1.abs()).fold(0.0, f64::max);
            let mut covered = 0;
            for (px, disc, c) in discs {
                let is_inside = inside(cone, px);
                if disc.abs() > 1e-3 * largest {
                    assert_eq!(is_inside, disc >= 0.0, "body {body}");
                }
                let value = conic.of(px).of(px).s();
                assert!((value - -disc / (c * c)).abs() < 1e-8, "{value} {disc}");
                if is_inside {
                    covered += 1;
                    if covered % 10 == 0 {
                        let depth = reproject(conic, polar, px);
                        let p = hit(identity, depth, px);
                        assert!((p & surface.of(p)).s().abs() < 1e-8);
                    }
                }
            }
            assert!(covered > 0, "body {body} is in view");
        }
    }

    /// The bodies past a quarter turn (at 2.1 and 2.6 rad) cover more of the view than the one at
    /// 1.4 rad: the great circles from the eye reconverge on its antipode.
    #[test]
    fn the_farthest_body_looms_larger() {
        let identity = eye(0.0);
        let scene = walk(identity);
        let count = |surface: Quadric| {
            let cone = outline(identity, surface);
            let mut n = 0;
            for r in 0..90 {
                for col in 0..120 {
                    let px = chart(scene.fov, col as f64, r as f64, 120.0, 90.0);
                    if inside(cone, px) {
                        n += 1;
                    }
                }
            }
            n
        };
        let sizes: Vec<usize> = scene.surfaces.iter().map(|s| count(*s)).collect();
        assert!(sizes[2] > sizes[1] && sizes[3] > sizes[1], "{sizes:?}");
    }

    #[test]
    fn a_frame_draws() {
        let mut draw = super::draw;
        let c = gax_numga_examples::app::frame(
            &gax_numga_examples::Anim::new("t", 1.0).size(120, 90),
            1.0,
            &mut draw,
        );
        assert!(c.mean()[0] > 0.0);
    }
}
