//! numga's examples, ported to gax: shared drawing code. Each example is a binary in
//! `src/bin/` that computes with gax and draws its frames on a [`Canvas`]; [`run`] shows them in
//! a window or writes a GIF or PNG.
//!
//! The ports follow the examples of numga (Eelco Hoogendoorn,
//! <https://github.com/EelcoHoogendoorn/numga>), one binary each, with their tests.

pub mod app;
pub mod canvas;
pub mod colormap;
pub mod contour;
pub mod font;
pub mod palette;
pub mod plot;
pub mod scene3;
pub mod view;

pub use app::{Anim, run};
pub use canvas::{Canvas, Px, Rgb};
pub use font::Align;
pub use plot::{Axes, Marker};
pub use scene3::Scene3;
pub use view::{Camera, Lens, View2};

/// The usual backdrop: a dark vertical gradient.
pub fn backdrop(c: &mut Canvas) {
    c.backdrop(palette::top(), palette::bottom());
}

/// A title in the top left corner, and an optional caption under it.
pub fn caption(c: &mut Canvas, title: &str, sub: &str) {
    let s = (c.height as f32 / 30.0).clamp(10.0, 22.0);
    c.text(title, s * 0.8, s * 1.6, s, palette::ink(), Align::Left);
    if !sub.is_empty() {
        c.text(sub, s * 0.8, s * 3.0, s * 0.7, palette::grid(), Align::Left);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_covers_its_pixels_and_not_others() {
        let mut c = Canvas::new(20, 10);
        c.line([2.0, 5.0], [18.0, 5.0], 2.0, [1.0; 3], 1.0);
        assert!(c.get(10, 4)[0] > 0.9 && c.get(10, 5)[0] > 0.9);
        assert!(c.get(10, 1)[0] < 1e-6 && c.get(0, 5)[0] < 1e-6);
    }

    #[test]
    fn a_square_fills_its_area() {
        let mut c = Canvas::new(10, 10);
        c.fill(
            &[[2.0, 2.0], [8.0, 2.0], [8.0, 8.0], [2.0, 8.0]],
            [1.0; 3],
            1.0,
        );
        assert!((c.mean()[0] - 0.36).abs() < 1e-3);
    }

    #[test]
    fn a_circle_s_contour_is_a_closed_ring_of_segments() {
        let segs = contour::of_fn(|x, y| x * x + y * y, [-2.0, 2.0], [-2.0, 2.0], 41, 1.0);
        assert!(segs.len() > 20);
        for [a, b] in segs {
            for p in [a, b] {
                assert!(
                    ((p[0] * p[0] + p[1] * p[1]).sqrt() - 1.0).abs() < 0.02,
                    "{p:?}"
                );
            }
        }
    }

    #[test]
    fn axes_map_data_to_pixels_and_back() {
        let ax = Axes::new([10.0, 20.0, 110.0, 220.0], [0.0, 1.0], [-1.0, 1.0]);
        assert_eq!(ax.px([0.0, -1.0]), [10.0, 220.0]);
        assert_eq!(ax.px([1.0, 1.0]), [110.0, 20.0]);
        let back = ax.data(ax.px([0.25, 0.5]));
        assert!((back[0] - 0.25).abs() < 1e-6 && (back[1] - 0.5).abs() < 1e-6);
        let log = Axes::new([0.0, 0.0, 100.0, 100.0], [0.0, 1.0], [1e-3, 1.0]).log_y();
        assert!((log.px([0.0, 1e-2])[1] - 200.0 / 3.0).abs() < 1e-3);
    }

    #[test]
    fn a_scene_draws_far_to_near() {
        let cam = Camera::looking(
            64,
            64,
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            Lens::Perspective(0.8),
        );
        let mut s = Scene3::new(cam);
        // A red square in front of a green one: the centre is red.
        s.quad(
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 0.0, 0.0],
            1.0,
        );
        s.quad(
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [0.0, 1.0, 0.0],
            1.0,
        );
        let mut c = Canvas::new(64, 64);
        s.draw(&mut c);
        let p = c.get(32, 32);
        assert!(p[0] > 0.5 && p[1] < 0.01, "{p:?}");
    }

    #[test]
    fn the_camera_does_not_mirror() {
        // Seen from above (+z), the x axis points right and the y axis up on screen.
        let cam = Camera::looking(
            100,
            100,
            [0.0, -0.001, 10.0],
            [0.0, 0.0, 0.0],
            Lens::Perspective(0.8),
        );
        let o = cam.px([0.0, 0.0, 0.0]).expect("in view");
        let x = cam.px([1.0, 0.0, 0.0]).expect("in view");
        let y = cam.px([0.0, 1.0, 0.0]).expect("in view");
        assert!(x[0] > o[0] + 1.0, "{o:?} {x:?}");
        assert!(y[1] < o[1] - 1.0, "{o:?} {y:?}");
        // The ray through a point's pixel passes through the point.
        let (origin, dir) = cam.ray(x);
        let [ox, oy, oz] = origin.to_euclidean();
        let t = oz / -dir.e021();
        let hit = [ox + t * dir.e032(), oy + t * dir.e013()];
        assert!(
            (hit[0] - 1.0).abs() < 1e-3 && hit[1].abs() < 1e-3,
            "{hit:?}"
        );
    }
}
