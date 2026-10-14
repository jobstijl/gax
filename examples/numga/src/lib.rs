//! numga's examples, ported to gax: shared drawing code. Each example is a binary in
//! `src/bin/` that computes with gax and draws its frames on a [`Canvas`]; [`run`] shows them in
//! a window or writes a GIF or PNG.
//!
//! The drawing code's geometry goes through gax too, as in `examples/warp`: distances are the
//! norms of joins, turns are rotation motors, the direction at an angle and the angle between
//! two directions are gax's (`direction_from_angle`, `angle_between`, `angle_to`), shading is
//! the inner product of planes. Square roots and trigonometry on floats are denied here
//! (`clippy.toml`); normalizing a random direction is the one exception.
//!
//! The ports follow the examples of numga (Eelco Hoogendoorn,
//! <https://github.com/EelcoHoogendoorn/numga>), one binary each, with their tests.

#![deny(clippy::disallowed_methods)]

pub mod app;
pub mod canvas;
pub mod colormap;
pub mod contour;
pub mod disc;
pub mod font;
pub mod measure;
pub mod mesh;
pub mod palette;
pub mod parallel;
pub mod plot;
pub mod points;
pub mod rng;
pub mod scene3;
pub mod signal;
pub mod view;

pub use app::{Anim, run};
pub use canvas::{Canvas, Rect};
pub use font::Align;
pub use gax_colour::{Light, light};
pub use parallel::par_map;
pub use plot::{Axes, Marker};
pub use points::{
    Dir2, Dir3, ORIGIN2, ORIGIN3, Point2, Point3, Pos2, Pos3, from_above, reach2, reach3, vector2,
    vector3,
};
pub use scene3::Scene3;
pub use view::{Camera, Lens, View2};

/// The usual backdrop: a dark vertical gradient.
pub fn backdrop(c: &mut Canvas) {
    c.backdrop(palette::top(), palette::bottom());
}

/// A title in the top left corner, and an optional caption under it. Placed by the canvas's
/// height (the panels below are laid out by it); the text shrinks to fit the canvas's width.
pub fn caption(c: &mut Canvas, title: &str, sub: &str) {
    let screen = c.rect();
    let s = (screen.height() / 30.0).min(22.0);
    let room = screen.width() - 1.6 * s;
    let fit = |size: f32, text: &str| size.min(room / font::width(text, 1.0).max(1e-6));
    let ink = palette::ink();
    c.text(
        title,
        Point2::xy(s * 0.8, s * 1.6),
        fit(s, title),
        ink,
        Align::Left,
    );
    if !sub.is_empty() {
        let at = Point2::xy(s * 0.8, s * 3.0);
        c.text(sub, at, fit(s * 0.7, sub), ink.faded(0.45), Align::Left);
    }
}

/// A label under a panel's middle: a `rows`-th of the panel's height (between 7 and 12 pixels),
/// shrunk to fit the `share` of its width (less where panels sit close).
pub fn label_under(c: &mut Canvas, rect: Rect, label: &str, rows: f32, share: f32) {
    let fitting = share * rect.width() / font::width(label, 1.0);
    let size = (rect.height() / rows).clamp(7.0, 12.0).min(fitting);
    let at = rect.bottom_middle() - Point2::direction(0.0, size * 0.6);
    c.text(label, at, size, palette::ink(), Align::Center);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// White light at intensity 1.
    const WHITE: Light = light(1.0, 1.0, 1.0, 1.0);

    /// Two pixel points within `eps` of each other.
    fn near(a: Point2, b: Point2, eps: f32) -> bool {
        (a.unitized() & b.unitized()).norm() < eps
    }

    #[test]
    fn a_line_glows_around_a_solid_core() {
        let mut c = Canvas::with_scale(40, 30, 1.0);
        c.line(Point2::xy(5.0, 15.0), Point2::xy(35.0, 15.0), 2.0, WHITE);
        // The core is solid, the glow faint and fading, and beyond its reach nothing.
        assert!((c.get(20, 14)).luma() > 0.95 && (c.get(20, 15)).luma() > 0.95);
        let glow = (c.get(20, 20)).luma();
        assert!(glow > 0.0 && glow < 0.1, "{glow}");
        assert_eq!((c.get(20, 0)).luma(), 0.0);
    }

    #[test]
    fn a_polyline_is_one_stroke() {
        // Its joints do not shine twice: the light where two segments meet is the light along
        // either.
        let mut c = Canvas::with_scale(40, 20, 1.0);
        let pts = [
            Point2::xy(5.0, 10.0),
            Point2::xy(20.0, 10.0),
            Point2::xy(35.0, 10.0),
        ];
        c.polyline(&pts, 2.0, WHITE, false);
        assert!(((c.get(20, 10)).luma() - (c.get(12, 10)).luma()).abs() < 1e-3);
    }

    #[test]
    fn fills_cover_their_area() {
        // A triangle, and a star with a hole by winding.
        let mut c = Canvas::with_scale(40, 40, 1.0);
        let corners = [
            Point2::xy(0.0, 0.0),
            Point2::xy(40.0, 0.0),
            Point2::xy(0.0, 40.0),
        ];
        c.fill(&corners, WHITE, 1.0);
        assert!((c.mean().luma() - 0.5).abs() < 2e-3, "{:?}", c.mean());
        let mut c = Canvas::with_scale(60, 60, 1.0);
        // A pentagram: every second corner of a pentagon, by turns of two fifths.
        let centre = Point2::xy(30.0, 30.0);
        let star: Vec<Point2> = (0..5)
            .map(|k| {
                let turn = gax::pga2d::Motor::rotation(
                    centre,
                    core::f32::consts::TAU * (k * 2) as f32 / 5.0,
                );
                turn >> Point2::xy(55.0, 30.0)
            })
            .collect();
        c.fill(&star, WHITE, 1.0);
        assert!(
            (c.get(30, 30)).luma() > 0.99,
            "non-zero winding fills the centre"
        );
    }

    #[test]
    fn a_square_fills_its_area() {
        let mut c = Canvas::with_scale(10, 10, 1.0);
        let r = Rect::new(2.0, 2.0, 8.0, 8.0);
        c.fill(&[r.lo, r.top_right(), r.hi, r.bottom_left()], WHITE, 1.0);
        assert!((c.mean().luma() - 0.36).abs() < 1e-3);
    }

    #[test]
    fn a_viewport_centres_the_view_in_its_panel() {
        let cam = Camera::looking(
            Rect::new(0.0, 0.0, 100.0, 100.0),
            Point3::xyz(0.0, -5.0, 0.0),
            ORIGIN3,
            Lens::Perspective(0.8),
        )
        .viewport(Rect::new(200.0, 50.0, 300.0, 150.0));
        let o = cam.px(ORIGIN3).expect("in view");
        assert!(near(o, Point2::xy(250.0, 100.0), 1e-3), "{o:?}");
        let (origin, dir) = cam.ray(Point2::xy(250.0, 100.0));
        // From the eye, straight ahead along +y.
        assert!(
            (origin & Point3::xyz(0.0, -5.0, 0.0)).norm() < 1e-4,
            "{origin:?}"
        );
        assert!(
            (dir - Point3::direction(0.0, 1.0, 0.0)).ideal_norm() < 1e-5,
            "{dir:?}"
        );
    }

    #[test]
    fn the_generator_draws_standard_normals() {
        use rng::Draw;
        let mut r = rng::rng(7);
        let n = 20000;
        let xs: Vec<f64> = (0..n).map(|_| r.normal()).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        assert!(
            mean.abs() < 0.03 && (var - 1.0).abs() < 0.05,
            "{mean} {var}"
        );
    }

    #[test]
    fn a_circle_closes_at_its_radius_and_a_drawn_point_reads_back() {
        let centre = Point2::xy(2.0, -1.0);
        let ring = points::circle(centre, 3.0, 12);
        assert_eq!(ring.len(), 13);
        assert!(near(ring[0], ring[12], 1e-5) && near(ring[3], Point2::xy(2.0, 2.0), 1e-5));
        for p in &ring {
            assert!(((*p & centre).norm() - 3.0).abs() < 1e-5, "{p:?}");
        }
        let v = gax::vga2d::Vector::new(0.25, -1.5);
        assert_eq!(vector2(reach2(v)), v);
        let w = gax::vga3d::Vector::new(0.25, -1.5, 0.0);
        assert_eq!(vector3(from_above(reach3(w))), w);
    }

    #[test]
    fn a_grid_reads_between_its_samples() {
        // Two rows of three: the value is the column plus ten times the row.
        let grid = [0.0, 1.0, 2.0, 10.0, 11.0, 12.0];
        assert_eq!(plot::bilinear(&grid, 3, 1.5, 0.5), 6.5);
        assert_eq!(plot::bilinear(&grid, 3, 2.0, 1.0), 12.0);
        assert_eq!(plot::bilinear(&grid, 3, -1.0, 4.0), 10.0, "past the edge");
    }

    #[test]
    fn a_circle_s_contour_is_a_closed_ring_of_segments() {
        // The unit circle: the level 1 of the squared distance from the origin.
        let r2 = |p: Point2| (p & ORIGIN2).norm_squared();
        let segs = contour::of_fn(r2, [-2.0, 2.0], [-2.0, 2.0], 41, 1.0);
        assert!(segs.len() > 20);
        for p in segs.into_iter().flatten() {
            let r = (p & ORIGIN2).norm();
            assert!((r - 1.0).abs() < 0.02, "{p:?}");
        }
    }

    #[test]
    fn axes_map_data_to_pixels_and_back() {
        let ax = Axes::new(Rect::new(10.0, 20.0, 110.0, 220.0), [0.0, 1.0], [-1.0, 1.0]);
        assert!(near(
            ax.px(Point2::xy(0.0, -1.0)),
            Point2::xy(10.0, 220.0),
            1e-4
        ));
        assert!(near(
            ax.px(Point2::xy(1.0, 1.0)),
            Point2::xy(110.0, 20.0),
            1e-4
        ));
        let back = ax.data(ax.px(Point2::xy(0.25, 0.5)));
        assert!(near(back, Point2::xy(0.25, 0.5), 1e-6), "{back:?}");
        assert!(near(ax.at(0.5, 1.0), Point2::xy(0.5, 1.0), 1e-6));
        let log = Axes::new(Rect::new(0.0, 0.0, 100.0, 100.0), [0.0, 1.0], [1e-3, 1.0]).log_y();
        let p = log.px(Point2::xy(0.0, 1e-2));
        assert!(near(p, Point2::xy(0.0, 200.0 / 3.0), 1e-3), "{p:?}");
    }

    #[test]
    fn a_scene_draws_far_to_near() {
        let cam = Camera::looking(
            Rect::new(0.0, 0.0, 64.0, 64.0),
            Point3::xyz(5.0, 0.0, 0.0),
            ORIGIN3,
            Lens::Perspective(0.8),
        );
        let mut s = Scene3::new(cam);
        // A red square in front of a green one: the centre is red.
        let square = |x: f32| {
            [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map(|(y, z)| Point3::xyz(x, y, z))
        };
        let [a, b, c, d] = square(1.0);
        s.quad(a, b, c, d, light(1.0, 0.0, 0.0, 1.0), 1.0);
        let [a, b, c, d] = square(-1.0);
        s.quad(a, b, c, d, light(0.0, 1.0, 0.0, 1.0), 1.0);
        let mut c = Canvas::with_scale(64, 64, 1.0);
        s.draw(&mut c);
        let p = c.get(32, 32);
        let [r, g, _, _] = p.to_premultiplied();
        assert!(r > 0.5 && g < 0.01, "{p:?}");
    }

    #[test]
    fn the_camera_does_not_mirror() {
        // Seen from above (+z), the x axis points right and the y axis up on screen.
        let cam = Camera::looking(
            Rect::new(0.0, 0.0, 100.0, 100.0),
            Point3::xyz(0.0, -0.001, 10.0),
            ORIGIN3,
            Lens::Perspective(0.8),
        );
        let o = cam.px(ORIGIN3).expect("in view");
        let x = cam.px(Point3::xyz(1.0, 0.0, 0.0)).expect("in view");
        let y = cam.px(Point3::xyz(0.0, 1.0, 0.0)).expect("in view");
        assert!((x - o).e20() > 1.0, "{o:?} {x:?}");
        assert!((y - o).e01() < -1.0, "{o:?} {y:?}");
        // The ray through a point's pixel meets the ground plane `z = 0` at the point.
        let (origin, dir) = cam.ray(x);
        let ray = origin & (origin + dir);
        let hit = ray ^ gax::pga3d::Plane::from_normal([0.0, 0.0, 1.0], 0.0);
        assert!(
            (hit & Point3::xyz(1.0, 0.0, 0.0)).norm() < 1e-3 * hit.e123().abs(),
            "{hit:?}"
        );
    }

    #[test]
    fn a_right_angle_field_of_view_reaches_the_edges() {
        // 90 degrees: half the view rises one unit per unit ahead, so the point one unit up at
        // distance one is on the top edge.
        let cam = Camera::looking(
            Rect::new(0.0, 0.0, 100.0, 80.0),
            ORIGIN3,
            Point3::xyz(1.0, 0.0, 0.0),
            Lens::Perspective(core::f32::consts::FRAC_PI_2),
        );
        let top = cam.px(Point3::xyz(1.0, 0.0, 1.0)).expect("in view");
        assert!(near(top, Point2::xy(50.0, 0.0), 1e-3), "{top:?}");
    }
}
