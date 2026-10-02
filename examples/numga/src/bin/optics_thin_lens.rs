//! numga's `optics/thin_lens`: Gaussian optics in PGA2D. A ray is a line, and a thin lens is a
//! linear map on lines (an extensor): at the origin it shears a line's `x` coefficient by the
//! line's incidence with the origin over the focal length, `L ↦ L - (L ∨ O) x / f`. A lens
//! elsewhere is that map conjugated by a motor, and a system of elements is a composition, one
//! map for the whole train. The animation shows a train of a lens, a prism, a lens and a
//! mirror with the first lens sliding and tilting and the mirror rocking; the image stays sharp
//! because every element is a collineation.

use gax::pga2d::{Line, Motor, Point};
use gax_numga_examples::{Anim, Axes, Canvas, Marker, backdrop, caption, palette, plot, run};

mod optics {
    use super::*;

    pub type L = Line<(), f64>;
    pub type P = Point<(), f64>;
    /// A map on lines.
    pub type LineMap = Line<(Line,), f64>;
    pub type M = gax::Unit<Motor<(), f64>>;

    /// The origin.
    pub fn origin() -> P {
        Point::xy(0.0, 0.0)
    }

    /// The home plane `x = 0`, where every train element is defined.
    pub fn home() -> L {
        Line::new(1.0, 0.0, 0.0)
    }

    /// The vertical line `x = a`.
    pub fn vertical(a: f64) -> L {
        Line::new(1.0, 0.0, -a)
    }

    /// A thin lens of focal length `focal` in the home plane: `L - (L ∨ O) x / f`.
    pub fn lens_at_home(focal: f64) -> LineMap {
        let l = Line::slot();
        l - (l & origin()) * home().gp(1.0 / focal)
    }

    /// A thin lens in a vertical `plane`: the lens at home conjugated by the translation onto
    /// the plane.
    pub fn thin_lens(plane: L, focal: f64) -> LineMap {
        // The plane's signed offset from the origin along x.
        let offset = -(plane & origin()).s() / plane.e1();
        let shift = Motor::translation(offset, 0.0);
        shift >> lens_at_home(focal).of(shift << Line::slot())
    }

    /// The train's elements, each a map on lines defined in the home plane: a lens of focal
    /// length 1, a thin prism (the same slope change for every height: a shear by the incidence
    /// with an ideal point), a lens of focal length -2, and a flat mirror (the sandwich by the
    /// home plane).
    pub fn elements() -> [LineMap; 4] {
        let l = Line::slot();
        let up = Point::direction(0.0, 1.0);
        [
            lens_at_home(1.0),
            l - (l & up) * home().gp(0.15),
            lens_at_home(-2.0),
            home().normalized() >> Line::slot(),
        ]
    }

    /// Place each element by its motor and pass the bundle through: the element planes, the
    /// bundle before and after each element, and the train as one map.
    pub fn trace(
        rays: &[L],
        motors: &[M; 4],
        elements: &[LineMap; 4],
    ) -> (Vec<L>, Vec<Vec<L>>, LineMap) {
        let mut train = Line::slot();
        let mut legs = vec![rays.to_vec()];
        for (motor, element) in motors.iter().zip(elements) {
            let placed = *motor >> element.of(*motor << Line::slot());
            let next = legs
                .last()
                .expect("a leg")
                .iter()
                .map(|r| placed.of(*r))
                .collect();
            legs.push(next);
            train = placed.of(train);
        }
        (motors.iter().map(|m| *m >> home()).collect(), legs, train)
    }

    /// The point where two lines meet.
    pub fn meet(a: L, b: L) -> P {
        a ^ b
    }

    /// One frame of the train at phase `t` (radians): the subject, the planes, the legs, the
    /// composed train and the image.
    pub fn train(t: f64) -> (P, Vec<L>, Vec<Vec<L>>, LineMap, P) {
        let subject = Point::xy(-1.0, 0.5);
        let fan: Vec<L> = (0..7)
            .map(|k| subject & Point::xy(1.0, -0.6 + 1.2 * k as f64 / 6.0))
            .collect();
        let o = origin();
        let motors = [
            // The first lens slides and tilts.
            Motor::translation(1.0 + 0.3 * t.sin(), 0.0)
                * Motor::rotation(o, 0.3 * (2.0 * t).sin()),
            // The prism and the second lens are fixed.
            Motor::translation(1.9, 0.0),
            Motor::translation(2.2, 0.0),
            // The mirror rocks about its pivot, turned clockwise so that it sends the bundle up.
            Motor::translation(3.2, 0.0)
                * Motor::rotation(o, -(core::f64::consts::FRAC_PI_4 + 0.1 * t.cos())),
        ];
        let (planes, legs, composed) = trace(&fan, &motors, &elements());
        let back: Vec<L> = fan.iter().map(|r| composed.of(*r)).collect();
        let image = meet(back[0], back[back.len() - 1]);
        (subject, planes, legs, composed, image)
    }

    /// One lens imaging a point, and two lenses focusing parallel rays: the bundles leg by leg
    /// as `(rays, start plane, stop plane)`, the lens planes, and the image and focus.
    #[allow(clippy::type_complexity)]
    pub fn lenses() -> (
        Vec<(Vec<L>, L, L)>,
        Vec<(Vec<L>, L, L)>,
        P,
        P,
        [(L, f64); 2],
    ) {
        let (plane_1, focal_1) = (vertical(1.5), 1.0);
        let (plane_2, focal_2) = (vertical(2.1), 0.5);
        let (lens_1, lens_2) = (thin_lens(plane_1, focal_1), thin_lens(plane_2, focal_2));
        // Rays from an object point through a pupil of points on the lens plane; every ray
        // after the lens passes through one point, the image.
        let obj = Point::xy(-2.0, 0.5);
        let pupil: Vec<P> = (0..9)
            .map(|k| Point::xy(1.5, -0.8 + 1.6 * k as f64 / 8.0))
            .collect();
        let rays: Vec<L> = pupil.iter().map(|p| obj & *p).collect();
        let out: Vec<L> = rays.iter().map(|r| lens_1.of(*r)).collect();
        let image = meet(out[0], out[1]);
        // Two lenses: one map. Parallel rays meet after it in the back focal point.
        let system = lens_2.of(lens_1);
        let along_x = Point::direction(1.0, 0.0);
        let parallel: Vec<L> = pupil.iter().map(|p| along_x & *p).collect();
        let focused: Vec<L> = parallel.iter().map(|r| system.of(*r)).collect();
        let focus = meet(focused[0], focused[1]);
        let after = |p: P| vertical(p.to_euclidean()[0] + 0.25);
        let one = vec![
            (rays, vertical(-2.0), plane_1),
            (out, plane_1, after(image)),
        ];
        let two = vec![
            (parallel.clone(), vertical(1.0), plane_1),
            (
                parallel.iter().map(|r| lens_1.of(*r)).collect(),
                plane_1,
                plane_2,
            ),
            (focused, plane_2, after(focus)),
        ];
        (
            one,
            two,
            image,
            focus,
            [(plane_1, focal_1), (plane_2, focal_2)],
        )
    }
}

use optics::*;

fn xy(p: P) -> [f32; 2] {
    let [x, y] = p.to_euclidean();
    [x as f32, y as f32]
}

/// The bundle between two planes.
fn rays(ax: &Axes, c: &mut Canvas, rays: &[L], start: L, stop: L, color: gax_numga_examples::Rgb) {
    for r in rays {
        ax.line(c, xy(*r ^ start), xy(*r ^ stop), 1.2, color, 0.9);
    }
}

/// An element's plane between heights `-h` and `h`.
fn plane(ax: &Axes, c: &mut Canvas, plane: L, h: f64) {
    let a = xy(plane ^ Line::new(0.0, 1.0, h));
    let b = xy(plane ^ Line::new(0.0, 1.0, -h));
    ax.line(c, a, b, 2.5, palette::grid(), 1.0);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.width as f32, c.height as f32);
    let phase = f64::from(t) / 4.32 * core::f64::consts::TAU;
    // The train, animated, on the left.
    let ax = Axes::equal(
        plot::inset([0.0, 0.0, w * 0.64, h], 20.0, 70.0, 10.0, 20.0),
        [1.7, 0.85],
        2.2,
    );
    let (subject, planes, legs, _, image) = train(phase);
    // Each ray's direction of travel, carried along: a line has none of its own, so each leg
    // takes the direction of its line that continues forward through a lens or prism, and
    // that turns back across the plane at the mirror (the last element).
    let along = |l: L| {
        let (a, b) = (l.e1() as f32, l.e2() as f32);
        let n = (a * a + b * b).sqrt();
        [-b / n, a / n]
    };
    let dot = |u: [f32; 2], v: [f32; 2]| u[0] * v[0] + u[1] * v[1];
    let mut start: Vec<[f32; 2]> = vec![xy(subject); legs[0].len()];
    let mut dir: Vec<[f32; 2]> = legs[0]
        .iter()
        .map(|r| {
            let d = along(*r);
            let to = xy(*r ^ planes[0]);
            let s = xy(subject);
            if dot(d, [to[0] - s[0], to[1] - s[1]]) >= 0.0 {
                d
            } else {
                [-d[0], -d[1]]
            }
        })
        .collect();
    for (k, pl) in planes.iter().enumerate() {
        let colour = palette::series(k);
        let normal = [pl.e1() as f32, pl.e2() as f32];
        for (i, r) in legs[k].iter().enumerate() {
            let stop = xy(*r ^ *pl);
            ax.line(c, start[i], stop, 1.3, colour, 0.9);
            start[i] = xy(legs[k + 1][i] ^ *pl);
            let d = along(legs[k + 1][i]);
            let mirror = k == planes.len() - 1;
            // Through: the same side of the plane as before; at the mirror: the other side.
            let same = dot(d, normal) * dot(dir[i], normal) >= 0.0;
            dir[i] = if same != mirror { d } else { [-d[0], -d[1]] };
        }
        plane(&ax, c, *pl, 0.8);
    }
    for (s, d) in start.iter().zip(&dir) {
        ax.line(
            c,
            *s,
            [s[0] + 2.5 * d[0], s[1] + 2.5 * d[1]],
            1.3,
            palette::series(4),
            0.9,
        );
    }
    ax.scatter(c, &[xy(subject)], Marker::Dot, 9.0, palette::series(0), 1.0);
    ax.scatter(c, &[xy(image)], Marker::Star, 13.0, palette::yellow(), 1.0);
    caption(
        c,
        "THIN LENS: AN OPTICAL TRAIN AS ONE MAP ON LINES",
        "LENS, PRISM, LENS, MIRROR (PGA2D)",
    );
    // One lens and two lenses, still, on the right.
    let (one, two, image_1, focus, lens_planes) = lenses();
    let right = [w * 0.64, 0.0, w, h];
    for (k, (legs, title)) in [(one, "ONE LENS"), (two, "TWO LENSES")]
        .into_iter()
        .enumerate()
    {
        let top = right[1] + 60.0 + k as f32 * (h - 60.0) / 2.0;
        let rect = plot::inset(
            [right[0], top, right[2], top + (h - 60.0) / 2.0],
            14.0,
            24.0,
            14.0,
            14.0,
        );
        let ax = Axes::equal(rect, [0.4, 0.0], 1.1);
        for (i, (r, start, stop)) in legs.iter().enumerate() {
            rays(&ax, c, r, *start, *stop, palette::series(i + 1));
        }
        for (pl, _) in &lens_planes[..=k] {
            plane(&ax, c, *pl, 1.0);
        }
        let mark = if k == 0 { image_1 } else { focus };
        ax.scatter(c, &[xy(mark)], Marker::Star, 11.0, palette::yellow(), 1.0);
        ax.text(
            c,
            [ax.x[0] + 0.1, ax.y[1] - 0.25],
            title,
            11.0,
            palette::ink(),
            gax_numga_examples::Align::Left,
        );
    }
}

fn main() {
    run(Anim::new("thin lens", 4.32).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::optics::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// The composed train maps the fan onto the last leg, and every output ray passes through
    /// the image, for every placement.
    #[test]
    fn train_images_the_subject_through_every_placement() {
        for k in 0..12 {
            let t = core::f64::consts::TAU * k as f64 / 12.0;
            let (_, _, legs, train, image) = train(t);
            for (first, last) in legs[0].iter().zip(&legs[legs.len() - 1]) {
                let out = train.of(*first);
                assert!((out ^ image).c[0].abs() < 1e-10 * (1.0 + image.e12().abs()));
                for (a, b) in out.c.iter().zip(last.c) {
                    assert!(close(*a, b, 1e-10), "{out:?} vs {last:?}");
                }
            }
        }
    }

    /// The thin lens equation for one lens, and Gullstrand's back focal distance for two; the
    /// focus lies on the axis.
    #[test]
    fn lens_equations() {
        let (one, _, image, focus, [(plane_1, f1), (plane_2, f2)]) = lenses();
        for r in &one[1].0 {
            assert!((*r ^ image).c[0].abs() < 1e-11);
        }
        let x = |p: P| p.to_euclidean()[0];
        let lens_1_x = 1.5;
        let d_obj = lens_1_x - (-2.0);
        let d_img = x(image) - lens_1_x;
        assert!(close(1.0 / d_obj + 1.0 / d_img, 1.0 / f1, 1e-12));
        let gap = 2.1 - 1.5;
        let f_eff = 1.0 / (1.0 / f1 + 1.0 / f2 - gap / (f1 * f2));
        let back_focal = f_eff * (f1 - gap) / f1;
        assert!(close(x(focus) - 2.1, back_focal, 1e-12));
        assert!(focus.to_euclidean()[1].abs() < 1e-12);
        let _ = (plane_1, plane_2);
    }

    #[test]
    fn a_frame_draws() {
        let mut draw = super::draw;
        let c = gax_numga_examples::app::frame(
            &gax_numga_examples::Anim::new("t", 1.0).size(320, 180),
            0.5,
            &mut draw,
        );
        assert!(c.mean()[0] > 0.0);
    }
}
