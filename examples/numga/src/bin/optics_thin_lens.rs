//! numga's `optics/thin_lens`: Gaussian optics in PGA2D. A ray is a line, and a thin lens is a
//! linear map on lines (an extensor): at the origin it shears a line's `x` coefficient by the
//! line's incidence with the origin over the focal length, `L ↦ L - (L ∨ O) x / f`. A lens
//! elsewhere is that map conjugated by a motor, and a system of elements is a composition, one
//! map for the whole train. The animation shows a train of a lens, a prism, a lens and a
//! mirror with the first lens sliding and tilting and the mirror rocking; the image stays sharp
//! because every element is a collineation.

use gax::pga2d::{Line, Motor, Point};

use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, Point2, Rect, backdrop, caption, palette, run,
    signal::{phasor, wave},
};

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

    /// A thin lens in a `plane`: the lens at home conjugated by the motion that carries the
    /// home plane onto it.
    pub fn thin_lens(plane: L, focal: f64) -> LineMap {
        placed(Motor::between(home(), plane), lens_at_home(focal))
    }

    /// An element defined at home, placed by a motor: `m >> element(m << line)`.
    pub fn placed(m: M, element: LineMap) -> LineMap {
        m >> element.of(m << Line::slot())
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
            let placed = placed(*motor, *element);
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
            Motor::translation(1.0 + 0.3 * wave(t), 0.0) * Motor::rotation(o, 0.3 * wave(2.0 * t)),
            // The prism and the second lens are fixed.
            Motor::translation(1.9, 0.0),
            Motor::translation(2.2, 0.0),
            // The mirror rocks about its pivot, turned clockwise so that it sends the bundle up.
            Motor::translation(3.2, 0.0)
                * Motor::rotation(o, -(core::f64::consts::FRAC_PI_4 + 0.1 * phasor(t).e20())),
        ];
        let (planes, legs, composed) = trace(&fan, &motors, &elements());
        let back: Vec<L> = fan.iter().map(|r| composed.of(*r)).collect();
        let image = back[0] ^ back[back.len() - 1];
        (subject, planes, legs, composed, image)
    }

    /// A bundle of rays between two planes.
    pub struct Leg {
        pub rays: Vec<L>,
        pub start: L,
        pub stop: L,
    }

    /// One lens imaging a point, and two lenses focusing parallel rays.
    pub struct Bench {
        /// The bundle through one lens, leg by leg.
        pub one: Vec<Leg>,
        /// The bundle through two lenses, leg by leg.
        pub two: Vec<Leg>,
        /// Where the one lens images the object point.
        pub image: P,
        /// Where the two lenses focus the parallel rays.
        pub focus: P,
        /// The lens planes and focal lengths.
        pub lenses: [(L, f64); 2],
    }

    pub fn lenses() -> Bench {
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
        let image = out[0] ^ out[1];
        // Two lenses: one map. Parallel rays meet after it in the back focal point.
        let system = lens_2.of(lens_1);
        let along_x = Point::direction(1.0, 0.0);
        let parallel: Vec<L> = pupil.iter().map(|p| along_x & *p).collect();
        let focused: Vec<L> = parallel.iter().map(|r| system.of(*r)).collect();
        let focus = focused[0] ^ focused[1];
        // The bundles stop a little past the point they meet in.
        let after = |p: P| Motor::translation(0.25, 0.0) >> (p & Point::direction(0.0, 1.0));
        let leg = |rays: Vec<L>, start: L, stop: L| Leg { rays, start, stop };
        let through_1 = parallel.iter().map(|r| lens_1.of(*r)).collect();
        Bench {
            one: vec![
                leg(rays, vertical(-2.0), plane_1),
                leg(out, plane_1, after(image)),
            ],
            two: vec![
                leg(parallel, vertical(1.0), plane_1),
                leg(through_1, plane_1, plane_2),
                leg(focused, plane_2, after(focus)),
            ],
            image,
            focus,
            lenses: [(plane_1, focal_1), (plane_2, focal_2)],
        }
    }
}

use optics::*;

/// The bundle between two planes.
fn rays(ax: &Axes, c: &mut Canvas, leg: &Leg, color: Light) {
    for r in &leg.rays {
        ax.line(c, *r ^ leg.start, *r ^ leg.stop, 1.2, color.faded(0.9));
    }
}

/// An element's plane between heights `-h` and `h`.
fn plane(ax: &Axes, c: &mut Canvas, plane: L, h: f64) {
    let a = plane ^ Line::new(0.0, 1.0, h);
    let b = plane ^ Line::new(0.0, 1.0, -h);
    ax.line(c, a, b, 2.5, palette::grid());
}

/// A ray's heading: the unit direction along a line (its point at infinity), flipped to the
/// side of `plane` that `towards` gives, the sign of its pairing with the plane.
fn heading(ray: L, plane: L, towards: f64) -> P {
    let d = ray ^ Line::new(0.0, 0.0, 1.0);
    let d = d.gp(d.ideal_norm().recip());
    if (plane & d).s() * towards >= 0.0 {
        d
    } else {
        -d
    }
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let phase = f64::from(t) / 4.32 * core::f64::consts::TAU;
    // The train, animated, on the left.
    let left = Rect::new(0.0, 0.0, w * 0.64, h);
    let ax = Axes::equal(
        left.inset(20.0, 70.0, 10.0, 20.0),
        Point::xy(1.7, 0.85),
        2.2,
    );
    let (subject, planes, legs, _, image) = train(phase);
    // Each ray's direction of travel, carried along: a line has none of its own, so each leg
    // takes the direction of its line that continues forward through a lens or prism, and
    // that turns back across the plane at the mirror (the last element). The side of a plane
    // a direction points to is the sign of its pairing with the plane.
    let mut start: Vec<P> = vec![subject; legs[0].len()];
    // From the subject towards the first plane: the side the subject is not on.
    let mut dir: Vec<P> = legs[0]
        .iter()
        .map(|r| heading(*r, planes[0], -(planes[0] & subject).s()))
        .collect();
    for (k, pl) in planes.iter().enumerate() {
        let colour = palette::series(k);
        let mirror = k == planes.len() - 1;
        for (i, r) in legs[k].iter().enumerate() {
            ax.line(c, start[i], *r ^ *pl, 1.3, colour.faded(0.9));
            start[i] = (legs[k + 1][i] ^ *pl).unitized();
            // Through: on to the side it was heading; at the mirror: back.
            let side = (*pl & dir[i]).s();
            dir[i] = heading(legs[k + 1][i], *pl, if mirror { -side } else { side });
        }
        plane(&ax, c, *pl, 0.8);
    }
    for (s, d) in start.iter().zip(&dir) {
        ax.line(c, *s, *s + d.gp(2.5), 1.3, (palette::series(4)).faded(0.9));
    }
    ax.scatter(c, &[subject], Marker::Dot, 9.0, palette::series(0));
    ax.scatter(c, &[image], Marker::Star, 13.0, palette::yellow());
    caption(
        c,
        "THIN LENS: AN OPTICAL TRAIN AS ONE MAP ON LINES",
        "LENS, PRISM, LENS, MIRROR (PGA2D)",
    );
    // One lens and two lenses, still, on the right.
    let bench = lenses();
    // Below the caption, the right side in two halves, one above the other.
    let right = Rect::new(w * 0.64, 60.0, w, h);
    let half = Point2::direction(0.0, right.height() / 2.0);
    for (k, (legs, title)) in [(&bench.one, "ONE LENS"), (&bench.two, "TWO LENSES")]
        .into_iter()
        .enumerate()
    {
        let lo = right.lo + half.gp(k as f32);
        let hi = lo + Point2::direction(right.width(), 0.0) + half;
        let rect = Rect { lo, hi }.inset(14.0, 24.0, 14.0, 14.0);
        // Both benches at one scale, from the object to a little past the image.
        let corners = [Point::xy(-2.0, -0.85), Point::xy(3.2, 0.85)];
        let ax = Axes::fitting(rect, corners, 1.0);
        for (i, leg) in legs.iter().enumerate() {
            rays(&ax, c, leg, palette::series(i + 1));
        }
        for (pl, _) in &bench.lenses[..=k] {
            plane(&ax, c, *pl, 1.0);
        }
        let mark = if k == 0 { bench.image } else { bench.focus };
        ax.scatter(c, &[mark], Marker::Star, 11.0, palette::yellow());
        ax.text(
            c,
            // Just inside the top left corner of the data box.
            ax.at(0.0, 1.0) + Point2::direction(0.1, -0.25),
            title,
            11.0,
            palette::ink(),
            Align::Left,
        );
    }
}

fn main() {
    run(Anim::new("thin lens", 4.32).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::optics::*;
    use gax::ApproxEq;

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
                assert!(out.max_abs_diff(last) < 1e-10, "{out:?} vs {last:?}");
            }
        }
    }

    /// The thin lens equation for one lens, and Gullstrand's back focal distance for two; the
    /// focus lies on the axis.
    #[test]
    fn lens_equations() {
        let Bench {
            one,
            image,
            focus,
            lenses: [(_, f1), (_, f2)],
            ..
        } = lenses();
        for r in &one[1].rays {
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
        // The focus is on the axis, the line `y = 0`.
        let axis = gax::pga2d::Line::new(0.0, 1.0, 0.0);
        assert!((axis & focus.unitized()).s().abs() < 1e-12);
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
