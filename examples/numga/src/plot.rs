//! Plot axes on a canvas: a rectangle of the canvas showing a range of data, linear or
//! logarithmic on each axis, with a frame, ticks and labels; lines, markers, arrows, filled
//! shapes, level lines, images and legends in data coordinates.
//!
//! The axes are a map of the plane: the data box (in logarithms where an axis is logarithmic)
//! onto the rectangle, its lower left corner to the rectangle's bottom left. Everything is
//! placed through it, and pixels are points too.

use crate::canvas::{Canvas, Rect};
use crate::font::{self, Align};
use crate::points::{Dir2, Map2, Point2, Pos2, box_map, finite};
use crate::{contour, palette};
use gax::pga2d::Motor;
use gax_colour::Light;

/// A marker's shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Marker {
    /// A glowing dot.
    Dot,
    /// A hollow circle.
    Ring,
    /// A filled square.
    Square,
    /// A five-pointed star.
    Star,
    /// A diagonal cross.
    Cross,
    /// A filled triangle, pointing up.
    Triangle,
}

/// Axes: data `x` and `y` ranges shown in the rectangle `rect` of the canvas.
#[derive(Clone, Copy, Debug)]
pub struct Axes {
    /// Where on the canvas.
    pub rect: Rect,
    /// The data range across.
    pub x: [f32; 2],
    /// The data range up.
    pub y: [f32; 2],
    /// Logarithmic across.
    pub log_x: bool,
    /// Logarithmic up.
    pub log_y: bool,
    /// The (scaled) data plane onto the canvas, and back.
    to_px: Map2,
    from_px: Map2,
}

/// Panel `i` of `n` side by side on a canvas.
pub fn panel(c: &Canvas, i: usize, n: usize) -> Rect {
    c.rect().column(i, n)
}

/// A round step for about `n` ticks over `span`: 1, 2 or 5 times a power of ten.
fn nice_step(span: f32, n: f32) -> f32 {
    let raw = (span / n).abs().max(1e-30);
    let mag = 10f32.powf(raw.log10().floor());
    let f = raw / mag;
    mag * if f < 1.5 {
        1.0
    } else if f < 3.5 {
        2.0
    } else if f < 7.5 {
        5.0
    } else {
        10.0
    }
}

/// A tick's label: as many decimals as the step needs.
fn label(v: f32, step: f32) -> String {
    if v.abs() < step * 1e-3 {
        return "0".into();
    }
    let digits = (-step.log10().floor()).max(0.0) as usize;
    if v.abs() >= 1e4 || v.abs() < 1e-3 {
        format!("{v:.1e}")
    } else {
        format!("{v:.digits$}")
    }
}

/// The ticks over the range `r`: round steps, or the powers of ten on a logarithmic axis.
fn ticks(r: [f32; 2], log: bool) -> Vec<f32> {
    let (lo, hi) = (r[0].min(r[1]), r[0].max(r[1]));
    if log {
        let (a, b) = (lo.log10().ceil() as i32, hi.log10().floor() as i32);
        (a..=b).map(|k| 10f32.powi(k)).collect()
    } else {
        let step = nice_step(hi - lo, 5.0);
        let first = (lo / step).ceil() as i64;
        let last = (hi / step).floor() as i64;
        (first..=last).map(|k| k as f32 * step).collect()
    }
}

impl Axes {
    /// Axes showing `x` by `y` in `rect`.
    pub fn new(rect: Rect, x: [f32; 2], y: [f32; 2]) -> Axes {
        Axes {
            rect,
            x,
            y,
            log_x: false,
            log_y: false,
            to_px: Map2::zero(),
            from_px: Map2::zero(),
        }
        .mapped()
    }

    /// The same axes with their map built: the data box's lower left and upper right corners
    /// onto the rectangle's bottom left and top right.
    fn mapped(self) -> Axes {
        let lo = self.scaled(Point2::xy(self.x[0], self.y[0]));
        let hi = self.scaled(Point2::xy(self.x[1], self.y[1]));
        let to_px = box_map([lo, hi], [self.rect.bottom_left(), self.rect.top_right()]);
        Axes {
            to_px,
            from_px: to_px.inverse(),
            ..self
        }
    }

    /// A data point as the map takes it: each logarithmic coordinate replaced by its logarithm
    /// (a logarithmic axis is one on each coordinate by definition).
    fn scaled(&self, p: Point2) -> Point2 {
        if !(self.log_x || self.log_y) {
            return p;
        }
        let [x, y] = p.to_euclidean();
        let log = |v: f32, on: bool| if on { v.max(1e-30).ln() } else { v };
        Point2::xy(log(x, self.log_x), log(y, self.log_y))
    }

    /// The inverse of [`Axes::scaled`].
    fn unscaled(&self, p: Point2) -> Point2 {
        if !(self.log_x || self.log_y) {
            return p;
        }
        let [x, y] = p.to_euclidean();
        let exp = |v: f32, on: bool| if on { v.exp() } else { v };
        Point2::xy(exp(x, self.log_x), exp(y, self.log_y))
    }

    /// Axes with equal scales across and up, centred on `centre`, `half_height` units from the
    /// middle to the top (for geometry).
    pub fn equal(rect: Rect, centre: impl Pos2, half_height: f32) -> Axes {
        let half = Point2::direction(half_height * rect.width() / rect.height(), half_height);
        let centre = centre.point2().unitized();
        let ([x0, y0], [x1, y1]) = (
            (centre - half).to_euclidean(),
            (centre + half).to_euclidean(),
        );
        Axes::new(rect, [x0, x1], [y0, y1])
    }

    /// Axes with equal scales, centred on the box around `points` and large enough to show it
    /// whole in `rect`, with `margin` to spare (`1.1` leaves a tenth).
    pub fn fitting<P: Pos2>(rect: Rect, points: impl IntoIterator<Item = P>, margin: f32) -> Axes {
        // The box: its lower left and upper right corners.
        let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
        for p in points {
            for (i, v) in p.point2().to_euclidean().into_iter().enumerate() {
                (lo[i], hi[i]) = (lo[i].min(v), hi[i].max(v));
            }
        }
        let (lo, hi) = (Point2::xy(lo[0], lo[1]), Point2::xy(hi[0], hi[1]));
        // Its diagonal, fitted to the rectangle's aspect.
        let span = hi - lo;
        let aspect = rect.height() / rect.width();
        let half = 0.5 * span.e01().max(span.e20() * aspect) * margin;
        Axes::equal(rect, (lo + hi).unitized(), half)
    }

    /// The same, logarithmic up.
    pub fn log_y(self) -> Axes {
        Axes {
            log_y: true,
            ..self
        }
        .mapped()
    }

    /// The same, logarithmic across.
    pub fn log_x(self) -> Axes {
        Axes {
            log_x: true,
            ..self
        }
        .mapped()
    }

    /// The pixel of data point `p`.
    pub fn px(&self, p: impl Pos2) -> Point2 {
        self.to_px.of(self.scaled(p.point2().unitized()))
    }

    /// The data point at pixel `q`.
    pub fn data(&self, q: Point2) -> Point2 {
        self.unscaled(self.from_px.of(q).unitized())
    }

    /// The data point a fraction `fx` across and `fy` up the axes (for labels and notes).
    pub fn at(&self, fx: f32, fy: f32) -> Point2 {
        let r = self.rect;
        self.data(r.bottom_left() + Point2::direction(fx * r.width(), -fy * r.height()))
    }

    /// Pixels per data unit across (linear axes).
    pub fn scale(&self) -> f32 {
        self.rect.width() / (self.x[1] - self.x[0])
    }

    /// Confine drawing to the axes.
    pub fn clip(&self, c: &mut Canvas) {
        c.clip(self.rect);
    }

    /// A frame with ticks and tick labels, a title above and axis labels.
    pub fn frame(&self, c: &mut Canvas, title: &str, xlabel: &str, ylabel: &str) {
        c.unclip();
        let (ink, dim) = (palette::ink(), palette::grid());
        let r = self.rect;
        // Small panels get small text, a miniature rather than an overflow.
        let s = (r.height() / 26.0).clamp(3.0, 14.0);
        c.polyline(
            &[r.lo, r.top_right(), r.hi, r.bottom_left()],
            1.0,
            dim,
            true,
        );
        let (up, right) = (Point2::direction(0.0, -1.0), Point2::direction(1.0, 0.0));
        // A tick's foot: where the line through its pixel, across the edge, meets the edge.
        let foot_on = |edge: [Point2; 2], along: Point2| {
            move |px: Point2| ((px & (px + along)) ^ (edge[0] & edge[1])).unitized()
        };
        let bottom = foot_on([r.bottom_left(), r.hi], up);
        let left = foot_on([r.lo, r.bottom_left()], right);
        let step_x = nice_step(self.x[1] - self.x[0], 5.0);
        for v in ticks(self.x, self.log_x) {
            let foot = bottom(self.px(Point2::xy(v, self.y[0])));
            c.line(foot, foot + up.gp(4.0), 1.0, dim);
            let text = if self.log_x {
                format!("1E{}", v.log10().round())
            } else {
                label(v, step_x)
            };
            c.text(&text, foot - up.gp(s * 1.3), s * 0.8, ink, Align::Center);
        }
        let step_y = nice_step(self.y[1] - self.y[0], 5.0);
        for v in ticks(self.y, self.log_y) {
            let foot = left(self.px(Point2::xy(self.x[0], v)));
            c.line(foot, foot + right.gp(4.0), 1.0, dim);
            let text = if self.log_y {
                format!("1E{}", v.log10().round())
            } else {
                label(v, step_y)
            };
            let at = foot - right.gp(s * 0.4) - up.gp(s * 0.4);
            c.text(&text, at, s * 0.8, ink, Align::Right);
        }
        let top_middle = (r.lo + r.top_right()).unitized();
        let bottom_middle = (r.bottom_left() + r.hi).unitized();
        if !title.is_empty() {
            c.text(
                title,
                top_middle + up.gp(s * 0.6),
                s * 1.1,
                ink,
                Align::Center,
            );
        }
        if !xlabel.is_empty() {
            c.text(
                xlabel,
                bottom_middle - up.gp(s * 2.7),
                s * 0.9,
                ink,
                Align::Center,
            );
        }
        if !ylabel.is_empty() {
            let above = s * 0.6 + if title.is_empty() { 0.0 } else { s * 1.6 };
            c.text(ylabel, r.lo + up.gp(above), s * 0.9, ink, Align::Left);
        }
    }

    /// A segment in data coordinates.
    pub fn line(&self, c: &mut Canvas, a: impl Pos2, b: impl Pos2, width: f32, l: Light) {
        self.clip(c);
        c.line(self.px(a), self.px(b), width, l);
        c.unclip();
    }

    /// The runs of `pts` between non-finite points, as pixels.
    fn runs(&self, pts: &[impl Pos2]) -> Vec<Vec<Point2>> {
        let mut runs = vec![Vec::new()];
        for p in pts {
            let p = p.point2();
            if finite(p) {
                runs.last_mut().expect("a run").push(self.px(p));
            } else if !runs.last().expect("a run").is_empty() {
                runs.push(Vec::new());
            }
        }
        runs
    }

    /// Segments in data coordinates as one stroke (a grid, spokes, a set of dashes).
    pub fn stroke<P: Pos2>(&self, c: &mut Canvas, segs: &[[P; 2]], width: f32, l: Light) {
        self.clip(c);
        let px: Vec<[Point2; 2]> = segs
            .iter()
            .map(|&[a, b]| [self.px(a), self.px(b)])
            .collect();
        c.stroke(&px, width, l);
        c.unclip();
    }

    /// A polyline through data points, one stroke; non-finite points break it.
    pub fn polyline(&self, c: &mut Canvas, pts: &[impl Pos2], width: f32, l: Light) {
        self.clip(c);
        for run in self.runs(pts) {
            c.polyline(&run, width, l, false);
        }
        c.unclip();
    }

    /// A dashed polyline: dashes of `dash` pixels with equal gaps, measured along it on the
    /// canvas, all one stroke.
    pub fn dashed(&self, c: &mut Canvas, pts: &[impl Pos2], width: f32, dash: f32, l: Light) {
        self.clip(c);
        let mut dashes: Vec<[Point2; 2]> = Vec::new();
        for run in self.runs(pts) {
            let mut along = 0.0f32;
            for w in run.windows(2) {
                let (a, b) = (w[0].unitized(), w[1].unitized());
                let len = (a & b).norm();
                // The point `t` pixels from `a` towards `b`.
                let at = |t: f32| a + (b - a).gp(t / len.max(1e-12));
                let mut s = 0.0;
                while s < len {
                    let phase = (along + s) % (2.0 * dash);
                    let on = phase < dash;
                    let e = (s + if on { dash - phase } else { 2.0 * dash - phase }).min(len);
                    if on {
                        dashes.push([at(s), at(e)]);
                    }
                    s = e;
                }
                along += len;
            }
        }
        c.stroke(&dashes, width, l);
        c.unclip();
    }

    /// A marker of `size` at each data point.
    pub fn scatter(&self, c: &mut Canvas, pts: &[impl Pos2], marker: Marker, size: f32, l: Light) {
        self.clip(c);
        for p in pts {
            let p = p.point2();
            if finite(p) {
                mark(c, self.px(p), marker, size, l);
            }
        }
        c.unclip();
    }

    /// An arrow from `from` to `to` (data coordinates) with a head of `head`.
    pub fn arrow(
        &self,
        c: &mut Canvas,
        from: impl Pos2,
        to: impl Pos2,
        width: f32,
        head: f32,
        l: Light,
    ) {
        self.clip(c);
        arrow(c, self.px(from), self.px(to), width, head, l);
        c.unclip();
    }

    /// The infinite line through `p` along `d`, clipped to the axes.
    pub fn axline(&self, c: &mut Canvas, p: impl Pos2, d: impl Dir2, width: f32, l: Light) {
        let big = 4.0 * ((self.x[1] - self.x[0]).abs() + (self.y[1] - self.y[0]).abs());
        let d = d.dir2();
        let reach = d.gp(big / d.ideal_norm().max(1e-30));
        let p = p.point2().unitized();
        self.line(c, p - reach, p + reach, width, l);
    }

    /// A filled polygon in data coordinates, covering what is below by `opacity`.
    pub fn fill(&self, c: &mut Canvas, poly: &[impl Pos2], l: Light, opacity: f32) {
        self.clip(c);
        let px: Vec<Point2> = poly.iter().map(|&p| self.px(p)).collect();
        c.fill(&px, l, opacity);
        c.unclip();
    }

    /// Text at a data point.
    pub fn text(&self, c: &mut Canvas, at: impl Pos2, s: &str, size: f32, l: Light, align: Align) {
        c.text(s, self.px(at), size, l, align);
    }

    /// The level line `f(p) = level`, from `n` x `n` samples over the axes, one stroke.
    pub fn contour(
        &self,
        c: &mut Canvas,
        f: impl Fn(Point2) -> f32,
        n: usize,
        level: f32,
        width: f32,
        l: Light,
    ) {
        self.clip(c);
        let segs: Vec<[Point2; 2]> = contour::of_fn(f, self.x, self.y, n, level)
            .into_iter()
            .map(|[a, b]| [self.px(a), self.px(b)])
            .collect();
        c.stroke(&segs, width, l);
        c.unclip();
    }

    /// An image over the axes: `f` of the data point under each pixel (`samples` x `samples`
    /// each), `None` showing the canvas through.
    pub fn image(
        &self,
        c: &mut Canvas,
        samples: usize,
        f: impl Fn(Point2) -> Option<Light> + Sync,
    ) {
        self.clip(c);
        let me = *self;
        c.shade(samples, |q| f(me.data(q)));
        c.unclip();
    }

    /// A legend in the top right corner: a short line of each light and its label.
    pub fn legend(&self, c: &mut Canvas, entries: &[(&str, Light)]) {
        let s = (self.rect.height() / 30.0).clamp(3.0, 12.0);
        let w = entries
            .iter()
            .map(|(t, _)| font::width(t, s))
            .fold(0.0, f32::max)
            + s * 3.5;
        let (right, down) = (Point2::direction(1.0, 0.0), Point2::direction(0.0, 1.0));
        let corner = self.rect.top_right() + (down - right).gp(s * 0.6);
        let lo = corner - right.gp(w);
        let hi = corner + down.gp(s * (1.6 * entries.len() as f32 + 0.4));
        let card = Rect { lo, hi };
        c.fill(
            &[card.lo, card.top_right(), card.hi, card.bottom_left()],
            palette::bottom(),
            0.75,
        );
        for (i, (t, l)) in entries.iter().enumerate() {
            let baseline = lo + down.gp(s * (1.2 + 1.6 * i as f32));
            let mid = baseline - down.gp(s * 0.35);
            c.line(mid + right.gp(s * 0.5), mid + right.gp(s * 2.2), 2.0, *l);
            c.text(
                t,
                baseline + right.gp(s * 2.8),
                s,
                palette::ink(),
                Align::Left,
            );
        }
    }
}

/// A marker at pixel `p`: dots and rings glow; square, triangle and star are filled and edged
/// with a glowing outline; the cross is one stroke.
pub fn mark(c: &mut Canvas, p: Point2, marker: Marker, size: f32, l: Light) {
    let r = size * 0.5;
    let at = |x: f32, y: f32| p + gax::pga2d::Point::direction(x, y);
    let shape = |c: &mut Canvas, corners: &[Point2]| {
        c.fill(corners, l.faded(0.6), 1.0);
        c.polyline(corners, 1.0, l, true);
    };
    match marker {
        Marker::Dot => c.disk(p, r * 0.85, l),
        Marker::Ring => c.ring(p, r, (size / 6.0).max(1.0), l),
        Marker::Square => shape(c, &[at(-r, -r), at(r, -r), at(r, r), at(-r, r)]),
        Marker::Triangle => shape(c, &[at(0.0, -r), at(r, r * 0.8), at(-r, r * 0.8)]),
        Marker::Cross => c.stroke(
            &[[at(-r, -r), at(r, r)], [at(-r, r), at(r, -r)]],
            (size / 5.0).max(1.0),
            l,
        ),
        Marker::Star => {
            // Ten corners a tenth of a turn apart, alternately long and short, from the top.
            let p = p.unitized();
            let pts: Vec<Point2> = (0..10)
                .map(|k| {
                    let rr = if k % 2 == 0 { r * 1.2 } else { r * 0.5 };
                    let turn = Motor::rotation(p, core::f32::consts::TAU * k as f32 / 10.0);
                    turn >> (p + gax::pga2d::Point::direction(0.0, -rr))
                })
                .collect();
            shape(c, &pts);
        }
    }
}

/// An arrow between pixels with a head of `head`: the shaft one stroke, the head a
/// filled triangle.
pub fn arrow(c: &mut Canvas, a: Point2, b: Point2, width: f32, head: f32, l: Light) {
    let (pa, pb) = (a.unitized(), b.unitized());
    // The shaft's line: its norm is the length, its normal `(e1, e2)` the head's crossbar.
    let shaft = pa & pb;
    let len = shaft.norm();
    if len.is_nan() || len <= 1e-6 {
        return;
    }
    let h = head.min(len * 0.6);
    let base = pb - (pb - pa).gp(h / len);
    let across = gax::pga2d::Point::direction(shaft.e1(), shaft.e2()).gp(h * 0.45 / len);
    c.line(pa, base, width, l);
    c.fill(&[pb, base + across, base - across], l, 1.0);
}

/// The 2σ ellipse of a planar Gaussian about `centre`, from its variances and principal axes
/// (unit directions, as `eigh` gives them), as `n + 1` points around it: the unit circle turned
/// by rotations and stretched by `2√variance` along each axis (a sum of dyads).
pub fn ellipse(
    centre: gax::pga2d::Point<(), f64>,
    variances: [f64; 2],
    axes: [gax::vga2d::Vector<(), f64>; 2],
    n: usize,
) -> Vec<gax::pga2d::Point<(), f64>> {
    use gax::pga2d::{Line, Point};
    let mut stretch = Point::<(Point,), f64>::zero();
    for (a, v) in axes.iter().zip(variances) {
        let along = Point::direction(a.c[0], a.c[1]);
        stretch += along.gp(2.0 * gax::Real::sqrt(v.max(0.0))) * (Line::from(*a) & Point::slot());
    }
    let (origin, east) = (Point::xy(0.0, 0.0), Point::direction(1.0, 0.0));
    (0..=n)
        .map(|k| {
            let turn = Motor::rotation(origin, core::f64::consts::TAU * k as f64 / n as f64);
            centre + stretch.of(turn >> east)
        })
        .collect()
}
