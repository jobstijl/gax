//! Plot axes on a canvas: a rectangle of pixels showing a range of data, linear or logarithmic
//! on each axis, with a frame, ticks and labels; lines, markers, arrows, filled shapes, level
//! lines, images and legends in data coordinates.

use crate::canvas::{Canvas, Px, Rgb};
use crate::coords::{Dir2, Point2, Pos2, finite};
use crate::font::Align;
use crate::{contour, palette};
use gax::pga2d::{Motor, Point};

/// A marker's shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Marker {
    /// A filled disk.
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

/// Axes: data `x` and `y` ranges shown in the pixel rectangle `rect` (`[x0, y0, x1, y1]`).
#[derive(Clone, Copy, Debug)]
pub struct Axes {
    /// The pixel rectangle.
    pub rect: [f32; 4],
    /// The data range across.
    pub x: [f32; 2],
    /// The data range up.
    pub y: [f32; 2],
    /// Logarithmic across.
    pub log_x: bool,
    /// Logarithmic up.
    pub log_y: bool,
}

/// `[x0, y0, x1, y1]` of panel `i` of `n` side by side on a canvas, with a margin.
pub fn panel(c: &Canvas, i: usize, n: usize) -> [f32; 4] {
    let w = c.width as f32 / n as f32;
    [i as f32 * w, 0.0, (i + 1) as f32 * w, c.height as f32]
}

/// `rect` shrunk by `left`, `top`, `right`, `bottom` pixels (room for ticks and titles).
pub fn inset(rect: [f32; 4], left: f32, top: f32, right: f32, bottom: f32) -> [f32; 4] {
    [
        rect[0] + left,
        rect[1] + top,
        rect[2] - right,
        rect[3] - bottom,
    ]
}

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

impl Axes {
    /// Axes showing `x` by `y` in `rect`.
    pub fn new(rect: [f32; 4], x: [f32; 2], y: [f32; 2]) -> Axes {
        Axes {
            rect,
            x,
            y,
            log_x: false,
            log_y: false,
        }
    }

    /// Axes with equal scales across and up, centred on `centre`, `half_height` units from the
    /// middle to the top (for geometry).
    pub fn equal(rect: [f32; 4], centre: impl Pos2, half_height: f32) -> Axes {
        let centre = centre.point2().to_euclidean();
        let (w, h) = (rect[2] - rect[0], rect[3] - rect[1]);
        let half_width = half_height * w / h;
        Axes::new(
            rect,
            [centre[0] - half_width, centre[0] + half_width],
            [centre[1] - half_height, centre[1] + half_height],
        )
    }

    /// Axes with equal scales, centred on the box around `points` and large enough to show it
    /// whole in `rect`, with `margin` to spare (`1.1` leaves a tenth).
    pub fn fitting<P: Pos2>(
        rect: [f32; 4],
        points: impl IntoIterator<Item = P>,
        margin: f32,
    ) -> Axes {
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
        let aspect = (rect[3] - rect[1]) / (rect[2] - rect[0]);
        let half = 0.5 * span.e01().max(span.e20() * aspect) * margin;
        Axes::equal(rect, (lo + hi).unitized(), half)
    }

    /// The same, logarithmic up.
    pub fn log_y(self) -> Axes {
        Axes {
            log_y: true,
            ..self
        }
    }

    /// The same, logarithmic across.
    pub fn log_x(self) -> Axes {
        Axes {
            log_x: true,
            ..self
        }
    }

    fn t(v: f32, r: [f32; 2], log: bool) -> f32 {
        if log {
            (v.max(1e-30).ln() - r[0].ln()) / (r[1].ln() - r[0].ln())
        } else {
            (v - r[0]) / (r[1] - r[0])
        }
    }

    /// The pixel of data point `p`.
    pub fn px(&self, p: impl Pos2) -> Px {
        let [x, y] = p.point2().to_euclidean();
        let tx = Self::t(x, self.x, self.log_x);
        let ty = Self::t(y, self.y, self.log_y);
        [
            self.rect[0] + tx * (self.rect[2] - self.rect[0]),
            self.rect[3] - ty * (self.rect[3] - self.rect[1]),
        ]
    }

    /// The data point at pixel `q`.
    pub fn data(&self, q: Px) -> Point2 {
        let tx = (q[0] - self.rect[0]) / (self.rect[2] - self.rect[0]);
        let ty = (self.rect[3] - q[1]) / (self.rect[3] - self.rect[1]);
        let v = |t: f32, r: [f32; 2], log: bool| {
            if log {
                (r[0].ln() + t * (r[1].ln() - r[0].ln())).exp()
            } else {
                r[0] + t * (r[1] - r[0])
            }
        };
        Point2::xy(v(tx, self.x, self.log_x), v(ty, self.y, self.log_y))
    }

    /// Pixels per data unit across (linear axes).
    pub fn scale(&self) -> f32 {
        (self.rect[2] - self.rect[0]) / (self.x[1] - self.x[0])
    }

    /// Confine drawing to the axes.
    pub fn clip(&self, c: &mut Canvas) {
        c.clip(self.rect);
    }

    /// A frame with ticks and tick labels, a title above and axis labels.
    pub fn frame(&self, c: &mut Canvas, title: &str, xlabel: &str, ylabel: &str) {
        c.unclip();
        let ink = palette::ink();
        let dim = palette::grid();
        let s = ((self.rect[3] - self.rect[1]) / 26.0).clamp(8.0, 14.0);
        let [x0, y0, x1, y1] = self.rect;
        c.polyline(
            &[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            1.0,
            dim,
            1.0,
            true,
        );
        let ticks = |r: [f32; 2], log: bool| -> Vec<f32> {
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
        };
        let step_x = nice_step(self.x[1] - self.x[0], 5.0);
        for v in ticks(self.x, self.log_x) {
            let [px, _] = self.px(Point2::xy(v, self.y[0]));
            c.line([px, y1], [px, y1 - 4.0], 1.0, dim, 1.0);
            let text = if self.log_x {
                format!("1E{}", v.log10().round())
            } else {
                label(v, step_x)
            };
            c.text(&text, px, y1 + s * 1.3, s * 0.8, ink, Align::Center);
        }
        let step_y = nice_step(self.y[1] - self.y[0], 5.0);
        for v in ticks(self.y, self.log_y) {
            let [_, py] = self.px(Point2::xy(self.x[0], v));
            c.line([x0, py], [x0 + 4.0, py], 1.0, dim, 1.0);
            let text = if self.log_y {
                format!("1E{}", v.log10().round())
            } else {
                label(v, step_y)
            };
            c.text(
                &text,
                x0 - s * 0.4,
                py + s * 0.4,
                s * 0.8,
                ink,
                Align::Right,
            );
        }
        if !title.is_empty() {
            c.text(
                title,
                (x0 + x1) * 0.5,
                y0 - s * 0.6,
                s * 1.1,
                ink,
                Align::Center,
            );
        }
        if !xlabel.is_empty() {
            c.text(
                xlabel,
                (x0 + x1) * 0.5,
                y1 + s * 2.7,
                s * 0.9,
                ink,
                Align::Center,
            );
        }
        if !ylabel.is_empty() {
            c.text(
                ylabel,
                x0,
                y0 - s * 0.6 - if title.is_empty() { 0.0 } else { s * 1.6 },
                s * 0.9,
                ink,
                Align::Left,
            );
        }
    }

    /// A segment in data coordinates.
    pub fn line(
        &self,
        c: &mut Canvas,
        a: impl Pos2,
        b: impl Pos2,
        width: f32,
        color: Rgb,
        alpha: f32,
    ) {
        self.clip(c);
        c.line(self.px(a), self.px(b), width, color, alpha);
        c.unclip();
    }

    /// A polyline through data points; non-finite points break it.
    pub fn polyline(&self, c: &mut Canvas, pts: &[impl Pos2], width: f32, color: Rgb, alpha: f32) {
        let pts: Vec<Point2> = pts.iter().map(|p| p.point2()).collect();
        self.clip(c);
        for w in pts.windows(2) {
            if finite(w[0]) && finite(w[1]) {
                c.line(self.px(w[0]), self.px(w[1]), width, color, alpha);
            }
        }
        c.unclip();
    }

    /// A dashed polyline: dashes of `dash` pixels with equal gaps.
    pub fn dashed(
        &self,
        c: &mut Canvas,
        pts: &[impl Pos2],
        width: f32,
        dash: f32,
        color: Rgb,
        alpha: f32,
    ) {
        let pts: Vec<Point2> = pts.iter().map(|p| p.point2()).collect();
        self.clip(c);
        let mut along = 0.0f32;
        for w in pts.windows(2) {
            let (a, b) = (self.px(w[0]), self.px(w[1]));
            let (pa, pb) = (Point::xy(a[0], a[1]), Point::xy(b[0], b[1]));
            let len = (pa & pb).norm();
            let mut s = 0.0;
            while s < len {
                let phase = (along + s) % (2.0 * dash);
                let run = if phase < dash {
                    dash - phase
                } else {
                    2.0 * dash - phase
                };
                let e = (s + run).min(len);
                if phase < dash {
                    let p = |t: f32| (pa + (pb - pa).gp(t / len)).to_euclidean();
                    c.line(p(s), p(e), width, color, alpha);
                }
                s = e;
            }
            along += len;
        }
        c.unclip();
    }

    /// A marker of `size` pixels at each data point.
    pub fn scatter(
        &self,
        c: &mut Canvas,
        pts: &[impl Pos2],
        marker: Marker,
        size: f32,
        color: Rgb,
        alpha: f32,
    ) {
        let pts: Vec<Point2> = pts.iter().map(|p| p.point2()).collect();
        self.clip(c);
        for p in pts {
            if finite(p) {
                mark(c, self.px(p), marker, size, color, alpha);
            }
        }
        c.unclip();
    }

    /// An arrow from `from` to `to` (data coordinates) with a head of `head` pixels.
    pub fn arrow(
        &self,
        c: &mut Canvas,
        from: impl Pos2,
        to: impl Pos2,
        width: f32,
        head: f32,
        color: Rgb,
    ) {
        self.clip(c);
        arrow(c, self.px(from), self.px(to), width, head, color, 1.0);
        c.unclip();
    }

    /// The infinite line through `p` along `d`, clipped to the axes.
    pub fn axline(
        &self,
        c: &mut Canvas,
        p: impl Pos2,
        d: impl Dir2,
        width: f32,
        color: Rgb,
        alpha: f32,
    ) {
        let big = 4.0 * ((self.x[1] - self.x[0]).abs() + (self.y[1] - self.y[0]).abs());
        let d = d.dir2();
        let reach = d.gp(big / d.ideal_norm().max(1e-30));
        let p = p.point2().unitized();
        self.line(c, p - reach, p + reach, width, color, alpha);
    }

    /// A filled polygon in data coordinates.
    pub fn fill(&self, c: &mut Canvas, poly: &[impl Pos2], color: Rgb, alpha: f32) {
        self.clip(c);
        let px: Vec<Px> = poly.iter().map(|&p| self.px(p)).collect();
        c.fill(&px, color, alpha);
        c.unclip();
    }

    /// Text at a data point.
    pub fn text(
        &self,
        c: &mut Canvas,
        at: impl Pos2,
        s: &str,
        size: f32,
        color: Rgb,
        align: Align,
    ) {
        let [x, y] = self.px(at);
        c.text(s, x, y, size, color, align);
    }

    /// The level line `f(p) = level`, from `n` x `n` samples over the axes.
    pub fn contour(
        &self,
        c: &mut Canvas,
        f: impl Fn(Point2) -> f32,
        n: usize,
        level: f32,
        width: f32,
        color: Rgb,
    ) {
        self.clip(c);
        for [a, b] in contour::of_fn(f, self.x, self.y, n, level) {
            c.line(self.px(a), self.px(b), width, color, 1.0);
        }
        c.unclip();
    }

    /// An image over the axes: `f` of the data point under each pixel (`samples` x `samples`
    /// each), `None` showing the canvas through.
    pub fn image(&self, c: &mut Canvas, samples: usize, f: impl Fn(Point2) -> Option<Rgb> + Sync) {
        self.clip(c);
        let me = *self;
        c.shade(samples, |x, y| f(me.data([x, y])));
        c.unclip();
    }

    /// A legend in the top right corner: a short line of each colour and its label.
    pub fn legend(&self, c: &mut Canvas, entries: &[(&str, Rgb)]) {
        let s = ((self.rect[3] - self.rect[1]) / 30.0).clamp(7.0, 12.0);
        let w = entries
            .iter()
            .map(|(t, _)| crate::font::width(t, s))
            .fold(0.0, f32::max)
            + s * 3.5;
        let (x1, y0) = (self.rect[2] - s * 0.6, self.rect[1] + s * 0.6);
        let x0 = x1 - w;
        c.fill(
            &[
                [x0, y0],
                [x1, y0],
                [x1, y0 + s * 1.6 * entries.len() as f32 + s * 0.4],
                [x0, y0 + s * 1.6 * entries.len() as f32 + s * 0.4],
            ],
            palette::bottom(),
            0.75,
        );
        for (i, (t, col)) in entries.iter().enumerate() {
            let y = y0 + s * (1.2 + 1.6 * i as f32);
            c.line(
                [x0 + s * 0.5, y - s * 0.35],
                [x0 + s * 2.2, y - s * 0.35],
                2.0,
                *col,
                1.0,
            );
            c.text(t, x0 + s * 2.8, y, s, palette::ink(), Align::Left);
        }
    }
}

/// A marker at pixel `p`.
pub fn mark(c: &mut Canvas, p: Px, marker: Marker, size: f32, color: Rgb, alpha: f32) {
    let r = size * 0.5;
    match marker {
        Marker::Dot => c.disk(p, r, color, alpha),
        Marker::Ring => c.ring(p, r, (size / 6.0).max(1.0), color, alpha),
        Marker::Square => c.fill(
            &[
                [p[0] - r, p[1] - r],
                [p[0] + r, p[1] - r],
                [p[0] + r, p[1] + r],
                [p[0] - r, p[1] + r],
            ],
            color,
            alpha,
        ),
        Marker::Triangle => c.fill(
            &[
                [p[0], p[1] - r],
                [p[0] + r, p[1] + r * 0.8],
                [p[0] - r, p[1] + r * 0.8],
            ],
            color,
            alpha,
        ),
        Marker::Cross => {
            c.line(
                [p[0] - r, p[1] - r],
                [p[0] + r, p[1] + r],
                (size / 5.0).max(1.0),
                color,
                alpha,
            );
            c.line(
                [p[0] - r, p[1] + r],
                [p[0] + r, p[1] - r],
                (size / 5.0).max(1.0),
                color,
                alpha,
            );
        }
        Marker::Star => {
            // Ten corners a tenth of a turn apart, alternately long and short, from the top.
            let centre = Point::xy(p[0], p[1]);
            let pts: Vec<Px> = (0..10)
                .map(|k| {
                    let rr = if k % 2 == 0 { r * 1.2 } else { r * 0.5 };
                    let turn = Motor::rotation(centre, core::f32::consts::TAU * k as f32 / 10.0);
                    (turn >> Point::xy(p[0], p[1] - rr)).to_euclidean()
                })
                .collect();
            c.fill(&pts, color, alpha);
        }
    }
}

/// An arrow between pixels with a head of `head` pixels.
pub fn arrow(c: &mut Canvas, a: Px, b: Px, width: f32, head: f32, color: Rgb, alpha: f32) {
    let (pa, pb) = (Point::xy(a[0], a[1]), Point::xy(b[0], b[1]));
    // The shaft's line: its norm is the length, its normal `(e1, e2)` the head's crossbar.
    let shaft = pa & pb;
    let len = shaft.norm();
    if len.is_nan() || len <= 1e-6 {
        return;
    }
    let h = head.min(len * 0.6);
    let base = pb - (pb - pa).gp(h / len);
    let across = Point::direction(shaft.e1(), shaft.e2()).gp(h * 0.45 / len);
    c.line(a, base.to_euclidean(), width, color, alpha);
    c.fill(
        &[
            b,
            (base + across).to_euclidean(),
            (base - across).to_euclidean(),
        ],
        color,
        alpha,
    );
}

/// The 2σ ellipse of a planar Gaussian about `centre`, from its variances and principal axes
/// (unit directions, as `eigh` gives them), as `n + 1` points around it: the unit circle turned
/// by rotations and stretched by `2√variance` along each axis (a sum of dyads).
pub fn ellipse(
    centre: Point<(), f64>,
    variances: [f64; 2],
    axes: [gax::vga2d::Vector<(), f64>; 2],
    n: usize,
) -> Vec<Point<(), f64>> {
    use gax::pga2d::Line;
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
