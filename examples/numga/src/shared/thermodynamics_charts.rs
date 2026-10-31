//! The drawing shared by the thermodynamics examples (numga's `examples/thermodynamics/rendering.py`):
//! diagrams of one quantity against another, with their families of level lines, tangents,
//! arrows, shaded areas and names, and the apparatus, a cylinder of gas under a piston, in 3D.
//!
//! A diagram's chart is a plane whose points carry its two quantities: a chart point is the PGA2D
//! point `(across, up)`, and the diagram's [`Axes`] map it onto the canvas, each quantity at its
//! own scale. Curves are polylines through chart points, areas polygons, a tangent the join of a
//! point with a direction, sampled where it meets the lines of fixed value across, and a foot the
//! meet of the line through a point along the quantity up with the axis. The level lines of a
//! function sampled on a grid are found by marching squares (`contour`) and chained into
//! polylines. Names are written in the canvas's stroke font over a dark halo; a floating name
//! takes, among candidate places along its lines, the one where it covers the least light drawn
//! before it.
//!
//! The apparatus is a few boxes in a [`Scene3`], lit, the gas a translucent box coloured along
//! the thermal ramp from icy blue to red heat.

#![allow(dead_code)]

use std::collections::HashMap;

use gax::Extensor;
use gax::pga2d::Line;
use gax_colour::{Light, Srgb, light};
use gax_numga_examples::canvas::{Canvas, Rect};
use gax_numga_examples::font::{self, Align};
use gax_numga_examples::plot::{self, Axes};
use gax_numga_examples::points::{ORIGIN2, Point2, Point3, box_map};
use gax_numga_examples::scene3::{Scene3, panel3};
use gax_numga_examples::view::{Camera, Lens};
use gax_numga_examples::{colormap, contour, palette};

// --- colours --------------------------------------------------------------------------

/// Text, the axes and the lines a state is drawn along.
pub const INK: Light = palette::INK;
/// The axes and ticks.
pub const AXIS: Light = light(0.75, 0.85, 1.0, 0.75);
/// numga's blue, its orange, its green.
pub const BLUE: Light = palette::SKY;
pub const ORANGE: Light = palette::ORANGE;
pub const GREEN: Light = palette::GREEN;
/// Grey: the apparatus's frame, quiet lines.
pub const GREY: Light = light(0.55, 0.62, 0.78, 0.75);
/// Phase boundaries, the lines the regions of a diagram part along: bright, as numga's are black.
pub const BOUNDARY: Light = light(0.85, 0.9, 1.0, 1.7);

/// The families of lines a diagram is read against, one colour a family (numga's colours, as
/// lights): isotherms red, isobars purple, isochores green, adiabats teal, levels of humidity
/// blue, of enthalpy brown, other levels grey-blue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Temperature,
    Pressure,
    Volume,
    Entropy,
    Humidity,
    Enthalpy,
    Level,
}

impl Family {
    /// The family's light.
    pub fn light(self) -> Light {
        let (code, intensity) = match self {
            Family::Temperature => (0xf0604c, 1.1),
            Family::Pressure => (0xa47ae0, 1.2),
            Family::Volume => (0x5ec266, 1.0),
            Family::Entropy => (0x3fc4b8, 1.0),
            Family::Humidity => (0x5a9af0, 1.2),
            Family::Enthalpy => (0xd09040, 1.0),
            Family::Level => (0x9ab4c8, 0.8),
        };
        gax_colour::hex(code, intensity)
    }
}

/// The thermal ramp, from icy blue for the coldest shown, through a pale middle, to red heat for
/// the hottest, a share of the way along it.
pub fn thermal(fraction: f64) -> Light {
    let stops = [
        Srgb::rgb(0.122, 0.435, 0.773),
        Srgb::rgb(0.561, 0.816, 0.969),
        Srgb::rgb(0.949, 0.933, 0.910),
        Srgb::rgb(0.965, 0.627, 0.302),
        Srgb::rgb(0.816, 0.122, 0.090),
    ];
    colormap::stops(fraction.clamp(0.0, 1.0) as f32, &stops)
}

/// The share along the thermal ramp of a temperature a gap above a reference, positive or
/// negative, which the widest gap shown puts at the ramp's ends: half by the gap itself, half by
/// its logarithm over as many decades as shown, so that a gap closing exponentially keeps moving
/// along the ramp to the end without its start crowding together.
pub fn relaxing(gap: f64, widest: f64, decades: f64) -> f64 {
    let share = gap.abs() / widest;
    let depth = 0.5 * share.clamp(0.0, 1.0)
        + 0.5 * ((share + 1e-300).log10() / decades + 1.0).clamp(0.0, 1.0);
    0.5 + 0.5 * gap.signum() * depth
}

// --- widths and sizes, in numga's pixels (a chart 300 rows tall) ----------------------

/// The widths of the strokes: the lines a diagram is read against, a curve, and the path the
/// diagram follows.
pub const THIN: f32 = 1.0;
pub const MEDIUM: f32 = 1.5;
pub const BOLD: f32 = 2.4;
/// The size of the names and labels, and of the numbers at the ticks.
pub const TEXT: f32 = 9.5;
pub const TICK: f32 = 8.0;
/// Room around a diagram's plot: for the name up on the left, the one across below, and a little
/// on the other two sides.
const MARGINS: [f32; 4] = [30.0, 16.0, 12.0, 24.0];
/// The radius of a dot marking a state, and of the dark ring around it.
const DOT: f32 = 4.0;
const RING: f32 = 5.5;

// --- diagrams -------------------------------------------------------------------------

/// The chart point at values of its two quantities: the point `(across, up)` of the plane.
pub fn at(across: f64, up: f64) -> Point2 {
    Point2::xy(across as f32, up as f32)
}

/// The direction of a chart's plane along changes of its two quantities.
pub fn toward(across: f64, up: f64) -> Point2 {
    Point2::direction(across as f32, up as f32)
}

/// Where a name is placed: its corner or middle at the point, as numga's Pillow anchors (`"lb"`
/// puts the text's left bottom there).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Anchor(pub Align, pub Vertical);

/// A name's vertical placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vertical {
    Top,
    Middle,
    Bottom,
}

pub const LB: Anchor = Anchor(Align::Left, Vertical::Bottom);
pub const LT: Anchor = Anchor(Align::Left, Vertical::Top);
pub const LM: Anchor = Anchor(Align::Left, Vertical::Middle);
pub const RB: Anchor = Anchor(Align::Right, Vertical::Bottom);
pub const RT: Anchor = Anchor(Align::Right, Vertical::Top);
pub const RM: Anchor = Anchor(Align::Right, Vertical::Middle);
pub const MT: Anchor = Anchor(Align::Center, Vertical::Top);
pub const MM: Anchor = Anchor(Align::Center, Vertical::Middle);

/// A diagram of one quantity across against another up: the axes it is drawn in, the values at
/// which its two axes cross, and the names of the two quantities (a subscript after `_`).
#[derive(Clone, Copy, Debug)]
pub struct Diagram {
    pub ax: Axes,
    pub crossing: [f64; 2],
    pub names: [&'static str; 2],
    /// Canvas units per numga pixel: a chart 300 of them tall.
    pub unit: f32,
}

impl Diagram {
    /// A diagram in the panel `rect`, as tall as it is wide at most and centred in it, showing the
    /// ranges `across` and `up`.
    pub fn new(
        rect: Rect,
        across: [f64; 2],
        up: [f64; 2],
        crossing: [f64; 2],
        names: [&'static str; 2],
    ) -> Diagram {
        let height = (rect.height() * 0.86).min(rect.width() * 0.97);
        let middle = rect.centre();
        let picture = Rect::around(middle, Point2::direction(rect.width() * 0.5, height * 0.5));
        let unit = height / 300.0;
        let [left, top, right, bottom] = MARGINS.map(|m| m * unit);
        let plot = picture.inset(left, top, right, bottom);
        let ax = Axes::new(
            plot,
            [across[0] as f32, across[1] as f32],
            [up[0] as f32, up[1] as f32],
        );
        Diagram {
            ax,
            crossing,
            names,
            unit,
        }
    }

    /// The canvas point of a chart point.
    pub fn px(&self, p: Point2) -> Point2 {
        self.ax.px(p)
    }

    /// The two axes, crossing where the diagram says, each ending in an arrowhead, and the
    /// quantities' names at their ends.
    pub fn axes(&self, c: &mut Canvas) {
        let [x0, y0] = self.crossing;
        let corner = self.px(at(x0, y0));
        let right = self.px(at(f64::from(self.ax.x[1]), y0));
        let top = self.px(at(x0, f64::from(self.ax.y[1])));
        let u = self.unit;
        for end in [right, top] {
            plot::arrow(c, corner, end, u, 7.0 * u, AXIS);
        }
        let [across, up] = self.names;
        self.write(c, right, across, INK, TEXT, RT, [0.0, -5.0]);
        self.write(c, top, up, INK, TEXT, LT, [8.0, 4.0]);
    }

    /// Short marks on the axes at values to read off, and their numbers beside them.
    pub fn ticks(&self, c: &mut Canvas, across: &[f64], up: &[f64]) {
        let [x0, y0] = self.crossing;
        let u = self.unit;
        let (down, left) = (
            Point2::direction(0.0, 4.0 * u),
            Point2::direction(-4.0 * u, 0.0),
        );
        let mut marks = Vec::new();
        for &v in across {
            let p = self.px(at(v, y0));
            marks.push([p, p + down]);
            self.write(c, p, &number(v), INK, TICK, MT, [0.0, -6.0]);
        }
        for &v in up {
            let p = self.px(at(x0, v));
            marks.push([p, p + left]);
            self.write(c, p, &number(v), INK, TICK, RM, [-6.0, 0.0]);
        }
        c.stroke(&marks, u, AXIS);
    }

    /// A polyline through chart points, cut to the diagram's ranges.
    pub fn polyline(&self, c: &mut Canvas, pts: &[Point2], width: f32, l: Light) {
        self.ax.polyline(c, pts, width * self.unit, l);
    }

    /// Polylines through chart points, in one light.
    pub fn lines(&self, c: &mut Canvas, lines: &[Vec<Point2>], width: f32, l: Light) {
        for line in lines {
            self.polyline(c, line, width, l);
        }
    }

    /// A family of lines, thin, in the family's light.
    pub fn family(&self, c: &mut Canvas, lines: &[Vec<Point2>], family: Family) {
        self.lines(c, lines, THIN, family.light());
    }

    /// A dashed polyline, dashes and gaps `dash` numga pixels long.
    pub fn dashed(&self, c: &mut Canvas, pts: &[Point2], width: f32, dash: f32, l: Light) {
        let u = self.unit;
        self.ax.dashed(c, pts, width * u, dash * u, l);
    }

    /// A note under the diagram, centred, in the ink's fainter light.
    pub fn note(&self, c: &mut Canvas, text: &str) {
        let r = self.ax.rect;
        let at = r.bottom_middle() + Point2::direction(0.0, 36.0 * self.unit);
        // As large as the names, or smaller to fit the picture's width.
        let room = r.width();
        let size = (TEXT * self.unit).min(room / font::width(text, 1.0).max(1e-6));
        c.text(text, at, size, INK.faded(0.6), Align::Center);
    }

    /// The area inside an outline through chart points, closed from its last corner back to its
    /// first, covering what is below by `opacity`.
    pub fn area(&self, c: &mut Canvas, corners: &[Point2], l: Light, opacity: f32) {
        self.ax.fill(c, corners, l, opacity);
    }

    /// Dots at chart points, each in a dark ring.
    pub fn dots(&self, c: &mut Canvas, pts: &[Point2], l: Light) {
        let u = self.unit;
        for p in pts {
            if p.is_finite() {
                let q = self.px(*p);
                c.disk(q, RING * u, palette::BOTTOM);
                c.disk(q, DOT * u, l);
            }
        }
    }

    /// An arrow `length` numga pixels long from a chart point toward another.
    pub fn arrow(
        &self,
        c: &mut Canvas,
        from: Point2,
        aim: Point2,
        length: f32,
        width: f32,
        l: Light,
    ) {
        let (tail, head) = (self.px(from), self.px(aim));
        let along = head - tail;
        let size = along.ideal_norm();
        if size > 1e-6 {
            let tip = tail + along.gp(length * self.unit / size);
            self.ax.clip(c);
            plot::arrow(c, tail, tip, width * self.unit, 7.0 * self.unit, l);
            c.unclip();
        }
    }

    /// A name at a chart point, in its light and size, placed by its anchor and shifted a few
    /// numga pixels right and up.
    #[allow(clippy::too_many_arguments)]
    pub fn label(
        &self,
        c: &mut Canvas,
        place: Point2,
        text: &str,
        l: Light,
        size: f32,
        anchor: Anchor,
        shift: [f32; 2],
    ) {
        self.write(c, self.px(place), text, l, size, anchor, shift);
    }

    /// A name at a canvas point (see [`Diagram::label`]): a subscript after an underscore at the
    /// size of the ticks' numbers and lower, the two placed together by the anchor; over a dark
    /// halo that keeps it readable over the lines under it.
    #[allow(clippy::too_many_arguments)]
    fn write(
        &self,
        c: &mut Canvas,
        place: Point2,
        text: &str,
        l: Light,
        size: f32,
        anchor: Anchor,
        shift: [f32; 2],
    ) {
        for (segs, width) in self.text_segments(place, text, size, anchor, shift) {
            c.outline(&segs, width + 2.5 * self.unit, palette::BOTTOM, 0.85);
            c.stroke(&segs, width, l);
        }
    }

    /// The strokes of a name placed as [`Diagram::write`] places it, with their width: the base,
    /// and the subscript if any.
    fn text_segments(
        &self,
        place: Point2,
        text: &str,
        size: f32,
        anchor: Anchor,
        shift: [f32; 2],
    ) -> Vec<(Vec<[Point2; 2]>, f32)> {
        let u = self.unit;
        let (base, subscript) = text.split_once('_').unwrap_or((text, ""));
        let (big, small) = (size * u, TICK * 0.85 * u);
        let wide = font::width(base, big)
            + if subscript.is_empty() {
                0.0
            } else {
                font::width(subscript, small) + 0.25 * big
            };
        let Anchor(align, vertical) = anchor;
        let left = match align {
            Align::Left => 0.0,
            Align::Center => -0.5 * wide,
            Align::Right => -wide,
        };
        let below = match vertical {
            Vertical::Top => big,
            Vertical::Middle => 0.5 * big,
            Vertical::Bottom => 0.0,
        };
        // The baseline's left end: canvas y runs down, numga's shift up.
        let start = place + Point2::direction(left + shift[0] * u, below - shift[1] * u);
        let placed = |s: &str, at: Point2, size: f32| {
            let k = size / 6.0;
            let map = box_map(
                [ORIGIN2, Point2::xy(1.0, 1.0)],
                [at, at + Point2::direction(k, -k)],
            );
            let segs: Vec<[Point2; 2]> = font::segments(s, Align::Left)
                .into_iter()
                .map(|[a, b]| [map.of(a), map.of(b)])
                .collect();
            (segs, (size / 9.0).max(0.8))
        };
        let mut out = vec![placed(base, start, big)];
        if !subscript.is_empty() {
            let after = start + Point2::direction(font::width(base, big) + 0.25 * big, 0.3 * big);
            out.push(placed(subscript, after, small));
        }
        out
    }

    /// A floating name, placed and written at once (see [`Diagram::place`]).
    pub fn float(
        &self,
        c: &mut Canvas,
        candidates: &[Point2],
        text: &str,
        l: Light,
        anchor: Anchor,
    ) {
        let name = Floating {
            candidates: candidates.to_vec(),
            text: text.to_string(),
            light: l,
            anchor,
        };
        let placed = self.place(c, &[name]);
        self.write_names(c, &placed);
    }

    /// Places for floating names: for each, among its candidate chart points, the one where the
    /// text, placed by its anchor, covers the least light drawn so far, inside the plot and
    /// clear of the names placed before it. Placed before what moves is drawn, they hold still
    /// across the frames of an animation; written after it ([`Diagram::write_names`]), they stay
    /// readable over it.
    pub fn place(&self, c: &Canvas, names: &[Floating]) -> Vec<Placed> {
        let rect = self.ax.rect;
        let [columns, rows] = c.pixels();
        let scale = columns as f32 / c.rect().width();
        let pad = 2.0 * self.unit;
        let [left, top] = rect.lo.to_euclidean();
        let [right, bottom] = rect.hi.to_euclidean();
        let mut taken: Vec<[[f32; 2]; 2]> = Vec::new();
        let mut placed = Vec::new();
        for name in names {
            let mut best: Option<(f32, Point2, [[f32; 2]; 2])> = None;
            for &p in name.candidates.iter().filter(|p| p.is_finite()) {
                let place = self.px(p);
                let segs: Vec<Point2> = self
                    .text_segments(place, &name.text, TEXT, name.anchor, [0.0, 0.0])
                    .into_iter()
                    .flat_map(|(s, _)| s.into_iter().flatten())
                    .collect();
                let [lo, hi] = gax_numga_examples::points::bounds(segs);
                let (lo, hi) = ([lo[0] - pad, lo[1] - pad], [hi[0] + pad, hi[1] + pad]);
                let inside = lo[0] >= left && hi[0] <= right && lo[1] >= top && hi[1] <= bottom;
                let clear = taken
                    .iter()
                    .all(|[a, b]| hi[0] < a[0] || lo[0] > b[0] || hi[1] < a[1] || lo[1] > b[1]);
                if !(inside && clear) {
                    continue;
                }
                let index = |v: f32, n: usize| ((v * scale).max(0.0) as usize).min(n);
                let (x0, x1) = (index(lo[0], columns), index(hi[0], columns));
                let (y0, y1) = (index(lo[1], rows), index(hi[1], rows));
                let mut ink = 0.0;
                for y in y0..y1 {
                    for x in x0..x1 {
                        ink += c.get(x, y).luma();
                    }
                }
                if best.is_none_or(|(least, ..)| ink < least) {
                    best = Some((ink, place, [lo, hi]));
                }
            }
            if let Some((_, place, bounds)) = best {
                taken.push(bounds);
                placed.push(Placed {
                    place,
                    text: name.text.clone(),
                    light: name.light,
                    anchor: name.anchor,
                });
            }
        }
        placed
    }

    /// Write placed names.
    pub fn write_names(&self, c: &mut Canvas, placed: &[Placed]) {
        for p in placed {
            self.write(c, p.place, &p.text, p.light, TEXT, p.anchor, [0.0, 0.0]);
        }
    }

    /// A tangent line through a chart point along a direction, drawn between the values
    /// `reach` across: the join of the point and the direction, met with the lines of those
    /// values.
    pub fn tangent(
        &self,
        c: &mut Canvas,
        point: Point2,
        direction: Point2,
        reach: [f64; 2],
        width: f32,
        l: Light,
    ) {
        let ends = tangent_points(point, direction, &reach);
        self.polyline(c, &ends, width, l);
    }
}

/// A name that finds its own place among candidate chart points (see [`Diagram::place`]).
#[derive(Clone, Debug)]
pub struct Floating {
    pub candidates: Vec<Point2>,
    pub text: String,
    pub light: Light,
    pub anchor: Anchor,
}

impl Floating {
    /// A name with candidate places every so many samples along lines.
    pub fn along(lines: &[Vec<Point2>], every: usize, text: &str, light: Light) -> Floating {
        Floating {
            candidates: along(lines, every),
            text: text.to_string(),
            light,
            anchor: LB,
        }
    }
}

/// A floating name where it was placed, a canvas point.
#[derive(Clone, Debug)]
pub struct Placed {
    pub place: Point2,
    pub text: String,
    pub light: Light,
    pub anchor: Anchor,
}

/// A tick's number: as few digits as it needs.
fn number(v: f64) -> String {
    let s = format!("{v}");
    if s.len() > 6 { format!("{v:.3}") } else { s }
}

/// The line of a chart's plane where the quantity across has the value `v`: `x - v = 0`.
pub fn across_line(v: f64) -> Line<(), f32> {
    Line::new(1.0, 0.0, -v as f32)
}

/// The line where the quantity up has the value `v`: `y - v = 0`.
pub fn up_line(v: f64) -> Line<(), f32> {
    Line::new(0.0, 1.0, -v as f32)
}

/// The line through a chart point along a direction (their join), sampled where it meets the
/// lines of the values across.
pub fn tangent_points(point: Point2, direction: Point2, across: &[f64]) -> Vec<Point2> {
    let line = point & direction;
    across
        .iter()
        .map(|&v| (line ^ across_line(v)).position())
        .collect()
}

/// The foot of a chart point on the line where the quantity up has the value `v`: the line
/// through the point along the quantity up, met with it.
pub fn foot(point: Point2, v: f64) -> Point2 {
    ((point & Point2::direction(0.0, 1.0)) ^ up_line(v)).position()
}

/// The point where the line through a chart point along the quantity across meets the line
/// where the quantity across has the value `v`.
pub fn level_with(point: Point2, v: f64) -> Point2 {
    ((point & Point2::direction(1.0, 0.0)) ^ across_line(v)).position()
}

/// Candidate places for a floating name along polylines of chart points: every so many of their
/// samples.
pub fn along(lines: &[Vec<Point2>], every: usize) -> Vec<Point2> {
    lines
        .iter()
        .flat_map(|line| line.iter().skip(every / 2).step_by(every.max(1)).copied())
        .collect()
}

// --- level lines ----------------------------------------------------------------------

/// Where a function sampled on a grid, `values[row * across.len() + column]` at the grid's
/// coordinates `across[column]` and `up[row]` (any spacing), takes a level: polylines of chart
/// points, the marching-squares segments chained where they share an end. Non-finite samples
/// break the lines.
pub fn level_lines(across: &[f64], up: &[f64], values: &[f64], level: f64) -> Vec<Vec<Point2>> {
    let grid: Vec<f32> = values.iter().map(|&v| (v - level) as f32).collect();
    let segs = contour::segments(&grid, across.len(), up.len(), 0.0);
    // A grid coordinate to its value: between the samples around it, in proportion.
    let read = |samples: &[f64], g: f32| {
        let i = (g.floor().max(0.0) as usize).min(samples.len() - 2);
        let f = f64::from(g) - i as f64;
        samples[i] + (samples[i + 1] - samples[i]) * f
    };
    let chart = |p: Point2| {
        let [g, h] = p.to_euclidean();
        at(read(across, g), read(up, h))
    };
    chain(&segs)
        .into_iter()
        .map(|line| line.into_iter().map(chart).collect())
        .collect()
}

/// Segments chained into polylines where their ends meet: within a thousandth of a cell (two
/// cells compute the crossing on the edge they share from opposite ends, which rounds
/// differently), found among the ends hashed into the neighbouring thousandths.
fn chain(segs: &[[Point2; 2]]) -> Vec<Vec<Point2>> {
    const TOLERANCE: f32 = 1e-3;
    let key = |p: Point2| {
        let [x, y] = p.to_euclidean();
        (
            (x / TOLERANCE).round() as i64,
            (y / TOLERANCE).round() as i64,
        )
    };
    let mut ends: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, [a, b]) in segs.iter().enumerate() {
        ends.entry(key(*a)).or_default().push(i);
        ends.entry(key(*b)).or_default().push(i);
    }
    let near = |p: Point2, q: Point2| (p & q).norm() <= TOLERANCE;
    let mut used = vec![false; segs.len()];
    let mut lines = Vec::new();
    for start in 0..segs.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let mut line = vec![segs[start][0], segs[start][1]];
        // Grow from the end, then from the start.
        for forward in [true, false] {
            loop {
                let end = if forward {
                    line[line.len() - 1]
                } else {
                    line[0]
                };
                let (kx, ky) = key(end);
                let next = (-1..=1)
                    .flat_map(|dx| (-1..=1).map(move |dy| (kx + dx, ky + dy)))
                    .filter_map(|k| ends.get(&k))
                    .flatten()
                    .copied()
                    .find(|&j| !used[j] && (near(segs[j][0], end) || near(segs[j][1], end)));
                let Some(j) = next else { break };
                used[j] = true;
                let [a, b] = segs[j];
                let other = if near(a, end) { b } else { a };
                if forward {
                    line.push(other);
                } else {
                    line.insert(0, other);
                }
            }
        }
        lines.push(line);
    }
    lines
}

// --- the charts of a gas --------------------------------------------------------------

/// A family of a gas's lines on its two charts: entropy across and temperature up, and volume
/// across and pressure up.
#[derive(Clone, Debug, Default)]
pub struct GasFamily {
    pub entropy_temperature: Vec<Vec<Point2>>,
    pub volume_pressure: Vec<Vec<Point2>>,
}

/// A gas's lines of equal volume, pressure, temperature and entropy.
#[derive(Clone, Debug, Default)]
pub struct GasLines {
    pub isochores: GasFamily,
    pub isobars: GasFamily,
    pub isotherms: GasFamily,
    pub adiabats: GasFamily,
}

/// A gas's state of entropy and volume on its two charts, its temperature and pressure from the
/// gas's functions `gas(entropy, volume) = [temperature, pressure]`.
pub fn gas_points(gas: &impl Fn(f64, f64) -> [f64; 2], entropy: f64, volume: f64) -> [Point2; 2] {
    let [temperature, pressure] = gas(entropy, volume);
    [at(entropy, temperature), at(volume, pressure)]
}

/// A gas's lines at the levels given (`[isochores, isobars, isotherms, adiabats]`): lines of
/// equal volume and of equal entropy run straight across the plane of states, sampled at the
/// entropies and volumes; lines of equal pressure and of equal temperature are level lines of
/// the gas's functions over it.
pub fn gas_lines(
    gas: &impl Fn(f64, f64) -> [f64; 2],
    entropies: &[f64],
    volumes: &[f64],
    levels: [&[f64]; 4],
) -> GasLines {
    let charted = |lines: Vec<Vec<(f64, f64)>>| {
        let mut family = GasFamily::default();
        for line in lines {
            let (a, b): (Vec<Point2>, Vec<Point2>) = line
                .iter()
                .map(|&(s, v)| {
                    let [p, q] = gas_points(gas, s, v);
                    (p, q)
                })
                .unzip();
            family.entropy_temperature.push(a);
            family.volume_pressure.push(b);
        }
        family
    };
    let grid: Vec<[f64; 2]> = volumes
        .iter()
        .flat_map(|&v| entropies.iter().map(move |&s| gas(s, v)))
        .collect();
    let level_family = |which: usize, values: &[f64]| {
        let samples: Vec<f64> = grid.iter().map(|g| g[which]).collect();
        let lines = values
            .iter()
            .flat_map(|&level| level_lines(entropies, volumes, &samples, level))
            .map(|line| {
                line.iter()
                    .map(|p| {
                        let [s, v] = p.to_euclidean();
                        (f64::from(s), f64::from(v))
                    })
                    .collect()
            })
            .collect();
        charted(lines)
    };
    let [isochores, isobars, isotherms, adiabats] = levels;
    GasLines {
        isochores: charted(
            isochores
                .iter()
                .map(|&v| entropies.iter().map(|&s| (s, v)).collect())
                .collect(),
        ),
        isobars: level_family(1, isobars),
        isotherms: level_family(0, isotherms),
        adiabats: charted(
            adiabats
                .iter()
                .map(|&s| volumes.iter().map(|&v| (s, v)).collect())
                .collect(),
        ),
    }
}

// --- panels ---------------------------------------------------------------------------

/// Panels side by side under the caption (the top 12% of the canvas), their widths in the
/// proportions `shares`.
pub fn row(c: &Canvas, shares: &[f32]) -> Vec<Rect> {
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let total: f32 = shares.iter().sum();
    let mut left = w * 0.01;
    let room = w * 0.98;
    shares
        .iter()
        .map(|s| {
            let width = room * s / total;
            let r = Rect::new(left, h * 0.14, left + width, h * 0.97);
            left += width;
            r
        })
        .collect()
}

// --- apparatus ------------------------------------------------------------------------

/// A box about a centre with half extents along x, y and z, its six faces lit, covering what is
/// behind by `opacity`, its edges a faint stroke of its light.
pub fn cuboid(s: &mut Scene3, centre: Point3, half: [f32; 3], l: Light, opacity: f32) {
    let centre = centre.unitized();
    let corner = |i: usize| {
        let sign = |bit: usize| if i & bit == 0 { -1.0 } else { 1.0 };
        centre + Point3::direction(sign(1) * half[0], sign(2) * half[1], sign(4) * half[2])
    };
    let p: [Point3; 8] = core::array::from_fn(corner);
    let faces = [
        [0, 2, 6, 4],
        [1, 5, 7, 3],
        [0, 4, 5, 1],
        [2, 3, 7, 6],
        [0, 1, 3, 2],
        [4, 6, 7, 5],
    ];
    for [a, b, cc, d] in faces {
        let shade = s.lit(p[a], p[b], p[cc], l);
        s.quad(p[a], p[b], p[cc], p[d], shade, opacity);
    }
    if opacity >= 1.0 {
        let edges = [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [0, 2],
            [1, 3],
            [4, 6],
            [5, 7],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ];
        let segs: Vec<[Point3; 2]> = edges.iter().map(|&[a, b]| [p[a], p[b]]).collect();
        s.stroke(&segs, 0.8, l.faded(0.6));
    }
}

/// A camera for an apparatus in the panel `rect`, at an elevation and azimuth (degrees; the
/// azimuth turned from looking along `+y`), its parallel lens fitted to show the corners whole.
pub fn apparatus_camera(rect: Rect, corners: &[Point3], azimuth: f32, elevation: f32) -> Camera {
    let n = corners.len() as f32;
    let middle: Point3 = corners
        .iter()
        .fold(Point3::direction(0.0, 0.0, 0.0), |sum, p| {
            sum + (p.unitized() - Point3::ORIGIN)
        })
        .gp(1.0 / n)
        + Point3::ORIGIN;
    let orbit = |lens| {
        Camera::orbit(
            rect,
            middle,
            30.0,
            (azimuth - 90.0).to_radians(),
            elevation.to_radians(),
            lens,
        )
    };
    let probe = orbit(Lens::Parallel(1.0));
    // How far the corners reach from the middle of the view, as a share of its half sizes.
    let centre = rect.centre();
    let reach = corners
        .iter()
        .filter_map(|p| probe.px(*p))
        .map(|q| {
            let d = q - centre;
            (d.e20().abs() / (0.5 * rect.width())).max(d.e01().abs() / (0.5 * rect.height()))
        })
        .fold(0.0f32, f32::max);
    orbit(Lens::Parallel(reach.max(1e-3) * 1.08))
}

/// The cylinder's inner half width and depth, its wall thickness, and the piston's thickness.
pub const BORE: f32 = 0.52;
pub const BORE_DEPTH: f32 = 0.43;
pub const WALL: f32 = 0.055;
pub const PLATE: f32 = 0.12;

/// A unit up.
fn rise(h: f32) -> Point3 {
    Point3::direction(0.0, 0.0, h)
}

/// A cylinder's base and walls, tall enough for a fill of `maximum`, and its piston resting on a
/// fill of `height`.
pub fn cylinder(s: &mut Scene3, height: f32, maximum: f32) {
    let grey = GREY.faded(0.55);
    cuboid(
        s,
        Point3::xyz(0.0, 0.0, -0.08),
        [BORE + 2.0 * WALL + 0.06, BORE_DEPTH + 0.17, 0.08],
        grey,
        1.0,
    );
    for side in [-1.0, 1.0] {
        cuboid(
            s,
            Point3::xyz(side * (BORE + WALL), 0.0, maximum * 0.5),
            [WALL, BORE_DEPTH + 0.07, maximum * 0.5],
            grey,
            1.0,
        );
    }
    cuboid(
        s,
        Point3::xyz(0.0, 0.0, PLATE * 0.5) + rise(height),
        [BORE + 0.06, BORE_DEPTH + 0.05, PLATE * 0.5],
        grey.faded(0.65),
        1.0,
    );
}

/// A weight on a piston resting on a fill of `height`: it holds the pressure fixed.
pub fn load(s: &mut Scene3, height: f32) {
    cuboid(
        s,
        Point3::xyz(0.0, 0.0, PLATE + 0.14) + rise(height),
        [0.31, 0.32, 0.14],
        INK.faded(0.5),
        1.0,
    );
}

/// A latch across a piston resting on a fill of `height`: it holds the volume fixed.
pub fn latch(s: &mut Scene3, height: f32) {
    cuboid(
        s,
        Point3::xyz(BORE + 0.12, 0.0, PLATE * 0.5) + rise(height),
        [0.14, BORE_DEPTH + 0.13, 0.05],
        ORANGE,
        1.0,
    );
}

/// The contents of the cylinder between two heights (at least a sliver apart), a translucent
/// box.
pub fn fill(s: &mut Scene3, bottom: f32, top: f32, l: Light, opacity: f32) {
    let top = bottom + (top - bottom).max(1e-3);
    cuboid(
        s,
        Point3::xyz(0.0, 0.0, 0.5 * (bottom + top)),
        [BORE, BORE_DEPTH, 0.5 * (top - bottom)],
        l,
        opacity,
    );
}

/// The camera on a cylinder tall enough for a fill of `maximum`, its piston and a weight on it.
pub fn cylinder_camera(rect: Rect, maximum: f32) -> Camera {
    let corners: Vec<Point3> = [-0.75, 0.75]
        .into_iter()
        .flat_map(|x| [-0.6, 0.6].map(|y| (x, y)))
        .flat_map(|(x, y)| [-0.16, maximum + 0.45].map(|z| Point3::xyz(x, y, z)))
        .collect();
    apparatus_camera(rect, &corners, -25.0, 18.0)
}

/// A cylinder of gas filled to a height, coloured by how warm it is, a share of the way from cold
/// to hot, under a piston either latched or loaded with a weight; framed for a fill of
/// `maximum`, in the panel `rect`.
pub fn gas_cylinder(
    c: &mut Canvas,
    rect: Rect,
    height: f32,
    warmth: f64,
    latched: bool,
    maximum: f32,
) {
    let cam = cylinder_camera(rect, maximum);
    panel3(c, rect, cam, |s| {
        cylinder(s, height, maximum);
        if latched {
            latch(s, height);
        } else {
            load(s, height);
        }
        fill(s, 0.0, height, thermal(warmth).faded(1.3), 0.7);
    });
}
