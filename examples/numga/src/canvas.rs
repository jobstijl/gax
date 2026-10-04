//! A software canvas of light, as `examples/warp` draws: every pixel holds a light (a PGA3D
//! point in linear RGB whose weight is its intensity, the `gax-colour` crate), and the canvas is
//! shown through AgX, warp's tonemapper.
//!
//! Strokes (lines, polylines, rings, text) and dots add their light: a solid core and a soft
//! glow, from the distance of each pixel centre to the shape. A stroke is one shape, however
//! many segments it has: each pixel takes its distance to the nearest segment, so the joints of
//! a polyline do not shine twice, while separate strokes add up where they cross. Fills and
//! images cover what is below by their opacity, an affine combination of lights.
//!
//! Pixel positions are PGA2D points (`x` to the right, `y` down, pixel centres at `+0.5`), and
//! distances are the norms of joins.

use crate::font::{self, Align};
use crate::points::{Point2, box_map};
use gax_colour::{DARK, Light, Srgb};

/// How far AgX's look moves colours away from grey (as warp's).
const SATURATION: f32 = 1.35;

/// A stroke's glow: its strength at the stroke's centre line, and its radius as a multiple of
/// the stroke's half width, plus a pixel and a half.
const GLOW: (f32, f32) = (0.16, 2.5);

/// A rectangle of the canvas: its top left and bottom right corners, in pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    /// The top left corner.
    pub lo: Point2,
    /// The bottom right corner.
    pub hi: Point2,
}

impl Rect {
    /// The rectangle from `(x0, y0)` to `(x1, y1)`, in pixels.
    pub fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
        Rect {
            lo: Point2::xy(x0, y0),
            hi: Point2::xy(x1, y1),
        }
    }

    /// The diagonal from the top left corner to the bottom right one: a direction.
    pub fn size(self) -> Point2 {
        self.hi - self.lo
    }

    /// The width in pixels.
    pub fn width(self) -> f32 {
        self.size().e20()
    }

    /// The height in pixels.
    pub fn height(self) -> f32 {
        self.size().e01()
    }

    /// The centre: the midpoint of the corners.
    pub fn centre(self) -> Point2 {
        (self.lo + self.hi).unitized()
    }

    /// The rectangle with the given centre and half diagonal (a direction).
    pub fn around(centre: Point2, half: Point2) -> Rect {
        let centre = centre.unitized();
        Rect {
            lo: centre - half,
            hi: centre + half,
        }
    }

    /// The part of the rectangle from the fractions `(x0, y0)` to `(x1, y1)` of its width and
    /// height (from the top left).
    pub fn part(self, x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
        let (w, h) = (self.width(), self.height());
        Rect {
            lo: self.lo + Point2::direction(x0 * w, y0 * h),
            hi: self.lo + Point2::direction(x1 * w, y1 * h),
        }
    }

    /// The middle of the top edge.
    pub fn top_middle(self) -> Point2 {
        (self.lo + self.top_right()).unitized()
    }

    /// The middle of the bottom edge.
    pub fn bottom_middle(self) -> Point2 {
        (self.bottom_left() + self.hi).unitized()
    }

    /// The middle of the left edge.
    pub fn left_middle(self) -> Point2 {
        (self.lo + self.bottom_left()).unitized()
    }

    /// The middle of the right edge.
    pub fn right_middle(self) -> Point2 {
        (self.top_right() + self.hi).unitized()
    }

    /// The bottom left corner.
    pub fn bottom_left(self) -> Point2 {
        self.lo + Point2::direction(0.0, self.height())
    }

    /// The top right corner.
    pub fn top_right(self) -> Point2 {
        self.lo + Point2::direction(self.width(), 0.0)
    }

    /// The rectangle shrunk by `left`, `top`, `right` and `bottom` pixels.
    pub fn inset(self, left: f32, top: f32, right: f32, bottom: f32) -> Rect {
        Rect {
            lo: self.lo + Point2::direction(left, top),
            hi: self.hi - Point2::direction(right, bottom),
        }
    }

    /// The `i`-th of `n` equal columns of the rectangle, left to right.
    pub fn column(self, i: usize, n: usize) -> Rect {
        let step = Point2::direction(self.width() / n as f32, 0.0);
        let lo = self.lo + step.gp(i as f32);
        Rect {
            lo,
            hi: lo + step + Point2::direction(0.0, self.height()),
        }
    }
}

/// A grid of lights.
#[derive(Clone)]
pub struct Canvas {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    px: Vec<Light>,
    /// Drawing is confined to these pixels: `[x0, y0, x1, y1]`, ends excluded.
    clip: [usize; 4],
    /// A stroke's coverage per pixel while it is drawn, and the pixels it touched.
    cover: Vec<f32>,
    touched: Vec<usize>,
}

/// The pixel indices `[x0, y0, x1, y1]` of the box around `pts`, grown by `pad` pixels and
/// limited to `clip` (ends excluded).
fn span(pts: &[Point2], pad: f32, clip: [usize; 4]) -> [usize; 4] {
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for p in pts {
        for (i, v) in p.to_euclidean().into_iter().enumerate() {
            (lo[i], hi[i]) = (lo[i].min(v - pad), hi[i].max(v + pad));
        }
    }
    let index = |v: f32, a: usize, b: usize| (v.max(a as f32) as usize).clamp(a, b);
    [
        index(lo[0].floor(), clip[0], clip[2]),
        index(lo[1].floor(), clip[1], clip[3]),
        index(hi[0].ceil() + 1.0, clip[0], clip[2]),
        index(hi[1].ceil() + 1.0, clip[1], clip[3]),
    ]
}

/// The centre of pixel `(x, y)`.
fn centre(x: usize, y: usize) -> Point2 {
    Point2::xy(x as f32 + 0.5, y as f32 + 0.5)
}

/// Whether a point is finite and not at infinity (a sample that did not blow up).
fn drawable(p: Point2) -> bool {
    p.c.iter().all(|x| x.is_finite()) && p.e12() != 0.0
}

/// The distance of `q` to the segment from `a` to `b` (unit points): to its line where the foot
/// falls between the ends (`q` lies between the perpendiculars `l | a` and `l | b`, so its
/// joins with them differ in sign), else to the nearer end. A join of unit points is a line
/// whose norm is their distance.
fn segment_distance(a: Point2, b: Point2, q: Point2) -> f32 {
    capped_distance(a, b, q, [true, true])
}

/// The same with each end round (`caps`) or flat: past a flat end the segment reaches no pixel,
/// so where a stroke is drawn in pieces, their flat ends meet without overlapping.
fn capped_distance(a: Point2, b: Point2, q: Point2, caps: [bool; 2]) -> f32 {
    let join = a & b;
    let len = join.norm();
    if len <= 1e-6 {
        return if caps[0] || caps[1] {
            (q & a).norm()
        } else {
            f32::INFINITY
        };
    }
    let l = join.gp(len.recip());
    let (beyond_a, beyond_b) = (((l | a) & q).s(), ((l | b) & q).s());
    if beyond_a * beyond_b <= 0.0 {
        return (l & q).s().abs();
    }
    // Past an end: the end nearer the pixel decides.
    let (near, cap) = if (q & a).norm() < (q & b).norm() {
        (a, caps[0])
    } else {
        (b, caps[1])
    };
    if cap {
        (q & near).norm()
    } else {
        f32::INFINITY
    }
}

/// The core of a shape `half` pixels wide on each side at a pixel `e` pixels from its middle,
/// anti-aliased over a pixel.
fn core(e: f32, half: f32) -> f32 {
    (half + 0.5 - e).clamp(0.0, 1.0)
}

/// The glow around it: a Gaussian.
fn glow(e: f32, half: f32) -> f32 {
    let g = GLOW.1 * half + 1.5;
    GLOW.0 * (-(e * e) / (g * g)).exp()
}

/// How much of a stroke's light reaches a pixel `e` pixels from its centre line: the core and
/// the glow.
fn stroke_light(e: f32, half: f32) -> f32 {
    core(e, half) + glow(e, half)
}

/// How far a stroke's glow reaches, in pixels.
fn reach(half: f32) -> f32 {
    half + 2.5 * (GLOW.1 * half + 1.5)
}

impl Canvas {
    /// A dark canvas.
    pub fn new(width: usize, height: usize) -> Canvas {
        Canvas {
            width,
            height,
            px: vec![DARK; width * height],
            clip: [0, 0, width, height],
            cover: vec![0.0; width * height],
            touched: Vec::new(),
        }
    }

    /// The canvas's unit of size: its height over 540 (the examples are laid out at 960x540),
    /// so that text, insets and markers scaled by it make a smaller canvas a miniature.
    pub fn unit(&self) -> f32 {
        self.height as f32 / 540.0
    }

    /// The whole canvas, as a rectangle.
    pub fn rect(&self) -> Rect {
        Rect::new(0.0, 0.0, self.width as f32, self.height as f32)
    }

    /// Confine drawing to `r` (a panel), until [`Canvas::unclip`].
    pub fn clip(&mut self, r: Rect) {
        let [x0, y0] = r.lo.to_euclidean();
        let [x1, y1] = r.hi.to_euclidean();
        let index = |v: f32, n: usize| (v.max(0.0) as usize).min(n);
        let (x0, y0) = (index(x0, self.width), index(y0, self.height));
        // An inverted rectangle (an inset larger than its panel) clips everything away.
        self.clip = [
            x0,
            y0,
            index(x1, self.width).max(x0),
            index(y1, self.height).max(y0),
        ];
    }

    /// Draw anywhere again.
    pub fn unclip(&mut self) {
        self.clip = [0, 0, self.width, self.height];
    }

    /// Copy `other` with its top left corner at pixel `(x, y)` (a panel drawn on its own canvas).
    pub fn blit(&mut self, other: &Canvas, x: usize, y: usize) {
        for row in 0..other.height.min(self.height.saturating_sub(y)) {
            let n = other.width.min(self.width.saturating_sub(x));
            let dst = (y + row) * self.width + x;
            self.px[dst..dst + n]
                .copy_from_slice(&other.px[row * other.width..row * other.width + n]);
        }
    }

    /// The light at pixel `(x, y)`.
    pub fn get(&self, x: usize, y: usize) -> Light {
        self.px[y * self.width + x]
    }

    /// Fill with one light.
    pub fn clear(&mut self, l: Light) {
        self.px.fill(l);
    }

    /// A vertical gradient from `top` to `bottom`.
    pub fn backdrop(&mut self, top: Light, bottom: Light) {
        let h = self.height.max(2) as f32 - 1.0;
        for y in 0..self.height {
            let l = top.mix_light(bottom, y as f32 / h);
            self.px[y * self.width..(y + 1) * self.width].fill(l);
        }
    }

    fn add(&mut self, i: usize, l: Light) {
        self.px[i] += l;
    }

    fn cover_with(&mut self, i: usize, l: Light, opacity: f32) {
        self.px[i] = self.px[i].mix_light(l, opacity);
    }

    /// The segments `segs` as one stroke `width` pixels wide: each pixel's coverage from its
    /// distance to the nearest segment, then the light added once. Thinner than a pixel, the
    /// stroke is drawn a pixel wide and fainter, keeping its light.
    pub fn stroke(&mut self, segs: &[[Point2; 2]], width: f32, l: Light) {
        let caps = vec![[true, true]; segs.len()];
        self.stroke_capped(segs, &caps, width, l);
    }

    /// [`Canvas::stroke`] with each segment's ends round or flat (`caps`): the pieces of a
    /// stroke drawn apart (a 3D curve split by what lies between its parts) take flat ends where
    /// they join, so the joints do not shine twice.
    pub fn stroke_capped(
        &mut self,
        segs: &[[Point2; 2]],
        caps: &[[bool; 2]],
        width: f32,
        l: Light,
    ) {
        let half = width.max(1.0) * 0.5;
        let l = l * width.min(1.0);
        for (&[a, b], &cap) in segs.iter().zip(caps) {
            if !(drawable(a) && drawable(b)) {
                continue;
            }
            let (a, b) = (a.unitized(), b.unitized());
            let [x0, y0, x1, y1] = span(&[a, b], reach(half), self.clip);
            for y in y0..y1 {
                for x in x0..x1 {
                    let k = stroke_light(capped_distance(a, b, centre(x, y), cap), half);
                    let i = y * self.width + x;
                    if k > 1e-3 && k > self.cover[i] {
                        if self.cover[i] == 0.0 {
                            self.touched.push(i);
                        }
                        self.cover[i] = k;
                    }
                }
            }
        }
        for n in 0..self.touched.len() {
            let i = self.touched[n];
            self.add(i, l * self.cover[i]);
            self.cover[i] = 0.0;
        }
        self.touched.clear();
    }

    /// The segments `segs` as one line `width` pixels wide that covers what is below by
    /// `opacity` (no glow): an edge or outline in a colour darker than what it crosses, which
    /// added light could not draw.
    pub fn outline(&mut self, segs: &[[Point2; 2]], width: f32, l: Light, opacity: f32) {
        let half = width.max(1.0) * 0.5;
        let opacity = opacity * width.min(1.0);
        for &[a, b] in segs {
            if !(drawable(a) && drawable(b)) {
                continue;
            }
            let (a, b) = (a.unitized(), b.unitized());
            let [x0, y0, x1, y1] = span(&[a, b], half + 1.0, self.clip);
            for y in y0..y1 {
                for x in x0..x1 {
                    let k = core(segment_distance(a, b, centre(x, y)), half);
                    let i = y * self.width + x;
                    if k > 0.0 && k > self.cover[i] {
                        if self.cover[i] == 0.0 {
                            self.touched.push(i);
                        }
                        self.cover[i] = k;
                    }
                }
            }
        }
        for n in 0..self.touched.len() {
            let i = self.touched[n];
            self.cover_with(i, l, opacity * self.cover[i]);
            self.cover[i] = 0.0;
        }
        self.touched.clear();
    }

    /// A segment `width` pixels wide.
    pub fn line(&mut self, a: Point2, b: Point2, width: f32, l: Light) {
        self.stroke(&[[a, b]], width, l);
    }

    /// One stroke through `pts`, closed back to the first when `closed`.
    pub fn polyline(&mut self, pts: &[Point2], width: f32, l: Light, closed: bool) {
        let mut segs: Vec<[Point2; 2]> = pts.windows(2).map(|w| [w[0], w[1]]).collect();
        if closed && pts.len() > 2 {
            segs.push([pts[pts.len() - 1], pts[0]]);
        }
        self.stroke(&segs, width, l);
    }

    /// A dot of radius `r`: a solid disk covering what is below, and its glow added around it,
    /// so that dots crowded together keep their own colours.
    pub fn disk(&mut self, centre_point: Point2, r: f32, l: Light) {
        if !drawable(centre_point) || !r.is_finite() {
            return;
        }
        let o = centre_point.unitized();
        let half = r.max(0.5);
        let [x0, y0, x1, y1] = span(&[o], reach(half), self.clip);
        for y in y0..y1 {
            for x in x0..x1 {
                let e = (centre(x, y) & o).norm();
                let i = y * self.width + x;
                let k = core(e, half);
                if k > 0.0 {
                    self.cover_with(i, l, k);
                }
                self.add(i, l * glow(e, half));
            }
        }
    }

    /// A circle of radius `r`, `width` pixels wide.
    pub fn ring(&mut self, centre_point: Point2, r: f32, width: f32, l: Light) {
        if !drawable(centre_point) || !r.is_finite() {
            return;
        }
        let o = centre_point.unitized();
        let half = width.max(1.0) * 0.5;
        let l = l * width.min(1.0);
        let [x0, y0, x1, y1] = span(&[o], r + reach(half), self.clip);
        for y in y0..y1 {
            for x in x0..x1 {
                let k = stroke_light(((centre(x, y) & o).norm() - r).abs(), half);
                if k > 1e-3 {
                    self.add(y * self.width + x, l * k);
                }
            }
        }
    }

    /// A filled polygon (non-zero winding) covering what is below by `opacity`, antialiased:
    /// four sub-rows per pixel row, each covering its spans exactly across. A sub-row is a line;
    /// each edge that crosses it meets it at a point, and the crossings, sorted, bound the
    /// inside spans, whose coverage goes to the pixels they overlap. The cost grows with the
    /// edges and the polygon's area, not their product.
    pub fn fill(&mut self, poly: &[Point2], l: Light, opacity: f32) {
        if poly.len() < 3 || !poly.iter().all(|p| drawable(*p)) {
            return;
        }
        let poly: Vec<Point2> = poly.iter().map(|p| p.unitized()).collect();
        let [x0, y0, x1, y1] = span(&poly, 0.0, self.clip);
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        let width = x1 - x0;
        let edges: Vec<(Point2, Point2)> = (0..poly.len())
            .map(|i| (poly[i], poly[(i + 1) % poly.len()]))
            .collect();
        let mut cover = vec![0.0f32; width];
        let mut cross: Vec<(f32, i32)> = Vec::new();
        const SUB: usize = 4;
        for y in y0..y1 {
            cover.iter_mut().for_each(|v| *v = 0.0);
            for sub in 0..SUB {
                let fy = y as f32 + (sub as f32 + 0.5) / SUB as f32;
                let row = Point2::xy(0.0, fy) & Point2::xy(1.0, fy);
                cross.clear();
                for &(a, b) in &edges {
                    let up = (b - a).e01();
                    if (a.e01() <= fy) != (b.e01() <= fy) {
                        let at = ((a & b) ^ row).to_euclidean()[0];
                        cross.push((at, if up > 0.0 { 1 } else { -1 }));
                    }
                }
                cross.sort_by(|p, q| p.0.total_cmp(&q.0));
                let mut wind = 0;
                for k in 0..cross.len() {
                    let before = wind;
                    wind += cross[k].1;
                    if before == 0 && wind != 0 {
                        // An inside span starts here and ends where the winding returns to 0.
                        let start = cross[k].0;
                        let mut w = wind;
                        let mut end = start;
                        for &(x, d) in &cross[k + 1..] {
                            w += d;
                            if w == 0 {
                                end = x;
                                break;
                            }
                        }
                        // Add the span's width in each pixel it overlaps.
                        let (a, b) = (start - x0 as f32, end - x0 as f32);
                        let first = a.max(0.0).floor() as usize;
                        let last = (b.min(width as f32).ceil() as usize).min(width);
                        for (px, v) in cover.iter_mut().enumerate().take(last).skip(first) {
                            let (l, r) = (px as f32, px as f32 + 1.0);
                            *v += (b.min(r) - a.max(l)).max(0.0) / SUB as f32;
                        }
                    }
                }
            }
            for (i, v) in cover.iter().enumerate() {
                if *v > 0.0 {
                    self.cover_with(y * self.width + x0 + i, l, opacity * v.min(1.0));
                }
            }
        }
    }

    /// Shade every pixel from `n` x `n` samples of `f` at points inside it, averaged; a sample
    /// of `None` leaves the canvas showing through. Rows are shaded in parallel.
    pub fn shade(&mut self, n: usize, f: impl Fn(Point2) -> Option<Light> + Sync) {
        let (w, n, clip) = (self.width, n.max(1), self.clip);
        let threads = std::thread::available_parallelism().map_or(4, |t| t.get());
        let rows_per = self.height.div_ceil(threads).max(1);
        std::thread::scope(|s| {
            for (chunk, rows) in self.px.chunks_mut(rows_per * w).enumerate() {
                let f = &f;
                s.spawn(move || {
                    for (i, p) in rows.iter_mut().enumerate() {
                        let (xi, yi) = (i % w, chunk * rows_per + i / w);
                        if xi < clip[0] || xi >= clip[2] || yi < clip[1] || yi >= clip[3] {
                            continue;
                        }
                        let corner = Point2::xy(xi as f32, yi as f32);
                        let (mut acc, mut hit) = (DARK, 0usize);
                        for sy in 0..n {
                            for sx in 0..n {
                                let inside = Point2::direction(
                                    (sx as f32 + 0.5) / n as f32,
                                    (sy as f32 + 0.5) / n as f32,
                                );
                                if let Some(l) = f(corner + inside) {
                                    acc += l;
                                    hit += 1;
                                }
                            }
                        }
                        if hit > 0 {
                            let opacity = hit as f32 / (n * n) as f32;
                            *p = (*p).mix_light(acc * (hit as f32).recip(), opacity);
                        }
                    }
                });
            }
        });
    }

    /// Text `size` pixels tall, its baseline's anchor at `at`, in the stroke font: one stroke.
    pub fn text(&mut self, s: &str, at: Point2, size: f32, l: Light, align: Align) {
        // The font's frame (glyph units, y up) onto the canvas (pixels, y down) at `at`.
        let k = size / 6.0;
        let corner = at + Point2::direction(k, -k);
        let place = box_map([Point2::xy(0.0, 0.0), Point2::xy(1.0, 1.0)], [at, corner]);
        let segs: Vec<[Point2; 2]> = font::segments(s, align)
            .into_iter()
            .map(|[a, b]| [place.of(a), place.of(b)])
            .collect();
        self.stroke(&segs, (size / 9.0).max(1.0), l);
    }

    /// Each pixel as shown: AgX of its light, the colour that makes on black, as sRGB bytes.
    fn display(&self) -> impl Iterator<Item = [u8; 3]> + '_ {
        self.px.iter().map(|l| {
            let shown: Srgb = l.agx(SATURATION).on_black().to();
            let [r, g, b, _] = shown.to_u8();
            [r, g, b]
        })
    }

    /// The pixels as sRGB `0x00RRGGBB` (for a window).
    pub fn to_xrgb(&self) -> Vec<u32> {
        self.display()
            .map(|[r, g, b]| (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b))
            .collect()
    }

    /// The pixels as sRGB `RGBA` bytes (for files).
    pub fn to_rgba(&self) -> Vec<u8> {
        self.display()
            .flat_map(|[r, g, b]| [r, g, b, 255])
            .collect()
    }

    /// The mean light (for tests: is anything drawn?).
    pub fn mean(&self) -> Light {
        let n = self.px.len().max(1) as f32;
        self.px.iter().fold(DARK, |m, l| m + *l) * n.recip()
    }
}
