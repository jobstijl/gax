//! A software canvas: linear-light RGB pixels, antialiased by coverage. Lines and disks take
//! their coverage from the distance of each pixel centre to the shape (in PGA2D: joins and their
//! norms), polygons and shaded images from sub-pixel samples; colours blend in linear light and
//! are encoded to sRGB once, when the canvas is shown or saved.

use crate::font::{self, Align};
use gax::pga2d::Point;

/// A colour in linear light, each channel in `[0, 1]`.
pub type Rgb = [f32; 3];

/// A colour from a hex code `0xRRGGBB` in sRGB.
pub fn hex(c: u32) -> Rgb {
    let b = |s: u32| ((c >> s) & 0xff) as f32 / 255.0;
    srgb(b(16), b(8), b(0))
}

/// A colour from sRGB components in `[0, 1]` (as colour pickers give them).
pub fn srgb(r: f32, g: f32, b: f32) -> Rgb {
    let lin = |c: f32| {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    [lin(r), lin(g), lin(b)]
}

/// `a` towards `b` by `t`.
pub fn mix(a: Rgb, b: Rgb, t: f32) -> Rgb {
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t)
}

/// `c` scaled by `k`.
pub fn scale(c: Rgb, k: f32) -> Rgb {
    c.map(|x| x * k)
}

fn encode(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

/// A point in pixel coordinates: `x` to the right, `y` down, pixel centres at `+0.5`.
pub type Px = [f32; 2];

/// A grid of linear-light pixels.
#[derive(Clone)]
pub struct Canvas {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    px: Vec<Rgb>,
    /// Drawing is confined to these pixels: `[x0, y0, x1, y1]`, ends excluded.
    clip: [usize; 4],
}

impl Canvas {
    /// A black canvas.
    pub fn new(width: usize, height: usize) -> Canvas {
        Canvas {
            width,
            height,
            px: vec![[0.0; 3]; width * height],
            clip: [0, 0, width, height],
        }
    }

    /// Confine drawing to the pixel rectangle `[x0, y0, x1, y1]` (a panel), until [`Canvas::unclip`].
    pub fn clip(&mut self, r: [f32; 4]) {
        let c = |v: f32, n: usize| (v.max(0.0) as usize).min(n);
        let (x0, y0) = (c(r[0], self.width), c(r[1], self.height));
        // An inverted rectangle (an inset larger than its panel) clips everything away.
        self.clip = [
            x0,
            y0,
            c(r[2], self.width).max(x0),
            c(r[3], self.height).max(y0),
        ];
    }

    /// Draw anywhere again.
    pub fn unclip(&mut self) {
        self.clip = [0, 0, self.width, self.height];
    }

    /// Copy `other` with its top left corner at `(x, y)` (a panel drawn on its own canvas).
    pub fn blit(&mut self, other: &Canvas, x: usize, y: usize) {
        for row in 0..other.height.min(self.height.saturating_sub(y)) {
            let n = other.width.min(self.width.saturating_sub(x));
            let dst = (y + row) * self.width + x;
            self.px[dst..dst + n]
                .copy_from_slice(&other.px[row * other.width..row * other.width + n]);
        }
    }

    /// The pixel at `(x, y)`.
    pub fn get(&self, x: usize, y: usize) -> Rgb {
        self.px[y * self.width + x]
    }

    /// Fill with one colour.
    pub fn clear(&mut self, c: Rgb) {
        self.px.fill(c);
    }

    /// A vertical gradient from `top` to `bottom`.
    pub fn backdrop(&mut self, top: Rgb, bottom: Rgb) {
        let h = self.height.max(2) as f32 - 1.0;
        for y in 0..self.height {
            let c = mix(top, bottom, y as f32 / h);
            self.px[y * self.width..(y + 1) * self.width].fill(c);
        }
    }

    fn blend(&mut self, x: usize, y: usize, c: Rgb, a: f32) {
        let p = &mut self.px[y * self.width + x];
        *p = mix(*p, c, a);
    }

    /// The pixel rows and columns that a box around `lo..hi` covers.
    fn span(&self, lo: Px, hi: Px) -> (core::ops::Range<usize>, core::ops::Range<usize>) {
        let [x0, y0, x1, y1] = self.clip;
        let clamp = |v: f32, a: usize, b: usize| (v.max(a as f32) as usize).clamp(a, b);
        (
            clamp(lo[0].floor(), x0, x1)..clamp(hi[0].ceil() + 1.0, x0, x1),
            clamp(lo[1].floor(), y0, y1)..clamp(hi[1].ceil() + 1.0, y0, y1),
        )
    }

    /// A segment `width` pixels wide, round-capped, with opacity `alpha`. Thinner than a pixel it
    /// is drawn a pixel wide and fainter, so that thin lines stay smooth.
    pub fn line(&mut self, a: Px, b: Px, width: f32, c: Rgb, alpha: f32) {
        if !(a.iter().chain(&b).all(|v| v.is_finite())) {
            return;
        }
        let half = width.max(1.0) * 0.5;
        let faint = alpha * width.min(1.0);
        let pad = half + 1.0;
        let (xs, ys) = self.span(
            [a[0].min(b[0]) - pad, a[1].min(b[1]) - pad],
            [a[0].max(b[0]) + pad, a[1].max(b[1]) + pad],
        );
        // The pixel's distance to the segment: to its line where the foot falls between the
        // ends (the pixel lies between the perpendiculars `l | a` and `l | b`, so their joins
        // with it differ in sign), else to the nearer end. A join of unit points is a line
        // whose norm is their distance.
        let (pa, pb) = (Point::xy(a[0], a[1]), Point::xy(b[0], b[1]));
        let join = pa & pb;
        let len = join.norm();
        let l = join.gp(len.max(1e-12).recip());
        let (ends_a, ends_b) = (l | pa, l | pb);
        for y in ys {
            for x in xs.clone() {
                let q = Point::xy(x as f32 + 0.5, y as f32 + 0.5);
                let between = len > 1e-6 && (ends_a & q).s() * (ends_b & q).s() <= 0.0;
                let e = if between {
                    (l & q).s().abs()
                } else {
                    (q & pa).norm().min((q & pb).norm())
                };
                let cover = (half + 0.5 - e).clamp(0.0, 1.0);
                if cover > 0.0 {
                    self.blend(x, y, c, cover * faint);
                }
            }
        }
    }

    /// Segments through `pts`, closed back to the first when `closed`.
    pub fn polyline(&mut self, pts: &[Px], width: f32, c: Rgb, alpha: f32, closed: bool) {
        for w in pts.windows(2) {
            self.line(w[0], w[1], width, c, alpha);
        }
        if closed && pts.len() > 2 {
            self.line(pts[pts.len() - 1], pts[0], width, c, alpha);
        }
    }

    /// A filled disk.
    pub fn disk(&mut self, centre: Px, r: f32, c: Rgb, alpha: f32) {
        self.ring_or_disk(centre, r, None, c, alpha);
    }

    /// A circle `width` pixels wide.
    pub fn ring(&mut self, centre: Px, r: f32, width: f32, c: Rgb, alpha: f32) {
        self.ring_or_disk(centre, r, Some(width), c, alpha);
    }

    fn ring_or_disk(&mut self, o: Px, r: f32, width: Option<f32>, c: Rgb, alpha: f32) {
        if !(o[0].is_finite() && o[1].is_finite() && r.is_finite()) {
            return;
        }
        let pad = r + width.unwrap_or(0.0) + 1.0;
        let centre = Point::xy(o[0], o[1]);
        let (xs, ys) = self.span([o[0] - pad, o[1] - pad], [o[0] + pad, o[1] + pad]);
        for y in ys {
            for x in xs.clone() {
                let d = (Point::xy(x as f32 + 0.5, y as f32 + 0.5) & centre).norm();
                let cover = match width {
                    None => (r + 0.5 - d).clamp(0.0, 1.0),
                    Some(w) => {
                        (w.max(1.0) * 0.5 + 0.5 - (d - r).abs()).clamp(0.0, 1.0) * w.min(1.0)
                    }
                };
                if cover > 0.0 {
                    self.blend(x, y, c, cover * alpha);
                }
            }
        }
    }

    /// A filled polygon (non-zero winding), antialiased: four sub-rows per pixel row, each
    /// covering its spans exactly across. Each sub-row finds where the edges cross it, sorts
    /// the crossings, and adds the inside spans' coverage to the pixels they overlap, so the
    /// cost grows with the edges and the polygon's area, not their product.
    pub fn fill(&mut self, poly: &[Px], c: Rgb, alpha: f32) {
        if poly.len() < 3 || !poly.iter().flatten().all(|v| v.is_finite()) {
            return;
        }
        let lo = poly
            .iter()
            .fold([f32::MAX; 2], |m, p| [m[0].min(p[0]), m[1].min(p[1])]);
        let hi = poly
            .iter()
            .fold([f32::MIN; 2], |m, p| [m[0].max(p[0]), m[1].max(p[1])]);
        let (xs, ys) = self.span(lo, hi);
        if xs.is_empty() || ys.is_empty() {
            return;
        }
        let (x0, width) = (xs.start, xs.end - xs.start);
        let mut cover = vec![0.0f32; width];
        let mut cross: Vec<(f32, i32)> = Vec::new();
        const SUB: usize = 4;
        for y in ys {
            cover.iter_mut().for_each(|v| *v = 0.0);
            for sub in 0..SUB {
                let fy = y as f32 + (sub as f32 + 0.5) / SUB as f32;
                cross.clear();
                for i in 0..poly.len() {
                    let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
                    if (a[1] <= fy) != (b[1] <= fy) {
                        let t = (fy - a[1]) / (b[1] - a[1]);
                        cross.push((a[0] + t * (b[0] - a[0]), if b[1] > a[1] { 1 } else { -1 }));
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
                    self.blend(x0 + i, y, c, alpha * v.min(1.0));
                }
            }
        }
    }

    /// Shade every pixel from `n` x `n` samples of `f(x, y)` (pixel coordinates), averaged; a
    /// sample of `None` leaves the canvas showing through. Rows are shaded in parallel.
    pub fn shade(&mut self, n: usize, f: impl Fn(f32, f32) -> Option<Rgb> + Sync) {
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
                        let (x, y) = (xi as f32, yi as f32);
                        let (mut acc, mut hit) = ([0.0f32; 3], 0usize);
                        for sy in 0..n {
                            for sx in 0..n {
                                let fx = x + (sx as f32 + 0.5) / n as f32;
                                let fy = y + (sy as f32 + 0.5) / n as f32;
                                if let Some(c) = f(fx, fy) {
                                    acc = [0, 1, 2].map(|k| acc[k] + c[k]);
                                    hit += 1;
                                }
                            }
                        }
                        if hit > 0 {
                            let a = hit as f32 / (n * n) as f32;
                            *p = mix(*p, acc.map(|v| v / hit as f32), a);
                        }
                    }
                });
            }
        });
    }

    /// Text `size` pixels tall, its baseline at `y`, in the stroke font.
    pub fn text(&mut self, s: &str, x: f32, y: f32, size: f32, c: Rgb, align: Align) {
        let width = (size / 9.0).max(1.0);
        for [x0, y0, x1, y1] in font::segments(s, x, 0.0, size, align) {
            self.line([x0, y - y0], [x1, y - y1], width, c, 1.0);
        }
    }

    /// The pixels as sRGB `0x00RRGGBB` (for a window).
    pub fn to_xrgb(&self) -> Vec<u32> {
        self.px
            .iter()
            .map(|c| {
                let [r, g, b] = c.map(encode);
                (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b)
            })
            .collect()
    }

    /// The pixels as sRGB `RGBA` bytes (for files).
    pub fn to_rgba(&self) -> Vec<u8> {
        self.px
            .iter()
            .flat_map(|c| {
                let [r, g, b] = c.map(encode);
                [r, g, b, 255]
            })
            .collect()
    }

    /// The mean of each channel (for tests: is anything drawn?).
    pub fn mean(&self) -> Rgb {
        let n = self.px.len().max(1) as f32;
        self.px
            .iter()
            .fold([0.0; 3], |m, c| [m[0] + c[0], m[1] + c[1], m[2] + c[2]])
            .map(|v| v / n)
    }
}
