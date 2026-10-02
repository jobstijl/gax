//! A software canvas: linear-light RGB pixels, antialiased by coverage. Lines and disks take
//! their coverage from the distance of each pixel centre to the shape, polygons and shaded
//! images from sub-pixel samples; colours blend in linear light and are encoded to sRGB once,
//! when the canvas is shown or saved.

use crate::font::{self, Align};

/// A colour in linear light, each channel in `[0, 1]`.
pub type Rgb = [f32; 3];

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
        self.clip = [
            c(r[0], self.width),
            c(r[1], self.height),
            c(r[2], self.width),
            c(r[3], self.height),
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
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len2 = (dx * dx + dy * dy).max(1e-12);
        for y in ys {
            for x in xs.clone() {
                let (px, py) = (x as f32 + 0.5 - a[0], y as f32 + 0.5 - a[1]);
                let t = ((px * dx + py * dy) / len2).clamp(0.0, 1.0);
                let (ex, ey) = (px - t * dx, py - t * dy);
                let cover = (half + 0.5 - (ex * ex + ey * ey).sqrt()).clamp(0.0, 1.0);
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
        let (xs, ys) = self.span([o[0] - pad, o[1] - pad], [o[0] + pad, o[1] + pad]);
        for y in ys {
            for x in xs.clone() {
                let (dx, dy) = (x as f32 + 0.5 - o[0], y as f32 + 0.5 - o[1]);
                let d = (dx * dx + dy * dy).sqrt();
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

    /// A filled polygon (non-zero winding), antialiased by 4 x 4 samples per pixel.
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
        let winding = |x: f32, y: f32| {
            let mut w = 0i32;
            for i in 0..poly.len() {
                let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
                let side = (b[0] - a[0]) * (y - a[1]) - (x - a[0]) * (b[1] - a[1]);
                if a[1] <= y && b[1] > y && side > 0.0 {
                    w += 1;
                } else if a[1] > y && b[1] <= y && side < 0.0 {
                    w -= 1;
                }
            }
            w != 0
        };
        for y in ys {
            for x in xs.clone() {
                let mut n = 0;
                for sy in 0..4 {
                    for sx in 0..4 {
                        let (fx, fy) = (
                            x as f32 + (sx as f32 + 0.5) / 4.0,
                            y as f32 + (sy as f32 + 0.5) / 4.0,
                        );
                        n += usize::from(winding(fx, fy));
                    }
                }
                if n > 0 {
                    self.blend(x, y, c, alpha * n as f32 / 16.0);
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
