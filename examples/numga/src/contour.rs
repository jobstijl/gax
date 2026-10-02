//! Level lines of a scalar field on a grid by marching squares: the segments where the field
//! crosses `level`, with the crossings placed by linear interpolation along the cell edges.

/// The segments `[[x0, y0], [x1, y1]]` (in grid coordinates: column, row) where `f` crosses
/// `level`, for `f` sampled on a `cols` x `rows` grid (`f[row * cols + col]`). Non-finite
/// samples break the line.
pub fn segments(f: &[f32], cols: usize, rows: usize, level: f32) -> Vec<[[f32; 2]; 2]> {
    let mut out = Vec::new();
    let at = |c: usize, r: usize| f[r * cols + c] - level;
    for r in 0..rows.saturating_sub(1) {
        for c in 0..cols.saturating_sub(1) {
            // Corners counterclockwise from the bottom left (in grid coordinates).
            let v = [at(c, r), at(c + 1, r), at(c + 1, r + 1), at(c, r + 1)];
            if v.iter().any(|x| !x.is_finite()) {
                continue;
            }
            let p = [
                [c as f32, r as f32],
                [c as f32 + 1.0, r as f32],
                [c as f32 + 1.0, r as f32 + 1.0],
                [c as f32, r as f32 + 1.0],
            ];
            let cross = |i: usize| {
                let j = (i + 1) % 4;
                let t = v[i] / (v[i] - v[j]);
                [
                    p[i][0] + t * (p[j][0] - p[i][0]),
                    p[i][1] + t * (p[j][1] - p[i][1]),
                ]
            };
            let edges: Vec<usize> = (0..4)
                .filter(|&i| (v[i] < 0.0) != (v[(i + 1) % 4] < 0.0))
                .collect();
            match edges.len() {
                2 => out.push([cross(edges[0]), cross(edges[1])]),
                4 => {
                    // A saddle: pair the crossings by the sign at the centre.
                    let centre = (v[0] + v[1] + v[2] + v[3]) * 0.25;
                    if (centre < 0.0) == (v[0] < 0.0) {
                        out.push([cross(0), cross(1)]);
                        out.push([cross(2), cross(3)]);
                    } else {
                        out.push([cross(3), cross(0)]);
                        out.push([cross(1), cross(2)]);
                    }
                }
                _ => {}
            }
        }
    }
    out
}

/// [`segments`] of a function sampled on a grid over `[x0, x1] x [y0, y1]` (`n` samples per
/// side), in the function's own coordinates.
pub fn of_fn(
    f: impl Fn(f32, f32) -> f32,
    x: [f32; 2],
    y: [f32; 2],
    n: usize,
    level: f32,
) -> Vec<[[f32; 2]; 2]> {
    let n = n.max(2);
    let step = |r: [f32; 2], i: usize| r[0] + (r[1] - r[0]) * i as f32 / (n - 1) as f32;
    let grid: Vec<f32> = (0..n * n)
        .map(|k| f(step(x, k % n), step(y, k / n)))
        .collect();
    let to = |g: [f32; 2]| {
        let s = |r: [f32; 2], v: f32| r[0] + (r[1] - r[0]) * v / (n - 1) as f32;
        [s(x, g[0]), s(y, g[1])]
    };
    segments(&grid, n, n, level)
        .into_iter()
        .map(|[a, b]| [to(a), to(b)])
        .collect()
}
