//! Level lines of a scalar field on a grid by marching squares: the segments where the field
//! crosses `level`, with the crossings placed by linear interpolation along the cell edges.

use crate::coords::Point2;

/// The segments (in grid coordinates: the point `(column, row)`) where `f` crosses `level`, for
/// `f` sampled on a `cols` x `rows` grid (`f[row * cols + col]`). Non-finite samples break the
/// line.
pub fn segments(f: &[f32], cols: usize, rows: usize, level: f32) -> Vec<[Point2; 2]> {
    let mut out = Vec::new();
    let at = |c: usize, r: usize| f[r * cols + c] - level;
    for r in 0..rows.saturating_sub(1) {
        for c in 0..cols.saturating_sub(1) {
            // Corners counterclockwise from the bottom left (in grid coordinates).
            let v = [at(c, r), at(c + 1, r), at(c + 1, r + 1), at(c, r + 1)];
            if v.iter().any(|x| !x.is_finite()) {
                continue;
            }
            let corner = Point2::xy(c as f32, r as f32);
            let p = [
                corner,
                corner + Point2::direction(1.0, 0.0),
                corner + Point2::direction(1.0, 1.0),
                corner + Point2::direction(0.0, 1.0),
            ];
            // Along the edge from corner `i` to the next, where the field reaches the level.
            let cross = |i: usize| {
                let j = (i + 1) % 4;
                let t = v[i] / (v[i] - v[j]);
                p[i] + (p[j] - p[i]).gp(t)
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

/// [`segments`] of a function of the plane sampled on a grid over the ranges `x` by `y` (`n`
/// samples per side), in the function's own coordinates.
pub fn of_fn(
    f: impl Fn(Point2) -> f32,
    x: [f32; 2],
    y: [f32; 2],
    n: usize,
    level: f32,
) -> Vec<[Point2; 2]> {
    let n = n.max(2);
    // Grid point `(i, j)` in the function's coordinates.
    let to = |g: Point2| {
        let [i, j] = g.to_euclidean();
        let s = |r: [f32; 2], v: f32| r[0] + (r[1] - r[0]) * v / (n - 1) as f32;
        Point2::xy(s(x, i), s(y, j))
    };
    let grid: Vec<f32> = (0..n * n)
        .map(|k| f(to(Point2::xy((k % n) as f32, (k / n) as f32))))
        .collect();
    segments(&grid, n, n, level)
        .into_iter()
        .map(|[a, b]| [to(a), to(b)])
        .collect()
}
