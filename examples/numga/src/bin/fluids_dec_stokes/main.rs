//! numga's `fluids/dec_stokes`: Stokes flow on a triangle mesh and its circumcentric dual, in
//! discrete exterior calculus, through a channel past an offset circular obstacle.
//!
//! The unknowns are the fluxes through the edges, numbers that balance on every triangle. Their
//! Hodge dual measures circulation along the dual edges, from circumcentre to circumcentre, and
//! summed around each dual cell by the adjugate of the vorticity's derivative it is the
//! vorticity at a vertex, a bivector. The vorticity's differences across the primal edges, paired
//! back through the adjugate of the Hodge map, are the viscous resistance, which the pressure
//! balances while it enforces zero net flux through every face: a saddle-point system, solved as
//! one sparse system. The incidences hold only signs; lengths and areas enter through the Hodge
//! maps alone, each the oriented patch of a dual cell over its primal cell's. Reversing an edge
//! changes the sign its flux is read with and nothing physical.
//!
//! Top, tracers ride the flow's streamlines over the pressure; below, the vorticity with the
//! same streamlines, and the primal mesh and its circumcentric dual about one vertex: its dual
//! cell, and an edge crossed by its dual edge.

mod scenes;
mod stokes;
#[cfg(test)]
mod tests;

#[path = "../../shared/simplicial.rs"]
mod simplicial;
#[path = "../../shared/surfaces_plane.rs"]
mod surfaces_plane;

use gax::field::Field;
use gax_numga_examples::{
    Anim, Axes, Canvas, Light, Marker, Point2, Rect, backdrop, caption, label_under, palette,
    run_with,
};
use scenes::*;
use simplicial::faces;
use stokes::*;
use surfaces_plane::Flat;

/// One loop: numga's 84 frames of 40 ms.
const SECONDS: f32 = 3.36;
/// The streamlines: the streamfunction's levels evenly between its values on the walls.
const STREAMLINES: usize = 13;
/// The tracer's step along the face velocities, and the trail's length in flow time.
const TRACER_STEP: f64 = 0.004;
const TRAIL: f64 = 0.06;
/// The vertex whose dual cell is drawn: the interior vertex nearest this point.
const COMPLEX_TARGET: [f64; 2] = [1.35, -0.35];

/// A tracer's path: its points and the flow time at each.
struct Path {
    points: Vec<Point2>,
    times: Vec<f64>,
}

struct Shown {
    channel: Channel,
    operators: Operators,
    flow: Flow,
    flat: Flat,
    streamfunction: Vec<f64>,
    levels: Vec<f64>,
    paths: Vec<Path>,
    /// The time between tracers on a path.
    period: f64,
}

/// A point as drawn.
fn at(p: Point) -> Point2 {
    let [x, y] = coordinates(p);
    Point2::xy(x as f32, y as f32)
}

/// The path of a tracer from a point, along the face velocities, until it leaves the channel.
fn trace(flat: &Flat, velocity: &Field<Point>, start: Point2) -> Path {
    let (mut points, mut times) = (vec![start], vec![0.0]);
    let mut p = start;
    let mut time = 0.0;
    while let Some((f, _)) = flat.locate(p) {
        let [vx, vy] = coordinates(velocity.site(f));
        let speed = gax::vga2d::Vector::new(vx, vy).norm();
        if speed < 1e-9 || points.len() > 20_000 {
            break;
        }
        let step = TRACER_STEP / speed;
        p += Point2::direction((vx * step) as f32, (vy * step) as f32);
        time += step;
        points.push(p);
        times.push(time);
    }
    Path { points, times }
}

fn shown() -> Shown {
    let channel = channel(DIVISIONS);
    let signs = vec![1.0; channel.mesh.len(1)];
    let (operators, _, flow) = flow(&channel, &signs);
    let streamfunction: Vec<f64> = operators
        .d0
        .lstsq(&flow.flux)
        .iter()
        .map(|s| s.s())
        .collect();
    let points: Vec<[f64; 2]> = channel.vertices.iter().map(coordinates).collect();
    let flat = Flat::new(&points, &faces(&channel.mesh));
    // The levels between the streamfunction's values on the two walls.
    let low = streamfunction.iter().copied().fold(f64::INFINITY, f64::min);
    let high = streamfunction
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let levels: Vec<f64> = (1..=STREAMLINES)
        .map(|k| low + (high - low) * k as f64 / (STREAMLINES + 1) as f64)
        .collect();
    // Each level's tracer starts where the parabolic inflow's streamfunction reaches it, a little
    // inside the inlet.
    let paths: Vec<Path> = (1..=STREAMLINES)
        .map(|k| {
            let share = k as f64 / (STREAMLINES + 1) as f64;
            // The inflow's streamfunction, `1.5 (y − 4 y³ / (3 H²))`, is a cubic in the height;
            // its share of the flux between the walls is reached by bisection.
            let stream = |y: f64| 1.5 * (y - 4.0 * y * y * y / (3.0 * HEIGHT * HEIGHT));
            let target =
                stream(-HEIGHT / 2.0) + share * (stream(HEIGHT / 2.0) - stream(-HEIGHT / 2.0));
            let y = gax_numga_examples::measure::bisect(-HEIGHT / 2.0, HEIGHT / 2.0, 60, |y| {
                stream(y) < target
            });
            trace(&flat, &flow.velocity, Point2::xy(0.02, y as f32))
        })
        .collect();
    let mut durations: Vec<f64> = paths
        .iter()
        .map(|p| *p.times.last().expect("a path"))
        .collect();
    durations.sort_by(f64::total_cmp);
    let period = durations[durations.len() / 2] / 6.0;
    Shown {
        channel,
        operators,
        flow,
        flat,
        streamfunction,
        levels,
        paths,
        period,
    }
}

/// The value of `|v|` below which a share `p` of the values lie.
fn percentile(values: &[f64], p: f64) -> f64 {
    let mut sorted: Vec<f64> = values.iter().map(|v| v.abs()).collect();
    sorted.sort_by(f64::total_cmp);
    sorted[(p * (sorted.len() - 1) as f64).round() as usize].max(1e-12)
}

/// A signed value's light: orange where it is positive, sky where negative.
fn diverging(v: f64) -> Light {
    let t = v.clamp(-1.0, 1.0) as f32;
    let accent = if t > 0.0 {
        palette::ORANGE
    } else {
        palette::SKY
    };
    palette::GRID.faded(0.5) + accent.faded(0.45 * t.abs())
}

/// The streamlines on a channel's axes.
fn streamlines(c: &mut Canvas, ax: &Axes, s: &Shown, l: Light) {
    for &level in &s.levels {
        ax.stroke(c, &s.flat.level(&s.streamfunction, level), 0.9, l);
    }
}

/// The channel's walls and the obstacle's rim.
fn rim(c: &mut Canvas, ax: &Axes, s: &Shown) {
    ax.stroke(c, &s.flat.rim(), 1.4, palette::INK.faded(0.8));
}

/// The flow: the pressure on each face, the streamlines, and the tracers riding them.
fn draw_flow(c: &mut Canvas, rect: Rect, s: &Shown, t: f32) {
    let ax = Axes::fitting(
        rect.part(0.0, 0.0, 1.0, 0.92),
        s.flat.points.iter().copied(),
        1.02,
    );
    let pressure: Vec<f64> = s.flow.pressure.iter().map(|p| p.s()).collect();
    let extent = percentile(&pressure, 0.97);
    s.flat
        .shade(c, &ax, 2, |f, _| Some(diverging(pressure[f] / extent)));
    streamlines(c, &ax, s, palette::INK.faded(0.18));
    rim(c, &ax, s);
    let phase = f64::from((t / SECONDS).rem_euclid(1.0));
    for (index, path) in s.paths.iter().enumerate() {
        let end = *path.times.last().expect("a path");
        // Tracers a period apart, each path's offset by the golden ratio's share.
        let offset = (index as f64 * 0.618_033_988_75).fract();
        let first = -1;
        let last = (end / s.period).ceil() as i64 + 1;
        for k in first..=last {
            let arrival = (k as f64 + phase + offset) * s.period;
            if arrival < 0.0 || arrival > end {
                continue;
            }
            let from = path.times.partition_point(|&x| x < arrival - TRAIL);
            let to = path.times.partition_point(|&x| x <= arrival);
            if to > from + 1 {
                ax.polyline(c, &path.points[from..to], 1.6, palette::INK.faded(0.8));
            }
            if to > 0 {
                ax.scatter(c, &path.points[to - 1..to], Marker::Dot, 3.5, palette::INK);
            }
        }
    }
    label_under(
        c,
        rect,
        "TRACERS ON THE STREAMLINES OVER THE PRESSURE",
        16.0,
        0.6,
    );
}

/// The vorticity at the vertices, with the streamlines.
fn draw_vorticity(c: &mut Canvas, rect: Rect, s: &Shown) {
    let ax = Axes::fitting(
        rect.part(0.0, 0.0, 1.0, 0.86),
        s.flat.points.iter().copied(),
        1.02,
    );
    let values: Vec<f64> = s.flow.vorticity.iter().map(|w| w.e12()).collect();
    let extent = percentile(&values, 0.97);
    s.flat
        .shade_field(c, &ax, &values, |v| diverging(v / extent));
    streamlines(c, &ax, s, palette::INK.faded(0.3));
    rim(c, &ax, s);
    label_under(c, rect, "THE VORTICITY AT THE VERTICES", 14.0, 0.8);
}

/// The primal mesh and its circumcentric dual about one vertex.
fn draw_complex(c: &mut Canvas, rect: Rect, s: &Shown) {
    let (mesh, vertices, o) = (&s.channel.mesh, &s.channel.vertices, &s.operators);
    let target = point(COMPLEX_TARGET[0], COMPLEX_TARGET[1]);
    let rim = boundary_edges(mesh);
    let on_rim: Vec<usize> = rim
        .iter()
        .flat_map(|&e| mesh.simplex(1, e).to_vec())
        .collect();
    let vertex = (0..mesh.len(0))
        .filter(|v| !on_rim.contains(v))
        .min_by(|&a, &b| {
            (vertices.site(a) - target)
                .norm()
                .total_cmp(&(vertices.site(b) - target).norm())
        })
        .expect("an interior vertex");
    let centre = vertices.site(vertex);
    let star: Vec<usize> = (0..mesh.len(2))
        .filter(|&f| mesh.simplex(2, f).contains(&vertex))
        .collect();
    let reach = star
        .iter()
        .flat_map(|&f| mesh.simplex(2, f).to_vec())
        .map(|v| (vertices.site(v) - centre).norm())
        .fold(0.0, f64::max);
    let radius = (1.25 * reach) as f32;
    let middle = at(centre);
    let [cx, cy] = [middle.e20() / middle.e12(), middle.e01() / middle.e12()];
    let half = Point2::direction(radius, radius);
    let ax = Axes::fitting(
        rect.part(0.0, 0.0, 1.0, 0.86),
        [middle - half, middle + half],
        1.0,
    );
    ax.clip(c);
    let near = |p: Point2| {
        let [x, y] = p.to_euclidean();
        (x - cx).abs() < 1.6 * radius && (y - cy).abs() < 1.6 * radius
    };
    // The dual cell: the circumcentres of the faces at the vertex, in turn about it.
    let reference = at(o.circumcentres.site(star[0])) - middle;
    let mut cell: Vec<(f32, Point2)> = star
        .iter()
        .map(|&f| {
            let p = at(o.circumcentres.site(f));
            (
                reference
                    .angle_to(p - middle)
                    .rem_euclid(core::f32::consts::TAU),
                p,
            )
        })
        .collect();
    cell.sort_by(|a, b| a.0.total_cmp(&b.0));
    let cell: Vec<Point2> = cell.into_iter().map(|(_, p)| p).collect();
    ax.fill(c, &cell, palette::ORANGE, 0.22);
    let nearest = star
        .iter()
        .copied()
        .min_by(|&a, &b| {
            (o.circumcentres.site(a) - centre)
                .norm()
                .total_cmp(&(o.circumcentres.site(b) - centre).norm())
        })
        .expect("a face");
    let corners = mesh
        .simplex(2, nearest)
        .iter()
        .map(|&v| at(vertices.site(v)))
        .collect::<Vec<_>>();
    ax.fill(c, &corners, palette::SKY, 0.22);
    let segments = o.dual_segments();
    let mut primal = Vec::new();
    let mut dual = Vec::new();
    for (e, segment) in segments.iter().enumerate() {
        let ends = mesh.simplex(1, e);
        let [a, b] = [ends[0], ends[1]].map(|v| at(vertices.site(v)));
        if near(a) && near(b) {
            primal.push([a, b]);
            dual.push(segment.map(at));
        }
    }
    ax.stroke(c, &primal, 1.0, palette::SKY.faded(0.6));
    ax.stroke(c, &dual, 1.0, palette::ORANGE.faded(0.8));
    let visible: Vec<Point2> = (0..mesh.len(0))
        .map(|v| at(vertices.site(v)))
        .filter(|&p| near(p))
        .collect();
    ax.scatter(c, &visible, Marker::Dot, 2.5, palette::SKY);
    let centres: Vec<Point2> = o
        .circumcentres
        .iter()
        .map(at)
        .filter(|&p| near(p))
        .collect();
    ax.scatter(c, &centres, Marker::Dot, 2.5, palette::ORANGE);
    ax.scatter(c, &[middle], Marker::Dot, 8.0, palette::SKY);
    // The nearest face's longest edge from its midpoint, and its dual edge from its middle.
    let longest = mesh
        .faces(2, nearest)
        .iter()
        .map(|&(e, _)| e)
        .max_by(|&a, &b| o.edges.site(a).norm().total_cmp(&o.edges.site(b).norm()))
        .expect("an edge");
    let ends = mesh.simplex(1, longest);
    let mid = (vertices.site(ends[0]) + vertices.site(ends[1])) * 0.5;
    let reach = 0.3;
    let edge = o.edges.site(longest);
    ax.arrow(
        c,
        at(mid - edge * reach),
        at(mid + edge * reach),
        1.6,
        8.0,
        palette::SKY,
    );
    let crossing = o.dual_edges.site(longest);
    ax.arrow(
        c,
        at(mid - crossing * reach),
        at(mid + crossing * reach),
        1.6,
        8.0,
        palette::ORANGE,
    );
    c.unclip();
    label_under(c, rect, "A DUAL CELL; AN EDGE AND ITS DUAL", 14.0, 0.95);
}

fn draw(c: &mut Canvas, s: &Shown, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    draw_flow(c, Rect::new(12.0, h * 0.13, w - 12.0, h * 0.66), s, t);
    let bottom = Rect::new(12.0, h * 0.68, w - 12.0, h * 0.98);
    draw_vorticity(c, bottom.part(0.0, 0.0, 0.68, 1.0), s);
    draw_complex(c, bottom.part(0.7, 0.0, 1.0, 1.0), s);
    caption(
        c,
        "STOKES FLOW IN DISCRETE EXTERIOR CALCULUS, PAST AN OBSTACLE",
        "EDGE FLUXES BALANCE ON EVERY TRIANGLE; THEIR DUAL CIRCULATION SUMS TO VORTICITY, WHOSE DIFFERENCES THE PRESSURE BALANCES (VGA2D)",
    );
}

fn main() {
    run_with(Anim::new("dec stokes", SECONDS).size(960, 540), shown, draw);
}
