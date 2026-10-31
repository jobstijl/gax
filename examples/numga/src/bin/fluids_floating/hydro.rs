//! The math: the water's push on a closed surface of triangles, the floating height, the forces,
//! the stiffness against small tilts; the hull and the box; and the scenes.

use gax::Unit;
use gax::pga3d::{Forque, Line, Motor, Plane, Point, Scalar, Spin};
use gax_numga_examples::shared::mechanics_simplex as simplex;

/// A triangle: its corners, wound outward or all inward.
pub type Triangle = [Point; 3];
/// The energy of a small turn about a line through the origin: a form on spins. numga's is on
/// tilts, the turns about horizontal lines through the origin (`e23`, `e31`), which have no
/// PGA3D kind in gax; its tilt block is read off this form (`Stiffness::tilt`).
pub type Stiffness = Scalar<(Spin, Spin)>;

/// The still water surface, the plane `z = 0`.
pub fn water() -> Plane {
    Plane::new(0.0, 0.0, 1.0, 0.0)
}

/// The point at infinity upward: the water's complement.
pub fn up() -> Point {
    water().dual()
}

/// The vertical shift: the plane at infinity meeting the water.
pub fn heave() -> Line {
    Plane::infinity() ^ water()
}

/// The axis of heel through the origin, along the body: the meet of `y = 0` and `z = 0`.
pub fn heel_axis() -> Line {
    Plane::new(0.0, 1.0, 0.0, 0.0) ^ water()
}

/// The axis of trim through the origin, across the body: the meet of `z = 0` and `x = 0`.
pub fn trim_axis() -> Line {
    water() ^ Plane::new(1.0, 0.0, 0.0, 0.0)
}

/// The water surface `height` above the body's origin.
pub fn surface_at(height: f64) -> Plane {
    water() - Plane::infinity() * height
}

/// A heel and a trim as spins: the parts of the axes of heel and trim through the origin.
pub fn tilts() -> [Spin; 2] {
    [heel_axis().cast::<Spin>(), trim_axis().cast::<Spin>()]
}

/// The wet part of a triangle below a water surface, as two triangles wound as it is, either or
/// both of no area: with the corners ordered from deepest to highest, the deepest corner fanned
/// to the points where the edges leave the water.
pub fn wet(triangle: &Triangle, surface: Plane) -> [Triangle; 2] {
    let depth = |p: Point| -(surface & p).s();
    let mut order = [0, 1, 2];
    order.sort_by(|a, b| depth(triangle[*b]).total_cmp(&depth(triangle[*a])));
    let [deep, middle, high] = order.map(|k| triangle[k]);
    let [first, second, third] = order.map(|k| depth(triangle[k]));
    // The point of an edge where it leaves the water, kept within the edge.
    let leaving = |start: Point, end: Point, start_depth: f64, end_depth: f64| {
        let drop = start_depth - end_depth;
        let share = (start_depth / if drop == 0.0 { 1.0 } else { drop }).clamp(0.0, 1.0);
        start + (end - start) * share
    };
    let turning = if second < 0.0 {
        leaving(deep, middle, first, second)
    } else {
        middle
    };
    let rising = if second < 0.0 {
        turning
    } else if third < 0.0 {
        leaving(middle, high, second, third)
    } else {
        high
    };
    let closing = if third < 0.0 {
        leaving(deep, high, first, third)
    } else {
        high
    };
    let parts = [[deep, turning, rising], [deep, rising, closing]];
    // An ordering that is not a rotation of the corners reverses the winding.
    if order[1] == (order[0] + 1) % 3 {
        parts
    } else {
        parts.map(|[a, b, c]| [c, b, a])
    }
}

/// Points halfway between each corner and the centroid of a triangle: they integrate anything
/// quadratic over it exactly, each standing for a third of its area (three times the simplex's
/// lumped-mass weights, `shared/mechanics_simplex.rs`).
fn rule() -> Vec<Vec<f64>> {
    simplex::weights(3)
        .into_iter()
        .map(|row| row.into_iter().map(|w| 3.0 * w).collect())
        .collect()
}

/// The sign of the volume a closed surface encloses, by its winding: positive exactly when its
/// faces' planes face outward.
pub fn winding(triangles: &[Triangle]) -> f64 {
    let o = Point::origin();
    let volume: f64 = triangles
        .iter()
        .map(|[a, b, c]| (o & *a & *b & *c).s())
        .sum();
    volume.signum()
}

/// Sample points of the wet surface below a water surface, and the water's push at each per
/// unit of pressure: a line through the point, square to the triangle, against its outward
/// normal, with a third of the part's area.
pub fn pressing(triangles: &[Triangle], surface: Plane) -> Vec<(Point, Forque)> {
    let rule = rule();
    let sign = winding(triangles);
    let mut out = Vec::new();
    for t in triangles {
        for part in wet(t, surface) {
            // The plane through the part's corners has twice its area as its weight; times the
            // winding's sign it faces outward.
            let plane = (part[0] & part[1] & part[2]) * sign;
            for row in &rule {
                let sample = part[0] * row[0] + part[1] * row[1] + part[2] * row[2];
                // The line through the sample square to the part: a force, intrinsic
                // (numga's `Forque`), whatever the inner product's orientation makes it.
                let line: Forque = ((sample | plane) * (-1.0 / 6.0)).with_orientation();
                out.push((sample, line));
            }
        }
    }
    out
}

/// The water's push on the body below a water surface, the pressure lines summed over its wet
/// surface, each pressure the fluid's weight times the depth; and how the push grows as the
/// surface rises.
pub fn buoyancy(triangles: &[Triangle], surface: Plane, fluid_weight: f64) -> (Forque, Forque) {
    let pressed = pressing(triangles, surface);
    let push = pressed
        .iter()
        .map(|(p, l)| *l * (fluid_weight * -(surface & *p).s()))
        .sum();
    let growth = pressed.iter().map(|(_, l)| *l * fluid_weight).sum();
    (push, growth)
}

/// The height of the water surface over the body's origin at which the body, turned by a motor,
/// floats, its buoyancy balancing its weight: Newton's method from the given height.
pub fn afloat(
    triangles: &[Triangle],
    turn: Unit<Motor>,
    weight: f64,
    mut height: f64,
    fluid_weight: f64,
    steps: usize,
) -> f64 {
    for _ in 0..steps {
        let (push, growth) = buoyancy(triangles, turn << surface_at(height), fluid_weight);
        let lifting = turn << heave();
        height += (weight - (push & lifting).s()) / (growth & lifting).s();
    }
    height
}

/// The motion taking the body's frame to the world's, the water at the world's `water`: the
/// turn, then down by the height of the water surface over the body's origin.
pub fn placement(turn: Unit<Motor>, height: f64) -> Unit<Motor> {
    (heave() * (height / 2.0)).exp() * turn
}

/// The forces on the body, turned and floating at a height, in the body's frame: its buoyancy,
/// and its weight, a line straight down through its centre of gravity. Their sum is the
/// righting couple.
pub fn forces(
    triangles: &[Triangle],
    turn: Unit<Motor>,
    height: f64,
    weight: f64,
    centre: Point,
    fluid_weight: f64,
) -> (Forque, Forque) {
    let (push, _) = buoyancy(triangles, turn << surface_at(height), fluid_weight);
    (push, (centre & (turn << -up())) * weight)
}

/// The energy of a small turn of the upright body floating at a height, about lines through the
/// origin: in the body's frame the turn tilts the water surface against the body, deepening each
/// wet point by its height above the change, and turns the weight with it; the change of the net
/// force, paired with the turn.
pub fn stiffness(
    triangles: &[Triangle],
    height: f64,
    weight: f64,
    centre: Point,
    fluid_weight: f64,
) -> Stiffness {
    let surface = surface_at(height);
    let tilted = surface.commutator(Spin::slot());
    let pushing: Forque<(Spin,)> = pressing(triangles, surface)
        .iter()
        .map(|(p, l)| (tilted & *p) * (-fluid_weight) * *l)
        .sum();
    let weighing = (centre & (-up()).commutator(Spin::slot())) * weight;
    Spin::slot() & (pushing + weighing)
}

/// The tilt block of a stiffness: its values on heel and trim, `[[heel, heel], [heel, trim]],
/// [[trim, heel], [trim, trim]]`.
pub fn tilt_block(k: Stiffness) -> [[f64; 2]; 2] {
    let t = tilts();
    [0, 1].map(|i| [0, 1].map(|j| k.of(t[i]).of(t[j]).s()))
}

/// The eigenvalues of a symmetric 2 × 2 block, ascending: half the trace, less and plus the
/// distance of the diagonal's half difference and the corner (gax has no 2 × 2 kind here).
#[cfg(test)] // numga's check of the metacentric heights
pub fn eigenvalues([[a, b], [_, d]]: [[f64; 2]; 2]) -> [f64; 2] {
    let mean = (a + d) / 2.0;
    let spread = gax::vga2d::Vector::new((a - d) / 2.0, b).norm();
    [mean - spread, mean + spread]
}

// --- shapes

/// The triangles of a grid of vertices, two per cell, wound from `u` towards `v`; `None` where a
/// vertex is missing.
fn patch(grid: &[Vec<Option<usize>>]) -> Vec<[Option<usize>; 3]> {
    let (nu, nv) = (grid.len(), grid[0].len());
    let mut first = Vec::new();
    let mut second = Vec::new();
    for i in 0..nu - 1 {
        for j in 0..nv - 1 {
            let (corner, along_u, across, along_v) = (
                grid[i][j],
                grid[i + 1][j],
                grid[i + 1][j + 1],
                grid[i][j + 1],
            );
            first.push([corner, along_u, across]);
            second.push([corner, across, along_v]);
        }
    }
    first.extend(second);
    first
}

/// A closed body: its vertices and its triangles, wound outward.
#[derive(Clone, Debug)]
pub struct Body {
    pub vertices: Vec<Point>,
    pub faces: Vec<[usize; 3]>,
}

impl Body {
    pub fn triangles(&self) -> Vec<Triangle> {
        self.faces
            .iter()
            .map(|f| f.map(|v| self.vertices[v]))
            .collect()
    }
}

/// A box of the given sizes along x, y and z about the origin, two triangles on each face.
pub fn cuboid(sizes: [f64; 3]) -> Body {
    // Each corner a choice of side along x, y and z, its index the choices read in binary.
    let vertices = (0..8)
        .map(|k| {
            let side = |axis: usize| f64::from((k >> (2 - axis)) & 1) - 0.5;
            Point::xyz(side(0) * sizes[0], side(1) * sizes[1], side(2) * sizes[2])
        })
        .collect();
    let mut faces = Vec::new();
    for axis in 0..3 {
        for side in 0..2 {
            // The face's u and v run along the next two axes, swapped on the negative side, so
            // that every face winds outward.
            let (first, second) = ((axis + 1) % 3, (axis + 2) % 3);
            let index = |u: usize, v: usize| {
                let mut bits = [0; 3];
                bits[axis] = side;
                (bits[first], bits[second]) = if side == 1 { (u, v) } else { (v, u) };
                Some(bits[0] * 4 + bits[1] * 2 + bits[2])
            };
            let grid: Vec<Vec<Option<usize>>> = (0..2)
                .map(|u| (0..2).map(|v| index(u, v)).collect())
                .collect();
            faces.extend(
                patch(&grid)
                    .into_iter()
                    .map(|f| f.map(|v| v.expect("a corner"))),
            );
        }
    }
    Body { vertices, faces }
}

/// A Wigley hull: below the waterline its half breadth is `beam / 2 (1 - (2 x / length)²)
/// (1 - (z / draft)²)`, above it the sides rise straight to a flat deck. Its vertices lie on
/// grids along the hull and up each side, and along the hull and across the deck.
pub fn wigley(length: f64, beam: f64, draft: f64, freeboard: f64, cells: usize) -> Body {
    let lin = gax_numga_examples::signal::linspace;
    let along = lin(-length / 2.0, length / 2.0, 2 * cells + 1);
    let mut heights = lin(-draft, 0.0, cells + 1);
    heights.extend(lin(0.0, freeboard, cells / 2 + 1).into_iter().skip(1));
    let across = lin(-1.0, 1.0, cells + 1);
    let (na, nh, nc) = (along.len(), heights.len(), across.len());
    let half = |x: f64, z: f64| {
        beam / 2.0 * (1.0 - (2.0 * x / length).powi(2)) * (1.0 - (z.min(0.0) / draft).powi(2))
    };
    let mut vertices = Vec::new();
    let mut add = |p: Point| {
        vertices.push(p);
        Some(vertices.len() - 1)
    };
    // The port side shares the keel and the stems with the starboard side, where the half
    // breadth vanishes; the deck shares its edges with the sides' tops and closes to the top of
    // each stem. The corner of the keel and the stern's stem is no vertex: the grid's triangles
    // there form a fin of no thickness, which the closed hull leaves out.
    let mut starboard = vec![vec![None; nh]; na];
    for i in 0..na {
        for j in 0..nh {
            if !(i == na - 1 && j == 0) {
                starboard[i][j] = add(Point::xyz(along[i], half(along[i], heights[j]), heights[j]));
            }
        }
    }
    let mut port = starboard.clone();
    for i in 1..na - 1 {
        for j in 1..nh {
            port[i][j] = add(Point::xyz(
                along[i],
                -half(along[i], heights[j]),
                heights[j],
            ));
        }
    }
    let mut deck = vec![vec![None; nc]; na];
    for i in 0..na {
        deck[i][0] = port[i][nh - 1];
        deck[i][nc - 1] = starboard[i][nh - 1];
    }
    // The bow's and the stern's rows close to the top of each stem.
    deck[0] = vec![starboard[0][nh - 1]; nc];
    deck[na - 1] = vec![starboard[na - 1][nh - 1]; nc];
    for i in 1..na - 1 {
        for k in 1..nc - 1 {
            let x = along[i];
            let y = across[k] * beam / 2.0 * (1.0 - (2.0 * x / length).powi(2));
            deck[i][k] = add(Point::xyz(x, y, freeboard));
        }
    }
    let transposed: Vec<Vec<Option<usize>>> = (0..nh)
        .map(|j| (0..na).map(|i| starboard[i][j]).collect())
        .collect();
    let mut faces = patch(&transposed);
    faces.extend(patch(&port));
    faces.extend(patch(&deck));
    // The closed hull's triangles: not on the fin, and not collapsed into the top of a stem.
    let faces = faces
        .into_iter()
        .filter_map(|[a, b, c]| match (a, b, c) {
            (Some(a), Some(b), Some(c)) if a != b && b != c && c != a => Some([a, b, c]),
            _ => None,
        })
        .collect();
    Body { vertices, faces }
}

// --- scenes

/// Fresh water's weight per cubic metre (N), and gravity.
pub const FLUID_WEIGHT: f64 = 9810.0;
pub const GRAVITY: f64 = 9.81;
/// The hull's length, beam, draft and freeboard (m), and its cells.
pub const LENGTH: f64 = 4.0;
pub const BEAM: f64 = 1.2;
pub const DRAFT: f64 = 0.5;
pub const FREEBOARD: f64 = 0.4;
pub const CELLS: usize = 8;
/// The heights of the hull's centre of gravity over its waterline, low ballast to high load.
pub const CENTRE_HEIGHTS: [f64; 5] = [-0.2, -0.1, 0.0, 0.1, 0.2];
/// The heels of the righting arm curves (degrees), every two from upright to upside down.
pub const HEEL_COUNT: usize = 91;
/// The cube's heels (degrees), past a quarter turn either side, and its densities.
pub const CUBE_HEEL_COUNT: usize = 56;
pub const DENSITY_COUNT: usize = 49;
pub const SHOWN_DENSITIES: [f64; 5] = [0.1, 0.23, 0.5, 0.77, 0.9];
/// Newton steps for the floating height, from upright and from the previous step.
pub const NEWTON_STEPS: usize = 6;
pub const WARM_STEPS: usize = 1;
/// The released hull: its loads, its heel at release, its radius of gyration about the axis of
/// heel as a share of the beam, the damping of its roll per second, the time step, the frames,
/// and the steps per frame.
pub const RELEASED: [usize; 2] = [1, 4];
pub const RELEASE_HEEL: f64 = 30.0;
pub const GYRATION: f64 = 0.4;
pub const DAMPING: f64 = 0.4;
pub const STEP: f64 = 0.03;
pub const FRAMES: usize = 100;
pub const STEPS_PER_FRAME: usize = 4;

pub fn heels() -> Vec<f64> {
    gax_numga_examples::signal::linspace(0.0, 180.0, HEEL_COUNT)
}
pub fn cube_heels() -> Vec<f64> {
    gax_numga_examples::signal::linspace(-10.0, 100.0, CUBE_HEEL_COUNT)
}
pub fn densities() -> Vec<f64> {
    gax_numga_examples::signal::linspace(0.02, 0.98, DENSITY_COUNT)
}

/// The turn heeling a body by an angle in degrees, about its axis of heel.
pub fn heeled(degrees: f64) -> Unit<Motor> {
    (heel_axis() * (-degrees.to_radians() / 2.0)).exp()
}

/// The hull, its weight (the water it displaces upright at its waterline) and its centres of
/// gravity.
pub fn ship() -> (Body, f64, Vec<Point>) {
    let body = wigley(LENGTH, BEAM, DRAFT, FREEBOARD, CELLS);
    let (upright, _) = buoyancy(&body.triangles(), water(), FLUID_WEIGHT);
    let centres = CENTRE_HEIGHTS
        .iter()
        .map(|h| Point::xyz(0.0, 0.0, *h))
        .collect();
    (body, (upright & heave()).s(), centres)
}

/// The righting couple with each load at each heel `[heels][loads]`, and the stiffness upright
/// with each load. Every load weighs the same, so the hull floats at one height per heel.
pub fn righting_curves(
    hull: &[Triangle],
    weight: f64,
    centres: &[Point],
) -> (Vec<Vec<Forque>>, Vec<Stiffness>) {
    let couples = heels()
        .iter()
        .map(|h| {
            let turn = heeled(*h);
            let height = afloat(hull, turn, weight, 0.0, FLUID_WEIGHT, NEWTON_STEPS);
            centres
                .iter()
                .map(|c| {
                    let (push, weighing) = forces(hull, turn, height, weight, *c, FLUID_WEIGHT);
                    push + weighing
                })
                .collect()
        })
        .collect();
    let stiffnesses = centres
        .iter()
        .map(|c| stiffness(hull, 0.0, weight, *c, FLUID_WEIGHT))
        .collect();
    (couples, stiffnesses)
}

/// The righting arm of a couple: its moment about the axis of heel, against the heel, per weight.
pub fn arm(couple: Forque, weight: f64) -> f64 {
    -(couple & heel_axis()).s() / weight
}

/// The unit cube's righting couple at each density and heel `[densities][heels]`, its centre of
/// gravity at its middle.
pub fn cube_couples(cube: &[Triangle], densities: &[f64]) -> Vec<Vec<Forque>> {
    densities
        .iter()
        .map(|d| {
            let weight = FLUID_WEIGHT * d;
            cube_heels()
                .iter()
                .map(|h| {
                    let turn = heeled(*h);
                    let height = afloat(cube, turn, weight, 0.0, FLUID_WEIGHT, NEWTON_STEPS);
                    let (push, weighing) =
                        forces(cube, turn, height, weight, Point::origin(), FLUID_WEIGHT);
                    push + weighing
                })
                .collect()
        })
        .collect()
}

/// The smallest heel from upright at which the cube rests stably, for a row of couples over the
/// cube's heels: where the couple turns from pushing it over to pushing it back, between the two
/// heels in proportion to the couples there.
pub fn stable_heel(couples: &[Forque]) -> f64 {
    let heels = cube_heels();
    let arms: Vec<f64> = couples.iter().map(|c| -(*c & heel_axis()).s()).collect();
    for k in 0..arms.len() - 1 {
        let (below, above) = (arms[k], arms[k + 1]);
        let share = -below / if above == below { 1.0 } else { above - below };
        let rest = heels[k] + share * (heels[k + 1] - heels[k]);
        if below < 0.0 && above >= 0.0 && rest > -1e-6 {
            return rest;
        }
    }
    heels[0]
}

/// The placement of a cube of a density, floating at a heel.
pub fn resting(cube: &[Triangle], density: f64, heel: f64) -> Unit<Motor> {
    let turn = heeled(heel);
    placement(
        turn,
        afloat(
            cube,
            turn,
            FLUID_WEIGHT * density,
            0.0,
            FLUID_WEIGHT,
            NEWTON_STEPS,
        ),
    )
}

/// A frame of the released hull: its roll angle (radians), placement and buoyancy (its weight
/// is a line down through its centre of gravity).
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub roll: f64,
    pub placed: Unit<Motor>,
    pub push: Forque,
}

/// The hull released from a heel with a load: at every frame its roll, placement and buoyancy. It turns about its axis of heel, floating at each moment at the height where its
/// buoyancy carries its weight, its roll driven by the righting couple and damped.
pub fn released(hull: &[Triangle], weight: f64, centre: Point) -> Vec<Frame> {
    let inertia = weight / GRAVITY * (GYRATION * BEAM).powi(2);
    // The roll and its rate are turns about the axis of heel, scaled by the angle and the rate.
    let mut roll = heel_axis() * RELEASE_HEEL.to_radians();
    let mut rate = heel_axis() * 0.0;
    let mut turn = (roll * -0.5).exp();
    let mut height = afloat(hull, turn, weight, 0.0, FLUID_WEIGHT, NEWTON_STEPS);
    let (mut push, mut weighing) = forces(hull, turn, height, weight, centre, FLUID_WEIGHT);
    let mut frames = Vec::new();
    for _ in 0..FRAMES {
        let angle = (roll & heel_axis().dual()).s();
        frames.push(Frame {
            roll: angle,
            placed: placement(turn, height),
            push,
        });
        for _ in 0..STEPS_PER_FRAME {
            // The righting couple about the axis of heel turns the roll, against its damping.
            let couple = ((push + weighing) & heel_axis()).s() / inertia;
            rate += (heel_axis() * couple - rate * DAMPING) * STEP;
            roll += rate * STEP;
            turn = (roll * -0.5).exp();
            height = afloat(hull, turn, weight, height, FLUID_WEIGHT, WARM_STEPS);
            (push, weighing) = forces(hull, turn, height, weight, centre, FLUID_WEIGHT);
        }
    }
    frames
}
