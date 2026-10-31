//! numga's `mechanics/odd_elasticity`: active bonds that push sideways when stretched, the work
//! they deliver around closed strain cycles, and the overdamped waves of a lattice of them
//! (VGA2D).
//!
//! Positions and displacements are directions, antivectors: in the plane the Hodge duals of
//! vectors, intrinsic. A force is a vector, extrinsic, read by its pairing with a displacement,
//! the work `force & displacement`. A bond reads its extension with the unit force along it and
//! answers with a force that has a sideways part, the unit force turned in the actuation plane;
//! its response, `Force <- Displacement`, is not symmetric, so it does net work around a closed
//! path: the activity times the area the path encloses, oriented by the plane. The adjugate of a
//! map of displacements carries forces back through it with their work kept, from the bonds to
//! the deformations that move them (two shear amplitudes) and to the particles at their ends.
//!
//! The animation shows a single bond whose free end runs around an ellipse, the area swept
//! shaded; a patch of the lattice driven around a circle of shear amplitudes, the loads on its
//! boundary; the work both deliver along their loops; and a wider lattice released from a
//! localized displacement, where drag alone meets the forces and the odd response turns the
//! relaxation into travelling, decaying waves (bonds coloured by their extension, one particle's
//! path traced).

use gax_numga_examples::shared::simplicial;
use gax_numga_examples::signal::frame;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, Point2, Rect, backdrop, caption, label_above,
    palette, run_with,
};

mod odd {
    use gax::field::{Field, SparseMap};
    use gax::mesh::Topology;
    use gax::vga2d;

    /// A scalar: a work, an extension, a drag (numga's `Scalar+`).
    pub type Scalar = vga2d::Scalar;
    /// A position or a displacement, the Hodge dual of its vector (numga's `Antivector-`).
    pub type Displacement<S = ()> = vga2d::intrinsic::Vector<S>;
    /// A force, read by its pairing with a displacement (numga's `Vector+`).
    pub type Force<S = ()> = vga2d::Vector<S>;
    /// The plane the sideways push turns in (numga's `Bivector+`).
    pub type Bivector = vga2d::Bivector;
    /// The two amplitudes of a shear, geometric like displacements.
    pub type Shear = Displacement;
    /// A response: displacements to forces.
    pub type Response = Force<(Displacement,)>;
    /// A map of displacements (a deformation, a lattice's evolution).
    pub type Motion = Displacement<(Displacement,)>;

    /// The point at `(x, y)`: the Hodge dual of its vector (numga's `x * mv.x.dual() + ...`).
    pub fn point(x: f64, y: f64) -> Displacement {
        Force::new(x, y).hodge()
    }

    /// The unit displacements along the two axes, which the unit forces along them read as one.
    pub fn x() -> Displacement {
        point(1.0, 0.0)
    }
    pub fn y() -> Displacement {
        point(0.0, 1.0)
    }

    /// A position's coordinates, read by pairing it with the unit forces along the axes.
    pub fn coordinates(p: Displacement) -> [f64; 2] {
        [
            (Force::new(1.0, 0.0) & p).s(),
            (Force::new(0.0, 1.0) & p).s(),
        ]
    }

    /// The unit force along a unit direction, which reads a displacement's component along it:
    /// the inverse of the Hodge dual (numga's `dual_inverse`; in the plane's Euclidean metric
    /// gax's metric-free `undual`).
    pub fn force_along(direction: Displacement) -> Force {
        direction.undual()
    }

    /// A triangular lattice with free edges: its triangles, whose edges are the bonds, and its
    /// positions, row by row. Between two rows each cell holds two triangles, counter-clockwise;
    /// the rows alternate in being shifted half a spacing to the right, which decides the
    /// diagonal the two triangles share.
    pub fn triangular_patch(
        n_columns: usize,
        n_rows: usize,
        spacing: f64,
    ) -> (Topology<3>, Field<Displacement>) {
        // A row up is the direction at sixty degrees less half a step across.
        let up = (Force::from_angle(core::f64::consts::PI / 3.0) - Force::new(0.5, 0.0)).hodge();
        let half = |n: usize| (n as f64 - 1.0) / 2.0;
        let positions = (0..n_rows * n_columns)
            .map(|k| {
                let (row, column) = (k / n_columns, k % n_columns);
                let across = column as f64 + (row % 2) as f64 / 2.0 - half(n_columns);
                (x() * across + up * (row as f64 - half(n_rows))) * spacing
            })
            .collect();
        let index = |row: usize, column: usize| row * n_columns + column;
        let mut triangles = Vec::new();
        for row in 0..n_rows - 1 {
            for column in 0..n_columns - 1 {
                let (lower_left, lower_right) = (index(row, column), index(row, column + 1));
                let (upper_left, upper_right) =
                    (index(row + 1, column), index(row + 1, column + 1));
                if row % 2 == 1 {
                    triangles.push([lower_left, upper_right, upper_left]);
                    triangles.push([lower_left, lower_right, upper_right]);
                } else {
                    triangles.push([lower_left, lower_right, upper_left]);
                    triangles.push([lower_right, upper_right, upper_left]);
                }
            }
        }
        let mesh = Topology::new(n_rows * n_columns, &triangles).expect("a triangle mesh");
        (mesh, positions)
    }

    /// The restoring and sideways force caused by a bond's endpoint displacement: the unit force
    /// along the bond reads the displacement's component along it, and the answer pulls back
    /// along the bond and pushes across it, turned in the actuation plane.
    pub fn bond_response(
        direction: Displacement,
        actuation_plane: Bivector,
        stiffness: f64,
        activity: f64,
    ) -> Response {
        let unit_force = force_along(direction);
        let push = unit_force * -stiffness + (unit_force | actuation_plane) * activity;
        push * (unit_force & Displacement::slot())
    }

    /// Displacement, force and work delivered along consecutive straight path segments: a
    /// linear force's work along a straight segment is its midpoint force paired with the
    /// segment's displacement.
    pub fn integrate_cycle(
        response: Response,
        cycle: &[Displacement],
    ) -> Vec<(Displacement, Force, Scalar)> {
        let mut work = Scalar::new(0.0);
        let mut states = vec![(cycle[0], response.of(cycle[0]), work)];
        for pair in cycle.windows(2) {
            let (start, end) = (pair[0], pair[1]);
            work += response.of((start + end) * 0.5) & (end - start);
            states.push((end, response.of(end), work));
        }
        states
    }

    /// The particles' velocities from their displacements, overdamped: no inertia, every force
    /// met by drag. Relative endpoint motion goes to each bond; its adjugate returns equal and
    /// opposite forces to the bond's ends; drag turns each particle's force into its velocity,
    /// along the force's metric direction, its dual.
    pub fn evolution(
        incidence: &SparseMap<Scalar>,
        response: &Field<Response>,
        drags: &Field<Scalar>,
    ) -> SparseMap<Motion> {
        let gradient = incidence * Displacement::slot();
        let mobility =
            SparseMap::from_diagonal(&drags.map(|d| Force::slot().hodge() * (1.0 / d.s())));
        let forces = gradient
            .adjugate()
            .compose(&SparseMap::from_diagonal(response).compose(&gradient));
        mobility.compose(&forces)
    }

    /// One fourth-order Runge–Kutta step of an overdamped displacement field.
    pub fn rk4_step(
        generator: &SparseMap<Motion>,
        displacement: &Field<Displacement>,
        dt: f64,
    ) -> Field<Displacement> {
        let first = generator.of(displacement);
        let second = generator.of(&(displacement + &first.scaled(dt / 2.0)));
        let third = generator.of(&(displacement + &second.scaled(dt / 2.0)));
        let fourth = generator.of(&(displacement + &third.scaled(dt)));
        let middle = (&second + &third).scaled(2.0);
        displacement + &(&(&first + &middle) + &fourth).scaled(dt / 6.0)
    }

    /// Each bond's extension: its relative endpoint displacement read along it.
    pub fn extensions(
        incidence: &SparseMap<Scalar>,
        bond_directions: &Field<Displacement>,
        displacement: &Field<Displacement>,
    ) -> Vec<f64> {
        let relative = incidence * displacement;
        bond_directions
            .iter()
            .zip(relative.iter())
            .map(|(d, r)| (force_along(d) & r).s())
            .collect()
    }

    /// The moving lattice and bond extensions during its free overdamped response,
    /// `steps_per_frame` steps per frame.
    #[allow(clippy::too_many_arguments)] // numga's signature
    pub fn wave_states(
        reference: &Field<Displacement>,
        incidence: &SparseMap<Scalar>,
        bond_directions: &Field<Displacement>,
        generator: &SparseMap<Motion>,
        initial: Field<Displacement>,
        frames: usize,
        steps_per_frame: usize,
        dt: f64,
    ) -> Vec<(Field<Displacement>, Vec<f64>)> {
        let mut displacement = initial;
        let mut states = Vec::with_capacity(frames);
        for _ in 0..frames {
            let stretched = extensions(incidence, bond_directions, &displacement);
            states.push((reference + &displacement, stretched));
            for _ in 0..steps_per_frame {
                displacement = rk4_step(generator, &displacement, dt);
            }
        }
        states
    }

    // --- scenes

    /// The bonds of the two work cycles: their stiffness, their activity, and the plane their
    /// sideways push turns in (`e12`).
    pub const STIFFNESS: f64 = 1.0;
    pub const ACTIVITY: f64 = 3.0;
    pub fn actuation_plane() -> Bivector {
        Bivector::new(1.0)
    }
    pub const CYCLE_STEPS: usize = 126;
    /// One bond: its rest length, and the radii of the ellipse its free end follows, along and
    /// across it.
    pub const REST_LENGTH: f64 = 1.0;
    pub const PARALLEL_RADIUS: f64 = 0.25;
    pub const TRANSVERSE_RADIUS: f64 = 0.18;
    /// A small patch sheared around a circle of shear amplitudes.
    pub const SPACING: f64 = 1.0;
    pub const SHEAR_AMPLITUDE: f64 = 0.14;
    pub const PATCH_COLUMNS: usize = 7;
    pub const PATCH_ROWS: usize = 7;
    /// A wider lattice, released from a localized displacement: its bonds, the drag on every
    /// particle, the displacement's amplitude, width and wavenumber, and the frames, the steps
    /// per frame and the time step.
    pub const WAVE_STIFFNESS: f64 = 0.2;
    pub const WAVE_ACTIVITY: f64 = 4.0;
    pub const DRAG: f64 = 1.0;
    pub const WAVE_AMPLITUDE: f64 = 0.15;
    pub const WAVE_WIDTH: f64 = 2.5;
    pub const WAVE_NUMBER: f64 = 1.2;
    pub const WAVE_COLUMNS: usize = 21;
    pub const WAVE_ROWS: usize = 11;
    pub const WAVE_FRAMES: usize = 180;
    pub const WAVE_STEPS_PER_FRAME: usize = 2;
    pub const TIME_STEP: f64 = 1.0 / 60.0;

    /// The turn by `angle` in the actuation plane.
    pub fn turned(direction: Displacement, angle: f64) -> Displacement {
        (actuation_plane() * (-angle / 2.0)).exp() >> direction
    }

    /// A unit direction turned once around the actuation plane, `steps + 1` of them.
    pub fn turns(steps: usize) -> Vec<Displacement> {
        gax_numga_examples::signal::linspace(0.0, core::f64::consts::TAU, steps + 1)
            .into_iter()
            .map(|phase| turned(x(), phase))
            .collect()
    }

    /// The sum of the triangles a closed polygon spans from the origin: its oriented area.
    pub fn enclosed(path: &[Displacement]) -> Bivector {
        path.windows(2).map(|w| w[0] ^ w[1]).sum::<Bivector>() * 0.5
    }

    /// An active bond as its endpoint follows an ellipse: its anchor, its rest end, the path
    /// of its end, and the end, force and work at each step.
    pub struct BondCycle {
        pub anchor: Displacement,
        pub reference: Displacement,
        pub path: Vec<Displacement>,
        pub states: Vec<(Displacement, Force, Scalar)>,
    }

    pub fn bond_cycle(steps: usize) -> BondCycle {
        let direction = x();
        let anchor = direction * 0.0;
        let reference = direction * REST_LENGTH;
        let response = bond_response(direction, actuation_plane(), STIFFNESS, ACTIVITY);
        let across = direction | actuation_plane();
        // The ellipse stretches the unit circle by its two radii, along the bond and across it.
        let open = Displacement::slot;
        let ellipse: Motion = direction * (force_along(direction) & open()) * PARALLEL_RADIUS
            + across * (force_along(across) & open()) * TRANSVERSE_RADIUS;
        let cycle: Vec<Displacement> = turns(steps).into_iter().map(|t| ellipse.of(t)).collect();
        let states = integrate_cycle(response, &cycle)
            .into_iter()
            .map(|(d, f, w)| (reference + d, f, w))
            .collect();
        BondCycle {
            anchor,
            reference,
            path: cycle.iter().map(|d| reference + *d).collect(),
            states,
        }
    }

    /// The deformation by two shear amplitudes: stretching along the axes, or along the
    /// diagonals between them, `Displacement <- (Shear, Displacement)`.
    pub fn strain() -> Displacement<(Shear, Displacement)> {
        let open = Displacement::slot;
        let (ex, ey) = (Force::new(1.0, 0.0), Force::new(0.0, 1.0));
        let horizontal: Motion = x() * (ex & open()) - y() * (ey & open());
        let diagonal: Motion = x() * (ey & open()) + y() * (ex & open());
        (ex & Shear::slot()) * horizontal + (ey & Shear::slot()) * diagonal
    }

    /// The forces that a strain cycle's bonds return to the shear amplitudes: each of the
    /// three bond families of a lattice cell moved by the shear, its force carried back by the
    /// adjugate of that motion with its work kept.
    pub fn shear_response(stiffness: f64, activity: f64, plane: Bivector) -> Force<(Shear,)> {
        let strain = strain();
        (0..3)
            .map(|k| {
                let direction = turned(x(), k as f64 * core::f64::consts::PI / 3.0);
                let response = bond_response(direction, plane, stiffness, activity);
                let motion = strain.at::<1>().of(direction * SPACING);
                motion.adjugate().of(response.of(motion))
            })
            .sum()
    }

    /// Two shear deformations drive a patch around a closed cycle; the work is per lattice
    /// cell. The positions and forces of the finite patch at each step, which reveal its
    /// boundary loads, and the work.
    pub struct ShearCycle {
        pub mesh: Topology<3>,
        pub reference: Field<Displacement>,
        pub states: Vec<(Field<Displacement>, Field<Force>, Scalar)>,
    }

    pub fn shear_cycle(steps: usize) -> ShearCycle {
        let (mesh, reference) = triangular_patch(PATCH_COLUMNS, PATCH_ROWS, SPACING);
        let strain = strain();
        let response = shear_response(STIFFNESS, ACTIVITY, actuation_plane());
        // Summing the same bond forces over the finite patch reveals its boundary loads.
        let incidence = mesh.d::<Scalar>(0);
        let gradient = &incidence * Displacement::slot();
        let edge_response = (&incidence * &reference)
            .map(|b| bond_response(b.normalized(), actuation_plane(), STIFFNESS, ACTIVITY));
        let patch_response = gradient
            .adjugate()
            .compose(&SparseMap::from_diagonal(&edge_response).compose(&gradient));
        let patch_motion = reference.map(|r| strain.at::<1>().of(r));
        let patch_forces: Field<Force<(Shear,)>> = patch_response.of(&patch_motion);
        let cycle: Vec<Shear> = turns(steps).iter().map(|t| *t * SHEAR_AMPLITUDE).collect();
        let states = integrate_cycle(response, &cycle)
            .into_iter()
            .map(|(shear, _, work)| {
                let moved = &reference + &patch_motion.map(|m| m.of(shear));
                (moved, patch_forces.map(|f| f.of(shear)), work)
            })
            .collect();
        ShearCycle {
            mesh,
            reference,
            states,
        }
    }

    /// A localized displacement excites damped waves in a free triangular patch: the lattice,
    /// its rest positions, and its positions and bond extensions at each frame.
    pub struct Wave {
        pub mesh: Topology<3>,
        pub reference: Field<Displacement>,
        pub frames: Vec<(Field<Displacement>, Vec<f64>)>,
    }

    pub fn lattice_wave(frames: usize, steps_per_frame: usize, dt: f64) -> Wave {
        let (mesh, reference) = triangular_patch(WAVE_COLUMNS, WAVE_ROWS, SPACING);
        let incidence = mesh.d::<Scalar>(0);
        let bond_directions = (&incidence * &reference).map(|b| b.normalized());
        let response = bond_directions
            .map(|d| bond_response(d, actuation_plane(), WAVE_STIFFNESS, WAVE_ACTIVITY));
        let drags = Field::from_fn(reference.len(), |_| Scalar::new(DRAG));
        let generator = evolution(&incidence, &response, &drags);
        let initial = reference.map(|r| {
            let envelope = (-r.norm_squared() / (2.0 * WAVE_WIDTH * WAVE_WIDTH)).exp();
            // The cosine of the phase along x: the unit direction at that angle, across.
            let phase = WAVE_NUMBER * coordinates(r)[0];
            x() * (WAVE_AMPLITUDE * envelope * Force::from_angle(phase).e1())
        });
        let frames = wave_states(
            &reference,
            &incidence,
            &bond_directions,
            &generator,
            initial,
            frames,
            steps_per_frame,
            dt,
        );
        Wave {
            mesh,
            reference,
            frames,
        }
    }
}

use odd::*;

/// The loop: the wave's 180 frames of 40 ms, and two strain cycles in it.
const SECONDS: f32 = 7.2;
/// The force arrows' lengths per unit force: the bond's, the patch's boundary loads.
const FORCE_SCALE: f64 = 0.55;
const PATCH_FORCE_SCALE: f64 = 1.3;
/// The wave's motion, exaggerated; the extension that saturates the colours.
const DISPLACEMENT_SCALE: f64 = 4.0;
const EXTENSION_SCALE: f64 = 0.04;
/// The frames of the tracked particle's trail.
const TRAIL_FRAMES: usize = 33;

/// The three scenes, computed once.
struct Scenes {
    bond: BondCycle,
    shear: ShearCycle,
    wave: Wave,
    /// The work each loop should deliver: the activity times the area it encloses (three
    /// bonds per cell for the patch).
    expected: [f64; 2],
}

fn scenes() -> Scenes {
    let bond = bond_cycle(CYCLE_STEPS);
    let ellipse: Vec<Displacement> = bond.path.iter().map(|p| *p - bond.reference).collect();
    let shears: Vec<Shear> = turns(CYCLE_STEPS)
        .iter()
        .map(|t| *t * SHEAR_AMPLITUDE)
        .collect();
    let plane = actuation_plane();
    let expected = [
        -ACTIVITY * plane.scalar_product(enclosed(&ellipse)).s(),
        -3.0 * ACTIVITY * plane.scalar_product(enclosed(&shears)).s(),
    ];
    Scenes {
        bond,
        shear: shear_cycle(CYCLE_STEPS),
        wave: lattice_wave(WAVE_FRAMES, WAVE_STEPS_PER_FRAME, TIME_STEP),
        expected,
    }
}

/// A position, to draw.
fn at(p: Displacement) -> Point2 {
    let [x, y] = coordinates(p);
    Point2::xy(x as f32, y as f32)
}

/// A force drawn along its metric direction, its dual, from where it acts.
fn tip(from: Displacement, force: Force, scale: f64) -> Point2 {
    at(from + force.hodge() * scale)
}

/// The bonds of a lattice at its positions.
fn segments(mesh: &gax::mesh::Topology<3>, positions: &[Point2]) -> Vec<[Point2; 2]> {
    simplicial::edges(mesh)
        .iter()
        .map(|&[a, b]| [positions[a], positions[b]])
        .collect()
}

/// An endpoint follows its closed path, the area it has swept about its rest position shaded,
/// while the bond's force changes.
fn draw_bond(c: &mut Canvas, rect: Rect, b: &BondCycle, step: usize) {
    let ends: Vec<Point2> = b.states.iter().map(|s| at(s.0)).collect();
    let tips: Vec<Point2> = b
        .states
        .iter()
        .map(|s| tip(s.0, s.1, FORCE_SCALE))
        .collect();
    let (anchor, rest) = (at(b.anchor), at(b.reference));
    let path: Vec<Point2> = b.path.iter().map(|p| at(*p)).collect();
    let all = path.iter().chain(&tips).copied().chain([anchor, rest]);
    let ax = Axes::fitting(rect, all, 1.12);
    let mut swept = vec![rest];
    swept.extend_from_slice(&ends[..=step]);
    ax.fill(c, &swept, palette::ORANGE, 0.16);
    ax.dashed(c, &[anchor, rest], 1.4, 5.0, palette::INK.faded(0.3));
    ax.polyline(c, &path, 1.2, palette::INK.faded(0.3));
    ax.polyline(c, &ends[..=step], 1.4, palette::SKY.faded(0.55));
    let end = ends[step];
    ax.line(c, anchor, end, 4.0, palette::SKY);
    ax.scatter(c, &[anchor], Marker::Dot, 9.0, palette::INK);
    ax.scatter(c, &[rest], Marker::Cross, 7.0, palette::INK.faded(0.6));
    ax.scatter(c, &[end], Marker::Dot, 9.0, palette::SKY.whitened(0.3));
    ax.arrow(c, end, tips[step], 2.2, 8.0, palette::ORANGE.whitened(0.2));
}

/// A deforming patch pushes on its boundary around its closed cycle.
fn draw_patch(c: &mut Canvas, rect: Rect, s: &ShearCycle, step: usize) {
    let rest: Vec<Point2> = s.reference.iter().map(at).collect();
    let boundary = simplicial::exterior(&s.mesh, 0);
    let tips = |k: usize| -> Vec<Point2> {
        let (places, forces, _) = &s.states[k];
        boundary
            .iter()
            .map(|&i| tip(places.site(i), forces.site(i), PATCH_FORCE_SCALE))
            .collect()
    };
    let reach = s
        .states
        .iter()
        .enumerate()
        .flat_map(|(k, st)| st.0.iter().map(at).chain(tips(k)).collect::<Vec<_>>());
    let ax = Axes::fitting(rect, reach.chain(rest.iter().copied()), 1.08);
    ax.stroke(c, &segments(&s.mesh, &rest), 1.0, palette::INK.faded(0.2));
    let (places, _, _) = &s.states[step];
    let now: Vec<Point2> = places.iter().map(at).collect();
    ax.stroke(c, &segments(&s.mesh, &now), 1.6, palette::SKY);
    ax.scatter(c, &now, Marker::Dot, 4.0, palette::INK);
    for (&i, t) in boundary.iter().zip(tips(step)) {
        ax.arrow(c, now[i], t, 1.2, 5.0, palette::ORANGE);
    }
}

/// The work delivered along both loops, against the share of the loop, with its expected
/// total: the activity times the oriented area enclosed, three bonds per cell for the patch.
fn draw_work(c: &mut Canvas, rect: Rect, s: &Scenes, step: usize) {
    let works: [Vec<f64>; 2] = [
        s.bond.states.iter().map(|st| st.2.s()).collect(),
        s.shear.states.iter().map(|st| st.2.s()).collect(),
    ];
    let top = s.expected.iter().copied().fold(0.0, f64::max) as f32;
    let ax = Axes::new(rect, [0.0, 1.0], [-0.05 * top, 1.3 * top]);
    let n = CYCLE_STEPS as f32;
    let lights = [palette::SKY, palette::GREEN];
    let names = ["ONE BOND: A x ELLIPSE", "PATCH: 3 A x CIRCLE, PER CELL"];
    let size = (rect.height() / 14.0).clamp(6.0, 9.0);
    for (k, w) in works.iter().enumerate() {
        let curve: Vec<Point2> = w
            .iter()
            .enumerate()
            .map(|(i, v)| Point2::xy(i as f32 / n, *v as f32))
            .collect();
        let total = s.expected[k] as f32;
        let level = [Point2::xy(0.0, total), Point2::xy(1.0, total)];
        ax.dashed(c, &level, 1.0, 4.0, lights[k].faded(0.45));
        let note = Point2::xy(0.02, total + 0.05 * top);
        ax.text(c, note, names[k], size, lights[k].faded(0.8), Align::Left);
        ax.polyline(c, &curve[..=step], 1.6, lights[k]);
        ax.scatter(c, &[curve[step]], Marker::Dot, 6.0, lights[k].whitened(0.3));
    }
    ax.frame(c, "WORK DELIVERED ROUND THE LOOP", "SHARE OF THE LOOP", "");
}

/// The colour of a bond's extension: compressed blue, unstretched dim, stretched orange.
fn extension_light(e: f64) -> Light {
    let t = (e / EXTENSION_SCALE).clamp(-1.0, 1.0) as f32;
    let middle = palette::SKY.faded(0.35);
    if t < 0.0 {
        middle.mix_light(palette::BLUE, -t)
    } else {
        middle.mix_light(palette::ORANGE, t)
    }
}

/// Moving bonds coloured by signed extension, the motion from the rest positions exaggerated,
/// with one particle's short trajectory marked.
fn draw_wave(c: &mut Canvas, rect: Rect, w: &Wave, k: usize) {
    let rest: Vec<Point2> = w.reference.iter().map(at).collect();
    let ax = Axes::fitting(rect, rest.iter().copied(), 1.06);
    ax.stroke(c, &segments(&w.mesh, &rest), 0.7, palette::INK.faded(0.15));
    let exaggerated = |positions: &gax::field::Field<Displacement>| -> Vec<Point2> {
        positions
            .iter()
            .zip(w.reference.iter())
            .map(|(p, r)| at(r + (p - r) * DISPLACEMENT_SCALE))
            .collect()
    };
    let (positions, stretched) = &w.frames[k];
    let now = exaggerated(positions);
    // The bonds in bins of extension, each bin one stroke.
    let bins = 17;
    let edges = simplicial::edges(&w.mesh);
    let mut binned: Vec<Vec<[Point2; 2]>> = vec![Vec::new(); bins];
    for (&[a, b], e) in edges.iter().zip(stretched) {
        let t = (e / EXTENSION_SCALE).clamp(-1.0, 1.0);
        let bin = (((t + 1.0) / 2.0) * (bins - 1) as f64).round() as usize;
        binned[bin].push([now[a], now[b]]);
    }
    for (bin, segs) in binned.iter().enumerate() {
        let e = (bin as f64 / (bins - 1) as f64 * 2.0 - 1.0) * EXTENSION_SCALE;
        ax.stroke(c, segs, 1.3, extension_light(e));
    }
    ax.scatter(c, &now, Marker::Dot, 2.5, palette::INK.faded(0.7));
    let tracked = (WAVE_ROWS / 2) * WAVE_COLUMNS + WAVE_COLUMNS / 2;
    let trail: Vec<Point2> = w.frames[k.saturating_sub(TRAIL_FRAMES - 1)..=k]
        .iter()
        .map(|(p, _)| exaggerated(p)[tracked])
        .collect();
    ax.polyline(c, &trail, 2.0, palette::YELLOW);
    ax.scatter(c, &[now[tracked]], Marker::Dot, 7.0, palette::YELLOW);
}

fn draw(c: &mut Canvas, s: &Scenes, t: f32) {
    backdrop(c);
    let phase = (t / SECONDS).rem_euclid(1.0);
    let wave_frame = frame(phase, WAVE_FRAMES);
    // Two strain cycles per loop, every step and back at the start.
    let step = frame((2.0 * phase).rem_euclid(1.0), CYCLE_STEPS + 1);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let size = (h / 50.0).clamp(6.0, 11.0);
    let body = Rect::new(0.0, h * 0.13, w, h);
    let left = body.part(0.0, 0.0, 0.56, 1.0);
    let right = body.part(0.56, 0.0, 1.0, 1.0);
    let bond = left.part(0.0, 0.0, 1.0, 0.46).inset(16.0, 18.0, 8.0, 4.0);
    draw_bond(c, bond, &s.bond, step);
    label_above(
        c,
        bond,
        "ONE ACTIVE BOND, ITS END ROUND AN ELLIPSE",
        size,
        4.0,
    );
    let wave = left.part(0.0, 0.46, 1.0, 1.0).inset(16.0, 22.0, 8.0, 8.0);
    draw_wave(c, wave, &s.wave, wave_frame);
    label_above(c, wave, "RELEASED: OVERDAMPED ODD WAVES", size, 4.0);
    let patch = right.part(0.0, 0.0, 1.0, 0.5).inset(8.0, 18.0, 16.0, 4.0);
    draw_patch(c, patch, &s.shear, step);
    label_above(c, patch, "A PATCH SHEARED ROUND A CIRCLE", size, 4.0);
    let work = right.part(0.0, 0.5, 1.0, 1.0).inset(48.0, 30.0, 18.0, 34.0);
    draw_work(c, work, s, step);
    caption(
        c,
        "ODD ELASTICITY: BONDS THAT PUSH SIDEWAYS",
        "RESPONSE = (-K F + A F|E12) (F & D); WORK ROUND A LOOP = A x AREA; ADJUGATES CARRY FORCES BACK (VGA2D)",
    );
}

fn main() {
    run_with(
        Anim::new("odd_elasticity", SECONDS).size(960, 540),
        scenes,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::odd::*;
    use gax::field::{Field, SparseMap};
    use gax_numga_examples::rng::{Draw, rng};
    use gax_numga_examples::signal::linspace;
    use gax_numga_examples::testing::{close, near};

    /// numga's fixture: an ellipse of 48 sides off the origin.
    fn cycle() -> Vec<Displacement> {
        let open = Displacement::slot;
        let (ex, ey) = (Force::new(1.0, 0.0), Force::new(0.0, 1.0));
        let stretch: Motion = x() * (ex & open()) * 0.18 + y() * (ey & open()) * 0.11;
        linspace(0.0, core::f64::consts::TAU, 49)
            .into_iter()
            .map(|a| stretch.of(turned(x(), a)) + x() * 0.07 - y() * 0.03)
            .collect()
    }

    /// The work of a linear force around a polygon, each segment's midpoint force paired with
    /// it.
    fn loop_work(response: Response, cycle: &[Displacement]) -> f64 {
        cycle
            .windows(2)
            .map(|w| (response.of((w[0] + w[1]) * 0.5) & (w[1] - w[0])).s())
            .sum()
    }

    fn reversed(cycle: &[Displacement]) -> Vec<Displacement> {
        cycle.iter().rev().copied().collect()
    }

    /// The direction of `(a, b)`, normalized.
    fn unit(a: f64, b: f64) -> Displacement {
        (x() * a + y() * b).normalized()
    }

    #[test]
    fn the_axes_read_back() {
        assert_eq!(coordinates(point(0.3, -2.0)), [0.3, -2.0]);
        assert_eq!(force_along(x()), Force::new(1.0, 0.0));
        assert_eq!(force_along(y()).hodge(), y());
    }

    #[test]
    fn cycle_work_is_oriented_area_and_changes_sign_on_reversal() {
        let cycle = cycle();
        let activity = 0.8;
        let direction = unit(1.0, 0.3);
        let plane = actuation_plane();
        let response = bond_response(direction, plane, 0.9, activity);
        let passive = bond_response(direction, plane, 0.9, 0.0);
        let work = loop_work(response, &cycle);
        let area_work = -activity * plane.scalar_product(enclosed(&cycle)).s();
        let states = integrate_cycle(response, &cycle);
        assert!(close(work, area_work, 1e-12), "{work} {area_work}");
        assert!(close(loop_work(passive, &cycle), 0.0, 1e-12));
        assert!(close(loop_work(response, &reversed(&cycle)), -work, 1e-12));
        assert!(close(states[states.len() - 1].2.s(), area_work, 1e-12));
        assert!(work > 0.04, "{work}");
    }

    #[test]
    fn mirror_transports_bond_force_and_preserves_cycle_work() {
        let cycle = cycle();
        let direction = unit(1.0, 0.4);
        let mirror = Force::new(1.0, -0.2).normalized();
        let plane = actuation_plane();
        let response = bond_response(direction, plane, 0.9, 0.8);
        let reflected_response = bond_response(mirror >> direction, mirror >> plane, 0.9, 0.8);
        let reflected_cycle: Vec<Displacement> = cycle.iter().map(|d| mirror >> *d).collect();
        for (d, r) in cycle.iter().zip(&reflected_cycle) {
            assert!(near(
                reflected_response.of(*r),
                mirror >> response.of(*d),
                1e-12
            ));
        }
        let (a, b) = (
            loop_work(reflected_response, &reflected_cycle),
            loop_work(response, &cycle),
        );
        assert!(close(a, b, 1e-12), "{a} {b}");
    }

    #[test]
    fn three_bond_families_pull_back_to_an_odd_shear_response() {
        let cycle = cycle();
        let (stiffness, activity) = (0.9, 0.8);
        let plane = actuation_plane();
        let shear = shear_response(stiffness, activity, plane);
        let along = Displacement::slot().undual();
        let expected = along * (-1.5 * stiffness) + (along | plane) * (1.5 * activity);
        assert!(near(shear, expected, 1e-12));
        // The work of the three bonds, each moved by the shear cycle, is the shear response's.
        let strain = strain();
        let bond_work: f64 = (0..3)
            .map(|k| {
                let direction = turned(x(), k as f64 * core::f64::consts::PI / 3.0);
                let response = bond_response(direction, plane, stiffness, activity);
                let motion = strain.at::<1>().of(direction);
                let moved: Vec<Displacement> = cycle.iter().map(|s| motion.of(*s)).collect();
                loop_work(response, &moved)
            })
            .sum();
        let expected_work = -(plane * (3.0 * activity))
            .scalar_product(enclosed(&cycle))
            .s();
        assert!(close(bond_work, loop_work(shear, &cycle), 1e-12));
        assert!(
            close(bond_work, expected_work, 1e-12),
            "{bond_work} {expected_work}"
        );
    }

    /// Random displacements of the given spread, one per site.
    fn random_displacements(n: usize, seed: u64) -> Field<Displacement> {
        let mut r = rng(seed);
        Field::from_fn(n, |_| point(0.1 * r.normal(), 0.1 * r.normal()))
    }

    fn unit_drags(n: usize) -> Field<Scalar> {
        Field::from_fn(n, |_| Scalar::new(1.0))
    }

    #[test]
    fn lattice_motion_is_covariant_and_has_the_expected_strain_dissipation() {
        for (stiffness, activity) in [(0.9, 0.0), (0.9, 1.3), (0.0, 1.3)] {
            let (columns, rows, spacing) = (5, 4, 0.8);
            let n = columns * rows;
            let (mesh, positions) = triangular_patch(columns, rows, spacing);
            let incidence = mesh.d::<Scalar>(0);
            let bonds = &incidence * &positions;
            let directions = bonds.map(|b| b.normalized());
            let gradient = &incidence * Displacement::slot();
            let drags = unit_drags(n);
            let plane = actuation_plane();
            let response = directions.map(|d| bond_response(d, plane, stiffness, activity));
            let generator = evolution(&incidence, &response, &drags);
            let displacement = random_displacements(n, 7);
            let velocity = generator.of(&displacement);
            let translation = positions.map(|_| x() * 0.4 - y() * 0.2);
            let rotation = positions.map(|p| p | plane);

            // The gradient of half the sum of squared bond extensions reads the extensions back
            // as longitudinal nodal loads. Sideways activity does no work against it.
            let along = directions.map(force_along);
            let readings = along.map(|a| a * (a & Displacement::slot()));
            let strain_map = gradient
                .adjugate()
                .compose(&SparseMap::from_diagonal(&readings).compose(&gradient));
            let strain_gradient: Field<Force> = strain_map.of(&displacement);
            let strain_rate: f64 = strain_gradient
                .iter()
                .zip(velocity.iter())
                .map(|(g, v)| (g & v).s())
                .sum();
            let passive_decay: f64 = -stiffness
                * strain_gradient
                    .iter()
                    .map(|g| g.norm_squared())
                    .sum::<f64>();

            let mirror = Force::new(1.0, 0.3).normalized();
            let reflected = directions
                .map(|d| bond_response(mirror >> d, mirror >> plane, stiffness, activity));
            let reflected_velocity =
                evolution(&incidence, &reflected, &drags).of(&displacement.map(|d| mirror >> d));

            for b in bonds.iter() {
                assert!(close(b.norm_squared(), spacing * spacing, 1e-12));
            }
            assert!(near(velocity.sum(), Displacement::zero(), 1e-12));
            assert!(generator.of(&translation).iter().all(|v| near(
                v,
                Displacement::zero(),
                1e-12
            )));
            assert!(
                generator
                    .of(&rotation)
                    .iter()
                    .all(|v| near(v, Displacement::zero(), 1e-12))
            );
            assert!(
                close(strain_rate, passive_decay, 1e-12),
                "{strain_rate} {passive_decay}"
            );
            for (r, v) in reflected_velocity.iter().zip(velocity.iter()) {
                assert!(near(r, mirror >> v, 1e-12));
            }
        }
    }

    #[test]
    fn purely_active_evolution_preserves_the_strain_quadratic() {
        let (columns, rows) = (5, 4);
        let n = columns * rows;
        let (steps, dt) = (25, 0.04);
        let (mesh, positions) = triangular_patch(columns, rows, 1.0);
        let incidence = mesh.d::<Scalar>(0);
        let directions = (&incidence * &positions).map(|b| b.normalized());
        let response = directions.map(|d| bond_response(d, actuation_plane(), 0.0, 1.2));
        let generator = evolution(&incidence, &response, &unit_drags(n));
        let initial = random_displacements(n, 5);
        let mut displacement = initial.clone();
        for _ in 0..steps {
            displacement = rk4_step(&generator, &displacement, dt);
        }
        let quadratic = |d: &Field<Displacement>| -> f64 {
            extensions(&incidence, &directions, d)
                .iter()
                .map(|e| e * e)
                .sum()
        };
        let (before, after) = (quadratic(&initial), quadratic(&displacement));
        assert!(
            ((after - before) / before).abs() <= 2e-3,
            "{before} {after}"
        );
        assert!((&displacement - &initial).norm() > 0.2);
        assert!(near(displacement.sum(), initial.sum(), 1e-12));
    }

    /// numga's scenario checks: around each loop the bonds deliver the activity times the area
    /// enclosed, oriented by the actuation plane: for one bond, the ellipse its end follows;
    /// for the patch, three bonds per cell over the circle of shear amplitudes.
    #[test]
    fn the_loops_deliver_the_activity_times_their_area() {
        let s = super::scenes();
        let last = |states: &[(Displacement, Force, Scalar)]| states[states.len() - 1].2.s();
        let bond = last(&s.bond.states);
        let shear = s.shear.states[s.shear.states.len() - 1].2.s();
        assert!(close(bond, s.expected[0], 1e-12), "{bond} {:?}", s.expected);
        assert!(
            close(shear, s.expected[1], 1e-12),
            "{shear} {:?}",
            s.expected
        );
        assert!(s.expected.iter().all(|w| *w > 0.0));
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scenes(), 1.0);
    }
}
