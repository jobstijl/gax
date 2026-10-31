//! numga's `math/penrose`: Penrose tilings by cut and project, a two-dimensional cut through the
//! integer lattice of five-dimensional space, in R(5,0).
//!
//! A Penrose tiling covers the plane with a thick and a thin rhombus in a pattern that never
//! repeats yet has five-fold symmetry everywhere. It is the shadow of the integer lattice of five
//! dimensions, cut by a plane tilted so that it never passes through a second lattice point:
//! each square face of the lattice the cut passes close enough to becomes a tile, laid flat onto
//! the cut. Shifting the cut without tilting it rearranges the tiling by local flips, phasons.
//!
//! The lattice's points and axes are antivectors, intrinsic like all geometry; the mirrors,
//! vectors, are planes that read them by the metric-free pairing `&`. Turning the five axes
//! into one another in a cycle is a rotor, the product of the four mirrors between neighbouring
//! axes. Its five powers as maps on directions, weighted by the cosine of a fifth of a turn times
//! the power, project onto the directions it turns by a fifth, `along`, and the cut is the join
//! of two axes laid flat. A face is a tile where the cut passes through the cube it is dual to:
//! the cube through its centre along the three other axes, read by the planes that pair to one
//! with one of those axes and to nothing with the other two and the cut, Cramer's rule with
//! joins alone. How often a face is a tile is the size of its dual cube's shadow square to the
//! cut, which equals the size of its tile: thick tiles outnumber thin ones by the golden ratio.
//!
//! The animation: left the tiling the cut selects; right the cut shifted round a loop square to
//! itself, the tiles that flipped lit.

use gax_numga_examples::{
    Anim, Axes, Canvas, Light, ORIGIN2, Point2, Rect, backdrop, caption, label_under, palette,
    run_with,
};

gax::algebra! {
    algebra r50 "Euclidean 5D, R(5,0): numga's `a+b+c+d+e+`, the lattice's space.";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = 1;
    kind Scalar = [1];
    versor Vector = [e1, e2, e3, e4, e5];
    kind Bivector = [e12, e13, e14, e15, e23, e24, e25, e34, e35, e45];
    kind Trivector = [e123, e124, e125, e134, e135, e145, e234, e235, e245, e345];
    kind Antivector = [e2345, e1345, e1245, e1235, e1234] intrinsic;
    kind Pseudoscalar = [e12345];
    versor Even = [1, e12, e13, e14, e15, e23, e24, e25, e34, e35, e45, e2345, e1345, e1245, e1235, e1234];
}

/// The cut and project construction.
mod penrose {
    use super::r50::{Antivector, Bivector, Even, Vector, intrinsic};
    use gax_numga_examples::signal::phasor;

    /// A plane: a mirror, or a reading of a point's offset.
    pub type Plane = Vector;
    /// A map on directions: the turn, its powers, the projections.
    pub type Projection = Antivector<(Antivector,)>;
    /// The join of two directions, an oriented area: the cut or a tile.
    pub type Join = intrinsic::Trivector;

    /// The faces of the lattice: each pair of axes, in numga's order.
    pub const SIDES: [[usize; 2]; 10] = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 3],
        [2, 4],
        [3, 4],
    ];

    /// The three other axes of each face.
    pub fn others(face: usize) -> [usize; 3] {
        let mut rest = (0..5).filter(|k| !SIDES[face].contains(k));
        core::array::from_fn(|_| rest.next().expect("three others"))
    }

    /// The mirror across axis `k`.
    pub fn mirror(k: usize) -> Plane {
        Vector::from_coeffs(core::array::from_fn(|i| if i == k { 1.0 } else { 0.0 }))
    }

    /// The axis `k`, the mirror's metric dual: an antivector.
    pub fn axis(k: usize) -> Antivector {
        mirror(k).hodge()
    }

    /// The turn of the five axes into one another: the product of the mirrors between
    /// neighbouring axes, each swapping two.
    pub fn turn() -> gax::Unit<Even> {
        let swap = |k: usize| (mirror(k) - mirror(k + 1)).unit();
        (swap(0) * swap(1)) * (swap(2) * swap(3))
    }

    /// The turn as a map on directions.
    pub fn turning() -> Projection {
        turn() >> Antivector::slot()
    }

    /// A map applied once, twice, and so on, `n` times.
    pub fn powers(cycle: Projection, n: usize) -> Vec<Projection> {
        let mut power = cycle;
        (0..n)
            .map(|_| {
                let now = power;
                power = cycle.of(power);
                now
            })
            .collect()
    }

    /// The projection onto the directions the turn turns by `harmonic` fifths of a turn: its five
    /// powers, the fifth the identity, weighted by the cosine of that turn times the power.
    pub fn projection(harmonic: usize) -> Projection {
        powers(turning(), 5)
            .into_iter()
            .enumerate()
            .fold(Projection::zero(), |sum, (k, power)| {
                let angle = core::f64::consts::TAU * (harmonic * (k + 1)) as f64 / 5.0;
                sum + power.gp(0.4 * phasor(angle).e20())
            })
    }

    /// The projection onto the cut, a fifth of a turn.
    pub fn along() -> Projection {
        projection(1)
    }

    /// The projection square to the cut and the diagonal, two fifths of a turn.
    pub fn across() -> Projection {
        projection(2)
    }

    /// The cut: the join of the first two axes laid flat.
    pub fn cut() -> Join {
        let along = along();
        along.of(axis(0)) & along.of(axis(1))
    }

    /// Per face, the reciprocal frame of its three other axes and the cut: the planes whose
    /// pairings with a point read how far along each of the three axes it is, blind to the cut.
    /// Cramer's rule with joins: per axis, the other two joined with the cut, over the volume
    /// all three span with the cut.
    pub fn reciprocal(cut: Join) -> [[Plane; 3]; 10] {
        core::array::from_fn(|face| {
            let o = others(face).map(axis);
            // Both joins are intrinsic: their quotient is a plane.
            let volume = (o[0] & o[1] & o[2] & cut).inverse();
            core::array::from_fn(|k| (o[(k + 1) % 3] & o[(k + 2) % 3] & cut) * volume)
        })
    }

    /// How far inside a face's dual cube the cut through `shift` passes, along each of its three
    /// other axes: the face is a tile where all three are positive.
    pub fn margins(shift: Antivector, centre: Antivector, frame: &[Plane; 3]) -> [f64; 3] {
        frame.map(|plane| {
            let offset = (plane & (shift - centre)).s();
            0.25 - offset * offset
        })
    }

    /// Whether a face is a tile: all three margins positive.
    pub fn is_tile(margins: [f64; 3]) -> bool {
        margins.iter().all(|m| *m > 0.0)
    }

    /// Per face: its area laid flat onto the cut, and the volume of the shadow its dual cube
    /// casts square to the cut, how often the face is a tile among faces of every kind.
    pub fn shadows(along: Projection) -> ([f64; 10], [f64; 10]) {
        let square = Antivector::slot() - along;
        let tiles = core::array::from_fn(|face| {
            let [a, b] = SIDES[face].map(axis);
            (along.of(a) & along.of(b)).norm()
        });
        let cubes = core::array::from_fn(|face| {
            let [a, b, c] = others(face).map(|k| square.of(axis(k)));
            (a & b & c).norm()
        });
        (tiles, cubes)
    }

    /// Whether each face is thick: its area above the mean.
    pub fn thick(areas: &[f64; 10]) -> [bool; 10] {
        let mean = areas.iter().sum::<f64>() / 10.0;
        areas.map(|a| a > mean)
    }

    /// The points of the integer lattice within `radius` of the origin along every axis.
    pub fn lattice(radius: i32) -> Vec<Antivector> {
        let side = (2 * radius + 1) as usize;
        (0..side.pow(5))
            .map(|mut n| {
                (0..5).fold(Antivector::zero(), |sum, k| {
                    let coordinate = (n % side) as f64 - f64::from(radius);
                    n /= side;
                    sum + axis(k).gp(coordinate)
                })
            })
            .collect()
    }

    /// The centre of a face at a lattice point.
    pub fn centre(point: Antivector, face: usize) -> Antivector {
        let [a, b] = SIDES[face].map(axis);
        point + (a + b).gp(0.5)
    }

    /// A face's corners about its centre, half a unit back and forth along both its axes.
    pub fn corners(centre: Antivector, face: usize) -> [Antivector; 4] {
        let [a, b] = SIDES[face].map(axis);
        [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
            .map(|(s, t)| centre + a.gp(s) + b.gp(t))
    }

    /// The point the cut passes through, its coordinates summing to zero (numga's).
    pub fn shift() -> Antivector {
        [0.12, 0.29, -0.11, 0.33, -0.63]
            .iter()
            .enumerate()
            .fold(Antivector::zero(), |sum, (k, x)| sum + axis(k).gp(*x))
    }

    /// How far the cut is shifted round its loop.
    pub const PHASON: f64 = 0.35;

    /// The point the cut passes through a share of the way round its loop: the starting point
    /// shifted round a circle among the directions square to the cut and to the diagonal.
    pub fn phason(share: f64) -> Antivector {
        let across = across();
        let start = across.of(axis(0));
        let plane: Bivector = start.commutator(across.of(axis(1)));
        let turn = plane.normalized().gp(core::f64::consts::PI * share).exp();
        shift() + ((turn >> start) - start).gp(PHASON / start.norm())
    }

    /// The planes that read the tiling in the cut: across the first axis laid flat, and across
    /// that turned a quarter turn in the cut (the commutator of two directions in the cut turns
    /// one towards the other within it).
    pub fn view(along: Projection) -> [Plane; 2] {
        let right = along.of(axis(0)).normalized();
        let turning: Bivector = right.commutator(along.of(axis(1)));
        let quarter = turning.normalized().gp(-core::f64::consts::FRAC_PI_4).exp();
        let up = quarter >> right;
        [right.hodge(), up.hodge()]
    }
}

use penrose::Plane;
use r50::Antivector;

/// The lattice points within this radius of the origin along every axis.
const LATTICE_RADIUS: i32 = 3;
/// numga's loop: 135 frames of 40 ms.
const SECONDS: f32 = 5.4;
/// Half the width of the view, in tile edges.
const EXTENT: f32 = 2.6;

/// A face of the lattice near the cut, ready to draw: its corners laid flat and read in the cut,
/// what its reciprocal frame reads of its centre, its kind, and whether the starting cut selects
/// it.
struct Face {
    corners: [Point2; 4],
    centre_reading: [f64; 3],
    face: usize,
    at_start: bool,
}

/// What the frames draw: the faces in view, the frames' readings, thick or thin.
struct Scene {
    faces: Vec<Face>,
    frames: [[Plane; 3]; 10],
    thick: [bool; 10],
}

impl Scene {
    fn new() -> Scene {
        let along = penrose::along();
        let frames = penrose::reciprocal(penrose::cut());
        let (areas, _) = penrose::shadows(along);
        let reading = penrose::view(along);
        let read = |p: Antivector| {
            let [x, y] = reading.map(|plane| (plane & p).s() as f32);
            Point2::xy(x, y)
        };
        let start = penrose::shift();
        let reach = EXTENT * 1.5;
        let mut faces = Vec::new();
        for point in penrose::lattice(LATTICE_RADIUS) {
            for (face, frame) in frames.iter().enumerate() {
                let centre = penrose::centre(point, face);
                let at = read(along.of(centre));
                if at.distance(ORIGIN2) > reach {
                    continue;
                }
                let centre_reading = frame.map(|plane| (plane & centre).s());
                let at_start = penrose::is_tile(penrose::margins(start, centre, frame));
                faces.push(Face {
                    corners: penrose::corners(centre, face).map(|c| read(along.of(c))),
                    centre_reading,
                    face,
                    at_start,
                });
            }
        }
        Scene {
            faces,
            frames,
            thick: penrose::thick(&areas),
        }
    }

    /// Whether a face is a tile for the cut through `shift`: its margins from what the frame
    /// reads of the shift less what it reads of the centre.
    fn selects(&self, f: &Face, readings: &[[f64; 3]; 10]) -> bool {
        let r = readings[f.face];
        (0..3).all(|k| {
            let offset = r[k] - f.centre_reading[k];
            0.25 - offset * offset > 0.0
        })
    }

    /// What each face's frame reads of a shift.
    fn readings(&self, shift: Antivector) -> [[f64; 3]; 10] {
        self.frames
            .map(|frame| frame.map(|plane| (plane & shift).s()))
    }
}

/// The thick tiles' colour, and the thin ones'.
fn colour(thick: bool) -> Light {
    if thick {
        Light::from_srgb(0.96, 0.62, 0.04, 1.6)
    } else {
        palette::SKY
    }
}

/// The tiles a cut selects in `rect`: filled by kind, outlined dark; with `lit`, the tiles not
/// selected at the start (flipped) brighter.
fn tiling_panel(c: &mut Canvas, scene: &Scene, rect: Rect, shift: Antivector, lit: bool) {
    let ax = Axes::equal(rect, ORIGIN2, EXTENT);
    let readings = scene.readings(shift);
    let mut sides: Vec<[Point2; 2]> = Vec::new();
    for f in scene.faces.iter().filter(|f| scene.selects(f, &readings)) {
        let flipped = lit && !f.at_start;
        let fade = if flipped { 0.6 } else { 0.22 };
        ax.fill(c, &f.corners, colour(scene.thick[f.face]).faded(fade), 1.0);
        let px = f.corners.map(|p| ax.px(p));
        sides.extend((0..4).map(|k| [px[k], px[(k + 1) % 4]]));
    }
    // The tiles' edges, dark, in one stroke.
    ax.clip(c);
    c.outline(&sides, 1.0, palette::BOTTOM, 0.9);
    c.unclip();
}

fn draw(c: &mut Canvas, scene: &Scene, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (left, right) = (
        screen.part(0.03, 0.14, 0.48, 0.92),
        screen.part(0.52, 0.14, 0.97, 0.92),
    );
    let room = |r: Rect| r.inset(0.0, 0.0, 0.0, 20.0);
    tiling_panel(c, scene, room(left), penrose::shift(), false);
    label_under(
        c,
        left,
        "THE FACES THE CUT PASSES CLOSE TO, LAID FLAT",
        26.0,
        0.9,
    );
    let shifted = penrose::phason(f64::from(t / SECONDS));
    tiling_panel(c, scene, room(right), shifted, true);
    label_under(
        c,
        right,
        "THE CUT SHIFTED ROUND A LOOP: TILES FLIP, LIT",
        26.0,
        0.9,
    );
    caption(
        c,
        "PENROSE TILINGS: A CUT THROUGH THE LATTICE OF FIVE DIMENSIONS (R(5,0))",
        "THICK AND THIN RHOMBI, THE SQUARE FACES NEAR A PLANE AT THE SLOPE OF A FIVE-FOLD TURN",
    );
}

fn main() {
    run_with(
        Anim::new("penrose", SECONDS).size(960, 540),
        Scene::new,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::penrose;
    use gax::ApproxEq;
    use gax_numga_examples::measure::roots;

    /// The golden ratio, the positive root of `t² - t - 1`.
    fn golden() -> f64 {
        let (middle, half) = roots(1.0, -0.5, -1.0).expect("real roots");
        middle + half
    }

    /// numga's first test: the shadow of each dual cube square to the cut is as large as its
    /// face laid flat, thick against thin in the golden ratio.
    #[test]
    fn faces_are_tiles_as_often_as_their_area_in_the_golden_ratio() {
        let (areas, volumes) = penrose::shadows(penrose::along());
        for (a, v) in areas.iter().zip(volumes) {
            assert!((a - v).abs() < 1e-12, "{a} {v}");
        }
        let thick = penrose::thick(&areas);
        for (a, ta) in areas.iter().zip(thick) {
            for (b, tb) in areas.iter().zip(thick) {
                if ta && !tb {
                    assert!((a / b - golden()).abs() < 1e-12 * golden());
                }
            }
        }
    }

    /// numga's second test: in a patch of a few hundred tiles, thick ones outnumber thin ones
    /// by the golden ratio, to within its edge.
    #[test]
    fn thick_tiles_outnumber_thin_ones_by_the_golden_ratio() {
        let (areas, _) = penrose::shadows(penrose::along());
        let thick = penrose::thick(&areas);
        let frames = penrose::reciprocal(penrose::cut());
        let shift = penrose::shift();
        let mut counts = [0usize; 2];
        for point in penrose::lattice(super::LATTICE_RADIUS) {
            for face in 0..10 {
                let centre = penrose::centre(point, face);
                if penrose::is_tile(penrose::margins(shift, centre, &frames[face])) {
                    counts[usize::from(thick[face])] += 1;
                }
            }
        }
        let ratio = counts[1] as f64 / counts[0] as f64;
        assert!((ratio - golden()).abs() < 5e-2 * golden(), "{counts:?}");
    }

    /// numga's third test: the reciprocal frames read nothing of the cut, so a shift within it
    /// only slides the tiling.
    #[test]
    fn sliding_the_cut_within_itself_selects_the_same_faces() {
        let frames = penrose::reciprocal(penrose::cut());
        let shift = penrose::shift();
        let along = penrose::along();
        let within = shift + along.of(penrose::axis(2).gp(0.7) - penrose::axis(4).gp(1.3));
        for point in penrose::lattice(2) {
            for (face, frame) in frames.iter().enumerate() {
                let centre = penrose::centre(point, face);
                let a = penrose::margins(within, centre, frame);
                let b = penrose::margins(shift, centre, frame);
                for (x, y) in a.iter().zip(b) {
                    assert!((x - y).abs() < 1e-11);
                }
            }
        }
    }

    /// The scenario's checks: the turn is of order five, and each face's reciprocal frame reads
    /// its three other axes as a unit frame and the cut's directions as nothing.
    #[test]
    fn the_turn_has_order_five_and_the_frames_read_the_axes() {
        let fifth = penrose::powers(penrose::turning(), 5)[4];
        for k in 0..5 {
            let axis = penrose::axis(k);
            assert!(fifth.of(axis).approx_eq(&axis, 1e-12));
        }
        let frames = penrose::reciprocal(penrose::cut());
        let along = penrose::along();
        for (face, frame) in frames.iter().enumerate() {
            let others = penrose::others(face);
            for (i, plane) in frame.iter().enumerate() {
                for (j, other) in others.iter().enumerate() {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(((*plane & penrose::axis(*other)).s() - expected).abs() < 1e-12);
                }
                for k in 0..5 {
                    assert!((*plane & along.of(penrose::axis(k))).s().abs() < 1e-12);
                }
            }
        }
    }

    /// numga's last test: the scenes draw, the still and two frames of the flips.
    #[test]
    fn scenes_draw() {
        let scene = super::Scene::new();
        for t in [0.0, 0.04] {
            gax_numga_examples::app::assert_draws_with(super::draw, &scene, t);
        }
    }
}
