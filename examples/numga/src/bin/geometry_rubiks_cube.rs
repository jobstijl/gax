//! numga's `geometry/rubiks_cube`: a Rubik's cube as points, turned, drawn and solved with the
//! same quarter-turn maps, in VGA3D.
//!
//! The cube is 27 cubies, their centres at -1, 0 and 1 along each axis, and a sticker is a
//! cubie's face on the cube's surface. Each sticker is a point, its cubie's centre plus half its
//! face, in units of half a cubie, `2 cubie + face`, so every coordinate is an integer, and so
//! are its corners. A face's layer is the stickers whose point reaches past 1 along the face.
//! A quarter turn is the rotor `1 + plane`, the face's plane (the face times the pseudoscalar),
//! as a map on vectors (`(1 + plane) >> Vector::slot()`), halved: the sandwich with an
//! unnormalized rotor scales by its squared norm, two, so the halved map turns by exactly a
//! quarter, clockwise seen from outside, with coefficients 0 and ±1; `1 - plane` turns back. A
//! turned point keeps integer coordinates, and the cube is solved when every sticker is back at
//! its own point.
//!
//! The same maps turn the whole frontier of a search: every state by both directions of every
//! face. The solver meets in the middle, widening the states reachable from the scramble and
//! from the solved cube a quarter turn at a time until they share one. Drawing turns a layer by
//! the rotor of a fraction of the quarter, `exp(plane π/4 f)`.
//!
//! In Singmaster's notation the faces along +x, -x, +y, -y, +z and -z read as R, L, B, F, U
//! and D. The animation plays a random scramble and then the solution the search found.

use gax_numga_examples::signal::smoothstep;
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Light, Point2, Rect, Scene3, backdrop, caption, light,
    palette, run,
};

mod cube {
    use gax::vga3d::{Bivector, Pseudoscalar, Rotor, Vector};
    use std::collections::{HashMap, HashSet};

    pub type V = Vector<(), f64>;
    /// A map on vectors.
    pub type Map = Vector<(Vector,), f64>;
    /// The cube's 54 sticker points.
    pub type Cube = [V; 54];
    /// A state's sticker coordinates as integers: exact, since they are.
    type Key = [i8; 162];

    /// A quarter turn: its direction (0 clockwise seen from outside, 1 back), and its face, a
    /// side (0 for +, 1 for -) and an axis.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Move {
        pub direction: usize,
        pub side: usize,
        pub axis: usize,
    }

    impl Move {
        /// Every move, directions outermost, then sides, then axes.
        pub fn all() -> impl Iterator<Item = Move> {
            (0..12).map(|i| Move {
                direction: i / 6,
                side: i / 3 % 2,
                axis: i % 3,
            })
        }

        /// The move that undoes this one.
        pub fn inverse(self) -> Move {
            Move {
                direction: 1 - self.direction,
                ..self
            }
        }

        /// The face along the move's side and axis.
        pub fn face(self) -> V {
            face(self.side, self.axis)
        }

        /// The face's plane, clockwise seen from outside: the face times the pseudoscalar.
        pub fn plane(self) -> Bivector<(), f64> {
            self.face() * Pseudoscalar::new(1.0)
        }

        /// The sign of the turn: plus one clockwise, minus one back.
        pub fn sign(self) -> f64 {
            [1.0, -1.0][self.direction]
        }

        /// The quarter turn as a map on vectors: the sandwich with `1 ± plane`, halved.
        pub fn turn(self) -> Map {
            let rotor = Rotor::new(1.0, 0.0, 0.0, 0.0) + Rotor::from(self.plane().gp(self.sign()));
            (rotor >> Vector::slot()).gp(0.5)
        }

        /// The move in Singmaster's notation.
        pub fn name(self) -> String {
            let face = ["R", "B", "U", "L", "F", "D"][self.side * 3 + self.axis];
            format!("{face}{}", ["", "'"][self.direction])
        }
    }

    /// The unit vector along a side (0 for +, 1 for -) of an axis.
    pub fn face(side: usize, axis: usize) -> V {
        let mut c = [0.0; 3];
        c[axis] = [1.0, -1.0][side];
        Vector::from_coeffs(c)
    }

    /// Whether a sticker lies in a face's layer: it reaches past 1 along the face.
    pub fn in_layer(sticker: V, face: V) -> bool {
        (sticker | face).s() > 1.0
    }

    /// The stickers after a quarter turn of a face: those in its layer turned, the rest left.
    pub fn turned(cube: &Cube, turn: Map, face: V) -> Cube {
        cube.map(|s| if in_layer(s, face) { turn.of(s) } else { s })
    }

    /// The cube after a move.
    pub fn apply(cube: &Cube, m: Move) -> Cube {
        turned(cube, m.turn(), m.face())
    }

    /// Every quarter turn of each of the given states, in the order of the moves (directions,
    /// sides, axes), each move over all the states.
    pub fn frontier(states: &[Cube]) -> Vec<Cube> {
        let moves: Vec<(Map, V)> = Move::all().map(|m| (m.turn(), m.face())).collect();
        moves
            .iter()
            .flat_map(|(turn, face)| states.iter().map(|s| turned(s, *turn, *face)))
            .collect()
    }

    /// The cubies' centres, at -1, 0 and 1 along each axis.
    pub fn cubies() -> Vec<V> {
        (0..27)
            .map(|i| [i / 9, i / 3 % 3, i % 3].map(|c| c as f64 - 1.0))
            .map(Vector::from_coeffs)
            .collect()
    }

    /// The sides (side, axis) a cubie shows on the cube's surface: those it lies on.
    pub fn outward(cubie: V) -> impl Iterator<Item = (usize, usize)> {
        (0..6)
            .map(|i| (i / 3, i % 3))
            .filter(move |&(side, axis)| (cubie | face(side, axis)).s() > 0.5)
    }

    /// The corners of a cubie's face about its centre, in order round the face: one corner on
    /// the face's side, then its quarter turns about it.
    pub fn square(side: usize, axis: usize) -> [V; 4] {
        let quarter = Move {
            direction: 0,
            side,
            axis,
        }
        .turn();
        let corner = Vector::new(1.0, 1.0, 1.0) - face(0, axis) + face(side, axis);
        let mut around = [corner; 4];
        for k in 1..4 {
            around[k] = quarter.of(around[k - 1]);
        }
        around
    }

    /// The solved cube: its 54 sticker points, their corners, and the face each belongs to,
    /// `side * 3 + axis`.
    pub fn solved() -> (Cube, [[V; 4]; 54], [usize; 54]) {
        let shown: Vec<(V, usize, usize)> = cubies()
            .into_iter()
            .flat_map(|c| outward(c).map(move |(side, axis)| (c, side, axis)))
            .collect();
        let points = core::array::from_fn(|i| {
            let (c, side, axis) = shown[i];
            c.gp(2.0) + face(side, axis)
        });
        let corners = core::array::from_fn(|i| {
            let (c, side, axis) = shown[i];
            square(side, axis).map(|v| c.gp(2.0) + v)
        });
        let colours = core::array::from_fn(|i| shown[i].1 * 3 + shown[i].2);
        (points, corners, colours)
    }

    fn key(cube: &Cube) -> Key {
        let mut k = [0; 162];
        for (i, v) in cube.iter().enumerate() {
            for (j, c) in v.c.iter().enumerate() {
                k[3 * i + j] = c.round() as i8;
            }
        }
        k
    }

    /// The states reached from one cube, a quarter turn at a time: each level's new states,
    /// with the state one turn back and the move that led there.
    struct Search {
        states: Vec<Vec<Cube>>,
        parents: Vec<Vec<usize>>,
        moves: Vec<Vec<Move>>,
        seen: HashSet<Key>,
    }

    impl Search {
        fn new(start: Cube) -> Search {
            Search {
                states: vec![vec![start]],
                parents: vec![vec![0]],
                moves: vec![vec![]],
                seen: HashSet::from([key(&start)]),
            }
        }

        fn last(&self) -> &[Cube] {
            self.states.last().expect("the start")
        }

        /// The states one turn on from the last level, not reached before (the first copy of
        /// each, in the frontier's order).
        fn widen(&mut self) {
            let count = self.last().len();
            let (mut states, mut parents, mut moves) = (vec![], vec![], vec![]);
            let all: Vec<Move> = Move::all().collect();
            for (i, s) in frontier(self.last()).into_iter().enumerate() {
                if self.seen.insert(key(&s)) {
                    states.push(s);
                    parents.push(i % count);
                    moves.push(all[i / count]);
                }
            }
            self.states.push(states);
            self.parents.push(parents);
            self.moves.push(moves);
        }

        /// The moves from the start to a state of the last level.
        fn path(&self, mut index: usize) -> Vec<Move> {
            let mut moves = vec![];
            for level in (1..self.states.len()).rev() {
                moves.push(self.moves[level][index]);
                index = self.parents[level][index];
            }
            moves.reverse();
            moves
        }
    }

    /// The first state of `these` that `those` share, as its index in each.
    fn meeting(these: &[Cube], those: &[Cube]) -> Option<(usize, usize)> {
        let index: HashMap<Key, usize> =
            those.iter().enumerate().map(|(i, s)| (key(s), i)).collect();
        these
            .iter()
            .enumerate()
            .find_map(|(i, s)| index.get(&key(s)).map(|j| (i, *j)))
    }

    /// The moves that take the scrambled cube to the solved one, found by widening the states
    /// reachable from each, a turn at a time and each side in turn, up to `depth` turns each,
    /// until they meet.
    pub fn solve(scramble: Cube, solved: Cube, depth: usize) -> Option<Vec<Move>> {
        let mut searches = [Search::new(scramble), Search::new(solved)];
        for level in 0..2 * depth {
            let (side, other) = (level % 2, 1 - level % 2);
            searches[side].widen();
            if let Some((here, there)) = meeting(searches[side].last(), searches[other].last()) {
                let (forward, backward) = if side == 0 {
                    (here, there)
                } else {
                    (there, here)
                };
                let mut moves = searches[0].path(forward);
                let back = searches[1].path(backward);
                moves.extend(back.into_iter().rev().map(Move::inverse));
                return Some(moves);
            }
        }
        None
    }
}

use cube::*;

/// numga's scramble length, and how many turns the search goes from each side.
const SCRAMBLE: usize = 10;
const DEPTH: usize = 5;
/// The seed of the scramble's moves.
const SEED: u64 = 4;
/// Seconds per quarter turn.
const PER_MOVE: f32 = 0.55;

/// A scramble of random quarter turns, none undoing the one before, and the moves that solve
/// it.
fn scrambled_and_solved() -> (Vec<Move>, Vec<Move>) {
    use rand::Rng;
    let mut r = gax_numga_examples::rng::rng(SEED);
    let all: Vec<Move> = Move::all().collect();
    let mut scramble: Vec<Move> = vec![];
    while scramble.len() < SCRAMBLE {
        let m = all[r.random_range(0..all.len())];
        if scramble.last() != Some(&m.inverse()) {
            scramble.push(m);
        }
    }
    let (solved, _, _) = solved();
    let scrambled = scramble.iter().fold(solved, |c, m| apply(&c, *m));
    let solution = solve(scrambled, solved, DEPTH).expect("solvable within the depth");
    (scramble, solution)
}

/// The scene's scramble and solution, found once.
fn scene() -> &'static (Vec<Move>, Vec<Move>) {
    static SCENE: std::sync::OnceLock<(Vec<Move>, Vec<Move>)> = std::sync::OnceLock::new();
    SCENE.get_or_init(scrambled_and_solved)
}

/// The face colours, by `side * 3 + axis`: +x red, +y blue, +z white, -x orange, -y green, -z
/// yellow.
fn sticker_colour(face: usize) -> Light {
    [
        light(1.0, 0.1, 0.16, 1.5),
        light(0.15, 0.35, 1.0, 1.7),
        light(0.95, 0.95, 1.0, 1.3),
        light(1.0, 0.4, 0.05, 1.6),
        light(0.1, 0.9, 0.45, 1.5),
        light(1.0, 0.85, 0.1, 1.5),
    ][face]
}

/// The cubies' plastic.
fn body() -> Light {
    light(0.02, 0.025, 0.05, 1.0)
}

/// A point of the cube, in half cubies, as a point of the scene, in cubies.
fn spot(v: V) -> gax::pga3d::Point<(), f64> {
    let [x, y, z] = v.c.map(|c| 0.5 * c);
    gax::pga3d::Point::xyz(x, y, z)
}

/// The share of a cubie's face its sticker covers, across.
const STICKER: f64 = 0.84;

/// A cubie's face: the sticker, inset, and around it a frame of dark plastic in four pieces
/// that do not overlap it, so that no depth order can put the plastic in front.
fn draw_sticker(s: &mut Scene3, corners: &[V; 4], colour: Light) {
    let centre = corners.iter().fold(V::zero(), |sum, v| sum + *v).gp(0.25);
    let outer = corners.map(spot);
    let inner = corners.map(|v| spot(centre + (v - centre).gp(STICKER)));
    for k in 0..4 {
        let next = (k + 1) % 4;
        s.quad(outer[k], outer[next], inner[next], inner[k], body(), 1.0);
    }
    let lit = s.lit(inner[0], inner[1], inner[2], colour);
    s.quad(inner[0], inner[1], inner[2], inner[3], lit, 1.0);
}

/// The cube in `rect` with the stickers' corners `corners`, and the dark cut under a turning
/// layer.
fn draw_cube(c: &mut Canvas, rect: Rect, corners: &[[V; 4]; 54], cut: &[[V; 4]]) {
    let cam = Camera::orbit(
        rect,
        gax_numga_examples::ORIGIN3,
        14.0,
        (-58f32).to_radians(),
        28f32.to_radians(),
        Lens::Perspective(0.36),
    );
    gax_numga_examples::scene3::panel3(c, rect, cam, |s| {
        let (_, _, colours) = solved();
        for (square, colour) in corners.iter().zip(colours) {
            draw_sticker(s, square, sticker_colour(colour));
        }
        for square in cut {
            let [a, b, cc, d] = square.map(spot);
            s.quad(a, b, cc, d, body(), 1.0);
        }
    });
}

/// The stickers' points and corners after whole moves.
fn settled(moves: &[Move]) -> (Cube, [[V; 4]; 54]) {
    let (mut stickers, mut corners, _) = solved();
    for m in moves {
        let (turn, face) = (m.turn(), m.face());
        for (s, square) in stickers.iter_mut().zip(corners.iter_mut()) {
            if in_layer(*s, face) {
                *s = turn.of(*s);
                *square = square.map(|v| turn.of(v));
            }
        }
    }
    (stickers, corners)
}

/// The faces of a move's layer's cubies towards the middle of the cube, seen while it turns.
fn cut(m: Move) -> Vec<[V; 4]> {
    let inner = square(1 - m.side, m.axis);
    let layer = cubies()
        .into_iter()
        .filter(|c| outward(*c).any(|f| f == (m.side, m.axis)));
    layer.map(|c| inner.map(|v| c.gp(2.0) + v)).collect()
}

/// The cube at a move, a fraction of the way through it: the stickers' corners, and the cut
/// under the turning layer on both sides, still and turned.
fn pose(moves: &[Move], index: usize, fraction: f64) -> ([[V; 4]; 54], Vec<[V; 4]>) {
    let (stickers, mut corners) = settled(&moves[..index]);
    let Some(m) = moves.get(index) else {
        return (corners, vec![]);
    };
    let rotor = m
        .plane()
        .gp(core::f64::consts::FRAC_PI_4 * fraction * m.sign())
        .exp();
    for (s, square) in stickers.iter().zip(corners.iter_mut()) {
        if in_layer(*s, m.face()) {
            *square = square.map(|v| rotor >> v);
        }
    }
    let still = cut(*m);
    let turned: Vec<[V; 4]> = still.iter().map(|q| q.map(|v| rotor >> v)).collect();
    (corners, [still, turned].concat())
}

/// The moves in Singmaster's notation, the current one lit.
fn draw_moves(c: &mut Canvas, at: Point2, label: &str, moves: &[Move], current: Option<usize>) {
    c.text(label, at, 11.0, palette::ink().faded(0.6), Align::Left);
    for (k, m) in moves.iter().enumerate() {
        let colour = if Some(k) == current {
            palette::orange()
        } else {
            palette::ink().faded(0.75)
        };
        let place = at + Point2::direction(36.0 * (k % 8) as f32, 22.0 * (1 + k / 8) as f32);
        c.text(&m.name(), place, 13.0, colour, Align::Left);
    }
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (scramble, solution) = scene();
    let moves = [scramble.as_slice(), solution].concat();
    let step = t / PER_MOVE;
    let index = (step as usize).min(moves.len());
    // The turn under way, eased in and out (numga's smoothstep).
    let (corners, cut) = pose(&moves, index, smoothstep(f64::from(step.fract())));
    draw_cube(c, screen.part(0.0, 0.05, 0.66, 1.0), &corners, &cut);
    let panel = screen.part(0.66, 0.25, 1.0, 1.0).lo;
    let current = |offset: usize, len: usize| index.checked_sub(offset).filter(|k| *k < len);
    draw_moves(c, panel, "SCRAMBLE", scramble, current(0, scramble.len()));
    let below = panel + Point2::direction(0.0, 110.0);
    let label = format!("SOLUTION, {} QUARTER TURNS", solution.len());
    draw_moves(
        c,
        below,
        &label,
        solution,
        current(scramble.len(), solution.len()),
    );
    caption(
        c,
        "RUBIK'S CUBE: QUARTER TURNS AS MAPS ON POINTS",
        "THE SANDWICH WITH 1 + PLANE, HALVED, TURNS INTEGER POINTS; SOLVED BY MEETING IN THE MIDDLE (VGA3D)",
    );
}

fn main() {
    let (scramble, solution) = scene();
    // Every move, and the time of two more on the solved cube.
    let seconds = PER_MOVE * (scramble.len() + solution.len() + 2) as f32;
    run(Anim::new("rubiks_cube", seconds).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::cube::*;

    /// Every quarter turn, four times over, is the identity, and a turn and its inverse undo
    /// each other, exactly.
    #[test]
    fn quarter_turns_are_exact_cycle_in_fours_and_undo() {
        let (solved, _, _) = solved();
        for m in Move::all() {
            let four = (0..4).fold(solved, |c, _| apply(&c, m));
            assert_eq!(four, solved);
            assert_eq!(apply(&apply(&solved, m), m.inverse()), solved);
        }
    }

    /// The frontier of the twelve states one turn from solved is every turn of every one of
    /// them, in order.
    #[test]
    fn the_frontier_is_every_turn_of_every_state() {
        let (solved, _, _) = solved();
        let states = frontier(&[solved]);
        let children = frontier(&states);
        for (i, m) in Move::all().enumerate() {
            for (parent, state) in states.iter().enumerate() {
                assert_eq!(children[i * 12 + parent], apply(state, m));
            }
        }
    }

    /// The scene's checks: every turn moves stickers onto sticker points, so the scrambled
    /// cube's points are the solved cube's; the solution takes the scrambled cube back exactly,
    /// in no more turns than the scramble; and the cube draws through a move.
    #[test]
    fn the_scene_solves_and_draws() {
        let (scramble, solution) = super::scrambled_and_solved();
        let (solved, _, _) = solved();
        let scrambled = scramble.iter().fold(solved, |c, m| apply(&c, *m));
        let sorted = |cube: &Cube| {
            let mut points: Vec<[i64; 3]> = cube.iter().map(|v| v.c.map(|x| x as i64)).collect();
            points.sort();
            points
        };
        assert_eq!(sorted(&scrambled), sorted(&solved));
        let back = solution.iter().fold(scrambled, |c, m| apply(&c, *m));
        assert_eq!(back, solved);
        assert!(solution.len() <= scramble.len());
        for t in [0.3, 5.75] {
            gax_numga_examples::app::assert_draws(super::draw, t);
        }
    }
}
