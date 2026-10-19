//! The bots: they play the `--smoke` run, the attract mode's demos when a table has no run
//! yet, the headless shots and the tests.

use crate::game::{Run, TunnelRun};
use crate::geom::{ORIGIN, clamp_length, with_length};
use crate::input::Flight;
use crate::signal::wave;
use crate::sim::body::turned;
use crate::{sim, tunnel};
use gax::pga2d::Point;

/// The bot's input for the run at time `t`: the Plane's controls, and the Tunnel's.
pub fn input(run: &Run, t: f32) -> (sim::Input, Flight) {
    match run {
        Run::Plane(r) => (plane(&r.world, t), Flight::default()),
        Run::Tunnel(r) => self::tunnel(r, t),
    }
}

/// The Plane's bot: strafe around the nearest enemy, aim at it, fire; bomb when crowded.
pub fn plane(w: &sim::World, t: f32) -> sim::Input {
    let p = w.ship.body.pos();
    let near = w
        .enemies
        .iter()
        .map(|e| e.body.pos())
        .min_by(|a, b| a.distance(p).total_cmp(&b.distance(p)));
    let mut input = sim::Input::default();
    // Wander: a direction turning with time.
    let mut go = Point::direction_from_angle(t * 0.9) * 0.6;
    if let Some(q) = near {
        let to = q - p;
        input.aim = with_length(to, 1.0);
        input.fire = true;
        if to.ideal_norm() < 8.0 {
            // Away and around: the way back from it, turned a little.
            go = with_length(turned(-to, -0.6), 1.0);
        }
        let crowd = w.enemies.iter().filter(|e| e.body.pos().distance(p) < 3.0);
        input.bomb = crowd.count() >= 5;
    }
    // Stay off the walls: pulled back towards the middle.
    let home = (ORIGIN - p) * (0.6 / 32.0);
    input.movement = clamp_length(go + home, 1.0);
    input
}

/// The Tunnel's bot: aims the cursor at the nearest enemy ahead, weaves, fires in bursts,
/// and rolls away from bolts and diving drones that come close.
pub fn tunnel(run: &TunnelRun, t: f32) -> (sim::Input, Flight) {
    let w = &run.world;
    let s = w.ship.s();
    let target = w.nearest_ahead(6.0).map(|e| e.pos);
    let cursor = target.and_then(|q| run.view.on_screen(w, q));
    let ahead = |p: tunnel::P| tunnel::arc(p) - s;
    let bolt = w
        .bolts
        .iter()
        .any(|b| ahead(b.pos) > 0.0 && ahead(b.pos) < 6.0);
    let dive = w
        .enemies
        .iter()
        .any(|e| e.flight == tunnel::Flight::Dive && ahead(e.pos) < 8.0);
    let input = sim::Input {
        movement: Point::direction_from_angle(t * 0.6) * 0.55,
        // In bursts, so that formations live long enough to be seen.
        fire: wave(t * 1.3) > 0.2,
        ..sim::Input::default()
    };
    let flight = Flight {
        cursor: cursor.or(Some(ORIGIN)),
        stick: None,
        roll: i8::from(bolt || dive),
        throttle: wave(t * 0.2),
        bomb: false,
    };
    (input, flight)
}
