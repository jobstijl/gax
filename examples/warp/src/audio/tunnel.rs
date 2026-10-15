//! The Tunnel as the sound hears it: its events become the Plane's, which the effects already
//! voice, each placed in the listener's frame.

use crate::render::tunnel::View;
use crate::sim::{self, Kind};
use crate::tunnel::{self, Event as T, Foe};

/// The Tunnel's events this tick as the sound hears them, into `out` (cleared first): each
/// position taken into the camera's frame (`cam << p`, the listener), where across is the pan
/// and the depth the distance.
pub fn sounds(w: &tunnel::World, view: &View, out: &mut Vec<sim::Event>) {
    out.clear();
    let at = |p: tunnel::P| view.listen(w.track.place(p));
    out.extend(w.events.iter().filter_map(|e| {
        Some(match *e {
            T::Fire { pos } => sim::Event::Fire {
                pos: at(pos),
                dir: sim::dir(0.0, 1.0),
            },
            T::Hit { pos, foe } => sim::Event::Hit {
                pos: at(pos),
                kind: foe.kind(),
            },
            T::Kill { pos, foe, points } => sim::Event::Kill {
                pos: at(pos),
                kind: foe.kind(),
                size: match foe {
                    Foe::Turret | Foe::Serpent => 2.0,
                    Foe::Singularity => 3.0,
                    _ => 1.0,
                },
                scored: points > 0,
                points,
                wreck: None,
            },
            T::Bolt { pos } => sim::Event::Deflect {
                pos: at(pos),
                kind: Kind::Warden,
            },
            T::Wall { pos } | T::GateMiss { pos } => sim::Event::Wall { pos: at(pos) },
            T::Warn { pos, foe } => sim::Event::Warn {
                pos: at(pos),
                kind: foe.kind(),
            },
            T::Dive { pos } => sim::Event::Warn {
                pos: at(pos),
                kind: Foe::Drone.kind(),
            },
            T::Pickup { pos, mult } => sim::Event::Pickup { pos: at(pos), mult },
            T::Gate { pos, chain, .. } => sim::Event::Pickup {
                pos: at(pos),
                mult: chain * 2,
            },
            T::Death { pos } => sim::Event::Death { pos: at(pos) },
            T::Bomb { pos } => sim::Event::Bomb { pos: at(pos) },
            T::Respawn => sim::Event::Respawn,
            T::Extra { life } => sim::Event::Extra { life },
            T::GameOver => sim::Event::GameOver,
            T::Absorb { pos, mass } => sim::Event::Absorb { pos: at(pos), mass },
            T::Burst { pos } => sim::Event::Burst { pos: at(pos) },
            T::Slingshot { .. } => sim::Event::Extra { life: false },
            T::Roll { .. } => return None,
        })
    }));
}
