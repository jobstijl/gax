//! The enemy families, and the numbers that set them apart. What a family does is code (the
//! matches in `enemy.rs`); how fast, how tough and how much it is worth is this table.

/// Enemy families.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// Wanders; harmless alone. Teaches shooting.
    Drifter,
    /// Homes in with a limited turn rate.
    Chaser,
    /// A gravity well: bends bullets, grid and particles, eats enemies and grows, bursts when
    /// overfed.
    Singularity,
    /// A small, fast chaser from a burst singularity or a carrier.
    Mote,
    /// Approaches, but sidesteps shots: the distance to each shot's line of flight.
    Evader,
    /// Tough; splits into three fragments.
    Splitter,
    /// A fast dart from a splitter.
    Fragment,
    /// A chain of segments following the head by motor interpolation; only the head is
    /// vulnerable, the body blocks shots.
    Serpent,
    /// Shielded in front (shots reflect off the shield line): flank it.
    Warden,
    /// Slow and tough; launches motes, and a ring of them when destroyed.
    Carrier,
}

/// Segments of a serpent's body.
pub const SERPENT_LEN: usize = 12;
/// Distance between serpent segments.
pub const SERPENT_GAP: f32 = 0.55;

/// A family's numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stats {
    /// Base score.
    pub value: u64,
    /// Collision radius (a singularity's grows as it eats).
    pub radius: f32,
    /// Hits it takes (a singularity gains more as it eats).
    pub hp: f32,
    /// The speed it lands with, in a random direction.
    pub launch: f32,
    /// Its own speed: cruising, or hunting.
    pub speed: f32,
    /// How fast it turns towards its target, in radians per second.
    pub turn_rate: f32,
    /// The range its spin is drawn from when it lands (equal ends draw nothing).
    pub spin: [f32; 2],
    /// Its spin while hunting (the evader and the serpent turn to face their way instead).
    pub hunting_spin: f32,
    /// The part of a shot's velocity a hit gives it (the heavy ones are pushed back).
    pub knockback: f32,
    /// How big its death is, for the effects (a singularity's grows with what it ate).
    pub size: f32,
    /// Shards a scored kill drops for sure (a singularity drops one more per mass eaten).
    pub shards: usize,
    /// The chance of one more shard.
    pub extra_shard: f32,
    /// Whether a singularity can eat it.
    pub edible: bool,
    /// Seconds of warning before it lands.
    pub warning: f32,
}

/// The common numbers, which the families change.
const BASE: Stats = Stats {
    value: 0,
    radius: 0.0,
    hp: 1.0,
    launch: 0.0,
    speed: 0.0,
    turn_rate: 0.0,
    spin: [0.0, 0.0],
    hunting_spin: 9.0,
    knockback: 0.0,
    size: 1.0,
    shards: 1,
    extra_shard: 0.35,
    edible: true,
    warning: 0.8,
};

impl Kind {
    /// The family's numbers.
    pub const fn stats(self) -> &'static Stats {
        match self {
            Kind::Drifter => &Stats {
                value: 50,
                radius: 0.62,
                launch: 2.2,
                speed: 2.2,
                spin: [-2.0, 2.0],
                ..BASE
            },
            Kind::Chaser => &Stats {
                value: 100,
                radius: 0.58,
                speed: 6.2,
                turn_rate: 2.6,
                hunting_spin: 3.0,
                ..BASE
            },
            Kind::Singularity => &Stats {
                value: 500,
                radius: 0.9,
                hp: 25.0,
                launch: 0.8,
                spin: [0.6, 0.6],
                knockback: 0.004,
                size: 2.0,
                shards: 8,
                extra_shard: 0.0,
                edible: false,
                warning: 1.4,
                ..BASE
            },
            Kind::Mote => &Stats {
                value: 25,
                radius: 0.32,
                launch: 6.0,
                speed: 8.5,
                turn_rate: 4.0,
                size: 0.5,
                shards: 0,
                extra_shard: 0.3,
                ..BASE
            },
            Kind::Evader => &Stats {
                value: 100,
                radius: 0.55,
                speed: 4.6,
                turn_rate: 3.0,
                hunting_spin: 0.0,
                ..BASE
            },
            Kind::Splitter => &Stats {
                value: 150,
                radius: 0.75,
                hp: 2.0,
                speed: 3.0,
                turn_rate: 1.4,
                spin: [2.0, 2.0],
                hunting_spin: 2.0,
                ..BASE
            },
            Kind::Fragment => &Stats {
                value: 50,
                radius: 0.34,
                launch: 6.0,
                speed: 8.0,
                turn_rate: 3.2,
                ..BASE
            },
            Kind::Serpent => &Stats {
                value: 400,
                radius: 0.55,
                hp: 4.0,
                speed: 5.2,
                turn_rate: 2.0,
                hunting_spin: 0.0,
                edible: false,
                warning: 1.4,
                ..BASE
            },
            Kind::Warden => &Stats {
                value: 250,
                radius: 0.8,
                speed: 2.4,
                turn_rate: 1.3,
                ..BASE
            },
            Kind::Carrier => &Stats {
                value: 600,
                radius: 1.15,
                hp: 14.0,
                launch: 1.4,
                speed: 1.4,
                spin: [0.7, 0.7],
                knockback: 0.004,
                edible: false,
                warning: 1.4,
                ..BASE
            },
        }
    }
}
