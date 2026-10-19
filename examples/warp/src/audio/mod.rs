//! Audio: the game's own synthesizer on the Firewheel graph, through bevy_seedling.
//!
//! The graph:
//!
//! ```text
//! SfxNode ──► sfx bus ─────┐
//!    └──► send ──► Freeverb ┤
//! MusicNode ─► music bus ──┤──► MainBus ──► Limiter ──► output
//!    └──► send ──► Freeverb ┘
//! ```
//!
//! Positional sound is computed with gax, not seedling's spatial nodes: each source is taken
//! into the camera's frame (`cam << p`), which gives its pan and distance gain.
//!
//! The DSP (`dsp`, `sfx`, `music`) is plain Rust that never allocates while rendering; the
//! nodes here only wrap it. That keeps it testable, and renderable offline to WAV. The game
//! talks to the nodes in fixed-size messages (`msg`).

pub mod dsp;
pub mod msg;
pub mod music;
pub mod offline;
pub mod sfx;
pub mod tunnel;

use crate::sim::body::Pose;
use crate::sim::{Event, Kind, World};
use bevy::prelude::*;
use bevy_seedling::firewheel::StreamInfo;
use bevy_seedling::firewheel::channel_config::{ChannelConfig, ChannelCount};
use bevy_seedling::firewheel::diff::{Diff, EventQueue, Patch};
use bevy_seedling::firewheel::event::{NodeEventType, ProcEvents};
use bevy_seedling::firewheel::node::{
    AudioNode, AudioNodeInfo, AudioNodeProcessor, ConstructProcessorContext, EmptyConfig,
    NodeError, ProcBuffers, ProcExtra, ProcInfo, ProcStreamCtx, ProcessStatus,
};
// Firewheel's derive macros name `firewheel_core`; it is seedling's re-export, not a separate
// dependency (a direct `firewheel` would risk a second copy of the engine).
use bevy_seedling::firewheel::core as firewheel_core;
use bevy_seedling::node::events::AudioEvents;
use bevy_seedling::prelude::*;
use dsp::Smooth;
use gax::pga2d::Point;
use msg::{MusicMsg, SfxMsg};
use music::{Controls, Music};
use sfx::{Family, Scale, Sfx, Trigger};

/// The sound-effect synthesizer node.
#[derive(Diff, Patch, Debug, Clone, Component)]
pub struct SfxNode {
    /// Output gain.
    pub gain: f32,
    /// Snap musical effects to the music's grid.
    pub on_beat: bool,
}

/// The music node.
#[derive(Diff, Patch, Debug, Clone, Component)]
pub struct MusicNode {
    /// Intensity, `0..1`.
    pub intensity: f32,
    /// Singularity proximity, `0..1`.
    pub darkness: f32,
    /// A run is being played.
    pub playing: bool,
    /// The multiplier's heat (`music::heat`).
    pub heat: f32,
    /// Output gain.
    pub gain: f32,
}

/// How long a volume change takes (the time constant of the nodes' gain), in seconds.
const GAIN_GLIDE: f32 = 0.02;

fn stereo_out(name: &'static str) -> AudioNodeInfo {
    AudioNodeInfo::new()
        .debug_name(name)
        .channel_config(ChannelConfig {
            num_inputs: ChannelCount::ZERO,
            num_outputs: ChannelCount::STEREO,
        })
}

impl AudioNode for SfxNode {
    type Configuration = EmptyConfig;

    fn info(&self, _: &EmptyConfig) -> Result<AudioNodeInfo, NodeError> {
        Ok(stereo_out("warp sfx"))
    }

    fn construct_processor(
        &self,
        _: &EmptyConfig,
        cx: ConstructProcessorContext,
    ) -> Result<impl AudioNodeProcessor, NodeError> {
        let sr = cx.stream_info.sample_rate.get() as f32;
        Ok(SfxProcessor {
            engine: Sfx::new(sr),
            params: self.clone(),
            gain: Smooth { value: self.gain },
            glide: Smooth::coefficient(GAIN_GLIDE, sr),
        })
    }
}

struct SfxProcessor {
    engine: Sfx,
    params: SfxNode,
    /// The output gain, gliding to `params.gain`.
    gain: Smooth,
    /// The gain's smoothing coefficient at the stream's rate.
    glide: f32,
}

impl AudioNodeProcessor for SfxProcessor {
    fn events(&mut self, _: &ProcInfo, events: &mut ProcEvents, _: &mut ProcExtra) {
        for e in events.drain() {
            match e {
                NodeEventType::CustomBytes(b) => {
                    if let Some(m) = SfxMsg::from_bytes(&b) {
                        m.apply(&mut self.engine);
                    }
                }
                other => {
                    if let Some(p) = SfxNode::patch_event(&other) {
                        self.params.apply(p);
                    }
                }
            }
        }
        self.engine.set_on_beat(self.params.on_beat);
    }

    fn process(
        &mut self,
        _: &ProcInfo,
        ProcBuffers { outputs, .. }: ProcBuffers,
        _: &mut ProcExtra,
    ) -> ProcessStatus {
        let (l, r) = outputs.split_at_mut(1);
        let (l, r) = (&mut *l[0], &mut *r[0]);
        l.fill(0.0);
        r.fill(0.0);
        if !self.engine.render(l, r) {
            // Silence: the gain can jump to where it is going.
            self.gain.value = self.params.gain;
            return ProcessStatus::ClearAllOutputs;
        }
        self.gain.scale(l, r, self.params.gain, self.glide);
        ProcessStatus::OutputsModified
    }

    fn new_stream(&mut self, info: &StreamInfo, _: &mut ProcStreamCtx) {
        let sr = info.sample_rate.get() as f32;
        self.engine.set_sample_rate(sr);
        self.glide = Smooth::coefficient(GAIN_GLIDE, sr);
    }
}

impl AudioNode for MusicNode {
    type Configuration = EmptyConfig;

    fn info(&self, _: &EmptyConfig) -> Result<AudioNodeInfo, NodeError> {
        Ok(stereo_out("warp music"))
    }

    fn construct_processor(
        &self,
        _: &EmptyConfig,
        cx: ConstructProcessorContext,
    ) -> Result<impl AudioNodeProcessor, NodeError> {
        let sr = cx.stream_info.sample_rate.get() as f32;
        Ok(MusicProcessor {
            engine: Music::new(1, sr),
            params: self.clone(),
            gain: Smooth { value: self.gain },
            glide: Smooth::coefficient(GAIN_GLIDE, sr),
        })
    }
}

struct MusicProcessor {
    engine: Music,
    params: MusicNode,
    /// The output gain, gliding to `params.gain`.
    gain: Smooth,
    /// The gain's smoothing coefficient at the stream's rate.
    glide: f32,
}

impl AudioNodeProcessor for MusicProcessor {
    fn events(&mut self, _: &ProcInfo, events: &mut ProcEvents, _: &mut ProcExtra) {
        for e in events.drain() {
            match e {
                NodeEventType::CustomBytes(b) => {
                    if let Some(m) = MusicMsg::from_bytes(&b) {
                        m.apply(&mut self.engine);
                    }
                }
                other => {
                    if let Some(p) = MusicNode::patch_event(&other) {
                        self.params.apply(p);
                    }
                }
            }
        }
        self.engine.set(Controls {
            intensity: self.params.intensity,
            darkness: self.params.darkness,
            playing: self.params.playing,
            heat: self.params.heat,
        });
    }

    fn process(
        &mut self,
        _: &ProcInfo,
        ProcBuffers { outputs, .. }: ProcBuffers,
        _: &mut ProcExtra,
    ) -> ProcessStatus {
        let (l, r) = outputs.split_at_mut(1);
        let (l, r) = (&mut *l[0], &mut *r[0]);
        self.engine.render(l, r);
        self.gain.scale(l, r, self.params.gain, self.glide);
        ProcessStatus::OutputsModified
    }

    fn new_stream(&mut self, info: &StreamInfo, _: &mut ProcStreamCtx) {
        let sr = info.sample_rate.get() as f32;
        self.engine.set_sample_rate(sr);
        self.glide = Smooth::coefficient(GAIN_GLIDE, sr);
    }
}

/// The plugin: seedling with an empty graph template, our nodes, and the graph.
pub struct AudioPlugin;

impl Plugin for AudioPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(AudioGraphTemplate::Empty)
            .add_plugins(SeedlingPlugins)
            .register_node::<SfxNode>()
            .register_node::<MusicNode>()
            .insert_resource(Sound::new())
            .add_systems(Startup, build_graph)
            .add_systems(
                PostStartup,
                add_events.after(SeedlingStartupSystems::StreamInitialization),
            )
            .add_systems(Last, flush);
    }
}

fn build_graph(mut commands: Commands) {
    // `WARP_MUTE` keeps the whole engine running with the master at silence (smoke tests).
    let master = if std::env::var_os("WARP_MUTE").is_some() {
        Volume::SILENT
    } else {
        Volume::UNITY_GAIN
    };
    commands
        .spawn((
            MainBus,
            VolumeNode {
                volume: master,
                ..Default::default()
            },
        ))
        .chain_node(LimiterNode::new(0.003, 0.15))
        .connect(AudioGraphOutput);
    let reverb = commands
        .spawn(FreeverbNode {
            room_size: 0.88,
            damping: 0.45,
            width: 1.0,
            ..Default::default()
        })
        .id();
    let sfx_bus = commands.spawn(VolumeNode::default()).id();
    let music_bus = commands.spawn(VolumeNode::default()).id();
    let send = |level: f32| VolumeNode {
        volume: Volume::Linear(level),
        ..Default::default()
    };
    let sfx_send = commands.spawn(send(0.22)).id();
    let music_send = commands.spawn(send(0.5)).id();
    let sfx = commands
        .spawn(SfxNode {
            gain: 1.0,
            on_beat: false,
        })
        .id();
    let music = commands
        .spawn(MusicNode {
            intensity: 0.1,
            darkness: 0.0,
            playing: false,
            heat: 0.0,
            gain: 1.0,
        })
        .id();
    for (from, to) in [
        (reverb, None),
        (sfx_bus, None),
        (music_bus, None),
        (sfx_send, Some(reverb)),
        (music_send, Some(reverb)),
        (sfx, Some(sfx_bus)),
        (sfx, Some(sfx_send)),
        (music, Some(music_bus)),
        (music, Some(music_send)),
    ] {
        match to {
            Some(t) => {
                commands.entity(from).connect(t);
            }
            None => {
                commands.entity(from).connect(MainBus);
            }
        }
    }
}

type SynthNodes<'w, 's> = Query<'w, 's, Entity, Or<(With<SfxNode>, With<MusicNode>)>>;

fn add_events(time: Res<Time<Audio>>, nodes: SynthNodes, mut commands: Commands) {
    for e in &nodes {
        commands.entity(e).insert(AudioEvents::new(&time));
    }
}

type MusicNodes<'w, 's> =
    Query<'w, 's, (&'static mut MusicNode, &'static mut AudioEvents), Without<SfxNode>>;

/// Send what the game queued to the audio nodes. A queue is emptied only once it has been
/// sent: while a node is missing its messages wait (`Sound` bounds the queues).
fn flush(
    mut sound: ResMut<Sound>,
    mut sfx: Query<(&mut SfxNode, &mut AudioEvents)>,
    mut music: MusicNodes,
) {
    let sound = &mut *sound;
    if let Ok((mut node, mut ev)) = sfx.single_mut() {
        for m in sound.to_sfx.drain(..) {
            ev.push(NodeEventType::CustomBytes(m.to_bytes()));
        }
        if node.gain != sound.gains[0] {
            node.gain = sound.gains[0];
        }
        if node.on_beat != sound.on_beat {
            node.on_beat = sound.on_beat;
        }
    }
    if let Ok((mut node, mut ev)) = music.single_mut() {
        if node.gain != sound.gains[1] {
            node.gain = sound.gains[1];
        }
        for m in sound.to_music.drain(..) {
            ev.push(NodeEventType::CustomBytes(m.to_bytes()));
        }
        let c = sound.controls;
        if node.intensity != c.intensity
            || node.darkness != c.darkness
            || node.playing != c.playing
            || node.heat != c.heat
        {
            node.intensity = c.intensity;
            node.darkness = c.darkness;
            node.playing = c.playing;
            node.heat = c.heat;
        }
    }
}

/// The game's side of the sound: turns simulation events into synthesizer triggers, placed and
/// pitched, and steers the music. Without an audio engine (headless runs) it keeps the same
/// bookkeeping and is never flushed.
#[derive(Resource)]
pub struct Sound {
    /// Output gains of the effects and the music (the volume settings).
    pub gains: [f32; 2],
    /// Effects on the music's beat (a setting).
    pub on_beat: bool,
    /// Messages waiting for the sound-effect node, in order.
    to_sfx: Vec<SfxMsg>,
    /// Messages waiting for the music node, in order.
    to_music: Vec<MusicMsg>,
    controls: Controls,
    scale: Scale,
    shots: u32,
    rng: dsp::Noise,
}

/// The detune generator's seed (mixed with a run's).
const DETUNE_SEED: u64 = 0x2468_ace1;
/// How many messages may wait for a node that is not there (headless runs) before the oldest
/// half goes.
const QUEUE: usize = 256;

impl Default for Sound {
    fn default() -> Sound {
        Sound::new()
    }
}

impl Sound {
    /// A new sound state.
    pub fn new() -> Sound {
        Sound {
            gains: [1.0, 1.0],
            on_beat: false,
            to_sfx: Vec::new(),
            to_music: Vec::new(),
            controls: Controls {
                intensity: 0.1,
                darkness: 0.0,
                playing: false,
                heat: 0.0,
            },
            scale: Scale::from_seed(1),
            shots: 0,
            rng: dsp::Noise::new(DETUNE_SEED),
        }
    }

    /// A run starts (a new run, a replay, a demo, whatever its seed): the music starts over
    /// from its first bar with the run's `seed` and `tempo`, the effects' beat grid with it,
    /// and the effects take the run's scale and detune, so that a replay sounds as the run did.
    pub fn begin_run(&mut self, seed: u64, tempo: f32) {
        self.scale = Scale::from_seed(seed);
        self.shots = 0;
        self.rng = dsp::Noise::new(DETUNE_SEED ^ seed);
        // Whatever the last run's music still had coming is moot.
        self.to_music.clear();
        self.to_music.push(MusicMsg::Seed { seed, tempo });
        self.to_sfx.retain(|m| !matches!(m, SfxMsg::Sync { .. }));
        self.to_sfx.push(SfxMsg::Sync { tempo });
    }

    fn push(
        &mut self,
        family: Family,
        note: f32,
        p: Point<(), f32>,
        cam: Pose,
        gain: f32,
        variant: f32,
    ) {
        // Into the listener's frame with gax: `x` across, `y` ahead. The ears sit `EAR` above
        // the plane of play, so a source is heard in the direction `(x, |(y, EAR)|)`, whose
        // equal-power gains `toward` reads off the rotor that turns left towards it.
        const EAR: f32 = 12.0;
        let off = (cam << p) - crate::geom::ORIGIN;
        let ahead = Point::direction(off.e01(), EAR).ideal_norm();
        let gains = dsp::toward(Point::direction(off.e20(), ahead));
        let d = off.ideal_norm();
        let fall = 1.0 / (1.0 + d * d / 1600.0);
        let detune = self.rng.next() * 0.08;
        self.to_sfx.push(SfxMsg::Trigger(Trigger {
            family,
            note: note + detune,
            gains,
            gain: gain * fall,
            variant,
        }));
        if self.to_sfx.len() > QUEUE {
            // Drop the oldest half of the sounds, never the beat grid's sync.
            let mut excess = QUEUE / 2;
            self.to_sfx.retain(|m| {
                let gone = excess > 0 && matches!(m, SfxMsg::Trigger(_));
                excess -= usize::from(gone);
                !gone
            });
        }
    }

    /// React to a Plane tick's events.
    pub fn on_events(&mut self, events: &[Event], w: &World, cam: Pose) {
        self.play(events, w.ledger.mult, w.ship.body.pos(), cam);
    }

    /// Play events (the Tunnel maps its own onto these), with the multiplier and the ship's
    /// position, placed through the listener `cam`.
    pub fn play(&mut self, events: &[Event], mult: u32, ship: Point<(), f32>, cam: Pose) {
        let s = self.scale;
        // Higher multipliers lift the effects by scale degrees.
        let lift = (mult.min(40) / 5) as i32;
        for e in events {
            match *e {
                Event::Fire { pos, .. } => {
                    self.shots = self.shots.wrapping_add(1);
                    let pattern = [0, 2, 4, 2, 5, 4, 2, 1];
                    let d = 21 + lift + pattern[(self.shots / 2) as usize % pattern.len()];
                    self.push(Family::Shot, s.note(d), pos, cam, 1.0, 0.0);
                }
                Event::Hit { pos, kind } => {
                    let d = if kind == Kind::Singularity {
                        7
                    } else {
                        16 + lift
                    };
                    self.push(Family::Hit, s.note(d), pos, cam, 1.0, 0.0);
                }
                Event::Kill {
                    pos,
                    kind,
                    size,
                    scored,
                    ..
                } => {
                    let gain = if scored { 1.0 } else { 0.6 };
                    match kind {
                        Kind::Singularity | Kind::Carrier => {
                            self.push(Family::KillBig, s.note(0), pos, cam, 1.0, size)
                        }
                        _ => {
                            // Each family has its register (scale degrees).
                            let d = match kind {
                                Kind::Serpent => 7,
                                Kind::Warden => 9,
                                Kind::Drifter => 11,
                                Kind::Splitter => 12,
                                Kind::Chaser => 14,
                                Kind::Evader => 16,
                                Kind::Fragment => 19,
                                _ => 18,
                            } + lift;
                            self.push(Family::Kill, s.note(d), pos, cam, gain, size);
                        }
                    }
                }
                Event::Wall { pos } => self.push(Family::Wall, s.note(28), pos, cam, 1.0, 0.0),
                Event::Deflect { pos, .. } => {
                    self.push(Family::Hit, s.note(25 + lift), pos, cam, 0.7, 0.0)
                }
                Event::Absorb { pos, .. } => {
                    self.push(Family::Absorb, s.note(7), pos, cam, 1.0, 0.0)
                }
                Event::Burst { pos } => self.push(Family::KillBig, s.note(0), pos, cam, 1.0, 3.0),
                Event::Warn { pos, kind } => {
                    let d = if kind == Kind::Singularity { 4 } else { 18 };
                    self.push(Family::Warn, s.note(d), pos, cam, 1.0, 0.0);
                }
                Event::Spawn { pos, kind } => {
                    let d = if kind == Kind::Singularity { 7 } else { 20 };
                    self.push(Family::Spawn, s.note(d), pos, cam, 1.0, 0.0);
                }
                Event::Pickup { pos, mult } => {
                    // The multiplier climbs the scale.
                    let d = 21 + (mult as i32 % 14);
                    self.push(Family::Pickup, s.note(d), pos, cam, 1.0, 0.0);
                }
                Event::Death { pos } => {
                    self.push(Family::Death, s.note(7), pos, cam, 1.0, 0.0);
                    self.music_event(MusicMsg::Death);
                }
                Event::Bomb { pos } => {
                    self.push(Family::Bomb, s.note(0), pos, cam, 1.0, 0.0);
                    self.music_event(MusicMsg::Bomb);
                }
                Event::Extra { .. } => self.push(Family::Extra, s.note(14), ship, cam, 1.0, 0.0),
                Event::Respawn => self.push(
                    Family::Spawn,
                    s.note(14),
                    crate::geom::ORIGIN,
                    cam,
                    1.5,
                    0.0,
                ),
                Event::GameOver => {}
            }
        }
    }

    fn music_event(&mut self, m: MusicMsg) {
        if self.to_music.len() >= QUEUE {
            // Nobody is listening (headless): only a run's start must not be lost.
            self.to_music.retain(|m| matches!(m, MusicMsg::Seed { .. }));
        }
        self.to_music.push(m);
    }

    /// Steer the music within a run: intensity, a singularity's darkness, the multiplier, and
    /// whether a run is being played. A run's start is `begin_run`.
    pub fn steer(&mut self, intensity: f32, darkness: f32, mult: u32, playing: bool) {
        self.controls = Controls {
            intensity,
            darkness,
            playing,
            heat: music::heat(mult),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::at;
    use crate::sim::body::Pose;

    /// Every message the game queues, through its bytes into the engines.
    fn deliver(sound: &mut Sound, sfx: &mut Sfx, music: &mut Music) {
        for m in sound.to_sfx.drain(..) {
            SfxMsg::from_bytes(&m.to_bytes()).unwrap().apply(sfx);
        }
        for m in sound.to_music.drain(..) {
            MusicMsg::from_bytes(&m.to_bytes()).unwrap().apply(music);
        }
    }

    /// A run's start restarts the music and the beat grid, also when the seed is the last
    /// run's (playing a run again, or a second demo); the game's events reach the engines as
    /// the sounds and the music's reactions they stand for.
    #[test]
    fn the_game_drives_the_engines_through_the_bytes() {
        let sr = 48000.0;
        let (mut sound, mut sfx, mut music) = (Sound::new(), Sfx::new(sr), Music::new(1, sr));
        sfx.set_on_beat(true);
        let (mut l, mut r) = (vec![0.0f32; 512], vec![0.0f32; 512]);
        for again in [false, true] {
            sound.begin_run(42, music::TUNNEL_TEMPO);
            sound.steer(0.6, 0.0, 1, true);
            for _ in 0..300 {
                music.render(&mut l, &mut r);
                sfx.render(&mut l, &mut r);
            }
            if again {
                // The same seed again: the music is far into the first run.
                assert!(music.step > 0);
            }
            deliver(&mut sound, &mut sfx, &mut music);
            assert_eq!(music.step, 0, "the music starts over (again: {again})");
            assert_eq!(music.tempo, music::TUNNEL_TEMPO);
            assert_eq!(music.scale, Scale::from_seed(42));
            assert_eq!(sfx.grid(), (music::TUNNEL_TEMPO, Some(0.0)));
            music.render(&mut l, &mut r);
        }
        // A death and a bomb: their sounds at once, and the music's reactions.
        let o = at(0.0, 0.0);
        sound.play(
            &[Event::Death { pos: o }, Event::Bomb { pos: at(30.0, 0.0) }],
            1,
            o,
            Pose::identity(),
        );
        deliver(&mut sound, &mut sfx, &mut music);
        assert_eq!(sfx.active_in(Family::Death), 1);
        assert_eq!(sfx.active_in(Family::Bomb), 1);
        assert_eq!(music.recovery(), (0.0, 0.35));
        // A shot waits for the grid's next 32nd.
        sound.play(
            &[Event::Fire {
                pos: o,
                dir: Point::direction(0.0, 1.0),
            }],
            1,
            o,
            Pose::identity(),
        );
        deliver(&mut sound, &mut sfx, &mut music);
        assert_eq!(sfx.active_in(Family::Shot), 0);
        sfx.render(&mut [0.0; 4096], &mut [0.0; 4096]);
        assert_eq!(sfx.active_in(Family::Shot), 1);
    }

    /// A source is heard from its side, at equal power, and a source to the right is louder
    /// on the right.
    #[test]
    fn sources_are_placed_by_direction() {
        let mut sound = Sound::new();
        let o = at(0.0, 0.0);
        for x in [-20.0, 0.0, 20.0] {
            sound.play(&[Event::Wall { pos: at(x, 5.0) }], 1, o, Pose::identity());
        }
        let gains: Vec<(f32, f32)> = sound
            .to_sfx
            .iter()
            .map(|m| match m {
                SfxMsg::Trigger(t) => t.gains,
                SfxMsg::Sync { .. } => unreachable!(),
            })
            .collect();
        for &(l, r) in &gains {
            assert!((l * l + r * r - 1.0).abs() < 1e-5);
        }
        assert!(gains[0].0 > gains[0].1 && gains[2].1 > gains[2].0);
        assert!((gains[1].0 - gains[1].1).abs() < 1e-6);
    }

    /// Without a node to send to, the queues stay bounded and keep a run's start.
    #[test]
    fn unsent_queues_stay_bounded() {
        let mut sound = Sound::new();
        sound.begin_run(3, music::TEMPO);
        let o = at(0.0, 0.0);
        for _ in 0..2000 {
            sound.play(&[Event::Death { pos: o }], 1, o, Pose::identity());
        }
        assert!(sound.to_sfx.len() <= QUEUE && sound.to_music.len() <= QUEUE);
        assert!(sound.to_sfx.contains(&SfxMsg::Sync {
            tempo: music::TEMPO
        }));
        assert!(sound.to_music.contains(&MusicMsg::Seed {
            seed: 3,
            tempo: music::TEMPO
        }));
    }
}
