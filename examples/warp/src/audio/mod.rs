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
//! nodes here only wrap it. That keeps it testable, and renderable offline to WAV.

pub mod dsp;
pub mod music;
pub mod offline;
pub mod sfx;

use crate::sim::body::Pose;
use crate::sim::{Event, Kind, Phase, World};
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
use gax::pga2d::Point;
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

/// The sfx node's sync event (`CustomBytes`, byte 0): the music starts its bar.
const SFX_SYNC: u8 = 255;

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

/// Events for the music node (`CustomBytes`, byte 0).
const MUSIC_DEATH: u8 = 1;
const MUSIC_BOMB: u8 = 2;
const MUSIC_SEED: u8 = 3;

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
        Ok(SfxProcessor {
            engine: Sfx::new(cx.stream_info.sample_rate.get() as f32),
            params: self.clone(),
        })
    }
}

struct SfxProcessor {
    engine: Sfx,
    params: SfxNode,
}

impl AudioNodeProcessor for SfxProcessor {
    fn events(&mut self, _: &ProcInfo, events: &mut ProcEvents, _: &mut ProcExtra) {
        for e in events.drain() {
            match e {
                NodeEventType::CustomBytes(b) if b[0] == SFX_SYNC => self.engine.sync(),
                NodeEventType::CustomBytes(b) => {
                    if let Some(t) = Trigger::from_bytes(&b) {
                        self.engine.submit(&t);
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
            return ProcessStatus::ClearAllOutputs;
        }
        let g = self.params.gain;
        for x in l.iter_mut().chain(r.iter_mut()) {
            *x *= g;
        }
        ProcessStatus::OutputsModified
    }

    fn new_stream(&mut self, info: &StreamInfo, _: &mut ProcStreamCtx) {
        self.engine.set_sample_rate(info.sample_rate.get() as f32);
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
        Ok(MusicProcessor {
            engine: Music::new(1, cx.stream_info.sample_rate.get() as f32),
            params: self.clone(),
            sr: cx.stream_info.sample_rate.get() as f32,
        })
    }
}

struct MusicProcessor {
    engine: Music,
    params: MusicNode,
    sr: f32,
}

impl AudioNodeProcessor for MusicProcessor {
    fn events(&mut self, _: &ProcInfo, events: &mut ProcEvents, _: &mut ProcExtra) {
        for e in events.drain() {
            match e {
                NodeEventType::CustomBytes(b) => match b[0] {
                    MUSIC_DEATH => self.engine.death(),
                    MUSIC_BOMB => self.engine.bomb(),
                    MUSIC_SEED => {
                        let seed =
                            u64::from_le_bytes([b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11]]);
                        // In place, keeping the echo's buffers: no allocation here.
                        self.engine.reseed(seed);
                    }
                    _ => {}
                },
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
        let g = self.params.gain;
        for x in l.iter_mut().chain(r.iter_mut()) {
            *x *= g;
        }
        ProcessStatus::OutputsModified
    }

    fn new_stream(&mut self, info: &StreamInfo, _: &mut ProcStreamCtx) {
        self.sr = info.sample_rate.get() as f32;
        self.engine.set_sample_rate(self.sr);
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

/// Send what the game queued to the audio nodes.
type MusicNodes<'w, 's> =
    Query<'w, 's, (&'static mut MusicNode, &'static mut AudioEvents), Without<SfxNode>>;

fn flush(
    mut sound: ResMut<Sound>,
    mut sfx: Query<(&mut SfxNode, &mut AudioEvents)>,
    mut music: MusicNodes,
) {
    if let Ok((mut node, mut ev)) = sfx.single_mut() {
        for b in sound.sfx_events.drain(..) {
            ev.push(NodeEventType::CustomBytes(b));
        }
        for t in sound.pending.drain(..) {
            ev.push(NodeEventType::CustomBytes(t.to_bytes()));
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
        for m in sound.music_events.drain(..) {
            ev.push(NodeEventType::CustomBytes(m));
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
    sound.pending.clear();
    sound.sfx_events.clear();
    sound.music_events.clear();
}

/// The game's side of the sound: turns simulation events into synthesizer triggers, placed and
/// pitched, and steers the music.
#[derive(Resource)]
pub struct Sound {
    /// Output gains of the effects and the music (the volume settings).
    pub gains: [f32; 2],
    /// Effects on the music's beat (a setting).
    pub on_beat: bool,
    pending: Vec<Trigger>,
    sfx_events: Vec<[u8; 36]>,
    music_events: Vec<[u8; 36]>,
    controls: Controls,
    scale: Scale,
    seed: u64,
    shots: u32,
    rng: dsp::Noise,
}

impl Sound {
    /// A new sound state.
    pub fn new() -> Sound {
        Sound {
            gains: [1.0, 1.0],
            on_beat: false,
            pending: Vec::new(),
            sfx_events: Vec::new(),
            music_events: Vec::new(),
            controls: Controls {
                intensity: 0.1,
                darkness: 0.0,
                playing: false,
                heat: 0.0,
            },
            scale: Scale::from_seed(1),
            seed: 1,
            shots: 0,
            rng: dsp::Noise(0x2468_ace1),
        }
    }

    /// A sound that plays nothing (headless runs): the same bookkeeping, never flushed.
    pub fn silent() -> Sound {
        Sound::new()
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
        // Into the camera's frame with gax: across gives the pan, the distance the gain.
        let off = (cam << p) - Point::xy(0.0, 0.0);
        let pan = (off.e20() / 30.0).clamp(-1.0, 1.0) * 0.8;
        let d = off.ideal_norm();
        let fall = 1.0 / (1.0 + d * d / 1600.0);
        let detune = self.rng.next() * 0.08;
        self.pending.push(Trigger {
            family,
            note: note + detune,
            pan,
            gain: gain * fall,
            variant,
        });
        if self.pending.len() > 256 {
            self.pending.drain(..128);
        }
    }

    /// React to a Plane tick's events.
    pub fn on_events(&mut self, events: &[Event], w: &World, cam: Pose) {
        self.play(events, w.mult, w.ship.body.pos(), cam);
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
                    self.music_event(MUSIC_DEATH);
                }
                Event::Bomb { pos } => {
                    self.push(Family::Bomb, s.note(0), pos, cam, 1.0, 0.0);
                    self.music_event(MUSIC_BOMB);
                }
                Event::Extra { .. } => self.push(Family::Extra, s.note(14), ship, cam, 1.0, 0.0),
                Event::Respawn => self.push(
                    Family::Spawn,
                    s.note(14),
                    Point::xy(0.0, 0.0),
                    cam,
                    1.5,
                    0.0,
                ),
                Event::GameOver => {}
            }
        }
    }

    fn music_event(&mut self, kind: u8) {
        let mut b = [0u8; 36];
        b[0] = kind;
        self.music_events.push(b);
    }

    /// Per frame, for the Plane: steer the music, and start a new run's music when the seed
    /// changes.
    pub fn update(&mut self, w: &World, playing: bool, _cam: Pose) {
        let ship = w.ship.body.pos();
        let darkness = w
            .wells()
            .map(|(p, _)| {
                let d = crate::sim::body::distance(p, ship);
                (1.0 - d / 14.0).clamp(0.0, 1.0)
            })
            .fold(0.0f32, f32::max);
        let alive = w.phase == Phase::Playing;
        let intensity = w.director.intensity * if alive { 1.0 } else { 0.5 };
        self.steer(w.seed, intensity, darkness, w.mult, playing);
    }

    /// Steer the music directly (the Tunnel): a run's seed, intensity, a singularity's
    /// darkness, the multiplier, and whether a run is being played.
    pub fn steer(&mut self, seed: u64, intensity: f32, darkness: f32, mult: u32, playing: bool) {
        if playing && seed != self.seed {
            self.seed = seed;
            self.scale = Scale::from_seed(seed);
            let mut b = [0u8; 36];
            b[0] = MUSIC_SEED;
            b[4..12].copy_from_slice(&seed.to_le_bytes());
            self.music_events.push(b);
            // The effects' beat grid starts with the music's first bar.
            let mut sync = [0u8; 36];
            sync[0] = SFX_SYNC;
            self.sfx_events.push(sync);
        }
        self.controls = Controls {
            intensity,
            darkness,
            playing,
            heat: music::heat(mult),
        };
        if self.music_events.len() > 64 {
            self.music_events.clear();
        }
    }
}
