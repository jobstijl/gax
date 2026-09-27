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
                NodeEventType::CustomBytes(b) => {
                    if let Some(t) = Trigger::from_bytes(&b) {
                        self.engine.trigger(&t);
                    }
                }
                other => {
                    if let Some(p) = SfxNode::patch_event(&other) {
                        self.params.apply(p);
                    }
                }
            }
        }
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
                        // Music has no heap state: rebuilding it here does not allocate.
                        self.engine = Music::new(seed, self.sr);
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
    let sfx = commands.spawn(SfxNode { gain: 1.0 }).id();
    let music = commands
        .spawn(MusicNode {
            intensity: 0.1,
            darkness: 0.0,
            playing: false,
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
    mut sfx: Query<&mut AudioEvents, With<SfxNode>>,
    mut music: MusicNodes,
) {
    if let Ok(mut ev) = sfx.single_mut() {
        for t in sound.pending.drain(..) {
            ev.push(NodeEventType::CustomBytes(t.to_bytes()));
        }
    }
    if let Ok((mut node, mut ev)) = music.single_mut() {
        for m in sound.music_events.drain(..) {
            ev.push(NodeEventType::CustomBytes(m));
        }
        let c = sound.controls;
        if node.intensity != c.intensity || node.darkness != c.darkness || node.playing != c.playing
        {
            node.intensity = c.intensity;
            node.darkness = c.darkness;
            node.playing = c.playing;
        }
    }
    sound.pending.clear();
    sound.music_events.clear();
}

/// The game's side of the sound: turns simulation events into synthesizer triggers, placed and
/// pitched, and steers the music.
#[derive(Resource)]
pub struct Sound {
    pending: Vec<Trigger>,
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
            pending: Vec::new(),
            music_events: Vec::new(),
            controls: Controls {
                intensity: 0.1,
                darkness: 0.0,
                playing: false,
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

    fn push(&mut self, family: Family, note: f32, p: [f32; 2], cam: Pose, gain: f32, variant: f32) {
        // Into the camera's frame with gax: its x gives the pan, its distance the gain.
        let local = (cam << Point::xy(p[0], p[1])).to_euclidean();
        let pan = (local[0] / 30.0).clamp(-1.0, 1.0) * 0.8;
        let d2 = local[0] * local[0] + local[1] * local[1];
        let fall = 1.0 / (1.0 + d2 / 1600.0);
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

    /// React to a tick's events.
    pub fn on_events(&mut self, events: &[Event], w: &World, cam: Pose) {
        let s = self.scale;
        // Higher multipliers lift the effects by scale degrees.
        let lift = (w.mult.min(40) / 5) as i32;
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
                } => {
                    let gain = if scored { 1.0 } else { 0.6 };
                    match kind {
                        Kind::Singularity => {
                            self.push(Family::KillBig, s.note(0), pos, cam, 1.0, size)
                        }
                        _ => {
                            let d = match kind {
                                Kind::Drifter => 11,
                                Kind::Chaser => 14,
                                _ => 18,
                            } + lift;
                            self.push(Family::Kill, s.note(d), pos, cam, gain, size);
                        }
                    }
                }
                Event::Wall { pos } => self.push(Family::Wall, s.note(28), pos, cam, 1.0, 0.0),
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
                Event::Extra { .. } => {
                    self.push(Family::Extra, s.note(14), w.ship.body.xy(), cam, 1.0, 0.0)
                }
                Event::Respawn => self.push(Family::Spawn, s.note(14), [0.0, 0.0], cam, 1.5, 0.0),
                Event::GameOver => {}
            }
        }
    }

    fn music_event(&mut self, kind: u8) {
        let mut b = [0u8; 36];
        b[0] = kind;
        self.music_events.push(b);
    }

    /// Per frame: steer the music, and start a new run's music when the seed changes.
    pub fn update(&mut self, w: &World, playing: bool, _cam: Pose) {
        if playing && w.seed != self.seed {
            self.seed = w.seed;
            self.scale = Scale::from_seed(w.seed);
            let mut b = [0u8; 36];
            b[0] = MUSIC_SEED;
            b[4..12].copy_from_slice(&w.seed.to_le_bytes());
            self.music_events.push(b);
        }
        let ship = w.ship.body.xy();
        let darkness = w
            .wells()
            .map(|(p, _)| {
                let d = ((p[0] - ship[0]).powi(2) + (p[1] - ship[1]).powi(2)).sqrt();
                (1.0 - d / 14.0).clamp(0.0, 1.0)
            })
            .fold(0.0f32, f32::max);
        let alive = w.phase == Phase::Playing;
        self.controls = Controls {
            intensity: if alive {
                w.director.intensity
            } else {
                w.director.intensity * 0.5
            },
            darkness,
            playing,
        };
        if self.music_events.len() > 64 {
            self.music_events.clear();
        }
    }
}
