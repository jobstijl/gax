//! The messages from the game to the synthesizers, and their one wire format.
//!
//! Each message travels to the audio thread as a Firewheel `CustomBytes` event: a fixed
//! `[u8; 36]`, so sending, decoding and applying one never allocates. The layouts live here
//! only (byte 0 is the message's tag, `0` is never one, multi-byte fields are little-endian):
//!
//! | message | bytes |
//! |---|---|
//! | `SfxMsg::Trigger` | tag 1, family at 1, note 4..8, gains 8..16, gain 16..20, variant 20..24 |
//! | `SfxMsg::Sync` | tag 2, tempo 4..8 |
//! | `MusicMsg::Death` | tag 1 |
//! | `MusicMsg::Bomb` | tag 2 |
//! | `MusicMsg::Seed` | tag 3, seed 4..12, tempo 12..16 |
//!
//! `apply` is what the nodes do with a message, kept apart from Firewheel so that a test can
//! follow one from the game through the bytes into an engine.

use super::music::Music;
use super::sfx::{Family, Sfx, Trigger};

/// The size of a Firewheel `CustomBytes` event.
pub const SIZE: usize = 36;

/// A message for the sound-effect node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SfxMsg {
    /// Play a sound.
    Trigger(Trigger),
    /// The music starts its first bar now, at `tempo`: align the beat grid with it.
    Sync {
        /// Beats per minute.
        tempo: f32,
    },
}

/// A message for the music node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MusicMsg {
    /// The ship was destroyed.
    Death,
    /// A bomb went off.
    Bomb,
    /// A run starts: the music starts over from its first bar.
    Seed {
        /// The run's seed.
        seed: u64,
        /// Beats per minute.
        tempo: f32,
    },
}

/// Little-endian fields in an event's bytes.
struct Wire([u8; SIZE]);

impl Wire {
    fn tagged(tag: u8) -> Wire {
        let mut b = [0; SIZE];
        b[0] = tag;
        Wire(b)
    }

    fn put(&mut self, at: usize, bytes: &[u8]) -> &mut Wire {
        self.0[at..at + bytes.len()].copy_from_slice(bytes);
        self
    }

    fn put_f32(&mut self, at: usize, x: f32) -> &mut Wire {
        self.put(at, &x.to_le_bytes())
    }

    fn f32(b: &[u8; SIZE], at: usize) -> f32 {
        f32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
    }

    fn u64(b: &[u8; SIZE], at: usize) -> u64 {
        let mut x = [0; 8];
        x.copy_from_slice(&b[at..at + 8]);
        u64::from_le_bytes(x)
    }
}

const SFX_TRIGGER: u8 = 1;
const SFX_SYNC: u8 = 2;
const MUSIC_DEATH: u8 = 1;
const MUSIC_BOMB: u8 = 2;
const MUSIC_SEED: u8 = 3;

impl SfxMsg {
    /// Encode as an event's bytes.
    pub fn to_bytes(self) -> [u8; SIZE] {
        match self {
            SfxMsg::Trigger(t) => {
                let mut w = Wire::tagged(SFX_TRIGGER);
                w.put(1, &[t.family as u8])
                    .put_f32(4, t.note)
                    .put_f32(8, t.gains.0)
                    .put_f32(12, t.gains.1)
                    .put_f32(16, t.gain)
                    .put_f32(20, t.variant);
                w.0
            }
            SfxMsg::Sync { tempo } => {
                let mut w = Wire::tagged(SFX_SYNC);
                w.put_f32(4, tempo);
                w.0
            }
        }
    }

    /// Decode an event's bytes (`None` for anything else).
    pub fn from_bytes(b: &[u8; SIZE]) -> Option<SfxMsg> {
        match b[0] {
            SFX_TRIGGER => Some(SfxMsg::Trigger(Trigger {
                family: Family::from_u8(b[1])?,
                note: Wire::f32(b, 4),
                gains: (Wire::f32(b, 8), Wire::f32(b, 12)),
                gain: Wire::f32(b, 16),
                variant: Wire::f32(b, 20),
            })),
            SFX_SYNC => Some(SfxMsg::Sync {
                tempo: Wire::f32(b, 4),
            }),
            _ => None,
        }
    }

    /// What the node does with it.
    #[inline]
    pub fn apply(self, sfx: &mut Sfx) {
        match self {
            SfxMsg::Trigger(t) => sfx.submit(&t),
            SfxMsg::Sync { tempo } => sfx.sync(tempo),
        }
    }
}

impl MusicMsg {
    /// Encode as an event's bytes.
    pub fn to_bytes(self) -> [u8; SIZE] {
        match self {
            MusicMsg::Death => Wire::tagged(MUSIC_DEATH).0,
            MusicMsg::Bomb => Wire::tagged(MUSIC_BOMB).0,
            MusicMsg::Seed { seed, tempo } => {
                let mut w = Wire::tagged(MUSIC_SEED);
                w.put(4, &seed.to_le_bytes()).put_f32(12, tempo);
                w.0
            }
        }
    }

    /// Decode an event's bytes (`None` for anything else).
    pub fn from_bytes(b: &[u8; SIZE]) -> Option<MusicMsg> {
        match b[0] {
            MUSIC_DEATH => Some(MusicMsg::Death),
            MUSIC_BOMB => Some(MusicMsg::Bomb),
            MUSIC_SEED => Some(MusicMsg::Seed {
                seed: Wire::u64(b, 4),
                tempo: Wire::f32(b, 12),
            }),
            _ => None,
        }
    }

    /// What the node does with it. A new seed rebuilds the music in place, keeping its echo's
    /// buffers: no allocation.
    #[inline]
    pub fn apply(self, music: &mut Music) {
        match self {
            MusicMsg::Death => music.death(),
            MusicMsg::Bomb => music.bomb(),
            MusicMsg::Seed { seed, tempo } => music.reseed(seed, tempo),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::dsp::pan;
    use crate::audio::music::{Controls, TEMPO, TUNNEL_TEMPO};
    use crate::audio::sfx::Scale;

    fn trigger(family: Family) -> Trigger {
        Trigger {
            family,
            note: 79.5,
            gains: pan(-0.3),
            gain: 0.8,
            variant: 2.0,
        }
    }

    /// Every message comes back from its bytes as it went in.
    #[test]
    fn every_message_round_trips_through_bytes() {
        for f in Family::ALL {
            let m = SfxMsg::Trigger(trigger(f));
            assert_eq!(SfxMsg::from_bytes(&m.to_bytes()), Some(m));
        }
        let m = SfxMsg::Sync { tempo: 132.5 };
        assert_eq!(SfxMsg::from_bytes(&m.to_bytes()), Some(m));
        for m in [
            MusicMsg::Death,
            MusicMsg::Bomb,
            MusicMsg::Seed {
                seed: 0xfedc_ba98_7654_3210,
                tempo: TUNNEL_TEMPO,
            },
        ] {
            assert_eq!(MusicMsg::from_bytes(&m.to_bytes()), Some(m));
        }
        // Zeros, an unknown tag and an unknown family are nothing.
        assert_eq!(SfxMsg::from_bytes(&[0; SIZE]), None);
        assert_eq!(MusicMsg::from_bytes(&[0; SIZE]), None);
        assert_eq!(MusicMsg::from_bytes(&[9; SIZE]), None);
        let mut b = SfxMsg::Trigger(trigger(Family::Shot)).to_bytes();
        b[1] = Family::ALL.len() as u8;
        assert_eq!(SfxMsg::from_bytes(&b), None);
    }

    /// A decoded message does to an engine what it says.
    #[test]
    fn decoded_messages_drive_the_engines() {
        let sr = 48000.0;
        let decode_sfx = |m: SfxMsg| SfxMsg::from_bytes(&m.to_bytes()).unwrap();
        let decode_music = |m: MusicMsg| MusicMsg::from_bytes(&m.to_bytes()).unwrap();
        let (mut l, mut r) = (vec![0.0f32; 1000], vec![0.0f32; 1000]);

        let mut sfx = Sfx::new(sr);
        decode_sfx(SfxMsg::Trigger(trigger(Family::Kill))).apply(&mut sfx);
        assert_eq!(sfx.active_in(Family::Kill), 1);
        // A sync restarts the beat grid at the new tempo.
        sfx.set_on_beat(true);
        sfx.render(&mut l, &mut r);
        assert_eq!(sfx.grid(), (TEMPO, Some(1000.0)));
        decode_sfx(SfxMsg::Sync {
            tempo: TUNNEL_TEMPO,
        })
        .apply(&mut sfx);
        assert_eq!(sfx.grid(), (TUNNEL_TEMPO, Some(0.0)));

        let mut music = Music::new(1, sr);
        music.set(Controls {
            intensity: 0.5,
            darkness: 0.0,
            playing: true,
            heat: 0.0,
        });
        for _ in 0..200 {
            music.render(&mut l, &mut r);
        }
        assert!(music.step > 0);
        decode_music(MusicMsg::Death).apply(&mut music);
        assert_eq!(music.recovery().0, 0.0);
        decode_music(MusicMsg::Bomb).apply(&mut music);
        assert_eq!(music.recovery().1, 0.35);
        decode_music(MusicMsg::Seed {
            seed: 6,
            tempo: TUNNEL_TEMPO,
        })
        .apply(&mut music);
        assert_eq!(music.step, 0);
        assert_eq!(music.tempo, TUNNEL_TEMPO);
        assert_eq!(music.scale, Scale::from_seed(6));
        assert_eq!(music.recovery(), (1.0, 1.0));
    }
}
