//! Offline renders for listening (`warp --music DIR`): two minutes of music at low, medium and
//! high intensity, and a run's arc with a death and a bomb, to WAV. The same engine as in the
//! game, without an audio device; the reverb is the same design as Firewheel's `FreeverbNode`.

use super::dsp::{Reverb, pan};
use super::music::{Controls, Music};
use super::sfx::{Family, Sfx, Trigger};
use std::path::Path;

const SR: f32 = 48000.0;
const BLOCK: usize = 512;

/// Render `seconds` of music for `seed`, with the intensity over time given by `intensity`, and
/// sound effects from `events` (at their times). Returns stereo samples.
pub fn render(
    seed: u64,
    seconds: f32,
    intensity: &dyn Fn(f32) -> f32,
    events: &[(f32, Event)],
) -> (Vec<f32>, Vec<f32>) {
    let mut music = Music::new(seed, SR);
    let mut sfx = Sfx::new(SR);
    let mut reverb = Reverb::new(SR);
    let n = (seconds * SR) as usize;
    let (mut out_l, mut out_r) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut ml, mut mr) = (vec![0.0f32; BLOCK], vec![0.0f32; BLOCK]);
    let (mut sl, mut sr) = (vec![0.0f32; BLOCK], vec![0.0f32; BLOCK]);
    let mut next = 0;
    let mut t = 0.0f32;
    while out_l.len() < n {
        while next < events.len() && events[next].0 <= t {
            match events[next].1 {
                Event::Death => {
                    music.death();
                    sfx.trigger(&Trigger {
                        family: Family::Death,
                        note: music.scale.note(7),
                        gains: pan(0.0),
                        gain: 1.0,
                        variant: 0.0,
                    });
                }
                Event::Bomb => {
                    music.bomb();
                    sfx.trigger(&Trigger {
                        family: Family::Bomb,
                        note: music.scale.note(0),
                        gains: pan(0.0),
                        gain: 1.0,
                        variant: 0.0,
                    });
                }
                Event::Sfx(tr) => sfx.trigger(&tr),
            }
            next += 1;
        }
        music.set(Controls {
            intensity: intensity(t),
            darkness: 0.0,
            playing: true,
            heat: intensity(t),
        });
        music.render(&mut ml, &mut mr);
        sl.fill(0.0);
        sr.fill(0.0);
        sfx.render(&mut sl, &mut sr);
        for i in 0..BLOCK {
            let (wl, wr) = reverb.tick(ml[i] * 0.5 + sl[i] * 0.22, mr[i] * 0.5 + sr[i] * 0.22);
            out_l.push(ml[i] + sl[i] + wl);
            out_r.push(mr[i] + sr[i] + wr);
        }
        t += BLOCK as f32 / SR;
    }
    out_l.truncate(n);
    out_r.truncate(n);
    (out_l, out_r)
}

/// Something that happens during an offline render.
#[derive(Clone, Copy, Debug)]
pub enum Event {
    /// The ship is destroyed.
    Death,
    /// A bomb.
    Bomb,
    /// A sound effect.
    Sfx(Trigger),
}

/// A 16-bit stereo WAV file, peak-limited softly.
pub fn wav(l: &[f32], r: &[f32]) -> Vec<u8> {
    let n = l.len().min(r.len());
    let mut out = Vec::with_capacity(44 + n * 4);
    let data = (n * 4) as u32;
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&(SR as u32).to_le_bytes());
    out.extend_from_slice(&(SR as u32 * 4).to_le_bytes());
    out.extend_from_slice(&4u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data.to_le_bytes());
    for (&a, &b) in l.iter().zip(r) {
        for x in [a, b] {
            let y = super::dsp::soft(x * 1.2) * 0.9;
            out.extend_from_slice(&((y * 32767.0) as i16).to_le_bytes());
        }
    }
    out
}

/// Write the review renders into `dir`.
pub fn run(args: &[String]) {
    let dir = args.first().map_or("music", String::as_str);
    let seconds: f32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(120.0);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(7);
    std::fs::create_dir_all(dir).expect("create the output directory");
    let scale = super::sfx::Scale::from_seed(seed);
    println!("seed {seed}: {} on MIDI {}", scale.name, scale.root);
    for (name, level) in [("low", 0.15f32), ("medium", 0.5), ("high", 0.9)] {
        let (l, r) = render(seed, seconds, &|_| level, &[]);
        let path = Path::new(dir).join(format!("music-{name}.wav"));
        std::fs::write(&path, wav(&l, &r)).expect("write WAV");
        println!("{}", path.display());
    }
    // A run's arc: calm, rising, a death at 45 s (drop to the drone, rebuild), a bomb at 80 s,
    // with shots and kills pitched to the scale.
    let mut events = vec![(45.0, Event::Death), (80.0, Event::Bomb)];
    let mut t = 5.0f32;
    let mut k = 0;
    while t < seconds {
        let pattern = [0, 2, 4, 2, 5, 4, 2, 1];
        events.push((
            t,
            Event::Sfx(Trigger {
                family: Family::Shot,
                note: scale.note(21 + pattern[k % 8]),
                gains: pan(0.1),
                gain: 1.0,
                variant: 0.0,
            }),
        ));
        if k % 9 == 4 {
            events.push((
                t,
                Event::Sfx(Trigger {
                    family: Family::Kill,
                    note: scale.note(14),
                    gains: pan(-0.4),
                    gain: 1.0,
                    variant: 1.0,
                }),
            ));
            events.push((
                t + 0.3,
                Event::Sfx(Trigger {
                    family: Family::Pickup,
                    note: scale.note(21 + (k as i32 / 9) % 14),
                    gains: pan(0.0),
                    gain: 1.0,
                    variant: 0.0,
                }),
            ));
        }
        k += 1;
        t += if (45.0..48.0).contains(&t) {
            3.0
        } else {
            1.0 / 11.0 * 2.0
        };
    }
    events.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (l, r) = render(seed, seconds, &|t| (0.1 + t / seconds).min(0.95), &events);
    let path = Path::new(dir).join("run-arc.wav");
    std::fs::write(&path, wav(&l, &r)).expect("write WAV");
    println!("{}", path.display());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::music::TEMPO;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Counts allocations made on a thread while its counter is armed.
    struct Counting;

    static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
    thread_local! {
        static ARMED: Cell<bool> = const { Cell::new(false) };
    }

    // SAFETY: forwards to the system allocator; only counts.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if ARMED.with(Cell::get) {
                ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            }
            // SAFETY: the caller's contract, passed on.
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: the caller's contract, passed on.
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    #[global_allocator]
    static COUNTING: Counting = Counting;

    /// The audio thread's rules: processing never allocates, however busy it gets. Everything
    /// reaches the engines as the nodes receive it, through the messages' bytes; for a stretch
    /// the effects wait for the beat (held, then released at the grid and when it is switched
    /// off); a new seed rebuilds the music in place; the volume glides.
    #[test]
    fn processing_does_not_allocate() {
        use crate::audio::dsp::Smooth;
        use crate::audio::msg::{MusicMsg, SfxMsg};
        let mut music = Music::new(5, SR);
        let mut sfx = Sfx::new(SR);
        let (mut l, mut r) = (vec![0.0f32; BLOCK], vec![0.0f32; BLOCK]);
        let mut volume = Smooth { value: 1.0 };
        let glide = Smooth::coefficient(0.02, SR);
        let to_sfx =
            |m: SfxMsg, sfx: &mut Sfx| SfxMsg::from_bytes(&m.to_bytes()).unwrap().apply(sfx);
        let to_music = |m: MusicMsg, music: &mut Music| {
            MusicMsg::from_bytes(&m.to_bytes()).unwrap().apply(music)
        };
        ARMED.with(|a| a.set(true));
        let before = ALLOCATIONS.load(Ordering::Relaxed);
        for block in 0..4000u32 {
            music.set(Controls {
                intensity: (block as f32 / 4000.0),
                darkness: if block % 700 < 100 { 0.8 } else { 0.0 },
                playing: true,
                heat: block as f32 / 4000.0,
            });
            if block % 500 == 250 {
                to_music(MusicMsg::Death, &mut music);
            }
            if block % 900 == 450 {
                to_music(MusicMsg::Bomb, &mut music);
            }
            // On the beat from block 1000 to 3000.
            sfx.set_on_beat((1000..3000).contains(&block));
            if block % 1000 == 0 {
                to_sfx(SfxMsg::Sync { tempo: TEMPO }, &mut sfx);
            }
            for (i, f) in Family::ALL.into_iter().enumerate() {
                if block % (3 + i as u32 * 7) == 0 {
                    let t = Trigger {
                        family: f,
                        note: 60.0 + i as f32,
                        gains: pan(0.0),
                        gain: 1.0,
                        variant: 1.0,
                    };
                    to_sfx(SfxMsg::Trigger(t), &mut sfx);
                }
            }
            music.render(&mut l, &mut r);
            sfx.render(&mut l, &mut r);
            volume.scale(
                &mut l,
                &mut r,
                if block % 200 < 100 { 1.0 } else { 0.3 },
                glide,
            );
        }
        to_music(
            MusicMsg::Seed {
                seed: 9,
                tempo: TEMPO,
            },
            &mut music,
        );
        music.render(&mut l, &mut r);
        let after = ALLOCATIONS.load(Ordering::Relaxed);
        ARMED.with(|a| a.set(false));
        assert_eq!(after - before, 0, "allocations while processing");
    }

    /// A seeded render is the same every time (replays reproduce the music).
    #[test]
    fn seeded_renders_are_deterministic() {
        let a = render(3, 4.0, &|t| t / 4.0, &[(1.0, Event::Bomb)]);
        let b = render(3, 4.0, &|t| t / 4.0, &[(1.0, Event::Bomb)]);
        assert!(a.0 == b.0 && a.1 == b.1);
        let c = render(4, 4.0, &|t| t / 4.0, &[(1.0, Event::Bomb)]);
        assert!(a.0 != c.0);
    }
}
