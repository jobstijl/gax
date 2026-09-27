# warp

A neon twin-stick shooter on warping space, and a real-load test of gax.

Everything that moves is gax:
* poses are unit PGA2D motors and velocities are twists;
* collisions are joins and signed distances;
* the camera follows on `log`/`exp`;
* the lattice of warped space and the particles are traced gax kernels, running in compute
  shaders with a CPU twin of the same programs;
* the shaders import gax's generated WGSL modules.

Bevy is plumbing only (the app, windows, input and gamepads). The renderer is the game's own
wgpu code, and the audio is its own synthesizer on the Firewheel graph through bevy_seedling.
[VERIFY.md](VERIFY.md) reports what the game found about gax.

## Play

```sh
cargo run --release
```

| | keyboard and mouse | gamepad |
|---|---|---|
| move | WASD / arrows | left stick |
| aim and fire | mouse, left button | right stick |
| bomb | space / right button | a trigger |
| pause | Esc | Select / B |
| start | Enter | Start / A |

Menus take arrows, WASD, the D-pad or a flick of the left stick. F3 toggles the debug overlay
(frame rate, simulation time, GPU time per pass, pose drift), and F11 toggles full screen.

**Settings** (from the title): screen shake in steps down to off, reduced flashes (softer
full-screen flashes, bursts and shock ripples), colour schemes that are safe for red–green and
for blue–yellow colour blindness (every enemy also has its own shape), volumes for the master,
music and effects, effects on the beat (shots, hits, kills and pickups wait for the music's
next 32nd, sample-accurately), and full screen.

**The music** is generated live from the run's seed: a pad with voice-led chords, bass and soft
percussion that follow the intensity, and layers the multiplier unlocks. Melodic fragments
(short cells on chord tones that repeat and develop, on an FM bell into a ping-pong echo) come
in from x4, texture (drifting air and high sparkles) from x10, and the fragments double an
octave up from x25. A replay plays its run's music again.

**High scores and replays.** Every run is recorded (its seed and every tick's input, about
0.8 KB a second). A run that makes the top ten asks for three initials and keeps its replay,
which the high-score table plays back (left and right change the speed). Settings, scores and
replays live in `~/.local/share/warp` (or the platform's equivalent, or `$WARP_DATA`); the last
run is always `replays/last.warp`.

**Rules.**
* Kills drop green shards. Collecting them raises the multiplier, which applies to every kill;
  losing a ship resets it.
* An extra life comes every 100 000 points, and an extra bomb every 150 000.
* A bomb clears the screen.
* Every spawn is announced by a warp-in, and never lands close to you.

**The enemies:**
* **drifters** (cyan) wander;
* **chasers** (magenta) home in with a limited turn rate;
* **singularities** (violet) are gravity wells: they bend your shots, the lattice, the
  particles and the other enemies. They eat what falls in and grow, and when overfed they burst
  into a swarm of motes. They take many hits.

Later in a run (the director unlocks them over the first minute and a half):
* **evaders** (yellow) read your line of fire and step out of it;
* **splitters** (orange) take two hits and break into three fast fragments;
* **wardens** (red) carry a shield that reflects shots back; hit them from behind;
* **serpents** (teal) wind toward you. The body blocks shots; only the head is vulnerable;
* **carriers** (blue) are slow and tough, launch motes as they go, and unload a ring of them
  when destroyed.

## Other modes

```sh
cargo run --release -- --shot DIR [SECONDS...]   # a scripted run, PNG snapshots (offscreen, wgpu validation on)
WARP_SCENE=roster cargo run --release -- --shot DIR   # every enemy kind in one still picture
WARP_SCENE=menus cargo run --release -- --shot DIR    # the menus
cargo run --release -- --music DIR [SECONDS] [SEED]  # the music offline: low, medium, high intensity, and a run's arc, as WAV
cargo run --release -- --smoke                   # play 8 s with a bot and quit (a startup test)
cargo run --release -- --replay FILE             # watch a replay
cargo run --release -- --verify FILE             # replay headless, checking every state hash
WARP_NO_AUDIO=1 cargo run --release              # without the audio engine
cargo test --release                             # the simulation, the audio, CPU against GPU kernels
./scripts/check-deps.sh                          # the dependency contract
```

## Rules for the code

* **The simulation (`src/sim`) is pure.** It has no Bevy or wgpu types, runs a fixed 120 Hz step
  from a seed, and a run replays from its seed and inputs. The simulation only ever sees
  quantized inputs (the replay's own encoding), so playback is exact on the same build; a hash
  of the world every second catches a divergence (`sim/replay.rs`, with tests, and one that
  plays through the whole game loop at uneven frame rates and watches the result).
* **Bevy is plumbing only.** `scripts/check-deps.sh` fails if `bevy_render`, `bevy_audio`,
  `bevy_ui`, `bevy_text`, `bevy_sprite`, `bevy_pbr`, `bevy_camera`, `bevy_mesh` or any other
  linear-algebra crate is in the tree. `bevy_image` is allowed, since `bevy_window` needs it for
  cursors.
* **Math goes through gax.** `clippy.toml` disallows glam's types and the methods that hand them
  out, as well as Bevy's `Transform` and seedling's spatial audio. The exception is `src/input.rs`,
  which turns stick and cursor vectors into gax types right away.
* **The audio thread does not allocate.** A test runs every synthesizer for 4000 blocks under a
  counting allocator.
