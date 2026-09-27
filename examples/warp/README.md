# warp

A neon twin-stick shooter on warping space, and a real-load test of gax.

Everything is gax:
* poses are unit PGA2D motors and velocities are twists;
* collisions are joins and signed distances, and the arena's walls are lines to reflect in;
* the camera follows on `log`/`exp`;
* the Tunnel's track is a chain of PGA3D screw motions, sampled by motor interpolation, and
  its camera is a PGA3D motor (`Motor::look_at`); the projection is homogeneous, and aiming
  goes back through it as a ray that meets the tunnel;
* colours are lights: homogeneous points in RGB space whose weight is their intensity, so
  adding them mixes them the way glow does (Grassmann's laws);
* the lattice of warped space, the particles, and the line renderer's geometry and light are
  traced gax kernels, running in shaders with a CPU twin of the same programs; the shaders
  import gax's generated WGSL modules;
* oscillators are phasors (a direction turned by a rotation motor every sample), and panning
  is a turn.

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

**Tunnel** (the second mode, a first slice): fly into the screen down a twisting tunnel of
warped space. The mouse (or right stick) places a reticle down the tunnel and the shots
converge on it; Q and E (or the bumpers) barrel-roll you around the tunnel with a moment of
invulnerability; Shift and Ctrl (or the triggers) boost and brake, and faster is worth more
points. Rings of drones close in around you (fly through their middle), mines drift in the
lane, and turrets on the wall fire bolts that grow and glow as they come. Your shadow on the
wall shows where you are. The reticle locks onto the enemy under it, at any depth, and your
shots lead it; the lead dot shows where they will cross its depth. Drone rings hold in a band
ahead of you and dive one at a time, after a warning flash.

**The enemies** (Plane):
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
WARP_SCENE=tunnel cargo run --release -- --shot DIR 5 20   # a bot flies the Tunnel
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
  which turns stick and cursor vectors into gax types right away. It also disallows geometry by
  hand: `sqrt`, `sin`, `cos`, `tan`, `atan2`, `asin`, `acos`, `sin_cos` and `hypot` on floats,
  everywhere, the audio included.
* **The audio thread does not allocate.** A test runs every synthesizer for 4000 blocks under a
  counting allocator.
