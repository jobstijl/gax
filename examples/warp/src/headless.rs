//! Headless runs: `warp --shot DIR [SECONDS...]` plays a scripted run with a simple bot and
//! writes PNG snapshots at the given times (and one of the title screen). It renders offscreen
//! with wgpu validation on, so it doubles as the renderer's smoke test.

use crate::render::Renderer;
use crate::{Game, Screen, advance, audio, input, render_game, sim};
use gax::pga2d::Point;
use std::path::Path;

const SIZE: [u32; 2] = [1600, 900];
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// A device without a window, with validation, failing loudly on any GPU error.
pub fn device() -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    try_device().expect("a GPU adapter")
}

/// [`device`], or `None` without an adapter.
pub fn try_device() -> Option<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    desc.flags |= wgpu::InstanceFlags::VALIDATION | wgpu::InstanceFlags::DEBUG;
    let instance = wgpu::Instance::new(desc);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("warp headless"),
        required_features: Renderer::wanted_features(&adapter),
        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
        ..Default::default()
    }))
    .ok()?;
    device.on_uncaptured_error(std::sync::Arc::new(|e| panic!("wgpu error: {e}")));
    Some((adapter, device, queue))
}

/// The bot's input: strafe around the nearest enemy, aim at it, fire; bomb when crowded.
/// A Tunnel bot: aims the cursor at the nearest enemy ahead, weaves, fires, and rolls away
/// from bolts that come close.
pub fn tunnel_bot(g: &Game, t: f32) -> (sim::Input, input::Flight) {
    let run = g.tunnel.as_ref().expect("tunnel");
    let w = &run.world;
    let s = w.ship.s();
    let target = w
        .enemies
        .iter()
        .filter(|e| crate::tunnel::arc(e.pos) > s + 6.0)
        .min_by(|a, b| crate::tunnel::arc(a.pos).total_cmp(&crate::tunnel::arc(b.pos)))
        .map(|e| e.pos);
    let cursor = target
        .and_then(|q| run.view.on_screen(w, q))
        .map(|p| p.to_euclidean());
    let danger = w.bolts.iter().any(|b| {
        let d = crate::tunnel::arc(b.pos) - s;
        d > 0.0 && d < 6.0
    }) || w
        .enemies
        .iter()
        .any(|e| e.flight == crate::tunnel::Flight::Dive && (crate::tunnel::arc(e.pos) - s) < 8.0);
    let input = sim::Input {
        movement: crate::sim::body::heading(t * 0.6, 0.55),
        // In bursts, so that formations live long enough to be seen.
        fire: crate::signal::wave(t * 1.3) > 0.2,
        ..sim::Input::default()
    };
    let flight = input::Flight {
        cursor: cursor.or(Some([0.0, 0.0])),
        stick: None,
        roll: if danger { 1 } else { 0 },
        throttle: crate::signal::wave(t * 0.2),
        bomb: false,
    };
    (input, flight)
}

pub fn bot_input(g: &Game, t: f32) -> sim::Input {
    bot(g, t)
}

fn bot(g: &Game, t: f32) -> sim::Input {
    use crate::sim::body::{distance, heading, turned, with_length};
    let s = &g.sim;
    let p = s.ship.body.pos();
    let near = s
        .enemies
        .iter()
        .map(|e| e.body.pos())
        .min_by(|a, b| distance(*a, p).total_cmp(&distance(*b, p)));
    let mut input = sim::Input::default();
    // Wander: a direction turning with time.
    let mut go = heading(t * 0.9, 0.6);
    if let Some(q) = near {
        let to = q - p;
        input.aim = with_length(to, 1.0);
        input.fire = true;
        if to.ideal_norm() < 8.0 {
            // Away and around: the way back from it, turned a little.
            go = with_length(turned(-to, -0.6), 1.0);
        }
        let crowd = s.enemies.iter().filter(|e| distance(e.body.pos(), p) < 3.0);
        input.bomb = crowd.count() >= 5;
    }
    // Stay off the walls: pulled back towards the middle.
    let home = (sim::at(0.0, 0.0) - p) * (0.6 / 32.0);
    let m = go + home;
    input.movement = if m.ideal_norm() > 1.0 {
        with_length(m, 1.0)
    } else {
        m
    };
    input
}

/// Write snapshots of a scripted run.
pub fn shots(args: &[String]) {
    let dir = args.first().map_or("shots", String::as_str);
    let mut times: Vec<f32> = args.iter().skip(1).filter_map(|a| a.parse().ok()).collect();
    if times.is_empty() {
        times = vec![4.0, 15.0, 45.0, 90.0];
    }
    std::fs::create_dir_all(dir).expect("create the output directory");
    let (adapter, device, queue) = device();
    let info = adapter.get_info();
    println!(
        "GPU: {} ({:?}, {:?})",
        info.name, info.device_type, info.backend
    );
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shot"),
        size: wgpu::Extent3d {
            width: SIZE[0],
            height: SIZE[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut renderer = Renderer::new(device, queue, FORMAT, SIZE, crate::grid_spec());
    let mut sound = audio::Sound::silent();
    let aspect = SIZE[0] as f32 / SIZE[1] as f32;
    let dt = 1.0 / 60.0;
    let none = input::Menu::default();

    // The title screen.
    let mut g = Game::new(crate::store::Store::new(None));
    for _ in 0..(3.0 / dt) as usize {
        g.time += dt;
        advance(&mut g, sim::Input::default(), none, dt, aspect, &mut sound);
        render_game(&mut g, &mut renderer, &view, SIZE);
    }
    save(&renderer, &target, &Path::new(dir).join("title.png"));

    // `WARP_SCENE=singularity`: one still singularity in the middle (for looking at it).
    if std::env::var("WARP_SCENE").as_deref() == Ok("singularity") {
        g.start_seeded(7);
        g.sim.director.enabled = false;
        g.sim.ship.body.shift(-20.0, -10.0);
        g.sim.spawn(sim::Kind::Singularity, sim::at(0.0, 0.0));
        g.sim.enemies[0].body.vel = Point::direction(0.0, 0.0);
        for _ in 0..120 {
            g.time += dt;
            advance(&mut g, sim::Input::default(), none, dt, aspect, &mut sound);
            g.sim.enemies[0].body.vel = Point::direction(0.0, 0.0);
            render_game(&mut g, &mut renderer, &view, SIZE);
        }
        save(&renderer, &target, &Path::new(dir).join("singularity.png"));
        return;
    }
    // `WARP_SCENE=tunnel`: a bot flies the Tunnel; snapshots at the given times.
    if std::env::var("WARP_SCENE").as_deref() == Ok("tunnel") {
        g.start_tunnel_seeded(11);
        let end = times.iter().copied().fold(0.0f32, f32::max);
        times.sort_by(f32::total_cmp);
        let (mut t, mut next) = (0.0f32, 0);
        while t <= end + dt {
            g.time += dt;
            t += dt;
            let (input, flight) = tunnel_bot(&g, t);
            g.flight = flight;
            advance(&mut g, input, none, dt, aspect, &mut sound);
            render_game(&mut g, &mut renderer, &view, SIZE);
            if next < times.len() && t >= times[next] {
                let w = &g.tunnel.as_ref().expect("tunnel").world;
                let name = format!("tunnel-{:03}s.png", times[next].round() as u32);
                save(&renderer, &target, &Path::new(dir).join(&name));
                let count = |f: crate::tunnel::Foe| w.enemies.iter().filter(|e| e.foe == f).count();
                println!(
                    "{name}: score {}, x{}, {} enemies ({} serpents, {} wells), {} gates, {} bolts, s {:.0}",
                    w.score,
                    w.mult,
                    w.enemies.len(),
                    count(crate::tunnel::Foe::Serpent),
                    count(crate::tunnel::Foe::Singularity),
                    w.gates
                        .iter()
                        .filter(|g| g.state == crate::tunnel::GateState::Open)
                        .count(),
                    w.bolts.len(),
                    w.ship.s()
                );
                next += 1;
            }
        }
        return;
    }
    // `WARP_SCENE=menus`: the title, settings, a score table, initials and the pause menu.
    if std::env::var("WARP_SCENE").as_deref() == Ok("menus") {
        let data = std::env::temp_dir().join(format!("warp-menus-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data);
        let store = crate::store::Store::new(Some(data.clone()));
        for (k, name) in ["NEO", "ARC", "VEX", "IO", "ZED"].iter().enumerate() {
            let mut r: sim::replay::Replay = sim::replay::Replay::new(k as u64);
            r.score = 480_000 / (k as u64 + 1) + 1234;
            r.inputs = vec![Default::default(); 120 * (300 - 40 * k)];
            store.finish(&r, Some(name));
        }
        let mut g = Game::new(store);
        let mut shot = |g: &mut Game, name: &str, menu: input::Menu, frames: u32| {
            for f in 0..frames {
                g.time += dt;
                let m = if f == 0 { menu } else { none };
                advance(g, sim::Input::default(), m, dt, aspect, &mut sound);
                render_game(g, &mut renderer, &view, SIZE);
            }
            save(
                &renderer,
                &target,
                &Path::new(dir).join(format!("{name}.png")),
            );
        };
        shot(&mut g, "menu-title", none, 60);
        g.screen = Screen::Settings;
        g.sel = 3;
        g.settings.music = 6;
        shot(&mut g, "menu-settings", none, 30);
        g.screen = Screen::Scores;
        g.sel = 1;
        g.highlight = Some(2);
        shot(&mut g, "menu-scores", none, 30);
        g.start_seeded(3);
        shot(&mut g, "menu-play", none, 240);
        g.screen = Screen::Paused;
        g.sel = 0;
        shot(&mut g, "menu-pause", none, 20);
        g.sim.score = 123_450;
        g.screen = Screen::Initials;
        g.sim.phase = sim::Phase::Over;
        g.initials = *b"ACE";
        g.cursor = 1;
        shot(&mut g, "menu-initials", none, 20);
        let _ = std::fs::remove_dir_all(&data);
        return;
    }
    // `WARP_SCENE=roster`: every enemy kind in a row, still, with a few shots in flight.
    if std::env::var("WARP_SCENE").as_deref() == Ok("roster") {
        g.start_seeded(7);
        g.sim.director.enabled = false;
        // `WARP_SCHEME=1` or `2`: the colour-blind schemes.
        g.settings.scheme = match std::env::var("WARP_SCHEME").as_deref() {
            Ok("1") => crate::store::Scheme::RedGreen,
            Ok("2") => crate::store::Scheme::BlueYellow,
            _ => crate::store::Scheme::Standard,
        };
        g.sim.ship.body.shift(0.0, -9.0);
        g.sim.ship.invulnerable = 1e9;
        use sim::Kind::*;
        let kinds = [
            Drifter, Chaser, Mote, Evader, Splitter, Fragment, Warden, Serpent, Carrier,
        ];
        for (k, kind) in kinds.iter().enumerate() {
            let x = -20.0 + k as f32 * 5.0;
            let y = if k % 2 == 0 { 3.0 } else { -1.0 };
            g.sim.spawn(*kind, sim::at(x, y));
        }
        // The warden faces the ship.
        for e in &mut g.sim.enemies {
            if e.kind == Warden {
                let here = e.body.pos();
                let facing = crate::sim::body::angle_of(sim::at(0.0, -9.0) - here);
                e.body = crate::sim::body::Body::new(crate::sim::body::place(here, facing));
            }
        }
        // Pinned in place (the serpent's chain still winds), so every shape is seen whole.
        let poses: Vec<_> = g.sim.enemies.iter().map(|e| (e.id, e.body.pose)).collect();
        for f in 0..150 {
            g.time += dt;
            for e in &mut g.sim.enemies {
                if let Some((_, p)) = poses.iter().find(|(id, _)| *id == e.id) {
                    e.body = crate::sim::body::Body::new(*p);
                }
            }
            // Fire at the warden's face at the end, to show the reflection.
            let input = sim::Input {
                aim: Point::direction(10.0, 12.0),
                fire: f > 110,
                ..sim::Input::default()
            };
            advance(&mut g, input, none, dt, aspect, &mut sound);
            render_game(&mut g, &mut renderer, &view, SIZE);
        }
        save(&renderer, &target, &Path::new(dir).join("roster.png"));
        return;
    }
    // A run.
    g.start_seeded(7);
    let end = times.iter().copied().fold(0.0f32, f32::max);
    let mut next = 0;
    times.sort_by(f32::total_cmp);
    let mut t = 0.0;
    while t <= end + dt {
        g.time += dt;
        t += dt;
        let input = bot(&g, t);
        // Show a singularity early in the scripted run.
        if (t - 8.0).abs() < dt * 0.5 {
            g.sim
                .announce(sim::Kind::Singularity, sim::at(12.0, 6.0), 1.2);
        }
        if g.screen == Screen::Over {
            g.start();
        }
        advance(&mut g, input, none, dt, aspect, &mut sound);
        render_game(&mut g, &mut renderer, &view, SIZE);
        if next < times.len() && t >= times[next] {
            let path = Path::new(dir).join(format!("play-{:03}s.png", times[next] as u32));
            save(&renderer, &target, &path);
            let gpu = renderer.timings().map_or(String::new(), |t| {
                format!(
                    ", GPU ms: compute {:.2} scene {:.2} bloom {:.2} post {:.2}",
                    t.compute, t.scene, t.bloom, t.composite
                )
            });
            println!(
                "{}: score {}, x{}, {} enemies, {} bullets, worst drift {:.1e}{gpu}",
                path.display(),
                g.sim.score,
                g.sim.mult,
                g.sim.enemies.len(),
                g.sim.bullets.len(),
                g.sim.worst_drift
            );
            next += 1;
        }
    }
}

/// Read the target back and write it as a PNG.
fn save(r: &Renderer, target: &wgpu::Texture, path: &Path) {
    let [w, h] = SIZE;
    let row = (w * 4).div_ceil(256) * 256;
    let buf = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shot readback"),
        size: u64::from(row * h),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(h),
            },
        },
        target.size(),
    );
    r.queue.submit([enc.finish()]);
    let slice = buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |res| res.expect("map the shot"));
    r.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the GPU finishes");
    let data = slice.get_mapped_range().expect("mapped").to_vec();
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let i = (y * row + x * 4) as usize;
            rgb.extend_from_slice(&data[i..i + 3]);
        }
    }
    std::fs::write(path, png(w, h, &rgb)).expect("write the PNG");
}

/// A PNG of 8-bit RGB pixels, with stored (uncompressed) deflate blocks: no dependencies.
pub fn png(w: u32, h: u32, rgb: &[u8]) -> Vec<u8> {
    let crc_table: Vec<u32> = (0..256u32)
        .map(|n| {
            (0..8).fold(n, |c, _| {
                if c & 1 == 1 {
                    0xedb8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                }
            })
        })
        .collect();
    let crc = |bytes: &[u8]| {
        !bytes.iter().fold(!0u32, |c, b| {
            crc_table[((c ^ u32::from(*b)) & 0xff) as usize] ^ (c >> 8)
        })
    };
    let chunk = |out: &mut Vec<u8>, ty: &[u8; 4], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut body = ty.to_vec();
        body.extend_from_slice(data);
        out.extend_from_slice(&body);
        out.extend_from_slice(&crc(&body).to_be_bytes());
    };
    let mut raw = Vec::with_capacity(((w * 3 + 1) * h) as usize);
    for y in 0..h as usize {
        raw.push(0); // no filter
        raw.extend_from_slice(&rgb[y * w as usize * 3..(y + 1) * w as usize * 3]);
    }
    let mut z = vec![0x78, 0x01];
    let chunks: Vec<&[u8]> = raw.chunks(65535).collect();
    for (i, c) in chunks.iter().enumerate() {
        z.push(u8::from(i + 1 == chunks.len()));
        z.extend_from_slice(&(c.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(c.len() as u16)).to_le_bytes());
        z.extend_from_slice(c);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for x in &raw {
        a = (a + u32::from(*x)) % 65521;
        b = (b + a) % 65521;
    }
    z.extend_from_slice(&((b << 16) | a).to_be_bytes());
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::cpu_grid::CpuGrid;
    use crate::render::{Frame, Node, Particle};
    use crate::sim::rng::Rng;

    /// The whole loop without a window: a bot plays through `advance` at uneven frame rates
    /// (with hit-stops and a death or two), ends the run from the pause menu, enters initials,
    /// and then watches its own replay, which must reach its end without diverging.
    #[test]
    fn a_played_run_goes_on_the_table_and_replays_exactly() {
        use crate::input::Menu;
        let dir = std::env::temp_dir().join(format!("warp-test-flow-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut g = Game::new(crate::store::Store::new(Some(dir.clone())));
        let mut sound = audio::Sound::silent();
        let none = Menu::default();
        let aspect = 16.0 / 9.0;
        g.start_seeded(99);
        let mut t = 0.0;
        let mut frame = 0u32;
        while g.screen == Screen::Playing && t < 70.0 {
            // Frames of 7 to 20 ms.
            let dt = 0.007 + 0.013 * ((frame * 7919) % 97) as f32 / 97.0;
            frame += 1;
            t += dt;
            let input = bot_input(&g, t);
            advance(&mut g, input, none, dt, aspect, &mut sound);
        }
        if g.screen == Screen::Playing {
            let step = |g: &mut Game, sound: &mut audio::Sound, m: Menu| {
                advance(g, sim::Input::default(), m, 0.016, aspect, sound);
            };
            step(&mut g, &mut sound, Menu { back: true, ..none });
            assert_eq!(g.screen, Screen::Paused);
            step(&mut g, &mut sound, Menu { down: true, ..none });
            step(
                &mut g,
                &mut sound,
                Menu {
                    start: true,
                    ..none
                },
            );
        }
        let score = g.sim.score;
        assert!(score > 0);
        assert_eq!(g.screen, Screen::Initials);
        for c in ['W', 'R', 'P'] {
            let m = Menu {
                letter: Some(c),
                ..none
            };
            advance(&mut g, sim::Input::default(), m, 0.016, aspect, &mut sound);
        }
        let m = Menu {
            start: true,
            ..none
        };
        advance(&mut g, sim::Input::default(), m, 0.016, aspect, &mut sound);
        assert_eq!(g.screen, Screen::Scores);
        assert_eq!(g.highlight, Some(0));
        let e = g.scores.entries[0].clone();
        assert_eq!((e.name.as_str(), e.score), ("WRP", score));
        // Watch it, fast.
        advance(&mut g, sim::Input::default(), m, 0.016, aspect, &mut sound);
        assert_eq!(g.screen, Screen::Watch);
        g.watch.as_mut().unwrap().speed = 8;
        while g.watch.as_ref().unwrap().end.is_none() {
            advance(
                &mut g,
                sim::Input::default(),
                none,
                0.05,
                aspect,
                &mut sound,
            );
        }
        assert_eq!(
            g.watch.as_ref().unwrap().end.as_deref(),
            Some("END OF REPLAY")
        );
        assert_eq!(g.sim.score, score);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The Tunnel through the game loop: chosen from the title, flown by the bot at uneven
    /// frame rates, ended from the pause menu, entered on the Tunnel's table, and watched
    /// back, which must reach the end with the same score; then back to the title.
    #[test]
    fn a_tunnel_run_through_the_loop() {
        use crate::input::Menu;
        let dir = std::env::temp_dir().join(format!("warp-test-tunnel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut g = Game::new(crate::store::Store::new(Some(dir.clone())));
        let mut sound = audio::Sound::silent();
        let none = Menu::default();
        let aspect = 16.0 / 9.0;
        let step = |g: &mut Game, sound: &mut audio::Sound, m: Menu| {
            advance(g, sim::Input::default(), m, 0.016, aspect, sound);
        };
        let start = Menu {
            start: true,
            ..none
        };
        step(&mut g, &mut sound, Menu { down: true, ..none });
        step(&mut g, &mut sound, start);
        assert_eq!((g.mode, g.screen), (crate::Mode::Tunnel, Screen::Playing));
        let mut t = 0.0;
        let mut frame = 0u32;
        while t < 25.0 && g.screen == Screen::Playing {
            let dt = 0.007 + 0.013 * ((frame * 7919) % 97) as f32 / 97.0;
            frame += 1;
            t += dt;
            let (input, flight) = tunnel_bot(&g, t);
            g.flight = flight;
            advance(&mut g, input, none, dt, aspect, &mut sound);
        }
        let w = &g.tunnel.as_ref().unwrap().world;
        let score = w.score;
        assert!(score > 0 && w.ship.s() > 300.0, "{score} at {}", w.ship.s());
        if g.screen == Screen::Playing {
            step(&mut g, &mut sound, Menu { back: true, ..none });
            step(&mut g, &mut sound, Menu { down: true, ..none });
            step(&mut g, &mut sound, start);
        }
        assert_eq!(g.screen, Screen::Initials);
        for c in ['T', 'U', 'N'] {
            let m = Menu {
                letter: Some(c),
                ..none
            };
            step(&mut g, &mut sound, m);
        }
        step(&mut g, &mut sound, start);
        assert_eq!((g.screen, g.table), (Screen::Scores, crate::Mode::Tunnel));
        assert_eq!(g.tunnel_scores.entries[0].score, score);
        assert!(g.scores.entries.is_empty(), "the Plane's table is separate");
        // Watch it, fast: the same score at the end, no divergence.
        step(&mut g, &mut sound, start);
        assert_eq!((g.mode, g.screen), (crate::Mode::Tunnel, Screen::Watch));
        g.watch.as_mut().unwrap().speed = 8;
        while g.watch.as_ref().unwrap().end.is_none() {
            advance(
                &mut g,
                sim::Input::default(),
                none,
                0.05,
                aspect,
                &mut sound,
            );
        }
        assert_eq!(
            g.watch.as_ref().unwrap().end.as_deref(),
            Some("END OF REPLAY")
        );
        assert_eq!(g.tunnel.as_ref().unwrap().world.score, score);
        step(&mut g, &mut sound, Menu { back: true, ..none });
        assert_eq!((g.mode, g.screen), (crate::Mode::Plane, Screen::Scores));
        assert!(g.tunnel.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The attract mode: idle on the title, a demo starts (the bot, with no table yet), any
    /// input ends it; the next demo plays the other game.
    #[test]
    fn the_attract_mode_plays_demos() {
        use crate::input::Menu;
        let mut g = Game::new(crate::store::Store::new(None));
        let mut sound = audio::Sound::silent();
        let none = Menu::default();
        let aspect = 16.0 / 9.0;
        let mut modes = Vec::new();
        for _ in 0..2 {
            let mut t = 0.0;
            while !g.demo && t < 30.0 {
                t += 0.05;
                advance(
                    &mut g,
                    sim::Input::default(),
                    none,
                    0.05,
                    aspect,
                    &mut sound,
                );
            }
            assert!(g.demo, "no demo after {t} s");
            modes.push(g.mode);
            for _ in 0..500 {
                g.time += 0.02;
                let input = if g.mode == crate::Mode::Tunnel {
                    let (i, f) = tunnel_bot(&g, g.time);
                    g.flight = f;
                    i
                } else {
                    bot_input(&g, g.time)
                };
                advance(&mut g, input, none, 0.02, aspect, &mut sound);
            }
            let score = match g.mode {
                crate::Mode::Tunnel => g.tunnel.as_ref().map_or(0, |r| r.world.score),
                crate::Mode::Plane => g.sim.score,
            };
            assert!(g.demo && score > 0, "the demo does not play");
            advance(
                &mut g,
                sim::Input::default(),
                Menu {
                    start: true,
                    ..none
                },
                0.02,
                aspect,
                &mut sound,
            );
            assert_eq!(g.screen, Screen::Title);
            assert!(!g.demo);
        }
        assert_eq!(modes, [crate::Mode::Tunnel, crate::Mode::Plane]);
    }

    fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
        match try_device() {
            Some((_, d, q)) => Some((d, q)),
            None if std::env::var_os("GAX_REQUIRE_GPU").is_some() => {
                panic!("no GPU adapter, and GAX_REQUIRE_GPU is set")
            }
            None => {
                eprintln!("no GPU adapter: skipping");
                None
            }
        }
    }

    fn frame<'a>(
        steps: &'a [Vec<[f32; 4]>],
        wells: &'a [[f32; 4]],
        spawn: &'a [Particle],
    ) -> Frame<'a> {
        let cam =
            crate::render::scene::camera(sim::body::pose_at(0.0, 0.0, 0.0), 20.0, [64, 64], 0.0);
        Frame {
            camera: cam,
            hud: cam,
            centre: [0.0, 0.0],
            world: &[],
            hud_lines: &[],
            grid_steps: steps,
            wells,
            spawn,
            grid_color: crate::light::light(0.1, 0.1, 0.5, 0.2),
            post: crate::fx::Fx::new().post(&crate::render::scene::view_map(
                sim::body::pose_at(0.0, 0.0, 0.0),
                20.0,
                [64, 64],
            )),
            plane: true,
        }
    }

    fn target(r: &Renderer) -> wgpu::TextureView {
        r.device
            .create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// The lattice kernel on the GPU and its CPU twin (the fused batch kernel) agree: one step
    /// to within a few f32 ulps of the motion's scale, and after 240 steps with blasts and a
    /// well they have not drifted apart.
    #[test]
    fn grid_kernels_agree_between_cpu_and_gpu() {
        let Some((device, queue)) = gpu() else { return };
        let spec = crate::grid_spec();
        let mut r = Renderer::new(device, queue, FORMAT, [64, 64], spec);
        let view = target(&r);
        let mut cpu = CpuGrid::new(spec);
        // A displaced, moving start.
        let mut rng = Rng::new(5);
        let start: Vec<Node> = cpu
            .p
            .iter()
            .map(|p| {
                let [x, y] = p.to_euclidean();
                let edge = x.abs() >= 31.9 || y.abs() >= 17.9;
                let d = if edge { 0.0 } else { 0.15 };
                Node {
                    p: Point::xy(x + rng.range(-d, d), y + rng.range(-d, d)).into(),
                    v: Point::direction(rng.range(-d, d) * 10.0, rng.range(-d, d) * 10.0).into(),
                }
            })
            .collect();
        r.write_grid(&start);
        cpu.set(&start);
        let sources = vec![[3.0, 2.0, -400.0, 4.0], [-10.0, 5.0, 80.0, 3.0]];
        let compare = |r: &Renderer, cpu: &CpuGrid| {
            let gpu = r.read_grid();
            let mut worst = 0.0f32;
            for (k, n) in gpu.iter().enumerate() {
                let (gp, gv): (Point<(), f32>, Point<(), f32>) = (n.p.into(), n.v.into());
                for (a, b) in
                    gp.c.iter()
                        .zip(&cpu.p[k].c)
                        .chain(gv.c.iter().zip(&cpu.v[k].c))
                {
                    worst = worst.max((a - b).abs() / (1.0 + b.abs()));
                }
            }
            worst
        };
        let steps = vec![sources.clone()];
        r.render(&view, &frame(&steps, &[], &[]));
        cpu.step(&sources, sim::DT);
        let one = compare(&r, &cpu);
        assert!(one < 1e-5, "one step: {one:e}");
        for _ in 0..240 {
            r.render(&view, &frame(&steps, &[], &[]));
            cpu.step(&sources, sim::DT);
        }
        let many = compare(&r, &cpu);
        println!("grid CPU vs GPU: one step {one:.1e}, 240 steps {many:.1e}");
        assert!(many < 1e-3, "240 steps: {many:e}");
    }

    /// The particle kernel likewise, with a well pulling.
    #[test]
    fn particle_kernels_agree_between_cpu_and_gpu() {
        let Some((device, queue)) = gpu() else { return };
        let mut r = Renderer::new(device, queue, FORMAT, [64, 64], crate::grid_spec());
        let view = target(&r);
        let mut rng = Rng::new(9);
        let ps: Vec<Particle> = (0..4096)
            .map(|_| Particle {
                p: Point::xy(rng.range(-20.0, 20.0), rng.range(-10.0, 10.0)).into(),
                v: Point::direction(rng.range(-9.0, 9.0), rng.range(-9.0, 9.0)).into(),
                color: crate::light::light(1.0, 1.0, 1.0, 1.0).into(),
                life: [0.0, 100.0, rng.range(0.5, 3.0), 0.03],
            })
            .collect();
        let wells = [[2.0f32, -1.0, 90.0, 1.2]];
        let steps = vec![Vec::new(); 3];
        r.render(&view, &frame(&steps, &wells, &ps));
        let gpu = r.read_particles(ps.len());
        let mut worst = 0.0f32;
        for (q, g) in ps.iter().zip(&gpu) {
            let (mut p, mut v): (Point<(), f32>, Point<(), f32>) = (q.p.into(), q.v.into());
            for _ in 0..3 {
                let f = crate::source_force(
                    p,
                    Point::xy(wells[0][0], wells[0][1]),
                    [wells[0][2], wells[0][3]],
                );
                (p, v) = crate::particle_step(p, v, f, [q.life[2], sim::DT]);
            }
            let (gp, gv): (Point<(), f32>, Point<(), f32>) = (g.p.into(), g.v.into());
            for (a, b) in gp.c.iter().zip(&p.c).chain(gv.c.iter().zip(&v.c)) {
                worst = worst.max((a - b).abs() / (1.0 + b.abs()));
            }
        }
        println!("particles CPU vs GPU after 3 steps: {worst:.1e}");
        assert!(worst < 1e-5, "{worst:e}");
    }

    /// The traced kernels equal the code they were traced from.
    #[test]
    fn traced_kernels_equal_their_source() {
        let mut rng = Rng::new(2);
        for _ in 0..500 {
            let mut pt = || Point::xy(rng.range(-5.0, 5.0), rng.range(-5.0, 5.0));
            let (p, rest, s) = (pt(), pt(), pt());
            let n = [pt(), pt(), pt(), pt()];
            let v = Point::direction(0.3, -0.7);
            let f = Point::direction(1.5, 0.2);
            let k = [95.0, 0.5, 5.0, 4.5, sim::DT];
            let (a, b) = (
                crate::grid_node(p, v, rest, n, f, k),
                crate::kernels::grid_node(p, v, rest, n, f, k),
            );
            for (x, y) in a.0.c.iter().zip(&b.0.c).chain(a.1.c.iter().zip(&b.1.c)) {
                assert!((x - y).abs() <= 1e-4 * (1.0 + y.abs()), "{x} {y}");
            }
            let (a, b) = (
                crate::particle_step(p, v, f, [2.0, sim::DT]),
                crate::kernels::particle_step(p, v, f, [2.0, sim::DT]),
            );
            for (x, y) in a.0.c.iter().zip(&b.0.c).chain(a.1.c.iter().zip(&b.1.c)) {
                assert!((x - y).abs() <= 1e-4 * (1.0 + y.abs()));
            }
            let (a, b) = (
                crate::source_force(p, s, [-40.0, 2.0]),
                crate::kernels::source_force(p, s, [-40.0, 2.0]),
            );
            for (x, y) in a.c.iter().zip(&b.c) {
                assert!((x - y).abs() <= 1e-4 * (1.0 + y.abs()));
            }
            // The line renderer's geometry, the lattice's heat, and the light kernels.
            let close = |x: f32, y: f32| (x - y).abs() <= 1e-4 * (1.0 + y.abs());
            let k = [1.0, -1.0, 0.3];
            let (a, b) = (
                crate::segment_corner(p, rest, k),
                crate::kernels::segment_corner(p, rest, k),
            );
            assert!(a.c.iter().zip(&b.c).all(|(x, y)| close(*x, *y)));
            assert!(close(
                crate::segment_distance(p, rest, s),
                crate::kernels::segment_distance(p, rest, s)
            ));
            let k = [0.5, 1.2, 0.012];
            assert!(close(
                crate::edge_heat(p, rest, v, f, k),
                crate::kernels::edge_heat(p, rest, v, f, k)
            ));
            let (a, b) = (
                crate::streak_tail(p, v, 0.03),
                crate::kernels::streak_tail(p, v, 0.03),
            );
            assert!(a.c.iter().zip(&b.c).all(|(x, y)| close(*x, *y)));
            let l1 = crate::light::light(rng.unit(), rng.unit(), rng.unit(), 2.0);
            let l2 = crate::light::light(rng.unit(), rng.unit(), rng.unit(), 0.5);
            let t = rng.unit();
            let pairs = [
                (
                    crate::light_mix(l1, l2, t),
                    crate::kernels::light_mix(l1, l2, t),
                ),
                (
                    crate::light_whiten(l1, t),
                    crate::kernels::light_whiten(l1, t),
                ),
                (crate::light_fade(l1, t), crate::kernels::light_fade(l1, t)),
            ];
            for (a, b) in pairs {
                assert!(a.c.iter().zip(&b.c).all(|(x, y)| close(*x, *y)));
            }
            // Post-processing: luminance, the tonemapper, the ripple, edge scaling, distance,
            // the phasor.
            assert!(close(crate::luma(l1), crate::kernels::luma(l1)));
            let (a, b) = (crate::agx(l1, 1.35), crate::kernels::agx(l1, 1.35));
            assert!(
                a.c.iter().zip(&b.c).all(|(x, y)| (x - y).abs() < 1e-3),
                "{a:?} {b:?}"
            );
            let (a, b) = (
                crate::ripple(p, s, [0.3, 0.05]),
                crate::kernels::ripple(p, s, [0.3, 0.05]),
            );
            assert!(a.c.iter().zip(&b.c).all(|(x, y)| close(*x, *y)));
            let (a, b) = (
                crate::scale_about(p, s, 0.9),
                crate::kernels::scale_about(p, s, 0.9),
            );
            assert!(a.c.iter().zip(&b.c).all(|(x, y)| close(*x, *y)));
            assert!(close(crate::distance(p, s), crate::kernels::distance(p, s)));
            assert!(close(crate::wave(t * 7.0), crate::kernels::wave(t * 7.0)));
        }
    }

    /// The traced segment distance is the Euclidean distance to the segment.
    #[test]
    fn segment_distance_by_joins_is_the_distance() {
        let (a, b) = (Point::xy(0.0f32, 0.0), Point::xy(4.0, 0.0));
        let d = |x: f32, y: f32| crate::segment_distance(a, b, Point::xy(x, y));
        assert!((d(2.0, 1.5) - 1.5).abs() < 1e-5);
        assert!((d(-3.0, 4.0) - 5.0).abs() < 1e-5);
        assert!((d(7.0, -4.0) - 5.0).abs() < 1e-5);
        // A degenerate segment is a point.
        assert!((crate::segment_distance(a, a, Point::xy(3.0, 4.0)) - 5.0).abs() < 1e-5);
    }
}
