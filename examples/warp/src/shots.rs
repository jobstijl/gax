//! `warp --shot DIR [SECONDS...]`: headless snapshots as PNG files. A bot plays a scripted run
//! (or, with `WARP_SCENE`, a scene: `singularity`, `tunnel`, `menus`, `roster`), rendered
//! offscreen with wgpu validation on, so it doubles as the renderer's smoke test.

use crate::audio::Sound;
use crate::frame::advance;
use crate::game::{Game, Mode, Screen};
use crate::gfx::{grid_spec, headless_device, render_game};
use crate::input::Menu;
use crate::render::Renderer;
use crate::store::{Scheme, Store};
use crate::{bot, sim, tunnel};
use gax::pga2d::Point;
use std::path::PathBuf;

const SIZE: [u32; 2] = [1600, 900];
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// The frames' time step.
const DT: f32 = 1.0 / 60.0;

/// Write the snapshots `args` ask for (`DIR [SECONDS...]`): the title, then the scene's.
pub fn shots(args: &[String]) {
    let dir = args.first().map_or("shots", String::as_str);
    let mut times: Vec<f32> = args.iter().skip(1).filter_map(|a| a.parse().ok()).collect();
    if times.is_empty() {
        times = vec![4.0, 15.0, 45.0, 90.0];
    }
    times.sort_by(f32::total_cmp);
    std::fs::create_dir_all(dir).expect("create the output directory");
    let mut cam = Camera::new(dir);
    let mut g = Game::new(Store::new(None));
    for _ in 0..(3.0 / DT) as usize {
        cam.frame(&mut g, sim::Input::default());
    }
    cam.save("title.png");
    match std::env::var("WARP_SCENE").as_deref() {
        Ok("singularity") => singularity(&mut cam, g),
        Ok("tunnel") => flight(&mut cam, g, &times),
        Ok("menus") => menus(&mut cam),
        Ok("roster") => roster(&mut cam, g),
        _ => run(&mut cam, g, &times),
    }
}

/// The offscreen target, the renderer, and where the pictures go.
struct Camera {
    dir: PathBuf,
    renderer: Renderer,
    target: wgpu::Texture,
    view: wgpu::TextureView,
    sound: Sound,
}

impl Camera {
    fn new(dir: &str) -> Camera {
        let (adapter, device, queue) = headless_device();
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
        Camera {
            dir: PathBuf::from(dir),
            renderer: Renderer::new(device, queue, FORMAT, SIZE, grid_spec()),
            target,
            view,
            sound: Sound::new(),
        }
    }

    /// A frame of the game with the Plane's `input`, drawn.
    fn frame(&mut self, g: &mut Game, input: sim::Input) {
        self.frame_with(g, input, Menu::default());
    }

    /// A frame with the Plane's `input` and the menu buttons `menu`, drawn.
    fn frame_with(&mut self, g: &mut Game, input: sim::Input, menu: Menu) {
        g.time += DT;
        let aspect = SIZE[0] as f32 / SIZE[1] as f32;
        advance(g, input, menu, DT, aspect, &mut self.sound);
        render_game(g, &mut self.renderer, &self.view, SIZE);
    }

    /// The GPU's time per pass, for the printouts.
    fn gpu_times(&self) -> String {
        self.renderer.timings().map_or(String::new(), |t| {
            format!(
                ", GPU ms: compute {:.2} scene {:.2} bloom {:.2} post {:.2}",
                t.compute, t.scene, t.bloom, t.composite
            )
        })
    }

    /// Read the target back and write it as the PNG `name`.
    fn save(&self, name: &str) {
        let r = &self.renderer;
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
            self.target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(h),
                },
            },
            self.target.size(),
        );
        r.queue.submit([enc.finish()]);
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |res| res.expect("map the shot"));
        r.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the GPU finishes");
        let data = slice.get_mapped_range().expect("mapped").to_vec();
        // Rows padded to `row` bytes, of RGBA pixels: their RGB.
        let rgb: Vec<u8> = data
            .chunks(row as usize)
            .take(h as usize)
            .flat_map(|line| line[..(w * 4) as usize].as_chunks::<4>().0)
            .flat_map(|&[r, g, b, _]| [r, g, b])
            .collect();
        let png = crate::png::encode(w, h, &rgb);
        std::fs::write(self.dir.join(name), png).expect("write the PNG");
    }
}

/// The Plane's world in a Plane run.
fn plane(g: &mut Game) -> &mut sim::World {
    &mut g.run.plane().expect("a Plane run").world
}

/// A scripted run: the bot plays from seed 7 (a new run when it loses), a singularity comes
/// early, and a snapshot is taken at each of `times`.
fn run(cam: &mut Camera, mut g: Game, times: &[f32]) {
    g.start_seeded(Mode::Plane, 7);
    let end = times.last().copied().unwrap_or(0.0);
    let (mut t, mut next) = (0.0, 0);
    while t <= end + DT {
        t += DT;
        if g.screen == Screen::Over {
            g.start(Mode::Plane);
        }
        // Show a singularity early in the run.
        if (t - 8.0).abs() < DT * 0.5 {
            plane(&mut g).announce(sim::Kind::Singularity, sim::at(12.0, 6.0), 1.2);
        }
        let input = bot::plane(plane(&mut g), t);
        cam.frame(&mut g, input);
        if next < times.len() && t >= times[next] {
            let name = format!("play-{:03}s.png", times[next] as u32);
            cam.save(&name);
            let w = plane(&mut g);
            println!(
                "{name}: score {}, x{}, {} enemies, {} bullets, worst drift {:.1e}{}",
                w.ledger.score,
                w.ledger.mult,
                w.enemies.len(),
                w.bullets.len(),
                w.worst_drift,
                cam.gpu_times()
            );
            next += 1;
        }
    }
}

/// `WARP_SCENE=singularity`: one still singularity in the middle (for looking at it).
fn singularity(cam: &mut Camera, mut g: Game) {
    g.start_seeded(Mode::Plane, 7);
    let w = plane(&mut g);
    w.director.enabled = false;
    w.ship.body.shift(sim::dir(-20.0, -10.0));
    w.spawn(sim::Kind::Singularity, sim::at(0.0, 0.0));
    for _ in 0..120 {
        plane(&mut g).enemies[0].intent = Point::direction(0.0, 0.0);
        cam.frame(&mut g, sim::Input::default());
    }
    cam.save("singularity.png");
}

/// `WARP_SCENE=tunnel`: the bot flies the Tunnel from seed 11; a snapshot at each of
/// `times`, with the CPU time of the last second's frames (the simulation, and building the
/// frame: the projection through the camera maps, the lines, and the submission).
fn flight(cam: &mut Camera, mut g: Game, times: &[f32]) {
    g.start_seeded(Mode::Tunnel, 11);
    let end = times.last().copied().unwrap_or(0.0);
    let (mut t, mut next) = (0.0f32, 0);
    let (mut sim_ms, mut draw_ms) = (Vec::new(), Vec::new());
    let aspect = SIZE[0] as f32 / SIZE[1] as f32;
    while t <= end + DT {
        g.time += DT;
        t += DT;
        let (input, flight) = bot::input(&g.run, t);
        g.flight = flight;
        let clock = std::time::Instant::now();
        advance(&mut g, input, Menu::default(), DT, aspect, &mut cam.sound);
        let mid = std::time::Instant::now();
        render_game(&mut g, &mut cam.renderer, &cam.view, SIZE);
        sim_ms.push((mid - clock).as_secs_f64() * 1e3);
        draw_ms.push(mid.elapsed().as_secs_f64() * 1e3);
        if sim_ms.len() > 60 {
            sim_ms.remove(0);
            draw_ms.remove(0);
        }
        if next < times.len() && t >= times[next] {
            let name = format!("tunnel-{:03}s.png", times[next].round() as u32);
            cam.save(&name);
            let w = &g.run.tunnel().expect("a Tunnel run").world;
            let count = |f: tunnel::Foe| w.enemies.iter().filter(|e| e.foe == f).count();
            let open = w
                .gates
                .iter()
                .filter(|g| g.state == tunnel::GateState::Open);
            println!(
                "{name}: score {}, x{}, {} enemies ({} serpents, {} wells), {} gates, {} bolts, s {:.0}",
                w.ledger.score,
                w.ledger.mult,
                w.enemies.len(),
                count(tunnel::Foe::Serpent),
                count(tunnel::Foe::Singularity),
                open.count(),
                w.bolts.len(),
                w.ship.s()
            );
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            println!(
                "    CPU ms per frame: simulation {:.2}, drawing {:.2}{}",
                mean(&sim_ms),
                mean(&draw_ms),
                cam.gpu_times()
            );
            next += 1;
        }
    }
}

/// `WARP_SCENE=menus`: the title, settings, a score table, play, the pause menu and initials.
fn menus(cam: &mut Camera) {
    let data = std::env::temp_dir().join(format!("warp-menus-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data);
    let store = Store::new(Some(data.clone()));
    for (k, name) in ["NEO", "ARC", "VEX", "IO", "ZED"].iter().enumerate() {
        let mut r: sim::replay::Replay = sim::replay::Replay::new(k as u64);
        r.score = 480_000 / (k as u64 + 1) + 1234;
        r.inputs = vec![Default::default(); 120 * (300 - 40 * k)];
        store.finish(&r, Some(name));
    }
    let mut g = Game::new(store);
    let mut shot = |g: &mut Game, name: &str, frames: u32| {
        for _ in 0..frames {
            cam.frame(g, sim::Input::default());
        }
        cam.save(&format!("{name}.png"));
    };
    shot(&mut g, "menu-title", 60);
    g.screen = Screen::Settings;
    g.sel = 3;
    g.settings.music = 6;
    shot(&mut g, "menu-settings", 30);
    g.screen = Screen::Scores;
    g.sel = 1;
    g.highlight = Some(2);
    shot(&mut g, "menu-scores", 30);
    g.start_seeded(Mode::Plane, 3);
    shot(&mut g, "menu-play", 240);
    g.screen = Screen::Paused;
    g.sel = 0;
    shot(&mut g, "menu-pause", 20);
    let w = plane(&mut g);
    w.ledger.score = 123_450;
    w.phase = sim::Phase::Over;
    g.screen = Screen::Initials;
    g.initials = *b"ACE";
    g.cursor = 1;
    shot(&mut g, "menu-initials", 20);
    let _ = std::fs::remove_dir_all(&data);
}

/// `WARP_SCENE=roster`: every enemy kind in a row, still, with a few shots in flight at the
/// end (at the warden's shield, to show the reflection). `WARP_SCHEME=1` or `2`: in the
/// colour-blind schemes.
fn roster(cam: &mut Camera, mut g: Game) {
    use sim::Kind::*;
    g.start_seeded(Mode::Plane, 7);
    g.settings.scheme = match std::env::var("WARP_SCHEME").as_deref() {
        Ok("1") => Scheme::RedGreen,
        Ok("2") => Scheme::BlueYellow,
        _ => Scheme::Standard,
    };
    let w = plane(&mut g);
    w.director.enabled = false;
    w.ship.body.shift(sim::dir(0.0, -9.0));
    w.ship.invulnerable = 1e9;
    let kinds = [
        Drifter, Chaser, Mote, Evader, Splitter, Fragment, Warden, Serpent, Carrier,
    ];
    for (k, kind) in kinds.iter().enumerate() {
        let x = -20.0 + k as f32 * 5.0;
        let y = if k % 2 == 0 { 3.0 } else { -1.0 };
        w.spawn(*kind, sim::at(x, y));
    }
    // The warden faces the ship.
    for e in w.enemies.iter_mut().filter(|e| e.kind == Warden) {
        let here = e.body.pos();
        let facing = Point::direction(1.0, 0.0).angle_to(sim::at(0.0, -9.0) - here);
        e.body = sim::body::Body::new(sim::body::place(here, facing));
    }
    // Pinned in place (the serpent's chain still winds), so every shape is seen whole.
    let poses: Vec<_> = w.enemies.iter().map(|e| (e.id, e.body.pose)).collect();
    for f in 0..150 {
        for e in &mut plane(&mut g).enemies {
            if let Some((_, p)) = poses.iter().find(|(id, _)| *id == e.id) {
                e.body = sim::body::Body::new(*p);
            }
        }
        let input = sim::Input {
            aim: Point::direction(10.0, 12.0),
            fire: f > 110,
            ..sim::Input::default()
        };
        cam.frame(&mut g, input);
    }
    cam.save("roster.png");
}
