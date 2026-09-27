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
pub fn bot_input(g: &Game, t: f32) -> sim::Input {
    bot(g, t)
}

fn bot(g: &Game, t: f32) -> sim::Input {
    let s = &g.sim;
    let p = s.ship.body.xy();
    let near = s
        .enemies
        .iter()
        .map(|e| {
            let q = e.body.xy();
            ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2), q)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0));
    let mut input = sim::Input::default();
    let (mut mx, mut my) = ((t * 0.7).cos() * 0.6, (t * 1.1).sin() * 0.6);
    if let Some((d2, q)) = near {
        let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
        let d = d2.sqrt().max(1e-3);
        input.aim = Point::direction(dx / d, dy / d);
        input.fire = true;
        if d < 8.0 {
            // Away and around.
            mx = -dx / d * 0.8 - dy / d * 0.6;
            my = -dy / d * 0.8 + dx / d * 0.6;
        }
        let crowd = s.enemies.iter().filter(|e| {
            let q = e.body.xy();
            (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) < 9.0
        });
        input.bomb = crowd.count() >= 5;
    }
    // Stay off the walls.
    mx -= p[0] / 32.0 * 0.6;
    my -= p[1] / 18.0 * 0.6;
    let l = (mx * mx + my * my).sqrt().max(1.0);
    input.movement = Point::direction(mx / l, my / l);
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
    let mut g = Game::new();
    for _ in 0..(3.0 / dt) as usize {
        g.time += dt;
        advance(&mut g, sim::Input::default(), none, dt, aspect, &mut sound);
        render_game(&mut g, &mut renderer, &view, SIZE);
    }
    save(&renderer, &target, &Path::new(dir).join("title.png"));

    // `WARP_SCENE=singularity`: one still singularity in the middle (for looking at it).
    if std::env::var("WARP_SCENE").as_deref() == Ok("singularity") {
        g.start();
        g.sim = sim::World::new(7);
        g.sim.director.enabled = false;
        g.sim.ship.body.shift(-20.0, -10.0);
        g.sim.spawn(sim::Kind::Singularity, [0.0, 0.0]);
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
    // A run.
    g.start();
    g.sim = sim::World::new(7);
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
            g.sim.announce(sim::Kind::Singularity, [12.0, 6.0], 1.2);
        }
        if g.screen == Screen::Over {
            g.start();
        }
        advance(&mut g, input, none, dt, aspect, &mut sound);
        render_game(&mut g, &mut renderer, &view, SIZE);
        if next < times.len() && t >= times[next] {
            let path = Path::new(dir).join(format!("play-{:03}s.png", times[next] as u32));
            save(&renderer, &target, &path);
            println!(
                "{}: score {}, x{}, {} enemies, {} bullets, worst drift {:.1e}",
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
            grid_color: [0.1, 0.1, 0.5, 0.2],
            post: crate::fx::Fx::new().post(&cam),
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
                color: [1.0, 1.0, 1.0, 1.0],
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
        }
    }
}
