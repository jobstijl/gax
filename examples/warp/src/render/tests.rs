//! The renderer on a real GPU (skipped without an adapter, unless `GAX_REQUIRE_GPU` is set):
//! the lattice and particle kernels against their CPU twins, frames as the game builds them,
//! every pipeline drawn, resizing, and the pixels of a line.

use super::cpu_grid::CpuGrid;
use super::gpu_types::Node;
use super::lattice::MAX_GRID_STEPS;
use super::particles::MAX_PARTICLES;
use super::scene::{self, seg};
use super::*;
use crate::sim::{self, body::Pose, rng::Rng};

/// The headless shots' format.
const SRGB: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    match crate::gfx::try_headless_device() {
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

/// A renderer of `size` into `format`, with the game's lattice.
fn renderer(format: wgpu::TextureFormat, size: [u32; 2]) -> Option<Renderer> {
    let (device, queue) = gpu()?;
    Some(Renderer::new(
        device,
        queue,
        format,
        size,
        crate::gfx::grid_spec(),
    ))
}

/// A target of `size` in `format`, to render into and read back.
fn target(r: &Renderer, format: wgpu::TextureFormat, [width, height]: [u32; 2]) -> wgpu::Texture {
    r.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn view(t: &wgpu::Texture) -> wgpu::TextureView {
    t.create_view(&wgpu::TextureViewDescriptor::default())
}

/// The camera every test frame looks through: 20 units above and below the origin.
fn camera(size: [u32; 2]) -> CameraUniform {
    scene::camera(Pose::identity(), 20.0, size, 0.0)
}

/// The game's post-processing at rest.
fn post(size: [u32; 2]) -> PostSettings {
    crate::fx::Fx::new().post(&scene::view_map(Pose::identity(), 20.0, size))
}

/// A Plane frame of 64×64 with only the lattice's `steps`, the particles' `wells` and `spawn`.
fn plane<'a>(steps: &'a GridSteps, wells: &'a [Source], spawn: &'a [Particle]) -> Frame<'a> {
    Frame {
        camera: camera([64, 64]),
        world: &[],
        hud_lines: &[],
        post: post([64, 64]),
        plane: Some(PlaneLayers {
            centre: crate::geom::ORIGIN,
            grid_steps: steps,
            grid_light: crate::light::light(0.1, 0.1, 0.5, 0.2),
            wells,
            spawn,
        }),
    }
}

/// A frame's steps with these sources.
fn steps(sources: &[Vec<Source>]) -> GridSteps {
    let mut steps = GridSteps::default();
    for s in sources {
        steps.push(s.iter().copied());
    }
    steps
}

/// Copy `bytes` of `buf` back (blocking).
fn read<T: bytemuck::Pod>(r: &Renderer, buf: &wgpu::Buffer, bytes: usize) -> Vec<T> {
    let read = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = r.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, 0, &read, 0, bytes as u64);
    r.queue.submit([enc.finish()]);
    let slice = read.slice(..);
    slice.map_async(wgpu::MapMode::Read, |res| res.expect("map"));
    r.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("GPU finishes");
    let out = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
    read.unmap();
    out
}

/// The lattice's current state.
fn read_grid(r: &Renderer) -> Vec<Node> {
    read(
        r,
        r.lattice.current(),
        crate::gfx::grid_spec().nodes() * size_of::<Node>(),
    )
}

/// The pixels of `t` (4 bytes each), row by row.
fn pixels(r: &Renderer, t: &wgpu::Texture) -> Vec<[u8; 4]> {
    let (w, h) = (t.width(), t.height());
    let row = (w * 4).div_ceil(256) * 256;
    let buf = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pixels"),
        size: u64::from(row * h),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = r.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        t.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(h),
            },
        },
        t.size(),
    );
    r.queue.submit([enc.finish()]);
    let slice = buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |res| res.expect("map"));
    r.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("GPU finishes");
    let data = slice.get_mapped_range().expect("mapped").to_vec();
    data.chunks(row as usize)
        .flat_map(|line| line[..(w * 4) as usize].as_chunks::<4>().0.to_vec())
        .collect()
}

/// A position and a velocity.
type Motion = (Point<(), f32>, Point<(), f32>);

/// The largest relative difference of `got`'s coefficients from `want`'s.
fn difference(got: Motion, want: Motion) -> f32 {
    (got.0.c.iter().zip(&want.0.c))
        .chain(got.1.c.iter().zip(&want.1.c))
        .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
        .fold(0.0, f32::max)
}

/// Sources as a frame's ticks bring them: wells and blasts, pulling and pushing, a different
/// number in every step.
fn random_sources(rng: &mut Rng, n: usize) -> Vec<Source> {
    (0..n)
        .map(|_| {
            let at = Point::xy(rng.range(-25.0, 25.0), rng.range(-14.0, 14.0));
            let pull = if rng.chance(0.5) {
                rng.range(20.0, 120.0)
            } else {
                rng.range(-900.0, -50.0)
            };
            Source::new(at, pull, rng.range(0.5, 9.0))
        })
        .collect()
}

/// The lattice kernel on the GPU and its CPU twin (the fused batch kernels) agree, frame by
/// frame as the game runs them: frames of 1, 2, 3 and 8 steps, each step with its own sources
/// (a different number each, so the GPU pads them to the frame's most), the parity flipping
/// inside a frame. One step agrees to within a few f32 ulps of the motion's scale, and after
/// 240 steps they have not drifted apart.
#[test]
fn grid_kernels_agree_between_cpu_and_gpu() {
    let Some(mut r) = renderer(SRGB, [64, 64]) else {
        return;
    };
    let out = target(&r, SRGB, [64, 64]);
    let spec = crate::gfx::grid_spec();
    let mut cpu = CpuGrid::new(spec);
    // A displaced, moving start; the border is pinned.
    let mut rng = Rng::new(5);
    let start: Vec<Node> = (0..spec.nodes())
        .map(|k| {
            let d = if spec.is_border(k) { 0.0 } else { 0.15 };
            let p = spec.rest(k) + Point::direction(rng.range(-d, d), rng.range(-d, d));
            let v = Point::direction(rng.range(-d, d) * 10.0, rng.range(-d, d) * 10.0);
            Node {
                p: p.into(),
                v: v.into(),
            }
        })
        .collect();
    r.queue
        .write_buffer(r.lattice.current(), 0, bytemuck::cast_slice(&start));
    cpu.set(&start);
    let compare = |r: &Renderer, cpu: &CpuGrid| {
        read_grid(r)
            .iter()
            .zip(cpu.p.iter().zip(&cpu.v))
            .map(|(n, (&p, &v))| difference((n.p.into(), n.v.into()), (p, v)))
            .fold(0.0, f32::max)
    };
    let (mut ran, mut worst, mut first) = (0, 0.0f32, None);
    for frame in [1, 2, 3, 8].iter().cycle() {
        let sources: Vec<Vec<Source>> = (0..*frame)
            .map(|s| random_sources(&mut rng, (s * 3 + ran) % 6))
            .collect();
        let steps = steps(&sources);
        r.render(&view(&out), &plane(&steps, &[], &[]));
        for s in &sources {
            cpu.step(s, sim::DT);
        }
        ran += frame;
        let d = compare(&r, &cpu);
        first.get_or_insert(d);
        worst = worst.max(d);
        if ran >= 240 {
            break;
        }
    }
    let one = first.unwrap_or_default();
    println!("grid CPU vs GPU: one step {one:.1e}, {ran} steps {worst:.1e}");
    assert!(one < 1e-5, "one step: {one:e}");
    assert!(worst < 1e-3, "{ran} steps: {worst:e}");
}

/// After frames without the Plane's layers (the Tunnel's), the lattice is back at rest.
#[test]
fn the_lattice_rests_after_the_tunnel() {
    let Some(mut r) = renderer(SRGB, [64, 64]) else {
        return;
    };
    let out = target(&r, SRGB, [64, 64]);
    let blast = steps(&vec![
        vec![Source::new(Point::xy(3.0, 2.0), -900.0, 4.0)];
        4
    ]);
    r.render(&view(&out), &plane(&blast, &[], &[]));
    let rest = CpuGrid::new(crate::gfx::grid_spec());
    let moved = |r: &Renderer| {
        read_grid(r)
            .iter()
            .zip(&rest.p)
            .map(|(n, &p)| difference((n.p.into(), n.v.into()), (p, Point::zero())))
            .fold(0.0, f32::max)
    };
    assert!(moved(&r) > 1e-3, "the blast moves the lattice");
    let tunnel = Frame {
        plane: None,
        ..plane(&blast, &[], &[])
    };
    r.render(&view(&out), &tunnel);
    r.render(&view(&out), &plane(&GridSteps::default(), &[], &[]));
    assert_eq!(moved(&r), 0.0);
}

/// The particle kernel likewise, with a well pulling, particles of every age (some dying
/// between the steps), written across the end of the ring in two writes. Dead particles leave
/// the live window.
#[test]
fn particle_kernels_agree_between_cpu_and_gpu() {
    let Some(mut r) = renderer(SRGB, [64, 64]) else {
        return;
    };
    let out = target(&r, SRGB, [64, 64]);
    // Dead filler up to 100 before the ring's end.
    let dead = Particle {
        p: Point::xy(0.0, 0.0).into(),
        v: Point::zero().into(),
        color: crate::light::light(1.0, 1.0, 1.0, 1.0).into(),
        life: [1.0, 1.0, 0.0, 0.0],
    };
    let filler = vec![dead; MAX_PARTICLES - 100];
    r.render(&view(&out), &plane(&GridSteps::default(), &[], &filler));
    let mut rng = Rng::new(9);
    let ps: Vec<Particle> = (0..4096)
        .map(|_| Particle {
            p: Point::xy(rng.range(-20.0, 20.0), rng.range(-10.0, 10.0)).into(),
            v: Point::direction(rng.range(-9.0, 9.0), rng.range(-9.0, 9.0)).into(),
            color: crate::light::light(1.0, 1.0, 1.0, 1.0).into(),
            life: [
                rng.range(0.0, 0.02),
                rng.range(0.0, 0.05),
                rng.range(0.5, 3.0),
                0.03,
            ],
        })
        .collect();
    let wells = [Source::new(Point::xy(2.0, -1.0), 90.0, 1.2)];
    let three = steps(&[Vec::new(), Vec::new(), Vec::new()]);
    r.render(&view(&out), &plane(&three, &wells, &ps));
    let first = MAX_PARTICLES - 100;
    assert_eq!(
        r.particles.live(),
        first as u64..(first + ps.len()) as u64,
        "the filler is dead"
    );
    let ring: Vec<Particle> = read(
        &r,
        r.particles.buffer(),
        MAX_PARTICLES * size_of::<Particle>(),
    );
    let zero = Point::zero();
    let worst = ps
        .iter()
        .enumerate()
        .map(|(i, q)| {
            let g = ring[(first + i) % MAX_PARTICLES];
            let (mut p, mut v): Motion = (q.p.into(), q.v.into());
            let mut age = q.life[0];
            for _ in 0..3 {
                if age < q.life[1] {
                    let f = crate::add_source_force(zero, p, wells[0].at, wells[0].k());
                    (p, v) = crate::particle_step(p, v, f, [q.life[2], sim::DT]);
                    age += sim::DT;
                }
            }
            assert_eq!(g.life[0], age, "particle {i}'s age");
            difference((g.p.into(), g.v.into()), (p, v))
        })
        .fold(0.0, f32::max);
    println!("particles CPU vs GPU after 3 steps: {worst:.1e}");
    assert!(worst < 1e-5, "{worst:e}");
    // Every one of them has died within its lifetime: the window is empty.
    let eight = steps(&vec![Vec::new(); MAX_GRID_STEPS]);
    r.render(&view(&out), &plane(&eight, &wells, &[]));
    assert!(r.particles.live().is_empty());
}

/// The frame of the pixel tests: a bright horizontal line through the world's origin, and a
/// HUD line 12 HUD units above the middle.
fn lines_frame<'a>(
    world: &'a [LineInstance],
    hud: &'a [LineInstance],
    size: [u32; 2],
) -> Frame<'a> {
    Frame {
        camera: camera(size),
        world,
        hud_lines: hud,
        post: post(size),
        plane: None,
    }
}

fn line(y: f32) -> LineInstance {
    let c = crate::light::light(0.6, 0.8, 1.0, 3.0);
    seg(
        Point::xy(-6.0, y),
        Point::xy(6.0, y),
        c,
        scene::Stroke::new(0.1, 0.3, 0.3),
        Pose::identity(),
    )
}

/// The line pipelines light the pixels on a line, world and HUD, and leave far ones dark.
#[test]
fn a_line_lights_its_pixels() {
    let size = [128, 128];
    let Some(mut r) = renderer(SRGB, size) else {
        return;
    };
    let out = target(&r, SRGB, size);
    let (world, hud) = ([line(0.0)], [line(12.0)]);
    r.render(&view(&out), &lines_frame(&world, &hud, size));
    let px = pixels(&r, &out);
    let at = |x: usize, y: usize| px[y * size[0] as usize + x];
    let bright = |p: [u8; 4]| p[0].max(p[1]).max(p[2]);
    // The world's line through the middle row; the HUD's at 12 of 18 units above it.
    let hud_row = (64.0 - 12.0 / scene::HUD_HALF_HEIGHT * 64.0) as usize;
    assert!(bright(at(64, 64)) > 150, "{:?}", at(64, 64));
    assert!(bright(at(64, hud_row)) > 150, "{:?}", at(64, hud_row));
    for (x, y) in [(2, 2), (125, 30), (64, 110), (3, 120)] {
        assert!(bright(at(x, y)) < 40, "({x}, {y}): {:?}", at(x, y));
    }
}

/// A target without sRGB encoding (a surface's fallback), drawn through its sRGB view, holds
/// the same pixels as an sRGB target, the HUD's added light included.
#[test]
fn an_output_without_srgb_viewed_as_srgb_is_the_same() {
    let size = [96, 64];
    let (world, hud) = ([line(0.0)], [line(0.5)]);
    let Some(mut r) = renderer(SRGB, size) else {
        return;
    };
    let mut images = Vec::new();
    for format in [SRGB, wgpu::TextureFormat::Rgba8Unorm] {
        let out = r.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[SRGB],
        });
        let srgb = out.create_view(&wgpu::TextureViewDescriptor {
            format: Some(SRGB),
            ..Default::default()
        });
        r.render(&srgb, &lines_frame(&world, &hud, size));
        images.push(pixels(&r, &out));
    }
    assert_eq!(images[0], images[1]);
}

/// Resized to the smallest outputs, every pass still draws (and wgpu's validation, on in the
/// tests, finds nothing).
#[test]
fn renders_at_any_size() {
    let Some(mut r) = renderer(SRGB, [64, 64]) else {
        return;
    };
    let (world, hud) = ([line(0.0)], [line(12.0)]);
    let spawn = [Particle {
        p: Point::xy(1.0, 1.0).into(),
        v: Point::direction(3.0, 0.0).into(),
        color: crate::light::light(1.0, 0.5, 0.2, 2.0).into(),
        life: [0.0, 1.0, 1.0, 0.03],
    }];
    let two = steps(&[
        vec![Source::new(Point::xy(0.0, 0.0), -300.0, 2.0)],
        Vec::new(),
    ]);
    for size in [[1, 1], [3, 2], [64, 64]] {
        r.resize(size);
        let out = target(&r, SRGB, size);
        let frame = Frame {
            world: &world,
            hud_lines: &hud,
            ..plane(&two, &[], &spawn)
        };
        r.render(&view(&out), &frame);
        assert_eq!(pixels(&r, &out).len(), (size[0] * size[1]) as usize);
    }
}
