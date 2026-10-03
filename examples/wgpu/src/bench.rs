//! `--bench`: motors against matrices for instancing, and the particle step on the GPU against
//! the CPU batch kernel (docs/performance.md, "GPU").

use crate::gfx::{Gfx, Instance, MAX_PARTICLES, MatrixInstance, Particle};
use crate::scene::{Rng, rate};
use gax::Unit;
use gax::pga2d::{Motor, Point};
use rand::Rng as _;
use std::time::Instant;

fn per_iter(iters: u32, mut f: impl FnMut()) -> f64 {
    f(); // warm up
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() / f64::from(iters)
}

/// Run the benchmarks and print a table.
pub fn run(gfx: &mut Gfx) {
    let n = 1 << 20;
    let mut rng = <Rng as rand::SeedableRng>::seed_from_u64(1);
    let motors: Vec<Motor<(), f32>> = (0..n)
        .map(|_| {
            // A product of unit motors is a unit motor.
            (Motor::translation(rng.random_range(-10.0..10.0), rng.random_range(-6.0..6.0))
                * Motor::rotation(
                    Point::xy(0.0, 0.0),
                    rng.random_range(0.0..std::f32::consts::TAU),
                ))
            .into_inner()
        })
        .collect();

    // Instancing: 2^20 triangles into a 1920x1080 target.
    let view = crate::target(gfx, 1920, 1080);
    gfx.set_view(12.0, 1920.0 / 1080.0);
    let color = [0.2, 0.6, 1.0, 1.0];
    let mut motor_data = Vec::new();
    let cpu_motor = per_iter(10, || {
        motor_data = motors
            .iter()
            .map(|m| Instance {
                motor: (*m).into(),
                color,
            })
            .collect();
    });
    let mut matrix_data = Vec::new();
    let cpu_matrix = per_iter(10, || {
        matrix_data = motors
            .iter()
            .map(|m| MatrixInstance {
                matrix: (Unit::new_unchecked(*m) >> Point::slot()).into(),
                color,
            })
            .collect();
    });
    let frame = |gfx: &mut Gfx, matrix: bool| {
        if matrix {
            gfx.set_instances(&matrix_data);
        } else {
            gfx.set_instances(&motor_data);
        }
        let mut enc = gfx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        gfx.draw(&mut enc, &view, matrix, 0);
        gfx.queue.submit([enc.finish()]);
    };
    let (mut gpu, mut draw) = ([0.0; 2], [0.0; 2]);
    for (k, matrix) in [false, true].into_iter().enumerate() {
        frame(gfx, matrix);
        gfx.wait();
        let t = Instant::now();
        for _ in 0..30 {
            frame(gfx, matrix);
        }
        gfx.wait();
        gpu[k] = t.elapsed().as_secs_f64() / 30.0;
        // The draw alone, with the instances already on the GPU.
        let t = Instant::now();
        for _ in 0..30 {
            let mut enc = gfx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            gfx.draw(&mut enc, &view, matrix, 0);
            gfx.queue.submit([enc.finish()]);
        }
        gfx.wait();
        draw[k] = t.elapsed().as_secs_f64() / 30.0;
    }
    println!("\ninstancing {n} triangles (1920x1080), per frame:");
    println!("| per instance | bytes | CPU preparation | upload + draw | draw only |");
    println!("|---|---|---|---|---|");
    println!(
        "| unit motor (vertex shader sandwich) | {} | {:.2} ms | {:.2} ms | {:.2} ms |",
        size_of::<Instance>(),
        cpu_motor * 1e3,
        gpu[0] * 1e3,
        draw[0] * 1e3
    );
    println!(
        "| matrix `m >> Point::slot()` | {} | {:.2} ms | {:.2} ms | {:.2} ms |",
        size_of::<MatrixInstance>(),
        cpu_matrix * 1e3,
        gpu[1] * 1e3,
        draw[1] * 1e3
    );

    // Particles: 2^20 steps of the traced kernel.
    let rates: Vec<Point<(), f32>> = (0..n)
        .map(|_| rate(rng.random_range(0.5..6.0), rng.random_range(-6.0..6.0)))
        .collect();
    let dt = [1.0f32 / 60.0];
    let mut out = vec![Motor::<(), f32>::zero(); n];
    let cpu_batch = per_iter(10, || {
        crate::particle_step_batch(&motors, &rates, &dt, &mut out)
    });
    let cpu_scalar = per_iter(3, || {
        for ((o, m), r) in out.iter_mut().zip(&motors).zip(&rates) {
            *o = crate::particle_step(*m, *r, dt[0]);
        }
    });
    assert!(n <= MAX_PARTICLES);
    let ps: Vec<Particle> = motors
        .iter()
        .zip(&rates)
        .map(|(m, r)| Particle {
            motor: (*m).into(),
            rate: (*r).into(),
            life: [0.0, 1e9, 0.0, 0.0],
        })
        .collect();
    gfx.write_particles(0, &ps);
    let step = |gfx: &Gfx| {
        let mut enc = gfx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        gfx.step_particles(&mut enc, dt[0], n);
        gfx.queue.submit([enc.finish()]);
    };
    step(gfx);
    gfx.wait();
    let t = Instant::now();
    for _ in 0..100 {
        step(gfx);
    }
    gfx.wait();
    let gpu_step = t.elapsed().as_secs_f64() / 100.0;
    let latency = per_iter(20, || {
        step(gfx);
        gfx.wait();
    });
    println!("\nparticle step, {n} particles (`particle_step`, traced), per step:");
    println!("| where | time |");
    println!("|---|---|");
    println!("| CPU, scalar loop | {:.2} ms |", cpu_scalar * 1e3);
    println!(
        "| CPU, batch kernel (SIMD, one thread) | {:.2} ms |",
        cpu_batch * 1e3
    );
    println!("| GPU compute, pipelined | {:.3} ms |", gpu_step * 1e3);
    println!("| GPU compute, submit and wait | {:.3} ms |", latency * 1e3);
}
