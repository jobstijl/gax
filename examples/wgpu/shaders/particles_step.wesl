// The particle update: the traced kernel, the same program as the CPU's `particle_step`.
import package::fused::particle_step;
import gax::pga2d::{Motor, Point};

struct Particle {
    motor: Motor,
    rate: Point,
    // x: age, y: lifetime, z: hue, w: unused.
    life: vec4<f32>,
}

struct Step {
    dt: f32,
    count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Step;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.count) {
        return;
    }
    var p = particles[id.x];
    p.motor = particle_step(p.motor, p.rate, params.dt);
    p.life.x = p.life.x + params.dt;
    particles[id.x] = p;
}
