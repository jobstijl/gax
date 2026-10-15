// Particle step: the traced `particle_step`, with the wells' pull from the traced
// `source_force` (the same programs the tests run on the CPU). Only the live window of the
// ring is dispatched, and a dead particle is left as it is.
import package::common::{Particle, Source};
import package::fused::{particle_step, add_source_force};
import gax::pga2d::point_new;

struct Step {
    // Particles in the live window, wells, the window's first index in the ring, unused.
    dims: vec4<u32>,
    // dt, gravity scale, unused, unused.
    k: vec4<f32>,
}

struct Wells {
    s: array<Source, 16>,
}

@group(0) @binding(0) var<uniform> step_params: Step;
@group(0) @binding(1) var<uniform> wells: Wells;
@group(0) @binding(2) var<storage, read_write> particles: array<Particle>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= step_params.dims.x) {
        return;
    }
    let at = (step_params.dims.z + id.x) % arrayLength(&particles);
    var q = particles[at];
    if (q.life.x >= q.life.y) {
        return;
    }
    var f = point_new(0.0, 0.0, 0.0);
    for (var s = 0u; s < step_params.dims.y; s++) {
        let w = wells.s[s];
        f = add_source_force(f, q.p, w.at, array<f32, 2>(w.k.x * step_params.k.y, w.k.y));
    }
    let r = particle_step(q.p, q.v, f, array<f32, 2>(q.life.z, step_params.k.x));
    q.p = r.f0;
    q.v = r.f1;
    q.life.x = q.life.x + step_params.k.x;
    particles[at] = q;
}
