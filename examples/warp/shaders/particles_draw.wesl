// Particles as short glowing streaks along their velocity, fading and cooling with age: a
// light that starts whitened and cools into its colour, dimmed as it dies (traced light
// kernels, the same as the CPU's). Instance `i` is the ring's particle `i` (modulo its size);
// a dead particle draws nothing.
import package::common::{Camera, LineVarying, Particle, line_vertex, line_shade, no_line};
import package::fused::{streak_tail, light_whiten, light_fade};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

@vertex
fn vs_main(@builtin(vertex_index) k: u32, @builtin(instance_index) i: u32) -> LineVarying {
    let q = particles[i % arrayLength(&particles)];
    if (q.life.x >= q.life.y) {
        return no_line();
    }
    let t = clamp(q.life.x / max(q.life.y, 1e-3), 0.0, 1.0);
    let tail = streak_tail(q.p, q.v, q.life.w);
    let fade = (1.0 - t) * (1.0 - t);
    let hot = light_whiten(q.color, 1.0 - clamp(t * 3.0, 0.0, 1.0));
    let light = light_fade(hot, fade);
    let style = vec4<f32>(0.045, 0.22, 0.35, 0.0);
    return line_vertex(cam, k, tail, q.p, light.c0, style);
}

@fragment
fn fs_main(v: LineVarying) -> @location(0) vec4<f32> {
    return line_shade(v, cam.viewport.z);
}
