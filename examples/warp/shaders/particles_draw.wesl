// Particles as short glowing streaks along their velocity, fading and cooling with age: a
// light that starts whitened and cools into its colour, dimmed as it dies (traced light
// kernels, the same as the CPU's).
import package::common::{Camera, LineVarying, line_vertex, line_shade};
import package::fused::{streak_tail, light_whiten, light_fade};
import gax::pga2d::Point;

struct Particle {
    p: Point,
    v: Point,
    light: gax::pga3d::Point,
    life: vec4<f32>,
}

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

@vertex
fn vs_main(@builtin(vertex_index) k: u32, @builtin(instance_index) i: u32) -> LineVarying {
    let q = particles[i];
    let t = clamp(q.life.x / max(q.life.y, 1e-3), 0.0, 1.0);
    let alive = select(0.0, 1.0, q.life.x < q.life.y);
    let tail = streak_tail(q.p, q.v, q.life.w);
    let fade = (1.0 - t) * (1.0 - t) * alive;
    let hot = light_whiten(q.light, 1.0 - clamp(t * 3.0, 0.0, 1.0));
    let light = light_fade(hot, fade);
    let style = vec4<f32>(0.045 * alive, 0.22 * alive, 0.35, 0.0);
    return line_vertex(cam, k, tail.c0.xy, q.p.c0.xy, light.c0, style);
}

@fragment
fn fs_main(v: LineVarying) -> @location(0) vec4<f32> {
    return line_shade(v, cam.viewport.z);
}
