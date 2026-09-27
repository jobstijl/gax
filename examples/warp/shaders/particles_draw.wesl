// Particles as short glowing streaks along their velocity, fading and cooling with age.
import package::common::{Camera, LineVarying, line_vertex, line_shade};
import gax::pga2d::Point;

struct Particle {
    p: Point,
    v: Point,
    color: vec4<f32>,
    life: vec4<f32>,
}

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

@vertex
fn vs_main(@builtin(vertex_index) k: u32, @builtin(instance_index) i: u32) -> LineVarying {
    let q = particles[i];
    let t = clamp(q.life.x / max(q.life.y, 1e-3), 0.0, 1.0);
    let alive = select(0.0, 1.0, q.life.x < q.life.y);
    let b = q.p.c0.xy;
    let v = q.v.c0.xy;
    let a = b - v * q.life.w;
    let fade = (1.0 - t) * (1.0 - t) * alive;
    // Hot particles start whiter and cool into their colour.
    let hot = mix(vec3<f32>(1.0, 1.0, 1.0), q.color.rgb, clamp(t * 3.0, 0.0, 1.0));
    let color = vec4<f32>(hot, q.color.a * fade);
    let style = vec4<f32>(0.045 * alive, 0.22 * alive, 0.35, 0.0);
    return line_vertex(cam, k, a, b, color, style);
}

@fragment
fn fs_main(v: LineVarying) -> @location(0) vec4<f32> {
    return line_shade(v, cam.viewport.z);
}
