// Particles drawn as small quads, pulled from the storage buffer by instance index.
import gax::pga2d::{Motor, Point, point_new, unit_motor_sandwich_point};

struct Particle {
    motor: Motor,
    rate: Point,
    life: vec4<f32>,
}

struct View {
    scale: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

fn hue(h: f32) -> vec3<f32> {
    let k = vec3<f32>(0.0, 2.0, 4.0);
    return 0.5 + 0.5 * cos(6.2831853 * (h + k / 6.0));
}

@vertex
fn vs_main(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VsOut {
    let p = particles[i];
    let t = clamp(p.life.x / p.life.y, 0.0, 1.0);
    // A thin streak along the particle's local x axis, shrinking as it ages.
    let corner = vec2<f32>(select(-0.06, 0.06, (v & 1u) == 1u), select(-0.012, 0.012, (v & 2u) == 2u)) * (1.0 - t);
    let q = unit_motor_sandwich_point(p.motor, point_new(corner.x, corner.y, 1.0));
    let alive = select(0.0, 1.0, p.life.x < p.life.y);
    let color = vec4<f32>(hue(p.life.z) * (1.0 - t) * alive, 1.0);
    return VsOut(vec4<f32>(q.c0.xy / q.c0.z * view.scale.xy, 0.0, 1.0), color);
}

@fragment
fn fs_main(v: VsOut) -> @location(0) vec4<f32> {
    return v.color;
}
