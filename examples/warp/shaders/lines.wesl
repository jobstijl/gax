// Line segments of shapes, placed by unit PGA2D motors in the vertex shader (gax's
// `unit_motor_sandwich_point`). World-space lines (text, borders) use the identity motor.
import package::common::{Camera, LineVarying, line_vertex, line_shade};
import gax::pga2d::{Motor, point_new, unit_motor_sandwich_point};

@group(0) @binding(0) var<uniform> cam: Camera;

fn place(m: Motor, p: vec2<f32>) -> vec2<f32> {
    let q = unit_motor_sandwich_point(m, point_new(p.x, p.y, 1.0));
    return q.c0.xy / q.c0.z;
}

@vertex
fn vs_main(
    @builtin(vertex_index) k: u32,
    @location(0) ab: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) style: vec4<f32>,
    @location(3) motor: vec4<f32>,
) -> LineVarying {
    let m = Motor(motor);
    return line_vertex(cam, k, place(m, ab.xy), place(m, ab.zw), color, style);
}

@fragment
fn fs_main(v: LineVarying) -> @location(0) vec4<f32> {
    return line_shade(v, cam.viewport.z);
}
