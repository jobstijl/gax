// Line segments of shapes, placed by unit PGA2D motors in the vertex shader (gax's
// `unit_motor_sandwich_point`). World-space lines (text, borders) use the identity motor.
import package::common::{Camera, LineVarying, line_vertex, line_shade};
import gax::pga2d::{Motor, Point, unit_motor_sandwich_point};

@group(0) @binding(0) var<uniform> cam: Camera;

fn place(m: Motor, p: Point) -> vec2<f32> {
    let q = unit_motor_sandwich_point(m, p);
    return q.c0.xy / q.c0.z;
}

@vertex
fn vs_main(
    @builtin(vertex_index) k: u32,
    @location(0) a: vec4<f32>,
    @location(1) b: vec4<f32>,
    @location(2) light: vec4<f32>,
    @location(3) style: vec4<f32>,
    @location(4) motor: vec4<f32>,
) -> LineVarying {
    let m = Motor(motor);
    return line_vertex(cam, k, place(m, Point(a)), place(m, Point(b)), light, style);
}

@fragment
fn fs_main(v: LineVarying) -> @location(0) vec4<f32> {
    return line_shade(v, cam.viewport.z);
}

// The HUD, drawn after the composite straight onto the display: never bloomed, tonemapped or
// fringed with the scene. A light too bright for the display is faded as a whole (its weight
// scaled by its brightest channel, a hue-keeping compression), so text keeps its colour
// instead of clipping to white.
@fragment
fn fs_overlay(v: LineVarying) -> @location(0) vec4<f32> {
    let c = line_shade(v, cam.viewport.z).xyz;
    let m = max(max(c.x, c.y), max(c.z, 1e-4));
    return vec4<f32>(c * ((1.0 - exp(-1.5 * m)) / m), 1.0);
}
