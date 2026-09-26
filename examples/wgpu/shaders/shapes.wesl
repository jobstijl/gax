// Instanced shapes, each placed by a unit PGA2D motor: 16 bytes per instance.
import gax::pga2d::{Motor, point_new, unit_motor_sandwich_point};

struct View {
    // World units to clip space: x scale, y scale, and unused.
    scale: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: View;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(@location(0) corner: vec2<f32>, @location(1) motor: vec4<f32>, @location(2) color: vec4<f32>) -> VsOut {
    // The motor moves the shape's corner, a point `x e20 + y e01 + 1 e12`.
    let p = unit_motor_sandwich_point(Motor(motor), point_new(corner.x, corner.y, 1.0));
    let xy = p.c0.xy / p.c0.z;
    return VsOut(vec4<f32>(xy * view.scale.xy, 0.0, 1.0), color);
}

@fragment
fn fs_main(v: VsOut) -> @location(0) vec4<f32> {
    return v.color;
}
