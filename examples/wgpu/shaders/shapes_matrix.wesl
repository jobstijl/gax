// The same shapes placed by a matrix per instance (the motor's `m >> Point::slot()`, computed
// on the CPU): 48 bytes per instance. Only for the benchmark.
struct View {
    scale: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: View;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(@location(0) corner: vec2<f32>, @location(1) c0: vec4<f32>, @location(2) c1: vec4<f32>, @location(3) c2: vec4<f32>, @location(4) color: vec4<f32>) -> VsOut {
    let m = mat3x3<f32>(c0.xyz, c1.xyz, c2.xyz);
    let p = m * vec3<f32>(corner, 1.0);
    return VsOut(vec4<f32>(p.xy / p.z * view.scale.xy, 0.0, 1.0), color);
}

@fragment
fn fs_main(v: VsOut) -> @location(0) vec4<f32> {
    return v.color;
}
