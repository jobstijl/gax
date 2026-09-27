// The lattice as glowing lines, pulled from the node buffer: instance `k` is an edge between
// two neighbouring nodes. Stretched or moving edges run hot: their light mixes towards a
// violet-white (traced kernels: `edge_heat` by joins and ideal norms, `light_mix`).
import package::common::{Camera, LineVarying, line_vertex, line_shade};
import package::fused::{edge_heat, light_mix, light_fade};
import gax::pga2d::Point;

struct Node {
    p: Point,
    v: Point,
}

struct GridLook {
    // Columns, rows, unused, unused.
    dims: vec4<u32>,
    // The lattice's light.
    light: gax::pga3d::Point,
    // Half width, glow radius, glow strength, rest length.
    style: vec4<f32>,
}

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> look: GridLook;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;

@vertex
fn vs_main(@builtin(vertex_index) k: u32, @builtin(instance_index) e: u32) -> LineVarying {
    let w = look.dims.x;
    let h = look.dims.y;
    let horizontal = (w - 1u) * h;
    var i0: u32;
    var i1: u32;
    var major = false;
    if (e < horizontal) {
        let i = e % (w - 1u);
        let j = e / (w - 1u);
        i0 = j * w + i;
        i1 = i0 + 1u;
        major = j % 4u == 0u;
    } else {
        let f = e - horizontal;
        let i = f % w;
        let j = f / w;
        i0 = j * w + i;
        i1 = i0 + w;
        major = i % 4u == 0u;
    }
    let a = nodes[i0];
    let b = nodes[i1];
    let hot = edge_heat(a.p, b.p, a.v, b.v, array<f32, 3>(look.style.w, 1.2, 0.012));
    let base = light_fade(look.light, select(0.35, 1.0, major));
    // Violet-white, at 1.6 times the lattice's own intensity.
    let glare = light_fade(gax::pga3d::Point(vec4<f32>(0.55, 0.45, 1.0, 1.0)), look.light.c0.w * 1.6);
    let light = light_mix(base, glare, hot);
    let style = vec4<f32>(look.style.x * select(0.6, 1.0, major), look.style.y, look.style.z * (1.0 + 0.6 * hot), 0.0);
    return line_vertex(cam, k, a.p.c0.xy, b.p.c0.xy, light.c0, style);
}

@fragment
fn fs_main(v: LineVarying) -> @location(0) vec4<f32> {
    return line_shade(v, cam.viewport.z);
}
