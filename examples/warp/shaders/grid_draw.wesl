// The lattice as glowing lines, pulled from the node buffer: instance `k` is an edge between
// two neighbouring nodes. Stretched edges glow brighter and shift towards white.
import package::common::{Camera, LineVarying, line_vertex, line_shade};
import gax::pga2d::Point;

struct Node {
    p: Point,
    v: Point,
}

struct GridLook {
    // Columns, rows, unused, unused.
    dims: vec4<u32>,
    // Base colour (rgb) and intensity.
    color: vec4<f32>,
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
    let a = nodes[i0].p.c0.xy;
    let b = nodes[i1].p.c0.xy;
    let stretch = clamp(abs(length(b - a) / look.style.w - 1.0) * 1.2, 0.0, 1.0);
    let energy = clamp((length(nodes[i0].v.c0.xy) + length(nodes[i1].v.c0.xy)) * 0.012, 0.0, 1.0);
    let hot = max(stretch, energy);
    let base = look.color.rgb * look.color.a * select(0.35, 1.0, major);
    let color = vec4<f32>(mix(base, vec3<f32>(0.55, 0.45, 1.0) * look.color.a * 2.5, hot), 1.0);
    let style = vec4<f32>(look.style.x * select(0.6, 1.0, major), look.style.y, look.style.z * (1.0 + 1.5 * hot), 0.0);
    return line_vertex(cam, k, a, b, color, style);
}

@fragment
fn fs_main(v: LineVarying) -> @location(0) vec4<f32> {
    return line_shade(v, cam.viewport.z);
}
