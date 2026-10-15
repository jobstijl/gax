// Shared by the shaders: the camera, the lattice's nodes, the sources that pull them, the
// particles, full-screen triangles, and thick anti-aliased lines drawn as quads with a capsule
// distance in the fragment shader. The geometry is gax, traced at build time
// (`src/kernels.rs`): the quad's corners (`segment_corner`) and the distance to the segment by
// joins (`segment_distance`). The structs mirror the renderer's (`src/render/gpu_types.rs`,
// whose test checks every member's offset).
//
// A line's colour is a light: a PGA3D point in RGB space whose coordinates are its radiance
// (`src/light.rs`).

import package::fused::{segment_corner, segment_distance};
import gax::pga2d::{Point, point_new};

// The camera: a gax map taking world points `(x, y, 1)` to clip space, uploaded as the WGSL
// matrix of `proj.of(view)` (columns are inputs, applied as `m * p`).
struct Camera {
    view_proj: mat3x3<f32>,
    // Width and height in pixels, world units per pixel, time in seconds.
    viewport: vec4<f32>,
}

// A lattice node.
struct Node {
    // Its position (a finite point).
    p: Point,
    // Its velocity (a direction).
    v: Point,
}

// A point that pulls the lattice or the particles (the traced `source_force`).
struct Source {
    at: Point,
    // Pull (negative: push), radius², unused, unused.
    k: vec4<f32>,
}

// A particle.
struct Particle {
    p: Point,
    v: Point,
    // Its light (a PGA3D point).
    color: gax::pga3d::Point,
    // Age, lifetime, drag, streak length (seconds of velocity).
    life: vec4<f32>,
}

// A full-screen triangle's corner, with its uv (`0..1`, y down).
struct Full {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Corner `k` (0..2) of the triangle that covers the screen.
fn full_screen(k: u32) -> Full {
    let uv = vec2<f32>(f32((k << 1u) & 2u), f32(k & 2u));
    return Full(vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0), uv);
}

// A linear colour for the display: as it is for an sRGB target (the hardware encodes it), or
// encoded with sRGB's transfer function for a target that would show it raw (`encode`).
fn display(c: vec3<f32>, encode: bool) -> vec3<f32> {
    if (!encode) {
        return c;
    }
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

struct LineVarying {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec2<f32>,
    @location(1) @interpolate(flat) a: vec2<f32>,
    @location(2) @interpolate(flat) b: vec2<f32>,
    // The light's coefficients.
    @location(3) @interpolate(flat) light: vec4<f32>,
    // Half width, glow radius, glow strength, unused (world units).
    @location(4) @interpolate(flat) style: vec4<f32>,
}

fn to_clip(cam: Camera, p: vec2<f32>) -> vec4<f32> {
    let c = cam.view_proj * vec3<f32>(p, 1.0);
    return vec4<f32>(c.xy / c.z, 0.0, 1.0);
}

// A finite point's Euclidean coordinates: divided by its weight.
fn euclidean(p: Point) -> vec2<f32> {
    return p.c0.xy / p.c0.z;
}

// Corner `k` (0..3, a triangle strip) of the quad around the segment `a`-`b` (finite points of
// any weight), grown to hold the core, the glow and two pixels of anti-aliasing.
fn line_vertex(cam: Camera, k: u32, a: Point, b: Point, light: vec4<f32>, style: vec4<f32>) -> LineVarying {
    let ea = euclidean(a);
    let eb = euclidean(b);
    let extent = style.x + style.y * 1.5 + 2.0 * cam.viewport.z;
    let along = select(-1.0, 1.0, (k & 1u) == 1u);
    let side = select(-1.0, 1.0, (k & 2u) == 2u);
    let corner = segment_corner(point_new(ea.x, ea.y, 1.0), point_new(eb.x, eb.y, 1.0), array<f32, 3>(along, side, extent));
    let world = euclidean(corner);
    return LineVarying(to_clip(cam, world), world, ea, eb, light, style);
}

// A quad that draws nothing: every corner at one point, outside the clip volume.
fn no_line() -> LineVarying {
    let nowhere = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    return LineVarying(nowhere, vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
}

// The line's light at this fragment: a solid anti-aliased core and a Gaussian glow, additive.
fn line_shade(v: LineVarying, px: f32) -> vec4<f32> {
    let d = segment_distance(point_new(v.a.x, v.a.y, 1.0), point_new(v.b.x, v.b.y, 1.0), point_new(v.world.x, v.world.y, 1.0));
    // A line thinner than a pixel keeps its energy: widen it, and dim it by the same factor.
    let w = max(v.style.x, 0.6 * px);
    let thin = v.style.x / w;
    let core = (1.0 - smoothstep(w - px, w + px, d)) * thin;
    let g = max(v.style.y, 1e-4);
    let glow = v.style.z * exp(-2.5 * d * d / (g * g));
    // Radiance: the light's first three coordinates.
    return vec4<f32>(v.light.xyz * (core + glow), 1.0);
}
