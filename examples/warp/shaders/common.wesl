// Shared by every line-like primitive (shapes, grid, particles, text): the camera, and thick
// anti-aliased lines drawn as quads with a capsule distance in the fragment shader. The
// geometry is gax, traced at build time (`src/kernels.rs`): the quad's corners
// (`segment_corner`) and the distance to the segment by joins (`segment_distance`).
//
// A line's colour is a light: a PGA3D point in RGB space whose coordinates are its radiance
// (`src/light.rs`).

import package::fused::{segment_corner, segment_distance};
import gax::pga2d::point_new;

// The camera: a gax map taking world points `(x, y, 1)` to clip space, uploaded as the WGSL
// matrix of `proj.of(view)` (columns are inputs, applied as `m * p`).
struct Camera {
    view_proj: mat3x3<f32>,
    // Width and height in pixels, world units per pixel, time in seconds.
    viewport: vec4<f32>,
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

// Corner `k` (0..3, a triangle strip) of the quad around the segment `a`-`b`, grown to hold
// the core, the glow and two pixels of anti-aliasing.
fn line_vertex(cam: Camera, k: u32, a: vec2<f32>, b: vec2<f32>, light: vec4<f32>, style: vec4<f32>) -> LineVarying {
    let extent = style.x + style.y * 1.5 + 2.0 * cam.viewport.z;
    let along = select(-1.0, 1.0, (k & 1u) == 1u);
    let side = select(-1.0, 1.0, (k & 2u) == 2u);
    let corner = segment_corner(point_new(a.x, a.y, 1.0), point_new(b.x, b.y, 1.0), array<f32, 3>(along, side, extent));
    let world = corner.c0.xy / corner.c0.z;
    return LineVarying(to_clip(cam, world), world, a, b, light, style);
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
