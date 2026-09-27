// Shared by every line-like primitive (shapes, grid, particles, text): the camera, and thick
// anti-aliased lines drawn as quads with an analytic capsule distance in the fragment shader.

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
    @location(3) @interpolate(flat) color: vec4<f32>,
    // Half width, glow radius, glow strength, unused (world units).
    @location(4) @interpolate(flat) style: vec4<f32>,
}

fn to_clip(cam: Camera, p: vec2<f32>) -> vec4<f32> {
    let c = cam.view_proj * vec3<f32>(p, 1.0);
    return vec4<f32>(c.xy / c.z, 0.0, 1.0);
}

// Corner `k` (0..3, a triangle strip) of the quad around the segment `a`-`b`, grown to hold
// the core, the glow and two pixels of anti-aliasing.
fn line_vertex(cam: Camera, k: u32, a: vec2<f32>, b: vec2<f32>, color: vec4<f32>, style: vec4<f32>) -> LineVarying {
    let px = cam.viewport.z;
    let e = style.x + style.y * 1.5 + 2.0 * px;
    let d = b - a;
    let len = length(d);
    let u = select(vec2<f32>(1.0, 0.0), d / len, len > 1e-6);
    let n = vec2<f32>(-u.y, u.x);
    let along = select(-e, len + e, (k & 1u) == 1u);
    let side = select(-e, e, (k & 2u) == 2u);
    let world = a + u * along + n * side;
    return LineVarying(to_clip(cam, world), world, a, b, color, style);
}

// The line's light at this fragment: a solid anti-aliased core and a Gaussian glow, additive.
fn line_shade(v: LineVarying, px: f32) -> vec4<f32> {
    let pa = v.world - v.a;
    let ba = v.b - v.a;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-12), 0.0, 1.0);
    let d = length(pa - ba * h);
    // A line thinner than a pixel keeps its energy: widen it, and dim it by the same factor.
    let w = max(v.style.x, 0.6 * px);
    let thin = v.style.x / w;
    let core = (1.0 - smoothstep(w - px, w + px, d)) * thin;
    let g = max(v.style.y, 1e-4);
    let glow = v.style.z * exp(-2.5 * d * d / (g * g));
    return vec4<f32>(v.color.rgb * v.color.a * (core + glow), 1.0);
}
