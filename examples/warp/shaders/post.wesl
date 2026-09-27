// Post-processing: bloom by a mip chain (Jimenez, "Next Generation Post Processing in Call of
// Duty"), then composite: bloom, AgX tonemapping with a saturated look for neon, and optional
// vignette, edge chromatic aberration, grain and a shock ripple.
//
// The colour and the geometry are traced gax kernels (`src/kernels.rs`): a pixel's light is a
// PGA3D point whose coordinates are its radiance; luminance is its pairing with a plane, the
// tonemapper applies gax maps and moves away from grey; the ripple pushes points along their
// radial direction; the edge effects scale about the screen's centre point.

import package::fused::{luma, agx, ripple, scale_about, distance, light_fade};
import gax::pga2d::point_new;

struct Post {
    // 1 / width, 1 / height of the source, upsample radius, unused.
    texel: vec4<f32>,
    // Karis average on (first downsample), bloom strength, exposure, time.
    k: vec4<f32>,
    // Vignette, grain, chromatic aberration, saturation.
    look: vec4<f32>,
    // Shock ripple: centre (uv), radius, strength.
    shock: vec4<f32>,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> post: Post;
@group(0) @binding(3) var bloom: texture_2d<f32>;

struct Full {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_full(@builtin(vertex_index) k: u32) -> Full {
    let uv = vec2<f32>(f32((k << 1u) & 2u), f32(k & 2u));
    return Full(vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0), uv);
}

// A radiance as a light of weight 1.
fn as_light(c: vec3<f32>) -> gax::pga3d::Point {
    return gax::pga3d::Point(vec4<f32>(c, 1.0));
}

fn karis(c: vec3<f32>) -> f32 {
    return 1.0 / (1.0 + luma(as_light(c)));
}

fn tap(uv: vec2<f32>, o: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(src, samp, uv + o * post.texel.xy, 0.0).rgb;
}

@fragment
fn fs_down(f: Full) -> @location(0) vec4<f32> {
    let uv = f.uv;
    let a = tap(uv, vec2<f32>(-2.0, 2.0));
    let b = tap(uv, vec2<f32>(0.0, 2.0));
    let c = tap(uv, vec2<f32>(2.0, 2.0));
    let d = tap(uv, vec2<f32>(-2.0, 0.0));
    let e = tap(uv, vec2<f32>(0.0, 0.0));
    let ff = tap(uv, vec2<f32>(2.0, 0.0));
    let g = tap(uv, vec2<f32>(-2.0, -2.0));
    let h = tap(uv, vec2<f32>(0.0, -2.0));
    let i = tap(uv, vec2<f32>(2.0, -2.0));
    let j = tap(uv, vec2<f32>(-1.0, 1.0));
    let k = tap(uv, vec2<f32>(1.0, 1.0));
    let l = tap(uv, vec2<f32>(-1.0, -1.0));
    let m = tap(uv, vec2<f32>(1.0, -1.0));
    // Five overlapping 2x2 boxes; on the first level each weighted by the Karis average, which
    // keeps single bright pixels from flickering into huge blooms.
    let g0 = (a + b + d + e) * 0.25;
    let g1 = (b + c + e + ff) * 0.25;
    let g2 = (d + e + g + h) * 0.25;
    let g3 = (e + ff + h + i) * 0.25;
    let g4 = (j + k + l + m) * 0.25;
    if (post.k.x > 0.5) {
        let w0 = karis(g0) * 0.125;
        let w1 = karis(g1) * 0.125;
        let w2 = karis(g2) * 0.125;
        let w3 = karis(g3) * 0.125;
        let w4 = karis(g4) * 0.5;
        let s = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
        return vec4<f32>(max(s, vec3<f32>(0.0)), 1.0);
    }
    return vec4<f32>(g0 * 0.125 + g1 * 0.125 + g2 * 0.125 + g3 * 0.125 + g4 * 0.5, 1.0);
}

@fragment
fn fs_up(f: Full) -> @location(0) vec4<f32> {
    // A 3x3 tent, added onto the next larger mip by the blend state.
    let r = post.texel.z;
    let uv = f.uv;
    var s = tap(uv, vec2<f32>(0.0, 0.0)) * 4.0;
    s += (tap(uv, vec2<f32>(-r, 0.0)) + tap(uv, vec2<f32>(r, 0.0)) + tap(uv, vec2<f32>(0.0, -r)) + tap(uv, vec2<f32>(0.0, r))) * 2.0;
    s += tap(uv, vec2<f32>(-r, -r)) + tap(uv, vec2<f32>(r, -r)) + tap(uv, vec2<f32>(-r, r)) + tap(uv, vec2<f32>(r, r));
    return vec4<f32>(s / 16.0, 1.0);
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// A screen point (uv) as a PGA2D point, and back.
fn at(uv: vec2<f32>) -> gax::pga2d::Point {
    return point_new(uv.x, uv.y, 1.0);
}

fn uv_of(p: gax::pga2d::Point) -> vec2<f32> {
    return p.c0.xy / p.c0.z;
}

@fragment
fn fs_composite(f: Full) -> @location(0) vec4<f32> {
    var p = at(f.uv);
    // Shock ripple: a ring pushing the image outwards.
    if (post.shock.w > 0.0) {
        p = ripple(p, at(post.shock.xy), array<f32, 2>(post.shock.z, post.shock.w));
    }
    let mid = at(vec2<f32>(0.5, 0.5));
    let r = distance(p, mid);
    let edge = r * r;
    // Chromatic aberration towards the edges: red and blue sampled a little in and out, the
    // point scaled about the screen's centre.
    let ca = post.look.z * edge;
    let uv = uv_of(p);
    var hdr = vec3<f32>(
        textureSampleLevel(src, samp, uv_of(scale_about(p, mid, 1.0 - ca)), 0.0).r,
        textureSampleLevel(src, samp, uv, 0.0).g,
        textureSampleLevel(src, samp, uv_of(scale_about(p, mid, 1.0 + ca)), 0.0).b,
    );
    hdr += textureSampleLevel(bloom, samp, uv, 0.0).rgb * post.k.y;
    var c = agx(light_fade(as_light(hdr), post.k.z), post.look.w);
    // The vignette dims the light towards the edges.
    c = light_fade(c, 1.0 - post.look.x * edge * 1.6);
    // Grain: a little grey light added (lights add as points).
    let n = (hash(f.pos.xy + vec2<f32>(post.k.w * 61.0, post.k.w * 17.0)) - 0.5) * post.look.y;
    let out = c.c0.xyz + vec3<f32>(n);
    return vec4<f32>(max(out, vec3<f32>(0.0)), 1.0);
}
