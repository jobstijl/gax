// Post-processing: bloom by a mip chain (Jimenez, "Next Generation Post Processing in Call of
// Duty"), then composite: bloom, AgX tonemapping with a saturated look for neon, and optional
// vignette, edge chromatic aberration, grain and a shock ripple.

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

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn karis(c: vec3<f32>) -> f32 {
    return 1.0 / (1.0 + luma(c));
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

// AgX (Troy Sobotka), in the minimal fitted form by Benjamin Wrensch.
fn agx_contrast(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

fn agx(c: vec3<f32>, saturation: f32) -> vec3<f32> {
    let inset = mat3x3<f32>(
        vec3<f32>(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
        vec3<f32>(0.0784335999999992, 0.878468636469772, 0.0784336),
        vec3<f32>(0.0792237451477643, 0.0791661274605434, 0.879142973793104),
    );
    let outset = mat3x3<f32>(
        vec3<f32>(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
        vec3<f32>(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
        vec3<f32>(-0.0990297440797205, -0.0989611768448433, 1.15107367264116),
    );
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var v = inset * max(c, vec3<f32>(1e-10));
    v = clamp(log2(v), vec3<f32>(min_ev), vec3<f32>(max_ev));
    v = (v - min_ev) / (max_ev - min_ev);
    v = agx_contrast(v);
    // Look: more saturation, so that neon stays neon.
    let l = luma(v);
    v = l + saturation * (v - l);
    v = outset * v;
    // To linear light (the surface encodes sRGB).
    return pow(max(v, vec3<f32>(0.0)), vec3<f32>(2.2));
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

@fragment
fn fs_composite(f: Full) -> @location(0) vec4<f32> {
    var uv = f.uv;
    // Shock ripple: a ring pushing the image outwards.
    if (post.shock.w > 0.0) {
        let d = uv - post.shock.xy;
        let r = length(d);
        let ring = exp(-pow((r - post.shock.z) * 18.0, 2.0));
        uv = uv - d / max(r, 1e-4) * ring * post.shock.w;
    }
    let centre = uv - 0.5;
    let edge = dot(centre, centre);
    // Chromatic aberration towards the edges.
    let ca = post.look.z * edge;
    var hdr = vec3<f32>(
        textureSampleLevel(src, samp, uv - centre * ca, 0.0).r,
        textureSampleLevel(src, samp, uv, 0.0).g,
        textureSampleLevel(src, samp, uv + centre * ca, 0.0).b,
    );
    hdr += textureSampleLevel(bloom, samp, uv, 0.0).rgb * post.k.y;
    hdr *= post.k.z;
    var c = agx(hdr, post.look.w);
    c *= 1.0 - post.look.x * edge * 1.6;
    let n = hash(f.pos.xy + vec2<f32>(post.k.w * 61.0, post.k.w * 17.0)) - 0.5;
    c += n * post.look.y;
    return vec4<f32>(max(c, vec3<f32>(0.0)), 1.0);
}
