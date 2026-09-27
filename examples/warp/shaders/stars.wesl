// A faint parallax starfield behind the arena: three layers of hashed stars, moving slower
// than the world as the camera pans.
import package::common::Camera;

struct Stars {
    // Camera centre (world), unused, unused.
    centre: vec4<f32>,
}

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> stars: Stars;

struct Full {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) k: u32) -> Full {
    let uv = vec2<f32>(f32((k << 1u) & 2u), f32(k & 2u));
    return Full(vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0), uv);
}

fn hash2(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(q) * 43758.5453);
}

@fragment
fn fs_main(f: Full) -> @location(0) vec4<f32> {
    let px = f.pos.xy;
    var c = vec3<f32>(0.0);
    for (var layer = 0; layer < 3; layer++) {
        let depth = 0.15 + 0.2 * f32(layer);
        let scale = 38.0 + 22.0 * f32(layer);
        let p = (px + stars.centre.xy * vec2<f32>(1.0, -1.0) / cam.viewport.z * depth) / scale;
        let cell = floor(p);
        let h = hash2(cell + f32(layer) * 17.0);
        let star = cell + 0.15 + 0.7 * h;
        let d = length(p - star) * scale;
        let twinkle = 0.7 + 0.3 * sin(cam.viewport.w * (1.0 + 2.0 * h.x) + h.y * 40.0);
        let bright = select(0.0, 1.0, h.x > 0.55) * (0.25 + h.y) * twinkle / (1.0 + f32(layer));
        let tint = mix(vec3<f32>(0.6, 0.7, 1.0), vec3<f32>(1.0, 0.8, 0.9), h.y);
        c += tint * bright * exp(-d * d * 0.9) * 0.08;
    }
    return vec4<f32>(c, 1.0);
}
