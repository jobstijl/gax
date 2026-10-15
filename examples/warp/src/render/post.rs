//! Post-processing: bloom down and up a mip chain of the HDR scene, then the composite into
//! the output (bloom, AgX tonemapping, vignette, chromatic aberration, grain and a shock
//! ripple; `shaders/post.wesl`). [`Post`] owns its pipelines, its uniforms and the targets
//! sized to the output.

use super::factory::{Blend, Factory, Target};
use super::gpu_types::PostUniform;
use super::timer::{Span, Timer};
use super::{HDR, begin};
use gax::pga2d::Point;

/// Bloom levels, each half the size of the one above.
const BLOOM_MIPS: usize = 6;

/// Post-processing settings.
#[derive(Clone, Copy, Debug)]
pub struct PostSettings {
    /// Bloom strength.
    pub bloom: f32,
    /// Exposure.
    pub exposure: f32,
    /// Vignette.
    pub vignette: f32,
    /// Film grain.
    pub grain: f32,
    /// Chromatic aberration at the edges.
    pub aberration: f32,
    /// Saturation of the tonemapper's look.
    pub saturation: f32,
    /// A shock ripple.
    pub shock: Option<Shock>,
}

/// A shock ripple across the screen.
#[derive(Clone, Copy, Debug)]
pub struct Shock {
    /// Its centre in the target's uv coordinates (`0..1`, y down).
    pub at: Point<(), f32>,
    /// Its radius, in uv.
    pub radius: f32,
    /// How far it displaces.
    pub strength: f32,
}

/// The textures sized to the output, and the bind groups that read them.
struct Targets {
    /// The HDR scene.
    hdr: wgpu::TextureView,
    /// The bloom levels and their sizes.
    mips: Vec<(wgpu::TextureView, [u32; 2])>,
    down: Vec<wgpu::BindGroup>,
    up: Vec<wgpu::BindGroup>,
    composite: wgpu::BindGroup,
    size: [u32; 2],
}

/// The passes: their pipelines, sampler and uniforms.
struct Passes {
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    /// One per pass: the downsamples, the upsamples, then the composite.
    uniforms: Vec<wgpu::Buffer>,
}

/// Bloom and the composite.
pub(super) struct Post {
    passes: Passes,
    targets: Targets,
}

impl Post {
    /// Post-processing into `format`, for an output of `size`.
    pub fn new(f: Factory, q: &wgpu::Queue, format: wgpu::TextureFormat, size: [u32; 2]) -> Post {
        let module = f.module("post", super::wgsl::POST);
        let full = |label, fs, target, blend| {
            f.full_screen(label, &module, ["vs_full", fs], target, blend)
        };
        let passes = Passes {
            down: full("bloom down", "fs_down", Target::Hdr, Blend::Replace),
            up: full("bloom up", "fs_up", Target::Hdr, Blend::Add),
            composite: full(
                "composite",
                "fs_composite",
                Target::Display(format),
                Blend::Replace,
            ),
            sampler: f.0.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("linear clamp"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            uniforms: (0..2 * BLOOM_MIPS)
                .map(|_| f.uniform("post", size_of::<PostUniform>()))
                .collect(),
        };
        let targets = passes.targets(f, size);
        let post = Post { passes, targets };
        post.write_bloom(q);
        post
    }

    /// The HDR scene's target.
    pub fn hdr(&self) -> &wgpu::TextureView {
        &self.targets.hdr
    }

    /// The output's size.
    pub fn size(&self) -> [u32; 2] {
        self.targets.size
    }

    /// The targets for an output of `size`.
    pub fn resize(&mut self, f: Factory, q: &wgpu::Queue, size: [u32; 2]) {
        self.targets = self.passes.targets(f, size);
        self.write_bloom(q);
    }

    /// The bloom passes' uniforms: the texel sizes of what they read.
    fn write_bloom(&self, q: &wgpu::Queue) {
        let texel = |[w, h]: [u32; 2]| [1.0 / w as f32, 1.0 / h as f32, 1.0, 0.0];
        let mips = &self.targets.mips;
        for k in 0..BLOOM_MIPS {
            let source = if k == 0 {
                self.targets.size
            } else {
                mips[k - 1].1
            };
            let u = PostUniform {
                texel: texel(source),
                // The Karis average on the first downsample.
                k: [if k == 0 { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
                ..Default::default()
            };
            q.write_buffer(&self.passes.uniforms[k], 0, bytemuck::bytes_of(&u));
        }
        for k in 0..BLOOM_MIPS - 1 {
            let u = PostUniform {
                texel: texel(mips[k + 1].1),
                ..Default::default()
            };
            q.write_buffer(
                &self.passes.uniforms[BLOOM_MIPS + k],
                0,
                bytemuck::bytes_of(&u),
            );
        }
    }

    /// Write the composite's uniform: `p` at `time`.
    pub fn write(&self, q: &wgpu::Queue, p: &PostSettings, time: f32) {
        let shock = p.shock.map_or([0.5, 0.5, 0.0, 0.0], |s| {
            let [u, v] = s.at.to_euclidean();
            [u, v, s.radius, s.strength]
        });
        let u = PostUniform {
            texel: [0.0; 4],
            k: [0.0, p.bloom, p.exposure, time],
            look: [p.vignette, p.grain, p.aberration, p.saturation],
            shock,
        };
        q.write_buffer(
            &self.passes.uniforms[2 * BLOOM_MIPS - 1],
            0,
            bytemuck::bytes_of(&u),
        );
    }

    /// Bloom down the chain and back up, then the composite into `output`.
    pub fn encode(
        &self,
        enc: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        timer: Option<&Timer>,
    ) {
        let (p, t) = (&self.passes, &self.targets);
        let clear = wgpu::LoadOp::Clear(wgpu::Color::BLACK);
        for k in 0..BLOOM_MIPS {
            let ts = Timer::render(timer, (k == 0).then_some(Span::Bloom), None);
            let mut pass = begin(enc, "bloom down", &t.mips[k].0, clear, ts);
            draw(&mut pass, &p.down, &t.down[k]);
        }
        // Up k reads level k + 1 and adds into level k.
        for k in (0..BLOOM_MIPS - 1).rev() {
            let ts = Timer::render(timer, None, (k == 0).then_some(Span::Bloom));
            let mut pass = begin(enc, "bloom up", &t.mips[k].0, wgpu::LoadOp::Load, ts);
            draw(&mut pass, &p.up, &t.up[k]);
        }
        let ts = Timer::render(timer, Some(Span::Composite), Some(Span::Composite));
        let mut pass = begin(enc, "composite", output, clear, ts);
        draw(&mut pass, &p.composite, &t.composite);
    }
}

impl Passes {
    /// The textures for an output of `size`, and the bind groups reading them.
    fn targets(&self, f: Factory, size: [u32; 2]) -> Targets {
        let size = size.map(|n| n.max(1));
        let hdr = texture(f, "hdr", size);
        let mut mips: Vec<(wgpu::TextureView, [u32; 2])> = Vec::new();
        let mut s = size;
        for _ in 0..BLOOM_MIPS {
            s = s.map(|n| (n / 2).max(1));
            mips.push((texture(f, "bloom", s), s));
        }
        // Down k reads the level above (the HDR scene for k = 0) into level k.
        let down = (0..BLOOM_MIPS)
            .map(|k| {
                let src = if k == 0 { &hdr } else { &mips[k - 1].0 };
                self.bind(f, &self.down, k, src, None)
            })
            .collect();
        let up = (0..BLOOM_MIPS - 1)
            .map(|k| self.bind(f, &self.up, BLOOM_MIPS + k, &mips[k + 1].0, None))
            .collect();
        let composite = self.bind(
            f,
            &self.composite,
            2 * BLOOM_MIPS - 1,
            &hdr,
            Some(&mips[0].0),
        );
        Targets {
            hdr,
            mips,
            down,
            up,
            composite,
            size,
        }
    }

    /// The bind group of a pass of `pipeline` with uniform `k`, reading `src` (and `bloom`).
    fn bind(
        &self,
        f: Factory,
        pipeline: &wgpu::RenderPipeline,
        k: usize,
        src: &wgpu::TextureView,
        bloom: Option<&wgpu::TextureView>,
    ) -> wgpu::BindGroup {
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(src),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&self.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: self.uniforms[k].as_entire_binding(),
            },
        ];
        if let Some(b) = bloom {
            entries.push(wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(b),
            });
        }
        f.0.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    }
}

/// An HDR texture of `size` to render into and sample.
fn texture(f: Factory, label: &str, [width, height]: [u32; 2]) -> wgpu::TextureView {
    f.0.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
    .create_view(&wgpu::TextureViewDescriptor::default())
}

/// One full-screen triangle with `pipeline` reading `group`.
fn draw(pass: &mut wgpu::RenderPass, pipeline: &wgpu::RenderPipeline, group: &wgpu::BindGroup) {
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.draw(0..3, 0..1);
}
