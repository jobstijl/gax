//! What the renderer's parts build on the device: shader modules, buffers, bind groups, and
//! its three kinds of pipeline (quads of lines, full-screen triangles, compute).

use super::HDR;
use wgpu::util::DeviceExt;

/// Builds GPU objects on a device.
#[derive(Clone, Copy)]
pub(super) struct Factory<'a>(pub &'a wgpu::Device);

/// Where a render pipeline draws.
#[derive(Clone, Copy)]
pub(super) enum Target {
    /// The HDR scene (or a bloom level), in linear light.
    Hdr,
    /// The display, in an sRGB `format`: a shader writes linear light, which the hardware
    /// encodes.
    Display(wgpu::TextureFormat),
}

/// How a render pipeline's output meets what is in its target.
#[derive(Clone, Copy)]
pub(super) enum Blend {
    /// Added, as light adds.
    Add,
    /// Replacing it.
    Replace,
}

impl Factory<'_> {
    /// A shader module from linked WGSL.
    pub fn module(self, label: &str, wgsl: &str) -> wgpu::ShaderModule {
        self.0.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        })
    }

    /// A uniform buffer of `size` bytes, written by the queue.
    pub fn uniform(self, label: &str, size: usize) -> wgpu::Buffer {
        self.0.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// A storage buffer holding `contents` (written by the queue, read back by the tests).
    pub fn storage(self, label: &str, contents: &[u8]) -> wgpu::Buffer {
        self.0
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
    }

    /// A storage buffer of `size` zero bytes.
    pub fn empty_storage(self, label: &str, size: usize) -> wgpu::Buffer {
        self.0.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// A bind group for group 0 of `layout`: buffer `k` at binding `k`.
    pub fn group(
        self,
        layout: &wgpu::BindGroupLayout,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = (0u32..)
            .zip(buffers)
            .map(|(binding, b)| wgpu::BindGroupEntry {
                binding,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.0.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &entries,
        })
    }

    /// A pipeline of lines: `vs_main` places the four corners of a quad (a triangle strip) per
    /// instance, `fs` shades it, and the light adds into `target`.
    pub fn quads(
        self,
        label: &str,
        module: &wgpu::ShaderModule,
        fs: &str,
        target: Target,
        instances: Option<wgpu::VertexBufferLayout>,
    ) -> wgpu::RenderPipeline {
        let topology = wgpu::PrimitiveTopology::TriangleStrip;
        let buffers: Vec<Option<wgpu::VertexBufferLayout>> =
            instances.into_iter().map(Some).collect();
        self.render(
            label,
            module,
            ["vs_main", fs],
            &buffers,
            topology,
            target,
            Blend::Add,
        )
    }

    /// A pipeline drawing one triangle over the whole of `target` (`vs` and `fs`).
    pub fn full_screen(
        self,
        label: &str,
        module: &wgpu::ShaderModule,
        [vs, fs]: [&str; 2],
        target: Target,
        blend: Blend,
    ) -> wgpu::RenderPipeline {
        let topology = wgpu::PrimitiveTopology::TriangleList;
        self.render(label, module, [vs, fs], &[], topology, target, blend)
    }

    #[allow(clippy::too_many_arguments)]
    fn render(
        self,
        label: &str,
        module: &wgpu::ShaderModule,
        [vs, fs]: [&str; 2],
        buffers: &[Option<wgpu::VertexBufferLayout>],
        topology: wgpu::PrimitiveTopology,
        target: Target,
        blend: Blend,
    ) -> wgpu::RenderPipeline {
        let format = match target {
            Target::Hdr => HDR,
            Target::Display(format) => format,
        };
        let blend = match blend {
            Blend::Add => Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::REPLACE,
            }),
            Blend::Replace => None,
        };
        self.0
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: None,
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers,
                },
                primitive: wgpu::PrimitiveState {
                    topology,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
    }

    /// A compute pipeline from `module`'s `main`.
    pub fn compute(self, label: &str, module: &wgpu::ShaderModule) -> wgpu::ComputePipeline {
        self.0
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
    }
}
