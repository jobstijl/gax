//! A minimal compute harness for testing gax's WGSL kernels on a GPU (docs/shaders.md, layer 5).
//!
//! [`Gpu::new`] finds an adapter (in CI, Mesa's software Vulkan driver lavapipe);
//! [`Gpu::run`] runs one compute entry point over storage buffers and reads the last one back.

use std::time::Duration;
use wgpu::util::DeviceExt;

/// A device and its queue.
pub struct Gpu {
    /// The device.
    pub device: wgpu::Device,
    /// Its queue.
    pub queue: wgpu::Queue,
    /// What the adapter is.
    pub info: wgpu::AdapterInfo,
    /// Whether the device has `shader-f16` (the `f16` modules need it).
    pub f16: bool,
}

impl Gpu {
    /// A device on the default adapter, or `None` when there is none.
    pub fn new() -> Option<Gpu> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        let info = adapter.get_info();
        let f16 = adapter.features().contains(wgpu::Features::SHADER_F16);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gax-gpu-tests"),
            required_limits: adapter.limits(),
            required_features: if f16 {
                wgpu::Features::SHADER_F16
            } else {
                wgpu::Features::empty()
            },
            ..Default::default()
        }))
        .ok()?;
        Some(Gpu {
            device,
            queue,
            info,
            f16,
        })
    }

    /// A device, or `None` with a message on stderr. With `GAX_REQUIRE_GPU` set (as in CI), a
    /// missing adapter is a failure instead.
    pub fn or_skip() -> Option<Gpu> {
        let gpu = Gpu::new();
        match &gpu {
            Some(g) => eprintln!(
                "GPU: {} ({:?}, {:?})",
                g.info.name, g.info.device_type, g.info.backend
            ),
            None if std::env::var_os("GAX_REQUIRE_GPU").is_some() => {
                panic!("no GPU adapter, and GAX_REQUIRE_GPU is set")
            }
            None => eprintln!("no GPU adapter: skipping (set GAX_REQUIRE_GPU to fail instead)"),
        }
        gpu
    }

    /// Run `entry` of the WGSL `source` with `invocations` invocations (workgroups of 64), with
    /// `inputs` bound read-only at bindings `0..n` of group 0 and an output of `out_bytes`
    /// bytes at binding `n`, and return the output.
    pub fn run(
        &self,
        source: &str,
        entry: &str,
        inputs: &[&[u8]],
        out_bytes: usize,
        invocations: u32,
    ) -> Vec<u8> {
        let d = &self.device;
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        let bufs: Vec<wgpu::Buffer> = inputs
            .iter()
            .map(|b| {
                // Storage buffers may not be empty.
                let data: Vec<u8> = if b.is_empty() {
                    vec![0; 16]
                } else {
                    b.to_vec()
                };
                d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: &data,
                    usage: wgpu::BufferUsages::STORAGE,
                })
            })
            .collect();
        let out = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: out_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read"),
            size: out_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: bufs.len() as u32,
            resource: out.as_entire_binding(),
        });
        let group = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut enc = d.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(invocations.div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &read, 0, out_bytes as u64);
        self.queue.submit([enc.finish()]);
        let slice = read.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map the output"));
        d.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(600)),
        })
        .expect("the GPU finishes");
        let bytes = slice
            .get_mapped_range()
            .expect("the mapped output")
            .to_vec();
        read.unmap();
        bytes
    }
}

/// `f32`s from little-endian bytes.
pub fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}
