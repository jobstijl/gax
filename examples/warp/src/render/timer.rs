//! GPU time per pass, from timestamp queries (when the adapter has them): every pass writes a
//! timestamp where a span begins or ends, and the results are read back a few frames later
//! without waiting for the GPU.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// GPU time per pass, in milliseconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct Timings {
    /// Grid and particle compute.
    pub compute: f32,
    /// The HDR scene.
    pub scene: f32,
    /// Bloom.
    pub bloom: f32,
    /// Tonemapping and composite.
    pub composite: f32,
}

/// A span of a frame that is timed: its timestamps are queries `2 span` and `2 span + 1`.
#[derive(Clone, Copy)]
pub(super) enum Span {
    Compute,
    Scene,
    Bloom,
    Composite,
}

const QUERIES: u32 = 8;
const BYTES: u64 = QUERIES as u64 * 8;

/// The timestamp queries and their readback.
pub(super) struct Timer {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    ready: Arc<AtomicBool>,
    pending: bool,
    period: f32,
    last: Timings,
}

impl Timer {
    /// A timer, if the device has timestamp queries.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Timer> {
        let buffer = |label, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: BYTES,
                usage,
                mapped_at_creation: false,
            })
        };
        let copy = wgpu::BufferUsages::COPY_SRC;
        let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| Timer {
                set: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: QUERIES,
                }),
                resolve: buffer(
                    "timestamp resolve",
                    wgpu::BufferUsages::QUERY_RESOLVE | copy,
                ),
                read: buffer("timestamp read", read),
                ready: Arc::new(AtomicBool::new(false)),
                pending: false,
                period: queue.get_timestamp_period(),
                last: Timings::default(),
            })
    }

    /// The latest timings read back.
    pub fn last(&self) -> Timings {
        self.last
    }

    /// Timestamps for a render pass where `begins` begins and `ends` ends (none if neither).
    pub fn render(
        timer: Option<&Timer>,
        begins: Option<Span>,
        ends: Option<Span>,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let timed = begins.is_some() || ends.is_some();
        timer
            .filter(|_| timed)
            .map(|t| wgpu::RenderPassTimestampWrites {
                query_set: &t.set,
                beginning_of_pass_write_index: begins.map(|s| 2 * s as u32),
                end_of_pass_write_index: ends.map(|s| 2 * s as u32 + 1),
            })
    }

    /// Timestamps for a compute pass that is `span`.
    pub fn compute(
        timer: Option<&Timer>,
        span: Span,
    ) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        timer.map(|t| wgpu::ComputePassTimestampWrites {
            query_set: &t.set,
            beginning_of_pass_write_index: Some(2 * span as u32),
            end_of_pass_write_index: Some(2 * span as u32 + 1),
        })
    }

    /// Take the results of an earlier frame, if they have arrived.
    pub fn poll(&mut self) {
        if !(self.pending && self.ready.swap(false, Ordering::Acquire)) {
            return;
        }
        {
            let view = self
                .read
                .slice(..)
                .get_mapped_range()
                .expect("mapped timestamps");
            let ts: &[u64] = bytemuck::cast_slice(&view);
            let ms = |s: Span| {
                let (a, b) = (ts[2 * s as usize], ts[2 * s as usize + 1]);
                (b.wrapping_sub(a) as f64 * f64::from(self.period) / 1e6) as f32
            };
            self.last = Timings {
                compute: ms(Span::Compute),
                scene: ms(Span::Scene),
                bloom: ms(Span::Bloom),
                composite: ms(Span::Composite),
            };
        }
        self.read.unmap();
        self.pending = false;
    }

    /// Resolve this frame's timestamps into the readback buffer, unless one is still pending.
    /// Returns whether it did, and [`Timer::map`] must follow the submission.
    pub fn resolve(&self, enc: &mut wgpu::CommandEncoder) -> bool {
        if self.pending {
            return false;
        }
        enc.resolve_query_set(&self.set, 0..QUERIES, &self.resolve, 0);
        enc.copy_buffer_to_buffer(&self.resolve, 0, &self.read, 0, BYTES);
        true
    }

    /// Map the readback buffer; [`Timer::poll`] takes the results when they arrive.
    pub fn map(&mut self) {
        let ready = self.ready.clone();
        self.read
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                if r.is_ok() {
                    ready.store(true, Ordering::Release);
                }
            });
        self.pending = true;
    }
}
