//! The renderer's data as the GPU sees it: `Pod` structs of gax's GPU types (`PointGpu`,
//! `MotorGpu`, `GpuMat`, whose layouts gax checks at compile time), each the mirror of a WGSL
//! struct in `shaders/`. A test parses the linked WGSL with naga and checks every member's
//! name and offset, and every struct's size, against these.

use super::lattice::Source;
use bytemuck::{Pod, Zeroable};
use gax::pga2d::{MotorGpu, PointGpu};
use gax::pga3d::PointGpu as Light3Gpu;

/// A line segment: endpoints in the local frame of `motor`, a light (a PGA3D point in RGB
/// space, `light.rs`), a style (half width, glow radius, glow strength, 0) in world units.
/// The line pipelines read it as per-instance vertex attributes ([`LINE_ATTRIBUTES`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LineInstance {
    /// The first endpoint: the WGSL `Point`.
    pub a: PointGpu,
    /// The second endpoint.
    pub b: PointGpu,
    /// The light: the WGSL `gax::pga3d::Point`.
    pub color: Light3Gpu,
    /// Half width, glow radius, glow strength, unused.
    pub style: [f32; 4],
    /// Where the segment's frame is: a unit PGA2D motor, the WGSL `Motor`.
    pub motor: MotorGpu,
}

impl LineInstance {
    /// Step back behind a menu: towards grey (its tint's foot on OkLab's lightness axis) and
    /// dimmer.
    pub fn recede(&mut self, k: f32) {
        let l = crate::light::Light::from(self.color).perceptually(|c| c.desaturated(0.75));
        self.color = l.faded(k).into();
    }
}

/// A `vec4<f32>` attribute at `offset` in a [`LineInstance`].
const fn attribute(location: u32, offset: usize) -> wgpu::VertexAttribute {
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: offset as u64,
        shader_location: location,
    }
}

/// [`LineInstance`]'s fields as the vertex attributes of `lines.wesl`'s `vs_main`, in order.
pub const LINE_ATTRIBUTES: [wgpu::VertexAttribute; 5] = [
    attribute(0, core::mem::offset_of!(LineInstance, a)),
    attribute(1, core::mem::offset_of!(LineInstance, b)),
    attribute(2, core::mem::offset_of!(LineInstance, color)),
    attribute(3, core::mem::offset_of!(LineInstance, style)),
    attribute(4, core::mem::offset_of!(LineInstance, motor)),
];

/// The camera uniform: the WGSL `Camera`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    /// World `(x, y, 1)` to clip space: gax's map `proj.of(view)` as a WGSL `mat3x3`.
    pub view_proj: gax::GpuMat<3>,
    /// Width, height (pixels), world units per pixel, time.
    pub viewport: [f32; 4],
}

/// A lattice node: the WGSL `Node`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Node {
    /// Position (a finite PGA2D point).
    pub p: PointGpu,
    /// Velocity (a direction).
    pub v: PointGpu,
}

/// A particle: the WGSL `Particle`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Particle {
    /// Position.
    pub p: PointGpu,
    /// Velocity.
    pub v: PointGpu,
    /// The light.
    pub color: Light3Gpu,
    /// Age, lifetime, drag, streak length (seconds of velocity).
    pub life: [f32; 4],
}

/// A source on the GPU: the WGSL `Source`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct SourceGpu {
    at: PointGpu,
    /// Pull, radius², unused, unused.
    k: [f32; 4],
}

impl From<Source> for SourceGpu {
    fn from(s: Source) -> SourceGpu {
        let [pull, radius2] = s.k();
        SourceGpu {
            at: s.at.into(),
            k: [pull, radius2, 0.0, 0.0],
        }
    }
}

/// `sources` as a full uniform array of `N`, the rest filled with [`Source::none`].
pub(super) fn source_array<const N: usize>(sources: &[Source]) -> [SourceGpu; N] {
    core::array::from_fn(|i| sources.get(i).copied().unwrap_or_else(Source::none).into())
}

/// One lattice step's uniform: `grid_step.wesl`'s `Grid`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct GridUniform {
    /// Columns, rows, sources, unused.
    pub dims: [u32; 4],
    /// `grid_node`'s `k`: spring stiffness, anchor stiffness, damping, dt.
    pub k: [f32; 4],
    /// The lower-left node's rest position.
    pub origin: PointGpu,
    /// The distance between nodes, unused, unused, unused.
    pub spacing: [f32; 4],
}

/// How the lattice looks: `grid_draw.wesl`'s `GridLook`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct GridLook {
    /// Columns, rows, unused, unused.
    pub dims: [u32; 4],
    /// The lattice's light.
    pub color: Light3Gpu,
    /// Half width, glow radius, glow strength, rest length.
    pub style: [f32; 4],
}

/// One particle step's uniform: `particles_step.wesl`'s `Step`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct StepUniform {
    /// Particles in the live window, wells, the window's first index in the ring, unused.
    pub dims: [u32; 4],
    /// dt, gravity scale, unused, unused.
    pub k: [f32; 4],
}

/// A post-processing pass's uniform: `post.wesl`'s `Post`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(super) struct PostUniform {
    /// 1 / width, 1 / height of the source, upsample radius, unused.
    pub texel: [f32; 4],
    /// Karis average on (the first downsample), bloom strength, exposure, time.
    pub k: [f32; 4],
    /// Vignette, grain, chromatic aberration, saturation.
    pub look: [f32; 4],
    /// The shock ripple: centre (uv), radius, strength.
    pub shock: [f32; 4],
}

#[cfg(test)]
mod tests {
    use super::super::particles::MAX_WELLS;
    use super::super::{MAX_SOURCES, wgsl};
    use super::*;
    use wgpu::naga;

    /// A Rust struct's layout: its size, and its fields' names and offsets in order.
    struct Layout {
        size: usize,
        members: Vec<(&'static str, usize)>,
    }

    macro_rules! layout {
        ($t:ty { $($field:ident),* }) => {
            Layout {
                size: size_of::<$t>(),
                members: vec![$((stringify!($field), core::mem::offset_of!($t, $field))),*],
            }
        };
    }

    /// The WGSL module of shader `name`, parsed.
    fn module(name: &str) -> naga::Module {
        let (_, src) = wgsl::ALL
            .iter()
            .find(|(n, _)| *n == name)
            .expect("a shader");
        naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    /// The struct a binding holds, and the length of the array it is in (`None` for a struct
    /// bound directly). Runtime-sized storage arrays have no length; a uniform array is the
    /// single member of a wrapper struct.
    fn bound(m: &naga::Module, binding: u32) -> (naga::Handle<naga::Type>, Option<u32>) {
        let var = m
            .global_variables
            .iter()
            .map(|(_, v)| v)
            .find(|v| v.binding.as_ref().is_some_and(|b| b.binding == binding))
            .expect("the binding");
        let element = |ty: naga::Handle<naga::Type>| match m.types[ty].inner {
            naga::TypeInner::Array { base, size, .. } => Some((base, size)),
            _ => None,
        };
        match (&m.types[var.ty].inner, element(var.ty)) {
            (_, Some((base, _))) => (base, None),
            (naga::TypeInner::Struct { members, .. }, _) if members.len() == 1 => {
                match element(members[0].ty) {
                    Some((base, naga::ArraySize::Constant(n))) => (base, Some(n.get())),
                    _ => (var.ty, None),
                }
            }
            _ => (var.ty, None),
        }
    }

    /// Check the struct at `binding` of shader `name` against `rust`, member by member.
    fn check(name: &str, binding: u32, rust: &Layout) -> Option<u32> {
        let m = module(name);
        let (ty, len) = bound(&m, binding);
        let naga::TypeInner::Struct { members, span } = &m.types[ty].inner else {
            panic!("{name}, binding {binding}: not a struct");
        };
        let wgsl: Vec<(&str, usize)> = members
            .iter()
            .map(|s| (s.name.as_deref().unwrap_or(""), s.offset as usize))
            .collect();
        assert_eq!(wgsl, rust.members, "{name}, binding {binding}: members");
        assert_eq!(*span as usize, rust.size, "{name}, binding {binding}: size");
        len
    }

    /// Every struct the shaders share with the renderer has the Rust struct's layout, and the
    /// uniform arrays have the renderer's lengths.
    #[test]
    fn the_shaders_structs_match_the_renderers() {
        let camera = layout!(CameraUniform {
            view_proj,
            viewport
        });
        let node = layout!(Node { p, v });
        let particle = layout!(Particle { p, v, color, life });
        let source = layout!(SourceGpu { at, k });
        for shader in ["grid_draw", "particles_draw", "lines", "stars"] {
            check(shader, 0, &camera);
        }
        check(
            "grid_step",
            0,
            &layout!(GridUniform {
                dims,
                k,
                origin,
                spacing
            }),
        );
        assert_eq!(check("grid_step", 1, &source), Some(MAX_SOURCES as u32));
        check("grid_step", 2, &node);
        check("grid_step", 3, &node);
        check("grid_draw", 1, &layout!(GridLook { dims, color, style }));
        check("grid_draw", 2, &node);
        check("particles_step", 0, &layout!(StepUniform { dims, k }));
        assert_eq!(check("particles_step", 1, &source), Some(MAX_WELLS as u32));
        check("particles_step", 2, &particle);
        check("particles_draw", 1, &particle);
        check(
            "post",
            2,
            &layout!(PostUniform {
                texel,
                k,
                look,
                shock
            }),
        );
        // The stars' uniform is the camera's centre, a point.
        let centre = Layout {
            size: size_of::<PointGpu>(),
            members: vec![("centre", 0)],
        };
        check("stars", 1, &centre);
    }

    /// The line pipelines' attributes are `vs_main`'s `vec4<f32>` inputs, location by location,
    /// and cover the whole instance.
    #[test]
    fn the_line_attributes_are_the_instance() {
        let m = module("lines");
        let vs = m
            .entry_points
            .iter()
            .find(|e| e.name == "vs_main")
            .expect("vs_main");
        let vec4 = |ty: naga::Handle<naga::Type>| {
            matches!(
                m.types[ty].inner,
                naga::TypeInner::Vector {
                    size: naga::VectorSize::Quad,
                    scalar: naga::Scalar::F32
                }
            )
        };
        let locations: Vec<u32> = vs
            .function
            .arguments
            .iter()
            .filter_map(|a| match a.binding {
                Some(naga::Binding::Location { location, .. }) => {
                    assert!(vec4(a.ty), "location {location}");
                    Some(location)
                }
                _ => None,
            })
            .collect();
        let ours: Vec<u32> = LINE_ATTRIBUTES.iter().map(|a| a.shader_location).collect();
        assert_eq!(locations, ours);
        let covered: u64 = LINE_ATTRIBUTES.iter().map(|a| a.format.size()).sum();
        assert_eq!(covered as usize, size_of::<LineInstance>());
    }
}
