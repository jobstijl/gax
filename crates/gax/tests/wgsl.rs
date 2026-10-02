//! The WGSL modules without a GPU (docs/shaders.md, test layers 2 and 3): every generated
//! module parses and validates with naga, and naga's layout of every kind's struct and of the
//! matrices equals the layout of the Rust types (`{Kind}Gpu`, `GpuMat`).

#![cfg(all(feature = "wgsl", feature = "bytemuck"))]
#![allow(clippy::needless_range_loop)]

use gax::wgsl::Module;
use naga::valid::{Capabilities, ValidationFlags, Validator};

fn parse(m: &Module) -> naga::Module {
    let module = naga::front::wgsl::parse_str(m.source)
        .unwrap_or_else(|e| panic!("{}: {}", m.path, e.emit_to_string(m.source)));
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}: {}", m.path, e.emit_to_string(m.source)));
    module
}

/// `(size, align, member offsets)` of the named struct.
fn layout(module: &naga::Module, name: &str) -> (u32, u32, Vec<u32>) {
    let mut layouter = naga::proc::Layouter::default();
    layouter.update(module.to_ctx()).expect("layout");
    let (h, ty) = module
        .types
        .iter()
        .find(|(_, t)| t.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no struct {name}"));
    let naga::TypeInner::Struct { members, .. } = &ty.inner else {
        panic!("{name} is not a struct");
    };
    let l = layouter[h];
    (
        l.size,
        l.alignment.round_up(1),
        members.iter().map(|m| m.offset).collect(),
    )
}

#[test]
fn every_module_validates() {
    assert_ne!(gax::wgsl::ALL, []);
    for m in gax::wgsl::ALL {
        let module = parse(m);
        let fns = module.functions.len();
        println!("{}: {fns} functions, {} types", m.path, module.types.len());
        assert!(fns > 20, "{}: {fns} functions", m.path);
    }
}

#[test]
fn every_f16_module_validates() {
    assert_eq!(gax::wgsl::ALL_F16.len(), gax::wgsl::ALL.len());
    for (m, m32) in gax::wgsl::ALL_F16.iter().zip(gax::wgsl::ALL) {
        assert!(m.source.starts_with("enable f16;"), "{}", m.path);
        let names = |m: &naga::Module| -> std::collections::BTreeSet<String> {
            m.functions
                .iter()
                .filter_map(|(_, f)| f.name.clone())
                .collect()
        };
        let (f16, f32) = (names(&parse(m)), names(&parse(m32)));
        // The same functions as the f32 module, and where `exp` is the closed form in f16 (it
        // computes in f32 up to its final assembly), scaling and squaring as `*_exp_squaring`.
        assert!(f32.is_subset(&f16), "{}", m.path);
        let extra: Vec<&String> = f16.difference(&f32).collect();
        assert!(
            extra.iter().all(|n| n.ends_with("_exp_squaring")),
            "{}: {extra:?}",
            m.path
        );
    }
}

/// An algebra's `GPU_LAYOUTS`.
type Layouts = &'static [(&'static str, usize, usize, usize, usize)];

/// `(module, the algebra's GPU_LAYOUTS)` for the enabled algebras.
#[allow(clippy::vec_init_then_push)] // one push per enabled algebra
fn layouts() -> Vec<(Module, Layouts)> {
    let mut v: Vec<(Module, Layouts)> = Vec::new();
    #[cfg(feature = "pga2d")]
    v.push((gax::wgsl::PGA2D, gax::pga2d::GPU_LAYOUTS));
    #[cfg(feature = "pga3d")]
    v.push((gax::wgsl::PGA3D, gax::pga3d::GPU_LAYOUTS));
    #[cfg(feature = "vga2d")]
    v.push((gax::wgsl::VGA2D, gax::vga2d::GPU_LAYOUTS));
    #[cfg(feature = "vga3d")]
    v.push((gax::wgsl::VGA3D, gax::vga3d::GPU_LAYOUTS));
    #[cfg(feature = "sta")]
    v.push((gax::wgsl::STA, gax::sta::GPU_LAYOUTS));
    #[cfg(feature = "cga2d")]
    v.push((gax::wgsl::CGA2D, gax::cga2d::GPU_LAYOUTS));
    #[cfg(feature = "cga3d")]
    v.push((gax::wgsl::CGA3D, gax::cga3d::GPU_LAYOUTS));
    #[cfg(feature = "stap")]
    v.push((gax::wgsl::STAP, gax::stap::GPU_LAYOUTS));
    #[cfg(feature = "csta")]
    v.push((gax::wgsl::CSTA, gax::csta::GPU_LAYOUTS));
    v
}

/// `(f16 module, the algebra's GPU_LAYOUTS_F16)` for the enabled algebras.
#[allow(clippy::vec_init_then_push)] // one push per enabled algebra
fn layouts16() -> Vec<(Module, Layouts)> {
    let mut v: Vec<(Module, Layouts)> = Vec::new();
    #[cfg(feature = "pga2d")]
    v.push((gax::wgsl::PGA2D_F16, gax::pga2d::GPU_LAYOUTS_F16));
    #[cfg(feature = "pga3d")]
    v.push((gax::wgsl::PGA3D_F16, gax::pga3d::GPU_LAYOUTS_F16));
    #[cfg(feature = "vga2d")]
    v.push((gax::wgsl::VGA2D_F16, gax::vga2d::GPU_LAYOUTS_F16));
    #[cfg(feature = "vga3d")]
    v.push((gax::wgsl::VGA3D_F16, gax::vga3d::GPU_LAYOUTS_F16));
    #[cfg(feature = "sta")]
    v.push((gax::wgsl::STA_F16, gax::sta::GPU_LAYOUTS_F16));
    #[cfg(feature = "cga2d")]
    v.push((gax::wgsl::CGA2D_F16, gax::cga2d::GPU_LAYOUTS_F16));
    #[cfg(feature = "cga3d")]
    v.push((gax::wgsl::CGA3D_F16, gax::cga3d::GPU_LAYOUTS_F16));
    #[cfg(feature = "stap")]
    v.push((gax::wgsl::STAP_F16, gax::stap::GPU_LAYOUTS_F16));
    #[cfg(feature = "csta")]
    v.push((gax::wgsl::CSTA_F16, gax::csta::GPU_LAYOUTS_F16));
    v
}

#[test]
fn struct_layouts_match_the_rust_types() {
    for (m, rust) in layouts().into_iter().chain(layouts16()) {
        let module = parse(&m);
        for &(kind, size, align, offset, stride) in rust {
            let (s, a, members) = layout(&module, kind);
            assert_eq!(
                (s as usize, a as usize),
                (size, align),
                "{}::{kind}: size and align",
                m.path
            );
            let rust_offsets: Vec<u32> = (0..members.len())
                .map(|k| (offset + k * stride) as u32)
                .collect();
            assert_eq!(members, rust_offsets, "{}::{kind}: member offsets", m.path);
        }
    }
}

#[test]
fn matrix_layouts_match_gpu_mat() {
    let src = "struct M3 { m: mat3x3<f32> }\nstruct M4 { m: mat4x4<f32> }\nstruct M34 { m: mat3x4<f32> }\nstruct M43 { m: mat4x3<f32> }\n";
    let module = naga::front::wgsl::parse_str(src).expect("parse");
    let size = |n| layout(&module, n).0 as usize;
    assert_eq!(size("M3"), core::mem::size_of::<gax::GpuMat<3>>());
    assert_eq!(size("M4"), core::mem::size_of::<gax::GpuMat<4>>());
    // matCxR: C columns (inputs), each with the stride of vec4.
    assert_eq!(size("M34"), core::mem::size_of::<gax::GpuMat<3>>());
    assert_eq!(size("M43"), core::mem::size_of::<gax::GpuMat<4>>());
    assert_eq!(
        layout(&module, "M4").1 as usize,
        core::mem::align_of::<gax::GpuMat<4>>()
    );
}

/// The Rust conversions put coefficient `i` at `c[i / 4][i % 4]`, and maps transpose into
/// columns, as the WGSL side reads them.
#[cfg(feature = "pga3d")]
#[test]
fn conversions_follow_the_layout() {
    use gax::pga3d::{Line, LineGpu, Motor, Point};
    let l = Line::<(), f32>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    let g = LineGpu::from(l);
    assert_eq!(g.c, [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 0.0, 0.0]]);
    assert_eq!(Line::from(g), l);
    assert_eq!(gax::bytemuck::bytes_of(&g).len(), 32);
    let m = Motor::<(), f32>::new(0.8, 0.6, 0.0, 0.0, 0.3, -0.1, 0.2, 0.225);
    let map: Point<(Point,), f32> = m >> Point::slot();
    let gm = gax::GpuMat::<4>::from(map);
    let p = Point::<(), f32>::new(0.3, -0.2, 0.5, 1.0);
    // m * p with columns as inputs.
    let mut mp = [0.0f32; 4];
    for (i, col) in gm.cols.iter().enumerate() {
        for o in 0..4 {
            mp[o] += col[o] * p.c[i];
        }
    }
    let want = map.of(p);
    for o in 0..4 {
        assert!((mp[o] - want.c[o]).abs() < 1e-6);
    }
    assert_eq!(Point::<(Point,), f32>::from(gm), map);
}
