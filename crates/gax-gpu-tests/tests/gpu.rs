//! gax's WGSL kernels on a GPU (docs/shaders.md, layer 5).
//!
//! * Every kernel of eight algebras runs over random inputs, and each output is compared with
//!   the kernel's exact value: within twice its computed `f32` error bound for straight-line
//!   kernels (WGSL's `+`, `-`, `*` are correctly rounded, and fusing only removes roundings),
//!   and within a relative `2⁻¹⁰` for kernels that call WGSL's elementary functions, whose
//!   specified accuracy is loose (`sin` and `cos` to an absolute `2⁻¹¹`).
//! * The scaling-and-squaring exponentials (loops, not straight-line kernels) against gax's Rust
//!   `exp`, within a relative `2⁻¹⁰`.
//! * CSTA's `unit_even_log` (the closed form, turning planes near a half turn first) against
//!   gax's Rust `log`, within a relative `2⁻¹⁰`; its kernels, defined on unit versors only, get
//!   `exp` of random bivectors in the kernel sweep.
//! * The matrix orientation: a map uploaded as `GpuMat` and applied as `m * x` in the shader
//!   equals the map applied in Rust.
//! * A layout round trip: kinds written by Rust are read field by field by the shader.
//! * A traced kernel (`FUSED_WESL`, linked with the `wesl` crate) equals its Rust twin.
//!
//! Without an adapter the tests pass with a message; with `GAX_REQUIRE_GPU` set they fail.

use gax_gen::emit::{Config, emit};
use gax_gen::emit_wgsl;
use gax_gen::kernel::{Kernel, Precision, Ty};
use gax_gen::spec::AlgebraSpec;
use gax_gpu_tests::{Gpu, floats};
use std::fmt::Write as _;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

const SAMPLES: usize = 64;
/// Kernels per pipeline, to keep each shader a reasonable size.
const CHUNK: usize = 24;

fn spec(name: &str) -> AlgebraSpec {
    let path = format!("{}/../gax/specs/{name}.gax", env!("CARGO_MANIFEST_DIR"));
    AlgebraSpec::parse(&std::fs::read_to_string(path).expect("spec")).expect("parse")
}

fn len(spec: &AlgebraSpec, ty: &Ty) -> usize {
    match ty {
        Ty::Scalar => 1,
        Ty::Kind(k) => spec
            .kinds
            .iter()
            .find(|x| &x.name == k)
            .expect("kind")
            .layout
            .len(),
        Ty::Mat { cols, rows } => cols * rows,
    }
}

/// A compute shader running `kernels` for each sample: inputs packed per sample from `inp`,
/// all outputs packed per sample into `outp`.
fn harness(
    module: &str,
    spec: &AlgebraSpec,
    kernels: &[&Kernel],
    in_stride: usize,
    out_stride: usize,
) -> String {
    harness_in(Precision::F32, module, spec, kernels, in_stride, out_stride)
}

/// [`harness`] for a module in the given precision: inputs and outputs stay `f32` in the
/// buffers, converted at the call.
fn harness_in(
    prec: Precision,
    module: &str,
    spec: &AlgebraSpec,
    kernels: &[&Kernel],
    in_stride: usize,
    out_stride: usize,
) -> String {
    let (to, from) = match prec {
        Precision::F32 => ("", ""),
        Precision::F16 => ("f16", "f32"),
    };
    let mut s = String::from(module);
    let _ = write!(
        s,
        "\n@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let n = id.x;
    if (n >= {SAMPLES}u) {{ return; }}
    let i = n * {in_stride}u;
    let o = n * {out_stride}u;
"
    );
    let (mut ip, mut op) = (0usize, 0usize);
    for k in kernels {
        let mut args = Vec::new();
        for (_, t) in &k.params {
            let n = len(spec, t);
            let e: Vec<String> = (0..n)
                .map(|j| format!("{to}(inp[i + {}u])", ip + j))
                .collect();
            ip += n;
            args.push(match t {
                Ty::Scalar => e[0].clone(),
                Ty::Kind(name) => gax_gen::kernel::wgsl_construct_in(prec, name, &e),
                Ty::Mat { .. } => unreachable!(),
            });
        }
        let _ = writeln!(
            s,
            "    {{\n        let r = {}({});",
            k.name,
            args.join(", ")
        );
        match &k.result {
            Ty::Scalar => {
                let _ = writeln!(s, "        outp[o + {op}u] = {from}(r);");
                op += 1;
            }
            Ty::Kind(name) => {
                for j in 0..len(spec, &Ty::Kind(name.clone())) {
                    let _ = writeln!(
                        s,
                        "        outp[o + {}u] = {from}({});",
                        op + j,
                        gax_gen::kernel::wgsl_coeff("r", j)
                    );
                }
                op += len(spec, &k.result);
            }
            Ty::Mat { cols, rows } => {
                for c in 0..*cols {
                    for r in 0..*rows {
                        let _ = writeln!(
                            s,
                            "        outp[o + {}u] = {from}(r[{c}][{r}]);",
                            op + c * rows + r
                        );
                    }
                }
                op += cols * rows;
            }
        }
        let _ = writeln!(s, "    }}");
    }
    s.push_str("}\n");
    s
}

/// `(kernels, worst ratio to the bound, worst relative error of Study kernels)`.
fn check_algebra(gpu: &Gpu, name: &str, committed: &str) -> (usize, f64, f64) {
    check_algebra_in(gpu, name, committed, Precision::F32)
}

/// [`check_algebra`] for the module in the given precision. For `f16`, the inputs are rounded
/// to `f16` first (so the exact value is of the same inputs), the bounds use `f16`'s unit
/// roundoff `2⁻¹¹` plus an absolute `2⁻¹³` (a GPU may flush `f16` subnormals, below `2⁻¹⁴`, to
/// zero), and the elementary-function kernels must be within a relative `2⁻⁷`.
#[allow(clippy::too_many_lines)]
fn check_algebra_in(gpu: &Gpu, name: &str, committed: &str, prec: Precision) -> (usize, f64, f64) {
    let (unit, slack, rel_tol) = match prec {
        Precision::F32 => (1.0 / f64::from(1u32 << 24), 1e-30, 1.0 / 1024.0),
        Precision::F16 => (1.0 / 2048.0, 1.0 / 8192.0, 1.0 / 128.0),
    };
    let spec = spec(name);
    let (_, stats) = emit(
        &spec,
        &Config {
            core: "crate".into(),
            batch: None,
            check_units: None,
            gpu: None,
        },
    );
    let module = emit_wgsl::module_in(&spec, &stats, true, prec);
    assert_eq!(
        module, committed,
        "{name}: the committed module is what the generator emits"
    );
    let all = emit_wgsl::kernels(&spec, &stats);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let (mut worst, mut worst_rel) = (0.0f64, 0.0f64);
    for chunk in all.chunks(CHUNK) {
        let kernels: Vec<&Kernel> = chunk.iter().collect();
        let in_stride: usize = kernels
            .iter()
            .flat_map(|k| k.params.iter().map(|(_, t)| len(&spec, t)))
            .sum();
        let out_stride: usize = kernels.iter().map(|k| len(&spec, &k.result)).sum();
        let src = harness_in(prec, &module, &spec, &kernels, in_stride, out_stride);
        let mut inputs: Vec<f32> = Vec::with_capacity(SAMPLES * in_stride);
        for _ in 0..SAMPLES {
            for k in &kernels {
                match versor_input(name, &k.name, &mut rng) {
                    Some(v) => inputs.extend(v),
                    None => {
                        for (_, t) in &k.params {
                            inputs.extend((0..len(&spec, t)).map(|_| rng.next()));
                        }
                    }
                }
            }
        }
        if prec == Precision::F16 {
            for x in &mut inputs {
                *x = gax::gpu::f16_to_f32(gax::gpu::f16_bits(*x));
            }
        }
        let out = floats(&gpu.run(
            &src,
            "main",
            &[bytemuck::cast_slice(&inputs)],
            SAMPLES * out_stride * 4,
            SAMPLES as u32,
        ));
        for n in 0..SAMPLES {
            let (mut ip, mut op) = (n * in_stride, n * out_stride);
            for k in &kernels {
                let args: Vec<Vec<f64>> = k
                    .params
                    .iter()
                    .map(|(_, t)| {
                        let l = len(&spec, t);
                        let a = inputs[ip..ip + l].iter().map(|x| f64::from(*x)).collect();
                        ip += l;
                        a
                    })
                    .collect();
                let got: Vec<f32> = match (&k.result, &k.entries) {
                    (Ty::Mat { rows, .. }, Some(entries)) => entries
                        .iter()
                        .map(|&(r, c)| out[op + c * rows + r])
                        .collect(),
                    _ => out[op..op + len(&spec, &k.result)].to_vec(),
                };
                op += len(&spec, &k.result);
                let exact = k.eval(&args);
                if exact.iter().any(|x| !x.is_finite()) {
                    continue;
                }
                match k.error_bound(&args, unit) {
                    Some(bound) => {
                        for (o, (g, e)) in got.iter().zip(&exact).enumerate() {
                            let d = (f64::from(*g) - e).abs();
                            assert!(
                                d <= 2.0 * bound[o] + slack,
                                "{name}: {} output {o}: {g} vs {e}, bound {:e}",
                                k.name,
                                bound[o]
                            );
                            // The share of what is allowed (twice the bound, plus the slack).
                            worst = worst.max(d / (2.0 * bound[o] + slack));
                        }
                    }
                    None => {
                        let scale = exact.iter().fold(1.0f64, |m, x| m.max(x.abs()));
                        for (o, (g, e)) in got.iter().zip(&exact).enumerate() {
                            let rel = (f64::from(*g) - e).abs() / scale;
                            assert!(rel < rel_tol, "{name}: {} output {o}: {g} vs {e}", k.name);
                            worst_rel = worst_rel.max(rel);
                        }
                    }
                }
            }
        }
    }
    (all.len(), worst, worst_rel)
}

#[test]
fn every_kernel_on_the_gpu() {
    let Some(gpu) = Gpu::or_skip() else { return };
    for (name, committed) in [
        ("pga2d", gax::wgsl::PGA2D.source),
        ("pga3d", gax::wgsl::PGA3D.source),
        ("vga3d", gax::wgsl::VGA3D.source),
        ("sta", gax::wgsl::STA.source),
        ("cga2d", gax::wgsl::CGA2D.source),
        ("cga3d", gax::wgsl::CGA3D.source),
        ("stap", gax::wgsl::STAP.source),
        ("csta", gax::wgsl::CSTA.source),
    ] {
        let (n, worst, rel) = check_algebra(&gpu, name, committed);
        println!(
            "{name}: {n} kernels x {SAMPLES} samples, worst {worst:.3} of the allowance, elementary-function kernels within {rel:.1e}"
        );
    }
}

#[test]
fn every_f16_kernel_on_the_gpu() {
    let Some(gpu) = Gpu::or_skip() else { return };
    if !gpu.f16 {
        eprintln!("the adapter has no shader-f16: skipping the f16 modules");
        return;
    }
    for (name, committed) in [
        ("pga2d", gax::wgsl::PGA2D_F16.source),
        ("pga3d", gax::wgsl::PGA3D_F16.source),
        ("vga3d", gax::wgsl::VGA3D_F16.source),
        ("sta", gax::wgsl::STA_F16.source),
        ("cga2d", gax::wgsl::CGA2D_F16.source),
        ("cga3d", gax::wgsl::CGA3D_F16.source),
        ("stap", gax::wgsl::STAP_F16.source),
        ("csta", gax::wgsl::CSTA_F16.source),
    ] {
        let (n, worst, rel) = check_algebra_in(&gpu, name, committed, Precision::F16);
        println!(
            "{name} (f16): {n} kernels x {SAMPLES} samples, worst {worst:.3} of the allowance, elementary-function kernels within {rel:.1e}"
        );
    }
}

#[test]
fn fallback_exponentials_on_the_gpu() {
    let Some(gpu) = Gpu::or_skip() else { return };
    let spec = spec("csta");
    let (_, stats) = emit(
        &spec,
        &Config {
            core: "crate".into(),
            batch: None,
            check_units: None,
            gpu: None,
        },
    );
    let module = gax::wgsl::CSTA.source;
    let mut rng = Rng(0x0dd_ba11);
    let fallbacks = emit_wgsl::fallback_exps(&spec, &stats);
    assert_ne!(fallbacks, []);
    for (k, e) in fallbacks {
        // The harness only needs the call's name and signature.
        let call = Kernel {
            name: format!("{}_exp", gax_gen::kernel::snake(&k.name)),
            doc: String::new(),
            params: vec![("x".into(), Ty::Kind(k.name.clone()))],
            result: Ty::Kind(e.name.clone()),
            steps: Vec::new(),
            entries: None,
        };
        let (n_in, n_out) = (k.layout.len(), e.layout.len());
        let src = harness(module, &spec, &[&call], n_in, n_out);
        let inputs: Vec<f32> = (0..SAMPLES * n_in).map(|_| rng.next()).collect();
        let out = floats(&gpu.run(
            &src,
            "main",
            &[bytemuck::cast_slice(&inputs)],
            SAMPLES * n_out * 4,
            SAMPLES as u32,
        ));
        let mut worst = 0.0f64;
        for n in 0..SAMPLES {
            let b = gax::csta::Bivector::<(), f64>::from_coeffs(core::array::from_fn(|i| {
                f64::from(inputs[n * n_in + i])
            }));
            let exact = b.exp().into_inner().c;
            let scale = exact.iter().fold(1.0f64, |m, x| m.max(x.abs()));
            for (o, e) in exact.iter().enumerate() {
                let g = f64::from(out[n * n_out + o]);
                let rel = (g - e).abs() / scale;
                assert!(
                    rel < 1.0 / 1024.0,
                    "csta: {} output {o}: {g} vs {e}",
                    call.name
                );
                worst = worst.max(rel);
            }
        }
        println!("csta: {} x {SAMPLES} samples within {worst:.1e}", call.name);
    }
}

/// A CSTA unit versor `exp(B)` rounded to `f32`, `B` random with entries up to `size`, redrawn
/// until `min < ⟨R⟩₀ < max`.
fn csta_versor(rng: &mut Rng, size: f64, min: f64, max: f64) -> Vec<f32> {
    loop {
        let b = gax::csta::Bivector::<(), f64>::from_coeffs(core::array::from_fn(|_| {
            size * f64::from(rng.next())
        }));
        let r = b.exp().into_inner().c;
        if min < r[0] && r[0] < max {
            return r.iter().map(|x| *x as f32).collect();
        }
    }
}

/// The argument of a kernel defined on unit versors only (the closed-form 6D logarithm, for
/// `⟨R⟩₀ > 1/16`, and the planes it turns below), or `None` for the others.
fn versor_input(algebra: &str, kernel: &str, rng: &mut Rng) -> Option<Vec<f32>> {
    match (algebra, kernel) {
        ("csta", "unit_even_log_closed") => Some(csta_versor(rng, 0.3, 0.125, f64::INFINITY)),
        ("csta", "unit_even_log_turning") => Some(csta_versor(rng, 1.0, f64::NEG_INFINITY, 0.0625)),
        _ => None,
    }
}

#[test]
fn csta_log_on_the_gpu() {
    use gax::Kind;
    type B = gax::csta::Bivector<(), f64>;
    let Some(gpu) = Gpu::or_skip() else { return };
    let spec = spec("csta");
    let module = gax::wgsl::CSTA.source;
    let call = Kernel {
        name: "unit_even_log".into(),
        doc: String::new(),
        params: vec![("x".into(), Ty::Kind("Even".into()))],
        result: Ty::Kind("Bivector".into()),
        steps: Vec::new(),
        entries: None,
    };
    let (n_in, n_out) = (32, 15);
    let src = harness(module, &spec, &[&call], n_in, n_out);
    let mut rng = Rng(0x0106_5d6d);
    // Every fourth sample a rotation towards a half turn, and every fourth below 1/16: turned.
    let e12 = <gax::csta::Bivector as Kind>::BLADES.iter().position(|n| *n == "e12").expect("e12");
    let mut inputs: Vec<f32> = Vec::with_capacity(SAMPLES * n_in);
    for n in 0..SAMPLES {
        if n % 4 == 3 {
            let mut b = [0.0f64; 15];
            for x in &mut b {
                *x = 0.05 * f64::from(rng.next());
            }
            b[e12] = core::f64::consts::FRAC_PI_2 - 0.01 * (1.0 + f64::from(rng.next()));
            inputs.extend(B::from_coeffs(b).exp().into_inner().c.iter().map(|x| *x as f32));
        } else if n % 4 == 2 {
            // Below 1/16, either sign: turned.
            inputs.extend(csta_versor(&mut rng, 1.0, f64::NEG_INFINITY, 0.0625));
        } else {
            inputs.extend(csta_versor(&mut rng, 0.3, f64::NEG_INFINITY, f64::INFINITY));
        }
    }
    let out = floats(&gpu.run(
        &src,
        "main",
        &[bytemuck::cast_slice(&inputs)],
        SAMPLES * n_out * 4,
        SAMPLES as u32,
    ));
    let mut worst = 0.0f64;
    for n in 0..SAMPLES {
        let r = gax::csta::Even::<(), f64>::from_coeffs(core::array::from_fn(|i| {
            f64::from(inputs[n * n_in + i])
        }));
        let exact: B = gax::Unit::new_unchecked(r).log();
        let scale = exact.c.iter().fold(1.0f64, |m, x| m.max(x.abs()));
        for (o, e) in exact.c.iter().enumerate() {
            let g = f64::from(out[n * n_out + o]);
            let rel = (g - e).abs() / scale;
            assert!(rel < 1.0 / 1024.0, "csta: unit_even_log sample {n} output {o}: {g} vs {e}");
            worst = worst.max(rel);
        }
    }
    println!("csta: unit_even_log x {SAMPLES} samples within {worst:.1e}");
}

#[test]
fn matrices_apply_as_m_times_x() {
    use gax::pga3d::{Line, Point};
    let Some(gpu) = Gpu::or_skip() else { return };
    let mut rng = Rng(7);
    let n = 256;
    let mut mats = Vec::new();
    let mut pts = Vec::new();
    let mut want = Vec::new();
    for _ in 0..n {
        let b = Line::<(), f32>::new(
            rng.next(),
            rng.next(),
            rng.next(),
            rng.next(),
            rng.next(),
            rng.next(),
        );
        let map: Point<(Point,), f32> = b.exp() >> Point::slot();
        let p = Point::<(), f32>::new(rng.next(), rng.next(), rng.next(), 1.0);
        mats.push(gax::GpuMat::<4>::from(map));
        pts.push(gax::pga3d::PointGpu::from(p));
        want.push(map.of(p));
    }
    let src = format!(
        "{}
@group(0) @binding(0) var<storage, read> mats: array<mat4x4<f32>>;
@group(0) @binding(1) var<storage, read> pts: array<Point>;
@group(0) @binding(2) var<storage, read_write> outp: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    if (id.x >= {n}u) {{ return; }}
    outp[id.x] = mats[id.x] * pts[id.x].c0;
}}
",
        gax::wgsl::PGA3D.source
    );
    let out = floats(&gpu.run(
        &src,
        "main",
        &[bytemuck::cast_slice(&mats), bytemuck::cast_slice(&pts)],
        n * 16,
        n as u32,
    ));
    for (k, w) in want.iter().enumerate() {
        for o in 0..4 {
            let g = out[k * 4 + o];
            assert!(
                (g - w.c[o]).abs() < 1e-5 * (1.0 + w.c[o].abs()),
                "point {k} coefficient {o}: {g} vs {}",
                w.c[o]
            );
        }
    }
}

#[test]
fn layouts_round_trip() {
    use gax::pga3d::{Line, LineGpu, Motor, MotorGpu};
    let Some(gpu) = Gpu::or_skip() else { return };
    let mut rng = Rng(11);
    let n = 128;
    let lines: Vec<Line<(), f32>> = (0..n)
        .map(|_| Line::from_coeffs(core::array::from_fn(|_| rng.next())))
        .collect();
    let motors: Vec<Motor<(), f32>> = (0..n)
        .map(|_| Motor::from_coeffs(core::array::from_fn(|_| rng.next())))
        .collect();
    let lg: Vec<LineGpu> = lines.iter().map(|l| (*l).into()).collect();
    let mg: Vec<MotorGpu> = motors.iter().map(|m| (*m).into()).collect();
    let mut body = String::new();
    for j in 0..6 {
        let _ = writeln!(
            body,
            "    outp[id.x * 14u + {j}u] = {};",
            gax_gen::kernel::wgsl_coeff("lines[id.x]", j)
        );
    }
    for j in 0..8 {
        let _ = writeln!(
            body,
            "    outp[id.x * 14u + {}u] = {};",
            6 + j,
            gax_gen::kernel::wgsl_coeff("motors[id.x]", j)
        );
    }
    let src = format!(
        "{}
@group(0) @binding(0) var<storage, read> lines: array<Line>;
@group(0) @binding(1) var<storage, read> motors: array<Motor>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    if (id.x >= {n}u) {{ return; }}
{body}}}
",
        gax::wgsl::PGA3D.source
    );
    let out = floats(&gpu.run(
        &src,
        "main",
        &[bytemuck::cast_slice(&lg), bytemuck::cast_slice(&mg)],
        n * 14 * 4,
        n as u32,
    ));
    for k in 0..n {
        assert_eq!(&out[k * 14..k * 14 + 6], &lines[k].c[..], "line {k}");
        assert_eq!(&out[k * 14 + 6..k * 14 + 14], &motors[k].c[..], "motor {k}");
    }
}

#[test]
fn traced_kernels_match_their_rust_twins() {
    use gax::Unit;
    use gax::pga3d::{Line, Motor, MotorGpu, Point, PointGpu};
    use wesl::resolver::VirtualResolver;
    let Some(gpu) = Gpu::or_skip() else { return };
    let main = "
import package::fused::compose_apply_fused;
import gax::pga3d::{Motor, Point};
@group(0) @binding(0) var<storage, read> ms: array<Motor>;
@group(0) @binding(1) var<storage, read> ps: array<Point>;
@group(0) @binding(2) var<storage, read_write> outp: array<Point>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&ps)) { return; }
    outp[id.x] = compose_apply_fused(ms[2u * id.x], ms[2u * id.x + 1u], ps[id.x]);
}
";
    let mut r = VirtualResolver::new();
    for m in gax::wgsl::ALL {
        r.add_module(m.path.parse().expect("path"), m.wesl_public().into());
    }
    r.add_module(
        "package::fused".parse().expect("path"),
        gax_example_traced::FUSED_WESL.into(),
    );
    r.add_module("package::main".parse().expect("path"), main.into());
    let options = wesl::CompileOptions {
        strip: true,
        ..Default::default()
    };
    let wgsl = wesl::compile(&"package::main".parse().expect("path"), &options, &r)
        .unwrap_or_else(|e| panic!("{e}"))
        .syntax
        .to_string();
    let mut rng = Rng(3);
    let n = 200;
    let mut ms = Vec::new();
    let mut ps = Vec::new();
    let mut want = Vec::new();
    for _ in 0..n {
        let mut line = || Line::<(), f32>::from_coeffs(core::array::from_fn(|_| rng.next())).exp();
        let (a, b): (Unit<Motor<(), f32>>, _) = (line(), line());
        let p = Point::<(), f32>::new(rng.next(), rng.next(), rng.next(), 1.0);
        ms.push(MotorGpu::from(a));
        ms.push(MotorGpu::from(b));
        ps.push(PointGpu::from(p));
        want.push(gax_example_traced::compose_apply_fused::<f32>(a, b, p));
    }
    let out = floats(&gpu.run(
        &wgsl,
        "main",
        &[bytemuck::cast_slice(&ms), bytemuck::cast_slice(&ps)],
        n * 16,
        n as u32,
    ));
    for (k, w) in want.iter().enumerate() {
        for o in 0..4 {
            let g = out[k * 4 + o];
            assert!(
                (g - w.c[o]).abs() < 1e-5 * (1.0 + w.c[o].abs()),
                "sample {k} coefficient {o}: {g} vs {}",
                w.c[o]
            );
        }
    }
}
