//! Every kernel of the WGSL modules, executed on the CPU by `wesl`'s evaluator (docs/shaders.md,
//! layer 4), against the same kernel evaluated exactly (in `f64`) from its program.
//!
//! * Kernels of one straight-line program (products, sandwiches, matrices, norms, Newton
//!   steps) must be within the program's computed `f32` forward error bound.
//! * Kernels with a Study step (`exp`, `log`, `normalized`) call elementary functions; they
//!   must agree within a relative `2⁻¹⁴` (the evaluator uses Rust's `f32` functions; a GPU's
//!   WGSL built-ins are looser, see the GPU tests).
//!
//! * The closed-form 6D logarithm is defined on unit versors only; it gets `exp` of random
//!   bivectors, and the module's `unit_even_log` (the closed form or, near a half turn, inverse
//!   scaling and squaring) is checked against gax's Rust `log`.
//!
//! The evaluator implements neither `fma` nor calls of non-`@const` functions, so the module
//! is evaluated with every function marked `@const` and `fma(a, b, c)` as `a * b + c`. That
//! changes rounding only, which the bounds cover; it does not test that a GPU fuses.

use gax_gen::emit::{Config, emit};
use gax_gen::emit_wgsl;
use gax_gen::kernel::{Kernel, Ty};
use gax_gen::spec::AlgebraSpec;
use wesl::CompileOptions;
use wesl::resolver::VirtualResolver;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn spec(name: &str) -> AlgebraSpec {
    let path = format!("{}/../gax/specs/{name}.gax", env!("CARGO_MANIFEST_DIR"));
    AlgebraSpec::parse(&std::fs::read_to_string(path).expect("spec")).expect("parse")
}

/// The module in evaluable form: every function `@const`, `fma` spelled out.
fn evaluable(src: &str) -> String {
    let mut out =
        String::from("@const fn fma_(a: f32, b: f32, c: f32) -> f32 { return a * b + c; }\n");
    for line in src.lines() {
        if line.starts_with("fn ") {
            out.push_str("@const ");
        }
        out.push_str(&line.replace("fma(", "fma_("));
        out.push('\n');
    }
    out
}

/// A WGSL expression for a value of the parameter type `ty` with coefficients `c`.
fn literal(ty: &Ty, c: &[f32]) -> String {
    let f = |x: f32| format!("{x:?}f");
    match ty {
        Ty::Scalar => f(c[0]),
        Ty::Kind(k) => {
            let fields: Vec<String> = c
                .chunks(4)
                .map(|ch| {
                    let mut v: Vec<String> = ch.iter().map(|x| f(*x)).collect();
                    v.resize(4, "0.0f".into());
                    format!("vec4<f32>({})", v.join(", "))
                })
                .collect();
            format!("{k}({})", fields.join(", "))
        }
        Ty::Mat { .. } => unreachable!("no kernel takes a matrix"),
    }
}

fn len(spec: &AlgebraSpec, ty: &Ty) -> usize {
    match ty {
        Ty::Scalar => 1,
        Ty::Kind(k) => spec
            .kinds
            .iter()
            .find(|x| &x.name == k)
            .expect("a kind")
            .layout
            .len(),
        Ty::Mat { .. } => unreachable!(),
    }
}

/// The kernel's outputs, in program order, from the evaluator's bytes.
fn outputs(k: &Kernel, bytes: &[u8]) -> Vec<f32> {
    let f: Vec<f32> = bytes
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    match (&k.result, &k.entries) {
        (Ty::Mat { .. }, Some(entries)) => entries.iter().map(|&(r, c)| f[c * 4 + r]).collect(),
        _ => f,
    }
}

/// `(kernels checked, worst ratio to the bound, worst relative error of Study kernels)`.
#[allow(clippy::too_many_lines)]
fn check(name: &str, committed: &str) -> (usize, f64, f64) {
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
    let module = emit_wgsl::module(&spec, &stats, true);
    assert_eq!(
        module, committed,
        "{name}: the committed module is what the generator emits"
    );
    let mut r = VirtualResolver::new();
    r.add_module(
        "package::main".parse().expect("path"),
        evaluable(&module).into(),
    );
    let options = CompileOptions {
        keep_main: true,
        ..CompileOptions::default()
    };
    let res = wesl::compile(&"package::main".parse().expect("path"), &options, &r)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let (mut count, mut worst, mut worst_rel) = (0, 0.0f64, 0.0f64);
    // `GAX_EVAL_KERNEL=csta::twist_exp GAX_EVAL_SAMPLES=1000`: one kernel, many inputs.
    let only = std::env::var("GAX_EVAL_KERNEL").ok();
    let samples: usize = std::env::var("GAX_EVAL_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    for k in emit_wgsl::kernels(&spec, &stats) {
        if only
            .as_ref()
            .is_some_and(|o| *o != format!("{name}::{}", k.name))
        {
            continue;
        }
        for _ in 0..samples {
            let args: Vec<Vec<f32>> = match versor_input(name, &k.name, &mut rng) {
                Some(v) => vec![v],
                None => k
                    .params
                    .iter()
                    .map(|(_, t)| (0..len(&spec, t)).map(|_| rng.next()).collect())
                    .collect(),
            };
            let wide: Vec<Vec<f64>> = args
                .iter()
                .map(|a| a.iter().map(|x| f64::from(*x)).collect())
                .collect();
            let exact = k.eval(&wide);
            if exact.iter().any(|x| !x.is_finite()) {
                continue; // outside the kernel's domain (a negative norm under a root)
            }
            let call: Vec<String> = k
                .params
                .iter()
                .zip(&args)
                .map(|((_, t), a)| literal(t, a))
                .collect();
            let expr = format!("{}({})", k.name, call.join(", "));
            let mut v = res
                .eval(&expr)
                .unwrap_or_else(|e| panic!("{name}: {expr}: {e}"));
            let got = outputs(&k, &v.to_buffer().expect("bytes"));
            if let Some(bound) = k.error_bound(&wide, 1.0 / f64::from(1u32 << 24)) {
                for (o, (g, e)) in got.iter().zip(&exact).enumerate() {
                    let d = (f64::from(*g) - e).abs();
                    assert!(
                        d <= bound[o],
                        "{name}: {} output {o}: {g} vs {e}, bound {:e}",
                        k.name,
                        bound[o]
                    );
                    if bound[o] > 0.0 {
                        worst = worst.max(d / bound[o]);
                    }
                }
            } else {
                let scale = exact.iter().fold(1.0f64, |m, x| m.max(x.abs()));
                for (o, (g, e)) in got.iter().zip(&exact).enumerate() {
                    let rel = (f64::from(*g) - e).abs() / scale;
                    assert!(
                        rel < 1.0 / 16384.0,
                        "{name}: {} output {o}: {g} vs {e} at {args:?}",
                        k.name
                    );
                    worst_rel = worst_rel.max(rel);
                }
            }
        }
        count += 1;
    }
    // The scaling-and-squaring exponentials (loops, not straight-line kernels), against gax's
    // Rust `exp` for the same kind.
    for (kind, _) in emit_wgsl::fallback_exps(&spec, &stats) {
        let name_fn = format!("{}_exp", gax_gen::kernel::snake(&kind.name));
        let n = kind.layout.len();
        for _ in 0..samples.max(8) {
            let c: Vec<f32> = (0..n).map(|_| rng.next()).collect();
            let exact = reference_exp(name, &kind.name, &c);
            let expr = format!("{name_fn}({})", literal(&Ty::Kind(kind.name.clone()), &c));
            let mut v = res
                .eval(&expr)
                .unwrap_or_else(|e| panic!("{name}: {expr}: {e}"));
            let bytes = v.to_buffer().expect("bytes");
            let got: Vec<f32> = bytes
                .chunks(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let scale = exact.iter().fold(1.0f64, |m, x| m.max(x.abs()));
            for (o, (g, e)) in got.iter().zip(&exact).enumerate() {
                let rel = (f64::from(*g) - e).abs() / scale;
                // Scaling and squaring doubles the relative error with each of its up to 8
                // squarings, so its bound is looser than a closed form's: 2⁻¹².
                assert!(
                    rel < 1.0 / 4096.0,
                    "{name}: {name_fn} output {o}: {g} vs {e} at {c:?}"
                );
                worst_rel = worst_rel.max(rel);
            }
        }
        count += 1;
    }
    // The logarithms with a fallback near a half turn, against gax's Rust `log`.
    for (_, e) in emit_wgsl::fallback_logs(&spec, &stats) {
        let name_fn = format!("unit_{}_log", gax_gen::kernel::snake(&e.name));
        for case in 0..samples.max(8) {
            let (c, exact) = reference_log(name, &e.name, &mut rng, case);
            let expr = format!("{name_fn}({})", literal(&Ty::Kind(e.name.clone()), &c));
            let mut v = res
                .eval(&expr)
                .unwrap_or_else(|e| panic!("{name}: {expr}: {e}"));
            let bytes = v.to_buffer().expect("bytes");
            let got: Vec<f32> = bytes
                .chunks(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let scale = exact.iter().fold(1.0f64, |m, x| m.max(x.abs()));
            for (o, (g, e)) in got.iter().zip(&exact).enumerate() {
                let rel = (f64::from(*g) - e).abs() / scale;
                assert!(
                    rel < 1.0 / 4096.0,
                    "{name}: {name_fn} output {o}: {g} vs {e} at {c:?}"
                );
                worst_rel = worst_rel.max(rel);
            }
        }
        count += 1;
    }
    (count, worst, worst_rel)
}

/// A CSTA unit versor `exp(B)` rounded to `f32`, `B` random with entries up to `size`, redrawn
/// until `⟨R⟩₀ > min` (clear of a half turn).
fn csta_versor(rng: &mut Rng, size: f64, min: f64) -> Vec<f32> {
    loop {
        let b = gax::csta::Bivector::<(), f64>::from_coeffs(core::array::from_fn(|_| {
            size * f64::from(rng.next())
        }));
        let r = b.exp().into_inner().c;
        if r[0] > min {
            return r.iter().map(|x| *x as f32).collect();
        }
    }
}

/// The argument of a kernel defined on unit versors only (the closed-form 6D logarithm, for
/// `⟨R⟩₀ > 1/16`), or `None` for the others.
fn versor_input(algebra: &str, kernel: &str, rng: &mut Rng) -> Option<Vec<f32>> {
    match (algebra, kernel) {
        ("csta", "unit_even_log_closed") => Some(csta_versor(rng, 0.3, 0.125)),
        _ => None,
    }
}

/// A unit versor (every fourth one a rotation towards a half turn, where the fallback takes
/// over) and gax's Rust `log` of it.
fn reference_log(algebra: &str, kind: &str, rng: &mut Rng, case: usize) -> (Vec<f32>, Vec<f64>) {
    use gax::Kind;
    match (algebra, kind) {
        ("csta", "Even") => {
            type B = gax::csta::Bivector<(), f64>;
            let c = if case % 4 == 3 {
                let e12 = <gax::csta::Bivector as Kind>::BLADES
                    .iter()
                    .position(|n| *n == "e12")
                    .expect("e12");
                let mut b = [0.0f64; 15];
                for x in &mut b {
                    *x = 0.05 * f64::from(rng.next());
                }
                b[e12] = core::f64::consts::FRAC_PI_2 - 0.01 * (1.0 + f64::from(rng.next()));
                let r = B::from_coeffs(b).exp().into_inner().c;
                r.iter().map(|x| *x as f32).collect()
            } else {
                csta_versor(rng, 0.3, -1.0)
            };
            let r =
                gax::csta::Even::<(), f64>::from_coeffs(core::array::from_fn(|i| f64::from(c[i])));
            let l: B = gax::Unit::new_unchecked(r).log();
            (c, l.c.to_vec())
        }
        _ => panic!("no reference log for {algebra}::{kind}"),
    }
}

/// gax's Rust `exp` of a kind whose WGSL `exp` is the scaling-and-squaring fallback.
fn reference_exp(algebra: &str, kind: &str, c: &[f32]) -> Vec<f64> {
    match (algebra, kind) {
        ("csta", "Bivector") => {
            let b = gax::csta::Bivector::<(), f64>::from_coeffs(core::array::from_fn(|i| {
                f64::from(c[i])
            }));
            b.exp().into_inner().c.to_vec()
        }
        _ => panic!("no reference exp for {algebra}::{kind}"),
    }
}

#[test]
fn every_kernel_evaluates_within_its_bound() {
    for (name, committed) in [
        ("pga2d", gax::wgsl::PGA2D.source),
        ("pga3d", gax::wgsl::PGA3D.source),
        ("vga2d", gax::wgsl::VGA2D.source),
        ("vga3d", gax::wgsl::VGA3D.source),
        ("sta", gax::wgsl::STA.source),
        ("cga2d", gax::wgsl::CGA2D.source),
        ("cga3d", gax::wgsl::CGA3D.source),
        ("stap", gax::wgsl::STAP.source),
        ("csta", gax::wgsl::CSTA.source),
    ] {
        let (n, worst, rel) = check(name, committed);
        println!(
            "{name}: {n} kernels, worst {worst:.3} of the bound, Study kernels within {rel:.1e}"
        );
        assert!(n > 20 || std::env::var_os("GAX_EVAL_KERNEL").is_some());
    }
}
