//! The WGSL modules of a 7D algebra declared with `algebra!` (its even kind has 64 coefficients,
//! the most the modules take): they validate, and `wesl`'s evaluator runs the closed-form
//! logarithm `unit_even_log` and exponential `bivector_exp` (turning planes near a half turn
//! first) and the plain sandwich kernels in `f32`, against the Rust `f64` results.

#![cfg(feature = "wgsl")]

mod common;

gax::algebra! {
    algebra r43 "Split signature R(4,3): rotations, boosts and loxodromic planes.";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = -1, e6 = -1, e7 = -1;
    kind Scalar = [1];
    versor Vector = [e1, e2, e3, e4, e5, e6, e7];
    kind Bivector = [e12, e13, e23, e14, e24, e34, e15, e25, e35, e45, e16, e26, e36, e46, e56, e17, e27, e37, e47, e57, e67];
    versor Even = [1, e12, e13, e23, e14, e24, e34, e15, e25, e35, e45, e16, e26, e36, e46, e56, e17, e27, e37, e47, e57, e67, e1234, e1235, e1245, e1345, e2345, e1236, e1246, e1346, e2346, e1256, e1356, e2356, e1456, e2456, e3456, e1237, e1247, e1347, e2347, e1257, e1357, e2357, e1457, e2457, e3457, e1267, e1367, e2367, e1467, e2467, e3467, e1567, e2567, e3567, e4567, e123456, e123457, e123467, e123567, e124567, e134567, e234567];
}

use common::Rng;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use r43::{Bivector, Even, Vector};
use wesl::CompileOptions;
use wesl::resolver::VirtualResolver;

fn validate(what: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("{what}: {}", e.emit_to_string(src)));
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{what}: {}", e.emit_to_string(src)));
}

/// The module in evaluable form: every function `@const`, `fma` spelled out (the evaluator
/// implements neither `fma` nor calls of other functions).
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

/// A WGSL value of the kind `name` with coefficients `c`.
fn literal(name: &str, c: &[f64]) -> String {
    let fields: Vec<String> = c
        .chunks(4)
        .map(|ch| {
            let mut v: Vec<String> = ch.iter().map(|x| format!("{:?}f", *x as f32)).collect();
            v.resize(4, "0.0f".into());
            format!("vec4<f32>({})", v.join(", "))
        })
        .collect();
    format!("{name}({})", fields.join(", "))
}

fn evaluate(res: &wesl::CompileResult, expr: &str) -> Vec<f64> {
    let mut v = res.eval(expr).unwrap_or_else(|e| panic!("{expr}: {e}"));
    v.to_buffer()
        .expect("bytes")
        .chunks(4)
        .map(|b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
        .collect()
}

#[test]
fn the_modules_validate() {
    assert_eq!(r43::WGSL_MODULE.path, "package::r43");
    validate("r43", r43::WGSL_MODULE.source);
    validate("r43_f16", r43::WGSL_MODULE_F16.source);
    for f in [
        "fn unit_even_log(",
        "fn bivector_exp(",
        "fn unit_even_sandwich_vector(",
    ] {
        assert!(r43::WGSL_MODULE.source.contains(f), "{f}");
    }
}

#[test]
fn log_exp_and_sandwiches_evaluate() {
    let mut r = VirtualResolver::new();
    r.add_module(
        "package::main".parse().expect("path"),
        evaluable(r43::WGSL_MODULE.source).into(),
    );
    let options = CompileOptions {
        keep_main: true,
        ..CompileOptions::default()
    };
    let res = wesl::compile(&"package::main".parse().expect("path"), &options, &r)
        .unwrap_or_else(|e| panic!("{e}"));
    let mut rng = Rng(0x7d_36_51);
    let (mut worst, mut worst_exp, mut turned) = (0.0f64, 0.0f64, 0);
    let half = core::f64::consts::FRAC_PI_2;
    for case in 0..16 {
        // Random versors, and every fourth near or past a half turn in the rotation e12.
        let mut b = Bivector::<(), f64>::from_coeffs(core::array::from_fn(|_| 0.4 * rng.next()));
        if case % 4 == 3 {
            b.c[0] = half + 0.3 * rng.next();
        }
        let v = b.exp();
        let x = v.into_inner();
        turned += usize::from(x.c[0] < 0.0625);
        // The f32 versor for the shader; the Rust log of the f64 one (a unit, also under
        // `check-units`; the rounding is far below the tolerance).
        let x32 = Even::<(), f64>::from_coeffs(core::array::from_fn(|i| f64::from(x.c[i] as f32)));
        let want: Bivector<(), f64> = v.log();
        let got = evaluate(&res, &format!("unit_even_log({})", literal("Even", &x32.c)));
        let scale = want.c.iter().fold(1.0f64, |m, c| m.max(c.abs()));
        for (g, w) in got.iter().zip(&want.c) {
            let e = (g - w).abs() / scale;
            assert!(e < 1.0 / 4096.0, "case {case}: {got:?} vs {want:?}");
            worst = worst.max(e);
        }
        // The closed-form exponential, against the Rust one of the same f32 bivector.
        let b32 =
            Bivector::<(), f64>::from_coeffs(core::array::from_fn(|i| f64::from(b.c[i] as f32)));
        let want = b32.exp().into_inner();
        let got = evaluate(&res, &format!("bivector_exp({})", literal("Bivector", &b32.c)));
        let scale = want.c.iter().fold(1.0f64, |m, c| m.max(c.abs()));
        for (g, w) in got.iter().zip(&want.c) {
            let e = (g - w).abs() / scale;
            assert!(e < 1.0 / 4096.0, "exp, case {case}: {got:?} vs {want:?}");
            worst_exp = worst_exp.max(e);
        }
        // The plain sandwich kernels (a versor over 32 coefficients).
        let p = Vector::<(), f64>::from_coeffs(core::array::from_fn(|_| rng.next()));
        let want = v >> p;
        let got = evaluate(
            &res,
            &format!(
                "unit_even_sandwich_vector({}, {})",
                literal("Even", &x32.c),
                literal("Vector", &p.c)
            ),
        );
        for (g, w) in got.iter().zip(&want.c) {
            assert!((g - w).abs() < 1e-5, "sandwich: {got:?} vs {want:?}");
        }
    }
    assert!(turned > 0, "no case turned planes");
    println!(
        "7D unit_even_log and bivector_exp in f32: within {worst:.1e} and {worst_exp:.1e} of f64"
    );
}
