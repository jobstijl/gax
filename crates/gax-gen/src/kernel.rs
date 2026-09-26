//! Kernels in a language-neutral form, for printers other than the Rust emitter.
//!
//! The Rust emitter writes its methods directly. For every kernel that another target (WGSL)
//! also needs, it records the same verified [`Program`]s here, together with how their
//! variables map to the kernel's parameters and which Study-number helper runs between them.
//! A printer then only renders, so the correctness of the programs carries over (ADR-028).

use crate::poly::Var;
use crate::slp::{Program, Target, render_to};
use std::fmt::Write as _;

/// A kernel parameter or result type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ty {
    /// A scalar, `f32` in WGSL.
    Scalar,
    /// A value of the named kind.
    Kind(String),
    /// A column-major matrix with `cols` columns (the input dimension) and `rows` rows.
    Mat {
        /// Columns: the input kind's dimension.
        cols: usize,
        /// Rows: the output kind's dimension.
        rows: usize,
    },
}

/// Where a program variable comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// Coefficient `index` of parameter `param`.
    Arg {
        /// The parameter.
        param: usize,
        /// The coefficient.
        index: usize,
    },
    /// A local produced by an earlier step, such as `c0` from a Study helper.
    Local(String),
}

/// A Study-number helper called between two programs (see `gax_core::study`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StudyFn {
    /// `[c0, c1, s0, s1] = exp_coeffs_rotation(lambda, mu)`.
    ExpRotation,
    /// `[h0, h1] = log_coeffs_rotation((c0, c1), (u0, u1))`.
    LogRotation,
    /// `[s0, s1] = rsqrt_nil(a, b)`: `(a + bI)^(-1/2)` with `I² = 0`.
    RsqrtNil,
    /// `[s0, s1] = rsqrt(isq, a, b)`: `(a + bI)^(-1/2)` with `I² = isq`, `isq = ±1`.
    Rsqrt(i8),
    /// `s0 = 1 / sqrt(|a|)`.
    RsqrtAbs,
}

impl StudyFn {
    /// The WGSL helper's name (defined in [`crate::emit_wgsl::study_source`]).
    pub fn wgsl_name(self) -> &'static str {
        match self {
            StudyFn::ExpRotation => "study_exp_rotation",
            StudyFn::LogRotation => "study_log_rotation",
            StudyFn::RsqrtNil => "study_rsqrt_nil",
            StudyFn::Rsqrt(1) => "study_rsqrt_split",
            StudyFn::Rsqrt(_) => "study_rsqrt_complex",
            StudyFn::RsqrtAbs => "study_rsqrt_abs",
        }
    }
}

/// One step of a kernel body.
#[derive(Clone, Debug)]
pub enum Step {
    /// Straight-line code; its temporaries are named `{prefix}{k}`.
    Lets {
        /// The program.
        prog: Program,
        /// Prefix of its temporaries.
        prefix: String,
        /// The source of each variable it reads.
        vars: Vec<(Var, Source)>,
    },
    /// `outs = func(args)`, where each argument is output `k` of the `Lets` step `step`.
    Study {
        /// The helper.
        func: StudyFn,
        /// `(step, output)` per argument.
        args: Vec<(usize, usize)>,
        /// Names of the results, readable by later steps as [`Source::Local`].
        outs: Vec<String>,
    },
}

/// A kernel: parameters, a body of steps, and a result made of the outputs of the last
/// `Lets` step.
#[derive(Clone, Debug)]
pub struct Kernel {
    /// The function name (see ADR-028 for the scheme).
    pub name: String,
    /// One line of documentation.
    pub doc: String,
    /// `(name, type)` per parameter.
    pub params: Vec<(String, Ty)>,
    /// The result type.
    pub result: Ty,
    /// The body.
    pub steps: Vec<Step>,
    /// For a matrix result, the `(row, column)` of each output of the last program; missing
    /// entries are zero.
    pub entries: Option<Vec<(usize, usize)>>,
}

/// `lowerCamel`/`UpperCamel` to `snake_case`: `Multivector` → `multivector`,
/// `PointPair` → `point_pair`.
pub fn snake(name: &str) -> String {
    let mut s = String::new();
    for (i, ch) in name.chars().enumerate() {
        if ch.is_ascii_uppercase() {
            if i > 0 {
                s.push('_');
            }
            s.push(ch.to_ascii_lowercase());
        } else {
            s.push(ch);
        }
    }
    s
}

/// Number of `vec4<f32>` fields of a kind with `n` coefficients.
pub fn vec4s(n: usize) -> usize {
    n.div_ceil(4)
}

/// WGSL expression for coefficient `i` of the kind value `name`: `name.c0.x`.
pub fn wgsl_coeff(name: &str, i: usize) -> String {
    format!("{name}.c{}.{}", i / 4, ["x", "y", "z", "w"][i % 4])
}

/// WGSL expression constructing a kind value from its coefficient expressions (zero-padded).
pub fn wgsl_construct(kind: &str, parts: &[String]) -> String {
    let fields: Vec<String> = parts
        .chunks(4)
        .map(|c| {
            let mut v: Vec<String> = c.to_vec();
            v.resize(4, "0.0".into());
            format!("vec4<f32>({})", v.join(", "))
        })
        .collect();
    format!("{kind}({})", fields.join(", "))
}

impl Ty {
    fn wgsl(&self) -> String {
        match self {
            Ty::Scalar => "f32".into(),
            Ty::Kind(k) => k.clone(),
            Ty::Mat { cols, rows } => format!("mat{cols}x{rows}<f32>"),
        }
    }
}

impl Kernel {
    /// The kernel as a WGSL function. `vis` is written before `fn` (for example `"public "`,
    /// or empty for plain WGSL).
    ///
    /// # Panics
    /// If the steps do not end with a `Lets` step, or a matrix result has no entries.
    pub fn wgsl(&self, fma: bool, vis: &str) -> String {
        let target = Target::Wgsl { fma };
        let mut s = String::new();
        let _ = writeln!(s, "// {}", self.doc);
        let params: Vec<String> = self
            .params
            .iter()
            .map(|(n, t)| format!("{n}: {}", t.wgsl()))
            .collect();
        let _ = writeln!(
            s,
            "{vis}fn {}({}) -> {} {{",
            self.name,
            params.join(", "),
            self.result.wgsl()
        );
        let var_of = |vars: &[(Var, Source)], v: Var| -> String {
            let (_, src) = vars
                .iter()
                .find(|(w, _)| *w == v)
                .unwrap_or_else(|| panic!("{}: variable {v} has no source", self.name));
            match src {
                Source::Arg { param, index } => wgsl_coeff(&self.params[*param].0, *index),
                Source::Local(n) => n.clone(),
            }
        };
        for (k, step) in self.steps.iter().enumerate() {
            match step {
                Step::Lets { prog, prefix, vars } => {
                    // Rendered even when last: its outputs may read the temporaries.
                    let _ = k;
                    prog.emit_lets_to(target, &|v| var_of(vars, v), prefix, &mut s);
                }
                Step::Study { func, args, outs } => {
                    let a: Vec<String> = args
                        .iter()
                        .map(|(st, o)| self.output(*st, *o, target, &var_of))
                        .collect();
                    if outs.len() == 1 {
                        let _ = writeln!(
                            s,
                            "    let {} = {}({});",
                            outs[0],
                            func.wgsl_name(),
                            a.join(", ")
                        );
                    } else {
                        let _ =
                            writeln!(s, "    let r{k} = {}({});", func.wgsl_name(), a.join(", "));
                        for (j, o) in outs.iter().enumerate() {
                            let _ = writeln!(s, "    let {o} = r{k}[{j}];");
                        }
                    }
                }
            }
        }
        let last = self.steps.len() - 1;
        let Step::Lets { prog, .. } = &self.steps[last] else {
            panic!("{}: a kernel ends with straight-line code", self.name);
        };
        let outs: Vec<String> = (0..prog.outputs.len())
            .map(|o| self.output(last, o, target, &var_of))
            .collect();
        let value = match (&self.result, &self.entries) {
            (Ty::Scalar, _) => outs[0].clone(),
            (Ty::Kind(k), _) => wgsl_construct(k, &outs),
            (Ty::Mat { cols, rows }, Some(entries)) => {
                let mut m = vec![vec!["0.0".to_string(); *rows]; *cols];
                for (e, &(r, c)) in outs.iter().zip(entries) {
                    m[c][r].clone_from(e);
                }
                let cols: Vec<String> = m
                    .iter()
                    .map(|col| format!("vec{rows}<f32>({})", col.join(", ")))
                    .collect();
                format!("{}({})", self.result.wgsl(), cols.join(", "))
            }
            (Ty::Mat { .. }, None) => panic!("{}: a matrix result needs its entries", self.name),
        };
        let _ = writeln!(s, "    return {value};\n}}");
        s
    }

    /// Evaluate the kernel in `f64`: `args[p]` holds parameter `p`'s coefficients. Returns the
    /// outputs of the last program (for a matrix, in the order of [`Kernel::entries`]).
    ///
    /// # Panics
    /// If a program reads a variable without a source.
    pub fn eval(&self, args: &[Vec<f64>]) -> Vec<f64> {
        use gax_core::study;
        let mut locals: Vec<(String, f64)> = Vec::new();
        let mut outs: Vec<Vec<f64>> = Vec::new();
        for step in &self.steps {
            match step {
                Step::Lets { prog, vars, .. } => {
                    let get = |v: Var| {
                        let (_, src) = vars.iter().find(|(w, _)| *w == v).expect("a source");
                        match src {
                            Source::Arg { param, index } => args[*param][*index],
                            Source::Local(n) => {
                                locals.iter().find(|(m, _)| m == n).expect("a local").1
                            }
                        }
                    };
                    outs.push(prog.eval(&get));
                }
                Step::Study {
                    func,
                    args: a,
                    outs: names,
                } => {
                    let x: Vec<f64> = a.iter().map(|(st, o)| outs[*st][*o]).collect();
                    let r: Vec<f64> = match func {
                        StudyFn::ExpRotation => study::exp_coeffs_rotation(x[0], x[1]).to_vec(),
                        StudyFn::LogRotation => {
                            study::log_coeffs_rotation((x[0], x[1]), (x[2], x[3])).to_vec()
                        }
                        StudyFn::RsqrtNil => study::rsqrt_nil(x[0], x[1]).to_vec(),
                        StudyFn::Rsqrt(isq) => study::rsqrt(*isq, x[0], x[1]).to_vec(),
                        StudyFn::RsqrtAbs => vec![1.0 / x[0].abs().sqrt()],
                    };
                    for (n, v) in names.iter().zip(r) {
                        locals.push((n.clone(), v));
                    }
                    outs.push(Vec::new());
                }
            }
        }
        outs.pop().expect("a kernel ends with straight-line code")
    }

    /// For a kernel of one straight-line program: the a-priori forward error bound of each
    /// output in a precision with unit roundoff `u` (see [`Program::error_bound`]); `None` for
    /// kernels with a Study step, whose elementary functions have no such bound.
    pub fn error_bound(&self, args: &[Vec<f64>], u: f64) -> Option<Vec<f64>> {
        let [Step::Lets { prog, vars, .. }] = self.steps.as_slice() else {
            return None;
        };
        let mag = |v: Var| {
            let (_, src) = vars.iter().find(|(w, _)| *w == v).expect("a source");
            match src {
                Source::Arg { param, index } => args[*param][*index].abs(),
                Source::Local(_) => f64::INFINITY,
            }
        };
        Some(prog.error_bound(&mag, u))
    }

    fn output(
        &self,
        step: usize,
        o: usize,
        target: Target,
        var_of: &impl Fn(&[(Var, Source)], Var) -> String,
    ) -> String {
        let Step::Lets { prog, prefix, vars } = &self.steps[step] else {
            panic!("{}: step {step} is not straight-line code", self.name);
        };
        render_to(target, &prog.outputs[o], &|v| var_of(vars, v), prefix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::Rational;
    use crate::slp::{Instr, Operand};

    #[test]
    fn a_kernel_in_wgsl() {
        // f(p) = Point(2 p0, p0 * p1 + p2, 1/3, ·) with a study step in between.
        let mut pre = Program::default();
        let t = pre.push(Instr::Mul(Operand::Var(0), Operand::Var(0)));
        pre.outputs = vec![t, Operand::Var(1)];
        let mut post = Program::default();
        let a = post.push(Instr::Mul(
            Operand::Var(10),
            Operand::Const(Rational::new(2, 1)),
        ));
        let b = post.push(Instr::Mul(Operand::Var(0), Operand::Var(11)));
        let c = post.push(Instr::Add(b, Operand::Var(2)));
        post.outputs = vec![a, c, Operand::Const(Rational::new(1, 3))];
        let arg = |i| Source::Arg { param: 0, index: i };
        let k = Kernel {
            name: "point_f".into(),
            doc: "A test kernel.".into(),
            params: vec![("p".into(), Ty::Kind("Point".into()))],
            result: Ty::Kind("Point".into()),
            steps: vec![
                Step::Lets {
                    prog: pre,
                    prefix: "p".into(),
                    vars: vec![(0, arg(0)), (1, arg(1))],
                },
                Step::Study {
                    func: StudyFn::RsqrtNil,
                    args: vec![(0, 0), (0, 1)],
                    outs: vec!["s0".into(), "s1".into()],
                },
                Step::Lets {
                    prog: post,
                    prefix: "t".into(),
                    vars: vec![
                        (0, arg(0)),
                        (2, arg(2)),
                        (10, Source::Local("s0".into())),
                        (11, Source::Local("s1".into())),
                    ],
                },
            ],
            entries: None,
        };
        assert_eq!(
            k.wgsl(true, "public "),
            "// A test kernel.
public fn point_f(p: Point) -> Point {
    let p0 = p.c0.x * p.c0.x;
    let r1 = study_rsqrt_nil(p0, p.c0.y);
    let s0 = r1[0];
    let s1 = r1[1];
    let t0 = s0 * 2.0;
    let t2 = fma(p.c0.x, s1, p.c0.z);
    return Point(vec4<f32>(t0, t2, (1.0 / 3.0), 0.0));
}
"
        );
    }

    #[test]
    fn names() {
        assert_eq!(snake("Multivector"), "multivector");
        assert_eq!(snake("PointPair"), "point_pair");
        assert_eq!(wgsl_coeff("m", 5), "m.c1.y");
    }
}
