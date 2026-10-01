//! Checks shared by the 7D, 8D and 9D algebras (each test file declares its own).

#![allow(dead_code)]

/// A small deterministic generator (xorshift), uniform in [-1, 1).
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// Uniform elements of the prime field `gax::fp::Fp` (splitmix64).
pub struct FieldRng(pub u64);
impl FieldRng {
    pub fn next(&mut self) -> gax::fp::Fp {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        gax::fp::Fp::new(z ^ (z >> 31))
    }
}

/// Every coefficient within `tol` of the other, relative to the largest (at least 1).
pub fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    worst(a, b) <= tol
}

/// The largest difference, relative to the largest coefficient (at least 1).
pub fn worst(a: &[f64], b: &[f64]) -> f64 {
    let scale = a.iter().chain(b).fold(1.0f64, |m, x| m.max(x.abs()));
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
        / scale
}

/// The same checks for each algebra, whose kinds share names: `$alg` declares `Vector`,
/// `Bivector` and `Even`, and `$planes` lists disjoint basis bivectors (their positions in
/// `Bivector`), one per invariant plane, with 1 where the plane is a rotation.
#[macro_export]
macro_rules! checks {
    ($alg:ident, $checks:ident, $planes:expr) => {
        mod $checks {
            use super::$alg::{Bivector, Even, Vector};
            use gax::Unit;
            use $crate::common::{Rng, close, worst};

            const PLANES: &[(usize, bool)] = &$planes;
            const HALF: f64 = core::f64::consts::FRAC_PI_2;

            /// `a` embedded in the kind of `b`.
            fn like<A, B: From<A>>(a: A, _: &B) -> B {
                B::from(a)
            }

            fn bivector(rng: &mut Rng, size: f64) -> Bivector<(), f64> {
                Bivector::from_coeffs(core::array::from_fn(|_| size * rng.next()))
            }

            fn planes(angles: &[f64]) -> Bivector<(), f64> {
                let mut c = [0.0f64; <Bivector as gax::kind::Kind>::N];
                for (&a, &(i, _)) in angles.iter().zip(PLANES) {
                    c[i] = a;
                }
                Bivector::from_coeffs(c)
            }

            /// Small enough that every plane is well below a half turn: `log(exp B) = B`.
            #[test]
            fn log_of_exp_is_the_bivector() {
                let mut rng = Rng(0x7d_1065);
                for size in [0.01, 0.1, 0.2] {
                    for _ in 0..30 {
                        let b = bivector(&mut rng, size);
                        let back: Bivector<(), f64> = b.exp().log();
                        assert!(
                            close(&back.c, &b.c, 1e-11),
                            "size {size}: {back:?} vs {b:?}"
                        );
                    }
                }
            }

            /// Larger versors, planes near and past half turns included (`⟨R⟩₀` of either
            /// sign): the log is a log of `R` and a fixed point of `log ∘ exp`.
            #[test]
            fn exp_of_log_is_the_versor() {
                let mut rng = Rng(0x7d_e7b0);
                let mut negative = 0;
                for size in [0.4, 0.7, 1.0, 1.4] {
                    for _ in 0..60 {
                        let r = bivector(&mut rng, size).exp();
                        negative += usize::from(r.into_inner().c[0] < 0.0);
                        let b: Bivector<(), f64> = r.log();
                        let again = b.exp();
                        assert!(
                            close(&again.into_inner().c, &r.into_inner().c, 1e-9),
                            "size {size}: exp(log R) differs from R by {:e}",
                            worst(&again.into_inner().c, &r.into_inner().c)
                        );
                        let b2: Bivector<(), f64> = again.log();
                        assert!(
                            close(&b2.c, &b.c, 1e-8),
                            "size {size}: log(exp(log R)) differs from log R by {:e}",
                            worst(&b2.c, &b.c)
                        );
                    }
                }
                // Rotations only: some samples are past a half turn (⟨R⟩₀ < 0).
                if PLANES.iter().all(|p| p.1) {
                    assert!(negative > 0, "no versor past a half turn was sampled");
                }
            }

            /// Each basis plane alone (coinciding invariants), and where it is a rotation,
            /// towards and past a half turn (turned first).
            #[test]
            fn single_planes() {
                for i in 0..<Bivector as gax::kind::Kind>::N {
                    let mut c = [0.0f64; <Bivector as gax::kind::Kind>::N];
                    c[i] = 0.7;
                    let b = Bivector::from_coeffs(c);
                    let back: Bivector<(), f64> = b.exp().log();
                    assert!(close(&back.c, &b.c, 1e-13), "plane {i}: {back:?}");
                    for angle in [HALF - 1e-6, HALF + 0.4] {
                        c[i] = angle;
                        let b = Bivector::from_coeffs(c);
                        let r = b.exp();
                        if r.into_inner().c[0].abs() < 0.5 {
                            let back: Bivector<(), f64> = r.log();
                            assert!(
                                close(&back.c, &b.c, 1e-12),
                                "plane {i} at {angle}: {back:?}"
                            );
                        }
                    }
                }
            }

            /// Several invariant planes at once: coinciding angles (isoclinic clusters of two,
            /// three and four), near and past half turns, and boosts where the plane is one.
            #[test]
            fn clusters_and_half_turns() {
                let sets: &[&[f64]] = &[
                    &[0.6, 0.6],
                    &[0.6, 0.6, 0.6],
                    &[0.6, 0.6, 0.6, 0.6],
                    &[0.3, 0.9, 1.2, 0.5],
                    &[HALF - 1e-7, 0.4],
                    &[HALF - 1e-2, HALF - 1e-2, 0.3],
                    &[HALF + 0.3, 0.5, 0.2],
                    &[HALF + 0.3, HALF + 0.3, 0.7],
                    &[HALF + 0.2, 0.4, 0.4, 0.4],
                    &[1.2, 1.2, 1.2, 0.2],
                    &[HALF - 0.01, 0.8, 0.8],
                ];
                let mut checked = 0;
                for angles in sets {
                    if angles.len() > PLANES.len() {
                        continue;
                    }
                    let b = planes(angles);
                    let r = b.exp();
                    let got: Bivector<(), f64> = r.log();
                    let again = got.exp();
                    assert!(
                        close(&again.into_inner().c, &r.into_inner().c, 1e-9),
                        "{angles:?}: exp(log R) differs from R by {:e}",
                        worst(&again.into_inner().c, &r.into_inner().c)
                    );
                    // On the principal branch (every rotation below a half turn, ⟨R⟩₀ > 0 and
                    // away from it), the log is B itself.
                    let principal = angles
                        .iter()
                        .zip(PLANES)
                        .all(|(&a, p)| !p.1 || a < HALF - 1e-3)
                        && r.into_inner().c[0] > 1e-3;
                    if principal {
                        assert!(close(&got.c, &b.c, 1e-9), "{angles:?}: {got:?}");
                    }
                    checked += 1;
                }
                assert!(checked >= 7);
            }

            /// Lanes of one SIMD value that take different paths (closed form, turning), each
            /// equal to its scalar result.
            #[cfg(feature = "batch")]
            #[test]
            fn lanes() {
                type L = gax::batch::Lanes<f64, 4>;
                let mut rng = Rng(0x7d_1a4e);
                let bs = [
                    bivector(&mut rng, 0.3),
                    planes(&[HALF - 1e-5, 0.2]),
                    bivector(&mut rng, 1.0),
                    planes(&[HALF + 0.3, 0.4, 0.4]),
                ];
                let rs: Vec<Even<(), f64>> = bs.iter().map(|b| b.exp().into_inner()).collect();
                let lanes = Even::<(), L>::from_coeffs(core::array::from_fn(|i| {
                    L::new([rs[0].c[i], rs[1].c[i], rs[2].c[i], rs[3].c[i]])
                }));
                let got: Bivector<(), L> = Unit::new_unchecked(lanes).log();
                for (k, r) in rs.iter().enumerate() {
                    let want: Bivector<(), f64> = Unit::new_unchecked(*r).log();
                    let lane: [f64; <Bivector as gax::kind::Kind>::N] =
                        core::array::from_fn(|i| got.c[i].v[k]);
                    assert!(
                        close(&lane, &want.c, 1e-12),
                        "lane {k}: {lane:?} vs {want:?}"
                    );
                }
            }

            /// `exp` of several basis planes is the product of their exponentials, each
            /// `cos θ + sin θ e` (rotation), `cosh + sinh` (boost) or `1 + a e` (null), at angles
            /// within a quarter turn, beyond it (turned back) and beyond three quarters (halved).
            #[test]
            fn exp_of_planes() {
                let even_of = |i: usize, a: f64| -> Even<(), f64> {
                    let mut c = [0.0f64; <Bivector as gax::kind::Kind>::N];
                    c[i] = 1.0;
                    let b = Bivector::<(), f64>::from_coeffs(c);
                    let sq = (b * b).c[0]; // e² = −1, +1 or 0
                    let (cs, sn) = if sq < -0.5 {
                        (a.cos(), a.sin())
                    } else if sq > 0.5 {
                        (a.cosh(), a.sinh())
                    } else {
                        (1.0, a)
                    };
                    let mut r = Even::<(), f64>::zero();
                    r.c[0] = cs;
                    let e: Even<(), f64> = b.into();
                    r + e.gp(sn)
                };
                for angles in [
                    &[0.3, 0.5, -0.2, 0.1][..],
                    &[1.2, 0.4, 0.7, -0.3],
                    &[2.9, -2.0, 0.9, 1.3],
                    &[5.0, 0.0, 0.0, 0.0],
                    &[HALF, HALF, 0.25, 0.6],
                ] {
                    let mut c = [0.0f64; <Bivector as gax::kind::Kind>::N];
                    let mut want = Even::<(), f64>::zero();
                    want.c[0] = 1.0;
                    for (&a, &(i, _)) in angles.iter().zip(PLANES) {
                        c[i] = a;
                        want = want * even_of(i, a);
                    }
                    let got = Bivector::<(), f64>::from_coeffs(c).exp().into_inner();
                    assert!(
                        close(&got.c, &want.c, 1e-12),
                        "{angles:?}: off by {:e}",
                        worst(&got.c, &want.c)
                    );
                }
            }

            /// `exp` of random bivectors: a unit versor, and the same as a Taylor series with
            /// scaling and squaring (written here, from products alone).
            #[test]
            fn exp_matches_a_series() {
                let mut rng = Rng(0x7d_e4b0);
                for size in [0.05, 0.3, 0.8, 1.5] {
                    for _ in 0..20 {
                        let b = bivector(&mut rng, size);
                        let got = b.exp().into_inner();
                        let norm: f64 = b.c.iter().map(|x| x.abs()).sum();
                        let mut s = 0;
                        while norm / f64::from(1u32 << s) > 0.25 {
                            s += 1;
                        }
                        let x: Even<(), f64> = b.gp(1.0 / f64::from(1u32 << s)).into();
                        let mut want = Even::<(), f64>::zero();
                        want.c[0] = 1.0;
                        let mut term = want;
                        for k in 1..30 {
                            term = (term * x).gp(1.0 / f64::from(k));
                            want += term;
                        }
                        for _ in 0..s {
                            want = want * want;
                        }
                        let scale = want.c.iter().fold(1.0f64, |m, v| m.max(v.abs()));
                        assert!(
                            close(&got.c, &want.c, 1e-12 * scale.max(1.0)),
                            "size {size}: off by {:e}",
                            worst(&got.c, &want.c)
                        );
                        let unit = got * got.reverse();
                        assert!(
                            (unit.c[0] - 1.0).abs() < 1e-11 * scale * scale,
                            "not unit: {:?}",
                            unit.c[0]
                        );
                    }
                }
            }

            #[cfg(feature = "batch")]
            #[test]
            fn exp_in_lanes() {
                type L = gax::batch::Lanes<f64, 4>;
                let mut rng = Rng(0x7d_e1a4);
                let bs = [
                    bivector(&mut rng, 0.2),
                    planes(&[2.9, 0.4]),
                    bivector(&mut rng, 1.2),
                    planes(&[HALF + 0.3, 0.4, 0.4]),
                ];
                let lanes = Bivector::<(), L>::from_coeffs(core::array::from_fn(|i| {
                    L::new([bs[0].c[i], bs[1].c[i], bs[2].c[i], bs[3].c[i]])
                }));
                let got = lanes.exp().into_inner();
                for (k, b) in bs.iter().enumerate() {
                    let want = b.exp().into_inner();
                    let lane: [f64; <Even as gax::kind::Kind>::N] =
                        core::array::from_fn(|i| got.c[i].v[k]);
                    assert!(
                        close(&lane, &want.c, 1e-12),
                        "lane {k}: off by {:e}",
                        worst(&lane, &want.c)
                    );
                }
            }

            #[test]
            fn exp_in_f32() {
                let mut rng = Rng(0x7d_e32f);
                for _ in 0..20 {
                    let b = bivector(&mut rng, 0.6);
                    let want = b.exp().into_inner();
                    let b32 =
                        Bivector::<(), f32>::from_coeffs(core::array::from_fn(|i| b.c[i] as f32));
                    let got = b32.exp().into_inner();
                    let got: [f64; <Even as gax::kind::Kind>::N] =
                        core::array::from_fn(|i| f64::from(got.c[i]));
                    assert!(
                        close(&got, &want.c, 2e-5),
                        "off by {:e}",
                        worst(&got, &want.c)
                    );
                }
            }

            #[test]
            fn in_f32() {
                let mut rng = Rng(0x7d_f32f);
                for _ in 0..30 {
                    let b = bivector(&mut rng, 0.3);
                    let r = b.exp().into_inner();
                    let r32 = Even::<(), f32>::from_coeffs(core::array::from_fn(|i| r.c[i] as f32));
                    let got: Bivector<(), f32> = Unit::new_unchecked(r32).log();
                    let got: [f64; <Bivector as gax::kind::Kind>::N] =
                        core::array::from_fn(|i| f64::from(got.c[i]));
                    assert!(close(&got, &b.c, 5e-4), "{got:?} vs {:?}", b.c);
                }
            }

            /// Exactly, over the prime field `ℤ/p` (`gax::fp::Fp`, `p = 2⁶¹ − 1`): unit even
            /// versors as products of reflections of a basis vector in uniform random vectors.
            /// The plain sandwich kernels equal the projection of `(v x) ~v`, their prepared map
            /// agrees, and unit versors compose. A law that fails is a nonzero polynomial of
            /// degree below `36 n` in the samples, so each sample misses the failure with
            /// probability below `2⁻⁵²` (Schwartz–Zippel; docs/laws.md, law L).
            #[test]
            fn exact_mod_p() {
                use gax::Coef;
                use gax::fp::Fp as F;
                let mut rng = $crate::common::FieldRng(0x7d_f1e1_d5eed);
                let n = <Vector as gax::kind::Kind>::N;
                let basis = |i: usize| {
                    Vector::<(), F>::from_coeffs(core::array::from_fn(|k| {
                        if k == i { F::one() } else { F::zero() }
                    }))
                };
                let u0 = (0..n)
                    .map(basis)
                    .find(|e| (*e | *e).s() == F::one())
                    .expect("a basis vector of square 1");
                let unit = |rng: &mut $crate::common::FieldRng| loop {
                    let w = Vector::<(), F>::from_coeffs(core::array::from_fn(|_| rng.next()));
                    let (d, q) = ((u0 | w).s(), (w | w).s());
                    if d != F::zero() && q != F::zero() {
                        break u0 - w.gp(d * q.inv() * F::from_i64(2));
                    }
                };
                // An even versor of the algebra: the product of 2 ⌊n/2⌋ unit vectors.
                let versor = |rng: &mut $crate::common::FieldRng| -> Unit<Even<(), F>> {
                    let mut v: Even<(), F> = unit(rng) * unit(rng);
                    for _ in 1..n / 2 {
                        v = v * (unit(rng) * unit(rng));
                    }
                    Unit::new_unchecked(v)
                };
                for _ in 0..4 {
                    let (a, b) = (versor(&mut rng), versor(&mut rng));
                    let x = Vector::<(), F>::from_coeffs(core::array::from_fn(|_| rng.next()));
                    let y = Bivector::<(), F>::from_coeffs(core::array::from_fn(|_| rng.next()));
                    let v = a.into_inner();
                    let want = v * x * v.reverse();
                    assert_eq!(like(a >> x, &want), want, "a >> x");
                    assert_eq!(gax::Prepare::<Vector>::prepare(a) >> x, a >> x, "prepared");
                    let want = v * y * v.reverse();
                    assert_eq!(like(a >> y, &want), want, "a >> y");
                    assert_eq!((a * b) >> x, a >> (b >> x), "composition");
                }
            }

            /// The plain sandwich kernels (versors over 32 coefficients): `v >> x` and its
            /// prepared map against the products `v x ~v`.
            #[test]
            fn sandwiches() {
                let mut rng = Rng(0x7d_5a4d);
                for _ in 0..10 {
                    let v = bivector(&mut rng, 0.5).exp();
                    let x = Vector::<(), f64>::from_coeffs(core::array::from_fn(|_| rng.next()));
                    let want = v.into_inner() * x * v.into_inner().reverse();
                    let got = v >> x;
                    let prepared = gax::Prepare::<Vector>::prepare(v) >> x;
                    assert!(close(&like(got, &want).c, &want.c, 1e-12), "{got:?}");
                    assert!(
                        close(&like(prepared, &want).c, &want.c, 1e-12),
                        "{prepared:?}"
                    );
                    let y = bivector(&mut rng, 1.0);
                    let want = v.into_inner() * y * v.into_inner().reverse();
                    let got = v >> y;
                    assert!(close(&like(got, &want).c, &want.c, 1e-12), "{got:?}");
                }
            }
        }
    };
}
