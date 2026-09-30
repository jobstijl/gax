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
                    &[HALF - 1e-4, HALF - 1e-4, 0.3],
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
