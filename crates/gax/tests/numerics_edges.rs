//! `exp`, `log` and `normalized` at their edges, not on uniform samples (docs/numerics.md,
//! "Transcendental functions"): small angles and the series boundaries, pure translations,
//! tiny rotations with large translations, `log` near and at a full turn (and the branch it
//! returns there), normalization near zero norm and with a large Study part, and the CSTA
//! `exp` by scaling and squaring across norms.
//!
//! Tolerances are in units of `ε = f64::EPSILON` times the magnitude involved; where a problem
//! is ill-conditioned (the axis of a nearly full turn) the tolerance carries its condition
//! number.

#![cfg(all(
    feature = "pga3d",
    feature = "pga2d",
    feature = "vga3d",
    feature = "csta"
))]

use gax::{Extensor, Kind, Unit};
use proptest::prelude::*;

const EPS: f64 = f64::EPSILON;

/// The largest coefficient difference, and the largest coefficient of either.
fn diff<M: Extensor<Slots = (), Coef = f64>>(a: &M, b: &M) -> (f64, f64) {
    let (a, b) = (a.coeffs().as_ref(), b.coeffs().as_ref());
    let d = a
        .iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
    let s = a.iter().chain(b).fold(0.0f64, |m, x| m.max(x.abs()));
    (d, s)
}

/// `‖m ~m − 1‖∞`, relative to `‖m‖∞²` (the size of the products that make it up).
fn unit_error<M>(m: M) -> f64
where
    M: Extensor<Slots = (), Coef = f64> + gax::Reverse<Output = M> + core::ops::Mul<Output = M>,
{
    let n = m * m.reverse();
    let blades = <M::Kind as Kind>::BLADES;
    let scale = m
        .coeffs()
        .as_ref()
        .iter()
        .fold(0.0f64, |s, x| s.max(x.abs()));
    let e = n
        .coeffs()
        .as_ref()
        .iter()
        .enumerate()
        .fold(0.0f64, |s, (i, c)| {
            s.max(if blades[i] == "1" {
                (c - 1.0).abs()
            } else {
                c.abs()
            })
        });
    e / (scale * scale)
}

/// Copy coefficients between kinds by blade name (the blades of `A` must all occur in `B`).
fn embed<A, B>(a: A) -> B
where
    A: Extensor<Slots = (), Coef = f64>,
    B: Extensor<Slots = (), Coef = f64>,
{
    let (from, to) = (<A::Kind as Kind>::BLADES, <B::Kind as Kind>::BLADES);
    let c = a.coeffs();
    B::from_coeffs(<B::Kind as Kind>::arr_from_fn(|i| {
        from.iter()
            .position(|b| *b == to[i])
            .map_or(0.0, |j| c.as_ref()[j])
    }))
}

mod pga3d {
    use super::*;
    use gax::pga3d::{Line, Motor};

    fn line(dir: [f64; 3], mom: [f64; 3]) -> Line<(), f64> {
        Line::new(dir[0], dir[1], dir[2], mom[0], mom[1], mom[2])
    }
    fn unit3() -> impl Strategy<Value = [f64; 3]> {
        prop::array::uniform3(-1.0f64..1.0)
            .prop_filter("nonzero", |v| v.iter().map(|x| x * x).sum::<f64>() > 1e-2)
            .prop_map(|v| {
                let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                [v[0] / n, v[1] / n, v[2] / n]
            })
    }
    fn scaled(v: [f64; 3], s: f64) -> [f64; 3] {
        [v[0] * s, v[1] * s, v[2] * s]
    }

    proptest! {
        /// Rotation half-angles from 10⁻¹² to 1 (log-uniform) with translations of any size up
        /// to 10³: `log(exp(B)) = B` and `exp(B)` is unit, to a few ε of the magnitudes.
        #[test]
        fn exp_log_roundtrip_small_angles(
            d in unit3(), m in prop::array::uniform3(-1.0f64..1.0),
            la in -12.0f64..0.0, lt in -3.0f64..3.0,
        ) {
            let b = line(scaled(d, 10f64.powf(la)), scaled(m, 10f64.powf(lt)));
            let r = b.exp();
            prop_assert!(unit_error(r.into_inner()) < 8.0 * EPS);
            let (e, s) = diff(&r.log(), &b);
            prop_assert!(e <= 16.0 * EPS * s, "log(exp B) − B = {e:e} (|B| = {s:e})");
        }

        /// Angles straddling each series boundary of the rotation fast path (`a² = 10⁻⁴` in
        /// `exp`'s `S`, `a² = 1/4` in `S'`, `t² = 10⁻⁶` and `t² = 1/25` in `log`): `exp(B)`
        /// agrees with `exp(B/2)²`, which evaluates the other branch.
        #[test]
        fn series_boundaries_agree(
            d in unit3(), m in prop::array::uniform3(-1.0f64..1.0),
            which in 0usize..4, rel in -1e-3f64..1e-3,
        ) {
            // The boundary for B itself, so that B/2 (a quarter of a²) falls on the series side.
            let a2 = [1e-4, 0.25, 1e-6, 1.0 / 25.0][which] * (1.0 + rel);
            let b = line(scaled(d, a2.sqrt()), m);
            let whole = b.exp().into_inner();
            let half = b.gp(0.5).exp().into_inner();
            let (e, s) = diff(&whole, &(half * half));
            prop_assert!(e <= 8.0 * EPS * s, "exp(B) vs exp(B/2)²: {e:e}");
            let (e, s) = diff(&b.exp().log(), &b);
            prop_assert!(e <= 16.0 * EPS * s, "log(exp B) − B: {e:e}");
        }

        /// Pure translations (`B² = 0`, a degenerate Study number): `exp(B) = 1 + B` and
        /// `log(1 + B) = B`, for any size.
        #[test]
        fn pure_translations(m in prop::array::uniform3(-1.0f64..1.0), lt in -12.0f64..8.0) {
            let b = line([0.0; 3], scaled(m, 10f64.powf(lt)));
            let r = b.exp().into_inner();
            let expected = Motor::new(1.0, 0.0, 0.0, 0.0, b.c[3], b.c[4], b.c[5], 0.0);
            prop_assert_eq!(r, expected);
            prop_assert_eq!(b.exp().log(), b);
        }

        /// `log` near a full turn (versor half-angle `π − δ`, so `R ≈ −1`). The branch is
        /// `θ ∈ [0, π]`, so `log(exp B) = B` still, but the axis is the direction of a part of
        /// size `sin δ`, and the pitch divides by it, so the error in `B`, and in `exp(log R)`,
        /// grows like `ε / δ`.
        #[test]
        fn log_near_a_full_turn(
            d in unit3(), m in prop::array::uniform3(-1.0f64..1.0), ld in -9.0f64..-1.0,
        ) {
            let delta = 10f64.powf(ld);
            let b = line(scaled(d, core::f64::consts::PI - delta), m);
            let r = b.exp();
            let l = r.log();
            let (e, s) = diff(&l, &b);
            prop_assert!(e <= 16.0 * EPS * s / delta, "log(exp B) − B = {e:e}, δ = {delta:e}");
            let (e, s) = diff(&l.exp().into_inner(), &r.into_inner());
            // The translation along the axis is divided by sin δ, so this is ε / δ too.
            prop_assert!(e <= 32.0 * EPS * s / delta, "exp(log R) − R = {e:e}");
        }

        /// A half turn of the motion (versor half-angle `π/2`, scalar part 0) is not special.
        #[test]
        fn log_at_a_half_turn(d in unit3(), m in prop::array::uniform3(-1.0f64..1.0)) {
            let b = line(scaled(d, core::f64::consts::FRAC_PI_2), m);
            let (e, s) = diff(&b.exp().log(), &b);
            prop_assert!(e <= 16.0 * EPS * s);
        }

        /// Exactly at `R = −T` (a full turn times a translation, including `R = −1`) no unique
        /// logarithm exists; `log` returns `log(−R)`, the same motion.
        #[test]
        fn log_at_minus_a_translation(m in prop::array::uniform3(-1.0f64..1.0)) {
            let t = line([0.0; 3], m).exp().into_inner();
            let minus = Unit::new_unchecked(-t);
            let l = minus.log();
            prop_assert_eq!(l, line([0.0; 3], m));
            prop_assert_eq!(l.exp().into_inner(), t);
        }

        /// `normalized` is scale-invariant down to a norm of 10⁻¹⁵⁰ (above that `x ~x`
        /// underflows; at zero the result is not finite).
        #[test]
        fn normalized_near_zero_norm(
            d in unit3(), m in prop::array::uniform3(-1.0f64..1.0), ls in -150.0f64..0.0,
        ) {
            let u = line(scaled(d, 0.7), m).exp().into_inner();
            let small = u.gp(10f64.powf(ls));
            let (e, s) = diff(&small.normalized().into_inner(), &u);
            prop_assert!(e <= 8.0 * EPS * s, "{e:e}");
        }

        /// Normalizing `R (s + p I)` gives back `R` even when the Study part `p` is large
        /// relative to `s`: the inverse square root of a dual number is exact in form, so the
        /// error grows only with `p/s`, the magnitude of the input.
        #[test]
        fn normalized_with_a_large_study_part(
            d in unit3(), m in prop::array::uniform3(-1.0f64..1.0), lp in -3.0f64..6.0,
        ) {
            let u = line(scaled(d, 0.7), m).exp().into_inner();
            let p = 10f64.powf(lp);
            let x = u * Motor::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p);
            let (e, s) = diff(&x.normalized().into_inner(), &u);
            prop_assert!(e <= 16.0 * EPS * s * (1.0 + p), "p = {p:e}: {e:e}");
            prop_assert!(unit_error(x.normalized().into_inner()) <= 16.0 * EPS * (1.0 + p));
        }
    }
}

mod pga2d_vga3d {
    use super::*;

    proptest! {
        #[test]
        fn pga2d_small_angles_and_boundaries(
            la in -12.0f64..0.0, x in -1.0f64..1.0, y in -1.0f64..1.0, which in 0usize..5,
        ) {
            use gax::pga2d::Point;
            let a = if which == 4 { 10f64.powf(la) } else { [1e-4f64, 0.25, 1e-6, 0.04][which].sqrt() };
            // The PGA2D bivector is a point; its e12 part is the rotation.
            let b = Point::<(), f64>::from_coeffs([x, y, a]);
            let (e, s) = diff(&b.exp().log(), &b);
            prop_assert!(e <= 16.0 * EPS * s);
            let half = b.gp(0.5).exp().into_inner();
            let (e, s) = diff(&b.exp().into_inner(), &(half * half));
            prop_assert!(e <= 8.0 * EPS * s);
        }

        #[test]
        fn vga3d_rotors_near_zero_and_a_full_turn(
            v in prop::array::uniform3(-1.0f64..1.0), la in -12.0f64..0.0, ld in -9.0f64..-1.0,
        ) {
            use gax::vga3d::Bivector;
            let n = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-3);
            let dir = [v[0] / n, v[1] / n, v[2] / n];
            for (a, cond) in [(10f64.powf(la), 1.0), (core::f64::consts::PI - 10f64.powf(ld), 10f64.powf(-ld))] {
                let b = Bivector::<(), f64>::new(dir[0] * a, dir[1] * a, dir[2] * a);
                let (e, s) = diff(&b.exp().log(), &b);
                prop_assert!(e <= 16.0 * EPS * s * cond, "a = {a}: {e:e}");
            }
        }
    }
}

mod csta {
    use super::*;
    use gax::csta::{Bivector, Even, Motor, Twist};

    /// The previous scheme: a fixed `B/256`, 8 Taylor terms, 8 squarings, no renormalization.
    fn exp_fixed(b: Bivector<(), f64>) -> Even<(), f64> {
        let x: Even<(), f64> = embed(b.gp(1.0 / 256.0));
        let mut one = Even::<(), f64>::zero();
        one.c[0] = 1.0;
        let mut r = one;
        for k in (1..=8).rev() {
            r = one + (x * r).gp(1.0 / f64::from(k));
        }
        for _ in 0..8 {
            r = r * r;
        }
        r
    }

    /// A deterministic direction per family: rotations, boosts, translations, special
    /// conformal (`e·o`), dilations (`eoi`), and all of them.
    fn direction(family: usize, seed: u64) -> Bivector<(), f64> {
        let mut s = seed | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let blades = <<Bivector<(), f64> as Extensor>::Kind as Kind>::BLADES;
        let keep = |b: &str| match family {
            0 => ["e23", "e31", "e12"].contains(&b),
            1 => ["e41", "e42", "e43"].contains(&b),
            2 => b.ends_with('i') && b != "eoi",
            3 => b.ends_with('o'),
            4 => b == "eoi",
            _ => true,
        };
        let c: Vec<f64> = blades
            .iter()
            .map(|b| if keep(b) { next() } else { 0.0 })
            .collect();
        let l1: f64 = c.iter().map(|x| x.abs()).sum();
        Bivector::from_coeffs(core::array::from_fn(|i| c[i] / l1))
    }

    /// `‖exp(B) exp(B)~ − 1‖` (relative to `‖exp B‖²`) across norms, for the adaptive scheme
    /// and the fixed one. The adaptive result must be unit to within the rounding of its
    /// squarings, and agree with the closed form of `Twist::exp` on the Poincaré bivectors.
    #[test]
    fn exp_by_scaling_and_squaring_across_norms() {
        let norms = [1e-8, 1e-3, 0.1, 1.0, 4.0, 16.0, 64.0];
        let names = [
            "rotation",
            "boost",
            "translation",
            "special",
            "dilation",
            "general",
        ];
        println!("family       ‖B‖₁    adaptive     fixed(1/256)");
        for (family, name) in names.iter().enumerate() {
            for (seed, rho) in norms.iter().enumerate() {
                let mut worst = (0.0f64, 0.0f64);
                for k in 0..8 {
                    let b = direction(family, 0x9e37_79b9 * (seed as u64 + 1) + k).gp(*rho);
                    let adaptive = unit_error(b.exp().into_inner());
                    let fixed = unit_error(exp_fixed(b));
                    worst = (worst.0.max(adaptive), worst.1.max(fixed));
                    // Rounding grows by about 2 per squaring, and there are log2(16 ‖B‖₁) of them
                    // where the final Newton step does not apply.
                    let bound = 16.0 * EPS * (16.0 * rho).max(1.0);
                    assert!(adaptive < bound, "{name} ‖B‖ = {rho}: {adaptive:e}");
                    if family < 3 {
                        // Twist blades: compare with the closed form.
                        let t: Twist<(), f64> = embed(b);
                        let closed: Even<(), f64> =
                            embed::<Motor<(), f64>, _>(t.exp().into_inner());
                        let (e, s) = diff(&b.exp().into_inner(), &closed);
                        let tol = 1e3 * EPS * s * rho.max(1.0);
                        assert!(
                            e <= tol,
                            "{name} ‖B‖ = {rho}: differs from the closed form by {e:e}"
                        );
                    }
                }
                println!("{name:<12} {rho:<7} {:<12.2e} {:.2e}", worst.0, worst.1);
            }
        }
    }
}
