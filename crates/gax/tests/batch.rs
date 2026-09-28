//! The batch kernels (`gax::batch`) against the scalar kernels, lane for lane, on every SIMD
//! level this CPU can run (and the portable path), with lengths that leave remainders.

#![cfg(all(feature = "batch", feature = "pga3d", feature = "pga2d"))]

use gax::batch::{self, Batch, BatchTransform, Kernel, Map, SandwichKernel, Soa};
use gax::{Coef, Extensor, Real, Unit};

/// Lengths around every lane count and block size.
const LENS: &[usize] = &[0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 31, 32, 33, 100];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
    fn value<M: Extensor<Slots = ()>>(&mut self) -> M {
        M::from_coeffs(<M::Kind as gax::Kind>::arr_from_fn(|_| {
            M::Coef::from_f64(self.next())
        }))
    }
    fn values<M: Extensor<Slots = ()>>(&mut self, n: usize) -> Vec<M> {
        (0..n).map(|_| self.value()).collect()
    }
}

fn coeffs<M: Extensor<Slots = ()>>(m: &M) -> Vec<f64>
where
    M::Coef: Into<f64>,
{
    m.coeffs().as_ref().iter().map(|&x| x.into()).collect()
}

/// Every lane matches the scalar result to `tol`, relative to the magnitude of the result.
fn assert_lanes<M: Extensor<Slots = ()>>(got: &[M], want: &[M], tol: f64, what: &str)
where
    M::Coef: Into<f64>,
{
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let (g, w) = (coeffs(g), coeffs(w));
        let scale = w.iter().fold(1.0f64, |m, x| m.max(x.abs()));
        for (a, b) in g.iter().zip(&w) {
            assert!(
                (a - b).abs() <= tol * scale,
                "{what} [{}], element {i}: {g:?} vs {w:?}",
                batch::level_name(batch::level())
            );
        }
    }
}

fn each_level(mut f: impl FnMut()) {
    for level in batch::levels() {
        batch::with_level(level, &mut f);
    }
}

/// Uniform, pairwise, and struct-of-arrays sandwiches of versor type `V` on kind `M`.
fn check_sandwich<V, M>(rng: &mut Rng, make_versor: impl Fn(&mut Rng) -> V, tol: f64, what: &str)
where
    V: BatchTransform + gax::Transform<M, Output = M>,
    M: Extensor<Slots = ()>,
    M::Coef: gax::batch::LaneElem + Into<f64>,
    V::Kind: SandwichKernel<M::Kind, V::Cert, Versor<M::Coef> = V, Y = M::Kind>,
    <V::Kind as gax::Kind>::Mv<(), M::Coef>: Extensor<Slots = ()>,
    V: Copy,
    <M::Kind as gax::Kind>::Mv<(), M::Coef>: Into<M> + From<M>,
{
    each_level(|| {
        for &n in LENS {
            let v = make_versor(rng);
            let xs: Vec<M> = rng.values(n);
            let want: Vec<M> = xs.iter().map(|&x| v.transform(x)).collect();
            // Array of structs, one versor.
            let mut out: Vec<batch::Mv<M::Kind, M::Coef>> = vec![zero(); n];
            v.transform_slice(&xs, &mut out);
            let got: Vec<M> = out.iter().map(|&y| y.into()).collect();
            assert_lanes(&got, &want, tol, &format!("{what} transform_slice n={n}"));
            // Struct of arrays, one versor.
            let soa: Soa<M::Kind, M::Coef> = xs.iter().map(|&x| x.into()).collect();
            let mut out_soa = Soa::new();
            v.transform_soa(&soa, &mut out_soa);
            let got: Vec<M> = out_soa.to_vec().into_iter().map(Into::into).collect();
            assert_lanes(&got, &want, tol, &format!("{what} transform_soa n={n}"));
            // Pairs.
            let vs: Vec<V> = (0..n).map(|_| make_versor(rng)).collect();
            let want: Vec<M> = vs.iter().zip(&xs).map(|(&v, &x)| v.transform(x)).collect();
            let mut out: Vec<batch::Mv<M::Kind, M::Coef>> = vec![zero(); n];
            V::transform_each(&vs, &xs, &mut out);
            let got: Vec<M> = out.iter().map(|&y| y.into()).collect();
            assert_lanes(&got, &want, tol, &format!("{what} transform_each n={n}"));
            let vsoa: Soa<V::Kind, M::Coef> = vs
                .iter()
                .map(|&v| <V::Kind as SandwichKernel<M::Kind, V::Cert>>::unwrap(v))
                .collect();
            V::transform_each_soa(&vsoa, &soa, &mut out_soa);
            let got: Vec<M> = out_soa.to_vec().into_iter().map(Into::into).collect();
            assert_lanes(
                &got,
                &want,
                tol,
                &format!("{what} transform_each_soa n={n}"),
            );
        }
    });
}

fn zero<M: Extensor<Slots = ()>>() -> M {
    M::from_coeffs(<M::Kind as gax::Kind>::arr_from_fn(|_| M::Coef::zero()))
}

mod pga3d {
    use super::*;
    use gax::pga3d::{Line, Motor, Plane, Point};

    fn unit_motor<T: Real>(rng: &mut Rng) -> Unit<Motor<(), T>> {
        let m: Motor<(), T> = rng.value();
        (m + Motor::from_coeffs(core::array::from_fn(|i| {
            if i == 0 { T::from_f64(2.0) } else { T::zero() }
        })))
        .normalized()
    }

    #[test]
    fn sandwiches_f32() {
        let mut rng = Rng(1);
        check_sandwich::<Unit<Motor>, Point>(&mut rng, unit_motor, 2e-5, "Unit<Motor> >> Point");
        check_sandwich::<Unit<Motor>, Line>(&mut rng, unit_motor, 2e-5, "Unit<Motor> >> Line");
        check_sandwich::<Unit<Motor>, Plane>(&mut rng, unit_motor, 2e-5, "Unit<Motor> >> Plane");
        check_sandwich::<Motor, Point>(&mut rng, Rng::value, 2e-5, "Motor >> Point");
        check_sandwich::<Plane, Line>(&mut rng, Rng::value, 2e-5, "Plane >> Line");
    }

    #[test]
    fn sandwiches_f64() {
        let mut rng = Rng(2);
        check_sandwich::<Unit<Motor<(), f64>>, Point<(), f64>>(
            &mut rng,
            unit_motor,
            1e-12,
            "Unit<Motor> >> Point (f64)",
        );
        check_sandwich::<Motor<(), f64>, Line<(), f64>>(
            &mut rng,
            Rng::value,
            1e-12,
            "Motor >> Line (f64)",
        );
    }

    /// A built map applied to many values, against `of` value by value: a square map (a
    /// motor then a central projection) and a non-square one (the lines through a point).
    #[test]
    fn maps_of_slices() {
        use gax::batch::BatchOf;
        let mut rng = Rng(5);
        each_level(|| {
            for &n in LENS {
                let m = unit_motor::<f32>(&mut rng);
                let eye: Point = rng.value();
                let screen: Plane = rng.value();
                let square: Point<(Point,)> = (eye & (m >> Point::slot())) ^ screen;
                let wide: Line<(Point,)> = eye & Point::slot();
                let xs: Vec<Point> = rng.values(n);
                let want: Vec<Point> = xs.iter().map(|&x| square.of(x)).collect();
                let mut got = vec![Point::zero(); n];
                square.of_slice(&xs, &mut got);
                assert_lanes(&got, &want, 2e-5, &format!("Point <- Point of_slice n={n}"));
                let soa: Soa<Point> = xs.iter().copied().collect();
                let mut out = Soa::new();
                square.of_soa(&soa, &mut out);
                assert_lanes(&out.to_vec(), &want, 2e-5, &format!("of_soa n={n}"));
                let want: Vec<Line> = xs.iter().map(|&x| wide.of(x)).collect();
                let mut got = vec![Line::zero(); n];
                wide.of_slice(&xs, &mut got);
                assert_lanes(&got, &want, 2e-5, &format!("Line <- Point of_slice n={n}"));
                let map64: Point<(Point,), f64> = unit_motor::<f64>(&mut rng) >> Point::slot();
                let xs: Vec<Point<(), f64>> = rng.values(n);
                let want: Vec<Point<(), f64>> = xs.iter().map(|&x| map64.of(x)).collect();
                let mut got = vec![Point::zero(); n];
                map64.of_slice(&xs, &mut got);
                assert_lanes(&got, &want, 1e-12, &format!("f64 of_slice n={n}"));
            }
        });
    }

    struct ExpLog;
    impl Map for ExpLog {
        type X = Line;
        type Y = Line;
        #[inline(always)]
        fn call<T: Real>(&self, b: Line<(), T>) -> Line<(), T> {
            let m = b.exp();
            // Round trip through the logarithm, then the motor's action on the input.
            gax::Log::log(m) + (m >> b)
        }
    }

    #[test]
    fn exp_and_log_map() {
        let mut rng = Rng(3);
        each_level(|| {
            for &n in LENS {
                let bs: Vec<Line> = rng.values(n);
                let want: Vec<Line> = bs.iter().map(|&b| ExpLog.call(b)).collect();
                let mut got = vec![Line::zero(); n];
                batch::map(&ExpLog, &bs, &mut got);
                assert_lanes(&got, &want, 1e-5, &format!("exp/log n={n}"));
                let bs: Vec<Line<(), f64>> = rng.values(n);
                let want: Vec<Line<(), f64>> = bs.iter().map(|&b| ExpLog.call(b)).collect();
                let mut got = vec![Line::zero(); n];
                batch::map(&ExpLog, &bs, &mut got);
                assert_lanes(&got, &want, 1e-12, &format!("exp/log f64 n={n}"));
            }
        });
    }
}

mod pga2d {
    use super::*;
    use gax::pga2d::{Line, Motor, Point};

    #[test]
    fn sandwiches() {
        let mut rng = Rng(4);
        let unit = |r: &mut Rng| {
            let m: Motor = r.value();
            (m + Motor::new(2.0, 0.0, 0.0, 0.0)).normalized()
        };
        check_sandwich::<Unit<Motor>, Point>(&mut rng, unit, 2e-5, "pga2d Unit<Motor> >> Point");
        check_sandwich::<Unit<Motor>, Line>(&mut rng, unit, 2e-5, "pga2d Unit<Motor> >> Line");
        check_sandwich::<Line, Point>(&mut rng, Rng::value, 2e-5, "pga2d Line >> Point");
    }
}

/// The vectorized elementary functions agree with [`batch::math`] bit for bit, lane for lane.
struct Elementary<'a>(&'a [f32], &'a [f32]);
impl Kernel<f32> for Elementary<'_> {
    type Output = ();
    #[inline(always)]
    fn run<L: Batch<Elem = f32>>(self) {
        let n = L::LANES;
        let (xs, ys) = (self.0, self.1);
        for (x, y) in xs.chunks_exact(n).zip(ys.chunks_exact(n)) {
            let (lx, ly) = (L::load(x), L::load(y));
            let (s, c) = lx.sin_cos();
            let results = [
                (s, "sin"),
                (c, "cos"),
                (lx.sinh(), "sinh"),
                (lx.cosh(), "cosh"),
                (lx.abs().ln(), "ln"),
                (ly.atan2(lx), "atan2"),
            ];
            for l in 0..n {
                let (bs, bc) = batch::math::sin_cos(x[l]);
                let want = [
                    bs,
                    bc,
                    batch::math::sinh(x[l]),
                    batch::math::cosh(x[l]),
                    batch::math::ln(x[l].abs()),
                    batch::math::atan2(y[l], x[l]),
                ];
                for ((got, name), want) in results.iter().zip(want) {
                    let got = got.lane(l);
                    assert!(
                        got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                        "{name}({}, {}) [{}]: {got} vs {want}",
                        x[l],
                        y[l],
                        batch::level_name(batch::level())
                    );
                }
            }
        }
    }
}

#[test]
fn elementary_functions_match_the_scalar_ones() {
    let mut xs: Vec<f32> = (0..4096).map(|i| (i as f32 - 2048.0) * 0.0371).collect();
    xs.extend([
        0.0,
        -0.0,
        1e-40,
        -1e-40,
        f32::MIN_POSITIVE,
        1.0,
        -1.0,
        88.0,
        -88.0,
        100.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        3.0e5,
        -1e-3,
        0.5,
    ]);
    let ys: Vec<f32> = xs.iter().rev().copied().collect();
    each_level(|| batch::run(Elementary(&xs, &ys)));
}
