//! The batch forms of the traced kernels agree with the scalar fused kernels, lane for lane,
//! on every SIMD level this CPU can run, for lengths with remainders and with broadcast
//! arguments.

use gax::batch;
use gax::pga3d::{Line, Motor, Plane, Point};
use gax::{Extensor, Unit};
use gax_example_traced::*;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn point(&mut self) -> Point {
        Point::xyz(self.next(), self.next(), self.next())
    }
    fn motor(&mut self) -> Unit<Motor> {
        Motor::new(
            2.0,
            self.next(),
            self.next(),
            self.next(),
            self.next(),
            self.next(),
            self.next(),
            self.next(),
        )
        .normalized()
    }
    fn line(&mut self) -> Line {
        Line::from_coeffs(core::array::from_fn(|_| self.next()))
    }
}

fn assert_close<M: Extensor<Slots = (), Coef = f32>>(got: &[M], want: &[M], what: &str) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let (g, w) = (g.coeffs().as_ref(), w.coeffs().as_ref());
        let scale = w.iter().fold(1.0f32, |m, x| m.max(x.abs()));
        for (a, b) in g.iter().zip(w) {
            assert!(
                (a - b).abs() <= 1e-4 * scale,
                "{what} [{}] element {i}: {g:?} vs {w:?}",
                batch::level_name(batch::level())
            );
        }
    }
}

const LENS: &[usize] = &[1, 2, 7, 8, 9, 16, 17, 31, 33, 100];

#[test]
fn batch_forms_match_the_fused_kernels() {
    let mut rng = Rng(7);
    for level in batch::levels() {
        batch::with_level(level, || {
            for &n in LENS {
                let ms: Vec<Unit<Motor>> = (0..n).map(|_| rng.motor()).collect();
                let ls: Vec<Point> = (0..n).map(|_| rng.point()).collect();
                let ps: Vec<Point> = (0..n).map(|_| rng.point()).collect();
                let bs: Vec<Line> = (0..n).map(|_| rng.line()).collect();
                let g = [Plane::new(0.1, 0.2, 1.0, -0.3)];
                let mut out = vec![Point::zero(); n];

                // Every argument per element, and a broadcast plane.
                shadow_of_moved_fused_batch(&ms, &ls, &g, &ps, &mut out);
                let want: Vec<Point> = (0..n)
                    .map(|i| shadow_of_moved_fused(ms[i], ls[i], g[0], ps[i]))
                    .collect();
                assert_close(&out, &want, &format!("shadow_of_moved n={n}"));

                // One motor for all points.
                compose_apply_fused_batch(&ms[..1], &ms, &ps, &mut out);
                let want: Vec<Point> = (0..n)
                    .map(|i| compose_apply_fused(ms[0], ms[i], ps[i]))
                    .collect();
                assert_close(&out, &want, &format!("compose_apply n={n}"));

                euclidean_fused_batch(&ps, &mut out);
                let want: Vec<Point> = ps.iter().map(|&p| euclidean_fused(p)).collect();
                assert_close(&out, &want, &format!("euclidean n={n}"));

                // Vectorized sines and cosines inside.
                screw_apply_fused_batch(&bs, &ps, &mut out);
                let want: Vec<Point> = (0..n).map(|i| screw_apply_fused(bs[i], ps[i])).collect();
                assert_close(&out, &want, &format!("screw_apply n={n}"));
            }
        });
    }
}

#[test]
#[should_panic(expected = "argument lengths")]
fn mismatched_lengths_panic() {
    let ps = [Point::<(), f32>::zero(); 3];
    let mut out = [Point::<(), f32>::zero(); 2];
    euclidean_fused_batch(&ps, &mut out);
}
