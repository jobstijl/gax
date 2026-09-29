//! The tracer agrees with the generic code on random programs: each is traced, its verified
//! program run in `f64` (`Tracer::eval`), and compared with the program run directly on `f64`,
//! at several random inputs. The `trace_programs` fuzz target does the same on arbitrary bytes.

#![cfg(all(feature = "trace", feature = "pga3d"))]

#[path = "support/random_program.rs"]
mod random_program;

use gax::pga3d::{Line, Motor, Point};
use gax::trace::{Sym, Tracer};
use random_program::{OPS, agrees, run};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

#[test]
fn traced_programs_agree_with_the_generic_code() {
    let mut rng = Rng(0x5eed_1234_abcd);
    let mut compared = 0;
    for k in 0..24 {
        let len = 1 + (rng.next() % 6) as usize;
        let ops: Vec<u8> = (0..3 * len).map(|_| (rng.next() % 256) as u8).collect();
        let clock = std::time::Instant::now();
        let mut t = Tracer::new();
        let o = ops.clone();
        t.kernel(
            "p",
            move |m: Motor<(), Sym>, p: Point<(), Sym>, q: Point<(), Sym>, l: Line<(), Sym>| {
                run(&o, m, p, q, l)
            },
        );
        if std::env::var_os("GAX_GEN_PROFILE").is_some() {
            eprintln!(
                "program {k} {:?}: traced in {:.2} s",
                ops.chunks(3).map(|c| c[0] % OPS).collect::<Vec<_>>(),
                clock.elapsed().as_secs_f64()
            );
        }
        for _ in 0..8 {
            let x: [f64; 22] = core::array::from_fn(|_| 2.0 * rng.unit());
            let got = t.eval("p", &[&x[..8], &x[8..12], &x[12..16], &x[16..]]);
            assert!(
                agrees(&ops, &x, &got, 1e-7),
                "program {k} {:?} at {x:?}: traced {got:?}",
                ops.chunks(3).map(|c| c[0] % OPS).collect::<Vec<_>>()
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 24 * 8);
}
