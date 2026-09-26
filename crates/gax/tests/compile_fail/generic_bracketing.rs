// For generic slot lists the compiler cannot see that `Cat<Cat<A, B>, C>` equals
// `Cat<A, Cat<B, C>>`; `gax::slots::reassoc` is the witness that converts between them.
use gax::pga3d::Point;
use gax::{Cat, Slots};

fn triple<A: Slots, B: Slots, C: Slots>(
    a: Point<A, f64>,
    b: Point<B, f64>,
    c: Point<C, f64>,
) -> Point<Cat<A, Cat<B, C>>, f64> {
    (a * b) * c
}

fn main() {
    let p = Point::xyz(1.0, 0.0, 0.0);
    let _ = triple(p, p, p);
}
