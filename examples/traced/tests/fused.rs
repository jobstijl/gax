//! The fused kernels agree with the generic functions they were traced from.

use gax::Unit;
use gax::pga3d::{Motor, Plane, Point, Rotor, Translator};
use gax_example_traced::*;

fn motor(angle: f64, axis: [f64; 3], t: [f64; 3]) -> Unit<Motor<(), f64>> {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let (s, c) = (angle / 2.0).sin_cos();
    let r = Rotor::new(c, -s * axis[0] / n, -s * axis[1] / n, -s * axis[2] / n);
    let tr = Translator::new(1.0, -t[0] / 2.0, -t[1] / 2.0, -t[2] / 2.0);
    // A translator times a rotor is a motor (the product's kind).
    Unit::new_unchecked(tr * r)
}

fn close(a: &[f64], b: &[f64]) {
    let scale = a.iter().chain(b).fold(1.0f64, |m, x| m.max(x.abs()));
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= 1e-12 * scale, "{a:?} vs {b:?}");
    }
}

#[test]
fn fused_matches_generic() {
    for k in 0..20 {
        let f = k as f64;
        let m = motor(0.3 + f, [1.0, f.sin(), 2.0], [f.cos(), 0.5, -1.0]);
        let m2 = motor(1.1 * f, [f, 1.0, -0.5], [0.2, f.sin(), 3.0]);
        let light = Point::new(0.5, 1.0, 4.0, 1.0);
        let ground = Plane::new(0.1, 0.2, 1.0, 0.3);
        let p = Point::new(f.sin(), f.cos(), 0.5 * f, 1.0 + 0.1 * f);
        close(
            &shadow_of_moved_fused(m, light, ground, p).c,
            &kernels::shadow_of_moved(m, light, ground, p).c,
        );
        close(
            &compose_apply_fused(m, m2, p).c,
            &kernels::compose_apply(m, m2, p).c,
        );
        close(&euclidean_fused(p).c, &kernels::euclidean(p).c);
        close(
            &shadow_on_floor_fused(m, light, p).c,
            &kernels::shadow_on_floor(m, light, p).c,
        );
    }
}

#[test]
fn generic_kernel_also_builds_the_map() {
    // The same generic function, given an open slot, returns the whole transformation.
    let m = motor(0.7, [1.0, 2.0, 3.0], [0.5, -1.0, 2.0]);
    let light = Point::new(0.5, 1.0, 4.0, 1.0);
    let ground = Plane::new(0.0, 0.0, 1.0, 0.0);
    let map: Point<(Point,), f64> = kernels::shadow_of_moved(m, light, ground, Point::slot());
    let p = Point::new(1.0, 2.0, 3.0, 1.0);
    close(&map.of(p).c, &shadow_of_moved_fused(m, light, ground, p).c);
}

/// `Motor::between`, fused: the same motor as the generic code, and it carries `a` onto `b`.
#[test]
fn between_fused_matches_generic() {
    use gax::pga3d::Line;
    for k in 0..20 {
        let f = k as f64 + 0.5;
        let (pa, pb) = (
            Point::new(f.sin(), 0.5, f.cos(), 1.0),
            Point::new(1.0, f.cos(), -0.5 * f, 2.0),
        );
        let (ga, gb) = (
            Plane::new(0.3, f.sin(), 1.0, 0.5 * f),
            Plane::new(f.cos(), 1.0, -0.2, 1.0),
        );
        let (la, lb) = (
            pa & Point::new(0.0, 1.0, f, 1.0),
            pb & Point::new(f, 0.0, 1.0, 1.0),
        );
        let cases: [(Vec<f64>, Vec<f64>); 3] = [
            (
                between_points_fused(pa, pb).c.to_vec(),
                kernels::between_points(pa, pb).c.to_vec(),
            ),
            (
                between_planes_fused(ga, gb).c.to_vec(),
                kernels::between_planes(ga, gb).c.to_vec(),
            ),
            (
                between_lines_fused(la, lb).c.to_vec(),
                kernels::between_lines(la, lb).c.to_vec(),
            ),
        ];
        for (fused, generic) in &cases {
            let scale = fused
                .iter()
                .chain(generic)
                .fold(1.0f64, |m, x| m.max(x.abs()));
            for (x, y) in fused.iter().zip(generic) {
                assert!((x - y).abs() <= 1e-9 * scale, "{fused:?} vs {generic:?}");
            }
        }
        // The motor carries each element onto the other (up to the elements' scale).
        let m = Unit::new_unchecked(between_points_fused(pa, pb));
        let moved = (m >> pa).unitized();
        close(&moved.c, &pb.unitized().c);
        let m = Unit::new_unchecked(between_planes_fused(ga, gb));
        let (moved, want) = ((m >> ga).normalized(), gb.normalized());
        close(&moved.c, &want.c);
        let m: Unit<Motor<(), f64>> = Unit::new_unchecked(between_lines_fused(la, lb));
        let moved: Line<(), f64> = (m >> la).normalized().into_inner();
        close(&moved.c, &lb.normalized().into_inner().c);
    }
}
