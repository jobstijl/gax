//! numga's `quadrics/quadrics`: an ellipsoid as a dual quadric in PGA3D, a linear map from
//! planes to points. It sends each tangent plane to its contact point; its inverse sends the
//! contact point back to the tangent plane (the polar reciprocity), and the plane at infinity to
//! the centre. Moving the ellipsoid by a motor moves the map: pull the plane into the body frame,
//! apply the body map, push the point back out, `M >> Q(M << plane)`. The animation sweeps the
//! tangent plane's normal around the ellipsoid in its body frame (left) and carries the
//! ellipsoid along a screw motion in the world (right); the world map's contact point and the
//! moved body contact point stay one point.

use gax::pga3d::{Motor, Point};
use gax_numga_examples::canvas::mix;
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Marker, Scene3, backdrop, caption, palette, run,
};

mod quadrics {
    use gax::pga3d::{Motor, Plane, Point};

    pub type P = Point<(), f64>;
    pub type Pl = Plane<(), f64>;
    /// A dual quadric: planes to points.
    pub type DualQuadric = Point<(Plane,), f64>;
    pub type M = gax::Unit<Motor<(), f64>>;

    /// The plane at infinity, which selects the affine chart.
    pub fn infinity() -> Pl {
        Plane::new(0.0, 0.0, 0.0, 1.0)
    }

    /// The ellipsoid with the given semi-axes (ideal points) about `centre`: the dyads of the
    /// semi-axes give the shape, and subtracting the centre's dyad closes the envelope of
    /// tangent planes.
    pub fn ellipsoid(semi_axes: &[P], centre: P) -> DualQuadric {
        let dyad = |p: P| p * (Plane::slot() & p);
        semi_axes
            .iter()
            .fold(DualQuadric::zero(), |q, a| q + dyad(*a))
            - dyad(centre)
    }

    /// The ellipsoid's tangent plane facing `normal` (its offset is ignored): through the
    /// centre with that normal, then shifted until it contains its own pole.
    pub fn support_plane(quadric: DualQuadric, normal: Pl) -> Pl {
        let inf = infinity();
        let normal = normal.normalized().into_inner();
        // The pole of the plane at infinity: the centre.
        let centre = quadric.of(inf);
        let through_centre = normal - inf * ((normal & centre).s() / (inf & centre).s());
        // Shift until `tangent ∨ quadric(tangent) == 0`; dividing by the centre's weight makes
        // the distance independent of the quadric's scale.
        let radius =
            (-(through_centre & quadric.of(through_centre)).s() / (inf & centre).s()).sqrt();
        through_centre - inf * radius
    }

    /// The ellipsoid with semi-axes 3, 2 and 1 along x, y and z, centred at the origin.
    pub fn body() -> DualQuadric {
        ellipsoid(
            &[
                Point::direction(3.0, 0.0, 0.0),
                Point::direction(0.0, 2.0, 0.0),
                Point::direction(0.0, 0.0, 1.0),
            ],
            Point::xyz(0.0, 0.0, 0.0),
        )
    }

    /// numga's motor `exp(0.5 xw + yw + 1.5 zw) exp(-π/12 xy)`: the translation by `(1, 2, 3)`
    /// after a turn of `π/6` about z.
    pub fn motor() -> M {
        Motor::translation(1.0, 2.0, 3.0)
            * Motor::rotation_about(0.0, 0.0, 1.0, core::f64::consts::FRAC_PI_6)
    }

    /// The tangent's normal: `(1, 2, 3)` swept around by the phase `t` (radians).
    pub fn normal(t: f64) -> Pl {
        let turn = Motor::rotation_about(0.0, 0.0, 1.0, t)
            * Motor::rotation_about(1.0, 0.0, 0.0, 0.6 * (2.0 * t).sin());
        turn >> Plane::new(1.0, 2.0, 3.0, 0.0)
    }

    /// Tangent, contact and the plane recovered from the contact by the inverse map.
    pub fn reciprocity(quadric: DualQuadric, normal: Pl) -> (Pl, P, Pl) {
        let tangent = support_plane(quadric, normal);
        let contact = quadric.of(tangent);
        (tangent, contact, quadric.solve(contact))
    }

    /// The body moved by `m`: the world map, the world tangent and contact from it, and the
    /// moved body contact.
    pub fn transport(m: M, normal: Pl) -> (DualQuadric, Pl, P, P) {
        let local = body();
        let tangent = support_plane(local, normal);
        let contact = local.of(tangent);
        // Pull the input plane into the body frame; push the output point into the world.
        let world = m >> local.of(m << Plane::slot());
        let world_tangent = m >> tangent;
        (world, world_tangent, world.of(world_tangent), m >> contact)
    }

    /// The ellipsoid sampled by sweeping the support normal over a sphere: `rows + 1` parallels
    /// of `cols + 1` contact points. The normal is the plane z turned toward x by the polar
    /// angle, then about z by the longitude.
    pub fn surface(quadric: DualQuadric, rows: usize, cols: usize) -> Vec<Vec<P>> {
        let pole = Plane::new(0.0, 0.0, 1.0, 0.0);
        (0..=rows)
            .map(|i| {
                let polar = core::f64::consts::PI * i as f64 / rows as f64;
                let tilt = Motor::rotation_about(0.0, 1.0, 0.0, polar);
                (0..=cols)
                    .map(|j| {
                        let longitude = core::f64::consts::TAU * j as f64 / cols as f64;
                        let n = (Motor::rotation_about(0.0, 0.0, 1.0, longitude) * tilt) >> pole;
                        quadric.of(support_plane(quadric, n))
                    })
                    .collect()
            })
            .collect()
    }
}

use quadrics::*;

/// One panel: the ellipsoid's wireframe, its centre, the tangent patch and the contact point.
fn panel(
    c: &mut Canvas,
    cam: Camera,
    quadric: DualQuadric,
    tangent: Pl,
    contact: P,
    ghost: Option<DualQuadric>,
    moved: Option<P>,
) {
    let mut s = Scene3::new(cam);
    if let Some(g) = ghost {
        for row in surface(g, 12, 32) {
            s.polyline(&row, 1.0, palette::grid(), 0.6);
        }
    }
    let rows = surface(quadric, 16, 48);
    for row in &rows {
        s.polyline(row, 1.0, palette::sky(), 0.55);
    }
    for j in (0..rows[0].len()).step_by(3) {
        let meridian: Vec<P> = rows.iter().map(|r| r[j]).collect();
        s.polyline(&meridian, 1.0, palette::sky(), 0.55);
    }
    s.dot(quadric.of(infinity()), Marker::Dot, 9.0, palette::sky());
    // A square of the tangent plane about the contact point: a square of the plane z = 0,
    // turned from z to the tangent's normal and carried to the contact point.
    let normal = Point::direction(tangent.e1(), tangent.e2(), tangent.e3());
    let normal = normal.gp(1.0 / normal.ideal_norm());
    let frame = Motor::between(Point::xyz(0.0, 0.0, 0.0), contact)
        * Motor::rotation_between(Point::direction(0.0, 0.0, 1.0), normal);
    let corner = |u: f64, v: f64| frame >> Point::xyz(1.3 * u, 1.3 * v, 0.0);
    s.quad(
        corner(-1.0, -1.0),
        corner(1.0, -1.0),
        corner(1.0, 1.0),
        corner(-1.0, 1.0),
        palette::red(),
        0.35,
    );
    s.arrow(contact, normal.gp(1.2), 2.0, 9.0, palette::orange());
    s.dot(contact, Marker::Dot, 11.0, palette::red());
    if let Some(m) = moved {
        s.dot(m, Marker::Ring, 20.0, palette::yellow());
    }
    s.draw(c);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.width, c.height);
    let phase = f64::from(t) / 12.0 * core::f64::consts::TAU;
    let azimuth = 0.66 + 0.35 * (phase.sin() as f32);
    let half = w / 2;
    let normal = normal(phase);

    // The body frame, on the left.
    let mut left = Canvas::new(half, h);
    backdrop(&mut left);
    let cam = Camera::orbit(
        half,
        h,
        Point::xyz(0.0, 0.0, 0.8),
        12.5,
        azimuth,
        0.4,
        Lens::Perspective(0.6),
    );
    let (tangent, contact, _) = reciprocity(body(), normal);
    panel(&mut left, cam, body(), tangent, contact, None, None);
    c.blit(&left, 0, 0);

    // The world frame, on the right: the screw from the identity to numga's motor and back.
    let s = 0.5 - 0.5 * (phase).cos();
    let m = Motor::interpolate(Motor::translation(0.0, 0.0, 0.0), motor(), s);
    let (world, world_tangent, world_contact, moved) = transport(m, normal);
    let mut right = Canvas::new(w - half, h);
    backdrop(&mut right);
    let cam = Camera::orbit(
        w - half,
        h,
        Point::xyz(0.5, 1.0, 2.4),
        17.0,
        azimuth,
        0.4,
        Lens::Perspective(0.6),
    );
    panel(
        &mut right,
        cam,
        world,
        world_tangent,
        world_contact,
        Some(body()),
        Some(moved),
    );
    c.blit(&right, half, 0);
    c.line(
        [half as f32, 60.0],
        [half as f32, h as f32 - 20.0],
        1.0,
        palette::grid(),
        1.0,
    );

    let (wf, hf) = (w as f32, h as f32);
    let label = |c: &mut Canvas, x: f32, s: &str| {
        c.text(s, x, hf - 18.0, 11.0, palette::ink(), Align::Center);
    };
    label(c, wf * 0.25, "BODY FRAME: TANGENT -> CONTACT -> TANGENT");
    label(c, wf * 0.75, "WORLD FRAME: M >> Q(M << PLANE)");
    let key = [
        ("CENTRE Q(INFINITY)", palette::sky()),
        ("CONTACT Q(TANGENT)", palette::red()),
        ("M >> CONTACT", palette::yellow()),
    ];
    for (i, (s, col)) in key.iter().enumerate() {
        let y = hf - 100.0 + 18.0 * i as f32;
        c.disk([wf - 210.0, y - 4.0], 4.0, *col, 1.0);
        c.text(
            s,
            wf - 200.0,
            y,
            11.0,
            mix(palette::ink(), *col, 0.3),
            Align::Left,
        );
    }
    caption(
        c,
        "AN ELLIPSOID AS A MAP FROM PLANES TO POINTS",
        "Q = SUM A (PLANE & A) - C (PLANE & C) IN PGA3D, MOVED BY A MOTOR",
    );
}

fn main() {
    run(Anim::new("quadrics", 12.0).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::quadrics::*;
    use gax::ApproxEq;
    use gax::pga3d::{Line, Motor, Plane};

    /// The motor is numga's product of exponentials (`xw = e1 e0 = -e01`; numga's `xy` is
    /// gax's `e12`).
    #[test]
    fn the_motor_is_numga_s() {
        let numga = Line::from_coeffs([0.0, 0.0, 0.0, -0.5, -1.0, -1.5]).exp()
            * Line::from_coeffs([0.0, 0.0, -core::f64::consts::PI / 12.0, 0.0, 0.0, 0.0]).exp();
        assert!(motor().into_inner().approx_eq(&numga.into_inner(), 1e-12));
    }

    /// Every support plane contains its own contact point and keeps the requested
    /// orientation: its Euclidean part is the unit normal.
    #[test]
    fn support_planes_are_tangent_and_face_their_normal() {
        let dual = body();
        let normals = [
            Plane::new(1.0, 0.0, 0.0, 0.0),
            Plane::new(0.0, 1.0, 0.0, 0.0),
            Plane::new(-1.0, 2.0, 0.5, 0.0),
            Plane::new(0.3, -0.4, 2.0, 0.0),
        ];
        for n in normals {
            let tangent = support_plane(dual, n);
            assert!((tangent & dual.of(tangent)).s().abs() < 1e-12);
            let unit = n.normalized().into_inner();
            assert!(((tangent | unit).s() - (unit | unit).s()).abs() < 1e-12);
        }
    }

    /// The scenario's checks: incidence is reciprocal, the inverse map recovers the tangent
    /// from the contact point.
    #[test]
    fn polar_reciprocity() {
        let (tangent, contact, recovered) = reciprocity(body(), Plane::new(1.0, 2.0, 3.0, 0.0));
        assert!((tangent & contact).s().abs() < 1e-12);
        assert!((recovered & contact).s().abs() < 1e-12);
        for k in 0..12 {
            let (_, contact, recovered) = reciprocity(body(), normal(0.5 * k as f64));
            assert!((recovered & contact).s().abs() < 1e-12);
        }
    }

    /// The scenario's check: the world map's contact point is the moved body contact point
    /// (the line joining them vanishes), along the whole screw.
    #[test]
    fn motor_transport() {
        let (_, _, world_contact, moved) = transport(motor(), Plane::new(1.0, 2.0, 3.0, 0.0));
        let joined = world_contact & moved;
        assert!((joined | joined).s().abs() < 1e-20);
        for k in 0..=8 {
            let m = Motor::interpolate(Motor::translation(0.0, 0.0, 0.0), motor(), k as f64 / 8.0);
            let (_, _, world_contact, moved) = transport(m, normal(0.7 * k as f64));
            let joined = world_contact & moved;
            assert!((joined | joined).s().abs() < 1e-20, "{joined:?}");
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
