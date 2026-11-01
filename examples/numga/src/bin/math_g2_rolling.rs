//! numga's `math/g2_rolling`: a ball rolled over a sphere three times its radius, its route
//! planned by the product of split octonions, in the geometric algebra of space.
//!
//! A configuration of the ball, its contact direction and its attitude, is a multivector of
//! space without scalar part: its even part is the half turn in the plane through the centre
//! square to the contact direction, its odd part the attitude followed by the reflection in that
//! plane. The split-octonion product multiplies even parts by the geometric product and lets odd
//! parts enter in swapped order, reversed for conjugation (the doubling of `math_octonions`,
//! with the signs that make it split). Two configurations are one roll apart (the ball rolling
//! without slipping or twisting along a great circle) exactly when their product vanishes: the
//! product of two configurations two rolls apart is the waypoint between them, and the kernel of
//! multiplication by a configuration holds every configuration one roll away, so two kernels and
//! one product plan a route of three rolls back to the same contact with a new attitude. Maps
//! preserving the product, the split form of G₂, carry routes to routes: a derivation built from
//! multiplications, exponentiated (its cube vanishes), moves a route to another that still rolls.
//!
//! The contact direction is intrinsic and the half turn extrinsic, so identifying the two takes
//! a sense of turning, an intrinsic scalar of one, `handedness`; the other sense negates every
//! odd part, which preserves the product.
//!
//! The animation draws numga's three scenes side by side, ray traced on the canvas: the ball
//! rolling two rolls through the product of its ends, three rolls back to its starting contact,
//! and that route moved by split G₂, with the routes inked on the sphere.

use gax_numga_examples::{
    Anim, Camera, Canvas, Lens, Light, ORIGIN3, Point2, Point3, Rect, backdrop, caption,
    label_under, light, palette, run_with,
};

gax::algebra! {
    algebra r30 "Euclidean 3D, R(3,0): numga's VGA3D, whose multivectors double into the split octonions.";
    basis e1 = 1, e2 = 1, e3 = 1;
    kind Scalar = [1];
    versor Vector = [e1, e2, e3];
    kind Bivector = [e23, e31, e12];
    versor Rotor = [1, e23, e31, e12];
    kind Odd = [e1, e2, e3, e123];
    kind Nonscalar = [e1, e2, e3, e23, e31, e12, e123];
    kind Multivector = [1, e1, e2, e3, e23, e31, e12, e123];
}

/// Rolling by split octonions.
mod rolling {
    use super::r30::{Bivector, Multivector, Nonscalar, Odd, Rotor, Scalar, Vector, intrinsic};

    /// A contact direction: intrinsic, a point of the unit sphere.
    pub type Point = intrinsic::Bivector;
    /// A plane through the centre.
    pub type Plane = Vector;
    /// A configuration: no scalar part.
    pub type State = Nonscalar;
    /// A split octonion: any multivector of space.
    pub type Octonion = Multivector;
    /// A map on configurations.
    pub type StateMap = Nonscalar<(Nonscalar,)>;
    /// A map on split octonions.
    pub type OctonionMap = Multivector<(Multivector,)>;

    /// The sense of turning that identifies a direction with the plane square to it.
    pub fn handedness() -> intrinsic::Scalar {
        intrinsic::Scalar::new(1.0)
    }

    /// The split norm's bilinear form, contact part minus attitude part: the scalar part of the
    /// Clifford conjugate times the other.
    #[cfg_attr(not(test), allow(dead_code))] // numga's checks of the norm and the ends
    pub fn pairing(first: Octonion, second: Octonion) -> f64 {
        first.conjugate().scalar_product(second).s()
    }

    /// The split-octonion product: the even parts multiply by the geometric product, the odd
    /// parts enter in the opposite order, reversal supplying their conjugation.
    pub fn multiply(first: Octonion, second: Octonion) -> Octonion {
        let (a, b) = (first.cast::<Rotor>(), first.cast::<Odd>());
        let (c, d) = (second.cast::<Rotor>(), second.cast::<Odd>());
        (a * c + d * b.reverse()).cast::<Multivector>()
            + (a.reverse() * d + c * b).cast::<Multivector>()
    }

    /// A configuration as a split octonion.
    pub fn octonion(s: State) -> Octonion {
        s.cast::<Multivector>()
    }

    /// The product of two configurations.
    pub fn product(first: State, second: State) -> Octonion {
        multiply(octonion(first), octonion(second))
    }

    /// The half turn in the plane square to a contact direction, of the attitude's own
    /// extrinsic kind, so that the two multiply as quaternions.
    pub fn half_turn(contact: Point) -> Bivector {
        handedness() * contact
    }

    /// A configuration's contact direction, of the length its half turn has.
    pub fn contact_of(state: State) -> Point {
        handedness() * state.cast::<Bivector>()
    }

    /// A unit contact direction and an attitude as one null configuration: the half turn less
    /// the reflection in the plane square to the contact, through the centre, after the
    /// attitude.
    pub fn encode(contact: Point, attitude: Rotor) -> State {
        let mirror: Plane = contact.hodge();
        half_turn(contact).cast::<Nonscalar>() - (mirror * attitude).cast::<Nonscalar>()
    }

    /// The representative of a configuration whose contact direction has unit length.
    pub fn normalize(state: State) -> State {
        state.gp(1.0 / state.cast::<Bivector>().norm())
    }

    /// The representative whose contact lies on the positive side of the plane `hemisphere`.
    pub fn lift(state: State, hemisphere: Plane) -> State {
        normalize(state.gp((hemisphere & contact_of(state)).s()))
    }

    /// The contact direction and attitude of a configuration.
    pub fn decode(state: State) -> (Point, Rotor) {
        let state = normalize(state);
        let contact = contact_of(state);
        let mirror: Plane = contact.hodge();
        let attitude = (mirror.inverse() * state.cast::<Odd>()).cast::<Rotor>();
        (contact, -attitude)
    }

    /// The basis configuration `j`.
    pub fn basis(j: usize) -> State {
        Nonscalar::from_coeffs(core::array::from_fn(|i| if i == j { 1.0 } else { 0.0 }))
    }

    /// The basis split octonion `j`.
    pub fn unit(j: usize) -> Octonion {
        Multivector::from_coeffs(core::array::from_fn(|i| if i == j { 1.0 } else { 0.0 }))
    }

    /// Multiplication by a configuration on the left, as a map on configurations.
    pub fn left_action(state: State) -> Multivector<(Nonscalar,)> {
        Multivector::<(Nonscalar,)>::from_images(core::array::from_fn(|j| product(state, basis(j))))
    }

    /// The projection onto the configurations one roll away from this one: the identity less
    /// the projection onto the row space of its multiplication.
    pub fn annihilator(state: State) -> StateMap {
        let action = left_action(state);
        Nonscalar::slot() - action.pinv().of(action)
    }

    /// Two intermediate configurations between ends that need three rolls, the first as near as
    /// the rolls allow to a contact at `preferred`: one roll from the start, square to the finish
    /// in the split pairing (so at most two rolls from it), and the product of the two the second.
    pub fn three_rolls(start: State, finish: State, preferred: Point) -> (State, State) {
        let reachable = annihilator(start);
        let remaining: Scalar<(Nonscalar,)> = finish.conjugate().scalar_product(reachable);
        let preference = half_turn(preferred).cast::<Nonscalar>();
        let first = reachable.of(preference - remaining.lstsq(remaining.of(preference)));
        let second = product(first, finish).cast::<Nonscalar>();
        (normalize(first), normalize(second))
    }

    /// Multiplication by `a` on the left and on the right, as maps on split octonions.
    fn left(a: Octonion) -> OctonionMap {
        OctonionMap::from_images(core::array::from_fn(|j| multiply(a, unit(j))))
    }

    fn right(a: Octonion) -> OctonionMap {
        OctonionMap::from_images(core::array::from_fn(|j| multiply(unit(j), a)))
    }

    /// An infinitesimal product-preserving map, assembled from multiplication actions and
    /// restricted to the configurations.
    pub fn derivation(first: Octonion, second: Octonion) -> StateMap {
        let (l1, l2, r1, r2) = (left(first), left(second), right(first), right(second));
        let action = l1.of(l2) - l2.of(l1) + l1.of(r2) - r2.of(l1) + r1.of(r2) - r2.of(r1);
        action
            .of(Nonscalar::slot().cast::<Multivector>())
            .cast::<Nonscalar>()
    }

    /// The symmetry `exp(amount generator)`, ended at the second power (the third vanishes).
    pub fn symmetry(generator: StateMap, amount: f64) -> StateMap {
        Nonscalar::slot() + generator.gp(amount) + generator.of(generator).gp(amount * amount / 2.0)
    }

    /// The configurations along the projective line between two whose product vanishes: one
    /// roll.
    pub fn roll(start: State, finish: State, fraction: f64) -> State {
        normalize(start.gp(1.0 - fraction) + finish.gp(fraction))
    }

    /// The contact on top, `z`, and on the side, `x`.
    pub fn top() -> Point {
        Vector::new(0.0, 0.0, 1.0).hodge()
    }

    pub fn side() -> Point {
        Vector::new(1.0, 0.0, 0.0).hodge()
    }

    /// The attitude turned by `angle` in the horizontal plane, numga's `exp(xy angle / 2)`.
    pub fn turned(angle: f64) -> Rotor {
        Bivector::new(0.0, 0.0, angle / 2.0).exp().into_inner()
    }

    /// The identity attitude.
    pub fn upright() -> Rotor {
        Rotor::new(1.0, 0.0, 0.0, 0.0)
    }

    /// numga's attitude angle between the ends of each route.
    pub const ATTITUDE_ANGLE: f64 = -2.4;
    /// How far the symmetry moves the route.
    pub const SYMMETRY_AMOUNT: f64 = 0.12;

    /// numga's two-roll route: the product of the ends supplies the waypoint.
    pub fn two_roll_route() -> Vec<State> {
        let start = encode(top(), upright());
        let finish = encode(side(), turned(-ATTITUDE_ANGLE));
        let waypoint = lift(
            product(start, finish).cast::<Nonscalar>(),
            Vector::new(1.0, 0.0, 0.0),
        );
        vec![start, waypoint, finish]
    }

    /// numga's three-roll route back to the top with a turned attitude, its waypoints on the
    /// positive side of `x`.
    pub fn three_roll_route() -> Vec<State> {
        let start = encode(top(), upright());
        let finish = encode(top(), turned(-ATTITUDE_ANGLE));
        let hemisphere = Vector::new(1.0, 0.0, 0.0);
        let (first, second) = three_rolls(start, finish, hemisphere.hodge());
        vec![
            start,
            lift(first, hemisphere),
            lift(second, hemisphere),
            finish,
        ]
    }

    /// The three-roll route moved by split G₂: the derivation from its start and `x`,
    /// exponentiated.
    pub fn moved_route() -> Vec<State> {
        let route = three_roll_route();
        let x = Vector::new(1.0, 0.0, 0.0).cast::<Multivector>();
        let map = symmetry(derivation(octonion(route[0]), x), SYMMETRY_AMOUNT);
        route.into_iter().map(|s| map.of(s)).collect()
    }
}

use r30::{Rotor, Vector};
use rolling::{Point, State};

/// The loop's seconds.
const SECONDS: f32 = 9.0;
/// The share of the loop each route holds still at each end.
const HOLD: f32 = 0.1;
/// The radii of the sphere and the ball.
const SPHERE: f64 = 3.0;
const BALL: f64 = 1.0;
/// The samples along each drawn leg.
const SAMPLES: usize = 120;
/// The view's elevation and azimuth, in degrees.
const ELEVATION: f32 = 35.0;
const AZIMUTH: f32 = -25.0;
/// The legs' colours.
const LEGS: [Light; 3] = [palette::SKY, palette::ORANGE, palette::PURPLE];

/// A route ready to draw: its stops and each leg's contacts on the sphere.
struct Route {
    stops: Vec<State>,
    legs: Vec<Vec<Vector>>,
}

impl Route {
    fn new(stops: Vec<State>) -> Route {
        let legs = stops
            .windows(2)
            .map(|pair| {
                (0..SAMPLES)
                    .map(|k| {
                        let f = k as f64 / (SAMPLES - 1) as f64;
                        direction(rolling::decode(rolling::roll(pair[0], pair[1], f)).0)
                    })
                    .collect()
            })
            .collect();
        Route { stops, legs }
    }

    /// The contact and attitude a share of the way through the loop: held at the ends, rolling
    /// leg by leg between.
    fn at(&self, share: f32) -> (Point, Rotor) {
        let legs = self.stops.len() - 1;
        let moving = ((share - HOLD) / (1.0 - 2.0 * HOLD)).clamp(0.0, 1.0) * legs as f32;
        let leg = (moving as usize).min(legs - 1);
        let fraction = f64::from(moving - leg as f32);
        rolling::decode(self.stops[leg].gp(1.0 - fraction) + self.stops[leg + 1].gp(fraction))
    }
}

/// A contact direction as the unit vector it points along.
fn direction(contact: Point) -> Vector {
    contact.hodge()
}

/// A vector of space as a point of the drawing's world.
fn world(v: Vector) -> Point3 {
    Point3::xyz(v.e1() as f32, v.e2() as f32, v.e3() as f32)
}

/// A drawing ray's origin and direction as vectors of space.
fn ray(cam: &Camera, q: Point2) -> (Vector, Vector) {
    let (o, d) = cam.ray(q);
    let [x, y, z] = o.to_euclidean();
    let at = Vector::new(f64::from(x), f64::from(y), f64::from(z));
    let along = Vector::new(
        f64::from(d.e032()),
        f64::from(d.e013()),
        f64::from(d.e021()),
    );
    (at, along)
}

/// How far along a unit direction a ray from `start` first meets the ball of `radius` about
/// `centre`, if it does: the nearer root of the squared distance less the squared radius.
fn meet(start: Vector, along: Vector, centre: Vector, radius: f64) -> Option<f64> {
    let offset = start - centre;
    let b = (along | offset).s();
    let c = offset.dot(offset).s() - radius * radius;
    let (middle, half) = gax_numga_examples::measure::roots(1.0, b, c)?;
    Some(middle - half).filter(|t| *t > 0.0)
}

/// The octants' colours, numga's, as lights.
fn octant(k: usize) -> Light {
    const SRGB: [[f32; 3]; 8] = [
        [0.19, 0.44, 0.66],
        [0.93, 0.48, 0.19],
        [0.94, 0.75, 0.32],
        [0.19, 0.64, 0.61],
        [0.81, 0.35, 0.39],
        [0.40, 0.68, 0.84],
        [0.61, 0.47, 0.69],
        [0.91, 0.89, 0.76],
    ];
    let [r, g, b] = SRGB[k];
    Light::from_srgb(r, g, b, 1.0)
}

/// One scene: the sphere, the route inked on its near side, and the ball at the loop's share,
/// its octants painted in its own frame, lit from up and to the left.
fn scene_panel(c: &mut Canvas, rect: Rect, route: &Route, share: f32, title: &str) {
    let half = ((SPHERE + 2.0 * BALL) * 1.07) as f32;
    let cam = Camera::orbit(
        rect,
        ORIGIN3,
        20.0,
        AZIMUTH.to_radians(),
        ELEVATION.to_radians(),
        Lens::Parallel(half),
    );
    let (_, view) = ray(&cam, rect.centre());
    let lamp = {
        let towards = cam.pose >> Point3::direction(-0.40, 0.48, -0.78);
        Vector::new(
            f64::from(towards.e032()),
            f64::from(towards.e013()),
            f64::from(towards.e021()),
        )
        .normalized()
    };
    let lit =
        |normal: Vector, floor: f64| (floor + (1.0 - floor) * (normal | lamp).s().max(0.0)) as f32;
    let surface = light(0.3, 0.42, 1.0, 0.35);
    c.clip(rect);
    c.shade(2, |q| {
        let (start, along) = ray(&cam, q);
        let t = meet(start, along, Vector::zero(), SPHERE)?;
        let normal = (start + along.gp(t)).gp(1.0 / SPHERE);
        Some(surface.faded(lit(normal, 0.25)))
    });
    // The route on the sphere's near side, and its ends.
    for (leg, points) in route.legs.iter().enumerate() {
        let near: Vec<Point2> = points
            .iter()
            .filter(|p| (**p | view).s() < 0.0)
            .filter_map(|p| cam.px(world(p.gp(SPHERE))))
            .collect();
        c.polyline(&near, 2.4, LEGS[leg], false);
        for end in [points[0], points[SAMPLES - 1]] {
            if (end | view).s() < 0.0
                && let Some(px) = cam.px(world(end.gp(SPHERE)))
            {
                c.disk(px, 4.0, palette::INK);
                c.disk(px, 3.0, LEGS[leg]);
            }
        }
    }
    let (contact, attitude) = route.at(share);
    let centre = direction(contact).gp(SPHERE + BALL);
    let seams = light(0.02, 0.03, 0.08, 1.0);
    c.shade(2, |q| {
        let (start, along) = ray(&cam, q);
        let t = meet(start, along, centre, BALL)?;
        if meet(start, along, Vector::zero(), SPHERE).is_some_and(|s| s < t) {
            return None;
        }
        let normal = (start + along.gp(t) - centre).gp(1.0 / BALL);
        // The normal in the ball's own frame: its coordinates' signs pick the octant, and the
        // seams lie where one is small.
        let body = attitude << normal;
        let [x, y, z] = [body.e1(), body.e2(), body.e3()];
        let k = usize::from(x >= 0.0) + 2 * usize::from(y >= 0.0) + 4 * usize::from(z >= 0.0);
        let seam = (1.0 - x.abs().min(y.abs()).min(z.abs()) / 0.035).clamp(0.0, 1.0) as f32;
        Some(octant(k).mix_light(seams, seam).faded(lit(normal, 0.47)))
    });
    c.unclip();
    label_under(c, rect, title, 22.0, 0.95);
}

/// The three routes, computed once.
struct Scenes {
    routes: [Route; 3],
}

impl Scenes {
    fn new() -> Scenes {
        Scenes {
            routes: [
                Route::new(rolling::two_roll_route()),
                Route::new(rolling::three_roll_route()),
                Route::new(rolling::moved_route()),
            ],
        }
    }
}

fn draw(c: &mut Canvas, scenes: &Scenes, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let share = t / SECONDS;
    let titles = [
        "TWO ROLLS: THE PRODUCT OF THE ENDS",
        "THREE ROLLS BACK TO THE TOP, TURNED",
        "THE SAME ROUTE MOVED BY SPLIT G2",
    ];
    for (k, (route, title)) in scenes.routes.iter().zip(titles).enumerate() {
        let cell = screen.part(0.01 + 0.33 * k as f32, 0.16, 0.33 + 0.33 * k as f32, 0.92);
        scene_panel(c, cell.inset(4.0, 0.0, 4.0, 0.0), route, share, title);
    }
    caption(
        c,
        "ROLLING A BALL BY SPLIT OCTONIONS: G2 (R(3,0))",
        "ONE ROLL APART WHEN THE PRODUCT VANISHES; THE PRODUCT, THE KERNEL AND SPLIT G2 PLAN AND MOVE ROUTES",
    );
}

fn main() {
    run_with(
        Anim::new("g2_rolling", SECONDS).size(960, 540),
        Scenes::new,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::r30::{Bivector, Multivector, Nonscalar, Rotor, Vector};
    use super::rolling::*;
    use gax::ApproxEq;
    use gax_numga_examples::rng::{Draw, rng};

    /// Every coefficient of a split octonion within `tolerance` of zero.
    fn vanishes(x: Octonion, tolerance: f64) -> bool {
        x.c.iter().all(|c| c.abs() < tolerance)
    }

    /// A split octonion of normal coefficients.
    fn random(r: &mut impl Draw) -> Octonion {
        Multivector::from_coeffs(core::array::from_fn(|_| r.normal()))
    }

    /// numga's first test: the split product preserves its norm and is alternative. numga's
    /// eight pairs are drawn from numpy's stream; these from a seeded one.
    #[test]
    fn split_product_is_alternative_and_preserves_its_norm() {
        let mut r = rng(8);
        for _ in 0..8 {
            let (a, b) = (random(&mut r), random(&mut r));
            let p = multiply(a, b);
            let error = pairing(p, p) - pairing(a, a) * pairing(b, b);
            assert!(error.abs() < 1e-10, "{error}");
            let twice = multiply(multiply(a, a), b) - multiply(a, multiply(a, b));
            assert!(vanishes(twice, 1e-10));
        }
    }

    /// numga's second test: ends two rolls apart pair to zero, their product (nonzero) is the
    /// waypoint, one roll from each end; and the finish decodes to its contact and attitude.
    #[test]
    fn product_finds_the_two_roll_waypoint() {
        let start = encode(top(), upright());
        let finish = encode(side(), turned(1.2));
        let waypoint = normalize(product(start, finish).cast::<Nonscalar>());
        assert!(pairing(octonion(start), octonion(finish)).abs() < 1e-12);
        let size: f64 = product(start, finish).c.iter().map(|c| c * c).sum();
        assert!(size > 0.01);
        assert!(vanishes(product(start, waypoint), 1e-12));
        assert!(vanishes(product(waypoint, finish), 1e-12));
        let (contact, attitude) = decode(finish);
        assert!(contact.approx_eq(&side(), 1e-12));
        assert!(attitude.approx_eq(&turned(1.2), 1e-12));
    }

    /// numga's third test: ends at the same contact are no two rolls apart; the configurations
    /// one roll from the start are a projection of rank three, killed by the start; the route
    /// of three rolls has every step one roll, and its stops unit contacts and attitudes.
    #[test]
    fn kernel_and_product_find_three_rolls_for_independent_endpoints() {
        let start = encode(top(), upright());
        let finish = encode(top(), turned(1.2));
        let (first, second) = three_rolls(start, finish, side());
        let reachable = annihilator(start);
        assert!(pairing(octonion(start), octonion(finish)).abs() > 0.1);
        let rank = reachable.svdvals().iter().filter(|s| **s > 1e-10).count();
        assert_eq!(rank, 3);
        for j in 0..7 {
            assert!(vanishes(product(start, reachable.of(basis(j))), 1e-12));
        }
        assert!(reachable.of(reachable).approx_eq(&reachable, 1e-12));
        for (a, b) in [(start, first), (first, second), (second, finish)] {
            assert!(vanishes(product(a, b), 1e-12));
        }
        for s in [start, first, second, finish] {
            let (contact, attitude) = decode(s);
            assert!((contact.norm_squared() - 1.0).abs() < 1e-12);
            assert!((attitude.norm_squared() - 1.0).abs() < 1e-12);
        }
    }

    /// numga's fourth test: the derivation's cube vanishes; its exponential preserves the
    /// product of any two configurations (the product's scalar kept); and both the route and the
    /// moved route roll physically: the centre, at radius four, moves as the rotation carries
    /// the contact point (no slip), and the rotation has no part about the contact (no twist).
    #[test]
    fn split_g2_deformation_preserves_products_and_physical_rolling() {
        let start = encode(top(), upright());
        let finish = encode(top(), turned(1.2));
        let (first, second) = three_rolls(start, finish, side());
        let x = Vector::new(1.0, 0.0, 0.0);
        let generator = derivation(octonion(start), x.cast::<Multivector>());
        let map = symmetry(generator, 0.12);
        let cube = generator.of(generator.of(generator));
        assert!(cube.approx_eq(&StateMap::zero(), 1e-12));

        let mut r = rng(17);
        for _ in 0..7 {
            let left = Nonscalar::from_coeffs(core::array::from_fn(|_| r.normal()));
            let right = Nonscalar::from_coeffs(core::array::from_fn(|_| r.normal()));
            let p = product(left, right);
            let mapped = p.cast::<super::r30::Scalar>().cast::<Multivector>()
                + octonion(map.of(p.cast::<Nonscalar>()));
            let moved = product(map.of(left), map.of(right));
            assert!(vanishes(mapped - moved, 1e-11));
        }

        let step = 1e-5;
        for transform in [Nonscalar::slot(), map] {
            let stops = [start, first, lift(second, x), finish].map(|s| transform.of(s));
            for pair in stops.windows(2) {
                for k in 0..5 {
                    let f = 0.15 + 0.7 * k as f64 / 4.0;
                    let (contact, attitude) = decode(roll(pair[0], pair[1], f));
                    let (c0, a0) = decode(roll(pair[0], pair[1], f - step));
                    let (c1, a1) = decode(roll(pair[0], pair[1], f + step));
                    let contact_rate = (c1 - c0).gp(1.0 / (2.0 * step));
                    let attitude_rate = (a1 - a0).gp(1.0 / (2.0 * step));
                    let rotation = (attitude_rate * attitude.reverse())
                        .cast::<Bivector>()
                        .gp(2.0);
                    let slip = contact_rate.gp(4.0) - rotation.commutator(contact);
                    let twist = rotation.scalar_product(contact).s();
                    assert!(slip.c.iter().all(|c| c.abs() < 1e-5), "{slip:?}");
                    assert!(twist.abs() < 1e-5, "{twist}");
                }
            }
        }
    }

    /// numga's fifth test: lifting routes commutes with mirroring: the mirrored ends' waypoints
    /// decode to the mirrored contacts and attitudes.
    #[test]
    fn physical_route_lifts_commute_with_mirroring() {
        let mirror = Vector::new(1.0, 1.0, 1.0).unit();
        let hemisphere = Vector::new(1.0, 0.0, 0.0);
        let finish_attitude = turned(1.2);
        let route = |m: &dyn Fn(Point) -> Point, r: &dyn Fn(Rotor) -> Rotor, h: Vector| {
            let start = encode(m(top()), r(upright()));
            let two = encode(m(side()), r(finish_attitude));
            let three = encode(m(top()), r(finish_attitude));
            let middle = lift(product(start, two).cast::<Nonscalar>(), h);
            let (first, second) = three_rolls(start, three, h.hodge());
            [middle, first, lift(second, h)].map(decode)
        };
        let plain = route(&|p| p, &|r| r, hemisphere);
        let mirrored = route(&|p| mirror >> p, &|r| mirror >> r, mirror >> hemisphere);
        for ((c, a), (mc, ma)) in plain.iter().zip(mirrored) {
            assert!(mc.approx_eq(&(mirror >> *c), 1e-11));
            assert!(ma.approx_eq(&(mirror >> *a), 1e-11));
        }
    }

    /// The scenario's checks: neighbouring stops of every route multiply to zero; and a frame
    /// draws.
    #[test]
    fn every_route_rolls_and_a_frame_draws() {
        for route in [two_roll_route(), three_roll_route(), moved_route()] {
            for pair in route.windows(2) {
                assert!(vanishes(product(pair[0], pair[1]), 1e-10));
            }
        }
        let scenes = super::Scenes::new();
        gax_numga_examples::app::assert_draws_with(super::draw, &scenes, 4.0);
    }
}
