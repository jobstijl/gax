//! numga's `electromagnetism/fresnel`: light's wave surface in a crystal, from extensors in
//! spacetime algebra (STA, `t² = 1`, `x² = y² = z² = -1`).
//!
//! A material is a map from the field, a bivector, to its excitation, built with the field left
//! open: `dielectric` weights each of the observer's electric planes `axis ∧ t` by its
//! permittivity, and the magnetic planes, which the observer's sandwich `t F t` keeps, by one
//! reluctivity, then takes the Hodge dual (numga's `dual`; gax's `dual` is the metric-free
//! complement, written `hodge` here as in the other STA ports). A potential `A` gives the field
//! `k ∧ A`, and the potential map `A ↦ (k ∧ χ(k ∧ A))*` is what is left of Maxwell's other
//! equation, a map `Vector <- Vector`. It always loses the potential along `k`; a wave is a second
//! potential it loses, and then it sends every volume to nothing: its outermorphism on
//! trivectors vanishes. The adjugate of that action, the map on vectors with
//! `adj(c) & v = c & volumes(v)` (a pairing solve, as gax has no adjugate method), holds two
//! factors of `k` and one number more: the Fresnel quartic, zero where light can travel.
//!
//! Along each spatial direction at a fixed frequency the quartic of a crystal at rest is a
//! quadratic in the squared radius, fixed by three samples, and its two roots are the two
//! sheets of the wave surface, one per polarization. The animation turns vacuum (one sphere)
//! into a biaxial crystal and back: the two sheets in space, and their sections through the
//! `xz` plane, which touch at the optic axes.

use gax_numga_examples::scene3::panel3;
use gax_numga_examples::signal::haversine;
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Light, ORIGIN3, Point2, Rect, Scene3, backdrop,
    caption, palette, run,
};

mod fresnel {
    use gax::sta::{Bivector, Trivector, Vector, intrinsic};
    use gax_numga_examples::measure::roots;

    pub type V = Vector<(), f64>;
    pub type B = Bivector<(), f64>;
    /// A material: the map from the field to its excitation, `Antibivector <- Bivector`
    /// (antibivectors are bivectors in four dimensions, of the other orientation).
    pub type Medium = intrinsic::Bivector<(Bivector,), f64>;
    /// A map on fields, `Bivector <- Bivector`, such as a material's response.
    pub type Response = Bivector<(Bivector,), f64>;
    /// A map on vectors, such as a wavevector's potential map.
    pub type VectorMap = Vector<(Vector,), f64>;
    /// A potential's field, `Bivector <- Vector`.
    pub type FieldMap = Bivector<(Vector,), f64>;

    pub const T: V = V::new(1.0, 0.0, 0.0, 0.0);
    pub const X: V = V::new(0.0, 1.0, 0.0, 0.0);
    pub const Y: V = V::new(0.0, 0.0, 1.0, 0.0);
    pub const Z: V = V::new(0.0, 0.0, 0.0, 1.0);
    /// numga's planes `zx` and `xy`, gax's `e31` and `e12`.
    pub const ZX: B = B::new(0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    pub const XY: B = B::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    /// The open field.
    pub fn open_field() -> Response {
        Bivector::slot()
    }

    /// The open vector.
    pub fn open_vector() -> VectorMap {
        Vector::slot()
    }

    /// The response of a reciprocal dielectric at rest along `t`, `Bivector <- Bivector`: its
    /// electric planes weighted by their permittivities, its magnetic planes by the reluctivity
    /// (the inverse permeability).
    pub fn response(permittivity: [f64; 3], reluctivity: f64) -> Response {
        // Each plane reads the field's part along itself and returns that plane, weighted.
        let electric: Response = [X, Y, Z]
            .into_iter()
            .zip(permittivity)
            .map(|(axis, weight)| {
                let plane = axis ^ T;
                (plane * (plane | open_field())) * weight
            })
            .sum();
        // The sandwich by t keeps the magnetic planes and turns the electric ones around: half
        // the sum with the identity keeps the magnetic planes alone.
        let magnetic = (open_field() + (T >> open_field())) * 0.5;
        electric + magnetic * reluctivity
    }

    /// The dielectric: its response, dualized to the excitation.
    pub fn dielectric(permittivity: [f64; 3], reluctivity: f64) -> Medium {
        response(permittivity, reluctivity).hodge()
    }

    /// The potential map of a wavevector: a potential's field `k ∧ A`, its excitation, and
    /// Maxwell's other equation on that, dualized. The wavevector always lies in its kernel.
    pub fn potential_map(medium: Medium, wavevector: V) -> VectorMap {
        (wavevector ^ medium.of(field_map(wavevector))).hodge()
    }

    /// The field of a potential in a wave of `wavevector`, `k ∧ A`.
    pub fn field_map(wavevector: V) -> FieldMap {
        wavevector ^ open_vector()
    }

    /// The adjugate of a map's action on volumes: the map on vectors that measures against a
    /// volume what the vector measures against the volume's image, `adj(c) & v = c & volumes(v)`.
    pub fn volume_adjugate(map: VectorMap) -> VectorMap {
        let volumes = map.outermorphism::<Trivector>();
        let incidence: intrinsic::Scalar<(Vector, Trivector), f64> =
            Vector::slot() & Trivector::slot();
        incidence.solve(Vector::slot() & volumes)
    }

    /// The Fresnel quartic of a wavevector, for its nonzero frequency as the unit timelike
    /// `observer` measures it.
    pub fn polynomial(medium: Medium, observer: V, wavevector: V) -> f64 {
        let adjugate = volume_adjugate(potential_map(medium, wavevector));
        let frequency = (observer | wavevector).s();
        -(observer | adjugate.of(observer)).s() / (frequency * frequency)
    }

    /// The two roots of `a s² + b s + c`, ascending; where the two coincide, rounding can leave
    /// no real pair, and they are taken as one double root.
    fn quadratic_roots(a: f64, b: f64, c: f64) -> [f64; 2] {
        let (middle, half) = roots(a, b / 2.0, c).unwrap_or((-b / (2.0 * a), 0.0));
        [middle - half, middle + half]
    }

    /// The root of a non-negative number: half the distance between the roots of `s² - r²`.
    fn root(squared: f64) -> f64 {
        roots(1.0, 0.0, -squared).map_or(0.0, |(_, half)| half)
    }

    /// The square roots of the three sampled squared radii, 0, 1 and 2.
    const SAMPLE_RADII: [f64; 3] = [0.0, 1.0, core::f64::consts::SQRT_2];

    /// Both wavevector sheets of a dielectric at rest, along a unit spatial `direction`: the
    /// inner sheet first. The quartic is even in the spatial radius, so three samples fix it as
    /// a quadratic in the radius squared, whose two roots are the two polarizations' sheets.
    pub fn radial_sheets(medium: Medium, direction: V, frequency: f64) -> [V; 2] {
        let [constant, unit, twice] =
            SAMPLE_RADII.map(|radius| polynomial(medium, T, T * frequency + direction * radius));
        let quadratic = (twice - 2.0 * unit + constant) / 2.0;
        let linear = unit - constant - quadratic;
        quadratic_roots(quadratic, linear, constant).map(|squared| direction * root(squared))
    }

    /// The direction at `angle` from `z` towards `x`, by a turn in numga's `zx` plane.
    pub fn meridian(angle: f64) -> V {
        (ZX * (angle / 2.0)).exp() >> Z
    }

    /// A sphere of directions, `latitudes` by `longitudes`: each meridian turned about `z` in
    /// numga's `xy` plane.
    pub fn sphere(latitudes: usize, longitudes: usize) -> Vec<Vec<V>> {
        let share = |i: usize, n: usize| i as f64 / (n - 1) as f64;
        (0..latitudes)
            .map(|i| {
                let polar = meridian(core::f64::consts::PI * share(i, latitudes));
                (0..longitudes)
                    .map(|j| (XY * (core::f64::consts::PI * share(j, longitudes))).exp() >> polar)
                    .collect()
            })
            .collect()
    }

    /// A circle of directions in the `xz` plane, from `z` round to `z`.
    pub fn circle(samples: usize) -> Vec<V> {
        let angle = |i: usize| core::f64::consts::TAU * i as f64 / (samples - 1) as f64;
        (0..samples).map(|i| meridian(angle(i))).collect()
    }

    // --- scenes ---------------------------------------------------------------------------

    /// Vacuum, a uniaxial and a biaxial crystal: the permittivities along the principal axes
    /// `x`, `y` and `z`, all with unit reluctivity.
    pub const CASES: [(&str, [f64; 3]); 3] = [
        ("VACUUM", [1.0, 1.0, 1.0]),
        ("UNIAXIAL CRYSTAL", [2.25, 2.25, 3.24]),
        ("BIAXIAL CRYSTAL", [1.44, 2.25, 3.24]),
    ];
    pub const RELUCTIVITY: f64 = 1.0;
    pub const FREQUENCY: f64 = 1.0;
    pub const LATITUDES: usize = 33;
    pub const SECTION_SAMPLES: usize = 4 * (LATITUDES - 1) + 1;
    pub const MOVIE_LATITUDES: usize = 19;
    pub const MOVIE_LONGITUDES: usize = 2 * (MOVIE_LATITUDES - 1) + 1;

    /// The permittivities a share `weight` of the way from vacuum to the biaxial crystal.
    pub fn between(weight: f64) -> [f64; 3] {
        let (vacuum, crystal) = (CASES[0].1, CASES[2].1);
        core::array::from_fn(|i| vacuum[i] + weight * (crystal[i] - vacuum[i]))
    }

    /// Both sheets along each of the directions.
    pub fn sheets(medium: Medium, directions: &[V]) -> Vec<[V; 2]> {
        directions
            .iter()
            .map(|d| radial_sheets(medium, *d, FREQUENCY))
            .collect()
    }
}

use fresnel::*;

/// One loop: numga's 32 frames of 90 ms, played at half speed.
const SECONDS: f32 = 2.88 * 2.0;
/// How far the views reach, in units of the frequency.
const LIMIT: f32 = 2.05;
/// Every how many latitudes and longitudes the wireframe draws a line.
const LATITUDE_STRIDE: usize = 3;
const LONGITUDE_STRIDE: usize = 4;
/// The view of the sheets.
const AZIMUTH: f32 = -0.92;
const ELEVATION: f32 = 0.38;
const DISTANCE: f32 = 7.6;

/// The inner and outer sheets' lights.
fn sheet_light(sheet: usize) -> Light {
    [palette::sky(), palette::orange()][sheet]
}

/// A spatial vector as a point of space, as drawn.
fn spatial(v: V) -> gax::pga3d::Point<(), f64> {
    gax::pga3d::Point::xyz(v.e1(), v.e2(), v.e3())
}

/// The sheets on a sphere of directions, in space: translucent and lit, with a wireframe.
fn surfaces(c: &mut Canvas, rect: Rect, grid: &[Vec<[V; 2]>]) {
    let cam = Camera::orbit(
        rect,
        ORIGIN3,
        DISTANCE,
        AZIMUTH,
        ELEVATION,
        Lens::Perspective(0.5),
    );
    panel3(c, rect, cam, |s: &mut Scene3| {
        for axis in [X, Y, Z] {
            let reach = spatial(axis * f64::from(LIMIT));
            s.seg(
                reach,
                spatial(axis * -f64::from(LIMIT)),
                1.0,
                palette::grid(),
            );
        }
        for sheet in 0..2 {
            let at = |i: usize, j: usize| spatial(grid[i][j][sheet]);
            let light = sheet_light(sheet);
            for (row, next) in grid.iter().zip(&grid[1..]) {
                for (here, there) in row.windows(2).zip(next.windows(2)) {
                    let [p0, p1, p2, p3] =
                        [here[0], there[0], there[1], here[1]].map(|k| spatial(k[sheet]));
                    let shade = s.lit(p0, p1, p2, light.faded(0.5));
                    s.quad(p0, p1, p2, p3, shade, 0.14);
                }
            }
            let wire = light.faded(0.6);
            for i in (0..grid.len()).step_by(LATITUDE_STRIDE) {
                let ring: Vec<_> = (0..grid[i].len()).map(|j| at(i, j)).collect();
                s.polyline(&ring, 1.0, wire);
            }
            for j in (0..grid[0].len()).step_by(LONGITUDE_STRIDE) {
                let line: Vec<_> = (0..grid.len()).map(|i| at(i, j)).collect();
                s.polyline(&line, 1.0, wire);
            }
        }
    });
}

/// The sheets' sections through the `xz` plane.
fn sections(c: &mut Canvas, rect: Rect, section: &[[V; 2]]) {
    let ax = Axes::equal(rect, Point2::xy(0.0, 0.0), LIMIT);
    ax.frame(c, "SECTION THROUGH XZ", "KX", "KZ");
    let faint = palette::grid().faded(2.0);
    ax.line(
        c,
        Point2::xy(-LIMIT * 2.0, 0.0),
        Point2::xy(LIMIT * 2.0, 0.0),
        1.0,
        faint,
    );
    ax.line(
        c,
        Point2::xy(0.0, -LIMIT),
        Point2::xy(0.0, LIMIT),
        1.0,
        faint,
    );
    // Each sheet's wavevectors in the plane of `x` (across) and `z` (up).
    let [inner, outer] = [0, 1].map(|sheet| {
        let in_plane = |k: &[V; 2]| Point2::xy(k[sheet].e1() as f32, k[sheet].e3() as f32);
        section.iter().map(in_plane).collect::<Vec<Point2>>()
    });
    ax.polyline(c, &inner, 2.0, sheet_light(0));
    ax.dashed(c, &outer, 2.0, 6.0, sheet_light(1));
    let legend = [
        ("INNER SHEET", sheet_light(0)),
        ("OUTER SHEET", sheet_light(1)),
    ];
    ax.legend(c, &legend);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let phase = core::f64::consts::TAU * f64::from(t / SECONDS);
    let weight = haversine(phase);
    let permittivity = between(weight);
    let medium = dielectric(permittivity, RELUCTIVITY);
    let grid: Vec<Vec<[V; 2]>> = sphere(MOVIE_LATITUDES, MOVIE_LONGITUDES)
        .iter()
        .map(|row| sheets(medium, row))
        .collect();
    let section = sheets(medium, &circle(SECTION_SAMPLES));
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let top = (h / 30.0).clamp(10.0, 22.0) * 3.4;
    let body = Rect::new(0.0, top, w, h);
    surfaces(c, body.part(0.0, 0.0, 0.55, 1.0), &grid);
    sections(c, body.part(0.6, 0.15, 0.97, 0.92), &section);
    let [ex, ey, ez] = permittivity;
    let label = format!("PERMITTIVITY {ex:.2} {ey:.2} {ez:.2}");
    let at = body.part(0.0, 0.0, 0.55, 0.97).bottom_middle();
    c.text(&label, at, 11.0, palette::ink(), Align::Center);
    caption(
        c,
        "FRESNEL WAVE SURFACES: VACUUM TO A BIAXIAL CRYSTAL",
        "WHERE THE ADJUGATE OF THE POTENTIAL MAP'S ACTION ON VOLUMES VANISHES: TWO SHEETS (STA)",
    );
}

fn main() {
    run(Anim::new("fresnel", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::fresnel::*;
    use gax::ApproxEq;
    use gax::sta::{Bivector, Vector, intrinsic};

    /// numga's comparison grid: twice as many longitudes as latitudes, round the sphere.
    const LONGITUDES: usize = 2 * (LATITUDES - 1) + 1;

    /// Whether `a` is within `tol` of `b`, coefficient by coefficient.
    fn near<X: ApproxEq>(a: X, b: X, tol: f64) -> bool {
        a.max_abs_diff(&b) <= tol
    }

    /// A rotor from numga's coefficients on `tx` (gax's `-e10`) and `yz`.
    fn rotor(tx: f64, yz: f64) -> gax::Unit<gax::sta::Even<(), f64>> {
        B::new(-tx, 0.0, 0.0, yz, 0.0, 0.0).exp()
    }

    /// The adjoint of a map on vectors under the metric: `M*(a) · b = a · M(b)` for all `a`,
    /// `b`.
    fn adjoint(map: VectorMap) -> VectorMap {
        let metric = Vector::slot() | Vector::slot();
        metric.solve(Vector::slot() | map)
    }

    /// The adjoint of a field map under the metric: `F*(B) · a = B · F(a)` for every bivector
    /// `B` and vector `a`.
    fn field_adjoint(map: FieldMap) -> Vector<(Bivector,), f64> {
        let metric = Vector::slot() | Vector::slot();
        metric.solve(Bivector::slot() | map)
    }

    /// The medium seen moved by a rotor: conjugated by it.
    fn moved(medium: Medium, r: gax::Unit<gax::sta::Even<(), f64>>) -> Medium {
        r >> medium.of(r << open_field())
    }

    #[test]
    fn vacuum_quartic_and_observer_independence() {
        let medium = dielectric([1.0; 3], 1.0);
        let wavevectors = [
            V::new(2.0, 0.3, 0.4, -0.2),
            V::new(1.0, 0.0, 0.0, 1.0),
            V::new(0.8, 1.1, -0.2, 0.3),
        ];
        let observer = rotor(0.23, 0.0) >> T;
        for k in wavevectors {
            let expected = k.norm_squared() * k.norm_squared();
            for frame in [T, observer] {
                let actual = polynomial(medium, frame, k);
                assert!((actual - expected).abs() < 1e-10, "{actual} {expected}");
            }
            assert!(near(potential_map(medium, k).of(k), V::zero(), 1e-12));
        }
        // numga's quartic of one wavevector in the biaxial crystal.
        let biaxial = dielectric(CASES[2].1, RELUCTIVITY);
        let probe = polynomial(biaxial, T, V::new(1.0, 0.9, 0.6, 0.4));
        assert!((probe - 1.708_452).abs() < 1e-6, "{probe}");
    }

    #[test]
    fn principal_wave_radii_match_transverse_material_weights() {
        let (reluctivity, frequency) = (0.7, 1.2);
        let medium = dielectric([2.0, 3.0, 4.0], reluctivity);
        let expected = [[3.0, 4.0], [2.0, 4.0], [2.0, 3.0]];
        for (axis, weights) in [X, Y, Z].into_iter().zip(expected) {
            let sheets = radial_sheets(medium, axis, frequency);
            for (k, weight) in sheets.into_iter().zip(weights) {
                let radius_squared = -k.norm_squared();
                let wanted = frequency * frequency * weight / reluctivity;
                assert!(
                    (radius_squared - wanted).abs() < 1e-10,
                    "{radius_squared} {wanted}"
                );
            }
        }
    }

    #[test]
    fn both_sheets_support_maxwell_fields_and_transform_with_the_medium() {
        let r = rotor(0.19, 0.13);
        for (_, permittivity) in CASES {
            let medium = dielectric(permittivity, RELUCTIVITY);
            let moved_medium = moved(medium, r);
            for direction in sphere(7, 13).into_iter().flatten() {
                for sheet in radial_sheets(medium, direction, FREQUENCY) {
                    let k = T * FREQUENCY + sheet;
                    let wave = potential_map(medium, k);
                    // The two potentials the map loses, besides none: its smallest singular
                    // vectors, and the fields they give.
                    let potentials = wave.svd().inputs;
                    let fields = [potentials[2], potentials[3]].map(|a| k ^ a);
                    let residual = polynomial(medium, T, k);
                    assert!(residual.abs() < 1e-9, "{residual}");
                    assert!(near(volume_adjugate(wave), VectorMap::zero(), 1e-9));
                    // Repeated radial roots lose half the significant digits to the square root.
                    for f in fields {
                        assert!(near(k ^ medium.of(f), intrinsic::Trivector::zero(), 1e-5));
                    }
                    let size: f64 = fields
                        .iter()
                        .map(|f| f.c.iter().map(|x| x * x).sum::<f64>())
                        .sum();
                    assert!(size > 0.1);
                    let moved_residual = polynomial(moved_medium, r >> T, r >> k);
                    assert!((moved_residual - residual).abs() < 1e-9);
                }
            }
        }
    }

    /// numga's scenario: every wavevector on the sheets of the three materials solves the
    /// quartic and lies in its potential map's kernel; and, from the notebook, the potential
    /// maps are self-adjoint, the materials' self-adjoint responses between the field map and
    /// its adjoint, the contraction with the wavevector.
    #[test]
    fn the_comparison_sheets_solve_the_quartic() {
        for (_, permittivity) in CASES {
            let medium = dielectric(permittivity, RELUCTIVITY);
            let response = response(permittivity, RELUCTIVITY);
            for row in sphere(LATITUDES, LONGITUDES) {
                for pair in sheets(medium, &row) {
                    for sheet in pair {
                        let k = T * FREQUENCY + sheet;
                        assert!(polynomial(medium, T, k).abs() < 1e-9);
                        let wave = potential_map(medium, k);
                        assert!(near(wave.of(k), V::zero(), 1e-11));
                        assert!(near(adjoint(wave), wave, 1e-12));
                        let field = field_map(k);
                        let back = field_adjoint(field);
                        assert!(near(back, Bivector::slot() | k, 1e-12));
                        assert!(near(back.of(response.of(field)), wave, 1e-12));
                    }
                }
            }
            let metric = Bivector::slot() | Bivector::slot();
            let response_adjoint: Response = metric.solve(Bivector::slot() | response);
            assert!(near(response_adjoint, response, 1e-12));
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
    }
}
