//! numga's `electromagnetism/fresnel`: light's wave surface in a crystal, from extensors in
//! spacetime algebra (STA, `x+y+z+t-`, as numga's).
//!
//! A material is a map from the field, a bivector, to its excitation, an antibivector, built
//! with the field left open as a sum of dyads: each patch that reads a part of the field, times
//! its reading `F ∨ patch`, weighted by its material constant. The planes each axis sweeps
//! through time read the electric field and carry the permittivities; the patches of the
//! planes of space read the magnetic field and share one reluctivity. A potential `A` gives the
//! field `k ∧ A`, and the potential map `A ↦ k ∧ χ(k ∧ A)` is what is left of Maxwell's other
//! equation, the current the potential would need: from potentials, extrinsic, to currents,
//! intrinsic, with no metric but the material's. It always loses the potential along `k`; a
//! wave is a second potential it loses, and then it sends every volume to nothing: its action
//! on volumes, from the volume of three potentials to the join of their currents, vanishes.
//! Read back on the observer's rest space, `rest ∨ volumes(rest)`, it holds two factors of the
//! frequency `k ∨ time` and one number more: the Fresnel quartic, zero where light can travel.
//! An axion term, a coupling times the field itself, meets the wave's own wavevector in
//! Maxwell's other equation and vanishes: light does not see it.
//!
//! Along each wavefront at a fixed frequency the quartic of a crystal at rest is a quadratic
//! in the squared wavenumber, fixed by three samples, and its two roots are the two sheets of
//! the wave surface, one per polarization. The animation turns vacuum (one sphere) into a
//! biaxial crystal and back: the two sheets in space, and their sections through the `xz`
//! plane, which touch at the optic axes.

use gax::sta::Vector;
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::signal::haversine;
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Light, ORIGIN2, ORIGIN3, Point2, Rect, Scene3,
    backdrop, caption, palette, run,
};

mod fresnel {
    // numga's API in full: the tests use parts the animation does not.
    #![cfg_attr(not(test), allow(dead_code))]

    use gax::sta::{Bivector, Trivector, Vector, intrinsic};
    use gax_numga_examples::measure::roots;

    /// A material: the map from the field to its excitation, `Antibivector <- Bivector`
    /// (antibivectors are bivectors in four dimensions, intrinsic).
    pub type Medium = intrinsic::Bivector<(Bivector,)>;
    /// An excitation, a patch of a plane: an antibivector.
    pub type Patch = intrinsic::Bivector;
    /// A current, or a direction in time: an antivector (a trivector, intrinsic).
    pub type Antivector = intrinsic::Trivector;
    /// A wavevector's potential map, from potentials to the currents they would need,
    /// `Antivector <- Vector`.
    pub type WaveMap = intrinsic::Trivector<(Vector,)>;
    /// A potential's field, `Bivector <- Vector`.
    pub type FieldMap = Bivector<(Vector,)>;
    /// A coupling of the excitation to the field itself, with a sense of its own: an intrinsic
    /// scalar (numga's antipseudoscalar).
    pub type AxionCoupling = intrinsic::Scalar;

    pub const X: Vector = Vector::new(1.0, 0.0, 0.0, 0.0);
    pub const Y: Vector = Vector::new(0.0, 1.0, 0.0, 0.0);
    pub const Z: Vector = Vector::new(0.0, 0.0, 1.0, 0.0);
    pub const T: Vector = Vector::new(0.0, 0.0, 0.0, 1.0);
    /// The planes of space `yz`, `zx` and `xy`. numga's sphere turns by `exp(-θ/2 zx)` and
    /// `exp(-φ/2 xy)`, z towards x and x towards y.
    pub const YZ: Bivector = Bivector::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    pub const ZX: Bivector = Bivector::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    pub const XY: Bivector = Bivector::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);

    /// The observer's direction in time, which pairs to one with `t`: `-t*`, the Hodge dual
    /// (numga's `dual`) carrying the minus of `t`'s square.
    pub fn time() -> Antivector {
        -T.hodge()
    }

    /// The observer's rest space, the volume of its three spatial potentials.
    pub fn rest_space() -> Trivector {
        X ^ Y ^ Z
    }

    /// The open field.
    pub fn open_field() -> Bivector<(Bivector,)> {
        Bivector::slot()
    }

    /// The open potential.
    pub fn open_vector() -> Vector<(Vector,)> {
        Vector::slot()
    }

    /// The planes each axis sweeps through time, `axis* ∨ time`, which read the electric
    /// field, and the patches of the planes of space, which read the magnetic field.
    pub fn swept_planes() -> [Patch; 3] {
        [X, Y, Z].map(|axis| axis.hodge() & time())
    }
    pub fn patches() -> [Patch; 3] {
        [YZ, ZX, XY].map(|plane| plane.hodge())
    }

    /// A reciprocal dielectric at rest along `time`, from its permittivity along each axis and
    /// its one reluctivity, the inverse permeability: the dyads of the patches that read the
    /// field, each a patch times its reading `F ∨ patch`, weighted by its material constant.
    /// Time squares to minus one, so the electric dyads enter with a minus. No metric enters
    /// but the material's own.
    pub fn dielectric(permittivity: [f64; 3], reluctivity: f64) -> Medium {
        let dyad = |patch: Patch| patch * (open_field() & patch);
        let electric: Medium = swept_planes()
            .into_iter()
            .zip(permittivity)
            .map(|(p, e)| dyad(p).gp(e))
            .sum();
        let magnetic: Medium = patches().into_iter().map(dyad).sum();
        magnetic.gp(reluctivity) - electric
    }

    /// The field of a potential in a wave of `wavevector`, `k ∧ A`.
    pub fn field_map(wavevector: Vector) -> FieldMap {
        wavevector ^ open_vector()
    }

    /// The potential map of a wavevector: from a potential to the current it would need, its
    /// excitation's residual in Maxwell's other equation, `k ∧ χ(k ∧ A)`: from potentials,
    /// extrinsic, to currents, intrinsic. The wavevector always lies in its kernel.
    pub fn potential_map(medium: Medium, wavevector: Vector) -> WaveMap {
        wavevector ^ medium.of(field_map(wavevector))
    }

    /// The potential map's action on volumes, applied to one: the volume `a ∧ b ∧ c` of three
    /// potentials goes to the join `M(a) ∨ M(b) ∨ M(c)` of their currents, numga's
    /// `wave.outermorphism(Trivector)`. gax's outermorphisms run from a kind to itself; this
    /// one, wedging in and joining out, is written out on the basis volumes `xyz yzt zxt xyt`.
    pub fn volumes(wave: WaveMap, volume: Trivector) -> intrinsic::Vector {
        let m = [X, Y, Z, T].map(|v| wave.of(v));
        let join = |i: usize, j: usize, k: usize| m[i] & m[j] & m[k];
        let [xyz, yzt, zxt, xyt] = volume.c;
        (join(0, 1, 2).gp(xyz) + join(1, 2, 3).gp(yzt))
            + (join(2, 0, 3).gp(zxt) + join(0, 1, 3).gp(xyt))
    }

    /// The Fresnel quartic of a wavevector, for its nonzero frequency as measured by the
    /// observer whose unit direction in time is `time` and whose unit rest space is
    /// `rest_space`: the rest space reads the join of the currents of its own three
    /// potentials, which holds two factors of the frequency `k ∨ time` and the quartic.
    pub fn polynomial(
        medium: Medium,
        time: Antivector,
        rest_space: Trivector,
        wavevector: Vector,
    ) -> f64 {
        let joined = volumes(potential_map(medium, wavevector), rest_space);
        let frequency = (wavevector & time).s();
        (rest_space & joined).s() / (frequency * frequency)
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

    /// The square roots of the three sampled squared wavenumbers, 0, 1 and 2.
    const SAMPLE_RADII: [f64; 3] = [0.0, 1.0, core::f64::consts::SQRT_2];

    /// Both wavevector sheets of a dielectric at rest, along a unit wavefront, a plane of space
    /// (its normal `wavefront`): the inner sheet first. The quartic is even in the wavenumber,
    /// so three samples fix it as a quadratic in the wavenumber squared, whose two roots are the
    /// two polarizations' sheets.
    pub fn radial_sheets(medium: Medium, wavefront: Vector, frequency: f64) -> [Vector; 2] {
        let [constant, unit, twice] = SAMPLE_RADII.map(|wavenumber| {
            polynomial(
                medium,
                time(),
                rest_space(),
                T * frequency + wavefront * wavenumber,
            )
        });
        let quadratic = (twice - 2.0 * unit + constant) / 2.0;
        let linear = unit - constant - quadratic;
        quadratic_roots(quadratic, linear, constant).map(|squared| wavefront * root(squared))
    }

    /// The wavefront at `angle` from `z` towards `x`, by a turn in the `zx` plane.
    pub fn meridian(angle: f64) -> Vector {
        (ZX * (-angle / 2.0)).exp() >> Z
    }

    /// A sphere of unit wavefronts, `latitudes` by `longitudes`: each meridian turned about `z`
    /// from x towards y.
    pub fn sphere(latitudes: usize, longitudes: usize) -> Vec<Vec<Vector>> {
        let share = |i: usize, n: usize| i as f64 / (n - 1) as f64;
        (0..latitudes)
            .map(|i| {
                let polar = meridian(core::f64::consts::PI * share(i, latitudes));
                (0..longitudes)
                    .map(|j| (XY * (-core::f64::consts::PI * share(j, longitudes))).exp() >> polar)
                    .collect()
            })
            .collect()
    }

    /// A circle of unit wavefronts in the `xz` plane, from `z` round to `z`.
    pub fn circle(samples: usize) -> Vec<Vector> {
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
    /// An axion coupling, of an illustrative size: light does not see it.
    pub const AXION_COUPLING: f64 = 0.3;

    /// The axion part of a material, a coupling times the field itself: it meets the field's
    /// own wavevector in Maxwell's other equation, `k ∧ (α k ∧ A)`, which vanishes.
    pub fn axion(coupling: f64) -> Medium {
        AxionCoupling::new(coupling) * open_field()
    }

    /// The permittivities a share `weight` of the way from vacuum to the biaxial crystal.
    pub fn between(weight: f64) -> [f64; 3] {
        let (vacuum, crystal) = (CASES[0].1, CASES[2].1);
        core::array::from_fn(|i| vacuum[i] + weight * (crystal[i] - vacuum[i]))
    }

    /// Both sheets along each of the directions.
    pub fn sheets(medium: Medium, directions: &[Vector]) -> Vec<[Vector; 2]> {
        directions
            .iter()
            .map(|d| radial_sheets(medium, *d, FREQUENCY))
            .collect()
    }
}

use fresnel::*;

/// One loop: numga's 72 frames of 40 ms, vacuum to the crystal and back, at half speed.
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
    [palette::SKY, palette::ORANGE][sheet]
}

/// A spatial vector as a point of space, as drawn.
fn spatial(v: Vector) -> gax::pga3d::Point {
    gax::pga3d::Point::xyz(v.e1(), v.e2(), v.e3())
}

/// The sheets on a sphere of directions, in space: translucent and lit, with a wireframe.
fn surfaces(c: &mut Canvas, rect: Rect, grid: &[Vec<[Vector; 2]>]) {
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
            s.seg(reach, spatial(axis * -f64::from(LIMIT)), 1.0, palette::GRID);
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
fn sections(c: &mut Canvas, rect: Rect, section: &[[Vector; 2]]) {
    let ax = Axes::equal(rect, ORIGIN2, LIMIT);
    ax.frame(c, "SECTION THROUGH XZ", "KX", "KZ");
    let faint = palette::GRID.faded(2.0);
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
        let in_plane = |k: &[Vector; 2]| Point2::xy(k[sheet].e1() as f32, k[sheet].e3() as f32);
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
    let grid: Vec<Vec<[Vector; 2]>> = sphere(MOVIE_LATITUDES, MOVIE_LONGITUDES)
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
    c.text(&label, at, 11.0, palette::INK, Align::Center);
    caption(
        c,
        "FRESNEL WAVE SURFACES: VACUUM TO A BIAXIAL CRYSTAL",
        "WHERE THE POTENTIAL MAP'S ACTION ON VOLUMES VANISHES: TWO SHEETS (STA)",
    );
}

fn main() {
    run(Anim::new("fresnel", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::fresnel::*;
    use gax::sta::{Bivector, Trivector, Vector, intrinsic};
    use gax_numga_examples::testing::near;

    /// numga's comparison grid: twice as many longitudes as latitudes, round the sphere.
    const LONGITUDES: usize = 2 * (LATITUDES - 1) + 1;

    /// A rotor from numga's coefficients on `tx` and `yz` (`tx = t ∧ x = -e14`).
    fn rotor(tx: f64, yz: f64) -> gax::Unit<gax::sta::Even> {
        Bivector::new(yz, 0.0, 0.0, -tx, 0.0, 0.0).exp()
    }

    /// The medium seen moved by a rotor: conjugated by it.
    fn moved(medium: Medium, r: gax::Unit<gax::sta::Even>) -> Medium {
        r >> medium.of(r << open_field())
    }

    /// The basis volumes `xyz yzt zxt xyt`.
    fn basis_volumes() -> [Trivector; 4] {
        [X ^ Y ^ Z, Y ^ Z ^ T, Z ^ X ^ T, X ^ Y ^ T]
    }

    #[test]
    fn vacuum_quartic_and_observer_independence() {
        let medium = dielectric([1.0; 3], 1.0);
        let wavevectors = [
            Vector::new(0.3, 0.4, -0.2, 2.0),
            Vector::new(0.0, 0.0, 1.0, 1.0),
            Vector::new(1.1, -0.2, 0.3, 0.8),
        ];
        let boost = rotor(0.23, 0.0);
        for k in wavevectors {
            let expected = k.norm_squared() * k.norm_squared();
            for (time, rest_space) in [
                (time(), rest_space()),
                (boost >> time(), boost >> rest_space()),
            ] {
                let actual = polynomial(medium, time, rest_space, k);
                assert!((actual - expected).abs() < 1e-10, "{actual} {expected}");
            }
            assert!(near(
                potential_map(medium, k).of(k),
                Antivector::zero(),
                1e-12
            ));
        }
    }

    /// numga 35797de's numbers for one wavevector in the biaxial crystal: the potential map,
    /// its action on the rest space, and the quartic.
    #[test]
    fn the_maps_read_numga_s_numbers() {
        let biaxial = dielectric(CASES[2].1, RELUCTIVITY);
        let probe = Vector::new(0.9, 0.6, 0.4, 1.0);
        let wave = potential_map(biaxial, probe);
        let columns = [
            [1.296, 0.92, 0.54, 0.36],
            [1.35, 0.54, 1.28, 0.24],
            [1.296, 0.36, 0.24, 2.07],
            [-2.4948, -1.296, -1.35, -1.296],
        ];
        for (v, column) in [X, Y, Z, T].into_iter().zip(columns) {
            let got: [f64; 4] = wave.of(v).c;
            assert!(
                got.iter().zip(column).all(|(a, b)| (a - b).abs() < 1e-12),
                "{got:?}"
            );
        }
        let joined: [f64; 4] = volumes(wave, rest_space()).c;
        let numga = [1.5376068, 1.0250712, 0.6833808, 1.708452];
        assert!(
            joined.iter().zip(numga).all(|(a, b)| (a - b).abs() < 1e-7),
            "{joined:?}"
        );
        let probe = polynomial(biaxial, time(), rest_space(), probe);
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
                let radius_squared = k.norm_squared();
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
            for wavefront in sphere(7, 13).into_iter().flatten() {
                for sheet in radial_sheets(medium, wavefront, FREQUENCY) {
                    let k = T * FREQUENCY + sheet;
                    let wave = potential_map(medium, k);
                    // The two potentials the map loses: its smallest singular vectors, and the
                    // fields they give.
                    let potentials = wave.svd().inputs;
                    let fields = [potentials[2], potentials[3]].map(|a| k ^ a);
                    let residual = polynomial(medium, time(), rest_space(), k);
                    assert!(residual.abs() < 1e-9, "{residual}");
                    // The action on volumes vanishes on every volume.
                    for v in basis_volumes() {
                        assert!(near(volumes(wave, v), intrinsic::Vector::zero(), 1e-9));
                    }
                    // Repeated radial roots lose half the significant digits to the square root.
                    for f in fields {
                        assert!(near(k ^ medium.of(f), intrinsic::Trivector::zero(), 1e-5));
                    }
                    let size: f64 = fields
                        .iter()
                        .map(|f| f.c.iter().map(|x| x * x).sum::<f64>())
                        .sum();
                    assert!(size > 0.1);
                    let moved_residual =
                        polynomial(moved_medium, r >> time(), r >> rest_space(), r >> k);
                    assert!((moved_residual - residual).abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn the_axion_part_leaves_light_alone_and_reverses_in_a_mirror() {
        let medium = dielectric([1.44, 2.25, 3.24], 1.0);
        let axion = axion(0.3);
        for axis in [X, Y, Z] {
            for sheet in radial_sheets(medium, axis, 1.0) {
                let k = T + sheet;
                assert!(near(potential_map(axion, k), WaveMap::zero(), 1e-12));
                let quartic = polynomial(medium + axion, time(), rest_space(), k);
                assert!(quartic.abs() < 1e-9, "{quartic}");
            }
        }
        // The mirror image of the material: the reflected map has the opposite coupling.
        let reflected = X >> axion.of(X << open_field());
        assert!(near(reflected + axion, Medium::zero(), 1e-12));
    }

    /// numga's scenario checks: every wavevector on the sheets of the three materials solves
    /// the quartic and lies in its potential map's kernel; the axion part adds nothing to any
    /// potential map and leaves every sheet where it is (where two sheets coincide, the square
    /// root of the vanishing discriminant loses half the digits). And, from the notebook, the
    /// potential maps are symmetric with no metric, each minus its own adjugate, and built as
    /// the material between the field map and its adjugate, the wedge with the wavevector.
    #[test]
    fn the_comparison_sheets_solve_the_quartic() {
        let axion = axion(AXION_COUPLING);
        for (_, permittivity) in CASES {
            let medium = dielectric(permittivity, RELUCTIVITY);
            for row in sphere(LATITUDES, LONGITUDES) {
                let coupled = row
                    .iter()
                    .map(|d| radial_sheets(medium + axion, *d, FREQUENCY));
                for (pair, coupled) in sheets(medium, &row).into_iter().zip(coupled) {
                    for (sheet, coupled) in pair.into_iter().zip(coupled) {
                        assert!(near(coupled, sheet, 1e-6));
                        let k = T * FREQUENCY + sheet;
                        assert!(polynomial(medium, time(), rest_space(), k).abs() < 1e-9);
                        let wave = potential_map(medium, k);
                        assert!(near(wave.of(k), Antivector::zero(), 1e-11));
                        assert!(near(potential_map(axion, k), WaveMap::zero(), 1e-12));
                        assert!(near(wave.adjugate(), -wave, 1e-12));
                        let field = field_map(k);
                        let back = field.adjugate();
                        assert!(near(back, k ^ Patch::slot(), 1e-12));
                        assert!(near(back.of(medium.of(field)), wave, 1e-12));
                    }
                }
            }
        }
    }

    /// The port's own: each dyad pairs two fields alike either way round, so the material's
    /// form `a ∨ χ(b)` is symmetric (the notebook's reason for the potential map's symmetry).
    #[test]
    fn the_material_form_is_symmetric() {
        let basis = |i: usize| -> Bivector {
            Bivector::from_coeffs(core::array::from_fn(|j| f64::from(i == j)))
        };
        for (_, permittivity) in CASES {
            let medium = dielectric(permittivity, 0.7);
            for i in 0..6 {
                for j in 0..6 {
                    let (a, b) = (basis(i), basis(j));
                    assert!(((a & medium.of(b)).s() - (b & medium.of(a)).s()).abs() < 1e-14);
                }
            }
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
    }
}
