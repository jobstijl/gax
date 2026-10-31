//! numga's `orientation/optical_rotation`: light's polarization turned by quartz and by a
//! magnetic field, out to a reflector and back, in numga's spacetime algebra `x+y+z+t-`.
//!
//! Light travels along a unit direction in spacetime, an intrinsic trivector (numga's
//! antivector, ADR-051). Its electric field is an extrinsic bivector: the field along its
//! polarization, the Hodge dual (numga's `dual`) of the plane the polarization sweeps through
//! the observer's time, `(polarization & time).hodge()`. A polarizer reads the field by pairing
//! it with the plane its axis sweeps, `field & (axis & time)`.
//!
//! An optically active medium, quartz or a sugar solution, turns the field across the path by
//! its rotary power per distance, in the plane the light sweeps through time, `direction &
//! time`: an intrinsic bivector oriented by the travel, which as a turn lies across the path.
//! An intrinsic bivector is not a rate (gax's `exp` refuses it); the rotary power makes it one.
//! The medium's turn has a sense of its own, so the rotary power is an intrinsic scalar, and
//! the product of the two is an extrinsic bivector that exponentiates: the turn reverses with
//! the direction of travel, and quartz turns the returning light back. A magnetic field, a
//! plane of space, turns the light by the Verdet constant times its flux through the beam's
//! cross-section, `magnetic_field & cross_section`, in the cross-section's plane: neither knows
//! which way the light goes, so the field keeps turning the returning light the same way. With
//! a 45 degree turn each way, the light comes back crossed with the polarizer it entered
//! through, which stops it: an optical isolator.
//!
//! Mirroring every input of the setup with `>>` (the beam, the medium's rotary power, the
//! magnetic field and the entering light) mirrors every result, with no sign rule added: the
//! mirror image of right-handed quartz is left-handed quartz.
//!
//! The animation unfolds each path along the beam, out to the reflector and back, quartz on the
//! left and the magnetic field on the right, the original setups above their mirror images.
//! The light runs out and back; each line is the polarization across the beam where the light
//! has reached, orange out and teal back, with the entering polarization dashed at both ends.

use gax_numga_examples::scene3::{Scene3, panel3};
use gax_numga_examples::{
    Anim, Camera, Canvas, Lens, Light, Point3, Rect, backdrop, caption, label_under, palette,
    reach3, run_with,
};

use gax_numga_examples::shared::spacetime;

mod rotation {
    use super::spacetime::{Bivector, Direction, Vector, intrinsic, space, t, time, x, y};

    /// An electric or magnetic field: an extrinsic bivector.
    pub type Field = Bivector;
    /// A rate of turning: an extrinsic bivector.
    pub type TurnRate = Bivector;
    /// The beam's cross-section, the patch two directions across the path span: intrinsic.
    pub type CrossSection = intrinsic::Bivector;
    /// A turn per distance with a sense of its own: an intrinsic scalar.
    pub type RotaryPower = intrinsic::Scalar;

    /// The path through the medium, mm, and the samples of the light along each leg.
    pub const LENGTH: f64 = 2.0;
    pub const SAMPLES: usize = 9;
    /// The rotary power in rad per mm (quartz turns 21.7 degrees per mm), and the Verdet constant
    /// in rad per mm per tesla, illustrative; the magnetic field's strength in tesla.
    pub const QUARTZ_ROTARY_POWER: f64 = 22.5 * core::f64::consts::PI / 180.0;
    pub const VERDET: f64 = 22.5 * core::f64::consts::PI / 180.0;
    pub const FIELD_STRENGTH: f64 = 1.0;

    /// The field along a direction, the Hodge dual of the plane the direction sweeps in time:
    /// the vacuum's identification of a direction with the field along it, which pairs to one
    /// with that plane for a unit direction.
    pub fn field_along(d: Direction) -> Field {
        (d & time()).hodge()
    }

    /// What a polarizer along `axis` reads of a field: its pairing with the plane the axis
    /// sweeps in time.
    pub fn reading(field: Field, axis: Direction) -> f64 {
        (field & (axis & time())).s()
    }

    /// The light's field at each depth on the way out, and on the way back after a reflection
    /// at the far end, `[legs][depths]`: the light leaves the medium where it entered.
    pub fn out_and_back(
        out_rate: TurnRate,
        back_rate: TurnRate,
        entering: Field,
    ) -> [Vec<Field>; 2] {
        let depths = || (0..SAMPLES).map(|k| LENGTH * k as f64 / (SAMPLES - 1) as f64);
        let out: Vec<Field> = depths()
            .map(|d| out_rate.gp(-d / 2.0).exp() >> entering)
            .collect();
        let far = out[SAMPLES - 1];
        let back = depths()
            .map(|d| back_rate.gp(-d / 2.0).exp() >> far)
            .collect();
        [out, back]
    }

    /// A setup: the beam's direction, its cross-section, the medium's rotary power, the magnetic
    /// field, and the light entering.
    #[derive(Clone, Copy)]
    pub struct Setup {
        pub direction: Direction,
        pub cross_section: CrossSection,
        pub rotary_power: RotaryPower,
        pub magnetic_field: Field,
        pub entering: Field,
    }

    impl Setup {
        /// The mirror image of the setup: every input reflected.
        pub fn mirrored(self, mirror: Vector) -> Setup {
            Setup {
                direction: mirror >> self.direction,
                cross_section: mirror >> self.cross_section,
                rotary_power: mirror >> self.rotary_power,
                magnetic_field: mirror >> self.magnetic_field,
                entering: mirror >> self.entering,
            }
        }
    }

    /// Light out and back through the optically active medium and through the magnetic field,
    /// `[media][legs][depths]`: the medium turns it by the direction it travels in, the field by
    /// the cross-section it crosses, the same out and back.
    pub fn through_media(s: Setup) -> [[Vec<Field>; 2]; 2] {
        // The plane the light sweeps through time, intrinsic, lies across the path as a turn;
        // the rotary power's own sense makes the product extrinsic, a rate that exponentiates.
        let active = out_and_back(
            (s.direction & time()) * s.rotary_power,
            (-s.direction & time()) * s.rotary_power,
            s.entering,
        );
        // The flux through the cross-section turns the light in the cross-section's plane,
        // numga's `dual_inverse`: minus the Hodge dual on bivectors.
        let plane = -s.cross_section.hodge();
        let faraday_rate = plane * (s.magnetic_field & s.cross_section) * VERDET;
        let faraday = out_and_back(faraday_rate, faraday_rate, s.entering);
        [active, faraday]
    }

    /// The light travels along `z`, through a polarizer along `x`.
    pub fn travel() -> Direction {
        space()[2]
    }
    pub fn polarizer_axis() -> Direction {
        space()[0]
    }
    /// The mirror: the plane `x = 0`, holding the path.
    pub fn mirror() -> Vector {
        x()
    }

    /// numga's setup: quartz turning 22.5 degrees per mm, a field across the path, and light
    /// polarized along `x`.
    pub fn setup() -> Setup {
        let xy = (x() ^ y()).cast::<Bivector>();
        Setup {
            direction: travel(),
            cross_section: xy.hodge(),
            rotary_power: RotaryPower::new(QUARTZ_ROTARY_POWER),
            magnetic_field: xy * FIELD_STRENGTH,
            entering: field_along(polarizer_axis()),
        }
    }

    /// numga's isolator: `[media][setups][legs][depths]`, the setups the original and its
    /// mirror image.
    pub fn isolator() -> [[[Vec<Field>; 2]; 2]; 2] {
        let [original_active, original_faraday] = through_media(setup());
        let [mirrored_active, mirrored_faraday] = through_media(setup().mirrored(mirror()));
        [
            [original_active, mirrored_active],
            [original_faraday, mirrored_faraday],
        ]
    }

    /// Each field's polarization: the plane it sweeps through time (numga's `dual_inverse`,
    /// minus the Hodge dual), met with the observer's slice of space.
    pub fn polarization(field: Field) -> Direction {
        (-field.hodge()) ^ t()
    }
}

use rotation::*;
use spacetime::{Bivector, Direction, Even, space};

/// The light out and back, then held.
const SECONDS: f32 = 6.0;
/// The share of the loop the light takes to run out and back.
const RUN: f32 = 0.7;
/// The drawn half-length of a polarization line, mm, and the entering polarization's and the
/// reflector's reach, in such half-lengths.
const HALF_LINE: f64 = 0.45;
const ENTERING_REACH: f64 = 1.3;
const REFLECTOR_REACH: f64 = 1.4;
/// The colours of the legs, out and back.
const LEGS: [Light; 2] = [palette::ORANGE, palette::SKY];

/// The beam's frame turned into the scene's, whose `z` is up: a half turn about the line along
/// `x + z` carries the beam, along `z`, onto `x`, across the picture, and the polarizations'
/// `x` onto `z`, up.
fn upright() -> Even {
    let axis = Bivector::new(1.0, 0.0, 1.0, 0.0, 0.0, 0.0);
    axis.gp(core::f64::consts::PI / (2.0 * core::f64::consts::SQRT_2))
        .exp()
        .into_inner()
}

/// A place or a direction in the beam's frame, in the scene.
fn scene(d: Direction) -> Point3 {
    reach3(upright() >> d)
}

/// One medium and setup in `rect`, the light having run `run` of the way out and back.
fn path_panel(c: &mut Canvas, rect: Rect, legs: &[Vec<Field>; 2], run: f64) {
    let beam = space()[2];
    let cam = Camera::orbit(
        rect,
        scene(beam.gp(LENGTH)),
        6.5,
        (-72f32).to_radians(),
        14f32.to_radians(),
        Lens::Perspective(0.5),
    );
    panel3(c, rect, cam, |s: &mut Scene3| {
        let [start, end] = [0.0, 2.0 * LENGTH].map(|a| scene(beam.gp(a)));
        s.seg(start, end, 0.9, palette::GRID.faded(3.0));
        // The reflector: a square across the beam at its far end.
        let across = |u: f64, v: f64| {
            let [ax, ay, _] = space();
            scene(ax.gp(u) + ay.gp(v) + beam.gp(LENGTH))
        };
        let r = REFLECTOR_REACH * HALF_LINE;
        s.quad(
            across(-r, -r),
            across(r, -r),
            across(r, r),
            across(-r, r),
            palette::INK.faded(0.3),
            0.2,
        );
        // The entering polarization, dashed at both ends.
        let entering = polarization(legs[0][0]).gp(ENTERING_REACH * HALF_LINE);
        for at in [0.0, 2.0 * LENGTH] {
            let middle = beam.gp(at);
            let dashes = 7;
            for k in 0..dashes {
                let share = |f: f64| -1.0 + 2.0 * (k as f64 + f) / dashes as f64;
                let [a, b] = [0.2, 0.8].map(|f| scene(middle + entering.gp(share(f))));
                s.seg(a, b, 1.1, palette::INK.faded(0.6));
            }
        }
        // The polarization where the light has reached.
        for (leg, (fields, colour)) in legs.iter().zip(LEGS).enumerate() {
            for (k, field) in fields.iter().enumerate() {
                let along = LENGTH * (leg as f64 + k as f64 / (SAMPLES - 1) as f64);
                if along > run * 2.0 * LENGTH + 1e-9 {
                    continue;
                }
                let middle = beam.gp(along);
                let half = polarization(*field).gp(HALF_LINE);
                s.seg(scene(middle - half), scene(middle + half), 2.2, colour);
            }
        }
    });
}

fn draw(c: &mut Canvas, fields: &[[[Vec<Field>; 2]; 2]; 2], t: f32) {
    backdrop(c);
    let screen = c.rect();
    let run = f64::from((t / (SECONDS * RUN)).min(1.0));
    let media = ["QUARTZ", "MAGNETIC FIELD"];
    let setups = ["", ", MIRRORED"];
    for (m, medium) in fields.iter().enumerate() {
        for (k, legs) in medium.iter().enumerate() {
            let rect = screen
                .part(
                    0.5 * m as f32,
                    0.14 + 0.43 * k as f32,
                    0.5 * (m + 1) as f32,
                    0.57 + 0.43 * k as f32,
                )
                .inset(8.0, 2.0, 8.0, 2.0);
            path_panel(c, rect, legs, run);
            let leaving = legs[1][SAMPLES - 1];
            let axis = if k == 0 {
                polarizer_axis()
            } else {
                mirror() >> polarizer_axis()
            };
            let passed = reading(leaving, axis).powi(2);
            let label = format!("{}{}: PASSES {:.0}%", media[m], setups[k], 100.0 * passed);
            label_under(c, rect, &label, 14.0, 0.8);
        }
    }
    caption(
        c,
        "OPTICAL ROTATION, OUT AND BACK",
        "QUARTZ TURNS BY ITS TRAVEL (INTRINSIC), THE FIELD BY ITS FLUX: THE FIELD MAKES AN ISOLATOR (STA)",
    );
}

fn main() {
    run_with(
        Anim::new("optical rotation", SECONDS).size(960, 540),
        isolator,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::rotation::*;
    use gax::ApproxEq;

    /// numga's test: by Malus's law the polarizer each setup's light entered through passes
    /// all of the light quartz returns, and none of what the magnetic field returns.
    #[test]
    fn isolator_and_its_mirror_image() {
        let fields = isolator();
        let axes = [polarizer_axis(), mirror() >> polarizer_axis()];
        for (medium, expected) in fields.iter().zip([1.0, 0.0]) {
            for (setup, axis) in medium.iter().zip(axes) {
                let passed = reading(setup[1][SAMPLES - 1], axis).powi(2);
                assert!((passed - expected).abs() < 1e-12, "{passed}");
            }
        }
    }

    /// numga's scenario checks: quartz returns the light as it entered; the magnetic field
    /// returns it crossed with the polarizer's axis; the mirrored setup is the mirror image of
    /// the original at every depth on both legs.
    #[test]
    fn the_mirrored_setup_mirrors_every_result() {
        let fields = isolator();
        let entering = field_along(polarizer_axis());
        assert!(fields[0][0][1][SAMPLES - 1].max_abs_diff(&entering) < 1e-12);
        assert!(reading(fields[1][0][1][SAMPLES - 1], polarizer_axis()).abs() < 1e-12);
        for medium in &fields {
            for (original, mirrored) in medium[0].iter().zip(&medium[1]) {
                for (a, b) in original.iter().zip(mirrored) {
                    assert!((mirror() >> *a).max_abs_diff(b) < 1e-12);
                }
            }
        }
    }

    /// The field along a unit direction pairs to one with the plane it sweeps, and its
    /// polarization is the direction; the quartz turns by 45 degrees on the way out.
    #[test]
    fn fields_and_polarizations() {
        let axis = polarizer_axis();
        assert!((reading(field_along(axis), axis) - 1.0).abs() < 1e-15);
        assert!(polarization(field_along(axis)).max_abs_diff(&axis) < 1e-15);
        let fields = isolator();
        let out = fields[0][0][0][SAMPLES - 1];
        assert!((reading(out, axis).powi(2) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &isolator(), 3.0);
    }
}
