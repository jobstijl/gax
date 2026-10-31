//! numga's `optics/black_sandwich`: a half-wave plate between crossed polarizers, a nonzero
//! optical map that squares to zero, in numga's spacetime algebra `x+y+z+t-`.
//!
//! The electric field of the light is an extrinsic bivector of spacetime, and the axes of the
//! polarizers and plates are directions, intrinsic trivectors (numga's antivectors, ADR-051).
//! Each axis sweeps a plane through the observer's time, `axis & time`; a field's component
//! along the axis is its pairing with that plane, and the field along a direction is the Hodge
//! dual (numga's `dual`) of the plane it sweeps. A polarizer is then a map from fields to
//! fields, `field_along(axis) * ((axis & time) & Field::slot())`, a `Bivector<(Bivector,)>`; a
//! half-wave plate keeps the field along its fast axis and reverses the field across it, twice
//! its polarizer less the identity. One stack, entrance polarizer, plate and crossed exit
//! polarizer, is their composition, `exit.of(plate.of(entrance))`.
//!
//! With the plate at 45 degrees one stack passes the whole field, turned onto the exit axis.
//! Two stacks in succession pass nothing, for every plate and every input: the stack is a
//! square root of zero (Berry and Dennis, "Black polarization sandwiches are square roots of
//! zero"). In Jones calculus the stack is `P_y H P_x`, a 2x2 matrix with a single entry off the
//! diagonal. Reflecting every component moves the whole map, `mirror >> stack.of(mirror <<
//! Field::slot())`, the guide's "Moving a map".
//!
//! The animation: light passing from left to right through two identical stacks while their
//! plates swing from the entrance axis to ninety degrees and back. The polarizers' axes are
//! dark lines, the plates' orange; above the beam is the field between components, and the
//! beam's strength follows the intensity. The second stack stays dark throughout.

use gax::ApproxEq;
use gax_numga_examples::signal::frame;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Point2, Rect, backdrop, caption, palette, run_with,
};

#[path = "../shared/spacetime.rs"]
mod spacetime;

mod sandwich {
    use super::spacetime::{Bivector, Direction, Vector, space, time};
    use gax_numga_examples::signal::haversine;

    /// The electric field: an extrinsic bivector.
    pub type Field = Bivector;
    /// A map from fields to fields.
    pub type OpticalMap = Bivector<(Bivector,)>;

    /// The field along a direction, the Hodge dual of the plane the direction sweeps in time:
    /// the vacuum's identification of a direction with the field along it, which pairs to one
    /// with that plane for a unit direction.
    pub fn field_along(d: Direction) -> Field {
        (d & time()).hodge()
    }

    /// The field's pairing with the plane a direction sweeps in time: its component along the
    /// direction.
    pub fn along(field: Field, d: Direction) -> f64 {
        (field & (d & time())).s()
    }

    /// The field along one direction, weighted by the field's pairing with the plane another
    /// sweeps in time; for a unit axis twice, the polarizer that transmits the field's
    /// component along the axis.
    pub fn polarizer(passing: Direction, reading: Direction) -> OpticalMap {
        field_along(passing) * ((reading & time()) & Field::slot())
    }

    /// The half-wave plate along `axis`: it keeps the field along its fast axis and reverses the
    /// field across it, up to a common phase that does not affect the intensity.
    pub fn plate(axis: Direction) -> OpticalMap {
        polarizer(axis, axis) * 2.0 - Field::slot()
    }

    /// The field before and after each component of two identical stacks, `[stages]`.
    pub fn through_pair(components: [OpticalMap; 3], incident: Field) -> [Field; 7] {
        let mut fields = [incident; 7];
        for k in 0..6 {
            fields[k + 1] = components[k % 3].of(fields[k]);
        }
        fields
    }

    /// The crossed polarizers' axes: the entrance along `x`, the exit along `y`.
    pub fn input_axis() -> Direction {
        space()[0]
    }
    pub fn output_axis() -> Direction {
        space()[1]
    }

    /// The frames of the plates' swing out, which the animation plays forward and back.
    pub const FRAMES: usize = 108 / 2 + 1;

    /// One frame of the swing: the plate's fast axis, the stack's map, and the fields.
    pub struct Frame {
        pub plate_axis: Direction,
        pub stack: OpticalMap,
        pub fields: [Field; 7],
    }

    /// The plate's fast axis turned from the entrance axis by `angle` in `xy`.
    pub fn plate_axis(angle: f64) -> Direction {
        let xy = Bivector::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        xy.gp(-angle / 2.0).exp() >> input_axis()
    }

    /// The stack of entrance, plate and exit, for a plate along `axis`, moved by a mirror.
    pub fn components(axis: Direction, mirror: Option<Vector>) -> [OpticalMap; 3] {
        let moved = |d: Direction| mirror.map_or(d, |m| m >> d);
        let (input, output) = (moved(input_axis()), moved(output_axis()));
        [
            polarizer(input, input),
            plate(moved(axis)),
            polarizer(output, output),
        ]
    }

    /// numga's scenario: the plates swung from the entrance axis to ninety degrees, easing in
    /// and out, the incident field along the entrance axis.
    pub fn rotating_plate() -> Vec<Frame> {
        (0..FRAMES)
            .map(|k| {
                let cycle = core::f64::consts::PI * k as f64 / (FRAMES - 1) as f64;
                let angle = haversine(cycle) * core::f64::consts::FRAC_PI_2;
                let axis = plate_axis(angle);
                let [entrance, plate, exit] = components(axis, None);
                Frame {
                    plate_axis: axis,
                    stack: exit.of(plate.of(entrance)),
                    fields: through_pair([entrance, plate, exit], field_along(input_axis())),
                }
            })
            .collect()
    }
}

use sandwich::*;
use spacetime::Direction;

/// numga's animation: 55 frames of 40 ms forward, then back.
const SECONDS: f32 = 2.0 * FRAMES as f32 * 0.04;
/// Where the components and the fields between them sit along the beam, and the height of the
/// fields' glyphs above it; the components' radius.
const COMPONENTS: [f32; 6] = [1.2, 3.2, 5.2, 8.2, 10.2, 12.2];
const FIELDS: [f32; 7] = [0.1, 2.2, 4.2, 6.7, 9.2, 11.2, 13.7];
const FIELD_HEIGHT: f32 = 1.25;
const RADIUS: f32 = 0.42;

/// A direction across the beam, in the panel's plane: its pairings with the slices `x` and `y`.
fn across(d: Direction) -> Point2 {
    let [x, y, _, _] = spacetime::coordinates(d);
    Point2::direction(x as f32, y as f32)
}

/// A field across the beam: its components along `x` and `y`.
fn field_across(f: Field) -> Point2 {
    let [ax, ay, _] = spacetime::space();
    Point2::direction(along(f, ax) as f32, along(f, ay) as f32)
}

/// A line across the middle of a glyph at `centre`, either way along `half`.
fn glyph(ax: &Axes, c: &mut Canvas, centre: Point2, half: Point2, width: f32, l: Light) {
    ax.line(c, centre - half, centre + half, width, l);
}

fn draw(c: &mut Canvas, frames: &[Frame], t: f32) {
    backdrop(c);
    let screen = c.rect();
    // Forward, then back.
    let phase = 1.0 - (2.0 * t / SECONDS - 1.0).abs();
    let now = &frames[frame(phase, frames.len())];
    let rect = Rect::new(
        0.0,
        screen.height() * 0.2,
        screen.width(),
        screen.height() * 0.8,
    );
    let ax = Axes::new(rect, [-0.5, 14.3], [-1.2, 2.3]);
    // The beam between components, its strength the field's square size there.
    let bounds: Vec<f32> = [FIELDS[0] - 0.3]
        .into_iter()
        .chain(COMPONENTS)
        .chain([FIELDS[6] + 0.3])
        .collect();
    for (k, field) in now.fields.iter().enumerate() {
        let [a, b] = [bounds[k], bounds[k + 1]].map(|x| Point2::xy(x, 0.0));
        ax.line(c, a, b, 1.0, palette::GRID.faded(2.0));
        let strength = field_across(*field).ideal_norm().powi(2).min(1.0);
        if strength > 1e-6 {
            ax.line(c, a, b, 4.0, palette::SKY.faded(strength));
        }
    }
    let scale = ax.scale();
    for (k, x) in COMPONENTS.into_iter().enumerate() {
        let centre = Point2::xy(x, 0.0);
        c.disk(ax.px(centre), RADIUS * scale, palette::BOTTOM);
        c.ring(ax.px(centre), RADIUS * scale, 1.0, palette::GRID.faded(4.0));
        let (axis, colour) = match k % 3 {
            0 => (input_axis(), palette::INK),
            1 => (now.plate_axis, palette::ORANGE),
            _ => (output_axis(), palette::INK),
        };
        glyph(&ax, c, centre, across(axis).gp(0.8 * RADIUS), 2.2, colour);
    }
    for (x, field) in FIELDS.into_iter().zip(&now.fields) {
        let centre = Point2::xy(x, FIELD_HEIGHT);
        c.ring(ax.px(centre), 0.38 * scale, 0.8, palette::GRID.faded(2.5));
        let half = field_across(*field).gp(0.34);
        if half.ideal_norm() > 1e-4 {
            glyph(&ax, c, centre, half, 2.2, palette::SKY);
        }
    }
    let passed = along(now.fields[3], output_axis()).powi(2);
    let twice = now.stack.of(now.stack);
    let notes = [
        format!("ONE STACK PASSES {:3.0}%", 100.0 * passed),
        format!(
            "TWO PASS NONE: STACK OF STACK = {:.0e}",
            twice.max_abs_diff(&OpticalMap::zero())
        ),
    ];
    for (note, x) in notes.iter().zip([3.2f32, 10.2]) {
        let at = ax.px(Point2::xy(x, -0.95));
        c.text(note, at, 13.0, palette::YELLOW, Align::Center);
    }
    caption(
        c,
        "BLACK POLARIZATION SANDWICHES",
        "A HALF-WAVE PLATE BETWEEN CROSSED POLARIZERS: A MAP OF FIELDS THAT SQUARES TO ZERO (STA)",
    );
}

fn main() {
    run_with(
        Anim::new("black sandwich", SECONDS).size(960, 540),
        rotating_plate,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::sandwich::*;
    use super::spacetime::{Vector, time, x, y};
    use gax::ApproxEq;

    const ROUND_OFF: f64 = 1e-12;

    /// numga's first test: the plate is its own inverse; one stack passes the field turned by
    /// twice the plate's angle, read along the exit axis, at most all of it; the stack is the
    /// dyad from the entrance to the exit axis times the strength the plate passes; two stacks
    /// pass nothing, and the second stack's fields are zero.
    #[test]
    fn half_wave_transmission_and_nilpotence() {
        let frames = rotating_plate();
        let dyad = field_along(output_axis()) * ((input_axis() & time()) & Field::slot());
        let mut largest = 0.0f64;
        for f in &frames {
            let plate = plate(f.plate_axis);
            assert!(plate.of(plate).max_abs_diff(&Field::slot()) < ROUND_OFF);
            let passed = along(f.fields[3], output_axis());
            let expected = 2.0 * (x() & f.plate_axis).s() * (y() & f.plate_axis).s();
            assert!((passed - expected).abs() < ROUND_OFF);
            largest = largest.max(passed);
            let strength = along(plate.of(field_along(input_axis())), output_axis());
            assert!(f.stack.max_abs_diff(&(dyad * strength)) < ROUND_OFF);
            assert!(f.stack.of(f.stack).max_abs_diff(&OpticalMap::zero()) < ROUND_OFF);
            for field in &f.fields[4..] {
                assert!(field.max_abs_diff(&Field::zero()) < ROUND_OFF);
            }
        }
        assert!((largest - 1.0).abs() < ROUND_OFF, "{largest}");
    }

    /// numga's second test: reflecting every component in a mirror moves the whole map, and
    /// the fields through the moved stacks are the mirror images of the fields.
    #[test]
    fn moving_the_components_moves_the_whole_map() {
        let mirror = Vector::new(0.6, 0.8, 0.0, 0.0);
        for f in rotating_plate() {
            let moved = components(f.plate_axis, Some(mirror));
            let [entrance, plate, exit] = moved;
            let rebuilt = exit.of(plate.of(entrance));
            let reflected = mirror >> f.stack.of(mirror << Field::slot());
            assert!(rebuilt.max_abs_diff(&reflected) < ROUND_OFF);
            let incident = field_along(mirror >> input_axis());
            for (a, b) in through_pair(moved, incident).iter().zip(&f.fields) {
                assert!(a.max_abs_diff(&(mirror >> *b)) < ROUND_OFF);
            }
            assert!(rebuilt.of(rebuilt).max_abs_diff(&OpticalMap::zero()) < ROUND_OFF);
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &rotating_plate(), 1.0);
    }
}
