//! numga's `relativity/twistors`: light rays carried by twistors, and a twisting family of rays
//! filling spacetime, in the conformal algebra of spacetime `Cl(4,2)`.
//!
//! The algebra has spacetime's directions `x y z t` (here `t` squares to minus one) and the
//! conformal pair `u`, `v` (`u` to plus one, `v` to minus one); gax does not ship it in this
//! basis, so it is declared below with `gax::algebra!` (gax's `csta` is the same algebra with a
//! null pair for `u` and `v`). An event `e` is the null conformal vector
//! `o + e + e² ∞ / 2`, with the origin `o = (v - u) / 2` and infinity `∞ = v + u`.
//!
//! Twistors are this algebra's spinors: three commuting idempotent factors,
//! `P = (1 + x)(1 + y t)(1 + z v) / 8`, cut its 64 dimensions to a left ideal of eight real
//! components, the multiples `ψ P` of the subalgebra of `y`, `z` and `u`. Every multivector acts
//! on twistors by multiplication from the left, `ψ ↦ 8 ⟨a ψ P⟩_twistor`: a map on twistors,
//! `Twistor<(Twistor,)>`, built from the images of the basis twistors. The conformal volume
//! acts as a complex structure (it squares to minus one and commutes with every even element),
//! and a pairing of two twistors, `8 ⟨y z (ψ P)~ I (φ P)⟩`, is kept by every conformal rotor.
//!
//! An event's action on twistors has a kernel: the twistors of the light rays through it. Two
//! null-separated events share one ray, the image of the action of their product (by
//! `svd`). From a twistor, the ray's null plane is a bilinear map of twistors,
//! `Bivector<(Twistor, Twistor)>`: the plane whose scalar product with every bivector `B` is the
//! pairing of the twistor with the volume times `B`'s action on it. Its meet with infinity is
//! the ray's direction, and with a slice of constant time, where the ray crosses it.
//!
//! A fixed twistor that is not null picks one light ray at every event, the one in the kernel
//! of the event's action on it: the Robinson congruence, straight rays twisting around each
//! other. At time zero its flow lines are the linked circles of the Hopf fibration on nested
//! tori (computed here in VGA3D, by the stereographic projection of rotors). A null
//! electromagnetic field, the Hopfion, is carried by the congruence: its field lines, moved
//! along the straight rays, stay field lines.
//!
//! The animation: the Hopfion's electric field lines carried along the Robinson rays through
//! time (left), the congruence's flow lines at time zero (top right), and a light ray recovered
//! from its twistor through two events, at rest and boosted (bottom right; `x`, `z` and `t`).
//!
//! `twistors.rs` holds the algebra and the twistors' geometry, `scenes.rs` numga's scenes.

mod scenes;
mod twistors;

use std::sync::OnceLock;

use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Light, Marker, ORIGIN3, Point2, Point3, Rect, backdrop,
    caption, colormap, palette, run, scene3::panel3,
};
use scenes::{
    FIBRE_SAMPLES, Incidence, PER_CIRCLE, POLARS, electric_turn, fibres, incidence, turned,
};
use twistors::{Event, Spatial, at, robinson, spatial};

/// One loop: the field lines carried from time `-1.25` to `1.25`.
const SECONDS: f32 = 8.0;
const FIRST_TIME: f64 = -1.25;
const LAST_TIME: f64 = 1.25;
/// Curves are broken where they leave a ball of this radius about the origin.
const LIMIT: f64 = 3.8;
/// The share of a turn the spatial views turn through in one loop.
const VIEW_TURN: f32 = 0.4;

/// What the frames share: the congruence's flow lines, the Hopfion's electric lines at time zero
/// with the Robinson direction at each of their points, and the incidence.
struct Scene {
    flow: Vec<Vec<Spatial>>,
    electric: Vec<Vec<(Event, Event)>>,
    incidence: Vec<Incidence>,
}

fn scene() -> &'static Scene {
    static SCENE: OnceLock<Scene> = OnceLock::new();
    SCENE.get_or_init(|| {
        let flow = fibres(&POLARS, PER_CIRCLE, FIBRE_SAMPLES);
        let electric = turned(&flow, electric_turn())
            .iter()
            .map(|curve| {
                curve
                    .iter()
                    .map(|p| {
                        let e = at(*p, 0.0);
                        (e, robinson(e))
                    })
                    .collect()
            })
            .collect();
        Scene {
            flow,
            electric,
            incidence: incidence(),
        }
    })
}

/// A point of space to draw, or a break (a non-finite point) beyond the shown ball.
fn shown(p: Spatial) -> Point3 {
    if p.norm() > LIMIT {
        Point3::xyz(f32::NAN, f32::NAN, f32::NAN)
    } else {
        Point3::xyz(p.e1() as f32, p.e2() as f32, p.e3() as f32)
    }
}

/// One colour per curve, kept through every frame.
fn hue(k: usize, count: usize) -> Light {
    colormap::hsv(k as f32 / count as f32).faded(0.7)
}

/// The fixed spatial view of the field lines and the flow lines in `rect`, `half` units from
/// the middle to the top, turned about `z` by `turn`.
fn spatial_camera(rect: Rect, turn: f32, half: f32) -> Camera {
    let azimuth = (-55f32).to_radians() + turn;
    Camera::orbit(
        rect,
        ORIGIN3,
        20.0,
        azimuth,
        22f32.to_radians(),
        Lens::Parallel(half),
    )
}

/// Curves of space in a 3D panel, each in its own colour.
fn draw_curves(c: &mut Canvas, rect: Rect, cam: Camera, curves: &[Vec<Point3>]) {
    panel3(c, rect, cam, |s| {
        for (k, curve) in curves.iter().enumerate() {
            s.polyline(curve, 1.0, hue(k, curves.len()));
        }
    });
}

/// A label at the bottom of a panel.
fn label(c: &mut Canvas, panel: Rect, text: &str, size: f32, colour: Light) {
    let at = panel.bottom_middle() - Point2::direction(0.0, size * 0.8);
    c.text(text, at, size, colour, Align::Center);
}

/// A light ray and its two events in `x`, `z` and `t`, drawn as space's three axes.
fn draw_incidence(c: &mut Canvas, rect: Rect, incidence: &Incidence) {
    let drawn = |e: Event| Point3::xyz(e.ex() as f32, e.ez() as f32, e.et() as f32);
    let (azimuth, elevation) = ((-68f32).to_radians(), 15f32.to_radians());
    let cam = Camera::orbit(rect, ORIGIN3, 20.0, azimuth, elevation, Lens::Parallel(2.3));
    let reach = 1.6;
    let axes = [
        Point3::direction(reach, 0.0, 0.0),
        Point3::direction(0.0, reach, 0.0),
        Point3::direction(0.0, 0.0, reach),
    ];
    panel3(c, rect, cam, |s| {
        for axis in axes {
            s.arrow(
                ORIGIN3 - axis,
                axis.gp(2.0),
                1.0,
                6.0,
                palette::grid().faded(2.0),
            );
        }
        let ray: Vec<Point3> = incidence.ray.iter().map(|e| drawn(*e)).collect();
        s.polyline(&ray, 1.8, palette::sky());
        for e in incidence.events {
            s.dot(drawn(e), Marker::Dot, 8.0, palette::orange());
        }
    });
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let s = scene();
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let share = f64::from(t / SECONDS);
    let time = FIRST_TIME + (LAST_TIME - FIRST_TIME) * share;
    let turn = VIEW_TURN * (t / SECONDS) * core::f32::consts::TAU;
    let size = (h / 42.0).clamp(6.0, 12.0);

    // The field lines, each point carried along its straight light ray.
    let lines: Vec<Vec<Point3>> = s
        .electric
        .iter()
        .map(|curve| {
            curve
                .iter()
                .map(|(e, d)| shown(spatial(*e + *d * time)))
                .collect()
        })
        .collect();
    let left = Rect::new(0.0, h * 0.12, w * 0.58, h);
    draw_curves(c, left, spatial_camera(left, turn, 2.6), &lines);
    let readout = format!("HOPFION: ELECTRIC LINES AT T = {time:+.2}");
    label(c, left, &readout, size, palette::yellow());

    let flow: Vec<Vec<Point3>> = s
        .flow
        .iter()
        .map(|curve| curve.iter().map(|p| shown(*p)).collect())
        .collect();
    let top_right = Rect::new(w * 0.58, h * 0.12, w, h * 0.62);
    draw_curves(c, top_right, spatial_camera(top_right, turn, 2.4), &flow);
    label(
        c,
        top_right,
        "ROBINSON CONGRUENCE AT T = 0",
        size,
        palette::ink(),
    );

    let bottom = Rect::new(w * 0.58, h * 0.62, w, h);
    for (k, (incidence, name)) in s.incidence.iter().zip(["AT REST", "BOOSTED"]).enumerate() {
        let panel = bottom.column(k, 2).inset(0.0, 0.0, 0.0, size * 1.6);
        draw_incidence(c, panel, incidence);
        label(c, panel, name, size * 0.8, palette::ink());
    }
    label(
        c,
        bottom,
        "A RAY FROM ITS TWISTOR (X, Z, T)",
        size,
        palette::ink(),
    );
    caption(
        c,
        "TWISTORS: LIGHT RAYS AND THE ROBINSON CONGRUENCE",
        "CL(4,2) SPINORS: A RAY IS THE KERNEL OF ITS EVENTS' ACTION; THE HOPFION RIDES THE RAYS",
    );
}

fn main() {
    run(Anim::new("twistors", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
