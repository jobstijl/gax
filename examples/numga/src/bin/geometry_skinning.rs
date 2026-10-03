//! numga's `geometry/skinning`: linear blend skinning in PGA3D, three ways. A bone's transform is
//! a motor `m`, and its action on points is the map `m >> Point`. Matrix skinning blends the
//! bones' point maps and applies the blend; motor skinning blends the motors themselves and
//! renormalizes (dual quaternion blending), or follows the geodesic between them (the log of the
//! relative motor, scaled and exponentiated). In gax the maps and the motors are the same kind of
//! object, so each blend is one line, and the well-known artefact of the matrix blend, the skin
//! collapsing under a twist, is one number. The animation twists the second bone back and forth
//! up to 150 degrees: both motor blends keep the cylinder's radius, the matrix blend pinches it
//! towards `cos 75°` in the middle.

use gax::Unit;
use gax::pga3d::{Line, Motor, Point};
use gax_numga_examples::canvas::mix;
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Scene3, backdrop, caption, colormap, palette, run,
};

mod skinning {
    use super::*;

    pub type P = Point<(), f64>;
    pub type M = Unit<Motor<(), f64>>;

    /// The three skins: normalized motor blending, motor slerp, and map (matrix) blending, each
    /// vertex weighted `w` towards the tip bone.
    pub fn blend_skin(
        vertices: &[P],
        weights: &[f64],
        root: M,
        tip: M,
    ) -> (Vec<P>, Vec<P>, Vec<P>) {
        let (r, t) = (root.into_inner(), tip.into_inner());
        // The bones' point maps: the matrices of computer graphics.
        let root_map = root >> Point::slot();
        let tip_map = tip >> Point::slot();
        // The geodesic from the root to the tip: the log of the relative motor.
        let relative: Line<(), f64> = (tip * root.inverse()).log();
        let mut motor_skin = Vec::with_capacity(vertices.len());
        let mut slerp_skin = Vec::with_capacity(vertices.len());
        let mut matrix_skin = Vec::with_capacity(vertices.len());
        for (v, &w) in vertices.iter().zip(weights) {
            // Blend the motors and renormalize: a chord of the geodesic, still rigid.
            motor_skin.push((r + (t - r).gp(w)).normalized() >> *v);
            // Follow the geodesic: rigid, with the twist spread evenly along the blend.
            slerp_skin.push((relative.gp(w).exp() * root) >> *v);
            // Blend the maps: no longer a motion, so the skin shrinks where the bones disagree.
            matrix_skin.push((root_map + (tip_map - root_map).gp(w)).of(*v));
        }
        (motor_skin, slerp_skin, matrix_skin)
    }

    /// The bone axis, the x axis: the join of the origin and a point along x.
    pub fn bone_axis() -> Line<(), f64> {
        Point::xyz(0.0, 0.0, 0.0) & Point::xyz(1.0, 0.0, 0.0)
    }

    /// The distance of a vertex from the bone axis: the norm of the plane joining them.
    pub fn radius(v: P) -> f64 {
        (v.unitized() & bone_axis()).norm()
    }

    /// A unit-radius skin around the x axis for `x` from 0 to 1, ring by ring, and each vertex's
    /// weight of the second bone, its `x`.
    pub fn cylinder(rings: usize, around: usize) -> (Vec<P>, Vec<f64>) {
        let mut skin = Vec::with_capacity(rings * around);
        let mut weights = Vec::with_capacity(rings * around);
        for i in 0..rings {
            let x = i as f64 / (rings - 1) as f64;
            for j in 0..around {
                let turn = core::f64::consts::TAU * j as f64 / around as f64;
                // The skin at y = 1, turned about the x axis.
                skin.push(Motor::rotation_about(1.0, 0.0, 0.0, turn) >> Point::xyz(x, 1.0, 0.0));
                weights.push(x);
            }
        }
        (skin, weights)
    }

    /// The second bone, turned by `angle` about the shared axis.
    pub fn twist(angle: f64) -> M {
        Motor::rotation_about(1.0, 0.0, 0.0, angle)
    }

    pub const RINGS: usize = 12;
    pub const AROUND: usize = 24;

    /// The scene: the cylinder skinned three ways with the tip bone turned by `angle`.
    pub fn skinning(angle: f64) -> [Vec<P>; 3] {
        let (skin, weights) = cylinder(RINGS, AROUND);
        let root = Motor::<(), f64>::translation(0.0, 0.0, 0.0);
        let (motor, slerp, matrix) = blend_skin(&skin, &weights, root, twist(angle));
        [motor, slerp, matrix]
    }
}

use skinning::*;

const TITLES: [&str; 3] = [
    "MOTOR BLEND (NORMALIZED LERP)",
    "MOTOR BLEND (SLERP)",
    "MATRIX BLEND",
];

fn xyz(p: P) -> [f32; 3] {
    let [x, y, z] = p.to_euclidean();
    [x as f32, y as f32, z as f32]
}

/// One skin as a lit quad mesh, striped around its length, with the bone axis inside.
fn panel(c: &mut Canvas, skin: &[P], angle: f64, azimuth: f32, title: &str) {
    backdrop(c);
    let cam = Camera::orbit(
        c.width,
        c.height,
        [0.5, 0.0, 0.0],
        6.2,
        azimuth,
        0.45,
        Lens::Perspective(0.55),
    );
    let mut s = Scene3::new(cam);
    // The skin, drawn here rather than through the scene: one polygon per quad (the scene
    // splits quads in two, and each triangle's fill scans the quad's whole box), far to near,
    // leaving out the quads that face away (their outward normal, radial from the bone axis,
    // points away from the eye). The bones go under it, seen faintly through.
    let at = |i: usize, j: usize| xyz(skin[i * AROUND + j % AROUND]);
    let eye = cam.eye();
    let mut quads = Vec::with_capacity((RINGS - 1) * AROUND);
    for i in 0..RINGS - 1 {
        for j in 0..AROUND {
            let q = [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)];
            let mid = [0, 1, 2].map(|k| q.iter().map(|p| p[k]).sum::<f32>() / 4.0);
            let to_eye = [0, 1, 2].map(|k| eye[k] - mid[k]);
            if mid[1] * to_eye[1] + mid[2] * to_eye[2] < 0.0 {
                continue;
            }
            let colour = s.lit(
                q[0],
                q[1],
                q[2],
                colormap::viridis(j as f32 / AROUND as f32),
            );
            quads.push((cam.local(mid)[2], q, colour));
        }
    }
    quads.sort_by(|a, b| b.0.total_cmp(&a.0));
    // The bones along the axis, and each bone's frame: the root's y axis, and the tip's, turned.
    s.seg(
        [-0.25, 0.0, 0.0],
        [1.25, 0.0, 0.0],
        2.0,
        palette::ink(),
        0.8,
    );
    s.arrow(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.5],
        2.0,
        8.0,
        palette::orange(),
    );
    let up = xyz(twist(angle) >> Point::xyz(1.0, 0.0, 1.5));
    s.arrow(
        [1.0, 0.0, 0.0],
        [up[0] - 1.0, up[1], up[2]],
        2.0,
        8.0,
        palette::orange(),
    );
    s.draw(c);
    for (_, q, colour) in quads {
        let px: Option<Vec<_>> = q.iter().map(|p| cam.px(*p)).collect();
        if let Some(px) = px {
            c.fill(&px, colour, 0.92);
            c.polyline(&px, 0.6, palette::bottom(), 0.6, true);
        }
    }
    let size = (c.height as f32 / 30.0).clamp(8.0, 13.0);
    let mid = c.width as f32 * 0.5;
    c.text(
        title,
        mid,
        c.height as f32 * 0.17,
        size,
        palette::ink(),
        Align::Center,
    );
    let least = skin.iter().map(|v| radius(*v)).fold(f64::MAX, f64::min);
    let colour = mix(palette::red(), palette::green(), least as f32);
    c.text(
        &format!("MIN RADIUS {least:.3}"),
        mid,
        c.height as f32 * 0.92,
        size,
        colour,
        Align::Center,
    );
}

fn draw(c: &mut Canvas, t: f32) {
    let phase = f64::from(t) / SECONDS * core::f64::consts::TAU;
    // The twist rises to 150 degrees and returns.
    let angle = 150f64.to_radians() * 0.5 * (1.0 - phase.cos());
    let azimuth = 0.9 + 0.25 * (phase as f32).sin();
    let skins = skinning(angle);
    let w = c.width / 3;
    for (i, (skin, title)) in skins.iter().zip(TITLES).enumerate() {
        let mut sub = Canvas::new(w, c.height);
        panel(&mut sub, skin, angle, azimuth, title);
        c.blit(&sub, i * w, 0);
    }
    caption(
        c,
        "SKINNING: BLEND THE MOTORS, OR BLEND THEIR MAPS",
        &format!(
            "A CYLINDER ON TWO BONES, THE TIP TWISTED {:.0} DEG",
            angle.to_degrees()
        ),
    );
}

const SECONDS: f64 = 6.0;

fn main() {
    run(Anim::new("skinning", SECONDS as f32).size(960, 400), draw);
}

#[cfg(test)]
mod tests {
    use super::skinning::*;

    /// numga's scenario checks: both motor blends are rigid about the axis, so every vertex
    /// keeps its unit radius.
    #[test]
    fn motor_blends_keep_the_radius() {
        let [motor, slerp, _] = skinning(150f64.to_radians());
        for v in motor.iter().chain(&slerp) {
            assert!((radius(*v) - 1.0).abs() < 1e-12, "{}", radius(*v));
        }
    }

    /// The matrix blend of the identity and a turn by `θ` scales the radial part by
    /// `|(1 - w) + w e^{iθ}|`, which is `cos(θ / 2)` halfway: numga prints the minimum radius
    /// against `cos 75°`.
    #[test]
    fn matrix_blend_collapses_as_the_formula_says() {
        let theta = 150f64.to_radians();
        let (skin, weights) = cylinder(RINGS, AROUND);
        let [_, _, matrix] = skinning(theta);
        let mut least = f64::MAX;
        for ((v, w), m) in skin.iter().zip(&weights).zip(&matrix) {
            let expected =
                ((1.0 - w) * (1.0 - w) + w * w + 2.0 * w * (1.0 - w) * theta.cos()).sqrt();
            assert!((radius(*m) - expected).abs() < 1e-12);
            // It stays on its ring: x is unchanged.
            assert!((m.to_euclidean()[0] - v.to_euclidean()[0]).abs() < 1e-12);
            least = least.min(radius(*m));
        }
        // No ring sits exactly halfway (the rings are at k / 11), so the least radius is a
        // little above cos 75°.
        let half = 75f64.to_radians().cos();
        assert!(least >= half - 1e-12 && least < half + 0.02, "{least}");
    }

    /// At the ends of the blend the three skins agree: the root and the tip bone.
    #[test]
    fn the_blends_agree_at_the_bones() {
        let [a, b, c] = skinning(2.0);
        for k in (0..AROUND).chain((RINGS - 1) * AROUND..RINGS * AROUND) {
            let (x, y, z) = (
                a[k].to_euclidean(),
                b[k].to_euclidean(),
                c[k].to_euclidean(),
            );
            for i in 0..3 {
                assert!((x[i] - y[i]).abs() < 1e-12 && (x[i] - z[i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
