//! numga's `quantum/process_tomography`: learning a qubit's noisy gate from prepared states and
//! measured probabilities, in the Pauli algebra of space (VGA3D). A state is a scalar plus a
//! vector, `(1 + r) / 2` (VGA3D's `Paravector`); its scalar part keeps the normalization, so rotations and relaxation toward a
//! preferred state are linear maps on states. Unobserved alternatives act by sandwiches whose
//! results add: a noisy gate is a sum of sandwiches (Kraus operators), a map `Paravector <- Paravector`.
//!
//! Four preparations spanning the states determine such a map. A measurement reads a state by
//! its scalar product with an effect, and a dual frame undoes the overlap between these readouts:
//! the outputs are reconstructed from their probabilities, then paired with the dual
//! preparations, which gives the channel as a sum of dyads with the input left open.
//!
//! The animation applies each learned channel (a rotation; with phase noise; with relaxation
//! toward the north pole) over and over to the sphere of pure states: the rotation keeps the
//! sphere, dephasing flattens it toward the axis, relaxation shrinks it onto the north pole.
//! Below, the probabilities the learned maps predict after as many uses, for states the
//! reconstruction never saw, against the exact ones, and the measured probability tables.

use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Marker, Rgb, backdrop, caption, colormap, palette, run,
};
use std::sync::OnceLock;

mod tomography {
    use gax::vga3d::{Bivector, Multivector, Paravector, Vector};

    /// A state `(1 + r) / 2`, and a channel: a map on states.
    pub type St = Paravector<(), f64>;
    pub type Channel = Paravector<(Paravector,), f64>;
    pub type Bi = Bivector<(), f64>;
    /// A Bloch vector.
    pub type Space = Vector<(), f64>;

    pub fn one() -> St {
        Paravector::new(1.0, 0.0, 0.0, 0.0)
    }
    pub fn x() -> Space {
        Vector::new(1.0, 0.0, 0.0)
    }
    pub fn z() -> Space {
        Vector::new(0.0, 0.0, 1.0)
    }
    pub fn xy() -> Bi {
        Bivector::new(0.0, 0.0, 1.0)
    }

    /// The open state.
    fn open() -> Channel {
        Paravector::slot()
    }

    /// The state with Bloch vector `r`.
    pub fn state(r: Space) -> St {
        (one() + r.cast::<Paravector>()) * 0.5
    }

    /// The Bloch vector of a state: twice its vector part.
    pub fn bloch(rho: St) -> Space {
        rho.cast::<Vector>() * 2.0
    }

    /// The sandwich by one path of a channel, `k rho ~k`. A path may be any multivector whose
    /// sandwich keeps a state (its own reverse) a state, such as a scalar plus a vector plus a
    /// bivector, or a rotor; the other grades of the product are zero and dropped.
    fn sandwich(k: Multivector<(), f64>) -> Channel {
        (k * open() * k.reverse()).cast::<Paravector>()
    }

    /// A coherent turn in `plane` after independent phase noise and relaxation toward the north
    /// pole: each a sum of sandwiches, composed. The noise's alternatives are not observed:
    /// their output states add, not their amplitudes.
    pub fn noisy_gate(angle: f64, phase_flip: f64, loss: f64, plane: Bi) -> Channel {
        // Kept, or flipped in the xy plane; the square roots are amplitudes.
        let keep = one().cast::<Multivector>() * (1.0 - phase_flip).sqrt();
        let flip = xy().cast::<Multivector>() * phase_flip.sqrt();
        let dephasing = sandwich(keep) + sandwich(flip);
        let (north, south) = (state(z()), state(-z()));
        // The first path keeps north and attenuates south; the second carries south to north.
        let stay = (north + south * (1.0 - loss).sqrt()).cast::<Multivector>();
        let fall = x() * south * loss.sqrt();
        let damping = sandwich(stay) + sandwich(fall);
        let rotation = (plane * (-angle / 2.0)).exp();
        // A map applied to another map composes them; the rotor then turns every output.
        sandwich(rotation.cast::<Multivector>()).of(damping.of(dephasing))
    }

    /// The dyad that reads a state by its scalar product with `along` and sends the reading
    /// out along `out`. (The product of a scalar plus a vector with a scalar is a state again;
    /// gax types it as a multivector, so the empty grades are dropped.)
    pub fn dyad(out: St, along: St) -> Channel {
        (out * along.scalar_product(open())).cast::<Paravector>()
    }

    /// The overlap of a set of states: the sum of their dyads, each measuring the overlap with
    /// its state and sending it back along it.
    pub fn overlap(states: &[St]) -> Channel {
        states
            .iter()
            .fold(Channel::zero(), |acc, s| acc + dyad(*s, *s))
    }

    /// Weights that recover any state from its scalar products with a spanning set of states:
    /// the dual frame undoes the overlap.
    pub fn dual_frame(states: &[St]) -> Vec<St> {
        let overlap = overlap(states);
        states.iter().map(|s| overlap.solve(*s)).collect()
    }

    /// Each outcome's probability for each preparation passed through the channel,
    /// `[preparation][outcome]`.
    pub fn probabilities(device: Channel, prepared: &[St], effects: &[St]) -> Vec<Vec<f64>> {
        prepared
            .iter()
            .map(|p| {
                let out = device.of(*p);
                effects
                    .iter()
                    .map(|e| 2.0 * e.scalar_product(out).s())
                    .collect()
            })
            .collect()
    }

    /// The outputs recovered from their probabilities, then the channel from its outputs. The
    /// readout's scalar product includes the Born rule's factor of two.
    pub fn reconstruct(prepared: &[St], effects: &[St], measured: &[Vec<f64>]) -> Channel {
        let twice: Vec<St> = effects.iter().map(|e| *e * 2.0).collect();
        let measurement_dual = dual_frame(&twice);
        let preparation_dual = dual_frame(prepared);
        measured
            .iter()
            .zip(&preparation_dual)
            .fold(Channel::zero(), |acc, (row, dual)| {
                let output = row
                    .iter()
                    .zip(&measurement_dual)
                    .fold(St::zero(), |o, (p, m)| o + *m * *p);
                // Leaving the input open extends the measured action to every state.
                acc + dyad(output, *dual)
            })
    }

    /// One use of the channel, then two, three, and so on, as composed maps.
    pub fn powers(device: Channel, steps: usize) -> Vec<Channel> {
        let mut out = vec![device];
        for _ in 1..steps {
            let last = *out.last().expect("one use");
            out.push(device.of(last));
        }
        out
    }

    /// Pure states on a closed latitude-longitude mesh of the Bloch sphere, `[latitude][longitude]`:
    /// z tilted toward x, then turned about z.
    pub fn sphere(latitudes: usize, longitudes: usize) -> Vec<Vec<St>> {
        (0..=latitudes)
            .map(|i| {
                let inclination = core::f64::consts::PI * i as f64 / latitudes as f64;
                let tilt = ((z() ^ x()) * (-inclination / 2.0)).exp();
                (0..=longitudes)
                    .map(|j| {
                        let azimuth = core::f64::consts::TAU * j as f64 / longitudes as f64;
                        let turn = (xy() * (-azimuth / 2.0)).exp();
                        state((turn * tilt) >> z())
                    })
                    .collect()
            })
            .collect()
    }

    // --- the scenario ---

    pub const LABELS: [&str; 3] = ["ROTATION", "ROTATION + DEPHASING", "ROTATION + RELAXATION"];
    pub const TURNS: [f64; 3] = [0.075, 0.075, 0.075];
    pub const PHASE_FLIP: [f64; 3] = [0.0, 0.06, 0.0];
    pub const LOSS: [f64; 3] = [0.0, 0.0, 0.10];
    pub const STEPS: usize = 48;

    pub fn rotation_plane() -> Bi {
        Bivector::new(1.0, 1.0, 1.0).normalized().into_inner()
    }

    /// The tetrahedral preparations, and the detector's effects: half of them.
    pub fn prepared() -> Vec<St> {
        tetrahedron().into_iter().map(state).collect()
    }

    /// The unit directions to a regular tetrahedron's corners.
    pub fn tetrahedron() -> [Space; 4] {
        [
            Vector::new(1.0, 1.0, 1.0),
            Vector::new(1.0, -1.0, -1.0),
            Vector::new(-1.0, 1.0, -1.0),
            Vector::new(-1.0, -1.0, 1.0),
        ]
        .map(|d| d.normalized().into_inner())
    }
    pub fn effects() -> Vec<St> {
        prepared().iter().map(|p| *p * 0.5).collect()
    }

    /// The three gates as they are.
    pub fn devices() -> Vec<Channel> {
        (0..3)
            .map(|c| noisy_gate(TURNS[c], PHASE_FLIP[c], LOSS[c], rotation_plane()))
            .collect()
    }

    /// The prepared states, their ideal measurement probabilities `[channel][prep][outcome]`, and
    /// the reconstructed maps.
    #[allow(clippy::type_complexity)]
    pub fn run_tomography() -> (Vec<St>, Vec<Vec<Vec<f64>>>, Vec<Channel>) {
        let (prepared, effects) = (prepared(), effects());
        let measured: Vec<Vec<Vec<f64>>> = devices()
            .iter()
            .map(|d| probabilities(*d, &prepared, &effects))
            .collect();
        let learned = measured
            .iter()
            .map(|m| reconstruct(&prepared, &effects, m))
            .collect();
        (prepared, measured, learned)
    }

    /// Pure and mixed states absent from the reconstruction.
    pub fn heldout() -> Vec<St> {
        let axes = [x(), Vector::new(0.0, 1.0, 0.0), z()];
        let lengths = [1.0, 0.7, 0.3, -1.0, -0.7, 0.0];
        axes.iter()
            .cycle()
            .zip(lengths)
            .map(|(d, l)| state(*d * l))
            .collect()
    }

    /// Exact and predicted outcome probabilities for the held-out states, after `uses` uses of
    /// each gate: `[channel]` of `([state][outcome], [state][outcome])`.
    #[allow(clippy::type_complexity)]
    pub fn validation(learned: &[Channel], uses: usize) -> Vec<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let (states, effects) = (heldout(), effects());
        devices()
            .iter()
            .zip(learned)
            .map(|(actual, l)| {
                let actual = powers(*actual, uses)[uses - 1];
                let predicted = powers(*l, uses)[uses - 1];
                (
                    probabilities(actual, &states, &effects),
                    probabilities(predicted, &states, &effects),
                )
            })
            .collect()
    }

    /// The singular values of the tetrahedron's overlap, and of two opposite probes': how many
    /// state directions each determines.
    pub fn completeness() -> [[f64; 4]; 2] {
        let opposite = [state(z()), state(-z())];
        [overlap(&prepared()).svdvals(), overlap(&opposite).svdvals()]
    }
}

use tomography::*;

const SECONDS: f32 = 9.6;
const LATITUDES: usize = 12;
const LONGITUDES: usize = 24;

/// What the frames share, computed once: the experiment, the learned maps and their powers.
struct Data {
    prepared: Vec<St>,
    measured: Vec<Vec<Vec<f64>>>,
    learned: Vec<Channel>,
    powers: Vec<Vec<Channel>>,
    surface: Vec<Vec<St>>,
}

fn data() -> &'static Data {
    static D: OnceLock<Data> = OnceLock::new();
    D.get_or_init(|| {
        let (prepared, measured, learned) = run_tomography();
        let powers = learned.iter().map(|l| powers(*l, STEPS)).collect();
        Data {
            prepared,
            measured,
            learned,
            powers,
            surface: sphere(LATITUDES, LONGITUDES),
        }
    })
}

/// A table of values as coloured cells (a heat map), `[row][column]`, with the values written in
/// and the rows and columns named.
fn table(
    c: &mut Canvas,
    rect: [f32; 4],
    cells: &[Vec<f64>],
    names: &[&str],
    colour: impl Fn(f64) -> Rgb,
) {
    let rows = cells.len() as f32;
    let cols = cells[0].len() as f32;
    let (cw, ch) = ((rect[2] - rect[0]) / cols, (rect[3] - rect[1]) / rows);
    let size = (ch * 0.32).clamp(6.0, 10.0);
    for (i, row) in cells.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            let (x0, y0) = (rect[0] + j as f32 * cw, rect[1] + i as f32 * ch);
            let quad = [[x0, y0], [x0 + cw, y0], [x0 + cw, y0 + ch], [x0, y0 + ch]];
            c.fill(&quad, colour(*v), 1.0);
            c.text(
                // Probabilities, without the leading zero, to fit the cells.
                format!("{v:.2}").trim_start_matches('0'),
                x0 + cw * 0.5,
                y0 + ch * 0.5 + size * 0.5,
                size,
                palette::bottom(),
                Align::Center,
            );
        }
        c.text(
            names[i],
            rect[0] - 4.0,
            rect[1] + (i as f32 + 0.5) * ch + size * 0.5,
            size,
            palette::ink(),
            Align::Right,
        );
    }
    for (j, name) in names.iter().enumerate().take(cols as usize) {
        c.text(
            name,
            rect[0] + (j as f32 + 0.5) * cw,
            rect[3] + size * 1.4,
            size,
            palette::ink(),
            Align::Center,
        );
    }
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let d = data();
    let (w, h) = (c.width, c.height);
    let (wf, hf) = (w as f32, h as f32);
    let phase = (t / SECONDS).clamp(0.0, 0.9999);
    let uses = 1 + (phase * STEPS as f32) as usize;
    let colours = [palette::sky(), palette::orange(), palette::purple()];
    let probe_colours = [
        palette::sky(),
        palette::orange(),
        palette::green(),
        palette::purple(),
    ];
    let small = (hf / 50.0).clamp(7.0, 11.0);

    // Each learned channel, applied `uses` times to the sphere of pure states and to the probes.
    let top = (hf * 0.13) as usize;
    let bottom = (hf * 0.66) as usize;
    let third = w / 3;
    let azimuth = (-54.0f32).to_radians() + 0.5 * (core::f32::consts::TAU * phase).sin();
    for k in 0..3 {
        let map = d.powers[k][uses - 1];
        let image: Vec<Vec<Space>> = d
            .surface
            .iter()
            .map(|row| row.iter().map(|s| bloch(map.of(*s))).collect())
            .collect();
        let rect = [k * third, top, (k + 1) * third, bottom].map(|v| v as f32);
        let cam = Camera::orbit(
            third,
            bottom - top,
            [0.0; 3],
            4.6,
            azimuth,
            24.0f32.to_radians(),
            Lens::Perspective(0.62),
        );
        panel3(c, rect, cam, |s| {
            s.sphere_wire([0.0; 3], 1.0, 16, palette::grid(), 0.45);
            for a in 0..3 {
                let mut v = [0.0; 3];
                v[a] = 1.0;
                s.seg(v.map(|x| -x), v, 1.0, palette::grid(), 0.8);
            }
            // The image: a translucent surface with its mesh lines.
            for i in 0..LATITUDES {
                for j in 0..LONGITUDES {
                    let (p, q, r, u) = (
                        image[i][j],
                        image[i + 1][j],
                        image[i + 1][j + 1],
                        image[i][j + 1],
                    );
                    let lit = s.lit(p, q, r, colours[k]);
                    s.quad(p, q, r, u, lit, 0.16);
                }
            }
            for row in &image {
                s.polyline(row, 1.0, colours[k], 0.85);
            }
            for j in (0..LONGITUDES).step_by(2) {
                let meridian: Vec<Space> = image.iter().map(|row| row[j]).collect();
                s.polyline(&meridian, 1.0, colours[k], 0.85);
            }
            // Hollow and filled markers pair each preparation with its image.
            for (p, pc) in d.prepared.iter().zip(probe_colours) {
                s.dot(bloch(*p), Marker::Ring, 8.0, pc);
                s.dot(bloch(map.of(*p)), Marker::Dot, 6.0, pc);
            }
        });
        c.text(
            LABELS[k],
            (k as f32 + 0.5) * third as f32,
            top as f32 + 2.0,
            small * 1.15,
            colours[k],
            Align::Center,
        );
    }
    c.text(
        &format!("{uses} USES OF EACH LEARNED CHANNEL"),
        wf * 0.5,
        bottom as f32 - small * 0.8,
        small * 1.1,
        palette::ink(),
        Align::Center,
    );
    // How many state directions each probe set determines: the singular values of its dyads.
    for (k, (name, spectrum)) in ["TETRAHEDRON", "TWO PROBES"]
        .iter()
        .zip(completeness())
        .enumerate()
    {
        let values: Vec<String> = spectrum.iter().map(|v| format!("{v:.2}")).collect();
        c.text(
            &format!("{name}: {}", values.join(" ")),
            if k == 0 { wf * 0.02 } else { wf * 0.98 },
            bottom as f32 + small * 0.5,
            small * 0.85,
            palette::grid(),
            if k == 0 { Align::Left } else { Align::Right },
        );
    }

    // Predictions after as many uses for unseen states, against the exact probabilities.
    let quarter = wf / 4.0;
    let y0 = hf * 0.76;
    let y1 = hf * 0.94;
    let ax = Axes::equal(
        [quarter * 0.3, y0, quarter * 0.3 + (y1 - y0), y1],
        [0.27, 0.27],
        0.27,
    );
    ax.frame(c, "UNSEEN STATES", "EXACT", "");
    ax.line(c, [0.0, 0.0], [0.54, 0.54], 1.0, palette::grid(), 1.0);
    for ((exact, predicted), colour) in validation(&d.learned, uses).iter().zip(colours) {
        let pts: Vec<[f64; 2]> = exact
            .iter()
            .flatten()
            .zip(predicted.iter().flatten())
            .map(|(a, b)| [*a, *b])
            .collect();
        ax.scatter(c, &pts, Marker::Dot, 4.5, colour, 0.85);
    }

    // The measured probability tables: preparation by outcome, coloured on a log scale.
    let names = ["A", "B", "C", "D"];
    let lowest = d
        .measured
        .iter()
        .flatten()
        .flatten()
        .copied()
        .filter(|v| *v > 0.0)
        .fold(f64::INFINITY, f64::min);
    let shade = |v: f64| {
        let t = (v.max(lowest).ln() - lowest.ln()) / (0.5f64.ln() - lowest.ln());
        colormap::turbo(t as f32)
    };
    for (k, table_values) in d.measured.iter().enumerate() {
        let x0 = quarter * (k as f32 + 1.0) + quarter * 0.22;
        let size = (y1 - y0).min(quarter * 0.7);
        table(
            c,
            [x0, y0, x0 + size, y0 + size],
            table_values,
            &names,
            shade,
        );
        c.text(
            LABELS[k],
            x0 + size * 0.5,
            y0 - small * 0.6,
            small,
            colours[k],
            Align::Center,
        );
    }

    caption(
        c,
        "PROCESS TOMOGRAPHY: A NOISY GATE FROM FOUR PROBES",
        "CHANNELS AS SUMS OF SANDWICHES, LEARNED BY DUAL FRAMES",
    );
}

fn main() {
    run(
        Anim::new("process tomography", SECONDS).size(960, 540),
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::tomography::*;
    use gax::ApproxEq;
    use gax::vga3d::{Bivector, Multivector, Pseudoscalar, Vector};
    use gax_numga_examples::rng::{Draw, rng};

    /// Bloch vectors in random directions with lengths spread evenly over `[0, 1]`.
    fn blochs(seed: u64, n: usize) -> Vec<Space> {
        let mut r = rng(seed);
        (0..n)
            .map(|k| Vector::from_coeffs(r.direction()) * (k as f64 / (n - 1) as f64))
            .collect()
    }

    /// Dephasing and amplitude damping followed by Rodrigues' rotation about the unit `axis`,
    /// written out with the cross product (the vector at right angles to the plane `axis ^ v`)
    /// rather than a rotor.
    fn analytical_bloch(r: Space, angle: f64, phase_flip: f64, loss: f64, axis: Space) -> Space {
        let transverse = (1.0 - 2.0 * phase_flip) * (1.0 - loss).sqrt();
        let v = Vector::new(
            transverse * r.e1(),
            transverse * r.e2(),
            (1.0 - loss) * r.e3() + loss,
        );
        let cross = -(Pseudoscalar::new(1.0) * (axis ^ v));
        let along = (axis | v).s();
        v * angle.cos() + cross * angle.sin() + axis * ((1.0 - angle.cos()) * along)
    }

    fn close(a: St, b: St, tol: f64) -> bool {
        a.max_abs_diff(&b) <= tol
    }

    #[test]
    fn channels_match_the_independent_bloch_law_and_normalized_born_probabilities() {
        let angles = [0.0, 0.47, -0.9, 0.3];
        let phase_flip = [0.0, 0.22, 0.08, 0.5];
        let loss = [0.0, 0.45, 1.0, 0.3];
        let axis = Vector::new(2.0, -1.0, 3.0).normalized().into_inner();
        let plane = Bivector::new(2.0, -1.0, 3.0).normalized().into_inner();
        let blochs = blochs(180, 29);
        let detector = tetrahedron();
        let effects: Vec<St> = detector.iter().map(|d| state(*d) * 0.5).collect();
        for c in 0..4 {
            let device = noisy_gate(angles[c], phase_flip[c], loss[c], plane);
            let prepared: Vec<St> = blochs.iter().map(|r| state(*r)).collect();
            let probs = probabilities(device, &prepared, &effects);
            for (r, (p, row)) in blochs.iter().zip(prepared.iter().zip(&probs)) {
                let out = device.of(*p);
                let want = analytical_bloch(*r, angles[c], phase_flip[c], loss[c], axis);
                assert!(close(out, state(want), 1e-8), "{out:?} vs {want:?}");
                assert!((2.0 * out.s() - 1.0).abs() < 1e-8);
                assert!(bloch(out).norm() <= 1.0 + 1e-8);
                for (prob, dir) in row.iter().zip(&detector) {
                    let expected = (1.0 + (want | *dir).s()) / 4.0;
                    assert!((prob - expected).abs() < 1e-8);
                    assert!(*prob >= -1e-8);
                }
                assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn reconstruction_predicts_mixed_states_with_an_independent_measurement_frame() {
        let angles = [0.31, -0.76];
        let phase_flip = [0.12, 0.07];
        let loss = [0.26, 0.54];
        let plane = Bivector::new(1.0, 0.0, 1.0).normalized().into_inner();
        // Six preparations are redundant, while the four detector directions are rotated
        // independently: the reconstruction cannot rely on matching the two frames.
        let axes = [x(), Vector::new(0.0, 1.0, 0.0), z()];
        let prepared: Vec<St> = axes.iter().flat_map(|a| [state(*a), state(-*a)]).collect();
        let rotation = (xy() * -0.23).exp() * (Bivector::new(1.0, 0.0, 0.0) * 0.17).exp();
        let effects: Vec<St> = super::tomography::effects()
            .iter()
            .map(|e| (rotation >> *e).cast::<gax::vga3d::Paravector>())
            .collect();
        let unseen: Vec<St> = blochs(972, 37).into_iter().map(state).collect();
        for c in 0..2 {
            let device = noisy_gate(angles[c], phase_flip[c], loss[c], plane);
            let measured = probabilities(device, &prepared, &effects);
            let learned = reconstruct(&prepared, &effects, &measured);
            for s in &unseen {
                assert!(close(learned.of(*s), device.of(*s), 1e-12));
            }
        }
    }

    #[test]
    fn learned_channels_predict_unseen_measurements_and_repeated_applications() {
        let (prepared, measured, learned) = run_tomography();
        for (exact, predicted) in validation(&learned, 1) {
            for (a, b) in exact.iter().flatten().zip(predicted.iter().flatten()) {
                assert!((a - b).abs() < 1e-12);
            }
        }
        for (l, m) in learned.iter().zip(&measured) {
            let again = probabilities(*l, &prepared, &effects());
            for (a, b) in again.iter().flatten().zip(m.iter().flatten()) {
                assert!((a - b).abs() < 1e-12);
            }
        }
        let blochs = [
            Vector::new(0.2, -0.3, 0.4),
            Vector::new(-0.6, 0.1, 0.2),
            Vector::new(0.0, 0.0, -1.0),
        ];
        let axis = Vector::new(1.0, 1.0, 1.0).normalized().into_inner();
        for (c, l) in learned.iter().enumerate() {
            let mut expected = blochs;
            for accumulated in powers(*l, 4) {
                expected =
                    expected.map(|r| analytical_bloch(r, TURNS[c], PHASE_FLIP[c], LOSS[c], axis));
                for (r, want) in blochs.iter().zip(&expected) {
                    assert!(close(accumulated.of(state(*r)), state(*want), 1e-8));
                }
            }
        }
    }

    #[test]
    fn completeness_distinguishes_a_spanning_tetrahedron_from_two_axial_probes() {
        let expected = [[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0], [0.0, 0.0, 0.5, 0.5]];
        for (mut spectrum, want) in completeness().into_iter().zip(expected) {
            spectrum.sort_by(f64::total_cmp);
            for (a, b) in spectrum.iter().zip(want) {
                assert!((a - b).abs() < 1e-12, "{spectrum:?}");
            }
        }
    }

    /// The display surface consists of normalized pure states (idempotents) before any channel
    /// acts, and the composed channels animate: two frames differ.
    #[test]
    fn the_surface_is_pure_and_the_animation_draws() {
        for row in sphere(8, 16) {
            for s in row {
                let square = s * s;
                assert!(
                    square.max_abs_diff(&s.cast::<Multivector>()) < 1e-8,
                    "{square:?}"
                );
            }
        }
        let mut draw = super::draw;
        let anim = gax_numga_examples::Anim::new("t", super::SECONDS).size(480, 270);
        let a = gax_numga_examples::app::frame(&anim, 0.1, &mut draw);
        let b = gax_numga_examples::app::frame(&anim, 5.0, &mut draw);
        assert!(a.mean()[0] > 0.0);
        assert!(a.mean() != b.mean());
    }
}
