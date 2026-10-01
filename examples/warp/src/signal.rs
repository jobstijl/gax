//! Signals in time: an oscillation is a phasor, a point going round the unit circle, and its
//! value is the point's height. The turn is a rotation motor, as everywhere else.

/// The height of the unit phasor turned by `phase` from `(1, 0)`: a sine wave in time.
pub fn wave(phase: f32) -> f32 {
    crate::geom::phasor(phase).e01()
}

#[cfg(test)]
mod tests {
    #[test]
    #[allow(clippy::disallowed_methods)] // the reference it is checked against
    fn a_phasor_traces_a_sine() {
        for k in 0..64 {
            let p = k as f32 * 0.37;
            assert!((super::wave(p) - p.sin()).abs() < 1e-5);
        }
    }
}
