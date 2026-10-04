//! Gradients: colours along a path through a colour space.

use crate::colour::Colour;
use crate::space::Space;

/// How a gradient goes from one stop to the next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path {
    /// Straight through the space: [`Colour::mix`].
    Straight,
    /// Around the grey axis, the shorter way: [`Colour::mix_hue`].
    AroundGrey,
}

/// Colours evenly spaced over `[0, 1]`, joined by straight or turning paths in their space.
#[derive(Clone, Debug)]
pub struct Gradient<S: Space> {
    stops: Vec<Colour<S>>,
    path: Path,
}

impl<S: Space> Gradient<S> {
    /// Through `stops`, straight from each to the next.
    pub fn new(stops: impl IntoIterator<Item = Colour<S>>) -> Self {
        let stops: Vec<_> = stops.into_iter().collect();
        assert!(!stops.is_empty(), "a gradient needs a colour");
        Gradient {
            stops,
            path: Path::Straight,
        }
    }

    /// The same stops, joined around the grey axis.
    pub fn around_grey(self) -> Self {
        Gradient {
            path: Path::AroundGrey,
            ..self
        }
    }

    /// The colour at `t` (clamped to `[0, 1]`).
    pub fn at(&self, t: f32) -> Colour<S> {
        let n = self.stops.len();
        if n == 1 || !t.is_finite() {
            return self.stops[0];
        }
        let x = t.clamp(0.0, 1.0) * (n - 1) as f32;
        let i = (x.floor() as usize).min(n - 2);
        let (a, b, f) = (self.stops[i], self.stops[i + 1], x - i as f32);
        match self.path {
            Path::Straight => a.mix(b, f),
            Path::AroundGrey => a.mix_hue(b, f),
        }
    }
}

/// A range of two colours as a gradient, as `bevy_color`'s `ColorRange`.
pub trait ColourRange<S: Space> {
    /// The colour at `t` of the way from the start to the end.
    fn at(&self, t: f32) -> Colour<S>;
}

impl<S: Space> ColourRange<S> for core::ops::Range<Colour<S>> {
    fn at(&self, t: f32) -> Colour<S> {
        self.start.mix(self.end, t)
    }
}
