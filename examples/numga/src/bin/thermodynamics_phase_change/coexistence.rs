//! The math of the phase change: coexistence of a van der Waals fluid's liquid and vapour by
//! Newton's method on states of `l0g0`, and the fluid heated along its isobar.

use super::phases::{Scalar, Vector, intrinsic};
use gax::Real;
use gax::jet::jacobian;

/// A state of coexistence: the two volumes, an intrinsic vector (numga's `Antivector-`).
pub type State<T = f64> = intrinsic::Vector<(), T>;
/// `Scalar <- State`.
pub type Reading = Scalar<(intrinsic::Vector,), f64>;

pub fn l() -> Vector {
    Vector::new(1.0, 0.0)
}
pub fn g() -> Vector {
    Vector::new(0.0, 1.0)
}
pub fn state(liquid: f64, vapour: f64) -> State {
    l().dual() * liquid + g().dual() * vapour
}
pub fn liquid_volume() -> Reading {
    l() & State::slot()
}
pub fn vapour_volume() -> Reading {
    g() & State::slot()
}

/// The fluid: its heat capacity, the strength of its molecules' attraction, and the volume
/// they exclude.
#[derive(Clone, Copy, Debug)]
pub struct Fluid {
    pub capacity: f64,
    pub attraction: f64,
    pub exclusion: f64,
}

impl Fluid {
    /// The van der Waals equation of state.
    pub fn pressure<T: Real>(&self, volume: T, temperature: f64) -> T {
        let c = T::from_f64;
        c(temperature) / (volume - c(self.exclusion)) - c(self.attraction) / (volume * volume)
    }

    /// The Helmholtz energy, whose slope along the volume is minus the pressure.
    pub fn helmholtz_energy<T: Real>(&self, volume: T, temperature: f64) -> T {
        let c = T::from_f64;
        c(-self.capacity * temperature * temperature.ln())
            - c(temperature) * (volume - c(self.exclusion)).ln()
            - c(self.attraction) / volume
    }

    /// The Helmholtz energy plus the work of making room at the pressure: where the tangent
    /// at a volume crosses zero volume.
    pub fn chemical_potential<T: Real>(&self, volume: T, temperature: f64) -> T {
        self.helmholtz_energy(volume, temperature) + self.pressure(volume, temperature) * volume
    }

    /// The energy, the heat capacity's share less the attraction's, plus the work of making
    /// room at the pressure: the heat taken in at fixed pressure is its change.
    pub fn enthalpy(&self, volume: f64, temperature: f64) -> f64 {
        self.capacity * temperature - self.attraction / volume
            + self.pressure(volume, temperature) * volume
    }

    /// The equation of state solved for the temperature: where the isobar at a pressure
    /// passes a volume.
    pub fn isobar_temperature<T: Real>(&self, volume: T, pressure: f64) -> T {
        let c = T::from_f64;
        (c(pressure) + c(self.attraction) / (volume * volume)) * (volume - c(self.exclusion))
    }

    /// How far the isobar at a pressure is from a temperature at a volume.
    fn isobar_gap<T: Real>(&self, volume: T, pressure: f64, temperature: f64) -> T {
        self.isobar_temperature(volume, pressure) - T::from_f64(temperature)
    }

    /// The volume where the isobar at a pressure reaches a temperature, by Newton's method on
    /// the algebra's scalars from a guess on the liquid's or the vapour's side, `steps` steps.
    pub fn volume_at(&self, temperature: f64, pressure: f64, guess: f64, steps: usize) -> f64 {
        let mut volume = Scalar::new(guess);
        for _ in 0..steps {
            let (gap, slope) = jacobian(
                |v: Scalar<(), _>| Scalar::new(self.isobar_gap(v.s(), pressure, temperature)),
                volume,
            );
            volume -= slope.solve(gap);
        }
        volume.s()
    }

    /// How far the phases at the state's two volumes are from coexisting: their difference in
    /// pressure along the liquid's volume, in chemical potential along the vapour's.
    pub fn mismatch<T: Real>(&self, x: State<T>, temperature: f64) -> State<T> {
        let read = |v: Vector| (v.map_coefs(T::from_f64) & x).s();
        let (liquid, vapour) = (read(l()), read(g()));
        let pressure_gap = self.pressure(liquid, temperature) - self.pressure(vapour, temperature);
        let chemical_gap = self.chemical_potential(liquid, temperature)
            - self.chemical_potential(vapour, temperature);
        l().dual().map_coefs(T::from_f64).gp(pressure_gap)
            + g().dual().map_coefs(T::from_f64).gp(chemical_gap)
    }

    /// The coexisting volumes at a temperature, by Newton's method: each of the `steps` steps
    /// solves the mismatch's derivative, a map from a correction of the state to the change of
    /// the mismatch, for the correction that cancels the mismatch.
    pub fn coexistence(&self, temperature: f64, steps: usize) -> State {
        // Below the critical point the two volumes part as the square root of the distance
        // from it, the liquid's slowly, the vapour's fast; this starts each on its own side.
        #[allow(clippy::disallowed_methods)] // a scalar starting guess, not geometry
        let below = (1.0 - temperature).sqrt();
        let mut x = state(1.0 / (1.0 + 2.0 * below), 1.0 / (1.0 - below).powi(3));
        for _ in 0..steps {
            let (gap, derivative) = jacobian(|y| self.mismatch(y, temperature), x);
            x -= derivative.solve(gap);
        }
        x
    }

    /// The volume and the enthalpy of a mixture with a fraction of its amount as vapour, each
    /// shared out between the phases by that fraction, and the volume its liquid takes up.
    pub fn mixture(&self, coexisting: State, fraction: f64, temperature: f64) -> [f64; 3] {
        let liquid = liquid_volume().of(coexisting).s();
        let vapour = vapour_volume().of(coexisting).s();
        let volume = (1.0 - fraction) * liquid + fraction * vapour;
        let enthalpy = (1.0 - fraction) * self.enthalpy(liquid, temperature)
            + fraction * self.enthalpy(vapour, temperature);
        [volume, enthalpy, (1.0 - fraction) * liquid]
    }
}

// --- the scenario: a fluid heated at a steady rate under a loaded piston -------------

pub const FLUID: Fluid = Fluid {
    capacity: 1.5,
    attraction: 9.0 / 8.0,
    exclusion: 1.0 / 3.0,
};
pub const NEWTON: usize = 8;

pub fn coexistence_at(temperature: f64) -> State {
    FLUID.coexistence(temperature, NEWTON)
}
/// The saturation pressure at a temperature.
#[cfg_attr(not(test), allow(dead_code))]
pub fn saturation_at(temperature: f64) -> f64 {
    let liquid = liquid_volume().of(coexistence_at(temperature)).s();
    FLUID.pressure(liquid, temperature)
}

/// A step of the heating: the fluid's volume, temperature, enthalpy and the volume its liquid
/// takes up.
#[derive(Clone, Copy, Debug)]
pub struct Heated {
    pub volume: f64,
    pub temperature: f64,
    pub enthalpy: f64,
    pub liquid: f64,
}

/// `n` values from `a` toward `b`, `b` itself included only if `last`.
fn spaced(a: f64, b: f64, n: usize, last: bool) -> Vec<f64> {
    let d = if last { n.max(2) - 1 } else { n } as f64;
    (0..n).map(|k| a + (b - a) * k as f64 / d).collect()
}

/// The fluid in equal steps of heat at the saturation pressure of the boiling temperature:
/// from liquid at the coldest temperature, through boiling, to vapour at the hottest. Each
/// stage takes frames in proportion to the heat it takes.
pub fn heating(boiling: f64, coldest: f64, hottest: f64, frames: usize) -> Vec<Heated> {
    let coexisting = coexistence_at(boiling);
    let liquid = liquid_volume().of(coexisting).s();
    let vapour = vapour_volume().of(coexisting).s();
    let saturation = FLUID.pressure(liquid, boiling);
    let volume_at = |t: f64, guess: f64| FLUID.volume_at(t, saturation, guess, NEWTON);
    let ends = [
        FLUID.enthalpy(volume_at(coldest, liquid), coldest),
        FLUID.enthalpy(liquid, boiling),
        FLUID.enthalpy(vapour, boiling),
        FLUID.enthalpy(volume_at(hottest, vapour), hottest),
    ];
    let spans: Vec<f64> = ends.windows(2).map(|w| w[1] - w[0]).collect();
    let total: f64 = spans.iter().sum();
    let counts: Vec<usize> = (spans.iter())
        .map(|s| (frames as f64 * s / total).round() as usize)
        .collect();
    let single = |t: f64, guess: f64, keeps_liquid: bool| {
        let volume = volume_at(t, guess);
        Heated {
            volume,
            temperature: t,
            enthalpy: FLUID.enthalpy(volume, t),
            liquid: if keeps_liquid { volume } else { 0.0 },
        }
    };
    let mut out: Vec<Heated> = (spaced(coldest, boiling, counts[0], false).into_iter())
        .map(|t| single(t, liquid, true))
        .collect();
    for fraction in spaced(0.0, 1.0, counts[1], false) {
        let [volume, enthalpy, liquid_share] = FLUID.mixture(coexisting, fraction, boiling);
        out.push(Heated {
            volume,
            temperature: boiling,
            enthalpy,
            liquid: liquid_share,
        });
    }
    out.extend(
        (spaced(boiling, hottest, counts[2], true).into_iter()).map(|t| single(t, vapour, false)),
    );
    out
}
