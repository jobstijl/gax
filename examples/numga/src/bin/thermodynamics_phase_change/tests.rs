//! numga's tests of the phase change: coexistence, the latent heat and Clapeyron's equation,
//! and the heated fluid's walk along its isobar.

use super::coexistence::*;

const ROUND_OFF: f64 = 1e-10;

/// At nine temperatures the coexisting volumes have equal pressure and chemical potential,
/// the liquid's below the critical volume and the vapour's above it.
#[test]
fn coexistence_has_equal_pressure_and_chemical_potential_on_distinct_branches() {
    for k in 0..9 {
        let temperature = 0.6 + (0.99 - 0.6) * k as f64 / 8.0;
        let x = coexistence_at(temperature);
        let gap = FLUID.mismatch(x, temperature);
        assert!(gap.c.iter().all(|c| c.abs() < ROUND_OFF), "{gap:?}");
        assert!(liquid_volume().of(x).s() < 1.0 && vapour_volume().of(x).s() > 1.0);
    }
}

/// The latent heat is the temperature times the jump in entropy, and the saturation curve's
/// slope obeys Clapeyron's equation.
#[test]
fn latent_heat_is_temperature_times_entropy_jump_and_obeys_clapeyron() {
    let (temperature, step) = (0.85, 1e-4);
    let x = coexistence_at(temperature);
    let (liquid, vapour) = (liquid_volume().of(x).s(), vapour_volume().of(x).s());
    let latent = FLUID.enthalpy(vapour, temperature) - FLUID.enthalpy(liquid, temperature);
    let b = FLUID.exclusion;
    let expected = temperature * ((vapour - b) / (liquid - b)).ln();
    assert!((latent - expected).abs() < ROUND_OFF);
    let rise =
        (saturation_at(temperature + step) - saturation_at(temperature - step)) / (2.0 * step);
    let clapeyron = latent / (temperature * (vapour - liquid));
    assert!(
        (rise - clapeyron).abs() <= 1e-6 * clapeyron.abs(),
        "{rise} {clapeyron}"
    );
}

/// The heated fluid stays on its isobar outside boiling, its enthalpy and volume always
/// rising, and its liquid boils away.
#[test]
fn heating_walks_the_isobar_in_rising_enthalpy() {
    let temperature = 0.85;
    let level = saturation_at(temperature);
    let steps = heating(temperature, 0.7, 0.98, 30);
    for s in steps.iter().filter(|s| s.temperature != temperature) {
        assert!((FLUID.pressure(s.volume, s.temperature) - level).abs() < ROUND_OFF);
    }
    for pair in steps.windows(2) {
        assert!(pair[1].enthalpy > pair[0].enthalpy && pair[1].volume > pair[0].volume);
    }
    assert_eq!(steps[steps.len() - 1].liquid, 0.0);
    let boiling: Vec<f64> = (steps.iter())
        .filter(|s| s.temperature == temperature)
        .map(|s| s.liquid)
        .collect();
    assert!(boiling.windows(2).all(|p| p[1] < p[0]));
}

#[test]
fn a_frame_draws() {
    gax_numga_examples::app::assert_draws_with(super::draw, &super::scene(), 2.0);
}
