// The logarithm is defined for certified unit versors only.
use gax::pga3d::Motor;

fn main() {
    let m = Motor::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let _ = m.log();
}
