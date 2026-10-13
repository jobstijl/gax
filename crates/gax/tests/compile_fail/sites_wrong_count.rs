// A slot of `Sites<Point, 3>` takes a field over three sites, not four.
use gax::pga3d::Point;
use gax::sites::{SiteField, Sites};

fn main() {
    let map: SiteField<Point, 3, (Sites<Point, 3>,), f64> = SiteField::zero();
    let four: SiteField<Point, 4, (), f64> = SiteField::zero();
    let _ = map.of(four);
}
