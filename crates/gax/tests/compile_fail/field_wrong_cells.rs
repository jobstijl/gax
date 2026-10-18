// A field map's cells are typed: map cells `Point <- Point` take a field of points, not lines.
use gax::field::{Field, SparseMap};
use gax::pga3d::{Line, Point};

fn main() {
    let cells: SparseMap<Point<(Point,), f64>> =
        SparseMap::new(vec![Point::slot()], vec![0], vec![0], (1, 1));
    let lines: Field<Line<(), f64>> = Field::zeros(1);
    let _ = cells.of(&lines);
}
