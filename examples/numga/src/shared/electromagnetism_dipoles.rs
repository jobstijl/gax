//! What the induced-dipole examples share: the field map coupling every particle to every other,
//! and the check that such a map is its own adjoint.

use gax::field::SparseMap;

/// The field map over `sites` sites from each site to every other, through the cell
/// `cell(i, j)` from site `j` to site `i`: no site couples to itself.
pub fn between_others<C: Copy>(
    sites: usize,
    mut cell: impl FnMut(usize, usize) -> C,
) -> SparseMap<C> {
    let pairs: Vec<(usize, usize)> = (0..sites)
        .flat_map(|i| (0..sites).filter(move |&j| j != i).map(move |j| (i, j)))
        .collect();
    let cells = pairs.iter().map(|&(i, j)| cell(i, j)).collect();
    let (rows, columns) = pairs.into_iter().unzip();
    SparseMap::new(cells, rows, columns, (sites, sites))
}

/// Whether a field map is its own adjoint to within `tolerance`: the couplings between each two
/// sites (added up where they repeat) are the transposes of those back. For the dipoles' maps,
/// the field of one particle's dipole at another is the field of the other's at the one, and
/// each coupling is symmetric.
pub fn is_self_adjoint<C, A>(map: &SparseMap<C>, tolerance: f64) -> bool
where
    C: gax::Extensor<Slots = (A,), Coef = f64> + core::ops::Add<Output = C>,
    A: gax::Kind<Mv<(C::Kind,), f64> = C>,
{
    use std::collections::BTreeMap;
    let merged = |m: &SparseMap<C>| {
        let mut cells: BTreeMap<(usize, usize), C> = BTreeMap::new();
        for (row, column, cell) in m.couplings() {
            cells
                .entry((row, column))
                .and_modify(|sum| *sum = *sum + cell)
                .or_insert(cell);
        }
        cells
    };
    let (forth, back) = (merged(map), merged(&map.transpose()));
    forth.keys().eq(back.keys())
        && forth
            .iter()
            .all(|(k, cell)| gax::ApproxEq::max_abs_diff(cell, &back[k]) < tolerance)
}
