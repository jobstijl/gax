//! A sparse field map of map cells laid out for SIMD (feature `batch`): sliced ELLPACK.

extern crate std;

use super::{Field, SparseMap};
use crate::batch::{BLOCK, Batch, Kernel, LaneElem, Mv, run, to_array};

use crate::kind::{Extensor, Kind};
use std::vec;
use std::vec::Vec;

/// A [`SparseMap`] of square map cells `K <- K`, packed once for the batch kernels.
///
/// The rows are taken in slices of [`BLOCK`] (16). A slice is padded to its longest row, and its
/// `j`-th couplings are stored coefficient by coefficient, 16 rows side by side, as a `Soa`
/// stores values. A SIMD lane works on one row, so the rows' sums need no scatter, each cell
/// coefficient is one vector load, and only the values at the columns are gathered. Padding
/// couplings are masked out, so a non-finite value elsewhere in the field cannot leak in.
///
/// Build it once for a map applied many times, such as the step of an explicit simulation. It
/// pays most with 8 lanes (`f32`): on a string of 1024 sites it applies in 0.78 of the scalar
/// loop's time (0.66 with `target-cpu=native`), and in `f64` it is about even. Solves stay with
/// [`SparseMap::solve`], whose vector arithmetic weighs as much as its products, so packing
/// them did not pay (ADR-046). Rows of very different lengths pad a slice to its longest; the
/// layout suits meshes, whose rows are alike.
///
/// ```
/// use gax::field::{Field, SparseMap};
/// use gax::pga3d::Point;
/// let n: usize = 40;
/// let (mut cells, mut rows, mut cols) = (vec![], vec![], vec![]);
/// for i in 0..n {
///     for j in [i.saturating_sub(1), i, (i + 1).min(n - 1)] {
///         cells.push(Point::<(), f32>::slot().gp(if i == j { 3.0 } else { -1.0 }));
///         rows.push(i);
///         cols.push(j);
///     }
/// }
/// let map = SparseMap::new(cells, rows, cols, (n, n));
/// let x = Field::from_fn(n, |i| Point::xyz(i as f32, 1.0, 0.0));
/// let packed = map.packed();
/// let (a, b) = (map.of(&x), packed.of(&x));
/// assert!(a.sites().iter().zip(b.sites()).all(|(p, q)| (*p - *q).c.iter().all(|d| d.abs() < 1e-4)));
/// ```
#[derive(Clone, Debug)]
pub struct PackedSparseMap<C: Extensor> {
    /// Per slice: its padded width, its shortest row's length (couplings below it need no
    /// mask), and the offset of its columns.
    slices: Vec<(usize, usize, usize)>,
    /// The cells: per slice, per coupling `j`, per coefficient, 16 rows.
    cells: Vec<C::Coef>,
    /// The columns: per slice, per coupling `j`, 16 rows.
    cols: Vec<u32>,
    /// Per slice, 16 rows: the number of couplings of each row, as a coefficient.
    lens: Vec<C::Coef>,
    /// The unpacked map, for the diagonal and the shape.
    map: SparseMap<C>,
}

impl<C, K, E> SparseMap<C>
where
    C: Extensor<Kind = K, Slots = (K,), Coef = E>,
    K: Kind,
    E: LaneElem,
{
    /// The cells packed for the batch kernels ([`PackedSparseMap`]).
    ///
    /// # Panics
    /// If a column index does not fit in `u32`.
    #[must_use]
    pub fn packed(&self) -> PackedSparseMap<C> {
        let q: usize = K::N * K::N;
        let n: usize = self.shape.0;
        let nslices = n.div_ceil(BLOCK);
        let mut slices = Vec::with_capacity(nslices);
        let (mut cells, mut cols, mut lens) = (Vec::new(), Vec::new(), Vec::new());
        for s in 0..nslices {
            let rows = s * BLOCK..((s + 1) * BLOCK).min(n);
            let width: usize = rows
                .clone()
                .map(|r| self.starts[r + 1] - self.starts[r])
                .max()
                .unwrap_or(0);
            let shortest: usize = rows
                .clone()
                .map(|r| self.starts[r + 1] - self.starts[r])
                .min()
                .unwrap_or(0);
            // A partial last slice pads with empty rows.
            let shortest = if rows.len() < BLOCK { 0 } else { shortest };
            slices.push((width, shortest, cols.len()));
            let base_cells = cells.len();
            cells.resize(base_cells + width * q * BLOCK, E::zero());
            cols.resize(cols.len() + width * BLOCK, 0u32);
            let base_cols = cols.len() - width * BLOCK;
            for l in 0..BLOCK {
                let r = s * BLOCK + l;
                let len = if r < n {
                    self.starts[r + 1] - self.starts[r]
                } else {
                    0
                };
                lens.push(E::from_i64(len as i64));
                for j in 0..len {
                    let k = self.starts[r] + j;
                    cols[base_cols + j * BLOCK + l] =
                        u32::try_from(self.cols[k]).expect("a column index beyond u32");
                    let cell = self.cells[k].coeffs().as_ref();
                    for o in 0..K::N {
                        for i in 0..K::N {
                            cells[base_cells + (j * q + o * K::N + i) * BLOCK + l] =
                                cell[o].as_ref()[i];
                        }
                    }
                }
            }
        }
        PackedSparseMap {
            slices,
            cells,
            cols,
            lens,
            map: self.clone(),
        }
    }
}

impl<C, K, E> PackedSparseMap<C>
where
    C: Extensor<Kind = K, Slots = (K,), Coef = E>,
    K: Kind,
    E: LaneElem,
{
    /// The numbers of output and input sites.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.map.shape
    }

    /// The map that was packed.
    #[must_use]
    pub fn unpacked(&self) -> &SparseMap<C> {
        &self.map
    }

    /// [`SparseMap::of`] on SIMD lanes of the current batch level, one row per lane. The
    /// kernels fuse multiply-adds where the level has them, so the last bits can differ from
    /// `of`.
    ///
    /// # Panics
    /// If `x` is not over the map's input sites.
    #[must_use]
    #[track_caller]
    pub fn of(&self, x: &Field<Mv<K, E>>) -> Field<Mv<K, E>> {
        self.map.check_input(x.len());
        let mut out = vec![crate::field::zero::<Mv<K, E>>(); self.map.shape.0];
        self.run_kernel(x.sites(), &mut out);
        Field::new(out)
    }

    fn run_kernel(&self, x: &[Mv<K, E>], y: &mut [Mv<K, E>]) {
        struct Run<'a, K: Kind, E: LaneElem> {
            slices: &'a [(usize, usize, usize)],
            cells: &'a [E],
            cols: &'a [u32],
            lens: &'a [E],
            x: &'a [Mv<K, E>],
            y: &'a mut [Mv<K, E>],
        }
        impl<K: Kind, E: LaneElem> Kernel<E> for Run<'_, K, E> {
            type Output = ();
            #[inline(always)]
            fn run<L: Batch<Elem = E>>(self) {
                let Run {
                    slices,
                    cells,
                    cols,
                    lens,
                    x,
                    y,
                } = self;
                let q = K::N * K::N;
                let n = y.len();
                let mut cell_base = 0;
                for (s, &(width, shortest, col_base)) in slices.iter().enumerate() {
                    for g in (0..BLOCK).step_by(L::LANES) {
                        let r0 = s * BLOCK + g;
                        if r0 >= n {
                            break;
                        }
                        let len = L::load(&lens[s * BLOCK + g..]);
                        let mut acc: K::Arr<L> = K::arr_from_fn(
                            #[inline(always)]
                            |_| L::zero(),
                        );
                        let blades: K::Arr<usize> = K::arr_from_fn(
                            #[inline(always)]
                            |o| o,
                        );
                        // Every row of the slice has its first `shortest` couplings.
                        for j in 0..shortest {
                            acc = K::arr_zip(
                                &acc,
                                &product::<K, E, L>(
                                    x,
                                    &cols[col_base + j * BLOCK + g..],
                                    &cells[cell_base + j * q * BLOCK + g..],
                                    &blades,
                                ),
                                #[inline(always)]
                                |a, v| *a + *v,
                            );
                        }
                        // Beyond, padding couplings (j past a row's length) add nothing.
                        for j in shortest..width {
                            let at = L::splat(E::from_i64(j as i64));
                            acc = K::arr_zip(
                                &acc,
                                &product::<K, E, L>(
                                    x,
                                    &cols[col_base + j * BLOCK + g..],
                                    &cells[cell_base + j * q * BLOCK + g..],
                                    &blades,
                                ),
                                #[inline(always)]
                                |a, v| *a + L::select_lt(at, len, *v, L::zero()),
                            );
                        }
                        let lanes = K::arr_map(
                            &acc,
                            #[inline(always)]
                            |a| to_array(*a),
                        );
                        for l in 0..L::LANES.min(n - r0) {
                            y[r0 + l] = Extensor::from_coeffs(K::arr_from_fn(
                                #[inline(always)]
                                |o| lanes.as_ref()[o][l],
                            ));
                        }
                    }
                    cell_base += width * q * BLOCK;
                }
            }
        }
        run(Run::<K, E> {
            slices: &self.slices,
            cells: &self.cells,
            cols: &self.cols,
            lens: &self.lens,
            x,
            y,
        });
    }
}

/// Coupling `j` of a lane group's rows: each cell times the value at its column. `cols` and
/// `cell` start at the group's first lane of coupling `j`.
#[inline(always)]
fn product<K: Kind, E: LaneElem, L: Batch<Elem = E>>(
    x: &[Mv<K, E>],
    cols: &[u32],
    cell: &[E],
    blades: &K::Arr<usize>,
) -> K::Arr<L> {
    // The values at this coupling's columns, one coefficient per lane.
    let xs: K::Arr<L> = K::arr_from_fn(
        #[inline(always)]
        |i| {
            L::from_fn(
                #[inline(always)]
                |l| x[cols[l] as usize].coeffs().as_ref()[i],
            )
        },
    );
    let xc = xs.as_ref();
    K::arr_map(
        blades,
        #[inline(always)]
        |&o| {
            let mut v = L::load(&cell[(o * K::N) * BLOCK..]) * xc[0];
            for i in 1..K::N {
                v = L::load(&cell[(o * K::N + i) * BLOCK..]).mul_add(xc[i], v);
            }
            v
        },
    )
}
