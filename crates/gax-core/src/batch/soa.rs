//! Struct-of-arrays storage, in blocks.

use super::{Batch, LaneElem, Mv};
use crate::kind::{Extensor, Kind};
use core::fmt;
use core::marker::PhantomData;
extern crate std;
use std::vec::Vec;

/// The number of values per block of a [`Soa`]: a multiple of every lane count.
pub const BLOCK: usize = 16;

/// Values of one kind in struct-of-arrays form, in blocks of [`BLOCK`] values: a block holds
/// coefficient 0 of its 16 values, then coefficient 1, and so on.
///
/// Kernels load a batch of a coefficient with one vector load where an array of structs
/// needs a transpose, at offsets the compiler knows within a block. The last block is
/// padded; the padding holds unspecified values that kernels compute on and discard.
///
/// ```
/// use gax::batch::Soa;
/// use gax::pga3d::Point;
/// let mut s: Soa<Point> = (0..20).map(|i| Point::xyz(i as f32, 0.0, 0.0)).collect();
/// assert_eq!((s.len(), s.blocks()), (20, 2));
/// assert_eq!(s.get(17), Point::xyz(17.0, 0.0, 0.0));
/// s.set(3, Point::xyz(0.0, 1.0, 0.0));
/// assert_eq!(s.column(1).nth(3), Some(1.0));
/// ```
#[derive(Clone)]
pub struct Soa<K: Kind, E = f32> {
    data: Vec<E>,
    len: usize,
    marker: PhantomData<fn() -> K>,
}

impl<K: Kind, E: LaneElem> PartialEq for Soa<K, E> {
    fn eq(&self, o: &Self) -> bool {
        self.len == o.len && (0..self.len).all(|i| self.get(i) == o.get(i))
    }
}

impl<K: Kind, E: LaneElem> fmt::Debug for Soa<K, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries((0..self.len).map(|i| self.get(i)))
            .finish()
    }
}

impl<K: Kind, E: LaneElem> Default for Soa<K, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Kind, E: LaneElem> Soa<K, E> {
    /// No values.
    #[must_use]
    pub fn new() -> Self {
        Soa {
            data: Vec::new(),
            len: 0,
            marker: PhantomData,
        }
    }

    /// `n` zero values.
    #[must_use]
    pub fn zeros(n: usize) -> Self {
        let mut s = Self::new();
        s.resize(n);
        s
    }

    /// The number of values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of blocks, `ceil(len / BLOCK)`.
    #[must_use]
    pub fn blocks(&self) -> usize {
        self.len.div_ceil(BLOCK)
    }

    #[inline(always)]
    fn index(i: usize, k: usize) -> usize {
        (i / BLOCK) * K::N * BLOCK + k * BLOCK + i % BLOCK
    }

    /// Resize to `n` values, filling with zeros.
    pub fn resize(&mut self, n: usize) {
        self.data
            .resize(n.div_ceil(BLOCK) * K::N * BLOCK, E::zero());
        for i in self.len..n.min(self.blocks() * BLOCK) {
            for k in 0..K::N {
                self.data[Self::index(i, k)] = E::zero();
            }
        }
        self.len = n;
    }

    /// Remove all values.
    pub fn clear(&mut self) {
        self.data.clear();
        self.len = 0;
    }

    /// Append a value.
    pub fn push(&mut self, x: Mv<K, E>) {
        if self.len.is_multiple_of(BLOCK) {
            self.data.resize(self.data.len() + K::N * BLOCK, E::zero());
        }
        self.len += 1;
        self.set(self.len - 1, x);
    }

    /// Value `i`.
    ///
    /// # Panics
    /// If `i >= self.len()`.
    #[must_use]
    pub fn get(&self, i: usize) -> Mv<K, E> {
        assert!(i < self.len, "Soa::get: index {i} out of bounds");
        Extensor::from_coeffs(K::arr_from_fn(|k| self.data[Self::index(i, k)]))
    }

    /// Set value `i`.
    ///
    /// # Panics
    /// If `i >= self.len()`.
    pub fn set(&mut self, i: usize, x: Mv<K, E>) {
        assert!(i < self.len, "Soa::set: index {i} out of bounds");
        for (k, x) in x.coeffs().as_ref().iter().enumerate() {
            self.data[Self::index(i, k)] = *x;
        }
    }

    /// Coefficient `k` of every value.
    pub fn column(&self, k: usize) -> impl Iterator<Item = E> + '_ {
        (0..self.len).map(move |i| self.data[Self::index(i, k)])
    }

    /// The values as an array of structs.
    #[must_use]
    pub fn to_vec(&self) -> Vec<Mv<K, E>> {
        (0..self.len).map(|i| self.get(i)).collect()
    }

    /// The blocks: `K::N * BLOCK` values each, coefficient-major.
    #[must_use]
    pub fn as_blocks(&self) -> &[E] {
        &self.data
    }

    /// The blocks, mutably.
    pub fn as_blocks_mut(&mut self) -> &mut [E] {
        &mut self.data
    }

    /// Block `b`: `K::N * BLOCK` values, coefficient-major.
    ///
    /// # Panics
    /// If `b >= self.blocks()`.
    #[inline(always)]
    #[must_use]
    pub fn block(&self, b: usize) -> &[E] {
        &self.data[b * K::N * BLOCK..(b + 1) * K::N * BLOCK]
    }

    /// Block `b`, mutably.
    ///
    /// # Panics
    /// If `b >= self.blocks()`.
    #[inline(always)]
    pub fn block_mut(&mut self, b: usize) -> &mut [E] {
        &mut self.data[b * K::N * BLOCK..(b + 1) * K::N * BLOCK]
    }

    /// The lanes of values `start..start + L::LANES` (padding past the end).
    ///
    /// # Panics
    /// If `start` is not a multiple of `L::LANES` or not below `blocks() * BLOCK`.
    #[inline(always)]
    #[must_use]
    pub fn load<L: Batch<Elem = E>>(&self, start: usize) -> Mv<K, L> {
        assert!(start.is_multiple_of(L::LANES) && BLOCK.is_multiple_of(L::LANES));
        Extensor::from_coeffs(K::arr_from_fn(|k| {
            L::load(&self.data[Self::index(start, k)..])
        }))
    }

    /// Store lanes at `start..start + L::LANES` (padding past the end).
    ///
    /// # Panics
    /// If `start` is not a multiple of `L::LANES` or not below `blocks() * BLOCK`.
    #[inline(always)]
    pub fn store<L: Batch<Elem = E>>(&mut self, start: usize, v: &Mv<K, L>) {
        assert!(start.is_multiple_of(L::LANES) && BLOCK.is_multiple_of(L::LANES));
        for (k, x) in v.coeffs().as_ref().iter().enumerate() {
            x.store(&mut self.data[Self::index(start, k)..]);
        }
    }
}

impl<K: Kind, E: LaneElem> FromIterator<Mv<K, E>> for Soa<K, E> {
    fn from_iter<I: IntoIterator<Item = Mv<K, E>>>(iter: I) -> Self {
        let mut s = Self::new();
        for x in iter {
            s.push(x);
        }
        s
    }
}

impl<K: Kind, E: LaneElem> From<&[Mv<K, E>]> for Soa<K, E> {
    fn from(xs: &[Mv<K, E>]) -> Self {
        xs.iter().copied().collect()
    }
}

/// Call `f` with lanes of every value of `xs` and write the results to `out`, block by
/// block (the driver of the batch kernels on [`Soa`]).
///
/// # Panics
/// If `L::LANES` does not divide [`BLOCK`], or `out` has fewer blocks than `xs`.
#[inline(always)]
#[allow(clippy::chunks_exact_to_as_chunks)]
pub fn soa_map<X: Kind, Y: Kind, L: Batch>(
    xs: &Soa<X, L::Elem>,
    out: &mut Soa<Y, L::Elem>,
    mut f: impl FnMut(Mv<X, L>) -> Mv<Y, L>,
) {
    assert!(BLOCK.is_multiple_of(L::LANES) && out.blocks() >= xs.blocks());
    let blocks = xs.data.chunks_exact(X::N * BLOCK);
    for (xb, yb) in blocks.zip(out.data.chunks_exact_mut(Y::N * BLOCK)) {
        let mut j = 0;
        while j < BLOCK {
            let y = f(load_block::<X, L>(xb, j));
            store_block::<Y, L>(yb, j, &y);
            j += L::LANES;
        }
    }
}

/// [`soa_map`] with two inputs, which must have the same number of blocks.
///
/// # Panics
/// If `L::LANES` does not divide [`BLOCK`], the inputs differ in blocks, or `out` has fewer.
#[inline(always)]
#[allow(clippy::chunks_exact_to_as_chunks)]
pub fn soa_map2<A: Kind, B: Kind, Y: Kind, L: Batch>(
    a: &Soa<A, L::Elem>,
    b: &Soa<B, L::Elem>,
    out: &mut Soa<Y, L::Elem>,
    mut f: impl FnMut(Mv<A, L>, Mv<B, L>) -> Mv<Y, L>,
) {
    assert!(
        BLOCK.is_multiple_of(L::LANES) && a.blocks() == b.blocks() && out.blocks() >= a.blocks()
    );
    let blocks = a
        .data
        .chunks_exact(A::N * BLOCK)
        .zip(b.data.chunks_exact(B::N * BLOCK));
    for ((ab, bb), yb) in blocks.zip(out.data.chunks_exact_mut(Y::N * BLOCK)) {
        let mut j = 0;
        while j < BLOCK {
            let x: Mv<A, L> =
                Extensor::from_coeffs(A::arr_from_fn(|k| L::load(&ab[k * BLOCK + j..])));
            let z: Mv<B, L> =
                Extensor::from_coeffs(B::arr_from_fn(|k| L::load(&bb[k * BLOCK + j..])));
            let y = f(x, z);
            for (k, c) in y.coeffs().as_ref().iter().enumerate() {
                c.store(&mut yb[k * BLOCK + j..]);
            }
            j += L::LANES;
        }
    }
}

/// The lanes at offset `j` (a multiple of `L::LANES` below [`BLOCK`]) of one block of a `Soa`.
///
/// # Panics
/// If the block is shorter than `K::N * BLOCK` or `j + L::LANES > BLOCK`.
#[inline(always)]
pub fn load_block<K: Kind, L: Batch>(block: &[L::Elem], j: usize) -> Mv<K, L> {
    let block = &block[..K::N * BLOCK];
    Extensor::from_coeffs(K::arr_from_fn(
        #[inline(always)]
        |k| L::load(&block[k * BLOCK + j..]),
    ))
}

/// Store lanes at offset `j` of one block of a `Soa` (see [`load_block`]).
///
/// # Panics
/// If the block is shorter than `K::N * BLOCK` or `j + L::LANES > BLOCK`.
#[inline(always)]
pub fn store_block<K: Kind, L: Batch>(block: &mut [L::Elem], j: usize, v: &Mv<K, L>) {
    let block = &mut block[..K::N * BLOCK];
    for (k, c) in v.coeffs().as_ref().iter().enumerate() {
        c.store(&mut block[k * BLOCK + j..]);
    }
}
