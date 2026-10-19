//! Work spread over the cores: the examples' scans and runs computed once at start-up are
//! independent items, each a function of its own input, so they come out the same in parallel;
//! an image's rows are shaded in bands.

/// The cores to spread work over.
fn threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

/// `f` on `data` in bands of whole rows of `row` items, one band per core, in parallel: each
/// call gets the index of its band's first row and the band.
pub fn par_rows<T: Send>(data: &mut [T], row: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let per = (data.len() / row.max(1)).div_ceil(threads()).max(1);
    std::thread::scope(|s| {
        for (k, band) in data.chunks_mut(per * row).enumerate() {
            let f = &f;
            s.spawn(move || f(k * per, band));
        }
    });
}

/// `f` of each item, on all cores, in the items' order: the items are dealt out in turn, so that
/// costly runs of them spread evenly.
pub fn par_map<T: Sync, U: Send>(items: &[T], f: impl Fn(&T) -> U + Sync) -> Vec<U> {
    let threads = threads();
    let mut parts: Vec<std::vec::IntoIter<U>> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads)
            .map(|k| {
                let f = &f;
                s.spawn(move || {
                    let dealt = items.iter().skip(k).step_by(threads);
                    dealt.map(f).collect::<Vec<U>>()
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| w.join().expect("a worker").into_iter())
            .collect()
    });
    (0..items.len())
        .map(|i| parts[i % threads].next().expect("dealt"))
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_results_keep_the_items_order() {
        let items: Vec<usize> = (0..37).collect();
        assert_eq!(
            super::par_map(&items, |k| k * k),
            items.iter().map(|k| k * k).collect::<Vec<_>>()
        );
    }
}
