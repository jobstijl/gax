//! Parallel map for the generator's independent jobs (sandwich kernels, value methods, laws).
//!
//! The jobs are pure functions of the algebra, so they run on all cores and their results are
//! written in the original order: the output is byte for byte the sequential one. Threads take
//! the next job from a shared counter, so a few expensive kernels do not hold up the rest.
//!
//! `GAX_GEN_THREADS=1` runs everything on the calling thread (for profiling).

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The number of worker threads: `GAX_GEN_THREADS`, or the machine's parallelism.
fn threads() -> usize {
    std::env::var("GAX_GEN_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from))
        .max(1)
}

/// Report a job that took over half a second, when `GAX_GEN_PROFILE` is set.
pub fn timed<R>(label: impl FnOnce() -> String, f: impl FnOnce() -> R) -> R {
    if std::env::var_os("GAX_GEN_PROFILE").is_none() {
        return f();
    }
    let t = std::time::Instant::now();
    let r = f();
    let e = t.elapsed().as_secs_f64();
    if e > 0.5 {
        eprintln!("    [{e:.1} s] {}", label());
    }
    r
}

/// Report a phase's time since `since`, when `GAX_GEN_PROFILE` is set.
pub fn report(what: &str, since: std::time::Instant) {
    if std::env::var_os("GAX_GEN_PROFILE").is_some() {
        eprintln!("    phase {what}: {:.1} s", since.elapsed().as_secs_f64());
    }
}

/// `(a(), b())`, side by side.
pub fn join<A: Send, B: Send>(
    a: impl FnOnce() -> A + Send,
    b: impl FnOnce() -> B + Send,
) -> (A, B) {
    if threads() <= 1 {
        return (a(), b());
    }
    std::thread::scope(|s| {
        let hb = s.spawn(b);
        let ra = a();
        (ra, hb.join().expect("no job panicked"))
    })
}

/// `items.iter().map(f).collect()`, on all cores, in order.
pub fn map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    map_heaviest_first(items, |_| 0, f)
}

/// [`map`], with the jobs started heaviest first by `weight` (an estimate of their cost), so
/// that the longest ones do not start last and set the wall time. Results stay in order.
pub fn map_heaviest_first<T: Sync, R: Send>(
    items: &[T],
    weight: impl Fn(&T) -> usize,
    f: impl Fn(&T) -> R + Sync,
) -> Vec<R> {
    let n = threads().min(items.len());
    if n <= 1 {
        return items.iter().map(f).collect();
    }
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(weight(&items[i])));
    let next = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<R>>> = items.iter().map(|_| Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for _ in 0..n {
            s.spawn(|| {
                while let Some(&i) = order.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let item = &items[i];
                    let r = f(item);
                    *slots[i].lock().expect("no job panics while holding a slot") = Some(r);
                }
            });
        }
    });
    slots
        .into_iter()
        .map(|m| {
            m.into_inner()
                .expect("no job panicked")
                .expect("every job ran")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn keeps_the_order() {
        let items: Vec<u64> = (0..1000).collect();
        let out = super::map(&items, |&x| {
            // Uneven work, so threads finish out of order.
            (0..(x % 7) * 1000).fold(x, |a, b| a.wrapping_add(b % 3))
        });
        let want: Vec<u64> = items
            .iter()
            .map(|&x| (0..(x % 7) * 1000).fold(x, |a, b| a.wrapping_add(b % 3)))
            .collect();
        assert_eq!(out, want);
    }
}
