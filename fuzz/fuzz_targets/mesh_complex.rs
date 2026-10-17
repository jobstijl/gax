//! Complexes from arbitrary lists of simplices: an error or a complex with its invariants, never
//! a panic (`crates/gax/tests/meshes.rs` runs the same checks on fixed samples).
#![no_main]

#[path = "../../crates/gax/tests/support/complex_checks.rs"]
mod complex_checks;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Some((&head, rest)) = data.split_first() else {
        return;
    };
    // The first byte picks the dimension and the number of vertices; each simplex takes N bytes.
    let vertices = usize::from(head >> 2) % 24;
    match head % 4 {
        0 => complex_checks::check::<2>(vertices, &simplices(rest)),
        1 => complex_checks::check::<3>(vertices, &simplices(rest)),
        2 => complex_checks::check::<4>(vertices, &simplices(rest)),
        _ => complex_checks::check::<5>(vertices, &simplices(rest)),
    }
});

fn simplices<const N: usize>(bytes: &[u8]) -> Vec<[usize; N]> {
    bytes
        .chunks_exact(N)
        .take(64)
        .map(|c| core::array::from_fn(|i| usize::from(c[i] % 26)))
        .collect()
}
