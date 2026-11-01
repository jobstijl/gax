//! Type errors users should get, with the messages they see (`tests/compile_fail/*.stderr`).
//! Regenerate the expected messages with
//! `TRYBUILD=overwrite cargo test -p gax --features all-algebras --test compile_fail`.
//!
//! The messages are pinned with every standard algebra enabled, as CI's all-algebras step
//! builds them: rustc names a type by its full path where its bare name is ambiguous in the
//! build (PGA3D's `Tilt` beside VGA3D's), so a smaller feature set prints other paths.

#![cfg(all(
    feature = "pga2d",
    feature = "pga3d",
    feature = "vga2d",
    feature = "vga3d",
    feature = "sta",
    feature = "sta2d",
    feature = "cga2d",
    feature = "cga3d",
    feature = "stap",
    feature = "csta"
))]

#[test]
fn compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}
