//! Type errors users should get, with the messages they see (`tests/compile_fail/*.stderr`).
//! Regenerate the expected messages with `TRYBUILD=overwrite cargo test --test compile_fail`.

#[test]
fn compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}
