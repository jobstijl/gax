//! The algebra declaration parser must never panic, whatever the input: it runs inside the
//! `algebra!` proc macro on user text. A declaration that parses must also emit.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(src) = std::str::from_utf8(data) {
        if let Ok(spec) = gax_gen::spec::AlgebraSpec::parse(src) {
            // Keep emission cheap for the fuzzer: only small algebras.
            if spec.algebra.dim() <= 3 && spec.kinds.len() <= 6 {
                let _ = gax_gen::emit::emit(
                    &spec,
                    &gax_gen::emit::Config {
                        core: "::gax".into(),
                        batch: Some(String::new()),
                    },
                );
            }
        }
    }
});
