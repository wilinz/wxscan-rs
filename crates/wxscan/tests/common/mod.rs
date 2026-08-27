//! Locates the prebuilt weights, which no longer ship inside any crate.
//!
//! They live in the wxscan-weights repository, checked out beside this one.
//! `WXSCAN_WEIGHTS_DIR` overrides the location; when nothing is found the
//! caller skips rather than fails, so a clone without the sibling checkout
//! still has a green test run.

use std::path::PathBuf;

pub fn dir() -> Option<PathBuf> {
    let candidate = match std::env::var_os("WXSCAN_WEIGHTS_DIR") {
        Some(dir) => PathBuf::from(dir),
        // From crates/wxscan, the sibling checkout is three levels up.
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../wxscan-weights/models"),
    };
    candidate.is_dir().then_some(candidate)
}

pub fn load(name: &str) -> Option<Vec<u8>> {
    let path = dir()?.join(name);
    std::fs::read(&path).ok()
}

/// Prints why a test did nothing, so a skip is never mistaken for a pass.
pub fn skip(what: &str) {
    eprintln!(
        "skipping {what}: no weights. Clone https://github.com/wilinz/wxscan-weights \
         beside this repository, or set WXSCAN_WEIGHTS_DIR."
    );
}
