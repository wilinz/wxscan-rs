//! `Instant`, except on wasm.
//!
//! The stage timings below feed the profiling recorders, which are compiled in
//! unconditionally. `std::time::Instant::now()` panics on
//! `wasm32-unknown-unknown` — the platform has no clock the standard library
//! can reach — so there it becomes a zero-sized stand-in and every stage
//! reports zero microseconds.

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::time::Instant;

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy)]
pub(crate) struct Instant;

#[cfg(target_arch = "wasm32")]
impl Instant {
    pub(crate) fn now() -> Self {
        Instant
    }
}

#[cfg(target_arch = "wasm32")]
impl core::ops::Sub for Instant {
    type Output = Elapsed;

    fn sub(self, _: Instant) -> Elapsed {
        Elapsed
    }
}

/// What subtracting two [`Instant`]s gives on wasm: nothing, measured.
#[cfg(target_arch = "wasm32")]
pub(crate) struct Elapsed;

#[cfg(target_arch = "wasm32")]
impl Elapsed {
    pub(crate) fn as_micros(&self) -> u128 {
        0
    }
}
