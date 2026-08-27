//! The wxscan C ABI, compiled to WebAssembly for the browser.
//!
//! Everything callable is in [`wxscan_ffi`]; this crate adds the three things a
//! wasm module needs that a native library gets from its platform, and produces
//! the `cdylib` that a browser can instantiate.
//!
//! * `malloc` and `free`, so the host can put an image into linear memory and
//!   take a result out. A wasm module has no other way to be handed bytes.
//! * A source of randomness for `getrandom`, which tract pulls in transitively.
//! * A reference to each exported function, so the linker does not discard the
//!   ABI as unreachable — nothing inside this crate calls it.
//!
//! By default inference is the host's job — see [`wxscan_ffi::host_net`] — which
//! is what keeps this module at a quarter of a megabyte. The `tract` feature
//! compiles an ONNX engine in instead, for a module of about twelve.

// The document writer is ordinary Rust and is tested on every target; the
// module around it only means anything on wasm.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
mod json;

#[cfg(target_arch = "wasm32")]
mod module;
