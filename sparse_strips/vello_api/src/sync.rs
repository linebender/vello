// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Interior mutability primitives used by implementations of [`Renderer`](crate::Renderer).
//!
//! This module is intended to be used only by implementors of Vello API (such as Vello Hybrid); if you're writing an
//! application this module is only implementation details.
//!
//! The [`Renderer`](crate::Renderer) trait makes heavy use of interior mutability.
//! This is because it is expected to be a cross-thread singleton in most apps, and it significantly improves
//! ergonomics for textures (and similar) to be reference counted.
//! Vello API needs to support targets as varied as embedded devices, desktop applications and Webassembly (both in-browser and headless).
//! However, there is not a single thread-safe interior mutability pattern suitable for all these platforms.
//!
//! This module provides a [`Lock`] type, which implements the most suitable practical locking on this platform.
//! It also provides the [`Share`] trait, which abstracts the thread safety guarantees provided by the lock.
//! These types are provided inside the Vello API crate, as [`Share`] is used as a marker supertrait
//! of `Renderer`.
//! As such, each implementations of `Renderer` needs to be written to the same sharing standard, to avoid
//! diverging feature flags causing issues.
//!
//! Internally, there are two implementations, a single threaded and a multi-threaded mode.
//! The multi-threaded implementation uses the standard library [`Mutex`](std::sync::Mutex), and [`Share`] is
//! an alias for <code>[Send] + [Sync]</code>.
//! The single-threaded implementation uses [`RefCell`](core::cell::RefCell), and [`Share`] is implemented for all types.
//!
//! We determine therefore that multi-threading can only be supported if the `std` feature of this crate is enabled
//! (as we cannot use the `std` `Mutex` without using the standard library!).
//!
//! For WebAssembly, we currently hardcode to only being in single threaded mode, even if the renderer could theoretically use (e.g. web workers).
//! This is done because all JavaScript objects cannot be sent between threads, and so Vello Hybrid cannot support cross-thread operation.
//! Additionally, Wasm multi-threading is not currently very well developed, and it isn't supported to block the thread in a lock.
//! All of this makes supporting multithreaded rendering in Wasm through the abstraction currently impractical.
//! If the need arises, we can revisit these concerns.
//!
//! This does lead to a few tradeoffs:
//!
//! - Implementations of the API need to remember to use the synchronisation primitives from this
//!   module rather than the standard library equivalents.
//! - Code needs to be tested with both versions.
//! - If the user forgets to enable the `std` feature of this library and try to use it across multiple threads, they may
//!   get confusing errors relating to `Send`/`Sync`.
//!
//! However, we believe that this is worthwhile to make resource management significantly more ergonomic, whilst still
//! allowing cross-thread usage.

// There are arguments for type aliases for `Arc` (which is `Rc`) on single threading.
// However, we choose not to do that because it increases the chance that implementations of
// Vello API would expose this detail to end users, which is to be avoided if possible.

#[cfg(all(not(target_arch = "wasm32"), feature = "std"))]
pub use multi::{Lock, Share};
#[cfg(any(target_arch = "wasm32", not(feature = "std")))]
pub use single::{Lock, Share};

#[cfg(any(target_arch = "wasm32", not(feature = "std")))]
mod single {
    /// An alias trait for the most practical subset of `Send + Sync` which supports
    /// interior mutability.
    ///
    /// In the current compilation, this is implemented for every type.
    /// See the [module level documentation](crate::sync) for more information.
    ///
    /// When the standard library is enabled, this is an alias for `Send + Sync`.
    /// If not (or on Wasm), this is implemented by every type.
    pub trait Share {}

    impl<T: ?Sized> Share for T {}

    /// The most practical locking primitive for interior mutability in 2D renderers.
    ///
    /// In the current compilation, this is a thin wrapper around [`RefCell`](core::cell::RefCell).
    ///
    /// When the standard library is enabled, this is a wrapper around `Mutex`.
    /// If not (or on Wasm), this is a wrapper around [`RefCell`](core::cell::RefCell).
    pub struct Lock<T> {
        inner: core::cell::RefCell<T>,
    }

    impl<T> Lock<T> {
        pub fn new(val: T) -> Self {
            Self {
                inner: core::cell::RefCell::new(val),
            }
        }
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "std"))]
mod multi {
    extern crate std;

    /// An alias trait for the most practical subset of `Send + Sync` which supports
    /// interior mutability.
    ///
    /// In the current compilation, this an alias for `Send + Sync`.
    /// See the [module level documentation](crate::sync) for more information.
    ///
    /// When the standard library is enabled, this is an alias for `Send + Sync`.
    /// If not (or on Wasm), this is implemented by every type.
    pub trait Share: Send + Sync {}

    impl<T: Send + Sync + ?Sized> Share for T {}

    #[derive(Debug)]
    /// The most practical locking primitive for interior mutability in 2D renderers.
    ///
    /// In the current compilation, this is a thin wrapper around [`Mutex`](std::sync::Mutex).
    ///
    /// When the standard library is enabled, this is a wrapper around `Mutex`.
    /// If not (or on Wasm), this is a wrapper around [`RefCell`](core::cell::RefCell).
    pub struct Lock<T> {
        inner: std::sync::Mutex<T>,
    }

    impl<T> Lock<T> {
        pub fn new(val: T) -> Self {
            Self {
                inner: std::sync::Mutex::new(val),
            }
        }
    }
}
