// Copyright 2018 the Vello Authors and The Pathfinder Project Developers
// SPDX-License-Identifier: Apache-2.0 OR MIT

pub mod dummy;

pub use dummy as default;

/*
TODO: All of this ...
#[cfg(target_os = "macos")]
pub use self::core_animation as default;
#[cfg(target_family = "windows")]
pub use self::direct_composition as default;
#[cfg(target_os = "linux")]
pub use self::wayland as default;

#[cfg(target_os = "macos")]
#[path = "core-animation.rs"]
pub mod core_animation;
#[cfg(target_family = "windows")]
#[path = "direct-composition.rs"]
pub mod direct_composition;
#[cfg(any(target_os = "linux"))]
pub mod wayland;
*/

#[cfg(target_vendor = "apple")]
pub mod core_animation;

// Special backends
pub mod alternate;
