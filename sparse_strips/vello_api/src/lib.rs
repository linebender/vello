// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// After you edit the crate's doc comment, run this command, then check README.md for any missing links
// cargo rdme --workspace-project=vello_api

//! Vello API is the rendering API of the 2d renderers in the Vello project.
//!
//! There are currently two [supported Vello renderers](#renderers), each with different tradeoffs.
//! This crate allows you to write the majority of your application logic to support either of those renderers.
//! These renderers are [Vello CPU](todo) and [Vello Hybrid](todo).
//!
//! # Usage
//!
//! The main entry-point in this crate is the [`Renderer`] trait, which is implemented by [`VelloCPU`](todo) and [`VelloHybrid`](todo).
//! Once you have a created a renderer, you then create scenes.
//! These are then scheduled to be run against specific textures.
//! You can also make textures from CPU content.
//!
//! TODO: This is a stub just to have an outline to push.
//!
//! # Renderers
//!
//! The Vello renderers which support this API are:
//!
//! - Vello CPU, an extremely portable 2d renderer which does not require a GPU.
//!   It is one of the fastest CPU-only 2d renderers in Rust.
//! - Vello Hybrid, which runs the most compute intensive portions of rendering on the GPU, improving performance over Vello CPU.
//!   It has wide compatibility with most devices, so long as they have a GPU, and it runs well on the web.
//! <!-- We might also have, to be determined:
//! - Vello Classic, which performs almost all rendering on the GPU, which gives great performance on devices with decent GPUs.
//!   However, it cannot run well on devices with weak GPUs, or in contexts without support for compute shaders, such as the web.
//!   It also has unavoidably high memory usage, and can silently fail to render if the scene gets too big.
//! -->
//!
//! As a general guide for consumers, you should prefer Vello Hybrid for applications, and Vello CPU for headless use cases
//! (e.g. screenshot tests or server-rendered previews).
//! Note that applications using Vello Hybrid might need to support falling back to Vello CPU for compatibility or performance reasons.
//!
//! This abstraction is tailored for the Vello renderers, as we believe that these have a sufficiently broad coverage of the trade-off
//! space to be viable for any consumer.
//! Vello API guarantees identical rendering between these renderers, barring subpixel differences due to precision/different rounding.
//! <!-- TODO: Is ^ true? -->
//!
//! # Abstraction Boundaries
//!
//! The abstractions in this crate are focused on 2d rendering, and the resources required to perform that.
//! In particular, this does abstract over strategies for:
//!
//! - creating the renderer.
//! - bringing external content into the renderer (for example, already resident GPU textures); nor
//! - presenting rendered content to an operating system window.
//!
//! These functionalities are however catered for where applicable by APIs on the specific renderers.
//! The renderer API supports downcasting to the specific renderer, so that these extensions can be called.
//! Each supported renderer will/does have examples showing how to achieve this yourself.
//!
//! # Text
//!
//! Vello API does not handle text/glyph rendering itself.
//! This allows for improved resource sharing of intermediate text layout data, for hinting and ink splitting underlines.
//!
//! Text can be rendered to Vello API scenes using the "Parley Draw" crate.
//! We also support rendering using using traditional glyph atlases, which may be preferred by some consumers.
//! This is especially useful to achieve subpixel rendering, such as ClearType, which Vello doesn't currently support directly.

#![forbid(unsafe_code)]
#![no_std]
#![expect(missing_docs, reason = "This code is very experimental.")]
#![expect(clippy::result_unit_err, reason = "This code is very experimental.")]

extern crate alloc;

mod design;
mod download;
mod painter;
mod renderer;

pub mod baseline;
pub mod dynamic;
pub mod prepared;
pub mod recording;
pub mod texture;

pub use self::download::DownloadId;
pub use self::painter::{PaintScene, SceneOptions};
pub use self::renderer::Renderer;

pub use ::peniko;
