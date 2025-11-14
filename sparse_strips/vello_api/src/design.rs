// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Considerations for Vello API; use cases, required features, etc.
//!
//! In scope:
//!
//! - Fill/Stroke paths.
//! - Draw images/gradients; images must be pre-uploaded, maybe gradients too.
//! - Filter effects, a core required set and support for externally registered effects.
//!   Externally registered effects *should* be able to:
//!     - Have a `static` identifier (this allows a third-party filter effect library)
//!     - Support detecting whether or not they're available, somehow.
//!     - Have runtime configuration options, ideally type-checked.
//!     - All of this suggests that using [`TypeId`](core::any::TypeId)s and a trait for validation is the way.
//! - Blurred rounded rectangles (we should maybe test it against the actual blurring)
//! - Image atlasing, i.e. drawing from a subregion of a registered image.
//!   - Needed for atlas based glyph rendering
//! - Split the renderer and the "scene".
//! - Layers. These need:
//!   - A "clip path" (this being optional, i.e. falling back to the viewport is useful)
//!   - A blend mode.
//!   - An opacity.
//!   - Being non-isolated.
//!   - Filters
//! - Downcasting, for renderer-specific options (e.g. [`set_aliasing_threshold`]).
//! - Cached intermediates for subpaths (i.e. for glyph caching if outlines are drawn "inline").
//!   - Support for applying a transform to this cache, including recalculation if needed.
//! - Transforms.
//!
//! Medium Scope:
//!
//! - "Upload" CPU images "immediately", or at least from a value.
//!   - This importantly means that we also need to support cleanup.
//! - There are arguments for supporting "global"/`static` image ids, like filters.
//! - Mipmaps, both explicit and automatic.
//! - Output image from one render "pass" and use it in another.
//!   This should work "immediately".
//!   Again, this needs support for cleanup.
//! - Download image back to the CPU. This must be "async".
//!   - We should think about how this interacts with tests.
//! - Masks: How do they work?
//!   - Do we want to support rendering *something* in 1 channel?
//!   - What about RGB-only?
//!
//! As yet undetermined:
//!
//! - Fully generic/backend-interchangable recording
//!   - What are the use cases for this?
//! - Precision, in intermediate calculations and similar.
//! - Rendering color space/HDR/Tonemapping.
//! - Rendering "over" a previous image.
//!   Support for such invalidation might be valuable for external textures.
//! - Per-path blendmodes.
//!
//! Out of scope:
//!
//! - Creating the actual renderer.
//! - Use image already on GPU (i.e. `wgpu::Texture`); this a renderer-specific extension.
//! - Glyph rendering, outlining and caching (Scoped to "Parley Draw").
//! - Registering custom filter effects.
//! - Drive a GPU surface (i.e. output to a buffer).
//!   Note that we *do* need to not *block* this.
//!
//! [`set_aliasing_threshold`]: https://docs.rs/vello_cpu/latest/vello_cpu/struct.RenderContext.html#method.set_aliasing_threshold

// This is a documentation only module
