// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

//! Raw scene encoding.

mod draw;
mod encoding;
mod glyph;
mod glyph_cache;
mod image_cache;
mod math;
mod monoid;
mod path;
mod ramp_cache;
mod resolve;

pub use draw::{
    DrawBeginClip, DrawColor, DrawImage, DrawLinearGradient, DrawMonoid, DrawRadialGradient,
    DrawTag,
};
pub use encoding::{Encoding, StreamOffsets};
pub use glyph::{Glyph, GlyphRun};
pub use math::Transform;
pub use monoid::Monoid;
pub use path::{PathBbox, PathEncoder, PathMonoid, PathSegment, PathSegmentType, PathTag};
pub use ramp_cache::Ramps;
pub use resolve::{Config, Layout, Patch, Resolver};
