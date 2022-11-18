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

// The DrawMonoid is computed as a prefix sum to aid in decoding
// the variable-length encoding of draw objects.
struct DrawMonoid {
    // The number of paths preceding this draw object.
    path_ix: u32,
    // The number of clip operations preceding this draw object.
    clip_ix: u32,
    // The offset of the encoded draw object in the scene (u32s).
    scene_offset: u32,
    // The offset of the associated info.
    info_offset: u32,
}

// Each draw object has a 32-bit draw tag, which is a bit-packed
// version of the draw monoid.
let DRAWTAG_NOP = 0u;
let DRAWTAG_FILL_COLOR = 0x44u;
let DRAWTAG_FILL_LIN_GRADIENT = 0x114u;
let DRAWTAG_FILL_RAD_GRADIENT = 0x2dcu;
let DRAWTAG_FILL_IMAGE = 0x48u;
let DRAWTAG_BEGIN_CLIP = 0x05u;
let DRAWTAG_END_CLIP = 0x25u;

fn draw_monoid_identity() -> DrawMonoid {
    return DrawMonoid();
}

fn combine_draw_monoid(a: DrawMonoid, b: DrawMonoid) -> DrawMonoid {
    var c: DrawMonoid;
    c.path_ix = a.path_ix + b.path_ix;
    c.clip_ix = a.clip_ix + b.clip_ix;
    c.scene_offset = a.scene_offset + b.scene_offset;
    c.info_offset = a.info_offset + b.info_offset;
    return c;
}

fn map_draw_tag(tag_word: u32) -> DrawMonoid {
    var c: DrawMonoid;
    c.path_ix = u32(tag_word != DRAWTAG_NOP);
    c.clip_ix = tag_word & 1u;
    c.scene_offset = (tag_word >> 2u) & 0x07u;
    c.info_offset = (tag_word >> 6u) & 0x0fu;
    return c;
}
