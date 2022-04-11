// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Common data structures and functions for the draw tag stream.

// Design of draw tag: & 0x1c gives scene size in bytes
// & 1 gives clip
// (tag >> 4) & 0x3c is info size in bytes

#define Drawtag_Nop 0
#define Drawtag_FillColor 0x44
#define Drawtag_FillLinGradient 0x114
#define Drawtag_FillRadGradient 0x2dc
#define Drawtag_FillImage 0x48
#define Drawtag_BeginClip 0x05
#define Drawtag_EndClip 0x25

struct DrawMonoid {
    uint path_ix;
    uint clip_ix;
    uint scene_offset;
    uint info_offset;
};

DrawMonoid draw_monoid_identity() {
    return DrawMonoid(0, 0, 0, 0);
}

DrawMonoid combine_draw_monoid(DrawMonoid a, DrawMonoid b) {
    DrawMonoid c;
    c.path_ix = a.path_ix + b.path_ix;
    c.clip_ix = a.clip_ix + b.clip_ix;
    c.scene_offset = a.scene_offset + b.scene_offset;
    c.info_offset = a.info_offset + b.info_offset;
    return c;
}

DrawMonoid map_tag(uint tag_word) {
    // TODO: at some point, EndClip should not generate a path
    uint has_path = uint(tag_word != Drawtag_Nop);
    return DrawMonoid(has_path, tag_word & 1, tag_word & 0x1c, (tag_word >> 4) & 0x3c);
}
