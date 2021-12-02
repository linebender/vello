// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Common data structures and functions for the draw tag stream.

struct DrawMonoid {
    uint path_ix;
    uint clip_ix;
};

DrawMonoid tag_monoid_identity() {
    return DrawMonoid(0, 0);
}

DrawMonoid combine_tag_monoid(DrawMonoid a, DrawMonoid b) {
    DrawMonoid c;
    c.path_ix = a.path_ix + b.path_ix;
    c.clip_ix = a.clip_ix + b.clip_ix;
    return c;
}

#ifdef Element_size
DrawMonoid map_tag(uint tag_word) {
    switch (tag_word) {
    case Element_FillColor:
    case Element_FillLinGradient:
    case Element_FillImage:
        return DrawMonoid(1, 0);
    case Element_BeginClip:
        return DrawMonoid(1, 1);
    case Element_EndClip:
        return DrawMonoid(0, 1);
    default:
        return DrawMonoid(0, 0);
    }
}
#endif
