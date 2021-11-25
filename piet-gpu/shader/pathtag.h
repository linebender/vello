// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Common data structures and functions for the path tag stream.

// This is the layout for tag bytes in the path stream. See
// doc/pathseg.md for an explanation.

#define PATH_TAG_PATHSEG_BITS 0xf
#define PATH_TAG_PATH 0x10
#define PATH_TAG_TRANSFORM 0x20
#define PATH_TAG_LINEWIDTH 0x40

struct TagMonoid {
    uint trans_ix;
    uint linewidth_ix;
    uint pathseg_ix;
    uint path_ix;
    uint pathseg_offset;
};

TagMonoid tag_monoid_identity() {
    return TagMonoid(0, 0, 0, 0, 0);
}

TagMonoid combine_tag_monoid(TagMonoid a, TagMonoid b) {
    TagMonoid c;
    c.trans_ix = a.trans_ix + b.trans_ix;
    c.linewidth_ix = a.linewidth_ix + b.linewidth_ix;
    c.pathseg_ix = a.pathseg_ix + b.pathseg_ix;
    c.path_ix = a.path_ix + b.path_ix;
    c.pathseg_offset = a.pathseg_offset + b.pathseg_offset;
    return c;
}

TagMonoid reduce_tag(uint tag_word) {
    TagMonoid c;
    // Some fun bit magic here, see doc/pathseg.md for explanation.
    uint point_count = tag_word & 0x3030303;
    c.pathseg_ix = bitCount((point_count * 7) & 0x4040404);
    c.linewidth_ix = bitCount(tag_word & (PATH_TAG_LINEWIDTH * 0x1010101));
    c.path_ix = bitCount(tag_word & (PATH_TAG_PATH * 0x1010101));
    c.trans_ix = bitCount(tag_word & (PATH_TAG_TRANSFORM * 0x1010101));
    uint n_points = point_count + ((tag_word >> 2) & 0x1010101);
    uint a = n_points + (n_points & (((tag_word >> 3) & 0x1010101) * 15));
    a += a >> 8;
    a += a >> 16;
    c.pathseg_offset = a & 0xff;
    return c;
}
