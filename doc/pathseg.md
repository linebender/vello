# Path segment encoding

The new (November 2021) element processing pipeline has a particularly clever approach to path segment encoding, and this document explains that.

By way of motivation, in the old scene encoding, all elements take a fixed amount of space, currently 36 bytes, but that's at risk of expanding if a new element type requires even more space. The new design is based on stream compaction. The input is separated into multiple streams, so in particular path segment data gets its own stream. Further, that stream can be packed.

As explained in [#119], the path stream is separated into one stream for tag bytes, and another stream for the path segment data.

## Prefix sum for unpacking

The key to this encoding is a prefix sum over the size of each element's payload. The payload size can be readily derived from the tag byte itself (see below for details on this), then an exclusive prefix sum gives the start offset of the packed encoding for each element. The combination of the tag byte and that offset gives you everything needed to unpack a segment.

## Tag byte encoding

Bits 0-1 indicate the type of path segment: 1 is line, 2 is quadratic bezier, 3 is cubic bezier.

Bit 2 indicates whether this is the last segment in a subpath; see below.

Bit 3 indicates whether the coordinates are i16 or f32.

Thus, values of 1-7 indicate the following combinations in a 16 bit encoding, so `size` counts both points and u32 indices.

```
value op             size
    1 lineto         1
    2 quadto         2
    3 curveto        3
    5 lineto + end   2
    6 quadto + end   3
    7 curveto + end  4
```

Values of 9-15 are the same but with a 32 bit encoding, so double `size` to compute the size in u32 units.

A value of 0 indicates no path segment present; it may be a nop, for example padding at the end of the stream to make it an integral number of workgroups, or other bits in the tag byte might indicate a transform, end path, or line width marker (with one bit left for future expansion). Values of 4, 8, and 12 are unused.

In addition to path segments, bits 4-6 are "one hot" encodings of other element types. Bit 4 set (0x10) is a path (encoded after all path segments). Bit 5 set (0x20) is a transform. Bit 6 set (0x40) is a line width setting. Transforms and line widths have their own streams in the encoded scene buffer, so prefix sums of the counts serve as indices into those streams.

### End subpath handling

In the previous encoding, every path segment was encoded independently; the segments could be shuffled within a path without affecting the results. However, that encoding failed to take advantage of the fact that subpaths are continuous, meaning that the start point of each segment is equal to the end point of the previous segment. Thus, there was redundancy in the encoding, and more CPU-side work for the encoder.

This encoding fixes that. Bit 2 of the tag byte indicates whether the segment is the last one in the subpath. If it is set, then the size encompasses all the points in the segment. If not, then it is short one, which leaves the offset for the next segment pointing at the last point in this one.

There is a relatively straightforward state maching to convert the usual moveto/lineto representation to this one. In short, the point for the moveto is encoded, a moveto or closepath sets the end bit for the previously encoded segment (if any), and the end bit is also set for the last segment in the path. Certain cases, such as a lone moveto, must be avoided.

### Bit magic

The encoding is carefully designed for fast calculation based on bits, in particular to quickly compute a sum of counts based on all four tag bytes in a u32.

To count whether a path segment is present, compute `(tag | (tag >> 1)) & 1`. Thus, the number of path segments in a 4-byte word is `bitCount((tag | (tag >> 1)) & 0x1010101)`. Also note: `((tag & 3) * 7) & 4` counts the same number of bits and might save one instruction given that `tag & 3` can be reused below.

The number of points (ie the value of the table above) is `(tag & 3) + ((tag >> 2) & 1)`. The value `(tag >> 3) & 1` is 0 for 16 bit encodings and 1 for 32 bit encodings. Thus, `points + (point & (((tag >> 3) & 1) * 7))` is the number of u32 words. All these operations can be performed in parallel on the 4 bytes in a word, justifying the following code:

```glsl
    uint point_count = (tag & 0x3030303) + ((tag >> 2) & 0x1010101);
    uint word_count = point_count + (point_count & (((tag >> 3) & 0x1010101) * 15));
    word_count += word_count >> 8;
    word_count += word_count >> 16;
    word_count &= 0xff;
```

One possible optimization to explore is packing multiple tags into a byte by or'ing together the flags. This would add a small amount of complexity into the interpretation (mostly in pathseg), and increase utilization a bit.

[#119]: https://github.com/linebender/piet-gpu/issues/119
