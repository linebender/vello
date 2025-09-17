// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use bytemuck::{Pod, Zeroable};
use peniko::{Extend, ImageData};
use std::ops::Range;
use std::sync::Arc;

use super::{DrawTag, Encoding, PathTag, StreamOffsets, Style, Transform};

use crate::glyph_cache::GlyphCache;
use crate::image_cache::{ImageCache, Images};
use crate::ramp_cache::{RampCache, Ramps};

/// Layout of a packed encoding.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct Layout {
    /// Number of draw objects.
    pub n_draw_objects: u32,
    /// Number of paths.
    pub n_paths: u32,
    /// Number of clips.
    pub n_clips: u32,
    /// Start of binning data.
    pub bin_data_start: u32,
    /// Start of path tag stream.
    pub path_tag_base: u32,
    /// Start of path data stream.
    pub path_data_base: u32,
    /// Start of draw tag stream.
    pub draw_tag_base: u32,
    /// Start of draw data stream.
    pub draw_data_base: u32,
    /// Start of transform stream.
    pub transform_base: u32,
    /// Start of style stream.
    pub style_base: u32,
}

impl Layout {
    /// Creates a zeroed layout.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the path tag stream.
    pub fn path_tags<'a>(&self, data: &'a [u8]) -> &'a [PathTag] {
        let start = self.path_tag_base as usize * 4;
        let end = self.path_data_base as usize * 4;
        bytemuck::cast_slice(&data[start..end])
    }

    pub fn path_tags_size(&self) -> u32 {
        let start = self.path_tag_base * 4;
        let end = self.path_data_base * 4;
        end - start
    }

    /// Returns the path tag stream in chunks of 4.
    pub fn path_tags_chunked<'a>(&self, data: &'a [u8]) -> &'a [u32] {
        let start = self.path_tag_base as usize * 4;
        let end = self.path_data_base as usize * 4;
        bytemuck::cast_slice(&data[start..end])
    }

    /// Returns the path data stream.
    pub fn path_data<'a>(&self, data: &'a [u8]) -> &'a [u8] {
        let start = self.path_data_base as usize * 4;
        let end = self.draw_tag_base as usize * 4;
        bytemuck::cast_slice(&data[start..end])
    }

    /// Returns the draw tag stream.
    pub fn draw_tags<'a>(&self, data: &'a [u8]) -> &'a [DrawTag] {
        let start = self.draw_tag_base as usize * 4;
        let end = self.draw_data_base as usize * 4;
        bytemuck::cast_slice(&data[start..end])
    }

    /// Returns the draw data stream.
    pub fn draw_data<'a>(&self, data: &'a [u8]) -> &'a [u32] {
        let start = self.draw_data_base as usize * 4;
        let end = self.transform_base as usize * 4;
        bytemuck::cast_slice(&data[start..end])
    }

    /// Returns the transform stream.
    pub fn transforms<'a>(&self, data: &'a [u8]) -> &'a [Transform] {
        let start = self.transform_base as usize * 4;
        let end = self.style_base as usize * 4;
        bytemuck::cast_slice(&data[start..end])
    }

    /// Returns the style stream.
    pub fn styles<'a>(&self, data: &'a [u8]) -> &'a [Style] {
        let start = self.style_base as usize * 4;
        bytemuck::cast_slice(&data[start..])
    }
}

/// Resolves and packs an encoding that contains only paths with solid color
/// fills.
///
/// Panics if the encoding contains any late bound resources (gradients, images
/// or glyph runs).
pub fn resolve_solid_paths_only(encoding: &Encoding, packed: &mut Vec<u8>) -> Layout {
    assert!(
        encoding.resources.patches.is_empty(),
        "this resolve function doesn't support late bound resources"
    );
    let data = packed;
    data.clear();
    let mut layout = Layout {
        n_paths: encoding.n_paths,
        n_clips: encoding.n_clips,
        ..Layout::default()
    };
    let SceneBufferSizes {
        buffer_size,
        path_tag_padded,
    } = SceneBufferSizes::new(encoding, &StreamOffsets::default());
    data.reserve(buffer_size);
    // Path tag stream
    layout.path_tag_base = size_to_words(data.len());
    data.extend_from_slice(bytemuck::cast_slice(&encoding.path_tags));
    for _ in 0..encoding.n_open_clips {
        data.extend_from_slice(bytemuck::bytes_of(&PathTag::PATH));
    }
    data.resize(path_tag_padded, 0);
    // Path data stream
    layout.path_data_base = size_to_words(data.len());
    data.extend_from_slice(bytemuck::cast_slice(&encoding.path_data));
    // Draw tag stream
    layout.draw_tag_base = size_to_words(data.len());
    // Bin data follows draw info
    layout.bin_data_start = encoding.draw_tags.iter().map(|tag| tag.info_size()).sum();
    data.extend_from_slice(bytemuck::cast_slice(&encoding.draw_tags));
    for _ in 0..encoding.n_open_clips {
        data.extend_from_slice(bytemuck::bytes_of(&DrawTag::END_CLIP));
    }
    // Draw data stream
    layout.draw_data_base = size_to_words(data.len());
    data.extend_from_slice(bytemuck::cast_slice(&encoding.draw_data));
    // Transform stream
    layout.transform_base = size_to_words(data.len());
    data.extend_from_slice(bytemuck::cast_slice(&encoding.transforms));
    // Style stream
    layout.style_base = size_to_words(data.len());
    data.extend_from_slice(bytemuck::cast_slice(&encoding.styles));
    layout.n_draw_objects = layout.n_paths;
    assert_eq!(buffer_size, data.len());
    layout
}

/// Resolver for late bound resources.
#[derive(Default)]
pub struct Resolver {
    glyph_cache: GlyphCache,
    glyphs: Vec<Arc<Encoding>>,
    ramp_cache: RampCache,
    image_cache: ImageCache,
    pending_images: Vec<PendingImage>,
    patches: Vec<ResolvedPatch>,
}

impl Resolver {
    /// Creates a new resource cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolves late bound resources and packs an encoding. Returns the packed
    /// layout and computed ramp data.
    pub fn resolve<'a>(
        &'a mut self,
        encoding: &Encoding,
        packed: &mut Vec<u8>,
    ) -> (Layout, Ramps<'a>, Images<'a>) {
        let resources = &encoding.resources;
        if resources.patches.is_empty() {
            let layout = resolve_solid_paths_only(encoding, packed);
            return (layout, Ramps::default(), Images::default());
        }
        let patch_sizes = self.resolve_patches(encoding);
        self.resolve_pending_images();
        let data = packed;
        data.clear();
        let mut layout = Layout {
            n_paths: encoding.n_paths,
            n_clips: encoding.n_clips,
            ..Layout::default()
        };
        let SceneBufferSizes {
            buffer_size,
            path_tag_padded,
        } = SceneBufferSizes::new(encoding, &patch_sizes);
        data.reserve(buffer_size);
        // Path tag stream
        layout.path_tag_base = size_to_words(data.len());
        {
            let mut pos = 0;
            let stream = &encoding.path_tags;
            for patch in &self.patches {
                if let ResolvedPatch::GlyphRun { index, glyphs, .. } = patch {
                    layout.n_paths += 1;
                    let stream_offset = resources.glyph_runs[*index].stream_offsets.path_tags;
                    if pos < stream_offset {
                        data.extend_from_slice(bytemuck::cast_slice(&stream[pos..stream_offset]));
                        pos = stream_offset;
                    }
                    for glyph in &self.glyphs[glyphs.clone()] {
                        data.extend_from_slice(bytemuck::bytes_of(&PathTag::TRANSFORM));
                        data.extend_from_slice(bytemuck::cast_slice(&glyph.path_tags));
                    }
                    data.extend_from_slice(bytemuck::bytes_of(&PathTag::PATH));
                }
            }
            if pos < stream.len() {
                data.extend_from_slice(bytemuck::cast_slice(&stream[pos..]));
            }
            for _ in 0..encoding.n_open_clips {
                data.extend_from_slice(bytemuck::bytes_of(&PathTag::PATH));
            }
            data.resize(path_tag_padded, 0);
        }
        // Path data stream
        layout.path_data_base = size_to_words(data.len());
        {
            let mut pos = 0;
            let stream = &encoding.path_data;
            for patch in &self.patches {
                if let ResolvedPatch::GlyphRun { index, glyphs, .. } = patch {
                    let stream_offset = encoding.resources.glyph_runs[*index]
                        .stream_offsets
                        .path_data;
                    if pos < stream_offset {
                        data.extend_from_slice(bytemuck::cast_slice(&stream[pos..stream_offset]));
                        pos = stream_offset;
                    }
                    for glyph in &self.glyphs[glyphs.clone()] {
                        data.extend_from_slice(bytemuck::cast_slice(&glyph.path_data));
                    }
                }
            }
            if pos < stream.len() {
                data.extend_from_slice(bytemuck::cast_slice(&stream[pos..]));
            }
        }
        // Draw tag stream
        layout.draw_tag_base = size_to_words(data.len());
        // Bin data follows draw info
        layout.bin_data_start = encoding.draw_tags.iter().map(|tag| tag.info_size()).sum();
        {
            data.extend_from_slice(bytemuck::cast_slice(&encoding.draw_tags));
            for _ in 0..encoding.n_open_clips {
                data.extend_from_slice(bytemuck::bytes_of(&DrawTag::END_CLIP));
            }
        }
        // Draw data stream
        layout.draw_data_base = size_to_words(data.len());
        {
            let mut pos = 0;
            let stream = &encoding.draw_data;
            for patch in &self.patches {
                match patch {
                    ResolvedPatch::Ramp {
                        draw_data_offset,
                        ramp_id,
                        extend,
                    } => {
                        if pos < *draw_data_offset {
                            data.extend_from_slice(bytemuck::cast_slice(
                                &encoding.draw_data[pos..*draw_data_offset],
                            ));
                        }
                        let index_mode = (ramp_id << 2) | *extend as u32;
                        data.extend_from_slice(bytemuck::bytes_of(&index_mode));
                        pos = *draw_data_offset + 1;
                    }
                    ResolvedPatch::GlyphRun { .. } => {}
                    ResolvedPatch::Image {
                        index,
                        draw_data_offset,
                    } => {
                        if pos < *draw_data_offset {
                            data.extend_from_slice(bytemuck::cast_slice(
                                &encoding.draw_data[pos..*draw_data_offset],
                            ));
                        }
                        if let Some((x, y)) = self.pending_images[*index].xy {
                            let xy = (x << 16) | y;
                            data.extend_from_slice(bytemuck::bytes_of(&xy));
                            pos = *draw_data_offset + 1;
                        } else {
                            // If we get here, we failed to allocate a slot for this image in the atlas.
                            // In this case, let's zero out the dimensions so we don't attempt to render
                            // anything.
                            // TODO: a better strategy: texture array? downsample large images?
                            data.extend_from_slice(&[0_u8; 8]);
                            pos = *draw_data_offset + 2;
                        }
                    }
                }
            }
            if pos < stream.len() {
                data.extend_from_slice(bytemuck::cast_slice(&stream[pos..]));
            }
        }
        // Transform stream
        layout.transform_base = size_to_words(data.len());
        {
            let mut pos = 0;
            let stream = &encoding.transforms;
            for patch in &self.patches {
                if let ResolvedPatch::GlyphRun {
                    index,
                    glyphs: _,
                    transform,
                    scale,
                    hint,
                } = patch
                {
                    let run = &resources.glyph_runs[*index];
                    let stream_offset = run.stream_offsets.transforms;
                    if pos < stream_offset {
                        data.extend_from_slice(bytemuck::cast_slice(&stream[pos..stream_offset]));
                        pos = stream_offset;
                    }
                    if let Some(glyph_transform) = run.glyph_transform {
                        for glyph in &resources.glyphs[run.glyphs.clone()] {
                            let mut xform = *transform
                                * Transform {
                                    matrix: [1.0, 0.0, 0.0, -1.0],
                                    translation: [glyph.x * scale, glyph.y * scale],
                                }
                                * glyph_transform;
                            if *hint {
                                xform.translation[1] = xform.translation[1].round();
                            }
                            data.extend_from_slice(bytemuck::bytes_of(&xform));
                        }
                    } else {
                        for glyph in &resources.glyphs[run.glyphs.clone()] {
                            let mut xform = *transform
                                * Transform {
                                    matrix: [1.0, 0.0, 0.0, -1.0],
                                    translation: [glyph.x * scale, glyph.y * scale],
                                };
                            if *hint {
                                xform.translation[1] = xform.translation[1].round();
                            }
                            data.extend_from_slice(bytemuck::bytes_of(&xform));
                        }
                    }
                }
            }
            if pos < stream.len() {
                data.extend_from_slice(bytemuck::cast_slice(&stream[pos..]));
            }
        }
        // Style stream
        layout.style_base = size_to_words(data.len());
        {
            let mut pos = 0;
            let stream = &encoding.styles;
            for patch in &self.patches {
                if let ResolvedPatch::GlyphRun { index, glyphs, .. } = patch {
                    let stream_offset = resources.glyph_runs[*index].stream_offsets.styles;
                    if pos < stream_offset {
                        data.extend_from_slice(bytemuck::cast_slice(&stream[pos..stream_offset]));
                        pos = stream_offset;
                    }
                    for glyph in &self.glyphs[glyphs.clone()] {
                        data.extend_from_slice(bytemuck::cast_slice(&glyph.styles));
                    }
                }
            }
            if pos < stream.len() {
                data.extend_from_slice(bytemuck::cast_slice(&stream[pos..]));
            }
        }
        self.glyphs.clear();
        layout.n_draw_objects = layout.n_paths;
        assert_eq!(buffer_size, data.len());
        (layout, self.ramp_cache.ramps(), self.image_cache.images())
    }

    fn resolve_patches(&mut self, encoding: &Encoding) -> StreamOffsets {
        self.ramp_cache.maintain();
        self.glyphs.clear();
        self.glyph_cache.maintain();
        self.image_cache.clear();
        self.pending_images.clear();
        self.patches.clear();
        let mut sizes = StreamOffsets::default();
        let resources = &encoding.resources;
        for patch in &resources.patches {
            match patch {
                Patch::Ramp {
                    draw_data_offset,
                    stops,
                    extend,
                } => {
                    let ramp_id = self.ramp_cache.add(&resources.color_stops[stops.clone()]);
                    self.patches.push(ResolvedPatch::Ramp {
                        draw_data_offset: *draw_data_offset + sizes.draw_data,
                        ramp_id,
                        extend: *extend,
                    });
                }
                Patch::GlyphRun { index } => {
                    let mut run_sizes = StreamOffsets::default();
                    let run = &resources.glyph_runs[*index];
                    let glyphs = &resources.glyphs[run.glyphs.clone()];
                    let coords = &resources.normalized_coords[run.normalized_coords.clone()];
                    let mut hint = run.hint;
                    let mut font_size = run.font_size;
                    let mut transform = run.transform;
                    let mut scale = 1.0;
                    if hint {
                        // If hinting was requested and our transform matrix is just a uniform
                        // scale, then adjust our font size and cancel out the matrix. Otherwise,
                        // disable hinting entirely.
                        if transform.matrix[0] == transform.matrix[3]
                            && transform.matrix[1] == 0.0
                            && transform.matrix[2] == 0.0
                        {
                            scale = transform.matrix[0];
                            font_size *= scale;
                            transform.matrix = [1.0, 0.0, 0.0, 1.0];
                        } else {
                            hint = false;
                        }
                    }
                    let Some(mut session) = self.glyph_cache.session(
                        &run.font,
                        bytemuck::cast_slice(coords),
                        font_size,
                        hint,
                        &run.style,
                    ) else {
                        continue;
                    };
                    let glyph_start = self.glyphs.len();
                    for glyph in glyphs {
                        let (encoding, stream_sizes) =
                            session.get_or_insert(glyph.id).unwrap_or_else(|| {
                                // HACK: We pretend that the encoding was empty.
                                // In theory, we should be able to skip this glyph, but there is also
                                // a corresponding entry in `resources`, which means that we would
                                // need to make the patching process skip this glyph.
                                (Arc::new(Encoding::new()), StreamOffsets::default())
                            });
                        run_sizes.add(&stream_sizes);
                        self.glyphs.push(encoding);
                    }
                    let glyph_end = self.glyphs.len();
                    run_sizes.path_tags += glyphs.len() + 1;
                    run_sizes.transforms += glyphs.len();
                    sizes.add(&run_sizes);
                    self.patches.push(ResolvedPatch::GlyphRun {
                        index: *index,
                        glyphs: glyph_start..glyph_end,
                        transform,
                        scale,
                        hint,
                    });
                }
                Patch::Image {
                    draw_data_offset,
                    image,
                } => {
                    let index = self.pending_images.len();
                    self.pending_images.push(PendingImage {
                        image: image.clone(),
                        xy: None,
                    });
                    self.patches.push(ResolvedPatch::Image {
                        index,
                        draw_data_offset: *draw_data_offset + sizes.draw_data,
                    });
                }
            }
        }
        sizes
    }

    fn resolve_pending_images(&mut self) {
        self.image_cache.clear();
        'outer: loop {
            // Loop over the images, attempting to allocate them all into the atlas.
            for pending_image in &mut self.pending_images {
                if let Some(xy) = self.image_cache.get_or_insert(&pending_image.image) {
                    pending_image.xy = Some(xy);
                } else {
                    // We failed to allocate. Try to bump the atlas size.
                    if self.image_cache.bump_size() {
                        // We were able to increase the atlas size. Restart the outer loop.
                        continue 'outer;
                    } else {
                        // If the atlas is already maximum size, there's nothing we can do. Set
                        // the xy field to None so this image isn't rendered and then carry on--
                        // other images might still fit.
                        pending_image.xy = None;
                    }
                }
            }
            // If we made it here, we've either successfully allocated all images or we reached
            // the maximum atlas size.
            break;
        }
    }
}

/// Patch for a late bound resource.
#[derive(Clone)]
pub enum Patch {
    /// Gradient ramp resource.
    Ramp {
        /// Byte offset to the ramp id in the draw data stream.
        draw_data_offset: usize,
        /// Range of the gradient stops in the resource set.
        stops: Range<usize>,
        /// Extend mode for the gradient.
        extend: Extend,
    },
    /// Glyph run resource.
    GlyphRun {
        /// Index in the glyph run buffer.
        index: usize,
    },
    /// Image resource.
    Image {
        /// Offset to the atlas coordinates in the draw data stream.
        draw_data_offset: usize,
        /// Underlying image data.
        image: ImageData,
    },
}

/// Image to be allocated in the atlas.
#[derive(Clone, Debug)]
struct PendingImage {
    image: ImageData,
    xy: Option<(u32, u32)>,
}

#[derive(Clone, Debug)]
enum ResolvedPatch {
    Ramp {
        /// Offset to the ramp id in draw data stream.
        draw_data_offset: usize,
        /// Resolved ramp index.
        ramp_id: u32,
        /// Extend mode for the gradient.
        extend: Extend,
    },
    GlyphRun {
        /// Index of the original glyph run in the encoding.
        index: usize,
        /// Range into the glyphs encoding range buffer.
        glyphs: Range<usize>,
        /// Global transform.
        transform: Transform,
        /// Additional scale factor to apply to translation.
        scale: f32,
        /// Whether the glyph was hinted.
        ///
        /// This determines whether the y-coordinate of the final position
        /// needs to be rounded.
        hint: bool,
    },
    Image {
        /// Index of pending image element.
        index: usize,
        /// Offset to the atlas location in the draw data stream.
        draw_data_offset: usize,
    },
}

struct SceneBufferSizes {
    /// Full size of the scene buffer in bytes.
    buffer_size: usize,
    /// Padded length of the path tag stream in bytes.
    path_tag_padded: usize,
}

impl SceneBufferSizes {
    /// Computes common scene buffer sizes for the given encoding and patch
    /// stream sizes.
    fn new(encoding: &Encoding, patch_sizes: &StreamOffsets) -> Self {
        let n_path_tags =
            encoding.path_tags.len() + patch_sizes.path_tags + encoding.n_open_clips as usize;
        let path_tag_padded = align_up(n_path_tags, 4 * crate::config::PATH_REDUCE_WG);
        let buffer_size = path_tag_padded
            + slice_size_in_bytes(&encoding.path_data, patch_sizes.path_data)
            + slice_size_in_bytes(
                &encoding.draw_tags,
                patch_sizes.draw_tags + encoding.n_open_clips as usize,
            )
            + slice_size_in_bytes(&encoding.draw_data, patch_sizes.draw_data)
            + slice_size_in_bytes(&encoding.transforms, patch_sizes.transforms)
            + slice_size_in_bytes(&encoding.styles, patch_sizes.styles);
        Self {
            buffer_size,
            path_tag_padded,
        }
    }
}

fn slice_size_in_bytes<T: Sized>(slice: &[T], extra: usize) -> usize {
    (slice.len() + extra) * size_of::<T>()
}

fn size_to_words(byte_size: usize) -> u32 {
    (byte_size / size_of::<u32>()) as u32
}

fn align_up(len: usize, alignment: u32) -> usize {
    len + (len.wrapping_neg() & (alignment as usize - 1))
}
