// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Running a render graph has three important steps:
//!
//! 1) Resolving the paintings to be rendered, and importantly their sizes.
//!    Note that this *might* involve splitting the tree, because of [`OutputSize::Inferred`]
//! 2) Creating a graph to find the order in which those are to be rendered.
//!    Note that this doesn't have to dictate the order that their commands
//!    are encoded, only the order in which they are submitted.
//! 3) Running that graph. This involves encoding all the commands, and submitting them
//!    in the order calculated in step 2.

use std::{
    collections::HashMap,
    sync::{Arc, MutexGuard},
};

use peniko::Color;
use wgpu::{
    Device, Origin3d, Queue, TexelCopyTextureInfoBase, Texture, TextureDescriptor,
    TextureDimension, TextureFormat, TextureViewDescriptor,
};

use super::{Gallery, OutputSize, PaintInner, Painting, PaintingId, PaintingSource, Vello};

pub(super) struct RenderOrder {
    painting: PaintingId,
    should_paint: bool,
}

#[must_use]
pub struct RenderDetails<'a> {
    root: Painting,
    gallery: MutexGuard<'a, HashMap<PaintingId, PaintInner>>,
    order: Vec<RenderOrder>,
}

impl std::fmt::Debug for RenderDetails<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderDetails")
            .field("gallery", &"elided")
            .field("root", &self.root)
            .finish()
    }
}

/// For inter-frame caching, we keep the same Vello struct around.
impl Vello {
    /// Prepare a rendering operation.
    ///
    /// # Panics
    ///
    /// If the graph has a loop.
    pub fn prepare_render<'a>(
        &mut self,
        of: Painting,
        gallery: &'a mut Gallery,
    ) -> RenderDetails<'a> {
        self.scratch_paint_order.clear();
        // TODO: Nicer error reporting
        assert_eq!(
            of.inner.gallery.as_ptr(),
            Arc::as_ptr(&gallery.inner),
            "{of:?} isn't from {gallery:?}."
        );
        assert_eq!(
            gallery.inner.device, self.device,
            "Gallery is not for the same device as the renderer"
        );

        let mut gallery = gallery.inner.lock_paintings();
        // Perform a depth-first resolution of the root node.
        resolve_recursive(
            &mut gallery,
            &of,
            &mut self.scratch_paint_order,
            &mut Vec::with_capacity(16),
        );

        RenderDetails {
            root: of,
            gallery,
            order: std::mem::take(&mut self.scratch_paint_order),
        }
    }

    /// Run a rendering operation.
    ///
    /// A queue submission might be needed after this
    ///
    /// # Panics
    ///
    /// If rendering fails
    pub fn render_to_texture(
        &mut self,
        device: &Device,
        queue: &Queue,
        RenderDetails {
            mut gallery,
            root,
            order,
        }: RenderDetails<'_>,
    ) -> Texture {
        // TODO: Ideally `render_to_texture` wouldn't do its own submission.
        // let buffer = device.create_command_encoder(&CommandEncoderDescriptor {
        //     label: Some("Vello Render Graph Runner"),
        // });
        // TODO: In future, we can parallelise some of these batches.
        let gallery = &mut *gallery;
        for node in &order {
            if !node.should_paint {
                continue;
            }
            let painting_id = node.painting;
            let paint = gallery.get_mut(&painting_id).unwrap();

            Self::validate_update_texture(device, paint);
            let (target_tex, target_view) =
                (paint.texture.clone().unwrap(), paint.view.clone().unwrap());

            // Take the source for borrow checker purposes
            let dimensions = paint.dimensions;
            let source = paint
                .source
                .take()
                .expect("A sourceless painting should have `should_paint` unset");
            match &source {
                PaintingSource::Image(image) => {
                    let block_size = target_tex
                        .format()
                        .block_copy_size(None)
                        .expect("ImageFormat must have a valid block size");
                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &target_tex,
                            mip_level: 0,
                            origin: Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        image.data.data(),
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(image.width * block_size),
                            rows_per_image: None,
                        },
                        wgpu::Extent3d {
                            width: image.width,
                            height: image.height,
                            depth_or_array_layers: 1,
                        },
                    );
                }
                PaintingSource::Canvas(canvas, new_output_size) => {
                    debug_assert_eq!(
                        new_output_size, &dimensions,
                        "Incorrect size determined in first pass."
                    );
                    for (image_id, dependency) in &canvas.paintings {
                        let dep_paint = gallery.get(&dependency.inner.id).expect(
                            "We know we previously made a cached version of this dependency",
                        );
                        let Some(texture) = dep_paint.texture.clone() else {
                            // TODO: This happens if a texture is used as a dependency, but
                            // (for example), its source is never set.
                            // Error instead (maybe somewhere much earlier)?
                            continue;
                        };
                        self.vector_renderer.engine.image_overrides.insert(
                            *image_id,
                            TexelCopyTextureInfoBase {
                                // TODO: Ideally, we wouldn't need to `Arc` the textures, because they
                                // are only used temporarily here. This is something we need to fix in Vello's API.
                                texture,
                                aspect: wgpu::TextureAspect::All,
                                mip_level: 0,
                                origin: Origin3d::ZERO,
                            },
                        );
                    }
                    self.vector_renderer
                        .render_to_texture(
                            device,
                            queue,
                            &canvas.scene,
                            &target_view,
                            &crate::RenderParams {
                                width: dimensions.width,
                                height: dimensions.height,
                                // TODO: Configurable somewhere
                                base_color: Color::BLACK,
                                antialiasing_method: crate::AaConfig::Area,
                            },
                        )
                        .unwrap();
                }
                PaintingSource::Blur(dependency) => {
                    let dependency_paint = gallery.get(&dependency.inner.id).unwrap();
                    self.blur.blur_into(
                        device,
                        queue,
                        dependency_paint.view.as_ref().unwrap(),
                        &target_view,
                        dimensions,
                    );
                } // PaintingSource::Region {
                  //     painting,
                  //     x,
                  //     y,
                  //     size,
                  // } => todo!(),
            }
        }
        self.scratch_paint_order = order;
        gallery
            .get_mut(&root.inner.id)
            .unwrap()
            .texture
            .clone()
            .unwrap()
    }

    fn validate_update_texture(device: &Device, paint: &mut PaintInner) {
        if let Some(texture) = paint.texture.as_ref() {
            // TODO: Some reasoning about 3d textures?
            if texture.width() == paint.dimensions.width
                && texture.height() == paint.dimensions.height
            {
                debug_assert_eq!(
                    texture.usage(),
                    paint.usages,
                    "Texture usages in a painting are immutable."
                );
                return;
            }
        }
        // Either recreate the texture with corrected dimensions, or create the first texture.
        let texture = device.create_texture(&TextureDescriptor {
            label: Some(&*paint.label),
            size: wgpu::Extent3d {
                width: paint.dimensions.width,
                height: paint.dimensions.height,
                depth_or_array_layers: 1,
            },
            // TODO: Ideally we support mipmapping here? Should this be part of the painting?
            mip_level_count: 1,
            // TODO: What does it even mean to be multisampled?
            sample_count: 1,
            dimension: TextureDimension::D2,
            // TODO: How would we determine this format in an HDR world?
            format: TextureFormat::Rgba8Unorm,
            usage: paint.usages,
            view_formats: &[],
        });
        // TODO: Should we just be creating this ad-hoc?
        let view = texture.create_view(&TextureViewDescriptor {
            label: Some(&*paint.label),
            ..Default::default()
        });
        paint.texture = Some(texture);
        paint.view = Some(view);
    }
}

/// Returns whether this painting will be repainted.
fn resolve_recursive(
    gallery: &mut HashMap<PaintingId, PaintInner>,
    painting: &Painting,
    rendering_preorder: &mut Vec<RenderOrder>,
    scratch_stack: &mut Vec<Painting>,
) -> bool {
    let painting_id = painting.inner.id;
    let paint = gallery
        .get_mut(&painting_id)
        .expect("Painting exists and is associated with this gallery, so should be present here");
    if paint.resolving {
        // TODO: Improved debugging information (path to `self`?)
        // Is there a nice way to hook into backtrace reporting/unwinding to print that?
        panic!("Infinite loop in render graph at {painting:?}.")
    }
    // If we've already scheduled this painting this round, there's nothing to do.
    if let Some(x) = rendering_preorder
        // as_slice is *only* needed to fix rust-analyzer's analysis. I don't know why
        .as_slice()
        .get(paint.paint_index)
    {
        return x.should_paint;
    };

    paint.resolving = true;
    let Some(source) = paint.source.as_ref() else {
        let idx = rendering_preorder.len();
        rendering_preorder.push(RenderOrder {
            painting: painting_id,
            should_paint: false,
        });
        paint.paint_index = idx;
        // What does this return value represent?
        // We know that there are no dependencies, but do we need to still add this to the preorder?
        return false;
    };
    let mut size_matches_dependency = None;
    let mut dimensions = OutputSize {
        height: u32::MAX,
        width: u32::MAX,
    };
    let dependencies_start_idx = scratch_stack.len();
    match source {
        PaintingSource::Image(image) => {
            dimensions = OutputSize {
                height: image.height,
                width: image.width,
            };
        }
        PaintingSource::Canvas(canvas, size) => {
            for dependency in canvas.paintings.values() {
                scratch_stack.push(dependency.clone());
            }
            dimensions = *size;
        }
        PaintingSource::Blur(dependency) => {
            scratch_stack.push(dependency.clone());
            size_matches_dependency = Some(dependency.inner.id);
        }
    };

    paint.resolving = true;
    let mut dependency_changed = false;
    for idx in dependencies_start_idx..scratch_stack.len() {
        let dependency = scratch_stack[idx].clone();
        let will_paint = resolve_recursive(gallery, &dependency, rendering_preorder, scratch_stack);
        dependency_changed |= will_paint;
    }
    scratch_stack.truncate(dependencies_start_idx);
    if let Some(size_matches) = size_matches_dependency {
        let new_size = gallery
            .get(&size_matches)
            .expect("We just resolved this")
            .dimensions;
        dimensions = new_size;
    }
    debug_assert_ne!(
        dimensions.height,
        u32::MAX,
        "Dimensions should have been initialised properly"
    );
    #[expect(
        clippy::shadow_unrelated,
        reason = "Same reference, different lifetime."
    )]
    let paint = gallery
        .get_mut(&painting_id)
        .expect("Painting exists and is associated with this gallery, so should be present here");
    paint.resolving = false;
    paint.dimensions = dimensions;

    // For certain scene types, if the source hasn't changed but its dependencies have,
    // we could retain the path data and *only* perform "compositing".
    // That is, handle
    // We don't have that kind of infrastructure set up currently.
    // Of course for filters and other resamplings, that is pretty meaningless, as
    // there is no metadata.
    let should_paint = paint.source_dirty || dependency_changed;
    paint.source_dirty = false;
    let idx = rendering_preorder.len();
    rendering_preorder.push(RenderOrder {
        painting: painting_id,
        should_paint,
    });
    paint.paint_index = idx;
    should_paint
}
