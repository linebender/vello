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
    collections::{hash_map::Entry, HashMap},
    sync::Arc,
};

use dagga::{Dag, Node, Schedule};
use peniko::Color;
use wgpu::{
    Device, ImageCopyTexture, ImageCopyTextureBase, Origin3d, Queue, Texture, TextureDescriptor,
    TextureDimension, TextureFormat, TextureView, TextureViewDescriptor,
};

use super::{Gallery, Generation, OutputSize, Painting, PaintingId, PaintingSource, Vello};

#[must_use]
pub struct RenderDetails<'a> {
    schedule: Schedule<Node<Painting, PaintingId>>,
    union: HashMap<PaintingId, Intermediary<'a>>,
    root: Painting,
}

impl std::fmt::Debug for RenderDetails<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderDetails")
            .field("schedule", &self.schedule.batches)
            .field("union", &self.union)
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
        galleries: &'a mut [Gallery],
    ) -> RenderDetails<'a> {
        for graph in galleries.iter_mut() {
            // TODO: Also clean up `cache`?
            graph.gc();
        }
        // We only need exclusive access to the galleries for garbage collection.
        let galleries = &galleries[..];

        let mut union = HashMap::new();
        // Create a map of references to all paintings in the provided galleries.
        union.extend(galleries.iter().flat_map(|it| {
            it.paintings
                .iter()
                .map(|(id, source)| (*id, Intermediary::new(source)))
        }));

        let mut dag = Dag::default();
        // Perform a depth-first resolution of the root node.
        resolve_recursive(&mut union, &of, &self.cache, &mut dag);

        // TODO: Error reporting in case of loop.
        let schedule = dag.build_schedule().unwrap();
        RenderDetails {
            schedule,
            union,
            root: of,
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
            schedule,
            union,
            root,
        }: RenderDetails<'_>,
    ) -> Arc<Texture> {
        // TODO: Ideally `render_to_texture` wouldn't do its own submission.
        // let buffer = device.create_command_encoder(&CommandEncoderDescriptor {
        //     label: Some("Vello Render Graph Runner"),
        // });
        // TODO: In future, we can parallelise some of these batches.
        for batch in schedule.batches {
            for node in batch {
                let painting = node.into_inner();
                let details = union.get(&painting.inner.id).unwrap();
                let generation = details.generation.clone();
                let output_size = details.dimensions;

                Self::resolve_or_update_cache(
                    device,
                    &painting,
                    generation,
                    output_size,
                    self.cache.entry(painting.inner.id),
                );
                let value = self
                    .cache
                    .get(&painting.inner.id)
                    .expect("create_texture_if_needed created this Painting");
                let (target_tex, target_view) = (&value.0, &value.1);
                match details.source {
                    PaintingSource::Image(image) => {
                        let block_size = target_tex
                            .format()
                            .block_copy_size(None)
                            .expect("ImageFormat must have a valid block size");
                        queue.write_texture(
                            ImageCopyTexture {
                                texture: target_tex,
                                mip_level: 0,
                                origin: Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            image.data.data(),
                            wgpu::ImageDataLayout {
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
                            new_output_size, &output_size,
                            "Incorrect size determined in first pass."
                        );
                        for (image_id, dependency) in &canvas.paintings {
                            let cached = self.cache.get(&dependency.inner.id).expect(
                                "We know we previously made a cached version of this dependency",
                            );
                            self.vector_renderer.engine.image_overrides.insert(
                                *image_id,
                                ImageCopyTextureBase {
                                    // TODO: Ideally, we wouldn't need to `Arc` the textures, because they
                                    // are only used temporarily here.
                                    // OTOH, `Texture` will be `Clone` soon (https://github.com/gfx-rs/wgpu/pull/6665)
                                    texture: Arc::clone(&cached.0),
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
                                target_view,
                                &crate::RenderParams {
                                    width: output_size.width,
                                    height: output_size.height,
                                    // TODO: Config
                                    base_color: Color::BLACK,
                                    antialiasing_method: crate::AaConfig::Area,
                                },
                            )
                            .unwrap();
                    }
                    PaintingSource::Blur(dependency) => {
                        let cached = self.cache.get(&dependency.inner.id).expect(
                            "We know we previously made a cached version of this dependency",
                        );
                        self.blur.blur_into(
                            device,
                            queue,
                            &cached.1,
                            target_view,
                            details.dimensions,
                        );
                    } // PaintingSource::Region {
                      //     painting,
                      //     x,
                      //     y,
                      //     size,
                      // } => todo!(),
                }
            }
        }
        self.cache
            .get(&root.inner.id)
            .map(|(ret, ..)| ret.clone())
            .expect("We should have created an updated value")
    }

    fn resolve_or_update_cache(
        device: &Device,
        painting: &Painting,
        generation: Generation,
        output_size: OutputSize,
        mut cached: Entry<'_, PaintingId, (Arc<Texture>, TextureView, Generation)>,
    ) {
        if let Entry::Occupied(cache) = &mut cached {
            let cache = cache.get_mut();
            cache.2 = generation.clone();
            if cache.0.width() == output_size.width && cache.0.height() == output_size.height {
                return;
            }
        }

        // Either recreate the texture with the right dimensions, or create the first texture.
        let texture = device.create_texture(&TextureDescriptor {
            label: Some(&*painting.inner.label),
            size: wgpu::Extent3d {
                width: output_size.width,
                height: output_size.height,
                depth_or_array_layers: 1,
            },
            // TODO: Ideally we support mipmapping here? Should this be part of the painting?
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            // TODO: How would we determine this format?
            format: TextureFormat::Rgba8Unorm,
            usage: painting.inner.usages,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor {
            label: Some(&*painting.inner.label),
            ..Default::default()
        });
        cached.insert_entry((Arc::new(texture), view, generation));
    }
}

#[derive(Debug)]
struct Intermediary<'a> {
    source: &'a PaintingSource,
    generation: Generation,
    added: bool,
    dimensions: OutputSize,
}
impl<'a> Intermediary<'a> {
    fn new((source, generation): &'a (PaintingSource, Generation)) -> Self {
        Self {
            source,
            generation: generation.clone(),
            added: false,
            // These will be overwritten
            dimensions: OutputSize {
                height: u32::MAX,
                width: u32::MAX,
            },
        }
    }
}

fn resolve_recursive(
    union: &mut HashMap<PaintingId, Intermediary<'_>>,
    painting: &Painting,
    cache: &HashMap<PaintingId, (Arc<Texture>, TextureView, Generation)>,
    dag: &mut Dag<Painting, PaintingId>,
) -> Option<PaintingId> {
    let Some(Intermediary {
        ref source,
        ref generation,
        ref mut added,
        ref mut dimensions,
    }) = union.get_mut(&painting.inner.id)
    else {
        // TODO: Better error reporting? Continue?
        panic!("Failed to get painting: {painting:?}");
    };
    let generation = generation.clone();
    if *added {
        // If this node has already been added, there's nothing to do.
        return Some(painting.inner.id);
    }

    // Denote that the node has been (will be) added to the graph.
    // This means that a loop doesn't cause infinite recursion.
    *added = true;
    // Maybe a smallvec?
    let mut dependencies = Vec::new();
    let mut size_matches_dependency = None;
    // Collect dependencies, and size if possible
    match source {
        PaintingSource::Image(image) => {
            *dimensions = OutputSize {
                height: image.height,
                width: image.width,
            };
        }
        PaintingSource::Canvas(canvas, size) => {
            for dependency in canvas.paintings.values() {
                dependencies.push(dependency.clone());
            }
            *dimensions = *size;
        }
        PaintingSource::Blur(dependency) => {
            dependencies.push(dependency.clone());
            size_matches_dependency = Some(dependency.inner.id);
        } // PaintingSource::Resample(source, size)
          // | PaintingSource::Region {
          //     painting: source,
          //     size,
          //     ..
          // } => {
          //     dependencies.push(source.clone());
          //     *size
          // }
    };
    // Hmm. Maybe we should alloc an output texture here?
    // The advantage of that would be that it makes creating the
    // command-encoders in parallel lock-free.

    // If the dependency was already cached, we return `None` from the recursive function
    // so there won't be a corresponding node.
    dependencies.retain(|dependency| resolve_recursive(union, dependency, cache, dag).is_some());
    if let Some(size_matches) = size_matches_dependency {
        let new_size = union
            .get(&size_matches)
            .expect("We just resolved this")
            .dimensions;
        union.get_mut(&painting.inner.id).unwrap().dimensions = new_size;
    }
    // If all dependencies were cached, then we can also use the cache.
    // If any dependencies needed to be repainted, we have to repaint.
    if let Some((_, _, cache_generation)) = cache.get(&painting.inner.id) {
        if cache_generation == &generation {
            if dependencies.is_empty() {
                // Nothing to do, because this exact painting has already been rendered.
                // We don't add it to the graph, because it's effectively already complete
                // at the start of the run.
                return None;
            } else {
                // For certain scene types, we could retain the path data and *only* perform "compositing".
                // We don't have that kind of infrastructure set up currently.
                // Of course for filters and other resamplings, that is pretty meaningless, as
                // there is no metadata.
            }
        }
    }

    let node = Node::new(painting.clone())
        .with_name(&*painting.inner.label)
        .with_reads(dependencies.iter().map(|it| it.inner.id))
        .with_result(painting.inner.id);
    dag.add_node(node);
    Some(painting.inner.id)
}
