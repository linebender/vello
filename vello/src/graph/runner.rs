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
    collections::{btree_map::OccupiedEntry, hash_map::Entry, HashMap},
    u32,
};

use dagga::{Dag, Node, Schedule};
use wgpu::{
    Device, Queue, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureView,
};

use super::{Gallery, Generation, OutputSize, Painting, PaintingId, PaintingSource, Vello};

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
    /// Run a rendering operation.
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

    pub fn render_to_texture(
        &mut self,
        device: &Device,
        queue: &Queue,
        texture: &TextureView,
        RenderDetails {
            schedule,
            union,
            root,
        }: RenderDetails<'_>,
    ) {
        // TODO: In future, we can parallelise some of these batches.
        for batch in schedule.batches {
            for node in batch {
                let painting = node.into_inner();
                let details = union.get(&painting.inner.id).unwrap();
                let generation = details.generation.clone();
                let output_size = details.dimensions;
                let cached = self.cache.entry(painting.inner.id);
                let texture = if root.inner.id == painting.inner.id {
                    texture
                } else {
                    Self::cached_or_create_texture(
                        device,
                        painting,
                        generation,
                        output_size,
                        cached,
                    );
                    todo!();
                };
            }
        }
    }

    fn cached_or_create_texture<'cached>(
        device: &Device,
        painting: Painting,
        generation: Generation,
        output_size: OutputSize,
        mut cached: Entry<'cached, PaintingId, (Texture, TextureView, Generation)>,
    ) -> &'cached TextureView {
        if let Entry::Occupied(cache) = &mut cached {
            let cache = cache.get_mut();
            cache.2 = generation;
            if cache.0.width() == output_size.width && cache.0.height() == output_size.height {
                let Entry::Occupied(cache) = cached else {
                    unreachable!();
                };
                return &cache.into_mut().1;
            }
        }

        // Either create a new texture with the right dimensions, or create the first texture.
        let texture = device.create_texture(&TextureDescriptor {
            label: Some(&*painting.inner.label),
            size: wgpu::Extent3d {
                width: output_size.width,
                height: output_size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            // Hmmm. How do we decide this?
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            // | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ret = cached.insert_entry(todo!());
        &ret.get().1
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
    cache: &HashMap<PaintingId, (Texture, TextureView, Generation)>,
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
    *dimensions = match source {
        PaintingSource::Image(image) => OutputSize {
            height: image.height,
            width: image.width,
        },
        PaintingSource::Canvas(canvas, size) => {
            for dependency in canvas.paintings.values() {
                dependencies.push(dependency.clone());
            }
            *size
        }
        PaintingSource::Resample(source, size)
        | PaintingSource::Region {
            painting: source,
            size,
            ..
        } => {
            dependencies.push(source.clone());
            *size
        }
    };
    // Hmm. Maybe we should alloc an output texture here?
    // The advantage of that would be that it makes creating the
    // command-encoders in parallel lock-free.

    // If the dependency was already cached, we return `None` from the recursive function
    // so there won't be a corresponding node.
    dependencies.retain(|dependency| resolve_recursive(union, dependency, cache, dag).is_some());
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
