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

use std::collections::HashMap;

use dagga::{Dag, Node};
use wgpu::Texture;

use super::{Gallery, Generation, Painting, PaintingId, PaintingSource, Vello};

/// For inter-frame caching, we keep the same Vello struct around.
impl Vello {
    /// Run a rendering operation.
    ///
    /// Roots indicate
    pub fn run_render(&mut self, root: Painting, texture: Texture, galleries: &mut [Gallery]) {
        for graph in galleries.iter_mut() {
            graph.gc();
        }
        let galleries = &galleries[..];
        let mut dag = resolve_graph(&[root], galleries);
    }
}

struct Intermediary<'a> {
    source: &'a PaintingSource,
    generation: Generation,
    added: bool,
}
impl<'a> Intermediary<'a> {
    fn new((source, generation): &'a (PaintingSource, Generation)) -> Self {
        Self {
            source,
            generation: generation.clone(),
            added: false,
        }
    }
}

fn resolve_graph(roots: &[Painting], graphs: &[Gallery]) -> Dag<PaintingId, PaintingId> {
    let cache = HashMap::<PaintingId, (Texture, Generation)>::new();
    let mut union = HashMap::new();
    // Create a map of references to all paintings in the provided galleries.
    union.extend(graphs.iter().flat_map(|it| {
        it.paintings
            .iter()
            .map(|(id, source)| (*id, Intermediary::new(source)))
    }));

    let mut dag = Dag::default();
    // Perform a depth-first resolution of the root nodes.
    for painting in roots {
        resolve_recursive(&mut union, painting, &cache, &mut dag);
    }
    dag
}

fn resolve_recursive(
    union: &mut HashMap<PaintingId, Intermediary<'_>>,
    painting: &Painting,
    cache: &HashMap<PaintingId, (Texture, Generation)>,
    dag: &mut Dag<PaintingId, PaintingId>,
) -> Option<PaintingId> {
    let Some(Intermediary {
        ref source,
        ref generation,
        ref mut added,
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
    *added = true;
    // Maybe a smallvec?
    let mut dependencies = Vec::new();
    match source {
        PaintingSource::Image(_) => {}
        PaintingSource::Canvas(_scene, _) => {
            // for painting in _scene.resources.paintings {
            //    // push to vec (smallvec?)
            // }
        }
        PaintingSource::Resample(source, _)
        | PaintingSource::Region {
            painting: source, ..
        } => dependencies.push(source.clone()),
    };
    // If the dependency was already cached, we return `None` from the recursive function
    // so there won't be a corresponding node.
    dependencies.retain(|dependency| resolve_recursive(union, dependency, cache, dag).is_some());
    // If all dependencies were cached, then we can also use the cache.
    // If any dependencies needed to be repainted, we have to repaint.
    if let Some((_, cache_generation)) = cache.get(&painting.inner.id) {
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
    let node = Node::new(painting.inner.id)
        .with_reads(dependencies.iter().map(|it| it.inner.id))
        .with_result(painting.inner.id);
    dag.add_node(node);
    Some(painting.inner.id)
}
