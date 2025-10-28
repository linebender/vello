// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Render graph for multi-pass rendering with filter effects.

use crate::coarse::Bbox;
use crate::filter_effects::Filter;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

/// The ID of a node in the render graph.
pub type NodeId = usize;
/// The ID of a layer in the render graph.
pub type LayerId = u32;

/// A render graph containing nodes and edges representing the rendering pipeline.
///
/// This graph tracks layers with filter effects and their dependencies, enabling
/// multi-pass rendering where filter outputs can be used as inputs to subsequent operations.
#[derive(Debug, Clone)]
pub struct RenderGraph {
    /// All render operations (layers) in the graph.
    pub nodes: Vec<RenderNode>,
    /// Dependencies between nodes, defining execution order.
    pub edges: Vec<RenderEdge>,
    /// Counter for generating unique node IDs.
    next_node_id: NodeId,
}

/// A node in the render graph representing a single render operation.
#[derive(Debug, Clone)]
pub struct RenderNode {
    /// Unique identifier for this node.
    pub id: NodeId,
    /// The type of render operation this node performs.
    pub kind: RenderNodeKind,
}

/// The type of operation a render node performs.
#[derive(Debug, Clone)]
pub enum RenderNodeKind {
    /// The root (backdrop) layer that serves as the base for compositing.
    RootLayer {
        /// ID of the root layer.
        layer_id: LayerId,
        /// Bounding box in wide tile coordinates covered by this layer.
        wtile_bbox: Bbox,
    },
    /// A layer with filter effects applied.
    ///
    /// This combines multiple passes: render geometry → apply filter → blend result.
    /// Only layers with filters are added to the render graph.
    FilterLayer {
        /// ID of this filtered layer.
        layer_id: LayerId,
        /// The filter effect to apply.
        filter: Filter,
        /// Bounding box in wide tile coordinates containing geometry for this layer.
        wtile_bbox: Bbox,
    },
}

/// An edge representing a dependency between two nodes.
#[derive(Debug, Clone)]
pub struct RenderEdge {
    /// The source node that must complete first.
    pub from: NodeId,
    /// The destination node that depends on the source.
    pub to: NodeId,
    /// The type of dependency relationship.
    pub kind: DependencyKind,
}

/// The type of dependency between nodes.
#[derive(Debug, Clone)]
pub enum DependencyKind {
    /// Data dependency: the destination node consumes output from the source node.
    ///
    /// For example, a filter may read from a layer buffer produced by a previous operation.
    DataDependency {
        /// The layer whose data is being consumed.
        layer_id: LayerId,
    },
    /// Ordering dependency: execution order matters but no data is transferred.
    ///
    /// Used to ensure correct compositing order when filter effects are involved.
    OrderingDependency,
}

impl RenderGraph {
    /// Create a new empty render graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            next_node_id: 0,
        }
    }

    /// Add a node to the graph and return its ID.
    pub fn add_node(&mut self, kind: RenderNodeKind) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;

        self.nodes.push(RenderNode { id, kind });
        id
    }

    /// Add an edge between two nodes.
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, kind: DependencyKind) {
        self.edges.push(RenderEdge { from, to, kind });
    }

    /// Check if the graph contains any filter passes.
    ///
    /// Returns `true` if any layers have filter effects that need processing.
    pub fn has_filters(&self) -> bool {
        self.nodes
            .iter()
            .any(|node| matches!(node.kind, RenderNodeKind::FilterLayer { .. }))
    }

    /// Topologically sort nodes based on their dependencies.
    ///
    /// Returns nodes in execution order, ensuring all dependencies are satisfied.
    pub fn topological_sort(&self) -> Vec<NodeId> {
        // Track how many incoming edges each node has
        let mut in_degree = vec![0; self.nodes.len()];
        // Map each node to its dependents
        let mut adj_list: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();

        // Build adjacency list and count incoming edges
        for edge in &self.edges {
            adj_list.entry(edge.from).or_default().push(edge.to);
            in_degree[edge.to] += 1;
        }

        // Start with nodes that have no dependencies (in-degree = 0)
        let mut queue: Vec<NodeId> = (0..self.nodes.len())
            .filter(|&i| in_degree[i] == 0)
            .collect();

        let mut result = Vec::new();

        // Process nodes in dependency order
        while let Some(node_id) = queue.pop() {
            result.push(node_id);

            // For each dependent, reduce its in-degree
            if let Some(neighbors) = adj_list.get(&node_id) {
                for &neighbor in neighbors {
                    in_degree[neighbor] -= 1;
                    // If all dependencies satisfied, add to queue
                    if in_degree[neighbor] == 0 {
                        queue.push(neighbor);
                    }
                }
            }
        }

        // If we didn't process all nodes, there's a cycle
        assert_eq!(
            result.len(),
            self.nodes.len(),
            "Cycle detected in render graph"
        );
        result
    }
}

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
    }
}
