// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Render graph for multi-pass rendering with filter effects.
//!
//! This module provides a directed acyclic graph (DAG) representation of the rendering pipeline,
//! enabling efficient multi-pass rendering when layers have filter effects applied.
//!
//! # Overview
//!
//! The render graph tracks dependencies between rendering operations (nodes) and ensures they
//! execute in the correct order. Each node represents either:
//! - The root (backdrop) layer that forms the base for compositing
//! - A filtered layer that renders geometry, applies effects, and blends the result
//!
//! # Key Features
//!
//! - **Incremental execution order**: The graph builds its execution order as layers are popped
//!   during scene encoding, avoiding the need for a full topological sort at render time.
//! - **Data dependencies**: Edges represent dependencies where one layer's output is consumed
//!   as input to another operation (e.g., a filter reading from a previous layer's buffer).
//! - **Efficient traversal**: The pre-computed execution order allows O(1) iteration over nodes
//!   in dependency order, with children always processed before their parents.
//!
//! # Usage Pattern
//!
//! 1. During scene encoding, nodes are added via [`RenderGraph::add_node`] as layers are created
//! 2. When layers with dependencies are popped, edges are added via [`RenderGraph::add_edge`]
//! 3. As each layer completes, it's recorded in execution order via [`RenderGraph::record_node_for_execution`]
//! 4. At render time, iterate nodes in order using [`RenderGraph::execution_order`]
//!
//! # Example Structure
//!
//! ```text
//! RootLayer (id=0) ← backdrop
//!     │
//!     └─→ FilterLayer (id=1, blur filter)
//!             │
//!             └─→ FilterLayer (id=2, drop shadow filter)
//!
//! Execution order: [2, 1, 0] (deepest children first, then parents)
//! ```
//!
//! In this example, `FilterLayer 2` (drop shadow) is the most deeply nested child, so it
//! executes first. Then `FilterLayer 1` (blur) executes and can reference `FilterLayer 2`'s output
//! if needed. Finally, the `RootLayer` composites all filtered results into the final image.
//!
//! # Future Improvements
//!
//! TODO: Consider using `render_graph` for `FilterGraph` execution for filters. The render graph
//! infrastructure could potentially be extended to handle the internal DAG structure within
//! individual filter effects, not just layer-to-layer dependencies.

use crate::coarse::Bbox;
use crate::filter_effects::Filter;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

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
    /// Pre-computed execution order collected during `pop_layer` calls.
    /// This is built incrementally as layers are popped (children before parents),
    /// avoiding the need for topological sort at render time.
    node_execution_order: Vec<NodeId>,
    /// The root layer node ID, tracked separately since it's never popped.
    /// This is appended to the execution order when iterating.
    root_node: Option<NodeId>,
}

/// The type of dependency between nodes.
#[derive(Debug, Clone)]
pub enum DependencyKind {
    /// Sequential execution: the `from` node must execute before the `to` node.
    ///
    /// The `to` node consumes output from the `from` node via the specified layer.
    /// For example, a parent filter reads from a child layer buffer produced by a previous operation.
    Sequential {
        /// The layer that connects the two nodes, whose output flows from `from` to `to`.
        layer_id: LayerId,
    },
}

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderGraph {
    /// Creates a new empty render graph.
    ///
    /// The graph starts with no nodes or edges. Nodes are added via [`add_node`](Self::add_node)
    /// as layers are created during scene encoding.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            next_node_id: 0,
            node_execution_order: Vec::new(),
            root_node: None,
        }
    }

    /// Add a node to the graph and return its ID.
    ///
    /// This is called during scene encoding when a new layer is created. The returned ID
    /// should be stored with the layer to establish edges later when dependencies are known.
    pub fn add_node(&mut self, kind: RenderNodeKind) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;

        // Track root node separately since it's never popped and executes last
        if matches!(kind, RenderNodeKind::RootLayer { .. }) {
            self.root_node = Some(id);
        }

        self.nodes.push(RenderNode { id, kind });
        id
    }

    /// Add an edge between two nodes, establishing a dependency relationship.
    ///
    /// This is called when a layer with dependencies is popped during scene encoding.
    /// The edge ensures that `from` node executes before `to` node, typically because
    /// `to` needs to read data produced by `from` (e.g., a filter reading from a layer buffer).
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, kind: DependencyKind) {
        self.edges.push(RenderEdge { from, to, kind });
    }

    /// Record a node as ready for execution in topological order.
    ///
    /// This is called from `pop_layer` to incrementally build the execution order
    /// as layers are popped. Since layers are popped in a depth-first manner (children
    /// before parents), this naturally builds the correct dependency order without
    /// requiring a separate topological sort at render time.
    ///
    /// # Example
    ///
    /// For nested layers A → B → C (where C is the deepest child):
    /// - C is popped first → recorded first
    /// - B is popped second → recorded second  
    /// - A is popped last → recorded last
    ///
    /// This gives execution order [C, B, A], which respects dependencies.
    pub fn record_node_for_execution(&mut self, node_id: NodeId) {
        self.node_execution_order.push(node_id);
    }

    /// Check if the graph contains any filter passes.
    ///
    /// Returns `true` if any layers have filter effects that need processing.
    /// This is useful for determining whether to use the multi-pass rendering
    /// pipeline (with filter support) or a simpler single-pass approach.
    pub fn has_filters(&self) -> bool {
        self.nodes
            .iter()
            .any(|node| matches!(node.kind, RenderNodeKind::FilterLayer { .. }))
    }

    /// Get an iterator over nodes in execution order.
    ///
    /// Returns filter layers followed by the root layer, in the order they should be executed.
    /// This order is built incrementally during `pop_layer` calls (children before parents),
    /// avoiding the need for topological sort at render time. Returns an iterator to avoid allocation.
    pub fn execution_order(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.node_execution_order
            .iter()
            .copied()
            .chain(self.root_node)
    }

    /// Topologically sort nodes based on their dependencies using Kahn's algorithm.
    ///
    /// Returns nodes in execution order, ensuring all dependencies are satisfied before
    /// each node is processed. This implements a standard topological sort with cycle detection.
    ///
    /// # Performance
    ///
    /// This is primarily used for validation and debugging. For normal rendering,
    /// prefer [`execution_order`](Self::execution_order) which is pre-computed during
    /// scene encoding and avoids the O(V+E) sorting overhead.
    ///
    /// # Panics
    ///
    /// Panics if the graph contains a cycle, which should never happen in a valid render graph.
    pub fn topological_sort(&self) -> Vec<NodeId> {
        // Track how many incoming edges each node has (dependencies it's waiting for)
        let mut in_degree = vec![0; self.nodes.len()];

        // Map each node to its dependents (nodes that depend on it)
        let mut adj_list: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();

        // Build adjacency list and count incoming edges for each node
        for edge in &self.edges {
            adj_list.entry(edge.from).or_default().push(edge.to);
            in_degree[edge.to] += 1;
        }

        // Start with nodes that have no dependencies (in-degree = 0)
        // These are safe to execute immediately
        let mut queue: Vec<NodeId> = (0..self.nodes.len())
            .filter(|&i| in_degree[i] == 0)
            .collect();

        let mut result = Vec::new();

        // Process nodes in dependency order (Kahn's algorithm)
        while let Some(node_id) = queue.pop() {
            result.push(node_id);

            // For each node that depends on this one, mark this dependency as satisfied
            if let Some(neighbors) = adj_list.get(&node_id) {
                for &neighbor in neighbors {
                    in_degree[neighbor] -= 1;
                    // If all dependencies are now satisfied, this node is ready to execute
                    if in_degree[neighbor] == 0 {
                        queue.push(neighbor);
                    }
                }
            }
        }

        // If we didn't process all nodes, there's a cycle (which should never happen)
        assert_eq!(
            result.len(),
            self.nodes.len(),
            "Cycle detected in render graph"
        );
        result
    }
}

/// A node in the render graph representing a single render operation.
#[derive(Debug, Clone)]
pub struct RenderNode {
    /// Unique identifier for this node.
    pub id: NodeId,
    /// The type of render operation this node performs.
    pub kind: RenderNodeKind,
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

/// The type of operation a render node performs.
///
/// Each variant represents a different kind of rendering pass in the pipeline.
/// Nodes are executed in dependency order to ensure correct compositing and filtering.
#[derive(Debug, Clone)]
pub enum RenderNodeKind {
    /// The root (backdrop) layer that serves as the base for compositing.
    ///
    /// This layer is always present and is rendered last, after all filter layers have
    /// been processed. It contains all non-filtered geometry and serves as the final
    /// compositing target.
    RootLayer {
        /// ID of the root layer.
        layer_id: LayerId,
        /// Bounding box in wide tile coordinates covered by this layer.
        wtile_bbox: Bbox,
    },
    /// A layer with filter effects applied.
    ///
    /// This combines multiple passes: render geometry → apply filter → blend result.
    /// Layers are added to the render graph only if they have filters. The filter
    /// output may be consumed by other filter layers or composited into the root layer.
    FilterLayer {
        /// ID of this filtered layer.
        layer_id: LayerId,
        /// The filter effect to apply.
        filter: Filter,
        /// Bounding box in wide tile coordinates containing geometry for this layer.
        wtile_bbox: Bbox,
    },
}

/// Unique identifier for a layer in the rendering system.
///
/// Layer IDs are assigned by the encoding system and used to track layer buffers
/// across multiple render passes. A layer may or may not have a corresponding node
/// in the render graph (only layers with filters create graph nodes).
pub type LayerId = u32;

/// Unique identifier for a node in the render graph.
///
/// Node IDs are assigned sequentially as nodes are added during scene encoding.
/// They are used to reference nodes when adding edges and during execution order traversal.
pub type NodeId = usize;
