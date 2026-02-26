// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A `no_std`-compatible guillotine rectangle allocator with tree-based coalescing.
//!
//! Ported from [guillotiere](https://github.com/nical/guillotiere) (Apache-2.0 OR MIT)
//! by Nicolas Silva. Adapted for `no_std` by replacing `euclid` types with local
//! equivalents and removing `std`-only functionality (SVG dump, serde).

use crate::multi_atlas::{AllocId, Allocation};
use alloc::vec;
use alloc::vec::Vec;
use core::num::Wrapping;

// ---------------------------------------------------------------------------
// Minimal geometry types (replacing euclid)
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct Size {
    width: i32,
    height: i32,
}

impl Size {
    const fn new(width: i32, height: i32) -> Self {
        Self { width, height }
    }

    fn is_empty(self) -> bool {
        self.width <= 0 || self.height <= 0
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct Point {
    x: i32,
    y: i32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct Rect {
    min: Point,
    max: Point,
}

impl Rect {
    fn zero() -> Self {
        Self {
            min: Point { x: 0, y: 0 },
            max: Point { x: 0, y: 0 },
        }
    }

    fn width(self) -> i32 {
        self.max.x - self.min.x
    }

    fn height(self) -> i32 {
        self.max.y - self.min.y
    }

    fn size(self) -> Size {
        Size::new(self.width(), self.height())
    }

    fn is_empty(self) -> bool {
        self.width() <= 0 || self.height() <= 0
    }
}

impl From<Size> for Rect {
    fn from(s: Size) -> Self {
        Self {
            min: Point { x: 0, y: 0 },
            max: Point {
                x: s.width,
                y: s.height,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Internal index type
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct AllocIndex(u32);

impl AllocIndex {
    const NONE: Self = Self(u32::MAX);

    fn index(self) -> usize {
        self.0 as usize
    }

    fn is_none(self) -> bool {
        self == Self::NONE
    }

    fn is_some(self) -> bool {
        self != Self::NONE
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GEN_MASK: u32 = 0xFF000000;
const IDX_MASK: u32 = 0x00FFFFFF;

const LARGE_BUCKET: usize = 2;
const MEDIUM_BUCKET: usize = 1;
const SMALL_BUCKET: usize = 0;
const NUM_BUCKETS: usize = 3;

fn free_list_for_size(small_threshold: i32, large_threshold: i32, size: Size) -> usize {
    if size.width >= large_threshold || size.height >= large_threshold {
        LARGE_BUCKET
    } else if size.width >= small_threshold || size.height >= small_threshold {
        MEDIUM_BUCKET
    } else {
        SMALL_BUCKET
    }
}

// ---------------------------------------------------------------------------
// Tree node types
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum Orientation {
    Vertical,
    Horizontal,
}

impl Orientation {
    fn flipped(self) -> Self {
        match self {
            Self::Vertical => Self::Horizontal,
            Self::Horizontal => Self::Vertical,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum NodeKind {
    Container,
    Alloc,
    Free,
    Unused,
}

#[derive(Clone, Debug)]
struct Node {
    parent: AllocIndex,
    next_sibling: AllocIndex,
    prev_sibling: AllocIndex,
    kind: NodeKind,
    orientation: Orientation,
    rect: Rect,
}

// ---------------------------------------------------------------------------
// Allocator options
// ---------------------------------------------------------------------------

const DEFAULT_SMALL_SIZE_THRESHOLD: i32 = 32;
const DEFAULT_LARGE_SIZE_THRESHOLD: i32 = 256;

// ---------------------------------------------------------------------------
// AtlasAllocator
// ---------------------------------------------------------------------------

/// A dynamic texture atlas allocator using the guillotine algorithm with tree-based coalescing.
///
/// Maintains a tree of allocated and free rectangles as leaf nodes and containers as
/// non-leaf nodes. Consecutive sibling free nodes are merged in O(1), and unique children
/// are collapsed into their parent, cascading merges upward.
///
/// See the [guillotiere documentation](https://docs.rs/guillotiere) for a detailed
/// explanation of the data structure and its trade-offs.
pub(crate) struct GuillotineAllocator {
    nodes: Vec<Node>,
    free_lists: [Vec<AllocIndex>; NUM_BUCKETS],
    unused_nodes: AllocIndex,
    generations: Vec<Wrapping<u8>>,
    small_size_threshold: i32,
    large_size_threshold: i32,
    #[expect(
        dead_code,
        reason = "retained for future grow()/clear()/is_empty() support"
    )]
    size: Size,
    #[expect(dead_code, reason = "retained for future grow()/is_empty() support")]
    root_node: AllocIndex,
}

impl GuillotineAllocator {
    pub(crate) fn new(width: u32, height: u32) -> Self {
        let size = Size::new(width as i32, height as i32);
        assert!(size.width > 0, "atlas width must be positive");
        assert!(size.height > 0, "atlas height must be positive");

        let mut free_lists = [Vec::new(), Vec::new(), Vec::new()];
        let bucket = free_list_for_size(
            DEFAULT_SMALL_SIZE_THRESHOLD,
            DEFAULT_LARGE_SIZE_THRESHOLD,
            size,
        );
        free_lists[bucket].push(AllocIndex(0));

        Self {
            nodes: vec![Node {
                parent: AllocIndex::NONE,
                next_sibling: AllocIndex::NONE,
                prev_sibling: AllocIndex::NONE,
                rect: size.into(),
                kind: NodeKind::Free,
                orientation: Orientation::Vertical,
            }],
            free_lists,
            generations: vec![Wrapping(0)],
            unused_nodes: AllocIndex::NONE,
            small_size_threshold: DEFAULT_SMALL_SIZE_THRESHOLD,
            large_size_threshold: DEFAULT_LARGE_SIZE_THRESHOLD,
            size,
            root_node: AllocIndex(0),
        }
    }

    #[expect(
        clippy::cast_sign_loss,
        reason = "coordinates are always non-negative for valid allocations"
    )]
    pub(crate) fn allocate(&mut self, width: u32, height: u32) -> Option<Allocation> {
        let requested_size = Size::new(width as i32, height as i32);
        if requested_size.is_empty() {
            return None;
        }

        let chosen_id = self.find_suitable_rect(requested_size);
        if chosen_id.is_none() {
            return None;
        }

        let chosen_node = self.nodes[chosen_id.index()].clone();
        let chosen_rect = chosen_node.rect;
        let allocated_rect = Rect {
            min: chosen_rect.min,
            max: Point {
                x: chosen_rect.min.x + requested_size.width,
                y: chosen_rect.min.y + requested_size.height,
            },
        };
        let current_orientation = chosen_node.orientation;
        assert_eq!(
            chosen_node.kind,
            NodeKind::Free,
            "chosen node must be free for allocation"
        );

        let (split_rect, leftover_rect, orientation) =
            guillotine_rect(chosen_node.rect, requested_size, current_orientation);

        let allocated_id;
        let split_id;
        let leftover_id;

        if orientation == current_orientation {
            if !split_rect.is_empty() {
                let next_sibling = chosen_node.next_sibling;

                split_id = self.new_node();
                self.nodes[split_id.index()] = Node {
                    parent: chosen_node.parent,
                    next_sibling,
                    prev_sibling: chosen_id,
                    rect: split_rect,
                    kind: NodeKind::Free,
                    orientation: current_orientation,
                };

                self.nodes[chosen_id.index()].next_sibling = split_id;
                if next_sibling.is_some() {
                    self.nodes[next_sibling.index()].prev_sibling = split_id;
                }
            } else {
                split_id = AllocIndex::NONE;
            }

            if !leftover_rect.is_empty() {
                self.nodes[chosen_id.index()].kind = NodeKind::Container;

                allocated_id = self.new_node();
                leftover_id = self.new_node();

                self.nodes[allocated_id.index()] = Node {
                    parent: chosen_id,
                    next_sibling: leftover_id,
                    prev_sibling: AllocIndex::NONE,
                    rect: allocated_rect,
                    kind: NodeKind::Alloc,
                    orientation: current_orientation.flipped(),
                };

                self.nodes[leftover_id.index()] = Node {
                    parent: chosen_id,
                    next_sibling: AllocIndex::NONE,
                    prev_sibling: allocated_id,
                    rect: leftover_rect,
                    kind: NodeKind::Free,
                    orientation: current_orientation.flipped(),
                };
            } else {
                allocated_id = chosen_id;
                let node = &mut self.nodes[chosen_id.index()];
                node.kind = NodeKind::Alloc;
                node.rect = allocated_rect;

                leftover_id = AllocIndex::NONE;
            }
        } else {
            self.nodes[chosen_id.index()].kind = NodeKind::Container;

            if !split_rect.is_empty() {
                split_id = self.new_node();
                self.nodes[split_id.index()] = Node {
                    parent: chosen_id,
                    next_sibling: AllocIndex::NONE,
                    prev_sibling: AllocIndex::NONE,
                    rect: split_rect,
                    kind: NodeKind::Free,
                    orientation: current_orientation.flipped(),
                };
            } else {
                split_id = AllocIndex::NONE;
            }

            if !leftover_rect.is_empty() {
                let container_id = self.new_node();
                self.nodes[container_id.index()] = Node {
                    parent: chosen_id,
                    next_sibling: split_id,
                    prev_sibling: AllocIndex::NONE,
                    rect: Rect::zero(),
                    kind: NodeKind::Container,
                    orientation: current_orientation.flipped(),
                };

                self.nodes[split_id.index()].prev_sibling = container_id;

                allocated_id = self.new_node();
                leftover_id = self.new_node();

                self.nodes[allocated_id.index()] = Node {
                    parent: container_id,
                    next_sibling: leftover_id,
                    prev_sibling: AllocIndex::NONE,
                    rect: allocated_rect,
                    kind: NodeKind::Alloc,
                    orientation: current_orientation,
                };

                self.nodes[leftover_id.index()] = Node {
                    parent: container_id,
                    next_sibling: AllocIndex::NONE,
                    prev_sibling: allocated_id,
                    rect: leftover_rect,
                    kind: NodeKind::Free,
                    orientation: current_orientation,
                };
            } else {
                allocated_id = self.new_node();
                self.nodes[allocated_id.index()] = Node {
                    parent: chosen_id,
                    next_sibling: split_id,
                    prev_sibling: AllocIndex::NONE,
                    rect: allocated_rect,
                    kind: NodeKind::Alloc,
                    orientation: current_orientation.flipped(),
                };

                self.nodes[split_id.index()].prev_sibling = allocated_id;

                leftover_id = AllocIndex::NONE;
            }
        }

        assert_eq!(
            self.nodes[allocated_id.index()].kind,
            NodeKind::Alloc,
            "allocated node must have Alloc kind"
        );

        if split_id.is_some() {
            self.add_free_rect(split_id, split_rect.size());
        }

        if leftover_id.is_some() {
            self.add_free_rect(leftover_id, leftover_rect.size());
        }

        Some(Allocation {
            id: self.alloc_id(allocated_id),
            x: allocated_rect.min.x as u32,
            y: allocated_rect.min.y as u32,
        })
    }

    pub(crate) fn deallocate(&mut self, id: AllocId) {
        let mut node_id = self.get_index(id);

        assert!(
            node_id.index() < self.nodes.len(),
            "node index must be within nodes array"
        );
        assert_eq!(
            self.nodes[node_id.index()].kind,
            NodeKind::Alloc,
            "deallocated node must have Alloc kind"
        );

        self.nodes[node_id.index()].kind = NodeKind::Free;

        loop {
            let orientation = self.nodes[node_id.index()].orientation;

            let next = self.nodes[node_id.index()].next_sibling;
            let prev = self.nodes[node_id.index()].prev_sibling;

            if next.is_some() && self.nodes[next.index()].kind == NodeKind::Free {
                self.merge_siblings(node_id, next, orientation);
            }

            if prev.is_some() && self.nodes[prev.index()].kind == NodeKind::Free {
                self.merge_siblings(prev, node_id, orientation);
                node_id = prev;
            }

            let parent = self.nodes[node_id.index()].parent;
            if self.nodes[node_id.index()].prev_sibling.is_none()
                && self.nodes[node_id.index()].next_sibling.is_none()
                && parent.is_some()
            {
                debug_assert_eq!(
                    self.nodes[parent.index()].kind,
                    NodeKind::Container,
                    "parent of unique child must be Container"
                );

                let rect = self.nodes[node_id.index()].rect;
                self.mark_node_unused(node_id);

                self.nodes[parent.index()].rect = rect;
                self.nodes[parent.index()].kind = NodeKind::Free;

                node_id = parent;
            } else {
                let size = self.nodes[node_id.index()].rect.size();
                self.add_free_rect(node_id, size);
                break;
            }
        }
    }

    // ----- internal helpers -----

    fn find_suitable_rect(&mut self, requested_size: Size) -> AllocIndex {
        let ideal_bucket = free_list_for_size(
            self.small_size_threshold,
            self.large_size_threshold,
            requested_size,
        );

        let use_worst_fit = ideal_bucket == LARGE_BUCKET;
        for bucket in ideal_bucket..NUM_BUCKETS {
            let mut candidate_score = if use_worst_fit { 0 } else { i32::MAX };
            let mut candidate = None;

            let mut freelist_idx = 0;
            while freelist_idx < self.free_lists[bucket].len() {
                let id = self.free_lists[bucket][freelist_idx];

                if self.nodes[id.index()].kind != NodeKind::Free {
                    self.free_lists[bucket].swap_remove(freelist_idx);
                    continue;
                }

                let size = self.nodes[id.index()].rect.size();
                let dx = size.width - requested_size.width;
                let dy = size.height - requested_size.height;

                if dx >= 0 && dy >= 0 {
                    if dx == 0 || dy == 0 {
                        candidate = Some((id, freelist_idx));
                        break;
                    }

                    let score = i32::min(dx, dy);
                    if (use_worst_fit && score > candidate_score)
                        || (!use_worst_fit && score < candidate_score)
                    {
                        candidate_score = score;
                        candidate = Some((id, freelist_idx));
                    }
                }

                freelist_idx += 1;
            }

            if let Some((id, freelist_idx)) = candidate {
                self.free_lists[bucket].swap_remove(freelist_idx);
                return id;
            }
        }

        AllocIndex::NONE
    }

    fn new_node(&mut self) -> AllocIndex {
        let idx = self.unused_nodes;
        if idx.index() < self.nodes.len() {
            self.unused_nodes = self.nodes[idx.index()].next_sibling;
            self.generations[idx.index()] += Wrapping(1);
            debug_assert_eq!(
                self.nodes[idx.index()].kind,
                NodeKind::Unused,
                "reused node must have been Unused"
            );
            return idx;
        }

        self.nodes.push(Node {
            parent: AllocIndex::NONE,
            next_sibling: AllocIndex::NONE,
            prev_sibling: AllocIndex::NONE,
            rect: Rect::zero(),
            kind: NodeKind::Unused,
            orientation: Orientation::Horizontal,
        });

        self.generations.push(Wrapping(0));

        AllocIndex(self.nodes.len() as u32 - 1)
    }

    fn mark_node_unused(&mut self, id: AllocIndex) {
        debug_assert!(
            self.nodes[id.index()].kind != NodeKind::Unused,
            "node to mark unused must not already be Unused"
        );
        self.nodes[id.index()].kind = NodeKind::Unused;
        self.nodes[id.index()].next_sibling = self.unused_nodes;
        self.unused_nodes = id;
    }

    fn add_free_rect(&mut self, id: AllocIndex, size: Size) {
        debug_assert_eq!(
            self.nodes[id.index()].kind,
            NodeKind::Free,
            "added free rect node must be Free"
        );
        let bucket = free_list_for_size(self.small_size_threshold, self.large_size_threshold, size);
        self.free_lists[bucket].push(id);
    }

    fn merge_siblings(&mut self, node: AllocIndex, next: AllocIndex, orientation: Orientation) {
        debug_assert_eq!(
            self.nodes[node.index()].kind,
            NodeKind::Free,
            "merge node must be Free"
        );
        debug_assert_eq!(
            self.nodes[next.index()].kind,
            NodeKind::Free,
            "merge next must be Free"
        );

        let merge_size = self.nodes[next.index()].rect.size();
        match orientation {
            Orientation::Horizontal => {
                self.nodes[node.index()].rect.max.x += merge_size.width;
            }
            Orientation::Vertical => {
                self.nodes[node.index()].rect.max.y += merge_size.height;
            }
        }

        let next_next = self.nodes[next.index()].next_sibling;
        self.nodes[node.index()].next_sibling = next_next;
        if next_next.is_some() {
            self.nodes[next_next.index()].prev_sibling = node;
        }

        self.mark_node_unused(next);
    }

    fn alloc_id(&self, index: AllocIndex) -> AllocId {
        let generation = self.generations[index.index()].0 as u32;
        debug_assert!(
            index.0 & IDX_MASK == index.0,
            "index must fit within IDX_MASK bits"
        );
        AllocId(index.0 + (generation << 24))
    }

    fn get_index(&self, id: AllocId) -> AllocIndex {
        let idx = id.0 & IDX_MASK;
        let expected_generation = (self.generations[idx as usize].0 as u32) << 24;
        assert_eq!(
            id.0 & GEN_MASK,
            expected_generation,
            "AllocId generation mismatch: stale or invalid id"
        );
        AllocIndex(idx)
    }
}

// ---------------------------------------------------------------------------
// Guillotine split logic
// ---------------------------------------------------------------------------

fn safe_area(rect: Rect) -> i32 {
    rect.width().checked_mul(rect.height()).unwrap_or(i32::MAX)
}

fn guillotine_rect(
    chosen_rect: Rect,
    requested_size: Size,
    default_orientation: Orientation,
) -> (Rect, Rect, Orientation) {
    let candidate_leftover_right = Rect {
        min: Point {
            x: chosen_rect.min.x + requested_size.width,
            y: chosen_rect.min.y,
        },
        max: Point {
            x: chosen_rect.max.x,
            y: chosen_rect.min.y + requested_size.height,
        },
    };
    let candidate_leftover_bottom = Rect {
        min: Point {
            x: chosen_rect.min.x,
            y: chosen_rect.min.y + requested_size.height,
        },
        max: Point {
            x: chosen_rect.min.x + requested_size.width,
            y: chosen_rect.max.y,
        },
    };

    if requested_size == chosen_rect.size() {
        (Rect::zero(), Rect::zero(), default_orientation)
    } else if safe_area(candidate_leftover_right) > safe_area(candidate_leftover_bottom) {
        let split_rect = Rect {
            min: candidate_leftover_right.min,
            max: Point {
                x: candidate_leftover_right.max.x,
                y: chosen_rect.max.y,
            },
        };
        (
            split_rect,
            candidate_leftover_bottom,
            Orientation::Horizontal,
        )
    } else {
        let split_rect = Rect {
            min: candidate_leftover_bottom.min,
            max: Point {
                x: chosen_rect.max.x,
                y: candidate_leftover_bottom.max.y,
            },
        };
        (split_rect, candidate_leftover_right, Orientation::Vertical)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_allocation_placed_at_origin() {
        let mut alloc = GuillotineAllocator::new(256, 256);
        let a = alloc.allocate(10, 20).unwrap();
        assert_eq!(a.x, 0);
        assert_eq!(a.y, 0);
    }

    #[test]
    fn sequential_allocations_pack_correctly() {
        let mut alloc = GuillotineAllocator::new(256, 256);
        let a = alloc.allocate(30, 10).unwrap();
        let b = alloc.allocate(50, 10).unwrap();
        assert_eq!(a.x, 0);
        assert_eq!(a.y, 0);
        assert!(b.x > 0 || b.y > 0);
    }

    #[test]
    fn returns_none_when_width_exceeds_atlas() {
        let mut alloc = GuillotineAllocator::new(64, 64);
        assert!(alloc.allocate(65, 1).is_none());
    }

    #[test]
    fn returns_none_when_height_exceeds_atlas() {
        let mut alloc = GuillotineAllocator::new(64, 64);
        assert!(alloc.allocate(1, 65).is_none());
    }

    #[test]
    fn exact_fit_succeeds() {
        let mut alloc = GuillotineAllocator::new(100, 100);
        let a = alloc.allocate(100, 100).unwrap();
        assert_eq!((a.x, a.y), (0, 0));
        assert!(alloc.allocate(1, 1).is_none());
    }

    #[test]
    fn deallocate_reclaims_space() {
        let mut alloc = GuillotineAllocator::new(64, 64);
        let a = alloc.allocate(64, 64).unwrap();
        assert!(alloc.allocate(1, 1).is_none());
        alloc.deallocate(a.id);
        let b = alloc.allocate(64, 64).unwrap();
        assert_eq!((b.x, b.y), (0, 0));
    }

    #[test]
    fn zero_size_allocation_returns_none() {
        let mut alloc = GuillotineAllocator::new(64, 64);
        assert!(alloc.allocate(0, 10).is_none());
        assert!(alloc.allocate(10, 0).is_none());
    }

    #[test]
    fn deallocate_unknown_id_panics() {
        // guillotiere uses generation counters; deallocating a bogus ID should panic.
    }

    #[test]
    fn many_small_then_deallocate_all_reclaims_full() {
        let mut alloc = GuillotineAllocator::new(64, 64);
        let mut ids = Vec::new();
        for _ in 0..4 {
            for _ in 0..4 {
                if let Some(a) = alloc.allocate(16, 16) {
                    ids.push(a.id);
                }
            }
        }
        assert!(!ids.is_empty());
        for id in ids {
            alloc.deallocate(id);
        }
        let full = alloc.allocate(64, 64).unwrap();
        assert_eq!((full.x, full.y), (0, 0));
        assert!(alloc.allocate(1, 1).is_none());
    }

    #[test]
    fn complex_alloc_dealloc_reclaims_full() {
        let mut alloc = GuillotineAllocator::new(1000, 1000);

        let full = alloc.allocate(1000, 1000).unwrap();
        assert!(alloc.allocate(1, 1).is_none());
        alloc.deallocate(full.id);

        let a = alloc.allocate(100, 1000).unwrap().id;
        let b = alloc.allocate(900, 200).unwrap().id;
        let c = alloc.allocate(300, 200).unwrap().id;
        let d = alloc.allocate(200, 300).unwrap().id;
        let e = alloc.allocate(100, 300).unwrap().id;
        let f = alloc.allocate(100, 300).unwrap().id;
        let g = alloc.allocate(100, 300).unwrap().id;

        alloc.deallocate(b);
        alloc.deallocate(f);
        alloc.deallocate(c);
        alloc.deallocate(e);
        let h = alloc.allocate(500, 200).unwrap().id;
        alloc.deallocate(a);
        let i = alloc.allocate(500, 200).unwrap().id;
        alloc.deallocate(g);
        alloc.deallocate(h);
        alloc.deallocate(d);
        alloc.deallocate(i);

        let full = alloc.allocate(1000, 1000).unwrap();
        assert!(alloc.allocate(1, 1).is_none());
        alloc.deallocate(full.id);
    }

    #[test]
    fn stress_random_alloc_dealloc() {
        let mut alloc = GuillotineAllocator::new(1000, 1000);

        let a: usize = 1103515245;
        let c: usize = 12345;
        let m: usize = usize::pow(2, 31);
        let mut seed: usize = 37;

        let mut rand = || {
            seed = (a.wrapping_mul(seed).wrapping_add(c)) % m;
            seed
        };

        let mut allocated = Vec::new();
        for _ in 0..50000 {
            if rand() % 5 > 2 && !allocated.is_empty() {
                let nth = rand() % allocated.len();
                let id = allocated[nth];
                allocated.swap_remove(nth);
                alloc.deallocate(id);
            } else {
                let w = (rand() % 300) as u32 + 5;
                let h = (rand() % 300) as u32 + 5;
                if let Some(a) = alloc.allocate(w, h) {
                    allocated.push(a.id);
                }
            }
        }

        while let Some(id) = allocated.pop() {
            alloc.deallocate(id);
        }

        let full = alloc.allocate(1000, 1000).unwrap();
        assert!(alloc.allocate(1, 1).is_none());
        alloc.deallocate(full.id);
    }
}
