use crate::{Rectangle, Size};
use euclid::{point2, size2, vec2};

use std::num::Wrapping;

const LARGE_BUCKET: usize = 2;
const MEDIUM_BUCKET: usize = 1;
const SMALL_BUCKET: usize = 0;
const NUM_BUCKETS: usize = 3;

fn free_list_for_size(small_threshold: i32, large_threshold: i32, size: &Size) -> usize {
    if size.width >= large_threshold || size.height >= large_threshold {
        LARGE_BUCKET
    } else if size.width >= small_threshold || size.height >= small_threshold {
        MEDIUM_BUCKET
    } else {
        SMALL_BUCKET
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct AllocIndex(u32);
impl AllocIndex {
    const NONE: AllocIndex = AllocIndex(std::u32::MAX);

    fn index(self) -> usize {
        self.0 as usize
    }

    fn is_none(self) -> bool {
        self == AllocIndex::NONE
    }

    fn is_some(self) -> bool {
        self != AllocIndex::NONE
    }
}

/// ID referring to an allocated rectangle.
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct AllocId(pub(crate) u32);

impl AllocId {
    pub fn serialize(&self) -> u32 {
        self.0
    }

    pub fn deserialize(bytes: u32) -> Self {
        AllocId(bytes)
    }
}

const GEN_MASK: u32 = 0xFF000000;
const IDX_MASK: u32 = 0x00FFFFFF;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum Orientation {
    Vertical,
    Horizontal,
}

impl Orientation {
    fn flipped(self) -> Self {
        match self {
            Orientation::Vertical => Orientation::Horizontal,
            Orientation::Horizontal => Orientation::Vertical,
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Container,
    Alloc,
    Free,
    Unused,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
struct Node {
    parent: AllocIndex,
    next_sibling: AllocIndex,
    prev_sibling: AllocIndex,
    kind: NodeKind,
    orientation: Orientation,
    rect: Rectangle,
}

/// Options to tweak the behavior of the atlas allocator.
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct AllocatorOptions {
    /// Round the rectangle sizes up to a multiple of this value.
    ///
    /// Width and height alignments must be superior to zero.
    ///
    /// Default value: (1, 1),
    pub alignment: Size,

    /// Value below which a size is considered small.
    ///
    /// This is value is used to speed up the storage and lookup of free rectangles.
    /// This value must be inferior or equal to `large_size_threshold`
    ///
    /// Default value: 32,
    pub small_size_threshold: i32,

    /// Value above which a size is considered large.
    ///
    /// This is value is used to speed up the storage and lookup of free rectangles.
    /// This value must be inferior or equal to `large_size_threshold`
    ///
    /// Default value: 256,
    pub large_size_threshold: i32,
}

pub const DEFAULT_OPTIONS: AllocatorOptions = AllocatorOptions {
    alignment: size2(1, 1),
    large_size_threshold: 256,
    small_size_threshold: 32,
};

impl Default for AllocatorOptions {
    fn default() -> Self {
        DEFAULT_OPTIONS
    }
}

/// A dynamic texture atlas allocator using the guillotine algorithm.
///
/// The guillotine algorithm is assisted by a data structure that keeps track of
/// neighboring rectangles to provide fast deallocation and coalescing.
///
/// ## Goals
///
/// Coalescing free rectangles, in the context of dynamic atlas allocation can be
/// prohibitively expensive under real-time constraints if the algorithm needs to
/// visit a large amount of free rectangles to find merge candidates.
///
/// This implementation proposes a compromise with fast (constant time) search
/// for merge candidates at the expense of some (constant time) bookkeeping overhead
/// when allocating and removing rectangles and imperfect defragmentation (see the
/// "Limitations" section below.
///
/// The subdivision scheme uses the worst fit variant of the guillotine algorithm
/// for its simplicity and CPU efficiency.
///
/// ## The data structure
///
/// We maintain a tree with allocated and free rectangles as leaf nodes and
/// containers as non-leaf nodes.
///
/// The direct children of a Containers's form an ordered horizontal or vertical
/// sequence of rectangles that cover exactly their parent container's area.
///
/// For example, a subdivision such as this one:
///
/// ```ascii
/// +-----------+----------+---+---+--+---------+---+
/// |           |          | C | D |E | F       | G |
/// |           |          +---+---+--+---------+---+
/// |     A     |    B     |                        |
/// |           |          |           H            |
/// |           |          |                        |
/// +------+----+----------+-+----------------------+
/// |      |        J        |                      |
/// |  I   +-----------------+          L           |
/// |      |        K        |                      |
/// +------+-----------------+----------------------+
/// ```
///
/// Would have a tree of the form:
///
/// ```ascii
///
///  Tree                | Layout
/// ---------------------+------------
///                      |
///           #          |
///           |          |
///      +----+----+. . .|. vertical
///      |         |     |
///      #         #     |
///      |         |     |
///    +-+-+ . . +-+-+. .|. horizontal
///    | | |     | | |   |
///    A B #     I # L   |
///        |       |     |
///      +-+-+ . +-+-+. .|. vertical
///      |   |   |   |   |
///      #   H   J   K   |
///      |               |
///  +-+-+-+-+. . . . . .|. horizontal
///  | | | | |           |
///  C D E F G           |
/// ```
///
/// Where container nodes are represented with "#".
///
/// Note that if a horizontal container is the direct child of another
/// horizontal container, we can merge the two into a single horizontal
/// sequence.
/// We use this property to always keep the tree in its simplest form.
/// In practice this means that the orientation of a container is always
/// the opposite of the orientation of its parent, if any.
///
/// The goal of this data structure is to quickly find neighboring free
/// rectangles that can be coalesced into fewer rectangles.
/// This structure guarantees that two consecutive children of the same
/// container, if both rectangles are free, can be coalesced into a single
/// one.
///
/// An important thing to note about this tree structure is that we only
/// use it to visit neighbor and parent nodes. As a result we don't care
/// about whether the tree is balanced, although flat sequences of children
/// tend to offer more opportunity for coalescing than deeply nested structures
/// Either way, the cost of finding potential merges is the same because
/// each node stores the indices of their siblings, and we never have to
/// traverse any global list of free rectangle nodes.
///
/// ### Merging siblings
///
/// As soon as two consecutive sibling nodes are marked as "free", they are coalesced
/// into a single node.
///
/// In the example below, we just deallocated the rectangle `B`, which is a sibling of
/// `A` which is free and `C` which is still allocated. `A` and `B` are merged and this
/// change is reflected on the tree as shown below:
///
/// ```ascii
/// +---+---+---+         #               +-------+---+         #
/// |   |   |///|         |               |       |///|         |
/// | A | B |/C/|     +---+---+           | AB    |/C/|     +---+---+
/// |   |   |///|     |       |           |       |///|     |       |
/// +---+---+---+     #       D           +-------+---+     #       D
/// | D         |     |            ->     | D         |     |
/// |           |   +-+-+                 |           |   +-+-+
/// |           |   | | |                 |           |   |   |
/// +-----------+   A B C                 +-----------+   AB  C
/// ```
///
/// ### Merging unique children with their parents
///
/// In the previous example `C` was an allocated slot. Let's now deallocate it:
///
/// ```ascii
/// +-------+---+         #               +-----------+         #                 #
/// |       |   |         |               |           |         |                 |
/// | AB    | C |     +---+---+           | ABC       |     +---+---+         +---+---+
/// |       |   |     |       |           |           |     |       |         |       |
/// +-------+---+     #       D           +-----------+     #       D        ABC      D
/// | D         |     |            ->     | D         |     |           ->
/// |           |   +-+-+                 |           |     +
/// |           |   |   |                 |           |     |
/// +-----------+   AB  C                 +-----------+    ABC
/// ```
///
/// Deallocating `C` allowed it to merge with the free rectangle `AB`, making the
/// resulting node `ABC` the only child of its parent container. As a result the
/// node `ABC` was lifted up the tree to replace its parent.
///
/// In this example, assuming `D` to also be a free rectangle, `ABC` and `D` would
/// be immediately merged and the resulting node `ABCD`, also being only child of
/// its parent container, would replace its parent, turning the tree into a single
/// node `ABCD`.
///
/// ### Limitations
///
/// This strategy can miss some opportunities for coalescing free rectangles
/// when the two sibling containers are split exactly the same way.
///
/// For example:
///
/// ```ascii
/// +---------+------+
/// |    A    |  B   |
/// |         |      |
/// +---------+------+
/// |    C    |  D   |
/// |         |      |
/// +---------+------+
/// ```
///
/// Could be the result of either a vertical followed with two horizontal splits,
/// or an horizontal then two vertical splits.
///
/// ```ascii
///  Tree            | Layout             Tree            | Layout
/// -----------------+------------       -----------------+------------
///         #        |                           #        |
///         |        |                           |        |
///     +---+---+ . .|. Vertical             +---+---+ . .|. Horizontal
///     |       |    |                       |       |    |
///     #       #    |               or      #       #    |
///     |       |    |                       |       |    |
///   +-+-+ . +-+-+ .|. Horizontal         +-+-+ . +-+-+ .|. Vertical
///   |   |   |   |  |                     |   |   |   |  |
///   A   B   C   D  |                     A   C   B   D  |
/// ```
///
/// In the former case A can't be merged with C nor B with D because they are not siblings.
///
/// For a lot of workloads it is rather rare for two consecutive sibling containers to be
/// subdivided exactly the same way. In this situation losing the ability to merge rectangles
/// that aren't under the same container is good compromise between the CPU cost of coalescing
/// and the fragmentation of the atlas.
///
/// This algorithm is, however, not the best solution for very "structured" grid-like
/// subdivision patterns where the ability to merge across containers would have provided
/// frequent defragmentation opportunities.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct AtlasAllocator {
    nodes: Vec<Node>,
    /// Free lists are split into a small a medium and a large bucket for faster lookups.
    free_lists: [Vec<AllocIndex>; NUM_BUCKETS],

    /// Index of the first element of an intrusive linked list of unused nodes.
    /// The `next_sibling` member of unused node serves as the linked list link.
    unused_nodes: AllocIndex,

    /// We keep a per-node generation counter to reduce the likelihood of ID reuse bugs
    /// going unnoticed.
    generations: Vec<Wrapping<u8>>,

    /// See `AllocatorOptions`.
    alignment: Size,

    /// See `AllocatorOptions`.
    small_size_threshold: i32,

    /// See `AllocatorOptions`.
    large_size_threshold: i32,

    /// Total size of the atlas.
    size: Size,

    /// Index of one of the top-level nodes in the tree.
    root_node: AllocIndex,
}

// Some notes about the atlas's tree data structure:
//
//      (AllocIndex::NONE)                (AllocIndex::NONE)
//              ^                                 ^
//              | parent                          | parent
//           +---------+ next sibling         +---------+ next sibling
// ... ------|Container|---------------------->|Free     |---> (AllocIndex::NONE)
//     ----->|         |<----------------------|         |
//           +---------+     previous sibling +---------+
//              ^ ^
//              |  \____________________________
//              |                               \
//              | parent                         \ parent
//           +---------+ next sibling         +---------+ next sibling
// ... ------|Alloc    |---------------------->|Container|---> (AllocIndex::NONE)
//     ----->|         |<----------------------|         |
//           +---------+     previous sibling +---------+
//                                               ^ ^ ^
//                                              /  |  \
//                                                ...
//
// - Nodes are stored in a contiguous vector.
// - Links between the nodes are indices in the vector (AllocIndex), with a magic value
//   AllocIndex::NONE that means no link.
// - Nodes have a link to their parent, but parents do not have a link to any of its children because
//   we never need to traverse the structure from parent to child.
// - All nodes with the same parent are "siblings". An intrusive linked list allows traversing siblings
//   in order. Consecutive siblings share an edge and can be merged if they are both "free".
// - There isn't necessarily a single root node. The top-most level of the tree can have several siblings
//   and their parent index is equal to AllocIndex::NONE. AtlasAllocator::root_node only needs to refer
//   to one of these top-level nodes.
// - After a rectangle has been deallocated, the slot for its node in the vector is not part of the
//   tree anymore in the sense that no node from the tree points to it with its sibling list or parent
//   index. This unused node is available for reuse in a future allocation, and is placed in another
//   linked list (also using AllocIndex), a singly linked list this time, which reuses the next_sibling
//   member of the node. So depending on whether the node kind is Unused or not, the next_sibling
//   member is used different things.
// - We reuse nodes aggressively to avoid growing the vector whenever possible. This is important because
//   the memory footprint of this data structure depends on the capacity of its vectors which don't
//   get deallocated during the lifetime of the atlas.
// - Because nodes are aggressively reused, the same node indices will come up often. To avoid id reuse
//   bugs, a parallel vector of generation counters is maintained.
// - The difference between AllocIndex and AllocId is that the latter embeds a generation ID to help
//   finding id reuse bugs. AllocIndex however only contains the node offset. Internal links in the
//   data structure use AllocIndex, and external users of the data structure only get to see AllocId.

impl AtlasAllocator {
    /// Create an atlas allocator.
    pub fn new(size: Size) -> Self {
        AtlasAllocator::with_options(size, &DEFAULT_OPTIONS)
    }

    /// Create an atlas allocator with the provided options.
    pub fn with_options(size: Size, options: &AllocatorOptions) -> Self {
        assert!(options.alignment.width > 0);
        assert!(options.alignment.height > 0);
        assert!(size.width > 0);
        assert!(size.height > 0);
        assert!(options.large_size_threshold >= options.small_size_threshold);

        let mut free_lists = [Vec::new(), Vec::new(), Vec::new()];
        let bucket = free_list_for_size(
            options.small_size_threshold,
            options.large_size_threshold,
            &size,
        );
        free_lists[bucket].push(AllocIndex(0));

        AtlasAllocator {
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
            alignment: options.alignment,
            small_size_threshold: options.small_size_threshold,
            large_size_threshold: options.large_size_threshold,
            size,
            root_node: AllocIndex(0),
        }
    }

    /// The total size of the atlas.
    pub fn size(&self) -> Size {
        self.size
    }

    /// Allocate a rectangle in the atlas.
    pub fn allocate(&mut self, mut requested_size: Size) -> Option<Allocation> {
        if requested_size.is_empty() {
            return None;
        }

        adjust_size(self.alignment.width, &mut requested_size.width);
        adjust_size(self.alignment.height, &mut requested_size.height);

        // Find a suitable free rect.
        let chosen_id = self.find_suitable_rect(&requested_size);

        if chosen_id.is_none() {
            //println!("failed to allocate {:?}", requested_size);
            //self.print_free_rects();

            // No suitable free rect!
            return None;
        }

        let chosen_node = self.nodes[chosen_id.index()].clone();
        let chosen_rect = chosen_node.rect;
        let allocated_rect = Rectangle {
            min: chosen_rect.min,
            max: chosen_rect.min + requested_size.to_vector(),
        };
        let current_orientation = chosen_node.orientation;
        assert_eq!(chosen_node.kind, NodeKind::Free);

        let (split_rect, leftover_rect, orientation) =
            guillotine_rect(&chosen_node.rect, requested_size, current_orientation);

        // Update the tree.

        let allocated_id;
        let split_id;
        let leftover_id;

        //println!("{:?} -> {:?}", current_orientation, orientation);
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
                // No need to split for the leftover area, we can allocate directly in the chosen node.
                allocated_id = chosen_id;
                let node = &mut self.nodes[chosen_id.index()];
                node.kind = NodeKind::Alloc;
                node.rect = allocated_rect;

                leftover_id = AllocIndex::NONE
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
                    rect: Rectangle::zero(),
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

        assert_eq!(self.nodes[allocated_id.index()].kind, NodeKind::Alloc);

        if split_id.is_some() {
            self.add_free_rect(split_id, &split_rect.size());
        }

        if leftover_id.is_some() {
            self.add_free_rect(leftover_id, &leftover_rect.size());
        }

        //println!("allocated {:?}     split: {:?} leftover: {:?}", allocated_rect, split_rect, leftover_rect);
        //self.print_free_rects();

        #[cfg(feature = "checks")]
        self.check_tree();

        Some(Allocation {
            id: self.alloc_id(allocated_id),
            rectangle: allocated_rect,
        })
    }

    /// Deallocate a rectangle in the atlas.
    pub fn deallocate(&mut self, node_id: AllocId) {
        let mut node_id = self.get_index(node_id);

        assert!(node_id.index() < self.nodes.len());
        assert_eq!(self.nodes[node_id.index()].kind, NodeKind::Alloc);

        self.nodes[node_id.index()].kind = NodeKind::Free;

        loop {
            let orientation = self.nodes[node_id.index()].orientation;

            let next = self.nodes[node_id.index()].next_sibling;
            let prev = self.nodes[node_id.index()].prev_sibling;

            // Try to merge with the next node.
            if next.is_some() && self.nodes[next.index()].kind == NodeKind::Free {
                self.merge_siblings(node_id, next, orientation);
            }

            // Try to merge with the previous node.
            if prev.is_some() && self.nodes[prev.index()].kind == NodeKind::Free {
                self.merge_siblings(prev, node_id, orientation);
                node_id = prev;
            }

            // If this node is now a unique child. We collapse it into its parent and try to merge
            // again at the parent level.
            let parent = self.nodes[node_id.index()].parent;
            if self.nodes[node_id.index()].prev_sibling.is_none()
                && self.nodes[node_id.index()].next_sibling.is_none()
                && parent.is_some()
            {
                debug_assert_eq!(self.nodes[parent.index()].kind, NodeKind::Container);

                self.mark_node_unused(node_id);

                // Replace the parent container with a free node.
                self.nodes[parent.index()].rect = self.nodes[node_id.index()].rect;
                self.nodes[parent.index()].kind = NodeKind::Free;

                // Start again at the parent level.
                node_id = parent;
            } else {
                let size = self.nodes[node_id.index()].rect.size();
                self.add_free_rect(node_id, &size);
                break;
            }
        }

        #[cfg(feature = "checks")]
        self.check_tree();
    }

    pub fn is_empty(&self) -> bool {
        let root = &self.nodes[self.root_node.index()];

        root.kind == NodeKind::Free && root.next_sibling.is_none()
    }

    /// Drop all rectangles, clearing the atlas to its initial state.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.nodes.push(Node {
            parent: AllocIndex::NONE,
            next_sibling: AllocIndex::NONE,
            prev_sibling: AllocIndex::NONE,
            rect: self.size.into(),
            kind: NodeKind::Free,
            orientation: Orientation::Vertical,
        });

        self.root_node = AllocIndex(0);

        self.generations.clear();
        self.generations.push(Wrapping(0));

        self.unused_nodes = AllocIndex::NONE;

        let bucket = free_list_for_size(
            self.small_size_threshold,
            self.large_size_threshold,
            &self.size,
        );
        for i in 0..NUM_BUCKETS {
            self.free_lists[i].clear();
        }
        self.free_lists[bucket].push(AllocIndex(0));
    }

    /// Clear the allocator and reset its size and options.
    pub fn reset(&mut self, size: Size, options: &AllocatorOptions) {
        self.alignment = options.alignment;
        self.small_size_threshold = options.small_size_threshold;
        self.large_size_threshold = options.large_size_threshold;
        self.size = size;

        self.clear();
    }

    /// Recompute the allocations in the atlas and returns a list of the changes.
    ///
    /// Previous ids and rectangles are not valid anymore after this operation as each id/rectangle
    /// pair is assigned to new values which are communicated in the returned change list.
    /// Rearranging the atlas can help reduce fragmentation.
    pub fn rearrange(&mut self) -> ChangeList {
        let size = self.size;
        self.resize_and_rearrange(size)
    }

    /// Identical to `AtlasAllocator::rearrange`, also allowing to change the size of the atlas.
    pub fn resize_and_rearrange(&mut self, new_size: Size) -> ChangeList {
        let mut allocs = Vec::with_capacity(self.nodes.len());
        for (i, node) in self.nodes.iter().enumerate() {
            if node.kind != NodeKind::Alloc {
                continue;
            }
            let id = self.alloc_id(AllocIndex(i as u32));
            allocs.push(Allocation {
                id,
                rectangle: node.rect,
            });
        }

        allocs.sort_by_key(|alloc| safe_area(&alloc.rectangle));
        allocs.reverse();

        self.size = new_size;
        self.clear();

        let mut changes = Vec::new();
        let mut failures = Vec::new();

        for old in allocs {
            let size = old.rectangle.size();
            if let Some(new) = self.allocate(size) {
                changes.push(Change { old, new });
            } else {
                failures.push(old);
            }
        }

        ChangeList { changes, failures }
    }

    /// Resize the atlas without changing the allocations.
    ///
    /// This method is not allowed to shrink the width or height of the atlas.
    pub fn grow(&mut self, new_size: Size) {
        assert!(new_size.width >= self.size.width);
        assert!(new_size.height >= self.size.height);

        let old_size = self.size;
        self.size = new_size;

        let dx = new_size.width - old_size.width;
        let dy = new_size.height - old_size.height;

        // If there is only one node and it is free, just grow it.
        let root = &mut self.nodes[self.root_node.index()];
        if root.kind == NodeKind::Free && root.rect.size() == old_size {
            root.rect.max = root.rect.min + new_size.to_vector();
            return;
        }

        let root_orientation = root.orientation;
        let grows_in_root_orientation = match root_orientation {
            Orientation::Horizontal => dx > 0,
            Orientation::Vertical => dy > 0,
        };

        // If growing along the orientation of the root node, find the right-or-bottom-most sibling
        // and either grow it (if it is free) or append a free node next.
        if grows_in_root_orientation {
            let mut sibling = self.root_node;
            while self.nodes[sibling.index()].next_sibling != AllocIndex::NONE {
                sibling = self.nodes[sibling.index()].next_sibling;
            }
            let node = &mut self.nodes[sibling.index()];
            if node.kind == NodeKind::Free {
                node.rect.max += match root_orientation {
                    Orientation::Horizontal => vec2(dx, 0),
                    Orientation::Vertical => vec2(0, dy),
                };
            } else {
                let rect = match root_orientation {
                    Orientation::Horizontal => {
                        let min = point2(node.rect.max.x, node.rect.min.y);
                        let max = min + vec2(dx, node.rect.height());
                        Rectangle { min, max }
                    }
                    Orientation::Vertical => {
                        let min = point2(node.rect.min.x, node.rect.max.y);
                        let max = min + vec2(node.rect.width(), dy);
                        Rectangle { min, max }
                    }
                };

                let next = self.new_node();
                self.nodes[sibling.index()].next_sibling = next;
                self.nodes[next.index()] = Node {
                    kind: NodeKind::Free,
                    rect,
                    prev_sibling: sibling,
                    next_sibling: AllocIndex::NONE,
                    parent: AllocIndex::NONE,
                    orientation: root_orientation,
                };

                self.add_free_rect(next, &rect.size());
            }
        }

        let grows_in_opposite_orientation = match root_orientation {
            Orientation::Horizontal => dy > 0,
            Orientation::Vertical => dx > 0,
        };

        if grows_in_opposite_orientation {
            let free_node = self.new_node();
            let new_root = self.new_node();

            let old_root = self.root_node;
            self.root_node = new_root;

            let new_root_orientation = root_orientation.flipped();

            let min = match new_root_orientation {
                Orientation::Horizontal => point2(old_size.width, 0),
                Orientation::Vertical => point2(0, old_size.height),
            };
            let max = point2(new_size.width, new_size.height);
            let rect = Rectangle { min, max };

            self.nodes[free_node.index()] = Node {
                parent: AllocIndex::NONE,
                prev_sibling: new_root,
                next_sibling: AllocIndex::NONE,
                kind: NodeKind::Free,
                rect,
                orientation: new_root_orientation,
            };

            self.nodes[new_root.index()] = Node {
                parent: AllocIndex::NONE,
                prev_sibling: AllocIndex::NONE,
                next_sibling: free_node,
                kind: NodeKind::Container,
                rect: Rectangle::zero(),
                orientation: new_root_orientation,
            };

            self.add_free_rect(free_node, &rect.size());

            // Update the nodes that need to be re-parented to the new-root.

            let mut iter = old_root;
            while iter != AllocIndex::NONE {
                self.nodes[iter.index()].parent = new_root;
                iter = self.nodes[iter.index()].next_sibling;
            }

            // That second loop might not be necessary, I think that the root is always the first
            // sibling.
            let mut iter = self.nodes[old_root.index()].next_sibling;
            while iter != AllocIndex::NONE {
                self.nodes[iter.index()].parent = new_root;
                iter = self.nodes[iter.index()].prev_sibling;
            }
        }

        #[cfg(feature = "checks")]
        self.check_tree();
    }

    /// Invoke a callback for each free rectangle in the atlas.
    pub fn for_each_free_rectangle<F>(&self, mut callback: F)
    where
        F: FnMut(&Rectangle),
    {
        for node in &self.nodes {
            if node.kind == NodeKind::Free {
                callback(&node.rect);
            }
        }
    }

    /// Invoke a callback for each allocated rectangle in the atlas.
    pub fn for_each_allocated_rectangle<F>(&self, mut callback: F)
    where
        F: FnMut(AllocId, &Rectangle),
    {
        for (i, node) in self.nodes.iter().enumerate() {
            if node.kind != NodeKind::Alloc {
                continue;
            }

            let id = self.alloc_id(AllocIndex(i as u32));

            callback(id, &node.rect);
        }
    }

    fn find_suitable_rect(&mut self, requested_size: &Size) -> AllocIndex {
        let ideal_bucket = free_list_for_size(
            self.small_size_threshold,
            self.large_size_threshold,
            requested_size,
        );

        let use_worst_fit = ideal_bucket == LARGE_BUCKET;
        for bucket in ideal_bucket..NUM_BUCKETS {
            let mut candidate_score = if use_worst_fit { 0 } else { std::i32::MAX };
            let mut candidate = None;

            let mut freelist_idx = 0;
            while freelist_idx < self.free_lists[bucket].len() {
                let id = self.free_lists[bucket][freelist_idx];

                // During tree simplification we don't remove merged nodes from the free list, so we have
                // to handle it here.
                // This is a tad awkward, but lets us avoid having to maintain a doubly linked list for
                // the free list (which would be needed to remove nodes during tree simplification).
                if self.nodes[id.index()].kind != NodeKind::Free {
                    // remove the element from the free list
                    self.free_lists[bucket].swap_remove(freelist_idx);
                    continue;
                }

                let size = self.nodes[id.index()].rect.size();
                let dx = size.width - requested_size.width;
                let dy = size.height - requested_size.height;

                if dx >= 0 && dy >= 0 {
                    if dx == 0 || dy == 0 {
                        // Perfect fit!
                        candidate = Some((id, freelist_idx));
                        break;
                    }

                    // Favor the largest minimum dimension, except for small
                    // allocations.
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
            debug_assert_eq!(self.nodes[idx.index()].kind, NodeKind::Unused);
            return idx;
        }

        self.nodes.push(Node {
            parent: AllocIndex::NONE,
            next_sibling: AllocIndex::NONE,
            prev_sibling: AllocIndex::NONE,
            rect: Rectangle::zero(),
            kind: NodeKind::Unused,
            orientation: Orientation::Horizontal,
        });

        self.generations.push(Wrapping(0));

        AllocIndex(self.nodes.len() as u32 - 1)
    }

    fn mark_node_unused(&mut self, id: AllocIndex) {
        debug_assert!(self.nodes[id.index()].kind != NodeKind::Unused);
        self.nodes[id.index()].kind = NodeKind::Unused;
        self.nodes[id.index()].next_sibling = self.unused_nodes;
        self.unused_nodes = id;
    }

    #[allow(dead_code)]
    fn print_free_rects(&self) {
        println!("Large:");
        for &id in &self.free_lists[LARGE_BUCKET] {
            if self.nodes[id.index()].kind == NodeKind::Free {
                println!(" - {:?} #{:?}", self.nodes[id.index()].rect, id);
            }
        }
        println!("Medium:");
        for &id in &self.free_lists[MEDIUM_BUCKET] {
            if self.nodes[id.index()].kind == NodeKind::Free {
                println!(" - {:?} #{:?}", self.nodes[id.index()].rect, id);
            }
        }
        println!("Small:");
        for &id in &self.free_lists[SMALL_BUCKET] {
            if self.nodes[id.index()].kind == NodeKind::Free {
                println!(" - {:?} #{:?}", self.nodes[id.index()].rect, id);
            }
        }
    }

    #[cfg(feature = "checks")]
    fn check_siblings(&self, id: AllocIndex, next: AllocIndex, orientation: Orientation) {
        if next.is_none() {
            return;
        }

        if self.nodes[next.index()].prev_sibling != id {
            panic!(
                "error: #{:?}'s next sibling #{:?} has prev sibling #{:?}",
                id,
                next,
                self.nodes[next.index()].prev_sibling
            );
        }
        assert_eq!(self.nodes[next.index()].prev_sibling, id);

        match self.nodes[id.index()].kind {
            NodeKind::Container | NodeKind::Unused => {
                return;
            }
            _ => {}
        }
        match self.nodes[next.index()].kind {
            NodeKind::Container | NodeKind::Unused => {
                return;
            }
            _ => {}
        }

        let r1 = self.nodes[id.index()].rect;
        let r2 = self.nodes[next.index()].rect;
        match orientation {
            Orientation::Horizontal => {
                assert_eq!(r1.min.y, r2.min.y);
                assert_eq!(r1.max.y, r2.max.y);
            }
            Orientation::Vertical => {
                assert_eq!(r1.min.x, r2.min.x);
                assert_eq!(r1.max.x, r2.max.x);
            }
        }
    }

    #[cfg(feature = "checks")]
    fn check_tree(&self) {
        for node_idx in 0..self.nodes.len() {
            let node = &self.nodes[node_idx];

            if node.kind == NodeKind::Unused {
                if node.next_sibling.is_some() {
                    assert_eq!(self.nodes[node.next_sibling.index()].kind, NodeKind::Unused);
                }
                continue;
            }

            let mut iter = node.next_sibling;
            while iter.is_some() {
                assert_eq!(self.nodes[iter.index()].orientation, node.orientation);
                assert_eq!(self.nodes[iter.index()].parent, node.parent);
                assert!(self.nodes[iter.index()].kind != NodeKind::Unused);
                let next = self.nodes[iter.index()].next_sibling;

                #[cfg(feature = "checks")]
                self.check_siblings(iter, next, node.orientation);

                iter = next;
            }

            if node.parent.is_some() {
                if self.nodes[node.parent.index()].kind != NodeKind::Container {
                    panic!("error: child: {:?} parent: {:?}", node_idx, node.parent);
                }
                assert_eq!(
                    self.nodes[node.parent.index()].orientation,
                    node.orientation.flipped()
                );
                assert_eq!(self.nodes[node.parent.index()].kind, NodeKind::Container);
            }
        }
    }

    fn add_free_rect(&mut self, id: AllocIndex, size: &Size) {
        debug_assert_eq!(self.nodes[id.index()].kind, NodeKind::Free);
        let bucket = free_list_for_size(self.small_size_threshold, self.large_size_threshold, size);
        //println!("add free rect #{:?} size {} bucket {}", id, size, bucket);
        self.free_lists[bucket].push(id);
    }

    // Merge `next` into `node` and append `next` to a list of available `nodes`vector slots.
    fn merge_siblings(&mut self, node: AllocIndex, next: AllocIndex, orientation: Orientation) {
        debug_assert_eq!(self.nodes[node.index()].kind, NodeKind::Free);
        debug_assert_eq!(self.nodes[next.index()].kind, NodeKind::Free);
        let r1 = self.nodes[node.index()].rect;
        let r2 = self.nodes[next.index()].rect;
        //println!("merge {} #{:?} and {} #{:?}       {:?}", r1, node, r2, next, orientation);
        let merge_size = self.nodes[next.index()].rect.size();
        match orientation {
            Orientation::Horizontal => {
                debug_assert_eq!(r1.min.y, r2.min.y);
                debug_assert_eq!(r1.max.y, r2.max.y);
                self.nodes[node.index()].rect.max.x += merge_size.width;
            }
            Orientation::Vertical => {
                debug_assert_eq!(r1.min.x, r2.min.x);
                debug_assert_eq!(r1.max.x, r2.max.x);
                self.nodes[node.index()].rect.max.y += merge_size.height;
            }
        }

        // Remove the merged node from the sibling list.
        let next_next = self.nodes[next.index()].next_sibling;
        self.nodes[node.index()].next_sibling = next_next;
        if next_next.is_some() {
            self.nodes[next_next.index()].prev_sibling = node;
        }

        // Add the merged node to the list of available slots in the nodes vector.
        self.mark_node_unused(next);
    }

    fn alloc_id(&self, index: AllocIndex) -> AllocId {
        let generation = self.generations[index.index()].0 as u32;
        debug_assert!(index.0 & IDX_MASK == index.0);
        AllocId(index.0 + (generation << 24))
    }

    fn get_index(&self, id: AllocId) -> AllocIndex {
        let idx = id.0 & IDX_MASK;
        let expected_generation = (self.generations[idx as usize].0 as u32) << 24;
        assert_eq!(id.0 & GEN_MASK, expected_generation);
        AllocIndex(idx)
    }
}

impl std::ops::Index<AllocId> for AtlasAllocator {
    type Output = Rectangle;
    fn index(&self, index: AllocId) -> &Rectangle {
        let idx = self.get_index(index);

        &self.nodes[idx.index()].rect
    }
}

/// A simpler atlas allocator implementation that can allocate rectangles but not deallocate them.
pub struct SimpleAtlasAllocator {
    free_rects: [Vec<Rectangle>; 3],
    alignment: Size,
    small_size_threshold: i32,
    large_size_threshold: i32,
    size: Size,
}

impl SimpleAtlasAllocator {
    /// Create a simple atlas allocator with default options.
    pub fn new(size: Size) -> Self {
        Self::with_options(size, &DEFAULT_OPTIONS)
    }

    /// Create a simple atlas allocator with the provided options.
    pub fn with_options(size: Size, options: &AllocatorOptions) -> Self {
        let bucket = free_list_for_size(
            options.small_size_threshold,
            options.large_size_threshold,
            &size,
        );

        let mut free_rects = [Vec::new(), Vec::new(), Vec::new()];
        free_rects[bucket].push(size.into());

        SimpleAtlasAllocator {
            free_rects,
            alignment: options.alignment,
            small_size_threshold: options.small_size_threshold,
            large_size_threshold: options.large_size_threshold,
            size,
        }
    }

    /// Drop all rectangles, clearing the atlas to its initial state.
    pub fn clear(&mut self) {
        for i in 0..NUM_BUCKETS {
            self.free_rects[i].clear();
        }

        let bucket = free_list_for_size(
            self.small_size_threshold,
            self.large_size_threshold,
            &self.size,
        );

        self.free_rects[bucket].push(self.size.into());
    }

    /// Clear the allocator and reset its size and options.
    pub fn reset(&mut self, size: Size, options: &AllocatorOptions) {
        self.alignment = options.alignment;
        self.small_size_threshold = options.small_size_threshold;
        self.large_size_threshold = options.large_size_threshold;
        self.size = size;

        self.clear();
    }

    pub fn is_empty(&self) -> bool {
        for b in 0..NUM_BUCKETS {
            for rect in &self.free_rects[b] {
                return rect.size() == self.size;
            }
        }

        // This should be unreachable.
        return false;
    }

    /// The total size of the atlas.
    pub fn size(&self) -> Size {
        self.size
    }

    /// Allocate a rectangle in the atlas.
    pub fn allocate(&mut self, mut requested_size: Size) -> Option<Rectangle> {
        if requested_size.is_empty() {
            return None;
        }

        adjust_size(self.alignment.width, &mut requested_size.width);
        adjust_size(self.alignment.height, &mut requested_size.height);

        let ideal_bucket = free_list_for_size(
            self.small_size_threshold,
            self.large_size_threshold,
            &requested_size,
        );

        let use_worst_fit = ideal_bucket == LARGE_BUCKET;

        let mut chosen_rect = None;
        for bucket in ideal_bucket..NUM_BUCKETS {
            let mut candidate_score = if use_worst_fit { 0 } else { std::i32::MAX };
            let mut candidate = None;

            for (index, rect) in self.free_rects[bucket].iter().enumerate() {
                let dx = rect.width() - requested_size.width;
                let dy = rect.height() - requested_size.height;

                if dx >= 0 && dy >= 0 {
                    if dx == 0 || dy == 0 {
                        // Perfect fit!
                        candidate = Some(index);
                        break;
                    }

                    let score = i32::min(dx, dy);
                    if (use_worst_fit && score > candidate_score)
                        || (!use_worst_fit && score < candidate_score)
                    {
                        candidate_score = score;
                        candidate = Some(index);
                    }
                }
            }

            if let Some(index) = candidate {
                let rect = self.free_rects[bucket].remove(index);
                chosen_rect = Some(rect);
                break;
            }
        }

        if let Some(rect) = chosen_rect {
            let (split_rect, leftover_rect, _) =
                guillotine_rect(&rect, requested_size, Orientation::Vertical);
            self.add_free_rect(&split_rect);
            self.add_free_rect(&leftover_rect);

            return Some(Rectangle {
                min: rect.min,
                max: rect.min + requested_size.to_vector(),
            });
        }

        None
    }

    /// Resize the atlas without changing the allocations.
    ///
    /// This method is not allowed to shrink the width or height of the atlas.
    pub fn grow(&mut self, new_size: Size) {
        assert!(new_size.width >= self.size.width);
        assert!(new_size.height >= self.size.height);

        let (split_rect, leftover_rect, _) =
            guillotine_rect(&new_size.into(), self.size, Orientation::Vertical);

        self.size = new_size;

        self.add_free_rect(&split_rect);
        self.add_free_rect(&leftover_rect);
    }

    /// Initialize this simple allocator with the content of an atlas allocator.
    pub fn init_from_allocator(&mut self, src: &AtlasAllocator) {
        self.size = src.size;
        self.small_size_threshold = src.small_size_threshold;
        self.large_size_threshold = src.large_size_threshold;

        for bucket in 0..NUM_BUCKETS {
            for id in src.free_lists[bucket].iter() {
                // During tree simplification we don't remove merged nodes from the free list, so we have
                // to handle it here.
                // This is a tad awkward, but lets us avoid having to maintain a doubly linked list for
                // the free list (which would be needed to remove nodes during tree simplification).
                if src.nodes[id.index()].kind != NodeKind::Free {
                    continue;
                }

                self.free_rects[bucket].push(src.nodes[id.index()].rect);
            }
        }
    }

    fn add_free_rect(&mut self, rect: &Rectangle) {
        if rect.width() < self.alignment.width || rect.height() < self.alignment.height {
            return;
        }

        let bucket = free_list_for_size(
            self.small_size_threshold,
            self.large_size_threshold,
            &rect.size(),
        );

        self.free_rects[bucket].push(*rect);
    }
}

fn adjust_size(alignment: i32, size: &mut i32) {
    let rem = *size % alignment;
    if rem > 0 {
        *size += alignment - rem;
    }
}

/// Compute the area, saturating at i32::MAX instead of overflowing.
fn safe_area(rect: &Rectangle) -> i32 {
    rect.width()
        .checked_mul(rect.height())
        .unwrap_or(std::i32::MAX)
}

fn guillotine_rect(
    chosen_rect: &Rectangle,
    requested_size: Size,
    default_orientation: Orientation,
) -> (Rectangle, Rectangle, Orientation) {
    // Decide whether to split horizontally or vertically.
    //
    // If the chosen free rectangle is bigger than the requested size, we subdivide it
    // into an allocated rectangle, a split rectangle and a leftover rectangle:
    //
    // +-----------+-------------+
    // |///////////|             |
    // |/allocated/|             |
    // |///////////|             |
    // +-----------+             |
    // |                         |
    // |          chosen         |
    // |                         |
    // +-------------------------+
    //
    // Will be split into either:
    //
    // +-----------+-------------+
    // |///////////|             |
    // |/allocated/|  leftover   |
    // |///////////|             |
    // +-----------+-------------+
    // |                         |
    // |          split          |
    // |                         |
    // +-------------------------+
    //
    // or:
    //
    // +-----------+-------------+
    // |///////////|             |
    // |/allocated/|             |
    // |///////////|    split    |
    // +-----------+             |
    // |           |             |
    // | leftover  |             |
    // |           |             |
    // +-----------+-------------+

    let candidate_leftover_rect_to_right = Rectangle {
        min: chosen_rect.min + vec2(requested_size.width, 0),
        max: point2(chosen_rect.max.x, chosen_rect.min.y + requested_size.height),
    };
    let candidate_leftover_rect_to_bottom = Rectangle {
        min: chosen_rect.min + vec2(0, requested_size.height),
        max: point2(chosen_rect.min.x + requested_size.width, chosen_rect.max.y),
    };

    let split_rect;
    let leftover_rect;
    let orientation;
    if requested_size == chosen_rect.size() {
        // Perfect fit.
        orientation = default_orientation;
        split_rect = Rectangle::zero();
        leftover_rect = Rectangle::zero();
    } else if safe_area(&candidate_leftover_rect_to_right)
        > safe_area(&candidate_leftover_rect_to_bottom)
    {
        leftover_rect = candidate_leftover_rect_to_bottom;
        split_rect = Rectangle {
            min: candidate_leftover_rect_to_right.min,
            max: point2(candidate_leftover_rect_to_right.max.x, chosen_rect.max.y),
        };
        orientation = Orientation::Horizontal;
    } else {
        leftover_rect = candidate_leftover_rect_to_right;
        split_rect = Rectangle {
            min: candidate_leftover_rect_to_bottom.min,
            max: point2(chosen_rect.max.x, candidate_leftover_rect_to_bottom.max.y),
        };
        orientation = Orientation::Vertical;
    }

    (split_rect, leftover_rect, orientation)
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Allocation {
    pub id: AllocId,
    pub rectangle: Rectangle,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Change {
    pub old: Allocation,
    pub new: Allocation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChangeList {
    pub changes: Vec<Change>,
    pub failures: Vec<Allocation>,
}

impl ChangeList {
    pub fn empty() -> Self {
        ChangeList {
            changes: Vec::new(),
            failures: Vec::new(),
        }
    }
}

/// Dump a visual representation of the atlas in SVG format.
pub fn dump_svg(atlas: &AtlasAllocator, output: &mut dyn std::io::Write) -> std::io::Result<()> {
    use svg_fmt::*;

    writeln!(
        output,
        "{}",
        BeginSvg {
            w: atlas.size.width as f32,
            h: atlas.size.height as f32
        }
    )?;

    dump_into_svg(atlas, None, output)?;

    writeln!(output, "{}", EndSvg)
}

/// Dump a visual representation of the atlas in SVG, omitting the beginning and end of the
/// SVG document, so that it can be included in a larger document.
///
/// If a rectangle is provided, translate and scale the output to fit it.
pub fn dump_into_svg(
    atlas: &AtlasAllocator,
    rect: Option<&Rectangle>,
    output: &mut dyn std::io::Write,
) -> std::io::Result<()> {
    use svg_fmt::*;

    let (sx, sy, tx, ty) = if let Some(rect) = rect {
        (
            rect.width() as f32 / atlas.size.width as f32,
            rect.height() as f32 / atlas.size.height as f32,
            rect.min.x as f32,
            rect.min.y as f32,
        )
    } else {
        (1.0, 1.0, 0.0, 0.0)
    };

    for node in &atlas.nodes {
        let color = match node.kind {
            NodeKind::Free => rgb(50, 50, 50),
            NodeKind::Alloc => rgb(70, 70, 180),
            _ => {
                continue;
            }
        };

        let (x, y) = node.rect.min.to_f32().to_tuple();
        let (w, h) = node.rect.size().to_f32().to_tuple();

        writeln!(
            output,
            r#"    {}"#,
            rectangle(tx + x * sx, ty + y * sy, w * sx, h * sy)
                .fill(color)
                .stroke(Stroke::Color(black(), 1.0))
        )?;
    }

    Ok(())
}

#[test]
fn atlas_basic() {
    let mut atlas = AtlasAllocator::new(size2(1000, 1000));

    let full = atlas.allocate(size2(1000, 1000)).unwrap().id;
    assert!(atlas.allocate(size2(1, 1)).is_none());

    atlas.deallocate(full);

    let a = atlas.allocate(size2(100, 1000)).unwrap().id;
    let b = atlas.allocate(size2(900, 200)).unwrap().id;
    let c = atlas.allocate(size2(300, 200)).unwrap().id;
    let d = atlas.allocate(size2(200, 300)).unwrap().id;
    let e = atlas.allocate(size2(100, 300)).unwrap().id;
    let f = atlas.allocate(size2(100, 300)).unwrap().id;
    let g = atlas.allocate(size2(100, 300)).unwrap().id;

    atlas.deallocate(b);
    atlas.deallocate(f);
    atlas.deallocate(c);
    atlas.deallocate(e);
    let h = atlas.allocate(size2(500, 200)).unwrap().id;
    atlas.deallocate(a);
    let i = atlas.allocate(size2(500, 200)).unwrap().id;
    atlas.deallocate(g);
    atlas.deallocate(h);
    atlas.deallocate(d);
    atlas.deallocate(i);

    let full = atlas.allocate(size2(1000, 1000)).unwrap().id;
    assert!(atlas.allocate(size2(1, 1)).is_none());
    atlas.deallocate(full);
}

#[test]
fn atlas_random_test() {
    let mut atlas = AtlasAllocator::with_options(
        size2(1000, 1000),
        &AllocatorOptions {
            alignment: size2(5, 2),
            ..DEFAULT_OPTIONS
        },
    );

    let a = 1103515245;
    let c = 12345;
    let m = usize::pow(2, 31);
    let mut seed: usize = 37;

    let mut rand = || {
        seed = (a * seed + c) % m;
        seed
    };

    let mut n: usize = 0;
    let mut misses: usize = 0;

    let mut allocated = Vec::new();
    for _ in 0..500000 {
        if rand() % 5 > 2 && !allocated.is_empty() {
            // deallocate something
            let nth = rand() % allocated.len();
            let id = allocated[nth];
            allocated.remove(nth);

            atlas.deallocate(id);
        } else {
            // allocate something
            let size = size2((rand() % 300) as i32 + 5, (rand() % 300) as i32 + 5);

            if let Some(alloc) = atlas.allocate(size) {
                allocated.push(alloc.id);
                n += 1;
            } else {
                misses += 1;
            }
        }
    }

    while let Some(id) = allocated.pop() {
        atlas.deallocate(id);
    }

    println!("added/removed {} rectangles, {} misses", n, misses);
    println!(
        "nodes.cap: {}, free_list.cap: {}/{}/{}",
        atlas.nodes.capacity(),
        atlas.free_lists[LARGE_BUCKET].capacity(),
        atlas.free_lists[MEDIUM_BUCKET].capacity(),
        atlas.free_lists[SMALL_BUCKET].capacity(),
    );

    let full = atlas.allocate(size2(1000, 1000)).unwrap().id;
    assert!(atlas.allocate(size2(1, 1)).is_none());
    atlas.deallocate(full);
}

#[test]
fn test_grow() {
    let mut atlas = AtlasAllocator::new(size2(1000, 1000));

    atlas.grow(size2(2000, 2000));

    let full = atlas.allocate(size2(2000, 2000)).unwrap().id;
    assert!(atlas.allocate(size2(1, 1)).is_none());
    atlas.deallocate(full);

    let a = atlas.allocate(size2(100, 100)).unwrap().id;

    atlas.grow(size2(3000, 3000));

    let b = atlas.allocate(size2(1000, 2900)).unwrap().id;

    atlas.grow(size2(4000, 4000));

    atlas.deallocate(b);
    atlas.deallocate(a);

    let full = atlas.allocate(size2(4000, 4000)).unwrap().id;
    assert!(atlas.allocate(size2(1, 1)).is_none());
    atlas.deallocate(full);
}

#[test]
fn clear_empty() {
    let mut atlas = AtlasAllocator::new(size2(1000, 1000));

    assert!(atlas.is_empty());

    assert!(atlas.allocate(size2(10, 10)).is_some());
    assert!(!atlas.is_empty());

    atlas.clear();
    assert!(atlas.is_empty());

    let a = atlas.allocate(size2(10, 10)).unwrap().id;
    let b = atlas.allocate(size2(20, 20)).unwrap().id;
    assert!(!atlas.is_empty());

    atlas.deallocate(b);
    atlas.deallocate(a);
    assert!(atlas.is_empty());

    atlas.clear();
    assert!(atlas.is_empty());

    atlas.clear();
    assert!(atlas.is_empty());
}

#[test]
fn simple_atlas() {
    let mut atlas = SimpleAtlasAllocator::new(size2(1000, 1000));

    assert!(atlas.allocate(size2(1, 1001)).is_none());
    assert!(atlas.allocate(size2(1001, 1)).is_none());

    let mut rectangles = Vec::new();
    rectangles.push(atlas.allocate(size2(100, 1000)).unwrap());
    rectangles.push(atlas.allocate(size2(900, 200)).unwrap());
    rectangles.push(atlas.allocate(size2(300, 200)).unwrap());
    rectangles.push(atlas.allocate(size2(200, 300)).unwrap());
    rectangles.push(atlas.allocate(size2(100, 300)).unwrap());
    rectangles.push(atlas.allocate(size2(100, 300)).unwrap());
    rectangles.push(atlas.allocate(size2(100, 300)).unwrap());
    assert!(atlas.allocate(size2(800, 800)).is_none());

    for i in 0..rectangles.len() {
        for j in 0..rectangles.len() {
            if i == j {
                continue;
            }

            assert!(!rectangles[i].intersects(&rectangles[j]));
        }
    }
}

#[test]
fn allocate_zero() {
    let mut atlas = SimpleAtlasAllocator::new(size2(1000, 1000));

    assert!(atlas.allocate(size2(0, 0)).is_none());
}

#[test]
fn allocate_negative() {
    let mut atlas = SimpleAtlasAllocator::new(size2(1000, 1000));

    assert!(atlas.allocate(size2(-1, 1)).is_none());
    assert!(atlas.allocate(size2(1, -1)).is_none());
    assert!(atlas.allocate(size2(-1, -1)).is_none());

    assert!(atlas.allocate(size2(-167114179, -718142)).is_none());
}

#[test]
fn issue_25() {
    let mut allocator = AtlasAllocator::new(Size::new(65536, 65536));
    allocator.allocate(Size::new(2, 2));
    allocator.allocate(Size::new(65500, 2));
    allocator.allocate(Size::new(2, 65500));
}
