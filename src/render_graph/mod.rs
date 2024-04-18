use slotmap::{new_key_type, SecondaryMap, SlotMap};

use crate::Recording;

use self::nodes::RenderNode;

pub mod nodes;

new_key_type! {
    pub struct RenderNodeId;
}

pub struct RenderGraph {
    nodes: SlotMap<RenderNodeId, Box<dyn RenderNode>>,
    dependencies: SecondaryMap<RenderNodeId, Vec<RenderNodeId>>,
}

impl RenderGraph {
    pub fn new() -> Self {
        RenderGraph {
            nodes: SlotMap::with_key(),
            dependencies: SecondaryMap::new(),
        }
    }

    pub fn insert_node(
        &mut self,
        node: impl RenderNode + 'static,
        dependencies: &[RenderNodeId],
    ) -> RenderNodeId {
        let id = self.nodes.insert(Box::new(node));
        self.dependencies.insert(id, dependencies.to_vec());
        id
    }

    pub fn process(&self) -> Recording {
        let mut recording = Recording::default();
        recording
    }
}
