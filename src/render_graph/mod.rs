use slotmap::{new_key_type, SecondaryMap, SlotMap};

use crate::{Recording, ResourceProxy};

use self::nodes::RenderNode;

pub mod nodes;

new_key_type! {
    pub struct RenderNodeId;
}

pub struct RenderGraph {
    nodes: SlotMap<RenderNodeId, Box<dyn RenderNode>>,
    dependencies: SecondaryMap<RenderNodeId, Vec<RenderNodeId>>,
    resource_manager: ResourceManager,
}

impl RenderGraph {
    pub fn new() -> Self {
        RenderGraph {
            nodes: SlotMap::with_key(),
            dependencies: SecondaryMap::new(),
            resource_manager: ResourceManager::new(),
        }
    }

    pub fn insert_node<F, N>(&mut self, node: F, dependencies: &[RenderNodeId]) -> RenderNodeId
    where
        F: FnOnce(&mut ResourceHinter) -> N,
        N: RenderNode + 'static,
    {
        let mut resource_hinter = ResourceHinter();
        let node = node(&mut resource_hinter);
        let id = self.nodes.insert(Box::new(node));
        self.dependencies.insert(id, dependencies.to_vec());
        id
    }

    pub fn manage_resource(&mut self, resource: Option<ResourceProxy>) -> ManagedResource {
        self.resource_manager.resources.insert(resource)
    }

    pub fn process(&self) -> Recording {
        let mut recording = Recording::default();
        recording
    }
}

new_key_type! {
    pub struct ManagedResource;
}

pub struct ResourceManager {
    resources: SlotMap<ManagedResource, Option<ResourceProxy>>,
}

impl ResourceManager {
    pub fn new() -> Self {
        Self {
            resources: SlotMap::with_key(),
        }
    }

    pub fn get(&self, res: ResourceRef) -> &Option<ResourceProxy> {
        self.resources.get(res.0).unwrap()
    }

    pub fn get_mut(&mut self, res: ResourceRef) -> &mut Option<ResourceProxy> {
        self.resources.get_mut(res.0).unwrap()
    }
}

pub struct ResourceHinter();

impl ManagedResource {
    pub fn hint(&self, rh: &mut ResourceHinter) -> ResourceRef {
        ResourceRef(*self)
    }
}

#[derive(Clone, Copy)]
pub struct ResourceRef(ManagedResource);
