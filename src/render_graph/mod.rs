use slotmap::{new_key_type, SecondaryMap, SlotMap};

use crate::{BufferProxy, ImageProxy, Recording, ResourceProxy};

use self::passes::RenderPass;

pub mod passes;

new_key_type! {
    pub struct PassId;
}

pub struct RenderGraph {
    nodes: SlotMap<PassId, Box<dyn RenderPass>>,
    dependencies: SecondaryMap<PassId, Vec<PassId>>,
    pub resources: ResourceManager,
}

impl RenderGraph {
    pub fn new() -> Self {
        RenderGraph {
            nodes: SlotMap::with_key(),
            dependencies: SecondaryMap::new(),
            resources: ResourceManager::new(),
        }
    }

    pub fn insert_pass(
        &mut self,
        pass: impl RenderPass + 'static,
        dependencies: &[PassId],
    ) -> PassId {
        let id = self.nodes.insert(Box::new(pass));
        self.dependencies.insert(id, dependencies.to_vec());
        id
    }

    pub fn process(&self) -> Recording {
        let mut recording = Recording::default();
        recording
    }
}

new_key_type! {
    pub struct ResourceId;
}

#[derive(Clone, Copy)]
pub struct Handle<T> {
    id: ResourceId,
    proxy: T,
}

impl Into<ResourceProxy> for Handle<ImageProxy> {
    fn into(self) -> ResourceProxy {
        ResourceProxy::Image(self.proxy)
    }
}
impl Into<ResourceProxy> for Handle<BufferProxy> {
    fn into(self) -> ResourceProxy {
        ResourceProxy::Buffer(self.proxy)
    }
}

pub struct ResourceManager {
    resources: SlotMap<ResourceId, ()>,
}

impl ResourceManager {
    pub fn new() -> Self {
        Self {
            resources: SlotMap::with_key(),
        }
    }

    pub fn import_image(&mut self, image: ImageProxy) -> Handle<ImageProxy> {
        let id = self.resources.insert(());
        Handle { id, proxy: image }
    }

    pub fn import_buffer(&mut self, buffer: BufferProxy) -> Handle<BufferProxy> {
        let id = self.resources.insert(());
        Handle { id, proxy: buffer }
    }
}
