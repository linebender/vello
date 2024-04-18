use vello_encoding::RenderConfig;

use crate::{render_graph::ResourceManager, FullShaders, Recording, RenderParams};

use super::RenderNode;

pub struct VelloCoarse {}

impl RenderNode for VelloCoarse {
    fn recording(
        &mut self,
        resources: &mut ResourceManager,
        config: &RenderConfig,
        params: &RenderParams,
        shaders: &FullShaders,
    ) -> Recording {
        todo!()
    }
}
