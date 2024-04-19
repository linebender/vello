use vello_encoding::RenderConfig;

use crate::{render_graph::ResourceManager, FullShaders, Recording, RenderParams};

use super::RenderPass;

pub struct VelloCoarse {}

impl RenderPass for VelloCoarse {
    fn record(
        self,
        resources: &mut ResourceManager,
        config: &RenderConfig,
        params: &RenderParams,
        shaders: &FullShaders,
    ) -> Recording {
        todo!()
    }
}
