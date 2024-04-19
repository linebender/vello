mod coarse;
pub use coarse::*;

mod fine;
pub use fine::*;
use vello_encoding::RenderConfig;

use crate::{FullShaders, Recording, RenderParams};

use super::ResourceManager;

pub trait RenderPass: Send + Sync {
    fn record(
        &mut self,
        resources: &mut ResourceManager,
        config: &RenderConfig,
        params: &RenderParams,
        shaders: &FullShaders,
    ) -> Recording;
}
