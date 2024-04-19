use crate::{
    render_graph::{Handle, PassContext},
    BufferProxy, ImageProxy, Recording,
};

use super::RenderPass;

pub struct VelloCoarse {}

#[derive(Clone, Copy)]
pub struct CoarseOutput {
    pub config_buf: Handle<BufferProxy>,
    pub bump_buf: Handle<BufferProxy>,
    pub tile_buf: Handle<BufferProxy>,
    pub segments_buf: Handle<BufferProxy>,
    pub ptcl_buf: Handle<BufferProxy>,
    pub gradient_image: Handle<BufferProxy>,
    pub info_bin_data_buf: Handle<BufferProxy>,
    pub image_atlas: Handle<BufferProxy>,

    pub out_image: Handle<ImageProxy>,
}

impl RenderPass for VelloCoarse {
    type Output = CoarseOutput;

    fn record(self, cx: PassContext<'_>) -> (Recording, Self::Output)
    where
        Self: Sized,
    {
        todo!()
    }
}
