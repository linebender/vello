use vello_encoding::{make_mask_lut, make_mask_lut_16, RenderConfig};

use crate::{
    render_graph::{Handle, ResourceManager},
    AaConfig, BufferProxy, FullShaders, ImageProxy, Recording, RenderParams,
};

use super::RenderPass;

pub struct VelloFine {
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

impl RenderPass for VelloFine {
    fn record(
        self,
        resources: &mut ResourceManager,
        config: &RenderConfig,
        params: &RenderParams,
        shaders: &FullShaders,
    ) -> Recording {
        let mut recording = Recording::default();

        match params.antialiasing_method {
            AaConfig::Area => {
                recording.dispatch(
                    shaders
                        .fine_area
                        .expect("shaders not configured to support AA mode: area"),
                    config.workgroup_counts.fine,
                    (
                        self.config_buf,
                        self.segments_buf,
                        self.ptcl_buf,
                        self.info_bin_data_buf,
                        self.out_image,
                        self.gradient_image,
                        self.image_atlas,
                    ),
                );
            }
            _ => {
                let mask_lut = match params.antialiasing_method {
                    AaConfig::Msaa16 => make_mask_lut_16(),
                    AaConfig::Msaa8 => make_mask_lut(),
                    _ => unreachable!(),
                };
                let mask_buf = recording.upload("mask lut", mask_lut);

                let fine_shader = match params.antialiasing_method {
                    AaConfig::Msaa16 => shaders
                        .fine_msaa16
                        .expect("shaders not configured to support AA mode: msaa16"),
                    AaConfig::Msaa8 => shaders
                        .fine_msaa8
                        .expect("shaders not configured to support AA mode: msaa8"),
                    _ => unreachable!(),
                };
                recording.dispatch(
                    fine_shader,
                    config.workgroup_counts.fine,
                    (
                        self.config_buf,
                        self.segments_buf,
                        self.ptcl_buf,
                        self.info_bin_data_buf,
                        self.out_image,
                        self.gradient_image,
                        self.image_atlas,
                        mask_buf,
                    ),
                );
                // TODO: make mask buf persistent
                // could we move mask_buf out of this and make a util that handles
                // this resource with the graph and easily creates on AaConfig change?
                recording.free_resource(mask_buf.into());
            }
        }

        recording
    }
}
