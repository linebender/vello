use vello_encoding::{make_mask_lut, make_mask_lut_16, RenderConfig};

use crate::{
    render_graph::{ResourceManager, ResourceRef},
    AaConfig, FullShaders, ImageFormat, ImageProxy, Recording, RenderParams, ResourceProxy,
};

use super::RenderPass;

pub struct VelloFine {
    pub config_buf: ResourceRef,
    pub bump_buf: ResourceRef,
    pub tile_buf: ResourceRef,
    pub segments_buf: ResourceRef,
    pub ptcl_buf: ResourceRef,
    pub gradient_image: ResourceRef,
    pub info_bin_data_buf: ResourceRef,
    pub image_atlas: ResourceRef,

    // TODO: change this back to Image (and maybe the other to buffers :3)
    pub out_image: ResourceRef,

    pub mask_buf: Option<ResourceProxy>,
}

impl RenderPass for VelloFine {
    fn record(
        &mut self,
        resources: &mut ResourceManager,
        config: &RenderConfig,
        params: &RenderParams,
        shaders: &FullShaders,
    ) -> Recording {
        let mut recording = Recording::default();

        assert!(
            resources
                .get_mut(self.out_image)
                .replace(ResourceProxy::Image(ImageProxy::new(
                    params.width,
                    params.height,
                    ImageFormat::Rgba8
                )))
                .is_none(),
            "VelloFine's out_image was already created"
        );

        match params.antialiasing_method {
            AaConfig::Area => {
                recording.dispatch(
                    shaders
                        .fine_area
                        .expect("shaders not configured to support AA mode: area"),
                    config.workgroup_counts.fine,
                    [
                        resources
                            .get(self.config_buf)
                            .expect("VelloCoarse should have already initialized this"),
                        resources
                            .get(self.segments_buf)
                            .expect("VelloCoarse should have already initialized this"),
                        resources
                            .get(self.ptcl_buf)
                            .expect("VelloCoarse should have already initialized this"),
                        resources
                            .get(self.info_bin_data_buf)
                            .expect("VelloCoarse should have already initialized this"),
                        resources.get(self.out_image).unwrap(),
                        resources
                            .get(self.gradient_image)
                            .expect("VelloCoarse should have already initialized this"),
                        resources
                            .get(self.image_atlas)
                            .expect("VelloCoarse should have already initialized this"),
                    ],
                );
            }
            _ => {
                if self.mask_buf.is_none() {
                    let mask_lut = match params.antialiasing_method {
                        AaConfig::Msaa16 => make_mask_lut_16(),
                        AaConfig::Msaa8 => make_mask_lut(),
                        _ => unreachable!(),
                    };
                    let buf = recording.upload("mask lut", mask_lut);
                    self.mask_buf = Some(buf.into());
                }
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
                    [
                        resources
                            .get(self.config_buf)
                            .expect("VelloCoarse should have already initialized this"),
                        resources
                            .get(self.segments_buf)
                            .expect("VelloCoarse should have already initialized this"),
                        resources
                            .get(self.ptcl_buf)
                            .expect("VelloCoarse should have already initialized this"),
                        resources
                            .get(self.info_bin_data_buf)
                            .expect("VelloCoarse should have already initialized this"),
                        resources.get(self.out_image).unwrap(),
                        resources
                            .get(self.gradient_image)
                            .expect("VelloCoarse should have already initialized this"),
                        resources
                            .get(self.image_atlas)
                            .expect("VelloCoarse should have already initialized this"),
                        self.mask_buf.unwrap(),
                    ],
                );
            }
        }
        // TODO: make mask buf persistent
        // could we move mask_buf out of this and make a util that handles
        // this resource with the graph and easily creates on AaConfig change?
        if let Some(mask_buf) = self.mask_buf.take() {
            recording.free_resource(mask_buf);
        }

        recording
    }
}
