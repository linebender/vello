/// Trait for types that can write image data directly to the atlas texture.
///
/// This allows efficient uploading from different sources:
/// - `Pixmap`: Direct upload without intermediate texture
/// - `Texture`: Texture-to-texture copy
/// - Custom implementations for other image sources
pub trait AtlasWriter {
    /// Get the width of the image.
    fn width(&self) -> u16;
    /// Get the height of the image.
    fn height(&self) -> u16;

    /// Write image data to an atlas texture at the specified offset.
    fn write_to_atlas(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        atlas_texture: &Texture,
        offset: [u16; 2],
        width: u16,
        height: u16,
    );
}

/// Implementation for `wgpu::Texture` - uses texture-to-texture copy
impl AtlasWriter for Texture {
    fn width(&self) -> u16 {
        u16::try_from(self.width()).expect("texture width exceeds the u16 atlas domain")
    }

    fn height(&self) -> u16 {
        u16::try_from(self.height()).expect("texture height exceeds the u16 atlas domain")
    }

    fn write_to_atlas(
        &self,
        _device: &Device,
        _queue: &Queue,
        encoder: &mut CommandEncoder,
        atlas_texture: &Texture,
        offset: [u16; 2],
        width: u16,
        height: u16,
    ) {
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: self,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: u32::from(offset[0]),
                    y: u32::from(offset[1]),
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            Extent3d {
                width: u32::from(width),
                height: u32::from(height),
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Implementation for `Pixmap` - direct upload to atlas
impl AtlasWriter for Pixmap {
    fn width(&self) -> u16 {
        self.width()
    }

    fn height(&self) -> u16 {
        self.height()
    }

    fn write_to_atlas(
        &self,
        _device: &Device,
        queue: &Queue,
        _encoder: &mut CommandEncoder,
        atlas_texture: &Texture,
        offset: [u16; 2],
        width: u16,
        height: u16,
    ) {
        let width = u32::from(width);
        let height = u32::from(height);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: u32::from(offset[0]),
                    y: u32::from(offset[1]),
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            self.data_as_u8_slice(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Implementation for `Arc<Pixmap>`
impl AtlasWriter for Arc<Pixmap> {
    fn width(&self) -> u16 {
        self.as_ref().width()
    }

    fn height(&self) -> u16 {
        self.as_ref().height()
    }

    fn write_to_atlas(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        atlas_texture: &Texture,
        offset: [u16; 2],
        width: u16,
        height: u16,
    ) {
        self.as_ref()
            .write_to_atlas(device, queue, encoder, atlas_texture, offset, width, height);
    }
}

