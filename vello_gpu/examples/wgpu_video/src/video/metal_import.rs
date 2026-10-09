// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Imports decoded pixel-buffer planes into wgpu without copying, going from
//! CVPixelBuffer to CVMetalTexture to MTLTexture to `wgpu::Texture`.
//!
//! Nothing is cached across frames; caching per IOSurface would save the import cost
//! but isn't needed for this example.

use std::ptr::{self, NonNull};

use objc2_core_foundation::CFRetained;
use objc2_core_video::{
    CVImageBuffer, CVMetalTexture, CVMetalTextureCache, CVPixelBuffer, kCVReturnSuccess,
};
use objc2_metal::MTLPixelFormat;

use super::error::AvError;

/// The imported planes of one NV12 pixel buffer.
pub(crate) struct Nv12Planes {
    pub(crate) y: wgpu::Texture,
    pub(crate) uv: wgpu::Texture,
    /// Keep the IOSurface out of the decoder's buffer pool while these are alive.
    _cv_textures: [CFRetained<CVMetalTexture>; 2],
    _pixel_buffer: CFRetained<CVPixelBuffer>,
}

/// Imports pixel-buffer planes as wgpu textures through a texture cache tied to the
/// wgpu device.
pub(crate) struct MetalImporter {
    cache: CFRetained<CVMetalTextureCache>,
    device: wgpu::Device,
}

impl MetalImporter {
    /// Fails unless `device` uses the Metal backend.
    pub(crate) fn new(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Result<Self, AvError> {
        let backend = adapter.get_info().backend;
        let unexpected_backend = AvError::UnexpectedBackend {
            expected: "Metal",
            got: backend,
        };
        if backend != wgpu::Backend::Metal {
            return Err(unexpected_backend);
        }

        // SAFETY: the hal device is valid while `device` is alive, and the cloned
        // `MTLDevice` is only passed to `CVMetalTextureCache::create`, which retains it.
        let mtl_device = unsafe {
            let hal = device
                .as_hal::<wgpu::hal::api::Metal>()
                .ok_or(unexpected_backend)?;
            hal.raw_device().clone()
        };

        let mut cache_ptr: *mut CVMetalTextureCache = ptr::null_mut();
        // SAFETY: `cache_ptr` is writable and `mtl_device` is valid; nil attributes
        // mean defaults.
        let status = unsafe {
            CVMetalTextureCache::create(
                None,
                None,
                &mtl_device,
                None,
                NonNull::new_unchecked(&mut cache_ptr),
            )
        };
        if status != kCVReturnSuccess || cache_ptr.is_null() {
            return Err(AvError::backend("CVMetalTextureCacheCreate failed", status));
        }
        // SAFETY: on success, `create` stores a retained cache in `cache_ptr`.
        let cache = unsafe { CFRetained::from_raw(NonNull::new_unchecked(cache_ptr)) };

        Ok(Self {
            cache,
            device: device.clone(),
        })
    }

    /// Imports the Y and Cb/Cr planes of an NV12 pixel buffer.
    pub(crate) fn import_nv12(
        &self,
        pixel_buffer: CFRetained<CVPixelBuffer>,
    ) -> Result<Nv12Planes, AvError> {
        let plane_count = objc2_core_video::CVPixelBufferGetPlaneCount(&pixel_buffer);
        if plane_count != 2 {
            return Err(AvError::backend(
                format!("expected NV12 (2 planes), got {plane_count}"),
                0,
            ));
        }

        // Releases cache entries whose textures have all been dropped. Without this
        // the cache holds on to every IOSurface it has seen.
        self.cache.flush(0);

        let (y_cv, y) = self.import_plane(&pixel_buffer, 0, NV12_Y)?;
        let (uv_cv, uv) = self.import_plane(&pixel_buffer, 1, NV12_UV)?;
        Ok(Nv12Planes {
            y,
            uv,
            _cv_textures: [y_cv, uv_cv],
            _pixel_buffer: pixel_buffer,
        })
    }

    fn import_plane(
        &self,
        pixel_buffer: &CVPixelBuffer,
        plane_index: usize,
        format: PlaneFormat,
    ) -> Result<(CFRetained<CVMetalTexture>, wgpu::Texture), AvError> {
        let width = objc2_core_video::CVPixelBufferGetWidthOfPlane(pixel_buffer, plane_index);
        let height = objc2_core_video::CVPixelBufferGetHeightOfPlane(pixel_buffer, plane_index);
        let image_buffer: &CVImageBuffer = pixel_buffer;
        let mut cv_tex_ptr: *mut CVMetalTexture = ptr::null_mut();
        // SAFETY: `cache` and `image_buffer` are valid, nil attributes mean defaults, and
        // `import_nv12` has checked that `plane_index` exists.
        let status = unsafe {
            CVMetalTextureCache::create_texture_from_image(
                None,
                &self.cache,
                image_buffer,
                None,
                format.metal,
                width,
                height,
                plane_index,
                NonNull::new_unchecked(&mut cv_tex_ptr),
            )
        };
        if status != kCVReturnSuccess || cv_tex_ptr.is_null() {
            return Err(AvError::backend(
                format!("CVMetalTextureCacheCreateTextureFromImage failed (plane {plane_index})"),
                status,
            ));
        }
        // SAFETY: the call succeeded and stored a retained texture in `cv_tex_ptr`.
        let cv_tex: CFRetained<CVMetalTexture> =
            unsafe { CFRetained::from_raw(NonNull::new_unchecked(cv_tex_ptr)) };

        let mtl_texture = objc2_core_video::CVMetalTextureGetTexture(&cv_tex)
            .ok_or_else(|| AvError::backend("CVMetalTextureGetTexture returned nil", 0))?;

        let size = wgpu::Extent3d {
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: 1,
        };
        // SAFETY: `mtl_texture` is retained and was created with `format.metal` at `size`.
        let texture = unsafe { wrap_metal_texture(&self.device, mtl_texture, format.wgpu, size) };
        Ok((cv_tex, texture))
    }
}

#[derive(Clone, Copy)]
struct PlaneFormat {
    metal: MTLPixelFormat,
    wgpu: wgpu::TextureFormat,
}

const NV12_Y: PlaneFormat = PlaneFormat {
    metal: MTLPixelFormat::R8Unorm,
    wgpu: wgpu::TextureFormat::R8Unorm,
};
const NV12_UV: PlaneFormat = PlaneFormat {
    metal: MTLPixelFormat::RG8Unorm,
    wgpu: wgpu::TextureFormat::Rg8Unorm,
};

/// Wraps a Metal texture as a `wgpu::Texture` through wgpu's Metal hal API.
///
/// # Safety
///
/// `mtl_texture` must be a 2D, single-mip texture matching `format` and `size`. Its
/// retain is handed over to wgpu.
unsafe fn wrap_metal_texture(
    device: &wgpu::Device,
    mtl_texture: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>>,
    format: wgpu::TextureFormat,
    size: wgpu::Extent3d,
) -> wgpu::Texture {
    // SAFETY: the caller guarantees `mtl_texture` matches this description; wgpu
    // releases it when the hal texture drops.
    let hal_texture = unsafe {
        wgpu::hal::metal::Device::texture_from_raw(
            mtl_texture,
            format,
            objc2_metal::MTLTextureType::Type2D,
            1,
            1,
            wgpu::hal::CopyExtent {
                width: size.width,
                height: size.height,
                depth: 1,
            },
            None,
        )
    };

    let descriptor = wgpu::TextureDescriptor {
        label: Some("vello_gpu_wgpu_video imported plane"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    // SAFETY: `hal_texture` was just created to match `descriptor`.
    unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Metal>(
            hal_texture,
            &descriptor,
            wgpu::TextureUses::RESOURCE,
        )
    }
}
