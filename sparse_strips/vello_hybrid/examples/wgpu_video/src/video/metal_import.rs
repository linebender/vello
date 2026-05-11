// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Zero-copy Metal import: `CVPixelBuffer` -> `CVMetalTexture` -> `MTLTexture` ->
//! `wgpu::Texture`.
//!
//! Trimmed port of `lightspeed_av/backend/apple/metal_import.rs`:
//! - **No IOSurface-keyed plane cache**: every frame imports its planes fresh. The
//!   source pipeline caches per IOSurface to amortise the
//!   `CVMetalTextureCacheCreateTextureFromImage` + wgpu hal interop cost; we skip
//!   that here to keep the surface area small. Re-introducing the cache is a future
//!   optimisation if/when the example becomes performance-sensitive.
//! - **Lazy plane resolution** through a `OnceCell`: the renderer typically
//!   accesses both planes per frame, but doing the import lazily keeps the cost
//!   off the decode path and out of the way if the renderer ever decides to skip
//!   a frame.

use std::cell::OnceCell;
use std::ptr::{self, NonNull};
use std::sync::Arc;

use objc2_core_foundation::CFRetained;
use objc2_core_video::{CVMetalTexture, CVMetalTextureCache, CVPixelBuffer, kCVReturnSuccess};
use objc2_metal::MTLPixelFormat;

use super::error::AvError;

/// Per-frame retain bundle: keeps the `CVPixelBuffer` and (lazily) its NV12
/// `CVMetalTexture` plane wrappers alive for as long as any wgpu texture derived from
/// it is alive.
pub(crate) struct PlatformVideoFrameRetain {
    importer: Arc<MetalImporter>,
    pixel_buffer: CFRetained<CVPixelBuffer>,
    /// Per-frame memoisation of the imported plane textures. Renderers that
    /// access both planes will hit the populated cell after the first call.
    planes: OnceCell<CachedNv12Planes>,
}

// SAFETY: holds an `Arc<MetalImporter>` (which carries a CFType
// `CVMetalTextureCache` + a `wgpu::Device`, both atomic-refcount safe) and a
// `CFRetained<CVPixelBuffer>` (CFType, atomic retain/release). The pixel buffer's
// IOSurface backing is read-only after VT decode finishes, so transferring ownership
// across threads is safe. `OnceCell` lazy init isn't synchronised, but
// `PlatformVideoFrameRetain` is observed from one thread at a time (decode -> render).
unsafe impl Send for PlatformVideoFrameRetain {}

impl std::fmt::Debug for PlatformVideoFrameRetain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlatformVideoFrameRetain")
            .field("planes_resolved", &self.planes.get().is_some())
            .finish_non_exhaustive()
    }
}

impl PlatformVideoFrameRetain {
    pub(crate) fn new_nv12(
        importer: Arc<MetalImporter>,
        pixel_buffer: CFRetained<CVPixelBuffer>,
    ) -> Self {
        Self {
            importer,
            pixel_buffer,
            planes: OnceCell::new(),
        }
    }

    pub(crate) fn y_plane(&self) -> &wgpu::Texture {
        self.resolve_planes().y_plane()
    }

    pub(crate) fn uv_plane(&self) -> &wgpu::Texture {
        self.resolve_planes().uv_plane()
    }

    fn resolve_planes(&self) -> &CachedNv12Planes {
        self.planes.get_or_init(|| {
            self.importer
                .import_nv12_planes(&self.pixel_buffer)
                .unwrap_or_else(|e| panic!("NV12 plane import failed: {e}"))
        })
    }
}

/// Both NV12 plane wgpu textures plus the `CVMetalTexture` retains backing them.
pub(crate) struct CachedNv12Planes {
    y: PlaneSlot,
    uv: PlaneSlot,
}

impl CachedNv12Planes {
    fn y_plane(&self) -> &wgpu::Texture {
        &self.y.texture
    }
    fn uv_plane(&self) -> &wgpu::Texture {
        &self.uv.texture
    }
}

struct PlaneSlot {
    texture: wgpu::Texture,
    /// Kept alive so the underlying IOSurface isn't recycled while wgpu samples from it.
    _cv_texture: CFRetained<CVMetalTexture>,
}

/// Wraps a `CVMetalTextureCache` and the wgpu `Device` it imports against.
///
/// One importer is shared by both the decoder (NV12 plane import) and the RGBA
/// converter (BGRA import).
pub(crate) struct MetalImporter {
    cache: CFRetained<CVMetalTextureCache>,
    pub(crate) device: wgpu::Device,
}

// SAFETY: `CVMetalTextureCache` is a CFType (atomic retain/release; methods are
// documented thread-safe); `wgpu::Device` is already `Send + Sync`. We need both
// `Send` and `Sync` so `Arc<MetalImporter>` is itself `Send`.
unsafe impl Send for MetalImporter {}
unsafe impl Sync for MetalImporter {}

impl MetalImporter {
    pub(crate) fn new(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Result<Self, AvError> {
        let backend = adapter.get_info().backend;
        if backend != wgpu::Backend::Metal {
            return Err(AvError::UnexpectedBackend {
                expected: "Metal",
                got: backend,
            });
        }

        // SAFETY: `as_hal::<Metal>()` returns a guard that derefs to the hal device
        // while the wgpu `Device` is alive; we use the borrowed `MTLDevice` only for
        // the duration of `CVMetalTextureCache::create`, which retains it internally.
        let mtl_device = unsafe {
            let hal =
                device
                    .as_hal::<wgpu::hal::api::Metal>()
                    .ok_or(AvError::UnexpectedBackend {
                        expected: "Metal",
                        got: backend,
                    })?;
            hal.raw_device().clone()
        };

        let mut cache_ptr: *mut CVMetalTextureCache = ptr::null_mut();
        // SAFETY: typed binding. `cache_ptr` is a writable out-pointer; `mtl_device`
        // is a valid `MTLDevice` reference; nil attribute dictionaries select default
        // behaviour.
        let status = unsafe {
            CVMetalTextureCache::create(
                None,
                None,
                &mtl_device,
                None,
                NonNull::new_unchecked(&mut cache_ptr),
            )
        };
        if status != kCVReturnSuccess {
            return Err(AvError::backend("CVMetalTextureCacheCreate failed", status));
        }
        // SAFETY: the call above succeeded and populated `cache_ptr` with a +1
        // retained CFType.
        let cache = unsafe { CFRetained::from_raw(NonNull::new_unchecked(cache_ptr)) };

        Ok(Self {
            cache,
            device: device.clone(),
        })
    }

    /// Validate that `pixel_buffer` is a 2-plane (NV12) buffer.
    pub(crate) fn validate_nv12(&self, pixel_buffer: &CVPixelBuffer) -> Result<(), AvError> {
        let plane_count = objc2_core_video::CVPixelBufferGetPlaneCount(pixel_buffer);
        if plane_count != 2 {
            return Err(AvError::backend(
                format!("expected NV12 (2 planes), got {plane_count}"),
                0,
            ));
        }
        Ok(())
    }

    /// Import the Y + UV planes of an NV12 `CVPixelBuffer` as two `wgpu::Texture`s.
    pub(crate) fn import_nv12_planes(
        &self,
        pixel_buffer: &CVPixelBuffer,
    ) -> Result<CachedNv12Planes, AvError> {
        let y_width = objc2_core_video::CVPixelBufferGetWidthOfPlane(pixel_buffer, 0);
        let y_height = objc2_core_video::CVPixelBufferGetHeightOfPlane(pixel_buffer, 0);
        let (y_cv_tex, y_texture) =
            self.import_plane(pixel_buffer, 0, y_width, y_height, NV12_Y)?;

        let uv_width = objc2_core_video::CVPixelBufferGetWidthOfPlane(pixel_buffer, 1);
        let uv_height = objc2_core_video::CVPixelBufferGetHeightOfPlane(pixel_buffer, 1);
        let (uv_cv_tex, uv_texture) =
            self.import_plane(pixel_buffer, 1, uv_width, uv_height, NV12_UV)?;

        Ok(CachedNv12Planes {
            y: PlaneSlot {
                texture: y_texture,
                _cv_texture: y_cv_tex,
            },
            uv: PlaneSlot {
                texture: uv_texture,
                _cv_texture: uv_cv_tex,
            },
        })
    }

    fn import_plane(
        &self,
        pixel_buffer: &CVPixelBuffer,
        plane_index: usize,
        width: usize,
        height: usize,
        format: PlaneFormat,
    ) -> Result<(CFRetained<CVMetalTexture>, wgpu::Texture), AvError> {
        // CVPixelBuffer is a CVImageBuffer typedef in CoreVideo; deref coercion
        // surfaces the right type.
        let image_buffer: &objc2_core_video::CVImageBuffer = pixel_buffer;
        let mut cv_tex_ptr: *mut CVMetalTexture = ptr::null_mut();
        // SAFETY: typed binding. `cache`, `image_buffer` are valid; nil texture
        // attributes select default behaviour; `plane_index` is in range (caller
        // checked for NV12, BGRA always passes 0).
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
        // SAFETY: the call above succeeded and populated `cv_tex_ptr` with a +1
        // retained CFType.
        let cv_tex: CFRetained<CVMetalTexture> =
            unsafe { CFRetained::from_raw(NonNull::new_unchecked(cv_tex_ptr)) };

        // The typed accessor returns `Option<Retained<…>>` already retained on our
        // behalf (objc_retain_autoreleased), so no manual balancing needed.
        let mtl_texture = objc2_core_video::CVMetalTextureGetTexture(&cv_tex)
            .ok_or_else(|| AvError::backend("CVMetalTextureGetTexture returned nil", 0))?;

        // SAFETY: `mtl_texture` is a freshly-retained `id<MTLTexture>`; we hand its
        // retain to wgpu hal which takes ownership. The descriptor matches the
        // underlying texture's storage layout.
        let texture = unsafe {
            import_metal_texture(
                &self.device,
                mtl_texture,
                format.wgpu,
                width as u32,
                height as u32,
            )?
        };
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

/// Wrap a typed `id<MTLTexture>` as a `wgpu::Texture` via the wgpu hal Metal API.
///
/// # Safety
///
/// `mtl_texture` must be a valid retained `id<MTLTexture>` whose backing resource
/// matches `format` / `width` / `height`. The caller transfers ownership of one retain
/// into wgpu hal.
unsafe fn import_metal_texture(
    device: &wgpu::Device,
    mtl_texture: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>>,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> Result<wgpu::Texture, AvError> {
    use wgpu::hal::api::Metal;

    let descriptor = wgpu::TextureDescriptor {
        label: Some("vello_hybrid_wgpu_video imported"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    };

    let copy_extent = wgpu::hal::CopyExtent {
        width,
        height,
        depth: 1,
    };

    // SAFETY: `mtl_texture` is a valid retained MTLTexture; wgpu hal takes ownership
    // of that retain and releases it when the hal Texture drops.
    let hal_texture = unsafe {
        wgpu::hal::metal::Device::texture_from_raw(
            mtl_texture,
            format,
            objc2_metal::MTLTextureType::Type2D,
            1,
            1,
            copy_extent,
        )
    };

    // SAFETY: `hal_texture` is a freshly-constructed wgpu hal Metal texture matching
    // the descriptor we pass.
    let wgpu_tex = unsafe { device.create_texture_from_hal::<Metal>(hal_texture, &descriptor) };
    Ok(wgpu_tex)
}
