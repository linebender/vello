// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

#![warn(clippy::doc_markdown, clippy::semicolon_if_nothing_returned)]

mod cpu_dispatch;
mod cpu_shader;
#[cfg(all(feature = "debug_layers", feature = "wgpu"))]
mod debug;
mod engine;
mod render;
mod scene;
mod shaders;
#[cfg(feature = "wgpu")]
mod wgpu_engine;

/// Styling and composition primitives.
pub use peniko;
/// 2D geometry, with a focus on curves.
pub use peniko::kurbo;

#[doc(hidden)]
pub use skrifa;

pub mod glyph;

#[cfg(feature = "wgpu")]
pub mod util;

pub use render::Render;
pub use scene::{DrawGlyphs, Scene};
#[cfg(feature = "wgpu")]
pub use util::block_on_wgpu;

pub use engine::{
    BindType, BufProxy, Command, Id, ImageFormat, ImageProxy, Recording, ResourceProxy, ShaderId,
};
pub use shaders::FullShaders;
#[cfg(feature = "wgpu")]
use wgpu_engine::{ExternalResource, WgpuEngine};

#[cfg(all(feature = "debug_layers", feature = "wgpu"))]
pub use debug::DebugLayers;
/// Temporary export, used in `with_winit` for stats
pub use vello_encoding::BumpAllocators;
#[cfg(feature = "wgpu")]
use wgpu::{Device, Queue, SurfaceTexture, TextureFormat, TextureView};
#[cfg(feature = "wgpu-profiler")]
use wgpu_profiler::{GpuProfiler, GpuProfilerSettings};

/// Catch-all error type.
pub type Error = Box<dyn std::error::Error>;

/// Specialization of `Result` for our catch-all error type.
pub type Result<T> = std::result::Result<T, Error>;

/// Represents the antialiasing method to use during a render pass.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum AaConfig {
    Area,
    Msaa8,
    Msaa16,
}

/// Represents the set of antialiasing configurations to enable during pipeline creation.
pub struct AaSupport {
    pub area: bool,
    pub msaa8: bool,
    pub msaa16: bool,
}

impl AaSupport {
    pub fn all() -> Self {
        Self {
            area: true,
            msaa8: true,
            msaa16: true,
        }
    }

    pub fn area_only() -> Self {
        Self {
            area: true,
            msaa8: false,
            msaa16: false,
        }
    }
}

/// Renders a scene into a texture or surface.
#[cfg(feature = "wgpu")]
pub struct Renderer {
    #[cfg_attr(not(feature = "hot_reload"), allow(dead_code))]
    options: RendererOptions,
    engine: WgpuEngine,
    shaders: FullShaders,
    blit: Option<BlitPipeline>,
    #[cfg(feature = "debug_layers")]
    debug: Option<debug::DebugRenderer>,
    target: Option<TargetTexture>,
    #[cfg(feature = "wgpu-profiler")]
    profiler: GpuProfiler,
    #[cfg(feature = "wgpu-profiler")]
    pub profile_result: Option<Vec<wgpu_profiler::GpuTimerQueryResult>>,
}

/// Parameters used in a single render that are configurable by the client.
pub struct RenderParams {
    /// The background color applied to the target. This value is only applicable to the full
    /// pipeline.
    pub base_color: peniko::Color,

    /// Dimensions of the rasterization target
    pub width: u32,
    pub height: u32,

    /// The anti-aliasing algorithm. The selected algorithm must have been initialized while
    /// constructing the `Renderer`.
    pub antialiasing_method: AaConfig,

    #[cfg(feature = "debug_layers")]
    /// Options for debug layer rendering.
    pub debug: DebugLayers,
}

#[cfg(feature = "wgpu")]
pub struct RendererOptions {
    /// The format of the texture used for surfaces with this renderer/device
    /// If None, the renderer cannot be used with surfaces
    pub surface_format: Option<TextureFormat>,

    /// If true, run all stages up to fine rasterization on the CPU.
    // TODO: Consider evolving this so that the CPU stages can be configured dynamically via
    // `RenderParams`.
    pub use_cpu: bool,

    /// Represents the enabled set of AA configurations. This will be used to determine which
    /// pipeline permutations should be compiled at startup.
    pub antialiasing_support: AaSupport,
}

struct RenderResult {
    bump: Option<BumpAllocators>,
    #[cfg(feature = "debug_layers")]
    captured: Option<render::CapturedBuffers>,
}

#[cfg(feature = "wgpu")]
impl Renderer {
    /// Creates a new renderer for the specified device.
    pub fn new(device: &Device, options: RendererOptions) -> Result<Self> {
        let mut engine = WgpuEngine::new(options.use_cpu);
        let shaders = shaders::full_shaders(device, &mut engine, &options)?;
        let blit = options
            .surface_format
            .map(|surface_format| BlitPipeline::new(device, surface_format, &mut engine));
        #[cfg(feature = "debug_layers")]
        let debug = options
            .surface_format
            .map(|surface_format| debug::DebugRenderer::new(device, surface_format, &mut engine));

        Ok(Self {
            options,
            engine,
            shaders,
            blit,
            #[cfg(feature = "debug_layers")]
            debug,
            target: None,
            // Use 3 pending frames
            #[cfg(feature = "wgpu-profiler")]
            profiler: GpuProfiler::new(GpuProfilerSettings {
                ..Default::default()
            })?,
            #[cfg(feature = "wgpu-profiler")]
            profile_result: None,
        })
    }

    /// Renders a scene to the target texture.
    ///
    /// The texture is assumed to be of the specified dimensions and have been created with
    /// the [`wgpu::TextureFormat::Rgba8Unorm`] format and the [`wgpu::TextureUsages::STORAGE_BINDING`]
    /// flag set.
    pub fn render_to_texture(
        &mut self,
        device: &Device,
        queue: &Queue,
        scene: &Scene,
        texture: &TextureView,
        params: &RenderParams,
    ) -> Result<()> {
        let (recording, target) = render::render_full(scene, &self.shaders, params);
        let external_resources = [ExternalResource::Image(
            *target.as_image().unwrap(),
            texture,
        )];
        self.engine.run_recording(
            device,
            queue,
            &recording,
            &external_resources,
            "render_to_texture",
            #[cfg(feature = "wgpu-profiler")]
            &mut self.profiler,
        )?;
        Ok(())
    }

    /// Renders a scene to the target surface.
    ///
    /// This renders to an intermediate texture and then runs a render pass to blit to the
    /// specified surface texture.
    ///
    /// The surface is assumed to be of the specified dimensions and have been configured with
    /// the same format passed in the constructing [`RendererOptions`]' `surface_format`.
    /// Panics if `surface_format` was `None`
    pub fn render_to_surface(
        &mut self,
        device: &Device,
        queue: &Queue,
        scene: &Scene,
        surface: &SurfaceTexture,
        params: &RenderParams,
    ) -> Result<()> {
        let width = params.width;
        let height = params.height;
        let mut target = self
            .target
            .take()
            .unwrap_or_else(|| TargetTexture::new(device, width, height));
        // TODO: implement clever resizing semantics here to avoid thrashing the memory allocator
        // during resize, specifically on metal.
        if target.width != width || target.height != height {
            target = TargetTexture::new(device, width, height);
        }
        self.render_to_texture(device, queue, scene, &target.view, params)?;
        let blit = self
            .blit
            .as_ref()
            .expect("renderer should have configured surface_format to use on a surface");
        let mut recording = Recording::default();
        let target_proxy = ImageProxy::new(width, height, ImageFormat::from_wgpu(target.format));
        let surface_proxy = ImageProxy::new(
            width,
            height,
            ImageFormat::from_wgpu(surface.texture.format()),
        );
        recording.draw(
            blit.0,
            1,
            6,
            None,
            [ResourceProxy::Image(target_proxy)],
            surface_proxy,
            Some([0., 0., 0., 0.]),
        );

        let surface_view = surface
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let external_resources = [
            ExternalResource::Image(target_proxy, &target.view),
            ExternalResource::Image(surface_proxy, &surface_view),
        ];
        self.engine.run_recording(
            device,
            queue,
            &recording,
            &external_resources,
            "blit (render_to_surface_async)",
            #[cfg(feature = "wgpu-profiler")]
            &mut self.profiler,
        )?;

        #[cfg(feature = "wgpu-profiler")]
        {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            self.profiler.resolve_queries(&mut encoder);
            queue.submit(Some(encoder.finish()));
            self.profiler.end_frame().unwrap();
            if let Some(result) = self
                .profiler
                .process_finished_frame(queue.get_timestamp_period())
            {
                self.profile_result = Some(result);
            }
        }

        self.target = Some(target);
        Ok(())
    }

    /// Reload the shaders. This should only be used during `vello` development
    #[cfg(feature = "hot_reload")]
    pub async fn reload_shaders(&mut self, device: &Device) -> Result<()> {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut engine = WgpuEngine::new(self.options.use_cpu);
        let shaders = shaders::full_shaders(device, &mut engine, &self.options)?;
        let blit = self
            .options
            .surface_format
            .map(|surface_format| BlitPipeline::new(device, surface_format, &mut engine));
        #[cfg(feature = "debug_layers")]
        let debug = self
            .options
            .surface_format
            .map(|format| debug::DebugRenderer::new(device, format, &mut engine));
        let error = device.pop_error_scope().await;
        if let Some(error) = error {
            return Err(error.into());
        }
        self.engine = engine;
        self.shaders = shaders;
        self.blit = blit;
        #[cfg(feature = "debug_layers")]
        {
            self.debug = debug;
        }
        Ok(())
    }

    /// Renders a scene to the target texture.
    ///
    /// The texture is assumed to be of the specified dimensions and have been created with
    /// the [`wgpu::TextureFormat::Rgba8Unorm`] format and the [`wgpu::TextureUsages::STORAGE_BINDING`]
    /// flag set.
    ///
    /// The return value is the value of the `BumpAllocators` in this rendering, which is currently used
    /// for debug output.
    ///
    /// This return type is not stable, and will likely be changed when a more principled way to access
    /// relevant statistics is implemented
    pub async fn render_to_texture_async(
        &mut self,
        device: &Device,
        queue: &Queue,
        scene: &Scene,
        texture: &TextureView,
        params: &RenderParams,
    ) -> Result<Option<BumpAllocators>> {
        let result = self
            .render_to_texture_async_internal(device, queue, scene, texture, params)
            .await?;
        #[cfg(feature = "debug_layers")]
        {
            // TODO: it would be better to improve buffer ownership tracking so that it's not
            // necessary to submit a whole new Recording to free the captured buffers.
            if let Some(captured) = result.captured {
                let mut recording = Recording::default();
                // TODO: this sucks. better to release everything in a helper
                self.engine.free_download(captured.lines);
                captured.release_buffers(&mut recording);
                self.engine.run_recording(
                    device,
                    queue,
                    &recording,
                    &[],
                    "free memory",
                    #[cfg(feature = "wgpu-profiler")]
                    &mut self.profiler,
                )?;
            }
        }
        Ok(result.bump)
    }

    async fn render_to_texture_async_internal(
        &mut self,
        device: &Device,
        queue: &Queue,
        scene: &Scene,
        texture: &TextureView,
        params: &RenderParams,
    ) -> Result<RenderResult> {
        let mut render = Render::new();
        let encoding = scene.encoding();
        // TODO: turn this on; the download feature interacts with CPU dispatch.
        // Currently this is always enabled when the `debug_layers` setting is enabled as the bump
        // counts are used for debug visualiation.
        let robust = cfg!(feature = "debug_layers");
        let recording = render.render_encoding_coarse(encoding, &self.shaders, params, robust);
        let target = render.out_image();
        let bump_buf = render.bump_buf();
        #[cfg(feature = "debug_layers")]
        let captured = render.take_captured_buffers();
        self.engine.run_recording(
            device,
            queue,
            &recording,
            &[],
            "t_async_coarse",
            #[cfg(feature = "wgpu-profiler")]
            &mut self.profiler,
        )?;

        let mut bump: Option<BumpAllocators> = None;
        if let Some(bump_buf) = self.engine.get_download(bump_buf) {
            let buf_slice = bump_buf.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            buf_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
            if let Some(recv_result) = receiver.receive().await {
                recv_result?;
            } else {
                return Err("channel was closed".into());
            }
            let mapped = buf_slice.get_mapped_range();
            bump = Some(bytemuck::pod_read_unaligned(&mapped));
        }
        // TODO: apply logic to determine whether we need to rerun coarse, and also
        // allocate the blend stack as needed.
        self.engine.free_download(bump_buf);
        // Maybe clear to reuse allocation?
        let mut recording = Recording::default();
        render.record_fine(&self.shaders, &mut recording);
        let external_resources = [ExternalResource::Image(target, texture)];
        self.engine.run_recording(
            device,
            queue,
            &recording,
            &external_resources,
            "t_async_fine",
            #[cfg(feature = "wgpu-profiler")]
            &mut self.profiler,
        )?;
        Ok(RenderResult {
            bump,
            #[cfg(feature = "debug_layers")]
            captured,
        })
    }

    /// See [`Self::render_to_surface`]
    pub async fn render_to_surface_async(
        &mut self,
        device: &Device,
        queue: &Queue,
        scene: &Scene,
        surface: &SurfaceTexture,
        params: &RenderParams,
    ) -> Result<Option<BumpAllocators>> {
        let width = params.width;
        let height = params.height;
        let mut target = self
            .target
            .take()
            .unwrap_or_else(|| TargetTexture::new(device, width, height));
        // TODO: implement clever resizing semantics here to avoid thrashing the memory allocator
        // during resize, specifically on metal.
        if target.width != width || target.height != height {
            target = TargetTexture::new(device, width, height);
        }
        let result = self
            .render_to_texture_async_internal(device, queue, scene, &target.view, params)
            .await?;
        let blit = self
            .blit
            .as_ref()
            .expect("renderer should have configured surface_format to use on a surface");
        let mut recording = Recording::default();
        let target_proxy = ImageProxy::new(width, height, ImageFormat::from_wgpu(target.format));
        let surface_proxy = ImageProxy::new(
            width,
            height,
            ImageFormat::from_wgpu(surface.texture.format()),
        );
        recording.draw(
            blit.0,
            1,
            6,
            None,
            [ResourceProxy::Image(target_proxy)],
            surface_proxy,
            Some([0., 0., 0., 0.]),
        );

        #[cfg(feature = "debug_layers")]
        {
            if let Some(captured) = result.captured {
                let debug = self
                    .debug
                    .as_ref()
                    .expect("renderer should have configured surface_format to use on a surface");
                let bump = result.bump.as_ref().unwrap();
                // TODO: We could avoid this download if `DebugLayers::VALIDATION` is unset.
                let downloads = DebugDownloads::map(&mut self.engine, &captured, bump).await?;
                debug.render(
                    &mut recording,
                    surface_proxy,
                    &captured,
                    bump,
                    &params,
                    &downloads,
                );

                // TODO: this sucks. better to release everything in a helper
                // TODO: it would be much better to have a way to safely destroy a buffer.
                self.engine.free_download(captured.lines);
                captured.release_buffers(&mut recording);
            }
        }

        let surface_view = surface
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let external_resources = [
            ExternalResource::Image(target_proxy, &target.view),
            ExternalResource::Image(surface_proxy, &surface_view),
        ];
        self.engine.run_recording(
            device,
            queue,
            &recording,
            &external_resources,
            "blit (render_to_surface_async)",
            #[cfg(feature = "wgpu-profiler")]
            &mut self.profiler,
        )?;

        #[cfg(feature = "wgpu-profiler")]
        {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            self.profiler.resolve_queries(&mut encoder);
            queue.submit(Some(encoder.finish()));
            self.profiler.end_frame().unwrap();
            if let Some(result) = self
                .profiler
                .process_finished_frame(queue.get_timestamp_period())
            {
                self.profile_result = Some(result);
            }
        }

        self.target = Some(target);
        Ok(result.bump)
    }
}

#[cfg(feature = "wgpu")]
struct TargetTexture {
    view: TextureView,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
}

#[cfg(feature = "wgpu")]
impl TargetTexture {
    pub fn new(device: &Device, width: u32, height: u32) -> Self {
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            format,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            view,
            width,
            height,
            format,
        }
    }
}

#[cfg(feature = "wgpu")]
struct BlitPipeline(ShaderId);

#[cfg(feature = "wgpu")]
impl BlitPipeline {
    fn new(device: &Device, format: TextureFormat, engine: &mut WgpuEngine) -> Self {
        const SHADERS: &str = r#"
            @vertex
            fn vs_main(@builtin(vertex_index) ix: u32) -> @builtin(position) vec4<f32> {
                // Generate a full screen quad in NDCs
                var vertex = vec2(-1.0, 1.0);
                switch ix {
                    case 1u: {
                        vertex = vec2(-1.0, -1.0);
                    }
                    case 2u, 4u: {
                        vertex = vec2(1.0, -1.0);
                    }
                    case 5u: {
                        vertex = vec2(1.0, 1.0);
                    }
                    default: {}
                }
                return vec4(vertex, 0.0, 1.0);
            }

            @group(0) @binding(0)
            var fine_output: texture_2d<f32>;

            @fragment
            fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
                let rgba_sep = textureLoad(fine_output, vec2<i32>(pos.xy), 0);
                return vec4(rgba_sep.rgb * rgba_sep.a, rgba_sep.a);
            }
        "#;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit shaders"),
            source: wgpu::ShaderSource::Wgsl(SHADERS.into()),
        });
        let shader_id = engine.add_render_shader(
            device,
            "blit",
            &module,
            "vs_main",
            "fs_main",
            wgpu::PrimitiveTopology::TriangleList,
            wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            },
            None,
            &[(
                BindType::ImageRead(ImageFormat::from_wgpu(format)),
                wgpu::ShaderStages::FRAGMENT,
            )],
        );
        Self(shader_id)
    }
}

#[cfg(all(feature = "debug_layers", feature = "wgpu"))]
pub(crate) struct DebugDownloads<'a> {
    pub lines: wgpu::BufferSlice<'a>,
}

#[cfg(all(feature = "debug_layers", feature = "wgpu"))]
impl<'a> DebugDownloads<'a> {
    pub async fn map(
        engine: &'a WgpuEngine,
        captured: &render::CapturedBuffers,
        bump: &BumpAllocators,
    ) -> Result<DebugDownloads<'a>> {
        use vello_encoding::LineSoup;

        let Some(lines_buf) = engine.get_download(captured.lines) else {
            return Err("could not download LineSoup buffer".into());
        };

        let lines = lines_buf.slice(..bump.lines as u64 * std::mem::size_of::<LineSoup>() as u64);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        lines.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        if let Some(recv_result) = receiver.receive().await {
            recv_result?;
        } else {
            return Err("channel was closed".into());
        }
        Ok(Self { lines })
    }
}
