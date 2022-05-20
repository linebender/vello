mod blend;
mod encoder;
pub mod glyph_render;
mod gradient;
mod pico_svg;
mod render_ctx;
pub mod stages;
pub mod test_scenes;
mod text;

use bytemuck::Pod;
use std::convert::TryInto;

pub use blend::{Blend, BlendMode, CompositionMode};
pub use encoder::EncodedSceneRef;
pub use gradient::Colrv1RadialGradient;
pub use render_ctx::PietGpuRenderContext;

use piet::kurbo::Vec2;
use piet::{ImageFormat, RenderContext};

use piet_gpu_hal::{
    include_shader, BindType, Buffer, BufferUsage, CmdBuf, ComputePassDescriptor, DescriptorSet,
    Error, Image, ImageLayout, Pipeline, QueryPool, Session,
};

pub use pico_svg::PicoSvg;
use stages::{ClipBinding, ElementBinding, ElementCode};

use crate::stages::{ClipCode, Config, ElementStage};

const TILE_W: usize = 16;
const TILE_H: usize = 16;

const PTCL_INITIAL_ALLOC: usize = 1024;

#[allow(unused)]
fn dump_scene(buf: &[u8]) {
    for i in 0..(buf.len() / 4) {
        let mut buf_u32 = [0u8; 4];
        buf_u32.copy_from_slice(&buf[i * 4..i * 4 + 4]);
        println!("{:4x}: {:8x}", i * 4, u32::from_le_bytes(buf_u32));
    }
}

#[allow(unused)]
pub fn dump_k1_data(k1_buf: &[u32]) {
    for i in 0..k1_buf.len() {
        if k1_buf[i] != 0 {
            println!("{:4x}: {:8x}", i * 4, k1_buf[i]);
        }
    }
}

pub struct RenderConfig {
    width: usize,
    height: usize,
    format: PixelFormat,
}

// Should we just use the enum from piet-gpu-hal?
pub enum PixelFormat {
    A8,
    Rgba8,
}

pub struct Renderer {
    // These sizes are aligned to tile boundaries, though at some point
    // we'll want to have a good strategy for dealing with odd sizes.
    width: usize,
    height: usize,

    pub image_dev: Image, // resulting image

    // The reference is held by the pipelines. We will be changing
    // this to make the scene upload dynamic.
    scene_bufs: Vec<Buffer>,

    memory_buf_host: Vec<Buffer>,
    memory_buf_dev: Buffer,

    // Staging buffers
    config_bufs: Vec<Buffer>,
    // Device config buf
    config_buf: Buffer,

    // New element pipeline
    element_code: ElementCode,
    element_stage: ElementStage,
    element_bindings: Vec<ElementBinding>,

    clip_code: ClipCode,
    clip_binding: ClipBinding,

    tile_pipeline: Pipeline,
    tile_ds: Vec<DescriptorSet>,

    path_pipeline: Pipeline,
    path_ds: DescriptorSet,

    backdrop_pipeline: Pipeline,
    backdrop_ds: DescriptorSet,
    backdrop_y: u32,

    bin_pipeline: Pipeline,
    bin_ds: DescriptorSet,

    coarse_pipeline: Pipeline,
    coarse_ds: Vec<DescriptorSet>,

    k4_pipeline: Pipeline,
    k4_ds: DescriptorSet,

    n_transform: usize,
    n_drawobj: usize,
    n_paths: usize,
    n_pathseg: usize,
    n_pathtag: usize,
    n_clip: u32,

    // Keep a reference to the image so that it is not destroyed.
    _bg_image: Image,

    gradient_bufs: Vec<Buffer>,
    gradients: Image,
}

impl RenderConfig {
    pub fn new(width: usize, height: usize) -> RenderConfig {
        RenderConfig {
            width,
            height,
            format: PixelFormat::Rgba8,
        }
    }

    pub fn pixel_format(mut self, format: PixelFormat) -> Self {
        self.format = format;
        self
    }
}

impl Renderer {
    /// The number of query pool entries needed to run the renderer.
    pub const QUERY_POOL_SIZE: u32 = 12;

    pub unsafe fn new(
        session: &Session,
        width: usize,
        height: usize,
        n_bufs: usize,
    ) -> Result<Self, Error> {
        let config = RenderConfig::new(width, height);
        Self::new_from_config(session, config, n_bufs)
    }

    /// Create a new renderer.
    pub unsafe fn new_from_config(
        session: &Session,
        config: RenderConfig,
        n_bufs: usize,
    ) -> Result<Self, Error> {
        // For now, round up to tile alignment
        let width = config.width;
        let height = config.height;
        let width = width + (width.wrapping_neg() & (TILE_W - 1));
        let height = height + (height.wrapping_neg() & (TILE_W - 1));
        let dev = BufferUsage::STORAGE | BufferUsage::COPY_DST;
        let host_upload = BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC;

        // This may be inadequate for very complex scenes (paris etc)
        // TODO: separate staging buffer (if needed)
        let scene_bufs = (0..n_bufs)
            .map(|_| session.create_buffer(8 * 1024 * 1024, host_upload).unwrap())
            .collect::<Vec<_>>();

        let image_format = match config.format {
            PixelFormat::A8 => piet_gpu_hal::ImageFormat::A8,
            PixelFormat::Rgba8 => piet_gpu_hal::ImageFormat::Rgba8,
        };
        let image_dev = session.create_image2d(width as u32, height as u32, image_format)?;

        const CONFIG_BUFFER_SIZE: u64 = std::mem::size_of::<Config>() as u64;
        let config_buf = session.create_buffer(CONFIG_BUFFER_SIZE, dev).unwrap();
        let config_bufs = (0..n_bufs)
            .map(|_| {
                session
                    .create_buffer(CONFIG_BUFFER_SIZE, host_upload)
                    .unwrap()
            })
            .collect();

        let memory_buf_host = (0..n_bufs)
            .map(|_| session.create_buffer(2 * 4, host_upload).unwrap())
            .collect();
        let memory_buf_dev = session.create_buffer(128 * 1024 * 1024, dev)?;

        let element_code = ElementCode::new(session);
        let element_stage = ElementStage::new(session, &element_code);
        let element_bindings = scene_bufs
            .iter()
            .map(|scene_buf| {
                element_stage.bind(
                    session,
                    &element_code,
                    &config_buf,
                    scene_buf,
                    &memory_buf_dev,
                )
            })
            .collect();

        let clip_code = ClipCode::new(session);
        let clip_binding = ClipBinding::new(session, &clip_code, &config_buf, &memory_buf_dev);

        let tile_alloc_code = include_shader!(session, "../shader/gen/tile_alloc");
        let tile_pipeline = session.create_compute_pipeline(
            tile_alloc_code,
            &[
                BindType::Buffer,
                BindType::BufReadOnly,
                BindType::BufReadOnly,
            ],
        )?;
        let tile_ds = scene_bufs
            .iter()
            .map(|scene_buf| {
                session.create_simple_descriptor_set(
                    &tile_pipeline,
                    &[&memory_buf_dev, &config_buf, scene_buf],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let path_alloc_code = include_shader!(session, "../shader/gen/path_coarse");
        let path_pipeline = session
            .create_compute_pipeline(path_alloc_code, &[BindType::Buffer, BindType::BufReadOnly])?;
        let path_ds = session
            .create_simple_descriptor_set(&path_pipeline, &[&memory_buf_dev, &config_buf])?;

        let (backdrop_code, backdrop_y) =
            if session.gpu_info().workgroup_limits.max_invocations >= 1024 {
                (include_shader!(session, "../shader/gen/backdrop_lg"), 4)
            } else {
                println!("using small workgroup backdrop kernel");
                (include_shader!(session, "../shader/gen/backdrop"), 1)
            };
        let backdrop_pipeline = session
            .create_compute_pipeline(backdrop_code, &[BindType::Buffer, BindType::BufReadOnly])?;
        let backdrop_ds = session
            .create_simple_descriptor_set(&backdrop_pipeline, &[&memory_buf_dev, &config_buf])?;

        // TODO: constants
        let bin_code = include_shader!(session, "../shader/gen/binning");
        let bin_pipeline = session
            .create_compute_pipeline(bin_code, &[BindType::Buffer, BindType::BufReadOnly])?;
        let bin_ds =
            session.create_simple_descriptor_set(&bin_pipeline, &[&memory_buf_dev, &config_buf])?;

        let coarse_code = include_shader!(session, "../shader/gen/coarse");
        let coarse_pipeline = session.create_compute_pipeline(
            coarse_code,
            &[
                BindType::Buffer,
                BindType::BufReadOnly,
                BindType::BufReadOnly,
            ],
        )?;
        let coarse_ds = scene_bufs
            .iter()
            .map(|scene_buf| {
                session.create_simple_descriptor_set(
                    &coarse_pipeline,
                    &[&memory_buf_dev, &config_buf, scene_buf],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bg_image = Self::make_test_bg_image(&session);

        const GRADIENT_BUF_SIZE: usize =
            crate::gradient::N_GRADIENTS * crate::gradient::N_SAMPLES * 4;
        let gradient_bufs = (0..n_bufs)
            .map(|_| {
                session
                    .create_buffer(GRADIENT_BUF_SIZE as u64, host_upload)
                    .unwrap()
            })
            .collect();
        let gradients = Self::make_gradient_image(&session);

        let k4_code = match config.format {
            PixelFormat::A8 => include_shader!(session, "../shader/gen/kernel4_gray"),
            PixelFormat::Rgba8 => include_shader!(session, "../shader/gen/kernel4"),
        };
        let k4_pipeline = session.create_compute_pipeline(
            k4_code,
            &[
                BindType::Buffer,
                BindType::BufReadOnly,
                BindType::Image,
                BindType::ImageRead,
                BindType::ImageRead,
            ],
        )?;
        let k4_ds = session
            .descriptor_set_builder()
            .add_buffers(&[&memory_buf_dev, &config_buf])
            .add_images(&[&image_dev])
            .add_textures(&[&bg_image, &gradients])
            .build(&session, &k4_pipeline)?;

        Ok(Renderer {
            width,
            height,
            scene_bufs,
            memory_buf_host,
            memory_buf_dev,
            config_buf,
            config_bufs,
            image_dev,
            element_code,
            element_stage,
            element_bindings,
            clip_code,
            clip_binding,
            tile_pipeline,
            tile_ds,
            path_pipeline,
            path_ds,
            backdrop_pipeline,
            backdrop_ds,
            backdrop_y,
            bin_pipeline,
            bin_ds,
            coarse_pipeline,
            coarse_ds,
            k4_pipeline,
            k4_ds,
            n_transform: 0,
            n_drawobj: 0,
            n_paths: 0,
            n_pathseg: 0,
            n_pathtag: 0,
            n_clip: 0,
            _bg_image: bg_image,
            gradient_bufs,
            gradients,
        })
    }

    /// Convert the scene in the render context to GPU resources.
    ///
    /// At present, this requires that any command buffer submission has completed.
    /// A future evolution will handle staging of the next frame's scene while the
    /// rendering of the current frame is in flight.
    pub fn upload_render_ctx(
        &mut self,
        render_ctx: &mut PietGpuRenderContext,
        buf_ix: usize,
    ) -> Result<(), Error> {
        let (mut config, mut alloc) = render_ctx.stage_config();
        let n_drawobj = render_ctx.n_drawobj();
        // TODO: be more consistent in size types
        let n_path = render_ctx.n_path() as usize;
        self.n_paths = n_path;
        self.n_transform = render_ctx.n_transform();
        self.n_drawobj = render_ctx.n_drawobj();
        self.n_pathseg = render_ctx.n_pathseg() as usize;
        self.n_pathtag = render_ctx.n_pathtag();
        self.n_clip = render_ctx.n_clip();

        // These constants depend on encoding and may need to be updated.
        // Perhaps we can plumb these from piet-gpu-derive?
        const PATH_SIZE: usize = 12;
        const BIN_SIZE: usize = 8;
        let width_in_tiles = self.width / TILE_W;
        let height_in_tiles = self.height / TILE_H;
        let tile_base = alloc;
        alloc += ((n_path + 3) & !3) * PATH_SIZE;
        let bin_base = alloc;
        alloc += ((n_drawobj + 255) & !255) * BIN_SIZE;
        let ptcl_base = alloc;
        alloc += width_in_tiles * height_in_tiles * PTCL_INITIAL_ALLOC;

        config.width_in_tiles = width_in_tiles as u32;
        config.height_in_tiles = height_in_tiles as u32;
        config.tile_alloc = tile_base as u32;
        config.bin_alloc = bin_base as u32;
        config.ptcl_alloc = ptcl_base as u32;
        unsafe {
            // TODO: reallocate scene buffer if size is inadequate
            {
                let mut mapped_scene = self.scene_bufs[buf_ix].map_write(..)?;
                render_ctx.write_scene(&mut mapped_scene);
            }
            self.config_bufs[buf_ix].write(&[config])?;
            self.memory_buf_host[buf_ix].write(&[alloc as u32, 0 /* Overflow flag */])?;

            // Upload gradient data.
            let ramp_data = render_ctx.get_ramp_data();
            if !ramp_data.is_empty() {
                assert!(
                    self.gradient_bufs[buf_ix].size() as usize
                        >= std::mem::size_of_val(&*ramp_data)
                );
                self.gradient_bufs[buf_ix].write(&ramp_data)?;
            }
        }
        Ok(())
    }

    pub fn upload_scene<T: Copy + Pod>(
        &mut self,
        scene: &EncodedSceneRef<T>,
        buf_ix: usize,
    ) -> Result<(), Error> {
        let (mut config, mut alloc) = scene.stage_config();
        let n_drawobj = scene.n_drawobj();
        // TODO: be more consistent in size types
        let n_path = scene.n_path() as usize;
        self.n_paths = n_path;
        self.n_transform = scene.n_transform();
        self.n_drawobj = scene.n_drawobj();
        self.n_pathseg = scene.n_pathseg() as usize;
        self.n_pathtag = scene.n_pathtag();
        self.n_clip = scene.n_clip();

        // These constants depend on encoding and may need to be updated.
        // Perhaps we can plumb these from piet-gpu-derive?
        const PATH_SIZE: usize = 12;
        const BIN_SIZE: usize = 8;
        let width_in_tiles = self.width / TILE_W;
        let height_in_tiles = self.height / TILE_H;
        let tile_base = alloc;
        alloc += ((n_path + 3) & !3) * PATH_SIZE;
        let bin_base = alloc;
        alloc += ((n_drawobj + 255) & !255) * BIN_SIZE;
        let ptcl_base = alloc;
        alloc += width_in_tiles * height_in_tiles * PTCL_INITIAL_ALLOC;

        config.width_in_tiles = width_in_tiles as u32;
        config.height_in_tiles = height_in_tiles as u32;
        config.tile_alloc = tile_base as u32;
        config.bin_alloc = bin_base as u32;
        config.ptcl_alloc = ptcl_base as u32;
        unsafe {
            // TODO: reallocate scene buffer if size is inadequate
            {
                let mut mapped_scene = self.scene_bufs[buf_ix].map_write(..)?;
                scene.write_scene(&mut mapped_scene);
            }
            self.config_bufs[buf_ix].write(&[config])?;
            self.memory_buf_host[buf_ix].write(&[alloc as u32, 0 /* Overflow flag */])?;

            // Upload gradient data.
            if !scene.ramp_data.is_empty() {
                assert!(
                    self.gradient_bufs[buf_ix].size() as usize
                        >= std::mem::size_of_val(&*scene.ramp_data)
                );
                self.gradient_bufs[buf_ix].write(scene.ramp_data)?;
            }
        }
        Ok(())
    }

    pub unsafe fn record(&self, cmd_buf: &mut CmdBuf, query_pool: &QueryPool, buf_ix: usize) {
        cmd_buf.copy_buffer(&self.config_bufs[buf_ix], &self.config_buf);
        cmd_buf.copy_buffer(&self.memory_buf_host[buf_ix], &self.memory_buf_dev);
        cmd_buf.memory_barrier();
        cmd_buf.image_barrier(
            &self.image_dev,
            ImageLayout::Undefined,
            ImageLayout::General,
        );
        // TODO: make gradient upload optional, only if it's changed
        cmd_buf.image_barrier(
            &self.gradients,
            ImageLayout::Undefined,
            ImageLayout::BlitDst,
        );
        cmd_buf.copy_buffer_to_image(&self.gradient_bufs[buf_ix], &self.gradients);
        cmd_buf.image_barrier(&self.gradients, ImageLayout::BlitDst, ImageLayout::General);
        cmd_buf.reset_query_pool(&query_pool);
        cmd_buf.begin_debug_label("Element bounding box calculation");
        let mut pass = cmd_buf.begin_compute_pass(&ComputePassDescriptor::timer(&query_pool, 0, 1));
        self.element_stage.record(
            &mut pass,
            &self.element_code,
            &self.element_bindings[buf_ix],
            self.n_transform as u64,
            self.n_paths as u32,
            self.n_pathtag as u32,
            self.n_drawobj as u64,
        );
        pass.end();
        cmd_buf.end_debug_label();
        cmd_buf.memory_barrier();
        let mut pass = cmd_buf.begin_compute_pass(&ComputePassDescriptor::timer(&query_pool, 2, 3));
        pass.begin_debug_label("Clip bounding box calculation");
        self.clip_binding
            .record(&mut pass, &self.clip_code, self.n_clip as u32);
        pass.end_debug_label();
        pass.begin_debug_label("Element binning");
        pass.dispatch(
            &self.bin_pipeline,
            &self.bin_ds,
            (((self.n_paths + 255) / 256) as u32, 1, 1),
            (256, 1, 1),
        );
        pass.end_debug_label();
        pass.memory_barrier();
        pass.begin_debug_label("Tile allocation");
        pass.dispatch(
            &self.tile_pipeline,
            &self.tile_ds[buf_ix],
            (((self.n_paths + 255) / 256) as u32, 1, 1),
            (256, 1, 1),
        );
        pass.end_debug_label();
        pass.end();
        cmd_buf.begin_debug_label("Path flattening");
        cmd_buf.memory_barrier();
        let mut pass = cmd_buf.begin_compute_pass(&ComputePassDescriptor::timer(&query_pool, 4, 5));
        pass.dispatch(
            &self.path_pipeline,
            &self.path_ds,
            (((self.n_pathseg + 31) / 32) as u32, 1, 1),
            (32, 1, 1),
        );
        pass.end();
        cmd_buf.end_debug_label();
        cmd_buf.memory_barrier();
        cmd_buf.begin_debug_label("Backdrop propagation");
        let mut pass = cmd_buf.begin_compute_pass(&ComputePassDescriptor::timer(&query_pool, 6, 7));
        pass.dispatch(
            &self.backdrop_pipeline,
            &self.backdrop_ds,
            (((self.n_paths + 255) / 256) as u32, 1, 1),
            (256, self.backdrop_y, 1),
        );
        pass.end();
        cmd_buf.end_debug_label();
        // TODO: redo query accounting
        cmd_buf.memory_barrier();
        cmd_buf.begin_debug_label("Coarse raster");
        let mut pass = cmd_buf.begin_compute_pass(&ComputePassDescriptor::timer(&query_pool, 8, 9));
        pass.dispatch(
            &self.coarse_pipeline,
            &self.coarse_ds[buf_ix],
            (
                (self.width as u32 + 255) / 256,
                (self.height as u32 + 255) / 256,
                1,
            ),
            (256, 1, 1),
        );
        pass.end();
        cmd_buf.end_debug_label();
        cmd_buf.memory_barrier();
        cmd_buf.begin_debug_label("Fine raster");
        let mut pass =
            cmd_buf.begin_compute_pass(&ComputePassDescriptor::timer(&query_pool, 10, 11));
        pass.dispatch(
            &self.k4_pipeline,
            &self.k4_ds,
            (
                (self.width / TILE_W) as u32,
                (self.height / TILE_H) as u32,
                1,
            ),
            (8, 4, 1),
        );
        pass.end();
        cmd_buf.end_debug_label();
        cmd_buf.memory_barrier();
        cmd_buf.image_barrier(&self.image_dev, ImageLayout::General, ImageLayout::BlitSrc);
    }

    pub fn make_image(
        session: &Session,
        width: usize,
        height: usize,
        buf: &[u8],
        format: ImageFormat,
    ) -> Result<Image, Error> {
        unsafe {
            if format != ImageFormat::RgbaPremul {
                return Err("unsupported image format".into());
            }
            let buffer = session.create_buffer_init(&buf, BufferUsage::COPY_SRC)?;
            const RGBA: piet_gpu_hal::ImageFormat = piet_gpu_hal::ImageFormat::Rgba8;
            let image = session.create_image2d(width.try_into()?, height.try_into()?, RGBA)?;
            let mut cmd_buf = session.cmd_buf()?;
            cmd_buf.begin();
            cmd_buf.image_barrier(&image, ImageLayout::Undefined, ImageLayout::BlitDst);
            cmd_buf.copy_buffer_to_image(&buffer, &image);
            cmd_buf.image_barrier(&image, ImageLayout::BlitDst, ImageLayout::General);
            cmd_buf.finish();
            // Make sure not to drop the buffer and image until the command buffer completes.
            cmd_buf.add_resource(&buffer);
            cmd_buf.add_resource(&image);
            let _ = session.run_cmd_buf(cmd_buf, &[], &[]);
            // We let the session reclaim the fence.
            Ok(image)
        }
    }

    /// Make a test image.
    fn make_test_bg_image(session: &Session) -> Image {
        const WIDTH: usize = 256;
        const HEIGHT: usize = 256;
        let mut buf = vec![255u8; WIDTH * HEIGHT * 4];
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let r = x as u8;
                let g = y as u8;
                let b = r ^ g;
                buf[(y * WIDTH + x) * 4] = r;
                buf[(y * WIDTH + x) * 4 + 1] = g;
                buf[(y * WIDTH + x) * 4 + 2] = b;
            }
        }
        Self::make_image(session, WIDTH, HEIGHT, &buf, ImageFormat::RgbaPremul).unwrap()
    }

    fn make_gradient_image(session: &Session) -> Image {
        unsafe {
            const RGBA: piet_gpu_hal::ImageFormat = piet_gpu_hal::ImageFormat::Rgba8;
            session
                .create_image2d(
                    gradient::N_SAMPLES as u32,
                    gradient::N_GRADIENTS as u32,
                    RGBA,
                )
                .unwrap()
        }
    }
}
