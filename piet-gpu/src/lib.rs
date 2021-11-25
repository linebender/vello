mod gradient;
mod pico_svg;
mod render_ctx;
pub mod stages;
pub mod test_scenes;
mod text;

use std::convert::TryInto;

pub use render_ctx::PietGpuRenderContext;

use piet::kurbo::Vec2;
use piet::{ImageFormat, RenderContext};

use piet_gpu_types::encoder::Encode;

use piet_gpu_hal::{
    BindType, Buffer, BufferUsage, CmdBuf, DescriptorSet, Error, Image, ImageLayout, Pipeline,
    QueryPool, Session, ShaderCode,
};

use pico_svg::PicoSvg;

use crate::stages::Config;

const TILE_W: usize = 16;
const TILE_H: usize = 16;

const PTCL_INITIAL_ALLOC: usize = 1024;

const MAX_BLEND_STACK: usize = 128;

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

    state_buf: Buffer,

    // Staging buffers
    config_bufs: Vec<Buffer>,
    // Device config buf
    config_buf: Buffer,

    el_pipeline: Pipeline,
    el_ds: Vec<DescriptorSet>,

    tile_pipeline: Pipeline,
    tile_ds: DescriptorSet,

    path_pipeline: Pipeline,
    path_ds: DescriptorSet,

    backdrop_pipeline: Pipeline,
    backdrop_ds: DescriptorSet,

    bin_pipeline: Pipeline,
    bin_ds: DescriptorSet,

    coarse_pipeline: Pipeline,
    coarse_ds: DescriptorSet,

    k4_pipeline: Pipeline,
    k4_ds: DescriptorSet,

    n_elements: usize,
    n_paths: usize,
    n_pathseg: usize,

    // Keep a reference to the image so that it is not destroyed.
    _bg_image: Image,

    gradient_bufs: Vec<Buffer>,
    gradients: Image,
}

impl Renderer {
    /// Create a new renderer.
    pub unsafe fn new(
        session: &Session,
        width: usize,
        height: usize,
        n_bufs: usize,
    ) -> Result<Self, Error> {
        // For now, round up to tile alignment
        let width = width + (width.wrapping_neg() & (TILE_W - 1));
        let height = height + (height.wrapping_neg() & (TILE_W - 1));
        let dev = BufferUsage::STORAGE | BufferUsage::COPY_DST;
        let host_upload = BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC;

        // This may be inadequate for very complex scenes (paris etc)
        // TODO: separate staging buffer (if needed)
        let scene_bufs = (0..n_bufs)
            .map(|_| session.create_buffer(8 * 1024 * 1024, host_upload).unwrap())
            .collect();

        let state_buf = session.create_buffer(1 * 1024 * 1024, dev)?;
        let image_dev = session.create_image2d(width as u32, height as u32)?;

        // Note: this must be updated when the config struct size changes.
        const CONFIG_BUFFER_SIZE: u64 = std::mem::size_of::<Config>() as u64;
        let config_buf = session.create_buffer(CONFIG_BUFFER_SIZE, dev).unwrap();
        // TODO: separate staging buffer (if needed)
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

        let el_code = ShaderCode::Spv(include_bytes!("../shader/elements.spv"));
        let el_pipeline = session.create_compute_pipeline(
            el_code,
            &[
                BindType::Buffer,
                BindType::Buffer,
                BindType::Buffer,
                BindType::Buffer,
            ],
        )?;
        let mut el_ds = Vec::with_capacity(n_bufs);
        for scene_buf in &scene_bufs {
            el_ds.push(session.create_simple_descriptor_set(
                &el_pipeline,
                &[&memory_buf_dev, &config_buf, scene_buf, &state_buf],
            )?);
        }

        let tile_alloc_code = ShaderCode::Spv(include_bytes!("../shader/tile_alloc.spv"));
        let tile_pipeline = session
            .create_compute_pipeline(tile_alloc_code, &[BindType::Buffer, BindType::Buffer])?;
        let tile_ds = session
            .create_simple_descriptor_set(&tile_pipeline, &[&memory_buf_dev, &config_buf])?;

        let path_alloc_code = ShaderCode::Spv(include_bytes!("../shader/path_coarse.spv"));
        let path_pipeline = session
            .create_compute_pipeline(path_alloc_code, &[BindType::Buffer, BindType::Buffer])?;
        let path_ds = session
            .create_simple_descriptor_set(&path_pipeline, &[&memory_buf_dev, &config_buf])?;

        let backdrop_code = if session.gpu_info().workgroup_limits.max_invocations >= 1024 {
            ShaderCode::Spv(include_bytes!("../shader/backdrop_lg.spv"))
        } else {
            println!("using small workgroup backdrop kernel");
            ShaderCode::Spv(include_bytes!("../shader/backdrop.spv"))
        };
        let backdrop_pipeline = session
            .create_compute_pipeline(backdrop_code, &[BindType::Buffer, BindType::Buffer])?;
        let backdrop_ds = session
            .create_simple_descriptor_set(&backdrop_pipeline, &[&memory_buf_dev, &config_buf])?;

        // TODO: constants
        let bin_code = ShaderCode::Spv(include_bytes!("../shader/binning.spv"));
        let bin_pipeline =
            session.create_compute_pipeline(bin_code, &[BindType::Buffer, BindType::Buffer])?;
        let bin_ds =
            session.create_simple_descriptor_set(&bin_pipeline, &[&memory_buf_dev, &config_buf])?;

        let coarse_code = ShaderCode::Spv(include_bytes!("../shader/coarse.spv"));
        let coarse_pipeline =
            session.create_compute_pipeline(coarse_code, &[BindType::Buffer, BindType::Buffer])?;
        let coarse_ds = session
            .create_simple_descriptor_set(&coarse_pipeline, &[&memory_buf_dev, &config_buf])?;

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

        let k4_code = ShaderCode::Spv(include_bytes!("../shader/kernel4.spv"));
        let k4_pipeline = session.create_compute_pipeline(
            k4_code,
            &[
                BindType::Buffer,
                BindType::Buffer,
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
            state_buf,
            config_buf,
            config_bufs,
            image_dev,
            el_pipeline,
            el_ds,
            tile_pipeline,
            tile_ds,
            path_pipeline,
            path_ds,
            backdrop_pipeline,
            backdrop_ds,
            bin_pipeline,
            bin_ds,
            coarse_pipeline,
            coarse_ds,
            k4_pipeline,
            k4_ds,
            n_elements: 0,
            n_paths: 0,
            n_pathseg: 0,
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
        let n_paths = render_ctx.path_count();
        let n_pathseg = render_ctx.pathseg_count();
        let n_trans = render_ctx.trans_count();
        self.n_paths = n_paths;
        self.n_pathseg = n_pathseg;

        // These constants depend on encoding and may need to be updated.
        // Perhaps we can plumb these from piet-gpu-derive?
        const PATH_SIZE: usize = 12;
        const BIN_SIZE: usize = 8;
        const PATHSEG_SIZE: usize = 52;
        const ANNO_SIZE: usize = 40;
        const TRANS_SIZE: usize = 24;
        let width_in_tiles = self.width / TILE_W;
        let height_in_tiles = self.height / TILE_H;
        let mut alloc = 0;
        let tile_base = alloc;
        alloc += ((n_paths + 3) & !3) * PATH_SIZE;
        let bin_base = alloc;
        alloc += ((n_paths + 255) & !255) * BIN_SIZE;
        let ptcl_base = alloc;
        alloc += width_in_tiles * height_in_tiles * PTCL_INITIAL_ALLOC;
        let pathseg_base = alloc;
        alloc += (n_pathseg * PATHSEG_SIZE + 3) & !3;
        let anno_base = alloc;
        alloc += (n_paths * ANNO_SIZE + 3) & !3;
        let trans_base = alloc;
        alloc += (n_trans * TRANS_SIZE + 3) & !3;
        let config = Config {
            n_elements: n_paths as u32,
            n_pathseg: n_pathseg as u32,
            width_in_tiles: width_in_tiles as u32,
            height_in_tiles: height_in_tiles as u32,
            tile_alloc: tile_base as u32,
            bin_alloc: bin_base as u32,
            ptcl_alloc: ptcl_base as u32,
            pathseg_alloc: pathseg_base as u32,
            anno_alloc: anno_base as u32,
            trans_alloc: trans_base as u32,
            n_trans: n_trans as u32,
            // We'll fill the rest of the fields in when we hook up the new element pipeline.
            ..Default::default()
        };
        unsafe {
            let scene = render_ctx.get_scene_buf();
            self.n_elements = scene.len() / piet_gpu_types::scene::Element::fixed_size();
            // TODO: reallocate scene buffer if size is inadequate
            assert!(self.scene_bufs[buf_ix].size() as usize >= scene.len());
            self.scene_bufs[buf_ix].write(scene)?;
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

    pub unsafe fn record(&self, cmd_buf: &mut CmdBuf, query_pool: &QueryPool, buf_ix: usize) {
        cmd_buf.copy_buffer(&self.config_bufs[buf_ix], &self.config_buf);
        cmd_buf.copy_buffer(&self.memory_buf_host[buf_ix], &self.memory_buf_dev);
        cmd_buf.clear_buffer(&self.state_buf, None);
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
        cmd_buf.write_timestamp(&query_pool, 0);
        cmd_buf.dispatch(
            &self.el_pipeline,
            &self.el_ds[buf_ix],
            (((self.n_elements + 127) / 128) as u32, 1, 1),
            (128, 1, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 1);
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(
            &self.tile_pipeline,
            &self.tile_ds,
            (((self.n_paths + 255) / 256) as u32, 1, 1),
            (256, 1, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 2);
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(
            &self.path_pipeline,
            &self.path_ds,
            (((self.n_pathseg + 31) / 32) as u32, 1, 1),
            (32, 1, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 3);
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(
            &self.backdrop_pipeline,
            &self.backdrop_ds,
            (((self.n_paths + 255) / 256) as u32, 1, 1),
            (256, 1, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 4);
        // Note: this barrier is not needed as an actual dependency between
        // pipeline stages, but I am keeping it in so that timer queries are
        // easier to interpret.
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(
            &self.bin_pipeline,
            &self.bin_ds,
            (((self.n_paths + 255) / 256) as u32, 1, 1),
            (256, 1, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 5);
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(
            &self.coarse_pipeline,
            &self.coarse_ds,
            (
                (self.width as u32 + 255) / 256,
                (self.height as u32 + 255) / 256,
                1,
            ),
            (256, 256, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 6);
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(
            &self.k4_pipeline,
            &self.k4_ds,
            (
                (self.width / TILE_W) as u32,
                (self.height / TILE_H) as u32,
                1,
            ),
            (8, 4, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 7);
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
            let image = session.create_image2d(width.try_into()?, height.try_into()?)?;
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
            session
                .create_image2d(gradient::N_SAMPLES as u32, gradient::N_GRADIENTS as u32)
                .unwrap()
        }
    }
}
