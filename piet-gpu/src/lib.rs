mod pico_svg;
mod render_ctx;

use std::convert::TryInto;

pub use render_ctx::PietGpuRenderContext;

use rand::{Rng, RngCore};

use piet::kurbo::{BezPath, Circle, Point, Vec2};
use piet::{Color, ImageFormat, RenderContext};

use piet_gpu_types::encoder::Encode;

use piet_gpu_hal::{SamplerParams, hub};
use piet_gpu_hal::{CmdBuf, Error, ImageLayout, MemFlags};

use pico_svg::PicoSvg;

pub const WIDTH: usize = TILE_W * WIDTH_IN_TILES;
pub const HEIGHT: usize = TILE_H * HEIGHT_IN_TILES;

const TILE_W: usize = 16;
const TILE_H: usize = 16;

const WIDTH_IN_TILES: usize = 128;
const HEIGHT_IN_TILES: usize = 96;
const PTCL_INITIAL_ALLOC: usize = 1024;

const N_CIRCLES: usize = 0;

pub fn render_svg(rc: &mut impl RenderContext, filename: &str, scale: f64) {
    let xml_str = std::fs::read_to_string(filename).unwrap();
    let start = std::time::Instant::now();
    let svg = PicoSvg::load(&xml_str, scale).unwrap();
    println!("parsing time: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    svg.render(rc);
    println!("flattening and encoding time: {:?}", start.elapsed());
}

pub fn render_scene(rc: &mut impl RenderContext) {
    let mut rng = rand::thread_rng();
    for _ in 0..N_CIRCLES {
        let color = Color::from_rgba32_u32(rng.next_u32());
        let center = Point::new(
            rng.gen_range(0.0, WIDTH as f64),
            rng.gen_range(0.0, HEIGHT as f64),
        );
        let radius = rng.gen_range(0.0, 50.0);
        let circle = Circle::new(center, radius);
        rc.fill(circle, &color);
    }
    let _ = rc.save();
    let mut path = BezPath::new();
    path.move_to((200.0, 150.0));
    path.line_to((100.0, 200.0));
    path.line_to((150.0, 250.0));
    path.close_path();
    rc.clip(path);

    let mut path = BezPath::new();
    path.move_to((100.0, 150.0));
    path.line_to((200.0, 200.0));
    path.line_to((150.0, 250.0));
    path.close_path();
    rc.fill(path, &Color::rgb8(128, 0, 128));
    let _ = rc.restore();
    rc.stroke(
        piet::kurbo::Line::new((100.0, 100.0), (200.0, 150.0)),
        &Color::WHITE,
        5.0,
    );
    //render_cardioid(rc);
    render_clip_test(rc);
    //render_tiger(rc);
}

#[allow(unused)]
fn render_cardioid(rc: &mut impl RenderContext) {
    let n = 601;
    let dth = std::f64::consts::PI * 2.0 / (n as f64);
    let center = Point::new(1024.0, 768.0);
    let r = 750.0;
    let mut path = BezPath::new();
    for i in 1..n {
        let p0 = center + Vec2::from_angle(i as f64 * dth) * r;
        let p1 = center + Vec2::from_angle(((i * 2) % n) as f64 * dth) * r;
        //rc.fill(&Circle::new(p0, 8.0), &Color::WHITE);
        path.move_to(p0);
        path.line_to(p1);
        //rc.stroke(Line::new(p0, p1), &Color::BLACK, 2.0);
    }
    rc.stroke(&path, &Color::BLACK, 2.0);
}

#[allow(unused)]
fn render_clip_test(rc: &mut impl RenderContext) {
    const N: usize = 16;
    const X0: f64 = 50.0;
    const Y0: f64 = 450.0;
    // Note: if it gets much larger, it will exceed the 1MB scratch buffer.
    // But this is a pretty demanding test.
    const X1: f64 = 550.0;
    const Y1: f64 = 950.0;
    let step = 1.0 / ((N + 1) as f64);
    for i in 0..N {
        let t = ((i + 1) as f64) * step;
        rc.save();
        let mut path = BezPath::new();
        path.move_to((X0, Y0));
        path.line_to((X1, Y0));
        path.line_to((X1, Y0 + t * (Y1 - Y0)));
        path.line_to((X1 + t * (X0 - X1), Y1));
        path.line_to((X0, Y1));
        path.close_path();
        rc.clip(path);
    }
    let rect = piet::kurbo::Rect::new(X0, Y0, X1, Y1);
    rc.fill(rect, &Color::BLACK);
    for _ in 0..N {
        rc.restore();
    }
}

fn render_tiger(rc: &mut impl RenderContext) {
    let xml_str = std::str::from_utf8(include_bytes!("../Ghostscript_Tiger.svg")).unwrap();
    let start = std::time::Instant::now();
    let svg = PicoSvg::load(xml_str, 8.0).unwrap();
    println!("parsing time: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    svg.render(rc);
    println!("flattening and encoding time: {:?}", start.elapsed());
}

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
    pub image_dev: hub::Image, // resulting image

    scene_buf_host: hub::Buffer,
    scene_buf_dev: hub::Buffer,

    memory_buf_host: hub::Buffer,
    memory_buf_dev: hub::Buffer,

    state_buf: hub::Buffer,

    config_buf_host: hub::Buffer,
    config_buf_dev: hub::Buffer,

    el_pipeline: hub::Pipeline,
    el_ds: hub::DescriptorSet,

    tile_pipeline: hub::Pipeline,
    tile_ds: hub::DescriptorSet,

    path_pipeline: hub::Pipeline,
    path_ds: hub::DescriptorSet,

    backdrop_pipeline: hub::Pipeline,
    backdrop_ds: hub::DescriptorSet,

    bin_pipeline: hub::Pipeline,
    bin_ds: hub::DescriptorSet,

    coarse_pipeline: hub::Pipeline,
    coarse_ds: hub::DescriptorSet,

    k4_pipeline: hub::Pipeline,
    k4_ds: hub::DescriptorSet,

    n_elements: usize,
    n_paths: usize,
    n_pathseg: usize,

    bg_image: hub::Image,
}

impl Renderer {
    pub unsafe fn new(
        session: &hub::Session,
        scene: &[u8],
        n_paths: usize,
        n_pathseg: usize,
        n_trans: usize,
    ) -> Result<Self, Error> {
        let host = MemFlags::host_coherent();
        let dev = MemFlags::device_local();

        let n_elements = scene.len() / piet_gpu_types::scene::Element::fixed_size();
        println!(
            "scene: {} elements, {} paths, {} path_segments, {} transforms",
            n_elements, n_paths, n_pathseg, n_trans
        );

        let mut scene_buf_host = session
            .create_buffer(std::mem::size_of_val(&scene[..]) as u64, host)
            .unwrap();
        let scene_buf_dev = session
            .create_buffer(std::mem::size_of_val(&scene[..]) as u64, dev)
            .unwrap();
        scene_buf_host.write(&scene)?;

        let state_buf = session.create_buffer(1 * 1024 * 1024, dev)?;
        let image_dev = session.create_image2d(WIDTH as u32, HEIGHT as u32, dev)?;

        const CONFIG_SIZE: u64 = 10*4; // Size of Config in setup.h.
        let mut config_buf_host = session.create_buffer(CONFIG_SIZE, host)?;
        let config_buf_dev = session.create_buffer(CONFIG_SIZE, dev)?;

        // TODO: constants
        const PATH_SIZE: usize = 12;
        const BIN_SIZE: usize = 8;
        const PATHSEG_SIZE: usize = 52;
        const ANNO_SIZE: usize = 28;
        const TRANS_SIZE: usize = 24;
        let mut alloc = 0;
        let tile_base = alloc;
        alloc += ((n_paths + 3) & !3) * PATH_SIZE;
        let bin_base = alloc;
        alloc += ((n_paths + 255) & !255) * BIN_SIZE;
        let ptcl_base = alloc;
        alloc += WIDTH_IN_TILES * HEIGHT_IN_TILES * PTCL_INITIAL_ALLOC;
        let pathseg_base = alloc;
        alloc += (n_pathseg * PATHSEG_SIZE + 3) & !3;
        let anno_base = alloc;
        alloc += (n_paths * ANNO_SIZE + 3) & !3;
        let trans_base = alloc;
        alloc += (n_trans * TRANS_SIZE + 3) & !3;
        config_buf_host.write(&[n_paths as u32, n_pathseg as u32, WIDTH_IN_TILES as u32, HEIGHT_IN_TILES as u32, tile_base as u32, bin_base as u32, ptcl_base as u32, pathseg_base as u32, anno_base as u32, trans_base as u32])?;

        let mut memory_buf_host = session.create_buffer(2*4, host)?;
        let memory_buf_dev = session.create_buffer(128 * 1024 * 1024, dev)?;
        memory_buf_host.write(&[alloc as u32, 0 /* Overflow flag */])?;

        let el_code = include_bytes!("../shader/elements.spv");
        let el_pipeline = session.create_simple_compute_pipeline(el_code, 4)?;
        let el_ds = session.create_simple_descriptor_set(
            &el_pipeline,
            &[&memory_buf_dev, &config_buf_dev, &scene_buf_dev, &state_buf],
        )?;

        let tile_alloc_code = include_bytes!("../shader/tile_alloc.spv");
        let tile_pipeline = session.create_simple_compute_pipeline(tile_alloc_code, 2)?;
        let tile_ds = session.create_simple_descriptor_set(
            &tile_pipeline,
            &[&memory_buf_dev, &config_buf_dev],
        )?;

        let path_alloc_code = include_bytes!("../shader/path_coarse.spv");
        let path_pipeline = session.create_simple_compute_pipeline(path_alloc_code, 2)?;
        let path_ds = session.create_simple_descriptor_set(
            &path_pipeline,
            &[&memory_buf_dev, &config_buf_dev],
        )?;

        let backdrop_alloc_code = include_bytes!("../shader/backdrop.spv");
        let backdrop_pipeline = session.create_simple_compute_pipeline(backdrop_alloc_code, 2)?;
        let backdrop_ds = session.create_simple_descriptor_set(
            &backdrop_pipeline,
            &[&memory_buf_dev, &config_buf_dev],
        )?;

        // TODO: constants
        let bin_code = include_bytes!("../shader/binning.spv");
        let bin_pipeline = session.create_simple_compute_pipeline(bin_code, 2)?;
        let bin_ds = session.create_simple_descriptor_set(
            &bin_pipeline,
            &[&memory_buf_dev, &config_buf_dev],
        )?;

        let coarse_code = include_bytes!("../shader/coarse.spv");
        let coarse_pipeline = session.create_simple_compute_pipeline(coarse_code, 2)?;
        let coarse_ds = session.create_simple_descriptor_set(
            &coarse_pipeline,
            &[&memory_buf_dev, &config_buf_dev],
        )?;

        let bg_image = Self::make_test_bg_image(&session);

        let k4_code = include_bytes!("../shader/kernel4.spv");
        // This is an arbitrary limit on the number of textures that can be referenced by
        // the fine rasterizer. To set it for real, we probably want to pay attention both
        // to the device limit (maxDescriptorSetSampledImages) but also to the number of
        // images encoded (I believe there's an cost when allocating descriptor pools). If
        // it can't be satisfied, then for compatibility we'll probably want to fall back
        // to an atlasing approach.
        let max_textures = 256;
        let sampler = session.create_sampler(SamplerParams::Linear)?;
        let k4_pipeline = session
            .pipeline_builder()
            .add_buffers(2)
            .add_images(1)
            .add_textures(max_textures)
            .create_compute_pipeline(&session, k4_code)?;
        let k4_ds = session
            .descriptor_set_builder()
            .add_buffers(&[&memory_buf_dev, &config_buf_dev])
            .add_images(&[&image_dev])
            .add_textures(&[&bg_image], &sampler)
            .build(&session, &k4_pipeline)?;

        Ok(Renderer {
            scene_buf_host,
            scene_buf_dev,
            memory_buf_host,
            memory_buf_dev,
            state_buf,
            config_buf_host,
            config_buf_dev,
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
            n_elements,
            n_paths,
            n_pathseg,
            bg_image,
        })
    }

    pub unsafe fn record(&self, cmd_buf: &mut hub::CmdBuf, query_pool: &hub::QueryPool) {
        cmd_buf.copy_buffer(self.scene_buf_host.vk_buffer(), self.scene_buf_dev.vk_buffer());
        cmd_buf.copy_buffer(
            self.config_buf_host.vk_buffer(),
            self.config_buf_dev.vk_buffer(),
        );
        cmd_buf.copy_buffer(
            self.memory_buf_host.vk_buffer(),
            self.memory_buf_dev.vk_buffer(),
        );
        cmd_buf.clear_buffer(self.state_buf.vk_buffer(), None);
        cmd_buf.memory_barrier();
        cmd_buf.image_barrier(
            self.image_dev.vk_image(),
            ImageLayout::Undefined,
            ImageLayout::General,
        );
        cmd_buf.reset_query_pool(&query_pool);
        cmd_buf.write_timestamp(&query_pool, 0);
        cmd_buf.dispatch(
            &self.el_pipeline,
            &self.el_ds,
            (((self.n_elements + 127) / 128) as u32, 1, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 1);
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(
            &self.tile_pipeline,
            &self.tile_ds,
            (((self.n_paths + 255) / 256) as u32, 1, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 2);
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(
            &self.path_pipeline,
            &self.path_ds,
            (((self.n_pathseg + 31) / 32) as u32, 1, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 3);
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(
            &self.backdrop_pipeline,
            &self.backdrop_ds,
            (((self.n_paths + 255) / 256) as u32, 1, 1),
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
        );
        cmd_buf.write_timestamp(&query_pool, 5);
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(
            &self.coarse_pipeline,
            &self.coarse_ds,
            ((WIDTH as u32 + 255) / 256, (HEIGHT as u32 + 255) / 256, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 6);
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(
            &self.k4_pipeline,
            &self.k4_ds,
            ((WIDTH / TILE_W) as u32, (HEIGHT / TILE_H) as u32, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 7);
        cmd_buf.memory_barrier();
        cmd_buf.image_barrier(
            self.image_dev.vk_image(),
            ImageLayout::General,
            ImageLayout::BlitSrc,
        );
    }

    pub fn make_image(
        session: &hub::Session,
        width: usize,
        height: usize,
        buf: &[u8],
        format: ImageFormat,
    ) -> Result<hub::Image, Error> {
        unsafe {
            if format != ImageFormat::RgbaPremul {
                return Err("unsupported image format".into());
            }
            let host_mem_flags = MemFlags::host_coherent();
            let dev_mem_flags = MemFlags::device_local();
            let mut buffer = session.create_buffer(buf.len() as u64, host_mem_flags)?;
            buffer.write(buf)?;
            let image =
                session.create_image2d(width.try_into()?, height.try_into()?, dev_mem_flags)?;
            let mut cmd_buf = session.cmd_buf()?;
            cmd_buf.begin();
            cmd_buf.image_barrier(
                image.vk_image(),
                ImageLayout::Undefined,
                ImageLayout::BlitDst,
            );
            cmd_buf.copy_buffer_to_image(buffer.vk_buffer(), image.vk_image());
            cmd_buf.image_barrier(image.vk_image(), ImageLayout::BlitDst, ImageLayout::ShaderRead);
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
    fn make_test_bg_image(session: &hub::Session) -> hub::Image {
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
}
