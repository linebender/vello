mod pico_svg;
mod render_ctx;

pub use render_ctx::PietGpuRenderContext;

use rand::{Rng, RngCore};

use piet::kurbo::{BezPath, Circle, Line, Point, Vec2};
use piet::{Color, RenderContext};

use piet_gpu_types::encoder::Encode;

use piet_gpu_hal::{CmdBuf, Device, Error, ImageLayout, MemFlags};

use pico_svg::PicoSvg;

pub const WIDTH: usize = TILE_W * WIDTH_IN_TILES;
pub const HEIGHT: usize = TILE_H * HEIGHT_IN_TILES;

const TILE_W: usize = 16;
const TILE_H: usize = 16;

const WIDTH_IN_TILEGROUPS: usize = 4;
const HEIGHT_IN_TILEGROUPS: usize = 96;
const TILEGROUP_STRIDE: usize = 2048;

const WIDTH_IN_TILES: usize = 128;
const HEIGHT_IN_TILES: usize = 96;
const PTCL_INITIAL_ALLOC: usize = 1024;

const K2_PER_TILE_SIZE: usize = 8;

const N_CIRCLES: usize = 0;

const N_WG: u32 = 16;

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
    /*
    let mut path = BezPath::new();
    path.move_to((100.0, 1150.0));
    path.line_to((200.0, 1200.0));
    path.line_to((150.0, 1250.0));
    path.close_path();
    rc.fill(path, &Color::rgb8(128, 0, 128));
    */
    rc.stroke(
        Line::new((100.0, 100.0), (200.0, 150.0)),
        &Color::WHITE,
        5.0,
    );
    //render_cardioid(rc);
    render_tiger(rc);
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

pub struct Renderer<D: Device> {
    pub image_dev: D::Image, // resulting image

    scene_buf: D::Buffer,
    scene_dev: D::Buffer,

    pub state_buf: D::Buffer,
    pub anno_buf: D::Buffer,
    pub pathseg_buf: D::Buffer,
    pub tile_buf: D::Buffer,
    pub bin_buf: D::Buffer,
    pub ptcl_buf: D::Buffer,

    el_pipeline: D::Pipeline,
    el_ds: D::DescriptorSet,

    tile_pipeline: D::Pipeline,
    tile_ds: D::DescriptorSet,

    path_pipeline: D::Pipeline,
    path_ds: D::DescriptorSet,

    tile_alloc_buf_host: D::Buffer,
    tile_alloc_buf_dev: D::Buffer,

    bin_pipeline: D::Pipeline,
    bin_ds: D::DescriptorSet,

    bin_alloc_buf_host: D::Buffer,
    bin_alloc_buf_dev: D::Buffer,

    coarse_pipeline: D::Pipeline,
    coarse_ds: D::DescriptorSet,

    coarse_alloc_buf_host: D::Buffer,
    coarse_alloc_buf_dev: D::Buffer,

    k4_pipeline: D::Pipeline,
    k4_ds: D::DescriptorSet,

    n_elements: usize,
    n_paths: usize,
    n_pathseg: usize,
}

impl<D: Device> Renderer<D> {
    pub unsafe fn new(device: &D, scene: &[u8], n_paths: usize, n_pathseg: usize) -> Result<Self, Error> {
        let host = MemFlags::host_coherent();
        let dev = MemFlags::device_local();

        let n_elements = scene.len() / piet_gpu_types::scene::Element::fixed_size();
        println!("scene: {} elements", n_elements);

        let scene_buf = device
            .create_buffer(std::mem::size_of_val(&scene[..]) as u64, host)
            .unwrap();
        let scene_dev = device
            .create_buffer(std::mem::size_of_val(&scene[..]) as u64, dev)
            .unwrap();
        device.write_buffer(&scene_buf, &scene)?;

        let state_buf = device.create_buffer(1 * 1024 * 1024, dev)?;
        let anno_buf = device.create_buffer(64 * 1024 * 1024, dev)?;
        let pathseg_buf = device.create_buffer(64 * 1024 * 1024, dev)?;
        let tile_buf = device.create_buffer(64 * 1024 * 1024, dev)?;
        let bin_buf = device.create_buffer(64 * 1024 * 1024, dev)?;
        let ptcl_buf = device.create_buffer(48 * 1024 * 1024, dev)?;
        let image_dev = device.create_image2d(WIDTH as u32, HEIGHT as u32, dev)?;

        let el_code = include_bytes!("../shader/elements.spv");
        let el_pipeline = device.create_simple_compute_pipeline(el_code, 4, 0)?;
        let el_ds = device.create_descriptor_set(
            &el_pipeline,
            &[&scene_dev, &state_buf, &anno_buf, &pathseg_buf],
            &[],
        )?;

        let tile_alloc_buf_host = device.create_buffer(12, host)?;
        let tile_alloc_buf_dev = device.create_buffer(12, dev)?;

        // TODO: constants
        const PATH_SIZE: usize = 12;
        let tile_alloc_start = ((n_paths + 31) & !31) * PATH_SIZE;
        device.write_buffer(
            &tile_alloc_buf_host,
            &[n_paths as u32, n_pathseg as u32, tile_alloc_start as u32],
        )?;
        let tile_alloc_code = include_bytes!("../shader/tile_alloc.spv");
        let tile_pipeline = device.create_simple_compute_pipeline(tile_alloc_code, 3, 0)?;
        let tile_ds = device.create_descriptor_set(
            &tile_pipeline,
            &[&anno_buf, &tile_alloc_buf_dev, &tile_buf],
            &[],
        )?;

        let path_alloc_code = include_bytes!("../shader/path_coarse.spv");
        let path_pipeline = device.create_simple_compute_pipeline(path_alloc_code, 3, 0)?;
        let path_ds = device.create_descriptor_set(
            &path_pipeline,
            &[&pathseg_buf, &tile_alloc_buf_dev, &tile_buf],
            &[],
        )?;

        let bin_alloc_buf_host = device.create_buffer(12, host)?;
        let bin_alloc_buf_dev = device.create_buffer(12, dev)?;

        // TODO: constants
        let bin_alloc_start = ((n_paths + 255) & !255) * 8;
        device.write_buffer(
            &bin_alloc_buf_host,
            &[n_paths as u32, 0, bin_alloc_start as u32],
        )?;
        let bin_code = include_bytes!("../shader/binning.spv");
        let bin_pipeline = device.create_simple_compute_pipeline(bin_code, 4, 0)?;
        let bin_ds = device.create_descriptor_set(
            &bin_pipeline,
            &[&anno_buf, &state_buf, &bin_alloc_buf_dev, &bin_buf],
            &[],
        )?;

        let coarse_alloc_buf_host = device.create_buffer(8, host)?;
        let coarse_alloc_buf_dev = device.create_buffer(8, dev)?;

        let coarse_alloc_start = WIDTH_IN_TILES * HEIGHT_IN_TILES * PTCL_INITIAL_ALLOC;
        device.write_buffer(
            &coarse_alloc_buf_host,
            &[n_elements as u32, coarse_alloc_start as u32],
        )?;
        let coarse_code = include_bytes!("../shader/coarse.spv");
        let coarse_pipeline = device.create_simple_compute_pipeline(coarse_code, 5, 0)?;
        let coarse_ds = device.create_descriptor_set(
            &coarse_pipeline,
            &[&anno_buf, &bin_buf, &tile_buf, &coarse_alloc_buf_dev, &ptcl_buf],
            &[],
        )?;

        let k4_code = include_bytes!("../shader/kernel4.spv");
        let k4_pipeline = device.create_simple_compute_pipeline(k4_code, 2, 1)?;
        let k4_ds = device.create_descriptor_set(
            &k4_pipeline, 
            &[&ptcl_buf, &tile_buf], 
            &[&image_dev]
        )?;

        Ok(Renderer {
            scene_buf,
            scene_dev,
            image_dev,
            el_pipeline,
            el_ds,
            tile_pipeline,
            tile_ds,
            path_pipeline,
            path_ds,
            bin_pipeline,
            bin_ds,
            coarse_pipeline,
            coarse_ds,
            k4_pipeline,
            k4_ds,
            state_buf,
            anno_buf,
            pathseg_buf,
            tile_buf,
            bin_buf,
            ptcl_buf,
            tile_alloc_buf_host,
            tile_alloc_buf_dev,
            bin_alloc_buf_host,
            bin_alloc_buf_dev,
            coarse_alloc_buf_host,
            coarse_alloc_buf_dev,
            n_elements,
            n_paths,
            n_pathseg,
        })
    }

    pub unsafe fn record(&self, cmd_buf: &mut impl CmdBuf<D>, query_pool: &D::QueryPool) {
        cmd_buf.copy_buffer(&self.scene_buf, &self.scene_dev);
        cmd_buf.copy_buffer(&self.tile_alloc_buf_host, &self.tile_alloc_buf_dev);
        cmd_buf.copy_buffer(&self.bin_alloc_buf_host, &self.bin_alloc_buf_dev);
        cmd_buf.copy_buffer(&self.coarse_alloc_buf_host, &self.coarse_alloc_buf_dev);
        cmd_buf.clear_buffer(&self.state_buf);
        cmd_buf.memory_barrier();
        cmd_buf.image_barrier(
            &self.image_dev,
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
            (((self.n_paths + 31) / 32) as u32, 1, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 2);
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(
            &self.path_pipeline,
            &self.path_ds,
            (((self.n_pathseg + 31) / 32) as u32, 1, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 3);
        // Note: this barrier is not needed as an actual dependency between
        // pipeline stages, but I am keeping it in so that timer queries are
        // easier to interpret.
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(
            &self.bin_pipeline,
            &self.bin_ds,
            (((self.n_paths + 255) / 256) as u32, 1, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 4);
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(
            &self.coarse_pipeline,
            &self.coarse_ds,
            (WIDTH as u32 / 256, HEIGHT as u32 / 256, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 5);
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(
            &self.k4_pipeline,
            &self.k4_ds,
            ((WIDTH / TILE_W) as u32, (HEIGHT / TILE_H) as u32, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 6);
        cmd_buf.memory_barrier();
        cmd_buf.image_barrier(&self.image_dev, ImageLayout::General, ImageLayout::BlitSrc);
    }
}
