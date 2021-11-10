//! Android example
//!
//! Run using `cargo apk run --example android`
//!
//! Requires the [cargo-apk] tool.
//! [cargo-apk]: https://crates.io/crates/cargo-apk

use raw_window_handle::android::AndroidHandle;
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};

use ndk::native_window::NativeWindow;
use ndk_glue::Event;

use piet_gpu_hal::{
    CmdBuf, Error, ImageLayout, Instance, QueryPool, Semaphore, Session, SubmittedCmdBuf, Surface,
    Swapchain,
};

use piet::kurbo::Point;
use piet::{RenderContext, Text, TextAttribute, TextLayoutBuilder};

use piet_gpu::{test_scenes, PietGpuRenderContext, Renderer};

#[cfg_attr(target_os = "android", ndk_glue::main(backtrace = "on"))]
fn main() {
    my_main().unwrap();
}

struct MyHandle {
    handle: AndroidHandle,
}

// State required to render and present the contents
struct GfxState {
    session: Session,
    renderer: Renderer,
    swapchain: Swapchain,
    current_frame: usize,
    submitted: [Option<SubmittedCmdBuf>; NUM_FRAMES],
    cmd_bufs: [Option<CmdBuf>; NUM_FRAMES],
    query_pools: Vec<QueryPool>,
    present_semaphores: Vec<Semaphore>,
}

const NUM_FRAMES: usize = 2;

fn my_main() -> Result<(), Error> {
    let mut gfx_state = None;
    loop {
        for event in ndk_glue::poll_events() {
            println!("got event {:?}", event);
            match event {
                Event::WindowCreated => {
                    let window = ndk_glue::native_window();
                    if let Some(window) = &*window {
                        let width = window.width() as usize;
                        let height = window.height() as usize;
                        let handle = get_handle(window);
                        let (instance, surface) = Instance::new(Some(&handle), Default::default())?;
                        gfx_state =
                            Some(GfxState::new(&instance, surface.as_ref(), width, height)?);
                    } else {
                        println!("native window is sadly none");
                    }
                }
                Event::WindowRedrawNeeded => {
                    if let Some(gfx_state) = gfx_state.as_mut() {
                        for _ in 0..1000 {
                            gfx_state.redraw();
                        }
                    }
                }
                _ => (),
            }
        }
    }
}

fn get_handle(window: &NativeWindow) -> MyHandle {
    println!(
        "window = {:?}, {}x{}",
        window.ptr(),
        window.width(),
        window.height()
    );
    let mut handle = AndroidHandle::empty();
    handle.a_native_window = window.ptr().as_ptr() as *mut std::ffi::c_void;
    MyHandle { handle }
}

unsafe impl HasRawWindowHandle for MyHandle {
    fn raw_window_handle(&self) -> RawWindowHandle {
        RawWindowHandle::Android(self.handle)
    }
}

impl GfxState {
    fn new(
        instance: &Instance,
        surface: Option<&Surface>,
        width: usize,
        height: usize,
    ) -> Result<GfxState, Error> {
        unsafe {
            let device = instance.device(surface)?;
            let swapchain = instance.swapchain(width, height, &device, surface.unwrap())?;
            let session = Session::new(device);
            let current_frame = 0;
            let present_semaphores = (0..NUM_FRAMES)
                .map(|_| session.create_semaphore())
                .collect::<Result<Vec<_>, Error>>()?;
            let query_pools = (0..NUM_FRAMES)
                .map(|_| session.create_query_pool(8))
                .collect::<Result<Vec<_>, Error>>()?;
            let submitted = Default::default();
            let cmd_bufs = Default::default();

            let renderer = Renderer::new(&session, width, height, NUM_FRAMES)?;

            Ok(GfxState {
                session,
                renderer,
                swapchain,
                current_frame,
                submitted,
                cmd_bufs,
                query_pools,
                present_semaphores,
            })
        }
    }

    fn redraw(&mut self) {
        println!("redraw");
        unsafe {
            let frame_idx = self.current_frame % NUM_FRAMES;
            let mut info_string = String::new();

            if let Some(submitted) = self.submitted[frame_idx].take() {
                self.cmd_bufs[frame_idx] = submitted.wait().unwrap();
                let ts = self
                    .session
                    .fetch_query_pool(&self.query_pools[frame_idx])
                    .unwrap();
                info_string = format!("{:.1}ms", ts.last().unwrap() * 1e3);
                println!("render time: {:?}", ts);
            }
            let mut ctx = PietGpuRenderContext::new();
            test_scenes::render_anim_frame(&mut ctx, self.current_frame);
            //test_scenes::render_tiger(&mut ctx);
            render_info_string(&mut ctx, &info_string);
            if let Err(e) = self.renderer.upload_render_ctx(&mut ctx, frame_idx) {
                println!("error in uploading: {}", e);
            }
            let (image_idx, acquisition_semaphore) = self.swapchain.next().unwrap();
            let swap_image = self.swapchain.image(image_idx);
            let query_pool = &self.query_pools[frame_idx];
            let mut cmd_buf = self.cmd_bufs[frame_idx]
                .take()
                .unwrap_or_else(|| self.session.cmd_buf().unwrap());
            cmd_buf.begin();
            self.renderer.record(&mut cmd_buf, &query_pool, frame_idx);

            // Image -> Swapchain
            cmd_buf.image_barrier(&swap_image, ImageLayout::Undefined, ImageLayout::BlitDst);
            cmd_buf.blit_image(&self.renderer.image_dev, &swap_image);
            cmd_buf.image_barrier(&swap_image, ImageLayout::BlitDst, ImageLayout::Present);
            cmd_buf.finish();

            self.submitted[frame_idx] = Some(
                self.session
                    .run_cmd_buf(
                        cmd_buf,
                        &[&acquisition_semaphore],
                        &[&self.present_semaphores[frame_idx]],
                    )
                    .unwrap(),
            );

            self.swapchain
                .present(image_idx, &[&self.present_semaphores[frame_idx]])
                .unwrap();

            self.current_frame += 1;
        }
    }
}

fn render_info_string(rc: &mut impl RenderContext, info: &str) {
    let layout = rc
        .text()
        .new_text_layout(info.to_string())
        .default_attribute(TextAttribute::FontSize(60.0))
        .build()
        .unwrap();
    rc.draw_text(&layout, Point::new(110.0, 120.0));
}
