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
    Error, ImageLayout, Instance, QueryPool, Semaphore, Session, SubmittedCmdBuf, Surface,
    Swapchain,
};

use piet_gpu::{render_scene, PietGpuRenderContext, Renderer};

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
    last_frame_idx: usize,
    submitted: Option<SubmittedCmdBuf>,
    query_pools: Vec<QueryPool>,
    present_semaphores: Vec<Semaphore>,
}

const WIDTH: usize = 1080;
const HEIGHT: usize = 2280;
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
                        let handle = get_handle(window);
                        let (instance, surface) = Instance::new(Some(&handle))?;
                        gfx_state = Some(GfxState::new(&instance, surface.as_ref())?);
                    } else {
                        println!("native window is sadly none");
                    }
                }
                Event::WindowRedrawNeeded => {
                    if let Some(gfx_state) = gfx_state.as_mut() {
                        for _ in 0..10 {
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
    fn new(instance: &Instance, surface: Option<&Surface>) -> Result<GfxState, Error> {
        unsafe {
            let device = instance.device(surface)?;
            let mut swapchain =
                instance.swapchain(WIDTH / 2, HEIGHT / 2, &device, surface.unwrap())?;
            let session = Session::new(device);
            let mut current_frame = 0;
            let present_semaphores = (0..NUM_FRAMES)
                .map(|_| session.create_semaphore())
                .collect::<Result<Vec<_>, Error>>()?;
            let query_pools = (0..NUM_FRAMES)
                .map(|_| session.create_query_pool(8))
                .collect::<Result<Vec<_>, Error>>()?;

            let mut ctx = PietGpuRenderContext::new();
            render_scene(&mut ctx);
            let n_paths = ctx.path_count();
            let n_pathseg = ctx.pathseg_count();
            let n_trans = ctx.pathseg_count();
            let scene = ctx.get_scene_buf();

            let renderer = Renderer::new(&session, scene, n_paths, n_pathseg, n_trans)?;

            let submitted: Option<SubmittedCmdBuf> = None;
            let current_frame = 0;
            let last_frame_idx = 0;
            Ok(GfxState {
                session,
                renderer,
                swapchain,
                current_frame,
                last_frame_idx,
                submitted,
                query_pools,
                present_semaphores,
            })
        }
    }

    fn redraw(&mut self) {
        println!("redraw");
        unsafe {
            if let Some(submitted) = self.submitted.take() {
                submitted.wait().unwrap();

                let ts = self
                    .session
                    .fetch_query_pool(&self.query_pools[self.last_frame_idx])
                    .unwrap();
                println!("render time: {:?}", ts);
            }
            let frame_idx = self.current_frame % NUM_FRAMES;
            let (image_idx, acquisition_semaphore) = self.swapchain.next().unwrap();
            let swap_image = self.swapchain.image(image_idx);
            let query_pool = &self.query_pools[frame_idx];
            let mut cmd_buf = self.session.cmd_buf().unwrap();
            cmd_buf.begin();
            self.renderer.record(&mut cmd_buf, &query_pool);

            // Image -> Swapchain
            cmd_buf.image_barrier(&swap_image, ImageLayout::Undefined, ImageLayout::BlitDst);
            cmd_buf.blit_image(&self.renderer.image_dev, &swap_image);
            cmd_buf.image_barrier(&swap_image, ImageLayout::BlitDst, ImageLayout::Present);
            cmd_buf.finish();

            self.submitted = Some(
                self.session
                    .run_cmd_buf(
                        cmd_buf,
                        &[&acquisition_semaphore],
                        &[&self.present_semaphores[frame_idx]],
                    )
                    .unwrap(),
            );
            self.last_frame_idx = frame_idx;

            self.swapchain
                .present(image_idx, &[&self.present_semaphores[frame_idx]])
                .unwrap();

            self.current_frame += 1;
        }
    }
}
