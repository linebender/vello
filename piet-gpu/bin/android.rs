#![cfg(target_os = "android")]
//! Android example
//!
//! Run using `cargo apk run --example android`
//!
//! Requires the [cargo-apk] tool.
//! [cargo-apk]: https://crates.io/crates/cargo-apk

use raw_window_handle::{
    AndroidDisplayHandle, AndroidNdkWindowHandle, RawDisplayHandle, RawWindowHandle,
};

use ndk_glue::Event;

use piet_gpu_hal::{
    Error, ImageLayout, Instance, InstanceFlags, Semaphore, Session, Surface, Swapchain,
};

use piet_gpu::{samples, RenderDriver, Renderer, SimpleText};
use piet_scene::{ResourceContext, Scene, SceneBuilder};

#[cfg_attr(target_os = "android", ndk_glue::main(backtrace = "on"))]
fn main() {
    my_main().unwrap();
}

// State required to render and present the contents
struct GfxState {
    session: Session,
    render_driver: RenderDriver,
    swapchain: Swapchain,
    current_frame: usize,
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
                        let instance = Instance::new(InstanceFlags::default())?;
                        let mut android_handle = AndroidNdkWindowHandle::empty();
                        android_handle.a_native_window = window.ptr().as_ptr() as *mut _;
                        let window_handle = RawWindowHandle::AndroidNdk(android_handle);
                        let display_handle =
                            RawDisplayHandle::Android(AndroidDisplayHandle::empty());
                        let surface = unsafe { instance.surface(display_handle, window_handle)? };
                        gfx_state = Some(GfxState::new(&instance, Some(&surface), width, height)?);
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

impl GfxState {
    fn new(
        instance: &Instance,
        surface: Option<&Surface>,
        width: usize,
        height: usize,
    ) -> Result<GfxState, Error> {
        unsafe {
            let device = instance.device()?;
            let swapchain = instance.swapchain(width, height, &device, surface.unwrap())?;
            let session = Session::new(device);
            let current_frame = 0;
            let present_semaphores = (0..NUM_FRAMES)
                .map(|_| session.create_semaphore())
                .collect::<Result<Vec<_>, Error>>()?;

            let renderer = Renderer::new(&session, width, height, NUM_FRAMES)?;
            let render_driver = RenderDriver::new(&session, NUM_FRAMES, renderer);

            Ok(GfxState {
                session,
                render_driver,
                swapchain,
                current_frame,
                present_semaphores,
            })
        }
    }

    fn redraw(&mut self) {
        println!("redraw");
        unsafe {
            let frame_idx = self.current_frame % NUM_FRAMES;
            let mut info_string = String::new();

            if self.current_frame >= NUM_FRAMES {
                let stats = self
                    .render_driver
                    .get_timing_stats(&self.session, frame_idx);
                info_string = stats.short_summary();
                println!("{}", info_string);
            }
            let mut text = SimpleText::new();
            let mut scene = Scene::default();
            let mut rcx = ResourceContext::default();
            let mut builder = SceneBuilder::for_scene(&mut scene, &mut rcx);
            samples::render_anim_frame(&mut builder, self.current_frame);
            //samples::render_tiger(&mut builder, false);
            render_info(&mut text, &mut builder, &info_string);
            builder.finish();
            if let Err(e) = self.render_driver.upload_scene(&self.session, &scene, &rcx) {
                println!("error in uploading: {}", e);
            }
            let (image_idx, acquisition_semaphore) = self.swapchain.next().unwrap();
            let swap_image = self.swapchain.image(image_idx);
            self.render_driver.run_coarse(&self.session).unwrap();
            let target = self.render_driver.record_fine(&self.session).unwrap();
            let cmd_buf = target.cmd_buf;

            // Image -> Swapchain
            cmd_buf.image_barrier(&swap_image, ImageLayout::Undefined, ImageLayout::BlitDst);
            cmd_buf.blit_image(target.image, &swap_image);
            cmd_buf.image_barrier(&swap_image, ImageLayout::BlitDst, ImageLayout::Present);

            self.render_driver
                .submit(
                    &self.session,
                    &[&acquisition_semaphore],
                    &[&self.present_semaphores[frame_idx]],
                )
                .unwrap();

            self.swapchain
                .present(image_idx, &[&self.present_semaphores[frame_idx]])
                .unwrap();

            self.render_driver.next_buffer();
            self.current_frame += 1;
        }
    }
}

fn render_info(simple_text: &mut SimpleText, sb: &mut SceneBuilder, info: &str) {
    simple_text.add(
        sb,
        None,
        60.0,
        None,
        piet_scene::Affine::translate(110.0, 120.0),
        info,
    );
}
