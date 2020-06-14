use piet_gpu_hal::vulkan::VkInstance;
use piet_gpu_hal::{CmdBuf, Device, Error, ImageLayout};

use piet_gpu::{render_scene, PietGpuRenderContext, Renderer, HEIGHT, WIDTH};

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const NUM_FRAMES: usize = 2;

fn main() -> Result<(), Error> {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(winit::dpi::LogicalSize {
            width: (WIDTH / 2) as f64,
            height: (HEIGHT / 2) as f64,
        })
        .with_resizable(false) // currently not supported
        .build(&event_loop)?;

    let (instance, surface) = VkInstance::new(Some(&window))?;
    unsafe {
        let device = instance.device(surface.as_ref())?;
        let mut swapchain = instance.swapchain(&device, surface.as_ref().unwrap())?;

        let mut current_frame = 0;
        let present_semaphores = (0..NUM_FRAMES)
            .map(|_| device.create_semaphore())
            .collect::<Result<Vec<_>, Error>>()?;
        let frame_fences = (0..NUM_FRAMES)
            .map(|_| device.create_fence(false))
            .collect::<Result<Vec<_>, Error>>()?;
        let mut cmd_buffers = (0..NUM_FRAMES)
            .map(|_| device.create_cmd_buf())
            .collect::<Result<Vec<_>, Error>>()?;
        let query_pools = (0..NUM_FRAMES)
            .map(|_| device.create_query_pool(8))
            .collect::<Result<Vec<_>, Error>>()?;

        let mut ctx = PietGpuRenderContext::new();
        render_scene(&mut ctx);
        let n_paths = ctx.path_count();
        let n_pathseg = ctx.pathseg_count();
        let scene = ctx.get_scene_buf();

        let renderer = Renderer::new(&device, scene, n_paths, n_pathseg)?;

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll; // `ControlFlow::Wait` if only re-render on event

            match event {
                Event::WindowEvent { event, window_id } if window_id == window.id() => {
                    match event {
                        WindowEvent::CloseRequested => {
                            *control_flow = ControlFlow::Exit;
                        }
                        _ => (),
                    }
                }
                Event::MainEventsCleared => {
                    window.request_redraw();
                }
                Event::RedrawRequested(window_id) if window_id == window.id() => {
                    let frame_idx = current_frame % NUM_FRAMES;
                    let query_pool = &query_pools[frame_idx];

                    if current_frame >= NUM_FRAMES {
                        device.wait_and_reset(&[frame_fences[frame_idx]]).unwrap();

                        let ts = device.reap_query_pool(query_pool).unwrap();
                        window.set_title(&format!(
                            "{:.3}ms :: e:{:.3}ms|alloc:{:.3}ms|cp:{:.3}ms|bd:{:.3}ms|bin:{:.3}ms|cr:{:.3}ms|r:{:.3}ms",
                            ts[6] * 1e3,
                            ts[0] * 1e3,
                            (ts[1] - ts[0]) * 1e3,
                            (ts[2] - ts[1]) * 1e3,
                            (ts[3] - ts[2]) * 1e3,
                            (ts[4] - ts[3]) * 1e3,
                            (ts[5] - ts[4]) * 1e3,
                            (ts[6] - ts[5]) * 1e3,
                        ));
                    }

                    let (image_idx, acquisition_semaphore) = swapchain.next().unwrap();
                    let swap_image = swapchain.image(image_idx);
                    let cmd_buf = &mut cmd_buffers[frame_idx];
                    cmd_buf.begin();
                    renderer.record(cmd_buf, &query_pool);

                    // Image -> Swapchain
                    cmd_buf.image_barrier(
                        &swap_image,
                        ImageLayout::Undefined,
                        ImageLayout::BlitDst,
                    );
                    cmd_buf.blit_image(&renderer.image_dev, &swap_image);
                    cmd_buf.image_barrier(&swap_image, ImageLayout::BlitDst, ImageLayout::Present);
                    cmd_buf.finish();

                    device
                        .run_cmd_buf(
                            &cmd_buf,
                            &[acquisition_semaphore],
                            &[present_semaphores[frame_idx]],
                            Some(&frame_fences[frame_idx]),
                        )
                        .unwrap();

                    swapchain
                        .present(image_idx, &[present_semaphores[frame_idx]])
                        .unwrap();

                    current_frame += 1;
                }
                _ => (),
            }
        })
    }
}
