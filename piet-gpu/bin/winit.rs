use piet_gpu::{samples, PicoSvg, RenderDriver, Renderer, SimpleText};
use piet_gpu_hal::{Error, ImageLayout, Instance, InstanceFlags, Session};
use piet_scene::{Scene, SceneBuilder};

use clap::{App, Arg};

use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const NUM_FRAMES: usize = 2;

const WIDTH: usize = 2048;
const HEIGHT: usize = 1536;

fn main() -> Result<(), Error> {
    let matches = App::new("piet-gpu test")
        .arg(Arg::with_name("INPUT").index(1))
        .arg(Arg::with_name("flip").short('f').long("flip"))
        .arg(
            Arg::with_name("scale")
                .short('s')
                .long("scale")
                .takes_value(true),
        )
        .get_matches();

    // Collect SVG if input
    let svg = match matches.value_of("INPUT") {
        Some(file) => {
            let mut scale = matches
                .value_of("scale")
                .map(|scale| scale.parse().unwrap())
                .unwrap_or(8.0);
            if matches.is_present("flip") {
                scale = -scale;
            }
            let xml_str = std::fs::read_to_string(file).unwrap();
            let start = std::time::Instant::now();
            let svg = PicoSvg::load(&xml_str, scale).unwrap();
            println!("parsing time: {:?}", start.elapsed());
            Some(svg)
        }
        None => None,
    };

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(winit::dpi::LogicalSize {
            width: (WIDTH / 2) as f64,
            height: (HEIGHT / 2) as f64,
        })
        .with_resizable(false) // currently not supported
        .build(&event_loop)?;

    let instance = Instance::new(InstanceFlags::default())?;
    let mut info_string = "info".to_string();
    let mut scene = Scene::default();
    let mut simple_text = piet_gpu::SimpleText::new();
    unsafe {
        let display_handle = window.raw_display_handle();
        let window_handle = window.raw_window_handle();
        let surface = instance.surface(display_handle, window_handle)?;
        let device = instance.device()?;
        let mut swapchain = instance.swapchain(WIDTH / 2, HEIGHT / 2, &device, &surface)?;
        let session = Session::new(device);

        let mut current_frame = 0;
        let present_semaphores = (0..NUM_FRAMES)
            .map(|_| session.create_semaphore())
            .collect::<Result<Vec<_>, Error>>()?;

        let renderer = Renderer::new(&session, WIDTH, HEIGHT, NUM_FRAMES)?;
        let mut render_driver = RenderDriver::new(&session, NUM_FRAMES, renderer);
        let mut sample_index = 0usize;

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll; // `ControlFlow::Wait` if only re-render on event

            match event {
                Event::WindowEvent { event, window_id } if window_id == window.id() => {
                    use winit::event::{ElementState, VirtualKeyCode};
                    match event {
                        WindowEvent::CloseRequested => {
                            *control_flow = ControlFlow::Exit;
                        }
                        WindowEvent::KeyboardInput { input, .. } => {
                            if input.state == ElementState::Pressed {
                                match input.virtual_keycode {
                                    Some(VirtualKeyCode::Left) => {
                                        sample_index = sample_index.saturating_sub(1)
                                    }
                                    Some(VirtualKeyCode::Right) => {
                                        sample_index = sample_index.saturating_add(1)
                                    }
                                    _ => {}
                                }
                            }
                        }
                        _ => (),
                    }
                }
                Event::MainEventsCleared => {
                    window.request_redraw();
                }
                Event::RedrawRequested(window_id) if window_id == window.id() => {
                    let frame_idx = current_frame % NUM_FRAMES;

                    if current_frame >= NUM_FRAMES {
                        let stats = render_driver.get_timing_stats(&session, frame_idx);
                        info_string = stats.short_summary();
                    }

                    if let Some(svg) = &svg {
                        let mut builder = SceneBuilder::for_scene(&mut scene);
                        samples::render_svg(&mut builder, svg, false);
                        render_info(&mut simple_text, &mut builder, &info_string);
                        builder.finish();
                        if let Err(e) = render_driver.upload_scene(&session, &scene) {
                            println!("error in uploading: {}", e);
                        }
                    } else {
                        let mut builder = SceneBuilder::for_scene(&mut scene);

                        const N_SAMPLES: usize = 4;
                        match sample_index % N_SAMPLES {
                            0 => samples::render_anim_frame(
                                &mut builder,
                                &mut simple_text,
                                current_frame,
                            ),
                            1 => samples::render_blend_grid(&mut builder),
                            2 => samples::render_tiger(&mut builder, false),
                            _ => samples::render_scene(&mut builder),
                        }
                        render_info(&mut simple_text, &mut builder, &info_string);
                        builder.finish();
                        if let Err(e) = render_driver.upload_scene(&session, &scene) {
                            println!("error in uploading: {}", e);
                        }
                    }

                    let (image_idx, acquisition_semaphore) = swapchain.next().unwrap();
                    let swap_image = swapchain.image(image_idx);
                    render_driver.run_coarse(&session).unwrap();
                    let target = render_driver.record_fine(&session).unwrap();
                    let cmd_buf = target.cmd_buf;

                    // Image -> Swapchain
                    cmd_buf.image_barrier(
                        &swap_image,
                        ImageLayout::Undefined,
                        ImageLayout::BlitDst,
                    );
                    cmd_buf.blit_image(target.image, &swap_image);
                    cmd_buf.image_barrier(&swap_image, ImageLayout::BlitDst, ImageLayout::Present);
                    render_driver
                        .submit(
                            &session,
                            &[&acquisition_semaphore],
                            &[&present_semaphores[frame_idx]],
                        )
                        .unwrap();

                    swapchain
                        .present(image_idx, &[&present_semaphores[frame_idx]])
                        .unwrap();

                    render_driver.next_buffer();
                    current_frame += 1;
                }
                Event::LoopDestroyed => {
                    render_driver.wait_all(&session);
                }
                _ => (),
            }
        })
    }
}

fn render_info(simple_text: &mut SimpleText, sb: &mut SceneBuilder, info: &str) {
    simple_text.add(
        sb,
        None,
        40.0,
        None,
        piet_scene::Affine::translate(110.0, 50.0),
        info,
    );
}
