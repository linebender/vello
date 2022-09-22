use piet::kurbo::Point;
use piet::{RenderContext, Text, TextAttribute, TextLayoutBuilder};
use piet_gpu_hal::{Error, ImageLayout, Instance, InstanceFlags, Session};

use piet_gpu::{test_scenes, PicoSvg, PietGpuRenderContext, RenderDriver, Renderer};

use clap::{App, Arg};

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
    unsafe {
        let surface = instance.surface(&window)?;
        let device = instance.device()?;
        let mut swapchain = instance.swapchain(WIDTH / 2, HEIGHT / 2, &device, &surface)?;
        let session = Session::new(device);

        let mut current_frame = 0;
        let present_semaphores = (0..NUM_FRAMES)
            .map(|_| session.create_semaphore())
            .collect::<Result<Vec<_>, Error>>()?;

        let renderer = Renderer::new(&session, WIDTH, HEIGHT, NUM_FRAMES)?;
        let mut render_driver = RenderDriver::new(&session, NUM_FRAMES, renderer);
        let mut mode = 0usize;

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
                                    Some(VirtualKeyCode::Left) => mode = mode.wrapping_sub(1),
                                    Some(VirtualKeyCode::Right) => mode = mode.wrapping_add(1),
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

                    let mut ctx = PietGpuRenderContext::new();
                    let test_blend = false;
                    if let Some(svg) = &svg {
                        test_scenes::render_svg(&mut ctx, svg);
                    } else if test_blend {
                        use piet_gpu::{Blend, BlendMode::*, CompositionMode::*};
                        let blends = [
                            Blend::new(Normal, SrcOver),
                            Blend::new(Multiply, SrcOver),
                            Blend::new(Screen, SrcOver),
                            Blend::new(Overlay, SrcOver),
                            Blend::new(Darken, SrcOver),
                            Blend::new(Lighten, SrcOver),
                            Blend::new(ColorDodge, SrcOver),
                            Blend::new(ColorBurn, SrcOver),
                            Blend::new(HardLight, SrcOver),
                            Blend::new(SoftLight, SrcOver),
                            Blend::new(Difference, SrcOver),
                            Blend::new(Exclusion, SrcOver),
                            Blend::new(Hue, SrcOver),
                            Blend::new(Saturation, SrcOver),
                            Blend::new(Color, SrcOver),
                            Blend::new(Luminosity, SrcOver),
                            Blend::new(Normal, Clear),
                            Blend::new(Normal, Copy),
                            Blend::new(Normal, Dest),
                            Blend::new(Normal, SrcOver),
                            Blend::new(Normal, DestOver),
                            Blend::new(Normal, SrcIn),
                            Blend::new(Normal, DestIn),
                            Blend::new(Normal, SrcOut),
                            Blend::new(Normal, DestOut),
                            Blend::new(Normal, SrcAtop),
                            Blend::new(Normal, DestAtop),
                            Blend::new(Normal, Xor),
                            Blend::new(Normal, Plus),
                        ];
                        let blend = blends[mode % blends.len()];
                        test_scenes::render_blend_test(&mut ctx, current_frame, blend);
                        info_string = format!("{:?}", blend);
                    } else {
                        test_scenes::render_anim_frame(&mut ctx, current_frame);
                    }
                    render_info_string(&mut ctx, &info_string);
                    if let Err(e) = render_driver.upload_render_ctx(&session, &mut ctx) {
                        println!("error in uploading: {}", e);
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

fn render_info_string(rc: &mut impl RenderContext, info: &str) {
    let layout = rc
        .text()
        .new_text_layout(info.to_string())
        .default_attribute(TextAttribute::FontSize(40.0))
        .build()
        .unwrap();
    rc.draw_text(&layout, Point::new(110.0, 50.0));
}
