use piet::kurbo::Point;
use piet::{RenderContext, Text, TextAttribute, TextLayoutBuilder};
use piet_gpu_hal::{CmdBuf, Error, ImageLayout, Instance, Session, SubmittedCmdBuf};

use piet_gpu::{test_scenes, EncodedSceneRef, PietGpuRenderContext, Renderer};

use piet_scene::resource::ResourceContext;
use piet_scene::scene::{build_scene, Scene};

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
        .arg(Arg::with_name("flip").short("f").long("flip"))
        .arg(
            Arg::with_name("scale")
                .short("s")
                .long("scale")
                .takes_value(true),
        )
        .get_matches();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(winit::dpi::LogicalSize {
            width: (WIDTH / 2) as f64,
            height: (HEIGHT / 2) as f64,
        })
        .with_resizable(false) // currently not supported
        .build(&event_loop)?;

    let (instance, surface) = Instance::new(Some(&window), Default::default())?;
    let mut info_string = "info".to_string();

    let mut scene = Scene::default();
    let mut rcx = ResourceContext::new();
    unsafe {
        let device = instance.device(surface.as_ref())?;
        let mut swapchain =
            instance.swapchain(WIDTH / 2, HEIGHT / 2, &device, surface.as_ref().unwrap())?;
        let session = Session::new(device);

        let mut current_frame = 0;
        let present_semaphores = (0..NUM_FRAMES)
            .map(|_| session.create_semaphore())
            .collect::<Result<Vec<_>, Error>>()?;
        let query_pools = (0..NUM_FRAMES)
            .map(|_| session.create_query_pool(8))
            .collect::<Result<Vec<_>, Error>>()?;
        let mut cmd_bufs: [Option<CmdBuf>; NUM_FRAMES] = Default::default();
        let mut submitted: [Option<SubmittedCmdBuf>; NUM_FRAMES] = Default::default();

        let mut renderer = Renderer::new(&session, WIDTH, HEIGHT, NUM_FRAMES)?;
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

                    if let Some(submitted) = submitted[frame_idx].take() {
                        cmd_bufs[frame_idx] = submitted.wait().unwrap();
                        let ts = session.fetch_query_pool(&query_pools[frame_idx]).unwrap();
                        if !ts.is_empty() {
                            info_string = format!(
                                "{:.3}ms :: e:{:.3}ms|alloc:{:.3}ms|cp:{:.3}ms|bd:{:.3}ms|bin:{:.3}ms|cr:{:.3}ms|r:{:.3}ms",
                                ts[6] * 1e3,
                                ts[0] * 1e3,
                                (ts[1] - ts[0]) * 1e3,
                                (ts[2] - ts[1]) * 1e3,
                                (ts[3] - ts[2]) * 1e3,
                                (ts[4] - ts[3]) * 1e3,
                                (ts[5] - ts[4]) * 1e3,
                                (ts[6] - ts[5]) * 1e3,
                            );
                        }
                    }

                    let mut ctx = PietGpuRenderContext::new();
                    if let Some(input) = matches.value_of("INPUT") {
                        let mut scale = matches
                            .value_of("scale")
                            .map(|scale| scale.parse().unwrap())
                            .unwrap_or(8.0);
                        if matches.is_present("flip") {
                            scale = -scale;
                        }
                        test_scenes::render_svg(&mut ctx, input, scale);
                    } else {
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
                    }
                    render_info_string(&mut ctx, &info_string);

                    ctx = PietGpuRenderContext::new();
                    test_scene1_old(&mut ctx);
                    let mut encoded_scene_old = ctx.encoded_scene();
                    let ramp_data = ctx.get_ramp_data();
                    encoded_scene_old.ramp_data = &ramp_data;
                    test_scene1(&mut scene, &mut rcx);
                    let encoded_scene = scene_to_encoded_scene(&scene, &rcx);
                    // println!("{:?}\n============\n{:?}", encoded_scene_old, encoded_scene);
                    // panic!();
                    let res = if mode & 1 == 0 {
                        render_info_string(&mut ctx, &info_string);
                        renderer.upload_render_ctx(&mut ctx, frame_idx)
                    } else {
                        renderer.upload_scene(&encoded_scene, frame_idx)
                    };
                    if let Err(e) = res {
                        println!("error in uploading: {}", e);
                    }

                    let (image_idx, acquisition_semaphore) = swapchain.next().unwrap();
                    let swap_image = swapchain.image(image_idx);
                    let query_pool = &query_pools[frame_idx];
                    let mut cmd_buf = cmd_bufs[frame_idx].take().unwrap_or_else(|| session.cmd_buf().unwrap());
                    cmd_buf.begin();
                    renderer.record(&mut cmd_buf, &query_pool, frame_idx);

                    // Image -> Swapchain
                    cmd_buf.image_barrier(
                        &swap_image,
                        ImageLayout::Undefined,
                        ImageLayout::BlitDst,
                    );
                    cmd_buf.blit_image(&renderer.image_dev, &swap_image);
                    cmd_buf.image_barrier(&swap_image, ImageLayout::BlitDst, ImageLayout::Present);
                    cmd_buf.finish();

                    submitted[frame_idx] = Some(session
                        .run_cmd_buf(
                            cmd_buf,
                            &[&acquisition_semaphore],
                            &[&present_semaphores[frame_idx]],
                        )
                        .unwrap());

                    swapchain
                        .present(image_idx, &[&present_semaphores[frame_idx]])
                        .unwrap();

                    current_frame += 1;
                }
                Event::LoopDestroyed => {
                    for cmd_buf in &mut submitted {
                        // Wait for command list submission, otherwise dropping of renderer may
                        // cause validation errors (and possibly crashes).
                        if let Some(cmd_buf) = cmd_buf.take() {
                            cmd_buf.wait().unwrap();
                        }
                    }
                }
                _ => (),
            }
        })
    }
}

fn test_scene1_old(ctx: &mut PietGpuRenderContext) {
    use piet::kurbo::{Affine, Rect, Vec2};
    use piet::{Color, GradientStop};
    ctx.transform(Affine::translate(Vec2::new(200., 200.)) * Affine::rotate(45f64.to_radians()));
    let linear = ctx
        .gradient(piet::FixedGradient::Linear(piet::FixedLinearGradient {
            start: Point::new(0., 0.),
            end: Point::new(200., 100.),
            stops: vec![
                GradientStop {
                    pos: 0.0,
                    color: Color::rgb8(0, 0, 255),
                },
                GradientStop {
                    pos: 0.5,
                    color: Color::rgb8(0, 255, 0),
                },
                GradientStop {
                    pos: 1.0,
                    color: Color::rgb8(255, 0, 0),
                },
            ],
        }))
        .unwrap();
    let radial = ctx
        .gradient(piet::FixedGradient::Radial(piet::FixedRadialGradient {
            center: Point::new(50., 50.),
            origin_offset: Vec2::new(0., 0.),
            radius: 240.,
            stops: vec![
                GradientStop {
                    pos: 0.0,
                    color: Color::rgb8(0, 0, 255),
                },
                GradientStop {
                    pos: 0.5,
                    color: Color::rgb8(0, 255, 0),
                },
                GradientStop {
                    pos: 1.0,
                    color: Color::rgb8(255, 0, 0),
                },
            ],
        }))
        .unwrap();
    ctx.fill_transform(
        Rect {
            x0: 0.,
            y0: 0.,
            x1: 200.,
            y1: 100.,
        },
        // &piet::PaintBrush::Color(piet::Color::rgb8(0, 255, 0)),
        &radial,
        // &piet::FixedGradient::Linear(piet::FixedLinearGradient {
        //     start: Point::new(0., 0.),
        //     end: Point::new(200., 100.),
        //     stops: vec![
        //         GradientStop {
        //             pos: 0.0,
        //             color: Color::rgb8(0, 0, 255)
        //         },
        //         GradientStop {
        //             pos: 0.5,
        //             color: Color::rgb8(0, 255, 0)
        //         },
        //         GradientStop {
        //             pos: 1.0,
        //             color: Color::rgb8(255, 0, 0)
        //         },
        //     ],
        // }),
        Affine::default(), // rotate(90f64.to_radians()),
    );
}

fn test_scene1(scene: &mut Scene, rcx: &mut ResourceContext) {
    use piet_scene::brush::*;
    use piet_scene::geometry::{Affine, Point, Rect};
    use piet_scene::scene::*;
    let mut fragment = Fragment::default();
    let mut b = build_fragment(&mut fragment);
    let linear = Brush::LinearGradient(LinearGradient {
        start: Point::new(0., 0.),
        end: Point::new(200., 100.),
        extend: Extend::Pad,
        stops: (&[
            Stop {
                offset: 0.,
                color: Color::rgb8(0, 0, 255),
            },
            Stop {
                offset: 0.5,
                color: Color::rgb8(0, 255, 0),
            },
            Stop {
                offset: 1.,
                color: Color::rgb8(255, 0, 0),
            },
        ][..])
            .into(),
    });
    let radial = Brush::RadialGradient(RadialGradient {
        center0: Point::new(50., 50.),
        center1: Point::new(50., 50.),
        radius0: 0.,
        radius1: 240.,
        extend: Extend::Pad,
        stops: (&[
            Stop {
                offset: 0.,
                color: Color::rgb8(0, 0, 255),
            },
            Stop {
                offset: 0.5,
                color: Color::rgb8(0, 255, 0),
            },
            Stop {
                offset: 1.,
                color: Color::rgb8(255, 0, 0),
            },
        ][..])
            .into(),
    });
    //b.push_transform(Affine::translate(200., 200.) * Affine::rotate(45f32.to_radians()));
    b.fill(
        Fill::NonZero,
        // &Brush::Solid(Color::rgba8(0, 255, 0, 255)),
        &radial,
        None, //Some(Affine::rotate(90f32.to_radians())),
        Rect {
            min: Point::new(0., 0.),
            max: Point::new(200., 100.),
        }
        .elements(),
    );
    b.finish();
    let mut b = build_scene(scene, rcx);
    b.push_transform(Affine::translate(200., 200.) * Affine::rotate(45f32.to_radians()));
    b.append(&fragment);
    b.pop_transform();
    b.push_transform(Affine::translate(400., 600.));
    b.append(&fragment);
    b.finish();
}

fn scene_to_encoded_scene<'a>(
    scene: &'a Scene,
    rcx: &'a ResourceContext,
) -> EncodedSceneRef<'a, piet_scene::geometry::Affine> {
    let d = scene.data();
    EncodedSceneRef {
        transform_stream: &d.transform_stream,
        tag_stream: &d.tag_stream,
        pathseg_stream: &d.pathseg_stream,
        linewidth_stream: &d.linewidth_stream,
        drawtag_stream: &d.drawtag_stream,
        drawdata_stream: &d.drawdata_stream,
        n_path: d.n_path,
        n_pathseg: d.n_pathseg,
        n_clip: d.n_clip,
        ramp_data: rcx.ramp_data(),
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

// use piet_scene::geometry::*;
// use piet_scene::path::*;
// use piet_scene::scene::*;
// use piet_scene::{geometry::*, path::*, resource::ResourceContext, scene::*};

// fn main() {
//     let mut scene = Scene::default();
//     let mut rcx = ResourceContext::new();
//     let mut sb = build_scene(&mut scene, &mut rcx);

//     sb.push_layer(Blend::default(), Rect::default().elements());

//     // let mut path = Path::new();
//     // let mut b = PathBuilder::new(&mut path);
//     // b.move_to(100., 100.);
//     // b.line_to(200., 200.);
//     // b.close_path();
//     // b.move_to(50., 50.);
//     // b.line_to(600., 150.);
//     // b.move_to(4., 2.);
//     // b.quad_to(8., 8., 9., 9.);
//     // b.close_path();
//     // println!("{:?}", path);
//     // for el in path.elements() {
//     //     println!("{:?}", el);
//     // }
//     //sb.push_layer(path.elements(), BlendMode::default());

//     sb.push_layer(Blend::default(), [Element::MoveTo((0., 0.).into())]);
// }
