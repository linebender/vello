// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Simple example.

use anyhow::Result;
use parley::{
    Alignment, AlignmentOptions, FontContext, FontFamily, FontStack, LayoutContext, LineHeight,
    PositionedLayoutItem, StyleProperty,
};
use std::borrow::Cow;
use std::sync::Arc;
use vello::kurbo::{Affine, Circle, Ellipse, Line, RoundedRect, Stroke};
use vello::peniko::color::palette;
use vello::peniko::{Blob, BrushRef, Color, Fill};
use vello::util::{RenderContext, RenderSurface};
use vello::{AaConfig, Renderer, RendererOptions, Scene};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::Window;

use vello::wgpu;

#[derive(Debug)]
enum RenderState<'s> {
    /// `RenderSurface` and `Window` for active rendering.
    Active {
        // The `RenderSurface` and the `Window` must be in this order, so that the surface is dropped first.
        surface: Box<RenderSurface<'s>>,
        window: Arc<Window>,
    },
    /// Cache a window so that it can be reused when the app is resumed after being suspended.
    Suspended(Option<Arc<Window>>),
}

struct SimpleVelloApp<'s> {
    // The vello RenderContext which is a global context that lasts for the
    // lifetime of the application
    context: RenderContext,

    // The FontContext which is used to manage fonts
    font_context: FontContext,

    // The LayoutContext which is used to manage text layouts
    layout_context: LayoutContext<ColorBrush>,

    // An array of renderers, one per wgpu device
    renderers: Vec<Option<Renderer>>,

    // State for our example where we store the winit Window and the wgpu Surface
    state: RenderState<'s>,

    // A vello Scene which is a data structure which allows one to build up a
    // description a scene to be drawn (with paths, fills, images, text, etc)
    // which is then passed to a renderer for rendering
    scene: Scene,
}

impl ApplicationHandler for SimpleVelloApp<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let RenderState::Suspended(cached_window) = &mut self.state else {
            return;
        };

        // Get the winit window cached in a previous Suspended event or else create a new window
        let window = cached_window
            .take()
            .unwrap_or_else(|| create_winit_window(event_loop));

        // Create a vello Surface
        let size = window.inner_size();
        let surface_future = self.context.create_surface(
            window.clone(),
            size.width,
            size.height,
            wgpu::PresentMode::AutoVsync,
        );
        let surface = pollster::block_on(surface_future).expect("Error creating surface");

        // Create a vello Renderer for the surface (using its device id)
        self.renderers
            .resize_with(self.context.devices.len(), || None);
        self.renderers[surface.dev_id]
            .get_or_insert_with(|| create_vello_renderer(&self.context, &surface));

        // Save the Window and Surface to a state variable
        self.state = RenderState::Active {
            surface: Box::new(surface),
            window,
        };
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        if let RenderState::Active { window, .. } = &self.state {
            self.state = RenderState::Suspended(Some(window.clone()));
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        // Only process events for our window, and only when we have a surface.
        let surface = match &mut self.state {
            RenderState::Active { surface, window } if window.id() == window_id => surface,
            _ => return,
        };

        match event {
            // Exit the event loop when a close is requested (e.g. window's close button is pressed)
            WindowEvent::CloseRequested => event_loop.exit(),

            // Resize the surface when the window is resized
            WindowEvent::Resized(size) => {
                self.context
                    .resize_surface(surface, size.width, size.height);
            }

            // This is where all the rendering happens
            WindowEvent::RedrawRequested => {
                // Empty the scene of objects to draw. You could create a new Scene each time, but in this case
                // the same Scene is reused so that the underlying memory allocation can also be reused.
                self.scene.reset();

                // Re-add the objects to draw to the scene.
                add_shapes_to_scene(
                    &mut self.scene,
                    &mut self.font_context,
                    &mut self.layout_context,
                );

                // Get the window size
                let width = surface.config.width;
                let height = surface.config.height;

                // Get a handle to the device
                let device_handle = &self.context.devices[surface.dev_id];

                // Render to a texture, which we will later copy into the surface
                self.renderers[surface.dev_id]
                    .as_mut()
                    .unwrap()
                    .render_to_texture(
                        &device_handle.device,
                        &device_handle.queue,
                        &self.scene,
                        &surface.target_view,
                        &vello::RenderParams {
                            base_color: palette::css::BLACK, // Background color
                            width,
                            height,
                            antialiasing_method: AaConfig::Msaa16,
                        },
                    )
                    .expect("failed to render to surface");

                // Get the surface's texture
                let surface_texture = surface
                    .surface
                    .get_current_texture()
                    .expect("failed to get surface texture");

                // Perform the copy
                let mut encoder =
                    device_handle
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Surface Blit"),
                        });
                surface.blitter.copy(
                    &device_handle.device,
                    &mut encoder,
                    &surface.target_view,
                    &surface_texture
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default()),
                );
                device_handle.queue.submit([encoder.finish()]);
                // Queue the texture to be presented on the surface
                surface_texture.present();

                device_handle.device.poll(wgpu::PollType::Poll).unwrap();
            }
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    // By default the FontContext will be empty
    let mut font_context = FontContext::new();

    // Register the Roboto font from the assets directory
    font_context.collection.register_fonts(
        Blob::new(Arc::new(include_bytes!(
            "../../assets/roboto/Roboto-Regular.ttf"
        ))),
        None,
    );

    // Check if the font was registered successfully
    font_context
        .collection
        .family_id("roboto")
        .expect("Failed to register font family");

    // Setup a bunch of state:
    let mut app = SimpleVelloApp {
        context: RenderContext::new(),
        font_context,
        layout_context: LayoutContext::new(),
        renderers: vec![],
        state: RenderState::Suspended(None),
        scene: Scene::new(),
    };

    // Create and run a winit event loop
    let event_loop = EventLoop::new()?;
    event_loop
        .run_app(&mut app)
        .expect("Couldn't run event loop");
    Ok(())
}

/// Helper function that creates a Winit window and returns it (wrapped in an Arc for sharing between threads)
fn create_winit_window(event_loop: &ActiveEventLoop) -> Arc<Window> {
    let attr = Window::default_attributes()
        .with_inner_size(LogicalSize::new(1044, 800))
        .with_resizable(true)
        .with_title("Vello Shapes");
    Arc::new(event_loop.create_window(attr).unwrap())
}

/// Helper function that creates a vello `Renderer` for a given `RenderContext` and `RenderSurface`
fn create_vello_renderer(render_cx: &RenderContext, surface: &RenderSurface<'_>) -> Renderer {
    Renderer::new(
        &render_cx.devices[surface.dev_id].device,
        RendererOptions::default(),
    )
    .expect("Couldn't create renderer")
}

/// Add shapes to a vello scene. This does not actually render the shapes, but adds them
/// to the Scene data structure which represents a set of objects to draw.
fn add_shapes_to_scene(
    scene: &mut Scene,
    font_context: &mut FontContext,
    layout_context: &mut LayoutContext<ColorBrush>,
) {
    // Draw an outlined rectangle
    let stroke = Stroke::new(6.0);
    let rect = RoundedRect::new(10.0, 10.0, 240.0, 240.0, 20.0);
    let rect_stroke_color = Color::new([0.9804, 0.702, 0.5294, 1.]);
    scene.stroke(&stroke, Affine::IDENTITY, rect_stroke_color, None, &rect);

    // Draw a filled circle
    let circle = Circle::new((420.0, 200.0), 120.0);
    let circle_fill_color = Color::new([0.9529, 0.5451, 0.6588, 1.]);
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        circle_fill_color,
        None,
        &circle,
    );

    // Draw a filled ellipse
    let ellipse = Ellipse::new((250.0, 420.0), (100.0, 160.0), -90.0);
    let ellipse_fill_color = Color::new([0.7961, 0.651, 0.9686, 1.]);
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        ellipse_fill_color,
        None,
        &ellipse,
    );

    // Draw a straight line
    let line = Line::new((260.0, 20.0), (620.0, 100.0));
    let line_stroke_color = Color::new([0.5373, 0.7059, 0.9804, 1.]);
    scene.stroke(&stroke, Affine::IDENTITY, line_stroke_color, None, &line);

    add_text_to_scene(scene, font_context, layout_context);
}

// A Simple color brush. There is no default brush in Vello/parley, so we define one here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ColorBrush {
    pub(crate) color: Color,
}

impl ColorBrush {
    pub(crate) fn new(color: Color) -> Self {
        let rgba8 = color.to_rgba8();
        Self {
            color: Color::from_rgba8(rgba8.r, rgba8.g, rgba8.b, rgba8.a),
        }
    }
}

// This is important, as properties in parley must implement `Default` to be used in styles.
impl Default for ColorBrush {
    fn default() -> Self {
        Self {
            color: Color::BLACK,
        }
    }
}

fn add_text_to_scene(
    scene: &mut Scene,
    font_context: &mut FontContext,
    layout_context: &mut LayoutContext<ColorBrush>,
) {
    let text = "This is a simple example of rendering text with Vello.\n\
                It supports custom fonts, colors, and styles.\n\
                Enjoy creating beautiful graphics!";

    // The `text` passed here is just used as a length hint for the layout.
    let mut builder = layout_context.ranged_builder(font_context, text, 1.0, true);

    builder.push_default(StyleProperty::Brush(ColorBrush::new(palette::css::RED)));

    builder.push_default(StyleProperty::FontStack(FontStack::Single(
        FontFamily::Named(Cow::Borrowed("Roboto")),
    )));

    builder.push_default(StyleProperty::LineHeight(LineHeight::FontSizeRelative(1.0)));

    // This is where the text is built according to your string
    let mut layout = builder.build(text);
    layout.break_all_lines(None);
    layout.align(None, Alignment::Start, AlignmentOptions::default());

    for line in layout.lines() {
        for item in line.items() {
            let PositionedLayoutItem::GlyphRun(glyph_run) = item else {
                continue;
            };

            let style = glyph_run.style();
            let mut x = glyph_run.offset();
            let y = glyph_run.baseline();
            let run = glyph_run.run();
            let font = run.font();
            let font_size = run.font_size();
            let synthesis = run.synthesis();
            let glyph_xform = synthesis
                .skew()
                .map(|angle| Affine::skew(angle.to_radians().tan() as f64, 0.0));
            scene
                .draw_glyphs(font)
                .brush(BrushRef::Solid(style.brush.color))
                .hint(true)
                .transform(Affine::translate((50.0, 700.0)))
                .glyph_transform(glyph_xform)
                .font_size(font_size)
                .normalized_coords(run.normalized_coords())
                .draw(
                    Fill::NonZero,
                    glyph_run.glyphs().map(|glyph| {
                        let gx = x + glyph.x;
                        let gy = y - glyph.y;
                        x += glyph.advance;
                        vello::Glyph {
                            id: glyph.id as _,
                            x: gx,
                            y: gy,
                        }
                    }),
                );
        }
    }
}
