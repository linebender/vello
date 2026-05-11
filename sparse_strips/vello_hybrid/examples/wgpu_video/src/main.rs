// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Vello Hybrid example that renders decoded video frames through Vello's
//! native NV12 (YCbCr) external-texture API on a wgpu backend.
//!
//! macOS-only: scans `~/Downloads/videos/` for `.mp4` / `.mov` files and opens each
//! through a real video pipeline (`AVAsset` demuxer -> `VideoToolbox` decoder).
//! Each decoded NV12 frame is bound directly to Vello as a pair of plane
//! `wgpu::Texture` views (Y in `R8Unorm`, Cb/Cr in `Rg8Unorm`); the YCbCr → RGB
//! conversion happens inside Vello's shader. All videos are laid out in a grid
//! and play back simultaneously, paced by their own presentation timestamps so
//! they run at native frame rate regardless of how fast the host renders.

#[cfg(target_os = "macos")]
mod frame_source;
#[cfg(target_os = "macos")]
mod render_context;
#[cfg(target_os = "macos")]
mod video;

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!(
        "vello_hybrid_wgpu_video currently only runs on macOS \
         (uses VideoToolbox + IOSurface for zero-copy video import)."
    );
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
fn main() {
    macos::run();
}

#[cfg(target_os = "macos")]
mod macos {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Instant;

    use glifo::Glyph;
    use skrifa::MetadataProvider;
    use skrifa::raw::FileRef;
    use vello_common::color::palette::css::YELLOW;
    use vello_common::geometry::RectU16;
    use vello_common::kurbo::{Affine, BezPath, Circle, Ellipse, Point, Shape, Vec2};
    use vello_common::peniko::{Blob, FontData, ImageQuality};
    use vello_hybrid::{
        ExternalTextureFormat, RenderSize, Resources, SampleRect, Scene, TextureBindings, TextureId,
    };
    use winit::application::ApplicationHandler;
    use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::keyboard::{Key, NamedKey};
    use winit::window::{Window, WindowId};

    use crate::frame_source::{FrameSource, VideoFileSource};
    use crate::render_context::RenderContext;

    /// Directory scanned for `.mp4` / `.mov` files, relative to `$HOME`.
    const VIDEOS_RELATIVE_DIR: &str = "Downloads/videos";

    const INITIAL_WIDTH: u32 = 1920;
    const INITIAL_HEIGHT: u32 = 1080;

    const ZOOM_STEP: f64 = 0.1;

    /// Rotation step per `[` / `]` keypress, in radians (15°). 30°/2 = 15°.
    const ROTATION_STEP_RADIANS: f64 = std::f64::consts::FRAC_PI_6 * 0.5;

    /// Roboto Regular embedded at build time and used to render the "Mr. Bean"
    /// overlay on top of the video grid.
    const ROBOTO_FONT: &[u8] =
        include_bytes!("../../../../../examples/assets/roboto/Roboto-Regular.ttf");

    /// Text drawn as a large overlay across the whole scene.
    const OVERLAY_TEXT: &str = "Mr. Bean";

    /// Reference font size used to lay out the cached overlay glyph run. We
    /// rescale this layout per-frame to hit a target text width.
    const OVERLAY_REF_FONT_SIZE: f32 = 100.0;

    /// Fraction of the scene width the overlay text should span before clamping.
    const OVERLAY_TARGET_WIDTH_FRACTION: f64 = 0.8;

    /// Clamp range for the final overlay font size (in pixels). Keeps the text
    /// readable when the window is tiny and avoids absurd allocations when it's
    /// huge.
    const OVERLAY_MIN_FONT_SIZE: f32 = 24.0;
    const OVERLAY_MAX_FONT_SIZE: f32 = 400.0;

    pub(super) fn run() {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .format_timestamp(Some(env_logger::TimestampPrecision::Millis))
            .init();

        let event_loop = EventLoop::new().expect("Failed to create event loop");
        let mut app = App::default();
        event_loop.run_app(&mut app).expect("Event loop failed");
    }

    /// One cell in the video grid: a source pipeline plus its assigned texture slot.
    struct VideoSlot {
        source: VideoFileSource,
        texture_id: TextureId,
    }

    /// Pre-laid out glyph run for the "Mr. Bean" overlay text.
    ///
    /// Layout is done once at startup at [`OVERLAY_REF_FONT_SIZE`] with the
    /// baseline at `y = 0`. At render time we just scale the cached positions
    /// and translate the run to center it in the scene.
    struct OverlayText {
        font: FontData,
        glyphs: Vec<Glyph>,
        /// Total advance width of [`Self::glyphs`] at the reference font size.
        width_at_ref_size: f32,
        /// Ascent at the reference font size, used for vertical centering.
        ascent_at_ref_size: f32,
        /// Descent at the reference font size (typically negative).
        descent_at_ref_size: f32,
    }

    impl OverlayText {
        /// Lay out [`OVERLAY_TEXT`] in Roboto Regular at [`OVERLAY_REF_FONT_SIZE`].
        ///
        /// Mirrors the pattern used by `vello_sparse_tests::util::layout_glyphs`:
        /// resolve the `FontRef` (handling both single-font and collection
        /// files), then walk the chars and accumulate advances.
        fn new() -> Self {
            let font = FontData::new(Blob::new(Arc::new(ROBOTO_FONT)), 0);
            let font_ref = {
                let file_ref =
                    FileRef::new(font.data.as_ref()).expect("Roboto-Regular.ttf is a valid font");
                match file_ref {
                    FileRef::Font(f) => f,
                    FileRef::Collection(collection) => collection
                        .get(font.index)
                        .expect("font index in range for collection"),
                }
            };
            let size = skrifa::instance::Size::new(OVERLAY_REF_FONT_SIZE);
            let variations: Vec<(&str, f32)> = vec![];
            let var_loc = font_ref.axes().location(variations.as_slice());
            let charmap = font_ref.charmap();
            let metrics = font_ref.metrics(size, &var_loc);
            let glyph_metrics = font_ref.glyph_metrics(size, &var_loc);

            let mut pen_x = 0_f32;
            let glyphs: Vec<Glyph> = OVERLAY_TEXT
                .chars()
                .map(|ch| {
                    let gid = charmap.map(ch).unwrap_or_default();
                    let advance = glyph_metrics.advance_width(gid).unwrap_or_default();
                    let x = pen_x;
                    pen_x += advance;
                    Glyph {
                        id: gid.to_u32(),
                        x,
                        y: 0.0,
                    }
                })
                .collect();

            log::info!(
                "overlay text initialised: '{OVERLAY_TEXT}' \
                 ({} glyphs, ref width {:.1} px @ {OVERLAY_REF_FONT_SIZE} px)",
                glyphs.len(),
                pen_x,
            );

            Self {
                font,
                glyphs,
                width_at_ref_size: pen_x,
                ascent_at_ref_size: metrics.ascent,
                descent_at_ref_size: metrics.descent,
            }
        }

        /// Draw the cached glyph run into `scene`, scaled so its advance width
        /// is about [`OVERLAY_TARGET_WIDTH_FRACTION`] of `scene_w` and centered
        /// both horizontally and vertically. The font size is clamped to
        /// `[OVERLAY_MIN_FONT_SIZE, OVERLAY_MAX_FONT_SIZE]`.
        ///
        /// Free-standing (not a method on [`Active`]) so the caller can split
        /// borrows of `scene` / `resources` / `transform` from the rest of
        /// `Active` — those fields would otherwise alias with the immutable
        /// borrow of `self.slots` that survives in `TextureBindings` for the
        /// remainder of `Active::render`.
        #[allow(
            clippy::cast_possible_truncation,
            reason = "f64 scene-space pixel coordinates are converted to f32 \
                      for glyph positions; for realistic window sizes the \
                      values comfortably fit in f32"
        )]
        fn draw(
            &self,
            scene: &mut Scene,
            resources: &mut Resources,
            transform: Affine,
            scene_w: f64,
            scene_h: f64,
        ) {
            // Avoid divide-by-zero on a pathological empty layout.
            if self.width_at_ref_size <= 0.0 {
                return;
            }

            let width_at_ref_size = f64::from(self.width_at_ref_size);
            let ref_font_size = f64::from(OVERLAY_REF_FONT_SIZE);

            let target_text_width = scene_w * OVERLAY_TARGET_WIDTH_FRACTION;
            let desired_font_size = (target_text_width / width_at_ref_size) * ref_font_size;
            let font_size_f64 = desired_font_size.clamp(
                f64::from(OVERLAY_MIN_FONT_SIZE),
                f64::from(OVERLAY_MAX_FONT_SIZE),
            );
            let font_size = font_size_f64 as f32;
            let scale_f64 = font_size_f64 / ref_font_size;
            let scale = scale_f64 as f32;

            let text_w_px = width_at_ref_size * scale_f64;
            let ascent_px = f64::from(self.ascent_at_ref_size) * scale_f64;
            let descent_px = f64::from(self.descent_at_ref_size) * scale_f64;
            let text_h_px = ascent_px - descent_px;

            let offset_x = (scene_w - text_w_px) * 0.5;
            // `glyph_run` expects y coordinates to be on the baseline. Place
            // the baseline so the ink box `[ -ascent, -descent ]` is vertically
            // centered in the scene.
            let baseline_y = (scene_h - text_h_px) * 0.5 + ascent_px;

            let offset_x_f32 = offset_x as f32;
            let baseline_y_f32 = baseline_y as f32;
            let positioned_glyphs = self.glyphs.iter().map(move |g| Glyph {
                id: g.id,
                x: g.x * scale + offset_x_f32,
                y: g.y * scale + baseline_y_f32,
            });

            // Set the transform again defensively so future intermediate
            // transforms added to `Active::render` can't accidentally leak into
            // the overlay's placement. `transform` is already set at the top of
            // `render`.
            scene.set_transform(transform);
            scene.set_paint(YELLOW);
            scene
                .glyph_run(resources, &self.font)
                .font_size(font_size)
                .hint(true)
                .fill_glyphs(positioned_glyphs);
        }
    }

    #[derive(Default)]
    struct App<'window> {
        active: Option<Active<'window>>,
    }

    struct Active<'window> {
        window: Arc<Window>,
        render_context: RenderContext<'window>,
        scene: Scene,
        resources: Resources,
        slots: Vec<VideoSlot>,
        /// Cached layout for the "Mr. Bean" text overlay drawn on top of all
        /// videos every frame.
        overlay: OverlayText,
        // --- interaction state ---
        transform: Affine,
        mouse_down: bool,
        last_cursor_position: Option<Point>,
        /// When set, sources that have reached end-of-stream are restarted at the
        /// top of the next render so videos loop indefinitely. Toggled with `L`.
        auto_replay: bool,
        // --- FPS counter ---
        last_frame_time: Option<Instant>,
        frame_count: u32,
        fps_update_time: Instant,
        /// Total wall-clock between consecutive [`Self::render`] entries.
        accumulated_frame_time: f64,
        /// CPU time spent inside the render function (decode + scene build +
        /// `Renderer::render`), excluding the blocking `surface.present()`.
        /// Distinct from `accumulated_frame_time` so you can see actual render
        /// cost separately from `VSync` wait when present mode is `Fifo`.
        accumulated_render_time: f64,
    }

    impl ApplicationHandler for App<'_> {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.active.is_some() {
                return;
            }

            let attrs = Window::default_attributes()
                .with_title("Vello Hybrid — wgpu video")
                .with_inner_size(winit::dpi::PhysicalSize::new(INITIAL_WIDTH, INITIAL_HEIGHT));
            let window = Arc::new(
                event_loop
                    .create_window(attrs)
                    .expect("Failed to create window"),
            );
            let render_context = RenderContext::new(window.clone());

            let slots = build_video_slots(&render_context);
            log::info!("{} video(s) loaded into grid", slots.len());

            let size = window.inner_size();
            let scene = Scene::new(
                u16::try_from(size.width.max(1)).unwrap_or(u16::MAX),
                u16::try_from(size.height.max(1)).unwrap_or(u16::MAX),
            );

            self.active = Some(Active {
                window,
                render_context,
                scene,
                resources: Resources::new(),
                slots,
                overlay: OverlayText::new(),
                transform: Affine::IDENTITY,
                mouse_down: false,
                last_cursor_position: None,
                auto_replay: true,
                last_frame_time: None,
                frame_count: 0,
                fps_update_time: Instant::now(),
                accumulated_frame_time: 0.0,
                accumulated_render_time: 0.0,
            });
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            window_id: WindowId,
            event: WindowEvent,
        ) {
            let Some(active) = self.active.as_mut() else {
                return;
            };
            if active.window.id() != window_id {
                return;
            }

            match event {
                WindowEvent::CloseRequested => event_loop.exit(),

                // ---- keyboard ----
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            logical_key,
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                } => match logical_key {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::Space) => {
                        active.transform = Affine::IDENTITY;
                        active.window.request_redraw();
                    }
                    Key::Character(ref ch) if ch.eq_ignore_ascii_case("r") => {
                        for slot in &mut active.slots {
                            slot.source.restart();
                        }
                        active.transform = Affine::IDENTITY;
                        active.window.request_redraw();
                    }
                    Key::Character(ref ch) if ch.eq_ignore_ascii_case("l") => {
                        active.auto_replay = !active.auto_replay;
                        log::info!(
                            "auto-replay: {}",
                            if active.auto_replay { "ON" } else { "OFF" }
                        );
                    }
                    Key::Character(ref ch) if ch.as_str() == "[" || ch.as_str() == "]" => {
                        let theta = if ch.as_str() == "[" {
                            -ROTATION_STEP_RADIANS
                        } else {
                            ROTATION_STEP_RADIANS
                        };
                        let center = active.last_cursor_position.unwrap_or(Point::new(
                            f64::from(active.render_context.surface_config.width) * 0.5,
                            f64::from(active.render_context.surface_config.height) * 0.5,
                        ));
                        active.transform = active.transform.then_rotate_about(theta, center);
                        active.window.request_redraw();
                    }
                    _ => {}
                },

                // ---- mouse drag (pan) ----
                WindowEvent::MouseInput {
                    state,
                    button: MouseButton::Left,
                    ..
                } => {
                    active.mouse_down = state == ElementState::Pressed;
                    if !active.mouse_down {
                        active.last_cursor_position = None;
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    let current = Point::new(position.x, position.y);
                    if active.mouse_down
                        && let Some(last) = active.last_cursor_position
                    {
                        active.transform = active.transform.then_translate(current - last);
                        active.window.request_redraw();
                    }
                    active.last_cursor_position = Some(current);
                }

                // ---- scroll wheel (zoom) ----
                WindowEvent::MouseWheel { delta, .. } => {
                    let delta_y = match delta {
                        MouseScrollDelta::LineDelta(_, y) => f64::from(y),
                        MouseScrollDelta::PixelDelta(pos) => pos.y / 100.0,
                    };
                    if let Some(cursor) = active.last_cursor_position {
                        let factor = (1.0 + delta_y * ZOOM_STEP).max(0.1);
                        active.transform = active.transform.then_scale_about(factor, cursor);
                        active.window.request_redraw();
                    }
                }

                // ---- trackpad pinch (zoom) ----
                WindowEvent::PinchGesture { delta, .. } => {
                    let factor = 1.0 + delta * ZOOM_STEP * 5.0;
                    let center = active.last_cursor_position.unwrap_or(Point::new(
                        f64::from(active.render_context.surface_config.width) * 0.5,
                        f64::from(active.render_context.surface_config.height) * 0.5,
                    ));
                    active.transform = active.transform.then_scale_about(factor, center);
                    active.window.request_redraw();
                }

                // ---- resize ----
                WindowEvent::Resized(size) => {
                    active.render_context.resize(size.width, size.height);
                    active.scene = Scene::new(
                        u16::try_from(size.width.max(1)).unwrap_or(u16::MAX),
                        u16::try_from(size.height.max(1)).unwrap_or(u16::MAX),
                    );
                    active.window.request_redraw();
                }

                WindowEvent::RedrawRequested => {
                    active.render();
                }
                _ => {}
            }
        }
    }

    impl Active<'_> {
        fn render(&mut self) {
            self.update_fps();

            let render_start = Instant::now();

            // Phase 1: advance every source (mutable borrows).
            // When auto-replay is on, restart sources that just reached EOS so the
            // grid keeps looping. `restart()` pre-decodes the first frame of the
            // new playback into `current`, so we skip `next_frame()` to avoid
            // immediately advancing past it.
            for slot in &mut self.slots {
                slot.source.next_frame();
                if self.auto_replay && slot.source.is_eos() {
                    slot.source.restart();
                }
            }

            // Phase 2: build scene + texture bindings (immutable borrows).
            self.scene.reset();
            self.scene.set_transform(self.transform);
            let mut bindings = TextureBindings::new();

            let (cols, rows) = grid_dims(self.slots.len());
            let scene_w = f64::from(self.scene.width());
            let scene_h = f64::from(self.scene.height());
            let cell_w = scene_w / cols as f64;
            let cell_h = scene_h / rows as f64;

            for (i, slot) in self.slots.iter().enumerate() {
                let frame = slot.source.current_frame();
                bindings.insert_ycbcr_nv12(slot.texture_id, frame.y_view(), frame.uv_view());

                let col = i % cols;
                let row = i / cols;
                let cell_x = col as f64 * cell_w;
                let cell_y = row as f64 * cell_h;

                let (fw, fh) = frame.dimensions();
                if fw == 0 || fh == 0 {
                    continue;
                }
                let scale = (cell_w / f64::from(fw)).min(cell_h / f64::from(fh));
                let dst_w = f64::from(fw) * scale;
                let dst_h = f64::from(fh) * scale;
                let tx = cell_x + (cell_w - dst_w) * 0.5;
                let ty = cell_y + (cell_h - dst_h) * 0.5;

                // Pick a different clip shape per cell, cycling through the
                // [`ClipShape`] variants. Clips are built in scene-space and are
                // therefore transformed by the user's pan/zoom
                // (`Scene::set_transform` above).
                let center = Point::new(tx + dst_w * 0.5, ty + dst_h * 0.5);
                let clip = ClipShape::for_index(i).build(center, dst_w, dst_h);
                self.scene.push_clip_path(&clip);

                let sample = SampleRect {
                    source_region: RectU16::new(0, 0, fw, fh),
                    transform: Affine::translate((tx, ty)) * Affine::scale(scale),
                };
                self.scene.draw_texture_rects(
                    slot.texture_id,
                    ImageQuality::Medium,
                    ExternalTextureFormat::YCbCrNv12 {
                        color_space: frame.color_space(),
                    },
                    [sample],
                );

                self.scene.pop_clip_path();
            }

            // Draw the "Mr. Bean" overlay text on top of every video cell. The
            // scene transform set above (= the user's pan/zoom/rotate) applies
            // to the glyph run too, so the overlay pans and zooms with the
            // grid. Disjoint field borrows so we don't alias the immutable
            // borrow of `self.slots` still held by `bindings`.
            self.overlay.draw(
                &mut self.scene,
                &mut self.resources,
                self.transform,
                scene_w,
                scene_h,
            );

            // Phase 3: GPU submit.
            let surface_texture = match self.render_context.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t) => t,
                wgpu::CurrentSurfaceTexture::Outdated
                | wgpu::CurrentSurfaceTexture::Suboptimal(_) => {
                    self.render_context.surface.configure(
                        &self.render_context.device,
                        &self.render_context.surface_config,
                    );
                    self.window.request_redraw();
                    return;
                }
                wgpu::CurrentSurfaceTexture::Occluded | wgpu::CurrentSurfaceTexture::Timeout => {
                    self.window.request_redraw();
                    return;
                }
                wgpu::CurrentSurfaceTexture::Lost => panic!("Surface lost"),
                wgpu::CurrentSurfaceTexture::Validation => panic!("Surface validation error"),
            };

            let surface_view = surface_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            let mut encoder = self.render_context.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("vello_hybrid_wgpu_video frame"),
                },
            );

            let render_size = RenderSize {
                width: self.render_context.surface_config.width,
                height: self.render_context.surface_config.height,
            };
            self.render_context
                .renderer
                .render(
                    &self.scene,
                    &mut self.resources,
                    &self.render_context.device,
                    &self.render_context.queue,
                    &mut encoder,
                    &render_size,
                    &surface_view,
                    &bindings,
                )
                .expect("Vello Hybrid render failed");

            // Stop the render timer before submit+present: under VSync `present()` blocks
            // until the next VBlank, which would otherwise dominate the measurement.
            self.accumulated_render_time += render_start.elapsed().as_secs_f64() * 1000.0;

            self.render_context.queue.submit([encoder.finish()]);
            surface_texture.present();

            self.window.request_redraw();
        }

        fn update_fps(&mut self) {
            let now = Instant::now();
            if let Some(last) = self.last_frame_time {
                let frame_ms = now.duration_since(last).as_secs_f64() * 1000.0;
                self.accumulated_frame_time += frame_ms;
                self.frame_count += 1;

                if now.duration_since(self.fps_update_time).as_secs_f64() >= 1.0 {
                    let count = f64::from(self.frame_count);
                    let avg_frame_ms = self.accumulated_frame_time / count;
                    let avg_render_ms = self.accumulated_render_time / count;
                    let avg_fps = 1000.0 / avg_frame_ms;
                    let loop_label = if self.auto_replay {
                        "loop ON"
                    } else {
                        "loop OFF"
                    };
                    let title = format!(
                        "Vello Hybrid — wgpu video — \
                         {avg_fps:.1} FPS  |  frame {avg_frame_ms:.2} ms  |  render {avg_render_ms:.2} ms  |  {loop_label}"
                    );
                    self.window.set_title(&title);
                    self.frame_count = 0;
                    self.accumulated_frame_time = 0.0;
                    self.accumulated_render_time = 0.0;
                    self.fps_update_time = now;
                }
            }
            self.last_frame_time = Some(now);
        }
    }

    /// One of the four clip shapes we cycle through, one per video cell.
    #[derive(Copy, Clone)]
    enum ClipShape {
        Circle,
        Triangle,
        Pentagon,
        Ellipse,
    }

    impl ClipShape {
        fn for_index(i: usize) -> Self {
            match i % 4 {
                0 => Self::Circle,
                1 => Self::Triangle,
                2 => Self::Pentagon,
                _ => Self::Ellipse,
            }
        }

        /// Build the clip path inscribed in a `dst_w x dst_h` box centered on
        /// `center`. Coordinates are in scene-space.
        fn build(self, center: Point, dst_w: f64, dst_h: f64) -> BezPath {
            // Tolerance for curve flattening (Circle / Ellipse). Picked tight enough
            // to look smooth at typical zoom levels but loose enough to keep strip
            // generation cheap.
            const TOLERANCE: f64 = 0.1;
            let r = dst_w.min(dst_h) * 0.5;
            match self {
                Self::Circle => Circle::new(center, r).to_path(TOLERANCE),
                Self::Ellipse => {
                    Ellipse::new(center, (dst_w * 0.5, dst_h * 0.5), 0.0).to_path(TOLERANCE)
                }
                Self::Triangle => regular_polygon(center, r, 3),
                Self::Pentagon => regular_polygon(center, r, 5),
            }
        }
    }

    /// Build a regular `sides`-gon inscribed in a circle of `radius` centered at
    /// `center`. The first vertex points straight up.
    fn regular_polygon(center: Point, radius: f64, sides: u32) -> BezPath {
        debug_assert!(
            sides >= 3,
            "regular_polygon needs at least 3 sides, got {sides}"
        );
        let mut path = BezPath::new();
        let n = f64::from(sides);
        for k in 0..sides {
            // -PI/2 puts the first vertex at 12 o'clock.
            let theta = -std::f64::consts::FRAC_PI_2 + std::f64::consts::TAU * f64::from(k) / n;
            let v = center + Vec2::new(radius * theta.cos(), radius * theta.sin());
            if k == 0 {
                path.move_to(v);
            } else {
                path.line_to(v);
            }
        }
        path.close_path();
        path
    }

    /// Compute a `(cols, rows)` grid that fits `n` items, roughly square.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "small grid counts fit in usize trivially"
    )]
    fn grid_dims(n: usize) -> (usize, usize) {
        if n == 0 {
            return (1, 1);
        }
        let cols = (n as f64).sqrt().ceil() as usize;
        let rows = n.div_ceil(cols);
        (cols, rows)
    }

    /// Scan `~/Downloads/videos/` for `.mp4` / `.mov` files and open a
    /// [`VideoFileSource`] for each one. Files that fail to open are logged and
    /// skipped.
    fn build_video_slots(render_context: &RenderContext<'_>) -> Vec<VideoSlot> {
        let home = std::env::var_os("HOME").expect("$HOME is unset");
        let dir = PathBuf::from(home).join(VIDEOS_RELATIVE_DIR);
        assert!(
            dir.is_dir(),
            "{} is not a directory — create it and drop some .mp4 / .mov files in",
            dir.display()
        );

        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                let ext = path.extension()?.to_ascii_lowercase();
                if ext == "mp4" || ext == "mov" {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();
        paths.sort();

        assert!(
            !paths.is_empty(),
            "no .mp4 / .mov files found in {}",
            dir.display()
        );

        log::info!("found {} video file(s) in {}", paths.len(), dir.display());

        let mut slots = Vec::with_capacity(paths.len());
        for (i, path) in paths.iter().enumerate() {
            log::info!("  [{}] opening {}", i, path.display());
            match VideoFileSource::open(path, &render_context.adapter, &render_context.device) {
                Ok(source) => {
                    slots.push(VideoSlot {
                        source,
                        texture_id: TextureId(i as u64),
                    });
                }
                Err(err) => {
                    log::warn!("  [{}] skipping {}: {err}", i, path.display());
                }
            }
        }

        assert!(
            !slots.is_empty(),
            "all video files in {} failed to open",
            dir.display()
        );

        slots
    }
}
