// Derived from vello_editor
// Copyright 2024 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! An example of running Vello Hybrid on Android

#![deny(unsafe_op_in_unsafe_fn)]

use access_ids::WINDOW_ID;
use accesskit::{ActionRequest, ActivationHandler, Node, Role, Tree, TreeUpdate};
use accesskit_android::ActionHandlerWithAndroidContext;
use android_view::{
    jni::{
        JNIEnv, JavaVM,
        objects::JObject,
        sys::{JNI_VERSION_1_6, JavaVM as RawJavaVM, jfloat, jint, jlong},
    },
    ndk::native_window::NativeWindow,
    *,
};
use anyhow::Result;
use log::LevelFilter;
use peniko::kurbo::{Affine, Stroke};
use std::{collections::HashMap, ffi::c_void};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use vello_common::pico_svg::{Item, PicoSvg};
use vello_hybrid::{RenderParams, Renderer, RendererOptions, Scene};
use wgpu::{
    self, DeviceDescriptor, Features, InstanceDescriptor, Limits, RenderPassDescriptor,
    RequestAdapterOptions, TextureFormat,
    rwh::{DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, WindowHandle},
};

mod access_ids;

// From VelloCompose
struct AndroidWindowHandle {
    window: NativeWindow,
}

impl HasDisplayHandle for AndroidWindowHandle {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        Ok(DisplayHandle::android())
    }
}

impl HasWindowHandle for AndroidWindowHandle {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        self.window.window_handle()
    }
}

struct EditorAccessTreeSource<'a> {
    _render_surface: &'a Option<RenderSurface<'static>>,
}

impl EditorAccessTreeSource<'_> {
    fn build_initial_tree(&mut self) -> TreeUpdate {
        let mut update = TreeUpdate {
            nodes: vec![],
            tree: Some(Tree::new(WINDOW_ID)),
            focus: WINDOW_ID,
        };
        let node = Node::new(Role::Window);
        update.nodes.push((WINDOW_ID, node));
        update
    }
}

impl ActivationHandler for EditorAccessTreeSource<'_> {
    fn request_initial_tree(&mut self) -> Option<TreeUpdate> {
        Some(self.build_initial_tree())
    }
}

struct EditorAccessActionHandler<'a> {
    _render_surface: &'a Option<RenderSurface<'static>>,
}

impl ActionHandlerWithAndroidContext for EditorAccessActionHandler<'_> {
    fn do_action<'local>(
        &mut self,
        _env: &mut JNIEnv<'local>,
        _view: &JObject<'local>,
        _request: ActionRequest,
    ) {
    }
}

struct RenderSurface<'a> {
    surface: wgpu::Surface<'a>,
    width: u16,
    height: u16,
    format: TextureFormat,
    // The scene cares about its height, so we need to handle it here
    scene: Scene,
}

struct RendererDevice {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderers: HashMap<TextureFormat, Renderer>,
}

impl RendererDevice {
    async fn new(instance: &wgpu::Instance, surface: &wgpu::Surface<'_>) -> Self {
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                compatible_surface: Some(surface),
                ..Default::default()
            })
            .await
            .expect("Could find an adapter, as Vulkan is mandatory since Android API level 24");
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("sparse_strips_demo.device"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                },
                None,
            )
            .await
            .expect("Device would be wgpu compatible");
        device.on_uncaptured_error(Box::new(|err| {
            tracing::error!("Error from wgpu: {err}");
            panic!("Android View should be catching the panic from wgpu, but doesn't currently");
        }));
        Self {
            adapter,
            device,
            queue,
            renderers: HashMap::new(),
        }
    }
}

struct DemoViewPeer {
    instance: wgpu::Instance,
    devices: Option<RendererDevice>,

    /// State for our example where we store the winit Window and the wgpu Surface.
    render_surface: Option<RenderSurface<'static>>,

    access_adapter: accesskit_android::Adapter,
    svg: PicoSvg,
}

impl DemoViewPeer {
    fn enqueue_render_if_needed<'local>(&mut self, env: &mut JNIEnv<'local>, view: &View<'local>) {
        if self.render_surface.is_none() {
            return;
        }
        view.post_frame_callback(env);
    }

    fn render<'local>(&mut self, env: &mut JNIEnv<'local>, view: &View<'local>) {
        let Some(surface) = self.render_surface.as_mut() else {
            log::warn!("Tried to render without a scene");
            return;
        };
        let Some(devices) = self.devices.as_mut() else {
            log::warn!("Tried to render without a device");
            return;
        };

        // Get the window size.
        let width = surface.width;
        let height = surface.height;
        tracing::info!("Drawing at: w:{width}xh:{height}");
        // Get the surface's texture.
        let surface_texture = surface
            .surface
            .get_current_texture()
            .expect("failed to get surface texture");

        // Empty the scene of objects to draw. You could create a new Scene each time, but in this case
        // the same Scene is reused so that the underlying memory allocation can also be reused.
        surface.scene.reset();

        fn render_svg(ctx: &mut Scene, scale: f64, items: &[Item]) {
            fn render_svg_inner(ctx: &mut Scene, items: &[Item], transform: Affine) {
                ctx.set_transform(transform);
                for item in items {
                    match item {
                        Item::Fill(fill_item) => {
                            ctx.set_paint(fill_item.color.into());
                            ctx.fill_path(&fill_item.path);
                        }
                        Item::Stroke(stroke_item) => {
                            let style = Stroke::new(stroke_item.width);
                            ctx.set_stroke(style);
                            ctx.set_paint(stroke_item.color.into());
                            ctx.stroke_path(&stroke_item.path);
                        }
                        Item::Group(group_item) => {
                            render_svg_inner(
                                ctx,
                                &group_item.children,
                                transform * group_item.affine,
                            );
                            ctx.set_transform(transform);
                        }
                    }
                }
            }

            render_svg_inner(ctx, items, Affine::scale(scale));
        }

        render_svg(&mut surface.scene, 1.0, &self.svg.items);
        let view_class = env.get_object_class(&view.0).unwrap();
        self.access_adapter.update_if_active(
            || TreeUpdate {
                nodes: vec![],
                tree: None,
                focus: WINDOW_ID,
            },
            env,
            &view_class,
            &view.0,
        );

        let params = RenderParams {
            width: width.into(),
            height: height.into(),
        };
        let renderer = devices.renderers.entry(surface.format).or_insert_with(|| {
            Renderer::new(
                &devices.device,
                &RendererOptions {
                    format: surface.format,
                },
            )
        });
        // Render to the surface's texture.
        renderer.prepare(&devices.device, &devices.queue, &surface.scene, &params);
        let surface_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        // Copy texture to buffer
        let mut encoder = devices
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Vello Render To Buffer"),
            });
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            renderer.render(&surface.scene, &mut pass, &params);
        }
        devices.queue.submit([encoder.finish()]);
        surface_texture.present();

        devices.device.poll(wgpu::Maintain::Poll);
    }
}

impl ViewPeer for DemoViewPeer {
    // TODO

    fn on_focus_changed<'local>(
        &mut self,
        env: &mut JNIEnv<'local>,
        view: &View<'local>,
        _gain_focus: bool,
        _direction: jint,
        _previously_focused_rect: Option<&Rect<'local>>,
    ) {
        self.enqueue_render_if_needed(env, view);
    }

    fn surface_changed<'local>(
        &mut self,
        env: &mut JNIEnv<'local>,
        view: &View<'local>,
        holder: &SurfaceHolder<'local>,
        _format: jint,
        width: jint,
        height: jint,
    ) {
        // TODO: If this is the same native window as last time, we only need to reconfigure?
        let window = holder.surface(env).to_native_window(env);
        self.render_surface = None;
        let wgpu_surface = self
            .instance
            .create_surface(wgpu::SurfaceTarget::from(AndroidWindowHandle { window }))
            .expect("Error creating surface");
        if !self
            .devices
            .as_ref()
            .is_some_and(|it| it.adapter.is_surface_supported(&wgpu_surface))
        {
            // We recreate the device here because
            // It's reasonable to use block_on because this is "fake async"; none of these futures are actually async.
            self.devices = Some(pollster::block_on(RendererDevice::new(
                &self.instance,
                &wgpu_surface,
            )));
        }
        let devices = self.devices.as_mut().expect("Just created if not present");
        let capabilities = wgpu_surface.get_capabilities(&devices.adapter);

        let present_mode = if capabilities
            .present_modes
            .contains(&wgpu::PresentMode::Mailbox)
        {
            // We only start a render upon vsync start, so MailBox is appropriate.
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::AutoVsync
        };

        let width: u16 = width.try_into().expect("Reasonable height");
        let height: u16 = height.try_into().expect("Reasonable height");
        let mut config = wgpu_surface
            .get_default_config(&devices.adapter, width.into(), height.into())
            .expect("We just validated that this surface is supported by this adapter");
        /* We're on Android, see https://raphlinus.github.io/ui/graphics/gpu/2021/10/22/swapchain-frame-pacing.html */
        config.desired_maximum_frame_latency = 3;
        config.present_mode = present_mode;
        wgpu_surface.configure(&devices.device, &config);

        self.render_surface = Some(RenderSurface {
            surface: wgpu_surface,
            width,
            height,
            format: config.format,
            scene: Scene::new(width, height),
        });
        devices.renderers.entry(config.format).or_insert_with(|| {
            Renderer::new(
                &devices.device,
                &RendererOptions {
                    format: config.format,
                },
            )
        });
        self.render(env, view);
    }

    fn surface_destroyed<'local>(
        &mut self,
        env: &mut JNIEnv<'local>,
        view: &View<'local>,
        _holder: &SurfaceHolder<'local>,
    ) {
        self.render_surface = None;
        view.remove_frame_callback(env);
        view.remove_delayed_callbacks(env);
    }

    fn do_frame<'local>(
        &mut self,
        env: &mut JNIEnv<'local>,
        view: &View<'local>,
        _frame_time_nanos: jlong,
    ) {
        self.render(env, view);
    }

    fn delayed_callback<'local>(&mut self, env: &mut JNIEnv<'local>, view: &View<'local>) {
        self.enqueue_render_if_needed(env, view);
    }

    fn populate_accessibility_node_info<'local>(
        &mut self,
        env: &mut JNIEnv<'local>,
        view: &View<'local>,
        host_screen_x: jint,
        host_screen_y: jint,
        virtual_view_id: jint,
        node_info: &JObject<'local>,
    ) -> bool {
        let mut tree_source = EditorAccessTreeSource {
            _render_surface: &self.render_surface,
        };
        self.access_adapter
            .populate_node_info(
                &mut tree_source,
                env,
                &view.0,
                host_screen_x,
                host_screen_y,
                virtual_view_id,
                node_info,
            )
            .unwrap()
    }

    fn input_focus<'local>(&mut self, env: &mut JNIEnv<'local>, view: &View<'local>) -> jint {
        let mut tree_source = EditorAccessTreeSource {
            _render_surface: &self.render_surface,
        };
        self.access_adapter
            .input_focus(&mut tree_source, env, &view.0)
    }

    fn virtual_view_at_point<'local>(
        &mut self,
        env: &mut JNIEnv<'local>,
        view: &View<'local>,
        x: jfloat,
        y: jfloat,
    ) -> jint {
        let mut tree_source = EditorAccessTreeSource {
            _render_surface: &self.render_surface,
        };
        self.access_adapter
            .virtual_view_at_point(&mut tree_source, env, &view.0, x, y)
    }

    fn perform_accessibility_action<'local>(
        &mut self,
        env: &mut JNIEnv<'local>,
        view: &View<'local>,
        virtual_view_id: jint,
        action: jint,
    ) -> bool {
        let mut action_handler = EditorAccessActionHandler {
            _render_surface: &self.render_surface,
        };
        self.access_adapter.perform_action(
            &mut action_handler,
            env,
            &view.0,
            virtual_view_id,
            action,
        )
    }

    fn accessibility_set_text_selection<'local>(
        &mut self,
        env: &mut JNIEnv<'local>,
        view: &View<'local>,
        virtual_view_id: jint,
        anchor: jint,
        focus: jint,
    ) -> bool {
        let mut action_handler = EditorAccessActionHandler {
            _render_surface: &self.render_surface,
        };
        let view_class = env.get_object_class(&view.0).unwrap();
        self.access_adapter.set_text_selection(
            &mut action_handler,
            env,
            &view_class,
            &view.0,
            virtual_view_id,
            anchor,
            focus,
        )
    }

    fn accessibility_collapse_text_selection<'local>(
        &mut self,
        env: &mut JNIEnv<'local>,
        view: &View<'local>,
        virtual_view_id: jint,
    ) -> bool {
        let mut action_handler = EditorAccessActionHandler {
            _render_surface: &self.render_surface,
        };
        let view_class = env.get_object_class(&view.0).unwrap();
        self.access_adapter.collapse_text_selection(
            &mut action_handler,
            env,
            &view_class,
            &view.0,
            virtual_view_id,
        )
    }

    fn accessibility_traverse_text<'local>(
        &mut self,
        env: &mut JNIEnv<'local>,
        view: &View<'local>,
        virtual_view_id: jint,
        granularity: jint,
        forward: bool,
        extend_selection: bool,
    ) -> bool {
        let mut action_handler = EditorAccessActionHandler {
            _render_surface: &self.render_surface,
        };
        let view_class = env.get_object_class(&view.0).unwrap();
        self.access_adapter.traverse_text(
            &mut action_handler,
            env,
            &view_class,
            &view.0,
            virtual_view_id,
            granularity,
            forward,
            extend_selection,
        )
    }
}

extern "system" fn new_view_peer<'local>(
    _env: JNIEnv<'local>,
    _view: View<'local>,
    _context: Context<'local>,
) -> jlong {
    let ghosttiger = include_str!("../../../examples/assets/Ghostscript_Tiger.svg");
    let svg = PicoSvg::load(ghosttiger, 6.0).expect("error parsing SVG");
    let peer = DemoViewPeer {
        devices: None,
        // Ideally, we'd use from_env_or_default, but it's not possible to set environment variables...
        instance: wgpu::Instance::new(&InstanceDescriptor::default()),
        render_surface: None,
        svg,
        access_adapter: accesskit_android::Adapter::default(),
    };
    register_view_peer(peer)
}

#[unsafe(no_mangle)]
/// # Safety
///
/// Ideally this is called by the Android system with reasonable values
pub unsafe extern "system" fn JNI_OnLoad(vm: *mut RawJavaVM, _: *mut c_void) -> jint {
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(LevelFilter::Trace)
            .with_tag("android-view-demo"),
    );
    // This will try to create a "log" logger, and error because one was already created above
    // We therefore ignore the error
    // Ideally, we'd only ignore the SetLoggerError, but the only way that's possible is to inspect
    // `Debug/Display` on the TryInitError, which is awful.
    let _ = tracing_subscriber::registry()
        .with(tracing_android_trace::AndroidTraceLayer::new())
        .try_init();

    let vm = unsafe { JavaVM::from_raw(vm) }.unwrap();
    let mut env = vm.get_env().unwrap();
    register_view_class(
        &mut env,
        "org/linebender/vello/sparse_strips_demo/DemoView",
        new_view_peer,
    );
    JNI_VERSION_1_6
}
