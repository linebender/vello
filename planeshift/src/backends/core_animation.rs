// Copyright 2018 the Vello Authors and The Pathfinder Project Developers
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Core Animation native system implementation.

use block::ConcreteBlock;
use cocoa::base::{NO, YES, id, nil};
use cocoa::foundation::{NSPoint, NSRect, NSSize};
use cocoa::quartzcore::{CALayer, transaction};
use core_graphics::base::CGFloat;
use core_graphics::geometry::{CG_ZERO_POINT, CGPoint, CGRect, CGSize};
use core_graphics::window::{self, CGWindowID, kCGWindowImageBestResolution};
use core_graphics::window::{kCGWindowImageBoundsIgnoreFraming, kCGWindowListOptionAll};
use image::RgbaImage;
use objc::{msg_send, sel, sel_impl};
use std::sync::Mutex;

#[cfg(feature = "enable-winit")]
use winit::{
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::Window,
};

use crate::{Connection, ConnectionError, LayerContainerInfo};
use crate::{LayerGeometryInfo, LayerId, LayerMap, LayerParent, LayerSurfaceInfo, LayerTreeInfo};
use crate::{Promise, Rect, SurfaceOptions};

pub struct Backend {
    native_component: LayerMap<NativeInfo>,
}

impl crate::Backend for Backend {
    type NativeConnection = ();
    type Host = id;

    fn new(_connection: Connection<Self::NativeConnection>) -> Result<Backend, ConnectionError> {
        Ok(Backend {
            native_component: LayerMap::new(),
        })
    }

    fn begin_transaction(&self) {
        transaction::begin();

        // Disable implicit animations.
        transaction::set_disable_actions(true);
    }

    fn end_transaction(
        &mut self,
        promise: &Promise<()>,
        _: &LayerMap<LayerTreeInfo>,
        _: &LayerMap<LayerContainerInfo>,
        _: &LayerMap<LayerGeometryInfo>,
        _: &LayerMap<LayerSurfaceInfo>,
    ) {
        let promise = Mutex::new(Some((*promise).clone()));
        transaction::set_completion_block(ConcreteBlock::new(move || {
            (*promise.lock().unwrap()).take().unwrap().resolve(())
        }));

        transaction::commit();
    }

    fn add_container_layer(&mut self, new_layer: LayerId) {
        let layer = CALayer::new();
        layer.set_anchor_point(&CG_ZERO_POINT);

        self.native_component.add(
            new_layer,
            NativeInfo {
                host: nil,
                core_animation_layer: layer,
            },
        );
    }

    fn add_surface_layer(&mut self, new_layer: LayerId) {
        self.add_container_layer(new_layer);
    }

    fn delete_layer(&mut self, layer: LayerId) {
        self.native_component.remove_if_present(layer);
    }

    fn insert_before(
        &mut self,
        parent: LayerId,
        new_child: LayerId,
        reference: Option<LayerId>,
        tree_component: &LayerMap<LayerTreeInfo>,
        container_component: &LayerMap<LayerContainerInfo>,
        geometry_component: &LayerMap<LayerGeometryInfo>,
    ) {
        let parent = &self.native_component[parent].core_animation_layer;
        let new_core_animation_child = &self.native_component[new_child].core_animation_layer;
        match reference {
            None => parent.add_sublayer(new_core_animation_child),
            Some(reference) => {
                let reference = &self.native_component[reference].core_animation_layer;
                parent.insert_sublayer_below(new_core_animation_child, reference);
            }
        }

        self.update_layer_subtree_bounds(
            new_child,
            tree_component,
            container_component,
            geometry_component,
        );
    }

    fn remove_from_superlayer(
        &mut self,
        layer: LayerId,
        _: LayerId,
        _: &LayerMap<LayerTreeInfo>,
        _: &LayerMap<LayerGeometryInfo>,
    ) {
        self.native_component[layer]
            .core_animation_layer
            .remove_from_superlayer()
    }

    // Increases the reference count of `hosting_view`.
    unsafe fn host_layer(
        &mut self,
        layer: LayerId,
        host: Self::Host,
        tree_component: &LayerMap<LayerTreeInfo>,
        container_component: &LayerMap<LayerContainerInfo>,
        geometry_component: &LayerMap<LayerGeometryInfo>,
    ) {
        let native_component = &mut self.native_component[layer];
        debug_assert_eq!(native_component.host, nil);

        let core_animation_layer = &native_component.core_animation_layer;
        let _: id = msg_send![host, retain];
        let _: () = msg_send![host, setLayer:core_animation_layer.id()];
        let _: () = msg_send![host, setWantsLayer:YES];

        native_component.host = host;

        self.update_layer_subtree_bounds(
            layer,
            tree_component,
            container_component,
            geometry_component,
        );
    }

    fn unhost_layer(&mut self, layer: LayerId) {
        let native_component = &mut self.native_component[layer];
        debug_assert_ne!(native_component.host, nil);

        unsafe {
            let _: () = msg_send![native_component.host, setWantsLayer:NO];
            let _: () = msg_send![native_component.host, setLayer:nil];
            let _: id = msg_send![native_component.host, release];
        }

        native_component.host = nil;
    }

    fn set_layer_bounds(
        &mut self,
        layer: LayerId,
        _: &Rect<f32>,
        tree_component: &LayerMap<LayerTreeInfo>,
        _: &LayerMap<LayerContainerInfo>,
        geometry_component: &LayerMap<LayerGeometryInfo>,
    ) {
        self.update_layer_bounds(layer, tree_component, geometry_component);
    }

    fn set_layer_surface_options(
        &mut self,
        layer: LayerId,
        surface_component: &LayerMap<LayerSurfaceInfo>,
    ) {
        let surface_options = surface_component[layer].options;

        let core_animation_layer = &mut self.native_component[layer].core_animation_layer;
        let opaque = surface_options.contains(SurfaceOptions::OPAQUE);
        core_animation_layer.set_opaque(opaque);
        core_animation_layer.set_contents_opaque(opaque);
    }

    // Screenshots

    fn screenshot_hosted_layer(
        &mut self,
        layer: LayerId,
        transaction_promise: &Promise<()>,
        _: &LayerMap<LayerTreeInfo>,
        _: &LayerMap<LayerContainerInfo>,
        _: &LayerMap<LayerGeometryInfo>,
        _: &LayerMap<LayerSurfaceInfo>,
    ) -> Promise<RgbaImage> {
        let result_promise = Promise::new();
        let result_promise_to_return = result_promise.clone();

        let hosting_view = self.native_component[layer].host as usize;
        transaction_promise.then(Box::new(move |()| {
            let hosting_view: id = hosting_view as id;
            let image;
            unsafe {
                let view_bounds: NSRect = msg_send![hosting_view, bounds];
                let mut view_frame: NSRect =
                    msg_send![hosting_view, convertRect:view_bounds toView:nil];

                let window: id = msg_send![hosting_view, window];
                let window_id: CGWindowID = msg_send![window, windowNumber];

                let window_frame: NSRect = msg_send![window, frame];
                view_frame.origin.x += window_frame.origin.x;
                view_frame.origin.y += window_frame.origin.y;

                let screen: id = msg_send![window, screen];
                let screen_frame: NSRect = msg_send![screen, frame];
                let screen_rect = CGRect::new(
                    &CGPoint::new(
                        view_frame.origin.x,
                        screen_frame.size.height - view_frame.origin.y - view_frame.size.height,
                    ),
                    &CGSize::new(view_frame.size.width, view_frame.size.height),
                );

                image = window::create_image(
                    screen_rect,
                    kCGWindowListOptionAll,
                    window_id,
                    kCGWindowImageBoundsIgnoreFraming | kCGWindowImageBestResolution,
                )
                .unwrap();
            }

            let (width, height) = (
                u32::try_from(image.width()).unwrap(),
                u32::try_from(image.height()).unwrap(),
            );
            let mut data = image.data().bytes().to_vec();
            data.chunks_mut(4).for_each(|pixel| pixel.swap(0, 2));
            result_promise.resolve(RgbaImage::from_vec(width, height, data).unwrap());
        }));

        result_promise_to_return
    }

    // `winit` integration

    #[cfg(feature = "enable-winit")]
    fn host_layer_in_window(
        &mut self,
        layer: LayerId,
        window: &Window,
        tree_component: &LayerMap<LayerTreeInfo>,
        container_component: &LayerMap<LayerContainerInfo>,
        geometry_component: &LayerMap<LayerGeometryInfo>,
    ) -> Result<(), ()> {
        let nsview = match window.window_handle().unwrap().as_raw() {
            RawWindowHandle::AppKit(handle) => handle.ns_view.cast().as_ptr(),
            _ => panic!("Unsupported platform."),
        };
        unsafe {
            self.host_layer(
                layer,
                nsview,
                tree_component,
                container_component,
                geometry_component,
            );
        }
        Ok(())
    }
}

impl Backend {
    fn hosting_view(&self, layer: LayerId, tree_component: &LayerMap<LayerTreeInfo>) -> Option<id> {
        match tree_component.get(layer) {
            None => None,
            Some(LayerTreeInfo {
                parent: LayerParent::Layer(parent_layer),
                ..
            }) => self.hosting_view(*parent_layer, tree_component),
            Some(LayerTreeInfo {
                parent: LayerParent::NativeHost,
                ..
            }) => Some(self.native_component[layer].host),
        }
    }

    fn update_layer_bounds_with_hosting_view(
        &mut self,
        layer: LayerId,
        hosting_view: id,
        geometry_component: &LayerMap<LayerGeometryInfo>,
    ) {
        let new_bounds: Rect<CGFloat> = match geometry_component.get(layer) {
            None => return,
            Some(geometry_info) => geometry_info.bounds.to_f64(),
        };

        let new_appkit_bounds = NSRect::new(
            NSPoint::new(new_bounds.origin.x, new_bounds.origin.y),
            NSSize::new(new_bounds.size.width, new_bounds.size.height),
        );
        let new_appkit_bounds: NSRect =
            unsafe { msg_send![hosting_view, convertRectFromBacking:new_appkit_bounds] };

        let new_core_animation_bounds = CGRect::new(
            &CG_ZERO_POINT,
            &CGSize::new(new_appkit_bounds.size.width, new_appkit_bounds.size.height),
        );

        let core_animation_layer = &self.native_component[layer].core_animation_layer;
        core_animation_layer.set_bounds(&new_core_animation_bounds);
        core_animation_layer.set_position(&CGPoint::new(
            new_appkit_bounds.origin.x,
            new_appkit_bounds.origin.y,
        ));
    }

    fn update_layer_subtree_bounds_with_hosting_view(
        &mut self,
        layer: LayerId,
        hosting_view: id,
        tree_component: &LayerMap<LayerTreeInfo>,
        container_component: &LayerMap<LayerContainerInfo>,
        geometry_component: &LayerMap<LayerGeometryInfo>,
    ) {
        self.update_layer_bounds_with_hosting_view(layer, hosting_view, geometry_component);

        if let Some(container_info) = container_component.get(layer) {
            let mut maybe_kid = container_info.first_child;
            while let Some(kid) = maybe_kid {
                self.update_layer_subtree_bounds_with_hosting_view(
                    kid,
                    hosting_view,
                    tree_component,
                    container_component,
                    geometry_component,
                );
                maybe_kid = tree_component[kid].next_sibling;
            }
        }
    }

    fn update_layer_subtree_bounds(
        &mut self,
        layer: LayerId,
        tree_component: &LayerMap<LayerTreeInfo>,
        container_component: &LayerMap<LayerContainerInfo>,
        geometry_component: &LayerMap<LayerGeometryInfo>,
    ) {
        if let Some(hosting_view) = self.hosting_view(layer, tree_component) {
            self.update_layer_subtree_bounds_with_hosting_view(
                layer,
                hosting_view,
                tree_component,
                container_component,
                geometry_component,
            )
        }
    }

    fn update_layer_bounds(
        &mut self,
        layer: LayerId,
        tree_component: &LayerMap<LayerTreeInfo>,
        geometry_component: &LayerMap<LayerGeometryInfo>,
    ) {
        if let Some(hosting_view) = self.hosting_view(layer, tree_component) {
            self.update_layer_bounds_with_hosting_view(layer, hosting_view, geometry_component)
        }
    }
}

struct NativeInfo {
    host: id,
    core_animation_layer: CALayer,
}

pub type LayerNativeHost = id;

impl Default for NativeInfo {
    fn default() -> NativeInfo {
        NativeInfo {
            host: nil,
            core_animation_layer: CALayer::new(),
        }
    }
}

impl Drop for NativeInfo {
    fn drop(&mut self) {
        unsafe {
            if !std::ptr::eq(self.host, nil) {
                let _: id = msg_send![self.host, release];
                self.host = nil;
            }
        }
    }
}
