// Copyright 2018 the Vello Authors and The Pathfinder Project Developers
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A backend that tries one backend, and if it fails, tries the other.
//!
//! If backend A fails to initialize, then it tries to initialize backend B. Note that more than
//! two backends can be chained together by making backend A or backend B itself a `Chain`.

use image::RgbaImage;

#[cfg(feature = "enable-winit")]
use winit::window::Window;

use crate::Rect;
use crate::{Connection, ConnectionError, LayerContainerInfo};
use crate::{LayerGeometryInfo, LayerId, LayerMap, LayerSurfaceInfo, LayerTreeInfo, Promise};

pub enum Backend<A, B>
where
    A: crate::Backend,
    B: crate::Backend,
{
    A(A),
    B(B),
}

impl<A, B> crate::Backend for Backend<A, B>
where
    A: crate::Backend,
    B: crate::Backend,
{
    type NativeConnection = NativeConnection<A, B>;
    type Host = Host<A, B>;

    // Constructor

    fn new(connection: Connection<Self::NativeConnection>) -> Result<Self, ConnectionError> {
        match connection {
            Connection::Native(NativeConnection::A(native_connection)) => {
                Ok(Backend::A(A::new(Connection::Native(native_connection))?))
            }
            Connection::Native(NativeConnection::B(native_connection)) => {
                Ok(Backend::B(B::new(Connection::Native(native_connection))?))
            }
            #[cfg(feature = "enable-winit")]
            Connection::Winit() => todo!(),
            /*
            Connection::Winit(window_builder, event_loop) => {
                match A::new(Connection::Winit(window_builder, event_loop)) {
                    Ok(backend) => Ok(Backend::A(backend)),
                    Err(err) => {
                        match B::new(Connection::Winit(err.window_builder.unwrap(), event_loop)) {
                            Ok(backend) => Ok(Backend::B(backend)),
                            Err(err) => Err(err),
                        }
                    }
                }
            }
            */
        }
    }

    // Transactions

    fn begin_transaction(&self) {
        match *self {
            Backend::A(ref this) => this.begin_transaction(),
            Backend::B(ref this) => this.begin_transaction(),
        }
    }

    fn end_transaction(
        &mut self,
        promise: &Promise<()>,
        tree_component: &LayerMap<LayerTreeInfo>,
        container_component: &LayerMap<LayerContainerInfo>,
        geometry_component: &LayerMap<LayerGeometryInfo>,
        surface_component: &LayerMap<LayerSurfaceInfo>,
    ) {
        match *self {
            Backend::A(ref mut this) => this.end_transaction(
                promise,
                tree_component,
                container_component,
                geometry_component,
                surface_component,
            ),
            Backend::B(ref mut this) => this.end_transaction(
                promise,
                tree_component,
                container_component,
                geometry_component,
                surface_component,
            ),
        }
    }

    // Layer creation and destruction

    fn add_container_layer(&mut self, new_layer: LayerId) {
        match *self {
            Backend::A(ref mut this) => this.add_container_layer(new_layer),
            Backend::B(ref mut this) => this.add_container_layer(new_layer),
        }
    }

    fn add_surface_layer(&mut self, new_layer: LayerId) {
        match *self {
            Backend::A(ref mut this) => this.add_surface_layer(new_layer),
            Backend::B(ref mut this) => this.add_surface_layer(new_layer),
        }
    }

    fn delete_layer(&mut self, layer: LayerId) {
        match *self {
            Backend::A(ref mut this) => this.delete_layer(layer),
            Backend::B(ref mut this) => this.delete_layer(layer),
        }
    }

    // Layer tree management

    fn insert_before(
        &mut self,
        parent: LayerId,
        new_child: LayerId,
        reference: Option<LayerId>,
        tree_component: &LayerMap<LayerTreeInfo>,
        container_component: &LayerMap<LayerContainerInfo>,
        geometry_component: &LayerMap<LayerGeometryInfo>,
    ) {
        match *self {
            Backend::A(ref mut this) => this.insert_before(
                parent,
                new_child,
                reference,
                tree_component,
                container_component,
                geometry_component,
            ),
            Backend::B(ref mut this) => this.insert_before(
                parent,
                new_child,
                reference,
                tree_component,
                container_component,
                geometry_component,
            ),
        }
    }

    fn remove_from_superlayer(
        &mut self,
        layer: LayerId,
        parent: LayerId,
        tree_component: &LayerMap<LayerTreeInfo>,
        geometry_component: &LayerMap<LayerGeometryInfo>,
    ) {
        match *self {
            Backend::A(ref mut this) => {
                this.remove_from_superlayer(layer, parent, tree_component, geometry_component)
            }
            Backend::B(ref mut this) => {
                this.remove_from_superlayer(layer, parent, tree_component, geometry_component)
            }
        }
    }

    // Native hosting

    unsafe fn host_layer(
        &mut self,
        layer: LayerId,
        host: Self::Host,
        tree_component: &LayerMap<LayerTreeInfo>,
        container_component: &LayerMap<LayerContainerInfo>,
        geometry_component: &LayerMap<LayerGeometryInfo>,
    ) {
        match *self {
            Backend::A(ref mut this) => match host {
                Host::A(host) => unsafe {
                    this.host_layer(
                        layer,
                        host,
                        tree_component,
                        container_component,
                        geometry_component,
                    )
                },
                Host::B(_) => panic!("host_layer(): mismatched backend and host"),
            },
            Backend::B(ref mut this) => match host {
                Host::B(host) => unsafe {
                    this.host_layer(
                        layer,
                        host,
                        tree_component,
                        container_component,
                        geometry_component,
                    )
                },
                Host::A(_) => panic!("host_layer(): mismatched backend and host"),
            },
        }
    }

    fn unhost_layer(&mut self, layer: LayerId) {
        match *self {
            Backend::A(ref mut this) => this.unhost_layer(layer),
            Backend::B(ref mut this) => this.unhost_layer(layer),
        }
    }

    // Geometry

    fn set_layer_bounds(
        &mut self,
        layer: LayerId,
        old_bounds: &Rect<f32>,
        tree_component: &LayerMap<LayerTreeInfo>,
        container_component: &LayerMap<LayerContainerInfo>,
        geometry_component: &LayerMap<LayerGeometryInfo>,
    ) {
        match *self {
            Backend::A(ref mut this) => this.set_layer_bounds(
                layer,
                old_bounds,
                tree_component,
                container_component,
                geometry_component,
            ),
            Backend::B(ref mut this) => this.set_layer_bounds(
                layer,
                old_bounds,
                tree_component,
                container_component,
                geometry_component,
            ),
        }
    }

    // Miscellaneous layer flags

    fn set_layer_surface_options(
        &mut self,
        layer: LayerId,
        surface_component: &LayerMap<LayerSurfaceInfo>,
    ) {
        match *self {
            Backend::A(ref mut this) => this.set_layer_surface_options(layer, surface_component),
            Backend::B(ref mut this) => this.set_layer_surface_options(layer, surface_component),
        }
    }

    // Screenshots

    fn screenshot_hosted_layer(
        &mut self,
        layer: LayerId,
        transaction_promise: &Promise<()>,
        tree_component: &LayerMap<LayerTreeInfo>,
        container_component: &LayerMap<LayerContainerInfo>,
        geometry_component: &LayerMap<LayerGeometryInfo>,
        surface_component: &LayerMap<LayerSurfaceInfo>,
    ) -> Promise<RgbaImage> {
        match *self {
            Backend::A(ref mut this) => this.screenshot_hosted_layer(
                layer,
                transaction_promise,
                tree_component,
                container_component,
                geometry_component,
                surface_component,
            ),
            Backend::B(ref mut this) => this.screenshot_hosted_layer(
                layer,
                transaction_promise,
                tree_component,
                container_component,
                geometry_component,
                surface_component,
            ),
        }
    }

    // `winit` integration

    #[cfg(feature = "enable-winit")]
    fn window(&self) -> Option<&Window> {
        match *self {
            Backend::A(ref this) => this.window(),
            Backend::B(ref this) => this.window(),
        }
    }

    #[cfg(feature = "enable-winit")]
    fn host_layer_in_window(
        &mut self,
        layer: LayerId,
        tree_component: &LayerMap<LayerTreeInfo>,
        container_component: &LayerMap<LayerContainerInfo>,
        geometry_component: &LayerMap<LayerGeometryInfo>,
    ) -> Result<(), ()> {
        match *self {
            Backend::A(ref mut this) => this.host_layer_in_window(
                layer,
                tree_component,
                container_component,
                geometry_component,
            ),
            Backend::B(ref mut this) => this.host_layer_in_window(
                layer,
                tree_component,
                container_component,
                geometry_component,
            ),
        }
    }
}

pub enum NativeConnection<A, B>
where
    A: crate::Backend,
    B: crate::Backend,
{
    A(A::NativeConnection),
    B(B::NativeConnection),
}

pub enum Host<A, B>
where
    A: crate::Backend,
    B: crate::Backend,
{
    A(A::Host),
    B(B::Host),
}
