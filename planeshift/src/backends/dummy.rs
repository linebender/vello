// Copyright 2018 the Vello Authors and The Pathfinder Project Developers
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Dummy implementation that integrates with nothing

use image::RgbaImage;

#[cfg(feature = "enable-winit")]
use winit::window::Window;

use crate::{Connection, ConnectionError, LayerContainerInfo};
use crate::{LayerGeometryInfo, LayerId, LayerMap, LayerSurfaceInfo, LayerTreeInfo};
use crate::{Promise, Rect};

pub struct Backend {
    native_component: LayerMap<()>,
}

impl crate::Backend for Backend {
    type NativeConnection = ();
    type Host = ();

    fn new(_connection: Connection<Self::NativeConnection>) -> Result<Backend, ConnectionError> {
        Ok(Backend {
            native_component: LayerMap::new(),
        })
    }

    fn begin_transaction(&self) {
        unimplemented!();
    }

    fn end_transaction(
        &mut self,
        _promise: &Promise<()>,
        _: &LayerMap<LayerTreeInfo>,
        _: &LayerMap<LayerContainerInfo>,
        _: &LayerMap<LayerGeometryInfo>,
        _: &LayerMap<LayerSurfaceInfo>,
    ) {
        unimplemented!();
    }

    fn add_container_layer(&mut self, new_layer: LayerId) {
        self.native_component.add(new_layer, ());
    }

    fn add_surface_layer(&mut self, new_layer: LayerId) {
        self.add_container_layer(new_layer);
    }

    fn delete_layer(&mut self, layer: LayerId) {
        self.native_component.remove_if_present(layer);
    }

    fn insert_before(
        &mut self,
        _parent: LayerId,
        _new_child: LayerId,
        _reference: Option<LayerId>,
        _tree_component: &LayerMap<LayerTreeInfo>,
        _container_component: &LayerMap<LayerContainerInfo>,
        _geometry_component: &LayerMap<LayerGeometryInfo>,
    ) {
        unimplemented!();
    }

    fn remove_from_superlayer(
        &mut self,
        _layer: LayerId,
        _: LayerId,
        _: &LayerMap<LayerTreeInfo>,
        _: &LayerMap<LayerGeometryInfo>,
    ) {
        unimplemented!();
    }

    // Increases the reference count of `hosting_view`.
    unsafe fn host_layer(
        &mut self,
        _layer: LayerId,
        _host: Self::Host,
        _tree_component: &LayerMap<LayerTreeInfo>,
        _container_component: &LayerMap<LayerContainerInfo>,
        _geometry_component: &LayerMap<LayerGeometryInfo>,
    ) {
        unimplemented!();
    }

    fn unhost_layer(&mut self, _layer: LayerId) {
        unimplemented!();
    }

    fn set_layer_bounds(
        &mut self,
        _layer: LayerId,
        _: &Rect<f32>,
        _tree_component: &LayerMap<LayerTreeInfo>,
        _: &LayerMap<LayerContainerInfo>,
        _geometry_component: &LayerMap<LayerGeometryInfo>,
    ) {
        unimplemented!();
    }

    fn set_layer_surface_options(
        &mut self,
        _layer: LayerId,
        _surface_component: &LayerMap<LayerSurfaceInfo>,
    ) {
        unimplemented!();
    }

    // Screenshots

    fn screenshot_hosted_layer(
        &mut self,
        _layer: LayerId,
        _transaction_promise: &Promise<()>,
        _: &LayerMap<LayerTreeInfo>,
        _: &LayerMap<LayerContainerInfo>,
        _: &LayerMap<LayerGeometryInfo>,
        _: &LayerMap<LayerSurfaceInfo>,
    ) -> Promise<RgbaImage> {
        unimplemented!()
    }

    // `winit` integration

    #[cfg(feature = "enable-winit")]
    fn host_layer_in_window(
        &mut self,
        _layer: LayerId,
        _window: &Window,
        _tree_component: &LayerMap<LayerTreeInfo>,
        _container_component: &LayerMap<LayerContainerInfo>,
        _geometry_component: &LayerMap<LayerGeometryInfo>,
    ) -> Result<(), ()> {
        unimplemented!();
    }
}
