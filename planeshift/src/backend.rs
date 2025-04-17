// Copyright 2018 the Vello Authors and The Pathfinder Project Developers
// SPDX-License-Identifier: Apache-2.0 OR MIT

use image::RgbaImage;

#[cfg(feature = "enable-winit")]
use winit::window::Window;

use crate::Rect;
use crate::{Connection, ConnectionError, LayerContainerInfo};
use crate::{LayerGeometryInfo, LayerId, LayerMap, LayerSurfaceInfo, LayerTreeInfo, Promise};

// Backend definition

pub trait Backend: Sized {
    type NativeConnection;
    type Host;

    // Constructor
    fn new(connection: Connection<Self::NativeConnection>) -> Result<Self, ConnectionError>;

    // Transactions
    fn begin_transaction(&self);
    fn end_transaction(
        &mut self,
        promise: &Promise<()>,
        tree_component: &LayerMap<LayerTreeInfo>,
        container_component: &LayerMap<LayerContainerInfo>,
        geometry_component: &LayerMap<LayerGeometryInfo>,
        surface_component: &LayerMap<LayerSurfaceInfo>,
    );

    // Layer creation and destruction
    fn add_container_layer(&mut self, new_layer: LayerId);
    fn add_surface_layer(&mut self, new_layer: LayerId);
    fn delete_layer(&mut self, layer: LayerId);

    // Layer tree management
    fn insert_before(
        &mut self,
        parent: LayerId,
        new_child: LayerId,
        reference: Option<LayerId>,
        tree_component: &LayerMap<LayerTreeInfo>,
        container_component: &LayerMap<LayerContainerInfo>,
        geometry_component: &LayerMap<LayerGeometryInfo>,
    );
    fn remove_from_superlayer(
        &mut self,
        layer: LayerId,
        parent: LayerId,
        tree_component: &LayerMap<LayerTreeInfo>,
        geometry_component: &LayerMap<LayerGeometryInfo>,
    );

    // Native hosting
    unsafe fn host_layer(
        &mut self,
        layer: LayerId,
        host: Self::Host,
        tree_component: &LayerMap<LayerTreeInfo>,
        container_component: &LayerMap<LayerContainerInfo>,
        geometry_component: &LayerMap<LayerGeometryInfo>,
    );
    fn unhost_layer(&mut self, layer: LayerId);

    // Geometry
    fn set_layer_bounds(
        &mut self,
        layer: LayerId,
        old_bounds: &Rect<f32>,
        tree_component: &LayerMap<LayerTreeInfo>,
        container_component: &LayerMap<LayerContainerInfo>,
        geometry_component: &LayerMap<LayerGeometryInfo>,
    );

    // Miscellaneous layer flags
    fn set_layer_surface_options(
        &mut self,
        layer: LayerId,
        surface_component: &LayerMap<LayerSurfaceInfo>,
    );

    // Screenshots
    fn screenshot_hosted_layer(
        &mut self,
        layer: LayerId,
        transaction_promise: &Promise<()>,
        tree_component: &LayerMap<LayerTreeInfo>,
        container_component: &LayerMap<LayerContainerInfo>,
        geometry_component: &LayerMap<LayerGeometryInfo>,
        surface_component: &LayerMap<LayerSurfaceInfo>,
    ) -> Promise<RgbaImage>;

    // `winit` integration
    #[cfg(feature = "enable-winit")]
    fn window(&self) -> Option<&Window>;
    #[cfg(feature = "enable-winit")]
    fn host_layer_in_window(
        &mut self,
        layer: LayerId,
        tree_component: &LayerMap<LayerTreeInfo>,
        container_component: &LayerMap<LayerContainerInfo>,
        geometry_component: &LayerMap<LayerGeometryInfo>,
    ) -> Result<(), ()>;
}
