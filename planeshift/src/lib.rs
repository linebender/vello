// Copyright 2018 the Vello Authors and The Pathfinder Project Developers
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(
    dead_code,
    missing_debug_implementations,
    missing_docs,
    reason = "Deferred"
)]
#![expect(
    clippy::mem_replace_option_with_none,
    clippy::mem_replace_with_default,
    clippy::missing_assert_message,
    clippy::missing_safety_doc,
    clippy::new_without_default,
    clippy::semicolon_if_nothing_returned,
    clippy::use_self,
    reason = "Deferred"
)]
#![cfg_attr(
    feature = "enable-winit",
    expect(clippy::result_unit_err, clippy::todo, reason = "Deferred")
)]
#![cfg_attr(target_vendor = "apple", expect(unexpected_cfgs, reason = "Deferred"))]

use image::RgbaImage;
use std::fmt::{self, Debug, Formatter};
use std::mem;
use std::ops::{Index, IndexMut};
use std::sync::{Arc, Mutex};

#[cfg(feature = "enable-winit")]
use winit::window::Window;

use crate::backend::Backend;

pub mod backend;
pub mod backends;

pub type Rect<T> = euclid::Rect<T, euclid::UnknownUnit>;

/// Manages all the layers.
pub struct LayerContext<B = backends::default::Backend>
where
    B: Backend,
{
    next_layer_id: LayerId,
    transaction: Option<TransactionInfo>,

    tree_component: LayerMap<LayerTreeInfo>,
    container_component: LayerMap<LayerContainerInfo>,
    geometry_component: LayerMap<LayerGeometryInfo>,
    surface_component: LayerMap<LayerSurfaceInfo>,

    backend: B,
}

/// A unique identifier for a layer.
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, Debug)]
pub struct LayerId(pub u32);

#[doc(hidden)]
#[derive(Debug)]
pub struct LayerMap<T>(pub Vec<Option<T>>);

// Public structures

/// A connection to the OS display server.
pub enum Connection<N> {
    /// A native connection.
    Native(N),
    /// A connection managed by `winit`.
    #[cfg(feature = "enable-winit")]
    Winit(), // TODO: Fix this ...
}

bitflags::bitflags! {
    /// Specifies the type of GPU surface or surfaces to be allocated for a surface layer.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct SurfaceOptions: u8 {
        /// The layer is opaque.
        ///
        /// The OS may be able to optimize composition of opaque layers, for example by not
        /// composing any content underneath them
        const OPAQUE = 0x01;

        /// The surface includes a 24-bit depth or Z-buffer.
        const DEPTH = 0x02;

        /// The surface includes an 8-bit stencil buffer.
        const STENCIL = 0x04;
    }
}

/// Represents the result of a pending operation.
///
/// This is similar to a Rust future, but it always uses the native OS event loop for dispatch.
/// Note that it is your responsibility to pump the OS event loop. (If using `winit`, this is the
/// `EventLoop` object.)
///
/// Use the `then` method to attach handlers.
#[derive(Clone)]
pub struct Promise<T>(Arc<Mutex<PromiseData<T>>>)
where
    T: 'static + Clone + Send;

// Components

#[doc(hidden)]
pub struct LayerTreeInfo {
    parent: LayerParent,
    prev_sibling: Option<LayerId>,
    next_sibling: Option<LayerId>,
}

#[doc(hidden)]
pub struct LayerContainerInfo {
    first_child: Option<LayerId>,
    last_child: Option<LayerId>,
}

#[doc(hidden)]
pub struct LayerGeometryInfo {
    bounds: Rect<f32>,
}

#[doc(hidden)]
pub struct LayerSurfaceInfo {
    options: SurfaceOptions,
}

// Other data structures

#[derive(PartialEq, Debug)]
pub enum LayerParent {
    Layer(LayerId),
    NativeHost,
}

struct PromiseData<T>
where
    T: Clone + Send,
{
    on_fulfilled: Vec<Box<dyn FnMut(T) + Send>>,
    on_rejected: Vec<Box<dyn FnMut() + Send>>,
    result: PromiseResult<T>,
}

enum PromiseResult<T>
where
    T: Clone + Send,
{
    Pending,
    Resolved(T),
    Rejected,
}

// Public API for the context

impl<B> LayerContext<B>
where
    B: Backend,
{
    // Core functions

    /// Creates a layer context from a connection to the display server.
    ///
    /// This method allows you to specify a backend explicitly.
    pub fn with_backend_connection(
        connection: Connection<B::NativeConnection>,
    ) -> Result<LayerContext<B>, ConnectionError> {
        Ok(LayerContext {
            backend: Backend::new(connection)?,

            next_layer_id: LayerId(0),
            transaction: None,

            tree_component: LayerMap::new(),
            container_component: LayerMap::new(),
            geometry_component: LayerMap::new(),
            surface_component: LayerMap::new(),
        })
    }

    // Transactions

    /// Opens a new atomic transaction.
    ///
    /// All layer manipulations must take place between calls to `begin_transaction` and
    /// `end_transaction`, unless otherwise specified. The layer context will panic otherwise.
    ///
    /// Transactions may be nested. No operations happen until the final `end_transaction` call is
    /// issued.
    pub fn begin_transaction(&mut self) {
        match self.transaction {
            None => {
                self.transaction = Some(TransactionInfo {
                    level: 1,
                    promise: Promise::new(),
                });
                self.backend.begin_transaction();
            }
            Some(ref mut transaction) => {
                transaction.level += 1;
            }
        }
    }

    /// Ends the current transaction and submits it to the display server.
    ///
    /// This method is *not* synchronous; it merely flushes the pending operations the server,
    /// ensuring that they will complete in finite time.
    pub fn end_transaction(&mut self) {
        {
            let transaction = self
                .transaction
                .as_mut()
                .expect("end_transaction(): Not in a transaction!");
            transaction.level -= 1;
            if transaction.level > 0 {
                return;
            }
        }

        // If we got here, we're done with the transaction.
        let transaction = self.transaction.take().unwrap();
        self.backend.end_transaction(
            &transaction.promise,
            &self.tree_component,
            &self.container_component,
            &self.geometry_component,
            &self.surface_component,
        );
    }

    /// Returns true if a transaction is in process and false otherwise.
    ///
    /// In other words, this returns true if and only if `begin_transaction` has been called
    /// without a matching `end_transaction`.
    #[inline]
    fn in_transaction(&self) -> bool {
        self.transaction.is_some()
    }

    // Layer tree management system

    /// Creates a new container layer and returns its ID.
    ///
    /// Container layers, as their name implies, contain other layers. They are invisible and
    /// cannot be rendered to. OpenGL contexts also cannot be attached to them.
    ///
    /// Initially, the newly-created layer is off-screen, with neither position nor size.
    pub fn add_container_layer(&mut self) -> LayerId {
        debug_assert!(self.in_transaction());

        let layer = self.next_layer_id;
        self.next_layer_id.0 += 1;

        self.container_component.add(
            layer,
            LayerContainerInfo {
                first_child: None,
                last_child: None,
            },
        );
        self.backend.add_container_layer(layer);
        layer
    }

    /// Creates a new surface layer and returns its ID.
    ///
    /// Surface layers can be rendered to, and OpenGL contexts can be attached to them. They may
    /// not contain other layers; therefore, they must be leaves of the layer tree.
    ///
    /// Initially, the newly-created layer is off-screen, with neither position nor size.
    pub fn add_surface_layer(&mut self) -> LayerId {
        debug_assert!(self.in_transaction());

        let layer = self.next_layer_id;
        self.next_layer_id.0 += 1;

        self.surface_component.add(
            layer,
            LayerSurfaceInfo {
                options: SurfaceOptions::empty(),
            },
        );

        self.backend.add_surface_layer(layer);
        layer
    }

    /// Returns the parent of the given layer, if it is on-screen.
    pub fn parent_of(&self, layer: LayerId) -> Option<&LayerParent> {
        self.tree_component.get(layer).map(|info| &info.parent)
    }

    /// Adds a layer to a container layer, optionally before a specific sibling.
    ///
    /// The specified parent layer must be a container layer. The new child layer must be
    /// off-screen (i.e. not in the tree).
    ///
    /// If `reference` is specified, it must name an immediate child of the given parent layer. The
    /// new child layer will be added before that reference in the parent's child list. If
    /// `reference` is `None`, then the new child is added to the end of the parent's child list.
    pub fn insert_before(
        &mut self,
        parent: LayerId,
        new_child: LayerId,
        reference: Option<LayerId>,
    ) {
        debug_assert!(self.in_transaction());

        if let Some(reference) = reference {
            debug_assert_eq!(self.parent_of(reference), Some(&LayerParent::Layer(parent)));
        }

        let new_prev_sibling = match reference {
            Some(reference) => self.tree_component[reference].prev_sibling,
            None => self.container_component[parent].last_child,
        };

        self.tree_component.add(
            new_child,
            LayerTreeInfo {
                parent: LayerParent::Layer(parent),
                prev_sibling: new_prev_sibling,
                next_sibling: reference,
            },
        );

        match reference {
            Some(reference) => self.tree_component[reference].next_sibling = Some(new_child),
            None => self.container_component[parent].last_child = Some(new_child),
        }

        if self.tree_component[new_child].prev_sibling.is_none() {
            self.container_component[parent].first_child = Some(new_child)
        }

        self.backend.insert_before(
            parent,
            new_child,
            reference,
            &self.tree_component,
            &self.container_component,
            &self.geometry_component,
        );
    }

    /// Adds a layer to the end of a container layer's child list.
    ///
    /// This is equivalent to `insert_before` with `reference` set to `None`.
    #[inline]
    pub fn append_child(&mut self, parent: LayerId, new_child: LayerId) {
        self.insert_before(parent, new_child, None)
    }

    #[inline]
    pub unsafe fn host_layer(&mut self, host: B::Host, layer: LayerId) {
        debug_assert!(self.in_transaction());

        self.tree_component.add(
            layer,
            LayerTreeInfo {
                parent: LayerParent::NativeHost,
                prev_sibling: None,
                next_sibling: None,
            },
        );

        unsafe {
            self.backend.host_layer(
                layer,
                host,
                &self.tree_component,
                &self.container_component,
                &self.geometry_component,
            );
        }
    }

    pub fn remove_from_parent(&mut self, old_child: LayerId) {
        debug_assert!(self.in_transaction());

        let old_tree = self.tree_component.take(old_child);
        match old_tree.parent {
            LayerParent::NativeHost => self.backend.unhost_layer(old_child),

            LayerParent::Layer(parent_layer) => {
                self.backend.remove_from_superlayer(
                    old_child,
                    parent_layer,
                    &self.tree_component,
                    &self.geometry_component,
                );

                match old_tree.prev_sibling {
                    None => {
                        self.container_component[parent_layer].first_child = old_tree.next_sibling
                    }
                    Some(prev_sibling) => {
                        self.tree_component[prev_sibling].next_sibling = old_tree.next_sibling
                    }
                }
                match old_tree.next_sibling {
                    None => {
                        self.container_component[parent_layer].last_child = old_tree.prev_sibling
                    }
                    Some(next_sibling) => {
                        self.tree_component[next_sibling].prev_sibling = old_tree.prev_sibling
                    }
                }
            }
        }
    }

    /// Deletes a layer and destroys all graphics resources associated with it.
    ///
    /// The layer must be offscreen (i.e. removed from the tree) first.
    pub fn delete_layer(&mut self, layer: LayerId) {
        debug_assert!(self.in_transaction());

        // TODO(pcwalton): Use a free list to recycle IDs.
        debug_assert!(self.parent_of(layer).is_none());

        self.tree_component.remove_if_present(layer);
        self.container_component.remove_if_present(layer);
        self.geometry_component.remove_if_present(layer);
        self.surface_component.remove_if_present(layer);

        self.backend.delete_layer(layer);
    }

    // Geometry system

    /// Returns the boundaries of the layer relative to its parent.
    ///
    /// The rectangle origin specifies the top left corner of the layer.
    pub fn layer_bounds(&self, layer: LayerId) -> Rect<f32> {
        debug_assert!(self.in_transaction());

        match self.geometry_component.get(layer) {
            None => Rect::zero(),
            Some(geometry) => geometry.bounds,
        }
    }

    /// Sets the boundaries of the layer relative to its parent.
    ///
    /// The rectangle origin specifies the top left corner of the layer.
    ///
    /// If this call causes the size of the layer to change, it may cause associated GPU resources
    /// to be reallocated.
    pub fn set_layer_bounds(&mut self, layer: LayerId, new_bounds: &Rect<f32>) {
        debug_assert!(self.in_transaction());

        let old_bounds = mem::replace(
            &mut self.geometry_component.get_mut_default(layer).bounds,
            *new_bounds,
        );

        self.backend.set_layer_bounds(
            layer,
            &old_bounds,
            &self.tree_component,
            &self.container_component,
            &self.geometry_component,
        );
    }

    // Miscellaneous layer flags

    /// Sets options for this surface layer.
    ///
    /// Any GL contexts attached to this layer must have the same surface options as the layer
    /// itself.
    ///
    /// The `layer` parameter must refer to a surface layer, not a container layer.
    pub fn set_layer_surface_options(&mut self, layer: LayerId, surface_options: SurfaceOptions) {
        debug_assert!(self.in_transaction());

        self.surface_component[layer].options = surface_options;
        self.backend
            .set_layer_surface_options(layer, &self.surface_component);
    }

    // Screenshots

    pub fn screenshot_hosted_layer(&mut self, layer: LayerId) -> Promise<RgbaImage> {
        debug_assert!(self.in_transaction());
        assert_eq!(self.tree_component[layer].parent, LayerParent::NativeHost);

        let transaction_promise = self.transaction.as_ref().unwrap().promise.clone();
        self.backend.screenshot_hosted_layer(
            layer,
            &transaction_promise,
            &self.tree_component,
            &self.container_component,
            &self.geometry_component,
            &self.surface_component,
        )
    }

    // `winit` integration

    #[cfg(feature = "enable-winit")]
    pub fn window(&self) -> Option<&Window> {
        self.backend.window()
    }

    #[cfg(feature = "enable-winit")]
    pub fn host_layer_in_window(&mut self, layer: LayerId) -> Result<(), ()> {
        debug_assert!(self.in_transaction());

        self.tree_component.add(
            layer,
            LayerTreeInfo {
                parent: LayerParent::NativeHost,
                prev_sibling: None,
                next_sibling: None,
            },
        );

        self.backend.host_layer_in_window(
            layer,
            &self.tree_component,
            &self.container_component,
            &self.geometry_component,
        )
    }
}

impl LayerContext<backends::default::Backend> {
    #[inline]
    pub fn new(
        connection: Connection<<backends::default::Backend as Backend>::NativeConnection>,
    ) -> Result<LayerContext<backends::default::Backend>, ConnectionError> {
        LayerContext::with_backend_connection(connection)
    }
}

// Errors

pub struct ConnectionError {
    #[cfg(feature = "enable-winit")]
    window_builder: Option<()>, // TODO
}

impl Debug for ConnectionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        "ConnectionError".fmt(formatter)
    }
}

impl ConnectionError {
    #[inline]
    pub fn new() -> ConnectionError {
        ConnectionError {
            #[cfg(feature = "enable-winit")]
            window_builder: None,
        }
    }
}

// Promise infrastructure

impl<T> Promise<T>
where
    T: 'static + Clone + Send,
{
    fn new() -> Promise<T> {
        Promise(Arc::new(Mutex::new(PromiseData {
            on_fulfilled: vec![],
            on_rejected: vec![],
            result: PromiseResult::Pending,
        })))
    }

    fn all(promises: Vec<Promise<T>>) -> Promise<Vec<T>> {
        let result_promise = Promise::new();
        let all = Arc::new(Mutex::new(All {
            result_promise: result_promise.clone(),
            promises,
            results: vec![],
        }));
        wait(all);
        return result_promise;

        fn wait<T>(all: Arc<Mutex<All<T>>>)
        where
            T: 'static + Clone + Send,
        {
            let next_promise;
            {
                let mut all = all.lock().unwrap();
                if all.results.len() == all.promises.len() {
                    let results = mem::replace(&mut all.results, vec![]);
                    all.result_promise.resolve(results);
                    return;
                }
                next_promise = all.promises[all.results.len()].clone();
            }

            next_promise.then(Box::new(move |result| {
                all.lock().unwrap().results.push(result);
                wait(all.clone());
            }));
        }

        struct All<T>
        where
            T: 'static + Clone + Send,
        {
            result_promise: Promise<Vec<T>>,
            promises: Vec<Promise<T>>,
            results: Vec<T>,
        }
    }

    pub fn then(&self, mut on_fulfilled: Box<dyn FnMut(T) + Send>) {
        let mut this = self.0.lock().unwrap();
        match this.result {
            PromiseResult::Rejected => {}
            PromiseResult::Resolved(ref result) => on_fulfilled((*result).clone()),
            PromiseResult::Pending => this.on_fulfilled.push(on_fulfilled),
        }
    }

    pub fn or_else(&self, mut on_rejected: Box<dyn FnMut() + Send>) {
        let mut this = self.0.lock().unwrap();
        match this.result {
            PromiseResult::Rejected => on_rejected(),
            PromiseResult::Resolved(_) => {}
            PromiseResult::Pending => this.on_rejected.push(on_rejected),
        }
    }

    fn resolve(&self, result: T) {
        let mut this = self.0.lock().unwrap();
        this.result = PromiseResult::Resolved(result.clone());
        for mut on_fulfilled in this.on_fulfilled.drain(..) {
            on_fulfilled(result.clone())
        }
    }

    fn reject(&self) {
        let mut this = self.0.lock().unwrap();
        this.result = PromiseResult::Rejected;
        for mut on_rejected in this.on_rejected.drain(..) {
            on_rejected()
        }
    }
}

struct TransactionInfo {
    level: u32,
    promise: Promise<()>,
}

// Entity-component system infrastructure

impl<T> LayerMap<T> {
    #[inline]
    fn new() -> LayerMap<T> {
        LayerMap(vec![])
    }

    fn add(&mut self, layer_id: LayerId, element: T) {
        while self.0.len() <= (layer_id.0 as usize) {
            self.0.push(None)
        }
        debug_assert!(self.0[layer_id.0 as usize].is_none());
        self.0[layer_id.0 as usize] = Some(element);
    }

    fn has(&self, layer_id: LayerId) -> bool {
        (layer_id.0 as usize) < self.0.len() && self.0[layer_id.0 as usize].is_some()
    }

    fn take(&mut self, layer_id: LayerId) -> T {
        debug_assert!(self.has(layer_id));
        mem::replace(&mut self.0[layer_id.0 as usize], None).unwrap()
    }

    fn remove(&mut self, layer_id: LayerId) {
        drop(self.take(layer_id))
    }

    fn remove_if_present(&mut self, layer_id: LayerId) {
        if self.has(layer_id) {
            self.remove(layer_id)
        }
    }

    fn get(&self, layer_id: LayerId) -> Option<&T> {
        if (layer_id.0 as usize) >= self.0.len() {
            None
        } else {
            self.0[layer_id.0 as usize].as_ref()
        }
    }

    fn get_mut(&mut self, layer_id: LayerId) -> Option<&mut T> {
        if (layer_id.0 as usize) >= self.0.len() {
            None
        } else {
            self.0[layer_id.0 as usize].as_mut()
        }
    }
}

impl<T> LayerMap<T>
where
    T: Default,
{
    fn get_mut_default(&mut self, layer_id: LayerId) -> &mut T {
        while self.0.len() <= (layer_id.0 as usize) {
            self.0.push(None)
        }
        if self.0[layer_id.0 as usize].is_none() {
            self.0[layer_id.0 as usize] = Some(T::default());
        }
        self.0[layer_id.0 as usize].as_mut().unwrap()
    }
}

impl<T> Index<LayerId> for LayerMap<T> {
    type Output = T;

    #[inline]
    fn index(&self, layer_id: LayerId) -> &T {
        self.0[layer_id.0 as usize].as_ref().unwrap()
    }
}

impl<T> IndexMut<LayerId> for LayerMap<T> {
    #[inline]
    fn index_mut(&mut self, layer_id: LayerId) -> &mut T {
        self.0[layer_id.0 as usize].as_mut().unwrap()
    }
}

// Specific type infrastructure

impl<N> Connection<N> {
    #[cfg(feature = "enable-winit")]
    pub fn into_window(self) -> Option<Window> {
        match self {
            Connection::Native(_) => None,
            Connection::Winit() => todo!(),
        }
    }
}

// Specific component infrastructure

impl Default for LayerGeometryInfo {
    fn default() -> LayerGeometryInfo {
        LayerGeometryInfo {
            bounds: Rect::zero(),
        }
    }
}
