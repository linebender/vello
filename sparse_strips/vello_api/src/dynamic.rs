// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A dyn-compatible wrapper around [`Renderer`]s

// use alloc::boxed::Box;
// use core::any::Any;

// use crate::{PaintScene, Renderer, prepared::PreparePaths};

// pub trait AnyRenderer: Any + Send {}

// impl<T: Renderer + Any> AnyRenderer for T {}
// impl Renderer for Box<dyn AnyRenderer> {
//     type ScenePainter = Box<dyn AnyScenePainter>;

//     type PathPreparer = Box<dyn AnyPathPreparer>;

//     fn create_texture(descriptor: crate::texture::TextureDescriptor) -> crate::texture::Texture {
//         todo!()
//     }

//     fn create_scene(
//         &mut self,
//         to: &crate::texture::Texture,
//         options: crate::SceneOptions,
//     ) -> Self::ScenePainter {
//         todo!()
//     }

//     fn queue_render(&mut self, from: Self::ScenePainter) {
//         todo!()
//     }

//     fn queue_download(&mut self, texture: &crate::texture::Texture) -> crate::DownloadId {
//         todo!()
//     }

//     fn upload_image(
//         to: &crate::texture::Texture,
//         data: peniko::ImageData,
//         region: Option<(u16, u16, u16, u16)>,
//     ) -> Result<(), ()> {
//         todo!()
//     }

//     fn create_path_cache(&mut self) -> crate::prepared::PreparedPaths {
//         todo!()
//     }

//     fn fill_path_cache<R>(
//         &mut self,
//         cache: &crate::prepared::PreparedPaths,
//         f: impl FnOnce(&mut Self::PathPreparer) -> R,
//     ) -> Result<R, ()> {
//         todo!()
//     }
// }

// pub trait AnyScenePainter: Any {}
// impl<T: PaintScene + Any> AnyScenePainter for T {}

// impl PaintScene for Box<dyn AnyScenePainter> {}

// pub trait AnyPathPreparer: Any {}
// impl<T: PreparePaths> AnyPathPreparer for T {}

// impl PreparePaths for Box<dyn AnyPathPreparer> {
//     fn prepare_fill(
//         &mut self,
//         fill_rule: peniko::Fill,
//         meta: crate::prepared::PreparedPathMeta,
//         shape: &impl peniko::kurbo::Shape,
//         // TODO: Result? For example, if we have a fixed storage size
//     ) -> crate::prepared::PreparedPathIndex {
//         todo!()
//     }

//     fn prepare_stroke(
//         &mut self,
//         stroke_rule: peniko::kurbo::Stroke,
//         meta: crate::prepared::PreparedPathMeta,
//         stroke: &impl peniko::kurbo::Shape,
//     ) -> crate::prepared::PreparedPathIndex {
//         todo!()
//     }
// }
