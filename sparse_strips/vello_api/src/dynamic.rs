// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The place where a dyn-compatible wrapper around [`Renderer`](crate::Renderer)s will go.

// pub trait AnyRenderer: Any + Send {}

// impl<T: Renderer> AnyRenderer for T {}
// impl Renderer for dyn AnyRenderer {}
// impl Renderer for Box<dyn AnyRenderer> {
//     type PaintScene = AnyScenePainter;
//}

// pub trait AnyScenePainter: Any {}

// impl<T: PaintScene> AnyScenePainter for T {}
// impl PaintScene for Box<dyn AnyScenePainter> {}

// pub trait AnyPathPreparer<Target: PaintScene>: Any {}

// The "PreparePaths" needs to know what to downcast the type to, so we store it in a specialised struct.
// struct DynamicPathPreparer<Target, Preparer>(PhantomData<Target>, Preparer);
// impl<T: PreparePathsDirect<Target>, Target> AnyPathPreparer<Target> for DynamicPathPreparer<Target, Preparer> {}
// impl<Target> PreparePathsDirect<Target> for Box<dyn AnyPathPreparer<Target>> {}
