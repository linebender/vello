// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::sync::Arc;
use core::{fmt::Debug, hash::Hash};

use crate::{Renderer, paths::PathGroupId};

#[derive(Debug, Clone)]
pub struct PathGroup {
    inner: Arc<PathGroupInner>,
}

impl PathGroup {
    pub fn id(&self) -> PathGroupId {
        self.inner.id
    }

    // Probably return a structure instead.
    pub fn with(&self, index: u32) -> (Self, u32) {
        (self.clone(), index)
    }

    pub(crate) fn new(renderer: Arc<dyn Renderer>, id: PathGroupId) -> Self {
        Self {
            inner: Arc::new(PathGroupInner { renderer, id }),
        }
    }
}

impl Hash for PathGroup {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.inner).hash(state);
    }
}
impl PartialEq for PathGroup {
    fn eq(&self, other: &Self) -> bool {
        Arc::as_ptr(&self.inner) == Arc::as_ptr(&other.inner)
    }
}
impl Eq for PathGroup {}

struct PathGroupInner {
    renderer: Arc<dyn Renderer>,
    id: PathGroupId,
}

impl Drop for PathGroupInner {
    fn drop(&mut self) {
        self.renderer.free_paths(self.id);
    }
}

impl Debug for PathGroupInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PathGroupInner")
            .field("renderer", &"elided")
            .field("id", &self.id)
            .finish()
    }
}
