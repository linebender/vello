// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

mod collection;
mod resource;

pub use self::collection::{Operation, PathMeta, PathSet, PreparedPathMeta};
pub use self::resource::PathGroup;

#[derive(Debug, Clone, Copy)]
pub struct PathGroupId(pub u32);

impl PathGroupId {
    /// The `PathGroup` used for 'local' paths, i.e. those which are stored in the current scene.
    pub const LOCAL: Self = Self(u32::MAX);
}

#[derive(Debug, Clone, Copy)]
pub struct StoredPathId(pub PathGroupId, pub u32);

impl StoredPathId {
    pub fn local(of: usize) -> Self {
        let conv = u32::try_from(of).expect("Shouldn't be having more than u32::MAX paths");
        Self(PathGroupId::LOCAL, conv)
    }
}
