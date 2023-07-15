//! Support for CPU implementations of compute shaders.

use std::{
    cell::{RefCell, RefMut},
    ops::Deref,
};

#[derive(Clone, Copy)]
pub enum CpuBinding<'a> {
    Buffer(&'a [u8]),
    BufferRW(&'a RefCell<Vec<u8>>),
    Texture(&'a CpuTexture),
}

pub enum CpuBufGuard<'a> {
    Slice(&'a [u8]),
    Interior(RefMut<'a, Vec<u8>>),
}

impl<'a> Deref for CpuBufGuard<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            CpuBufGuard::Slice(s) => s,
            CpuBufGuard::Interior(r) => &*r,
        }
    }
}

impl<'a> CpuBufGuard<'a> {
    /// Get a mutable reference to the buffer.
    ///
    /// Panics if the underlying resource is read-only.
    pub fn as_mut(&mut self) -> &mut [u8] {
        match self {
            CpuBufGuard::Interior(r) => &mut *r,
            _ => panic!("tried to borrow immutable buffer as mutable"),
        }
    }
}

impl<'a> CpuBinding<'a> {
    pub fn as_buf(&self) -> CpuBufGuard {
        match self {
            CpuBinding::Buffer(b) => CpuBufGuard::Slice(b),
            CpuBinding::BufferRW(b) => CpuBufGuard::Interior(b.borrow_mut()),
            _ => panic!("resource type mismatch"),
        }
    }

    // TODO: same guard as buf to make mutable
    pub fn as_tex(&self) -> &CpuTexture {
        match self {
            CpuBinding::Texture(t) => t,
            _ => panic!("resource type mismatch"),
        }
    }
}

/// Structure used for binding textures to CPU shaders.
pub struct CpuTexture {
    pub width: usize,
    pub height: usize,
    // In RGBA format. May expand in the future.
    pub pixels: Vec<u32>,
}
