// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Support for CPU implementations of compute shaders.
//!
//! Note that while this CPU implementation is useful for testing and debugging,
//! a full CPU fallback for targets without wgpu hasn't been implemented yet.

use std::cell::{Ref, RefCell, RefMut};
use std::ops::{Deref, DerefMut};

use bytemuck::Pod;

#[derive(Clone, Copy)]
pub enum CpuBinding<'a> {
    Buffer(&'a [u8]),
    BufferRW(&'a RefCell<Vec<u8>>),
    #[allow(unused)]
    Texture(&'a CpuTexture),
}

pub enum TypedBufGuard<'a, T: ?Sized> {
    Slice(&'a T),
    Interior(Ref<'a, T>),
}

pub enum TypedBufGuardMut<'a, T: ?Sized> {
    #[allow(dead_code)]
    Slice(&'a mut T),
    Interior(RefMut<'a, T>),
}

impl<'a, T: ?Sized> Deref for TypedBufGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            TypedBufGuard::Slice(s) => s,
            TypedBufGuard::Interior(r) => r,
        }
    }
}

impl<'a, T: ?Sized> Deref for TypedBufGuardMut<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            TypedBufGuardMut::Slice(s) => s,
            TypedBufGuardMut::Interior(r) => r,
        }
    }
}

impl<'a, T: ?Sized> DerefMut for TypedBufGuardMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            TypedBufGuardMut::Slice(s) => s,
            TypedBufGuardMut::Interior(r) => r,
        }
    }
}

impl<'a> CpuBinding<'a> {
    pub fn as_typed<T: Pod>(&self) -> TypedBufGuard<T> {
        match self {
            CpuBinding::Buffer(b) => TypedBufGuard::Slice(bytemuck::from_bytes(b)),
            CpuBinding::BufferRW(b) => {
                TypedBufGuard::Interior(Ref::map(b.borrow(), |buf| bytemuck::from_bytes(buf)))
            }
            _ => panic!("resource type mismatch"),
        }
    }

    pub fn as_typed_mut<T: Pod>(&self) -> TypedBufGuardMut<T> {
        match self {
            CpuBinding::Buffer(_) => panic!("can't borrow external buffer mutably"),
            CpuBinding::BufferRW(b) => {
                TypedBufGuardMut::Interior(RefMut::map(b.borrow_mut(), |buf| {
                    bytemuck::from_bytes_mut(buf)
                }))
            }
            _ => panic!("resource type mismatch"),
        }
    }

    pub fn as_slice<T: Pod>(&self) -> TypedBufGuard<[T]> {
        match self {
            CpuBinding::Buffer(b) => TypedBufGuard::Slice(bytemuck::cast_slice(b)),
            CpuBinding::BufferRW(b) => {
                TypedBufGuard::Interior(Ref::map(b.borrow(), |buf| bytemuck::cast_slice(buf)))
            }
            _ => panic!("resource type mismatch"),
        }
    }

    pub fn as_slice_mut<T: Pod>(&self) -> TypedBufGuardMut<[T]> {
        match self {
            CpuBinding::Buffer(_) => panic!("can't borrow external buffer mutably"),
            CpuBinding::BufferRW(b) => {
                TypedBufGuardMut::Interior(RefMut::map(b.borrow_mut(), |buf| {
                    bytemuck::cast_slice_mut(buf)
                }))
            }
            _ => panic!("resource type mismatch"),
        }
    }

    // TODO: same guard as buf to make mutable
    #[allow(unused)]
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
