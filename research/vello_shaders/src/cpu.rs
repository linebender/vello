// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Support for CPU implementations of compute shaders.
//!
//! Note that while this CPU implementation is useful for testing and debugging,
//! a full CPU fallback as an alternative to GPU shaders is not provided.

// Allow un-idiomatic Rust to more closely match shaders
#![expect(
    clippy::needless_range_loop,
    reason = "Keeps code easily comparable to GPU shaders"
)]

mod backdrop;
mod bbox_clear;
mod binning;
mod clip_leaf;
mod clip_reduce;
mod coarse;
mod draw_leaf;
mod draw_reduce;
mod euler;
mod fine;
mod flatten;
mod path_count;
mod path_count_setup;
mod path_tiling;
mod path_tiling_setup;
mod pathtag_reduce;
mod pathtag_scan;
mod tile_alloc;
mod util;

pub use backdrop::backdrop;
pub use bbox_clear::bbox_clear;
pub use binning::binning;
pub use clip_leaf::clip_leaf;
pub use clip_reduce::clip_reduce;
pub use coarse::coarse;
pub use draw_leaf::draw_leaf;
pub use draw_reduce::draw_reduce;
pub use flatten::flatten;
pub use path_count::path_count;
pub use path_count_setup::path_count_setup;
pub use path_tiling::path_tiling;
pub use path_tiling_setup::path_tiling_setup;
pub use pathtag_reduce::pathtag_reduce;
pub use pathtag_scan::pathtag_scan;
pub use tile_alloc::tile_alloc;

use std::cell::{Ref, RefCell, RefMut};
use std::ops::{Deref, DerefMut};

use bytemuck::Pod;

#[derive(Clone, Copy)]
pub enum CpuBinding<'a> {
    Buffer(&'a [u8]),
    BufferRW(&'a RefCell<Vec<u8>>),
    Texture(&'a CpuTexture),
}

pub enum TypedBufGuard<'a, T: ?Sized> {
    Slice(&'a T),
    Interior(Ref<'a, T>),
}

pub enum TypedBufGuardMut<'a, T: ?Sized> {
    Slice(&'a mut T),
    Interior(RefMut<'a, T>),
}

impl<T: ?Sized> Deref for TypedBufGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            TypedBufGuard::Slice(s) => s,
            TypedBufGuard::Interior(r) => r,
        }
    }
}

impl<T: ?Sized> Deref for TypedBufGuardMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            TypedBufGuardMut::Slice(s) => s,
            TypedBufGuardMut::Interior(r) => r,
        }
    }
}

impl<T: ?Sized> DerefMut for TypedBufGuardMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            TypedBufGuardMut::Slice(s) => s,
            TypedBufGuardMut::Interior(r) => r,
        }
    }
}

impl CpuBinding<'_> {
    pub fn as_typed<T: Pod>(&self) -> TypedBufGuard<'_, T> {
        match self {
            CpuBinding::Buffer(b) => TypedBufGuard::Slice(bytemuck::from_bytes(b)),
            CpuBinding::BufferRW(b) => {
                TypedBufGuard::Interior(Ref::map(b.borrow(), |buf| bytemuck::from_bytes(buf)))
            }
            _ => panic!("resource type mismatch"),
        }
    }

    pub fn as_typed_mut<T: Pod>(&self) -> TypedBufGuardMut<'_, T> {
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

    pub fn as_slice<T: Pod>(&self) -> TypedBufGuard<'_, [T]> {
        match self {
            CpuBinding::Buffer(b) => TypedBufGuard::Slice(bytemuck::cast_slice(b)),
            CpuBinding::BufferRW(b) => {
                TypedBufGuard::Interior(Ref::map(b.borrow(), |buf| bytemuck::cast_slice(buf)))
            }
            _ => panic!("resource type mismatch"),
        }
    }

    pub fn as_slice_mut<T: Pod>(&self) -> TypedBufGuardMut<'_, [T]> {
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

// Common internal definitions

const PTCL_INITIAL_ALLOC: u32 = 64;

// Tags for PTCL commands
const CMD_END: u32 = 0;
const CMD_FILL: u32 = 1;
//const CMD_STROKE: u32 = 2;
const CMD_SOLID: u32 = 3;
const CMD_COLOR: u32 = 5;
const CMD_LIN_GRAD: u32 = 6;
const CMD_RAD_GRAD: u32 = 7;
const CMD_SWEEP_GRAD: u32 = 8;
const CMD_IMAGE: u32 = 9;
const CMD_BEGIN_CLIP: u32 = 10;
const CMD_END_CLIP: u32 = 11;
const CMD_JUMP: u32 = 12;
const CMD_BLUR_RECT: u32 = 13;

// The following are computed in draw_leaf from the generic gradient parameters
// encoded in the scene, and stored in the gradient's info struct, for
// consumption during fine rasterization.

// Radial gradient kinds
const RAD_GRAD_KIND_CIRCULAR: u32 = 1;
const RAD_GRAD_KIND_STRIP: u32 = 2;
const RAD_GRAD_KIND_FOCAL_ON_CIRCLE: u32 = 3;
const RAD_GRAD_KIND_CONE: u32 = 4;

// Radial gradient flags
const RAD_GRAD_SWAPPED: u32 = 1;
