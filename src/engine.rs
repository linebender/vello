// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

use std::{
    num::NonZeroU64,
    sync::atomic::{AtomicU64, Ordering},
};

pub type Error = Box<dyn std::error::Error>;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ShaderId(pub usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id(pub NonZeroU64);

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Default)]
pub struct Recording {
    pub commands: Vec<Command>,
}

#[derive(Clone, Copy)]
pub struct BufProxy {
    pub size: u64,
    pub id: Id,
    pub name: &'static str,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum ImageFormat {
    Rgba8,
    #[allow(unused)]
    Bgra8,
}

#[derive(Clone, Copy)]
pub struct ImageProxy {
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub id: Id,
}

#[derive(Clone, Copy)]
pub enum ResourceProxy {
    Buf(BufProxy),
    BufRange {
        proxy: BufProxy,
        offset: u64,
        size: u64,
    },
    Image(ImageProxy),
}

pub enum Command {
    Upload(BufProxy, Vec<u8>),
    UploadUniform(BufProxy, Vec<u8>),
    UploadImage(ImageProxy, Vec<u8>),
    WriteImage(ImageProxy, [u32; 4], Vec<u8>),

    // Discussion question: third argument is vec of resources?
    // Maybe use tricks to make more ergonomic?
    // Alternative: provide bufs & images as separate sequences
    Dispatch(ShaderId, (u32, u32, u32), Vec<ResourceProxy>),
    DispatchIndirect(ShaderId, BufProxy, u64, Vec<ResourceProxy>),
    Draw {
        shader_id: ShaderId,
        instance_count: u32,
        vertex_count: u32,
        vertex_buffer: Option<BufProxy>,
        resources: Vec<ResourceProxy>,
        target: ImageProxy,
        clear_color: Option<[f32; 4]>,
    },

    Download(BufProxy),
    Clear(BufProxy, u64, Option<u64>),
    FreeBuf(BufProxy),
    FreeImage(ImageProxy),
}

/// The type of resource that will be bound to a slot in a shader.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BindType {
    /// A storage buffer with read/write access.
    Buffer,
    /// A storage buffer with read only access.
    BufReadOnly,
    /// A small storage buffer to be used as uniforms.
    Uniform,
    /// A storage image.
    Image(ImageFormat),
    /// A storage image with read only access.
    ImageRead(ImageFormat),
    // TODO: Uniform, Sampler, maybe others
}

impl Recording {
    pub fn push(&mut self, cmd: Command) {
        self.commands.push(cmd);
    }

    pub fn upload(&mut self, name: &'static str, data: impl Into<Vec<u8>>) -> BufProxy {
        let data = data.into();
        let buf_proxy = BufProxy::new(data.len() as u64, name);
        self.push(Command::Upload(buf_proxy, data));
        buf_proxy
    }

    pub fn upload_uniform(&mut self, name: &'static str, data: impl Into<Vec<u8>>) -> BufProxy {
        let data = data.into();
        let buf_proxy = BufProxy::new(data.len() as u64, name);
        self.push(Command::UploadUniform(buf_proxy, data));
        buf_proxy
    }

    pub fn upload_image(
        &mut self,
        width: u32,
        height: u32,
        format: ImageFormat,
        data: impl Into<Vec<u8>>,
    ) -> ImageProxy {
        let data = data.into();
        let image_proxy = ImageProxy::new(width, height, format);
        self.push(Command::UploadImage(image_proxy, data));
        image_proxy
    }

    pub fn write_image(
        &mut self,
        image: ImageProxy,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        data: impl Into<Vec<u8>>,
    ) {
        let data = data.into();
        self.push(Command::WriteImage(image, [x, y, width, height], data));
    }

    pub fn dispatch<R>(&mut self, shader: ShaderId, wg_size: (u32, u32, u32), resources: R)
    where
        R: IntoIterator,
        R::Item: Into<ResourceProxy>,
    {
        let r = resources.into_iter().map(|r| r.into()).collect();
        self.push(Command::Dispatch(shader, wg_size, r));
    }

    /// Do an indirect dispatch.
    ///
    /// Dispatch a compute shader where the size is determined dynamically.
    /// The `buf` argument contains the dispatch size, 3 `u32` values beginning
    /// at the given byte `offset`.
    #[allow(unused)]
    pub fn dispatch_indirect<R>(
        &mut self,
        shader: ShaderId,
        buf: BufProxy,
        offset: u64,
        resources: R,
    ) where
        R: IntoIterator,
        R::Item: Into<ResourceProxy>,
    {
        let r = resources.into_iter().map(|r| r.into()).collect();
        self.push(Command::DispatchIndirect(shader, buf, offset, r));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw<R>(
        &mut self,
        shader_id: ShaderId,
        instance_count: u32,
        vertex_count: u32,
        vertex_buffer: Option<BufProxy>,
        resources: R,
        target: ImageProxy,
        clear_color: Option<[f32; 4]>,
    ) where
        R: IntoIterator,
        R::Item: Into<ResourceProxy>,
    {
        let r = resources.into_iter().map(|r| r.into()).collect();
        self.push(Command::Draw {
            shader_id,
            instance_count,
            vertex_count,
            vertex_buffer,
            resources: r,
            target,
            clear_color,
        });
    }

    /// Prepare a buffer for downloading.
    ///
    /// Currently this copies to a download buffer. The original buffer can be freed
    /// immediately after.
    pub fn download(&mut self, buf: BufProxy) {
        self.push(Command::Download(buf));
    }

    pub fn clear_all(&mut self, buf: BufProxy) {
        self.push(Command::Clear(buf, 0, None));
    }

    pub fn free_buf(&mut self, buf: BufProxy) {
        self.push(Command::FreeBuf(buf));
    }

    pub fn free_image(&mut self, image: ImageProxy) {
        self.push(Command::FreeImage(image));
    }

    pub fn free_resource(&mut self, resource: ResourceProxy) {
        match resource {
            ResourceProxy::Buf(buf) => self.free_buf(buf),
            ResourceProxy::BufRange {
                proxy,
                offset: _,
                size: _,
            } => self.free_buf(proxy),
            ResourceProxy::Image(image) => self.free_image(image),
        }
    }

    pub fn into_commands(self) -> Vec<Command> {
        self.commands
    }
}

impl BufProxy {
    pub fn new(size: u64, name: &'static str) -> Self {
        let id = Id::next();
        debug_assert!(size > 0);
        BufProxy { id, size, name }
    }
}

impl ImageFormat {
    #[cfg(feature = "wgpu")]
    pub fn to_wgpu(self) -> wgpu::TextureFormat {
        match self {
            Self::Rgba8 => wgpu::TextureFormat::Rgba8Unorm,
            Self::Bgra8 => wgpu::TextureFormat::Bgra8Unorm,
        }
    }

    #[cfg(feature = "wgpu")]
    pub fn from_wgpu(format: wgpu::TextureFormat) -> Self {
        match format {
            wgpu::TextureFormat::Rgba8Unorm => Self::Rgba8,
            wgpu::TextureFormat::Bgra8Unorm => Self::Bgra8,
            _ => unimplemented!(),
        }
    }
}

impl ImageProxy {
    pub fn new(width: u32, height: u32, format: ImageFormat) -> Self {
        let id = Id::next();
        ImageProxy {
            width,
            height,
            format,
            id,
        }
    }
}

impl ResourceProxy {
    pub fn new_buf(size: u64, name: &'static str) -> Self {
        Self::Buf(BufProxy::new(size, name))
    }

    pub fn new_image(width: u32, height: u32, format: ImageFormat) -> Self {
        Self::Image(ImageProxy::new(width, height, format))
    }

    pub fn as_buf(&self) -> Option<&BufProxy> {
        match self {
            Self::Buf(proxy) => Some(proxy),
            _ => None,
        }
    }

    pub fn as_image(&self) -> Option<&ImageProxy> {
        match self {
            Self::Image(proxy) => Some(proxy),
            _ => None,
        }
    }
}

impl From<BufProxy> for ResourceProxy {
    fn from(value: BufProxy) -> Self {
        Self::Buf(value)
    }
}

impl From<ImageProxy> for ResourceProxy {
    fn from(value: ImageProxy) -> Self {
        Self::Image(value)
    }
}

impl Id {
    pub fn next() -> Id {
        let val = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        // could use new_unchecked
        Id(NonZeroU64::new(val + 1).unwrap())
    }
}
