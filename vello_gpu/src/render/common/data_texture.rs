// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Textures that shaders read packed data from.

/// The texel format of a [`DataTexture`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DataTextureFormat {
    Rgba8,
    Rgba32Uint,
}

impl DataTextureFormat {
    pub(crate) fn bytes_per_texel(self) -> u32 {
        match self {
            Self::Rgba8 => 4,
            Self::Rgba32Uint => 16,
        }
    }
}

/// A texture of a fixed width that shaders read packed data from, and whose height grows to fit
/// the data uploaded to it.
#[derive(Debug)]
pub(crate) struct DataTexture<T> {
    texture: T,
    format: DataTextureFormat,
    width: u32,
    rows: u32,
}

impl<T> DataTexture<T> {
    /// Wrap `texture`, which is `width` texels wide and has room for `rows` rows of data.
    pub(crate) fn new(texture: T, format: DataTextureFormat, width: u32, rows: u32) -> Self {
        Self {
            texture,
            format,
            width,
            rows,
        }
    }

    pub(crate) fn texture(&self) -> &T {
        &self.texture
    }

    pub(crate) fn format(&self) -> DataTextureFormat {
        self.format
    }

    #[cfg(feature = "webgl")]
    pub(crate) fn width(&self) -> u32 {
        self.width
    }

    pub(crate) fn rows_for_bytes(&self, len: usize) -> u32 {
        u32::try_from(len)
            .unwrap()
            .div_ceil(self.width * self.format.bytes_per_texel())
    }

    #[cfg(feature = "webgl")]
    pub(crate) fn rows_len(&self, rows: u32) -> usize {
        self.width as usize * rows as usize * self.format.bytes_per_texel() as usize
    }

    /// Grow the texture with `create` if it has no room for `len` bytes, and return whether it did.
    pub(crate) fn ensure_bytes<E>(
        &mut self,
        len: usize,
        create: impl FnOnce(DataTextureFormat, u32, u32) -> Result<T, E>,
    ) -> Result<bool, E> {
        self.ensure_rows(self.rows_for_bytes(len), create)
    }

    /// Grow the texture with `create` if it has no room for `texels` texels, and return whether it
    /// did.
    pub(crate) fn ensure_texels<E>(
        &mut self,
        texels: u32,
        create: impl FnOnce(DataTextureFormat, u32, u32) -> Result<T, E>,
    ) -> Result<bool, E> {
        self.ensure_rows(texels.div_ceil(self.width), create)
    }

    fn ensure_rows<E>(
        &mut self,
        rows: u32,
        create: impl FnOnce(DataTextureFormat, u32, u32) -> Result<T, E>,
    ) -> Result<bool, E> {
        if rows <= self.rows {
            return Ok(false);
        }

        self.texture = create(self.format, self.width, rows)?;
        self.rows = rows;
        Ok(true)
    }
}
