// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Textures that shaders read packed data from.

use crate::ResourceTexture;

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
/// the data uploaded to it, up to its width.
#[derive(Debug)]
pub(crate) struct DataTexture<T> {
    texture: T,
    kind: ResourceTexture,
    format: DataTextureFormat,
    width: u32,
    rows: u32,
}

/// The data for a [`DataTexture`] needs more rows than the texture's width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DataTextureTooLarge {
    pub(crate) texture: ResourceTexture,
    pub(crate) rows: u32,
    pub(crate) max_rows: u32,
}

impl<T> DataTexture<T> {
    /// Wrap `texture`, which is `width` texels wide and has room for `rows` rows of data.
    pub(crate) fn new(
        texture: T,
        kind: ResourceTexture,
        format: DataTextureFormat,
        width: u32,
        rows: u32,
    ) -> Self {
        Self {
            texture,
            kind,
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
    pub(crate) fn ensure_bytes<E: From<DataTextureTooLarge>>(
        &mut self,
        len: usize,
        create: impl FnOnce(DataTextureFormat, u32, u32) -> Result<T, E>,
    ) -> Result<bool, E> {
        self.ensure_rows(self.rows_for_bytes(len), create)
    }

    /// Grow the texture with `create` if it has no room for `texels` texels, and return whether it
    /// did.
    pub(crate) fn ensure_texels<E: From<DataTextureTooLarge>>(
        &mut self,
        texels: u32,
        create: impl FnOnce(DataTextureFormat, u32, u32) -> Result<T, E>,
    ) -> Result<bool, E> {
        self.ensure_rows(texels.div_ceil(self.width), create)
    }

    fn ensure_rows<E: From<DataTextureTooLarge>>(
        &mut self,
        rows: u32,
        create: impl FnOnce(DataTextureFormat, u32, u32) -> Result<T, E>,
    ) -> Result<bool, E> {
        if rows <= self.rows {
            return Ok(false);
        }

        // Data textures are as wide as the maximum resource texture dimension, which also limits
        // their height.
        if rows > self.width {
            return Err(DataTextureTooLarge {
                texture: self.kind,
                rows,
                max_rows: self.width,
            }
            .into());
        }

        self.texture = create(self.format, self.width, rows)?;
        self.rows = rows;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data_texture() -> DataTexture<u32> {
        DataTexture::new(
            1,
            ResourceTexture::Gradients,
            DataTextureFormat::Rgba32Uint,
            4,
            1,
        )
    }

    #[test]
    fn grows_to_whole_rows() {
        let mut texture = data_texture();
        let grown = texture.ensure_bytes::<DataTextureTooLarge>(65, |_, width, rows| {
            assert_eq!((width, rows), (4, 2));
            Ok(2)
        });
        assert_eq!(grown, Ok(true));
        assert_eq!(*texture.texture(), 2);
    }

    #[test]
    fn keeps_texture_when_data_fits() {
        let mut texture = data_texture();
        let grown = texture.ensure_texels::<DataTextureTooLarge>(4, |_, _, _| unreachable!());
        assert_eq!(grown, Ok(false));
        assert_eq!(*texture.texture(), 1);
    }

    #[test]
    fn accepts_as_many_rows_as_width() {
        let mut texture = data_texture();
        let grown = texture.ensure_texels::<DataTextureTooLarge>(16, |_, width, rows| {
            assert_eq!((width, rows), (4, 4));
            Ok(2)
        });
        assert_eq!(grown, Ok(true));
        assert_eq!(*texture.texture(), 2);
    }

    #[test]
    fn rejects_more_rows_than_width() {
        let mut texture = data_texture();
        let grown = texture.ensure_texels::<DataTextureTooLarge>(17, |_, _, _| unreachable!());
        assert_eq!(
            grown,
            Err(DataTextureTooLarge {
                texture: ResourceTexture::Gradients,
                rows: 5,
                max_rows: 4,
            })
        );
        assert_eq!(*texture.texture(), 1);
        assert_eq!(texture.rows, 1);
    }
}
