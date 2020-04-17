//  Copyright 2020 The xi-editor authors.

//! New-style encoders (supporting proc macros)

pub struct A;

/// A reference to an encoded object within a buffer
#[derive(Clone, Copy, Debug)]
pub struct Ref<T> {
    offset: u32,
    _phantom: std::marker::PhantomData<T>,
}

pub struct Encoder {
    buf: Vec<u8>,
}

// TODO: we probably do want to encode slices, get rid of Sized bound
pub trait Encode: Sized {
    /// Size if it's a fixed-size object, otherwise 0.
    fn fixed_size() -> usize;

    /// Encoded size, for both fixed and variable sized objects.
    fn encoded_size(&self) -> usize {
        Self::fixed_size()
    }

    /// Encode into a buffer; panics if not appropriately sized.
    fn encode_to(&self, buf: &mut [u8]);

    /// Allocate a chunk and encode, returning a reference.
    fn encode(&self, encoder: &mut Encoder) -> Ref<Self> {
        let size = self.encoded_size();
        let (offset, buf) = encoder.alloc_chunk(size as u32);
        self.encode_to(buf);
        Ref::new(offset)
    }
}

impl<T> Ref<T> {
    fn new(offset: u32) -> Ref<T> {
        Ref {
            offset,
            _phantom: Default::default(),
        }
    }

    pub fn offset(&self) -> u32 {
        self.offset
    }

    pub fn transmute<U>(&self) -> Ref<U> {
        Ref::new(self.offset)
    }
}

impl Encoder {
    pub fn new() -> Encoder {
        Encoder { buf: Vec::new() }
    }

    pub fn alloc_chunk(&mut self, size: u32) -> (u32, &mut [u8]) {
        let offset = self.buf.len();
        self.buf.resize(size as usize + offset, 0);
        (offset as u32, &mut self.buf[offset..])
    }

    pub fn buf(&self) -> &[u8] {
        &self.buf
    }

    pub fn buf_mut(&mut self) -> &mut [u8] {
        &mut self.buf
    }
}

impl<T> Encode for Ref<T> {
    fn fixed_size() -> usize {
        4
    }

    fn encode_to(&self, buf: &mut [u8]) {
        buf[0..4].copy_from_slice(&self.offset.to_le_bytes());
    }
}

// Encode impls for scalar and small vector types are as needed; it's a finite set of
// possibilities, so we could do it all with macros, but by hand is expedient.

impl Encode for u32 {
    fn fixed_size() -> usize {
        4
    }

    fn encode_to(&self, buf: &mut [u8]) {
        buf[0..4].copy_from_slice(&self.to_le_bytes());
    }
}

impl Encode for f32 {
    fn fixed_size() -> usize {
        4
    }

    fn encode_to(&self, buf: &mut [u8]) {
        buf[0..4].copy_from_slice(&self.to_le_bytes());
    }
}

impl Encode for [u16; 4] {
    fn fixed_size() -> usize {
        8
    }

    fn encode_to(&self, buf: &mut [u8]) {
        buf[0..2].copy_from_slice(&self[0].to_le_bytes());
        buf[2..4].copy_from_slice(&self[1].to_le_bytes());
        buf[4..6].copy_from_slice(&self[2].to_le_bytes());
        buf[6..8].copy_from_slice(&self[3].to_le_bytes());
    }
}

impl Encode for [f32; 2] {
    fn fixed_size() -> usize {
        8
    }

    fn encode_to(&self, buf: &mut [u8]) {
        buf[0..4].copy_from_slice(&self[0].to_le_bytes());
        buf[4..8].copy_from_slice(&self[1].to_le_bytes());
    }
}

// TODO: make this work for slices too, but need to deal with Sized bound
//
// Note: only works for vectors of fixed size objects.
impl<T: Encode> Encode for Vec<T> {
    fn fixed_size() -> usize {
        0
    }
    fn encoded_size(&self) -> usize {
        self.len() * T::fixed_size()
    }

    fn encode_to(&self, buf: &mut [u8]) {
        let size = T::fixed_size();
        for (ix, val) in self.iter().enumerate() {
            val.encode_to(&mut buf[ix * size..]);
        }
    }
}
