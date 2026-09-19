// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared packing/unpacking helpers.

fn unpack_u16_pair(value: u32) -> vec2<u32> {
    return vec2<u32>(value & 0xffffu, value >> 16u);
}
