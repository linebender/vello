use fearless_simd::{f32x16, u8x16, Simd, SimdInto};

// TODO: Remove `f32_to_u8` from `vello_cpu` and use this one!
/// Convert `f32x16` to `u8x16`.
#[inline(always)]
pub fn f32_to_u8<S: Simd>(val: f32x16<S>) -> u8x16<S> {
    let simd = val.simd;
    let converted = simd.cvt_u32_f32x16(val);

    [
        converted[0] as u8,
        converted[1] as u8,
        converted[2] as u8,
        converted[3] as u8,
        converted[4] as u8,
        converted[5] as u8,
        converted[6] as u8,
        converted[7] as u8,
        converted[8] as u8,
        converted[9] as u8,
        converted[10] as u8,
        converted[11] as u8,
        converted[12] as u8,
        converted[13] as u8,
        converted[14] as u8,
        converted[15] as u8,
    ]
        .simd_into(val.simd)
}