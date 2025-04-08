// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

pub(crate) mod scalar {
    /// Perform an approximate division by 255.
    ///
    /// There are three reasons for having this method.
    /// 1) Divisions are slower than shifting + adding, and the compiler does not seem to replace
    ///    divisions by 255 with an equivalent (this was verified by benchmarking; doing / 255 was
    ///    significantly slower).
    /// 2) Integer divisions are usually not available in SIMD, so this provides a good baseline
    ///    implementation.
    /// 3) There are two options for performing the division: One is to perform the division
    ///    in a way that completely preserves the rounding semantics of a integer division by
    ///    255. This could be achieved using the implementation `(val + 1 + (val >> 8)) >> 8`.
    ///    The second approach (used here) has slightly different rounding behavior to a
    ///    normal division by 255, but is much faster (see https://github.com/linebender/vello/issues/904)
    ///    and therefore preferable for the high-performance pipeline.
    ///
    /// Two properties worth mentioning:
    /// - Rounding errors do not appear for values divisible by 255, i.e. any call div_255(x * 255) will always yield x.
    /// - If there is a discrepancy, this division will always yield a value 1 higher than the original.
    #[inline(always)]
    pub(crate) const fn div_255(val: u16) -> u16 {
        (val + 255) >> 8
    }

    #[cfg(test)]
    mod tests {
        use crate::util::scalar::div_255;

        #[test]
        fn division() {
            for i in 0u16..=255 {
                for j in 0u16..=255 {
                    let mulled = i * j;
                    let expected = mulled / 255;
                    let actual = div_255(mulled);

                    let diff = (expected as i32 - actual as i32).abs();

                    // Rounding error shouldn't be higher than 1.
                    assert!(diff <= 1);

                    if i == 255 || j == 255 {
                        // Division should be accurate for multiples of 255.
                        assert!(diff <= 0);
                    }
                }
            }
        }
    }
}
