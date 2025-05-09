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
    ///    normal division by 255, but is much faster (see <https://github.com/linebender/vello/issues/904>)
    ///    and therefore preferable for the high-performance pipeline.
    ///
    /// Four properties worth mentioning:
    /// - This actually calculates the ceiling of `val / 256`.
    /// - Within the allowed range for `val`, rounding errors do not appear for values divisible by 255, i.e. any call `div_255(val * 255)` will always yield `val`.
    /// - If there is a discrepancy, this division will always yield a value 1 higher than the original.
    /// - This holds for values of `val` up to and including `65279`. You should not call this function with higher values.
    #[inline(always)]
    pub(crate) const fn div_255(val: u16) -> u16 {
        debug_assert!(
            val < 65280,
            "the properties of `div_255` do not hold for values of `65280` or greater"
        );
        (val + 255) >> 8
    }

    #[cfg(test)]
    mod tests {
        use crate::util::scalar::div_255;

        #[test]
        fn div_255_properties() {
            for i in 0_u16..256 * 255 {
                let expected = i / 255;
                let actual = div_255(i);

                assert!(
                    expected <= actual,
                    "In case of a discrepancy, the division should yield a value higher than the original."
                );

                let diff = expected.abs_diff(actual);
                assert!(diff <= 1, "Rounding error shouldn't be higher than 1.");

                if i % 255 == 0 {
                    assert_eq!(diff, 0, "Division should be accurate for multiples of 255.");
                }
            }
        }
    }
}
