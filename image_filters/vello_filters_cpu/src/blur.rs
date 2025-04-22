// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Implementation of blur filters.
//!
//! Blurs are a well of complexity, with both multiple possible operations and multiple implementation strategies.
//! The operation most commonly called "blur" is a [Gaussian blur][wikipedia::gaussian_blur], however there are also several others.
//!
//! ## Gaussian blurs
//!
//! A theoretically perfect Gaussian blur would take into account all pixels in the input image, with a different
//! weighting per output pixel.
//! This would have abyssmal performance, as it would be quadratic in the number of pixels in the image.
//! However, because this blur also has an exponential decay, it turns out to be reasonably easy to approximate.
//! There are however several possible approximations which we can use.
//! A good overview of possible implementation strategies is Gwosdek, P., Grewenig, S., Bruhn, A., Weickert, J. (2012).
//! (Theoretical Foundations of Gaussian Convolution by Extended Box Filtering).
//!
//! This can be found at
//! <https://www.mia.uni-saarland.de/Publications/gwosdek-ssvm11.pdf>.
//! Their performance numbers are based on an approximately 20 year old CPU
//! (I have been unable to determine the exact model number), but give a flavour for the performance tradeoffs.
//!
//! We currently implement only the "conventional box" filter described in that paper.
//! We plan to implement at least the truncated Gaussian form (useful for small values of sigma),
//! and potentially the extended box style.
//!
//! [wikipedia::gaussian_blur]: https://en.wikipedia.org/wiki/Gaussian_blur.

use tracing::warn;

use crate::ColorInterpolationFilters;

pub mod conventional_box;
// pub mod gaussian_truncated;

pub struct BlurParams {
    quality: BlurQuality,
    interpolation: ColorInterpolationFilters,
}

pub enum BlurQuality {
    Approx,
    Max,
    AlwaysApprox,
}

pub fn gaussian_blur(params: &BlurParams) {
    match params.quality {
        BlurQuality::Max | BlurQuality::Approx => {
            warn!("High quality blurs not yet implemented. Falling back to three-pass box blur");
        }
        BlurQuality::AlwaysApprox => {}
    }
}
