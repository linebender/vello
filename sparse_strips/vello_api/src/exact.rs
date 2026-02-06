// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Turning shapes into Bézier paths without approximation.

use kurbo::{BezPath, CubicBez, Line, QuadBez, Rect, Segments, Shape, Triangle, segments};
use peniko::kurbo::{self, PathEl};

/// A generic trait for shapes that can be mapped exactly to Bézier path elements (i.e., without
/// approximation).
///
/// The methods on [`PaintScene`](crate::PaintScene) use this trait ensure that consistent behaviour
/// is maintained when Vello API is used to render content which might be rescaled.
///
/// This is implemented for [`Shape`]s from Kurbo that can be exactly mapped to Bézier path elements.
/// To convert a [`Shape`] which requires approximation (such as [`Circle`](kurbo::Circle) or
/// [`RoundedRect`](kurbo::RoundedRect)), you can use [`within`].
/// This however requires you to provide the tolerance.
/// See the docs of `within` for more details.
///
/// It is a requirement of this trait that [`Shape::path_elements`] returns the same iterator
/// as [`ExactPathElements::exact_path_elements`] for any provided tolerance value.
pub trait ExactPathElements {
    /// The iterator returned by the [`Self::exact_path_elements`] method.
    ///
    /// In many cases, this will be the same iterator as [`Shape::path_elements`]
    type ExactPathElementsIter<'iter>: Iterator<Item = PathEl> + 'iter
    where
        Self: 'iter;

    /// Returns an iterator over this shape expressed as exact [`PathEl`]s;
    /// that is, as exact Bézier path _elements_.
    ///
    /// These path elements are exact, in the sense that no approximation is required
    /// to calculate them. This is not possible for all shapes, but is possible for all
    /// finite polygons and other piecewise-cubic parametric curves. Some shapes will
    /// need to be approximated, which [`Shape::path_elements`] does instead.
    /// The [`within`] helper function allows the approximated shape to be used
    /// where an exact value is required, by providing the tolerance used.
    ///
    /// In many cases, shapes are able to iterate their elements without
    /// allocating; however creating a [`BezPath`] object always allocates.
    /// If you need an owned [`BezPath`] you can use [`BezPath::from_iter`] (or
    /// [`Iterator::collect`]).
    fn exact_path_elements(&self) -> Self::ExactPathElementsIter<'_>;

    /// Returns an iterator over this shape expressed as exact Bézier path
    /// _segments_ ([`PathSeg`]s).
    ///
    /// The allocation behaviour is the same as for [`ExactPathElements::exact_path_elements`].
    ///
    /// [`PathSeg`]: kurbo::PathSeg
    #[inline]
    fn exact_path_segments(&self) -> Segments<Self::ExactPathElementsIter<'_>> {
        segments(self.exact_path_elements())
    }
}

impl<T: ExactPathElements> ExactPathElements for &T {
    type ExactPathElementsIter<'iter>
        = T::ExactPathElementsIter<'iter>
    where
        Self: 'iter;

    #[inline]
    fn exact_path_elements(&self) -> Self::ExactPathElementsIter<'_> {
        (*self).exact_path_elements()
    }
}

/// Use an approximated shape where an [`ExactPathElements`] is required, by approximating it to
/// within the given tolerance.
///
/// *WARNING*: Unlike [`ExactPathElements`], which will produce correct renderings for any scale, this
/// approximation is only valid for a fixed range of transforms.
///
/// As the user of this function, you are responsible for determining the correct tolerance for your use case.
/// A reasonable approach might be to select a tolerance which allows scaling up ("zooming in") by 4x (for example;
/// you should evaluate the correct value yourself) and remaining within your intended tolerance bound.
/// If the user zoomed past that limit, you would then recomputing the [`Scene`](crate::Scene) from your base data representation once that is exceeded.
/// If you know that the shape will not be scaled, you can use [`UNSCALED_TOLERANCE`].
/// The resulting path will be within 1/10th of a pixel of the actual shape, which is a negligible
/// difference for rendering.
///
/// This is useful for drawing shapes such as [`Circle`](kurbo::Circle)s and [`RoundedRect`](kurbo::RoundedRect)s.
// TODO: Provide as an extension trait?
pub fn within<S: Shape>(shape: S, tolerance: f64) -> WithTolerance<S> {
    WithTolerance { shape, tolerance }
}

/// The type used to implement [`within`].
#[derive(Debug, Clone, Copy)]
pub struct WithTolerance<S: Shape> {
    /// The shape to be approximated.
    pub shape: S,
    /// The tolerance to use when approximating it.
    pub tolerance: f64,
}

impl<S: Shape> ExactPathElements for WithTolerance<S> {
    type ExactPathElementsIter<'iter>
        = S::PathElementsIter<'iter>
    where
        S: 'iter;
    fn exact_path_elements(&self) -> Self::ExactPathElementsIter<'_> {
        self.shape.path_elements(self.tolerance)
    }
}

/// The recommended tolerance for approximating shapes which won't be rescaled as Bezier paths.
///
/// This can be used as the tolerance parameter to [`within`], when you know that
/// a shape will be drawn at its natural size, without scaling or skew.
pub const UNSCALED_TOLERANCE: f64 = 0.1;

// TODO: Provide an `fn ideal_tolerance(transform: Affine) -> f64`

/// Implement `ExactPathElements` for an existing [`Shape`], which we know will not be approximated in Kurbo.
// In theory, the impl is the wrong way around; instead the `Shape` impl should be in terms of `ExactPathElements`.
macro_rules! passthrough {
    ($ty: ty) => {
        impl ExactPathElements for $ty {
            type ExactPathElementsIter<'iter>
                = <$ty as Shape>::PathElementsIter<'iter>
            where
                $ty: 'iter;

            fn exact_path_elements(&self) -> Self::ExactPathElementsIter<'_> {
                // We use a tolerance of zero here because we know this to be exact.
                self.path_elements(0.)
            }
        }
    };
}

passthrough!(BezPath);
passthrough!(CubicBez);
passthrough!(Line);
passthrough!(QuadBez);
passthrough!(Rect);
passthrough!(Triangle);
