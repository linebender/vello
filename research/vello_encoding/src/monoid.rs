// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

/// Interface for a monoid. The default value must be the identity of
/// the monoid.
pub trait Monoid: Default {
    /// The source value for constructing the monoid.
    type SourceValue;

    /// Creates a monoid from a given value.
    fn new(value: Self::SourceValue) -> Self;

    /// Combines two monoids. This operation must be associative.
    #[must_use]
    fn combine(&self, other: &Self) -> Self;
}
