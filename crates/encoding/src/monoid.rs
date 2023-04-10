// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

/// Interface for a monoid. The default value must be the identity of
/// the monoid.
pub trait Monoid: Default {
    /// The source value for constructing the monoid.
    type SourceValue;

    /// Creates a monoid from a given value.
    fn new(value: Self::SourceValue) -> Self;

    /// Combines two monoids. This operation must be associative.
    fn combine(&self, other: &Self) -> Self;
}
