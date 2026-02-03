// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter effects API based on the W3C Filter Effects specification.
//!
//! This module provides a comprehensive filter system supporting both high-level
//! CSS filter functions and low-level SVG filter primitives. The API is designed
//! to follow the W3C Filter Effects Module Level 1 specification.
//!
//! See: <https://drafts.fxtf.org/filter-effects/>
//!
//! ## Implementation Status
//!
//! ### ✅ Implemented
//!
//! **Filter Functions:**
//! - `Blur` - Gaussian blur effect
//!
//! **Filter Primitives (Single Use Only):**
//! - `Flood` - Solid color fill
//! - `GaussianBlur` - Gaussian blur filter
//! - `DropShadow` - Drop shadow effect (compound primitive)
//! - `Offset` - Translation/shift (single primitive)
//!
//! **Note:** Currently only single primitive filters are supported. Filter graphs with
//! multiple connected primitives are not yet implemented.
//!
//! ### 🚧 Not Yet Implemented
//!
//! **Core Features:**
//! - `FilterGraph` execution - Chaining multiple filter primitives together
//! - `FilterInputs` - Connecting primitives to create complex effects
//!
//! **Filter Functions:**
//! - `Brightness`, `Contrast`, `Grayscale`, `HueRotate`, `Invert`,
//!   `Opacity`, `Saturate`, `Sepia`
//!
//! **Filter Primitives:**
//! - `ColorMatrix` - Matrix-based color transformation
//! - `Composite` - Porter-Duff compositing operations
//! - `Blend` - Blend mode operations
//! - `Morphology` - Dilate/erode operations
//! - `ConvolveMatrix` - Custom convolution kernels
//! - `Turbulence` - Perlin noise generation
//! - `DisplacementMap` - Pixel displacement
//! - `ComponentTransfer` - Per-channel transfer functions
//! - `Image` - External image reference
//! - `Tile` - Tiling operation
//! - `DiffuseLighting`, `SpecularLighting` - Lighting effects

use crate::color::{AlphaColor, Srgb};
use crate::kurbo::{Affine, Rect};
use alloc::sync::Arc;
use alloc::vec::Vec;
use smallvec::SmallVec;

/// The main filter system.
///
/// A filter combines a graph of filter primitives with optional spatial bounds.
/// If bounds are specified, the filter only applies within that region.
#[derive(Debug, Clone, PartialEq)]
pub struct Filter {
    /// Filter graph defining the effect pipeline.
    pub graph: Arc<FilterGraph>,
    // TODO: Add bounds restricting where the filter applies.
    // Optional bounds restricting where the filter applies.
    // If `None`, the filter applies to the entire filtered element.
    // pub bounds: Option<Rect>,
}

impl Filter {
    /// Create a simple filter system from a filter function.
    ///
    /// Converts a high-level CSS-style filter function into a filter graph.
    /// Use this for simple effects like blur, brightness, etc.
    pub fn from_function(function: FilterFunction) -> Self {
        // Convert function to primitive
        let primitive = match function {
            FilterFunction::Blur { radius } => FilterPrimitive::GaussianBlur {
                std_deviation: radius,
                edge_mode: EdgeMode::default(),
            },
            _ => unimplemented!("Filter function {:?} not supported", function),
        };

        Self::from_primitive(primitive)
    }

    /// Create a filter system from a filter primitive.
    ///
    /// Creates a simple filter graph with a single primitive.
    /// Use this for direct access to low-level SVG filter operations.
    pub fn from_primitive(primitive: FilterPrimitive) -> Self {
        let mut graph = FilterGraph::new();
        let filter_id = graph.add(primitive, None);
        graph.set_output(filter_id);

        Self {
            graph: Arc::new(graph),
        }
    }

    /// Calculate the bounds expansion for this filter in pixel/device space.
    ///
    /// Returns a `Rect` representing how many extra pixels are needed around the
    /// filtered region to correctly compute the filter effect. For example, a blur
    /// filter needs to sample beyond the original bounds to avoid edge artifacts.
    ///
    /// The expansion accounts for the transform (rotation, scale, and shear) to compute
    /// the correct axis-aligned bounding box expansion in device space.
    ///
    /// The returned rect is centered at origin:
    /// - x0: negative left expansion (in pixels)
    /// - y0: negative top expansion (in pixels)
    /// - x1: positive right expansion (in pixels)
    /// - y1: positive bottom expansion (in pixels)
    ///
    /// # Arguments
    /// * `transform` - The transform applied to this filter layer
    pub fn bounds_expansion(&self, transform: &Affine) -> Rect {
        let [a, b, c, d, _e, _f] = transform.as_coeffs();
        let linear_only = Affine::new([a, b, c, d, 0.0, 0.0]);

        self.graph.bounds_expansion(&linear_only)
    }
}

/// A directed acyclic graph (DAG) of filter operations.
///
/// The graph represents a pipeline of filter primitives where outputs of some
/// primitives can be used as inputs to others. Each primitive has a unique `FilterId`.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterGraph {
    /// All filter primitives in the graph, stored in insertion order.
    pub primitives: SmallVec<[FilterPrimitive; 1]>,
    /// The final output filter ID whose result is the output of this graph.
    pub output: FilterId,
    /// Next available filter ID (monotonically increasing counter).
    next_id: u16,
    /// Accumulated bounds expansion from all primitives in the graph, cached in user space.
    /// This is the axis-aligned bounding box of the expansion region (centered at origin),
    /// which can be transformed to device space when needed.
    expansion_rect: Rect,
}

impl Default for FilterGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl FilterGraph {
    /// Create a new empty filter graph.
    pub fn new() -> Self {
        Self {
            primitives: SmallVec::new(),
            output: FilterId(0),
            next_id: 0,
            expansion_rect: Rect::ZERO,
        }
    }

    /// Add a filter primitive with optional inputs.
    ///
    /// Returns a `FilterId` that can be referenced by other primitives.
    /// Automatically updates the accumulated bounds expansion based on the primitive's requirements.
    pub fn add(&mut self, primitive: FilterPrimitive, _inputs: Option<FilterInputs>) -> FilterId {
        let id = FilterId(self.next_id);
        self.next_id += 1;

        // Update accumulated expansion by taking the union of rects
        let primitive_rect = primitive.expansion_rect();
        self.expansion_rect = self.expansion_rect.union(primitive_rect);

        self.primitives.push(primitive);

        id
    }

    /// Set the output filter for the graph.
    pub fn set_output(&mut self, output: FilterId) {
        self.output = output;
    }

    /// Get the accumulated bounds expansion for all primitives in this graph.
    ///
    /// This returns the expansion required by all primitives in the graph,
    /// representing the padding needed to render all filter effects correctly.
    ///
    /// The expansion accounts for the transform (rotation, scale, and shear) to compute
    /// the correct axis-aligned bounding box expansion in device space.
    ///
    /// # Arguments
    /// * `transform` - The transform applied to this filter layer
    pub fn bounds_expansion(&self, transform: &Affine) -> Rect {
        // Transform the cached expansion rect to device space
        // transform_rect_bbox computes the axis-aligned bounding box of the transformed rect
        transform.transform_rect_bbox(self.expansion_rect)
    }
}

/// All possible filter effects.
///
/// This enum allows choosing between high-level filter functions (simple CSS-style effects)
/// and low-level filter primitives (complex SVG-style effects with full control).
/// Use `FilterFunction` for common effects like blur, and `FilterPrimitive` for
/// advanced composition and custom filter graphs.
#[derive(Debug, Clone)]
pub enum FilterEffect {
    /// Simple, high-level filter functions.
    Function(FilterFunction),
    /// Low-level filter primitives (granular control).
    Primitive(FilterPrimitive),
}

/// High-level filter functions for common effects (CSS filter functions).
///
/// These match the CSS Filter Effects specification and provide simple,
/// commonly-used visual effects without needing to construct a filter graph.
///
/// See: <https://drafts.fxtf.org/filter-effects/#filter-functions>
#[derive(Debug, Clone)]
pub enum FilterFunction {
    /// Gaussian blur effect.
    ///
    /// Applies a Gaussian blur to the input image. Larger radius values
    /// produce more blur. The blur is applied equally in all directions.
    ///
    /// Note: Per the W3C Filter Effects specification, this `radius` parameter
    /// represents the standard deviation (σ) of the Gaussian function, not the
    /// effective blur range. The effective blur range is approximately 3× this value.
    Blur {
        /// Standard deviation of the Gaussian blur in pixels. Must be non-negative.
        /// A value of 0 means no blur.
        ///
        /// Despite being called "radius" (to match CSS filter syntax), this is
        /// actually the standard deviation. The visible blur effect extends
        /// approximately 3 times this value in each direction.
        radius: f32,
    },
    //
    // ============================================================
    // TODO: The following filter functions are not yet implemented
    // ============================================================
    //
    /// Brightness adjustment.
    ///
    /// Adjusts the brightness of the input image using a linear multiplier.
    Brightness {
        /// Brightness amount: 0.0 = completely black, 1.0 = no change, >1.0 = brighter.
        /// Must be non-negative.
        amount: f32,
    },
    /// Contrast adjustment.
    ///
    /// Adjusts the contrast of the input image.
    Contrast {
        /// Contrast amount: 0.0 = uniform gray, 1.0 = no change, >1.0 = higher contrast.
        /// Must be non-negative.
        amount: f32,
    },
    /// Grayscale conversion.
    ///
    /// Converts the input to grayscale. Amount controls the strength of the conversion.
    Grayscale {
        /// Grayscale amount: 0.0 = original colors, 1.0 = full grayscale.
        /// Values should be in range [0.0, 1.0].
        amount: f32,
    },
    /// Hue rotation.
    ///
    /// Rotates the hue of all colors in the input image by the specified angle.
    HueRotate {
        /// Rotation angle in degrees. Can be negative.
        /// 0° = no change, 180° = opposite hue, 360° = back to original.
        angle: f32,
    },
    /// Color inversion.
    ///
    /// Inverts the colors of the input image.
    Invert {
        /// Inversion amount: 0.0 = original colors, 1.0 = fully inverted.
        /// Values should be in range [0.0, 1.0].
        amount: f32,
    },
    /// Opacity adjustment.
    ///
    /// Multiplies the alpha channel by the specified amount.
    Opacity {
        /// Opacity amount: 0.0 = fully transparent, 1.0 = no change.
        /// Values should be in range [0.0, 1.0].
        amount: f32,
    },
    /// Saturation adjustment.
    ///
    /// Adjusts the color saturation of the input image.
    Saturate {
        /// Saturation amount: 0.0 = completely desaturated (grayscale),
        /// 1.0 = no change, >1.0 = oversaturated.
        /// Must be non-negative.
        amount: f32,
    },
    /// Sepia tone effect.
    ///
    /// Applies a sepia tone effect (vintage/old photo appearance).
    Sepia {
        /// Sepia amount: 0.0 = original colors, 1.0 = full sepia tone.
        /// Values should be in range [0.0, 1.0].
        amount: f32,
    },
}

/// Edge mode for filter operations.
///
/// Determines how to extend the input image when filter operations require sampling
/// beyond the original image boundaries. This is particularly important for blur and
/// convolution operations near edges.
///
/// See: <https://drafts.fxtf.org/filter-effects/#element-attrdef-filter-primitive-edgemode>
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum EdgeMode {
    /// Extend by duplicating edge pixels (clamp to edge).
    ///
    /// The input image is extended along each border by replicating the color values
    /// at the given edge of the input image. This prevents dark halos around edges.
    Duplicate,
    /// Extend by wrapping to the opposite edge (repeat/tile).
    ///
    /// The input image is extended by taking color values from the opposite edge,
    /// creating a tiling effect.
    Wrap,
    /// Extend by mirroring across the edge.
    ///
    /// The input image is extended by taking color values mirrored across the edge.
    /// This creates seamless continuation at boundaries.
    Mirror,
    /// Extend with transparent black (zeros).
    ///
    /// The input image is extended with pixel values of zero for R, G, B and A.
    /// This is the default and most common mode, creating natural fade-to-transparent edges.
    #[default]
    None,
}

/// Low-level filter primitives for granular control (SVG filter primitives).
///
/// These are the building blocks for complex filter effects, corresponding to SVG
/// filter primitives. They can be combined in a `FilterGraph` to create sophisticated
/// visual effects.
///
/// See: <https://drafts.fxtf.org/filter-effects/#FilterPrimitivesOverview>
#[derive(Debug, Clone, PartialEq)]
pub enum FilterPrimitive {
    /// Generate a solid color fill.
    ///
    /// Creates a rectangle filled with the specified color, typically used as
    /// input to other filter operations (e.g., for colored shadows).
    Flood {
        /// Fill color with alpha channel.
        color: AlphaColor<Srgb>,
    },
    /// Gaussian blur filter.
    ///
    /// Applies a Gaussian blur using the specified standard deviation (σ).
    /// The effective blur range (distance over which pixels are sampled) is
    /// approximately 3 × `std_deviation`, as this captures ~99.7% of the
    /// Gaussian distribution.
    GaussianBlur {
        /// Standard deviation for the blur kernel. Larger values create more blur.
        /// Must be non-negative. A value of 0 means no blur.
        ///
        /// This directly corresponds to the σ (sigma) parameter in the Gaussian
        /// function. The visible blur effect extends approximately 3σ in each direction.
        ///
        /// TODO: Per the W3C specification, this should support separate x and y values.
        /// The spec allows `stdDeviation` to be either one number (applied to both axes)
        /// or two numbers (first for x-axis, second for y-axis). Currently only uniform
        /// blur is supported. Consider changing to `(f32, f32)` or a dedicated type.
        std_deviation: f32,
        /// Edge mode determining how pixels beyond the input bounds are handled.
        edge_mode: EdgeMode,
    },
    /// Drop shadow effect (compound primitive).
    ///
    /// Creates a drop shadow by blurring the input's alpha channel, offsetting it,
    /// and compositing it with the original. This is a compound operation that
    /// combines multiple primitive operations into one.
    ///
    /// See: <https://drafts.fxtf.org/filter-effects-2/#feDropShadowElement>
    DropShadow {
        /// Horizontal offset of the shadow in pixels. Positive values shift right.
        dx: f32,
        /// Vertical offset of the shadow in pixels. Positive values shift down.
        dy: f32,
        /// Blur standard deviation for the shadow. Larger values create softer shadows.
        std_deviation: f32,
        /// Shadow color with alpha channel. Alpha controls shadow opacity.
        color: AlphaColor<Srgb>,
        /// Edge mode for handling boundaries during blur operation.
        /// Default is `EdgeMode::None` per SVG spec.
        edge_mode: EdgeMode,
    },
    //
    // ============================================================
    // TODO: The following filter primitives are not yet implemented
    // ============================================================
    //
    /// Matrix-based color transformation.
    ///
    /// Applies a 4x5 matrix transformation to colors, allowing arbitrary
    /// color space transformations, hue shifts, and color adjustments.
    ColorMatrix {
        /// 4x5 color transformation matrix: 4 rows (R,G,B,A) × 5 columns (R,G,B,A,offset).
        /// Each output channel is computed as a linear combination of input channels plus offset.
        matrix: [f32; 20],
    },
    /// Geometric offset/translation.
    ///
    /// Shifts the input image by the specified offset. Useful for creating
    /// shadow effects or positioning elements in a filter graph.
    Offset {
        /// Horizontal offset in pixels. Positive values shift right.
        dx: f32,
        /// Vertical offset in pixels. Positive values shift down.
        dy: f32,
    },

    /// Composite two inputs using Porter-Duff compositing operations.
    ///
    /// Combines two input images using standard compositing operators
    /// (over, in, out, atop, xor) or custom arithmetic combination.
    Composite {
        /// Porter-Duff compositing operator to apply.
        operator: CompositeOperator,
    },
    /// Blend two inputs using blend modes.
    ///
    /// Combines two input images using Photoshop-style blend modes
    /// (multiply, screen, overlay, etc.).
    Blend {
        /// Blend mode determining how colors are combined.
        mode: BlendMode,
    },
    /// Morphological operations (dilate/erode).
    ///
    /// Expands (dilate) or contracts (erode) the shapes in the input image.
    /// Useful for creating outline effects or cleaning up edges.
    Morphology {
        /// Morphological operator determining whether to erode or dilate.
        operator: MorphologyOperator,
        /// Operation radius in pixels. Larger values create stronger effects.
        radius: f32,
    },
    /// Custom convolution kernel for image processing.
    ///
    /// Applies a custom convolution matrix to the input image, enabling
    /// effects like sharpening, edge detection, embossing, and custom filters.
    ConvolveMatrix {
        /// Convolution kernel specification including size, values, and normalization.
        kernel: ConvolutionKernel,
    },
    /// Generate Perlin noise/turbulence patterns.
    ///
    /// Creates procedural noise patterns useful for textures, clouds,
    /// marble effects, and other organic-looking randomness.
    Turbulence {
        /// Base frequency for noise generation. Higher values create finer detail.
        base_frequency: f32,
        /// Number of octaves for fractal noise. More octaves add finer detail.
        num_octaves: u32,
        /// Random seed for reproducible noise generation.
        seed: u32,
        /// Type of noise: smooth fractal or more chaotic turbulence.
        turbulence_type: TurbulenceType,
    },
    /// Displace pixels using a displacement map.
    ///
    /// Uses the color values from a second input to spatially displace pixels
    /// in the primary input, creating warping and distortion effects.
    DisplacementMap {
        /// Scale factor controlling the displacement intensity.
        scale: f32,
        /// Color channel from the displacement map used for X-axis displacement.
        x_channel: ColorChannel,
        /// Color channel from the displacement map used for Y-axis displacement.
        y_channel: ColorChannel,
    },
    /// Per-channel component transfer using lookup tables or functions.
    ///
    /// Applies independent transfer functions to each color channel,
    /// enabling color corrections, gamma adjustments, and custom mappings.
    ComponentTransfer {
        /// Transfer function applied to the red channel (None = identity).
        red_function: Option<TransferFunction>,
        /// Transfer function applied to the green channel (None = identity).
        green_function: Option<TransferFunction>,
        /// Transfer function applied to the blue channel (None = identity).
        blue_function: Option<TransferFunction>,
        /// Transfer function applied to the alpha channel (None = identity).
        alpha_function: Option<TransferFunction>,
    },
    /// Reference an external image as filter input.
    ///
    /// Allows using pre-existing images (from an atlas or resource) as
    /// input to filter operations, useful for texturing and overlays.
    Image {
        /// Identifier referencing an image in the resource atlas.
        image_id: u32,
        /// Optional 2D affine transformation matrix [a, b, c, d, e, f].
        /// Transforms the image before using it as filter input.
        transform: Option<[f32; 6]>,
    },
    /// Tile the input to fill the filter region.
    ///
    /// Repeats the input image to fill the entire filter primitive subregion,
    /// creating a tiling/repeating pattern.
    Tile,
    /// Diffuse lighting simulation.
    ///
    /// Creates a lighting effect by treating the input's alpha channel as a height map
    /// and calculating diffuse (matte) reflection from a light source.
    DiffuseLighting {
        /// Surface scale factor for converting alpha values to heights.
        surface_scale: f32,
        /// Diffuse reflection constant (kd). Controls lighting intensity.
        diffuse_constant: f32,
        /// Kernel unit length for gradient calculations in user space.
        kernel_unit_length: f32,
        /// Configuration of the light source (point, distant, or spot).
        light_source: LightSource,
    },
    /// Specular lighting simulation.
    ///
    /// Creates a lighting effect by treating the input's alpha channel as a height map
    /// and calculating specular (shiny) reflection highlights from a light source.
    SpecularLighting {
        /// Surface scale factor for converting alpha values to heights.
        surface_scale: f32,
        /// Specular reflection constant (ks). Controls highlight intensity.
        specular_constant: f32,
        /// Specular reflection exponent. Controls highlight sharpness (higher = sharper).
        specular_exponent: f32,
        /// Kernel unit length for gradient calculations in user space.
        kernel_unit_length: f32,
        /// Configuration of the light source (point, distant, or spot).
        light_source: LightSource,
    },
}

impl FilterPrimitive {
    /// Calculate the bounds expansion as a `Rect` in user space.
    ///
    /// Returns a rectangle centered at the origin representing how much the filter
    /// expands the processing region in each direction. The rect coordinates are:
    /// - x0: negative left expansion
    /// - y0: negative top expansion
    /// - x1: positive right expansion
    /// - y1: positive bottom expansion
    ///
    /// A `Rect::ZERO` means no expansion. This representation allows the expansion
    /// to be correctly transformed (including rotation) using standard rect transforms.
    ///
    /// For example, a blur filter needs additional pixels around the edges (3*sigma).
    /// Most filters that don't sample neighboring pixels return `Rect::ZERO`.
    pub fn expansion_rect(&self) -> Rect {
        match self {
            Self::GaussianBlur { std_deviation, .. } => {
                // Gaussian blur expands uniformly by 3*sigma (covers 99.7% of distribution)
                let radius = (*std_deviation * 3.0) as f64;
                Rect::new(-radius, -radius, radius, radius)
            }
            Self::Offset { dx, dy } => {
                // Offset shifts pixels; expand bounds asymmetrically so shifted content isn't cut.
                let dx = *dx as f64;
                let dy = *dy as f64;
                Rect::new(dx.min(0.0), dy.min(0.0), dx.max(0.0), dy.max(0.0))
            }
            Self::DropShadow {
                std_deviation,
                dx,
                dy,
                ..
            } => {
                // Drop shadow = blur + offset + composite with original
                // The expansion rect encompasses both the blur and the offset
                let blur_radius = (*std_deviation * 3.0) as f64;
                let dx = *dx as f64;
                let dy = *dy as f64;

                Rect::new(
                    -(blur_radius + (-dx).max(0.0)),
                    -(blur_radius + (-dy).max(0.0)),
                    blur_radius + dx.max(0.0),
                    blur_radius + dy.max(0.0),
                )
            }
            // Most other filters don't expand bounds
            _ => Rect::ZERO,
        }
    }
}

#[cfg(test)]
mod offset_expansion_tests {
    use super::FilterPrimitive;
    use crate::kurbo::Rect;

    #[test]
    fn offset_expands_in_direction_of_shift() {
        let p = FilterPrimitive::Offset { dx: 2.5, dy: -3.0 };
        assert_eq!(
            p.expansion_rect(),
            Rect::new(0.0, -3.0, 2.5, 0.0),
            "Offset expansion should be asymmetric and include the shift vector"
        );
    }
}

/// Unique identifier for a filter primitive in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FilterId(pub u16);

/// Input connections for a filter primitive.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterInputs {
    /// Primary input ("in" attribute in SVG).
    pub primary: FilterInput,
    /// Secondary input ("in2" attribute in SVG, for composite/blend operations).
    pub secondary: Option<FilterInput>,
}

impl FilterInputs {
    /// Create filter inputs with a single input.
    ///
    /// Use this for primitives that operate on a single source (blur, color matrix, etc.).
    pub fn single(input: FilterInput) -> Self {
        Self {
            primary: input,
            secondary: None,
        }
    }

    /// Create filter inputs with two inputs (for composite, blend, etc.).
    ///
    /// Use this for primitives that combine two sources (composite, blend, displacement map, etc.).
    pub fn dual(input1: FilterInput, input2: FilterInput) -> Self {
        Self {
            primary: input1,
            secondary: Some(input2),
        }
    }
}

/// A single filter input.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterInput {
    /// Input from a source (`SourceGraphic`, `SourceAlpha`, etc.).
    Source(FilterSource),
    /// Input from another filter's result.
    Result(FilterId),
}

/// Filter input sources.
///
/// Defines the various built-in sources that can be used as filter inputs,
/// matching the SVG filter primitive input types. These represent implicit
/// inputs available to any filter primitive without requiring previous operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterSource {
    /// The original graphic content being filtered.
    ///
    /// This is the default input - the rendered result of the element
    /// the filter is applied to, including all its fill, stroke, and content.
    SourceGraphic,
    /// Alpha channel only of the original graphic.
    ///
    /// Useful for creating effects based on shape/transparency, such as
    /// shadows that follow the element's outline.
    SourceAlpha,
    /// Background image content behind the filtered element.
    ///
    /// Allows filters to incorporate or blend with content behind the element.
    /// Not always available depending on the rendering context.
    BackgroundImage,
    /// Alpha channel only of the background image.
    ///
    /// The transparency mask of the background content.
    BackgroundAlpha,
    /// The fill paint of the element as an image input.
    ///
    /// For elements with gradient or pattern fills, this provides access
    /// to the fill as a filter input.
    FillPaint,
    /// The stroke paint of the element as an image input.
    ///
    /// For elements with gradient or pattern strokes, this provides access
    /// to the stroke as a filter input.
    StrokePaint,
}

/// Pre-built compound effects for common use cases.
///
/// These effects combine multiple filter primitives into commonly-used visual effects.
/// They provide a convenient high-level API for complex multi-step filter operations.
///
/// **Note:** These are planned but not yet implemented. Use `FilterGraph` to manually
/// construct these effects from primitives.
#[derive(Debug, Clone)]
pub enum CompoundFilter {
    /// Inner shadow effect (shadow inside the shape).
    ///
    /// Creates a shadow that appears inside the boundaries of the shape,
    /// giving a recessed or inset appearance. This is the opposite of a drop shadow.
    InnerShadow {
        /// Horizontal offset of the shadow in pixels. Positive values shift right.
        dx: f32,
        /// Vertical offset of the shadow in pixels. Positive values shift down.
        dy: f32,
        /// Blur radius for the shadow in pixels. Larger values create softer shadows.
        blur: f32,
        /// Shadow color with alpha channel.
        color: AlphaColor<Srgb>,
    },
    /// Glow effect around the shape.
    ///
    /// Creates a soft glowing halo around the shape by blurring and
    /// compositing a colored version with the original.
    Glow {
        /// Blur radius for the glow in pixels. Larger values create softer glows.
        blur: f32,
        /// Glow color with alpha channel.
        color: AlphaColor<Srgb>,
    },
    /// Bevel effect (3D raised/recessed appearance).
    ///
    /// Creates a 3D beveled edge effect using lighting simulation,
    /// making the shape appear raised or recessed from the surface.
    Bevel {
        /// Light source angle in degrees (0° = right, 90° = up).
        angle: f32,
        /// Width of the bevel edge in pixels.
        distance: f32,
        /// Color for the highlight (lit) side of the bevel.
        highlight: AlphaColor<Srgb>,
        /// Color for the shadow (dark) side of the bevel.
        shadow: AlphaColor<Srgb>,
    },
    /// Emboss effect for a raised relief appearance.
    ///
    /// Creates an embossed/stamped appearance by simulating lighting
    /// on a raised surface based on the shape's alpha channel.
    Emboss {
        /// Light angle in degrees determining emboss direction.
        angle: f32,
        /// Depth of the emboss effect.
        depth: f32,
        /// Overall strength/intensity of the effect (0.0 = none, 1.0 = full).
        amount: f32,
    },
}

/// Composite operators for combining filter inputs.
///
/// These are the Porter-Duff compositing operators used to combine two images.
/// Each operator defines how the source (input 1) and destination (input 2)
/// are combined based on their color and alpha values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompositeOperator {
    /// Source over destination (standard alpha blending).
    ///
    /// The source is composited over the destination. This is the most common
    /// blending mode where source alpha determines visibility.
    Over,
    /// Source in destination (intersection).
    ///
    /// The source is only visible where the destination is opaque.
    /// Result alpha = `source_alpha` × `dest_alpha`.
    In,
    /// Source out destination (subtract).
    ///
    /// The source is only visible where the destination is transparent.
    /// Useful for masking/cutting out regions.
    Out,
    /// Source atop destination.
    ///
    /// Source is composited over destination, but only where destination is opaque.
    Atop,
    /// Source XOR destination (exclusive or).
    ///
    /// Shows source where destination is transparent and vice versa,
    /// but not where both are opaque.
    Xor,
    /// Arithmetic combination with custom coefficients.
    ///
    /// Custom linear combination: result = k1*src*dst + k2*src + k3*dst + k4.
    /// Allows creating custom compositing operations beyond the standard Porter-Duff set.
    Arithmetic {
        /// Coefficient k1 for the (source * destination) term.
        k1: f32,
        /// Coefficient k2 for the source term.
        k2: f32,
        /// Coefficient k3 for the destination term.
        k3: f32,
        /// Constant offset k4 added to the result.
        k4: f32,
    },
}

/// Blend modes for combining colors.
///
/// These are blend modes that define how to combine the colors
/// of two layers. Unlike compositing operators which deal with alpha, blend modes
/// focus on color mixing while preserving the compositing behavior.
///
/// See: <https://drafts.fxtf.org/compositing/#blending>
pub type BlendMode = peniko::Mix;

/// Morphological operators for dilate/erode operations.
///
/// These operators modify the shape of objects by expanding or contracting them.
/// They work by examining neighborhoods of pixels and applying min/max operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphologyOperator {
    /// Erode operation (shrink/thin shapes).
    ///
    /// Makes objects smaller by removing pixels at the edges. Takes the minimum
    /// value in the neighborhood. Useful for removing noise or separating touching objects.
    Erode,
    /// Dilate operation (expand/thicken shapes).
    ///
    /// Makes objects larger by adding pixels at the edges. Takes the maximum
    /// value in the neighborhood. Useful for filling holes or connecting nearby objects.
    Dilate,
}

/// Convolution kernel for custom filtering operations.
///
/// Defines a square matrix of weights used for convolution-based image processing.
/// The kernel is applied to each pixel by multiplying surrounding pixels by the weights,
/// summing the results, dividing by the divisor, and adding the bias.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvolutionKernel {
    /// Kernel size (e.g., 3 for a 3×3 kernel, 5 for 5×5).
    /// The kernel must be square, so this defines both width and height.
    pub size: u32,
    /// Kernel weight values in row-major order.
    /// Length must equal size × size. Center of kernel is typically at (size/2, size/2).
    pub values: Vec<f32>,
    /// Normalization divisor applied to the convolution result.
    /// Common practice is to use the sum of all weights for averaging, or 1.0 otherwise.
    pub divisor: f32,
    /// Bias value added to the result after normalization.
    /// Useful for edge detection or emboss effects to shift the result range.
    pub bias: f32,
    /// Whether to preserve the alpha channel unchanged.
    /// If true, convolution only applies to RGB; if false, it applies to RGBA.
    pub preserve_alpha: bool,
}

/// Types of turbulence noise generation.
///
/// Determines the algorithm used for generating procedural noise patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurbulenceType {
    /// Fractal noise (smooth, natural-looking Perlin noise).
    ///
    /// Creates smooth, continuous patterns suitable for natural textures
    /// like clouds, marble, wood grain, or terrain.
    FractalNoise,
    /// Turbulence noise (more chaotic and energetic).
    ///
    /// Creates more chaotic patterns with sharper transitions,
    /// suitable for fire, smoke, or turbulent effects.
    Turbulence,
}

/// Color channels for displacement mapping and channel selection.
///
/// Specifies which color channel to use for operations that need to
/// extract or reference individual channels from an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorChannel {
    /// Red color channel (R component).
    Red,
    /// Green color channel (G component).
    Green,
    /// Blue color channel (B component).
    Blue,
    /// Alpha channel (transparency/opacity).
    Alpha,
}

/// Transfer functions for component transfer operations.
///
/// These functions map input color channel values to output values,
/// enabling gamma correction, color grading, and custom color curves.
/// Input and output values are typically in the range [0, 1].
#[derive(Debug, Clone, PartialEq)]
pub enum TransferFunction {
    /// Identity function (output = input, no change).
    Identity,
    /// Table lookup with linear interpolation.
    ///
    /// Maps input values using a lookup table with linear interpolation between entries.
    /// Input 0.0 maps to values\[0\], 1.0 maps to values\[n-1\], intermediate values interpolate.
    Table {
        /// Lookup table values defining the transfer curve.
        /// More values provide smoother curves. Minimum 2 values required.
        values: Vec<f32>,
    },
    /// Discrete step function (posterization).
    ///
    /// Maps input to discrete output values without interpolation, creating step/banding effects.
    /// Each segment gets a constant output value from the table.
    Discrete {
        /// Step values for each discrete output level.
        /// Input range is divided into len(values) segments, each mapping to one value.
        values: Vec<f32>,
    },
    /// Linear function: output = slope × input + intercept.
    ///
    /// Simple linear transformation of the input value.
    Linear {
        /// Slope coefficient (rate of change).
        slope: f32,
        /// Intercept offset (constant added to result).
        intercept: f32,
    },
    /// Gamma correction: output = amplitude × input^exponent + offset.
    ///
    /// Applies power-law transformation, commonly used for gamma correction and
    /// adjusting midtone brightness without affecting blacks or whites.
    Gamma {
        /// Amplitude multiplier applied to the result.
        amplitude: f32,
        /// Gamma exponent (< 1 brightens, > 1 darkens midtones).
        exponent: f32,
        /// Offset added to the final result.
        offset: f32,
    },
}

/// Light source configurations for lighting effects.
///
/// Defines different types of light sources used in diffuse and specular lighting
/// filter primitives. Each type has different characteristics and use cases.
#[derive(Debug, Clone, PartialEq)]
pub enum LightSource {
    /// Distant light source (infinitely far away, like the sun).
    ///
    /// All rays are parallel, creating uniform lighting across the surface.
    /// Direction is specified using spherical coordinates (azimuth and elevation).
    Distant {
        /// Azimuth angle in degrees (0° = pointing right, 90° = pointing up).
        /// Defines the horizontal direction of the light.
        azimuth: f32,
        /// Elevation angle in degrees (0° = horizon, 90° = directly overhead).
        /// Defines the vertical angle of the light source.
        elevation: f32,
    },
    /// Point light source at a specific 3D position.
    ///
    /// Light radiates uniformly in all directions from a single point.
    /// Intensity decreases with distance. Like a light bulb.
    Point {
        /// Light source X coordinate in user space.
        x: f32,
        /// Light source Y coordinate in user space.
        y: f32,
        /// Light source Z coordinate (height above the surface).
        /// Larger values create softer lighting across larger areas.
        z: f32,
    },
    /// Spot light with position, direction, and cone angle.
    ///
    /// Light emanates from a point in a specific direction with limited spread.
    /// Like a flashlight or stage spotlight with adjustable focus.
    Spot {
        /// Light source X coordinate in user space.
        x: f32,
        /// Light source Y coordinate in user space.
        y: f32,
        /// Light source Z coordinate (height above the surface).
        z: f32,
        /// X coordinate the spotlight is aimed at.
        points_at_x: f32,
        /// Y coordinate the spotlight is aimed at.
        points_at_y: f32,
        /// Z coordinate the spotlight is aimed at.
        points_at_z: f32,
        /// Specular exponent controlling the focus/sharpness of the spotlight beam.
        /// Higher values create tighter, more focused beams.
        specular_exponent: f32,
        /// Optional cone angle in degrees limiting the spotlight spread.
        /// If None, the light spreads based only on the specular exponent.
        limiting_cone_angle: Option<f32>,
    },
}

/// Common color transformation matrices.
///
/// These 4x5 matrices are used with the `ColorMatrix` filter primitive.
/// Each row transforms a color channel: [R, G, B, A, offset].
pub mod matrices {
    /// Identity matrix (no change).
    pub const IDENTITY: [f32; 20] = [
        1.0, 0.0, 0.0, 0.0, 0.0, // Red
        0.0, 1.0, 0.0, 0.0, 0.0, // Green
        0.0, 0.0, 1.0, 0.0, 0.0, // Blue
        0.0, 0.0, 0.0, 1.0, 0.0, // Alpha
    ];

    /// Extract alpha channel to RGB (for shadow effects).
    pub const ALPHA_TO_BLACK: [f32; 20] = [
        0.0, 0.0, 0.0, 1.0, 0.0, // Red = Alpha
        0.0, 0.0, 0.0, 1.0, 0.0, // Green = Alpha
        0.0, 0.0, 0.0, 1.0, 0.0, // Blue = Alpha
        0.0, 0.0, 0.0, 1.0, 0.0, // Alpha = Alpha
    ];

    /// Grayscale conversion matrix using luminosity weights.
    pub const GRAYSCALE: [f32; 20] = [
        0.2126, 0.7152, 0.0722, 0.0, 0.0, // Red
        0.2126, 0.7152, 0.0722, 0.0, 0.0, // Green
        0.2126, 0.7152, 0.0722, 0.0, 0.0, // Blue
        0.0, 0.0, 0.0, 1.0, 0.0, // Alpha
    ];

    /// Sepia tone matrix for vintage photo effect.
    pub const SEPIA: [f32; 20] = [
        0.393, 0.769, 0.189, 0.0, 0.0, // Red
        0.349, 0.686, 0.168, 0.0, 0.0, // Green
        0.272, 0.534, 0.131, 0.0, 0.0, // Blue
        0.0, 0.0, 0.0, 1.0, 0.0, // Alpha
    ];
}

/// Common convolution kernels.
///
/// These kernels are used with the `ConvolveMatrix` filter primitive
/// for various image processing effects. All provided kernels are 3x3.
pub mod kernels {
    use super::ConvolutionKernel;
    use alloc::vec;

    /// 3x3 Gaussian blur kernel for basic smoothing.
    pub fn gaussian_3x3() -> ConvolutionKernel {
        ConvolutionKernel {
            size: 3,
            values: vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0],
            divisor: 16.0,
            bias: 0.0,
            preserve_alpha: false,
        }
    }

    /// 3x3 Sharpen kernel to enhance edges and details.
    pub fn sharpen_3x3() -> ConvolutionKernel {
        ConvolutionKernel {
            size: 3,
            values: vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0],
            divisor: 1.0,
            bias: 0.0,
            preserve_alpha: true,
        }
    }

    /// 3x3 Edge detection kernel (Laplacian operator).
    pub fn edge_detect_3x3() -> ConvolutionKernel {
        ConvolutionKernel {
            size: 3,
            values: vec![-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0],
            divisor: 1.0,
            bias: 0.0,
            preserve_alpha: true,
        }
    }

    /// 3x3 Emboss kernel for creating a raised/beveled appearance.
    pub fn emboss_3x3() -> ConvolutionKernel {
        ConvolutionKernel {
            size: 3,
            values: vec![-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0],
            divisor: 1.0,
            bias: 0.5,
            preserve_alpha: true,
        }
    }
}
