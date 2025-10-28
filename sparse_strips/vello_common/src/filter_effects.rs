// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter effects API based on the W3C Filter Effects specification.
//!
//! This module provides a comprehensive filter system supporting both high-level
//! CSS filter functions and low-level SVG filter primitives. The API is designed
//! to follow the W3C Filter Effects Module Level 1 specification.
//!
//! See: <https://drafts.fxtf.org/filter-effects/>

use crate::color::{AlphaColor, Srgb};
use crate::kurbo::Rect;
use alloc::vec::Vec;
use std::collections::BTreeMap;

/// The main filter system.
#[derive(Debug, Clone, PartialEq)]
pub struct Filter {
    /// Filter graph defining the effect pipeline.
    pub graph: FilterGraph,
    /// Bounds where the filter should be applied.
    pub bounds: Option<Rect>,
}

impl Filter {
    /// Create a simple filter system from a filter function.
    ///
    /// Converts a high-level CSS-style filter function into a filter graph.
    /// Use this for simple effects like blur, brightness, etc.
    pub fn from_function(function: FilterFunction) -> Self {
        let mut graph = FilterGraph::new();

        // Convert function to primitive
        let primitive = match function {
            FilterFunction::Blur { radius } => FilterPrimitive::GaussianBlur {
                std_deviation: radius,
            },
            _ => unimplemented!("Filter function {:?} not supported", function),
        };

        let filter_id = graph.add(primitive, None);
        graph.set_output(filter_id);

        Self {
            graph,
            bounds: None,
        }
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
            graph,
            bounds: None,
        }
    }

    /// Calculate the total bounds expansion for this filter in pixels.
    /// This is cached and computed incrementally as primitives are added.
    pub fn bounds_expansion(&self) -> BoundsExpansion {
        self.graph.bounds_expansion()
    }
}

/// Describes how many pixels a filter expands the required processing region
/// in each direction. All values are non-negative.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundsExpansion {
    /// Extra pixels needed to the left
    pub left: f32,
    /// Extra pixels needed above
    pub top: f32,
    /// Extra pixels needed to the right
    pub right: f32,
    /// Extra pixels needed below
    pub bottom: f32,
}

impl BoundsExpansion {
    /// Create a zero expansion (no change).
    pub const fn zero() -> Self {
        Self {
            left: 0.0,
            top: 0.0,
            right: 0.0,
            bottom: 0.0,
        }
    }

    /// Create a uniform expansion in all directions.
    pub fn uniform(radius: f32) -> Self {
        Self {
            left: radius,
            top: radius,
            right: radius,
            bottom: radius,
        }
    }

    /// Combine two expansions by taking the maximum in each direction.
    /// Used when composing multiple filters.
    pub fn max(&self, other: &Self) -> Self {
        Self {
            left: self.left.max(other.left),
            top: self.top.max(other.top),
            right: self.right.max(other.right),
            bottom: self.bottom.max(other.bottom),
        }
    }
}

/// A directed acyclic graph (DAG) of filter operations.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterGraph {
    /// All filter primitives in the graph.
    pub primitives: Vec<FilterPrimitive>,
    /// Connections between filter primitives.
    pub connections: BTreeMap<FilterId, FilterInputs>,
    /// The final output filter.
    pub output: FilterId,
    /// Next available filter ID.
    next_id: u32,
    /// Accumulated bounds expansion from all primitives.
    /// Updated incrementally as primitives are added.
    expansion: BoundsExpansion,
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
            primitives: Vec::new(),
            connections: BTreeMap::new(),
            output: FilterId(0),
            next_id: 0,
            expansion: BoundsExpansion::zero(),
        }
    }

    /// Add a filter primitive with optional inputs.
    ///
    /// Returns a `FilterId` that can be referenced by other primitives.
    /// Automatically updates the accumulated bounds expansion based on the primitive's requirements.
    pub fn add(&mut self, primitive: FilterPrimitive, inputs: Option<FilterInputs>) -> FilterId {
        let id = FilterId(self.next_id);
        self.next_id += 1;

        // Update accumulated expansion
        let primitive_expansion = primitive.bounds_expansion();
        self.expansion = self.expansion.max(&primitive_expansion);

        self.primitives.push(primitive);

        if let Some(inputs) = inputs {
            self.connections.insert(id, inputs);
        }

        id
    }

    /// Set the output filter for the graph.
    pub fn set_output(&mut self, output: FilterId) {
        self.output = output;
    }

    /// Get the accumulated bounds expansion for all primitives in this graph.
    pub fn bounds_expansion(&self) -> BoundsExpansion {
        self.expansion
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
    // Custom shader effects (potential future feature)
    // Custom(CustomShaderFilter),
}

/// High-level filter functions for common effects (css filter functions).
/// @see <https://drafts.fxtf.org/filter-effects/#filter-functions>
#[derive(Debug, Clone)]
pub enum FilterFunction {
    /// Gaussian blur effect.
    Blur {
        /// Blur radius in pixels.
        radius: f32,
    },
    // ============================================================
    // TODO: The following filter functions are not yet implemented
    // ============================================================
    /// Brightness adjustment.
    Brightness {
        /// 0.0 = black, 1.0 = normal, >1.0 = brighter.
        amount: f32,
    },
    /// Contrast adjustment.
    Contrast {
        /// 0.0 = gray, 1.0 = normal, >1.0 = higher contrast.
        amount: f32,
    },
    /// Grayscale conversion.
    Grayscale {
        /// 0.0 = original, 1.0 = full grayscale.
        amount: f32,
    },
    /// Hue rotation.
    HueRotate {
        /// Angle in degrees.
        angle: f32,
    },
    /// Color inversion.
    Invert {
        /// 0.0 = original, 1.0 = fully inverted.
        amount: f32,
    },
    /// Opacity adjustment.
    Opacity {
        /// 0.0 = transparent, 1.0 = opaque.
        amount: f32,
    },
    /// Saturation adjustment.
    Saturate {
        /// 0.0 = grayscale, 1.0 = normal, >1.0 = oversaturated.
        amount: f32,
    },
    /// Sepia tone effect.
    Sepia {
        /// 0.0 = original, 1.0 = full sepia.
        amount: f32,
    },
}

/// Low-level filter primitives for granular control (svg filter primitives).
/// @see <https://drafts.fxtf.org/filter-effects/#FilterPrimitivesOverview>
#[derive(Debug, Clone, PartialEq)]
pub enum FilterPrimitive {
    /// Generate solid color fill.
    Flood {
        /// Fill color with alpha.
        color: AlphaColor<Srgb>,
    },
    /// Gaussian blur filter.
    GaussianBlur {
        /// Standard deviation for blur.
        std_deviation: f32,
    },
    /// Drop shadow effect (compound primitive).
    /// @see <https://drafts.fxtf.org/filter-effects-2/#feDropShadowElement>
    DropShadow {
        /// Horizontal offset in pixels.
        dx: f32,
        /// Vertical offset in pixels.
        dy: f32,
        /// Blur standard deviation.
        std_deviation: f32,
        /// Shadow color with alpha.
        color: AlphaColor<Srgb>,
    },
    // ============================================================
    // TODO: The following filter primitives are not yet implemented
    // ============================================================
    /// Matrix-based color transformation.
    ColorMatrix {
        /// 4x5 matrix: [r,g,b,a,offset] for each channel.
        matrix: [f32; 20],
    },
    /// Geometric offset/translation.
    Offset {
        /// Horizontal offset in pixels.
        dx: f32,
        /// Vertical offset in pixels.
        dy: f32,
    },

    /// Composite two inputs using Porter-Duff operations.
    Composite {
        /// Compositing operator to use.
        operator: CompositeOperator,
    },
    /// Blend two inputs using blend modes.
    Blend {
        /// Blend mode to apply.
        mode: BlendMode,
    },
    /// Morphological operations (dilate/erode).
    Morphology {
        /// Morphological operator (erode or dilate).
        operator: MorphologyOperator,
        /// Operation radius in pixels.
        radius: f32,
    },
    /// Custom convolution kernel.
    ConvolveMatrix {
        /// Convolution kernel definition.
        kernel: ConvolutionKernel,
    },
    /// Generate Perlin noise/turbulence.
    Turbulence {
        /// Base frequency for noise generation.
        base_frequency: f32,
        /// Number of octaves for fractal noise.
        num_octaves: u32,
        /// Random seed for noise generation.
        seed: u32,
        /// Type of turbulence (fractal or turbulence).
        turbulence_type: TurbulenceType,
    },
    /// Displace pixels using a displacement map.
    DisplacementMap {
        /// Displacement scale factor.
        scale: f32,
        /// Color channel to use for X displacement.
        x_channel: ColorChannel,
        /// Color channel to use for Y displacement.
        y_channel: ColorChannel,
    },
    /// Per-channel component transfer (lookup tables).
    ComponentTransfer {
        /// Transfer function for red channel.
        red_function: Option<TransferFunction>,
        /// Transfer function for green channel.
        green_function: Option<TransferFunction>,
        /// Transfer function for blue channel.
        blue_function: Option<TransferFunction>,
        /// Transfer function for alpha channel.
        alpha_function: Option<TransferFunction>,
    },
    /// Reference an external image.
    Image {
        /// Reference to image in atlas.
        image_id: u32,
        /// Optional 2D transform matrix.
        transform: Option<[f32; 6]>,
    },
    /// Tile input to fill region.
    Tile,
    /// Diffuse lighting simulation.
    DiffuseLighting {
        /// Surface scale factor for height calculation.
        surface_scale: f32,
        /// Diffuse reflection constant.
        diffuse_constant: f32,
        /// Kernel unit length for calculations.
        kernel_unit_length: f32,
        /// Light source configuration.
        light_source: LightSource,
    },
    /// Specular lighting simulation.
    SpecularLighting {
        /// Surface scale factor for height calculation.
        surface_scale: f32,
        /// Specular reflection constant.
        specular_constant: f32,
        /// Specular reflection exponent.
        specular_exponent: f32,
        /// Kernel unit length for calculations.
        kernel_unit_length: f32,
        /// Light source configuration.
        light_source: LightSource,
    },
}

impl FilterPrimitive {
    /// Calculate how many extra pixels are needed in each direction to correctly apply this filter.
    ///
    /// Returns the expansion required in the processing region to avoid edge artifacts.
    /// For example, a blur filter needs additional pixels around the edges (3*sigma).
    /// Most filters that don't sample neighboring pixels return zero expansion.
    pub fn bounds_expansion(&self) -> BoundsExpansion {
        match self {
            Self::GaussianBlur { std_deviation } => {
                // Gaussian blur expands uniformly by 3*sigma (covers 99.7% of distribution)
                let radius = std_deviation * 3.0;
                BoundsExpansion::uniform(radius)
            }
            Self::DropShadow {
                std_deviation,
                dx,
                dy,
                ..
            } => {
                // Drop shadow = blur + offset + composite with original
                // Need space for blur AND the offset direction
                let blur_radius = std_deviation * 3.0;
                BoundsExpansion {
                    left: blur_radius + (-dx).max(0.0),
                    top: blur_radius + (-dy).max(0.0),
                    right: blur_radius + (*dx).max(0.0),
                    bottom: blur_radius + (*dy).max(0.0),
                }
            }
            // Most other filters don't expand bounds
            _ => BoundsExpansion::zero(),
        }
    }
}

/// Unique identifier for a filter primitive in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FilterId(pub u32);

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
/// matching the SVG filter primitive input types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterSource {
    /// The original graphic content.
    SourceGraphic,
    /// Alpha channel of the original graphic.
    SourceAlpha,
    /// Background image content.
    BackgroundImage,
    /// Alpha channel of background image.
    BackgroundAlpha,
    /// Fill paint as input.
    FillPaint,
    /// Stroke paint as input.
    StrokePaint,
}

/// Pre-built compound effects for common use cases.
///
/// These effects combine multiple filter primitives into commonly-used visual effects.
/// They provide a convenient high-level API for complex multi-step filter operations.
#[derive(Debug, Clone)]
pub enum CompoundFilter {
    /// Inner shadow effect.
    InnerShadow {
        /// Horizontal offset in pixels.
        dx: f32,
        /// Vertical offset in pixels.
        dy: f32,
        /// Blur radius in pixels.
        blur: f32,
        /// Shadow color with alpha.
        color: AlphaColor<Srgb>,
    },
    /// Glow effect (blur + composite).
    Glow {
        /// Blur radius in pixels.
        blur: f32,
        /// Glow color with alpha.
        color: AlphaColor<Srgb>,
    },
    /// Bevel effect (lighting simulation).
    Bevel {
        /// Light angle in degrees.
        angle: f32,
        /// Bevel distance in pixels.
        distance: f32,
        /// Highlight color.
        highlight: AlphaColor<Srgb>,
        /// Shadow color.
        shadow: AlphaColor<Srgb>,
    },
    /// Emboss effect.
    Emboss {
        /// Emboss angle in degrees.
        angle: f32,
        /// Emboss depth.
        depth: f32,
        /// Effect amount/strength.
        amount: f32,
    },
}

/// Composite operators for combining filter inputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompositeOperator {
    /// Source over destination.
    Over,
    /// Source in destination.
    In,
    /// Source out destination.
    Out,
    /// Source atop destination.
    Atop,
    /// Source xor destination.
    Xor,
    /// Arithmetic combination with custom coefficients.
    Arithmetic {
        /// Coefficient for source * destination.
        k1: f32,
        /// Coefficient for source.
        k2: f32,
        /// Coefficient for destination.
        k3: f32,
        /// Constant offset.
        k4: f32,
    },
}

/// Blend modes for combining colors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Normal blending (no special effect).
    Normal,
    /// Multiply colors (darkening effect).
    Multiply,
    /// Screen colors (lightening effect).
    Screen,
    /// Overlay blending.
    Overlay,
    /// Darken colors.
    Darken,
    /// Lighten colors.
    Lighten,
    /// Color dodge effect.
    ColorDodge,
    /// Color burn effect.
    ColorBurn,
    /// Hard light blending.
    HardLight,
    /// Soft light blending.
    SoftLight,
    /// Difference blending.
    Difference,
    /// Exclusion blending.
    Exclusion,
    /// Hue blending.
    Hue,
    /// Saturation blending.
    Saturation,
    /// Color blending.
    Color,
    /// Luminosity blending.
    Luminosity,
}

/// Morphological operators for dilate/erode operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphologyOperator {
    /// Erode operation (shrink shapes).
    Erode,
    /// Dilate operation (expand shapes).
    Dilate,
}

/// Convolution kernel for custom filtering operations.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvolutionKernel {
    /// Kernel size (e.g., 3 for 3x3).
    pub size: u32,
    /// Kernel values.
    pub values: Vec<f32>,
    /// Normalization divisor.
    pub divisor: f32,
    /// Bias to add to result.
    pub bias: f32,
    /// Whether to preserve alpha channel.
    pub preserve_alpha: bool,
}

/// Types of turbulence noise generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurbulenceType {
    /// Fractal noise (smooth).
    FractalNoise,
    /// Turbulence noise (more chaotic).
    Turbulence,
}

/// Color channels for displacement mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorChannel {
    /// Red color channel.
    Red,
    /// Green color channel.
    Green,
    /// Blue color channel.
    Blue,
    /// Alpha channel.
    Alpha,
}

/// Transfer functions for component transfer operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TransferFunction {
    /// Identity function (no change).
    Identity,
    /// Table lookup function.
    Table {
        /// Lookup table values.
        values: Vec<f32>,
    },
    /// Discrete step function.
    Discrete {
        /// Step values.
        values: Vec<f32>,
    },
    /// Linear function: slope * input + intercept.
    Linear {
        /// Linear slope coefficient.
        slope: f32,
        /// Linear intercept offset.
        intercept: f32,
    },
    /// Gamma correction function.
    Gamma {
        /// Gamma amplitude.
        amplitude: f32,
        /// Gamma exponent.
        exponent: f32,
        /// Gamma offset.
        offset: f32,
    },
}

/// Light source configurations for lighting effects.
#[derive(Debug, Clone, PartialEq)]
pub enum LightSource {
    /// Distant light source (like the sun).
    Distant {
        /// Light azimuth angle in degrees.
        azimuth: f32,
        /// Light elevation angle in degrees.
        elevation: f32,
    },
    /// Point light source at specific coordinates.
    Point {
        /// Light X position.
        x: f32,
        /// Light Y position.
        y: f32,
        /// Light Z position.
        z: f32,
    },
    /// Spot light with direction and cone.
    Spot {
        /// Light X position.
        x: f32,
        /// Light Y position.
        y: f32,
        /// Light Z position.
        z: f32,
        /// X coordinate the light points at.
        points_at_x: f32,
        /// Y coordinate the light points at.
        points_at_y: f32,
        /// Z coordinate the light points at.
        points_at_z: f32,
        /// Specular reflection exponent.
        specular_exponent: f32,
        /// Optional cone angle limit in degrees.
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
