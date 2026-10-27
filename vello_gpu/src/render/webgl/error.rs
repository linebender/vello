// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::{format, string::String};
use core::fmt;
use thiserror::Error;
use web_sys::wasm_bindgen::JsValue;

use crate::RenderError;

/// Errors produced by the WebGL backend.
#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum WebGlError {
    /// Rendering failed for a certain reason.
    #[error(transparent)]
    Render(#[from] RenderError),
    /// An image upload was attempted without the `full_image_support` feature.
    #[error("uploaded images require the `full_image_support` feature")]
    FullImageSupportDisabled,
    /// A WebGL2 context could not be created for the canvas.
    #[error("WebGL2 context unavailable")]
    ContextUnavailable,
    /// A WebGL operation failed.
    #[error(
        "WebGL operation failed while {operation:?}{suffix}",
        suffix = MessageSuffix(.message.as_deref())
    )]
    OperationFailed {
        /// The failed operation.
        operation: WebGlOperation,
        /// Additional diagnostic information, when available.
        message: Option<String>,
    },
    /// A render pass contains too many instances for WebGL.
    #[error("too many instances")]
    TooManyInstances,
    /// A resource texture allocation exceeds the supported texture dimensions.
    #[error(
        "resource texture allocation {width}x{height} exceeds maximum dimension {max_dimension}"
    )]
    ResourceTextureTooLarge {
        /// The requested texture width.
        width: u32,
        /// The requested texture height.
        height: u32,
        /// The maximum supported width or height.
        max_dimension: u32,
    },
    /// The WebGL context is incompatible with the renderer.
    #[error("incompatible WebGL context: {0}")]
    IncompatibleContext(IncompatibleContextReason),
    /// Shader programs compiled from JavaScript cannot be adopted by the renderer.
    #[error("incompatible precompiled shaders: {0}")]
    IncompatiblePrecompiledShaders(IncompatiblePrecompiledShadersReason),
    /// A texture target cannot be rendered into.
    #[error("invalid texture target: {0}")]
    InvalidTextureTarget(InvalidTextureTargetReason),
}

/// A fallible operation performed by the WebGL backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WebGlOperation {
    /// A context operation.
    Context(WebGlContextOperation),
    /// Creation of a WebGL resource.
    ResourceCreation(WebGlResourceKind),
    /// Shader compilation or program linking.
    ShaderProgram(WebGlShaderProgramOperation),
    /// Resolution of a shader interface binding.
    ShaderInterface(WebGlShaderInterfaceOperation),
    /// A data-transfer operation.
    DataTransfer(WebGlDataTransferOperation),
    /// A renderer probe operation.
    #[cfg(feature = "probe")]
    Probe(WebGlProbeOperation),
}

/// WebGL context operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WebGlContextOperation {
    /// Creating a WebGL context.
    Create,
    /// Configuring context creation or state.
    Configure,
    /// Querying context state or limits.
    Query,
    /// Querying or invoking an extension.
    Extension,
}

/// Kinds of WebGL resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WebGlResourceKind {
    /// A texture.
    Texture,
    /// A sampler.
    Sampler,
    /// A buffer.
    Buffer,
    /// A framebuffer.
    Framebuffer,
    /// A renderbuffer.
    Renderbuffer,
    /// A synchronization object.
    Sync,
    /// A shader program.
    Program,
    /// A shader.
    Shader(WebGlShaderStage),
    /// A vertex array.
    VertexArray,
}

/// WebGL shader stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WebGlShaderStage {
    /// A vertex shader.
    Vertex,
    /// A fragment shader.
    Fragment,
}

/// Shader compilation and program linking operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WebGlShaderProgramOperation {
    /// Compiling a shader.
    Compile(WebGlShaderStage),
    /// Linking a shader program.
    Link,
}

/// Shader interface operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WebGlShaderInterfaceOperation {
    /// Resolving a uniform.
    Uniform,
    /// Resolving a uniform block.
    UniformBlock,
}

/// WebGL data-transfer operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WebGlDataTransferOperation {
    /// Uploading texture data.
    TextureUpload,
    /// Invalidating framebuffer attachments.
    FramebufferInvalidation,
}

/// WebGL renderer probe operations.
#[cfg(feature = "probe")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WebGlProbeOperation {
    /// Uploading the probe image.
    ImageUpload,
    /// Reading the probe result back.
    Readback,
}

/// Reasons a WebGL context can be incompatible with the renderer.
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum IncompatibleContextReason {
    /// Antialiasing is enabled on the WebGL context.
    #[error("antialiasing is enabled")]
    AntialiasingEnabled,
    /// The WebGL context's depth buffer does not have enough bits.
    #[error("depth buffer has {actual_bits} bits, but {required_bits} are required")]
    InsufficientDepthBuffer {
        /// Number of depth bits provided by the WebGL context.
        actual_bits: u32,
        /// Minimum number of depth bits required by the renderer.
        required_bits: u32,
    },
}

/// Reasons a texture target can be invalid.
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidTextureTargetReason {
    /// WebGL rejects the attachments, e.g. because of an unsupported format or differing sizes.
    #[error("incomplete framebuffer: {}", FramebufferStatus(*.status))]
    IncompleteFramebuffer {
        /// The status returned by `checkFramebufferStatus`.
        status: u32,
    },
    /// The depth buffer does not have enough bits.
    #[error("depth buffer has {actual_bits} bits, but {required_bits} are required")]
    InsufficientDepthBuffer {
        /// Number of depth bits provided by the depth buffer.
        actual_bits: u32,
        /// Minimum number of depth bits required by the renderer.
        required_bits: u32,
    },
}

/// Reasons shader programs compiled from JavaScript can be incompatible with the renderer.
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum IncompatiblePrecompiledShadersReason {
    /// The value is not a `compileVelloShaders` result for an `HTMLCanvasElement`, or is missing
    /// some of its fields.
    ///
    /// This includes `null`, which `compileVelloShaders` returns when WebGL2 is unavailable.
    #[error("the value is not a `compileVelloShaders` result")]
    NotCompileVelloShadersResult,
    /// The JavaScript module was generated by a different build of `vello_gpu_shaders` than the
    /// renderer's, for example with different shader features or from an older version.
    #[error("they were compiled by a different build of the `compileVelloShaders` module")]
    DifferentModule,
}

/// Converts JavaScript exceptions into WebGL operation errors.
pub(crate) trait WebGlResultExt<T> {
    /// Attach the WebGL operation that produced this JavaScript result.
    fn map_js_error(self, operation: WebGlOperation) -> Result<T, WebGlError>;
}

impl<T> WebGlResultExt<T> for Result<T, JsValue> {
    fn map_js_error(self, operation: WebGlOperation) -> Result<T, WebGlError> {
        self.map_err(|value| WebGlError::OperationFailed {
            operation,
            message: Some(value.as_string().unwrap_or_else(|| format!("{value:?}"))),
        })
    }
}

struct FramebufferStatus(u32);

impl fmt::Display for FramebufferStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use web_sys::WebGl2RenderingContext as Gl;

        match self.0 {
            Gl::FRAMEBUFFER_INCOMPLETE_ATTACHMENT => {
                f.write_str("FRAMEBUFFER_INCOMPLETE_ATTACHMENT")
            }
            Gl::FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT => {
                f.write_str("FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT")
            }
            Gl::FRAMEBUFFER_INCOMPLETE_DIMENSIONS => {
                f.write_str("FRAMEBUFFER_INCOMPLETE_DIMENSIONS")
            }
            Gl::FRAMEBUFFER_UNSUPPORTED => f.write_str("FRAMEBUFFER_UNSUPPORTED"),
            Gl::FRAMEBUFFER_INCOMPLETE_MULTISAMPLE => {
                f.write_str("FRAMEBUFFER_INCOMPLETE_MULTISAMPLE")
            }
            status => write!(f, "{status:#x}"),
        }
    }
}

struct MessageSuffix<'a>(Option<&'a str>);

impl fmt::Display for MessageSuffix<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(message) = self.0 {
            write!(f, ": {message}")?;
        }

        Ok(())
    }
}
