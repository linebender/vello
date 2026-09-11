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
    /// The WebGL context is incompatible with the renderer.
    #[error("incompatible WebGL context: {0:?}")]
    IncompatibleContext(IncompatibleContextReason),
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
    /// A buffer.
    Buffer,
    /// A framebuffer.
    Framebuffer,
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
    /// Creating the probe's pixel pack buffer.
    BufferCreation,
    /// Uploading the probe image.
    ImageUpload,
    /// Reading the probe result back.
    Readback,
    /// Creating synchronization state for the probe.
    Synchronization,
}

/// Reasons a WebGL context can be incompatible with the renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum IncompatibleContextReason {
    /// Antialiasing is enabled on the WebGL context.
    AntialiasingEnabled,
    /// The WebGL context's depth buffer does not have enough bits.
    InsufficientDepthBuffer,
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

struct MessageSuffix<'a>(Option<&'a str>);

impl fmt::Display for MessageSuffix<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(message) = self.0 {
            write!(f, ": {message}")?;
        }

        Ok(())
    }
}
