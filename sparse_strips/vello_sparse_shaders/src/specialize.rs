// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shader specialization via string substitution.
//!
//! The WGSL shader source declares shader build time constants with placeholder
//! values. At shader build time (during runtime on the app), [`ShaderConstants::specialize_wgsl`]
//! (and [`ShaderConstants::specialize_glsl`] for the WebGL backend) replace these
//! placeholders with device-specific literals, turning them into compile-time constants.
//!
//! Shader constants are supported by Naga, but would require to be shipped in the binary
//! at runtime for Naga to perform the substitution. We intentionally created `vello_sparse_shaders`
//! to avoid this binary bloat, so instead we perform this substitution ourselves.

use alloc::format;
use alloc::string::String;

/// Values that are constant for the lifetime of a renderer. At shader build time their
/// placeholder `const` declarations in the shader source are replaced with literals
/// via string substitution.
///
/// For example, for `ShaderConstants` with `strip_height` set to `16`,
/// ```wgsl
/// const STRIP_HEIGHT: u32 = 0u;
/// ```
/// Is string replaced with:
/// ```wgsl
/// const STRIP_HEIGHT: u32 = 16u;
/// ```
#[derive(Debug, Clone)]
pub struct ShaderConstants {
    /// Height of a strip.
    pub strip_height: u32,
    /// log2 of the alphas texture width (for bit-shift addressing).
    pub alphas_tex_width_bits: u32,
    /// log2 of the encoded-paints texture width (for bit-shift addressing).
    pub encoded_paints_tex_width_bits: u32,
    /// log2 of the gradient texture width (for bit-shift addressing).
    pub gradient_tex_width_bits: u32,
}

/// Assumes that each replacement is expected to appear at most once and to be fully contained within a single line.
fn replace_all_once(source: &str, replacements: &[(&str, &str)]) -> String {
    let mut result = String::with_capacity(source.len() + 128);
    for (i, line) in source.split('\n').enumerate() {
        if i > 0 {
            result.push('\n');
        }
        if let Some((pat, rep)) = replacements.iter().find(|(pat, _)| line.contains(pat)) {
            result.push_str(&line.replacen(pat, rep, 1));
        } else {
            result.push_str(line);
        }
    }
    result
}

impl ShaderConstants {
    /// Specialize a WGSL shader source by replacing build time `const` declarations.
    pub(crate) fn specialize_wgsl(&self, source: &str) -> String {
        let strip_height = format!("const STRIP_HEIGHT: u32 = {}u;", self.strip_height);
        let alphas = format!(
            "const ALPHAS_TEX_WIDTH_BITS: u32 = {}u;",
            self.alphas_tex_width_bits
        );
        let paints = format!(
            "const ENCODED_PAINTS_TEX_WIDTH_BITS: u32 = {}u;",
            self.encoded_paints_tex_width_bits
        );
        let gradient = format!(
            "const GRADIENT_TEX_WIDTH_BITS: u32 = {}u;",
            self.gradient_tex_width_bits
        );

        replace_all_once(
            source,
            &[
                ("const STRIP_HEIGHT: u32 = 0u;", &strip_height),
                ("const ALPHAS_TEX_WIDTH_BITS: u32 = 0u;", &alphas),
                ("const ENCODED_PAINTS_TEX_WIDTH_BITS: u32 = 0u;", &paints),
                ("const GRADIENT_TEX_WIDTH_BITS: u32 = 0u;", &gradient),
            ],
        )
    }

    /// Specialize a GLSL shader source by replacing build time `const` declarations.
    #[cfg(feature = "gles")]
    pub(crate) fn specialize_glsl(&self, source: &str) -> String {
        let strip_height = format!("const uint STRIP_HEIGHT = {}u;", self.strip_height);
        let alphas = format!(
            "const uint ALPHAS_TEX_WIDTH_BITS = {}u;",
            self.alphas_tex_width_bits
        );
        let paints = format!(
            "const uint ENCODED_PAINTS_TEX_WIDTH_BITS = {}u;",
            self.encoded_paints_tex_width_bits
        );
        let gradient = format!(
            "const uint GRADIENT_TEX_WIDTH_BITS = {}u;",
            self.gradient_tex_width_bits
        );

        replace_all_once(
            source,
            &[
                ("const uint STRIP_HEIGHT = 0u;", &strip_height),
                ("const uint ALPHAS_TEX_WIDTH_BITS = 0u;", &alphas),
                ("const uint ENCODED_PAINTS_TEX_WIDTH_BITS = 0u;", &paints),
                ("const uint GRADIENT_TEX_WIDTH_BITS = 0u;", &gradient),
            ],
        )
    }
}
