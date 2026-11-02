// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// The build script generates `webgl_bootstrap::DTS_MODULE` from this file. It declares the API of
// `webgl_bootstrap.js`, so it must change with that file's exports.

declare const precompiledVelloShadersBrand: unique symbol;

/**
 * Shader programs that started compiling before the WebAssembly module loaded.
 *
 * Pass it to `WebGlRenderer::begin_with_precompiled` with the canvas it was compiled on, at most
 * once: the renderer takes ownership of its WebGL objects.
 */
export type PrecompiledVelloShaders = { readonly [precompiledVelloShadersBrand]: never };

/**
 * Creates the canvas's WebGL2 context and starts compiling Vello's shader programs without
 * waiting for them to finish.
 *
 * If the canvas already has a WebGL2 context, it's used as is, and `contextAttributes` are
 * ignored. Returns `null` if WebGL2 is unavailable.
 */
export declare function compileVelloShaders(
  canvas: HTMLCanvasElement,
  contextAttributes?: WebGLContextAttributes,
): PrecompiledVelloShaders | null;
