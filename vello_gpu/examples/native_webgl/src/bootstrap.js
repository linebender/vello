// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// The build script appends this to the module defining `compileVelloShaders`.
//
// The WebAssembly glue imports this module before it fetches the WebAssembly binary, so the
// shaders compile while the binary downloads and instantiates.

const earlyCanvas = typeof document === "undefined" ? null : document.createElement("canvas");
const precompiledShaders = earlyCanvas && compileVelloShaders(earlyCanvas, { depth: true });

/** Returns the canvas whose shaders started compiling before the WebAssembly module loaded. */
export function earlyCanvasElement() {
  return earlyCanvas;
}

/** Returns the `compileVelloShaders` result for `earlyCanvasElement()`. */
export function earlyPrecompiledShaders() {
  return precompiledShaders;
}
