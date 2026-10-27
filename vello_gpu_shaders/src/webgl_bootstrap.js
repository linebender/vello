// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// The build script prepends `MODULE_HASH` and `PROGRAMS` to this file to generate
// `webgl_bootstrap::JS_MODULE`. The context attributes Vello enforces and the program creation
// must match `vello_gpu`'s WebGL backend; `webgl_contexts_created_by_javascript_and_rust_match` in
// `vello_tests` checks the attributes.

/**
 * Creates the canvas's WebGL2 context and starts compiling Vello's shader programs without
 * waiting for them to finish.
 *
 * The context keeps the attributes it is created with, so pass any attributes the application
 * needs in `contextAttributes`. `depth` defaults to `true`; with `depth: false` the renderer
 * doesn't use a depth buffer, which reduces GPU memory usage but can increase overdraw. Vello
 * always sets `antialias: false`, which it requires, and `failIfMajorPerformanceCaveat: true`,
 * which keeps it off slow software renderers.
 *
 * Pass the result to `WebGlRenderer::begin_with_precompiled` once; the renderer then owns its
 * shaders and programs and deletes them when it's dropped. Returns `null` if WebGL2 is
 * unavailable.
 */
export function compileVelloShaders(canvas, contextAttributes = {}) {
  const gl = canvas.getContext("webgl2", {
    depth: true,
    ...contextAttributes,
    antialias: false,
    failIfMajorPerformanceCaveat: true,
  });
  if (!gl) {
    return null;
  }
  // Must be enabled before compiling: besides the completion status that `vello_gpu` polls,
  // enabling it is what makes Chromium compile on background threads.
  gl.getExtension("KHR_parallel_shader_compile");

  const startTime = performance.now();
  const programs = {};
  for (const [name, [vertexSource, fragmentSource]] of Object.entries(PROGRAMS)) {
    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, vertexSource);
    gl.compileShader(vertexShader);
    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, fragmentSource);
    gl.compileShader(fragmentShader);
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    programs[name] = { program, vertexShader, fragmentShader };
  }
  // Browsers may buffer WebGL commands until the next flush, which would delay compilation
  // until the page yields.
  gl.flush();

  return { moduleHash: MODULE_HASH, startTime, gl, programs };
}
