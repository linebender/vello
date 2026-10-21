// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// The build script prepends `SOURCES_HASH` and `PROGRAMS` to this file to generate
// `webgl_bootstrap::JS_MODULE`. The default context attributes and the program creation must
// match `vello_gpu`'s WebGL backend.

/**
 * Creates the canvas's WebGL2 context and starts compiling Vello's shader programs without
 * waiting for them to finish.
 *
 * The context keeps the attributes it is created with, so pass any attributes the application
 * needs in `contextAttributes`. They are merged over Vello's defaults: `antialias: false` and
 * `depth: true` are required by Vello (with `depth: false` only if the renderer is created without
 * a depth buffer), while `failIfMajorPerformanceCaveat: true` keeps Vello off slow software
 * renderers.
 *
 * Pass the result to `WebGlShaders::from_precompiled` with the same canvas, using a
 * `use_depth_buffer` that matches `depth`. Returns `null` if WebGL2 is unavailable.
 */
export function compileVelloShaders(canvas, contextAttributes = {}) {
  const gl = canvas.getContext("webgl2", {
    antialias: false,
    depth: true,
    failIfMajorPerformanceCaveat: true,
    ...contextAttributes,
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

  return { sourcesHash: SOURCES_HASH, startTime, gl, programs };
}
