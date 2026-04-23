const canvas = document.getElementById("gl");
const meta = document.getElementById("meta");
const results = document.getElementById("results");
const runAllButton = document.getElementById("run-all");
const clearButton = document.getElementById("clear");

const gl = canvas.getContext("webgl2", {
  alpha: false,
  antialias: false,
  depth: false,
  stencil: false,
});

const BASE_VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location = 0) in vec2 a_position;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const TESTS = [
  {
    name: "Array Constructor",
    description:
      "Matches the generated float[3](...) initialization form.",
    fragmentSource: `#version 300 es
precision highp float;
precision highp int;
out vec4 outColor;
float sum_weights() {
  float weights[3] = float[3](0.0, 1.0, 2.0);
  return weights[0] + weights[1] + weights[2];
}
void main() {
  outColor = vec4(vec3(sum_weights() / 3.0), 1.0);
}
`,
  },
  {
    name: "Struct Constructor With Arrays",
    description:
      "Matches the generated BlurParams(..., array, array) pattern.",
    fragmentSource: `#version 300 es
precision highp float;
precision highp int;
out vec4 outColor;
struct BlurParams {
  uint n_linear_taps;
  float center_weight;
  float linear_weights[3];
  float linear_offsets[3];
};
BlurParams make_params() {
  float weights[3] = float[3](0.0, 1.0, 2.0);
  float offsets[3] = float[3](3.0, 4.0, 5.0);
  float copyA[3] = weights;
  float copyB[3] = offsets;
  return BlurParams(2u, 0.5, copyA, copyB);
}
void main() {
  BlurParams params = make_params();
  outColor = vec4(params.center_weight, params.linear_weights[1] / 4.0, params.linear_offsets[2] / 6.0, 1.0);
}
`,
  },
  {
    name: "Struct Constructor With Large Array",
    description:
      "Matches the generated GpuFilterData(uint[12](...)) pattern.",
    fragmentSource: `#version 300 es
precision highp float;
precision highp int;
out vec4 outColor;
struct GpuFilterData {
  uint data[12];
};
GpuFilterData load_data() {
  return GpuFilterData(uint[12](0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u));
}
void main() {
  GpuFilterData data = load_data();
  outColor = vec4(float(data.data[11]) / 11.0, 0.2, 0.4, 1.0);
}
`,
  },
  {
    name: "Large Struct Constructor",
    description:
      "Matches the generated FilterVertexOutput(...) fragment-local constructor form.",
    fragmentSource: `#version 300 es
precision highp float;
precision highp int;
out vec4 outColor;
struct BigState {
  vec4 position;
  uint filter_offset;
  uvec2 src_offset;
  uvec2 src_size;
  uvec2 dest_offset;
  uvec2 dest_size;
  uvec2 dest_atlas_size;
  uvec2 original_offset;
  uvec2 original_size;
  uint pass_kind;
};
void main() {
  BigState state = BigState(
    gl_FragCoord,
    1u,
    uvec2(2u, 3u),
    uvec2(4u, 5u),
    uvec2(6u, 7u),
    uvec2(8u, 9u),
    uvec2(10u, 11u),
    uvec2(12u, 13u),
    uvec2(14u, 15u),
    16u
  );
  outColor = vec4(float(state.pass_kind) / 16.0, float(state.filter_offset), 0.0, 1.0);
}
`,
  },
  {
    name: "Safe Rewrite",
    description:
      "Equivalent logic written with field assignments instead of constructor-heavy forms.",
    fragmentSource: `#version 300 es
precision highp float;
precision highp int;
out vec4 outColor;
struct BlurParams {
  uint n_linear_taps;
  float center_weight;
  float linear_weights[3];
  float linear_offsets[3];
};
struct GpuFilterData {
  uint data[12];
};
struct BigState {
  vec4 position;
  uint filter_offset;
  uvec2 src_offset;
  uvec2 src_size;
  uvec2 dest_offset;
  uvec2 dest_size;
  uvec2 dest_atlas_size;
  uvec2 original_offset;
  uvec2 original_size;
  uint pass_kind;
};
BlurParams make_params() {
  BlurParams result;
  result.n_linear_taps = 2u;
  result.center_weight = 0.5;
  result.linear_weights[0] = 0.0;
  result.linear_weights[1] = 1.0;
  result.linear_weights[2] = 2.0;
  result.linear_offsets[0] = 3.0;
  result.linear_offsets[1] = 4.0;
  result.linear_offsets[2] = 5.0;
  return result;
}
GpuFilterData load_data() {
  GpuFilterData result;
  result.data[0] = 0u;
  result.data[1] = 1u;
  result.data[2] = 2u;
  result.data[3] = 3u;
  result.data[4] = 4u;
  result.data[5] = 5u;
  result.data[6] = 6u;
  result.data[7] = 7u;
  result.data[8] = 8u;
  result.data[9] = 9u;
  result.data[10] = 10u;
  result.data[11] = 11u;
  return result;
}
void main() {
  BlurParams params = make_params();
  GpuFilterData data = load_data();
  BigState state;
  state.position = gl_FragCoord;
  state.filter_offset = 1u;
  state.src_offset = uvec2(2u, 3u);
  state.src_size = uvec2(4u, 5u);
  state.dest_offset = uvec2(6u, 7u);
  state.dest_size = uvec2(8u, 9u);
  state.dest_atlas_size = uvec2(10u, 11u);
  state.original_offset = uvec2(12u, 13u);
  state.original_size = uvec2(14u, 15u);
  state.pass_kind = 16u;
  outColor = vec4(params.center_weight, float(data.data[11]) / 11.0, float(state.pass_kind) / 16.0, 1.0);
}
`,
  },
];

if (!gl) {
  meta.innerHTML = `<div><strong>WebGL2:</strong> unavailable</div>`;
  results.innerHTML =
    `<div class="result"><h2>No WebGL2 context</h2><pre>This browser or device did not provide a WebGL2 context, so the repro cannot run.</pre></div>`;
} else {
  renderMeta();
  runAllTests();
}

runAllButton.addEventListener("click", () => runAllTests());
clearButton.addEventListener("click", () => {
  results.innerHTML = "";
});

function renderMeta() {
  const debugInfo = gl.getExtension("WEBGL_debug_renderer_info");
  const renderer = debugInfo
    ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL)
    : gl.getParameter(gl.RENDERER);
  const vendor = debugInfo
    ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL)
    : gl.getParameter(gl.VENDOR);
  const version = gl.getParameter(gl.VERSION);
  const shadingLanguageVersion = gl.getParameter(gl.SHADING_LANGUAGE_VERSION);

  meta.innerHTML = `
    <div><strong>Vendor:</strong> ${escapeHtml(String(vendor))}</div>
    <div><strong>Renderer:</strong> ${escapeHtml(String(renderer))}</div>
    <div><strong>WebGL Version:</strong> ${escapeHtml(String(version))}</div>
    <div><strong>GLSL Version:</strong> ${escapeHtml(String(shadingLanguageVersion))}</div>
  `;
}

function runAllTests() {
  results.innerHTML = "";
  TESTS.forEach((test) => {
    const result = compileProgram(BASE_VERTEX_SHADER, test.fragmentSource);
    results.appendChild(renderResult(test, result));
    console.group(`WebGL Constructor Repro: ${test.name}`);
    console.log(test.description);
    console.log("Success:", result.ok);
    if (!result.ok) {
      console.error(result.log);
    } else {
      console.log("Program linked successfully");
    }
    console.groupEnd();
  });
}

function compileProgram(vertexSource, fragmentSource) {
  const vertexShader = compileShader(gl.VERTEX_SHADER, vertexSource);
  if (!vertexShader.ok) {
    return {
      ok: false,
      stage: "vertex",
      log: vertexShader.log,
      source: vertexSource,
    };
  }

  const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);
  if (!fragmentShader.ok) {
    gl.deleteShader(vertexShader.shader);
    return {
      ok: false,
      stage: "fragment",
      log: fragmentShader.log,
      source: fragmentSource,
    };
  }

  const program = gl.createProgram();
  gl.attachShader(program, vertexShader.shader);
  gl.attachShader(program, fragmentShader.shader);
  gl.linkProgram(program);

  const ok = gl.getProgramParameter(program, gl.LINK_STATUS);
  const log = gl.getProgramInfoLog(program) || "";

  gl.deleteShader(vertexShader.shader);
  gl.deleteShader(fragmentShader.shader);
  gl.deleteProgram(program);

  return {
    ok: Boolean(ok),
    stage: "link",
    log,
    source: fragmentSource,
  };
}

function compileShader(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  const ok = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
  const log = gl.getShaderInfoLog(shader) || "";

  if (!ok) {
    gl.deleteShader(shader);
    return { ok: false, log, source };
  }

  return { ok: true, shader, log, source };
}

function renderResult(test, result) {
  const wrapper = document.createElement("article");
  wrapper.className = "result";

  const statusClass = result.ok ? "pass" : "fail";
  const statusText = result.ok ? "PASS" : `FAIL (${result.stage})`;
  const log = result.log.trim() || "No compiler or linker log.";

  wrapper.innerHTML = `
    <h2>${escapeHtml(test.name)}</h2>
    <div>${escapeHtml(test.description)}</div>
    <div class="status ${statusClass}">${statusText}</div>
    <pre>${escapeHtml(log)}</pre>
    <details>
      <summary>Fragment Shader</summary>
      <pre>${escapeHtml(test.fragmentSource)}</pre>
    </details>
  `;

  return wrapper;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}
