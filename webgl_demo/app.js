const canvas = document.getElementById("gl");
const meta = document.getElementById("meta");
const results = document.getElementById("results");
const runAllButton = document.getElementById("run-all");

const gl = canvas.getContext("webgl2", {
  alpha: false,
  antialias: false,
  depth: false,
  stencil: false,
});

const TARGET_UNIFORM = "_group_0_binding_2_fs";

const SIMPLE_VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location = 0) in vec2 a_position;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const TESTS = [
  {
    name: "Baseline",
    description:
      "Only the config block, the clip sampler, and a tiny slot/blend branch remain.",
    vertexPath: "./render_strips.vert.glsl",
    fragmentPath: "./render_strips_minimal.frag.glsl",
    inspectUniforms: [
      "_group_0_binding_0_fs",
      "_group_0_binding_2_fs",
      "_group_1_binding_0_fs",
      "_group_2_binding_0_fs",
      "_group_3_binding_0_fs",
    ],
  },
  {
    name: "2 Fields",
    description:
      "paint_and_rect_flag + position",
    vertexPath: "./render_strips.vert.glsl",
    fragmentPath: "./render_strips_fields_2.frag.glsl",
    inspectUniforms: [
      "_group_0_binding_0_fs",
      "_group_0_binding_2_fs",
      "_group_1_binding_0_fs",
      "_group_2_binding_0_fs",
      "_group_3_binding_0_fs",
    ],
  },
  {
    name: "Generated Render Strips",
    description:
      "Compiles the real generated render_strips vertex + fragment shaders and queries the same sampler uniform Vello unwraps.",
    vertexPath: "./render_strips_generated.vert.glsl",
    fragmentPath: "./render_strips_generated.frag.glsl",
    inspectUniforms: [
      "_group_0_binding_0_fs",
      "_group_0_binding_2_fs",
      "_group_1_binding_0_fs",
      "_group_2_binding_0_fs",
      "_group_3_binding_0_fs",
    ],
  },
  {
    name: "Generated Patched",
    description:
      "Same generated fragment shader, but patched to avoid fragment-local VertexOutput reconstruction.",
    vertexPath: "./render_strips_generated.vert.glsl",
    fragmentPath: "./render_strips_generated_patched.frag.glsl",
    inspectUniforms: [
      "_group_0_binding_0_fs",
      "_group_0_binding_2_fs",
      "_group_1_binding_0_fs",
      "_group_2_binding_0_fs",
      "_group_3_binding_0_fs",
    ],
  },
];

if (!gl) {
  meta.innerHTML = `<div><strong>WebGL2:</strong> unavailable</div>`;
  results.innerHTML =
    `<div class="result"><h2>No WebGL2 context</h2><pre>This browser or device did not provide a WebGL2 context, so the repro cannot run.</pre></div>`;
} else {
  renderMeta();
  void runAllTests();
}

runAllButton.addEventListener("click", () => {
  void runAllTests();
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
    <div><strong>WebGL:</strong> ${escapeHtml(String(version))}</div>
    <div><strong>GLSL:</strong> ${escapeHtml(String(shadingLanguageVersion))}</div>
  `;
}

async function runAllTests() {
  results.innerHTML = "";

  for (const test of TESTS) {
    const resolved = await resolveSources(test);
    const result = compileAndInspectProgram(
      resolved.vertexSource,
      resolved.fragmentSource,
      test.inspectUniforms,
    );
    results.appendChild(renderResult(test, resolved, result));

    console.group(`WebGL Uniform Repro: ${test.name}`);
    console.log(test.description);
    console.log("Link success:", result.ok);
    console.log("Target uniform:", TARGET_UNIFORM);
    console.log("Target location:", result.uniformLocations[TARGET_UNIFORM]);
    console.log("Active uniforms:", result.activeUniforms);
    if (resolved.loadError) {
      console.error("Source load error:", resolved.loadError);
    }
    if (result.log.trim()) {
      console.log("Program log:", result.log);
    }
    if (!result.ok) {
      console.error("Stage:", result.stage);
    }
    console.groupEnd();
  }
}

async function resolveSources(test) {
  if (test.vertexSource && test.fragmentSource) {
    return {
      vertexSource: test.vertexSource,
      fragmentSource: test.fragmentSource,
      loadError: "",
    };
  }

  try {
    const [vertexSource, fragmentSource] = await Promise.all([
      fetchText(test.vertexPath),
      fetchText(test.fragmentPath),
    ]);
    return { vertexSource, fragmentSource, loadError: "" };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      vertexSource: SIMPLE_VERTEX_SHADER,
      fragmentSource: `#version 300 es
precision highp float;
out vec4 outColor;
void main() { outColor = vec4(1.0, 0.0, 0.0, 1.0); }
`,
      loadError: message,
    };
  }
}

async function fetchText(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${path}: HTTP ${response.status}`);
  }
  return response.text();
}

function compileAndInspectProgram(vertexSource, fragmentSource, inspectUniforms) {
  const vertexShader = compileShader(gl.VERTEX_SHADER, vertexSource);
  if (!vertexShader.ok) {
    return failedResult("vertex", vertexShader.log, vertexSource, fragmentSource);
  }

  const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);
  if (!fragmentShader.ok) {
    gl.deleteShader(vertexShader.shader);
    return failedResult(
      "fragment",
      fragmentShader.log,
      vertexSource,
      fragmentSource,
    );
  }

  const program = gl.createProgram();
  gl.attachShader(program, vertexShader.shader);
  gl.attachShader(program, fragmentShader.shader);
  gl.linkProgram(program);

  const ok = Boolean(gl.getProgramParameter(program, gl.LINK_STATUS));
  const log = gl.getProgramInfoLog(program) || "";
  const activeUniforms = ok ? getActiveUniforms(program) : [];
  const uniformLocations = ok ? getUniformLocations(program, inspectUniforms) : {};

  gl.deleteShader(vertexShader.shader);
  gl.deleteShader(fragmentShader.shader);
  gl.deleteProgram(program);

  return {
    ok,
    stage: "link",
    log,
    vertexSource,
    fragmentSource,
    activeUniforms,
    uniformLocations,
  };
}

function failedResult(stage, log, vertexSource, fragmentSource) {
  return {
    ok: false,
    stage,
    log,
    vertexSource,
    fragmentSource,
    activeUniforms: [],
    uniformLocations: {},
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
    return { ok: false, log };
  }

  return { ok: true, shader, log };
}

function getActiveUniforms(program) {
  const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
  const uniforms = [];
  for (let i = 0; i < count; i += 1) {
    const info = gl.getActiveUniform(program, i);
    if (!info) {
      continue;
    }
    uniforms.push({
      name: info.name,
      size: info.size,
      type: enumName(info.type),
    });
  }
  return uniforms;
}

function getUniformLocations(program, names) {
  const locations = {};
  for (const name of names) {
    const location = gl.getUniformLocation(program, name);
    locations[name] = location ? "present" : "null";
  }
  return locations;
}

function renderResult(test, resolved, result) {
  const wrapper = document.createElement("article");
  wrapper.className = "result";

  let statusClass = "fail";
  let statusText = "missing";
  if (resolved.loadError) {
    statusText = "load error";
  } else if (!result.ok) {
    statusText = `compile/link error`;
  } else if (result.uniformLocations[TARGET_UNIFORM] === "present") {
    statusClass = "pass";
    statusText = "present";
  }

  wrapper.innerHTML = `
    <div style="display:flex;justify-content:space-between;gap:12px;align-items:baseline;">
      <h2>${escapeHtml(test.name)}</h2>
      <div class="status ${statusClass}">${escapeHtml(statusText)}</div>
    </div>
  `;

  return wrapper;
}

function enumName(value) {
  const entries = [
    ["SAMPLER_2D", gl.SAMPLER_2D],
    ["SAMPLER_2D_ARRAY", gl.SAMPLER_2D_ARRAY],
    ["SAMPLER_CUBE", gl.SAMPLER_CUBE],
    ["INT_SAMPLER_2D", gl.INT_SAMPLER_2D],
    ["UNSIGNED_INT_SAMPLER_2D", gl.UNSIGNED_INT_SAMPLER_2D],
    ["UNSIGNED_INT", gl.UNSIGNED_INT],
    ["FLOAT", gl.FLOAT],
  ];
  for (const [name, enumValue] of entries) {
    if (enumValue === value) {
      return name;
    }
  }
  return `0x${value.toString(16)}`;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}
