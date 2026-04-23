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

const SHARED_VERTEX_SHADER = `#version 300 es
precision highp float;
precision highp int;

layout(location = 0) in vec2 a_position;

flat out uint v0;
out vec2 v1;
out vec2 v2;
flat out uint v3;
flat out uint v4;
flat out uint v5;

void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
  v0 = 1u << 29u;
  v1 = vec2(0.25, 0.5);
  v2 = vec2(0.75, 0.125);
  v3 = 3u;
  v4 = 1u;
  v5 = 7u;
}
`;

const TESTS = [
  { name: "Minimal Pass", path: "./render_strips_min_pass.frag.glsl" },
  { name: "Minimal Fail", path: "./render_strips_min_fail.frag.glsl" },
  { name: "Render-Like Fail", path: "./render_strips_generated.frag.glsl" },
  { name: "Render-Like Fixed", path: "./render_strips_generated_patched.frag.glsl" },
];

if (!gl) {
  meta.innerHTML = `<div><strong>WebGL2:</strong> unavailable</div>`;
  results.innerHTML =
    `<div class="result"><h2>No WebGL2 context</h2><pre>This browser or device did not provide a WebGL2 context.</pre></div>`;
} else {
  renderMeta();
  void runAllTests();
}

runAllButton.addEventListener("click", () => {
  void runAllTests();
});

function renderMeta() {
  const debugInfo = gl.getExtension("WEBGL_debug_renderer_info");
  const vendor = debugInfo
    ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL)
    : gl.getParameter(gl.VENDOR);
  const renderer = debugInfo
    ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL)
    : gl.getParameter(gl.RENDERER);

  meta.innerHTML = `
    <div><strong>Vendor:</strong> ${escapeHtml(String(vendor))}</div>
    <div><strong>Renderer:</strong> ${escapeHtml(String(renderer))}</div>
  `;
}

async function runAllTests() {
  results.innerHTML = "";

  for (const test of TESTS) {
    const source = await loadFragmentSource(test.path);
    const result = compileAndInspectProgram(SHARED_VERTEX_SHADER, source.text);
    results.appendChild(renderResult(test.name, source.error, result));

    console.group(`WebGL struct repro: ${test.name}`);
    if (source.error) {
      console.error(source.error);
    }
    console.log("Link success:", result.ok);
    console.log("Target location:", result.uniformLocation);
    if (result.log.trim()) {
      console.log("Log:", result.log);
    }
    console.groupEnd();
  }
}

async function loadFragmentSource(path) {
  try {
    const response = await fetch(path, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to fetch ${path}: HTTP ${response.status}`);
    }
    return { text: await response.text(), error: "" };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      text: `#version 300 es
precision highp float;
out vec4 out_color;
void main() { out_color = vec4(1.0, 0.0, 0.0, 1.0); }
`,
      error: message,
    };
  }
}

function compileAndInspectProgram(vertexSource, fragmentSource) {
  const vertexShader = compileShader(gl.VERTEX_SHADER, vertexSource);
  if (!vertexShader.ok) {
    return failedResult("vertex", vertexShader.log);
  }

  const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);
  if (!fragmentShader.ok) {
    gl.deleteShader(vertexShader.shader);
    return failedResult("fragment", fragmentShader.log);
  }

  const program = gl.createProgram();
  gl.attachShader(program, vertexShader.shader);
  gl.attachShader(program, fragmentShader.shader);
  gl.linkProgram(program);

  const ok = Boolean(gl.getProgramParameter(program, gl.LINK_STATUS));
  const log = gl.getProgramInfoLog(program) || "";
  const uniformLocation = ok
    ? (gl.getUniformLocation(program, TARGET_UNIFORM) ? "present" : "null")
    : "null";

  gl.deleteShader(vertexShader.shader);
  gl.deleteShader(fragmentShader.shader);
  gl.deleteProgram(program);

  return { ok, log, uniformLocation };
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

function failedResult(stage, log) {
  return {
    ok: false,
    log: `${stage}: ${log}`,
    uniformLocation: "null",
  };
}

function renderResult(name, loadError, result) {
  const wrapper = document.createElement("article");
  wrapper.className = "result";

  let statusClass = "fail";
  let statusText = "missing";
  if (loadError) {
    statusText = "load error";
  } else if (!result.ok) {
    statusText = "compile/link error";
  } else if (result.uniformLocation === "present") {
    statusClass = "pass";
    statusText = "present";
  }

  wrapper.innerHTML = `
    <div style="display:flex;justify-content:space-between;gap:12px;align-items:baseline;">
      <h2>${escapeHtml(name)}</h2>
      <div class="status ${statusClass}">${escapeHtml(statusText)}</div>
    </div>
  `;

  return wrapper;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}
