const environment = document.querySelector("#environment");
const status = document.querySelector("#status");
const output = document.querySelector("#output");
const runButton = document.querySelector("#run");
const copyButton = document.querySelector("#copy");
const warmupInput = document.querySelector("#warmup");
const benchmarkTimeInput = document.querySelector("#benchmark-time");

let wasm;

const DIMENSIONS = [
  { width: 800, height: 500 },
  { width: 1920, height: 1080 },
  { width: 3840, height: 2160 },
];

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(resolve));
}

async function loadWasm() {
  const response = await fetch("./pkg/premultiply_web_bench.wasm");
  if (!response.ok) {
    throw new Error(`Could not load WebAssembly (${response.status}). Run ./build.sh first.`);
  }
  const bytes = await response.arrayBuffer();
  const module = await WebAssembly.instantiate(bytes);
  return module.instance.exports;
}

function formatDuration(milliseconds) {
  if (milliseconds < 0.001) return `${(milliseconds * 1e6).toFixed(2)} ns`;
  if (milliseconds < 1) return `${(milliseconds * 1e3).toFixed(2)} µs`;
  if (milliseconds < 1000) return `${milliseconds.toFixed(3)} ms`;
  return `${(milliseconds / 1000).toFixed(3)} s`;
}

function formatThroughput(bytesPerSecond) {
  const gibPerSecond = bytesPerSecond / (1024 ** 3);
  return `${gibPerSecond.toFixed(3)} GiB/s`;
}

async function runForDuration(name, benchmark, duration, phase) {
  const maximumChunkTime = 50;
  let elapsed = 0;
  let iterations = 0;
  let observed = 0;
  while (elapsed < duration) {
    wasm.reset();
    const chunkTarget = Math.min(maximumChunkTime, duration - elapsed);
    const start = performance.now();
    let chunkIterations = 0;
    do {
      observed ^= benchmark();
      chunkIterations++;
    } while (performance.now() - start < chunkTarget);
    elapsed += performance.now() - start;
    iterations += chunkIterations;
    const progress = Math.min(100, Math.round(elapsed / duration * 100));
    status.textContent = `${phase} ${name}: ${progress}% (${iterations.toLocaleString()} iterations)…`;
    await nextFrame();
  }
  window.__premultiplyBenchmarkResult = observed;
  return { elapsed, iterations };
}

function formatResult(result) {
  const average = result.elapsed / result.iterations;
  const throughput = result.byteLength / (average / 1000);
  return [
    result.name,
    `  time:       ${formatDuration(average)}`,
    `  thrpt:      ${formatThroughput(throughput)}`,
    `  iterations: ${result.iterations.toLocaleString()}`,
    `  elapsed:    ${formatDuration(result.elapsed)}`,
  ].join("\n");
}

function verifyImplementations() {
  const pointer = wasm.data_ptr();
  const length = wasm.data_len();
  wasm.reset();
  wasm.run_old_scalar();
  const oldPixels = new Uint8Array(wasm.memory.buffer, pointer, length).slice();
  wasm.reset();
  wasm.run_new_interleaved_64();
  const newPixels = new Uint8Array(wasm.memory.buffer, pointer, length);
  let differingBytes = 0;
  let maximumDifference = 0;
  for (let index = 0; index < length; index++) {
    const difference = Math.abs(oldPixels[index] - newPixels[index]);
    differingBytes += Number(difference !== 0);
    maximumDifference = Math.max(maximumDifference, difference);
  }
  if (maximumDifference > 1) {
    throw new Error(`Premultiply outputs differ by as much as ${maximumDifference}; expected at most one.`);
  }
  return differingBytes;
}

async function runBenchmarks() {
  runButton.disabled = true;
  copyButton.disabled = true;
  status.className = "";
  output.textContent = "";
  try {
    const warmup = Number(warmupInput.value);
    const benchmarkTime = Number(benchmarkTimeInput.value);
    if (!Number.isFinite(warmup) || warmup <= 0 || !Number.isFinite(benchmarkTime) || benchmarkTime <= 0) {
      throw new Error("Warm-up and benchmark time must both be positive numbers.");
    }
    const results = [];
    const verification = [];
    for (const { width, height } of DIMENSIONS) {
      const dimensions = `${width}x${height}`;
      if (wasm.set_dimensions(width, height) === 0) {
        throw new Error(`WebAssembly rejected the ${dimensions} benchmark dimensions.`);
      }
      status.textContent = `Checking ${dimensions} output…`;
      await nextFrame();
      verification.push([dimensions, verifyImplementations()]);
      const byteLength = wasm.data_len();
      const benchmarks = [
        [`premultiply_rgba8/${dimensions}/old_scalar`, wasm.run_old_scalar],
        [
          `premultiply_rgba8/${dimensions}/new_simd_interleaved_64_with_transparency`,
          wasm.run_new_interleaved_64,
        ],
      ];
      for (const [name, benchmark] of benchmarks) {
        await runForDuration(name, benchmark, warmup, "Warming up");
        const measurement = await runForDuration(name, benchmark, benchmarkTime, "Measuring");
        results.push({ name, byteLength, ...measurement });
        output.textContent = results.map(formatResult).join("\n\n");
      }
    }
    const differences = verification
      .map(([dimensions, count]) => `${dimensions}: ${count.toLocaleString()}`)
      .join(", ");
    status.textContent = `Benchmark complete. Channel bytes rounded one value above exact / 255 — ${differences}.`;
    copyButton.disabled = false;
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : String(error);
    status.className = "error";
  } finally {
    runButton.disabled = false;
  }
}

runButton.addEventListener("click", runBenchmarks);
copyButton.addEventListener("click", async () => {
  await navigator.clipboard.writeText(output.textContent);
  copyButton.textContent = "Copied";
  setTimeout(() => { copyButton.textContent = "Copy results"; }, 1200);
});

try {
  wasm = await loadWasm();
  const simd = wasm.simd_enabled() !== 0;
  environment.textContent = [
    DIMENSIONS.map(({ width, height }) => `${width}×${height}`).join(", "),
    simd ? "Wasm SIMD128 build" : "non-SIMD Wasm build",
    navigator.userAgent,
  ].join(" · ");
  runButton.disabled = false;
} catch (error) {
  environment.textContent = error instanceof Error ? error.message : String(error);
  environment.className = "error";
}
