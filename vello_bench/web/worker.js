const artifacts = new Map();

self.onmessage = async (event) => {
  const { request, type } = event.data;
  try {
    if (type === "init") {
      const manifests = {};
      for (const [name, url] of Object.entries(event.data.artifacts)) {
        const artifact = await loadArtifact(url);
        artifacts.set(name, artifact);
        manifests[name] = { cases: artifact.cases };
      }
      reply(request, { manifests });
    } else if (type === "sample") {
      const artifact = artifacts.get(event.data.artifact);
      if (!artifact) throw new Error(`unknown artifact: ${event.data.artifact}`);
      const index = artifact.cases.findIndex(({ id }) => id === event.data.id);
      if (index < 0) throw new Error(`unknown benchmark: ${event.data.id}`);
      const elapsedNanos = artifact.wasm.vello_bench_run_sample(index, event.data.iterations);
      reply(request, { elapsedNanos });
    }
  } catch (error) {
    reply(request, { error: String(error) });
  }
};

async function loadArtifact(url) {
  const imports = { vello_bench: { now: () => performance.now() } };
  const response = await fetch(url, { cache: "no-store" });
  const bytes = await response.arrayBuffer();
  const wasm = (await WebAssembly.instantiate(bytes, imports)).instance.exports;
  const decoder = new TextDecoder();
  const count = wasm.vello_bench_case_count();
  const cases = Array.from({ length: count }, (_, index) => {
    const pointer = wasm.vello_bench_case_name_ptr(index);
    const length = wasm.vello_bench_case_name_len(index);
    return {
      id: decoder.decode(new Uint8Array(wasm.memory.buffer, pointer, length)),
      extended: wasm.vello_bench_case_is_extended(index) !== 0,
      nonSimd: wasm.vello_bench_case_is_non_simd(index) !== 0,
      f32: wasm.vello_bench_case_is_f32(index) !== 0,
    };
  });
  return { wasm, cases };
}

function reply(request, message) {
  self.postMessage({ request, ...message });
}
