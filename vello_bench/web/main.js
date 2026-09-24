const MAX_ITERATIONS = 0xffff_ffff;
const MAX_CONSECUTIVE_ZERO_SAMPLES = 100;

class Runner {
  constructor(artifacts) {
    this.worker = new Worker("worker.js");
    this.nextRequest = 0;
    this.pending = new Map();
    this.worker.onmessage = ({ data }) => {
      const pending = this.pending.get(data.request);
      this.pending.delete(data.request);
      if (data.error) pending.reject(new Error(data.error));
      else pending.resolve(data);
    };
    this.ready = this.call("init", { artifacts });
  }

  call(type, message = {}) {
    const request = this.nextRequest++;
    return new Promise((resolve, reject) => {
      this.pending.set(request, { resolve, reject });
      this.worker.postMessage({ request, type, ...message });
    });
  }

  async sample(artifact, id, iterations) {
    return (await this.call("sample", { artifact, id, iterations })).elapsedNanos;
  }

  close() {
    this.worker.terminate();
    for (const { reject } of this.pending.values()) {
      reject(new Error("benchmark worker was replaced"));
    }
    this.pending.clear();
  }
}

const modeSummary = document.querySelector("#mode");
const showExtended = document.querySelector("#show-extended");
const showNonSimd = document.querySelector("#show-non-simd");
const showF32 = document.querySelector("#show-f32");
const selectAll = document.querySelector("#select-all");
const measurementTime = document.querySelector("#measurement-ms");
const sampleCount = document.querySelector("#sample-count");
const warmupTime = document.querySelector("#warmup-ms");
const selectionSummary = document.querySelector("#selection-summary");
const runButton = document.querySelector("#run");
const status = document.querySelector("#status");
const results = document.querySelector("#results");
let runner;
let comparing = false;
let loading = false;
let running = false;

async function loadBenchmarks() {
  loading = true;
  setStatus("Loading artifacts…");
  closeRunner();
  results.replaceChildren();
  setControls();

  let nextRunner;
  try {
    const artifactA = "./generated/vello_bench_a.wasm";
    const artifactB = "./generated/vello_bench_b.wasm";
    const isComparing = await artifactExists(artifactB);
    modeSummary.textContent = isComparing ? "Comparing artifacts A and B." : "Benchmarking artifact A.";
    const artifacts = { a: artifactA };
    if (isComparing) artifacts.b = artifactB;
    nextRunner = new Runner(artifacts);
    const { manifests } = await nextRunner.ready;
    const manifestA = manifests.a;
    const manifestB = manifests.b;
    if (manifestB && JSON.stringify(manifestA.cases) !== JSON.stringify(manifestB.cases)) {
      throw new Error("A and B expose different benchmark manifests");
    }

    runner = nextRunner;
    comparing = isComparing;
    document.body.classList.toggle("comparing", comparing);
    nextRunner = undefined;
    populateCases(manifestA.cases);
    const extendedCount = manifestA.cases.filter(({ extended }) => extended).length;
    const nonSimdCount = manifestA.cases.filter(({ nonSimd }) => nonSimd).length;
    const f32Count = manifestA.cases.filter(({ f32 }) => f32).length;
    showExtended.closest("label").hidden = extendedCount === 0;
    showNonSimd.closest("label").hidden = nonSimdCount === 0;
    showF32.closest("label").hidden = f32Count === 0;
    setStatus(
      `${manifestA.cases.length} benchmark variants loaded ` +
      `(${extendedCount} extended, ${nonSimdCount} non-SIMD, ${f32Count} f32).`,
    );
  } catch (error) {
    nextRunner?.close();
    modeSummary.textContent = "Unable to load benchmark artifacts.";
    setStatus(error.message);
  }
  loading = false;
  updateTreeSelection();
  setControls();
}

async function artifactExists(url) {
  return (await fetch(url, { method: "HEAD", cache: "no-store" })).ok;
}

for (const toggle of [showExtended, showNonSimd, showF32]) {
  toggle.onchange = () => {
    updateCaseVisibility();
    updateTreeSelection();
    setControls();
  };
}

selectAll.onchange = () => {
  for (const input of visibleCaseInputs()) input.checked = selectAll.checked;
  updateTreeSelection();
  setControls();
};

runButton.onclick = async () => {
  const ids = selectedVisibleCaseIds();
  const measurementMillis = Number(measurementTime.value);
  const samples = Number(sampleCount.value);
  const warmupMillis = Number(warmupTime.value);
  if (!Number.isFinite(measurementMillis) || measurementMillis <= 0) {
    setStatus("Measurement time must be greater than zero.");
    return;
  }
  if (!Number.isInteger(samples) || samples <= 0) {
    setStatus("Sample count must be a positive integer.");
    return;
  }
  if (!Number.isFinite(warmupMillis) || warmupMillis < 0) {
    setStatus("Warmup time must not be negative.");
    return;
  }
  const targetSampleNanos = measurementMillis * 1_000_000 / samples;
  clearResults();
  setRunning(true);
  try {
    for (const [index, id] of ids.entries()) {
      const prefix = `${index + 1}/${ids.length} ${id}`;
      setStatus(`${prefix}: warming up A…`);
      const iterationsA = await warmUp(runner, "a", id, warmupMillis, targetSampleNanos);
      let iterationsB = iterationsA;
      if (comparing) {
        setStatus(`${prefix}: warming up B…`);
        iterationsB = await warmUp(runner, "b", id, warmupMillis, targetSampleNanos);
      }

      const samplesA = [];
      const samplesB = [];
      const ratios = [];
      let completedBatches = 0;
      let consecutiveZeroSamples = 0;
      do {
        const progress = Math.round(100 * completedBatches / samples);
        const iterationSummary = comparing
          ? `${iterationsA} A / ${iterationsB} B iterations per sample`
          : `${iterationsA} iterations per sample`;
        setStatus(`${prefix}: measuring ${progress}% (${iterationSummary})…`);

        let elapsedA;
        let elapsedB;
        if (!comparing || completedBatches % 4 === 0 || completedBatches % 4 === 3) {
          elapsedA = await runner.sample("a", id, iterationsA);
          if (comparing) elapsedB = await runner.sample("b", id, iterationsB);
        } else {
          elapsedB = await runner.sample("b", id, iterationsB);
          elapsedA = await runner.sample("a", id, iterationsA);
        }

        // A zero can occur when a browser rounds both performance.now() calls to the same tick.
        // Discard the pair so A and B retain matching observations.
        if (elapsedA > 0 && (!comparing || elapsedB > 0)) {
          const nanosA = elapsedA / iterationsA;
          samplesA.push(nanosA);
          if (comparing) {
            const nanosB = elapsedB / iterationsB;
            samplesB.push(nanosB);
            ratios.push(nanosB / nanosA);
          }
          completedBatches++;
          consecutiveZeroSamples = 0;
        } else {
          consecutiveZeroSamples++;
          if (consecutiveZeroSamples >= MAX_CONSECUTIVE_ZERO_SAMPLES) {
            throw new Error(`browser timer could not measure a sample of ${id}`);
          }
        }
      } while (completedBatches < samples);

      showResult(
        id,
        average(samplesA),
        standardDeviation(samplesA),
        comparing ? average(samplesB) : null,
        comparing ? standardDeviation(samplesB) : null,
        comparing ? average(ratios) : null,
        comparing ? standardDeviation(ratios) : null,
      );
    }
    setStatus(`Finished ${ids.length} benchmark${ids.length === 1 ? "" : "s"}.`);
  } catch (error) {
    setStatus(error.message);
  } finally {
    setRunning(false);
  }
};

async function warmUp(runner, artifact, id, warmupMillis, targetSampleNanos) {
  const targetNanos = warmupMillis * 1_000_000;
  let warmedNanos = 0;
  let iterations = 1;
  do {
    const elapsedNanos = await runner.sample(artifact, id, iterations);
    warmedNanos += Math.max(1, elapsedNanos);
    const scale = elapsedNanos > 0
      ? Math.max(0.01, Math.min(100, targetSampleNanos / elapsedNanos))
      : 10;
    iterations = Math.max(
      1,
      Math.min(MAX_ITERATIONS, Math.round(iterations * scale)),
    );
  }
  while (warmedNanos < targetNanos);
  return iterations;
}

function populateCases(cases) {
  const root = { children: new Map() };
  for (const benchmark of cases) {
    const parts = benchmark.id.split("/");
    let parent = root;
    for (const part of parts.slice(0, -1)) {
      if (!parent.children.has(part)) {
        parent.children.set(part, { name: part, children: new Map() });
      }
      parent = parent.children.get(part);
    }
    parent.children.set(parts.at(-1), { name: parts.at(-1), benchmark });
  }

  appendTreeRows(root);
  updateCaseVisibility();
  updateTreeSelection();
}

function appendTreeRows(parent, depth = 0, parentPath = "") {
  for (const node of parent.children.values()) {
    const row = document.createElement("tr");
    const path = parentPath ? `${parentPath}/${node.name}` : node.name;
    if (node.benchmark) {
      row.className = "case";
      row.dataset.id = node.benchmark.id;
      row.dataset.extended = String(node.benchmark.extended);
      row.dataset.nonSimd = String(node.benchmark.nonSimd);
      row.dataset.f32 = String(node.benchmark.f32);

      const input = document.createElement("input");
      input.type = "checkbox";
      input.className = "case-checkbox";
      input.value = node.benchmark.id;
      input.ariaLabel = node.benchmark.id;
      input.onchange = () => {
        updateTreeSelection();
        setControls();
      };

      const name = document.createElement("th");
      name.className = "benchmark";
      name.scope = "row";
      name.style.paddingLeft = `${0.5 + depth * 1.25}rem`;
      name.append(node.name);
      if (node.benchmark.extended) {
        const marker = document.createElement("span");
        marker.className = "extended";
        marker.textContent = " extended";
        name.append(marker);
      }
      row.append(
        selectionCell(input),
        name,
        resultCell("result-a", "A"),
        resultCell("result-b", "B"),
        resultCell("result-change", "Change"),
      );
    } else {
      row.className = "category";
      row.dataset.path = path;
      const input = document.createElement("input");
      input.type = "checkbox";
      input.className = "category-checkbox";
      input.ariaLabel = `Select ${path}`;
      input.onchange = () => {
        for (const leaf of visibleCategoryInputs(row)) leaf.checked = input.checked;
        updateTreeSelection();
        setControls();
      };
      const name = document.createElement("th");
      name.className = "benchmark";
      name.colSpan = comparing ? 4 : 2;
      name.scope = "rowgroup";
      name.style.paddingLeft = `${0.5 + depth * 1.25}rem`;
      name.textContent = node.name;
      row.append(selectionCell(input), name);
    }
    results.append(row);
    if (!node.benchmark) appendTreeRows(node, depth + 1, path);
  }
}

function selectionCell(input) {
  const cell = document.createElement("td");
  cell.className = "selection";
  cell.append(input);
  return cell;
}

function resultCell(className, label) {
  const cell = document.createElement("td");
  cell.className = className;
  cell.dataset.label = label;
  cell.textContent = "—";
  return cell;
}

function updateCaseVisibility() {
  for (const row of results.querySelectorAll(".case")) {
    row.hidden = (row.dataset.extended === "true" && !showExtended.checked)
      || (row.dataset.nonSimd === "true" && !showNonSimd.checked)
      || (row.dataset.f32 === "true" && !showF32.checked);
  }
}

function updateTreeSelection() {
  const categories = [...results.querySelectorAll(".category")].reverse();
  for (const category of categories) {
    const leaves = visibleCategoryInputs(category);
    const checkbox = category.querySelector(".category-checkbox");
    const selected = leaves.filter(({ checked }) => checked).length;
    category.hidden = leaves.length === 0;
    checkbox.checked = leaves.length > 0 && selected === leaves.length;
    checkbox.indeterminate = selected > 0 && selected < leaves.length;
    checkbox.disabled = loading || running || leaves.length === 0;
  }
}

function caseInputs() {
  return [...results.querySelectorAll(".case-checkbox")];
}

function visibleCaseInputs() {
  return caseInputs().filter((input) => !input.closest(".case").hidden);
}

function visibleCategoryInputs(category) {
  const prefix = `${category.dataset.path}/`;
  return visibleCaseInputs().filter(({ value }) => value.startsWith(prefix));
}

function selectedVisibleCaseIds() {
  return visibleCaseInputs().filter(({ checked }) => checked).map(({ value }) => value);
}

function setRunning(value) {
  running = value;
  updateTreeSelection();
  setControls();
}

function setControls() {
  const shown = visibleCaseInputs();
  const selected = shown.filter(({ checked }) => checked).length;
  const loaded = Boolean(runner);
  const locked = loading || running;
  for (const toggle of [showExtended, showNonSimd, showF32]) toggle.disabled = locked;
  measurementTime.disabled = locked;
  sampleCount.disabled = locked;
  warmupTime.disabled = locked;
  for (const input of caseInputs()) input.disabled = locked;
  selectAll.checked = shown.length > 0 && selected === shown.length;
  selectAll.indeterminate = selected > 0 && selected < shown.length;
  selectAll.disabled = locked || !loaded || shown.length === 0;
  runButton.disabled = locked || !loaded || selected === 0;
  selectionSummary.textContent = loaded ? `${selected} of ${shown.length} shown selected` : "";
}

function closeRunner() {
  runner?.close();
  runner = undefined;
  comparing = false;
  document.body.classList.remove("comparing");
}

function showResult(
  id,
  averageA,
  standardDeviationA,
  averageB,
  standardDeviationB,
  ratio,
  ratioStandardDeviation,
) {
  const row = [...results.rows].find((candidate) => candidate.dataset.id === id);
  row.querySelector(".result-a").textContent = formatResult(averageA, standardDeviationA);
  row.querySelector(".result-b").textContent = averageB === null
    ? "—"
    : formatResult(averageB, standardDeviationB);

  const change = row.querySelector(".result-change");
  change.classList.remove("regression", "improvement");
  if (ratio === null) {
    change.textContent = "—";
  } else {
    const percentage = (ratio - 1) * 100;
    change.textContent = `${percentage.toFixed(2)}% ` +
      `(± ${(ratioStandardDeviation * 100).toFixed(2)} pp)`;
    if (percentage >= 5) change.classList.add("regression");
    else if (percentage <= -5) change.classList.add("improvement");
  }
}

function clearResults() {
  for (const cell of results.querySelectorAll(".result-a, .result-b, .result-change")) {
    cell.textContent = "—";
    cell.classList.remove("regression", "improvement");
  }
}

function standardDeviation(values) {
  if (values.length < 2) return 0;
  const mean = average(values);
  const squaredDeviations = values.reduce(
    (sum, value) => sum + (value - mean) ** 2,
    0,
  );
  return Math.sqrt(squaredDeviations / (values.length - 1));
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function formatResult(averageValue, standardDeviation) {
  const percentage = standardDeviation / averageValue * 100;
  return `${formatDuration(averageValue)} (± ${percentage.toFixed(2)}%)`;
}

function formatDuration(nanos) {
  if (nanos < 1_000) return `${nanos.toFixed(2)} ns`;
  if (nanos < 1_000_000) return `${(nanos / 1_000).toFixed(2)} µs`;
  return `${(nanos / 1_000_000).toFixed(2)} ms`;
}

function setStatus(message) {
  status.textContent = message;
}

loadBenchmarks();
