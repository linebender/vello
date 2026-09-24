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

const showExtended = document.querySelector("#show-extended");
const showNonSimd = document.querySelector("#show-non-simd");
const showF32 = document.querySelector("#show-f32");
const selectAll = document.querySelector("#select-all");
const measurementTime = document.querySelector("#measurement-ms");
const sampleCount = document.querySelector("#sample-count");
const warmupTime = document.querySelector("#warmup-ms");
const runButton = document.querySelector("#run");
const status = document.querySelector("#status");
const results = document.querySelector("#results");
let runner;
let comparing = false;
let loading = false;
let running = false;

async function loadBenchmarks() {
  loading = true;
  setStatus("Loading…");
  closeRunner();
  results.textContent = "";
  setControls();

  let nextRunner;
  try {
    const artifactA = "./generated/vello_bench_a.wasm";
    const artifactB = "./generated/vello_bench_b.wasm";
    const isComparing = await artifactExists(artifactB);
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
    setStatus("");
  } catch (error) {
    if (nextRunner) nextRunner.close();
    setStatus(error.message);
  }
  loading = false;
  updateGroupSelection();
  setControls();
}

async function artifactExists(url) {
  return (await fetch(url, { method: "HEAD", cache: "no-store" })).ok;
}

for (const toggle of [showExtended, showNonSimd, showF32]) {
  toggle.onchange = () => {
    updateCaseVisibility();
    updateGroupSelection();
    setControls();
  };
}

selectAll.onchange = () => {
  for (const input of visibleCaseInputs()) input.checked = selectAll.checked;
  updateGroupSelection();
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
    setStatus("");
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
  const groups = new Map();
  for (const benchmark of cases) {
    const slash = benchmark.id.lastIndexOf("/");
    const path = slash < 0 ? "Benchmarks" : benchmark.id.slice(0, slash);
    if (!groups.has(path)) groups.set(path, []);
    groups.get(path).push({ name: benchmark.id.slice(slash + 1), benchmark });
  }

  for (const [path, casesInGroup] of groups) appendGroup(path, casesInGroup);
  updateCaseVisibility();
  updateGroupSelection();
}

function appendGroup(path, cases) {
  const group = document.createElement("div");
  group.className = "group";
  group.setAttribute("role", "group");
  group.setAttribute("aria-label", path);
  const category = document.createElement("div");
  category.className = "category";
  const categoryInput = document.createElement("input");
  categoryInput.type = "checkbox";
  categoryInput.className = "category-checkbox";
  categoryInput.setAttribute("aria-label", `Select ${path}`);
  categoryInput.onchange = () => {
    for (const leaf of visibleCategoryInputs(category)) leaf.checked = categoryInput.checked;
    updateGroupSelection();
    setControls();
  };
  const categoryName = document.createElement("strong");
  categoryName.className = "benchmark";
  categoryName.textContent = path;
  category.append(selectionCell(categoryInput), categoryName);
  group.append(category);

  for (const { name, benchmark } of cases) {
    const row = document.createElement("div");
    row.className = "case";
    row.dataset.id = benchmark.id;
    row.dataset.extended = String(benchmark.extended);
    row.dataset.nonSimd = String(benchmark.nonSimd);
    row.dataset.f32 = String(benchmark.f32);

    const input = document.createElement("input");
    input.type = "checkbox";
    input.className = "case-checkbox";
    input.value = benchmark.id;
    input.setAttribute("aria-label", benchmark.id);
    input.onchange = () => {
      updateGroupSelection();
      setControls();
    };

    const caseName = document.createElement("span");
    caseName.className = "benchmark";
    caseName.textContent = name;
    row.append(
      selectionCell(input),
      caseName,
      resultCell("result-a", "A"),
      resultCell("result-b", "B"),
      resultCell("result-change", "Change"),
    );
    group.append(row);
  }
  results.append(group);
}

function selectionCell(input) {
  const cell = document.createElement("span");
  cell.className = "selection";
  cell.append(input);
  return cell;
}

function resultCell(className, label) {
  const cell = document.createElement("span");
  cell.className = className;
  cell.dataset.label = label;
  return cell;
}

function updateCaseVisibility() {
  for (const row of results.querySelectorAll(".case")) {
    row.hidden = (row.dataset.extended === "true" && !showExtended.checked)
      || (row.dataset.nonSimd === "true" && !showNonSimd.checked)
      || (row.dataset.f32 === "true" && !showF32.checked);
  }
}

function updateGroupSelection() {
  for (const category of results.querySelectorAll(".category")) {
    const leaves = visibleCategoryInputs(category);
    const checkbox = category.querySelector(".category-checkbox");
    const selected = leaves.filter(({ checked }) => checked).length;
    category.closest(".group").hidden = leaves.length === 0;
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
  return [...category.closest(".group").querySelectorAll(".case:not([hidden]) .case-checkbox")];
}

function selectedVisibleCaseIds() {
  return visibleCaseInputs().filter(({ checked }) => checked).map(({ value }) => value);
}

function setRunning(value) {
  running = value;
  updateGroupSelection();
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
}

function closeRunner() {
  if (runner) runner.close();
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
  const row = [...results.querySelectorAll(".case")].find((candidate) => candidate.dataset.id === id);
  row.querySelector(".result-a").textContent = formatResult(averageA, standardDeviationA);
  row.querySelector(".result-b").textContent = averageB === null
    ? ""
    : formatResult(averageB, standardDeviationB);

  const change = row.querySelector(".result-change");
  change.classList.remove("regression", "improvement");
  if (ratio === null) {
    change.textContent = "";
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
    cell.textContent = "";
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
