// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Saved benchmark runs (baselines) in IndexedDB, and the statistics used to compare against them.

const DB_NAME = 'glifo-harness';
const STORE = 'baselines';

/** Two-sided p-value below which a difference counts as real rather than noise. */
export const SIGNIFICANCE_LEVEL = 0.05;
/**
 * Default relative change below which a difference is reported as unchanged, even if significant.
 *
 * Repeating the same build in one browser session drifts by a few percent (GPU clocks, browser
 * warm-up), which no test over a single session can tell apart from a real change.
 */
export const DEFAULT_NOISE_THRESHOLD = 0.05;

let dbPromise = null;

function openDb() {
  dbPromise ??= new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE, { keyPath: 'name' });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  return dbPromise;
}

async function transact(mode, operation) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const request = operation(db.transaction(STORE, mode).objectStore(STORE));
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/** Returns all baselines, newest first. */
export async function listBaselines() {
  const all = await transact('readonly', (store) => store.getAll());
  return all.sort((a, b) => b.savedAt.localeCompare(a.savedAt));
}

/**
 * Saves `baseline`, replacing any baseline with the same name.
 *
 * `baseline` is `{ name, savedAt, environment, config, results }`, where each result carries the
 * per-frame `samples` that comparisons need.
 */
export function putBaseline(baseline) {
  return transact('readwrite', (store) => store.put(baseline));
}

export function deleteBaseline(name) {
  return transact('readwrite', (store) => store.delete(name));
}

export async function renameBaseline(from, to) {
  const baseline = await transact('readonly', (store) => store.get(from));
  if (!baseline) {
    throw new Error(`no baseline named ${from}`);
  }
  await putBaseline({ ...baseline, name: to });
  await deleteBaseline(from);
}

/** Identifies the same scenario across runs. */
export function scenarioKey({ workload, renderer, cache }) {
  return `${workload}/${renderer}/${cache ? 'on' : 'off'}`;
}

/**
 * Compares per-frame samples of a baseline and a current run with Welch's t-test.
 *
 * Returns `null` when either side has fewer than two finite samples. `change` is the relative
 * change of the mean, and `verdict` is `'better'`, `'worse'`, or `'same'`, where a change only
 * counts if it is both significant and at least `noiseThreshold`.
 */
export function compareSamples(baseSamples, currentSamples, { lowerIsBetter = true, noiseThreshold = DEFAULT_NOISE_THRESHOLD } = {}) {
  const base = moments(baseSamples);
  const current = moments(currentSamples);
  if (!base || !current || base.mean === 0) {
    return null;
  }
  const change = (current.mean - base.mean) / base.mean;
  const p = welchP(base, current);
  const significant = p < SIGNIFICANCE_LEVEL && Math.abs(change) >= noiseThreshold;
  const improved = lowerIsBetter ? change < 0 : change > 0;
  return {
    change,
    p,
    baseMean: base.mean,
    verdict: significant ? (improved ? 'better' : 'worse') : 'same',
  };
}

/**
 * Compares two series measured in the same rounds with a paired t-test on per-round log ratios.
 *
 * Rounds that ran slower overall (throttling, background work) slow both sides alike, so pairing
 * cancels that drift where `compareSamples` would count it as noise. `change` is the geometric
 * mean of the per-round relative changes; the result has the same shape as `compareSamples`.
 */
export function comparePaired(baseRounds, currentRounds, { lowerIsBetter = true, noiseThreshold = DEFAULT_NOISE_THRESHOLD } = {}) {
  const pairs = (baseRounds ?? []).map((base, i) => [base, currentRounds?.[i]]).filter(([a, b]) => a > 0 && b > 0);
  const ratios = moments(pairs.map(([a, b]) => Math.log(b / a)));
  if (!ratios || pairs.length !== baseRounds.length) {
    return null;
  }
  const change = Math.exp(ratios.mean) - 1;
  const se = Math.sqrt(ratios.variance / ratios.n);
  const df = ratios.n - 1;
  const t = se === 0 ? (ratios.mean === 0 ? 0 : Infinity) : ratios.mean / se;
  const p = Number.isFinite(t) ? incompleteBeta(df / (df + t * t), df / 2, 0.5) : 0;
  const significant = p < SIGNIFICANCE_LEVEL && Math.abs(change) >= noiseThreshold;
  const improved = lowerIsBetter ? change < 0 : change > 0;
  return {
    change,
    p,
    baseMean: pairs.reduce((sum, [a]) => sum + a, 0) / pairs.length,
    verdict: significant ? (improved ? 'better' : 'worse') : 'same',
  };
}

function moments(samples) {
  const values = (samples ?? []).filter(Number.isFinite);
  const n = values.length;
  if (n < 2) {
    return null;
  }
  const mean = values.reduce((sum, value) => sum + value, 0) / n;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (n - 1);
  return { n, mean, variance };
}

function welchP(a, b) {
  const va = a.variance / a.n;
  const vb = b.variance / b.n;
  const se = Math.sqrt(va + vb);
  if (se === 0) {
    // Both runs are constant, which happens with coarse timers on trivial workloads.
    return a.mean === b.mean ? 1 : 0;
  }
  const t = (b.mean - a.mean) / se;
  const df = (va + vb) ** 2 / (va ** 2 / (a.n - 1) + vb ** 2 / (b.n - 1));
  return incompleteBeta(df / (df + t * t), df / 2, 0.5);
}

/** Regularized incomplete beta function I_x(a, b). */
function incompleteBeta(x, a, b) {
  if (x <= 0) {
    return 0;
  }
  if (x >= 1) {
    return 1;
  }
  const front = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  // The continued fraction converges quickly only on this side of the mean.
  return x < (a + 1) / (a + b + 2)
    ? (front * betaContinuedFraction(x, a, b)) / a
    : 1 - (front * betaContinuedFraction(1 - x, b, a)) / b;
}

function betaContinuedFraction(x, a, b) {
  const tiny = 1e-300;
  let c = 1;
  let d = 1 - ((a + b) * x) / (a + 1);
  d = 1 / (Math.abs(d) < tiny ? tiny : d);
  let result = d;
  for (let m = 1; m <= 200; m++) {
    for (const numerator of [
      (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m)),
      (-(a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1)),
    ]) {
      d = 1 + numerator * d;
      d = 1 / (Math.abs(d) < tiny ? tiny : d);
      c = 1 + numerator / c;
      c = Math.abs(c) < tiny ? tiny : c;
      result *= d * c;
    }
    if (Math.abs(d * c - 1) < 1e-12) {
      break;
    }
  }
  return result;
}

/** Lanczos approximation of ln Γ(x) for x > 0. */
function logGamma(x) {
  const g = [
    676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }
  const z = x - 1;
  let sum = 0.99999999999980993;
  g.forEach((coefficient, index) => { sum += coefficient / (z + index + 1); });
  const t = z + g.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(sum);
}
