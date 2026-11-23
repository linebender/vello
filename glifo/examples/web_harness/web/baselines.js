// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Saved benchmark runs (baselines) in IndexedDB, and the statistics used to compare against them.

const DB_NAME = 'glifo-harness';
const STORE = 'baselines';

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
 * counts if it is both significant at the 5% level (`|t| >= tCritical`) and at least
 * `noiseThreshold`.
 */
export function compareSamples(baseSamples, currentSamples, { lowerIsBetter = true, noiseThreshold = DEFAULT_NOISE_THRESHOLD } = {}) {
  const base = moments(baseSamples);
  const current = moments(currentSamples);
  if (!base || !current || base.mean === 0) {
    return null;
  }
  const va = base.variance / base.n;
  const vb = current.variance / current.n;
  const df = (va + vb) ** 2 / (va ** 2 / (base.n - 1) + vb ** 2 / (current.n - 1));
  const change = (current.mean - base.mean) / base.mean;
  return verdict(change, tStatistic(current.mean - base.mean, Math.sqrt(va + vb)), df, base.mean, { lowerIsBetter, noiseThreshold });
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
  const baseMean = pairs.reduce((sum, [a]) => sum + a, 0) / pairs.length;
  const t = tStatistic(ratios.mean, Math.sqrt(ratios.variance / ratios.n));
  return verdict(Math.exp(ratios.mean) - 1, t, ratios.n - 1, baseMean, { lowerIsBetter, noiseThreshold });
}

function verdict(change, t, df, baseMean, { lowerIsBetter, noiseThreshold }) {
  const tCritical = criticalT(df);
  const significant = Math.abs(t) >= tCritical && Math.abs(change) >= noiseThreshold;
  const improved = lowerIsBetter ? change < 0 : change > 0;
  return { change, t, tCritical, baseMean, verdict: significant ? (improved ? 'better' : 'worse') : 'same' };
}

function tStatistic(difference, standardError) {
  // A zero standard error means both sides are constant, as with coarse timers on trivial workloads.
  return standardError === 0 ? (difference === 0 ? 0 : Infinity) : difference / standardError;
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

/** Two-sided 5% critical values of Student's t for 1 to 30 degrees of freedom. */
const T_CRITICAL = [
  12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
  2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
  2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
];

/**
 * Returns the two-sided 5% critical value of Student's t for `df` degrees of freedom.
 *
 * Fractional `df` (from Welch's test) interpolates linearly between table entries, which slightly
 * overestimates the critical value, so borderline changes err toward insignificant.
 */
function criticalT(df) {
  if (!(df >= 1)) {
    return Infinity;
  }
  if (df < 30) {
    const i = Math.floor(df) - 1;
    return T_CRITICAL[i] + (T_CRITICAL[i + 1] - T_CRITICAL[i]) * (df - i - 1);
  }
  if (df < 40) {
    return T_CRITICAL[29];
  }
  return df >= 120 ? 1.98 : df >= 60 ? 2.0 : 2.021;
}
