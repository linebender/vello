// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Shared helpers for the Glifo harness pages.

import init, { Harness } from './pkg/glifo_web_harness.js';

export { Harness };

let readyPromise = null;
/** Milliseconds spent fetching, compiling, and instantiating the wasm module. */
export let moduleInitMs = NaN;

export function ready() {
  readyPromise ??= (async () => {
    const start = performance.now();
    await init();
    moduleInitMs = performance.now() - start;
  })();
  return readyPromise;
}

export function workloads() {
  return Harness.workloads().map(([name, description]) => ({ name, description }));
}

/** Creates a harness for `renderer` ('gpu' or 'cpu'), measuring wall-clock creation time. */
export async function createHarness(renderer, canvas, { atlasSize = 4096 } = {}) {
  await ready();
  const start = performance.now();
  const harness = renderer === 'gpu'
    ? await Harness.createGpu(canvas, atlasSize)
    : await Harness.createCpu(canvas);
  return { harness, createMs: performance.now() - start };
}

/**
 * Releases a harness and its canvas context.
 *
 * After a Rust panic the harness stays borrowed and every method throws, so failures here are
 * ignored and the WebGL context is released through the canvas instead.
 */
export function disposeHarness(harness, canvas) {
  try {
    harness?.destroy();
    harness?.free();
  } catch {
    canvas?.getContext('webgl2')?.getExtension('WEBGL_lose_context')?.loseContext();
  }
}

/** Formats an error thrown by the harness, pointing at the console for Rust panics. */
export function describeError(error) {
  const text = String(error);
  return text.includes('unreachable') ? `${text} (Rust panic; see the browser console for the message)` : text;
}

/** Resolves on the next task, without the ~4 ms clamp of nested `setTimeout`. */
export function nextTask() {
  return new Promise((resolve) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = () => resolve();
    channel.port2.postMessage(null);
  });
}

// ---------------------------------------------------------------------------------------------
// Statistics

export function summarize(values) {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) {
    return { n: 0, mean: NaN, p50: NaN, p95: NaN, min: NaN, max: NaN };
  }
  const sorted = [...finite].sort((a, b) => a - b);
  const at = (q) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  const mean = finite.reduce((sum, value) => sum + value, 0) / finite.length;
  return { n: finite.length, mean, p50: at(0.5), p95: at(0.95), min: sorted[0], max: sorted.at(-1) };
}

export function median(values) {
  return summarize(values).p50;
}

/** Fixed-capacity history of samples. */
export class Ring {
  constructor(capacity) {
    this.capacity = capacity;
    this.values = [];
  }

  push(value) {
    this.values.push(value);
    if (this.values.length > this.capacity) {
      this.values.shift();
    }
  }

  clear() {
    this.values.length = 0;
  }

  summary(select = (value) => value) {
    return summarize(this.values.map(select));
  }
}

/** Copies the plain fields of a wasm-bindgen stats object so it survives `free()`. */
export function plainStats(stats) {
  const out = {};
  for (const key of [
    'frame', 'glyphs', 'encode_ms', 'render_ms', 'present_ms', 'sync_ms', 'gpu_ms', 'total_ms',
  ]) {
    out[key] = stats[key];
  }
  stats.free();
  return out;
}

export function plainInit(stats) {
  const out = {
    context_ms: stats.context_ms,
    compile_ms: stats.compile_ms,
    shader_count: stats.shader_count,
    poll_count: stats.poll_count,
    total_ms: stats.total_ms,
  };
  stats.free();
  return out;
}

// ---------------------------------------------------------------------------------------------
// Formatting

export function ms(value, digits = 2) {
  return Number.isFinite(value) ? `${value.toFixed(digits)} ms` : '–';
}

export function num(value, digits = 0) {
  if (!Number.isFinite(value)) {
    return '–';
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

export function pct(value, digits = 1) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : '–';
}

export function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (key === 'class') {
      node.className = value;
    } else if (key.startsWith('on')) {
      node.addEventListener(key.slice(2), value);
    } else if (value !== false && value != null) {
      node.setAttribute(key, value === true ? '' : value);
    }
  }
  node.append(...children.flat().filter((child) => child != null));
  return node;
}

export function download(filename, text, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([text], { type }));
  el('a', { href: url, download: filename }).click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------------------------------------------------------------------------------------------
// Stacked frame-time chart

export class FrameChart {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {{key: string, color: string, label: string}[]} series stacked bottom-up
   */
  constructor(canvas, series, capacity = 240) {
    this.canvas = canvas;
    this.series = series;
    this.history = new Ring(capacity);
  }

  push(stats) {
    this.history.push(stats);
  }

  clear() {
    this.history.clear();
  }

  draw() {
    const { canvas } = this;
    const dpr = window.devicePixelRatio || 1;
    const width = Math.round(canvas.clientWidth * dpr);
    const height = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);
    const values = this.history.values;
    if (values.length === 0) {
      return;
    }
    const totals = values.map((stats) => this.series.reduce((sum, s) => sum + (Number.isFinite(stats[s.key]) ? stats[s.key] : 0), 0));
    const peak = Math.max(16.7, ...totals) * 1.1;
    const barWidth = width / this.history.capacity;
    values.forEach((stats, index) => {
      let y = height;
      const x = index * barWidth;
      for (const s of this.series) {
        const value = Number.isFinite(stats[s.key]) ? stats[s.key] : 0;
        const barHeight = (value / peak) * height;
        ctx.fillStyle = s.color;
        ctx.fillRect(x, y - barHeight, Math.max(1, barWidth - 0.5), barHeight);
        y -= barHeight;
      }
    });
    ctx.font = `${10 * dpr}px ui-monospace, monospace`;
    for (const budget of [8.3, 16.7, 33.3]) {
      if (budget > peak) {
        continue;
      }
      const y = height - (budget / peak) * height;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
      ctx.fillStyle = 'rgba(255, 255, 255, 0.45)';
      ctx.fillText(`${budget} ms`, 4 * dpr, y - 3 * dpr);
    }
  }

  legend() {
    return el('div', { class: 'legend' },
      this.series.map((s) => el('span', {}, el('i', { style: `background:${s.color}` }), s.label)));
  }
}

/** Builds the navigation header shared by all pages. */
export function header(active) {
  const links = [
    ['index.html', 'Overview'],
    ['live.html', 'Live'],
    ['bench.html', 'Benchmark'],
  ];
  return el('header', {},
    el('span', { class: 'brand' }, 'Glifo harness'),
    el('nav', {}, links.map(([href, label]) => el('a', { href, class: href === active ? 'active' : null }, label))));
}
