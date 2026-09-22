// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Renders each scene with both backends and compares the results.

use crate::compare::{Mismatch, compare_images, write_diff_report};
use crate::config::{CONTINUE_ARTIFACT_PREFIX, DEDUP, TOLERANCE};
use crate::images::ImageTable;
use crate::replay::replay_scene;
use crate::scene::{FuzzScene, HEIGHT, WIDTH};
use crate::snapshot_test;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::panic::{AssertUnwindSafe, PanicHookInfo};
use std::path::{Path, PathBuf};
use vello_common::pixmap::Pixmap;
use vello_cpu::{Level, RenderMode};
use vello_tests::renderer::{CpuRenderer, GpuRenderer, Renderer};

/// A panic raised by one backend while rendering a scene.
struct Crash {
    backend: &'static str,
    message: String,
}

impl Crash {
    /// Fingerprint of the backend and panic location (the hook's first line, `panicked at
    /// file:line:col:`), so every input tripping the same assertion shares it.
    fn signature(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.backend.hash(&mut hasher);
        self.message.lines().next().unwrap_or("").hash(&mut hasher);
        hasher.finish()
    }
}

/// Paths of a newly saved finding and its sequence number in this run.
struct Recorded {
    number: usize,
    artifact: PathBuf,
    test_path: PathBuf,
}

thread_local! {
    /// Set while a backend renders in continue mode, so the panic hook records the panic instead
    /// of aborting the process.
    static CAPTURING_PANICS: Cell<bool> = const { Cell::new(false) };
    static CAPTURED_PANIC: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Replaces the `libfuzzer-sys` panic hook, which aborts before unwinding can reach a
/// `catch_unwind`. Panics outside a captured render (including on other threads) still go to the
/// previous hook and abort as before.
fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info: &PanicHookInfo<'_>| {
        if CAPTURING_PANICS.get() {
            CAPTURED_PANIC.set(Some(info.to_string()));
        } else {
            previous(info);
        }
    }));
}

/// Runs `f`, converting a panic into its message. Only active in continue mode; otherwise panics
/// propagate untouched so libFuzzer records them as usual.
fn catch_panics<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    if CONTINUE_ARTIFACT_PREFIX.is_none() {
        return Ok(f());
    }
    CAPTURING_PANICS.set(true);
    let result = std::panic::catch_unwind(AssertUnwindSafe(f));
    CAPTURING_PANICS.set(false);
    result.map_err(|payload| {
        CAPTURED_PANIC.take().unwrap_or_else(|| {
            payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic with non-string payload".to_owned())
        })
    })
}

/// A renderer together with the image sources registered on it. Registration is not undone by
/// `reset`, so it happens once per renderer instance.
struct Backend<R> {
    renderer: R,
    images: ImageTable,
}

impl<R: Renderer> Backend<R> {
    fn new(mut renderer: R) -> Self {
        let images = ImageTable::register(&mut renderer);
        Self { renderer, images }
    }

    fn render(&mut self, scene: &FuzzScene) -> Pixmap {
        self.renderer.reset();
        replay_scene(scene, &mut self.renderer, &self.images);
        self.renderer.flush();
        self.renderer.render();
        self.renderer.snapshot()
    }
}

fn new_cpu_backend() -> Backend<CpuRenderer> {
    Backend::new(CpuRenderer::new(
        WIDTH,
        HEIGHT,
        0,
        Level::fallback(),
        RenderMode::OptimizeQuality,
    ))
}

fn new_gpu_backend() -> Backend<GpuRenderer> {
    Backend::new(GpuRenderer::new(
        WIDTH,
        HEIGHT,
        0,
        Level::fallback(),
        RenderMode::OptimizeSpeed,
    ))
}

/// Holds both renderers for the lifetime of the fuzzing process; creating a GPU device per
/// iteration would dominate the run time.
pub(crate) struct DifferentialHarness {
    cpu: Backend<CpuRenderer>,
    gpu: Backend<GpuRenderer>,
    /// Hashes of inputs already recorded in continue mode, so replays are not saved twice.
    seen_findings: HashSet<u64>,
    /// Signature of each saved finding mapped to its number, for `--dedup`.
    seen_signatures: HashMap<u64, usize>,
}

impl DifferentialHarness {
    pub(crate) fn new() -> Self {
        if CONTINUE_ARTIFACT_PREFIX.is_some() {
            install_panic_hook();
        }
        Self {
            seen_findings: HashSet::new(),
            seen_signatures: HashMap::new(),
            cpu: new_cpu_backend(),
            gpu: new_gpu_backend(),
        }
    }

    /// Renders with both backends. In continue mode a panicking backend is reported as a
    /// [`Crash`] and rebuilt, since its state (and, for the GPU, the render mutex it holds) cannot
    /// be trusted afterwards.
    fn render(&mut self, scene: &FuzzScene) -> Result<(Pixmap, Pixmap), Crash> {
        let cpu_image = match catch_panics(|| self.cpu.render(scene)) {
            Ok(image) => image,
            Err(message) => {
                self.cpu = new_cpu_backend();
                return Err(Crash {
                    backend: "CPU",
                    message,
                });
            }
        };
        let gpu_image = match catch_panics(|| self.gpu.render(scene)) {
            Ok(image) => image,
            Err(message) => {
                self.gpu = new_gpu_backend();
                return Err(Crash {
                    backend: "GPU",
                    message,
                });
            }
        };
        Ok((cpu_image, gpu_image))
    }

    /// Panics on a mismatch so libFuzzer records the input, unless continue mode is enabled.
    pub(crate) fn run(&mut self, data: &[u8], scene: &FuzzScene) {
        let (cpu_image, gpu_image) = match self.render(scene) {
            Ok(images) => images,
            Err(crash) => {
                // `render` only returns a crash in continue mode.
                let artifact_prefix = CONTINUE_ARTIFACT_PREFIX
                    .as_ref()
                    .expect("crashes are only caught in continue mode");
                self.record_crash(data, scene, &crash, artifact_prefix);
                return;
            }
        };
        let Err(mismatch) = compare_images(&cpu_image, &gpu_image) else {
            return;
        };
        let Some(artifact_prefix) = CONTINUE_ARTIFACT_PREFIX.as_ref() else {
            panic!("{mismatch}");
        };
        self.record_mismatch(
            data,
            scene,
            &mismatch,
            cpu_image,
            gpu_image,
            artifact_prefix,
        );
    }

    /// Saves the input as `<prefix><kind>-<hash>` plus its `.rs` snapshot test. Returns `None`
    /// when this exact input was recorded before, or when `--dedup` is set and a finding with
    /// the same signature was already saved (printed as a duplicate of that finding).
    fn record_input(
        &mut self,
        data: &[u8],
        scene: &FuzzScene,
        kind: &str,
        signature: u64,
        artifact_prefix: &OsStr,
    ) -> Option<Recorded> {
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        let hash = hasher.finish();
        if self.seen_findings.contains(&hash) {
            return None;
        }
        if *DEDUP && let Some(original) = self.seen_signatures.get(&signature) {
            println!(
                "{}: duplicate of #{original}, not saved",
                kind.to_uppercase()
            );
            return None;
        }
        self.seen_findings.insert(hash);
        let number = self.seen_findings.len();
        self.seen_signatures.entry(signature).or_insert(number);

        let mut artifact = artifact_prefix.to_owned();
        artifact.push(format!("{kind}-{hash:016x}"));
        let artifact = PathBuf::from(artifact);
        if let Some(parent) = artifact.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(&artifact, data)
            .unwrap_or_else(|error| panic!("failed to write {}: {error}", artifact.display()));
        let test_path = with_extension(&artifact, ".rs");
        snapshot_test::write_to(scene, &test_path);
        Some(Recorded {
            number,
            artifact,
            test_path,
        })
    }

    /// Saves the input, its diff, and a snapshot test next to libFuzzer's own artifacts, instead
    /// of aborting.
    fn record_mismatch(
        &mut self,
        data: &[u8],
        scene: &FuzzScene,
        mismatch: &Mismatch,
        cpu_image: Pixmap,
        gpu_image: Pixmap,
        artifact_prefix: &OsStr,
    ) {
        let Some(recorded) = self.record_input(
            data,
            scene,
            "mismatch",
            mismatch.signature(),
            artifact_prefix,
        ) else {
            return;
        };
        let (image_path, json_path, _) =
            write_diff_report(cpu_image, gpu_image, &recorded.artifact);

        println!(
            "MISMATCH #{}: {mismatch}\n  input: {}\n  diff image: {}\n  diff report: {}\n  \
             snapshot test: {}\n  reproduce: cargo +nightly fuzz run cpu_gpu_differential {}",
            recorded.number,
            recorded.artifact.display(),
            image_path.display(),
            json_path.display(),
            recorded.test_path.display(),
            recorded.artifact.display()
        );
    }

    /// Saves the input, the panic message, and a snapshot test next to libFuzzer's own artifacts,
    /// instead of aborting.
    fn record_crash(
        &mut self,
        data: &[u8],
        scene: &FuzzScene,
        crash: &Crash,
        artifact_prefix: &OsStr,
    ) {
        let Some(recorded) =
            self.record_input(data, scene, "crash", crash.signature(), artifact_prefix)
        else {
            return;
        };
        let message_path = with_extension(&recorded.artifact, ".txt");
        let _ = std::fs::write(
            &message_path,
            format!("{} renderer panicked\n{}\n", crash.backend, crash.message),
        );

        println!(
            "CRASH #{}: {} renderer {}\n  input: {}\n  panic message: {}\n  snapshot test: {}\n  \
             reproduce: cargo +nightly fuzz run cpu_gpu_differential {}",
            recorded.number,
            crash.backend,
            crash.message.replace('\n', " "),
            recorded.artifact.display(),
            message_path.display(),
            recorded.test_path.display(),
            recorded.artifact.display()
        );
    }

    /// Writes a `cpu | diff | gpu` image and a per-pixel JSON report in the same format as the
    /// snapshot suite, using the fuzz target's tolerances.
    pub(crate) fn write_diff(&mut self, scene: &FuzzScene, stem: &Path) {
        let (cpu_image, gpu_image) = self
            .render(scene)
            .unwrap_or_else(|crash| panic!("{} renderer {}", crash.backend, crash.message));
        let (image_path, json_path, report) = write_diff_report(cpu_image, gpu_image, stem);
        println!(
            "CPU/GPU diff: {} pixels differ by more than {} (allowed: {}); max difference per \
             channel {:?}\n  diff image: {}\n  diff report: {}",
            report.pixel_count,
            TOLERANCE.channel,
            TOLERANCE.max_outlier_pixels,
            report.max_difference,
            image_path.display(),
            json_path.display()
        );
    }
}

/// Appends `extension` instead of replacing anything after a dot in the artifact name.
fn with_extension(artifact: &Path, extension: &str) -> PathBuf {
    let mut path = artifact.as_os_str().to_owned();
    path.push(extension);
    PathBuf::from(path)
}
