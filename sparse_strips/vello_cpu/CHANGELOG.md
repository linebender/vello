<!-- Instructions

This changelog follows the patterns described here: <https://keepachangelog.com/en/>.

Subheadings to categorize changes are `added, changed, deprecated, removed, fixed, security`.

-->

# Changelog

The latest published vello_cpu release is [0.0.6](#006---2026-01-15) which was released on 2026-01-15.
You can find its changes [documented below](#006---2026-01-15).

## [Unreleased]

This release has an [MSRV][] of 1.88.

### Added

- Added `composite_to_pixmap_at_offset` method to `RenderContext` for compositing at specific offsets within a larger pixmap, enabling spritesheet/atlas support. ([#1416][] by [@grebmeg][])

### Changed

- Improve performance of bilinear image sampling in the `RenderMode::OptimizeQuality` (`f32`) pipeline. ([#1343][] by [@tomcur][])

### Fixed

- Fixed rendering of blurred rounded rectangles with zero or very small blur standard deviations. ([#1422][] by [@tomcur][])

## [0.0.6][] - 2026-01-15

This release has an [MSRV][] of 1.88.

### Added

- Support for the "offset" filter has been added ([#1351] by [@waywardmonkeys])

### Changed

- Breaking change: Updated Peniko to [v0.6.0](https://github.com/linebender/peniko/releases/tag/v0.6.0). ([#1349][] by [@DJMcNab][])
  - This also updates Kurbo to [v0.13.0](https://github.com/linebender/kurbo/releases/tag/v0.13.0).
- Upgraded Skrifa to v0.40.0. ([#1353][] by [@waywardmonkeys][])
- Upgraded Hashbrown to v0.16.1. ([#1354][] by [@waywardmonkeys][])
- Perf: optimize image rendering for axis-aligned images ([#1335][] by [@grebmeg][])

## [0.0.5][] - 2026-01-08

This release has an [MSRV][] of 1.88.

### Added

- The `RenderContext` now has a `set_blend_mode` (and a corresponding `blend_mode`  getter method) that can be used to support non-isolated blending. ([#1159][] by [@LaurenzV])
- The `RenderContext` now contains a `push_clip_path` and `pop_clip_path` method for performing non-isolated clipping. ([#1203][] by [@LaurenzV])
- Experimental support for image filter effects: ([#1286][] by [@grebmeg][])
  - New filter API methods on `RenderContext`:
    - `set_filter_effect()` - Set a filter to be applied to subsequent drawing operations.
    - `push_filter_layer()` - Create a new layer with a filter effect.
  - `FilterEffect` trait providing both u8 and f32 precision variants for rendering across different backends.
  - Gaussian Blur filter with configurable standard deviation and edge modes (`None`, `Mirror`, `Wrap`, `Duplicate`).
    Uses an optimized decimated blur algorithm with automatic downsampling for performance.
  - Drop Shadow filter with customizable offset, blur radius, and shadow color.
  - Flood filter for solid color fills.
- Added a `set_mask` method to make it possible to mask rendered paths without inducing layer isolation. ([#1237][] by [@LaurenzV])
- Added support for conditionally disabling the u8 or f32 pipeline. ([#1294][] by [@nicoburns])
- Improve performance of rendering opaque images. ([#1327][] by [@grebmeg])

### Known Limitations

- Filter effects currently support only single-primitive filters; filter graphs with multiple chained primitives are not yet supported.
- Multithreaded rendering is not supported for filter effects; filters are only applied in single-threaded mode.

See also the [vello_hybrid 0.0.5](../vello_hybrid/CHANGELOG.md#005---2026-01-08) and [vello_common 0.0.5](../vello_common/CHANGELOG.md#005---2026-01-08) releases.

## [0.0.4][] - 2025-10-17

This release has an [MSRV][] of 1.86.

No changelog was kept for this release.

See also the [vello_hybrid 0.0.4](../vello_hybrid/CHANGELOG.md#004---2025-10-17) and [vello_common 0.0.4](../vello_common/CHANGELOG.md#004---2025-10-17) releases.

## [0.0.3][] - 2025-10-04

This release has an [MSRV][] of 1.86.

No changelog was kept for this release.

See also the [vello_common 0.0.3](../vello_common/CHANGELOG.md#003---2025-10-04) release.

## [0.0.2][] - 2025-09-22

This release has an [MSRV][] of 1.85.

No changelog was kept for this release.

See also the [vello_common 0.0.2](../vello_common/CHANGELOG.md#002---2025-09-22) release.

## [0.0.1][] - 2025-05-10

This release has an [MSRV][] of 1.85.

This is the initial release. No changelog was kept for this release.

See also the [vello_common 0.0.1](../vello_common/CHANGELOG.md#001---2025-05-10) release.

[@DJMcNab]: https://github.com/DJMcNab
[@grebmeg]: https://github.com/grebmeg
[@LaurenzV]: https://github.com/LaurenzV
[@nicoburns]: https://github.com/nicoburns
[@tomcur]: https://github.com/tomcur
[@waywardmonkeys]: https://github.com/waywardmonkeys

[#1159]: https://github.com/linebender/vello/pull/1159
[#1203]: https://github.com/linebender/vello/pull/1203
[#1237]: https://github.com/linebender/vello/pull/1237
[#1286]: https://github.com/linebender/vello/pull/1286
[#1294]: https://github.com/linebender/vello/pull/1294
[#1327]: https://github.com/linebender/vello/pull/1327
[#1335]: https://github.com/linebender/vello/pull/1335
[#1416]: https://github.com/linebender/vello/pull/1416
[#1343]: https://github.com/linebender/vello/pull/1343
[#1349]: https://github.com/linebender/vello/pull/1349
[#1351]: https://github.com/linebender/vello/pull/1351
[#1353]: https://github.com/linebender/vello/pull/1353
[#1354]: https://github.com/linebender/vello/pull/1354
[#1422]: https://github.com/linebender/vello/pull/1422

[Unreleased]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.6...HEAD
[0.0.6]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.5...sparse-strips-v0.0.6
[0.0.5]: https://github.com/linebender/vello/compare/sparse-stips-v0.0.4...sparse-strips-v0.0.5
[0.0.4]: https://github.com/linebender/vello/compare/sparse-stips-v0.0.3...sparse-strips-v0.0.4
[0.0.3]: https://github.com/linebender/vello/compare/sparse-stips-v0.0.2...sparse-strips-v0.0.3
[0.0.2]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.1...sparse-stips-v0.0.2
[0.0.1]: https://github.com/linebender/vello/compare/ca6b1e4c7f5b0d95953c3b524f5d3952d5669c5a...sparse-strips-v0.0.1

[MSRV]: README.md#minimum-supported-rust-version-msrv
