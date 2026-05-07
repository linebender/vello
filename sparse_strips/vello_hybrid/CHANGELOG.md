<!-- Instructions

This changelog follows the patterns described here: <https://keepachangelog.com/en/>.

Subheadings to categorize changes are `added, changed, deprecated, removed, fixed, security`.

-->

# Changelog

The latest published vello_hybrid release is [0.0.7](#007---2026-03-24) which was released on 2026-03-24.
You can find its changes [documented below](#007---2026-03-24).

## [Unreleased]

This release has an [MSRV][] of 1.88.

### Removed

- Support for recordings. This decision was made due to a number of downsides that
  came with the implementation. See the corresponding PR and Zulip thread for more information. ([#1611][] by [@LaurenzV][])

## [0.0.7][] - 2026-03-24

This release has an [MSRV][] of 1.92.

### Added

- Initial support for filter effects. ([#1494][] by [@LaurenzV][])
- `render_to_atlas` and `write_to_atlas` APIs for glyph caching. ([#1458][] by [@grebmeg][])
- `push_blend_layer`, `push_opacity_layer` and `push_mask_layer` methods to `Scene`. ([#1420][] by [@grebmeg][])
- Scene constraints for conditionally achieving better rendering performance. ([#1476][] by [@taj-p][])
- Support for image tinting. ([#1460][] by [@grebmeg][])

### Changed

- Updated `wgpu` to v28. ([#1492][] by [@xStrom][])
- Made text rendering an optional feature. ([#1455][] by [@LaurenzV][])

### Fixed

- Rendering artifacts sometimes present along the seam in sweep gradients by improving numerical robustness around the seam. ([#1352][] by [@tomcur][])
- Incorrect handling of scenes with many complex paints. ([#1467][] by [@taj-p][])

### Optimized

- Rendering axis-aligned rectangles. ([#1482][] by [@LaurenzV][])
- Rendering of opaque full-tile images. ([#1461][] by [@grebmeg][])
- Layer blending for src-over compositing. ([#1436][] by [@LaurenzV][])
- Coarse rasterization and scheduling of scenes without layers. ([#1454][] by [@LaurenzV][])
- Scenes with many changing gradients. ([#1496][] by [@LaurenzV][])

## [0.0.6][] - 2026-01-15

This release has an [MSRV][] of 1.88.

### Changed

- Breaking change: Updated Peniko to [v0.6.0](https://github.com/linebender/peniko/releases/tag/v0.6.0). ([#1349][] by [@DJMcNab][])
  - This also updates Kurbo to [v0.13.0](https://github.com/linebender/kurbo/releases/tag/v0.13.0).
- Upgraded Skrifa to v0.40.0. ([#1353][] by [@waywardmonkeys][])
- Upgraded Hashbrown to v0.16.1. ([#1354][] by [@waywardmonkeys][])

See also the [vello_cpu 0.0.6](../vello_cpu/CHANGELOG.md#006---2026-01-15) and [vello_common 0.0.6](../vello_common/CHANGELOG.md#006---2026-01-15) releases.

## [0.0.5][] - 2026-01-08

This release has an [MSRV][] of 1.88.

### Added

- The `Scene` now contains a `push_clip_path` and `pop_clip_path` method for performing non-isolated clipping. ([#1203][] by [@LaurenzV])

See also the [vello_cpu 0.0.5](../vello_cpu/CHANGELOG.md#005---2026-01-08) and [vello_common 0.0.5](../vello_common/CHANGELOG.md#005---2026-01-08) releases.

## [0.0.4][] - 2025-10-17

This release has an [MSRV][] of 1.86.

This is the initial release. No changelog was kept for this release.

See also the [vello_cpu 0.0.4](../vello_cpu/CHANGELOG.md#004---2025-10-17) and [vello_common 0.0.4](../vello_common/CHANGELOG.md#004---2025-10-17) releases.

[@DJMcNab]: https://github.com/DJMcNab
[@grebmeg]: https://github.com/grebmeg
[@LaurenzV]: https://github.com/LaurenzV
[@taj-p]: https://github.com/taj-p
[@tomcur]: https://github.com/tomcur
[@waywardmonkeys]: https://github.com/waywardmonkeys
[@xStrom]: https://github.com/xStrom

[#1203]: https://github.com/linebender/vello/pull/1203
[#1349]: https://github.com/linebender/vello/pull/1349
[#1352]: https://github.com/linebender/vello/pull/1352
[#1353]: https://github.com/linebender/vello/pull/1353
[#1354]: https://github.com/linebender/vello/pull/1354
[#1420]: https://github.com/linebender/vello/pull/1420
[#1436]: https://github.com/linebender/vello/pull/1436
[#1454]: https://github.com/linebender/vello/pull/1454
[#1455]: https://github.com/linebender/vello/pull/1455
[#1458]: https://github.com/linebender/vello/pull/1458
[#1460]: https://github.com/linebender/vello/pull/1460
[#1461]: https://github.com/linebender/vello/pull/1461
[#1467]: https://github.com/linebender/vello/pull/1467
[#1476]: https://github.com/linebender/vello/pull/1476
[#1482]: https://github.com/linebender/vello/pull/1482
[#1492]: https://github.com/linebender/vello/pull/1492
[#1494]: https://github.com/linebender/vello/pull/1494
[#1496]: https://github.com/linebender/vello/pull/1496
[#1611]: https://github.com/linebender/vello/pull/1611

[Unreleased]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.7...HEAD
[0.0.7]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.6...sparse-strips-v0.0.7
[0.0.6]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.5...sparse-strips-v0.0.6
[0.0.5]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.4...sparse-strips-v0.0.5
[0.0.4]: https://github.com/linebender/vello/compare/ca6b1e4c7f5b0d95953c3b524f5d3952d5669c5a...sparse-strips-v0.0.4

[MSRV]: README.md#minimum-supported-rust-version-msrv
