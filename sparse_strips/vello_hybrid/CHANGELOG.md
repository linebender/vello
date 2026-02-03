<!-- Instructions

This changelog follows the patterns described here: <https://keepachangelog.com/en/>.

Subheadings to categorize changes are `added, changed, deprecated, removed, fixed, security`.

-->

# Changelog

The latest published vello_hybrid release is [0.0.6](#006---2026-01-15) which was released on 2026-01-15.
You can find its changes [documented below](#006---2026-01-15).

## [Unreleased]

This release has an [MSRV][] of 1.88.

### Fixed

- Fixed rendering artifacts sometimes present along the seam in sweep gradients by improving numerical robustness around the seam. ([#1352][] by [@tomcur][])

## [0.0.6][] - 2026-01-15

This release has an [MSRV][] of 1.88.

### Changed

- Breaking change: Updated Peniko to [v0.6.0](https://github.com/linebender/peniko/releases/tag/v0.6.0). ([#1349][] by [@DJMcNab][])
  - This also updates Kurbo to [v0.13.0](https://github.com/linebender/kurbo/releases/tag/v0.13.0).
- Upgraded Skrifa to v0.40.0. ([#1353][] by [@waywardmonkeys][])
- Upgraded Hashbrown to v0.16.1. ([#1354][] by [@waywardmonkeys][])

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
[@LaurenzV]: https://github.com/LaurenzV
[@waywardmonkeys]: https://github.com/waywardmonkeys

[#1203]: https://github.com/linebender/vello/pull/1203
[#1349]: https://github.com/linebender/vello/pull/1349
[#1352]: https://github.com/linebender/vello/pull/1352
[#1353]: https://github.com/linebender/vello/pull/1353
[#1354]: https://github.com/linebender/vello/pull/1354

[Unreleased]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.6...HEAD
[0.0.6]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.5...sparse-strips-v0.0.6
[0.0.5]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.4...sparse-strips-v0.0.5
[0.0.4]: https://github.com/linebender/vello/compare/ca6b1e4c7f5b0d95953c3b524f5d3952d5669c5a...sparse-strips-v0.0.4

[MSRV]: README.md#minimum-supported-rust-version-msrv
