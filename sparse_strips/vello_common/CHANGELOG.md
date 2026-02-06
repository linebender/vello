<!-- Instructions

This changelog follows the patterns described here: <https://keepachangelog.com/en/>.

Subheadings to categorize changes are `added, changed, deprecated, removed, fixed, security`.

-->

# Changelog

The latest published vello_common release is [0.0.6](#006---2026-01-15) which was released on 2026-01-15.
You can find its changes [documented below](#006---2026-01-15).

## [Unreleased]

This release has an [MSRV][] of 1.88.

### Changed

- Improved Bézier flattening performance by catching more Béziers whose chords are immediately within rendering tolerance. ([#1216][] by [@tomcur][])
- Significantly improved rendering performance of scenes including blend layers by ensuring no commands are generated for wide tiles without layer content. ([#1399][] by [@tomcur][])

### Fixed

- Fixed rendering of blurred rounded rectangles with zero or very small blur standard deviations. ([#1422][] by [@tomcur][])

## [0.0.6][] - 2026-01-15

This release has an [MSRV][] of 1.88.

### Changed

- Breaking change: Updated Peniko to [v0.6.0](https://github.com/linebender/peniko/releases/tag/v0.6.0). ([#1349][] by [@DJMcNab][])
  - This also updates Kurbo to [v0.13.0](https://github.com/linebender/kurbo/releases/tag/v0.13.0).
- Upgraded Skrifa to v0.40.0. ([#1353][] by [@waywardmonkeys][])
- Upgraded Hashbrown to v0.16.1. ([#1354][] by [@waywardmonkeys][])
- Perf: track has_opacities to skip alpha blending ([#1329][] by [@grebmeg][])

## [0.0.5][] - 2026-01-08

This release has an [MSRV][] of 1.88.

### Added

- A new module `clip` has been added allowing for the possibility to intersect two strips to create a new strip representing their intersection. ([#1203][] by [@LaurenzV])
- An `extend` method has been added to `StripStorage` to extends its alphas/strips from another `StripStorage`. ([#1203][] by [@LaurenzV])
- Added a `from_parts` method for masks. ([#1237][] by [@LaurenzV])
- Add initial support for image filters. ([#1286][] by [@grebmeg])

### Changed

- `WideTile::generate` now takes an additional `BlendMode` as a parameter. ([#1159][] by [@LaurenzV])
- `CmdFill` and `CmdAlphaFill` now store a `BlendMode` instead of `Option<BlendMode>`. ([#1159][] by [@LaurenzV])
- `Strip` now implements `PartialEq` and `Eq`. ([#1203][] by [@LaurenzV])
- `Strip` now has a `is_sentinel` method. ([#1203][] by [@LaurenzV])
- `StripStorage` now implements `PartialEq` and `Eq`. ([#1203][] by [@LaurenzV])
- The `generate_filled_path` method of `StripGenerator` now takes an optional clip path as input. ([#1203][] by [@LaurenzV])
- A new trait for approximate integer division by 255 has been added. ([#1203][] by [@LaurenzV])
- The `generate` method of `Wide` now takes an optional mask as an additional argument. ([#1237][] by [@LaurenzV])
- `CmdFill` and `CmdAlphaFill` now store an optional mask. ([#1237][] by [@LaurenzV])
- Performance improvements for gradient rendering. ([#1301][] by [@valadaptive])
- Various changes to the logic for computing tile intersections and representation of tiles. ([#1293][], [#1317][], [#1318][] by [@b0nes164])
- Support for computing data necessary to implement multi-sampled anti-aliasing. ([#1319][], by [@b0nes164])
- Numerous performance and memory-efficiency improvements. ([#1325][] by [@LaurenzV], [#1327][] by [@grebmeg], [#1336][] by [@tomcur], [#1338][] by [@taj-p])

See also the [vello_hybrid 0.0.5](../vello_hybrid/CHANGELOG.md#005---2026-01-08) and [vello_cpu 0.0.5](../vello_cpu/CHANGELOG.md#005---2026-01-08) releases.

## [0.0.4][] - 2025-10-17

This release has an [MSRV][] of 1.86.

No changelog was kept for this release.

See also the [vello_hybrid 0.0.4](../vello_hybrid/CHANGELOG.md#004---2025-10-17) and [vello_cpu 0.0.4](../vello_cpu/CHANGELOG.md#004---2025-10-17) releases.

## [0.0.3][] - 2025-10-04

This release has an [MSRV][] of 1.86.

No changelog was kept for this release.

See also the [vello_cpu 0.0.3](../vello_cpu/CHANGELOG.md#003---2025-10-04) release.

## [0.0.2][] - 2025-09-22

This release has an [MSRV][] of 1.85.

No changelog was kept for this release.

See also the [vello_cpu 0.0.2](../vello_cpu/CHANGELOG.md#002---2025-09-22) release.

## [0.0.1][] - 2025-05-10

This release has an [MSRV][] of 1.85.

This is the initial release. No changelog was kept for this release.

See also the [vello_cpu 0.0.1](../vello_cpu/CHANGELOG.md#001---2025-05-10) release.

[@b0nes164]: https://github.com/b0nes164
[@DJMcNab]: https://github.com/waywardmonkeys
[@grebmeg]: https://github.com/grebmeg
[@LaurenzV]: https://github.com/LaurenzV
[@taj-p]: https://github.com/taj-p
[@tomcur]: https://github.com/tomcur
[@valadaptive]: https://github.com/valadaptive
[@waywardmonkeys]: https://github.com/waywardmonkeys

[#1159]: https://github.com/linebender/vello/pull/1159
[#1203]: https://github.com/linebender/vello/pull/1203
[#1216]: https://github.com/linebender/vello/pull/1216
[#1237]: https://github.com/linebender/vello/pull/1237
[#1286]: https://github.com/linebender/vello/pull/1286
[#1293]: https://github.com/linebender/vello/pull/1293
[#1301]: https://github.com/linebender/vello/pull/1301
[#1317]: https://github.com/linebender/vello/pull/1317
[#1318]: https://github.com/linebender/vello/pull/1318
[#1319]: https://github.com/linebender/vello/pull/1319
[#1325]: https://github.com/linebender/vello/pull/1325
[#1327]: https://github.com/linebender/vello/pull/1327
[#1329]: https://github.com/linebender/vello/pull/1329
[#1336]: https://github.com/linebender/vello/pull/1327
[#1338]: https://github.com/linebender/vello/pull/1327
[#1349]: https://github.com/linebender/vello/pull/1349
[#1353]: https://github.com/linebender/vello/pull/1353
[#1354]: https://github.com/linebender/vello/pull/1354
[#1399]: https://github.com/linebender/vello/pull/1399
[#1422]: https://github.com/linebender/vello/pull/1422

[Unreleased]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.6...HEAD
[0.0.6]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.5...sparse-strips-v0.0.6
[0.0.5]: https://github.com/linebender/vello/compare/sparse-stips-v0.0.4...sparse-strips-v0.0.5
[0.0.4]: https://github.com/linebender/vello/compare/sparse-stips-v0.0.3...sparse-strips-v0.0.4
[0.0.3]: https://github.com/linebender/vello/compare/sparse-stips-v0.0.2...sparse-strips-v0.0.3
[0.0.2]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.1...sparse-stips-v0.0.2
[0.0.1]: https://github.com/linebender/vello/compare/ca6b1e4c7f5b0d95953c3b524f5d3952d5669c5a...sparse-strips-v0.0.1

[MSRV]: README.md#minimum-supported-rust-version-msrv
