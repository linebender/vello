<!-- Instructions

This changelog follows the patterns described here: <https://keepachangelog.com/en/>.

Subheadings to categorize changes are `added, changed, deprecated, removed, fixed, security`.

-->

# Changelog

The latest published vello_common release is [0.0.6](#006---2026-01-15) which was released on 2026-01-15.
You can find its changes [documented below](#006---2026-01-15).

## [Unreleased]

This release has an [MSRV][] of 1.92.

### Added

- Added image tinting support. ([#1460][] by [@grebmeg][])
- Introduce a `RenderState` abstraction. ([#1486][] by [@LaurenzV][])
- Added some helpers for rendering filters. ([#1446][] by [@LaurenzV][])
- Added `ImageCache` and `MultiAtlasManager`. ([#1457][] by [@grebmeg][])

### Changed

- Improved Bézier flattening performance by catching more Béziers whose chords are immediately within rendering tolerance. ([#1216][] by [@tomcur][])
- Removed the Process Row Incremental Macro and Perfect Bit. ([#1384][] by [@b0nes164][])
- Improved flattening and tiling performance by culling out-of-viewport Béziers before flattening. ([#1341][] by [@tomcur][])
- Removed the `tolerance` parameter from the flattening method. ([#1399][] by [@LaurenzV][])
- Significantly improved rendering performance of scenes including blend layers by ensuring no commands are generated for wide tiles without layer content. ([#1403][] by [@tomcur][])
- Replaced custom blend mode representation with peniko `Mix` for filter effects. ([#1410][] by [@LaurenzV][])
- Further improved performance of scenes including blend layers by only allocating layer scratch buffers for a wide tile upon draw commands being performed in that wide tile. ([#1414][] by [@tomcur][])
- Improved analytic anti-aliasing performance. ([#1426][], [#1442][] by [@tomcur][])
- Removed the command annotation logic. ([#1438][] by [@LaurenzV][])
- Added a fast path for rendering pixel-aligned rectangles. ([#1453][] by [@grebmeg][])
- Added a fast-path for skipping coarse rasterization and scheduling for scenes without layers. ([#1454][] by [@LaurenzV][])
- Added an optimization to not reset wide tiles when not needed. ([#1484][] by [@LaurenzV][])

### Fixed

- Fixed double-scaling bug with non-hinted COLR glyphs. ([#1370][] by [@conor-93][])
- Fixed bug in handling of `pop_clip` commands and properly reset wide tiles. ([#1443][] by [@LaurenzV][])
- Fixed analytic AA performance regressions from `fearless_simd`'s tightened `max_precise` and `min_precise` semantics in v0.4.0. ([#1463][] and [#1464][] by [@tomcur][])
- Fixed layer ranges not being updated when a wide tile is reset with a background color. ([#1478][] by [@LaurenzV][])

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
[@conor-93]: https://github.com/conor-93
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
[#1341]: https://github.com/linebender/vello/pull/1341
[#1349]: https://github.com/linebender/vello/pull/1349
[#1353]: https://github.com/linebender/vello/pull/1353
[#1354]: https://github.com/linebender/vello/pull/1354
[#1399]: https://github.com/linebender/vello/pull/1399
[#1403]: https://github.com/linebender/vello/pull/1403
[#1414]: https://github.com/linebender/vello/pull/1414
[#1370]: https://github.com/linebender/vello/pull/1370
[#1384]: https://github.com/linebender/vello/pull/1384
[#1410]: https://github.com/linebender/vello/pull/1410
[#1426]: https://github.com/linebender/vello/pull/1426
[#1438]: https://github.com/linebender/vello/pull/1438
[#1442]: https://github.com/linebender/vello/pull/1442
[#1443]: https://github.com/linebender/vello/pull/1443
[#1446]: https://github.com/linebender/vello/pull/1446
[#1453]: https://github.com/linebender/vello/pull/1453
[#1454]: https://github.com/linebender/vello/pull/1454
[#1457]: https://github.com/linebender/vello/pull/1457
[#1460]: https://github.com/linebender/vello/pull/1460
[#1463]: https://github.com/linebender/vello/pull/1463
[#1464]: https://github.com/linebender/vello/pull/1464
[#1478]: https://github.com/linebender/vello/pull/1478
[#1484]: https://github.com/linebender/vello/pull/1484
[#1486]: https://github.com/linebender/vello/pull/1486

[Unreleased]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.6...HEAD
[0.0.6]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.5...sparse-strips-v0.0.6
[0.0.5]: https://github.com/linebender/vello/compare/sparse-stips-v0.0.4...sparse-strips-v0.0.5
[0.0.4]: https://github.com/linebender/vello/compare/sparse-stips-v0.0.3...sparse-strips-v0.0.4
[0.0.3]: https://github.com/linebender/vello/compare/sparse-stips-v0.0.2...sparse-strips-v0.0.3
[0.0.2]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.1...sparse-stips-v0.0.2
[0.0.1]: https://github.com/linebender/vello/compare/ca6b1e4c7f5b0d95953c3b524f5d3952d5669c5a...sparse-strips-v0.0.1

[MSRV]: README.md#minimum-supported-rust-version-msrv
