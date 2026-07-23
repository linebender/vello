<!-- Instructions

This changelog follows the patterns described here: <https://keepachangelog.com/en/>.

Subheadings to categorize changes are `added, changed, deprecated, removed, fixed, security`.

-->

# Changelog

## [Unreleased]

This release has an [MSRV][] of 1.88.

### Added

- `StripGenerator::set_cull_hint`: restrict strip generation to a bounding box; in-hint pixels are bit-identical. ([#1737][] by [@AdrianEddy][])
- `ClipContext` tracks clip stacks that form disjoint integer-rectangle sets (`is_int_rect_clip`, `effective_int_rect_set`, `path_as_integer_rect_set`), so consumers can clamp content to the set instead of intersecting with the clip mask. ([#1737][] by [@AdrianEddy][])

## [0.0.9][] - 2026-05-30

This release has an [MSRV][] of 1.88.

### Fixed

- Rendering of paths with high winding counts. ([#1673][] by [@b0nes164][])

### Optimized

- Tiling of line segments fully contained in the viewport. ([#1635][] by [@LaurenzV][])

## [0.0.8][] - 2026-05-15

This release has an [MSRV][] of 1.88.

### Added

- `TextureId`, a handle for user-provided textures resolved at render-time, and `EncodedExternalTexture` for encoding draws that sample from them. ([#1552][] by [@tomcur][])
- `RectU16`, a shared axis-aligned rectangle type for `u16` coordinate bounding boxes. ([#1559][] by [@tomcur][])

### Changed

- `WideTilesBbox` now uses `RectU16` internally, its `bbox` field is private, and `WideTilesBbox::new` now takes four `u16` coordinates instead of a `[u16; 4]`. ([#1559][] by [@tomcur][])
- Renamed image transparency hint APIs from `may_have_opacities` and `*_opacity_hint` to `may_have_transparency` and `*_transparency_hint`. ([#1613][] by [@upsuper][])
- Gradient LUTs no longer reserve an extra transparent entry for undefined radial-gradient regions; `EncodedGradient` now tracks whether those regions are possible. ([#1529][] by [@LaurenzV][])

### Removed

- Public `glyph` and `colr` modules and the `text` feature; sparse-strips text rendering now lives in `glifo` and the renderer crates. ([#1562][] by [@LaurenzV][])
- Support for recordings, including the shared recording API.
  This decision was made due to a number of downsides that came with the implementation.
  See the corresponding PR and Zulip thread for more information. ([#1611][] by [@LaurenzV][])

### Fixed

- Handling of zero-area rectangles in the rectangle fast path, avoiding out-of-viewport strips and downstream assertions. ([#1537][] by [@LaurenzV][])
- Invalid gradients with NaN or out-of-range final stop offsets now fall back to the first color instead of passing validation. ([#1531][] by [@LaurenzV][])
- Implicit subpaths that continue after `ClosePath` without an explicit `MoveTo`. ([#1544][] by [@tomcur][])
- Negative-sized rectangles during strip generation. ([#1589][] by [@LaurenzV][])

### Optimized

- Rendering performance for paths with line segments fully or partly left of the viewport by culling invisible left-of-viewport line work while preserving winding contributions. ([#1368][] by [@b0nes164][])
- Performance of clipped drawing by culling geometry outside the clip bounding box during flattening. ([#1519][] by [@tomcur][])
- Flattening and tiling performance through codegen improvements in the sparse strip pipeline. ([#1600][], [#1616][], [#1634][] by [@LaurenzV][])

## [0.0.7][] - 2026-03-24

This release has an [MSRV][] of 1.92.

### Added

- Support for image tinting. ([#1460][] by [@grebmeg][])
- `RenderState` abstraction. ([#1486][] by [@LaurenzV][])
- Helpers for rendering filters. ([#1446][] by [@LaurenzV][])
- `ImageCache` and `MultiAtlasManager`. ([#1457][] by [@grebmeg][])

### Changed

- Replaced custom blend mode representation with peniko `Mix` for filter effects. ([#1410][] by [@LaurenzV][])

### Removed

- Process Row Incremental Macro and Perfect Bit. ([#1384][] by [@b0nes164][])
- `tolerance` parameter from the flattening method. ([#1399][] by [@LaurenzV][])
- Command annotation logic. ([#1438][] by [@LaurenzV][])

### Fixed

- Double-scaling of non-hinted COLR glyphs. ([#1370][] by [@conor-93][])
- Handling of `pop_clip` commands and properly resetting wide tiles. ([#1443][] by [@LaurenzV][], [#1447][] by [@tomcur][])
- Layer ranges not being updated when a wide tile is reset with a background color. ([#1478][] by [@LaurenzV][])

### Optimized

- Bézier flattening performance by catching more Béziers whose chords are immediately within rendering tolerance. ([#1216][] by [@tomcur][])
- Flattening and tiling performance by culling out-of-viewport Béziers before flattening. ([#1341][] by [@tomcur][])
- Rendering performance of scenes including blend layers by ensuring no commands are generated for wide tiles without layer content. ([#1403][] by [@tomcur][])
- Scenes including blend layers by only allocating layer scratch buffers for a wide tile upon draw commands being performed in that wide tile. ([#1414][] by [@tomcur][])
- Analytic anti-aliasing. ([#1426][], [#1442][] by [@tomcur][])
- Filter application to layers. ([#1444][] by [@LaurenzV][])
- Rendering pixel-aligned rectangles. ([#1453][] by [@grebmeg][])
- Coarse rasterization and scheduling of scenes without layers. ([#1454][] by [@LaurenzV][])
- Resetting wide tiles. ([#1484][] by [@LaurenzV][])

## [0.0.6][] - 2026-01-15

This release has an [MSRV][] of 1.88.

### Changed

- Breaking change: Updated Peniko to [v0.6.0](https://github.com/linebender/peniko/releases/tag/v0.6.0). ([#1349][] by [@DJMcNab][])
  - This also updates Kurbo to [v0.13.0](https://github.com/linebender/kurbo/releases/tag/v0.13.0).
- Upgraded Skrifa to v0.40.0. ([#1353][] by [@waywardmonkeys][])
- Upgraded Hashbrown to v0.16.1. ([#1354][] by [@waywardmonkeys][])

### Optimized

- Alpha blending for fully opaque image fills. ([#1329][] by [@grebmeg][])

See also the [vello_hybrid 0.0.6](../vello_hybrid/CHANGELOG.md#006---2026-01-15) and [vello_cpu 0.0.6](../vello_cpu/CHANGELOG.md#006---2026-01-15) releases.

## [0.0.5][] - 2026-01-08

This release has an [MSRV][] of 1.88.

### Added

- A new module `clip` allowing for the possibility to intersect two strips to create a new strip representing their intersection. ([#1203][] by [@LaurenzV])
- An `extend` method to `StripStorage` to extend its alphas/strips from another `StripStorage`. ([#1203][] by [@LaurenzV])
- A new trait for approximate integer division by 255. ([#1203][] by [@LaurenzV])
- A `from_parts` method for masks. ([#1237][] by [@LaurenzV])
- Initial support for image filters. ([#1286][] by [@grebmeg])

### Changed

- `WideTile::generate` now takes an additional `BlendMode` as a parameter. ([#1159][] by [@LaurenzV])
- `CmdFill` and `CmdAlphaFill` now store a `BlendMode` instead of `Option<BlendMode>`. ([#1159][] by [@LaurenzV])
- `Strip` now implements `PartialEq` and `Eq`. ([#1203][] by [@LaurenzV])
- `Strip` now has a `is_sentinel` method. ([#1203][] by [@LaurenzV])
- `StripStorage` now implements `PartialEq` and `Eq`. ([#1203][] by [@LaurenzV])
- The `generate_filled_path` method of `StripGenerator` now takes an optional clip path as input. ([#1203][] by [@LaurenzV])
- The `generate` method of `Wide` now takes an optional mask as an additional argument. ([#1237][] by [@LaurenzV])
- `CmdFill` and `CmdAlphaFill` now store an optional mask. ([#1237][] by [@LaurenzV])
- Various changes to the logic for computing tile intersections and representation of tiles. ([#1293][], [#1317][], [#1318][] by [@b0nes164])
- Support for computing data necessary to implement multi-sampled anti-aliasing. ([#1319][], by [@b0nes164])

### Optimized

- Gradient rendering. ([#1301][] by [@valadaptive])
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

[@AdrianEddy]: https://github.com/AdrianEddy
[@b0nes164]: https://github.com/b0nes164
[@conor-93]: https://github.com/conor-93
[@DJMcNab]: https://github.com/waywardmonkeys
[@grebmeg]: https://github.com/grebmeg
[@LaurenzV]: https://github.com/LaurenzV
[@taj-p]: https://github.com/taj-p
[@tomcur]: https://github.com/tomcur
[@upsuper]: https://github.com/upsuper
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
[#1368]: https://github.com/linebender/vello/pull/1368
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
[#1444]: https://github.com/linebender/vello/pull/1444
[#1446]: https://github.com/linebender/vello/pull/1446
[#1447]: https://github.com/linebender/vello/pull/1447
[#1453]: https://github.com/linebender/vello/pull/1453
[#1454]: https://github.com/linebender/vello/pull/1454
[#1457]: https://github.com/linebender/vello/pull/1457
[#1460]: https://github.com/linebender/vello/pull/1460
[#1478]: https://github.com/linebender/vello/pull/1478
[#1484]: https://github.com/linebender/vello/pull/1484
[#1486]: https://github.com/linebender/vello/pull/1486
[#1519]: https://github.com/linebender/vello/pull/1519
[#1529]: https://github.com/linebender/vello/pull/1529
[#1531]: https://github.com/linebender/vello/pull/1531
[#1537]: https://github.com/linebender/vello/pull/1537
[#1544]: https://github.com/linebender/vello/pull/1544
[#1552]: https://github.com/linebender/vello/pull/1552
[#1559]: https://github.com/linebender/vello/pull/1559
[#1562]: https://github.com/linebender/vello/pull/1562
[#1589]: https://github.com/linebender/vello/pull/1589
[#1600]: https://github.com/linebender/vello/pull/1600
[#1611]: https://github.com/linebender/vello/pull/1611
[#1613]: https://github.com/linebender/vello/pull/1613
[#1616]: https://github.com/linebender/vello/pull/1616
[#1634]: https://github.com/linebender/vello/pull/1634
[#1635]: https://github.com/linebender/vello/pull/1635
[#1673]: https://github.com/linebender/vello/pull/1673
[#1737]: https://github.com/linebender/vello/pull/1737

[Unreleased]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.9...HEAD
[0.0.9]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.8...sparse-strips-v0.0.9
[0.0.8]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.7...sparse-strips-v0.0.8
[0.0.7]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.6...sparse-strips-v0.0.7
[0.0.6]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.5...sparse-strips-v0.0.6
[0.0.5]: https://github.com/linebender/vello/compare/sparse-stips-v0.0.4...sparse-strips-v0.0.5
[0.0.4]: https://github.com/linebender/vello/compare/sparse-stips-v0.0.3...sparse-strips-v0.0.4
[0.0.3]: https://github.com/linebender/vello/compare/sparse-stips-v0.0.2...sparse-strips-v0.0.3
[0.0.2]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.1...sparse-stips-v0.0.2
[0.0.1]: https://github.com/linebender/vello/compare/ca6b1e4c7f5b0d95953c3b524f5d3952d5669c5a...sparse-strips-v0.0.1

[MSRV]: README.md#minimum-supported-rust-version-msrv
