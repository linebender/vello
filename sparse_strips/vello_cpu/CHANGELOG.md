<!-- Instructions

This changelog follows the patterns described here: <https://keepachangelog.com/en/>.

Subheadings to categorize changes are `added, changed, deprecated, removed, fixed, security`.

-->

# Changelog

## [Unreleased]

TODO: Before making a 0.0.10 (or 0.1.0) release, resolve the following issue:
https://github.com/linebender/vello/pull/1665#issuecomment-4667033939!

This release has an [MSRV][] of 1.88.

### Changed
- The API for rendering into a pixmap. The methods `render_to_pixmap` and
  `composite_to_pixmap_at_offset` have been replaced with a single unified 
  `render` (and `render_with`) method that takes additional parameters for tweaking the behavior. 
  ([#1665][] by [@LaurenzV][])

## [0.0.9][] - 2026-05-30

This release has an [MSRV][] of 1.88.

### Fixed

- Rendering of glyphs with complex paints. ([#1668][] by [@LaurenzV][])
- Rendering of paths with high winding counts. ([#1673][] by [@b0nes164][])

### Optimized

- Optimized rendering of a number of blend modes. ([#1653][] by [@LaurenzV][])

## [0.0.8][] - 2026-05-15

This release has an [MSRV][] of 1.88.

### Added

- `render_decoration` on glyph run builders for rendering text decorations with skip-ink behavior. ([#1592][] by [@taj-p][])
- Support for rendering VARC glyphs in text paths. ([#1594][] by [@nicoburns][], [@oscargus][])
- Support for synthetic font emboldening in text glyph runs via `GlyphRunBuilder::font_embolden` and `glifo::FontEmbolden`. ([#1628][] by [@jrmoulton][])

### Changed

- Migrated text rendering to `glifo`; rendering, glyph, and image APIs now use an explicit `Resources` object for persistent image and glyph caches. `RenderContext::render_to_buffer`, `RenderContext::render_to_pixmap`, `RenderContext::composite_to_pixmap_at_offset`, and `RenderContext::glyph_run` now take resources, and image registration moved from `RenderContext` to `Resources`. ([#1562][] by [@LaurenzV][])
- Renamed image transparency hint APIs from `may_have_opacities` and `*_opacity_hint` to `may_have_transparency` and `*_transparency_hint`. ([#1613][] by [@upsuper][])

### Removed

- Experimental `vello_api` integration and `api` module. ([#1602][] by [@waywardmonkeys][])
- Support for recordings.
  This decision was made due to a number of downsides that came with the implementation.
  See the corresponding PR and Zulip thread for more information. ([#1611][] by [@LaurenzV][])

### Fixed

- Rendering of rare degenerate radial gradients with undefined regions, which should render those regions transparent. ([#1529][] by [@LaurenzV][])
- Incorrect rendering of filter layers containing nested clip layers. ([#1541][] by [@LaurenzV][])
- x86 artifacts in bicubic image rendering caused by negative interpolated color values wrapping during SIMD `f32` to `u8` conversion. ([#1563][] by [@LaurenzV][])
- Stroked glyph rendering when scaled glyph runs absorb the transform into the font size, preserving the expected stroke width. ([#1576][] by [@LaurenzV][])
- COLR glyph rendering, including glyphs using non-default blend modes. ([#1584][] by [@LaurenzV][])
- Rendering of inverted rectangles, including blurred rounded rectangles. ([#1589][] by [@LaurenzV][])

### Optimized

- CPU fine rasterization performance through improved SIMD codegen. ([#1616][] by [@LaurenzV][])

## [0.0.7][] - 2026-03-24

This release has an [MSRV][] of 1.92.

### Added

- `composite_to_pixmap_at_offset` method to `RenderContext` for compositing at specific offsets within a larger pixmap. ([#1416][] by [@grebmeg][])
- `ImageResolver` for resolving opaque image IDs at rasterization time. ([#1451][] by [@grebmeg][])
- Support for image tinting. ([#1460][] by [@grebmeg][])

### Fixed

- Rendering of blurred rounded rectangles with zero or very small blur standard deviations. ([#1422][] by [@tomcur][])
- Off-by-one error in gaussian blur decimation filter. ([#1488][] by [@LaurenzV][])
- Filter layers with zero clips. ([#1437][] by [@LaurenzV][])

### Optimized

- Bilinear image sampling in the `RenderMode::OptimizeQuality` (`f32`) pipeline. ([#1343][] by [@tomcur][])

## [0.0.6][] - 2026-01-15

This release has an [MSRV][] of 1.88.

### Added

- Support for the "offset" filter. ([#1351] by [@waywardmonkeys])

### Changed

- Breaking change: Updated Peniko to [v0.6.0](https://github.com/linebender/peniko/releases/tag/v0.6.0). ([#1349][] by [@DJMcNab][])
  - This also updates Kurbo to [v0.13.0](https://github.com/linebender/kurbo/releases/tag/v0.13.0).
- Upgraded Skrifa to v0.40.0. ([#1353][] by [@waywardmonkeys][])
- Upgraded Hashbrown to v0.16.1. ([#1354][] by [@waywardmonkeys][])
- Optimized image rendering for axis-aligned images. ([#1335][] by [@grebmeg][])

See also the [vello_hybrid 0.0.6](../vello_hybrid/CHANGELOG.md#006---2026-01-15) and [vello_common 0.0.6](../vello_common/CHANGELOG.md#006---2026-01-15) releases.

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
- A `set_mask` method to make it possible to mask rendered paths without inducing layer isolation. ([#1237][] by [@LaurenzV])
- Support for conditionally disabling the u8 or f32 pipeline. ([#1294][] by [@nicoburns])

### Changed

- Improved performance of rendering opaque images. ([#1327][] by [@grebmeg])

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
[@b0nes164]: https://github.com/b0nes164
[@grebmeg]: https://github.com/grebmeg
[@jrmoulton]: https://github.com/jrmoulton
[@LaurenzV]: https://github.com/LaurenzV
[@nicoburns]: https://github.com/nicoburns
[@oscargus]: https://github.com/oscargus
[@taj-p]: https://github.com/taj-p
[@tomcur]: https://github.com/tomcur
[@upsuper]: https://github.com/upsuper
[@waywardmonkeys]: https://github.com/waywardmonkeys

[#1159]: https://github.com/linebender/vello/pull/1159
[#1203]: https://github.com/linebender/vello/pull/1203
[#1237]: https://github.com/linebender/vello/pull/1237
[#1286]: https://github.com/linebender/vello/pull/1286
[#1294]: https://github.com/linebender/vello/pull/1294
[#1327]: https://github.com/linebender/vello/pull/1327
[#1335]: https://github.com/linebender/vello/pull/1335
[#1343]: https://github.com/linebender/vello/pull/1343
[#1349]: https://github.com/linebender/vello/pull/1349
[#1351]: https://github.com/linebender/vello/pull/1351
[#1353]: https://github.com/linebender/vello/pull/1353
[#1354]: https://github.com/linebender/vello/pull/1354
[#1416]: https://github.com/linebender/vello/pull/1416
[#1422]: https://github.com/linebender/vello/pull/1422
[#1437]: https://github.com/linebender/vello/pull/1437
[#1451]: https://github.com/linebender/vello/pull/1451
[#1460]: https://github.com/linebender/vello/pull/1460
[#1488]: https://github.com/linebender/vello/pull/1488
[#1529]: https://github.com/linebender/vello/pull/1529
[#1541]: https://github.com/linebender/vello/pull/1541
[#1562]: https://github.com/linebender/vello/pull/1562
[#1563]: https://github.com/linebender/vello/pull/1563
[#1576]: https://github.com/linebender/vello/pull/1576
[#1584]: https://github.com/linebender/vello/pull/1584
[#1589]: https://github.com/linebender/vello/pull/1589
[#1592]: https://github.com/linebender/vello/pull/1592
[#1594]: https://github.com/linebender/vello/pull/1594
[#1602]: https://github.com/linebender/vello/pull/1602
[#1611]: https://github.com/linebender/vello/pull/1611
[#1613]: https://github.com/linebender/vello/pull/1613
[#1616]: https://github.com/linebender/vello/pull/1616
[#1628]: https://github.com/linebender/vello/pull/1628
[#1653]: https://github.com/linebender/vello/pull/1653
[#1665]: https://github.com/linebender/vello/pull/1665
[#1668]: https://github.com/linebender/vello/pull/1668
[#1673]: https://github.com/linebender/vello/pull/1673

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
