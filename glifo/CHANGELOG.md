<!-- Instructions

This changelog follows the patterns described here: <https://keepachangelog.com/en/>.

Subheadings to categorize changes are `added, changed, deprecated, removed, fixed, security`.

-->

# Changelog

## [Unreleased]

This release has an [MSRV][] of 1.89.

## [0.5.0][] - 2026-11-11

This release has an [MSRV][] of 1.89.

### Changed

- Breaking change: Updated to `skrifa` 0.49, which is exposed through `ColorGlyph::skrifa_glyph`. ([#1996][] by [@nicoburns][])

### Fixed

- Synthetic emboldening of unhinted glyphs being scaled down to almost nothing. `FontEmbolden` amounts are now in pixels at the run's font size and scale with the run transform. ([#2049][] by [@nicoburns][])
- COLR glyphs without a `ClipBox`, which includes all COLRv0 glyphs, coming out lighter along the edges of their bounds. ([#2050][] by [@ump45nose][])
- Placement of `sbix` bitmap glyphs. ([#1996][] by [@nicoburns][])

## [0.4.0][] - 2026-10-02

This release has an [MSRV][] of 1.89.

### Changed

- Breaking change: Glyph filling and stroking methods now return `Result<(), GlyphRenderError>`. ([#1944][] by [@LaurenzV][])
- Breaking change: `GlyphAtlas::replay_pending_atlas_commands` now accepts a fallible callback. ([#1890][] by [@LaurenzV][])
- Bitmap glyphs now consistently use bilinear sampling (`ImageQuality::Medium`) instead of switching to bicubic sampling when downscaled. ([#1970][] by [@LaurenzV][])

### Removed

- Breaking change: `renderer::quality_for_scale`. ([#1970][] by [@LaurenzV][])

### Fixed

- Incorrect placement of non-Apple `sbix` glyphs caused by the Apple Color Emoji workaround. ([#1936][] by [@LaurenzV][])
- Incorrect `sbix` glyph placement caused by sign errors in bearing offsets. ([#1939][] by [@LaurenzV][])

## [0.3.0][] - 2026-08-07

This release has an [MSRV][] of 1.88.

### Changed

- Breaking change: Updated `vello_common` to v0.2.0.

## [0.2.0][] - 2026-07-29

This release has an [MSRV][] of 1.88.

### Changed

- Updated Skrifa to v0.44.0. ([#1774][] by [@LaurenzV][])

## [0.1.1][] - 2026-05-30

This release has an [MSRV][] of 1.88.

### Fixed

- Ignored paint transform when rendering glyphs. ([#1668][] by [@LaurenzV][])

### Optimized
- Caching behavior of glyph outlines. ([#1629][] by [@LaurenzV][])
- Performance of rendering COLR glyphs. ([#1672][] by [@LaurenzV][])

## [0.1.0][] - 2026-05-15

This release has an [MSRV][] of 1.88.

The first release of Glifo!

Glifo moved to the Vello repo in [#1539][] and was prepared for release by [@conor-93][], [@grebmeg][], [@jrmoulton][], [@LaurenzV][], [@nicoburns][], [@oscargus][], [@taj-p][], and [@xStrom][].

[@conor-93]: https://github.com/conor-93
[@grebmeg]: https://github.com/grebmeg
[@jrmoulton]: https://github.com/jrmoulton
[@LaurenzV]: https://github.com/LaurenzV
[@nicoburns]: https://github.com/nicoburns
[@oscargus]: https://github.com/oscargus
[@taj-p]: https://github.com/taj-p
[@ump45nose]: https://github.com/ump45nose
[@xStrom]: https://github.com/xStrom

[#1539]: https://github.com/linebender/vello/pull/1539
[#1629]: https://github.com/linebender/vello/pull/1629
[#1668]: https://github.com/linebender/vello/pull/1668
[#1672]: https://github.com/linebender/vello/pull/1672
[#1774]: https://github.com/linebender/vello/pull/1774
[#1890]: https://github.com/linebender/vello/pull/1890
[#1936]: https://github.com/linebender/vello/pull/1936
[#1939]: https://github.com/linebender/vello/pull/1939
[#1944]: https://github.com/linebender/vello/pull/1944
[#1970]: https://github.com/linebender/vello/pull/1970
[#1996]: https://github.com/linebender/vello/pull/1996
[#2049]: https://github.com/linebender/vello/pull/2049
[#2050]: https://github.com/linebender/vello/pull/2050

[Unreleased]: https://github.com/linebender/vello/compare/glifo-v0.5.0...HEAD
[0.5.0]: https://github.com/linebender/vello/compare/glifo-v0.4.0...glifo-v0.5.0
[0.4.0]: https://github.com/linebender/vello/compare/glifo-v0.3.0...glifo-v0.4.0
[0.3.0]: https://github.com/linebender/vello/compare/glifo-v0.2.0...glifo-v0.3.0
[0.2.0]: https://github.com/linebender/vello/compare/glifo-v0.1.1...glifo-v0.2.0
[0.1.1]: https://github.com/linebender/vello/compare/glifo-v0.1.0...glifo-v0.1.1
[0.1.0]: https://github.com/linebender/vello/compare/246912ae692cff7719cd95026107cc1aa077f205...glifo-v0.1.0

[MSRV]: README.md#minimum-supported-rust-version-msrv
