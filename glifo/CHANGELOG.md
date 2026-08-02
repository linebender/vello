<!-- Instructions

This changelog follows the patterns described here: <https://keepachangelog.com/en/>.

Subheadings to categorize changes are `added, changed, deprecated, removed, fixed, security`.

-->

# Changelog

## [Unreleased]

This release has an [MSRV][] of 1.88.

### Added

- `GlyphRenderer::glyph_coverage_contrast` (default `CoverageContrast::NONE`), applied to solid-painted outline glyphs with the weight term resolved per draw against the text color's luminance. Atlas-cached glyphs receive the transfer through their alpha-mask tint at sample time; glyphs drawn directly from their outlines receive the same transfer at fill time through the new `GlyphRenderer::fill_glyph_path` hook, so appearance does not depend on whether a glyph is cached. ([#1790][] by [@AdrianEddy][])

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
[@xStrom]: https://github.com/xStrom

[#1539]: https://github.com/linebender/vello/pull/1539
[#1629]: https://github.com/linebender/vello/pull/1629
[#1668]: https://github.com/linebender/vello/pull/1668
[#1672]: https://github.com/linebender/vello/pull/1672
[#1774]: https://github.com/linebender/vello/pull/1774

[#1790]: https://github.com/linebender/vello/pull/1790
[@AdrianEddy]: https://github.com/AdrianEddy
[Unreleased]: https://github.com/linebender/vello/compare/glifo-v0.2.0...HEAD
[0.2.0]: https://github.com/linebender/vello/compare/glifo-v0.1.1...glifo-v0.2.0
[0.1.1]: https://github.com/linebender/vello/compare/glifo-v0.1.0...glifo-v0.1.1
[0.1.0]: https://github.com/linebender/vello/compare/246912ae692cff7719cd95026107cc1aa077f205...glifo-v0.1.0

[MSRV]: README.md#minimum-supported-rust-version-msrv
