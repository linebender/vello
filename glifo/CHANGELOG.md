<!-- Instructions

This changelog follows the patterns described here: <https://keepachangelog.com/en/>.

Subheadings to categorize changes are `added, changed, deprecated, removed, fixed, security`.

-->

# Changelog

## [Unreleased]

This release has an [MSRV][] of 1.89.

### Fixed

- Glyph pixel-snapping decisions (the atlas quad's floor/fract split and the hinting baseline round) are now stable under last-ulp noise in the translation, so two passes that compute the same glyph position through different floating-point op orders (e.g. a live render and a recorded or cached one) place the glyph on the same pixel. ([#1785][] by [@AdrianEddy][])

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

[@AdrianEddy]: https://github.com/AdrianEddy
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
[#1785]: https://github.com/linebender/vello/pull/1785

[Unreleased]: https://github.com/linebender/vello/compare/glifo-v0.3.0...HEAD
[0.3.0]: https://github.com/linebender/vello/compare/glifo-v0.2.0...glifo-v0.3.0
[0.2.0]: https://github.com/linebender/vello/compare/glifo-v0.1.1...glifo-v0.2.0
[0.1.1]: https://github.com/linebender/vello/compare/glifo-v0.1.0...glifo-v0.1.1
[0.1.0]: https://github.com/linebender/vello/compare/246912ae692cff7719cd95026107cc1aa077f205...glifo-v0.1.0

[MSRV]: README.md#minimum-supported-rust-version-msrv
