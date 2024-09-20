<!-- Instructions

This changelog follows the patterns described here: <https://keepachangelog.com/en/>.

Subheadings to categorize changes are `added, changed, deprecated, removed, fixed, security`.

-->

# Changelog

The latest published Vello release is [0.2.1](#021---2024-07-16) which was released on 2024-07-16.
You can find its changes [documented below](#021---2024-07-16).

## [Unreleased]

This release has an [MSRV][] of 1.75.

### Highlights

- Support for most Emoji ([#615][], [#641][] by [@DJMcNab])

### Added

- Support blends more than four layers deep ([#657][] by [@DJMcNab][])

### Changed

- Breaking: Updated `wgpu` to 22.1.0. ([#635] by [@waywardmonkeys])

### Fixed

### Removed

- Breaking: `Pipelines` API from `vello_shaders` ([#612] by [@DJMcNab])

## [0.2.1][] - 2024-07-16

This release has an [MSRV][] of 1.75.

### Fixed

- Crash when there is no scene contents ([#630] by [@DJMcNab])

### Changed

- Updated `wgpu` to 0.20.1. ([#631] by [@waywardmonkeys])
- Document the MSRV of releases in the changelog ([#619] by [@DJMcNab])

## [0.2.0] - 2024-06-08

This release has an [MSRV][] of 1.75.

### Added

- Euler spiral based stroke expansion. ([#496] by [@raphlinus])
- Sweep gradients. ([#435] by [@dfrg])
- Bump allocation estimation. ([#436], [#454], [#522] by [@armansito])
- Impl `From<Encoding>` for `Scene`. ([#538] by [@waywardmonkeys])
- Glyph hinting support. ([#544] by [@dfrg])
- Better glyph caching. ([#555] by [@dfrg])
- `vello_shaders` crate to load and preprocess WGSL. ([#563] by [@armansito])
- Coverage-mask specialization. ([#540] by [@armansito])
- Support for the `#enable` post-process directive. ([#550] by [@armansito])

### Changed

- Better error types. ([#516] by [@DasLixou])
- `RenderContext::new()` no longer returns a `Result`. ([#547] by [@waywardmonkeys])
- Updated `wgpu` to 0.20. ([#560] by [@waywardmonkeys])

### Removed

- `force_rw_storage` feature. ([#540] by [@armansito])

### Fixed

- 64k draw object limit. ([#526] by [@raphlinus])
- Increased robustness of cubic params. ([#521] by [@raphlinus])
- Increased robustness of GPU shaders. ([#537] by [@raphlinus])
- `draw_leaf` uniformity. ([#535] by [@raphlinus])
- Bug in join estimates in `vello_encoding`. ([#573] by [@armansito])
- Incorrect use of numerical operators on atomics in binning. ([#539] by [@armansito])
- `path_reduced_scan` buffer size. ([#551] by [@armansito])
- Handling of upstream pipeline failure. ([#553] by [@armansito])
- Very slow shader compilation. ([#575] by [@DJMcNab], [@waywardmonkeys])
- Full system hang on Apple systems. ([#589] by [@raphlinus])

## [0.1.0] - 2024-03-04

- Initial release

[@raphlinus]: https://github.com/raphlinus
[@armansito]: https://github.com/armansito
[@DJMcNab]: https://github.com/DJMcNab
[@dfrg]: https://github.com/drfg
[@waywardmonkeys]: https://github.com/waywardmonkeys
[@DasLixou]: https://github.com/DasLixou

[#435]: https://github.com/linebender/vello/pull/435
[#436]: https://github.com/linebender/vello/pull/436
[#454]: https://github.com/linebender/vello/pull/454
[#496]: https://github.com/linebender/vello/pull/496
[#516]: https://github.com/linebender/vello/pull/516
[#521]: https://github.com/linebender/vello/pull/521
[#522]: https://github.com/linebender/vello/pull/522
[#526]: https://github.com/linebender/vello/pull/526
[#535]: https://github.com/linebender/vello/pull/535
[#537]: https://github.com/linebender/vello/pull/537
[#538]: https://github.com/linebender/vello/pull/538
[#539]: https://github.com/linebender/vello/pull/539
[#540]: https://github.com/linebender/vello/pull/540
[#544]: https://github.com/linebender/vello/pull/544
[#547]: https://github.com/linebender/vello/pull/547
[#550]: https://github.com/linebender/vello/pull/550
[#551]: https://github.com/linebender/vello/pull/551
[#553]: https://github.com/linebender/vello/pull/553
[#555]: https://github.com/linebender/vello/pull/555
[#560]: https://github.com/linebender/vello/pull/560
[#563]: https://github.com/linebender/vello/pull/563
[#573]: https://github.com/linebender/vello/pull/573
[#575]: https://github.com/linebender/vello/pull/575
[#589]: https://github.com/linebender/vello/pull/589
[#612]: https://github.com/linebender/vello/pull/612
[#615]: https://github.com/linebender/vello/pull/615
[#619]: https://github.com/linebender/vello/pull/619
[#630]: https://github.com/linebender/vello/pull/630
[#631]: https://github.com/linebender/vello/pull/631
[#635]: https://github.com/linebender/vello/pull/635
[#641]: https://github.com/linebender/vello/pull/641
[#657]: https://github.com/linebender/vello/pull/657

<!-- Note that this still comparing against 0.2.0, because 0.2.1 is a cherry-picked patch -->
[Unreleased]: https://github.com/linebender/vello/compare/v0.2.0...HEAD
[0.2.1]: https://github.com/linebender/vello/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/linebender/vello/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/linebender/vello/releases/tag/v0.1.0

[MSRV]: README.md#minimum-supported-rust-version-msrv
