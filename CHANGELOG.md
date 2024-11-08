<!-- Instructions

This changelog follows the patterns described here: <https://keepachangelog.com/en/>.

Subheadings to categorize changes are `added, changed, deprecated, removed, fixed, security`.

-->

# Changelog

The latest published Vello release is [0.3.0](#030---2024-10-04) which was released on 2024-10-04.
You can find its changes [documented below](#030---2024-10-04).

## [Unreleased]

This release has an [MSRV][] of 1.75.

### Fixed

- Offset in image rendering, and sampling outside correct atlas area ([#722][] by [@dfrg])

## [0.3.0][] - 2024-10-04

This release has an [MSRV][] of 1.75.

### Highlights

- Support for most Emoji ([#615][], [#641][] by [@DJMcNab])
- [GPU Friendly Stroke Expansion][stroke-expansion], which documents key parts of how Vello works, was released (by [@raphlinus], [@armansito])
- Blurred rounded rectangles are now supported, which can be used for box shadows ([#665][] by [@msiglreith][])
- Vello is no longer considered experimental 🎉 ([#691][] by [@waywardmonkeys])
  Note that Vello is still an alpha, but we believe that the direction has been proven.

### Added

- Access to the `Adapter` from the utils `DeviceHandle` ([#634][] by [@cfagot][])
- Support for compositing existing `wgpu::Texture`s into a Vello scene ([#636][], [#655][] by [@DJMcNab], [@TrueDoctor][])
- Utilities for constructing an `AaSupport` from a set of `AaConfig`s ([#654][] by [@simbleau][])
- An example which uses sdl2 ([#671][] by [@TheNachoBIT][])
- The underlying `Encoding` for a scene can now be modified, circumventing guardrails for advanced use-cases ([#701][] by [@timtom-dev][])

### Changed

- Breaking: Updated `wgpu` to 22.1.0 ([#635][] by [@waywardmonkeys])
- Clipping more than four layers deep is now supported ([#657][] by [@DJMcNab])
- Significantly improved automated testing ([#610][], [#643][] by [@DJMcNab])
- Preliminary debug layers for Vello's internal development ([#416][] by [@armansito])
- Examples now use the [`run_app`][] API from Winit ([#626][], [#628][] by [@yutannihilation][])
- Labels on GPU objects are now prefixed with `vello.` ([#677][] by [@waywardmonkeys])
- The async render implementations are deprecated and unstable; you should use the non-async versions as they enable greater throughput ([#706][] by [@DJMcNab])

### Fixed

- Example code in the repository README ([#627][] by [@kmoon2437][])
- A possible crash on iOS working around an invariant undocumented by Apple ([#639][] by [@DJMcNab][])
- Large number of clips now work ([#659][] by [@raphlinus])
- Empty clips now no longer cause artifacts ([#651][] by [@raphlinus])
- A potential panic in the presence of a weaker than default allocator ([#675][] by [@timtom-dev][])
- Watertightness breaks causing artifacts with some rounded rectangles ([#695][] by [@raphlinus])

### Removed

- Breaking: `Pipelines` API from `vello_shaders` ([#612][] by [@DJMcNab])
- The `wgpu_profiler` profiler feature is no longer stable ([#694][] by [@DJMcNab])
- Breaking: Moved the `Recording` abstraction into a `low_level` module, as almost all users should prefer the higher-level `Renderer` ([#711][] by [@DJMcNab])

## [0.2.1][] - 2024-07-16

This release has an [MSRV][] of 1.75.

### Fixed

- Crash when there is no scene contents ([#630][]by [@DJMcNab])

### Changed

- Updated `wgpu` to 0.20.1. ([#631][]by [@waywardmonkeys])
- Document the MSRV of releases in the changelog ([#619][] by [@DJMcNab])

## [0.2.0] - 2024-06-08

This release has an [MSRV][] of 1.75.

### Added

- Euler spiral based stroke expansion. ([#496][] by [@raphlinus])
- Sweep gradients. ([#435][] by [@dfrg])
- Bump allocation estimation. ([#436][], [#454][], [#522][] by [@armansito])
- Impl `From<Encoding>` for `Scene`. ([#538][] by [@waywardmonkeys])
- Glyph hinting support. ([#544][] by [@dfrg])
- Better glyph caching. ([#555][] by [@dfrg])
- `vello_shaders` crate to load and preprocess WGSL. ([#563][] by [@armansito])
- Coverage-mask specialization. ([#540][] by [@armansito])
- Support for the `#enable` post-process directive. ([#550][] by [@armansito])

### Changed

- Better error types. ([#516][] by [@DasLixou])
- `RenderContext::new()` no longer returns a `Result`. ([#547][] by [@waywardmonkeys])
- Updated `wgpu` to 0.20. ([#560][] by [@waywardmonkeys])

### Removed

- `force_rw_storage` feature. ([#540][] by [@armansito])

### Fixed

- 64k draw object limit. ([#526][] by [@raphlinus])
- Increased robustness of cubic params. ([#521][] by [@raphlinus])
- Increased robustness of GPU shaders. ([#537][] by [@raphlinus])
- `draw_leaf` uniformity. ([#535][] by [@raphlinus])
- Bug in join estimates in `vello_encoding`. ([#573][] by [@armansito])
- Incorrect use of numerical operators on atomics in binning. ([#539][] by [@armansito])
- `path_reduced_scan` buffer size. ([#551][] by [@armansito])
- Handling of upstream pipeline failure. ([#553][] by [@armansito])
- Very slow shader compilation. ([#575][] by [@DJMcNab], [@waywardmonkeys])
- Full system hang on Apple systems. ([#589][] by [@raphlinus])

## [0.1.0] - 2024-03-04

- Initial release

[@raphlinus]: https://github.com/raphlinus
[@armansito]: https://github.com/armansito
[@cfagot]: https://github.com/cfagot
[@DasLixou]: https://github.com/DasLixou
[@dfrg]: https://github.com/drfg
[@DJMcNab]: https://github.com/DJMcNab
[@kmoon2437]: https://github.com/kmoon2437
[@msiglreith]: https://github.com/msiglreith
[@simbleau]: https://github.com/simbleau
[@TheNachoBIT]: https://github.com/TheNachoBIT
[@timtom-dev]: https://github.com/timtom-dev
[@TrueDoctor]: https://github.com/TrueDoctor
[@waywardmonkeys]: https://github.com/waywardmonkeys
[@yutannihilation]: https://github.com/yutannihilation

[#416]: https://github.com/linebender/vello/pull/416
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
[#610]: https://github.com/linebender/vello/pull/610
[#612]: https://github.com/linebender/vello/pull/612
[#615]: https://github.com/linebender/vello/pull/615
[#619]: https://github.com/linebender/vello/pull/619
[#626]: https://github.com/linebender/vello/pull/626
[#627]: https://github.com/linebender/vello/pull/627
[#628]: https://github.com/linebender/vello/pull/628
[#630]: https://github.com/linebender/vello/pull/630
[#631]: https://github.com/linebender/vello/pull/631
[#634]: https://github.com/linebender/vello/pull/634
[#635]: https://github.com/linebender/vello/pull/635
[#636]: https://github.com/linebender/vello/pull/636
[#639]: https://github.com/linebender/vello/pull/639
[#641]: https://github.com/linebender/vello/pull/641
[#643]: https://github.com/linebender/vello/pull/643
[#651]: https://github.com/linebender/vello/pull/651
[#654]: https://github.com/linebender/vello/pull/654
[#655]: https://github.com/linebender/vello/pull/655
[#657]: https://github.com/linebender/vello/pull/657
[#659]: https://github.com/linebender/vello/pull/659
[#665]: https://github.com/linebender/vello/pull/665
[#671]: https://github.com/linebender/vello/pull/671
[#675]: https://github.com/linebender/vello/pull/675
[#677]: https://github.com/linebender/vello/pull/677
[#691]: https://github.com/linebender/vello/pull/691
[#694]: https://github.com/linebender/vello/pull/694
[#695]: https://github.com/linebender/vello/pull/695
[#701]: https://github.com/linebender/vello/pull/701
[#706]: https://github.com/linebender/vello/pull/706
[#711]: https://github.com/linebender/vello/pull/711
[#722]: https://github.com/linebender/vello/pull/722

[Unreleased]: https://github.com/linebender/vello/compare/v0.3.0...HEAD
<!-- Note that this still comparing against 0.2.0, because 0.2.1 is a cherry-picked patch -->
[0.3.0]: https://github.com/linebender/vello/compare/v0.2.0...v0.3.0
[0.2.1]: https://github.com/linebender/vello/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/linebender/vello/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/linebender/vello/releases/tag/v0.1.0

[MSRV]: README.md#minimum-supported-rust-version-msrv
[`run_app`]: https://docs.rs/winit/latest/winit/event_loop/struct.EventLoop.html#method.run_app
[stroke-expansion]: https://linebender.org/gpu-stroke-expansion-paper/
