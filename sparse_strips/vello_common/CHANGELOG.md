<!-- Instructions

This changelog follows the patterns described here: <https://keepachangelog.com/en/>.

Subheadings to categorize changes are `added, changed, deprecated, removed, fixed, security`.

-->

# Changelog

The latest published vello_common release is [0.0.4](#004---2025-10-17) which was released on 2025-10-17.
You can find its changes [documented below](#004---2025-10-17).

## [Unreleased]

This release has an [MSRV][] of 1.88.

### Added
- A new module `clip` has been added allowing for the possibility
  to intersect two strips to create a new strip representing their
  intersection ([#1203][] by [@LaurenzV])
- An `extend` method has been added to `StripStorage` to extends
  its alphas/strips from another `StripStorage` ([#1203][] by [@LaurenzV])
- Added a `from_parts` method for masks ([#1237][] by [@LaurenzV])

### Changed
- `WideTile::generate` now takes an additional `BlendMode` as a parameter ([#1159][] by [@LaurenzV])
- `CmdFill` and `CmdAlphaFill` now store a `BlendMode` instead of `Option<BlendMode>` ([#1159][] by [@LaurenzV])
- `Strip` now implements `PartialEq` and `Eq` ([#1203][] by [@LaurenzV])
- `Strip` now has a `is_sentinel` method ([#1203][] by [@LaurenzV])
- `StripStorage` now implements `PartialEq` and `Eq` ([#1203][] by [@LaurenzV])
- The `generate_filled_path` method of `StripGenerator` now takes
  an optional clip path as input ([#1203][] by [@LaurenzV])
- A new trait for approximate integer division by 255 has been added ([#1203][] by [@LaurenzV])
- The `generate` method of `Wide` now takes an optional mask as an
  additional argument ([#1237][] by [@LaurenzV])
- `CmdFill` and `CmdAlphaFill` now store an optional mask ([#1237][] by [@LaurenzV])

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

[@LaurenzV]: https://github.com/LaurenzV

[#1237]: https://github.com/linebender/vello/pull/1237
[#1203]: https://github.com/linebender/vello/pull/1203
[#1159]: https://github.com/linebender/vello/pull/1159

[Unreleased]: https://github.com/linebender/fearless_simd/compare/sparse-strips-v0.0.4...HEAD
[0.0.4]: https://github.com/linebender/vello/compare/sparse-stips-v0.0.3...sparse-strips-v0.0.4
[0.0.3]: https://github.com/linebender/vello/compare/sparse-stips-v0.0.2...sparse-strips-v0.0.3
[0.0.2]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.1...sparse-stips-v0.0.2
[0.0.1]: https://github.com/linebender/vello/compare/ca6b1e4c7f5b0d95953c3b524f5d3952d5669c5a...sparse-strips-v0.0.1

[MSRV]: README.md#minimum-supported-rust-version-msrv

