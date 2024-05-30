# Changelog

<!-- Instructions

This changelog follows the patterns described here: <https://keepachangelog.com/en/1.0.0/>.

Subheadings to categorize changes are `added, changed, deprecated, removed, fixed, security`.

-->

## Unreleased

### Added

- [#435](https://github.com/linebender/vello/pull/435) - Sweep gradients by [@dfrg](https://github.com/drfg)
- [#544](https://github.com/linebender/vello/pull/544) - Restore glyph hinting support by [@dfrg](https://github.com/drfg)

### Changed

- [#547](https://github.com/linebender/vello/pull/547) - `RenderContext::new()` no longer returns a `Result` by [@waywardmonkeys](https://github.com/waywardmonkeys)

### Fixed

- [#496](https://github.com/linebender/vello/pull/496) - Performance optimizations for stroke-heavy scenes by [@raphlinus](https://github.com/raphlinus)
- [#521](https://github.com/linebender/vello/pull/521) - Increase robustness of cubic params by [@raphlinus](https://github.com/raphlinus)
- [#526](https://github.com/linebender/vello/pull/526) - Increases ~64k draw object limit by [@raphlinus](https://github.com/raphlinus)
- [#537](https://github.com/linebender/vello/pull/537) - Increase robustness of GPU shaders by [@raphlinus](https://github.com/raphlinus)

## 0.1.0 (2024-03-04)

- Initial release
