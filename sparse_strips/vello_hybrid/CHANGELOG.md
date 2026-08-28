<!-- Instructions

This changelog follows the patterns described here: <https://keepachangelog.com/en/>.

Subheadings to categorize changes are `added, changed, deprecated, removed, fixed, security`.

-->

# Changelog

## [Unreleased]

This release has an [MSRV][] of 1.88.

### Fixed

- Rectangles drawn through the GPU rect fast path now produce exactly the same bytes as the CPU strip path. The two rasterizers used to round anti-aliased coverage differently (up to one alpha step at fractional edges and corners, more at large coordinates), so pushing a clip (which routes rects through strips) changed pixels the clip didn't even touch. ([#1784][] by [@AdrianEddy][])

## [0.2.0][] - 2026-08-07

This release has an [MSRV][] of 1.88.

### Added

- Support for sampling from external textures through `Scene::draw_texture_rects` in the `webgl` backend, bringing it to parity with the `wgpu` backend. Textures are bound at render time through the new `WebGlTextureBindings`. ([#1792][] by [@grebmeg][])
- `ProbeResult::statistics`, `ProbeStatistics`, and `ProbeFeature` for diagnosing which renderer-probe features and pixels differ from the reference output. ([#1779][] by [@LaurenzV][])
- `PROBE_ELEMENTS` for inspecting the active features used by the WebGL compatibility probe. ([#1801][] by [@LaurenzV][])

### Changed

- Breaking change: `Renderer::new`, `Renderer::new_with`, `WebGlRenderer::new`, and `WebGlRenderer::new_with` now return the renderer and its associated `Resources` as a tuple. `Resources` can no longer be constructed directly and must only be used with the renderer instance that generated it. ([#1794][] by [@LaurenzV][])
- Breaking change: `Scene::new_with` now takes a `Level` instead of `RenderSettings`. ([#1794][] by [@LaurenzV][])
- Breaking change: `WebGlRenderer::render` and `WebGlRenderer::render_to_atlas` now take `WebGlTextureBindings`, providing the external textures referenced by the scene. Pass `&WebGlTextureBindings::new()` if the scene does not use any. ([#1792][] by [@grebmeg][])
- Breaking change: Image-atlas allocation errors now include allocation and free-space diagnostics, while intermediate-texture failures use the new `IntermediateTextureError` variants for oversized allocations and texture-count limits. ([#1778][] by [@grebmeg][])

### Fixed

- Image tints with zero opacity now make the image fully transparent. ([#1791][] by [@LaurenzV][])
- WebGL image-atlas allocation and growth on Mali-G52 GPUs, avoiding application-not-responding errors. ([#1800][] by [@grebmeg][])

### Optimized

- WebGL intermediate textures are destroyed before resized replacements are allocated, reducing peak texture memory while render targets grow. ([#1781][] by [@LaurenzV][])

## [0.1.0][] - 2026-07-29

This release has an [MSRV][] of 1.88.

### Added

- Inverse blurred rounded rectangles through the new `invert` parameter on `Scene::fill_blurred_rounded_rect`, enabling inset box shadows. ([#1715][] by [@nicoburns][])
- `Resources::new_with_config` for configuring the image and glyph atlas used by resources. ([#1750][] by [@grebmeg][])
- `FilterPrimitive::DropShadowOnly` for rendering a drop shadow without compositing the original input over it. ([#1763][] by [@LaurenzV][])
- `Scene::reset_and_resize`, allowing a scene to be reused after its target size changes. ([#1769][] by [@LaurenzV][])

### Changed

- Breaking change: `WebGlRenderer::probe` is now asynchronous. It returns a `WebGlPendingProbe`, which can be polled with `try_finish`, avoiding a synchronous GPU-to-CPU readback stall and providing specific WebGL error codes on failure. ([#1671][], [#1735][] by [@LaurenzV][])
- Breaking change: Vello Hybrid's core scene scheduling and rendering architecture has been rewritten, removing coarse rasterization. The rewrite improves layer performance, allows filter output beyond the root viewport, reduces memory use for filter-heavy scenes, and removes the need for scene constraints. ([#1759][] by [@LaurenzV][])
- Breaking change: `RenderSettings` now contains `MemorySettings`, with separate image-atlas and intermediate-layer configuration. `SceneConstraints` and `RenderError::SlotsExhausted` have been removed, and `LayersConfig` allows applications to limit and tune intermediate textures. ([#1759][] by [@LaurenzV][])
- WebGL APIs are now exported whenever the `webgl` feature is enabled, including on non-Wasm targets. ([#1682][] by [@jesses-canva][])
- Relaxed the minimum versions of `wasm-bindgen`-related dependencies. ([#1683][] by [@LaurenzV][])

### Fixed

- Opaque image rendering when blend modes or masks are active. ([#1697][] by [@LaurenzV][])
- Gaussian blur strength changing abruptly at decimation thresholds. ([#1720][] by [@grebmeg][])
- Fast rectangle rendering at the maximum `u16` coordinates. ([#1666][] by [@dipeshbabu][])
- WebGL renderer resources are now deleted when dropped instead of relying on garbage collection, allowing renderers to be recreated safely after context loss. ([#1674][] by [@LaurenzV][])
- WebGL probing after rendering filter layers no longer corrupts the renderer's atlas state. ([#1734][] by [@LaurenzV][])

### Optimized

- Image-atlas textures and intermediate filter/layer textures are allocated lazily, substantially reducing initial GPU memory use for scenes that do not use images, layers, or filters. ([#1747][], [#1757][], [#1759][] by [@grebmeg][], [@LaurenzV][])
- Clip-path intersection by skipping rows outside the paths' shared vertical extent and locating the first relevant strip with binary search. ([#1690][], [#1691][] by [@LaurenzV][])
- Gradient lookup-table generation. ([#1726][] by [@LaurenzV][])
- Rendering opaque `peniko::ImageData` by computing and retaining its transparency hint during conversion. ([#1760][] by [@tronical][])

## [0.0.9][] - 2026-05-30

This release has an [MSRV][] of 1.88.

### Fixed

- WebGL rendering on certain Adreno GPUs. ([#1619][], [#1620][], [#1659][] by [@LaurenzV][], [@grebmeg][])
- Rendering of glyphs with complex paints. ([#1668][] by [@LaurenzV][])
- Rendering of paths with high winding counts. ([#1673][] by [@b0nes164][])

## [0.0.8][] - 2026-05-15

This release has an [MSRV][] of 1.88.

### Added

- Support for sampling from external textures through `Scene::draw_texture_rects`. ([#1552][] by [@tomcur][])
  
  This method can sample many regions from a texture at once, allowing the texture to be used as an atlas.
  You bind the textures at render-time (as opposed to scene-construction time), allowing drawing of quickly-changing textures without interning them into the renderer.

  External textures are not yet supported by the `webgl` backend.

- Bicubic image sampling in the sparse shader path, improving high-quality image rendering. ([#1557][] by [@waywardmonkeys][])
- `WebGlRenderer::probe` behind the `probe` feature to sanity-check WebGL device compatibility by rendering a small reference scene and comparing the output. ([#1596][] by [@LaurenzV][])
- `render_decoration` on glyph run builders for rendering text decorations with skip-ink behavior. ([#1592][] by [@taj-p][])
- Support for rendering VARC glyphs in text paths. ([#1594][] by [@nicoburns][], [@oscargus][])
- Support for synthetic font emboldening in text glyph runs via `GlyphRunBuilder::font_embolden` and `glifo::FontEmbolden`. ([#1628][] by [@jrmoulton][])

### Changed

- Updated `wgpu` to v29. ([#1534][] by [@nicoburns][])
- Migrated text rendering to `glifo`; renderer resources are now managed through `Resources`, which must be passed to text, image upload, and render operations such as `Scene::glyph_run`, `Renderer::render`, and `Renderer::upload_image`. ([#1562][] by [@LaurenzV][])
- Renamed image transparency hint APIs from `may_have_opacities` and `*_opacity_hint` to `may_have_transparency` and `*_transparency_hint`. ([#1613][] by [@upsuper][])
- `Renderer::destroy_image` and `WebGlRenderer::destroy_image` now take `&mut Resources` instead of `&mut ImageCache`, allowing clients to destroy images through the public resource container. ([#1580][] by [@LaurenzV][])
- `TextureBindings` now owns `wgpu::TextureView`s, allowing bindings to be held across frames more easily.
  `insert` now takes a `TextureView` by value and `remove` returns the removed view if present. ([#1639][] by [@tomcur][])

### Removed

- Experimental `vello_api` integration and `api` module. ([#1602][] by [@waywardmonkeys][])
- Support for recordings.
  This decision was made due to a number of downsides that came with the implementation.
  See the corresponding PR and Zulip thread for more information. ([#1611][] by [@LaurenzV][])

### Fixed

- Rendering of rare degenerate radial gradients with undefined regions, which should render those regions transparent. ([#1529][] by [@LaurenzV][])
- Incorrect rendering of filter layers containing nested clip layers. ([#1541][] by [@LaurenzV][])
- Incorrect rendering when a background-clearing optimization ran after filter layers had been used. ([#1526][] by [@LaurenzV][])
- `SceneConstraints::default_blending_only()` now allows non-default blend modes inside nested layers, while still rejecting them on the root layer. ([#1554][] by [@LaurenzV][])
- Draw ordering for fast-path strips scheduled after nested layers when default blending constraints are enabled. ([#1555][] by [@LaurenzV][], [@taj-p][])
- Atlas configuration is now clamped to backend device limits, allowing `RenderSettings::default()` and oversized atlas settings to work on more devices. ([#1568][] by [@LaurenzV][])
- `wgpu` render targets are cleared before drawing, fixing stale pixels in transparent or untouched regions when rendering into reused offscreen targets. ([#1572][] by [@waywardmonkeys][])
- Bicubic image sampling now clamps sparse shader results correctly, avoiding invalid premultiplied color values. ([#1573][] by [@waywardmonkeys][])
- Stroked glyph rendering when scaled glyph runs absorb the transform into the font size, preserving the expected stroke width. ([#1576][] by [@LaurenzV][])
- COLR glyph rendering, including glyphs using non-default blend modes. ([#1584][] by [@LaurenzV][])
- `WebGlRenderer` construction no longer leaks framebuffer state, fixing a panic when creating a second WebGL renderer. ([#1603][] by [@LaurenzV][])
- Blur and filter shader compatibility on some mobile GPUs. ([#1601][], [#1612][] by [@LaurenzV][])
- Feature gating so `probe` no longer enables `std`, and `std` no longer implies `wgpu`, preserving WebGL-only configurations. ([#1614][] by [@LaurenzV][])

### Optimized

- Native WebGL rendering now avoids an intermediate framebuffer, improving performance especially for low-complexity scenes. ([#1546][] by [@LaurenzV][])
- Rectangle rendering, including large anti-aliased rectangles, clipped axis-aligned `fill_rect` calls, and blurred rounded rectangles. ([#1565][], [#1586][] by [@LaurenzV][], [#1610][] by [@waywardmonkeys][])
- Filter rendering by applying scissor rectangles to filter passes and intermediate layers. ([#1566][], [#1567][] by [@LaurenzV][])
- Rendering of opaque strips by adding a front-to-back opaque pass with depth testing to reduce overdraw. ([#1577][] by [@taj-p][])

## [0.0.7][] - 2026-03-24

This release has an [MSRV][] of 1.92.

### Added

- Initial support for filter effects. ([#1494][] by [@LaurenzV][])
- `render_to_atlas` and `write_to_atlas` APIs for glyph caching. ([#1458][] by [@grebmeg][])
- `push_blend_layer`, `push_opacity_layer` and `push_mask_layer` methods to `Scene`. ([#1420][] by [@grebmeg][])
- Scene constraints for conditionally achieving better rendering performance. ([#1476][] by [@taj-p][])
- Support for image tinting. ([#1460][] by [@grebmeg][])

### Changed

- Updated `wgpu` to v28. ([#1492][] by [@xStrom][])
- Made text rendering an optional feature. ([#1455][] by [@LaurenzV][])

### Fixed

- Rendering artifacts sometimes present along the seam in sweep gradients by improving numerical robustness around the seam. ([#1352][] by [@tomcur][])
- Incorrect handling of scenes with many complex paints. ([#1467][] by [@taj-p][])

### Optimized

- Rendering axis-aligned rectangles. ([#1482][] by [@LaurenzV][])
- Rendering of opaque full-tile images. ([#1461][] by [@grebmeg][])
- Layer blending for src-over compositing. ([#1436][] by [@LaurenzV][])
- Coarse rasterization and scheduling of scenes without layers. ([#1454][] by [@LaurenzV][])
- Scenes with many changing gradients. ([#1496][] by [@LaurenzV][])

## [0.0.6][] - 2026-01-15

This release has an [MSRV][] of 1.88.

### Changed

- Breaking change: Updated Peniko to [v0.6.0](https://github.com/linebender/peniko/releases/tag/v0.6.0). ([#1349][] by [@DJMcNab][])
  - This also updates Kurbo to [v0.13.0](https://github.com/linebender/kurbo/releases/tag/v0.13.0).
- Upgraded Skrifa to v0.40.0. ([#1353][] by [@waywardmonkeys][])
- Upgraded Hashbrown to v0.16.1. ([#1354][] by [@waywardmonkeys][])

See also the [vello_cpu 0.0.6](../vello_cpu/CHANGELOG.md#006---2026-01-15) and [vello_common 0.0.6](../vello_common/CHANGELOG.md#006---2026-01-15) releases.

## [0.0.5][] - 2026-01-08

This release has an [MSRV][] of 1.88.

### Added

- The `Scene` now contains a `push_clip_path` and `pop_clip_path` method for performing non-isolated clipping. ([#1203][] by [@LaurenzV])

See also the [vello_cpu 0.0.5](../vello_cpu/CHANGELOG.md#005---2026-01-08) and [vello_common 0.0.5](../vello_common/CHANGELOG.md#005---2026-01-08) releases.

## [0.0.4][] - 2025-10-17

This release has an [MSRV][] of 1.86.

This is the initial release. No changelog was kept for this release.

See also the [vello_cpu 0.0.4](../vello_cpu/CHANGELOG.md#004---2025-10-17) and [vello_common 0.0.4](../vello_common/CHANGELOG.md#004---2025-10-17) releases.

[@AdrianEddy]: https://github.com/AdrianEddy
[@DJMcNab]: https://github.com/DJMcNab
[@b0nes164]: https://github.com/b0nes164
[@dipeshbabu]: https://github.com/dipeshbabu
[@grebmeg]: https://github.com/grebmeg
[@jesses-canva]: https://github.com/jesses-canva
[@jrmoulton]: https://github.com/jrmoulton
[@LaurenzV]: https://github.com/LaurenzV
[@nicoburns]: https://github.com/nicoburns
[@oscargus]: https://github.com/oscargus
[@taj-p]: https://github.com/taj-p
[@tomcur]: https://github.com/tomcur
[@tronical]: https://github.com/tronical
[@upsuper]: https://github.com/upsuper
[@waywardmonkeys]: https://github.com/waywardmonkeys
[@xStrom]: https://github.com/xStrom

[#1203]: https://github.com/linebender/vello/pull/1203
[#1349]: https://github.com/linebender/vello/pull/1349
[#1352]: https://github.com/linebender/vello/pull/1352
[#1353]: https://github.com/linebender/vello/pull/1353
[#1354]: https://github.com/linebender/vello/pull/1354
[#1420]: https://github.com/linebender/vello/pull/1420
[#1436]: https://github.com/linebender/vello/pull/1436
[#1454]: https://github.com/linebender/vello/pull/1454
[#1455]: https://github.com/linebender/vello/pull/1455
[#1458]: https://github.com/linebender/vello/pull/1458
[#1460]: https://github.com/linebender/vello/pull/1460
[#1461]: https://github.com/linebender/vello/pull/1461
[#1467]: https://github.com/linebender/vello/pull/1467
[#1476]: https://github.com/linebender/vello/pull/1476
[#1482]: https://github.com/linebender/vello/pull/1482
[#1492]: https://github.com/linebender/vello/pull/1492
[#1494]: https://github.com/linebender/vello/pull/1494
[#1496]: https://github.com/linebender/vello/pull/1496
[#1526]: https://github.com/linebender/vello/pull/1526
[#1529]: https://github.com/linebender/vello/pull/1529
[#1534]: https://github.com/linebender/vello/pull/1534
[#1541]: https://github.com/linebender/vello/pull/1541
[#1546]: https://github.com/linebender/vello/pull/1546
[#1554]: https://github.com/linebender/vello/pull/1554
[#1555]: https://github.com/linebender/vello/pull/1555
[#1552]: https://github.com/linebender/vello/pull/1552
[#1557]: https://github.com/linebender/vello/pull/1557
[#1562]: https://github.com/linebender/vello/pull/1562
[#1565]: https://github.com/linebender/vello/pull/1565
[#1566]: https://github.com/linebender/vello/pull/1566
[#1567]: https://github.com/linebender/vello/pull/1567
[#1568]: https://github.com/linebender/vello/pull/1568
[#1572]: https://github.com/linebender/vello/pull/1572
[#1573]: https://github.com/linebender/vello/pull/1573
[#1576]: https://github.com/linebender/vello/pull/1576
[#1577]: https://github.com/linebender/vello/pull/1577
[#1580]: https://github.com/linebender/vello/pull/1580
[#1584]: https://github.com/linebender/vello/pull/1584
[#1586]: https://github.com/linebender/vello/pull/1586
[#1592]: https://github.com/linebender/vello/pull/1592
[#1594]: https://github.com/linebender/vello/pull/1594
[#1596]: https://github.com/linebender/vello/pull/1596
[#1601]: https://github.com/linebender/vello/pull/1601
[#1602]: https://github.com/linebender/vello/pull/1602
[#1603]: https://github.com/linebender/vello/pull/1603
[#1610]: https://github.com/linebender/vello/pull/1610
[#1611]: https://github.com/linebender/vello/pull/1611
[#1612]: https://github.com/linebender/vello/pull/1612
[#1613]: https://github.com/linebender/vello/pull/1613
[#1614]: https://github.com/linebender/vello/pull/1614
[#1619]: https://github.com/linebender/vello/pull/1619
[#1620]: https://github.com/linebender/vello/pull/1620
[#1628]: https://github.com/linebender/vello/pull/1628
[#1639]: https://github.com/linebender/vello/pull/1639
[#1659]: https://github.com/linebender/vello/pull/1659
[#1668]: https://github.com/linebender/vello/pull/1668
[#1666]: https://github.com/linebender/vello/pull/1666
[#1671]: https://github.com/linebender/vello/pull/1671
[#1673]: https://github.com/linebender/vello/pull/1673
[#1674]: https://github.com/linebender/vello/pull/1674
[#1682]: https://github.com/linebender/vello/pull/1682
[#1683]: https://github.com/linebender/vello/pull/1683
[#1690]: https://github.com/linebender/vello/pull/1690
[#1691]: https://github.com/linebender/vello/pull/1691
[#1697]: https://github.com/linebender/vello/pull/1697
[#1715]: https://github.com/linebender/vello/pull/1715
[#1720]: https://github.com/linebender/vello/pull/1720
[#1726]: https://github.com/linebender/vello/pull/1726
[#1734]: https://github.com/linebender/vello/pull/1734
[#1735]: https://github.com/linebender/vello/pull/1735
[#1747]: https://github.com/linebender/vello/pull/1747
[#1750]: https://github.com/linebender/vello/pull/1750
[#1757]: https://github.com/linebender/vello/pull/1757
[#1759]: https://github.com/linebender/vello/pull/1759
[#1760]: https://github.com/linebender/vello/pull/1760
[#1763]: https://github.com/linebender/vello/pull/1763
[#1769]: https://github.com/linebender/vello/pull/1769
[#1778]: https://github.com/linebender/vello/pull/1778
[#1779]: https://github.com/linebender/vello/pull/1779
[#1781]: https://github.com/linebender/vello/pull/1781
[#1784]: https://github.com/linebender/vello/pull/1784
[#1791]: https://github.com/linebender/vello/pull/1791
[#1792]: https://github.com/linebender/vello/pull/1792
[#1794]: https://github.com/linebender/vello/pull/1794
[#1800]: https://github.com/linebender/vello/pull/1800
[#1801]: https://github.com/linebender/vello/pull/1801

[Unreleased]: https://github.com/linebender/vello/compare/sparse-strips-v0.2.0...HEAD
[0.2.0]: https://github.com/linebender/vello/compare/sparse-strips-v0.1.0...sparse-strips-v0.2.0
[0.1.0]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.9...sparse-strips-v0.1.0
[0.0.9]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.8...sparse-strips-v0.0.9
[0.0.8]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.7...sparse-strips-v0.0.8
[0.0.7]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.6...sparse-strips-v0.0.7
[0.0.6]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.5...sparse-strips-v0.0.6
[0.0.5]: https://github.com/linebender/vello/compare/sparse-strips-v0.0.4...sparse-strips-v0.0.5
[0.0.4]: https://github.com/linebender/vello/compare/ca6b1e4c7f5b0d95953c3b524f5d3952d5669c5a...sparse-strips-v0.0.4

[MSRV]: README.md#minimum-supported-rust-version-msrv
