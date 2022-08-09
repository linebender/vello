/** Automatically generated from pgpu-render/src/lib.rs with cbindgen. **/

#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

enum class PgpuBrushKind {
  Solid = 0,
};

enum class PgpuFill {
  NonZero = 0,
  EvenOdd = 1,
};

enum class PgpuPathVerb {
  MoveTo = 0,
  LineTo = 1,
  QuadTo = 2,
  CurveTo = 3,
  Close = 4,
};

/// Encoded (possibly color) outline for a glyph.
struct PgpuGlyph;

/// Context for loading and scaling glyphs.
struct PgpuGlyphContext;

/// Context for loading a scaling glyphs from a specific font.
struct PgpuGlyphProvider;

/// State and resources for rendering a scene.
struct PgpuRenderer;

/// Encoded streams and resources describing a vector graphics scene.
struct PgpuScene;

/// Builder for constructing an encoded scene.
struct PgpuSceneBuilder;

/// Encoded streams and resources describing a vector graphics scene fragment.
struct PgpuSceneFragment;

/// Affine transformation matrix.
struct PgpuTransform {
  float xx;
  float yx;
  float xy;
  float yy;
  float dx;
  float dy;
};

struct PgpuColor {
  uint8_t r;
  uint8_t g;
  uint8_t b;
  uint8_t a;
};

union PgpuBrushData {
  PgpuColor solid;
};

struct PgpuBrush {
  PgpuBrushKind kind;
  PgpuBrushData data;
};

struct PgpuPoint {
  float x;
  float y;
};

struct PgpuPathElement {
  PgpuPathVerb verb;
  PgpuPoint points[3];
};

struct PgpuPathIter {
  void *context;
  bool (*next_element)(void*, PgpuPathElement*);
};

/// Tag and value for a font variation axis.
struct PgpuFontVariation {
  /// Tag that specifies the axis.
  uint32_t tag;
  /// Requested setting for the axis.
  float value;
};

/// Description of a font.
struct PgpuFontDesc {
  /// Pointer to the context of the font file.
  const uint8_t *data;
  /// Size of the font file data in bytes.
  uintptr_t data_len;
  /// Index of the requested font in the font file.
  uint32_t index;
  /// Unique identifier for the font.
  uint64_t unique_id;
  /// Requested size in pixels per em unit. Set to 0.0 for
  /// unscaled outlines.
  float ppem;
  /// Pointer to array of font variation settings.
  const PgpuFontVariation *variations;
  /// Number of font variation settings.
  uintptr_t variations_len;
};

/// Rectangle defined by minimum and maximum points.
struct PgpuRect {
  float x0;
  float y0;
  float x1;
  float y1;
};

extern "C" {

#if defined(__APPLE__)
/// Creates a new piet-gpu renderer for the specified Metal device and
/// command queue.
///
/// device: MTLDevice*
/// queue: MTLCommandQueue*
PgpuRenderer *pgpu_renderer_new(void *device, void *queue);
#endif

#if defined(__APPLE__)
/// Renders a prepared scene into a texture target. Commands for rendering are
/// recorded into the specified command buffer. Returns an id representing
/// resources that may have been allocated during this process. After the
/// command buffer has been retired, call `pgpu_renderer_release` with this id
/// to drop any associated resources.
///
/// target: MTLTexture*
/// cmdbuf: MTLCommandBuffer*
uint32_t pgpu_renderer_render(PgpuRenderer *renderer,
                              const PgpuScene *scene,
                              void *target,
                              void *cmdbuf);
#endif

/// Releases the internal resources associated with the specified id from a
/// previous render operation.
void pgpu_renderer_release(PgpuRenderer *renderer, uint32_t id);

/// Destroys the piet-gpu renderer.
void pgpu_renderer_destroy(PgpuRenderer *renderer);

/// Creates a new, empty piet-gpu scene.
PgpuScene *pgpu_scene_new();

/// Destroys the piet-gpu scene.
void pgpu_scene_destroy(PgpuScene *scene);

/// Creates a new, empty piet-gpu scene fragment.
PgpuSceneFragment *pgpu_scene_fragment_new();

/// Destroys the piet-gpu scene fragment.
void pgpu_scene_fragment_destroy(PgpuSceneFragment *fragment);

/// Creates a new builder for filling a piet-gpu scene. The specified scene
/// should not be accessed while the builder is live.
PgpuSceneBuilder *pgpu_scene_builder_for_scene(PgpuScene *scene);

/// Creates a new builder for filling a piet-gpu scene fragment. The specified
/// scene fragment should not be accessed while the builder is live.
PgpuSceneBuilder *pgpu_scene_builder_for_fragment(PgpuSceneFragment *fragment);

/// Adds a glyph with the specified transform to the underlying scene.
void pgpu_scene_builder_add_glyph(PgpuSceneBuilder *builder,
                                  const PgpuGlyph *glyph,
                                  const PgpuTransform *transform);

/// Sets the current absolute transform for the scene builder.
void pgpu_scene_builder_transform(PgpuSceneBuilder *builder, const PgpuTransform *transform);

/// Fills a path using the specified fill style and brush. If the brush
/// parameter is nullptr, a solid color white brush will be used. The
/// brush_transform may be nullptr.
void pgpu_scene_builder_fill_path(PgpuSceneBuilder *builder,
                                  PgpuFill fill,
                                  const PgpuBrush *brush,
                                  const PgpuTransform *brush_transform,
                                  PgpuPathIter *path);

/// Appends a scene fragment to the underlying scene or fragment. The
/// transform parameter represents an absolute transform to apply to
/// the fragment. If it is nullptr, the fragment will be appended to
/// the scene with an assumed identity transform regardless of the
/// current transform state.
void pgpu_scene_builder_append_fragment(PgpuSceneBuilder *builder,
                                        const PgpuSceneFragment *fragment,
                                        const PgpuTransform *transform);

/// Finalizes the scene builder, making the underlying scene ready for
/// rendering. This takes ownership and consumes the builder.
void pgpu_scene_builder_finish(PgpuSceneBuilder *builder);

/// Creates a new context for loading glyph outlines.
PgpuGlyphContext *pgpu_glyph_context_new();

/// Destroys the glyph context.
void pgpu_glyph_context_destroy(PgpuGlyphContext *gcx);

/// Creates a new glyph provider for the specified glyph context and font
/// descriptor. May return nullptr if the font data is invalid. Only one glyph
/// provider may be live for a glyph context.
PgpuGlyphProvider *pgpu_glyph_provider_new(PgpuGlyphContext *gcx, const PgpuFontDesc *font);

/// Returns an encoded outline for the specified glyph provider and glyph id.
/// May return nullptr if the requested glyph is not available.
PgpuGlyph *pgpu_glyph_provider_get(PgpuGlyphProvider *provider, uint16_t gid);

/// Returns an encoded color outline for the specified glyph provider, color
/// palette index and glyph id. May return nullptr if the requested glyph is
/// not available.
PgpuGlyph *pgpu_glyph_provider_get_color(PgpuGlyphProvider *provider,
                                         uint16_t palette_index,
                                         uint16_t gid);

/// Destroys the glyph provider.
void pgpu_glyph_provider_destroy(PgpuGlyphProvider *provider);

/// Computes the bounding box for the glyph after applying the specified
/// transform.
PgpuRect pgpu_glyph_bbox(const PgpuGlyph *glyph, const float (*transform)[6]);

/// Destroys the glyph.
void pgpu_glyph_destroy(PgpuGlyph *glyph);

} // extern "C"
