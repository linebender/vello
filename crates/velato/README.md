# velato

This is an experimental [Lottie](https://airbnb.io/lottie) animation renderer built for vello. It is
not currently feature complete, but is capable of rendering a large number of Lottie animations.

## Example usage
```rust
let lottie_data = std::fs::read("path/to/lottie.json").unwrap();
let composition = velato::Composition::from_bytes(&lottie_data).unwrap();
let mut renderer = velato::Renderer::new();
let scene = vello::Scene::new();
let mut builder = vello::SceneBuilder::for_scene(&mut scene);
let time_secs = 1.0;
let transform = kurbo::Affine::IDENTITY;
let alpha = 1.0;
renderer.render(&composition, time_secs, transform, alpha, &mut builder);
```
