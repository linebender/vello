# Vello Image Filters

In 2d Rendering, you often need to process images in ways which aren't just drawing of paths.
For example, by blurring a background, zooming in, or inverting colors.
This folder provides implementations of several different image filters, using different backends.

These backends are:

- The "reference" CPU backend, found in [`vello_filters_cpu`].
  We intend to publish this, and use it in Vello CPU.
- GPU shader code, without a driver.
  We haven't yet decided how this will be distributed.
  This is not yet implemented.
- Utilities and pipelines for using the GPU shader with wgpu for simple cases.
  We intend to publish this crate, and might use it in Vello Hybrid.
  This is not yet implemented.

We intend to have GPU implementations which are able to be executed on WebGL, with more optimised versions (e.g. using compute shaders) for more powerful GPUs.

In this README, we use the "Svg" prefix to refer to filters from the Filter Effects Module Level 1, found at <https://drafts.fxtf.org/filter-effects/>.
The planned filters are (at least), and the implementations we have working are:

| Filter                              | CPU | GPU (planned) | wgpu (planned) |
| ----------------------------------- | --- | ------------- | -------------- |
| [Svg Gaussian Blur][feGaussianBlur] |     |               |                |

(Please let us know if this table is out-of-date)

Our implementations have been based on the following sources:

- resvg

Filters which don't operate on multiple pixels are generally out-of-scope for initial versions of this crate.
These include:

- [Svg Blend][feBlend]. This will generally be implemented as part of your renderer directly, as it is an entirely local operation.
- [Svg Composite][feComposite]. This will generally be implemented as part of a renderer directly, as it is entirely local. It's not clear how the arithmetic operation should be implemented.

<!-- We expect to also adapt code from `imageproc`, but have not done so yet.
This will impact the licenses this code is available under. -->

[feGaussianBlur]: https://drafts.fxtf.org/filter-effects/#feGaussianBlurElement
[feBlend]: https://drafts.fxtf.org/filter-effects/#feBlendElement
[feComposite]: https://drafts.fxtf.org/filter-effects/#feCompositeElement
