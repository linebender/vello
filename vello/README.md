# vello

This crate is currently a highly experimental proof-of-concept port of the piet-gpu renderer to the WGSL shader language, so it could be run on WebGPU. Depending on how well it works out, it may become the authoritative source for piet-gpu.

The shaders are actually handlebars templates over WGSL, as it's important to share common data structures; it's likely we'll use the template mechanism to supply various parameters which are not supported by the WGSL language, for example to specify grayscale or RGBA buffer output for fine rasterization.

This crate also uses a very different approach to the GPU abstraction than piet-gpu. That is essentially a HAL that supports an immediate mode approach to creating resources and submitting commands. Here, we generate a `Recording`, which is basically a simple value type, then an `Engine` plays that recording to the actual GPU. The idea is that this can abstract easily over multiple GPU back-ends, without either the render logic needing to be polymorphic or having dynamic dispatch at the GPU abstraction. The goal is to be more agile.

Scene encoding is shared with piet-gpu, and currently uses piet-scene in the same repo with no changes.

This module is still an experimental work in progress. Contributions can be made with the same policy as the root repo, but expect things to change quickly.
