The `vello_shaders` crate provides a utility library to integrate the Vello shader modules into any
renderer project. The crate provides the necessary metadata to construct the individual compute
pipelines on any GPU API while leaving the responsibility of all API interactions (such as
resource management and command encoding) up to the client.

The shaders can be pre-compiled to any target shading language at build time based on feature flags.
Currently only WGSL and Metal Shading Language are supported.
