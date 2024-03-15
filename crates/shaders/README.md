# Vello Shaders

This is a utility library to help integrate the [Vello] shader modules into any renderer project.
It provides the necessary metadata to construct the individual compute pipelines on any GPU API while leaving the responsibility of all API interactions (such as resource management and command encoding) up to the client.

The shaders can be pre-compiled to any target shading language at build time based on feature flags.
Currently only WGSL and Metal Shading Language are supported.

## Minimum supported Rust Version

This version of Vello Shaders has been verified to compile with Rust 1.75 and later.

Note that since then, one of Vello Shaders' dependencies might have released a new version with a higher requirement.
Thus if you encounter a compilation issue due to a dependency, you should downgrade it via `cargo update -p package_name --precise 0.1.1` (with a real name and version).

Future versions of Vello Shaders might increase the Rust version requirement.
It will not be treated as a breaking change and as such can even happen with small patch releases.

[Vello]: https://github.com/linebender/vello
