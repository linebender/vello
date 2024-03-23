# Vello Encoding

This package contains types that represent data that [Vello] can render.

## Minimum supported Rust Version (MSRV)

This version of Vello Encoding has been verified to compile with **Rust 1.75** and later.

Future versions of Vello Encoding might increase the Rust version requirement.
It will not be treated as a breaking change and as such can even happen with small patch releases.

> [!TIP]
> As time has passed, some of Vello Encoding's dependencies could have released versions with a higher Rust requirement.
> If you encounter a compilation issue due to a dependency and don't want to upgrade your Rust toolchain, then you could downgrade the dependency.
> ```sh
> # Use the problematic dependency's name and version
> cargo update -p package_name --precise 0.1.1
> ```

[Vello]: https://github.com/linebender/vello
