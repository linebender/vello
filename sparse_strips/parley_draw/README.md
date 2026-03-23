## Parley Draw 

Parley Draw provides APIs for efficiently rendering glyphs and paint styles like underline.

### Goals

Parley Draw is under rapid development. Consider it experimental for now. Its goals are to:

- Provide an API surface that accepts glyphs and their positions and renders them to a surface.
- Cache those glyphs so that repeated renders of a glyph are fast.
- Support rendering paint styles like underline, strikethrough, and brush color.
- Share expensive structs and data between the shaper and renderer like the hinting instance and hinted advance.

## Minimum supported Rust Version (MSRV)

This version of Parley has been verified to compile with **Rust 1.88** and later.

Future versions of Parley might increase the Rust version requirement.
It will not be treated as a breaking change and as such can even happen with small patch releases.

## Community

Discussion of Parley development happens in the [Linebender Zulip](https://xi.zulipchat.com/), specifically the [#parley channel](https://xi.zulipchat.com/#narrow/channel/205635-parley).
All public content can be read without logging in.

Contributions are welcome by pull request. The [Rust code of conduct] applies.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache 2.0 license, shall be licensed as noted in the [License](#license) section, without any additional terms or conditions.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

[Rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
