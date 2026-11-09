# Twemoji Mozilla Subset

This folder contains a small subset of [Twemoji Mozilla](https://github.com/mozilla/twemoji-colr), a COLRv0 build of [Twemoji](https://github.com/twitter/twemoji).
The emoji art is licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) and the font build tooling under the Apache License 2.0, see [LICENSE.md](LICENSE.md).
We do not include the full font, because including the entire Emoji set would increase the repository size too much.
Included emoji are:

- ✅ Check Mark - \u{2705}/`:white_check_mark:`
- 👀 Eyes - \u{1f440}/`:eyes:`
- 🎉 Party Popper - \u{1f389}/`:party_popper:`
- 🤠 Face with Cowboy Hat - \u{1f920}/`cowboy_hat_face`

COLRv0 fonts have no `ClipList`, so every glyph in this font is rendered without a `ClipBox`, unlike the glyphs in the Noto Color Emoji subset.
See <https://github.com/linebender/vello/issues/1983>.

The font is `Twemoji.Mozilla.ttf` from the [v0.7.0 release](https://github.com/mozilla/twemoji-colr/releases/tag/v0.7.0) (SHA-256 `6d90152ee0d29e82fe2a87793af5aa4b7ad13e6538360889e141e81ed299ee8e`), subset with [fontTools](https://github.com/fonttools/fonttools) 4.65.0:

```sh
pyftsubset Twemoji.Mozilla.ttf --unicodes=U+2705,U+1F440,U+1F389,U+1F920 --output-file=Twemoji.Mozilla-Subset.ttf
```
