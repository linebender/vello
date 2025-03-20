# Setup

In order to run the integration benchmarks, you need to download three SVG files:
- [ghostscript tiger](https://upload.wikimedia.org/wikipedia/commons/f/fd/Ghostscript_Tiger.svg), name it `gs.svg`.
- [paris_30k](https://github.com/google/forma/blob/main/assets/svgs/paris-30k.svg), name it `paris_30k.svg`.
- [coat of arms](https://en.m.wikipedia.org/wiki/File:Coat_of_Arms_of_the_Edinburgh_City_Council.svg), name it `coat_of_arms.svg`

Run the `dump` script for each of the files: `cargo run --bin dump {PATH_TO_FILE}`.

Then, you can for example simply run `cargo bench -- tile` to benchmark tiling with all 3 SVGs, or
`cargo bench -- tile/gs` to only benchmark for the ghostscript tiger. The same for all other parts of the pipeline!