# piet-gpu-tests

This subdirectory contains a curated set of tests for GPU issues likely to affect piet-gpu compatibility or performance. To run, cd to the tests directory and do `cargo run --release`. There are a number of additional options, including:

* `--dx12` Prefer DX12 backend on windows.
* `--size {s,m,l}` Size of test to run.
* `--n_iter n` Number of iterations.
* `--verbose` Verbose output.

As usual, run `cargo run -- -h` for the current list.

Below is a description of individual tests.

## clear buffers

This is as simple as it says, it uses a compute shader to clear buffers. It's run first as a warmup, and is a simple test of raw memory bandwidth (reported as 4 byte elements/s).

## Prefix sum tests

There are several variations of the prefix sum test, first the [decoupled look-back] variant, then a more conservative tree reduction version. The decoupled look-back implemenation exercises advanced atomic features and depends on their correctness, including atomic coherence and correct scope of memory barriers.

None of the decoupled look-back tests are expected to pass on Metal, as that back-end lacks the appropriate barrier; the spirv-cross translation silently translates the GLSL version to a weaker one. All tests are expected to pass on both Vulkan and DX12.

The compatibility variant does all manipulation of the state buffer using non-atomic operations, with the buffer marked "volatile" and barriers to insure acquire/release ordering.

The atomic variant is similar, but uses atomicLoad and atomicStore (from the [memory scope semantics] extension to GLSL).

Finally, the vkmm (Vulkan memory model) variant uses explicit acquire and release semantics on the atomics instead of barriers, and only runs when the device reports that the memory model extension is available.

The tree reduction version of this test does not rely on advanced atomics and can be considered a baseline for both correctness and performance. The current implementation lacks configuration settings to handle odd-size buffers. On well-tuned hardware, the decoupled look-back implementation is expected to be 1.5x faster.

Note that the workgroup sizes and sequential iteration count parameters are hard-coded (and tuned for a desktop card I had handy). A useful future extension of this test suite would be iteration over several combinations of those parameters. (The main reason this is not done yet is that it would put a lot of strain on the shader build pipeline, and at the moment hand-editing the ninja file is adequate).

## Atomic tests

Decoupled look-back relies on the atomic message passing idiom; these tests exercise that in isolation.

The message passing tests basically do bunch of the basic message passing operation in parallel, and the "special sauce" is that the memory locations for both flags and data are permuted. That seems to do a lot better job finding violations than existing versions of the test.

The linked list test is mostly a bandwidth test of atomicExchange, and is a simplified version of what the coarse path rasterizer does in piet-gpu to build per-tile lists of path segments. The verification of the resulting lists is also a pretty good test of device scoped modification order (not that this is likely to fail).

## More tests

I'll be adding more tests specific to piet-gpu. I'm also open to tests being added here, feel free to file an issue.

[decoupled look-back]: https://raphlinus.github.io/gpu/2020/04/30/prefix-sum.html
[memory scope semantics]: https://github.com/KhronosGroup/GLSL/blob/master/extensions/khr/GL_KHR_memory_scope_semantics.txt
