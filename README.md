### NIH2

NIH2 is a recreational project where I play around with software rendering and learn Rust in the process.
The source code largely originates from [NIH1 written in C++](https://github.com/mikekazakov/nih),
but here it's "rewritten in Rust"(tm), optimized further and has vastly better test coverage.
It's not production-quality and has no practical purpose - it's just the result of tinkering and having fun.

#### Features:

- Multithreaded, tile-based triangle rasterization.
- Edge-functions and depth evaluation with fixed-point arithmetic.
- Watertight rasterization following the top-left fill rule.
- Normal and additive alpha-blending.
- Branchless texture sampling with nearest, bilinear and trilinear filtering.

#### Examples:

- `./examples/texture_filtering`  
  Drawing quads using different texture filtering in ~100 LoCs.
- [./examples/particles](examples/particles/README.md)  
  Rendering 1K particles in a single `main()` function in ~150 LoCs.
