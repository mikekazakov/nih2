### Particles Example

This example demonstrates a simple particle system that is set up, simulated, and rendered in about 150 lines of Rust
code.  
Each frame updates 1k particles, which are drawn as 2k triangles (6k vertices).  
The triangle rasterizer runs perspective-correct interpolation of colors and texture coordinates.  
The texture is sampled with mipmapping and bilinear interpolation.  
Sampled texture colors are then combined with interpolated vertex colors and additively blended with the background.  
On an Apple M1 CPU, it runs at ~50 fps at 720p:  
[![Watch the video](https://img.youtube.com/vi/qDWApw2w1VI/hqdefault.jpg)](https://www.youtube.com/watch?v=qDWApw2w1VI)
