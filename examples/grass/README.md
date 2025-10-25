### Grass Example

Real-time grass simulation and rendering using the NIH2 software rasterizer.

First, it generates about 300 X-shaped grass bushes with randomized placement, size, and orientation.  
All world positions, normals, and texture coordinates are allocated upfront and filled according to the generated
bushes.

Each frame, the top four vertices of each bush are animated.  
Their displacement is determined by a Perlin noise value multiplied by the wind direction and strength.  
A time-dependent wind offset is applied when sampling the noise.  
Normals are also tilted based on the scaled vertex displacement.

Once the vertices are updated, the grass triangles are rendered using a single rasterization command.  
During rasterization, the grass texture is sampled with bilinear filtering; texels are alpha-tested and blended using
normal alpha blending.

A deferred lighting pass is then applied per pixel.  
It uses per-fragment albedo and normals to apply simple Lambertian shading.  
Finally, the tiled buffer is flattened and blitted to the window.

Runs at ~60 FPS at 720p on an Apple M1 CPU:   
[![Watch the video](https://img.youtube.com/vi/tgiEEwdLdvk/hqdefault.jpg)](https://www.youtube.com/watch?v=tgiEEwdLdvk)
