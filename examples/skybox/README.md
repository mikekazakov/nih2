### Skybox Example

Building skybox by sampling the Hosek-Wilkie sky model and rendering it using the NIH2 software rasterizer.

Each frame it does the following:

- Shifts the Sun direction
- Updates five 512x512 skybox textures:
    - Initializes the sky model with the given Sun elevation, turbidity and ground albedo.
    - For each pixel:
        - Calculates theta and gamma.
        - Samples the model per each component.
        - Conditionally injects the Sun into the radiance values.
        - Compresses the HDR radiance with the Reinhard tone-mapping operator.
        - Applies gamma correction.
        - Adds a small noise for debanding.
    - Generates mipmaps for the texture.
- Renders the skybox as 12 triangles.
- Draws some flares depending on the Sun's screen position.

The code that builds the skybox is heavily SIMD-optimized, with computations performed per row in a branchless manner.
Rebuilding the skybox takes ~3.0 ms of frame time.

The example runs at ~150 FPS at 720p on an Apple M1 CPU:   
[![Watch the video](https://img.youtube.com/vi/KQ2byH93Rc0/hqdefault.jpg)](https://www.youtube.com/watch?v=KQ2byH93Rc0)



