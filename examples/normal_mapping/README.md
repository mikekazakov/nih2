### Normal Mapping Example

This example draws a single quad with a texture and a normal map.  
Both albedo and normals are sampled with bilinear filtering.  
After conversion to world space, the normals are written into the normals buffer.  
A basic Blinn-Phong lighting is then applied in a deferred pass.  
On Apple M1 CPU it runs at ~100 fps with 720p resolution:  
[![Watch the video](https://img.youtube.com/vi/SQAomM3PrKE/hqdefault.jpg)](https://www.youtube.com/watch?v=SQAomM3PrKE)
