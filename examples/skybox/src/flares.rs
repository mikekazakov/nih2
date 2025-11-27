use nih::math::*;
use nih::render::*;
use once_cell::sync::Lazy;
use std::sync::Arc;

#[derive(Clone)]
pub struct Flare {
    pub texture: Arc<Texture>,

    // Position = dist^1 * pos1 + dist^0 * pos0
    pub pos1: f32,
    pub pos0: f32,

    // Scale = dist^1 * scale1 + dist^0 * scale0
    pub scale1: f32,
    pub scale0: f32,

    // Angle = dist^1 * angle1 + dist^0 * angle0
    pub angle1: f32,
    pub angle0: f32,

    // Alpha = dist^1 * alpha1 + dist^0 * alpha0
    pub alpha1: f32,
    pub alpha0: f32,
}

static DUMMY_TEXTURE: Lazy<Arc<Texture>> = Lazy::new(|| {
    let texel = [42u8];
    let source = TextureSource { texels: &texel, width: 1, height: 1, format: TextureFormat::Grayscale };
    Texture::new(&source)
});

impl Default for Flare {
    fn default() -> Self {
        Self {
            texture: DUMMY_TEXTURE.clone(),
            pos1: 1.0,
            pos0: 0.0,
            scale1: 0.0,
            scale0: 0.2,
            angle1: 0.0,
            angle0: 0.0,
            alpha1: 0.0,
            alpha0: 1.0,
        }
    }
}

pub struct Flares {
    flares: Vec<Flare>,
}

impl Flares {
    pub fn new() -> Self {
        Self { flares: Vec::new() }
    }

    pub fn add_flare(&mut self, flare: Flare) {
        self.flares.push(flare);
    }

    pub fn commit(
        &self,
        viewport: Viewport,
        view_orientation: &Mat44,
        projection: &Mat44,
        sun_dir: Vec3,
        rasterizer: &mut Rasterizer,
    ) {
        let viewport_width: f32 = viewport.xmax as f32 - viewport.xmin as f32;
        let viewport_height: f32 = viewport.ymax as f32 - viewport.ymin as f32;
        let scale_y: f32 = if viewport_height > viewport_width {
            viewport_width / viewport_height
        } else {
            1.0
        };
        let scale_x: f32 = if viewport_width > viewport_height {
            viewport_height / viewport_width
        } else {
            1.0
        };

        let sun_elevation_intensity: f32 = (sun_dir.y * 4.0).min(1.0);
        if sun_elevation_intensity < 0.01 {
            return;
        }

        let sun_ndc: Vec4 = projection * view_orientation * sun_dir.as_point4();
        if sun_ndc.w < 0.0 {
            return;
        }

        let sun_ndc: Vec2 = (sun_ndc / sun_ndc.w).xy();

        let sun_screen_intensity: f32 =
            if sun_ndc.x >= -1.0 && sun_ndc.x <= 1.0 && sun_ndc.y >= -1.0 && sun_ndc.y <= 1.0 {
                1.0
            } else {
                1.0 - ((sun_ndc.x.abs().max(sun_ndc.y.abs())) - 1.0) * 4.0
            };

        let sun_intensity: f32 = sun_screen_intensity * sun_elevation_intensity;
        if sun_intensity < 0.01 {
            return;
        }

        let screen_center_ndc: Vec2 = Vec2::new(0.0, 0.0);
        let screen_flare_dir: Vec2 = (screen_center_ndc - sun_ndc).normalized();
        let screen_dist_ndc: f32 = (screen_center_ndc - sun_ndc).length();

        for flare in &self.flares {
            let scale: f32 = flare.scale1 * screen_dist_ndc + flare.scale0;
            if scale < 0.001 {
                continue;
            }
            let size_ndc_x: f32 = scale * scale_x;
            let size_ndc_y: f32 = scale * scale_y;
            let alpha: f32 = (flare.alpha1 * screen_dist_ndc + flare.alpha0) * sun_intensity;
            if alpha < 0.01 {
                continue;
            }
            let pos_offset: f32 = (flare.pos1 * screen_dist_ndc + flare.pos0);
            let pos: Vec2 = screen_center_ndc + screen_flare_dir * pos_offset;
            let angle: f32 = flare.angle1 * screen_dist_ndc + flare.angle0;

            rasterizer.commit(&RasterizationCommand {
                world_positions: &QUAD_POSITIONS,
                tex_coords: &QUAD_TEX_COORDS,
                texture: Some(flare.texture.clone()),
                sampling_filter: SamplerFilter::Bilinear,
                alpha_blending: AlphaBlendingMode::Additive,
                model: Mat34::translate(Vec3::new(pos.x, pos.y, 0.0))
                    * Mat34::scale_non_uniform(Vec3::new(size_ndc_x, size_ndc_y, 0.0))
                    * Mat34::rotate_xy(angle),
                color: Vec4::new(1.0, 1.0, 1.0, alpha.min(1.0)),
                ..Default::default()
            });
        }
    }
}

static QUAD_POSITIONS: [Vec3; 6] = [
    Vec3::new(-1.0, 1.0, 0.0),
    Vec3::new(-1.0, -1.0, 0.0),
    Vec3::new(1.0, 1.0, 0.0),
    Vec3::new(1.0, 1.0, 0.0),
    Vec3::new(-1.0, -1.0, 0.0),
    Vec3::new(1.0, -1.0, 0.0),
];

static QUAD_TEX_COORDS: [Vec2; 6] = [
    Vec2::new(0.0, 0.0),
    Vec2::new(0.0, 1.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.0, 1.0),
    Vec2::new(1.0, 1.0),
];
