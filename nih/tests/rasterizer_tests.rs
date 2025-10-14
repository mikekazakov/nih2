use nih::math::*;
use nih::render::*;

macro_rules! assert_rgba_eq {
    ($left:expr, $right:expr, $tol:expr $(,)?) => {{
        let l = $left;
        let r = $right;
        let tol: i16 = $tol as i16;

        let dr = (l.r as i16 - r.r as i16).abs();
        let dg = (l.g as i16 - r.g as i16).abs();
        let db = (l.b as i16 - r.b as i16).abs();
        let da = (l.a as i16 - r.a as i16).abs();

        if dr > tol || dg > tol || db > tol || da > tol {
            panic!("assertion failed: left != right within tol={}\n  left: {:?}\n right: {:?}", tol, l, r);
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba, RgbaImage};
    use rstest::rstest;
    use std::path::Path;

    fn reference_path<P: AsRef<Path>>(reference: P) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/reference_images/")
            .join(reference)
    }

    fn save_albedo_next_to_reference<P: AsRef<Path>>(result: &Buffer<u32>, reference: P) {
        let mut actual_path = reference_path(reference);
        actual_path.set_extension("actual.png");

        let raw_rgba: Vec<u8> = result
            .elems
            .iter()
            .flat_map(|&pixel| {
                let bytes = pixel.to_le_bytes();
                [bytes[0], bytes[1], bytes[2], bytes[3]]
            })
            .collect();
        let img1: ImageBuffer<Rgba<u8>, _> =
            ImageBuffer::from_raw(result.width as u32, result.height as u32, raw_rgba).unwrap();
        img1.save(actual_path).unwrap();
    }

    fn save_normals_next_to_reference<P: AsRef<Path>>(result: &Buffer<u32>, reference: P) {
        let mut actual_path = reference_path(reference);
        actual_path.set_extension("actual.png");

        let raw_rgba: Vec<u8> = result
            .elems
            .iter()
            .flat_map(|&pixel| {
                let bytes = pixel.to_le_bytes();
                [bytes[0], bytes[1], bytes[2], 255]
            })
            .collect();
        let img1: ImageBuffer<Rgba<u8>, _> =
            ImageBuffer::from_raw(result.width as u32, result.height as u32, raw_rgba).unwrap();
        img1.save(actual_path).unwrap();
    }

    fn save_depth_next_to_reference<P: AsRef<Path>>(result: &Buffer<u16>, reference: P) {
        let mut actual_path = reference_path(reference);
        actual_path.set_extension("actual.png");

        let raw_rgba: Vec<u8> = result
            .elems
            .iter()
            .flat_map(|&pixel| {
                let bytes = pixel.to_le_bytes();
                [bytes[1], bytes[0], 0, 255]
            })
            .collect();
        let img1: ImageBuffer<Rgba<u8>, _> =
            ImageBuffer::from_raw(result.width as u32, result.height as u32, raw_rgba).unwrap();
        img1.save(actual_path).unwrap();
    }

    fn compare_albedo_against_reference<P: AsRef<Path>>(result: &Buffer<u32>, reference: P) -> bool {
        const ERROR_TOLERANCE: u8 = 2; // acceptable difference per channel, 2 ~= 1%
        let reference_path = reference_path(reference);

        let raw_rgba: Vec<u8> = result
            .elems
            .iter()
            .flat_map(|&pixel| {
                let bytes = pixel.to_le_bytes();
                [bytes[0], bytes[1], bytes[2], bytes[3]]
            })
            .collect();
        let img1: ImageBuffer<Rgba<u8>, _> =
            ImageBuffer::from_raw(result.width as u32, result.height as u32, raw_rgba).unwrap();

        let img2: RgbaImage = image::open(reference_path).unwrap().into_rgba8();

        if img1.dimensions() != img2.dimensions() {
            return false;
        }

        img1.pixels().zip(img2.pixels()).all(|(p1, p2)| {
            let diff_r = (p1[0] as i16 - p2[0] as i16).abs() as u8;
            let diff_g = (p1[1] as i16 - p2[1] as i16).abs() as u8;
            let diff_b = (p1[2] as i16 - p2[2] as i16).abs() as u8;
            let diff_a = (p1[3] as i16 - p2[3] as i16).abs() as u8;
            diff_r <= ERROR_TOLERANCE
                && diff_g <= ERROR_TOLERANCE
                && diff_b <= ERROR_TOLERANCE
                && diff_a <= ERROR_TOLERANCE
        })
    }

    fn compare_normals_against_reference<P: AsRef<Path>>(result: &Buffer<u32>, reference: P) -> bool {
        const ERROR_TOLERANCE: u8 = 2; // acceptable difference per channel, 2 ~= 1%
        let reference_path = reference_path(reference);

        let raw_rgba: Vec<u8> = result
            .elems
            .iter()
            .flat_map(|&pixel| {
                let bytes = pixel.to_le_bytes();
                [bytes[0], bytes[1], bytes[2], bytes[3]]
            })
            .collect();
        let img1: ImageBuffer<Rgba<u8>, _> =
            ImageBuffer::from_raw(result.width as u32, result.height as u32, raw_rgba).unwrap();

        let img2: RgbaImage = image::open(reference_path).unwrap().into_rgba8();

        if img1.dimensions() != img2.dimensions() {
            return false;
        }

        img1.pixels().zip(img2.pixels()).all(|(p1, p2)| {
            let diff_r = (p1[0] as i16 - p2[0] as i16).abs() as u8;
            let diff_g = (p1[1] as i16 - p2[1] as i16).abs() as u8;
            let diff_b = (p1[2] as i16 - p2[2] as i16).abs() as u8;
            diff_r <= ERROR_TOLERANCE && diff_g <= ERROR_TOLERANCE && diff_b <= ERROR_TOLERANCE
        })
    }

    fn compare_depth_against_reference<P: AsRef<Path>>(result: &Buffer<u16>, reference: P) -> bool {
        let reference_path = reference_path(reference);
        let reference_image: RgbaImage = image::open(reference_path).unwrap().into_rgba8();
        if reference_image.width() != result.width as u32 || reference_image.height() != result.height as u32 {
            return false;
        }

        for (x, y, pixel) in reference_image.enumerate_pixels() {
            // reconstruct depth from R and G components
            let reference_depth = (((pixel[0] as u16) << 8) | (pixel[1] as u16)) as i32;
            let actual_depth = result.at(x as u16, y as u16) as i32;
            let diff = (reference_depth - actual_depth).abs();
            if diff > 100 {
                return false; // 100 / 65535 ~= 0.15% error tolerance
            }
        }
        true
    }

    fn assert_albedo_against_reference<P: AsRef<Path>>(result: &Buffer<u32>, reference: P) {
        let equal = compare_albedo_against_reference(result, &reference);
        if !equal {
            save_albedo_next_to_reference(result, &reference);
        }
        assert!(equal);
    }

    fn assert_depth_against_reference<P: AsRef<Path>>(result: &Buffer<u16>, reference: P) {
        let equal = compare_depth_against_reference(result, &reference);
        if !equal {
            save_depth_next_to_reference(result, &reference);
        }
        assert!(equal);
    }

    fn assert_normals_against_reference<P: AsRef<Path>>(result: &Buffer<u32>, reference: P) {
        let equal = compare_normals_against_reference(result, &reference);
        if !equal {
            save_normals_next_to_reference(result, &reference);
        }
        assert!(equal);
    }

    fn render_to_64x64_albedo(command: &RasterizationCommand) -> Buffer<u32> {
        let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(64, 64);
        color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
        let mut rasterizer = Rasterizer::new();
        rasterizer.setup(Viewport::new(0, 0, 64, 64));
        rasterizer.commit(&command);
        rasterizer.draw(&mut Framebuffer { color_buffer: Some(&mut color_buffer), ..Framebuffer::default() });
        color_buffer.as_flat_buffer()
    }

    fn render_to_64x64_albedo_wbg(command: &RasterizationCommand) -> Buffer<u32> {
        let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(64, 64);
        color_buffer.fill(RGBA::new(255, 255, 255, 255).to_u32());
        let mut rasterizer = Rasterizer::new();
        rasterizer.setup(Viewport::new(0, 0, 64, 64));
        rasterizer.commit(&command);
        rasterizer.draw(&mut Framebuffer { color_buffer: Some(&mut color_buffer), ..Framebuffer::default() });
        color_buffer.as_flat_buffer()
    }

    fn render_to_256x256_albedo(command: &RasterizationCommand) -> Buffer<u32> {
        let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(256, 256);
        color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
        let mut rasterizer = Rasterizer::new();
        rasterizer.setup(Viewport::new(0, 0, 256, 256));
        rasterizer.commit(&command);
        rasterizer.draw(&mut Framebuffer { color_buffer: Some(&mut color_buffer), ..Framebuffer::default() });
        color_buffer.as_flat_buffer()
    }

    // render_to_64x64_normals?

    fn render_to_64x64_depth(command: &RasterizationCommand) -> Buffer<u16> {
        // let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(64, 64);
        // color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
        let mut depth_buffer = TiledBuffer::<u16, 64, 64>::new(64, 64);
        depth_buffer.fill(65535);
        let mut framebuffer = Framebuffer::default();
        // framebuffer.color_buffer = Some(&mut color_buffer);
        framebuffer.depth_buffer = Some(&mut depth_buffer);

        let mut rasterizer = Rasterizer::new();
        rasterizer.setup(Viewport::new(0, 0, 64, 64));
        rasterizer.commit(&command);
        rasterizer.draw(&mut framebuffer);

        depth_buffer.as_flat_buffer()
    }

    #[rstest]
    #[case(Vec4::new(0.0, 0.0, 0.0, 1.0), "rasterizer/triangle/simple/black.png")]
    #[case(Vec4::new(1.0, 1.0, 1.0, 1.0), "rasterizer/triangle/simple/white.png")]
    #[case(Vec4::new(1.0, 0.0, 0.0, 1.0), "rasterizer/triangle/simple/red.png")]
    #[case(Vec4::new(0.0, 1.0, 0.0, 1.0), "rasterizer/triangle/simple/green.png")]
    #[case(Vec4::new(0.0, 0.0, 1.0, 1.0), "rasterizer/triangle/simple/blue.png")]
    #[case(Vec4::new(1.0, 1.0, 0.0, 1.0), "rasterizer/triangle/simple/yellow.png")]
    #[case(Vec4::new(1.0, 0.0, 1.0, 1.0), "rasterizer/triangle/simple/purple.png")]
    #[case(Vec4::new(0.0, 1.0, 1.0, 1.0), "rasterizer/triangle/simple/cyan.png")]
    fn triangle_simple(#[case] color: Vec4, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(0.0, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, -0.5, 0.0)],
            color,
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }

    #[rstest]
    #[case(Vec2::new(-1.0, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/top_0.png")]
    #[case(Vec2::new(-0.75, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/top_1.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/top_2.png")]
    #[case(Vec2::new(-0.25, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/top_3.png")]
    #[case(Vec2::new(0.0, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/top_4.png")]
    #[case(Vec2::new(0.25, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/top_5.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/top_6.png")]
    #[case(Vec2::new(0.75, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/top_7.png")]
    #[case(Vec2::new(1.0, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/top_8.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-1.0, -0.5), Vec2::new(0.5, 0.5), "rasterizer/triangle/orientation/bottom_0.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.75, -0.5), Vec2::new(0.5, 0.5), "rasterizer/triangle/orientation/bottom_1.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, 0.5), "rasterizer/triangle/orientation/bottom_2.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.25, -0.5), Vec2::new(0.5, 0.5), "rasterizer/triangle/orientation/bottom_3.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(0.0, -0.5), Vec2::new(0.5, 0.5), "rasterizer/triangle/orientation/bottom_4.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(0.25, -0.5), Vec2::new(0.5, 0.5), "rasterizer/triangle/orientation/bottom_5.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(0.5, -0.5), Vec2::new(0.5, 0.5), "rasterizer/triangle/orientation/bottom_6.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(0.75, -0.5), Vec2::new(0.5, 0.5), "rasterizer/triangle/orientation/bottom_7.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(1.0, -0.5), Vec2::new(0.5, 0.5), "rasterizer/triangle/orientation/bottom_8.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, 1.0), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/left_0.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, 0.75), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/left_1.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, 0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/left_2.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, 0.25), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/left_3.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, 0.0), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/left_4.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, -0.25), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/left_5.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/left_6.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, -0.75), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/left_7.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, -1.0), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/left_8.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, 1.0), "rasterizer/triangle/orientation/right_0.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, 0.75), "rasterizer/triangle/orientation/right_1.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, 0.5), "rasterizer/triangle/orientation/right_2.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, 0.25), "rasterizer/triangle/orientation/right_3.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, 0.0), "rasterizer/triangle/orientation/right_4.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.25), "rasterizer/triangle/orientation/right_5.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/right_6.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.75), "rasterizer/triangle/orientation/right_7.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -1.0), "rasterizer/triangle/orientation/right_8.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, 0.25), Vec2::new(-0.25, -0.25), "rasterizer/triangle/orientation/other_0.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.75, 0.25), Vec2::new(-0.25, -0.25), "rasterizer/triangle/orientation/other_1.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, 0.0), Vec2::new(-0.25, -0.25), "rasterizer/triangle/orientation/other_2.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.75, 0.0), Vec2::new(-0.25, -0.25), "rasterizer/triangle/orientation/other_3.png")]
    #[case(Vec2::new(0.25, 0.5), Vec2::new(-0.5, 0.25), Vec2::new(-0.25, -0.25), "rasterizer/triangle/orientation/other_4.png")]
    #[case(Vec2::new(0.25, 0.75), Vec2::new(-0.5, 0.25), Vec2::new(-0.25, -0.25), "rasterizer/triangle/orientation/other_5.png")]
    #[case(Vec2::new(0.25, 1.0), Vec2::new(-0.5, 0.25), Vec2::new(-0.25, -0.25), "rasterizer/triangle/orientation/other_6.png")]
    #[case(Vec2::new(0.5, 1.0), Vec2::new(-0.5, 0.25), Vec2::new(-0.25, -0.25), "rasterizer/triangle/orientation/other_7.png")]
    #[case(Vec2::new(0.5, 0.75), Vec2::new(-0.5, 0.25), Vec2::new(-0.25, -0.25), "rasterizer/triangle/orientation/other_8.png")]
    fn triangle_orientation(#[case] v0: Vec2, #[case] v1: Vec2, #[case] v2: Vec2, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(v0.x, v0.y, 0.0), Vec3::new(v1.x, v1.y, 0.0), Vec3::new(v2.x, v2.y, 0.0)],
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }

    #[rstest]
    #[case(Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.75), Vec2::new(1.0, -1.0), "rasterizer/triangle/thin/00.png")]
    #[case(Vec2::new(-0.75, 1.0), Vec2::new(-1.0, 1.0), Vec2::new(1.0, -1.0), "rasterizer/triangle/thin/01.png")]
    #[case(Vec2::new(1.0, 1.0), Vec2::new(0.75, 1.0), Vec2::new(-1.0, -1.0), "rasterizer/triangle/thin/02.png")]
    #[case(Vec2::new(1.0, 0.75), Vec2::new(1.0, 1.0), Vec2::new(-1.0, -1.0), "rasterizer/triangle/thin/03.png")]
    #[case(Vec2::new(1.0, -0.75), Vec2::new(1.0, -1.0), Vec2::new(-1.0, 1.0), "rasterizer/triangle/thin/04.png")]
    #[case(Vec2::new(1.0, -1.0), Vec2::new(0.75, -1.0), Vec2::new(-1.0, 1.0), "rasterizer/triangle/thin/05.png")]
    #[case(Vec2::new(-0.75, -1.0), Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0), "rasterizer/triangle/thin/06.png")]
    #[case(Vec2::new(-1.0, -1.0), Vec2::new(-1.0, -0.75), Vec2::new(1.0, 1.0), "rasterizer/triangle/thin/07.png")]
    #[case(Vec2::new(0.25, 1.0), Vec2::new(0.0, 1.0), Vec2::new(0.0, -1.0), "rasterizer/triangle/thin/08.png")]
    #[case(Vec2::new(0.25, 1.0), Vec2::new(-0.25, 1.0), Vec2::new(0.0, -1.0), "rasterizer/triangle/thin/09.png")]
    #[case(Vec2::new(0.0, 1.0), Vec2::new(-0.25, 1.0), Vec2::new(0.0, -1.0), "rasterizer/triangle/thin/10.png")]
    #[case(Vec2::new(-1.0, 0.25), Vec2::new(-1.0, 0.0), Vec2::new(1.0, 0.0), "rasterizer/triangle/thin/11.png")]
    #[case(Vec2::new(-1.0, 0.25), Vec2::new(-1.0, -0.25), Vec2::new(1.0, 0.0), "rasterizer/triangle/thin/12.png")]
    #[case(Vec2::new(-1.0, 0.0), Vec2::new(-1.0, -0.25), Vec2::new(1.0, 0.0), "rasterizer/triangle/thin/13.png")]
    #[case(Vec2::new(-0.25, -1.0), Vec2::new(0.0, -1.0), Vec2::new(0.0, 1.0), "rasterizer/triangle/thin/14.png")]
    #[case(Vec2::new(-0.25, -1.0), Vec2::new(0.25, -1.0), Vec2::new(0.0, 1.0), "rasterizer/triangle/thin/15.png")]
    #[case(Vec2::new(0.0, -1.0), Vec2::new(0.25, -1.0), Vec2::new(0.0, 1.0), "rasterizer/triangle/thin/16.png")]
    #[case(Vec2::new(-1.0, 0.0), Vec2::new(1.0, -0.25), Vec2::new(1.0, 0.0), "rasterizer/triangle/thin/17.png")]
    #[case(Vec2::new(-1.0, 0.0), Vec2::new(1.0, -0.25), Vec2::new(1.0, 0.25), "rasterizer/triangle/thin/18.png")]
    #[case(Vec2::new(-1.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.25), "rasterizer/triangle/thin/19.png")]
    fn triangle_thin(#[case] v0: Vec2, #[case] v1: Vec2, #[case] v2: Vec2, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(v0.x, v0.y, 0.0), Vec3::new(v1.x, v1.y, 0.0), Vec3::new(v2.x, v2.y, 0.0)],
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }

    #[rstest]
    #[case(Vec2::new(0.600891411, 0.600891411), Vec2::new(-0.600891411, -0.600891411), Vec2::new(0.600891411, -0.600891411), "rasterizer/triangle/fract/01.png")]
    #[case(Vec2::new(-0.600891411, 0.600891411), Vec2::new(-0.600891411, -0.600891411), Vec2::new(0.600891411, 0.600891411), "rasterizer/triangle/fract/02.png")]
    fn triangle_fract(#[case] v0: Vec2, #[case] v1: Vec2, #[case] v2: Vec2, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(v0.x, v0.y, 0.0), Vec3::new(v1.x, v1.y, 0.0), Vec3::new(v2.x, v2.y, 0.0)],
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_256x256_albedo(&command), filename);
    }

    #[rstest]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_00.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 0.75), "rasterizer/interpolation/depth/large_01.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 0.5), "rasterizer/interpolation/depth/large_02.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 0.25), "rasterizer/interpolation/depth/large_03.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 0.0), "rasterizer/interpolation/depth/large_04.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, -0.25), "rasterizer/interpolation/depth/large_05.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, -0.5), "rasterizer/interpolation/depth/large_06.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, -0.75), "rasterizer/interpolation/depth/large_07.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, -1.0), "rasterizer/interpolation/depth/large_08.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 0.75), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_09.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 0.5), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_10.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 0.25), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_11.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_12.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, -0.25), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_13.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, -0.5), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_14.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, -0.75), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_15.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_16.png")]
    #[case(Vec3::new(-1.0, 1.0, 0.75), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_17.png")]
    #[case(Vec3::new(-1.0, 1.0, 0.5), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_18.png")]
    #[case(Vec3::new(-1.0, 1.0, 0.25), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_19.png")]
    #[case(Vec3::new(-1.0, 1.0, 0.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_20.png")]
    #[case(Vec3::new(-1.0, 1.0, -0.25), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_21.png")]
    #[case(Vec3::new(-1.0, 1.0, -0.5), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_22.png")]
    #[case(Vec3::new(-1.0, 1.0, -0.75), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_23.png")]
    #[case(Vec3::new(-1.0, 1.0, -1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_24.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, -1.0), "rasterizer/interpolation/depth/large_25.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_00.png")]
    #[case(Vec3::new(-0.75, -0.75, 0.75), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_01.png")]
    #[case(Vec3::new(-0.75, -0.75, 0.5), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_02.png")]
    #[case(Vec3::new(-0.75, -0.75, 0.25), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_03.png")]
    #[case(Vec3::new(-0.75, -0.75, 0.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_04.png")]
    #[case(Vec3::new(-0.75, -0.75, -0.25), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_05.png")]
    #[case(Vec3::new(-0.75, -0.75, -0.5), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_06.png")]
    #[case(Vec3::new(-0.75, -0.75, -0.75), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_07.png")]
    #[case(Vec3::new(-0.75, -0.75, -1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_08.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 0.75), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_09.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 0.5), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_10.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 0.25), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_11.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 0.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_12.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, -0.25), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_13.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, -0.5), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_14.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, -0.75), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_15.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, -1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_16.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 0.75), "rasterizer/interpolation/depth/tilted_17.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 0.5), "rasterizer/interpolation/depth/tilted_18.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 0.25), "rasterizer/interpolation/depth/tilted_19.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 0.0), "rasterizer/interpolation/depth/tilted_20.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, -0.25), "rasterizer/interpolation/depth/tilted_21.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, -0.5), "rasterizer/interpolation/depth/tilted_22.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, -0.75), "rasterizer/interpolation/depth/tilted_23.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, -1.0), "rasterizer/interpolation/depth/tilted_24.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 0.0), Vec3::new(0.5, 0.75, -1.0), "rasterizer/interpolation/depth/tilted_25.png")]
    fn depth_interpolation(#[case] v0: Vec3, #[case] v1: Vec3, #[case] v2: Vec3, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[v0, v1, v2],
            projection: Mat44::orthographic(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
            ..Default::default()
        };
        assert_depth_against_reference(&render_to_64x64_depth(&command), filename);
    }

    #[rstest]
    #[case(Vec3::new(-0.75, -0.75, -1.5), Vec3::new(0.75, -0.75, -1.5), Vec3::new(0.0, 0.75, -1.5), "rasterizer/interpolation/color/simple_0.png")]
    #[case(Vec3::new(-1.75, -1.75, -3.5), Vec3::new(0.75, -0.75, -1.5), Vec3::new(0.0, 0.75, -1.5), "rasterizer/interpolation/color/simple_1.png")]
    #[case(Vec3::new(-3.75, -3.75, -7.5), Vec3::new(0.75, -0.75, -1.5), Vec3::new(0.0, 0.75, -1.5), "rasterizer/interpolation/color/simple_2.png")]
    #[case(Vec3::new(-0.75, -0.75, -1.5), Vec3::new(1.75, -1.75, -3.5), Vec3::new(0.0, 0.75, -1.5), "rasterizer/interpolation/color/simple_3.png")]
    #[case(Vec3::new(-0.75, -0.75, -1.5), Vec3::new(3.75, -3.75, -7.5), Vec3::new(0.0, 0.75, -1.5), "rasterizer/interpolation/color/simple_4.png")]
    #[case(Vec3::new(-0.75, -0.75, -1.5), Vec3::new(0.75, -0.75, -1.5), Vec3::new(0.0, 1.75, -3.5), "rasterizer/interpolation/color/simple_5.png")]
    #[case(Vec3::new(-0.75, -0.75, -1.5), Vec3::new(0.75, -0.75, -1.5), Vec3::new(0.0, 3.75, -7.5), "rasterizer/interpolation/color/simple_6.png")]
    fn color_interpolation_simple(#[case] v0: Vec3, #[case] v1: Vec3, #[case] v2: Vec3, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[v0, v1, v2],
            colors: &[Vec4::new(1.0, 0.0, 0.0, 1.0), Vec4::new(0.0, 1.0, 0.0, 1.0), Vec4::new(0.0, 0.0, 1.0, 1.0)],
            projection: Mat44::perspective(0.1, 10.0, std::f32::consts::PI / 3.0, 1.),
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }

    #[rstest]
    #[case(
        Vec4::new(1.0, 0.5, 0.0, 1.0),
        Vec4::new(0.0, 1.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
        "rasterizer/interpolation/color/mix_0.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 1.0, 0.5, 1.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
        "rasterizer/interpolation/color/mix_1.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 1.0, 0.0, 1.0),
        Vec4::new(0.5, 0.0, 1.0, 1.0),
        "rasterizer/interpolation/color/mix_2.png"
    )]
    #[case(
        Vec4::new(1.0, 0.2, 0.1, 1.0),
        Vec4::new(0.0, 1.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
        "rasterizer/interpolation/color/mix_3.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::new(0.2, 1.0, 0.1, 1.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
        "rasterizer/interpolation/color/mix_4.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 1.0, 0.0, 1.0),
        Vec4::new(0.2, 0.1, 1.0, 1.0),
        "rasterizer/interpolation/color/mix_5.png"
    )]
    #[case(
        Vec4::new(1.0, 0.2, 0.1, 1.0),
        Vec4::new(0.2, 1.0, 0.1, 1.0),
        Vec4::new(0.2, 0.1, 1.0, 1.0),
        "rasterizer/interpolation/color/mix_6.png"
    )]
    fn color_interpolation_mix(#[case] c0: Vec4, #[case] c1: Vec4, #[case] c2: Vec4, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(-0.75, -0.75, -1.5), Vec3::new(0.75, -0.75, -1.5), Vec3::new(0.0, 0.75, -1.5)],
            colors: &[c0, c1, c2],
            projection: Mat44::perspective(0.1, 10.0, std::f32::consts::PI / 3.0, 1.),
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }

    #[rstest]
    #[case(Viewport::new(0, 0, 64, 64), "rasterizer/viewport/0.png")]
    #[case(Viewport::new(0, 0, 32, 32), "rasterizer/viewport/1.png")]
    #[case(Viewport::new(32, 0, 64, 32), "rasterizer/viewport/2.png")]
    #[case(Viewport::new(0, 32, 32, 64), "rasterizer/viewport/3.png")]
    #[case(Viewport::new(32, 32, 64, 64), "rasterizer/viewport/4.png")]
    #[case(Viewport::new(0, 0, 32, 64), "rasterizer/viewport/5.png")]
    #[case(Viewport::new(32, 0, 64, 64), "rasterizer/viewport/6.png")]
    #[case(Viewport::new(0, 0, 64, 32), "rasterizer/viewport/7.png")]
    #[case(Viewport::new(0, 32, 64, 64), "rasterizer/viewport/8.png")]
    fn viewport(#[case] v: Viewport, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(0.0, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, -0.5, 0.0)],
            ..Default::default()
        };
        let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(64, 64);
        color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
        let mut framebuffer = Framebuffer::default();
        framebuffer.color_buffer = Some(&mut color_buffer);
        let mut rasterizer = Rasterizer::new();
        rasterizer.setup(v);
        rasterizer.commit(&command);
        rasterizer.draw(&mut framebuffer);
        assert_albedo_against_reference(&color_buffer.as_flat_buffer(), filename);
    }

    #[rstest]
    #[case(256, 256, Vec2::new(0.0, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/tiling/256x256_00.png")]
    #[case(256, 256, Vec2::new(-0.5, 0.75), Vec2::new(-0.75, -0.75), Vec2::new(-0.25, -0.75), "rasterizer/tiling/256x256_01.png")]
    #[case(256, 256, Vec2::new(0.5, 0.75), Vec2::new(0.25, -0.75), Vec2::new(0.75, -0.75), "rasterizer/tiling/256x256_02.png")]
    #[case(256, 256, Vec2::new(-0.75, 0.75), Vec2::new(-0.75, 0.25), Vec2::new(0.75, 0.5), "rasterizer/tiling/256x256_03.png")]
    #[case(256, 256, Vec2::new(-0.75, -0.25), Vec2::new(-0.75, -0.75), Vec2::new(0.75, -0.5), "rasterizer/tiling/256x256_04.png")]
    #[case(256, 256, Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.0), Vec2::new(1.0, -1.0), "rasterizer/tiling/256x256_05.png")]
    #[case(256, 256, Vec2::new(-1.0, 0.0), Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0), "rasterizer/tiling/256x256_06.png")]
    #[case(256, 256, Vec2::new(-1.0, 1.0), Vec2::new(-1.0, -2.0), Vec2::new(1.0, 1.0), "rasterizer/tiling/256x256_07.png")]
    #[case(256, 256, Vec2::new(1.0, 1.25), Vec2::new(-1.0, 0.0), Vec2::new(1.0, -1.25), "rasterizer/tiling/256x256_08.png")]
    #[case(256, 256, Vec2::new(0.25, 0.75), Vec2::new(-0.75, -0.25), Vec2::new(-0.25, -0.25), "rasterizer/tiling/256x256_09.png")]
    #[case(256, 256, Vec2::new(-0.85, 0.75), Vec2::new(0.65, -0.35), Vec2::new(0.85, -0.25), "rasterizer/tiling/256x256_10.png")]
    #[case(141, 79, Vec2::new(0.0, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/tiling/141x79_00.png")]
    #[case(141, 79, Vec2::new(-0.5, 0.75), Vec2::new(-0.75, -0.75), Vec2::new(-0.25, -0.75), "rasterizer/tiling/141x79_01.png")]
    #[case(141, 79, Vec2::new(0.5, 0.75), Vec2::new(0.25, -0.75), Vec2::new(0.75, -0.75), "rasterizer/tiling/141x79_02.png")]
    #[case(141, 79, Vec2::new(-0.75, 0.75), Vec2::new(-0.75, 0.25), Vec2::new(0.75, 0.5), "rasterizer/tiling/141x79_03.png")]
    #[case(141, 79, Vec2::new(-0.75, -0.25), Vec2::new(-0.75, -0.75), Vec2::new(0.75, -0.5), "rasterizer/tiling/141x79_04.png")]
    #[case(141, 79, Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.0), Vec2::new(1.0, -1.0), "rasterizer/tiling/141x79_05.png")]
    #[case(141, 79, Vec2::new(-1.0, 0.0), Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0), "rasterizer/tiling/141x79_06.png")]
    #[case(141, 79, Vec2::new(-1.0, 1.0), Vec2::new(-1.0, -2.0), Vec2::new(1.0, 1.0), "rasterizer/tiling/141x79_07.png")]
    #[case(141, 79, Vec2::new(1.0, 1.25), Vec2::new(-1.0, 0.0), Vec2::new(1.0, -1.25), "rasterizer/tiling/141x79_08.png")]
    #[case(141, 79, Vec2::new(0.25, 0.75), Vec2::new(-0.75, -0.25), Vec2::new(-0.25, -0.25), "rasterizer/tiling/141x79_09.png")]
    #[case(141, 79, Vec2::new(-0.85, 0.75), Vec2::new(0.65, -0.35), Vec2::new(0.85, -0.25), "rasterizer/tiling/141x79_10.png")]
    #[case(65, 65, Vec2::new(0.0, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/tiling/65x65_00.png")]
    #[case(65, 65, Vec2::new(-0.5, 0.75), Vec2::new(-0.75, -0.75), Vec2::new(-0.25, -0.75), "rasterizer/tiling/65x65_01.png")]
    #[case(65, 65, Vec2::new(0.5, 0.75), Vec2::new(0.25, -0.75), Vec2::new(0.75, -0.75), "rasterizer/tiling/65x65_02.png")]
    #[case(65, 65, Vec2::new(-0.75, 0.75), Vec2::new(-0.75, 0.25), Vec2::new(0.75, 0.5), "rasterizer/tiling/65x65_03.png")]
    #[case(65, 65, Vec2::new(-0.75, -0.25), Vec2::new(-0.75, -0.75), Vec2::new(0.75, -0.5), "rasterizer/tiling/65x65_04.png")]
    #[case(65, 65, Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.0), Vec2::new(1.0, -1.0), "rasterizer/tiling/65x65_05.png")]
    #[case(65, 65, Vec2::new(-1.0, 0.0), Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0), "rasterizer/tiling/65x65_06.png")]
    #[case(65, 65, Vec2::new(-1.0, 1.0), Vec2::new(-1.0, -2.0), Vec2::new(1.0, 1.0), "rasterizer/tiling/65x65_07.png")]
    #[case(65, 65, Vec2::new(1.0, 1.25), Vec2::new(-1.0, 0.0), Vec2::new(1.0, -1.25), "rasterizer/tiling/65x65_08.png")]
    #[case(65, 65, Vec2::new(0.25, 0.75), Vec2::new(-0.75, -0.25), Vec2::new(-0.25, -0.25), "rasterizer/tiling/65x65_09.png")]
    #[case(65, 65, Vec2::new(-0.85, 0.75), Vec2::new(0.65, -0.35), Vec2::new(0.85, -0.25), "rasterizer/tiling/65x65_10.png")]
    #[case(1024, 1024, Vec2::new(0.0, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/tiling/1024x1024_00.png")]
    #[case(1024, 1024, Vec2::new(-0.5, 0.75), Vec2::new(-0.75, -0.75), Vec2::new(-0.25, -0.75), "rasterizer/tiling/1024x1024_01.png")]
    #[case(1024, 1024, Vec2::new(0.5, 0.75), Vec2::new(0.25, -0.75), Vec2::new(0.75, -0.75), "rasterizer/tiling/1024x1024_02.png")]
    #[case(1024, 1024, Vec2::new(-0.75, 0.75), Vec2::new(-0.75, 0.25), Vec2::new(0.75, 0.5), "rasterizer/tiling/1024x1024_03.png")]
    #[case(1024, 1024, Vec2::new(-0.75, -0.25), Vec2::new(-0.75, -0.75), Vec2::new(0.75, -0.5), "rasterizer/tiling/1024x1024_04.png")]
    #[case(1024, 1024, Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.0), Vec2::new(1.0, -1.0), "rasterizer/tiling/1024x1024_05.png")]
    #[case(1024, 1024, Vec2::new(-1.0, 0.0), Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0), "rasterizer/tiling/1024x1024_06.png")]
    #[case(1024, 1024, Vec2::new(-1.0, 1.0), Vec2::new(-1.0, -2.0), Vec2::new(1.0, 1.0), "rasterizer/tiling/1024x1024_07.png")]
    #[case(1024, 1024, Vec2::new(1.0, 1.25), Vec2::new(-1.0, 0.0), Vec2::new(1.0, -1.25), "rasterizer/tiling/1024x1024_08.png")]
    #[case(1024, 1024, Vec2::new(0.25, 0.75), Vec2::new(-0.75, -0.25), Vec2::new(-0.25, -0.25), "rasterizer/tiling/1024x1024_09.png")]
    #[case(1024, 1024, Vec2::new(-0.85, 0.75), Vec2::new(0.65, -0.35), Vec2::new(0.85, -0.25), "rasterizer/tiling/1024x1024_10.png")]
    fn tiling(
        #[case] width: u16,
        #[case] height: u16,
        #[case] v0: Vec2,
        #[case] v1: Vec2,
        #[case] v2: Vec2,
        #[case] filename: &str,
    ) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(v0.x, v0.y, 0.0), Vec3::new(v1.x, v1.y, 0.0), Vec3::new(v2.x, v2.y, 0.0)],
            ..Default::default()
        };
        let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(width, height);
        color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
        let mut framebuffer = Framebuffer::default();
        framebuffer.color_buffer = Some(&mut color_buffer);
        let mut rasterizer = Rasterizer::new();
        rasterizer.setup(Viewport::new(0, 0, width, height));
        rasterizer.commit(&command);
        rasterizer.draw(&mut framebuffer);
        assert_albedo_against_reference(&color_buffer.as_flat_buffer(), filename);
    }

    #[rstest]
    #[case(
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, 1.0),
        "rasterizer/interpolation/normal/simple_0.png"
    )]
    #[case(
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        "rasterizer/interpolation/normal/simple_1.png"
    )]
    #[case(
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        "rasterizer/interpolation/normal/simple_2.png"
    )]
    #[case(
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, -1.0),
        "rasterizer/interpolation/normal/simple_3.png"
    )]
    #[case(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        "rasterizer/interpolation/normal/simple_4.png"
    )]
    #[case(
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        "rasterizer/interpolation/normal/simple_5.png"
    )]
    #[case(
        Vec3::new(0.3, 0.3, 0.9),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, 1.0),
        "rasterizer/interpolation/normal/simple_6.png"
    )]
    #[case(
        Vec3::new(-0.5, 0.5, 0.8),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, 1.0),
        "rasterizer/interpolation/normal/simple_7.png"
    )]
    #[case(
        Vec3::new(-0.5, 0.7, 0.6),
        Vec3::new(0.2, -0.4, 0.9),
        Vec3::new(0.9, 0.2, -0.3),
        "rasterizer/interpolation/normal/simple_8.png"
    )]
    fn normal_interpolation_simple(#[case] n0: Vec3, #[case] n1: Vec3, #[case] n2: Vec3, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(0.0, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, -0.5, 0.0)],
            normals: &[n0, n1, n2],
            ..Default::default()
        };
        let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(64, 64);
        color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
        let mut normal_buffer = TiledBuffer::<u32, 64, 64>::new(64, 64);
        normal_buffer.fill(0);
        let mut framebuffer = Framebuffer::default();
        framebuffer.color_buffer = Some(&mut color_buffer);
        framebuffer.normal_buffer = Some(&mut normal_buffer);
        let mut rasterizer = Rasterizer::new();
        rasterizer.setup(Viewport::new(0, 0, 64, 64));
        rasterizer.commit(&command);
        rasterizer.draw(&mut framebuffer);
        assert_normals_against_reference(&normal_buffer.as_flat_buffer(), filename);
    }

    fn checkerboard_rgb_texture_32x32() -> std::sync::Arc<Texture> {
        let width = 32;
        let height = 32;
        let mut texels = vec![0u8; width * height * 3];
        for y in 0..width {
            for x in 0..height {
                let offset = ((y * width + x) * 3) as usize;
                if (x + y) % 2 == 0 {
                    texels[offset] = (x * 8) as u8; // R
                    texels[offset + 1] = (y * 8) as u8; // G
                    texels[offset + 2] = ((62 - x - y) * 4) as u8; // B
                } else {
                    texels[offset] = 127;
                    texels[offset + 1] = 127;
                    texels[offset + 2] = 127;
                }
            }
        }
        let source =
            TextureSource { texels: &texels, width: width as u32, height: height as u32, format: TextureFormat::RGB };
        Texture::new(&source)
    }

    #[rstest]
    #[case(&[Vec3::new(-0.5, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, 0.5, 0.0),
             Vec3::new(0.5, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, -0.5, 0.0),],
           &[Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0),
             Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0),],
        "rasterizer/texturing/nearest_0.png"
    )]
    #[case(&[Vec3::new(-1.0, 1.0, 0.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 0.0),
             Vec3::new(1.0, 1.0, 0.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0),],
           &[Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0),
             Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0),],
        "rasterizer/texturing/nearest_1.png"
    )]
    #[case(&[Vec3::new(0.0, 1.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0),
             Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0),],
           &[Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0),
             Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0),],
        "rasterizer/texturing/nearest_2.png"
    )]
    #[case(&[Vec3::new(0.0, 1.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0),
             Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0),],
           &[Vec2::new(1.0, 0.0), Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0),
             Vec2::new(1.0, 1.0), Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0),],
        "rasterizer/texturing/nearest_3.png"
    )]
    #[case(&[Vec3::new(-1.0, 0.5, 0.0), Vec3::new(-1.0, -0.5, 0.0), Vec3::new(1.0, 0.5, 0.0),
             Vec3::new(1.0, 0.5, 0.0), Vec3::new(-1.0, -0.5, 0.0), Vec3::new(1.0, -0.5, 0.0),],
           &[Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0),
             Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0),],
        "rasterizer/texturing/nearest_4.png"
    )]
    #[case(&[Vec3::new(-0.5, 1.0, 0.0), Vec3::new(-0.5, -1.0, 0.0), Vec3::new(0.5, 1.0, 0.0),
             Vec3::new(0.5, 1.0, 0.0), Vec3::new(-0.5, -1.0, 0.0), Vec3::new(0.5, -1.0, 0.0),],
           &[Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0),
             Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0),],
        "rasterizer/texturing/nearest_5.png"
    )]
    #[case(&[Vec3::new(-1.0, 1.0, 0.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 0.0),
             Vec3::new(1.0, 1.0, 0.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0),],
           &[Vec2::new(0.0, 0.0), Vec2::new(0.0, 2.0), Vec2::new(2.0, 0.0),
             Vec2::new(2.0, 0.0), Vec2::new(0.0, 2.0), Vec2::new(2.0, 2.0),],
        "rasterizer/texturing/nearest_6.png"
    )]
    #[case(&[Vec3::new(-1.0, 1.0, 0.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 0.0),
             Vec3::new(1.0, 1.0, 0.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0),],
           &[Vec2::new(-0.5, -0.5), Vec2::new(-0.5, 1.5), Vec2::new(1.5, -0.5),
             Vec2::new(1.5, -0.5), Vec2::new(-0.5, 1.5), Vec2::new(1.5, 1.5),],
        "rasterizer/texturing/nearest_7.png"
    )]
    fn texturing_nearest(#[case] world_positions: &[Vec3], #[case] tex_coords: &[Vec2], #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions,
            tex_coords,
            texture: Some(checkerboard_rgb_texture_32x32()),
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }

    #[rstest]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0)],
        "rasterizer/texturing/mip_selection_00.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, -0.6), Vec2::new(0.6, 1.0)],
        "rasterizer/texturing/mip_selection_01.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, -0.4), Vec2::new(0.4, 1.0)],
        "rasterizer/texturing/mip_selection_02.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.0), Vec2::new(0.0, 1.0)],
        "rasterizer/texturing/mip_selection_03.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.2), Vec2::new(-0.2, 1.0)],
        "rasterizer/texturing/mip_selection_04.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.3), Vec2::new(-0.3, 1.0)],
        "rasterizer/texturing/mip_selection_05.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.5), Vec2::new(-0.5, 1.0)],
        "rasterizer/texturing/mip_selection_06.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.6), Vec2::new(-0.6, 1.0)],
        "rasterizer/texturing/mip_selection_07.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.65), Vec2::new(-0.65, 1.0)],
        "rasterizer/texturing/mip_selection_08.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.75), Vec2::new(-0.75, 1.0)],
        "rasterizer/texturing/mip_selection_09.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.87), Vec2::new(-0.87, 1.0)],
        "rasterizer/texturing/mip_selection_10.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.93), Vec2::new(-0.93, 1.0)],
        "rasterizer/texturing/mip_selection_11.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.96), Vec2::new(-0.96, 1.0)],
        "rasterizer/texturing/mip_selection_12.png"
    )]
    fn texturing_mip_selection(#[case] positions: &[Vec2], #[case] filename: &str) {
        let texture = Texture::new(&TextureSource {
            texels: &vec![255u8; 64 * 64],
            width: 64,
            height: 64,
            format: TextureFormat::Grayscale,
        });
        let command = RasterizationCommand {
            world_positions: &[
                Vec3::new(positions[0].x, positions[0].y, 0.0),
                Vec3::new(positions[1].x, positions[1].y, 0.0),
                Vec3::new(positions[2].x, positions[2].y, 0.0),
            ],
            tex_coords: &[Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0)],
            texture: Some(texture),
            sampling_filter: SamplerFilter::DebugMip,
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }

    #[rstest]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        "rasterizer/alpha_blend/vert_simple_00.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        "rasterizer/alpha_blend/vert_simple_01.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        "rasterizer/alpha_blend/vert_simple_02.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        "rasterizer/alpha_blend/vert_simple_03.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        "rasterizer/alpha_blend/vert_simple_04.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        "rasterizer/alpha_blend/vert_simple_05.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        "rasterizer/alpha_blend/vert_simple_06.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        "rasterizer/alpha_blend/vert_simple_07.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        "rasterizer/alpha_blend/vert_simple_08.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        "rasterizer/alpha_blend/vert_simple_09.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/vert_simple_10.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 1.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
        "rasterizer/alpha_blend/vert_simple_11.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.9),
        Vec4::new(0.0, 1.0, 0.0, 0.9),
        Vec4::new(0.0, 0.0, 1.0, 0.9),
        "rasterizer/alpha_blend/vert_simple_12.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.8),
        Vec4::new(0.0, 1.0, 0.0, 0.8),
        Vec4::new(0.0, 0.0, 1.0, 0.8),
        "rasterizer/alpha_blend/vert_simple_13.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.7),
        Vec4::new(0.0, 1.0, 0.0, 0.7),
        Vec4::new(0.0, 0.0, 1.0, 0.7),
        "rasterizer/alpha_blend/vert_simple_14.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.6),
        Vec4::new(0.0, 1.0, 0.0, 0.6),
        Vec4::new(0.0, 0.0, 1.0, 0.6),
        "rasterizer/alpha_blend/vert_simple_15.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.5),
        Vec4::new(0.0, 1.0, 0.0, 0.5),
        Vec4::new(0.0, 0.0, 1.0, 0.5),
        "rasterizer/alpha_blend/vert_simple_16.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.4),
        Vec4::new(0.0, 1.0, 0.0, 0.4),
        Vec4::new(0.0, 0.0, 1.0, 0.4),
        "rasterizer/alpha_blend/vert_simple_17.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.3),
        Vec4::new(0.0, 1.0, 0.0, 0.3),
        Vec4::new(0.0, 0.0, 1.0, 0.3),
        "rasterizer/alpha_blend/vert_simple_18.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.2),
        Vec4::new(0.0, 1.0, 0.0, 0.2),
        Vec4::new(0.0, 0.0, 1.0, 0.2),
        "rasterizer/alpha_blend/vert_simple_19.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.1),
        Vec4::new(0.0, 1.0, 0.0, 0.1),
        Vec4::new(0.0, 0.0, 1.0, 0.1),
        "rasterizer/alpha_blend/vert_simple_20.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        "rasterizer/alpha_blend/vert_simple_21.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/vert_simple_22.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/vert_simple_23.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        "rasterizer/alpha_blend/vert_simple_24.png"
    )]
    fn alpha_blend_simple(#[case] c0: Vec4, #[case] c1: Vec4, #[case] c2: Vec4, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(0.0, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, -0.5, 0.0)],
            colors: &[c0, c1, c2],
            alpha_blending: AlphaBlendingMode::Normal,
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }

    #[rstest]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
        "rasterizer/alpha_blend/vert_simple_wbg_00.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.9),
        Vec4::new(0.0, 0.0, 0.0, 0.9),
        Vec4::new(0.0, 0.0, 0.0, 0.9),
        "rasterizer/alpha_blend/vert_simple_wbg_01.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.8),
        Vec4::new(0.0, 0.0, 0.0, 0.8),
        Vec4::new(0.0, 0.0, 0.0, 0.8),
        "rasterizer/alpha_blend/vert_simple_wbg_02.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.7),
        Vec4::new(0.0, 0.0, 0.0, 0.7),
        Vec4::new(0.0, 0.0, 0.0, 0.7),
        "rasterizer/alpha_blend/vert_simple_wbg_03.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.6),
        Vec4::new(0.0, 0.0, 0.0, 0.6),
        Vec4::new(0.0, 0.0, 0.0, 0.6),
        "rasterizer/alpha_blend/vert_simple_wbg_04.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.5),
        Vec4::new(0.0, 0.0, 0.0, 0.5),
        Vec4::new(0.0, 0.0, 0.0, 0.5),
        "rasterizer/alpha_blend/vert_simple_wbg_05.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.4),
        Vec4::new(0.0, 0.0, 0.0, 0.4),
        Vec4::new(0.0, 0.0, 0.0, 0.4),
        "rasterizer/alpha_blend/vert_simple_wbg_06.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.3),
        Vec4::new(0.0, 0.0, 0.0, 0.3),
        Vec4::new(0.0, 0.0, 0.0, 0.3),
        "rasterizer/alpha_blend/vert_simple_wbg_07.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.2),
        Vec4::new(0.0, 0.0, 0.0, 0.2),
        Vec4::new(0.0, 0.0, 0.0, 0.2),
        "rasterizer/alpha_blend/vert_simple_wbg_08.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.1),
        Vec4::new(0.0, 0.0, 0.0, 0.1),
        Vec4::new(0.0, 0.0, 0.0, 0.1),
        "rasterizer/alpha_blend/vert_simple_wbg_09.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 0.0),
        "rasterizer/alpha_blend/vert_simple_wbg_10.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 1.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
        "rasterizer/alpha_blend/vert_simple_wbg_11.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.9),
        Vec4::new(0.0, 1.0, 0.0, 0.9),
        Vec4::new(0.0, 0.0, 1.0, 0.9),
        "rasterizer/alpha_blend/vert_simple_wbg_12.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.8),
        Vec4::new(0.0, 1.0, 0.0, 0.8),
        Vec4::new(0.0, 0.0, 1.0, 0.8),
        "rasterizer/alpha_blend/vert_simple_wbg_13.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.7),
        Vec4::new(0.0, 1.0, 0.0, 0.7),
        Vec4::new(0.0, 0.0, 1.0, 0.7),
        "rasterizer/alpha_blend/vert_simple_wbg_14.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.6),
        Vec4::new(0.0, 1.0, 0.0, 0.6),
        Vec4::new(0.0, 0.0, 1.0, 0.6),
        "rasterizer/alpha_blend/vert_simple_wbg_15.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.5),
        Vec4::new(0.0, 1.0, 0.0, 0.5),
        Vec4::new(0.0, 0.0, 1.0, 0.5),
        "rasterizer/alpha_blend/vert_simple_wbg_16.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.4),
        Vec4::new(0.0, 1.0, 0.0, 0.4),
        Vec4::new(0.0, 0.0, 1.0, 0.4),
        "rasterizer/alpha_blend/vert_simple_wbg_17.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.3),
        Vec4::new(0.0, 1.0, 0.0, 0.3),
        Vec4::new(0.0, 0.0, 1.0, 0.3),
        "rasterizer/alpha_blend/vert_simple_wbg_18.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.2),
        Vec4::new(0.0, 1.0, 0.0, 0.2),
        Vec4::new(0.0, 0.0, 1.0, 0.2),
        "rasterizer/alpha_blend/vert_simple_wbg_19.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.1),
        Vec4::new(0.0, 1.0, 0.0, 0.1),
        Vec4::new(0.0, 0.0, 1.0, 0.1),
        "rasterizer/alpha_blend/vert_simple_wbg_20.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        "rasterizer/alpha_blend/vert_simple_wbg_21.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 0.0),
        "rasterizer/alpha_blend/vert_simple_wbg_22.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 0.0, 0.0),
        "rasterizer/alpha_blend/vert_simple_wbg_23.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
        "rasterizer/alpha_blend/vert_simple_wbg_24.png"
    )]
    fn alpha_blend_simple_wbg(#[case] c0: Vec4, #[case] c1: Vec4, #[case] c2: Vec4, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(0.0, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, -0.5, 0.0)],
            colors: &[c0, c1, c2],
            alpha_blending: AlphaBlendingMode::Normal,
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo_wbg(&command), filename);
    }

    #[rstest]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        "rasterizer/alpha_blend/tex_red_solid_00.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        "rasterizer/alpha_blend/tex_red_solid_01.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        "rasterizer/alpha_blend/tex_red_solid_02.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        "rasterizer/alpha_blend/tex_red_solid_03.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        "rasterizer/alpha_blend/tex_red_solid_04.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        "rasterizer/alpha_blend/tex_red_solid_05.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        "rasterizer/alpha_blend/tex_red_solid_06.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        "rasterizer/alpha_blend/tex_red_solid_07.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        "rasterizer/alpha_blend/tex_red_solid_08.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        "rasterizer/alpha_blend/tex_red_solid_09.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_red_solid_10.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 1.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
        "rasterizer/alpha_blend/tex_red_solid_11.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.9),
        Vec4::new(0.0, 1.0, 0.0, 0.9),
        Vec4::new(0.0, 0.0, 1.0, 0.9),
        "rasterizer/alpha_blend/tex_red_solid_12.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.8),
        Vec4::new(0.0, 1.0, 0.0, 0.8),
        Vec4::new(0.0, 0.0, 1.0, 0.8),
        "rasterizer/alpha_blend/tex_red_solid_13.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.7),
        Vec4::new(0.0, 1.0, 0.0, 0.7),
        Vec4::new(0.0, 0.0, 1.0, 0.7),
        "rasterizer/alpha_blend/tex_red_solid_14.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.6),
        Vec4::new(0.0, 1.0, 0.0, 0.6),
        Vec4::new(0.0, 0.0, 1.0, 0.6),
        "rasterizer/alpha_blend/tex_red_solid_15.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.5),
        Vec4::new(0.0, 1.0, 0.0, 0.5),
        Vec4::new(0.0, 0.0, 1.0, 0.5),
        "rasterizer/alpha_blend/tex_red_solid_16.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.4),
        Vec4::new(0.0, 1.0, 0.0, 0.4),
        Vec4::new(0.0, 0.0, 1.0, 0.4),
        "rasterizer/alpha_blend/tex_red_solid_17.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.3),
        Vec4::new(0.0, 1.0, 0.0, 0.3),
        Vec4::new(0.0, 0.0, 1.0, 0.3),
        "rasterizer/alpha_blend/tex_red_solid_18.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.2),
        Vec4::new(0.0, 1.0, 0.0, 0.2),
        Vec4::new(0.0, 0.0, 1.0, 0.2),
        "rasterizer/alpha_blend/tex_red_solid_19.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.1),
        Vec4::new(0.0, 1.0, 0.0, 0.1),
        Vec4::new(0.0, 0.0, 1.0, 0.1),
        "rasterizer/alpha_blend/tex_red_solid_20.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_red_solid_21.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_red_solid_22.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_red_solid_23.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        "rasterizer/alpha_blend/tex_red_solid_24.png"
    )]
    fn alpha_blend_tex_red_solid(#[case] c0: Vec4, #[case] c1: Vec4, #[case] c2: Vec4, #[case] filename: &str) {
        let texture = Texture::new(&TextureSource {
            texels: &vec![255u8, 0u8, 0u8],
            width: 1,
            height: 1,
            format: TextureFormat::RGB,
        });
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(0.0, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, -0.5, 0.0)],
            tex_coords: &[Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0)],
            texture: Some(texture),
            colors: &[c0, c1, c2],
            alpha_blending: AlphaBlendingMode::Normal,
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }

    #[rstest]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        "rasterizer/alpha_blend/tex_mixed_solid_00.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        "rasterizer/alpha_blend/tex_mixed_solid_01.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        "rasterizer/alpha_blend/tex_mixed_solid_02.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        "rasterizer/alpha_blend/tex_mixed_solid_03.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        "rasterizer/alpha_blend/tex_mixed_solid_04.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        "rasterizer/alpha_blend/tex_mixed_solid_05.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        "rasterizer/alpha_blend/tex_mixed_solid_06.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        "rasterizer/alpha_blend/tex_mixed_solid_07.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        "rasterizer/alpha_blend/tex_mixed_solid_08.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        "rasterizer/alpha_blend/tex_mixed_solid_09.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_mixed_solid_10.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 1.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
        "rasterizer/alpha_blend/tex_mixed_solid_11.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.9),
        Vec4::new(0.0, 1.0, 0.0, 0.9),
        Vec4::new(0.0, 0.0, 1.0, 0.9),
        "rasterizer/alpha_blend/tex_mixed_solid_12.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.8),
        Vec4::new(0.0, 1.0, 0.0, 0.8),
        Vec4::new(0.0, 0.0, 1.0, 0.8),
        "rasterizer/alpha_blend/tex_mixed_solid_13.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.7),
        Vec4::new(0.0, 1.0, 0.0, 0.7),
        Vec4::new(0.0, 0.0, 1.0, 0.7),
        "rasterizer/alpha_blend/tex_mixed_solid_14.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.6),
        Vec4::new(0.0, 1.0, 0.0, 0.6),
        Vec4::new(0.0, 0.0, 1.0, 0.6),
        "rasterizer/alpha_blend/tex_mixed_solid_15.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.5),
        Vec4::new(0.0, 1.0, 0.0, 0.5),
        Vec4::new(0.0, 0.0, 1.0, 0.5),
        "rasterizer/alpha_blend/tex_mixed_solid_16.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.4),
        Vec4::new(0.0, 1.0, 0.0, 0.4),
        Vec4::new(0.0, 0.0, 1.0, 0.4),
        "rasterizer/alpha_blend/tex_mixed_solid_17.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.3),
        Vec4::new(0.0, 1.0, 0.0, 0.3),
        Vec4::new(0.0, 0.0, 1.0, 0.3),
        "rasterizer/alpha_blend/tex_mixed_solid_18.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.2),
        Vec4::new(0.0, 1.0, 0.0, 0.2),
        Vec4::new(0.0, 0.0, 1.0, 0.2),
        "rasterizer/alpha_blend/tex_mixed_solid_19.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.1),
        Vec4::new(0.0, 1.0, 0.0, 0.1),
        Vec4::new(0.0, 0.0, 1.0, 0.1),
        "rasterizer/alpha_blend/tex_mixed_solid_20.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_mixed_solid_21.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_mixed_solid_22.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_mixed_solid_23.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        "rasterizer/alpha_blend/tex_mixed_solid_24.png"
    )]
    fn alpha_blend_tex_mixed_solid(#[case] c0: Vec4, #[case] c1: Vec4, #[case] c2: Vec4, #[case] filename: &str) {
        let texture = Texture::new(&TextureSource {
            texels: &vec![0x0Bu8, 0xDAu8, 0x51u8],
            width: 1,
            height: 1,
            format: TextureFormat::RGB,
        });
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(0.0, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, -0.5, 0.0)],
            tex_coords: &[Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0)],
            texture: Some(texture),
            colors: &[c0, c1, c2],
            alpha_blending: AlphaBlendingMode::Normal,
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo_wbg(&command), filename);
    }

    #[rstest]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        "rasterizer/alpha_blend/tex_purple_half_00.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        "rasterizer/alpha_blend/tex_purple_half_01.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        "rasterizer/alpha_blend/tex_purple_half_02.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        "rasterizer/alpha_blend/tex_purple_half_03.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        "rasterizer/alpha_blend/tex_purple_half_04.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        "rasterizer/alpha_blend/tex_purple_half_05.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        "rasterizer/alpha_blend/tex_purple_half_06.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        "rasterizer/alpha_blend/tex_purple_half_07.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        "rasterizer/alpha_blend/tex_purple_half_08.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        "rasterizer/alpha_blend/tex_purple_half_09.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_purple_half_10.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 1.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
        "rasterizer/alpha_blend/tex_purple_half_11.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.9),
        Vec4::new(0.0, 1.0, 0.0, 0.9),
        Vec4::new(0.0, 0.0, 1.0, 0.9),
        "rasterizer/alpha_blend/tex_purple_half_12.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.8),
        Vec4::new(0.0, 1.0, 0.0, 0.8),
        Vec4::new(0.0, 0.0, 1.0, 0.8),
        "rasterizer/alpha_blend/tex_purple_half_13.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.7),
        Vec4::new(0.0, 1.0, 0.0, 0.7),
        Vec4::new(0.0, 0.0, 1.0, 0.7),
        "rasterizer/alpha_blend/tex_purple_half_14.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.6),
        Vec4::new(0.0, 1.0, 0.0, 0.6),
        Vec4::new(0.0, 0.0, 1.0, 0.6),
        "rasterizer/alpha_blend/tex_purple_half_15.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.5),
        Vec4::new(0.0, 1.0, 0.0, 0.5),
        Vec4::new(0.0, 0.0, 1.0, 0.5),
        "rasterizer/alpha_blend/tex_purple_half_16.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.4),
        Vec4::new(0.0, 1.0, 0.0, 0.4),
        Vec4::new(0.0, 0.0, 1.0, 0.4),
        "rasterizer/alpha_blend/tex_purple_half_17.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.3),
        Vec4::new(0.0, 1.0, 0.0, 0.3),
        Vec4::new(0.0, 0.0, 1.0, 0.3),
        "rasterizer/alpha_blend/tex_purple_half_18.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.2),
        Vec4::new(0.0, 1.0, 0.0, 0.2),
        Vec4::new(0.0, 0.0, 1.0, 0.2),
        "rasterizer/alpha_blend/tex_purple_half_19.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.1),
        Vec4::new(0.0, 1.0, 0.0, 0.1),
        Vec4::new(0.0, 0.0, 1.0, 0.1),
        "rasterizer/alpha_blend/tex_purple_half_20.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_purple_half_21.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_purple_half_22.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_purple_half_23.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        "rasterizer/alpha_blend/tex_purple_half_24.png"
    )]
    fn alpha_blend_tex_purple_half(#[case] c0: Vec4, #[case] c1: Vec4, #[case] c2: Vec4, #[case] filename: &str) {
        let texture = Texture::new(&TextureSource {
            texels: &vec![0x93u8, 0x70u8, 0xDBu8, 0x7Fu8],
            width: 1,
            height: 1,
            format: TextureFormat::RGBA,
        });
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(0.0, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, -0.5, 0.0)],
            tex_coords: &[Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0)],
            texture: Some(texture),
            colors: &[c0, c1, c2],
            alpha_blending: AlphaBlendingMode::Normal,
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo_wbg(&command), filename);
    }
}

#[cfg(test)]
mod tests_watertight {
    use super::*;
    use image::{ImageBuffer, Rgba};
    use std::path::Path;

    fn save_image<P: AsRef<Path>>(path: &P, image: &Buffer<u32>) {
        let raw_rgba: Vec<u8> = image
            .elems
            .iter()
            .flat_map(|&pixel| {
                let bytes = pixel.to_le_bytes();
                [bytes[0], bytes[1], bytes[2], bytes[3]]
            })
            .collect();
        let img1: ImageBuffer<Rgba<u8>, _> =
            ImageBuffer::from_raw(image.width as u32, image.height as u32, raw_rgba).unwrap();
        img1.save(path).unwrap();
    }

    #[test]
    fn fullscreen_quad() {
        // v0--v2v3|
        // |   //  |
        // |  //   |
        // | //    |
        // v1v4---v5
        let wp1 = vec![
            Vec3::new(-1.0, 1.0, 0.0),
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
        ];
        // v0v5--v4
        // | \\   |
        // |  \\  |
        // |   \\ |
        // v1--v2v3
        let wp2 = vec![
            Vec3::new(-1.0, 1.0, 0.0),
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, 1.0, 0.0),
        ];
        let colors = vec![
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            Vec4::new(0.0, 1.0, 0.0, 1.0),
            Vec4::new(0.0, 1.0, 0.0, 1.0),
            Vec4::new(0.0, 1.0, 0.0, 1.0),
        ];
        let mut rasterizer = Rasterizer::new();
        for dim in 1..=512 {
            for wp in [&wp1, &wp2] {
                let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(dim as u16, dim as u16);
                color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
                rasterizer.setup(Viewport::new(0, 0, dim as u16, dim as u16));
                rasterizer.commit(&RasterizationCommand { world_positions: wp, colors: &colors, ..Default::default() });
                rasterizer.draw(&mut Framebuffer { color_buffer: Some(&mut color_buffer), ..Default::default() });
                let flat = color_buffer.as_flat_buffer();
                let tight = flat.elems.iter().all(|&x| {
                    let c = RGBA::from_u32(x);
                    c.r > 0 || c.g > 0 || c.b > 0
                });
                if !tight {
                    save_image(
                        &Path::new(env!("CARGO_MANIFEST_DIR"))
                            .join(format!("tests/fullscreen_quad_{0}x{0}.actual.png", dim)),
                        &flat,
                    );
                }
                assert!(tight);
            }
        }
    }
}

#[cfg(test)]
mod tests_alpha_blending {
    use super::*;

    #[test]
    fn blending_none() {
        let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(1u16, 1u16);
        let mut rasterizer = Rasterizer::new();

        let pos = [Vec3::new(0.0, 1.0, 0.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0)];
        for background in (0..=255).step_by(51) {
            for foreground in (0..=255).step_by(51) {
                for alpha in (0..=255).step_by(51) {
                    rasterizer.setup(Viewport::new(0, 0, 1u16, 1u16));
                    rasterizer.commit(&RasterizationCommand {
                        world_positions: &pos,
                        color: Vec4::new(foreground as f32 / 255.0, 0.0, 0.0, alpha as f32 / 255.0),
                        alpha_blending: AlphaBlendingMode::None,
                        ..Default::default()
                    });
                    color_buffer.fill(RGBA::new(background as u8, 0, 0, 255).to_u32());
                    rasterizer.draw(&mut Framebuffer { color_buffer: Some(&mut color_buffer), ..Default::default() });
                    let expected: u8 = foreground as u8;
                    assert_rgba_eq!(RGBA::from_u32(color_buffer.at(0, 0)), RGBA::new(expected, 0, 0, 255), 2);
                }
            }
        }
    }

    #[test]
    fn blending_normal() {
        let mut rasterizer = Rasterizer::new();
        let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(1u16, 1u16);
        let pos = [Vec3::new(0.0, 1.0, 0.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0)];
        for background in (0..=255).step_by(51) {
            for foreground in (0..=255).step_by(51) {
                for alpha in (0..=255).step_by(51) {
                    rasterizer.setup(Viewport::new(0, 0, 1u16, 1u16));
                    rasterizer.commit(&RasterizationCommand {
                        world_positions: &pos,
                        color: Vec4::new(foreground as f32 / 255.0, 0.0, 0.0, alpha as f32 / 255.0),
                        alpha_blending: AlphaBlendingMode::Normal,
                        ..Default::default()
                    });
                    color_buffer.fill(RGBA::new(background as u8, 0, 0, 255).to_u32());
                    rasterizer.draw(&mut Framebuffer { color_buffer: Some(&mut color_buffer), ..Default::default() });
                    let expected: u8 = (((foreground as f32 / 255.0) * (alpha as f32 / 255.0)
                        + (background as f32 / 255.0) * ((255 - alpha) as f32 / 255.0))
                        * 255.0) as u8;
                    assert_rgba_eq!(RGBA::from_u32(color_buffer.at(0, 0)), RGBA::new(expected, 0, 0, 255), 20);
                }
            }
        }
    }

    #[test]
    fn blending_additive() {
        let mut rasterizer = Rasterizer::new();
        let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(1u16, 1u16);
        let pos = [Vec3::new(0.0, 1.0, 0.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0)];
        for background in (0..=255).step_by(51) {
            for foreground in (0..=255).step_by(51) {
                for alpha in (0..=255).step_by(51) {
                    rasterizer.setup(Viewport::new(0, 0, 1u16, 1u16));
                    rasterizer.commit(&RasterizationCommand {
                        world_positions: &pos,
                        color: Vec4::new(foreground as f32 / 255.0, 0.0, 0.0, alpha as f32 / 255.0),
                        alpha_blending: AlphaBlendingMode::Additive,
                        ..Default::default()
                    });
                    color_buffer.fill(RGBA::new(background as u8, 0, 0, 255).to_u32());
                    rasterizer.draw(&mut Framebuffer { color_buffer: Some(&mut color_buffer), ..Default::default() });
                    let expected: u8 =
                        (((foreground as f32 / 255.0) * (alpha as f32 / 255.0) + (background as f32 / 255.0)) * 255.0)
                            .min(255.0) as u8;
                    assert_rgba_eq!(RGBA::from_u32(color_buffer.at(0, 0)), RGBA::new(expected, 0, 0, 255), 2);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests_alpha_test {
    use super::*;

    #[test]
    fn alpha_test() {
        let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(1u16, 1u16);
        let mut depth_buffer = TiledBuffer::<u16, 64, 64>::new(1u16, 1u16);
        let mut normal_buffer = TiledBuffer::<u32, 64, 64>::new(1u16, 1u16);
        let mut rasterizer = Rasterizer::new();
        rasterizer.setup(Viewport::new(0, 0, 1u16, 1u16));
        let pos = [Vec3::new(0.0, 1.0, 0.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0)];
        let tex_coords = [Vec2::new(0.5, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0)];

        struct TC {
            texture_alpha: u8,
            alpha_test: u8,
            expected_discard: bool,
        }

        let test_cases = vec![
            TC { texture_alpha: 255u8, alpha_test: 255u8, expected_discard: false },
            TC { texture_alpha: 255u8, alpha_test: 127u8, expected_discard: false },
            TC { texture_alpha: 255u8, alpha_test: 0u8, expected_discard: false },
            TC { texture_alpha: 127u8, alpha_test: 255u8, expected_discard: true },
            TC { texture_alpha: 127u8, alpha_test: 128u8, expected_discard: true },
            TC { texture_alpha: 127u8, alpha_test: 127u8, expected_discard: false },
            TC { texture_alpha: 127u8, alpha_test: 0u8, expected_discard: false },
            TC { texture_alpha: 0u8, alpha_test: 255u8, expected_discard: true },
            TC { texture_alpha: 0u8, alpha_test: 127u8, expected_discard: true },
            TC { texture_alpha: 0u8, alpha_test: 1u8, expected_discard: true },
            TC { texture_alpha: 0u8, alpha_test: 0u8, expected_discard: false },
        ];
        for tc in test_cases {
            let texture = Texture::new(&TextureSource {
                texels: &[255u8, 255u8, 255u8, tc.texture_alpha],
                width: 1,
                height: 1,
                format: TextureFormat::RGBA,
            });
            color_buffer.fill(0u32);
            depth_buffer.fill(u16::MAX);
            normal_buffer.fill(0u32);
            rasterizer.reset();
            rasterizer.commit(&RasterizationCommand {
                world_positions: &pos,
                texture: Some(texture),
                tex_coords: &tex_coords,
                alpha_test: tc.alpha_test,
                ..Default::default()
            });
            rasterizer.draw(&mut Framebuffer {
                color_buffer: Some(&mut color_buffer),
                depth_buffer: Some(&mut depth_buffer),
                normal_buffer: Some(&mut normal_buffer),
                ..Default::default()
            });
            let color_discarded = color_buffer.at(0, 0) == 0;
            assert_eq!(color_discarded, tc.expected_discard);
            let depth_discarded = depth_buffer.at(0, 0) == u16::MAX;
            assert_eq!(depth_discarded, tc.expected_discard);
            let normal_discarded = normal_buffer.at(0, 0) == 0;
            assert_eq!(normal_discarded, tc.expected_discard);
        }
    }
}
