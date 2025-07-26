use super::super::math::*;
use super::*;
use arrayvec::ArrayVec;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawLinesCommand<'a> {
    pub lines: &'a [Vec3],
    pub color: Vec4,
    pub model: Mat34,
    pub view: Mat44,
    pub projection: Mat44,
}

impl Default for DrawLinesCommand<'_> {
    fn default() -> Self {
        Self {
            lines: &[],
            color: Vec4::new(1.0, 1.0, 1.0, 1.0),
            model: Mat34::identity(),
            view: Mat44::identity(),
            projection: Mat44::identity(),
        }
    }
}

fn vec4_to_rgba(c: Vec4) -> RGBA {
    fn float_to_u8(x: f32) -> u8 {
        let i = (x * 256.0) as i32;
        if i < 0 {
            0
        } else if i > 255 {
            255
        } else {
            i as u8
        }
    }

    RGBA { r: float_to_u8(c.x), g: float_to_u8(c.y), b: float_to_u8(c.z), a: float_to_u8(c.w) }
}

fn perspective_divide_to_vec3(v: Vec4) -> Vec3 {
    Vec3::new(v.x / v.w, v.y / v.w, v.z / v.w)
}

// TODO: convert into Mat34 or Mat23?
// This is stupidly slow
fn apply_viewport(viewport: &Viewport, v: Vec3) -> Vec3 {
    Vec3::new(
        viewport.xmin as f32 + ((viewport.xmax - viewport.xmin - 1) as f32) * (0.5 + 0.5 * v.x),
        viewport.ymin as f32 + ((viewport.ymax - viewport.ymin - 1) as f32) * (0.5 - 0.5 * v.y),
        v.z,
    )
}

fn blend(src: RGBA, dst: RGBA) -> RGBA {
    let a = src.a as u32;
    let ia = 255 - a;
    RGBA {
        r: ((src.r as u32 * a + dst.r as u32 * ia) >> 8) as u8,
        g: ((src.g as u32 * a + dst.g as u32 * ia) >> 8) as u8,
        b: ((src.b as u32 * a + dst.b as u32 * ia) >> 8) as u8,
        a: dst.a,
    }
}

pub fn draw_lines(framebuffer: &mut Framebuffer, viewport: &Viewport, command: &DrawLinesCommand) {
    let lines = command.lines;
    let len = lines.len();
    assert_eq!(len % 2, 0);
    if len == 0 {
        return;
    }

    let view_projection = &command.projection * &command.view;
    let rgba = vec4_to_rgba(command.color);
    let mut color_buf_opt = framebuffer.color_buffer.as_deref_mut();

    let mut i = 0;

    while i + 1 < len {
        let world = [
            &command.model * lines[i], //
            &command.model * lines[i + 1],
        ];
        let projected = [
            view_projection * world[0].as_point4(), //
            view_projection * world[1].as_point4(),
        ];
        let clipped = clip_line(&projected);
        if clipped.len() < 2 {
            i += 2;
            continue;
        };
        let perspective_divided = [
            perspective_divide_to_vec3(clipped[0]), //
            perspective_divide_to_vec3(clipped[1]),
        ];
        let screen = [
            apply_viewport(viewport, perspective_divided[0]), //
            apply_viewport(viewport, perspective_divided[1]),
        ];

        let mut x0 = screen[0].x as i32;
        let mut y0 = screen[0].y as i32;
        let mut x1 = screen[1].x as i32;
        let mut y1 = screen[1].y as i32;
        //        let mut z0 = screen[0].z;
        //         let mut z1 = screen[1].z;

        let steep = (y1 - y0).abs() > (x1 - x0).abs();
        if steep {
            std::mem::swap(&mut x0, &mut y0);
            std::mem::swap(&mut x1, &mut y1);
        }

        if x0 > x1 {
            std::mem::swap(&mut x0, &mut x1);
            std::mem::swap(&mut y0, &mut y1);
            // std::mem::swap(&mut z0, &mut z1);
        }

        let dx = x1 - x0;
        let dy = (y1 - y0).abs();
        let mut error = dx / 2;
        let y_step = if y0 < y1 { 1 } else { -1 };
        let mut y = y0;
        // let steps = (x1 - x0 + 1) as f32;

        for x in x0..=x1 {
            // let t = (x - x0) as f32 / steps;
            // let z = (1.0 - t) * z0 + t * z1;
            let screen_x = if steep { y } else { x };
            let screen_y = if steep { x } else { y };

            // if framebuffer.depth.at(screen_x, screen_y) > z {
            //     framebuffer.depth.set(screen_x, screen_y, z);
            //     let dst = framebuffer.color.at(screen_x, screen_y);
            //     let out = if color.a == 255 {
            //         color
            //     } else {
            //         blend(color, dst)
            //     };
            //     framebuffer.color.set(screen_x, screen_y, out);
            // }

            // if framebuffer.color_buffer.is_some() {
            //     let mut pixel = framebuffer
            //         .color_buffer
            //         .unwrap()
            //         .at_mut(screen_x as usize, screen_y as usize);
            //     *pixel = rgba.to_u32();
            // }

            // if let Some(color_buf) = framebuffer.color_buffer.as_mut() {

            // WTF????????!!!!
            // if let Some(color_buf) = &mut framebuffer.color_buffer {
            //     let pixel = color_buf.at_mut(screen_x as usize, screen_y as usize);
            //     *pixel = rgba.to_u32();
            // }

            // if let Some(color_buf) = color_buf_opt {
            //     let pixel = color_buf.at_mut(screen_x as usize, screen_y as usize);
            //     *pixel = rgba.to_u32();
            // }

            if let Some(ref mut buf) = color_buf_opt {
                let dst = buf.at_mut(screen_x as usize, screen_y as usize);
                if rgba.a == 255 {
                    *dst = rgba.to_u32();
                } else {
                    *dst = blend(rgba, RGBA::from_u32(*dst)).to_u32();
                }
            }

            error -= dy;
            if error < 0 {
                y += y_step;
                error += dx;
            }
        }

        i += 2;
    }
}

pub fn aabb_to_lines(aabb: AABB) -> ArrayVec<Vec3, 24> {
    let mut lines = ArrayVec::new();

    // bottom
    lines.push(aabb.min);
    lines.push(Vec3::new(aabb.min.x, aabb.min.y, aabb.max.z));
    lines.push(Vec3::new(aabb.min.x, aabb.min.y, aabb.max.z));
    lines.push(Vec3::new(aabb.max.x, aabb.min.y, aabb.max.z));
    lines.push(Vec3::new(aabb.max.x, aabb.min.y, aabb.max.z));
    lines.push(Vec3::new(aabb.max.x, aabb.min.y, aabb.min.z));
    lines.push(Vec3::new(aabb.max.x, aabb.min.y, aabb.min.z));
    lines.push(aabb.min);

    // top
    lines.push(aabb.max);
    lines.push(Vec3::new(aabb.min.x, aabb.max.y, aabb.max.z));
    lines.push(Vec3::new(aabb.min.x, aabb.max.y, aabb.max.z));
    lines.push(Vec3::new(aabb.min.x, aabb.max.y, aabb.min.z));
    lines.push(Vec3::new(aabb.min.x, aabb.max.y, aabb.min.z));
    lines.push(Vec3::new(aabb.max.x, aabb.max.y, aabb.min.z));
    lines.push(Vec3::new(aabb.max.x, aabb.max.y, aabb.min.z));
    lines.push(aabb.max);

    // vertical
    lines.push(aabb.min);
    lines.push(Vec3::new(aabb.min.x, aabb.max.y, aabb.min.z));
    lines.push(aabb.max);
    lines.push(Vec3::new(aabb.max.x, aabb.min.y, aabb.max.z));
    lines.push(Vec3::new(aabb.max.x, aabb.min.y, aabb.min.z));
    lines.push(Vec3::new(aabb.max.x, aabb.max.y, aabb.min.z));
    lines.push(Vec3::new(aabb.min.x, aabb.min.y, aabb.max.z));
    lines.push(Vec3::new(aabb.min.x, aabb.max.y, aabb.max.z));

    lines
}
