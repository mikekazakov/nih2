use super::*;

pub struct Framebuffer<'a> {
    pub color_buffer: Option<&'a mut ColorBuffer>,
}
