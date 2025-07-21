#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub xmin: u16,
    pub ymin: u16,
    pub xmax: u16,
    pub ymax: u16,
}

impl Viewport {
    pub fn new(xmin: u16, ymin: u16, xmax: u16, ymax: u16) -> Viewport {
        Viewport { xmin, ymin, xmax, ymax }
    }
}
