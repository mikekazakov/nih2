use bytemuck::{Pod, Zeroable};

pub struct Buffer<T> {
    /// Width of usable elements in the buffer
    pub width: usize,

    /// Height of usable elements in the buffer
    pub height: usize,

    /// Number of elements between the rows
    pub stride: usize,

    /// The actual elements in the buffer
    pub elems: Vec<T>,
}

impl<T: Copy + Zeroable + Pod> Buffer<T> {
    pub fn new(width: usize, height: usize) -> Self {
        let stride = width;
        let elems = vec![T::zeroed(); stride * height];
        Self { width, height, stride, elems }
    }

    pub fn at(&self, x: usize, y: usize) -> T {
        assert!(x < self.width, "x out of bounds: {} >= {}", x, self.width);
        assert!(y < self.height, "y out of bounds: {} >= {}", y, self.height);
        self.elems[y * self.stride + x]
    }

    pub fn at_mut(&mut self, x: usize, y: usize) -> &mut T {
        assert!(x < self.width, "x out of bounds: {} >= {}", x, self.width);
        assert!(y < self.height, "y out of bounds: {} >= {}", y, self.height);
        &mut self.elems[y * self.stride + x]
    }

    pub fn as_u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(&self.elems)
    }

    pub fn as_u8_slice_mut(&mut self) -> &mut [u8] {
        bytemuck::cast_slice_mut(&mut self.elems)
    }

    pub fn as_u32_slice(&self) -> &[T] {
        &self.elems
    }

    pub fn as_u32_slice_mut(&mut self) -> &mut [T] {
        &mut self.elems
    }

    // pub fn as_rgba_slice(&self) -> &[RGBA] {
    //     bytemuck::cast_slice(&self.pixels)
    // }
    //
    // pub fn as_rgba_slice_mut(&mut self) -> &mut [RGBA] {
    //     bytemuck::cast_slice_mut(&mut self.pixels)
    // }

    pub fn fill(&mut self, with: T) {
        // let raw = color.to_u32();
        for elem in self.elems.iter_mut() {
            *elem = with;
        }
    }
}
