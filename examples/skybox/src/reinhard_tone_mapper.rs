use nih::math::simd::*;

pub struct ReinhardToneMapper {
    luma_weights_r: F32x4,
    luma_weights_g: F32x4,
    luma_weights_b: F32x4,
    inv_white_point2: F32x4,
    exposure: F32x4,
}

impl ReinhardToneMapper {
    pub fn new(exposure: f32, white_point: f32) -> Self {
        Self {
            luma_weights_r: F32x4::splat(0.2126),
            luma_weights_g: F32x4::splat(0.7152),
            luma_weights_b: F32x4::splat(0.0722),
            inv_white_point2: F32x4::splat(1.0 / (white_point * white_point)),
            exposure: F32x4::splat(exposure),
        }
    }

    pub fn map(&self, r: &[f32], g: &[f32], b: &[f32], texels24: &mut [u8], y: usize) {
        assert!(r.len() == g.len() && r.len() == b.len());
        assert_eq!(r.len() % 4, 0);
        assert_eq!(texels24.len(), r.len() * 3);
        assert_eq!(texels24.len() % 12, 0);
        let mut r_ptr: *const f32 = r.as_ptr();
        let mut g_ptr: *const f32 = g.as_ptr();
        let mut b_ptr: *const f32 = b.as_ptr();
        let mut output_ptr: *mut u8 = texels24.as_mut_ptr();
        let steps: usize = r.len() / 4;
        let zero: F32x4 = F32x4::splat(0.0);
        let one: F32x4 = F32x4::splat(1.0);
        let to_255: F32x4 = F32x4::splat(255.0);
        let exposure: F32x4 = self.exposure;
        let luma_weights_r: F32x4 = self.luma_weights_r;
        let luma_weights_g: F32x4 = self.luma_weights_g;
        let luma_weights_b: F32x4 = self.luma_weights_b;
        let inv_white_point2: F32x4 = self.inv_white_point2;
        let noise_r: F32x4 = F32x4::load(NOISE_TABLE[(y + 0) % 16]);
        let noise_g: F32x4 = F32x4::load(NOISE_TABLE[(y + 1) % 16]);
        let noise_b: F32x4 = F32x4::load(NOISE_TABLE[(y + 2) % 16]);
        for _idx in 0..steps {
            // Load inputs in sRGB primaries with a linear gamma ramp
            let r: F32x4 = F32x4::load(unsafe { *(r_ptr as *const [f32; 4]) });
            let g: F32x4 = F32x4::load(unsafe { *(g_ptr as *const [f32; 4]) });
            let b: F32x4 = F32x4::load(unsafe { *(b_ptr as *const [f32; 4]) });

            // Expose
            let re: F32x4 = r * exposure;
            let ge: F32x4 = g * exposure;
            let be: F32x4 = b * exposure;

            // Calculate luminance: dot(rgb, luma_weights)
            let luma: F32x4 = re * luma_weights_r + ge * luma_weights_g + be * luma_weights_b;

            // Calculate white scale: (1 + luma / white_point^2) / (1 + luma)
            let scale: F32x4 = luma.fma(inv_white_point2, one) / (one + luma);

            // Map to SDR
            let rt: F32x4 = re * scale;
            let gt: F32x4 = ge * scale;
            let bt: F32x4 = be * scale;

            // Gamma-correction: v = v^(1.0/2.0)
            let rc: F32x4 = rt.sqrt();
            let gc: F32x4 = gt.sqrt();
            let bc: F32x4 = bt.sqrt();

            // Apply some noise for dithering
            let r_final: F32x4 = rc + noise_r;
            let g_final: F32x4 = gc + noise_g;
            let b_final: F32x4 = bc + noise_b;

            // Clamp the values to [0.0, 1.0] and convert to [0.0, 255.0]
            let r_out: F32x4 = r_final.min(one).max(zero) * to_255;
            let g_out: F32x4 = g_final.min(one).max(zero) * to_255;
            let b_out: F32x4 = b_final.min(one).max(zero) * to_255;

            // Convert to integers [0, 255]
            let r_u32: [u32; 4] = r_out.to_u32().store();
            let g_u32: [u32; 4] = g_out.to_u32().store();
            let b_u32: [u32; 4] = b_out.to_u32().store();

            // Store the output texels
            unsafe {
                *output_ptr.add(0) = r_u32[0] as u8;
                *output_ptr.add(1) = g_u32[0] as u8;
                *output_ptr.add(2) = b_u32[0] as u8;
                *output_ptr.add(3) = r_u32[1] as u8;
                *output_ptr.add(4) = g_u32[1] as u8;
                *output_ptr.add(5) = b_u32[1] as u8;
                *output_ptr.add(6) = r_u32[2] as u8;
                *output_ptr.add(7) = g_u32[2] as u8;
                *output_ptr.add(8) = b_u32[2] as u8;
                *output_ptr.add(9) = r_u32[3] as u8;
                *output_ptr.add(10) = g_u32[3] as u8;
                *output_ptr.add(11) = b_u32[3] as u8;
            };

            // Advance the input/output pointers
            r_ptr = unsafe { r_ptr.add(4) };
            g_ptr = unsafe { g_ptr.add(4) };
            b_ptr = unsafe { b_ptr.add(4) };
            output_ptr = unsafe { output_ptr.add(12) };
        }
    }
}

static NOISE_TABLE: [[f32; 4]; 16] = {
    const RAW_NOISE_TABLE: [[f32; 4]; 16] = [
        [0.732, -0.418, 0.091, -0.857],
        [-0.624, 0.289, 0.951, -0.103],
        [0.476, -0.812, 0.138, -0.369],
        [-0.295, 0.763, -0.581, 0.412],
        [0.684, -0.157, 0.328, -0.749],
        [-0.491, 0.856, -0.233, 0.617],
        [0.372, -0.689, 0.145, -0.514],
        [-0.267, 0.591, -0.804, 0.438],
        [0.723, -0.342, 0.019, -0.661],
        [0.214, -0.873, 0.596, -0.048],
        [-0.731, 0.382, -0.119, 0.905],
        [0.467, 0.058, -0.992, 0.341],
        [-0.553, -0.247, 0.784, -0.164],
        [0.129, 0.944, -0.403, 0.278],
        [-0.816, 0.256, 0.672, -0.294],
        [0.503, -0.957, 0.188, 0.421],
    ];
    let mut noise_table: [[f32; 4]; 16] = [[0.0; 4]; 16];
    let scale: f32 = 1.0 / 256.0;
    let mut i: usize = 0;
    while i < 16 {
        let mut j: usize = 0;
        while j < 4 {
            noise_table[i][j] = RAW_NOISE_TABLE[i][j] * scale;
            j += 1;
        }
        i += 1;
    }
    noise_table
};
