#[derive(Clone, Copy, Debug)]
pub struct U32x4 {
    #[cfg(target_arch = "x86_64")]
    inner: core::arch::x86_64::__m128i,

    #[cfg(target_arch = "aarch64")]
    inner: core::arch::aarch64::uint32x4_t,
}

impl U32x4 {
    /// Construct from array
    #[inline(always)]
    pub fn load(values: [u32; 4]) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                Self { inner: _mm_loadu_si128(values.as_ptr() as *const __m128i) }
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                Self { inner: vld1q_u32(values.as_ptr()) }
            }
        }
    }

    /// Store back into array
    #[inline(always)]
    pub fn store(self) -> [u32; 4] {
        let mut out = [0u32; 4];
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, self.inner);
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                vst1q_u32(out.as_mut_ptr(), self.inner);
            }
        }

        out
    }

    /// Add two vectors
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                Self { inner: _mm_add_epi32(self.inner, other.inner) }
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                Self { inner: vaddq_u32(self.inner, other.inner) }
            }
        }
    }

    /// Bitwise AND
    #[inline(always)]
    pub fn bitand(self, other: Self) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                Self { inner: _mm_and_si128(self.inner, other.inner) }
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                Self { inner: vandq_u32(self.inner, other.inner) }
            }
        }
    }

    /// Check if any lane is nonzero
    #[inline(always)]
    pub fn any_nonzero(self) -> bool {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                // Test if all bits are zero
                _mm_testz_si128(self.inner, self.inner) == 0
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                vmaxvq_u32(self.inner) != 0
            }
        }
    }

    #[inline(always)]
    pub fn all_zero(self) -> bool {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                // _mm_testz_si128 returns 1 if all bits are zero
                _mm_testz_si128(self.inner, self.inner) != 0
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                // all zero means no lane is nonzero
                vmaxvq_u32(self.inner) == 0
            }
        }
    }

    #[inline(always)]
    pub fn extract_lane0(self) -> u32 {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                _mm_cvtsi128_si32(self.inner) as u32
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                vgetq_lane_u32(self.inner, 0)
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct F32x4 {
    #[cfg(target_arch = "x86_64")]
    inner: core::arch::x86_64::__m128,

    #[cfg(target_arch = "aarch64")]
    inner: core::arch::aarch64::float32x4_t,
}

impl F32x4 {
    /// Construct from array
    #[inline(always)]
    pub fn load(values: [f32; 4]) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                Self { inner: _mm_loadu_ps(values.as_ptr()) }
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                Self { inner: vld1q_f32(values.as_ptr()) }
            }
        }
    }

    /// Store back into array
    #[inline(always)]
    pub fn store(self) -> [f32; 4] {
        let mut out = [0f32; 4];
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                _mm_storeu_ps(out.as_mut_ptr(), self.inner);
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                vst1q_f32(out.as_mut_ptr(), self.inner);
            }
        }
        out
    }

    /// Store back into array
    #[inline(always)]
    pub fn store_to(self, out: &mut [f32; 4]) {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                _mm_storeu_ps(out.as_mut_ptr(), self.inner);
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                vst1q_f32(out.as_mut_ptr(), self.inner);
            }
        }
    }

    /// Construct from a single value broadcasted to 4 lanes
    #[inline(always)]
    pub fn splat(value: f32) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                Self { inner: _mm_set1_ps(value) }
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                Self { inner: vdupq_n_f32(value) }
            }
        }
    }

    /// Convert to a 32-bit integer vector.
    #[inline(always)]
    pub fn to_u32(self) -> U32x4 {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                U32x4 { inner: _mm_cvttps_epi32(self.inner) }
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                U32x4 { inner: vcvtq_u32_f32(self.inner) }
            }
        }
    }

    /// Add two vectors
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                Self { inner: _mm_add_ps(self.inner, other.inner) }
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                Self { inner: vaddq_f32(self.inner, other.inner) }
            }
        }
    }

    /// Subtracts two vectors
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                Self { inner: _mm_sub_ps(self.inner, other.inner) }
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                Self { inner: vsubq_f32(self.inner, other.inner) }
            }
        }
    }

    /// Multiplies two vectors
    #[inline(always)]
    pub fn mul(self, other: Self) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                Self { inner: _mm_mul_ps(self.inner, other.inner) }
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                Self { inner: vmulq_f32(self.inner, other.inner) }
            }
        }
    }

    /// Divides two vectors
    #[inline(always)]
    pub fn div(self, other: Self) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                Self { inner: _mm_div_ps(self.inner, other.inner) }
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                Self { inner: vdivq_f32(self.inner, other.inner) }
            }
        }
    }

    /// Calculates x * a + b
    #[inline(always)]
    pub fn fma(self, a: Self, b: Self) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                Self { inner: _mm_fmadd_ps(self.inner, a.inner, b.inner) }
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                Self { inner: vfmaq_f32(b.inner, self.inner, a.inner) }
            }
        }
    }

    /// Calculates square root
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                Self { inner: _mm_sqrt_ps(self.inner) }
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                Self { inner: vsqrtq_f32(self.inner) }
            }
        }
    }

    /// Calculates a reciprocal square root approximation
    #[inline(always)]
    pub fn rsqrt(self) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                Self { inner: _mm_rsqrt_ps(self.inner) }
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                let mut reciprocal: float32x4_t = vrsqrteq_f32(self.inner);
                reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(self.inner, reciprocal), reciprocal), reciprocal);
                Self { inner: reciprocal }
            }
        }
    }

    /// Calculates an exponent function
    #[inline(always)]
    pub fn exp(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            // dummy for now
            let mut v: [f32; 4] = self.store();
            v[0] = v[0].exp();
            v[1] = v[1].exp();
            v[2] = v[2].exp();
            v[3] = v[3].exp();
            Self::load(v)
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self { inner: vexpq_neon_f32(self.inner) }
        }
    }

    /// Calculates a natural logarithm function
    #[inline(always)]
    pub fn log(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            // dummy for now
            let mut v: [f32; 4] = self.store();
            v[0] = v[0].ln();
            v[1] = v[1].ln();
            v[2] = v[2].ln();
            v[3] = v[3].ln();
            Self::load(v)
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self { inner: vlogq_neon_f32(self.inner) }
        }
    }

    // Calculates arccosine of x: [-1,1]
    // https://developer.download.nvidia.com/cg/acos.html
    #[inline(always)]
    pub fn acos(self) -> Self {
        let zero: F32x4 = Self::splat(0.0);
        let one: F32x4 = Self::splat(1.0);
        let negate: F32x4 = self.cmp_lt(zero).select(one, zero);
        let x: F32x4 = self.abs();
        let mut ret: F32x4 = Self::splat(-0.0187293);
        ret = ret.fma(x, Self::splat(0.0742610));
        ret = ret.fma(x, Self::splat(-0.2121144));
        ret = ret.fma(x, Self::splat(1.5707288));
        ret = ret * (one - x).sqrt();
        ret = ret * negate.fma(Self::splat(-2.0), one);
        negate.fma(Self::splat(std::f32::consts::PI), ret)
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                Self { inner: _mm_and_ps(self.inner, _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFF))) }
            }
            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                Self { inner: vabsq_f32(self.inner) }
            }
        }
    }

    /// Compares less than for each lane.
    #[inline(always)]
    pub fn cmp_lt(self, other: Self) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                Self { inner: _mm_cmplt_ps(self.inner, other.inner) }
            }
            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                Self { inner: vreinterpretq_f32_u32(vcltq_f32(self.inner, other.inner)) }
            }
        }
    }

    /// Select per-bit values from two vectors based on a mask.
    /// If the bit is 1, a value from the first vector is picked.
    /// e.g. select() => if { first } else { second }
    #[inline(always)]
    pub fn select(self, one: Self, zero: Self) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                return Self { inner: _mm_blendv_ps(zero.inner, one.inner, self.inner) };
            }
            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                Self { inner: vbslq_f32(vreinterpretq_u32_f32(self.inner), one.inner, zero.inner) }
            }
        }
    }

    /// Min
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                return Self { inner: _mm_min_ps(self.inner, other.inner) };
            }
            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                Self { inner: vminq_f32(self.inner, other.inner) }
            }
        }
    }

    /// Max
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                return Self { inner: _mm_max_ps(self.inner, other.inner) };
            }
            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                Self { inner: vmaxq_f32(self.inner, other.inner) }
            }
        }
    }
}


// https://github.com/ARM-software/EndpointAI/blob/master/Kernels/Migrating_to_Helium_from_Neon_Companion_SW/vmath.c
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(non_snake_case)]
fn vtaylor_polyq_f32(x: core::arch::aarch64::float32x4_t, coeffs: &[f32; 32]) -> core::arch::aarch64::float32x4_t {
    unsafe {
        use core::arch::aarch64::*;
        let coeffs: *const f32 = coeffs.as_ptr();
        let A: float32x4_t = vmlaq_f32(vld1q_f32(coeffs.add(4 * 0)), vld1q_f32(coeffs.add(4 * 4)), x);
        let B: float32x4_t = vmlaq_f32(vld1q_f32(coeffs.add(4 * 2)), vld1q_f32(coeffs.add(4 * 6)), x);
        let C: float32x4_t = vmlaq_f32(vld1q_f32(coeffs.add(4 * 1)), vld1q_f32(coeffs.add(4 * 5)), x);
        let D: float32x4_t = vmlaq_f32(vld1q_f32(coeffs.add(4 * 3)), vld1q_f32(coeffs.add(4 * 7)), x);
        let x2: float32x4_t = vmulq_f32(x, x);
        let x4: float32x4_t = vmulq_f32(x2, x2);
        let res: float32x4_t = vmlaq_f32(vmlaq_f32(A, B, x2), vmlaq_f32(C, D, x2), x4);
        res
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(non_snake_case)]
fn vexpq_neon_f32(x: core::arch::aarch64::float32x4_t) -> core::arch::aarch64::float32x4_t {
    unsafe {
        use core::arch::aarch64::*;
        // Perform range reduction [-log(2),log(2)]
        let m: int32x4_t = vcvtq_s32_f32(vmulq_f32(x, vdupq_n_f32(std::f32::consts::LOG2_E)));
        let val: float32x4_t = vmlsq_f32(x, vcvtq_f32_s32(m), vdupq_n_f32(std::f32::consts::LN_2));
        // Polynomial Approximation
        let mut poly: float32x4_t = vtaylor_polyq_f32(val, &EXP_TAB);
        // Reconstruct
        poly = vreinterpretq_f32_s32(vqaddq_s32(vreinterpretq_s32_f32(poly), vqshlq_n_s32(m, 23)));
        poly = vbslq_f32(vcltq_s32(m, vdupq_n_s32(-126)), vdupq_n_f32(0.0), poly);
        poly
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(non_snake_case)]
fn vlogq_neon_f32(x: core::arch::aarch64::float32x4_t) -> core::arch::aarch64::float32x4_t {
    unsafe {
        use core::arch::aarch64::*;
        // Extract exponent
        let m: int32x4_t = vsubq_s32(vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_f32(x), 23)), vdupq_n_s32(127));
        let val: float32x4_t = vreinterpretq_f32_s32(vsubq_s32(vreinterpretq_s32_f32(x), vshlq_n_s32(m, 23)));
        // Polynomial Approximation
        let mut poly: float32x4_t = vtaylor_polyq_f32(val, &LOG_TAB);
        // Reconstruct
        poly = vmlaq_f32(poly, vcvtq_f32_s32(m), vdupq_n_f32(std::f32::consts::LN_2));
        poly
    }
}

#[cfg(target_arch = "aarch64")]
static EXP_TAB: [f32; 32] = [
    1.0, 1.0, 1.0, 1.0,
    0.0416598916054, 0.0416598916054, 0.0416598916054, 0.0416598916054,
    0.500000596046, 0.500000596046, 0.500000596046, 0.500000596046,
    0.0014122662833, 0.0014122662833, 0.0014122662833, 0.0014122662833,
    1.00000011921, 1.00000011921, 1.00000011921, 1.00000011921,
    0.00833693705499, 0.00833693705499, 0.00833693705499, 0.00833693705499,
    0.166665703058, 0.166665703058, 0.166665703058, 0.166665703058,
    0.000195780929062, 0.000195780929062, 0.000195780929062, 0.000195780929062
];

#[cfg(target_arch = "aarch64")]
static LOG_TAB: [f32; 32] = [
    -2.29561495781, -2.29561495781, -2.29561495781, -2.29561495781,
    -2.47071170807, -2.47071170807, -2.47071170807, -2.47071170807,
    -5.68692588806, -5.68692588806, -5.68692588806, -5.68692588806,
    -0.165253549814, -0.165253549814, -0.165253549814, -0.165253549814,
    5.17591238022, 5.17591238022, 5.17591238022, 5.17591238022,
    0.844007015228, 0.844007015228, 0.844007015228, 0.844007015228,
    4.58445882797, 4.58445882797, 4.58445882797, 4.58445882797,
    0.0141278216615, 0.0141278216615, 0.0141278216615, 0.0141278216615
];

// F32x4 + F32x4
impl std::ops::Add for F32x4 {
    type Output = F32x4;
    #[inline(always)]
    fn add(self, other: F32x4) -> F32x4 {
        self.add(other)
    }
}

// F32x4 - F32x4
impl std::ops::Sub for F32x4 {
    type Output = F32x4;
    #[inline(always)]
    fn sub(self, other: F32x4) -> F32x4 {
        self.sub(other)
    }
}

// F32x4 * F32x4
impl std::ops::Mul for F32x4 {
    type Output = F32x4;
    #[inline(always)]
    fn mul(self, other: F32x4) -> F32x4 {
        self.mul(other)
    }
}

// F32x4 / F32x4
impl std::ops::Div for F32x4 {
    type Output = F32x4;
    #[inline(always)]
    fn div(self, other: F32x4) -> F32x4 {
        self.div(other)
    }
}

// F32x4 += F32x4
impl std::ops::AddAssign for F32x4 {
    #[inline(always)]
    fn add_assign(&mut self, other: F32x4) {
        *self = self.add(other);
    }
}