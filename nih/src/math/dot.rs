pub trait Dot {
    fn dot(self, rhs: Self) -> f32;
}

pub fn dot<V: Dot>(v1: V, v2: V) -> f32 {
    v1.dot(v2)
}
