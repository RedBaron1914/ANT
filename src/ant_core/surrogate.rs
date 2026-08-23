// Surrogate gradients for spiking neural networks

#[derive(Clone)]
pub struct Surrogate {
    pub alpha: f32,
    pub v_th: f32,
}

impl Surrogate {
    pub fn new(alpha: f32, v_th: f32) -> Self {
        Self { alpha, v_th }
    }

    /// Forward pass: Heaviside step function
    pub fn heaviside_step(&self, x: f32) -> f32 {
        if x > self.v_th {
            1.0
        } else {
            0.0
        }
    }

    /// Backward pass: Surrogate derivative
    /// f'(x) = 1 / (1 + alpha * |x - v_th|)^2
    pub fn surrogate_derivative(&self, x: f32) -> f32 {
        let diff = (x - self.v_th).abs();
        let denom = 1.0 + self.alpha * diff;
        1.0 / (denom * denom)
    }
}
