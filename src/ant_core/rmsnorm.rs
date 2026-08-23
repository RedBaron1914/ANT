use super::tensor::{BatchTensor, Tensor1D, MatExt};
use super::ewc::Parameterized;

#[derive(Clone, Debug)]
pub struct RMSNorm {
    pub weight: Tensor1D,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(dim: usize) -> Self {
        let mut weight = Tensor1D::new(dim);
        weight.data.fill(1.0);
        Self {
            weight,
            eps: 1e-6,
        }
    }

    pub fn forward(&self, input: &BatchTensor) -> BatchTensor {
        let b_size = input.data.nrows();
        let dim = input.data.ncols();
        let mut output = BatchTensor::new(b_size, dim);
        
        for b in 0..b_size {
            let mut sum_sq = 0.0;
            for i in 0..dim {
                let val = input.data.read(b, i);
                sum_sq += val * val;
            }
            let mean_sq = sum_sq / dim as f32;
            let rms_inv = 1.0 / (mean_sq + self.eps).sqrt();
            
            for i in 0..dim {
                let val = input.data.read(b, i);
                output.data.write(b, i, val * rms_inv * self.weight.data[i]);
            }
        }
        output
    }

    pub fn backward(&mut self, input: &BatchTensor, grad_output: &BatchTensor, grad_input: &mut BatchTensor) {
        let b_size = input.data.nrows();
        let dim = input.data.ncols();
        
        for b in 0..b_size {
            let mut sum_sq = 0.0;
            for i in 0..dim {
                let val = input.data.read(b, i);
                sum_sq += val * val;
            }
            let mean_sq = sum_sq / dim as f32;
            let rms_inv = 1.0 / (mean_sq + self.eps).sqrt();
            
            let mut sum_dy_x = 0.0;
            for i in 0..dim {
                let dy = grad_output.grad.read(b, i);
                let x = input.data.read(b, i);
                sum_dy_x += dy * self.weight.data[i] * x;
                
                // Accumulate gradient with respect to weight
                self.weight.grad[i] += dy * x * rms_inv;
            }
            
            for i in 0..dim {
                let dy = grad_output.grad.read(b, i);
                let x = input.data.read(b, i);
                let dx = rms_inv * (dy * self.weight.data[i] - x * rms_inv * rms_inv * sum_dy_x / dim as f32);
                let current_grad = grad_input.grad.read(b, i);
                grad_input.grad.write(b, i, current_grad + dx);
            }
        }
    }
}

impl Parameterized for RMSNorm {
    fn params(&self) -> Vec<&[f32]> {
        vec![&self.weight.data]
    }

    fn grads(&self) -> Vec<&[f32]> {
        vec![&self.weight.grad]
    }

    fn params_mut(&mut self) -> Vec<&mut [f32]> {
        vec![&mut self.weight.data]
    }

    fn grads_mut(&mut self) -> Vec<&mut [f32]> {
        vec![&mut self.weight.grad]
    }
}
