use super::tensor::{BatchTensor, Tensor1D, Tensor2D, MatExt};
use super::ewc::Parameterized;

#[derive(Clone)]
pub struct SparseGatingScratchpad {
    pub x: BatchTensor,
    pub pre_activation: BatchTensor,
    
    // Backward scratchpad variables
    pub d_pre: BatchTensor,
    pub grad_x: BatchTensor,
}

impl SparseGatingScratchpad {
    pub fn new(batch_size: usize, input_size: usize, hidden_size: usize) -> Self {
        Self {
            x: BatchTensor::new(batch_size, input_size),
            pre_activation: BatchTensor::new(batch_size, hidden_size),
            d_pre: BatchTensor::new(batch_size, hidden_size),
            grad_x: BatchTensor::new(batch_size, input_size),
        }
    }
}

#[derive(Clone)]
pub struct SparseGating {
    pub w1: Tensor2D,
    pub b1: Tensor1D,
    pub hidden: BatchTensor,
}

impl SparseGating {
    pub fn new(batch_size: usize, input_size: usize, hidden_size: usize) -> Self {
        Self {
            w1: Tensor2D::new(hidden_size, input_size),
            b1: Tensor1D::new(hidden_size),
            hidden: BatchTensor::new(batch_size, hidden_size),
        }
    }
    
    pub fn forward_with_cache(&mut self, input: &BatchTensor, cache: &mut SparseGatingScratchpad) {
        cache.x.data.copy_from(&input.data);
        
        self.w1.matmul_batch(input, &mut self.hidden);
        
        let h_size = self.hidden.data.ncols();
        let b_size = self.hidden.data.nrows();
        
        // Add bias + ReLU activation (Sparse representation)
        for b in 0..b_size {
            for i in 0..h_size {
                let val = self.hidden.data.read(b, i) + self.b1.data[i];
                cache.pre_activation.data.write(b, i, val);
                self.hidden.data.write(b, i, if val > 0.0 { val } else { 0.0 });
            }
        }
    }

    pub fn forward<'a>(&'a mut self, input: &BatchTensor, scratchpad: &mut SparseGatingScratchpad) -> &'a BatchTensor {
        self.forward_with_cache(input, scratchpad);
        &self.hidden
    }

    pub fn backward<'a>(&mut self, grad_output: &BatchTensor, cache: &'a mut SparseGatingScratchpad) -> &'a BatchTensor {
        let h_size = self.hidden.data.cols;
        let b_size = cache.x.data.rows;
        
        cache.d_pre.zero_grad();
        cache.grad_x.zero_grad();
        
        for b in 0..b_size {
            for i in 0..h_size {
                // ReLU derivative
                let d_relu = if cache.pre_activation.data.read(b, i) > 0.0 { 1.0 } else { 0.0 };
                let d_val = grad_output.grad.read(b, i) * d_relu;
                cache.d_pre.grad.write(b, i, d_val);
                self.b1.grad[i] += d_val;
            }
        }
        
        cache.grad_x.data.copy_from(&cache.x.data);
        self.w1.matmul_batch_backward(&mut cache.grad_x, &cache.d_pre);
        
        &cache.grad_x
    }

    pub fn randomize(&mut self, min: f32, max: f32) {
        self.w1.randomize(min, max);
        self.b1.randomize(min, max);
    }
}

impl Parameterized for SparseGating {
    fn params(&self) -> Vec<&[f32]> {
        vec![&self.w1.data, &self.b1.data]
    }

    fn grads(&self) -> Vec<&[f32]> {
        vec![&self.w1.grad, &self.b1.grad]
    }

    fn params_mut(&mut self) -> Vec<&mut [f32]> {
        vec![&mut self.w1.data, &mut self.b1.data]
    }

    fn grads_mut(&mut self) -> Vec<&mut [f32]> {
        vec![&mut self.w1.grad, &mut self.b1.grad]
    }
}

