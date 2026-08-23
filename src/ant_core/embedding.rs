use super::tensor::{BatchTensor, Tensor2D, MatExt};
use super::ewc::Parameterized;

pub struct Embedding {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub weight: Tensor2D,
}

impl Embedding {
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        let mut weight = Tensor2D::new(vocab_size, embed_dim);
        let limit = (3.0 / embed_dim as f32).sqrt();
        weight.randomize(-limit, limit);
        
        Self {
            vocab_size,
            embed_dim,
            weight,
        }
    }

    pub fn forward(&self, tokens: &[usize]) -> BatchTensor {
        let batch_size = tokens.len();
        let mut out = BatchTensor::new(batch_size, self.embed_dim);
        
        for b in 0..batch_size {
            let token_id = tokens[b];
            assert!(token_id < self.vocab_size, "token_id {} out of bounds {}", token_id, self.vocab_size);
            let offset = token_id * self.embed_dim;
            for i in 0..self.embed_dim {
                out.data.write(b, i, self.weight.data[offset + i]);
            }
        }
        out
    }

    pub fn backward(&mut self, tokens: &[usize], grad_output: &BatchTensor) {
        let batch_size = tokens.len();
        for b in 0..batch_size {
            let token_id = tokens[b];
            let offset = token_id * self.embed_dim;
            for i in 0..self.embed_dim {
                self.weight.grad[offset + i] += grad_output.grad.read(b, i);
            }
        }
    }
}

impl Parameterized for Embedding {
    fn params(&self) -> Vec<&[f32]> {
        vec![&self.weight.data]
    }
    
    fn params_mut(&mut self) -> Vec<&mut [f32]> {
        vec![&mut self.weight.data]
    }
    
    fn grads(&self) -> Vec<&[f32]> {
        vec![&self.weight.grad]
    }
    
    fn grads_mut(&mut self) -> Vec<&mut [f32]> {
        vec![&mut self.weight.grad]
    }
}
