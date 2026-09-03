use super::tensor::{BatchTensor, Tensor1D, MatExt};
use super::ewc::{Parameterized, LoraLinear, LoraScratchpad};
use super::embedding::Embedding;

#[derive(Clone)]
pub struct ReadoutScratchpad {
    pub h_proj: BatchTensor,
    pub hidden_grad: BatchTensor,
    pub d_hidden: BatchTensor,
    pub lora_scratch: LoraScratchpad,
}

impl ReadoutScratchpad {
    pub fn new(batch_size: usize, embed_dim: usize, hidden_size: usize) -> Self {
        let max_lora_rank = 8;
        Self {
            h_proj: BatchTensor::new(batch_size, embed_dim),
            hidden_grad: BatchTensor::new(batch_size, hidden_size),
            d_hidden: BatchTensor::new(batch_size, hidden_size),
            lora_scratch: LoraScratchpad::new(batch_size, embed_dim, max_lora_rank),
        }
    }
}

pub struct ReadoutLayer {
    pub hidden_size: usize,
    pub embed_dim: usize,
    pub vocab_size: usize,
    
    pub w_proj: LoraLinear, // (embed_dim, hidden_size)
    pub b_proj: Tensor1D, // (embed_dim)
}

impl ReadoutLayer {
    pub fn new(hidden_size: usize, embed_dim: usize, vocab_size: usize, lora_rank: usize, lora_alpha: f32) -> Self {
        let mut w_proj = LoraLinear::new(embed_dim, hidden_size, lora_rank, lora_alpha);
        
        let limit = (6.0 / (hidden_size as f32 + embed_dim as f32)).sqrt();
        w_proj.randomize(-limit, limit);
        
        Self {
            hidden_size,
            embed_dim,
            vocab_size,
            w_proj,
            b_proj: Tensor1D::new(embed_dim),
        }
    }
    
    pub fn forward(&self, hidden: &BatchTensor, embedding: &Embedding, scratch: &mut ReadoutScratchpad, logits: &mut BatchTensor) {
        let b_size = hidden.data.rows;
        
        // 1. h_proj = hidden * w_proj^T (using LoRA)
        let mut h_proj = BatchTensor::new(b_size, self.embed_dim);
        self.w_proj.forward(hidden, &mut h_proj, &mut scratch.lora_scratch);
        
        // Add bias
        for b in 0..b_size {
            for i in 0..self.embed_dim {
                let val = h_proj.data.read(b, i) + self.b_proj.data[i];
                h_proj.data.write(b, i, val);
            }
        }
        
        // 2. logits = h_proj * embedding.weight^T
        embedding.weight.matmul_batch(&h_proj, logits);
    }
    
    pub fn backward<'a>(&mut self, hidden: &BatchTensor, d_logits: &BatchTensor, embedding: &mut Embedding, scratch: &'a mut ReadoutScratchpad) -> &'a BatchTensor {
        let b_size = hidden.data.rows;
        
        scratch.d_hidden.zero_grad();
        scratch.h_proj.zero_grad();
        
        // 1. Backprop through tied embedding weights
        embedding.weight.matmul_batch_backward(&mut scratch.h_proj, d_logits);
        
        // 2. Backprop through w_proj and b_proj
        scratch.hidden_grad.zero_grad();
        scratch.hidden_grad.data.copy_from(&hidden.data);
        
        self.w_proj.backward(&mut scratch.hidden_grad, &scratch.h_proj, &mut scratch.lora_scratch);
        
        for b in 0..b_size {
            for i in 0..self.embed_dim {
                self.b_proj.grad[i] += scratch.h_proj.grad.read(b, i);
            }
            for j in 0..self.hidden_size {
                scratch.d_hidden.grad.write(b, j, scratch.hidden_grad.grad.read(b, j));
            }
        }
        
        &scratch.d_hidden
    }
}

impl Parameterized for ReadoutLayer {
    fn params(&self) -> Vec<&[f32]> {
        let mut res = Vec::new();
        res.extend(self.w_proj.params());
        res.push(&self.b_proj.data);
        res
    }
    
    fn params_mut(&mut self) -> Vec<&mut [f32]> {
        let mut res = Vec::new();
        res.extend(self.w_proj.params_mut());
        res.push(&mut self.b_proj.data);
        res
    }
    
    fn grads(&self) -> Vec<&[f32]> {
        let mut res = Vec::new();
        res.extend(self.w_proj.grads());
        res.push(&self.b_proj.grad);
        res
    }
    
    fn grads_mut(&mut self) -> Vec<&mut [f32]> {
        let mut res = Vec::new();
        res.extend(self.w_proj.grads_mut());
        res.push(&mut self.b_proj.grad);
        res
    }
}
