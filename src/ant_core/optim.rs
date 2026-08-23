use super::tensor::{Tensor1D, Tensor2D};
use super::rnn::{MinGRU, GatedDeltaNet2};
use super::rmsnorm::RMSNorm;
use super::sparse_gating::SparseGating;
use super::ewc::LoraLinear;

pub struct SGD {
    pub learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }

    pub fn step_1d(&self, tensor: &mut Tensor1D) {
        for i in 0..tensor.len() {
            let mut g = tensor.grad[i];
            if !g.is_finite() { g = 0.0; }
            tensor.data[i] -= self.learning_rate * g;
        }
    }

    pub fn step_2d(&self, tensor: &mut Tensor2D) {
        for i in 0..tensor.data.len() {
            let mut g = tensor.grad[i];
            if !g.is_finite() { g = 0.0; }
            tensor.data[i] -= self.learning_rate * g;
        }
    }

    pub fn step_lora(&self, w: &mut LoraLinear) {
        self.step_2d(&mut w.base);
        self.step_2d(&mut w.lora_a);
        self.step_2d(&mut w.lora_b);
    }

    pub fn step_mingru(&self, gru: &mut MinGRU) {
        self.step_2d(&mut gru.w_z); self.step_1d(&mut gru.b_z);
        self.step_2d(&mut gru.w_h); self.step_1d(&mut gru.b_h);
    }

    pub fn zero_grad_mingru(&self, gru: &mut MinGRU) {
        for g in gru.w_z.grad.iter_mut() { *g = 0.0; }
        for g in gru.b_z.grad.iter_mut() { *g = 0.0; }
        for g in gru.w_h.grad.iter_mut() { *g = 0.0; }
        for g in gru.b_h.grad.iter_mut() { *g = 0.0; }
    }

    pub fn step_deltanet(&self, dn: &mut GatedDeltaNet2) {
        self.step_2d(&mut dn.w_k);
        self.step_2d(&mut dn.w_v);
        self.step_2d(&mut dn.w_q);
        self.step_2d(&mut dn.w_b);
        self.step_2d(&mut dn.w_w);
        self.step_2d(&mut dn.w_alpha);
    }

    pub fn zero_grad_deltanet(&self, dn: &mut GatedDeltaNet2) {
        for g in dn.w_k.grad.iter_mut() { *g = 0.0; }
        for g in dn.w_v.grad.iter_mut() { *g = 0.0; }
        for g in dn.w_q.grad.iter_mut() { *g = 0.0; }
        for g in dn.w_b.grad.iter_mut() { *g = 0.0; }
        for g in dn.w_w.grad.iter_mut() { *g = 0.0; }
        for g in dn.w_alpha.grad.iter_mut() { *g = 0.0; }
    }

    pub fn step_rmsnorm(&self, norm: &mut RMSNorm) {
        for i in 0..norm.weight.len() {
            let mut g = norm.weight.grad[i];
            if !g.is_finite() { g = 0.0; }
            norm.weight.data[i] -= self.learning_rate * g;
        }
    }

    pub fn zero_grad_rmsnorm(&self, norm: &mut RMSNorm) {
        for g in norm.weight.grad.iter_mut() { *g = 0.0; }
    }

    pub fn step_sparse_gating(&self, sg: &mut SparseGating) {
        self.step_2d(&mut sg.w1);
        self.step_1d(&mut sg.b1);
    }

    pub fn clip_sparse_gating(&self, sg: &mut SparseGating, max_norm: f32) {
        let total_norm_sq = Self::norm_2d(&sg.w1) + Self::norm_1d(&sg.b1);
        let total_norm = total_norm_sq.sqrt();
        if total_norm > max_norm && total_norm.is_finite() && total_norm > 0.0 {
            let scale = max_norm / total_norm;
            Self::scale_2d(&mut sg.w1, scale);
            Self::scale_1d(&mut sg.b1, scale);
        } else if !total_norm.is_finite() {
            Self::clean_2d(&mut sg.w1);
            Self::clean_1d(&mut sg.b1);
        }
    }

    pub fn step_readout(&mut self, readout: &mut super::readout::ReadoutLayer) {
        self.step_lora(&mut readout.w_proj);
        self.step_1d(&mut readout.b_proj);
    }

    pub fn zero_grad_readout(&self, readout: &mut super::readout::ReadoutLayer) {
        for grad in readout.w_proj.base.grad.iter_mut() { *grad = 0.0; }
        for grad in readout.w_proj.lora_a.grad.iter_mut() { *grad = 0.0; }
        for grad in readout.w_proj.lora_b.grad.iter_mut() { *grad = 0.0; }
        for grad in readout.b_proj.grad.iter_mut() { *grad = 0.0; }
    }

    pub fn clip_readout(&self, readout: &mut crate::ant_core::readout::ReadoutLayer, max_norm: f32) {
        let total_norm_sq = Self::norm_lora(&readout.w_proj) + Self::norm_1d(&readout.b_proj);
        let total_norm = total_norm_sq.sqrt();
        if total_norm > max_norm && total_norm.is_finite() && total_norm > 0.0 {
            let scale = max_norm / total_norm;
            Self::scale_lora(&mut readout.w_proj, scale);
            Self::scale_1d(&mut readout.b_proj, scale);
        } else if !total_norm.is_finite() {
            Self::clean_lora(&mut readout.w_proj);
            Self::clean_1d(&mut readout.b_proj);
        }
    }

    pub fn step_embedding(&self, emb: &mut super::embedding::Embedding) {
        let rows = emb.weight.rows;
        let cols = emb.weight.cols;
        for r in 0..rows {
            let offset = r * cols;
            let mut has_nonzero = false;
            for i in 0..cols {
                if emb.weight.grad[offset + i] != 0.0 {
                    has_nonzero = true;
                    break;
                }
            }
            if has_nonzero {
                for i in 0..cols {
                    let g = emb.weight.grad[offset + i];
                    if g.is_finite() {
                        emb.weight.data[offset + i] -= self.learning_rate * g;
                    } else {
                        emb.weight.grad[offset + i] = 0.0;
                    }
                }
            } else {
                for i in 0..cols {
                    let g = emb.weight.grad[offset + i];
                    if !g.is_finite() {
                        emb.weight.grad[offset + i] = 0.0;
                    }
                }
            }
        }
    }

    pub fn step_memory_attention(&self, attn: &mut super::memory_attention::MemoryAttention) {
        self.step_lora(&mut attn.w_q); self.step_1d(&mut attn.b_q);
        self.step_lora(&mut attn.w_fuse); self.step_1d(&mut attn.b_fuse);
    }

    pub fn zero_grad_memory_attention(&self, attn: &mut super::memory_attention::MemoryAttention) {
        for grad in attn.w_q.base.grad.iter_mut() { *grad = 0.0; }
        for grad in attn.w_q.lora_a.grad.iter_mut() { *grad = 0.0; }
        for grad in attn.w_q.lora_b.grad.iter_mut() { *grad = 0.0; }
        for grad in attn.b_q.grad.iter_mut() { *grad = 0.0; }

        for grad in attn.w_fuse.base.grad.iter_mut() { *grad = 0.0; }
        for grad in attn.w_fuse.lora_a.grad.iter_mut() { *grad = 0.0; }
        for grad in attn.w_fuse.lora_b.grad.iter_mut() { *grad = 0.0; }
        for grad in attn.b_fuse.grad.iter_mut() { *grad = 0.0; }
    }

    pub fn clip_memory_attention(&self, attn: &mut super::memory_attention::MemoryAttention, max_norm: f32) {
        let mut total_norm_sq = 0.0;
        total_norm_sq += Self::norm_lora(&attn.w_q) + Self::norm_1d(&attn.b_q);
        total_norm_sq += Self::norm_lora(&attn.w_fuse) + Self::norm_1d(&attn.b_fuse);

        let total_norm = total_norm_sq.sqrt();
        if total_norm > max_norm && total_norm.is_finite() && total_norm > 0.0 {
            let scale = max_norm / total_norm;
            Self::scale_lora(&mut attn.w_q, scale); Self::scale_1d(&mut attn.b_q, scale);
            Self::scale_lora(&mut attn.w_fuse, scale); Self::scale_1d(&mut attn.b_fuse, scale);
        } else if !total_norm.is_finite() {
            Self::clean_lora(&mut attn.w_q); Self::clean_1d(&mut attn.b_q);
            Self::clean_lora(&mut attn.w_fuse); Self::clean_1d(&mut attn.b_fuse);
        }
    }

    pub fn zero_grad_sparse_gating(&self, sg: &mut SparseGating) {
        for g in sg.w1.grad.iter_mut() { *g = 0.0; }
        for g in sg.b1.grad.iter_mut() { *g = 0.0; }
    }

    fn norm_1d(tensor: &Tensor1D) -> f32 {
        tensor.grad.iter().map(|&g| if g.is_finite() { g * g } else { 0.0 }).sum()
    }

    fn norm_2d(tensor: &Tensor2D) -> f32 {
        tensor.grad.iter().map(|&g| if g.is_finite() { g * g } else { 0.0 }).sum()
    }

    fn norm_lora(w: &LoraLinear) -> f32 {
        Self::norm_2d(&w.base) + Self::norm_2d(&w.lora_a) + Self::norm_2d(&w.lora_b)
    }

    fn scale_1d(tensor: &mut Tensor1D, scale: f32) {
        for g in tensor.grad.iter_mut() {
            if g.is_finite() {
                *g *= scale;
            } else {
                *g = 0.0;
            }
        }
    }

    fn scale_2d(tensor: &mut Tensor2D, scale: f32) {
        for g in tensor.grad.iter_mut() {
            if g.is_finite() {
                *g *= scale;
            } else {
                *g = 0.0;
            }
        }
    }

    fn scale_lora(w: &mut LoraLinear, scale: f32) {
        Self::scale_2d(&mut w.base, scale);
        Self::scale_2d(&mut w.lora_a, scale);
        Self::scale_2d(&mut w.lora_b, scale);
    }

    fn clean_1d(tensor: &mut Tensor1D) {
        for g in tensor.grad.iter_mut() {
            if !g.is_finite() { *g = 0.0; }
        }
    }

    fn clean_2d(tensor: &mut Tensor2D) {
        for g in tensor.grad.iter_mut() {
            if !g.is_finite() { *g = 0.0; }
        }
    }

    fn clean_lora(w: &mut LoraLinear) {
        Self::clean_2d(&mut w.base);
        Self::clean_2d(&mut w.lora_a);
        Self::clean_2d(&mut w.lora_b);
    }

    pub fn zero_grad_embedding(&self, emb: &mut super::embedding::Embedding) {
        for g in emb.weight.grad.iter_mut() { *g = 0.0; }
    }

    pub fn clip_embedding(&self, emb: &mut super::embedding::Embedding, max_norm: f32) {
        let total_norm = Self::norm_2d(&emb.weight).sqrt();
        if total_norm > max_norm && total_norm.is_finite() && total_norm > 0.0 {
            let scale = max_norm / total_norm;
            Self::scale_2d(&mut emb.weight, scale);
        } else if !total_norm.is_finite() {
            Self::clean_2d(&mut emb.weight);
        }
    }

    pub fn clip_mingru(&self, gru: &mut MinGRU, max_norm: f32) {
        let mut total_norm_sq = 0.0;
        total_norm_sq += Self::norm_2d(&gru.w_z) + Self::norm_1d(&gru.b_z);
        total_norm_sq += Self::norm_2d(&gru.w_h) + Self::norm_1d(&gru.b_h);
        
        let total_norm = total_norm_sq.sqrt();
        if total_norm > max_norm && total_norm.is_finite() && total_norm > 0.0 {
            let scale = max_norm / total_norm;
            Self::scale_2d(&mut gru.w_z, scale); Self::scale_1d(&mut gru.b_z, scale);
            Self::scale_2d(&mut gru.w_h, scale); Self::scale_1d(&mut gru.b_h, scale);
        } else if !total_norm.is_finite() {
            Self::clean_2d(&mut gru.w_z); Self::clean_1d(&mut gru.b_z);
            Self::clean_2d(&mut gru.w_h); Self::clean_1d(&mut gru.b_h);
        }
    }

    pub fn clip_rmsnorm(&self, norm: &mut RMSNorm, max_norm: f32) {
        let total_norm = Self::norm_1d(&norm.weight).sqrt();
        if total_norm > max_norm && total_norm.is_finite() && total_norm > 0.0 {
            let scale = max_norm / total_norm;
            Self::scale_1d(&mut norm.weight, scale);
        } else if !total_norm.is_finite() {
            Self::clean_1d(&mut norm.weight);
        }
    }

    pub fn clip_deltanet(&self, dn: &mut GatedDeltaNet2, max_norm: f32) {
        let mut total_norm_sq = 0.0;
        total_norm_sq += Self::norm_2d(&dn.w_k);
        total_norm_sq += Self::norm_2d(&dn.w_v);
        total_norm_sq += Self::norm_2d(&dn.w_q);
        total_norm_sq += Self::norm_2d(&dn.w_b);
        total_norm_sq += Self::norm_2d(&dn.w_w);
        total_norm_sq += Self::norm_2d(&dn.w_alpha);
        
        let total_norm = total_norm_sq.sqrt();
        if total_norm > max_norm && total_norm.is_finite() && total_norm > 0.0 {
            let scale = max_norm / total_norm;
            Self::scale_2d(&mut dn.w_k, scale);
            Self::scale_2d(&mut dn.w_v, scale);
            Self::scale_2d(&mut dn.w_q, scale);
            Self::scale_2d(&mut dn.w_b, scale);
            Self::scale_2d(&mut dn.w_w, scale);
            Self::scale_2d(&mut dn.w_alpha, scale);
        } else if !total_norm.is_finite() {
            Self::clean_2d(&mut dn.w_k);
            Self::clean_2d(&mut dn.w_v);
            Self::clean_2d(&mut dn.w_q);
            Self::clean_2d(&mut dn.w_b);
            Self::clean_2d(&mut dn.w_w);
            Self::clean_2d(&mut dn.w_alpha);
        }
    }
}
