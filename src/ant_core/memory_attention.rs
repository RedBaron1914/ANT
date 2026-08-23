use super::tensor::{BatchTensor, Tensor1D, MatExt};
use super::memory_io::DiskKVMemory;
use super::ewc::{Parameterized, LoraLinear, LoraScratchpad};

#[derive(Clone)]
pub struct MemoryAttentionCache {
    pub query: BatchTensor,
    pub top_k_indices: Vec<Vec<usize>>,
    pub scores: Vec<Vec<f32>>,
    pub alpha: Vec<Vec<f32>>,
    pub mem_out: BatchTensor,
    pub delayed_mem_out: BatchTensor,
    
    // Backward scratchpad variables
    pub d_hidden: BatchTensor,
    pub d_q_in: BatchTensor,
    pub mem_out_grad: BatchTensor,
    pub hidden_grad: BatchTensor,

    // LoRA scratchpads
    pub lora_q_scratch: LoraScratchpad,
    pub lora_fuse_scratch: LoraScratchpad,
}

impl MemoryAttentionCache {
    pub fn new(batch_size: usize, key_dim: usize, hidden_dim: usize, top_k: usize) -> Self {
        let max_lora_rank = 8; // default Rank 8
        Self {
            query: BatchTensor::new(batch_size, key_dim),
            top_k_indices: vec![Vec::with_capacity(top_k); batch_size],
            scores: vec![Vec::with_capacity(top_k); batch_size],
            alpha: vec![Vec::with_capacity(top_k); batch_size],
            mem_out: BatchTensor::new(batch_size, hidden_dim),
            delayed_mem_out: BatchTensor::new(batch_size, hidden_dim),
            
            d_hidden: BatchTensor::new(batch_size, hidden_dim),
            d_q_in: BatchTensor::new(batch_size, key_dim),
            mem_out_grad: BatchTensor::new(batch_size, hidden_dim),
            hidden_grad: BatchTensor::new(batch_size, hidden_dim),

            lora_q_scratch: LoraScratchpad::new(batch_size, key_dim, max_lora_rank),
            lora_fuse_scratch: LoraScratchpad::new(batch_size, hidden_dim, max_lora_rank),
        }
    }
}

pub struct MemoryAttention {
    pub key_dim: usize,
    pub hidden_dim: usize,
    pub top_k_base: usize,
    pub top_k_user: usize,

    pub w_q: LoraLinear,     // (key_dim, hidden_dim) = (64, 256)
    pub b_q: Tensor1D,     // (key_dim)

    pub w_fuse: LoraLinear,  // (hidden_dim, hidden_dim) = (256, 256)
    pub b_fuse: Tensor1D,  // (hidden_dim)
}

const USER_MEM_FLAG: usize = 1 << 60;

impl MemoryAttention {
    pub fn new(key_dim: usize, hidden_dim: usize, top_k_base: usize, top_k_user: usize, lora_rank: usize, lora_alpha: f32) -> Self {
        let mut w_q = LoraLinear::new(key_dim, hidden_dim, lora_rank, lora_alpha);
        let limit_q = (6.0 / (key_dim + hidden_dim) as f32).sqrt();
        w_q.randomize(-limit_q, limit_q);

        let mut w_fuse = LoraLinear::new(hidden_dim, hidden_dim, lora_rank, lora_alpha);
        let limit_f = (6.0 / (2 * hidden_dim) as f32).sqrt();
        w_fuse.randomize(-limit_f, limit_f);

        Self {
            key_dim,
            hidden_dim,
            top_k_base,
            top_k_user,
            w_q,
            b_q: Tensor1D::new(key_dim),
            w_fuse,
            b_fuse: Tensor1D::new(hidden_dim),
        }
    }

    pub fn forward(
        &self,
        hidden_norm: &BatchTensor,
        hidden_raw: &BatchTensor,
        base_memory: &DiskKVMemory,
        user_memory: &DiskKVMemory,
        query_metadata: u64,
        cache: &mut MemoryAttentionCache,
    ) -> BatchTensor {
        cache.mem_out.data.fill(0.0);

        let b_size = hidden_norm.data.nrows();
        let scale = 1.0 / (self.key_dim as f32).sqrt();

        // 1. Query projection: q = tanh(W_q * h_norm + b_q)
        let mut q_in = BatchTensor::new(b_size, self.key_dim);
        self.w_q.forward(hidden_norm, &mut q_in, &mut cache.lora_q_scratch);
        
        for b in 0..b_size {
            for i in 0..self.key_dim {
                cache.query.data.write(b, i, (q_in.data.read(b, i) + self.b_q.data[i]).tanh());
            }
        }

        let query_polarity = query_metadata & 1;

        // Loop over batch for attention lookup
        for b in 0..b_size {
            let mut q_slice = Tensor1D::new(self.key_dim);
            for i in 0..self.key_dim { q_slice.data[i] = cache.query.data.read(b, i); }
            
            // 2. Dual-Memory lookup
            let (_, base_indices) = base_memory.lookup(&q_slice, query_metadata, self.top_k_base);
            let (_, user_indices) = user_memory.lookup(&q_slice, query_metadata, self.top_k_user);
            
            let mut candidates = Vec::new();
            for &idx in &base_indices {
                let key = base_memory.get_key(idx);
                let mut dot = 0.0;
                for j in 0..self.key_dim { dot += q_slice.data[j] * key[j]; }
                let polarity = base_memory.get_metadata(idx) & 1;
                if query_polarity != polarity {
                    dot -= 0.5;
                }
                candidates.push((idx, dot * scale));
            }
            for &idx in &user_indices {
                let key = user_memory.get_key(idx);
                let mut dot = 0.0;
                for j in 0..self.key_dim { dot += q_slice.data[j] * key[j]; }
                let polarity = user_memory.get_metadata(idx) & 1;
                if query_polarity != polarity {
                    dot -= 0.5;
                }
                candidates.push((idx | USER_MEM_FLAG, dot * scale));
            }

            // Sort descending by score
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            let total_k = std::cmp::min(self.top_k_base + self.top_k_user, candidates.len());
            let top_candidates = &candidates[0..total_k];
            
            cache.top_k_indices[b].clear();
            cache.scores[b].clear();
            for &(idx, score) in top_candidates {
                cache.top_k_indices[b].push(idx);
                cache.scores[b].push(score);
            }
            
            let k = cache.top_k_indices[b].len();
            if k == 0 {
                cache.alpha[b].clear();
                continue;
            }

            // 4. Softmax with Temperature Scaling (T=0.2) + Threshold (5%) + Renormalization
            cache.alpha[b].resize(k, 0.0);
            let temp = 0.2;
            let threshold = 0.05;
            
            let mut max_score = f32::NEG_INFINITY;
            for i in 0..k {
                let s = cache.scores[b][i] / temp;
                if s > max_score {
                    max_score = s;
                }
            }
            
            let mut sum_exp = 0.0;
            for i in 0..k {
                let e = (cache.scores[b][i] / temp - max_score).exp();
                cache.alpha[b][i] = e;
                sum_exp += e;
            }
            
            for i in 0..k {
                cache.alpha[b][i] /= sum_exp;
            }
            
            let mut active_sum = 0.0;
            for i in 0..k {
                if cache.alpha[b][i] < threshold {
                    cache.alpha[b][i] = 0.0;
                }
                active_sum += cache.alpha[b][i];
            }
            
            if active_sum > 0.0 {
                for i in 0..k {
                    cache.alpha[b][i] /= active_sum;
                }
            } else {
                let mut max_idx = 0;
                let mut max_val = f32::NEG_INFINITY;
                for i in 0..k {
                    let score = cache.scores[b][i];
                    if score > max_val {
                        max_val = score;
                        max_idx = i;
                    }
                }
                cache.alpha[b][max_idx] = 1.0;
            }

            // 5. Weighted sum over memory values
            for (i, &idx) in cache.top_k_indices[b].iter().enumerate() {
                let val = if (idx & USER_MEM_FLAG) != 0 {
                    user_memory.get_val(idx & !USER_MEM_FLAG)
                } else {
                    base_memory.get_val(idx)
                };
                let weight = cache.alpha[b][i];
                if weight > 0.0 {
                    for j in 0..self.hidden_dim {
                        let prev = cache.mem_out.data.read(b, j);
                        cache.mem_out.data.write(b, j, prev + val[j] * weight);
                    }
                }
            }
        }

        // 6. Residual fusion: fused_h = h_raw + W_fuse * mem_out + b_fuse
        let mut fuse_in = BatchTensor::new(b_size, self.hidden_dim);
        self.w_fuse.forward(&cache.mem_out, &mut fuse_in, &mut cache.lora_fuse_scratch);

        let mut fused_h = BatchTensor::new(b_size, self.hidden_dim);
        for b in 0..b_size {
            for i in 0..self.hidden_dim {
                fused_h.data.write(b, i, hidden_raw.data.read(b, i) + fuse_in.data.read(b, i) + self.b_fuse.data[i]);
            }
        }

        fused_h
    }

    pub fn backward<'a>(
        &mut self,
        hidden_norm: &mut BatchTensor,
        d_fused: &BatchTensor,
        base_memory: &DiskKVMemory,
        user_memory: &DiskKVMemory,
        cache: &'a mut MemoryAttentionCache,
        grad_hidden_raw: &mut BatchTensor,
    ) -> &'a BatchTensor {
        let b_size = hidden_norm.data.rows;
        cache.d_hidden.zero_grad();
        
        let mut has_mem = false;
        for b in 0..b_size {
            if !cache.top_k_indices[b].is_empty() {
                has_mem = true;
                break;
            }
        }
        
        if !has_mem {
            for b in 0..b_size {
                for j in 0..self.hidden_dim {
                    let dy = d_fused.grad.read(b, j);
                    let current = grad_hidden_raw.grad.read(b, j);
                    grad_hidden_raw.grad.write(b, j, current + dy);
                }
            }
            return &cache.d_hidden;
        }

        // 1. Gradient through residual fusion
        for b in 0..b_size {
            for j in 0..self.hidden_dim {
                let dy = d_fused.grad.read(b, j);
                let current = grad_hidden_raw.grad.read(b, j);
                grad_hidden_raw.grad.write(b, j, current + dy);
                self.b_fuse.grad[j] += dy;
            }
        }

        // Backprop through W_fuse to get d_mem_out
        cache.mem_out_grad.zero_grad();
        self.w_fuse.backward(&mut cache.mem_out, d_fused, &mut cache.lora_fuse_scratch);
        
        cache.d_q_in.zero_grad();
        let scale = 1.0 / (self.key_dim as f32).sqrt();

        for b in 0..b_size {
            let k = cache.top_k_indices[b].len();
            if k == 0 { continue; }

            // 2. Gradient through weighted sum: mem_out = sum(alpha[i] * val[i])
            let mut d_alpha = vec![0.0; k];
            for (i, &idx) in cache.top_k_indices[b].iter().enumerate() {
                let val = if (idx & USER_MEM_FLAG) != 0 {
                    user_memory.get_val(idx & !USER_MEM_FLAG)
                } else {
                    base_memory.get_val(idx)
                };
                let mut dot = 0.0;
                for j in 0..self.hidden_dim {
                    dot += cache.mem_out_grad.grad.read(b, j) * val[j];
                }
                d_alpha[i] = dot;
            }

            // 3. Softmax backward: d_scores[i] = (1 / temp) * alpha[i] * (d_alpha[i] - sum(alpha[j] * d_alpha[j]))
            let temp = 0.2;
            let mut sum_alpha_dalpha = 0.0;
            for j in 0..k {
                sum_alpha_dalpha += cache.alpha[b][j] * d_alpha[j];
            }

            let mut d_scores = vec![0.0; k];
            for i in 0..k {
                d_scores[i] = (1.0 / temp) * cache.alpha[b][i] * (d_alpha[i] - sum_alpha_dalpha);
            }

            // 4. Scaled dot product backward
            let mut d_query_b = vec![0.0; self.key_dim];
            for (i, &idx) in cache.top_k_indices[b].iter().enumerate() {
                let key = if (idx & USER_MEM_FLAG) != 0 {
                    user_memory.get_key(idx & !USER_MEM_FLAG)
                } else {
                    base_memory.get_key(idx)
                };
                let ds = d_scores[i] * scale;
                for j in 0..self.key_dim {
                    d_query_b[j] += ds * key[j];
                }
            }

            // 5. Query projection backward: q = tanh(q_in)
            for j in 0..self.key_dim {
                let q_val = cache.query.data.read(b, j);
                let dtanh = 1.0 - q_val * q_val;
                let grad_val = d_query_b[j] * dtanh;
                cache.d_q_in.grad.write(b, j, grad_val);
                self.b_q.grad[j] += grad_val;
            }
        }

        cache.hidden_grad.zero_grad();
        self.w_q.backward(hidden_norm, &cache.d_q_in, &mut cache.lora_q_scratch);

        for b in 0..b_size {
            for j in 0..self.hidden_dim {
                let prev = cache.d_hidden.grad.read(b, j);
                cache.d_hidden.grad.write(b, j, prev + cache.hidden_grad.grad.read(b, j));
            }
        }

        &cache.d_hidden
    }
}

impl Parameterized for MemoryAttention {
    fn params(&self) -> Vec<&[f32]> {
        let mut res = Vec::new();
        res.extend(self.w_q.params());
        res.push(&self.b_q.data);
        res.extend(self.w_fuse.params());
        res.push(&self.b_fuse.data);
        res
    }

    fn grads(&self) -> Vec<&[f32]> {
        let mut res = Vec::new();
        res.extend(self.w_q.grads());
        res.push(&self.b_q.grad);
        res.extend(self.w_fuse.grads());
        res.push(&self.b_fuse.grad);
        res
    }

    fn params_mut(&mut self) -> Vec<&mut [f32]> {
        let mut res = Vec::new();
        res.extend(self.w_q.params_mut());
        res.push(&mut self.b_q.data);
        res.extend(self.w_fuse.params_mut());
        res.push(&mut self.b_fuse.data);
        res
    }

    fn grads_mut(&mut self) -> Vec<&mut [f32]> {
        let mut res = Vec::new();
        res.extend(self.w_q.grads_mut());
        res.push(&mut self.b_q.grad);
        res.extend(self.w_fuse.grads_mut());
        res.push(&mut self.b_fuse.grad);
        res
    }
}
