#![allow(non_snake_case)]
use super::tensor::{BatchTensor, Tensor1D, Tensor2D, MatExt};
use super::ewc::Parameterized;

// =========================================================================
// MinGRU CPU Implementation
// =========================================================================

#[derive(Clone, Debug)]
pub struct MinGruScratchpad {
    pub x: BatchTensor,
    pub prev_h: BatchTensor,
    pub z: BatchTensor,
    pub h_tilde: BatchTensor,
    pub temp_z_in: BatchTensor,
    pub temp_h_in: BatchTensor,
    
    // Backward scratchpad variables
    pub grad_x: BatchTensor,
    pub grad_prev_h: BatchTensor,
    pub d_z_pre: BatchTensor,
    pub d_h_pre: BatchTensor,
}

impl MinGruScratchpad {
    pub fn new(batch_size: usize, input_size: usize, hidden_size: usize) -> Self {
        Self {
            x: BatchTensor::new(batch_size, input_size),
            prev_h: BatchTensor::new(batch_size, hidden_size),
            z: BatchTensor::new(batch_size, hidden_size),
            h_tilde: BatchTensor::new(batch_size, hidden_size),
            temp_z_in: BatchTensor::new(batch_size, hidden_size),
            temp_h_in: BatchTensor::new(batch_size, hidden_size),
            
            grad_x: BatchTensor::new(batch_size, input_size),
            grad_prev_h: BatchTensor::new(batch_size, hidden_size),
            d_z_pre: BatchTensor::new(batch_size, hidden_size),
            d_h_pre: BatchTensor::new(batch_size, hidden_size),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MinGRU {
    pub w_z: Tensor2D,
    pub b_z: Tensor1D,
    pub w_h: Tensor2D,
    pub b_h: Tensor1D,
    pub hidden_state: BatchTensor,
}

impl MinGRU {
    pub fn new(batch_size: usize, input_size: usize, hidden_size: usize) -> Self {
        Self {
            w_z: Tensor2D::new(hidden_size, input_size),
            b_z: Tensor1D::new(hidden_size),
            w_h: Tensor2D::new(hidden_size, input_size),
            b_h: Tensor1D::new(hidden_size),
            hidden_state: BatchTensor::new(batch_size, hidden_size),
        }
    }
    
    pub fn randomize(&mut self, min: f32, max: f32) {
        self.w_z.randomize(min, max);
        self.b_z.randomize(min, max);
        self.w_h.randomize(min, max);
        self.b_h.randomize(min, max);
    }

    pub fn forward_with_cache(&mut self, input: &BatchTensor, cache: &mut MinGruScratchpad) {
        let h_size = self.hidden_state.data.ncols();
        let b_size = input.data.nrows();
        
        cache.x.data.copy_from(&input.data);
        cache.prev_h.data.copy_from(&self.hidden_state.data);

        // Linear projections
        self.w_z.matmul_batch(input, &mut cache.temp_z_in);
        self.w_h.matmul_batch(input, &mut cache.temp_h_in);
        
        for b in 0..b_size {
            for i in 0..h_size {
                let z_val = cache.temp_z_in.data.read(b, i) + self.b_z.data[i];
                let h_tilde_val = cache.temp_h_in.data.read(b, i) + self.b_h.data[i];
                
                let z = 1.0 / (1.0 + (-z_val).exp());
                cache.z.data.write(b, i, z);
                cache.h_tilde.data.write(b, i, h_tilde_val);
                
                let prev_h = cache.prev_h.data.read(b, i);
                let next_h = (1.0 - z) * prev_h + z * h_tilde_val;
                self.hidden_state.data.write(b, i, next_h);
            }
        }
    }

    pub fn forward<'a>(&'a mut self, input: &BatchTensor, cache: &mut MinGruScratchpad) -> &'a BatchTensor {
        self.forward_with_cache(input, cache);
        &self.hidden_state
    }

    pub fn backward<'a>(&mut self, grad_h: &BatchTensor, cache: &'a mut MinGruScratchpad) -> (&'a BatchTensor, &'a BatchTensor) {
        let h_size = self.hidden_state.data.cols;
        let b_size = cache.x.data.rows;
        
        cache.grad_x.zero_grad();
        cache.grad_prev_h.zero_grad();
        cache.d_z_pre.zero_grad();
        cache.d_h_pre.zero_grad();
        
        for b in 0..b_size {
            for i in 0..h_size {
                let dh = grad_h.grad.read(b, i);
                let z = cache.z.data.read(b, i);
                let prev_h = cache.prev_h.data.read(b, i);
                let h_tilde = cache.h_tilde.data.read(b, i);
                
                let dz = dh * (h_tilde - prev_h);
                let dz_pre = dz * z * (1.0 - z);
                cache.d_z_pre.grad.write(b, i, dz_pre);
                
                let dh_pre = dh * z;
                cache.d_h_pre.grad.write(b, i, dh_pre);
                
                cache.grad_prev_h.grad.write(b, i, dh * (1.0 - z));
                
                self.b_z.grad[i] += dz_pre;
                self.b_h.grad[i] += dh_pre;
            }
        }
        
        cache.grad_x.data.copy_from(&cache.x.data);
        cache.grad_prev_h.data.copy_from(&cache.prev_h.data);

        self.w_z.matmul_batch_backward(&mut cache.grad_x, &cache.d_z_pre);
        self.w_h.matmul_batch_backward(&mut cache.grad_x, &cache.d_h_pre);

        (&cache.grad_x, &cache.grad_prev_h)
    }
}

impl Parameterized for MinGRU {
    fn params(&self) -> Vec<&[f32]> {
        vec![
            &self.w_z.data, &self.b_z.data,
            &self.w_h.data, &self.b_h.data,
        ]
    }

    fn grads(&self) -> Vec<&[f32]> {
        vec![
            &self.w_z.grad, &self.b_z.grad,
            &self.w_h.grad, &self.b_h.grad,
        ]
    }

    fn params_mut(&mut self) -> Vec<&mut [f32]> {
        vec![
            &mut self.w_z.data, &mut self.b_z.data,
            &mut self.w_h.data, &mut self.b_h.data,
        ]
    }

    fn grads_mut(&mut self) -> Vec<&mut [f32]> {
        vec![
            &mut self.w_z.grad, &mut self.b_z.grad,
            &mut self.w_h.grad, &mut self.b_h.grad,
        ]
    }
}


// =========================================================================
// Gated DeltaNet-2 CPU Implementation
// =========================================================================

#[derive(Clone, Debug)]
pub struct GatedDeltaNet2Scratchpad {
    pub k: BatchTensor,
    pub v: BatchTensor,
    pub q: BatchTensor,
    pub b: BatchTensor,
    pub w: BatchTensor,
    pub alpha: BatchTensor,
    
    pub temp_k: BatchTensor,
    pub temp_v: BatchTensor,
    pub temp_q: BatchTensor,
    pub temp_b: BatchTensor,
    pub temp_w: BatchTensor,
    pub temp_alpha: BatchTensor,
    
    pub S_prev: Vec<f32>, // batch_size * hidden_size * hidden_size
    pub S_curr: Vec<f32>, // batch_size * hidden_size * hidden_size
    pub y: BatchTensor,
    
    // Gradients
    pub grad_x: BatchTensor,
    pub d_k: BatchTensor,
    pub d_v: BatchTensor,
    pub d_q: BatchTensor,
    pub d_b: BatchTensor,
    pub d_w: BatchTensor,
    pub d_alpha: BatchTensor,
}

impl GatedDeltaNet2Scratchpad {
    pub fn new(batch_size: usize, input_size: usize, hidden_size: usize) -> Self {
        Self {
            k: BatchTensor::new(batch_size, hidden_size),
            v: BatchTensor::new(batch_size, hidden_size),
            q: BatchTensor::new(batch_size, hidden_size),
            b: BatchTensor::new(batch_size, hidden_size),
            w: BatchTensor::new(batch_size, hidden_size),
            alpha: BatchTensor::new(batch_size, hidden_size),
            
            temp_k: BatchTensor::new(batch_size, hidden_size),
            temp_v: BatchTensor::new(batch_size, hidden_size),
            temp_q: BatchTensor::new(batch_size, hidden_size),
            temp_b: BatchTensor::new(batch_size, hidden_size),
            temp_w: BatchTensor::new(batch_size, hidden_size),
            temp_alpha: BatchTensor::new(batch_size, hidden_size),
            
            S_prev: vec![0.0; batch_size * hidden_size * hidden_size],
            S_curr: vec![0.0; batch_size * hidden_size * hidden_size],
            y: BatchTensor::new(batch_size, hidden_size),
            
            grad_x: BatchTensor::new(batch_size, input_size),
            d_k: BatchTensor::new(batch_size, hidden_size),
            d_v: BatchTensor::new(batch_size, hidden_size),
            d_q: BatchTensor::new(batch_size, hidden_size),
            d_b: BatchTensor::new(batch_size, hidden_size),
            d_w: BatchTensor::new(batch_size, hidden_size),
            d_alpha: BatchTensor::new(batch_size, hidden_size),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GatedDeltaNet2 {
    pub w_k: Tensor2D,
    pub w_v: Tensor2D,
    pub w_q: Tensor2D,
    pub w_b: Tensor2D,
    pub w_w: Tensor2D,
    pub w_alpha: Tensor2D,
    
    // Recurrent state (for inference, size: batch_size * hidden_size * hidden_size)
    pub state: Vec<f32>,
    pub hidden_size: usize,
}

impl GatedDeltaNet2 {
    pub fn new(batch_size: usize, hidden_size: usize) -> Self {
        Self {
            w_k: Tensor2D::new(hidden_size, hidden_size),
            w_v: Tensor2D::new(hidden_size, hidden_size),
            w_q: Tensor2D::new(hidden_size, hidden_size),
            w_b: Tensor2D::new(hidden_size, hidden_size),
            w_w: Tensor2D::new(hidden_size, hidden_size),
            w_alpha: Tensor2D::new(hidden_size, hidden_size),
            state: vec![0.0; batch_size * hidden_size * hidden_size],
            hidden_size,
        }
    }

    pub fn randomize(&mut self, min: f32, max: f32) {
        self.w_k.randomize(min, max);
        self.w_v.randomize(min, max);
        self.w_q.randomize(min, max);
        self.w_b.randomize(min, max);
        self.w_w.randomize(min, max);
        self.w_alpha.randomize(min, max);
    }

    pub fn forward_step(&mut self, input: &BatchTensor, negation_token_encountered: bool, out: &mut BatchTensor) {
        let b_size = input.data.nrows();
        let d = self.hidden_size;
        
        let mut k = BatchTensor::new(b_size, d);
        let mut v = BatchTensor::new(b_size, d);
        let mut q = BatchTensor::new(b_size, d);
        let mut b_gate = BatchTensor::new(b_size, d);
        let mut w = BatchTensor::new(b_size, d);
        let mut alpha = BatchTensor::new(b_size, d);
        
        self.w_k.matmul_batch(input, &mut k);
        self.w_v.matmul_batch(input, &mut v);
        self.w_q.matmul_batch(input, &mut q);
        self.w_b.matmul_batch(input, &mut b_gate);
        self.w_w.matmul_batch(input, &mut w);
        self.w_alpha.matmul_batch(input, &mut alpha);
        
        let mut next_state = vec![0.0; b_size * d * d];
        
        for batch_idx in 0..b_size {
            let s_offset = batch_idx * d * d;
            
            let mut alpha_t = vec![0.0; d];
            let mut b_t = vec![0.0; d];
            let mut w_t = vec![0.0; d];
            for i in 0..d {
                alpha_t[i] = 1.0 / (1.0 + (-alpha.data.read(batch_idx, i)).exp());
                let mut b_val = 1.0 / (1.0 + (-b_gate.data.read(batch_idx, i)).exp());
                if negation_token_encountered {
                    let k_val = k.data.read(batch_idx, i);
                    b_val = b_val + (1.0 - b_val) * k_val.abs().tanh();
                }
                b_t[i] = b_val;
                w_t[i] = 1.0 / (1.0 + (-w.data.read(batch_idx, i)).exp());
            }
            
            let mut row_vec = vec![0.0; d];
            for j in 0..d {
                let mut sum = 0.0;
                for m in 0..d {
                    let s_val = self.state[s_offset + m * d + j];
                    sum += b_t[m] * k.data.read(batch_idx, m) * alpha_t[m] * s_val;
                }
                row_vec[j] = sum;
            }
            
            for i in 0..d {
                let k_val = k.data.read(batch_idx, i);
                for j in 0..d {
                    let s_val = self.state[s_offset + i * d + j];
                    let val = alpha_t[i] * s_val - k_val * row_vec[j] + k_val * w_t[j] * v.data.read(batch_idx, j);
                    let c = 5.0f32;
                    let s_ratio = val / c;
                    let s_sat = val / (1.0 + s_ratio * s_ratio).sqrt();
                    next_state[s_offset + i * d + j] = s_sat;
                }
            }
            
            for i in 0..d {
                let mut sum = 0.0;
                for j in 0..d {
                    sum += next_state[s_offset + i * d + j] * q.data.read(batch_idx, j);
                }
                out.data.write(batch_idx, i, sum);
            }
        }
        
        if self.state.len() < b_size * d * d {
            self.state.resize(b_size * d * d, 0.0);
        }
        for (idx, &val) in next_state.iter().enumerate() {
            self.state[idx] = val;
        }
    }

    /// Read-only forward projection through frozen associative state S_{t-1} without state mutation
    pub fn forward_step_readonly(&self, input: &BatchTensor, _negation_token_encountered: bool, out: &mut BatchTensor) {
        let b_size = input.data.nrows();
        let d = self.hidden_size;
        
        let mut q = BatchTensor::new(b_size, d);
        self.w_q.matmul_batch(input, &mut q);
        
        for batch_idx in 0..b_size {
            let s_offset = batch_idx * d * d;
            for i in 0..d {
                let mut sum = 0.0;
                for j in 0..d {
                    sum += self.state[s_offset + i * d + j] * q.data.read(batch_idx, j);
                }
                out.data.write(batch_idx, i, sum);
            }
        }
    }

    /// Commit single associative state update S_t from converged thought vector
    pub fn commit_state_step(&mut self, input: &BatchTensor, negation_token_encountered: bool) {
        let b_size = input.data.nrows();
        let d = self.hidden_size;
        
        let mut k = BatchTensor::new(b_size, d);
        let mut v = BatchTensor::new(b_size, d);
        let mut b_gate = BatchTensor::new(b_size, d);
        let mut w = BatchTensor::new(b_size, d);
        let mut alpha = BatchTensor::new(b_size, d);
        
        self.w_k.matmul_batch(input, &mut k);
        self.w_v.matmul_batch(input, &mut v);
        self.w_b.matmul_batch(input, &mut b_gate);
        self.w_w.matmul_batch(input, &mut w);
        self.w_alpha.matmul_batch(input, &mut alpha);
        
        let mut next_state = vec![0.0; b_size * d * d];
        
        for batch_idx in 0..b_size {
            let s_offset = batch_idx * d * d;
            
            let mut alpha_t = vec![0.0; d];
            let mut b_t = vec![0.0; d];
            let mut w_t = vec![0.0; d];
            for i in 0..d {
                alpha_t[i] = 1.0 / (1.0 + (-alpha.data.read(batch_idx, i)).exp());
                let mut b_val = 1.0 / (1.0 + (-b_gate.data.read(batch_idx, i)).exp());
                if negation_token_encountered {
                    let k_val = k.data.read(batch_idx, i);
                    b_val = b_val + (1.0 - b_val) * k_val.abs().tanh();
                }
                b_t[i] = b_val;
                w_t[i] = 1.0 / (1.0 + (-w.data.read(batch_idx, i)).exp());
            }
            
            let mut row_vec = vec![0.0; d];
            for j in 0..d {
                let mut sum = 0.0;
                for m in 0..d {
                    let s_val = self.state[s_offset + m * d + j];
                    sum += b_t[m] * k.data.read(batch_idx, m) * alpha_t[m] * s_val;
                }
                row_vec[j] = sum;
            }
            
            for i in 0..d {
                let k_val = k.data.read(batch_idx, i);
                for j in 0..d {
                    let s_val = self.state[s_offset + i * d + j];
                    let val = alpha_t[i] * s_val - k_val * row_vec[j] + k_val * w_t[j] * v.data.read(batch_idx, j);
                    let c = 5.0f32;
                    let s_ratio = val / c;
                    let s_sat = val / (1.0 + s_ratio * s_ratio).sqrt();
                    next_state[s_offset + i * d + j] = s_sat;
                }
            }
        }
        
        if self.state.len() < b_size * d * d {
            self.state.resize(b_size * d * d, 0.0);
        }
        for (idx, &val) in next_state.iter().enumerate() {
            self.state[idx] = val;
        }
    }

    pub fn forward_with_cache(&mut self, input: &BatchTensor, negation_token_encountered: bool, cache: &mut GatedDeltaNet2Scratchpad) {
        let b_size = input.data.nrows();
        let d = self.hidden_size;
        
        // Linear projections
        self.w_k.matmul_batch(input, &mut cache.temp_k);
        self.w_v.matmul_batch(input, &mut cache.temp_v);
        self.w_q.matmul_batch(input, &mut cache.temp_q);
        self.w_b.matmul_batch(input, &mut cache.temp_b);
        self.w_w.matmul_batch(input, &mut cache.temp_w);
        self.w_alpha.matmul_batch(input, &mut cache.temp_alpha);
        
        cache.S_prev.copy_from_slice(&self.state);
        
        let mut next_state = vec![0.0; b_size * d * d];
        
        for batch_idx in 0..b_size {
            let s_offset = batch_idx * d * d;
            
            let mut alpha_t = vec![0.0; d];
            let mut b_t = vec![0.0; d];
            let mut w_t = vec![0.0; d];
            let mut k_t = vec![0.0; d];
            let mut v_t = vec![0.0; d];
            let mut q_t = vec![0.0; d];
            
            for i in 0..d {
                k_t[i] = cache.temp_k.data.read(batch_idx, i);
                v_t[i] = cache.temp_v.data.read(batch_idx, i);
                q_t[i] = cache.temp_q.data.read(batch_idx, i);
                
                let b_val = cache.temp_b.data.read(batch_idx, i);
                let w_val = cache.temp_w.data.read(batch_idx, i);
                let a_val = cache.temp_alpha.data.read(batch_idx, i);
                
                let mut b_sig = 1.0 / (1.0 + (-b_val).exp());
                if negation_token_encountered {
                    b_sig = b_sig + (1.0 - b_sig) * k_t[i].abs().tanh();
                }
                let w_sig = 1.0 / (1.0 + (-w_val).exp());
                let a_sig = 1.0 / (1.0 + (-a_val).exp());
                
                cache.k.data.write(batch_idx, i, k_t[i]);
                cache.v.data.write(batch_idx, i, v_t[i]);
                cache.q.data.write(batch_idx, i, q_t[i]);
                cache.b.data.write(batch_idx, i, b_sig);
                cache.w.data.write(batch_idx, i, w_sig);
                cache.alpha.data.write(batch_idx, i, a_sig);
                
                alpha_t[i] = a_sig;
                b_t[i] = b_sig;
                w_t[i] = w_sig;
            }
            
            let mut row_vec = vec![0.0; d];
            for j in 0..d {
                let mut sum = 0.0;
                for m in 0..d {
                    let s_val = cache.S_prev[s_offset + m * d + j];
                    sum += b_t[m] * k_t[m] * alpha_t[m] * s_val;
                }
                row_vec[j] = sum;
            }
            
            for i in 0..d {
                let k_val = k_t[i];
                for j in 0..d {
                    let s_val = cache.S_prev[s_offset + i * d + j];
                    let val = alpha_t[i] * s_val - k_val * row_vec[j] + k_val * w_t[j] * v_t[j];
                    let c = 5.0f32;
                    let s_ratio = val / c;
                    let s_sat = val / (1.0 + s_ratio * s_ratio).sqrt();
                    next_state[s_offset + i * d + j] = s_sat;
                }
            }
            
            for i in 0..d {
                let mut sum = 0.0;
                for j in 0..d {
                    sum += next_state[s_offset + i * d + j] * q_t[j];
                }
                cache.y.data.write(batch_idx, i, sum);
            }
        }
        
        cache.S_curr.copy_from_slice(&next_state);
        self.state = next_state;
    }

    pub fn backward_step(
        &mut self,
        grad_y: &BatchTensor,
        d_S: &mut [f32], // in-out state gradient
        cache: &mut GatedDeltaNet2Scratchpad,
    ) -> BatchTensor {
        let b_size = grad_y.data.nrows();
        let d = self.hidden_size;
        
        cache.grad_x.zero_grad();
        cache.d_k.zero_grad();
        cache.d_v.zero_grad();
        cache.d_q.zero_grad();
        cache.d_b.zero_grad();
        cache.d_w.zero_grad();
        cache.d_alpha.zero_grad();
        
        let mut d_S_prev = vec![0.0; b_size * d * d];
        
        for batch_idx in 0..b_size {
            let s_offset = batch_idx * d * d;
            
            let s_prev = &cache.S_prev[s_offset..(s_offset + d * d)];
            
            let mut alpha_t = vec![0.0; d];
            let mut b_t = vec![0.0; d];
            let mut w_t = vec![0.0; d];
            let mut k_t = vec![0.0; d];
            let mut v_t = vec![0.0; d];
            let mut q_t = vec![0.0; d];
            
            for i in 0..d {
                alpha_t[i] = cache.alpha.data.read(batch_idx, i);
                b_t[i] = cache.b.data.read(batch_idx, i);
                w_t[i] = cache.w.data.read(batch_idx, i);
                k_t[i] = cache.k.data.read(batch_idx, i);
                v_t[i] = cache.v.data.read(batch_idx, i);
                q_t[i] = cache.q.data.read(batch_idx, i);
            }
            
            let mut row_vec = vec![0.0; d];
            for j in 0..d {
                let mut sum = 0.0;
                for m in 0..d {
                    sum += b_t[m] * k_t[m] * alpha_t[m] * s_prev[m * d + j];
                }
                row_vec[j] = sum;
            }
            
            let mut s_t = vec![0.0; d * d];
            for i in 0..d {
                for j in 0..d {
                    s_t[i * d + j] = alpha_t[i] * s_prev[i * d + j] - k_t[i] * row_vec[j] + k_t[i] * w_t[j] * v_t[j];
                }
            }
            
            let mut d_S_total = vec![0.0; d * d];
            for i in 0..d {
                let dy = grad_y.grad.read(batch_idx, i);
                for j in 0..d {
                    d_S_total[i * d + j] = d_S[s_offset + i * d + j] + dy * q_t[j];
                }
            }
            
            for j in 0..d {
                let mut sum = 0.0;
                for i in 0..d {
                    let dy = grad_y.grad.read(batch_idx, i);
                    sum += s_t[i * d + j] * dy;
                }
                cache.d_q.grad.write(batch_idx, j, sum);
            }
            
            let mut G = vec![0.0; d];
            for j in 0..d {
                let mut sum = 0.0;
                for i in 0..d {
                    sum += d_S_total[i * d + j] * k_t[i];
                }
                G[j] = sum;
            }
            
            let mut H = vec![0.0; d];
            for i in 0..d {
                let mut sum = 0.0;
                for j in 0..d {
                    sum += s_prev[i * d + j] * G[j];
                }
                H[i] = sum;
            }
            
            let mut R = vec![0.0; d];
            for j in 0..d {
                let mut sum_m = 0.0;
                for m in 0..d {
                    sum_m += b_t[m] * k_t[m] * alpha_t[m] * s_prev[m * d + j];
                }
                R[j] = w_t[j] * v_t[j] - sum_m;
            }
            
            for j in 0..d {
                cache.d_v.grad.write(batch_idx, j, w_t[j] * G[j]);
            }
            
            for j in 0..d {
                let dw_pre = G[j] * v_t[j] * w_t[j] * (1.0 - w_t[j]);
                cache.d_w.grad.write(batch_idx, j, dw_pre);
            }
            
            for i in 0..d {
                let db_pre = -k_t[i] * alpha_t[i] * H[i] * b_t[i] * (1.0 - b_t[i]);
                cache.d_b.grad.write(batch_idx, i, db_pre);
            }
            
            for i in 0..d {
                let mut sum = 0.0;
                let k_val = k_t[i];
                let b_val = b_t[i];
                for j in 0..d {
                    sum += s_prev[i * d + j] * (d_S_total[i * d + j] - b_val * k_val * G[j]);
                }
                let da_pre = sum * alpha_t[i] * (1.0 - alpha_t[i]);
                cache.d_alpha.grad.write(batch_idx, i, da_pre);
            }
            
            for i in 0..d {
                let mut sum = 0.0;
                for j in 0..d {
                    sum += d_S_total[i * d + j] * R[j];
                }
                let dk = sum - b_t[i] * alpha_t[i] * H[i];
                cache.d_k.grad.write(batch_idx, i, dk);
            }
            
            for i in 0..d {
                let a_val = alpha_t[i];
                let b_val = b_t[i];
                let k_val = k_t[i];
                for j in 0..d {
                    d_S_prev[s_offset + i * d + j] = a_val * (d_S_total[i * d + j] - b_val * k_val * G[j]);
                }
            }
        }
        
        d_S.copy_from_slice(&d_S_prev);
        
        self.w_k.matmul_batch_backward(&mut cache.grad_x, &cache.d_k);
        self.w_v.matmul_batch_backward(&mut cache.grad_x, &cache.d_v);
        self.w_q.matmul_batch_backward(&mut cache.grad_x, &cache.d_q);
        self.w_b.matmul_batch_backward(&mut cache.grad_x, &cache.d_b);
        self.w_w.matmul_batch_backward(&mut cache.grad_x, &cache.d_w);
        self.w_alpha.matmul_batch_backward(&mut cache.grad_x, &cache.d_alpha);
        
        cache.grad_x.clone()
    }
}

impl Parameterized for GatedDeltaNet2 {
    fn params(&self) -> Vec<&[f32]> {
        vec![
            &self.w_k.data, &self.w_v.data, &self.w_q.data,
            &self.w_b.data, &self.w_w.data, &self.w_alpha.data,
        ]
    }

    fn grads(&self) -> Vec<&[f32]> {
        vec![
            &self.w_k.grad, &self.w_v.grad, &self.w_q.grad,
            &self.w_b.grad, &self.w_w.grad, &self.w_alpha.grad,
        ]
    }

    fn params_mut(&mut self) -> Vec<&mut [f32]> {
        vec![
            &mut self.w_k.data, &mut self.w_v.data, &mut self.w_q.data,
            &mut self.w_b.data, &mut self.w_w.data, &mut self.w_alpha.data,
        ]
    }

    fn grads_mut(&mut self) -> Vec<&mut [f32]> {
        vec![
            &mut self.w_k.grad, &mut self.w_v.grad, &mut self.w_q.grad,
            &mut self.w_b.grad, &mut self.w_w.grad, &mut self.w_alpha.grad,
        ]
    }
}
