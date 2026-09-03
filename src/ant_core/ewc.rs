use super::tensor::{Tensor2D, BatchTensor, MatExt};

pub trait Parameterized {
    /// Returns flat list of references to all parameter data slices
    fn params(&self) -> Vec<&[f32]>;
    /// Returns flat list of references to all gradient data slices
    fn grads(&self) -> Vec<&[f32]>;
    /// Returns flat list of mutable references to all parameter data slices
    fn params_mut(&mut self) -> Vec<&mut [f32]>;
    /// Returns flat list of mutable references to all gradient data slices
    fn grads_mut(&mut self) -> Vec<&mut [f32]>;
}

pub struct EwcSnapshot {
    pub star_params: Vec<f32>,
    pub fisher: Vec<f32>,
    pub lambda: f32,
}

impl EwcSnapshot {
    /// Option A: consolidate from current gradients.
    pub fn consolidate_from_current_grads(model: &impl Parameterized, lambda: f32) -> Self {
        let mut star_params = Vec::new();
        let mut fisher = Vec::new();

        let params = model.params();
        let grads = model.grads();

        for p_slice in params {
            star_params.extend_from_slice(p_slice);
        }

        for g_slice in grads {
            for &g in g_slice {
                fisher.push(g * g);
            }
        }

        assert_eq!(star_params.len(), fisher.len(), "Params and grads size mismatch");

        Self {
            star_params,
            fisher,
            lambda,
        }
    }

    /// Option B: consolidate from a dataset.
    pub fn consolidate_from_dataset<P, F>(
        model: &mut P,
        lambda: f32,
        mut compute_grads_fn: F,
    ) -> Self
    where
        P: Parameterized,
        F: FnMut(&mut P) -> bool,
    {
        let mut star_params = Vec::new();
        let mut fisher = Vec::new();

        for p_slice in model.params() {
            star_params.extend_from_slice(p_slice);
            for _ in 0..p_slice.len() {
                fisher.push(0.0);
            }
        }

        let mut num_samples = 0;
        
        while compute_grads_fn(model) {
            let mut idx = 0;
            for g_slice in model.grads() {
                for &g in g_slice {
                    fisher[idx] += g * g;
                    idx += 1;
                }
            }
            num_samples += 1;
        }

        if num_samples > 0 {
            for f in fisher.iter_mut() {
                *f /= num_samples as f32;
            }
        }

        Self {
            star_params,
            fisher,
            lambda,
        }
    }

    pub fn apply_decay(&mut self, gamma: f32) {
        for f in self.fisher.iter_mut() {
            *f *= gamma;
        }
    }

    /// Calculates the current EWC penalty: lambda / 2 * sum( F_i * (theta_i - theta^*_i)^2 )
    pub fn penalty(&self, model: &impl Parameterized) -> f32 {
        let mut penalty = 0.0;
        let mut idx = 0;

        for p_slice in model.params() {
            for &p in p_slice {
                let diff = p - self.star_params[idx];
                penalty += self.fisher[idx] * diff * diff;
                idx += 1;
            }
        }

        self.lambda * 0.5 * penalty
    }

    /// Adds the EWC gradient penalty to the model's current gradients.
    pub fn apply_penalty_grads(&self, model: &mut impl Parameterized) {
        let mut idx = 0;

        let params_flat: Vec<f32> = model.params().into_iter().flat_map(|s| s.iter().copied()).collect();
        
        for g_slice in model.grads_mut() {
            for g in g_slice.iter_mut() {
                if idx < params_flat.len() {
                    let p = params_flat[idx];
                    let diff = p - self.star_params[idx];
                    let penalty_grad = self.lambda * self.fisher[idx] * diff;
                    *g += penalty_grad;
                }
                idx += 1;
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct LoraScratchpad {
    pub z: BatchTensor,                   // (batch_size, rank)
    pub scaled_grad_output: BatchTensor,  // (batch_size, rows)
    pub d_z: BatchTensor,                 // (batch_size, rank)
}

impl LoraScratchpad {
    pub fn new(batch_size: usize, rows: usize, rank: usize) -> Self {
        Self {
            z: BatchTensor::new(batch_size, rank),
            scaled_grad_output: BatchTensor::new(batch_size, rows),
            d_z: BatchTensor::new(batch_size, rank),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LoraLinear {
    pub base: Tensor2D,
    pub lora_a: Tensor2D, // (rank, cols)
    pub lora_b: Tensor2D, // (rows, rank)
    pub rank: usize,
    pub alpha: f32,
    pub enabled: bool,
}

impl LoraLinear {
    pub fn new(rows: usize, cols: usize, rank: usize, alpha: f32) -> Self {
        let mut lora_a = Tensor2D::new(rank, cols);
        let mut lora_b = Tensor2D::new(rows, rank);
        
        lora_a.randomize(-0.1, 0.1);
        // lora_b is initialized with zeros so that LoRA is initially a no-op
        for val in lora_b.data.iter_mut() {
            *val = 0.0;
        }
        
        Self {
            base: Tensor2D::new(rows, cols),
            lora_a,
            lora_b,
            rank,
            alpha,
            enabled: false,
        }
    }
    
    pub fn zero_grad(&mut self) {
        self.base.zero_grad();
        self.lora_a.zero_grad();
        self.lora_b.zero_grad();
    }
    
    pub fn randomize(&mut self, min: f32, max: f32) {
        self.base.randomize(min, max);
    }

    /// Merges adapter low-rank deltas (alpha/r * B * A) into base weights and resets low-rank factors
    pub fn merge_into_base(&mut self) {
        if self.rank == 0 { return; }
        let scaling = self.alpha / self.rank as f32;
        let rows = self.base.rows;
        let cols = self.base.cols;

        for i in 0..rows {
            for j in 0..cols {
                let mut delta = 0.0f32;
                for r in 0..self.rank {
                    delta += self.lora_b.data[i * self.lora_b.cols + r] * self.lora_a.data[r * self.lora_a.cols + j];
                }
                self.base.data[i * cols + j] += scaling * delta;
            }
        }

        self.lora_a.randomize(-0.1, 0.1);
        self.lora_b.data.fill(0.0);
    }
    
    pub fn forward(&self, input: &BatchTensor, out: &mut BatchTensor, scratch: &mut LoraScratchpad) {
        self.base.matmul_batch(input, out);
        
        if self.enabled {
            let b_size = input.data.nrows();
            scratch.z.data.fill(0.0);
            self.lora_a.matmul_batch(input, &mut scratch.z);
            
            let mut temp_rows = BatchTensor::new(b_size, self.base.rows);
            self.lora_b.matmul_batch(&scratch.z, &mut temp_rows);
            
            let scale = self.alpha / (self.rank as f32);
            for b in 0..b_size {
                for i in 0..self.base.rows {
                    let prev = out.data.read(b, i);
                    out.data.write(b, i, prev + temp_rows.data.read(b, i) * scale);
                }
            }
        }
    }

    pub fn backward(&mut self, input: &mut BatchTensor, grad_output: &BatchTensor, scratch: &mut LoraScratchpad) {
        let b_size = input.data.nrows();
        
        self.base.matmul_batch_backward(input, grad_output);
        
        if self.enabled {
            let scale = self.alpha / (self.rank as f32);
            
            // Forward temp_rank (Z = X * A^T)
            scratch.z.data.fill(0.0);
            self.lora_a.matmul_batch(input, &mut scratch.z);
            
            // Scaled grad_output
            scratch.scaled_grad_output.zero_grad();
            for b in 0..b_size {
                for i in 0..self.base.rows {
                    scratch.scaled_grad_output.grad.write(b, i, grad_output.grad.read(b, i) * scale);
                }
            }
            
            // d_Z = scaled_grad_output * B
            scratch.d_z.zero_grad();
            self.lora_b.matmul_batch_backward(&mut scratch.z, &scratch.scaled_grad_output);
            
            // d_input_lora += d_Z * A
            scratch.d_z.data.copy_from(&scratch.z.grad);
            scratch.d_z.zero_grad();
            
            self.lora_a.matmul_batch_backward(input, &scratch.d_z);
        }
    }
}

impl Parameterized for LoraLinear {
    fn params(&self) -> Vec<&[f32]> {
        vec![&self.base.data, &self.lora_a.data, &self.lora_b.data]
    }
    fn params_mut(&mut self) -> Vec<&mut [f32]> {
        vec![&mut self.base.data, &mut self.lora_a.data, &mut self.lora_b.data]
    }
    fn grads(&self) -> Vec<&[f32]> {
        vec![&self.base.grad, &self.lora_a.grad, &self.lora_b.grad]
    }
    fn grads_mut(&mut self) -> Vec<&mut [f32]> {
        vec![&mut self.base.grad, &mut self.lora_a.grad, &mut self.lora_b.grad]
    }
}
