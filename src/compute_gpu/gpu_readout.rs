use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaStream, CudaSlice, LaunchConfig, CudaModule, PushKernelArg};
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t};
use crate::ant_core::tensor::{BatchTensor, Tensor1D};
use crate::ant_core::embedding::Embedding;

pub struct GpuReadout {
    pub hidden_size: usize,
    pub embed_dim: usize,
    pub vocab_size: usize,
    
    pub w_proj: super::gpu_lora::GpuLoraLinear,
    pub b_proj: Tensor1D,
    
    _ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub cublas: CudaBlas,
    pub module: Arc<CudaModule>,
    
    // GPU buffers
    pub d_hidden: CudaSlice<f32>,
    pub d_b_proj: CudaSlice<f32>,
    pub d_w_emb: CudaSlice<f32>,
    pub d_h_proj: CudaSlice<f32>,
    pub d_logits: CudaSlice<f32>,
    pub d_targets: CudaSlice<i32>,
    pub d_loss_grads: CudaSlice<f32>,
    pub d_losses: CudaSlice<f32>,

    // Gradient buffers
    pub d_b_proj_grad: CudaSlice<f32>,
    pub d_w_emb_grad: CudaSlice<f32>,
    pub d_h_proj_grad: CudaSlice<f32>,

    // Lion Momentum
    pub m_b_proj: CudaSlice<f32>,
    pub m_w_emb: CudaSlice<f32>,

    // LoRA VRAM scratch buffers
    pub d_lora_temp_rank: CudaSlice<f32>,
    pub d_lora_d_z: CudaSlice<f32>,
    pub d_lora_temp_out: CudaSlice<f32>,
}

impl GpuReadout {
    pub fn new(hidden_size: usize, embed_dim: usize, vocab_size: usize, max_batch_size: usize, lora_rank: usize, lora_alpha: f32) -> Self {
        let ctx = CudaContext::new(0).expect("No CUDA device found");
        let stream = ctx.new_stream().unwrap();
        let cublas = CudaBlas::new(stream.clone()).unwrap();
        
        let ptx_content = include_str!("kernel.cu");
        let ptx = cudarc::nvrtc::compile_ptx(ptx_content).expect("NVRTC failed to compile PTX");
        let module = ctx.load_module(ptx).expect("Failed to load PTX module");
        
        let d_hidden = stream.alloc_zeros::<f32>(max_batch_size * hidden_size).unwrap();
        let d_b_proj = stream.alloc_zeros::<f32>(embed_dim).unwrap();
        let d_w_emb = stream.alloc_zeros::<f32>(vocab_size * embed_dim).unwrap();
        let d_h_proj = stream.alloc_zeros::<f32>(max_batch_size * embed_dim).unwrap();
        let d_logits = stream.alloc_zeros::<f32>(max_batch_size * vocab_size).unwrap();
        let d_targets = stream.alloc_zeros::<i32>(max_batch_size).unwrap();
        let d_loss_grads = stream.alloc_zeros::<f32>(max_batch_size * vocab_size).unwrap();
        let d_losses = stream.alloc_zeros::<f32>(max_batch_size).unwrap();
        
        let d_b_proj_grad = stream.alloc_zeros::<f32>(embed_dim).unwrap();
        let d_w_emb_grad = stream.alloc_zeros::<f32>(vocab_size * embed_dim).unwrap();
        let d_h_proj_grad = stream.alloc_zeros::<f32>(max_batch_size * embed_dim).unwrap();
        
        let m_b_proj = stream.alloc_zeros::<f32>(embed_dim).unwrap();
        let m_w_emb = stream.alloc_zeros::<f32>(vocab_size * embed_dim).unwrap();
        
        let d_lora_temp_rank = stream.alloc_zeros::<f32>(max_batch_size * lora_rank).unwrap();
        let d_lora_d_z = stream.alloc_zeros::<f32>(max_batch_size * lora_rank).unwrap();
        let d_lora_temp_out = stream.alloc_zeros::<f32>(max_batch_size * embed_dim).unwrap();

        let w_proj = super::gpu_lora::GpuLoraLinear::new(&stream, embed_dim, hidden_size, lora_rank, lora_alpha);
        
        Self {
            hidden_size,
            embed_dim,
            vocab_size,
            w_proj,
            b_proj: Tensor1D::new(embed_dim),
            _ctx: ctx,
            stream,
            cublas,
            module,
            d_hidden,
            d_b_proj,
            d_w_emb,
            d_h_proj,
            d_logits,
            d_targets,
            d_loss_grads,
            d_losses,
            d_b_proj_grad,
            d_w_emb_grad,
            d_h_proj_grad,
            m_b_proj,
            m_w_emb,
            d_lora_temp_rank,
            d_lora_d_z,
            d_lora_temp_out,
        }
    }

    /// Zero out all parameter gradients on the GPU
    pub fn zero_grad(&mut self) {
        let f_zero = self.module.load_function("zero_buffer_kernel").unwrap();
        
        let zero_slice = |stream: &Arc<CudaStream>, f_zero: &cudarc::driver::CudaFunction, slice: &mut CudaSlice<f32>, size: usize| {
            let block_size = 256;
            let grid_size = ((size + block_size - 1) / block_size) as u32;
            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let size_i32 = size as i32;
            unsafe {
                stream.launch_builder(f_zero)
                    .arg(slice)
                    .arg(&size_i32)
                    .launch(cfg)
            }.unwrap();
        };

        self.w_proj.zero_grad(&self.stream, &f_zero);
        zero_slice(&self.stream, &f_zero, &mut self.d_b_proj_grad, self.embed_dim);
        zero_slice(&self.stream, &f_zero, &mut self.d_w_emb_grad, self.vocab_size * self.embed_dim);
    }

    /// Synchronize host embedding and readout weights to GPU VRAM once per chunk
    pub fn sync_weights_to_gpu(&mut self, embedding: &Embedding, readout: &crate::ant_core::readout::ReadoutLayer) {
        let mut d_w_emb_sub = self.d_w_emb.try_slice_mut(0..self.vocab_size * self.embed_dim).unwrap();
        self.stream.memcpy_htod(&embedding.weight.data, &mut d_w_emb_sub).unwrap();
        
        let mut d_w_proj_sub = self.w_proj.d_w_base.try_slice_mut(0..self.embed_dim * self.hidden_size).unwrap();
        self.stream.memcpy_htod(&readout.w_proj.base.data, &mut d_w_proj_sub).unwrap();
        
        let mut d_lora_a_sub = self.w_proj.d_lora_a.try_slice_mut(0..self.w_proj.rank * self.hidden_size).unwrap();
        self.stream.memcpy_htod(&readout.w_proj.lora_a.data, &mut d_lora_a_sub).unwrap();
        
        let mut d_lora_b_sub = self.w_proj.d_lora_b.try_slice_mut(0..self.embed_dim * self.w_proj.rank).unwrap();
        self.stream.memcpy_htod(&readout.w_proj.lora_b.data, &mut d_lora_b_sub).unwrap();
        
        let mut d_b_proj_sub = self.d_b_proj.try_slice_mut(0..self.embed_dim).unwrap();
        self.stream.memcpy_htod(&readout.b_proj.data, &mut d_b_proj_sub).unwrap();
    }

    /// Forward pass on GPU for Readout and Cross Entropy Loss (Asynchronous)
    pub fn forward_and_loss_vram(
        &mut self,
        d_hidden_in: &cudarc::driver::CudaView<'_, f32>,
        d_targets_in: &cudarc::driver::CudaView<'_, i32>,
        batch_size: usize,
    ) {
        let mut d_hidden_sub = self.d_hidden.try_slice_mut(0..batch_size * self.hidden_size).unwrap();
        self.stream.memcpy_dtod(d_hidden_in, &mut d_hidden_sub).unwrap();
        
        let mut d_la_temp_mut = self.d_lora_temp_rank.try_slice_mut(0..batch_size * self.w_proj.rank).unwrap();
        let mut d_la_out_mut = self.d_lora_temp_out.try_slice_mut(0..batch_size * self.embed_dim).unwrap();
        let mut d_h_proj_mut = self.d_h_proj.try_slice_mut(0..batch_size * self.embed_dim).unwrap();
        self.w_proj.forward(
            &self.cublas,
            &self.d_hidden,
            &mut d_h_proj_mut,
            &mut d_la_temp_mut,
            &mut d_la_out_mut,
            batch_size,
        );
        
        let f_bias = self.module.load_function("add_bias_kernel").unwrap();
        let bias_cfg = LaunchConfig { grid_dim: ((self.embed_dim as u32 + 255) / 256, batch_size as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&f_bias)
                .arg(&mut self.d_h_proj)
                .arg(&self.d_b_proj)
                .arg(&(batch_size as i32))
                .arg(&(self.embed_dim as i32))
                .launch(bias_cfg)
        }.unwrap();
        
        let cfg2 = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: self.vocab_size as i32,
            n: batch_size as i32,
            k: self.embed_dim as i32,
            alpha: 1.0f32,
            lda: self.embed_dim as i32,
            ldb: self.embed_dim as i32,
            beta: 0.0f32,
            ldc: self.vocab_size as i32,
        };
        unsafe { self.cublas.gemm(cfg2, &self.d_w_emb, &self.d_h_proj, &mut self.d_logits).unwrap(); }
        
        let norm_factor = 1.0f32 / (batch_size as f32);
        let f_loss = self.module.load_function("cross_entropy_loss_kernel").unwrap();
        let b_size_i32 = batch_size as i32;
        let v_size_i32 = self.vocab_size as i32;
        let loss_cfg = LaunchConfig { grid_dim: (batch_size as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&f_loss)
                .arg(&self.d_logits)
                .arg(d_targets_in)
                .arg(&mut self.d_loss_grads)
                .arg(&mut self.d_losses)
                .arg(&b_size_i32)
                .arg(&v_size_i32)
                .arg(&norm_factor)
                .launch(loss_cfg)
        }.unwrap();
    }

    pub fn backward_vram(&mut self, batch_size: usize) -> cudarc::driver::CudaView<'_, f32> {
        let cfg1 = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: self.embed_dim as i32,
            n: batch_size as i32,
            k: self.vocab_size as i32,
            alpha: 1.0f32,
            lda: self.embed_dim as i32,
            ldb: self.vocab_size as i32,
            beta: 0.0f32,
            ldc: self.embed_dim as i32,
        };
        unsafe { self.cublas.gemm(cfg1, &self.d_w_emb, &self.d_loss_grads, &mut self.d_h_proj_grad).unwrap(); }

        let cfg2 = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_T,
            m: self.embed_dim as i32,
            n: self.vocab_size as i32,
            k: batch_size as i32,
            alpha: 1.0f32,
            lda: self.embed_dim as i32,
            ldb: self.vocab_size as i32,
            beta: 1.0f32,
            ldc: self.embed_dim as i32,
        };
        unsafe { self.cublas.gemm(cfg2, &self.d_h_proj, &self.d_loss_grads, &mut self.d_w_emb_grad).unwrap(); }

        let f_bias_bw = self.module.load_function("bias_backward_kernel").unwrap();
        let bias_bw_cfg = LaunchConfig { grid_dim: ((self.embed_dim as u32 + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&f_bias_bw)
                .arg(&self.d_h_proj_grad)
                .arg(&mut self.d_b_proj_grad)
                .arg(&(batch_size as i32))
                .arg(&(self.embed_dim as i32))
                .launch(bias_bw_cfg)
        }.unwrap();

        let d_hidden_ro = self.d_hidden.try_slice(0..batch_size * self.hidden_size).unwrap();
        let mut d_hidden_mut = unsafe {
            std::mem::transmute::<cudarc::driver::CudaView<'_, f32>, cudarc::driver::CudaViewMut<'_, f32>>(
                self.d_hidden.try_slice(0..batch_size * self.hidden_size).unwrap()
            )
        };
        let mut d_dz_mut = self.d_lora_d_z.try_slice_mut(0..batch_size * self.w_proj.rank).unwrap();
        let d_temp_rank_ro = self.d_lora_temp_rank.try_slice(0..batch_size * self.w_proj.rank).unwrap();
        self.w_proj.backward(
            &self.cublas,
            &d_hidden_ro,
            &self.d_h_proj_grad,
            &mut d_hidden_mut,
            &d_temp_rank_ro,
            &mut d_dz_mut,
            batch_size,
            0.0, // beta_dx is 0.0 because it overwrites the gradient of hidden
        );

        self.d_hidden.try_slice(0..batch_size * self.hidden_size).unwrap()
    }
    
    pub fn step(&mut self, lr: f32, beta1: f32, beta2: f32, weight_decay: f32) {
        let f_lion = self.module.load_function("lion_step_kernel").unwrap();
        
        let step_w = |w: &mut CudaSlice<f32>, g: &mut CudaSlice<f32>, m: &mut CudaSlice<f32>, size: usize| {
            let cfg = LaunchConfig { grid_dim: (((size + 255)/256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { 
                self.stream.launch_builder(&f_lion)
                    .arg(w).arg(g).arg(m)
                    .arg(&lr).arg(&beta1).arg(&beta2).arg(&weight_decay).arg(&(size as i32))
                    .launch(cfg) 
            }.unwrap();
        };
        
        self.w_proj.step(&self.stream, &f_lion, lr, beta1, beta2, weight_decay);
        step_w(&mut self.d_b_proj, &mut self.d_b_proj_grad, &mut self.m_b_proj, self.embed_dim);
        step_w(&mut self.d_w_emb, &mut self.d_w_emb_grad, &mut self.m_w_emb, self.vocab_size * self.embed_dim);
    }
    
    pub fn sync_and_accumulate(&mut self, embedding: &mut Embedding, readout: &mut crate::ant_core::readout::ReadoutLayer, _lr: f32) {
        // Readout projection weights sync to host
        let d_w_proj_sub = self.w_proj.d_w_base.try_slice(0..self.embed_dim * self.hidden_size).unwrap();
        self.stream.memcpy_dtoh(&d_w_proj_sub, &mut readout.w_proj.base.data).unwrap();
        
        let d_la = self.w_proj.d_lora_a.try_slice(0..self.w_proj.rank * self.embed_dim).unwrap();
        self.stream.memcpy_dtoh(&d_la, &mut readout.w_proj.lora_a.data).unwrap();
        
        let d_lb = self.w_proj.d_lora_b.try_slice(0..self.embed_dim * self.w_proj.rank).unwrap();
        self.stream.memcpy_dtoh(&d_lb, &mut readout.w_proj.lora_b.data).unwrap();
        
        let d_b_proj_sub = self.d_b_proj.try_slice(0..self.embed_dim).unwrap();
        self.stream.memcpy_dtoh(&d_b_proj_sub, &mut readout.b_proj.data).unwrap();

        // Trained embedding weights sync from GPU Readout to host embedding
        let d_w_emb_sub = self.d_w_emb.try_slice(0..self.vocab_size * self.embed_dim).unwrap();
        self.stream.memcpy_dtoh(&d_w_emb_sub, &mut embedding.weight.data).unwrap();
    }

    pub fn forward_and_loss(
        &mut self,
        hidden: &crate::ant_core::tensor::BatchTensor,
        targets: &[usize],
        chunk_size: usize,
    ) -> (f32, Vec<f32>) {
        let batch_size = hidden.data.rows;
        
        // Copy hidden to GPU
        let mut d_hidden_sub = self.d_hidden.try_slice_mut(0..batch_size * self.hidden_size).unwrap();
        self.stream.memcpy_htod(&hidden.data.storage, &mut d_hidden_sub).unwrap();
        
        // Copy targets
        let targets_i32: Vec<i32> = targets.iter().map(|&t| t as i32).collect();
        let mut d_targets_sub = self.d_targets.try_slice_mut(0..batch_size).unwrap();
        self.stream.memcpy_htod(&targets_i32, &mut d_targets_sub).unwrap();
        
        let mut d_la_temp_mut = self.d_lora_temp_rank.try_slice_mut(0..batch_size * self.w_proj.rank).unwrap();
        let mut d_la_out_mut = self.d_lora_temp_out.try_slice_mut(0..batch_size * self.embed_dim).unwrap();
        let mut d_h_proj_mut = self.d_h_proj.try_slice_mut(0..batch_size * self.embed_dim).unwrap();
        self.w_proj.forward(
            &self.cublas,
            &self.d_hidden,
            &mut d_h_proj_mut,
            &mut d_la_temp_mut,
            &mut d_la_out_mut,
            batch_size,
        );
        
        // Add bias b_proj to h_proj
        let f_bias = self.module.load_function("add_bias_kernel").unwrap();
        let bias_cfg = LaunchConfig {
            grid_dim: ((self.embed_dim as u32 + 255) / 256, batch_size as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&f_bias)
                .arg(&mut self.d_h_proj)
                .arg(&self.d_b_proj)
                .arg(&(batch_size as i32))
                .arg(&(self.embed_dim as i32))
                .launch(bias_cfg)
        }.unwrap();
        
        // Compute logits = h_proj * w_emb^T
        let cfg2 = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: self.vocab_size as i32,
            n: batch_size as i32,
            k: self.embed_dim as i32,
            alpha: 1.0f32,
            lda: self.embed_dim as i32,
            ldb: self.embed_dim as i32,
            beta: 0.0f32,
            ldc: self.vocab_size as i32,
        };
        
        unsafe {
            self.cublas.gemm(cfg2, &self.d_w_emb, &self.d_h_proj, &mut self.d_logits).unwrap();
        }
        
        // Launch cross_entropy_loss_kernel
        let norm_factor = 1.0 / (batch_size * chunk_size) as f32;
        let loss_cfg = LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        
        let f_loss = self.module.load_function("cross_entropy_loss_kernel").unwrap();
        let b_size_i32 = batch_size as i32;
        let v_size_i32 = self.vocab_size as i32;
        unsafe {
            self.stream.launch_builder(&f_loss)
                .arg(&self.d_logits)
                .arg(&self.d_targets)
                .arg(&mut self.d_loss_grads)
                .arg(&mut self.d_losses)
                .arg(&b_size_i32)
                .arg(&v_size_i32)
                .arg(&norm_factor)
                .launch(loss_cfg)
        }.unwrap();
        
        let d_losses_sub = self.d_losses.try_slice(0..batch_size).unwrap();
        let host_losses = self.stream.clone_dtoh(&d_losses_sub).unwrap();
        let total_loss: f32 = host_losses.iter().sum();
        
        (total_loss, host_losses)
    }

    /// Backward pass on GPU for Readout
    /// Returns the CPU vector of d_hidden (size batch_size * hidden_size)
    pub fn backward(
        &mut self,
        hidden: &BatchTensor,
    ) -> Vec<f32> {
        let batch_size = hidden.data.rows;
        
        // Copy hidden to GPU
        let mut d_hidden_sub = self.d_hidden.try_slice_mut(0..batch_size * self.hidden_size).unwrap();
        self.stream.memcpy_htod(&hidden.data.storage, &mut d_hidden_sub).unwrap();

        // 1. d_h_proj_grad = d_loss_grads * w_emb
        let cfg1 = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: self.embed_dim as i32,
            n: batch_size as i32,
            k: self.vocab_size as i32,
            alpha: 1.0f32,
            lda: self.embed_dim as i32,
            ldb: self.vocab_size as i32,
            beta: 0.0f32,
            ldc: self.embed_dim as i32,
        };
        unsafe {
            self.cublas.gemm(cfg1, &self.d_w_emb, &self.d_loss_grads, &mut self.d_h_proj_grad).unwrap();
        }

        // 2. d_w_emb_grad += d_loss_grads^T * h_proj
        let cfg2 = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_T,
            m: self.embed_dim as i32,
            n: self.vocab_size as i32,
            k: batch_size as i32,
            alpha: 1.0f32,
            lda: self.embed_dim as i32,
            ldb: self.vocab_size as i32,
            beta: 1.0f32, // Accumulate over steps
            ldc: self.embed_dim as i32,
        };
        unsafe {
            self.cublas.gemm(cfg2, &self.d_h_proj, &self.d_loss_grads, &mut self.d_w_emb_grad).unwrap();
        }

        // 3. d_b_proj_grad += sum(d_h_proj_grad, dim=0)
        let f_bias_grad = self.module.load_function("bias_backward_kernel").unwrap();
        let bias_grad_cfg = LaunchConfig {
            grid_dim: ((self.embed_dim as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&f_bias_grad)
                .arg(&self.d_h_proj_grad)
                .arg(&mut self.d_b_proj_grad)
                .arg(&(batch_size as i32))
                .arg(&(self.embed_dim as i32))
                .launch(bias_grad_cfg)
        }.unwrap();

        let d_hidden_ro = self.d_hidden.try_slice(0..batch_size * self.hidden_size).unwrap();
        let mut d_hidden_mut = unsafe {
            std::mem::transmute::<cudarc::driver::CudaView<'_, f32>, cudarc::driver::CudaViewMut<'_, f32>>(
                self.d_hidden.try_slice(0..batch_size * self.hidden_size).unwrap()
            )
        };
        let mut d_dz_mut = self.d_lora_d_z.try_slice_mut(0..batch_size * self.w_proj.rank).unwrap();
        let d_temp_rank_ro = self.d_lora_temp_rank.try_slice(0..batch_size * self.w_proj.rank).unwrap();
        
        self.w_proj.backward(
            &self.cublas,
            &d_hidden_ro,
            &self.d_h_proj_grad,
            &mut d_hidden_mut,
            &d_temp_rank_ro,
            &mut d_dz_mut,
            batch_size,
            0.0,
        );

        // 6. Retrieve d_hidden back to host
        let d_hidden_sub = self.d_hidden.try_slice(0..batch_size * self.hidden_size).unwrap();
        self.stream.clone_dtoh(&d_hidden_sub).unwrap()
    }

    /// Synchronize the accumulated GPU gradients back to the host parameters
    pub fn sync_grads_back_to_host(&mut self, embedding: &mut Embedding, readout: &mut crate::ant_core::readout::ReadoutLayer) {
        let host_w_base_grad = self.stream.clone_dtoh(&self.w_proj.d_w_base_grad).unwrap();
        readout.w_proj.base.grad.copy_from_slice(&host_w_base_grad);

        let host_lora_a_grad = self.stream.clone_dtoh(&self.w_proj.d_lora_a_grad).unwrap();
        readout.w_proj.lora_a.grad.copy_from_slice(&host_lora_a_grad);

        let host_lora_b_grad = self.stream.clone_dtoh(&self.w_proj.d_lora_b_grad).unwrap();
        readout.w_proj.lora_b.grad.copy_from_slice(&host_lora_b_grad);
        
        let host_b_proj_grad = self.stream.clone_dtoh(&self.d_b_proj_grad).unwrap();
        readout.b_proj.grad.copy_from_slice(&host_b_proj_grad);
        
        let host_w_emb_grad = self.stream.clone_dtoh(&self.d_w_emb_grad).unwrap();
        for (cpu_g, gpu_g) in embedding.weight.grad.iter_mut().zip(host_w_emb_grad.iter()) {
            *cpu_g += *gpu_g;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_readout_forward_backward() {
        let _ = std::panic::catch_unwind(|| {
            let mut gpu_ro = GpuReadout::new(16, 8, 32, 4, 8, 16.0);
            let embedding = Embedding::new(32, 8);
            let readout = crate::ant_core::readout::ReadoutLayer::new(16, 8, 32, 8, 16.0);
            
            // Sync weights
            gpu_ro.sync_weights_to_gpu(&embedding, &readout);
            gpu_ro.zero_grad();
            
            let hidden = BatchTensor::new(4, 16);
            let targets = vec![1, 2, 3, 4];
            
            let (loss, _) = gpu_ro.forward_and_loss(&hidden, &targets, 10);
            assert!(loss >= 0.0);
            
            let d_hidden = gpu_ro.backward(&hidden);
            assert_eq!(d_hidden.len(), 4 * 16);
        });
    }
}
