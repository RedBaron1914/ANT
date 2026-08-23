use cudarc::driver::{CudaStream, CudaSlice, LaunchConfig, DevicePtr, DevicePtrMut, PushKernelArg};
use cudarc::cublas::{CudaBlas, GemmConfig, sys::cublasOperation_t, Gemm};
use std::sync::Arc;

pub struct GpuLoraLinear {
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    pub alpha: f32,
    pub enabled: bool,
    
    // GPU buffers
    pub d_w_base: CudaSlice<f32>,
    pub d_lora_a: CudaSlice<f32>, // (rank, cols)
    pub d_lora_b: CudaSlice<f32>, // (rows, rank)
    
    pub d_w_base_grad: CudaSlice<f32>,
    pub d_lora_a_grad: CudaSlice<f32>,
    pub d_lora_b_grad: CudaSlice<f32>,
    
    // Lion momentum buffers
    pub m_w_base: CudaSlice<f32>,
    pub m_lora_a: CudaSlice<f32>,
    pub m_lora_b: CudaSlice<f32>,
}

impl GpuLoraLinear {
    pub fn new(stream: &Arc<CudaStream>, rows: usize, cols: usize, rank: usize, alpha: f32) -> Self {
        Self {
            rows,
            cols,
            rank,
            alpha,
            enabled: false,
            d_w_base: stream.alloc_zeros::<f32>(rows * cols).unwrap(),
            d_lora_a: stream.alloc_zeros::<f32>(rank * cols).unwrap(),
            d_lora_b: stream.alloc_zeros::<f32>(rows * rank).unwrap(),
            d_w_base_grad: stream.alloc_zeros::<f32>(rows * cols).unwrap(),
            d_lora_a_grad: stream.alloc_zeros::<f32>(rank * cols).unwrap(),
            d_lora_b_grad: stream.alloc_zeros::<f32>(rows * rank).unwrap(),
            m_w_base: stream.alloc_zeros::<f32>(rows * cols).unwrap(),
            m_lora_a: stream.alloc_zeros::<f32>(rank * cols).unwrap(),
            m_lora_b: stream.alloc_zeros::<f32>(rows * rank).unwrap(),
        }
    }

    pub fn forward(
        &self,
        cublas: &CudaBlas,
        d_x: &impl DevicePtr<f32>,
        d_y: &mut impl DevicePtrMut<f32>,
        d_temp_rank: &mut (impl DevicePtrMut<f32> + DevicePtr<f32>), // shape: (batch_size, rank)
        d_temp_out: &mut impl DevicePtrMut<f32>,  // shape: (batch_size, rows)
        batch_size: usize,
    ) {
        // 1. base forward: d_y = d_w_base^T * d_x
        // We use gemm_batch helper logic inline
        let cfg_base = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: self.rows as i32,
            n: batch_size as i32,
            k: self.cols as i32,
            alpha: 1.0,
            lda: self.cols as i32,
            ldb: self.cols as i32,
            beta: 0.0,
            ldc: self.rows as i32,
        };
        unsafe { cublas.gemm(cfg_base, &self.d_w_base, d_x, d_y).unwrap(); }
        
        if self.enabled {
            // 2. temp_rank = d_lora_a^T * d_x
            let cfg_la = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: self.rank as i32,
                n: batch_size as i32,
                k: self.cols as i32,
                alpha: 1.0,
                lda: self.cols as i32,
                ldb: self.cols as i32,
                beta: 0.0,
                ldc: self.rank as i32,
            };
            unsafe { cublas.gemm(cfg_la, &self.d_lora_a, d_x, d_temp_rank).unwrap(); }
            
            // 3. temp_out = d_lora_b^T * temp_rank
            let cfg_lb = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                m: self.rows as i32,
                n: batch_size as i32,
                k: self.rank as i32,
                alpha: 1.0,
                lda: self.rank as i32,
                ldb: self.rank as i32,
                beta: 0.0,
                ldc: self.rows as i32,
            };
            unsafe { cublas.gemm(cfg_lb, &self.d_lora_b, d_temp_rank, d_temp_out).unwrap(); }
            
            // 4. d_y += (alpha / rank) * temp_out
            let scale = self.alpha / (self.rank as f32);
            let cfg_accum = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: self.rows as i32,
                n: batch_size as i32,
                k: self.rank as i32,
                alpha: scale,
                lda: self.rank as i32,
                ldb: self.rank as i32,
                beta: 1.0, // accumulate!
                ldc: self.rows as i32,
            };
            unsafe { cublas.gemm(cfg_accum, &self.d_lora_b, d_temp_rank, d_y).unwrap(); }
        }
    }

    pub fn backward(
        &mut self,
        cublas: &CudaBlas,
        d_x: &impl DevicePtr<f32>,
        d_dy: &impl DevicePtr<f32>, // shape: (batch_size, rows)
        d_dx: &mut impl DevicePtrMut<f32>, // shape: (batch_size, cols)
        d_temp_rank: &impl DevicePtr<f32>, // Z = d_lora_a^T * d_x
        d_d_z: &mut (impl DevicePtrMut<f32> + DevicePtr<f32>), // shape: (batch_size, rank)
        batch_size: usize,
        beta_dx: f32, // beta parameter for input gradient accumulation
    ) {
        // 1. base backward:
        // d_w_base_grad += d_dy * d_x^T
        let cfg_w = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_T,
            m: self.cols as i32,
            n: self.rows as i32,
            k: batch_size as i32,
            alpha: 1.0,
            lda: self.cols as i32,
            ldb: self.rows as i32,
            beta: 1.0,
            ldc: self.cols as i32,
        };
        unsafe { cublas.gemm(cfg_w, d_x, d_dy, &mut self.d_w_base_grad).unwrap(); }
        
        // d_dx += d_w_base * d_dy
        let cfg_x = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: self.cols as i32,
            n: batch_size as i32,
            k: self.rows as i32,
            alpha: 1.0,
            lda: self.cols as i32,
            ldb: self.rows as i32,
            beta: beta_dx,
            ldc: self.cols as i32,
        };
        unsafe { cublas.gemm(cfg_x, &self.d_w_base, d_dy, d_dx).unwrap(); }
        
        if self.enabled {
            let scale = self.alpha / (self.rank as f32);
            
            // 2. d_lora_b_grad += scale * d_dy * temp_rank^T
            let cfg_b = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_T,
                m: self.rank as i32,
                n: self.rows as i32,
                k: batch_size as i32,
                alpha: scale,
                lda: self.rank as i32,
                ldb: self.rows as i32,
                beta: 1.0, // accumulate!
                ldc: self.rank as i32,
            };
            unsafe { cublas.gemm(cfg_b, d_temp_rank, d_dy, &mut self.d_lora_b_grad).unwrap(); }
            
            // 3. d_z = scale * d_lora_b * d_dy
            let cfg_z = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: self.rank as i32,
                n: batch_size as i32,
                k: self.rows as i32,
                alpha: scale,
                lda: self.rank as i32,
                ldb: self.rows as i32,
                beta: 0.0, // replace!
                ldc: self.rank as i32,
            };
            unsafe { cublas.gemm(cfg_z, &self.d_lora_b, d_dy, d_d_z).unwrap(); }
            
            // 4. d_lora_a_grad += d_z * d_x^T
            let cfg_a = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_T,
                m: self.cols as i32,
                n: self.rank as i32,
                k: batch_size as i32,
                alpha: 1.0,
                lda: self.cols as i32,
                ldb: self.rank as i32,
                beta: 1.0, // accumulate!
                ldc: self.cols as i32,
            };
            unsafe { cublas.gemm(cfg_a, d_x, d_d_z, &mut self.d_lora_a_grad).unwrap(); }
            
            // 5. d_dx += d_lora_a * d_z (accumulate on input grads)
            let cfg_dx_lora = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: self.cols as i32,
                n: batch_size as i32,
                k: self.rank as i32,
                alpha: 1.0,
                lda: self.cols as i32,
                ldb: self.rank as i32,
                beta: 1.0, // accumulate!
                ldc: self.cols as i32,
            };
            unsafe { cublas.gemm(cfg_dx_lora, &self.d_lora_a, d_d_z, d_dx).unwrap(); }
        }
    }

    pub fn zero_grad(&mut self, stream: &Arc<CudaStream>, f_zero: &cudarc::driver::CudaFunction) {
        let zero_slice = |slice: &mut CudaSlice<f32>, size: usize| {
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

        if self.enabled {
            zero_slice(&mut self.d_lora_a_grad, self.rank * self.cols);
            zero_slice(&mut self.d_lora_b_grad, self.rows * self.rank);
        } else {
            zero_slice(&mut self.d_w_base_grad, self.rows * self.cols);
        }
    }

    pub fn step(
        &mut self,
        stream: &Arc<CudaStream>,
        f_step: &cudarc::driver::CudaFunction,
        lr: f32,
        beta1: f32,
        beta2: f32,
        wd: f32,
    ) {
        let step_w = |d_w: &mut CudaSlice<f32>, d_g: &mut CudaSlice<f32>, d_m: &mut CudaSlice<f32>, size: usize| {
            let block_size = 256;
            let grid_size = ((size + block_size - 1) / block_size) as u32;
            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let size_i32 = size as i32;
            unsafe {
                stream.launch_builder(f_step)
                    .arg(d_w)
                    .arg(d_g)
                    .arg(d_m)
                    .arg(&lr)
                    .arg(&beta1)
                    .arg(&beta2)
                    .arg(&wd)
                    .arg(&size_i32)
                    .launch(cfg)
            }.unwrap();
        };

        if self.enabled {
            step_w(&mut self.d_lora_a, &mut self.d_lora_a_grad, &mut self.m_lora_a, self.rank * self.cols);
            step_w(&mut self.d_lora_b, &mut self.d_lora_b_grad, &mut self.m_lora_b, self.rows * self.rank);
        } else {
            step_w(&mut self.d_w_base, &mut self.d_w_base_grad, &mut self.m_w_base, self.rows * self.cols);
        }
    }
}
