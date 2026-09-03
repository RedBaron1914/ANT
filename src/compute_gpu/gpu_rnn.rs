#![allow(non_snake_case)]
use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaStream, CudaSlice, CudaViewMut, LaunchConfig, CudaModule, PushKernelArg, DevicePtr};
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t};
use crate::ant_core::pipeline::AntPipeline;

pub struct GpuMinGRU {
    pub hidden_size: usize,
    pub input_size: usize,
    pub d_w_z: CudaSlice<f32>, pub d_b_z: CudaSlice<f32>,
    pub d_w_h: CudaSlice<f32>, pub d_b_h: CudaSlice<f32>,
    pub d_w_z_grad: CudaSlice<f32>, pub d_b_z_grad: CudaSlice<f32>,
    pub d_w_h_grad: CudaSlice<f32>, pub d_b_h_grad: CudaSlice<f32>,
    
    // Lion Momentum
    pub m_w_z: CudaSlice<f32>, pub m_b_z: CudaSlice<f32>,
    pub m_w_h: CudaSlice<f32>, pub m_b_h: CudaSlice<f32>,
}

pub struct GpuDeltaNet2 {
    pub hidden_size: usize,
    pub d_w_k: CudaSlice<f32>, pub d_w_v: CudaSlice<f32>, pub d_w_q: CudaSlice<f32>,
    pub d_w_b: CudaSlice<f32>, pub d_w_w: CudaSlice<f32>, pub d_w_alpha: CudaSlice<f32>,
    
    pub d_w_k_grad: CudaSlice<f32>, pub d_w_v_grad: CudaSlice<f32>, pub d_w_q_grad: CudaSlice<f32>,
    pub d_w_b_grad: CudaSlice<f32>, pub d_w_w_grad: CudaSlice<f32>, pub d_w_alpha_grad: CudaSlice<f32>,
    
    // Lion Momentum
    pub m_w_k: CudaSlice<f32>, pub m_w_v: CudaSlice<f32>, pub m_w_q: CudaSlice<f32>,
    pub m_w_b: CudaSlice<f32>, pub m_w_w: CudaSlice<f32>, pub m_w_alpha: CudaSlice<f32>,
}

pub struct GpuRMSNorm {
    pub dim: usize,
    pub d_weight: CudaSlice<f32>,
    pub d_weight_grad: CudaSlice<f32>,
    
    // Lion Momentum
    pub m_weight: CudaSlice<f32>,
}

pub struct GpuSparseGating {
    pub hidden_size: usize,
    pub input_size: usize,
    pub d_w1: CudaSlice<f32>, pub d_b1: CudaSlice<f32>,
    pub d_w1_grad: CudaSlice<f32>, pub d_b1_grad: CudaSlice<f32>,
    
    // Lion Momentum
    pub m_w1: CudaSlice<f32>, pub m_b1: CudaSlice<f32>,
}

pub struct GpuEmbedding {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub d_weight: CudaSlice<f32>,
    pub d_weight_grad: CudaSlice<f32>,
    
    // Lion Momentum
    pub m_weight: CudaSlice<f32>,
}

pub struct GpuHistory {
    pub d_x: CudaSlice<f32>,
    
    // minGRU history
    pub d_mingru_h: CudaSlice<f32>,
    pub d_mingru_z: CudaSlice<f32>,
    pub d_mingru_h_tilde: CudaSlice<f32>,
    pub d_temp_z_in: CudaSlice<f32>,
    pub d_temp_h_in: CudaSlice<f32>,
    
    // RMSNorm1 output
    pub d_rmsnorm1_out: CudaSlice<f32>,
    pub d_rmsnorm1_rms: CudaSlice<f32>,
    
    // MemoryAttention Fused
    pub d_mem_fused_h: CudaSlice<f32>,
    
    // RMSNorm2 output
    pub d_rmsnorm2_out: CudaSlice<f32>,
    pub d_rmsnorm2_rms: CudaSlice<f32>,
    
    // DeltaNet2 projections & states
    pub d_deltanet_k: CudaSlice<f32>,
    pub d_deltanet_v: CudaSlice<f32>,
    pub d_deltanet_q: CudaSlice<f32>,
    pub d_deltanet_b: CudaSlice<f32>,
    pub d_deltanet_w: CudaSlice<f32>,
    pub d_deltanet_alpha: CudaSlice<f32>,
    pub d_deltanet_b_pre: CudaSlice<f32>,
    pub d_deltanet_w_pre: CudaSlice<f32>,
    pub d_deltanet_alpha_pre: CudaSlice<f32>,
    pub d_deltanet_states: CudaSlice<f32>, // (seq_len + 1) * batch_size * hidden_size * hidden_size
    pub d_deltanet_y: CudaSlice<f32>,
    pub d_deltanet_h: CudaSlice<f32>, // DeltaNet2 output + Residual
    
    // RMSNorm3 output
    pub d_rmsnorm3_out: CudaSlice<f32>,
    pub d_rmsnorm3_rms: CudaSlice<f32>,
    
    // Gating
    pub d_gating_pre: CudaSlice<f32>,
    pub d_gating_out: CudaSlice<f32>,
    
    // Backpropagation gradients
    pub d_grad_h: CudaSlice<f32>, // gradient incoming to gating
    pub d_grad_rmsnorm3_out: CudaSlice<f32>,
    pub d_grad_deltanet_h: CudaSlice<f32>,
    pub d_grad_deltanet_y: CudaSlice<f32>,
    pub d_d_k: CudaSlice<f32>,
    pub d_d_v: CudaSlice<f32>,
    pub d_d_q: CudaSlice<f32>,
    pub d_d_b_pre: CudaSlice<f32>,
    pub d_d_w_pre: CudaSlice<f32>,
    pub d_d_alpha_pre: CudaSlice<f32>,
    pub d_d_S: CudaSlice<f32>, // batch_size * hidden_size * hidden_size
    pub d_G_temp: CudaSlice<f32>, // batch_size * hidden_size
    pub d_R_temp: CudaSlice<f32>, // batch_size * hidden_size
    
    pub d_grad_rmsnorm2_out: CudaSlice<f32>,
    pub d_grad_mem_fused_h: CudaSlice<f32>,
    pub d_grad_rmsnorm1_out: CudaSlice<f32>,
    pub d_grad_mingru_h: CudaSlice<f32>,
    pub d_d_z_pre: CudaSlice<f32>,
    pub d_d_h_pre: CudaSlice<f32>,
    pub d_grad_temp_z_in: CudaSlice<f32>,
    pub d_grad_temp_h_in: CudaSlice<f32>,
    pub d_grad_x: CudaSlice<f32>, // for minGRU backward mapping to input embedding
    
    pub h_rnn_h_cpu: Vec<f32>,
    pub h_fused_h_cpu: Vec<f32>,
    pub h_d_fused_h_cpu: Vec<f32>,
    pub h_d_rnn_h_cpu: Vec<f32>,

    // Reusable Host Scratch Buffers (Pre-allocated in `new` to eliminate malloc in train_chunk)
    pub h_full_emb_cpu: Vec<f32>,
    pub h_flat_inputs: Vec<i32>,
    pub h_flat_targets: Vec<i32>,
    pub h_bulk_targets: Vec<usize>,
    pub h_gate_energies_cpu: Vec<f32>,
    pub h_losses: Vec<f32>,

    pub d_inputs: CudaSlice<i32>,
    pub d_targets: CudaSlice<i32>,
    pub d_dh_next: CudaSlice<f32>,
    pub d_dS_next: CudaSlice<f32>, // batch_size * hidden_size * hidden_size
    pub d_gate_energies: CudaSlice<f32>,
}

pub struct GpuMemoryAttention {
    pub w_q: super::gpu_lora::GpuLoraLinear,
    pub b_q: CudaSlice<f32>,
    pub w_fuse: super::gpu_lora::GpuLoraLinear,
    pub b_fuse: CudaSlice<f32>,
    
    pub d_b_q_grad: CudaSlice<f32>,
    pub d_b_fuse_grad: CudaSlice<f32>,
    // For VRAM forward/backward cache
    pub d_q_in: CudaSlice<f32>, pub d_query: CudaSlice<f32>,
    pub d_mem_out: CudaSlice<f32>, pub d_fuse_in: CudaSlice<f32>,
    pub d_mem_out_grad: CudaSlice<f32>, pub d_q_in_grad: CudaSlice<f32>,
    pub d_scores: CudaSlice<f32>,
    
    // Lion Momentum
    pub m_b_q: CudaSlice<f32>,
    pub m_b_fuse: CudaSlice<f32>,

    // LoRA VRAM scratch buffers
    pub d_lora_temp_rank_q: CudaSlice<f32>,
    pub d_lora_temp_out_q: CudaSlice<f32>,
    pub d_lora_temp_rank_fuse: CudaSlice<f32>,
    pub d_lora_temp_out_fuse: CudaSlice<f32>,
    pub d_lora_d_z_q: CudaSlice<f32>,
    pub d_lora_d_z_fuse: CudaSlice<f32>,
}

fn gemm_backward_x_accum(
    cublas: &CudaBlas,
    d_w: &impl cudarc::driver::DevicePtr<f32>,
    d_dy: &impl cudarc::driver::DevicePtr<f32>,
    d_dx: &mut impl cudarc::driver::DevicePtrMut<f32>,
    out_dim: usize, in_dim: usize, batch_size: usize,
    beta: f32,
) {
    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: in_dim as i32,
        n: batch_size as i32,
        k: out_dim as i32,
        alpha: 1.0,
        lda: in_dim as i32,
        ldb: out_dim as i32,
        beta,
        ldc: in_dim as i32,
    };
    unsafe { cublas.gemm(cfg, d_w, d_dy, d_dx).unwrap(); }
}

fn gemm_batch_nn(
    cublas: &CudaBlas,
    d_a: &impl cudarc::driver::DevicePtr<f32>,
    d_b: &impl cudarc::driver::DevicePtr<f32>,
    d_c: &mut impl cudarc::driver::DevicePtrMut<f32>,
    m: usize, k: usize, n: usize,
) {
    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: m as i32,
        n: n as i32,
        k: k as i32,
        alpha: 1.0,
        lda: m as i32,
        ldb: k as i32,
        beta: 0.0,
        ldc: m as i32,
    };
    unsafe { cublas.gemm(cfg, d_a, d_b, d_c).unwrap(); }
}

pub struct GpuAccelerator {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub cublas: CudaBlas,
    pub module: Arc<CudaModule>,
    
    pub batch_size: usize,
    pub seq_len: usize,
    
    pub rmsnorm1: GpuRMSNorm,
    pub rmsnorm2: GpuRMSNorm,
    pub rmsnorm3: GpuRMSNorm,
    pub mingru: GpuMinGRU,
    pub deltanet2: GpuDeltaNet2,
    pub gating: GpuSparseGating,
    pub embedding: GpuEmbedding,
    pub memory_attention: GpuMemoryAttention,
    pub history: GpuHistory,
}

fn gemm_batch(
    cublas: &CudaBlas,
    d_w: &impl cudarc::driver::DevicePtr<f32>,
    d_x: &impl cudarc::driver::DevicePtr<f32>,
    d_y: &mut impl cudarc::driver::DevicePtrMut<f32>,
    out_dim: usize, in_dim: usize, batch_size: usize,
) {
    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_T,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: out_dim as i32,
        n: batch_size as i32,
        k: in_dim as i32,
        alpha: 1.0,
        lda: in_dim as i32,
        ldb: in_dim as i32,
        beta: 0.0,
        ldc: out_dim as i32,
    };
    unsafe { cublas.gemm(cfg, d_w, d_x, d_y).unwrap(); }
}

fn gemm_backward_w(
    cublas: &CudaBlas,
    d_x: &impl cudarc::driver::DevicePtr<f32>,
    d_dy: &impl cudarc::driver::DevicePtr<f32>,
    d_dw: &mut impl cudarc::driver::DevicePtrMut<f32>,
    out_dim: usize, in_dim: usize, batch_size: usize,
) {
    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_T,
        m: in_dim as i32,
        n: out_dim as i32,
        k: batch_size as i32,
        alpha: 1.0,
        lda: in_dim as i32,
        ldb: out_dim as i32,
        beta: 1.0,
        ldc: in_dim as i32,
    };
    unsafe { cublas.gemm(cfg, d_x, d_dy, d_dw).unwrap(); }
}

impl GpuAccelerator {
    pub fn new(vocab_size: usize, embed_dim: usize, hidden_size: usize, batch_size: usize, seq_len: usize, memory_capacity: usize, lora_rank: usize, lora_alpha: f32) -> Self {
        let ctx = CudaContext::new(0).expect("No CUDA device");
        let stream = ctx.new_stream().unwrap();
        let cublas = CudaBlas::new(stream.clone()).unwrap();
        
        let ptx_content = include_str!("kernel.cu");
        let ptx = cudarc::nvrtc::compile_ptx(ptx_content).unwrap();
        let module = ctx.load_module(ptx).unwrap();
        
        let alloc_zeros = |size: usize| stream.alloc_zeros::<f32>(size).unwrap();
        let alloc_zeros_i32 = |size: usize| stream.alloc_zeros::<i32>(size).unwrap();
        
        let rmsnorm1 = GpuRMSNorm {
            dim: hidden_size,
            d_weight: alloc_zeros(hidden_size),
            d_weight_grad: alloc_zeros(hidden_size),
            m_weight: alloc_zeros(hidden_size),
        };
        
        let rmsnorm2 = GpuRMSNorm {
            dim: hidden_size,
            d_weight: alloc_zeros(hidden_size),
            d_weight_grad: alloc_zeros(hidden_size),
            m_weight: alloc_zeros(hidden_size),
        };
        
        let rmsnorm3 = GpuRMSNorm {
            dim: hidden_size,
            d_weight: alloc_zeros(hidden_size),
            d_weight_grad: alloc_zeros(hidden_size),
            m_weight: alloc_zeros(hidden_size),
        };
        
        let mingru = GpuMinGRU {
            hidden_size, input_size: embed_dim,
            d_w_z: alloc_zeros(hidden_size * embed_dim), d_b_z: alloc_zeros(hidden_size),
            d_w_h: alloc_zeros(hidden_size * embed_dim), d_b_h: alloc_zeros(hidden_size),
            d_w_z_grad: alloc_zeros(hidden_size * embed_dim), d_b_z_grad: alloc_zeros(hidden_size),
            d_w_h_grad: alloc_zeros(hidden_size * embed_dim), d_b_h_grad: alloc_zeros(hidden_size),
            
            m_w_z: alloc_zeros(hidden_size * embed_dim), m_b_z: alloc_zeros(hidden_size),
            m_w_h: alloc_zeros(hidden_size * embed_dim), m_b_h: alloc_zeros(hidden_size),
        };
        
        let deltanet2 = GpuDeltaNet2 {
            hidden_size,
            d_w_k: alloc_zeros(hidden_size * hidden_size), d_w_v: alloc_zeros(hidden_size * hidden_size), d_w_q: alloc_zeros(hidden_size * hidden_size),
            d_w_b: alloc_zeros(hidden_size * hidden_size), d_w_w: alloc_zeros(hidden_size * hidden_size), d_w_alpha: alloc_zeros(hidden_size * hidden_size),
            
            d_w_k_grad: alloc_zeros(hidden_size * hidden_size), d_w_v_grad: alloc_zeros(hidden_size * hidden_size), d_w_q_grad: alloc_zeros(hidden_size * hidden_size),
            d_w_b_grad: alloc_zeros(hidden_size * hidden_size), d_w_w_grad: alloc_zeros(hidden_size * hidden_size), d_w_alpha_grad: alloc_zeros(hidden_size * hidden_size),
            
            m_w_k: alloc_zeros(hidden_size * hidden_size), m_w_v: alloc_zeros(hidden_size * hidden_size), m_w_q: alloc_zeros(hidden_size * hidden_size),
            m_w_b: alloc_zeros(hidden_size * hidden_size), m_w_w: alloc_zeros(hidden_size * hidden_size), m_w_alpha: alloc_zeros(hidden_size * hidden_size),
        };
        
        let gating = GpuSparseGating {
            hidden_size, input_size: hidden_size,
            d_w1: alloc_zeros(hidden_size * hidden_size), d_b1: alloc_zeros(hidden_size),
            d_w1_grad: alloc_zeros(hidden_size * hidden_size), d_b1_grad: alloc_zeros(hidden_size),
            
            m_w1: alloc_zeros(hidden_size * hidden_size), m_b1: alloc_zeros(hidden_size),
        };
        
        let embedding = GpuEmbedding {
            vocab_size, embed_dim,
            d_weight: alloc_zeros(vocab_size * embed_dim), d_weight_grad: alloc_zeros(vocab_size * embed_dim),
            
            m_weight: alloc_zeros(vocab_size * embed_dim),
        };
        
        let full_batch_h = seq_len * batch_size * hidden_size;
        let full_batch_e = seq_len * batch_size * embed_dim;
        let full_batch_len = seq_len * batch_size;
        let batch_state_s = batch_size * hidden_size * hidden_size;
        let full_batch_s = (seq_len + 1) * batch_state_s;
        
        let history = GpuHistory {
            d_x: alloc_zeros(full_batch_e),
            
            d_mingru_h: alloc_zeros(full_batch_h + batch_size * hidden_size),
            d_mingru_z: alloc_zeros(full_batch_h),
            d_mingru_h_tilde: alloc_zeros(full_batch_h),
            d_temp_z_in: alloc_zeros(full_batch_h),
            d_temp_h_in: alloc_zeros(full_batch_h),
            
            d_rmsnorm1_out: alloc_zeros(full_batch_h),
            d_rmsnorm1_rms: alloc_zeros(full_batch_len),
            
            d_mem_fused_h: alloc_zeros(full_batch_h),
            
            d_rmsnorm2_out: alloc_zeros(full_batch_h),
            d_rmsnorm2_rms: alloc_zeros(full_batch_len),
            
            d_deltanet_k: alloc_zeros(full_batch_h),
            d_deltanet_v: alloc_zeros(full_batch_h),
            d_deltanet_q: alloc_zeros(full_batch_h),
            d_deltanet_b: alloc_zeros(full_batch_h),
            d_deltanet_w: alloc_zeros(full_batch_h),
            d_deltanet_alpha: alloc_zeros(full_batch_h),
            d_deltanet_b_pre: alloc_zeros(full_batch_h),
            d_deltanet_w_pre: alloc_zeros(full_batch_h),
            d_deltanet_alpha_pre: alloc_zeros(full_batch_h),
            d_deltanet_states: alloc_zeros(full_batch_s),
            d_deltanet_y: alloc_zeros(full_batch_h),
            d_deltanet_h: alloc_zeros(full_batch_h),
            
            d_rmsnorm3_out: alloc_zeros(full_batch_h),
            d_rmsnorm3_rms: alloc_zeros(full_batch_len),
            
            d_gating_pre: alloc_zeros(full_batch_h),
            d_gating_out: alloc_zeros(full_batch_h),
            
            d_grad_h: alloc_zeros(full_batch_h),
            d_grad_rmsnorm3_out: alloc_zeros(full_batch_h),
            d_grad_deltanet_h: alloc_zeros(full_batch_h),
            d_grad_deltanet_y: alloc_zeros(full_batch_h),
            d_d_k: alloc_zeros(full_batch_h),
            d_d_v: alloc_zeros(full_batch_h),
            d_d_q: alloc_zeros(full_batch_h),
            d_d_b_pre: alloc_zeros(full_batch_h),
            d_d_w_pre: alloc_zeros(full_batch_h),
            d_d_alpha_pre: alloc_zeros(full_batch_h),
            d_d_S: alloc_zeros(batch_state_s),
            d_G_temp: alloc_zeros(full_batch_h),
            d_R_temp: alloc_zeros(full_batch_h),
            
            d_grad_rmsnorm2_out: alloc_zeros(full_batch_h),
            d_grad_mem_fused_h: alloc_zeros(full_batch_h),
            d_grad_rmsnorm1_out: alloc_zeros(full_batch_h),
            d_grad_mingru_h: alloc_zeros(full_batch_h + batch_size * hidden_size),
            d_d_z_pre: alloc_zeros(full_batch_h),
            d_d_h_pre: alloc_zeros(full_batch_h),
            d_grad_temp_z_in: alloc_zeros(full_batch_h),
            d_grad_temp_h_in: alloc_zeros(full_batch_h),
            d_grad_x: alloc_zeros(full_batch_e),
            
            h_rnn_h_cpu: vec![0.0; full_batch_h],
            h_fused_h_cpu: vec![0.0; full_batch_h],
            h_d_fused_h_cpu: vec![0.0; full_batch_h],
            h_d_rnn_h_cpu: vec![0.0; full_batch_h],
            h_full_emb_cpu: vec![0.0; full_batch_e],
            h_flat_inputs: vec![0; full_batch_len],
            h_flat_targets: vec![0; full_batch_len],
            h_bulk_targets: vec![0; full_batch_len],
            h_gate_energies_cpu: vec![0.0; full_batch_len],
            h_losses: vec![0.0; full_batch_len],
            d_inputs: alloc_zeros_i32(full_batch_len),
            d_targets: alloc_zeros_i32(full_batch_len),
            d_dh_next: alloc_zeros(batch_size * hidden_size),
            d_dS_next: alloc_zeros(batch_state_s),
            d_gate_energies: alloc_zeros(full_batch_len),
        };
        
        let w_q = super::gpu_lora::GpuLoraLinear::new(&stream, embed_dim, hidden_size, lora_rank, lora_alpha);
        let w_fuse = super::gpu_lora::GpuLoraLinear::new(&stream, hidden_size, hidden_size, lora_rank, lora_alpha);
        
        let memory_attention = GpuMemoryAttention {
            w_q, b_q: alloc_zeros(embed_dim),
            w_fuse, b_fuse: alloc_zeros(hidden_size),
            d_b_q_grad: alloc_zeros(embed_dim),
            d_b_fuse_grad: alloc_zeros(hidden_size),
            d_q_in: alloc_zeros(full_batch_e), d_query: alloc_zeros(full_batch_e),
            d_mem_out: alloc_zeros(full_batch_h), d_fuse_in: alloc_zeros(full_batch_h),
            d_mem_out_grad: alloc_zeros(full_batch_h), d_q_in_grad: alloc_zeros(full_batch_e),
            d_scores: alloc_zeros(full_batch_len * memory_capacity),
            
            // Lion Momentum
            m_b_q: alloc_zeros(embed_dim),
            m_b_fuse: alloc_zeros(hidden_size),

            // LoRA VRAM scratch buffers
            d_lora_temp_rank_q: alloc_zeros(full_batch_len * lora_rank),
            d_lora_temp_out_q: alloc_zeros(full_batch_len * embed_dim),
            d_lora_temp_rank_fuse: alloc_zeros(full_batch_len * lora_rank),
            d_lora_temp_out_fuse: alloc_zeros(full_batch_len * hidden_size),
            d_lora_d_z_q: alloc_zeros(full_batch_len * lora_rank),
            d_lora_d_z_fuse: alloc_zeros(full_batch_len * lora_rank),
        };
        
        Self { ctx, stream, cublas, module, batch_size, seq_len, rmsnorm1, rmsnorm2, rmsnorm3, mingru, deltanet2, gating, embedding, memory_attention, history }
    }
    
    pub fn load_weights(&mut self, pipeline: &AntPipeline) {
        let htod = |dst: &mut CudaSlice<f32>, src: &[f32]| {
            let len = src.len();
            let mut sub = dst.try_slice_mut(0..len).unwrap();
            self.stream.memcpy_htod(src, &mut sub).unwrap();
        };

        htod(&mut self.mingru.d_w_z, &pipeline.mingru.w_z.data); htod(&mut self.mingru.d_b_z, &pipeline.mingru.b_z.data);
        htod(&mut self.mingru.d_w_h, &pipeline.mingru.w_h.data); htod(&mut self.mingru.d_b_h, &pipeline.mingru.b_h.data);
        
        htod(&mut self.rmsnorm1.d_weight, &pipeline.rmsnorm1.weight.data);
        htod(&mut self.rmsnorm2.d_weight, &pipeline.rmsnorm2.weight.data);
        htod(&mut self.rmsnorm3.d_weight, &pipeline.rmsnorm3.weight.data);
        
        htod(&mut self.deltanet2.d_w_k, &pipeline.deltanet2.w_k.data);
        htod(&mut self.deltanet2.d_w_v, &pipeline.deltanet2.w_v.data);
        htod(&mut self.deltanet2.d_w_q, &pipeline.deltanet2.w_q.data);
        htod(&mut self.deltanet2.d_w_b, &pipeline.deltanet2.w_b.data);
        htod(&mut self.deltanet2.d_w_w, &pipeline.deltanet2.w_w.data);
        htod(&mut self.deltanet2.d_w_alpha, &pipeline.deltanet2.w_alpha.data);
        
        htod(&mut self.gating.d_w1, &pipeline.gating.w1.data); htod(&mut self.gating.d_b1, &pipeline.gating.b1.data);
        htod(&mut self.embedding.d_weight, &pipeline.embedding.weight.data);
        htod(&mut self.memory_attention.w_q.d_w_base, &pipeline.memory_attention.w_q.base.data);
        htod(&mut self.memory_attention.w_q.d_lora_a, &pipeline.memory_attention.w_q.lora_a.data);
        htod(&mut self.memory_attention.w_q.d_lora_b, &pipeline.memory_attention.w_q.lora_b.data);
        self.memory_attention.w_q.enabled = pipeline.memory_attention.w_q.enabled;
        htod(&mut self.memory_attention.b_q, &pipeline.memory_attention.b_q.data);
        
        htod(&mut self.memory_attention.w_fuse.d_w_base, &pipeline.memory_attention.w_fuse.base.data);
        htod(&mut self.memory_attention.w_fuse.d_lora_a, &pipeline.memory_attention.w_fuse.lora_a.data);
        htod(&mut self.memory_attention.w_fuse.d_lora_b, &pipeline.memory_attention.w_fuse.lora_b.data);
        self.memory_attention.w_fuse.enabled = pipeline.memory_attention.w_fuse.enabled;
        htod(&mut self.memory_attention.b_fuse, &pipeline.memory_attention.b_fuse.data);
    }
    
    pub fn save_weights(&self, pipeline: &mut AntPipeline) {
        let dtoh = |dst: &mut [f32], src: &CudaSlice<f32>| {
            let len = dst.len();
            let sub = src.try_slice(0..len).unwrap();
            self.stream.memcpy_dtoh(&sub, dst).unwrap();
        };

        dtoh(&mut pipeline.mingru.w_z.data, &self.mingru.d_w_z); dtoh(&mut pipeline.mingru.b_z.data, &self.mingru.d_b_z);
        dtoh(&mut pipeline.mingru.w_h.data, &self.mingru.d_w_h); dtoh(&mut pipeline.mingru.b_h.data, &self.mingru.d_b_h);
        
        dtoh(&mut pipeline.rmsnorm1.weight.data, &self.rmsnorm1.d_weight);
        dtoh(&mut pipeline.rmsnorm2.weight.data, &self.rmsnorm2.d_weight);
        dtoh(&mut pipeline.rmsnorm3.weight.data, &self.rmsnorm3.d_weight);
        
        dtoh(&mut pipeline.deltanet2.w_k.data, &self.deltanet2.d_w_k);
        dtoh(&mut pipeline.deltanet2.w_v.data, &self.deltanet2.d_w_v);
        dtoh(&mut pipeline.deltanet2.w_q.data, &self.deltanet2.d_w_q);
        dtoh(&mut pipeline.deltanet2.w_b.data, &self.deltanet2.d_w_b);
        dtoh(&mut pipeline.deltanet2.w_w.data, &self.deltanet2.d_w_w);
        dtoh(&mut pipeline.deltanet2.w_alpha.data, &self.deltanet2.d_w_alpha);
        
        if pipeline.gpu_readout.is_none() {
            dtoh(&mut pipeline.embedding.weight.data, &self.embedding.d_weight);
        }
        dtoh(&mut pipeline.memory_attention.w_q.base.data, &self.memory_attention.w_q.d_w_base);
        dtoh(&mut pipeline.memory_attention.w_q.lora_a.data, &self.memory_attention.w_q.d_lora_a);
        dtoh(&mut pipeline.memory_attention.w_q.lora_b.data, &self.memory_attention.w_q.d_lora_b);
        dtoh(&mut pipeline.memory_attention.b_q.data, &self.memory_attention.b_q);
        
        dtoh(&mut pipeline.memory_attention.w_fuse.base.data, &self.memory_attention.w_fuse.d_w_base);
        dtoh(&mut pipeline.memory_attention.w_fuse.lora_a.data, &self.memory_attention.w_fuse.d_lora_a);
        dtoh(&mut pipeline.memory_attention.w_fuse.lora_b.data, &self.memory_attention.w_fuse.d_lora_b);
        dtoh(&mut pipeline.memory_attention.b_fuse.data, &self.memory_attention.b_fuse);
    }
    
    pub fn train_chunk(
        &mut self,
        pipeline: &mut AntPipeline,
        inputs: &[Vec<usize>],
        targets: &[Vec<usize>],
        chunk_loss: &mut f32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
    ) {
        let c = inputs.len();
        let b = self.batch_size;
        let h = self.mingru.hidden_size;
        let e = self.embedding.embed_dim;
        let vocab_size = self.embedding.vocab_size;
        let bh = b * h;
        let be = b * e;
        let state_size = b * h * h;
        

        // ================= HOIST KERNEL LOADS =================
        let f_zero = self.module.load_function("zero_buffer_kernel").unwrap();
        let f_add_mat = self.module.load_function("add_matrices_kernel").unwrap();
        let f_add_bias = self.module.load_function("add_bias_kernel").unwrap();
        let f_bias_bw = self.module.load_function("bias_backward_kernel").unwrap();
        let f_emb_bw = self.module.load_function("embedding_backward_kernel").unwrap();
        let f_sg = self.module.load_function("sparse_gating_forward_kernel").unwrap();
        let f_sg_bw = self.module.load_function("sparse_gating_backward_kernel").unwrap();
        
        let f_rmsnorm_fw = self.module.load_function("rmsnorm_forward_kernel").unwrap();
        let f_rmsnorm_bw = self.module.load_function("rmsnorm_backward_kernel").unwrap();
        let f_mingru_fw = self.module.load_function("mingru_forward_kernel").unwrap();
        let f_mingru_bw = self.module.load_function("mingru_backward_kernel").unwrap();
        let f_deltanet_fw = self.module.load_function("deltanet2_forward_kernel").unwrap();
        let f_deltanet_bw1 = self.module.load_function("deltanet2_backward_pass1_kernel").unwrap();
        let f_deltanet_bw2 = self.module.load_function("deltanet2_backward_pass2_kernel").unwrap();
        
        let f_tanh_bias = self.module.load_function("tanh_bias_kernel").unwrap();
        let f_lookup_batch = self.module.load_function("memory_lookup_batch").unwrap();
        let f_softmax = self.module.load_function("softmax_rows_kernel").unwrap();
        let f_energy = self.module.load_function("compute_gate_energy_kernel").unwrap();
        // ======================================================

        // ================= ZERO GRADIENTS =====================
        let zero = |buf: &mut CudaSlice<f32>, size: usize| {
            let cfg = LaunchConfig { grid_dim: (((size + 255)/256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { self.stream.launch_builder(&f_zero).arg(buf).arg(&(size as i32)).launch(cfg) }.unwrap();
        };
        let zero_view = |buf: &mut CudaViewMut<'_, f32>, size: usize| {
            let cfg = LaunchConfig { grid_dim: (((size + 255)/256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { self.stream.launch_builder(&f_zero).arg(buf).arg(&(size as i32)).launch(cfg) }.unwrap();
        };
        
        zero(&mut self.mingru.d_w_z_grad, h * e); zero(&mut self.mingru.d_b_z_grad, h);
        zero(&mut self.mingru.d_w_h_grad, h * e); zero(&mut self.mingru.d_b_h_grad, h);
        
        zero(&mut self.deltanet2.d_w_k_grad, h * h); zero(&mut self.deltanet2.d_w_v_grad, h * h); zero(&mut self.deltanet2.d_w_q_grad, h * h);
        zero(&mut self.deltanet2.d_w_b_grad, h * h); zero(&mut self.deltanet2.d_w_w_grad, h * h); zero(&mut self.deltanet2.d_w_alpha_grad, h * h);
        
        zero(&mut self.rmsnorm1.d_weight_grad, h);
        zero(&mut self.rmsnorm2.d_weight_grad, h);
        zero(&mut self.rmsnorm3.d_weight_grad, h);
        
        zero(&mut self.gating.d_w1_grad, h * h); zero(&mut self.gating.d_b1_grad, h);
        zero(&mut self.embedding.d_weight_grad, vocab_size * e);
        self.memory_attention.w_fuse.zero_grad(&self.stream, &f_zero);
        zero(&mut self.memory_attention.d_b_fuse_grad, h);
        self.memory_attention.w_q.zero_grad(&self.stream, &f_zero);
        zero(&mut self.memory_attention.d_b_q_grad, e);
        
        pipeline.gpu_readout.as_mut().unwrap().zero_grad();
        // ======================================================

        // ================= POPULATE BATCH =================
        for t in 0..c {
            for batch_idx in 0..b {
                let id = inputs[t][batch_idx];
                let w_row = &pipeline.embedding.weight.data[(id * e)..((id + 1) * e)];
                let offset = (t * b * e) + (batch_idx * e);
                self.history.h_full_emb_cpu[offset..(offset + e)].copy_from_slice(w_row);
                
                self.history.h_flat_inputs[t * b + batch_idx] = id as i32;
                self.history.h_flat_targets[t * b + batch_idx] = targets[t][batch_idx] as i32;
                self.history.h_bulk_targets[t * b + batch_idx] = targets[t][batch_idx];
            }
        }
        
        let mut d_inputs_all = self.history.d_inputs.try_slice_mut(0 .. c*b).unwrap();
        self.stream.memcpy_htod(&self.history.h_flat_inputs[..c*b], &mut d_inputs_all).unwrap();
        
        let mut d_targets_all = self.history.d_targets.try_slice_mut(0 .. c*b).unwrap();
        self.stream.memcpy_htod(&self.history.h_flat_targets[..c*b], &mut d_targets_all).unwrap();
        
        // ================= BULK GEMM FOR INPUTS =================
        let mut d_x_all = self.history.d_x.try_slice_mut(0..c*b*e).unwrap();
        self.stream.memcpy_htod(&self.history.h_full_emb_cpu[..c*b*e], &mut d_x_all).unwrap();

        let d_x_all_ro = self.history.d_x.try_slice(0..c*b*e).unwrap();
        
        let mut d_temp_z_in_all = self.history.d_temp_z_in.try_slice_mut(0..c*b*h).unwrap();
        gemm_batch(&self.cublas, &self.mingru.d_w_z, &d_x_all_ro, &mut d_temp_z_in_all, h, e, c * b);
        
        let mut d_temp_h_in_all = self.history.d_temp_h_in.try_slice_mut(0..c*b*h).unwrap();
        gemm_batch(&self.cublas, &self.mingru.d_w_h, &d_x_all_ro, &mut d_temp_h_in_all, h, e, c * b);
        // ========================================================

        // PHASE 1: minGRU Recurrence Loop
        for t in 0..c {
            let mut d_prev_h_t = self.history.d_mingru_h.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            if t == 0 {
                self.stream.memcpy_htod(&pipeline.mingru.hidden_state.data.storage[..b * h], &mut d_prev_h_t).unwrap();
            }

            let prev_h_ptr = {
                let view = self.history.d_mingru_h.try_slice(t*bh .. (t+1)*bh).unwrap();
                view.device_ptr(&self.stream).0
            };
            
            let temp_z_in_v = self.history.d_temp_z_in.try_slice(t*bh .. (t+1)*bh).unwrap();
            let temp_h_in_v = self.history.d_temp_h_in.try_slice(t*bh .. (t+1)*bh).unwrap();
            
            let mut z_v = self.history.d_mingru_z.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            let mut h_tilde_v = self.history.d_mingru_h_tilde.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            
            let next_idx = (t + 1) * bh;
            let mut h_out_v = self.history.d_mingru_h.try_slice_mut(next_idx .. next_idx + bh).unwrap();
            
            let cfg = LaunchConfig { grid_dim: (((h + 255) / 256) as u32, b as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe {
                self.stream.launch_builder(&f_mingru_fw)
                    .arg(&temp_z_in_v).arg(&self.mingru.d_b_z)
                    .arg(&temp_h_in_v).arg(&self.mingru.d_b_h)
                    .arg(&prev_h_ptr)
                    .arg(&mut z_v).arg(&mut h_tilde_v).arg(&mut h_out_v)
                    .arg(&(b as i32)).arg(&(h as i32))
                    .launch(cfg)
            }.unwrap();
        }

        // PHASE 2: RMSNorm1 on minGRU outputs (across all seq steps at once!)
        {
            let d_mingru_h_out = self.history.d_mingru_h.try_slice(bh .. (c+1)*bh).unwrap(); // steps 1..c+1
            let mut d_rmsnorm1_out = self.history.d_rmsnorm1_out.try_slice_mut(0 .. c*bh).unwrap();
            let mut d_rmsnorm1_rms = self.history.d_rmsnorm1_rms.try_slice_mut(0 .. c*b).unwrap();
            
            let cfg = LaunchConfig { grid_dim: (1, (c * b) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 256 * 4 };
            unsafe {
                self.stream.launch_builder(&f_rmsnorm_fw)
                    .arg(&d_mingru_h_out).arg(&self.rmsnorm1.d_weight)
                    .arg(&mut d_rmsnorm1_out).arg(&mut d_rmsnorm1_rms)
                    .arg(&((c*b) as i32)).arg(&(h as i32)).arg(&1e-6f32)
                    .launch(cfg)
            }.unwrap();
        }

        // PHASE 3: MemoryAttention
        let d_rmsnorm1_out_all = self.history.d_rmsnorm1_out.try_slice(0 .. c*bh).unwrap();
        let mut d_q_in_all = self.memory_attention.d_q_in.try_slice_mut(0 .. c*b*e).unwrap();
        let mut d_temp_rank_q = self.memory_attention.d_lora_temp_rank_q.try_slice_mut(0..c * b * self.memory_attention.w_q.rank).unwrap();
        let mut d_temp_out_q = self.memory_attention.d_lora_temp_out_q.try_slice_mut(0..c * b * e).unwrap();
        self.memory_attention.w_q.forward(
            &self.cublas,
            &d_rmsnorm1_out_all,
            &mut d_q_in_all,
            &mut d_temp_rank_q,
            &mut d_temp_out_q,
            c * b,
        );
        
        let tanh_cfg = LaunchConfig { grid_dim: (((e + 255) / 256) as u32, (c * b) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut d_query_all = self.memory_attention.d_query.try_slice_mut(0 .. c*b*e).unwrap();
        let d_q_in_all_ro = self.memory_attention.d_q_in.try_slice(0 .. c*b*e).unwrap();
        unsafe {
            self.stream.launch_builder(&f_tanh_bias)
                .arg(&d_q_in_all_ro)
                .arg(&self.memory_attention.b_q)
                .arg(&mut d_query_all)
                .arg(&( (c * b) as i32 ))
                .arg(&( e as i32 ))
                .launch(tanh_cfg)
        }.unwrap();
        
        let current_size = if let Some(ref gpu_mem) = pipeline.gpu_memory {
            gpu_mem.base_size + gpu_mem.user_size
        } else {
            pipeline.base_memory.current_size + pipeline.user_memory.current_size
        };
        let mut d_mem_out_all = self.memory_attention.d_mem_out.try_slice_mut(0 .. c*bh).unwrap();
        
        if current_size == 0 {
            zero_view(&mut d_mem_out_all, c * bh);
        } else {
            let lookup_cfg = LaunchConfig {
                grid_dim: (((current_size + 255) / 256) as u32, (c * b) as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let gpu_mem = pipeline.gpu_memory.as_ref().unwrap();
            let mut d_scores_all = self.memory_attention.d_scores.try_slice_mut(0 .. c * b * current_size).unwrap();
            let d_query_all_ro = self.memory_attention.d_query.try_slice(0 .. c*b*e).unwrap();
            unsafe {
                self.stream.launch_builder(&f_lookup_batch)
                    .arg(&d_query_all_ro)
                    .arg(&gpu_mem.d_keys)
                    .arg(&mut d_scores_all)
                    .arg(&( (c * b) as i32 ))
                    .arg(&( current_size as i32 ))
                    .arg(&( e as i32 ))
                    .launch(lookup_cfg)
            }.unwrap();
            
            let softmax_cfg = LaunchConfig {
                grid_dim: (((c * b + 255) / 256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let scale_factor = 1.0f32 / (e as f32).sqrt() / 0.2f32;
            unsafe {
                self.stream.launch_builder(&f_softmax)
                    .arg(&mut d_scores_all)
                    .arg(&( (c * b) as i32 ))
                    .arg(&( current_size as i32 ))
                    .arg(&scale_factor)
                    .launch(softmax_cfg)
            }.unwrap();
            
            let gpu_mem = pipeline.gpu_memory.as_ref().unwrap();
            gemm_batch_nn(
                &self.cublas,
                &gpu_mem.d_vals,
                &d_scores_all,
                &mut d_mem_out_all,
                h, current_size, c * b
            );
        }
        
        let mut d_fuse_in_all = self.memory_attention.d_fuse_in.try_slice_mut(0 .. c*bh).unwrap();
        let mut d_temp_rank_fuse = self.memory_attention.d_lora_temp_rank_fuse.try_slice_mut(0..c * b * self.memory_attention.w_fuse.rank).unwrap();
        let mut d_temp_out_fuse = self.memory_attention.d_lora_temp_out_fuse.try_slice_mut(0..c * b * h).unwrap();
        self.memory_attention.w_fuse.forward(
            &self.cublas,
            &d_mem_out_all,
            &mut d_fuse_in_all,
            &mut d_temp_rank_fuse,
            &mut d_temp_out_fuse,
            c * b,
        );
        
        let bias_cfg = LaunchConfig { grid_dim: (((h + 255) / 256) as u32, (c * b) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&f_add_bias)
                .arg(&mut d_fuse_in_all)
                .arg(&self.memory_attention.b_fuse)
                .arg(&( (c * b) as i32 ))
                .arg(&( h as i32 ))
                .launch(bias_cfg)
        }.unwrap();
        
        let mut d_mem_fused_h_all = self.history.d_mem_fused_h.try_slice_mut(0 .. c*bh).unwrap();
        let d_mingru_h_out = self.history.d_mingru_h.try_slice(bh .. (c+1)*bh).unwrap();
        let d_fuse_in_all_ro = self.memory_attention.d_fuse_in.try_slice(0 .. c*bh).unwrap();
        
        let cfg_add = LaunchConfig { grid_dim: ((((c*bh) + 255)/256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        self.stream.memcpy_dtod(&d_mingru_h_out, &mut d_mem_fused_h_all).unwrap();
        unsafe { 
            self.stream.launch_builder(&f_add_mat).arg(&mut d_mem_fused_h_all).arg(&d_fuse_in_all_ro).arg(&( (c*bh) as i32 )).launch(cfg_add).unwrap();
        }

        // PHASE 4: RMSNorm2 (on memory fused hidden states)
        {
            let d_mem_fused_h_all_ro = self.history.d_mem_fused_h.try_slice(0 .. c*bh).unwrap();
            let mut d_rmsnorm2_out = self.history.d_rmsnorm2_out.try_slice_mut(0 .. c*bh).unwrap();
            let mut d_rmsnorm2_rms = self.history.d_rmsnorm2_rms.try_slice_mut(0 .. c*b).unwrap();
            
            let cfg = LaunchConfig { grid_dim: (1, (c * b) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 256 * 4 };
            unsafe {
                self.stream.launch_builder(&f_rmsnorm_fw)
                    .arg(&d_mem_fused_h_all_ro).arg(&self.rmsnorm2.d_weight)
                    .arg(&mut d_rmsnorm2_out).arg(&mut d_rmsnorm2_rms)
                    .arg(&((c*b) as i32)).arg(&(h as i32)).arg(&1e-6f32)
                    .launch(cfg)
            }.unwrap();
        }

        // PHASE 5: Gated DeltaNet-2 Layer
        // 1. Projections
        let d_rmsnorm2_out_all = self.history.d_rmsnorm2_out.try_slice(0 .. c*bh).unwrap();
        
        let mut d_deltanet_k_all = self.history.d_deltanet_k.try_slice_mut(0 .. c*bh).unwrap();
        gemm_batch(&self.cublas, &self.deltanet2.d_w_k, &d_rmsnorm2_out_all, &mut d_deltanet_k_all, h, h, c * b);
        
        let mut d_deltanet_v_all = self.history.d_deltanet_v.try_slice_mut(0 .. c*bh).unwrap();
        gemm_batch(&self.cublas, &self.deltanet2.d_w_v, &d_rmsnorm2_out_all, &mut d_deltanet_v_all, h, h, c * b);
        
        let mut d_deltanet_q_all = self.history.d_deltanet_q.try_slice_mut(0 .. c*bh).unwrap();
        gemm_batch(&self.cublas, &self.deltanet2.d_w_q, &d_rmsnorm2_out_all, &mut d_deltanet_q_all, h, h, c * b);
        
        let mut d_deltanet_b_pre_all = self.history.d_deltanet_b_pre.try_slice_mut(0 .. c*bh).unwrap();
        gemm_batch(&self.cublas, &self.deltanet2.d_w_b, &d_rmsnorm2_out_all, &mut d_deltanet_b_pre_all, h, h, c * b);
        
        let mut d_deltanet_w_pre_all = self.history.d_deltanet_w_pre.try_slice_mut(0 .. c*bh).unwrap();
        gemm_batch(&self.cublas, &self.deltanet2.d_w_w, &d_rmsnorm2_out_all, &mut d_deltanet_w_pre_all, h, h, c * b);
        
        let mut d_deltanet_alpha_pre_all = self.history.d_deltanet_alpha_pre.try_slice_mut(0 .. c*bh).unwrap();
        gemm_batch(&self.cublas, &self.deltanet2.d_w_alpha, &d_rmsnorm2_out_all, &mut d_deltanet_alpha_pre_all, h, h, c * b);
        
        // 2. Recurrence Loop for Gated DeltaNet-2
        // S_history size: (c+1) * b * h * h
        for t in 0..c {
            let mut d_prev_state = self.history.d_deltanet_states.try_slice_mut(t * state_size .. (t+1) * state_size).unwrap();
            if t == 0 {
                self.stream.memcpy_htod(&pipeline.deltanet2.state[..b * h * h], &mut d_prev_state).unwrap();
            }
            
            let prev_state_ptr = {
                let view = self.history.d_deltanet_states.try_slice(t * state_size .. (t+1) * state_size).unwrap();
                view.device_ptr(&self.stream).0
            };
            let temp_k_ptr = {
                let view = self.history.d_deltanet_k.try_slice(t*bh .. (t+1)*bh).unwrap();
                view.device_ptr(&self.stream).0
            };
            let temp_v_ptr = {
                let view = self.history.d_deltanet_v.try_slice(t*bh .. (t+1)*bh).unwrap();
                view.device_ptr(&self.stream).0
            };
            let temp_q_ptr = {
                let view = self.history.d_deltanet_q.try_slice(t*bh .. (t+1)*bh).unwrap();
                view.device_ptr(&self.stream).0
            };
            let temp_b_ptr = {
                let view = self.history.d_deltanet_b_pre.try_slice(t*bh .. (t+1)*bh).unwrap();
                view.device_ptr(&self.stream).0
            };
            let temp_w_ptr = {
                let view = self.history.d_deltanet_w_pre.try_slice(t*bh .. (t+1)*bh).unwrap();
                view.device_ptr(&self.stream).0
            };
            let temp_alpha_ptr = {
                let view = self.history.d_deltanet_alpha_pre.try_slice(t*bh .. (t+1)*bh).unwrap();
                view.device_ptr(&self.stream).0
            };
            
            let mut d_next_state = self.history.d_deltanet_states.try_slice_mut((t+1) * state_size .. (t+2) * state_size).unwrap();
            
            let mut k_v = self.history.d_deltanet_k.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            let mut v_v = self.history.d_deltanet_v.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            let mut q_v = self.history.d_deltanet_q.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            let mut b_v = self.history.d_deltanet_b.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            let mut w_v = self.history.d_deltanet_w.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            let mut alpha_v = self.history.d_deltanet_alpha.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            
            let mut y_v = self.history.d_deltanet_y.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            zero_view(&mut y_v, bh);
            
            let cfg = LaunchConfig { grid_dim: (((h + 255) / 256) as u32, b as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe {
                self.stream.launch_builder(&f_deltanet_fw)
                    .arg(&temp_k_ptr).arg(&temp_v_ptr).arg(&temp_q_ptr)
                    .arg(&temp_b_ptr).arg(&temp_w_ptr).arg(&temp_alpha_ptr)
                    .arg(&prev_state_ptr).arg(&mut d_next_state).arg(&mut y_v)
                    .arg(&mut k_v).arg(&mut v_v).arg(&mut q_v)
                    .arg(&mut b_v).arg(&mut w_v).arg(&mut alpha_v)
                    .arg(&(b as i32)).arg(&(h as i32))
                    .launch(cfg)
            }.unwrap();
            
            // Add residual connection: deltanet_h = mem_fused_h + deltanet_y
            let mut d_deltanet_h_t = self.history.d_deltanet_h.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            let d_mem_fused_h_t = self.history.d_mem_fused_h.try_slice(t*bh .. (t+1)*bh).unwrap();
            let d_deltanet_y_t = self.history.d_deltanet_y.try_slice(t*bh .. (t+1)*bh).unwrap();
            
            self.stream.memcpy_dtod(&d_mem_fused_h_t, &mut d_deltanet_h_t).unwrap();
            let cfg_add = LaunchConfig { grid_dim: (((bh + 255)/256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe {
                self.stream.launch_builder(&f_add_mat).arg(&mut d_deltanet_h_t).arg(&d_deltanet_y_t).arg(&(bh as i32)).launch(cfg_add).unwrap();
            }
        }

        // PHASE 6: RMSNorm3 (on DeltaNet2 output + Residual)
        {
            let d_deltanet_h_all_ro = self.history.d_deltanet_h.try_slice(0 .. c*bh).unwrap();
            let mut d_rmsnorm3_out = self.history.d_rmsnorm3_out.try_slice_mut(0 .. c*bh).unwrap();
            let mut d_rmsnorm3_rms = self.history.d_rmsnorm3_rms.try_slice_mut(0 .. c*b).unwrap();
            
            let cfg = LaunchConfig { grid_dim: (1, (c * b) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 256 * 4 };
            unsafe {
                self.stream.launch_builder(&f_rmsnorm_fw)
                    .arg(&d_deltanet_h_all_ro).arg(&self.rmsnorm3.d_weight)
                    .arg(&mut d_rmsnorm3_out).arg(&mut d_rmsnorm3_rms)
                    .arg(&((c*b) as i32)).arg(&(h as i32)).arg(&1e-6f32)
                    .launch(cfg)
            }.unwrap();
        }

        // PHASE 7: SparseGating Forward
        for t in 0..c {
            let d_rmsnorm3_out_view = self.history.d_rmsnorm3_out.try_slice(t*bh .. (t+1)*bh).unwrap();
            let mut d_gating_out_t = self.history.d_gating_out.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            gemm_batch(&self.cublas, &self.gating.d_w1, &d_rmsnorm3_out_view, &mut d_gating_out_t, h, h, b);

            let cfg = LaunchConfig { grid_dim: (((h + 255) / 256) as u32, b as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let mut pre_act_t = self.history.d_gating_pre.try_slice_mut(t*bh .. (t+1)*bh).unwrap();

            unsafe {
                self.stream.launch_builder(&f_sg)
                    .arg(&mut d_gating_out_t).arg(&self.gating.d_b1).arg(&mut pre_act_t).arg(&(h as i32))
                    .launch(cfg)
            }.unwrap();
        }
        
        // Launch GPU-side gating energy calculation
        let energy_cfg = LaunchConfig { grid_dim: (((c * b + 255) / 256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let d_gating_pre_all = self.history.d_gating_pre.try_slice(0 .. c*bh).unwrap();
        let mut d_gate_energies_all = self.history.d_gate_energies.try_slice_mut(0 .. c*b).unwrap();
        unsafe {
            self.stream.launch_builder(&f_energy)
                .arg(&d_gating_pre_all)
                .arg(&mut d_gate_energies_all)
                .arg(&( (c * b) as i32 ))
                .arg(&( h as i32 ))
                .launch(energy_cfg)
        }.unwrap();

        // Readout & Loss (Asynchronous)
        let d_gating_out_all = self.history.d_gating_out.try_slice(0 .. c*bh).unwrap();
        let d_targets_all_ro = self.history.d_targets.try_slice(0 .. c*b).unwrap();
        
        let gpu_readout = pipeline.gpu_readout.as_mut().unwrap();
        gpu_readout.forward_and_loss_vram(&d_gating_out_all, &d_targets_all_ro, c * b);
        
        // =========================================================================
        // BACKWARD PASS
        // =========================================================================
        
        // 1. Readout Backward
        let d_hidden_grad_vram = gpu_readout.backward_vram(c * b);
        let mut d_grad_h_all = self.history.d_grad_h.try_slice_mut(0 .. c*bh).unwrap();
        self.stream.memcpy_dtod(&d_hidden_grad_vram, &mut d_grad_h_all).unwrap();

        // 2. SparseGating Backward
        let cfg = LaunchConfig { grid_dim: (((h + 255) / 256) as u32, b as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        for t in (0..c).rev() {
            let grad_h_t = self.history.d_grad_h.try_slice(t*bh .. (t+1)*bh).unwrap();
            let pre_act_t = self.history.d_gating_pre.try_slice(t*bh .. (t+1)*bh).unwrap();
            let mut d_fused_grad_t = self.history.d_grad_rmsnorm3_out.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            
            unsafe {
                self.stream.launch_builder(&f_sg_bw)
                    .arg(&grad_h_t).arg(&pre_act_t).arg(&mut d_fused_grad_t).arg(&self.gating.d_b1_grad).arg(&(h as i32))
                    .launch(cfg)
            }.unwrap();
            
            let d_rmsnorm3_out_view = self.history.d_rmsnorm3_out.try_slice(t*bh .. (t+1)*bh).unwrap();
            let d_fused_grad_t_ro = self.history.d_grad_rmsnorm3_out.try_slice(t*bh .. (t+1)*bh).unwrap();
            gemm_backward_w(&self.cublas, &d_rmsnorm3_out_view, &d_fused_grad_t_ro, &mut self.gating.d_w1_grad, h, h, b);
        }

        // 3. RMSNorm3 Backward (Output to d_grad_deltanet_h)
        {
            let d_deltanet_h_all_ro = self.history.d_deltanet_h.try_slice(0 .. c*bh).unwrap();
            let d_grad_rmsnorm3_out_ro = self.history.d_grad_rmsnorm3_out.try_slice(0 .. c*bh).unwrap();
            let d_rms3_ro = self.history.d_rmsnorm3_rms.try_slice(0 .. c*b).unwrap();
            let mut d_grad_deltanet_h_all = self.history.d_grad_deltanet_h.try_slice_mut(0 .. c*bh).unwrap();
            
            let norm_cfg = LaunchConfig { grid_dim: (1, (c * b) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 256 * 4 };
            unsafe {
                self.stream.launch_builder(&f_rmsnorm_bw)
                    .arg(&d_deltanet_h_all_ro).arg(&self.rmsnorm3.d_weight).arg(&d_grad_rmsnorm3_out_ro)
                    .arg(&d_rms3_ro).arg(&mut d_grad_deltanet_h_all).arg(&self.rmsnorm3.d_weight_grad)
                    .arg(&((c*b) as i32)).arg(&(h as i32))
                    .launch(norm_cfg)
            }.unwrap();
        }

        // 4. Gated DeltaNet-2 Backward
        // grad_y = d_grad_deltanet_h
        // gradient on mem_fused_h (input to layer) is d_grad_mem_fused_h. It inherits d_grad_deltanet_h (residual connection).
        self.stream.memcpy_dtod(&self.history.d_grad_deltanet_h, &mut self.history.d_grad_mem_fused_h).unwrap();
        
        // Zero state gradient next
        zero(&mut self.history.d_dS_next, state_size);
        
        let deltanet_bw_cfg = LaunchConfig { grid_dim: (((h + 255) / 256) as u32, b as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        for t in (0..c).rev() {
            let dy_t = self.history.d_grad_deltanet_h.try_slice(t*bh .. (t+1)*bh).unwrap();
            let S_prev_t = self.history.d_deltanet_states.try_slice(t*state_size .. (t+1)*state_size).unwrap();
            
            let k_t = self.history.d_deltanet_k.try_slice(t*bh .. (t+1)*bh).unwrap();
            let v_t = self.history.d_deltanet_v.try_slice(t*bh .. (t+1)*bh).unwrap();
            let q_t = self.history.d_deltanet_q.try_slice(t*bh .. (t+1)*bh).unwrap();
            let b_t = self.history.d_deltanet_b.try_slice(t*bh .. (t+1)*bh).unwrap();
            let w_t = self.history.d_deltanet_w.try_slice(t*bh .. (t+1)*bh).unwrap();
            let a_t = self.history.d_deltanet_alpha.try_slice(t*bh .. (t+1)*bh).unwrap();
            
            let mut dq_t = self.history.d_d_q.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            let mut dv_t = self.history.d_d_v.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            let mut dw_pre_t = self.history.d_d_w_pre.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            
            let mut d_G_temp = self.history.d_G_temp.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            let mut d_R_temp = self.history.d_R_temp.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            
            unsafe {
                self.stream.launch_builder(&f_deltanet_bw1)
                    .arg(&dy_t).arg(&S_prev_t).arg(&k_t).arg(&v_t).arg(&q_t)
                    .arg(&b_t).arg(&w_t).arg(&a_t).arg(&self.history.d_dS_next)
                    .arg(&mut dq_t).arg(&mut dv_t).arg(&mut dw_pre_t)
                    .arg(&mut d_G_temp).arg(&mut d_R_temp)
                    .arg(&(b as i32)).arg(&(h as i32))
                    .launch(deltanet_bw_cfg)
            }.unwrap();
            
            let G_temp_t = self.history.d_G_temp.try_slice(t*bh .. (t+1)*bh).unwrap();
            let R_temp_t = self.history.d_R_temp.try_slice(t*bh .. (t+1)*bh).unwrap();
            
            let dS_next_ptr = {
                let view = self.history.d_dS_next.try_slice(0 .. state_size).unwrap();
                view.device_ptr(&self.stream).0
            };
            
            let mut dk_t = self.history.d_d_k.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            let mut db_pre_t = self.history.d_d_b_pre.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            let mut dalpha_pre_t = self.history.d_d_alpha_pre.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            
            unsafe {
                self.stream.launch_builder(&f_deltanet_bw2)
                    .arg(&dy_t).arg(&S_prev_t).arg(&k_t).arg(&q_t).arg(&b_t).arg(&a_t)
                    .arg(&dS_next_ptr).arg(&G_temp_t).arg(&R_temp_t)
                    .arg(&dS_next_ptr).arg(&mut dk_t).arg(&mut db_pre_t).arg(&mut dalpha_pre_t)
                    .arg(&(b as i32)).arg(&(h as i32))
                    .launch(deltanet_bw_cfg)
            }.unwrap();
        }
        
        // DeltaNet weight gradients
        let d_rmsnorm2_out_ro = self.history.d_rmsnorm2_out.try_slice(0 .. c*bh).unwrap();
        let d_dk_all = self.history.d_d_k.try_slice(0 .. c*bh).unwrap();
        gemm_backward_w(&self.cublas, &d_rmsnorm2_out_ro, &d_dk_all, &mut self.deltanet2.d_w_k_grad, h, h, c * b);
        
        let d_dv_all = self.history.d_d_v.try_slice(0 .. c*bh).unwrap();
        gemm_backward_w(&self.cublas, &d_rmsnorm2_out_ro, &d_dv_all, &mut self.deltanet2.d_w_v_grad, h, h, c * b);
        
        let d_dq_all = self.history.d_d_q.try_slice(0 .. c*bh).unwrap();
        gemm_backward_w(&self.cublas, &d_rmsnorm2_out_ro, &d_dq_all, &mut self.deltanet2.d_w_q_grad, h, h, c * b);
        
        let d_db_pre_all = self.history.d_d_b_pre.try_slice(0 .. c*bh).unwrap();
        gemm_backward_w(&self.cublas, &d_rmsnorm2_out_ro, &d_db_pre_all, &mut self.deltanet2.d_w_b_grad, h, h, c * b);
        
        let d_dw_pre_all = self.history.d_d_w_pre.try_slice(0 .. c*bh).unwrap();
        gemm_backward_w(&self.cublas, &d_rmsnorm2_out_ro, &d_dw_pre_all, &mut self.deltanet2.d_w_w_grad, h, h, c * b);
        
        let d_dalpha_pre_all = self.history.d_d_alpha_pre.try_slice(0 .. c*bh).unwrap();
        gemm_backward_w(&self.cublas, &d_rmsnorm2_out_ro, &d_dalpha_pre_all, &mut self.deltanet2.d_w_alpha_grad, h, h, c * b);
        
        // Gradient backpropagated to input of DeltaNet2 (d_rmsnorm2_out)
        let mut d_grad_rmsnorm2_out_all = self.history.d_grad_rmsnorm2_out.try_slice_mut(0 .. c*bh).unwrap();
        gemm_backward_x_accum(&self.cublas, &self.deltanet2.d_w_k, &d_dk_all, &mut d_grad_rmsnorm2_out_all, h, h, c * b, 0.0);
        gemm_backward_x_accum(&self.cublas, &self.deltanet2.d_w_v, &d_dv_all, &mut d_grad_rmsnorm2_out_all, h, h, c * b, 1.0);
        gemm_backward_x_accum(&self.cublas, &self.deltanet2.d_w_q, &d_dq_all, &mut d_grad_rmsnorm2_out_all, h, h, c * b, 1.0);
        gemm_backward_x_accum(&self.cublas, &self.deltanet2.d_w_b, &d_db_pre_all, &mut d_grad_rmsnorm2_out_all, h, h, c * b, 1.0);
        gemm_backward_x_accum(&self.cublas, &self.deltanet2.d_w_w, &d_dw_pre_all, &mut d_grad_rmsnorm2_out_all, h, h, c * b, 1.0);
        gemm_backward_x_accum(&self.cublas, &self.deltanet2.d_w_alpha, &d_dalpha_pre_all, &mut d_grad_rmsnorm2_out_all, h, h, c * b, 1.0);

        // 5. RMSNorm2 Backward
        {
            let d_mem_fused_h_all_ro = self.history.d_mem_fused_h.try_slice(0 .. c*bh).unwrap();
            let d_grad_rmsnorm2_out_ro = self.history.d_grad_rmsnorm2_out.try_slice(0 .. c*bh).unwrap();
            let d_rms2_ro = self.history.d_rmsnorm2_rms.try_slice(0 .. c*b).unwrap();
            let mut d_grad_mem_fused_h_all = self.history.d_grad_mem_fused_h.try_slice_mut(0 .. c*bh).unwrap();
            
            let norm_cfg = LaunchConfig { grid_dim: (1, (c * b) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 256 * 4 };
            unsafe {
                self.stream.launch_builder(&f_rmsnorm_bw)
                    .arg(&d_mem_fused_h_all_ro).arg(&self.rmsnorm2.d_weight).arg(&d_grad_rmsnorm2_out_ro)
                    .arg(&d_rms2_ro).arg(&mut d_grad_mem_fused_h_all).arg(&self.rmsnorm2.d_weight_grad)
                    .arg(&((c*b) as i32)).arg(&(h as i32))
                    .launch(norm_cfg)
            }.unwrap();
        }

        // 6. MemoryAttention Backward
        // Fused path: d_grad_mem_fused_h has accumulated d_grad_deltanet_h.
        // We propagate d_grad_mem_fused_h through memory attention.
        let d_grad_mem_fused_h_ro = self.history.d_grad_mem_fused_h.try_slice(0 .. c*bh).unwrap();
        let d_mem_out_all_ro = self.memory_attention.d_mem_out.try_slice(0 .. c*bh).unwrap();
        
        let bias_bw_cfg = LaunchConfig { grid_dim: (((h + 255) / 256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&f_bias_bw)
                .arg(&d_grad_mem_fused_h_ro)
                .arg(&mut self.memory_attention.d_b_fuse_grad)
                .arg(&( (c * b) as i32 ))
                .arg(&( h as i32 ))
                .launch(bias_bw_cfg)
        }.unwrap();
        
        let mut d_mem_out_grad_all = self.memory_attention.d_mem_out_grad.try_slice_mut(0 .. c*bh).unwrap();
        let mut d_dz_fuse = self.memory_attention.d_lora_d_z_fuse.try_slice_mut(0 .. c*b * self.memory_attention.w_fuse.rank).unwrap();
        let d_temp_rank_fuse_ro = self.memory_attention.d_lora_temp_rank_fuse.try_slice(0 .. c*b * self.memory_attention.w_fuse.rank).unwrap();
        
        self.memory_attention.w_fuse.backward(
            &self.cublas,
            &d_mem_out_all_ro,
            &d_grad_mem_fused_h_ro,
            &mut d_mem_out_grad_all,
            &d_temp_rank_fuse_ro,
            &mut d_dz_fuse,
            c * b,
            0.0,
        );
        
        // Now backprop to queries in VRAM
        let _d_mem_out_grad_all_ro = self.memory_attention.d_mem_out_grad.try_slice(0 .. c*bh).unwrap();
        let mut d_q_in_grad_all = self.memory_attention.d_q_in_grad.try_slice_mut(0 .. c*b*e).unwrap();
        zero_view(&mut d_q_in_grad_all, c * b * e);
        
        if current_size > 0 {
            // Memory Attention backward lookup
            let _gpu_mem = pipeline.gpu_memory.as_ref().unwrap();
            let _d_scores_all_ro = self.memory_attention.d_scores.try_slice(0 .. c * b * current_size).unwrap();
            
            // grad_scores = d_mem_out_grad * V^T
            // cfg: CUBLAS_OP_T, CUBLAS_OP_N, m = current_size, n = c*b, k = h
            // But we can simplify: since memory lookup is detached during BPTT (as specified in "Detached Lookup" comment),
            // we don't backpropagate through key-value scores to keys/values themselves.
            // We only backpropagate through query projection:
            // dq_in_grad += d_query_grad * (1 - query^2)
            // Wait, query_grad = grad_scores * K
            let _lookup_bw_cfg = LaunchConfig {
                grid_dim: (((e + 255) / 256) as u32, (c * b) as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            
            // For now, let's do: d_q_in_grad = d_query_grad
            // W_q_grad = d_q_in_grad^T * d_rmsnorm1_out
            // b_q_grad = sum(d_q_in_grad)
        }
        
        // W_q_grad and b_q_grad
        let d_q_in_grad_ro = self.memory_attention.d_q_in_grad.try_slice(0 .. c*b*e).unwrap();
        unsafe {
            self.stream.launch_builder(&f_bias_bw)
                .arg(&d_q_in_grad_ro)
                .arg(&mut self.memory_attention.d_b_q_grad)
                .arg(&( (c * b) as i32 ))
                .arg(&( e as i32 ))
                .launch(LaunchConfig { grid_dim: (((e + 255) / 256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
        }.unwrap();
        
        let mut d_grad_rmsnorm1_out_all = self.history.d_grad_rmsnorm1_out.try_slice_mut(0 .. c*bh).unwrap();
        let mut d_dz_q = self.memory_attention.d_lora_d_z_q.try_slice_mut(0 .. c*b * self.memory_attention.w_q.rank).unwrap();
        let d_temp_rank_q_ro = self.memory_attention.d_lora_temp_rank_q.try_slice(0 .. c*b * self.memory_attention.w_q.rank).unwrap();
        
        self.memory_attention.w_q.backward(
            &self.cublas,
            &d_rmsnorm1_out_all,
            &d_q_in_grad_ro,
            &mut d_grad_rmsnorm1_out_all,
            &d_temp_rank_q_ro,
            &mut d_dz_q,
            c * b,
            0.0,
        );
        
        // 7. RMSNorm1 Backward (Output to d_grad_mingru_h)
        // Note: the residual path of MemoryAttention also adds gradient directly to d_grad_mingru_h!
        // We first run RMSNorm1 backward to get the gradient with respect to minGRU outputs:
        let mut d_grad_mingru_h_all = self.history.d_grad_mingru_h.try_slice_mut(bh .. (c+1)*bh).unwrap(); // step 1..c+1
        {
            let d_mingru_h_out = self.history.d_mingru_h.try_slice(bh .. (c+1)*bh).unwrap();
            let d_grad_rmsnorm1_out_ro = self.history.d_grad_rmsnorm1_out.try_slice(0 .. c*bh).unwrap();
            let d_rms1_ro = self.history.d_rmsnorm1_rms.try_slice(0 .. c*b).unwrap();
            
            let norm_cfg = LaunchConfig { grid_dim: (1, (c * b) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 256 * 4 };
            unsafe {
                self.stream.launch_builder(&f_rmsnorm_bw)
                    .arg(&d_mingru_h_out).arg(&self.rmsnorm1.d_weight).arg(&d_grad_rmsnorm1_out_ro)
                    .arg(&d_rms1_ro).arg(&mut d_grad_mingru_h_all).arg(&self.rmsnorm1.d_weight_grad)
                    .arg(&((c*b) as i32)).arg(&(h as i32))
                    .launch(norm_cfg)
            }.unwrap();
        }
        
        // Add the residual gradient from MemoryAttention (which is d_grad_mem_fused_h)
        unsafe {
            self.stream.launch_builder(&f_add_mat)
                .arg(&mut d_grad_mingru_h_all)
                .arg(&d_grad_mem_fused_h_ro)
                .arg(&( (c * bh) as i32 ))
                .launch(LaunchConfig { grid_dim: (((c * bh + 255)/256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
        }.unwrap();

        // 8. minGRU Recurrence Backward
        zero(&mut self.history.d_dh_next, bh);
        
        let mingru_bw_cfg = LaunchConfig { grid_dim: (((h + 255) / 256) as u32, b as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        for t in (0..c).rev() {
            let mut grad_h_t = self.history.d_grad_mingru_h.try_slice_mut((t+1)*bh .. (t+2)*bh).unwrap();
            // Add d_dh_next
            if t < c - 1 {
                unsafe {
                    self.stream.launch_builder(&f_add_mat)
                        .arg(&mut grad_h_t).arg(&self.history.d_dh_next)
                        .arg(&(bh as i32))
                        .launch(LaunchConfig { grid_dim: (((bh + 255)/256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                }.unwrap();
            }
            
            let grad_h_t_ro = self.history.d_grad_mingru_h.try_slice((t+1)*bh .. (t+2)*bh).unwrap();
            let z_t = self.history.d_mingru_z.try_slice(t*bh .. (t+1)*bh).unwrap();
            let prev_h_t = self.history.d_mingru_h.try_slice(t*bh .. (t+1)*bh).unwrap();
            let h_tilde_t = self.history.d_mingru_h_tilde.try_slice(t*bh .. (t+1)*bh).unwrap();
            
            let mut grad_prev_h_t = self.history.d_dh_next.try_slice_mut(0 .. bh).unwrap();
            let mut d_z_pre_t = self.history.d_d_z_pre.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            let mut d_h_pre_t = self.history.d_d_h_pre.try_slice_mut(t*bh .. (t+1)*bh).unwrap();
            
            unsafe {
                self.stream.launch_builder(&f_mingru_bw)
                    .arg(&grad_h_t_ro).arg(&z_t).arg(&prev_h_t).arg(&h_tilde_t)
                    .arg(&mut grad_prev_h_t).arg(&mut d_z_pre_t).arg(&mut d_h_pre_t)
                    .arg(&self.mingru.d_b_z_grad).arg(&self.mingru.d_b_h_grad)
                    .arg(&(b as i32)).arg(&(h as i32))
                    .launch(mingru_bw_cfg)
            }.unwrap();
        }
        
        // minGRU weight gradients
        let d_x_all_ro = self.history.d_x.try_slice(0 .. c*b*e).unwrap();
        let d_d_z_pre_all = self.history.d_d_z_pre.try_slice(0 .. c*bh).unwrap();
        gemm_backward_w(&self.cublas, &d_x_all_ro, &d_d_z_pre_all, &mut self.mingru.d_w_z_grad, h, e, c * b);
        
        let d_d_h_pre_all = self.history.d_d_h_pre.try_slice(0 .. c*bh).unwrap();
        gemm_backward_w(&self.cublas, &d_x_all_ro, &d_d_h_pre_all, &mut self.mingru.d_w_h_grad, h, e, c * b);
        
        // Gradient with respect to input (Embedding)
        let mut d_grad_x_all = self.history.d_grad_x.try_slice_mut(0 .. c*be).unwrap();
        gemm_backward_x_accum(&self.cublas, &self.mingru.d_w_z, &d_d_z_pre_all, &mut d_grad_x_all, h, e, c * b, 0.0);
        gemm_backward_x_accum(&self.cublas, &self.mingru.d_w_h, &d_d_h_pre_all, &mut d_grad_x_all, h, e, c * b, 1.0);
        
        // 9. Embedding Backward
        let emb_bw_cfg = LaunchConfig { grid_dim: (((e + 255) / 256) as u32, (c * b) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let d_inputs_all_ro = self.history.d_inputs.try_slice(0 .. c*b).unwrap();
        let d_grad_x_all_ro = self.history.d_grad_x.try_slice(0 .. c*be).unwrap();
        unsafe {
            self.stream.launch_builder(&f_emb_bw)
                .arg(&d_grad_x_all_ro)
                .arg(&d_inputs_all_ro)
                .arg(&mut self.embedding.d_weight_grad)
                .arg(&((c * b) as i32))
                .arg(&(e as i32))
                .arg(&(vocab_size as i32))
                .launch(emb_bw_cfg)
        }.unwrap();

        // PHASE 11: Lion Optimizer Step
        let f_lion = self.module.load_function("lion_step_kernel").unwrap();

        let step = |w: &mut CudaSlice<f32>, g: &mut CudaSlice<f32>, m: &mut CudaSlice<f32>, size: usize| {
            let cfg = LaunchConfig { grid_dim: (((size + 255)/256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { 
                self.stream.launch_builder(&f_lion)
                    .arg(w).arg(g).arg(m)
                    .arg(&lr).arg(&beta1).arg(&beta2).arg(&weight_decay).arg(&(size as i32))
                    .launch(cfg) 
            }.unwrap();
        };

        step(&mut self.mingru.d_w_z, &mut self.mingru.d_w_z_grad, &mut self.mingru.m_w_z, h * e);
        step(&mut self.mingru.d_b_z, &mut self.mingru.d_b_z_grad, &mut self.mingru.m_b_z, h);
        step(&mut self.mingru.d_w_h, &mut self.mingru.d_w_h_grad, &mut self.mingru.m_w_h, h * e);
        step(&mut self.mingru.d_b_h, &mut self.mingru.d_b_h_grad, &mut self.mingru.m_b_h, h);
        
        step(&mut self.rmsnorm1.d_weight, &mut self.rmsnorm1.d_weight_grad, &mut self.rmsnorm1.m_weight, h);
        step(&mut self.rmsnorm2.d_weight, &mut self.rmsnorm2.d_weight_grad, &mut self.rmsnorm2.m_weight, h);
        step(&mut self.rmsnorm3.d_weight, &mut self.rmsnorm3.d_weight_grad, &mut self.rmsnorm3.m_weight, h);
        
        step(&mut self.deltanet2.d_w_k, &mut self.deltanet2.d_w_k_grad, &mut self.deltanet2.m_w_k, h * h);
        step(&mut self.deltanet2.d_w_v, &mut self.deltanet2.d_w_v_grad, &mut self.deltanet2.m_w_v, h * h);
        step(&mut self.deltanet2.d_w_q, &mut self.deltanet2.d_w_q_grad, &mut self.deltanet2.m_w_q, h * h);
        step(&mut self.deltanet2.d_w_b, &mut self.deltanet2.d_w_b_grad, &mut self.deltanet2.m_w_b, h * h);
        step(&mut self.deltanet2.d_w_w, &mut self.deltanet2.d_w_w_grad, &mut self.deltanet2.m_w_w, h * h);
        step(&mut self.deltanet2.d_w_alpha, &mut self.deltanet2.d_w_alpha_grad, &mut self.deltanet2.m_w_alpha, h * h);
        
        step(&mut self.gating.d_w1, &mut self.gating.d_w1_grad, &mut self.gating.m_w1, h * h);
        step(&mut self.gating.d_b1, &mut self.gating.d_b1_grad, &mut self.gating.m_b1, h);
        
        step(&mut self.embedding.d_weight, &mut self.embedding.d_weight_grad, &mut self.embedding.m_weight, vocab_size * e);
        self.memory_attention.w_fuse.step(&self.stream, &f_lion, lr, beta1, beta2, weight_decay);
        step(&mut self.memory_attention.b_fuse, &mut self.memory_attention.d_b_fuse_grad, &mut self.memory_attention.m_b_fuse, h);
        self.memory_attention.w_q.step(&self.stream, &f_lion, lr, beta1, beta2, weight_decay);
        step(&mut self.memory_attention.b_q, &mut self.memory_attention.d_b_q_grad, &mut self.memory_attention.m_b_q, e);

        // =========================================================================
        // POSTPONED DOWNLOADS & CPU MEMORY WRITES
        // =========================================================================
        
        // 1. Loss Download
        let d_losses_sub = gpu_readout.d_losses.try_slice(0..c*b).unwrap();
        self.stream.memcpy_dtoh(&d_losses_sub, &mut self.history.h_losses[..c*b]).unwrap();
        *chunk_loss += self.history.h_losses[..c*b].iter().sum::<f32>();

        // 2. Gating Energy Download
        let d_gate_energies_sub = self.history.d_gate_energies.try_slice(0..c*b).unwrap();
        self.stream.memcpy_dtoh(&d_gate_energies_sub, &mut self.history.h_gate_energies_cpu[..c*b]).unwrap();

        // 3. minGRU hidden states download
        self.stream.memcpy_dtoh(&self.history.d_mingru_h.try_slice(bh .. (c+1)*bh).unwrap(), &mut self.history.h_rnn_h_cpu[..c*bh]).unwrap();

        // 4. Final states download (for carryover to next chunk)
        let final_h = self.history.d_mingru_h.try_slice(c * bh .. (c + 1) * bh).unwrap();
        self.stream.memcpy_dtoh(&final_h, &mut pipeline.mingru.hidden_state.data.storage[..]).unwrap();

        let final_s = self.history.d_deltanet_states.try_slice(c * state_size .. (c+1) * state_size).unwrap();
        self.stream.memcpy_dtoh(&final_s, &mut pipeline.deltanet2.state[..]).unwrap();

        // 5. Build memory writes on CPU
        let mut memory_writes = Vec::new();
        let mut local_current_size = pipeline.base_memory.current_size + pipeline.user_memory.current_size;
        
        for t in 0..c {
            for batch_idx in 0..b {
                let gate_energy = self.history.h_gate_energies_cpu[t * b + batch_idx];
                
                // Cross-Entropy loss for this token is exactly its surprisal: -ln(P(x))
                let token_surprise = self.history.h_losses[t * b + batch_idx];
                let delta_s = token_surprise - pipeline.surprise_mean;
                pipeline.surprise_mean += 0.01 * delta_s;
                pipeline.surprise_var = (1.0 - 0.01) * pipeline.surprise_var + 0.01 * delta_s * delta_s;
                let std_dev = pipeline.surprise_var.sqrt().max(0.1);
                let z_score = (token_surprise - pipeline.surprise_mean) / std_dev;

                let is_meaningful_surprise = z_score >= 0.8 && z_score <= 3.5;
                let should_write = (gate_energy > pipeline.consolidation_energy && is_meaningful_surprise) || local_current_size < 32;

                if should_write {
                    let mut key = crate::ant_core::tensor::Tensor1D::new(e);
                    let mut val = crate::ant_core::tensor::Tensor1D::new(h);
                    
                    let emb_offset = (t * b * e) + (batch_idx * e);
                    key.data.copy_from_slice(&self.history.h_full_emb_cpu[emb_offset .. emb_offset + e]);
                    
                    let offset = (t * b * h) + (batch_idx * h);
                    val.data.copy_from_slice(&self.history.h_rnn_h_cpu[offset .. offset + h]);
                    
                    memory_writes.push((key, val));
                    local_current_size += 1;
                }
            }
        }
        
        if let Some(ref mut gpu_mem) = pipeline.gpu_memory {
            let mut keys_flat = Vec::with_capacity(memory_writes.len() * e);
            let mut vals_flat = Vec::with_capacity(memory_writes.len() * h);
            for (k, v) in &memory_writes {
                keys_flat.extend_from_slice(&k.data);
                vals_flat.extend_from_slice(&v.data);
            }
            if !memory_writes.is_empty() {
                gpu_mem.async_add_memory(&keys_flat, &vals_flat, memory_writes.len());
            }
        }
        
        let mut query_polarity = 0u64;
        for t in 0..c {
            for batch_idx in 0..b {
                let token_id = inputs[t][batch_idx];
                if pipeline.negation_ids.contains(&token_id) {
                    query_polarity = 1;
                    break;
                }
            }
        }
        
        for (k, v) in memory_writes {
            pipeline.base_memory.add_memory(k, v, query_polarity);
        }
    }
}
