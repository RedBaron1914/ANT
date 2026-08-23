use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, CudaSlice, LaunchConfig, CudaModule, PushKernelArg};
use crate::ant_core::tensor::Tensor1D;

pub struct GpuKVMemory {
    pub capacity: usize,
    pub key_dim: usize,
    pub val_dim: usize,
    pub base_size: usize,
    pub user_size: usize,
    pub user_write_cursor: usize,
    pub base_capacity: usize,
    pub user_capacity: usize,
    
    _ctx: Arc<CudaContext>,
    pub stream_lookup: Arc<CudaStream>,
    pub stream_ingest: Arc<CudaStream>,
    pub module: Arc<CudaModule>,
    
    // GPU buffers
    pub d_keys: CudaSlice<f32>,
    pub d_vals: CudaSlice<f32>,
}

impl GpuKVMemory {
    pub fn new(base_capacity: usize, user_capacity: usize, key_dim: usize, val_dim: usize) -> Self {
        let ctx = CudaContext::new(0).expect("No CUDA device found");
        let stream_lookup = ctx.new_stream().unwrap();
        let stream_ingest = ctx.new_stream().unwrap();
        
        let ptx_content = include_str!("kernel.cu");
        let ptx = cudarc::nvrtc::compile_ptx(ptx_content).expect("NVRTC failed to compile PTX");
        let module = ctx.load_module(ptx).expect("Failed to load PTX module");
        
        let total_capacity = base_capacity + user_capacity;
        let d_keys = stream_ingest.alloc_zeros::<f32>(total_capacity * key_dim).unwrap();
        let d_vals = stream_ingest.alloc_zeros::<f32>(total_capacity * val_dim).unwrap();
        
        Self {
            capacity: total_capacity,
            key_dim,
            val_dim,
            base_size: 0,
            user_size: 0,
            user_write_cursor: 0,
            base_capacity,
            user_capacity,
            _ctx: ctx,
            stream_lookup,
            stream_ingest,
            module,
            d_keys,
            d_vals,
        }
    }

    pub fn load_dual_memories(
        &mut self,
        base_keys: &[f32],
        base_vals: &[f32],
        base_size: usize,
        user_keys: &[f32],
        user_vals: &[f32],
        user_size: usize,
        user_write_cursor: usize,
    ) {
        // Copy base keys and values
        if base_size > 0 {
            let mut d_k = self.d_keys.try_slice_mut(0..base_size * self.key_dim).unwrap();
            self.stream_ingest.memcpy_htod(&base_keys[..base_size * self.key_dim], &mut d_k).unwrap();
            
            let mut d_v = self.d_vals.try_slice_mut(0..base_size * self.val_dim).unwrap();
            self.stream_ingest.memcpy_htod(&base_vals[..base_size * self.val_dim], &mut d_v).unwrap();
        }

        // Copy user keys and values contiguously starting right after base keys
        if user_size > 0 {
            let start_key = base_size * self.key_dim;
            let end_key = start_key + user_size * self.key_dim;
            let mut d_k = self.d_keys.try_slice_mut(start_key..end_key).unwrap();
            self.stream_ingest.memcpy_htod(&user_keys[..user_size * self.key_dim], &mut d_k).unwrap();
            
            let start_val = base_size * self.val_dim;
            let end_val = start_val + user_size * self.val_dim;
            let mut d_v = self.d_vals.try_slice_mut(start_val..end_val).unwrap();
            self.stream_ingest.memcpy_htod(&user_vals[..user_size * self.val_dim], &mut d_v).unwrap();
        }

        self.base_size = base_size;
        self.user_size = user_size;
        self.user_write_cursor = user_write_cursor;
    }

    pub fn async_add_memory(&mut self, keys_flat: &[f32], vals_flat: &[f32], count: usize) {
        if count == 0 { return; }
        let copy_count = std::cmp::min(count, self.user_capacity);
        
        for item in 0..copy_count {
            let user_cursor = self.user_write_cursor;
            // Write into user slot (offset by base_size)
            let cursor = self.base_size + user_cursor;
            let key_offset = cursor * self.key_dim;
            let val_offset = cursor * self.val_dim;
            
            let key_end = (item + 1) * self.key_dim;
            let val_end = (item + 1) * self.val_dim;
            if key_end > keys_flat.len() || val_end > vals_flat.len() { break; }

            let key_slice = &keys_flat[item * self.key_dim .. key_end];
            let val_slice = &vals_flat[item * self.val_dim .. val_end];
            
            if let Some(mut d_keys_sub) = self.d_keys.try_slice_mut(key_offset..key_offset + self.key_dim) {
                let _ = self.stream_ingest.memcpy_htod(key_slice, &mut d_keys_sub);
            }
            if let Some(mut d_vals_sub) = self.d_vals.try_slice_mut(val_offset..val_offset + self.val_dim) {
                let _ = self.stream_ingest.memcpy_htod(val_slice, &mut d_vals_sub);
            }
            
            self.user_write_cursor = (self.user_write_cursor + 1) % self.user_capacity;
            if self.user_size < self.user_capacity {
                self.user_size += 1;
            }
        }
    }

    pub fn lookup(&self, query: &Tensor1D, top_k: usize) -> Vec<(usize, f32)> {
        let size = self.base_size + self.user_size;
        if size == 0 {
            return vec![];
        }

        let mut q_norm = query.clone();
        let mut norm_sq = 0.0;
        for val in q_norm.data.iter() { norm_sq += val * val; }
        let norm = norm_sq.sqrt();
        if norm > 1e-8 {
            for val in q_norm.data.iter_mut() { *val /= norm; }
        }

        let d_query = self.stream_lookup.clone_htod(&q_norm.data).unwrap();
        let mut d_scores = self.stream_lookup.alloc_zeros::<f32>(size).unwrap();
        
        let block_size = 256;
        let grid_size = (size as u32 + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        
        let f = self.module.load_function("memory_lookup").unwrap();
        
        let current_size_i32 = size as i32;
        let key_dim_i32 = self.key_dim as i32;
        
        unsafe {
            self.stream_lookup.launch_builder(&f)
                .arg(&d_query)
                .arg(&self.d_keys)
                .arg(&mut d_scores)
                .arg(&current_size_i32)
                .arg(&key_dim_i32)
                .launch(cfg)
        }.unwrap();
        
        let host_scores = self.stream_lookup.clone_dtoh(&d_scores).unwrap();
        
        let mut scores: Vec<(usize, f32)> = host_scores.into_iter().enumerate().collect();
        let actual_k = std::cmp::min(top_k, scores.len());
        
        if actual_k < scores.len() {
            scores.select_nth_unstable_by(actual_k, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        
        let top_scores = &mut scores[0..actual_k];
        top_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        top_scores.to_vec()
    }
}
