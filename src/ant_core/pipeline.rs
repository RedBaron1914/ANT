use std::path::Path;
use std::collections::HashSet;
use super::tensor::{BatchTensor, Tensor1D, MatExt};
use super::embedding::Embedding;
use super::rnn::{MinGRU, MinGruScratchpad, GatedDeltaNet2, GatedDeltaNet2Scratchpad};
use super::rmsnorm::RMSNorm;
use super::sparse_gating::{SparseGating, SparseGatingScratchpad};
use super::memory_io::DiskKVMemory;
use super::memory_attention::{MemoryAttention, MemoryAttentionCache};
use super::readout::{ReadoutLayer, ReadoutScratchpad};
use super::session_tape::{SessionTape, LocalFifoBuffer};

pub struct AntPipeline {
    pub embedding: Embedding,
    pub mingru: MinGRU,
    pub rmsnorm1: RMSNorm,
    pub memory_attention: MemoryAttention,
    pub rmsnorm2: RMSNorm,
    pub deltanet2: GatedDeltaNet2,
    pub rmsnorm3: RMSNorm,
    pub gating: SparseGating,
    pub base_memory: DiskKVMemory,
    pub user_memory: DiskKVMemory,
    pub gpu_memory: Option<crate::compute_gpu::gpu_memory::GpuKVMemory>,
    pub gpu_readout: Option<crate::compute_gpu::gpu_readout::GpuReadout>,
    pub readout: ReadoutLayer,
    pub consolidation_energy: f32,
    
    // Alex Graves' ACT Halting Head
    pub w_halt: Tensor1D,
    pub b_halt: f32,

    // Dynamic Z-Score Bayesian Surprisal Filter (EMA)
    pub surprise_mean: f32,
    pub surprise_var: f32,

    // Modular .antpack skill cartridges
    pub mounted_packs: Vec<DiskKVMemory>,

    // Tape & Buffers
    pub session_tape: SessionTape,
    pub fifo_buffer: LocalFifoBuffer,
    pub negation_ids: HashSet<usize>,
    pub is_training: bool,
    
    // Scratchpads to avoid allocations
    pub mingru_scratch: MinGruScratchpad,
    pub deltanet2_scratch: GatedDeltaNet2Scratchpad,
    pub gating_scratch: SparseGatingScratchpad,
    pub memory_attn_cache: MemoryAttentionCache,
    pub readout_scratch: ReadoutScratchpad,
    pub logits: BatchTensor,
}

pub fn get_negation_ids(tokenizer: &tokenizers::Tokenizer) -> HashSet<usize> {
    let mut ids = HashSet::new();
    let negation_words = ["not", "no", "never", "не", "нет", "без"];
    for &word in &negation_words {
        if let Some(id) = tokenizer.token_to_id(word) {
            ids.insert(id as usize);
        }
        let prefix_word = format!("Ġ{}", word);
        if let Some(id) = tokenizer.token_to_id(&prefix_word) {
            ids.insert(id as usize);
        }
    }
    ids
}

impl AntPipeline {
    pub fn new(
        base_mem_path: &str,
        user_mem_path: &str,
        vocab_size: usize,
        embed_dim: usize, 
        hidden_size: usize, 
        batch_size: usize, 
        base_capacity: usize,
        user_capacity: usize,
        top_k_base: usize,
        top_k_user: usize,
        consolidation_energy: f32,
        session_tape_capacity: usize,
        fifo_window_capacity: usize,
        lora_rank: usize,
        lora_alpha: f32,
        force_cpu: bool,
    ) -> std::io::Result<Self> {
        let base_memory = DiskKVMemory::new(base_mem_path, base_capacity, embed_dim, hidden_size)?;
        let user_memory = DiskKVMemory::new(user_mem_path, user_capacity, embed_dim, hidden_size)?;
        
        let embedding = Embedding::new(vocab_size, embed_dim);
        let mut mingru = MinGRU::new(batch_size, embed_dim, hidden_size);
        mingru.randomize(-0.1, 0.1);
        
        let rmsnorm1 = RMSNorm::new(hidden_size);
        let rmsnorm2 = RMSNorm::new(hidden_size);
        
        let mut deltanet2 = GatedDeltaNet2::new(batch_size, hidden_size);
        deltanet2.randomize(-0.1, 0.1);
        
        let rmsnorm3 = RMSNorm::new(hidden_size);
        
        let mut gating = SparseGating::new(batch_size, hidden_size, hidden_size);
        gating.randomize(-0.1, 0.1);

        let (gpu_memory, gpu_readout) = if force_cpu {
            (None, None)
        } else {
            match std::panic::catch_unwind(|| {
                let mut gm = crate::compute_gpu::gpu_memory::GpuKVMemory::new(base_capacity, user_capacity, embed_dim, hidden_size);
                
                // === SYNC DISK AND VRAM AT START ===
                let mut base_keys = Vec::new();
                let mut base_vals = Vec::new();
                if base_memory.current_size > 0 {
                    for i in 0..base_memory.capacity {
                        base_keys.extend_from_slice(base_memory.get_key(i));
                        base_vals.extend_from_slice(base_memory.get_val(i));
                    }
                }
                
                let mut user_keys = Vec::new();
                let mut user_vals = Vec::new();
                if user_memory.current_size > 0 {
                    for i in 0..user_memory.capacity {
                        user_keys.extend_from_slice(user_memory.get_key(i));
                        user_vals.extend_from_slice(user_memory.get_val(i));
                    }
                }
                
                gm.load_dual_memories(
                    &base_keys, &base_vals, base_memory.current_size,
                    &user_keys, &user_vals, user_memory.current_size,
                    user_memory.write_cursor,
                );
                // ===============================================

                let gr = crate::compute_gpu::gpu_readout::GpuReadout::new(hidden_size, embed_dim, vocab_size, batch_size * 100, lora_rank, lora_alpha);
                (gm, gr)
            }) {
                Ok((gm, gr)) => (Some(gm), Some(gr)),
                Err(_) => (None, None),
            }
        };

        let mut w_halt = Tensor1D::new(hidden_size);
        let limit_halt = (6.0 / (hidden_size + 1) as f32).sqrt();
        w_halt.randomize(-limit_halt, limit_halt);

        Ok(Self {
            embedding,
            mingru,
            rmsnorm1,
            memory_attention: MemoryAttention::new(embed_dim, hidden_size, top_k_base, top_k_user, lora_rank, lora_alpha),
            rmsnorm2,
            deltanet2,
            rmsnorm3,
            gating,
            base_memory,
            user_memory,
            gpu_memory,
            gpu_readout,
            readout: ReadoutLayer::new(hidden_size, embed_dim, vocab_size, lora_rank, lora_alpha),
            consolidation_energy,
            w_halt,
            b_halt: -1.0,
            surprise_mean: 3.5,
            surprise_var: 2.0,
            mounted_packs: Vec::new(),
            session_tape: SessionTape::new(session_tape_capacity),
            fifo_buffer: LocalFifoBuffer::new(fifo_window_capacity),
            negation_ids: HashSet::new(),
            is_training: false,
            mingru_scratch: MinGruScratchpad::new(batch_size, embed_dim, hidden_size),
            deltanet2_scratch: GatedDeltaNet2Scratchpad::new(batch_size, hidden_size, hidden_size),
            gating_scratch: SparseGatingScratchpad::new(batch_size, hidden_size, hidden_size),
            memory_attn_cache: MemoryAttentionCache::new(batch_size, embed_dim, hidden_size, top_k_base + top_k_user),
            readout_scratch: ReadoutScratchpad::new(batch_size, embed_dim, hidden_size),
            logits: BatchTensor::new(batch_size, vocab_size),
        })
    }

    /// Mounts all skill cartridges (.antpack) found in packs_dir
    pub fn load_mounted_packs(&mut self, packs_dir: &str) {
        self.mounted_packs.clear();
        let path = Path::new(packs_dir);
        if path.exists() && path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    let entry_path = entry.path();
                    if entry_path.is_file() && entry_path.extension().map_or(false, |ext| ext == "antpack") {
                        match DiskKVMemory::open_existing(&entry_path) {
                            Ok(pack) => {
                                println!("📦 Mounted skill cartridge: {:?}", entry_path.file_name().unwrap());
                                self.mounted_packs.push(pack);
                            }
                            Err(e) => {
                                eprintln!("⚠️ Failed to mount pack {:?}: {}", entry_path, e);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Forward pass through the entire ANT architecture given a batch of token IDs
    pub fn forward(&mut self, tokens: &[usize]) -> &BatchTensor {
        let b_size = tokens.len();
        
        if b_size == 1 {
            self.session_tape.append(tokens[0]);
            self.fifo_buffer.push(tokens[0]);
        }

        let emb = self.embedding.forward(tokens);
        
        self.mingru.forward_with_cache(&emb, &mut self.mingru_scratch);
        let mut h_1 = BatchTensor::new(b_size, self.deltanet2.hidden_size);
        for b in 0..b_size {
            for i in 0..self.deltanet2.hidden_size {
                h_1.data.write(b, i, self.mingru.hidden_state.data.read(b, i));
            }
        }
        
        let h_1_norm = self.rmsnorm1.forward(&h_1);
        
        // Negation detection
        let mut query_metadata = 0u64;
        for &tok in tokens {
            if self.negation_ids.contains(&tok) {
                query_metadata = 1;
            }
        }
        if query_metadata == 0 {
            let recent = self.fifo_buffer.get_tokens();
            let start_idx = recent.len().saturating_sub(5);
            for &tok in &recent[start_idx..] {
                if self.negation_ids.contains(&tok) {
                    query_metadata = 1;
                    break;
                }
            }
        }

        let h_2 = self.memory_attention.forward(&h_1_norm, &h_1, &self.base_memory, &self.user_memory, &self.mounted_packs, query_metadata, &mut self.memory_attn_cache);
        
        // Alex Graves' ACT Adaptive Deliberation & Krasnoselskii-Mann Attractor
        let mut h_thought = self.rmsnorm2.forward(&h_2);
        let max_thinking_steps = 4;
        let mut accumulated_prob = 0.0f32;

        for _step in 0..max_thinking_steps {
            let mut deltanet_out = BatchTensor::new(b_size, self.deltanet2.hidden_size);
            self.deltanet2.forward_step_readonly(&h_thought, query_metadata == 1, &mut deltanet_out);
            
            for b in 0..b_size {
                for j in 0..self.deltanet2.hidden_size {
                    let prev = h_thought.data.read(b, j);
                    let cand = prev + deltanet_out.data.read(b, j);
                    h_thought.data.write(b, j, prev * 0.5 + cand * 0.5);
                }
            }
            
            let mut p_halt_batch = 0.0f32;
            for b in 0..b_size {
                let mut dot = self.b_halt;
                for j in 0..self.deltanet2.hidden_size {
                    dot += h_thought.data.read(b, j) * self.w_halt.data[j];
                }
                let p_halt = 1.0 / (1.0 + (-dot).exp());
                p_halt_batch += p_halt;
            }
            p_halt_batch /= b_size as f32;
            accumulated_prob += p_halt_batch;
            
            if accumulated_prob >= 1.0 || p_halt_batch > 0.75 {
                break;
            }
        }

        // Final Commit: update S_t EXACTLY ONCE with stabilized thought h*
        self.deltanet2.commit_state_step(&h_thought, query_metadata == 1);
        
        let mut h_3 = BatchTensor::new(b_size, self.deltanet2.hidden_size);
        for b in 0..b_size {
            for i in 0..self.deltanet2.hidden_size {
                h_3.data.write(b, i, h_thought.data.read(b, i));
            }
        }
        
        let h_3_norm = self.rmsnorm3.forward(&h_3);

        self.gating.forward_with_cache(&h_3_norm, &mut self.gating_scratch);
        let gated_hidden = self.gating.hidden.clone();
        
        // ANT 2026: Asynchronous Event-Driven Memory Consolidation
        let h_cols = self.gating.hidden.data.cols;
        let mut memory_writes = Vec::new();
        for b in 0..b_size {
            let mut gate_energy = 0.0;
            for j in 0..h_cols {
                gate_energy += self.gating.hidden.data.read(b, j);
            }
            gate_energy /= h_cols as f32;

            // Dynamic Z-Score Bayesian Surprisal Filter (EMA)
            let token = tokens[b];
            let vocab_size = self.embedding.vocab_size;
            let mut max_logit = f32::NEG_INFINITY;
            for v in 0..vocab_size {
                let l = self.logits.data.read(b, v);
                if l > max_logit { max_logit = l; }
            }
            let mut sum_exp = 0.0;
            for v in 0..vocab_size {
                sum_exp += (self.logits.data.read(b, v) - max_logit).exp();
            }
            let prob = if sum_exp > 0.0 && token < vocab_size {
                (self.logits.data.read(b, token) - max_logit).exp() / sum_exp
            } else {
                1.0 / vocab_size as f32
            };
            let token_surprise = -(prob + 1e-7).ln();
            let delta = token_surprise - self.surprise_mean;
            self.surprise_mean += 0.01 * delta;
            self.surprise_var = (1.0 - 0.01) * self.surprise_var + 0.01 * delta * delta;
            let std_dev = self.surprise_var.sqrt().max(0.1);
            let z_score = (token_surprise - self.surprise_mean) / std_dev;

            let is_meaningful_surprise = z_score >= 0.8 && z_score <= 3.5;

            let is_empty = if self.is_training { self.base_memory.current_size == 0 } else { self.user_memory.current_size == 0 };
            let should_write = (gate_energy > self.consolidation_energy && is_meaningful_surprise) || is_empty;

            if should_write {
                let mut key = Tensor1D::new(self.embedding.embed_dim);
                let mut val = Tensor1D::new(self.mingru.hidden_state.data.cols);
                for i in 0..self.embedding.embed_dim {
                    key.data[i] = emb.data.read(b, i);
                }
                for j in 0..self.mingru.hidden_state.data.cols {
                    val.data[j] = h_1.data.read(b, j);
                }
                memory_writes.push((key, val));
            }
        }

        if let Some(ref mut gpu_mem) = self.gpu_memory {
            let mut keys_flat = Vec::with_capacity(memory_writes.len() * self.embedding.embed_dim);
            let mut vals_flat = Vec::with_capacity(memory_writes.len() * self.mingru.hidden_state.data.cols);
            for (k, v) in &memory_writes {
                keys_flat.extend_from_slice(&k.data);
                vals_flat.extend_from_slice(&v.data);
            }
            gpu_mem.async_add_memory(&keys_flat, &vals_flat, memory_writes.len());
        }

        for (k, v) in memory_writes {
            if self.is_training {
                self.base_memory.add_memory(k, v, query_metadata);
            } else {
                self.user_memory.add_memory(k, v, query_metadata);
            }
        }

        self.readout.forward(&gated_hidden, &self.embedding, &mut self.readout_scratch, &mut self.logits);
        &self.logits
    }
    
    pub fn forward_with_cache(&mut self, tokens: &[usize], mingru_scratch: &mut MinGruScratchpad) -> (&BatchTensor, BatchTensor) {
        let b_size = tokens.len();
        
        if b_size == 1 {
            self.session_tape.append(tokens[0]);
            self.fifo_buffer.push(tokens[0]);
        }

        let emb = self.embedding.forward(tokens);
        
        self.mingru.forward_with_cache(&emb, mingru_scratch);
        let mut h_1 = BatchTensor::new(b_size, self.deltanet2.hidden_size);
        for b in 0..b_size {
            for i in 0..self.deltanet2.hidden_size {
                h_1.data.write(b, i, self.mingru.hidden_state.data.read(b, i));
            }
        }
        
        let h_1_norm = self.rmsnorm1.forward(&h_1);
        
        // Negation detection
        let mut query_metadata = 0u64;
        for &tok in tokens {
            if self.negation_ids.contains(&tok) {
                query_metadata = 1;
            }
        }
        if query_metadata == 0 {
            let recent = self.fifo_buffer.get_tokens();
            let start_idx = recent.len().saturating_sub(5);
            for &tok in &recent[start_idx..] {
                if self.negation_ids.contains(&tok) {
                    query_metadata = 1;
                    break;
                }
            }
        }

        let h_2 = self.memory_attention.forward(&h_1_norm, &h_1, &self.base_memory, &self.user_memory, &self.mounted_packs, query_metadata, &mut self.memory_attn_cache);
        
        let h_2_norm = self.rmsnorm2.forward(&h_2);
        
        let mut deltanet_out = BatchTensor::new(b_size, self.deltanet2.hidden_size);
        self.deltanet2.forward_with_cache(&h_2_norm, query_metadata == 1, &mut self.deltanet2_scratch);
        deltanet_out.data.copy_from(&self.deltanet2_scratch.y.data);
        
        let mut h_3 = BatchTensor::new(b_size, self.deltanet2.hidden_size);
        for b in 0..b_size {
            for i in 0..self.deltanet2.hidden_size {
                h_3.data.write(b, i, h_2.data.read(b, i) + deltanet_out.data.read(b, i));
            }
        }
        
        let h_3_norm = self.rmsnorm3.forward(&h_3);

        self.gating.forward_with_cache(&h_3_norm, &mut self.gating_scratch);
        let gated_hidden = self.gating.hidden.clone();

        // ANT 2026: Asynchronous Event-Driven Memory Consolidation
        let h_cols = self.gating.hidden.data.cols;
        let mut memory_writes = Vec::new();
        for b in 0..b_size {
            let mut gate_energy = 0.0;
            for j in 0..h_cols {
                gate_energy += self.gating.hidden.data.read(b, j);
            }
            gate_energy /= h_cols as f32;

            // Dynamic Z-Score Bayesian Surprisal Filter (EMA)
            let token = tokens[b];
            let vocab_size = self.embedding.vocab_size;
            let mut max_logit = f32::NEG_INFINITY;
            for v in 0..vocab_size {
                let l = self.logits.data.read(b, v);
                if l > max_logit { max_logit = l; }
            }
            let mut sum_exp = 0.0;
            for v in 0..vocab_size {
                sum_exp += (self.logits.data.read(b, v) - max_logit).exp();
            }
            let prob = if sum_exp > 0.0 && token < vocab_size {
                (self.logits.data.read(b, token) - max_logit).exp() / sum_exp
            } else {
                1.0 / vocab_size as f32
            };
            let token_surprise = -(prob + 1e-7).ln();
            let delta = token_surprise - self.surprise_mean;
            self.surprise_mean += 0.01 * delta;
            self.surprise_var = (1.0 - 0.01) * self.surprise_var + 0.01 * delta * delta;
            let std_dev = self.surprise_var.sqrt().max(0.1);
            let z_score = (token_surprise - self.surprise_mean) / std_dev;

            let is_meaningful_surprise = z_score >= 0.8 && z_score <= 3.5;

            let is_empty = if self.is_training { self.base_memory.current_size == 0 } else { self.user_memory.current_size == 0 };
            let should_write = (gate_energy > self.consolidation_energy && is_meaningful_surprise) || is_empty;

            if should_write {
                let mut key = Tensor1D::new(self.embedding.embed_dim);
                let mut val = Tensor1D::new(self.mingru.hidden_state.data.cols);
                for i in 0..self.embedding.embed_dim {
                    key.data[i] = emb.data.read(b, i);
                }
                for j in 0..self.mingru.hidden_state.data.cols {
                    val.data[j] = h_1.data.read(b, j);
                }
                memory_writes.push((key, val));
            }
        }
        
        if let Some(ref mut gpu_mem) = self.gpu_memory {
            let mut keys_flat = Vec::with_capacity(memory_writes.len() * self.embedding.embed_dim);
            let mut vals_flat = Vec::with_capacity(memory_writes.len() * self.mingru.hidden_state.data.cols);
            for (k, v) in &memory_writes {
                keys_flat.extend_from_slice(&k.data);
                vals_flat.extend_from_slice(&v.data);
            }
            gpu_mem.async_add_memory(&keys_flat, &vals_flat, memory_writes.len());
        }

        for (k, v) in memory_writes {
            if self.is_training {
                self.base_memory.add_memory(k, v, query_metadata);
            } else {
                self.user_memory.add_memory(k, v, query_metadata);
            }
        }
        
        self.readout.forward(&gated_hidden, &self.embedding, &mut self.readout_scratch, &mut self.logits);
        
        (&self.logits, h_2)
    }

    /// Online adaptation (Reward-Modulated Hebbian Learning)
    pub fn adapt_memory(&mut self, reward: f32) {
        let mut h = Tensor1D::new(self.mingru.hidden_state.data.ncols());
        for i in 0..h.len() { h.data[i] = self.mingru.hidden_state.data.read(0, i); }
        if self.is_training {
            self.base_memory.update_memory(&h, reward, self.consolidation_energy);
        } else {
            self.user_memory.update_memory(&h, reward, self.consolidation_energy);
        }
    }

    /// Save weights to a binary file with ANT Header
    pub fn save_weights<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        use std::io::Write;
        use crate::ant_core::ewc::Parameterized;
        
        let mut file = std::fs::File::create(path)?;
        
        // 1. WRITE HEADER
        let header = crate::format::AntHeader::new(
            self.embedding.vocab_size,
            self.embedding.embed_dim,
            self.mingru.hidden_state.data.cols,
            self.base_memory.capacity,
        );
        file.write_all(bytemuck::bytes_of(&header))?;
        
        // 2. WRITE WEIGHTS
        let mut write_params = |model: &dyn Parameterized| -> std::io::Result<()> {
            for p_slice in model.params() {
                let bytes = bytemuck::cast_slice(p_slice);
                file.write_all(bytes)?;
            }
            Ok(())
        };
        
        write_params(&self.embedding)?;
        write_params(&self.mingru)?;
        write_params(&self.rmsnorm1)?;
        write_params(&self.memory_attention)?;
        write_params(&self.rmsnorm2)?;
        write_params(&self.deltanet2)?;
        write_params(&self.rmsnorm3)?;
        write_params(&self.gating)?;
        write_params(&self.readout)?;
        
        Ok(())
    }

    /// Load weights, checking the ANT Header first (supporting v2.0 backward compatibility)
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> std::io::Result<()> {
        use std::io::Read;
        use crate::ant_core::ewc::Parameterized;
        
        let mut file = std::fs::File::open(&path)?;
        let file_len = file.metadata()?.len();
        
        // 1. READ AND CHECK HEADER
        let mut header = crate::format::AntHeader::new(0, 0, 0, 0);
        file.read_exact(bytemuck::bytes_of_mut(&mut header))?;
        
        if &header.magic != b"ANT\0" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid .ant magic number"));
        }
        
        assert_eq!(header.vocab_size as usize, self.embedding.vocab_size, "Vocab size mismatch in .ant file!");
        assert_eq!(header.embed_dim as usize, self.embedding.embed_dim, "Embed dim mismatch in .ant file!");
        assert_eq!(header.hidden_size as usize, self.mingru.hidden_state.data.cols, "Hidden size mismatch in .ant file!");
        
        // Calculate expected base size (excluding lora parameters)
        let v = self.embedding.vocab_size;
        let e = self.embedding.embed_dim;
        let h = self.mingru.hidden_state.data.cols;
        
        let base_floats = 
            (v * e) + // embedding
            (h * e * 2 + h * 2) + // mingru
            h + // rmsnorm1
            (e * h + e + h * h + h) + // memory_attention (base)
            h + // rmsnorm2
            (h * h * 6) + // deltanet2
            h + // rmsnorm3
            (h * h + h) + // gating
            (e * h + e); // readout (base)
        let expected_base_size = 40 + base_floats * 4;
        
        let is_lora_present = file_len >= (expected_base_size + (2 * e * self.memory_attention.w_q.rank + 2 * h * self.memory_attention.w_fuse.rank + 2 * e * self.readout.w_proj.rank) * 4) as u64;
        
        // Load embedding
        for p_slice in self.embedding.params_mut() {
            file.read_exact(bytemuck::cast_slice_mut(p_slice))?;
        }
        // Load mingru
        for p_slice in self.mingru.params_mut() {
            file.read_exact(bytemuck::cast_slice_mut(p_slice))?;
        }
        // Load rmsnorm1
        for p_slice in self.rmsnorm1.params_mut() {
            file.read_exact(bytemuck::cast_slice_mut(p_slice))?;
        }
        
        // Load memory_attention
        if is_lora_present {
            for p_slice in self.memory_attention.params_mut() {
                file.read_exact(bytemuck::cast_slice_mut(p_slice))?;
            }
        } else {
            file.read_exact(bytemuck::cast_slice_mut(&mut self.memory_attention.w_q.base.data))?;
            file.read_exact(bytemuck::cast_slice_mut(&mut self.memory_attention.b_q.data))?;
            file.read_exact(bytemuck::cast_slice_mut(&mut self.memory_attention.w_fuse.base.data))?;
            file.read_exact(bytemuck::cast_slice_mut(&mut self.memory_attention.b_fuse.data))?;
            self.memory_attention.w_q.lora_a.randomize(-0.1, 0.1);
            self.memory_attention.w_q.lora_b.data.fill(0.0);
            self.memory_attention.w_fuse.lora_a.randomize(-0.1, 0.1);
            self.memory_attention.w_fuse.lora_b.data.fill(0.0);
        }
        
        // Load rmsnorm2
        for p_slice in self.rmsnorm2.params_mut() {
            file.read_exact(bytemuck::cast_slice_mut(p_slice))?;
        }
        // Load deltanet2
        for p_slice in self.deltanet2.params_mut() {
            file.read_exact(bytemuck::cast_slice_mut(p_slice))?;
        }
        // Load rmsnorm3
        for p_slice in self.rmsnorm3.params_mut() {
            file.read_exact(bytemuck::cast_slice_mut(p_slice))?;
        }
        // Load gating
        for p_slice in self.gating.params_mut() {
            file.read_exact(bytemuck::cast_slice_mut(p_slice))?;
        }
        
        // Load readout
        if is_lora_present {
            for p_slice in self.readout.params_mut() {
                file.read_exact(bytemuck::cast_slice_mut(p_slice))?;
            }
        } else {
            file.read_exact(bytemuck::cast_slice_mut(&mut self.readout.w_proj.base.data))?;
            file.read_exact(bytemuck::cast_slice_mut(&mut self.readout.b_proj.data))?;
            self.readout.w_proj.lora_a.randomize(-0.1, 0.1);
            self.readout.w_proj.lora_b.data.fill(0.0);
        }
        
        // Sync to GPU weights if GPU accelerator is loaded
        if let Some(ref mut gr) = self.gpu_readout {
            gr.sync_weights_to_gpu(&self.embedding, &self.readout);
        }
        
        Ok(())
    }
}
