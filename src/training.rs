#![allow(non_snake_case)]
use std::fs;
use indicatif::{ProgressBar, ProgressStyle};
use crate::ant_core::tensor::{MatExt, BatchTensor};
use crate::ant_core::pipeline::AntPipeline;
use crate::ant_core::optim::SGD;
use crate::ant_core::ewc::EwcSnapshot;

use tokenizers::Tokenizer;

pub struct TextDataset {
    pub tokens: Vec<usize>,
    pub vocab_size: usize,
}

impl TextDataset {
    pub fn new(path: &str, tokenizer_path: &str) -> std::io::Result<Self> {
        let bytes = fs::read(path)?;
        let text = String::from_utf8_lossy(&bytes);
        
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::NotFound, format!("Failed to load tokenizer from {}: {}", tokenizer_path, e)))?;
        
        let encoding = tokenizer.encode(text.as_ref(), false)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Encoding error: {}", e)))?;
            
        let tokens: Vec<usize> = encoding.get_ids().iter().map(|&id| id as usize).collect();
        let vocab_size = tokenizer.get_vocab_size(true);
        
        println!("Loaded dataset: {} tokens using 8K tokenizer (vocab_size: {})", tokens.len(), vocab_size);

        Ok(Self {
            tokens,
            vocab_size,
        })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn get(&self, index: usize) -> usize {
        self.tokens[index]
    }
}

pub struct Trainer {
    pub pipeline: AntPipeline,
    optimizer: SGD,
    opt_cfg: crate::OptimizerSection,
    pub cl_cfg: crate::ContinualLearningSection,
    pub model_path: String,
    ewc_snapshot: Option<EwcSnapshot>,
    dh_total: crate::ant_core::tensor::BatchTensor,
    dh_next: crate::ant_core::tensor::BatchTensor,
    pub gpu_accelerator: Option<crate::compute_gpu::gpu_rnn::GpuAccelerator>,
}

impl Trainer {
    pub fn new(pipeline: AntPipeline, opt_cfg: crate::OptimizerSection, cl_cfg: crate::ContinualLearningSection, model_path: String) -> Self {
        let learning_rate = opt_cfg.default_lr as f32;
        let batch_size = pipeline.mingru.hidden_state.data.rows;
        let hidden_size = pipeline.mingru.hidden_state.data.cols;
        
        let gpu_accelerator = if pipeline.gpu_readout.is_some() {
            Some(crate::compute_gpu::gpu_rnn::GpuAccelerator::new(
                pipeline.embedding.vocab_size, 
                pipeline.embedding.embed_dim, 
                hidden_size, 
                batch_size, 
                70, // Default seq_len
                pipeline.base_memory.capacity + pipeline.user_memory.capacity,
                cl_cfg.lora_rank,
                cl_cfg.lora_alpha,
            ))
        } else {
            None
        };

        Self {
            pipeline,
            optimizer: SGD::new(learning_rate),
            opt_cfg,
            cl_cfg,
            model_path,
            ewc_snapshot: None,
            dh_total: crate::ant_core::tensor::BatchTensor::new(batch_size, hidden_size),
            dh_next: crate::ant_core::tensor::BatchTensor::new(batch_size, hidden_size),
            gpu_accelerator,
        }
    }

    pub fn set_ewc_snapshot(&mut self, snapshot: EwcSnapshot) {
        self.ewc_snapshot = Some(snapshot);
    }

    pub fn sleep_phase(&mut self, _replay_dataset: &TextDataset, _seq_len: usize, steps: usize) {
        println!("[ANT Sleep Phase] Initiating Memory Consolidation...");
        
        self.pipeline.base_memory.compress_and_prune(0.95);
        self.pipeline.user_memory.compress_and_prune(0.95);

        println!("[ANT Sleep Phase] Consolidating neural pathways (EWC)...");
        
        if let Some(ref mut snapshot) = self.ewc_snapshot {
            snapshot.apply_decay(self.cl_cfg.fisher_decay);
        }
        
        let mut current_step = 0;
        let mut compute_grads = |pipeline: &mut AntPipeline| -> bool {
            if current_step >= steps { return false; }
            
            pipeline.embedding.weight.zero_grad();
            pipeline.mingru.w_z.zero_grad();
            pipeline.mingru.b_z.zero_grad();
            pipeline.mingru.w_h.zero_grad();
            pipeline.mingru.b_h.zero_grad();
            
            pipeline.rmsnorm1.weight.zero_grad();
            pipeline.rmsnorm2.weight.zero_grad();
            pipeline.rmsnorm3.weight.zero_grad();
            
            pipeline.deltanet2.w_k.zero_grad();
            pipeline.deltanet2.w_v.zero_grad();
            pipeline.deltanet2.w_q.zero_grad();
            pipeline.deltanet2.w_b.zero_grad();
            pipeline.deltanet2.w_w.zero_grad();
            pipeline.deltanet2.w_alpha.zero_grad();
            
            pipeline.memory_attention.w_q.zero_grad();
            pipeline.memory_attention.b_q.zero_grad();
            pipeline.memory_attention.w_fuse.zero_grad();
            pipeline.memory_attention.b_fuse.zero_grad();
            
            pipeline.gating.w1.zero_grad();
            pipeline.gating.b1.zero_grad();
            
            pipeline.readout.w_proj.zero_grad();
            pipeline.readout.b_proj.zero_grad();
            
            current_step += 1;
            true
        };

        while compute_grads(&mut self.pipeline) {}

        let snapshot = EwcSnapshot::consolidate_from_current_grads(&self.pipeline.mingru, 100.0);
        self.set_ewc_snapshot(snapshot);
        
        println!("[ANT Sleep Phase] Waking up. Agent is ready.");
    }

    pub fn train_epoch(&mut self, dataset: &TextDataset, seq_len: usize) {
        let batch_size = self.pipeline.mingru.hidden_state.data.rows;
        let total_tokens = dataset.len().saturating_sub(1);
        let tokens_per_batch = total_tokens / batch_size;
        let total_steps = tokens_per_batch;
        
        if total_steps == 0 {
            return;
        }

        // Reset hidden states at the start of each epoch
        self.pipeline.mingru.hidden_state.zero_grad();
        self.pipeline.mingru.hidden_state.data.fill(0.0);
        self.pipeline.deltanet2.state.fill(0.0);

        let pb = ProgressBar::new(total_steps as u64 * 2);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) - Loss: {msg}")
            .unwrap()
            .progress_chars("#>-"));

        let mut epoch_total_loss = 0.0;
        let mut epoch_steps = 0;
        
        let embed_dim = self.pipeline.embedding.embed_dim;
        let hidden_size = self.pipeline.mingru.hidden_state.data.cols;
        let top_k = self.pipeline.memory_attention.top_k_base + self.pipeline.memory_attention.top_k_user;
        
        // TBPTT histories
        let mut scratch_history = vec![crate::ant_core::rnn::MinGruScratchpad::new(batch_size, embed_dim, hidden_size); seq_len];
        let mut deltanet_scratch_history = vec![crate::ant_core::rnn::GatedDeltaNet2Scratchpad::new(batch_size, hidden_size, hidden_size); seq_len];
        
        let mut h_1_history = vec![crate::ant_core::tensor::BatchTensor::new(batch_size, hidden_size); seq_len];
        let mut h_1_norm_history = vec![crate::ant_core::tensor::BatchTensor::new(batch_size, hidden_size); seq_len];
        let mut h_2_history = vec![crate::ant_core::tensor::BatchTensor::new(batch_size, hidden_size); seq_len];
        let mut h_2_norm_history = vec![crate::ant_core::tensor::BatchTensor::new(batch_size, hidden_size); seq_len];
        let mut deltanet_out_history = vec![crate::ant_core::tensor::BatchTensor::new(batch_size, hidden_size); seq_len];
        let mut h_3_history = vec![crate::ant_core::tensor::BatchTensor::new(batch_size, hidden_size); seq_len];
        let mut h_3_norm_history = vec![crate::ant_core::tensor::BatchTensor::new(batch_size, hidden_size); seq_len];
        let mut gating_hidden_history = vec![crate::ant_core::tensor::BatchTensor::new(batch_size, hidden_size); seq_len];
        
        let mut gating_scratch_history = vec![crate::ant_core::sparse_gating::SparseGatingScratchpad::new(batch_size, hidden_size, hidden_size); seq_len];
        let mut attn_cache_history = vec![crate::ant_core::memory_attention::MemoryAttentionCache::new(batch_size, embed_dim, hidden_size, top_k); seq_len];
        let mut loss_grad_history = vec![crate::ant_core::tensor::BatchTensor::new(batch_size, dataset.vocab_size); seq_len];
        let mut input_history = vec![vec![0usize; batch_size]; seq_len];
        let mut readout_scratch_history = vec![crate::ant_core::readout::ReadoutScratchpad::new(batch_size, embed_dim, hidden_size); seq_len];

        let mut i = 0;
        while i < total_steps {
            let chunk_size = std::cmp::min(seq_len, total_steps - i);
            let mut chunk_loss = 0.0;
            
            let mut chunk_inputs = vec![vec![0usize; batch_size]; chunk_size];
            let mut chunk_targets = vec![vec![0usize; batch_size]; chunk_size];
            for t in 0..chunk_size {
                for b in 0..batch_size {
                    let idx = b * tokens_per_batch + i + t;
                    chunk_inputs[t][b] = dataset.get(idx);
                    chunk_targets[t][b] = dataset.get(idx + 1);
                }
            }

            // [Shadow GPU Trainer VRAM BPTT]
            if let Some(ref mut accelerator) = self.gpu_accelerator {
                accelerator.load_weights(&self.pipeline);
                if let Some(ref mut gpu_readout) = self.pipeline.gpu_readout {
                    gpu_readout.sync_weights_to_gpu(&self.pipeline.embedding, &self.pipeline.readout);
                    gpu_readout.zero_grad();
                }

                accelerator.train_chunk(
                    &mut self.pipeline, 
                    &chunk_inputs, 
                    &chunk_targets, 
                    &mut chunk_loss, 
                    self.optimizer.learning_rate,
                    self.opt_cfg.beta1,
                    self.opt_cfg.beta2,
                    self.opt_cfg.weight_decay,
                );
                
                accelerator.save_weights(&mut self.pipeline);
                
                if let Some(ref mut gpu_readout) = self.pipeline.gpu_readout {
                    gpu_readout.step(
                        self.optimizer.learning_rate,
                        self.opt_cfg.beta1,
                        self.opt_cfg.beta2,
                        self.opt_cfg.weight_decay,
                    );
                    gpu_readout.sync_and_accumulate(&mut self.pipeline.embedding, &mut self.pipeline.readout, self.optimizer.learning_rate);
                }
                
                if chunk_loss.is_nan() || chunk_loss.is_infinite() {
                    eprintln!("\n🚨 [NaN/Inf Guard] Warning: NaN or Inf detected in loss at step {}! Skipping step accumulation.", epoch_steps);
                } else {
                    epoch_total_loss += chunk_loss / (batch_size * chunk_size) as f32;
                }
                
                epoch_steps += 1;
                pb.inc((chunk_size * 2) as u64);
                
                i += chunk_size;
                
                if epoch_steps % 10 == 0 {
                    pb.set_message(format!("{:.4}", epoch_total_loss / epoch_steps as f32));
                }
                if epoch_steps % 100 == 0 && !chunk_loss.is_nan() && !chunk_loss.is_infinite() {
                    let _ = self.pipeline.save_weights(&self.model_path);
                }
                continue;
            }
            
            let has_gpu = self.pipeline.gpu_readout.is_some();
            if has_gpu {
                let emb = &self.pipeline.embedding;
                let ro = &self.pipeline.readout;
                let gr = self.pipeline.gpu_readout.as_mut().unwrap();
                gr.sync_weights_to_gpu(emb, ro);
                gr.zero_grad();
            }

            // 1. Forward Pass over the sequence chunk
            let mut bulk_hidden = BatchTensor::new(batch_size * chunk_size, hidden_size);
            let mut bulk_targets = vec![0usize; batch_size * chunk_size];

            for t in 0..chunk_size {
                let mut inputs = vec![0usize; batch_size];
                let mut targets = vec![0usize; batch_size];
                
                for b in 0..batch_size {
                    let idx = b * tokens_per_batch + i + t;
                    inputs[b] = dataset.get(idx);
                    targets[b] = dataset.get(idx + 1);
                }

                input_history[t] = inputs.clone();
                
                let emb = self.pipeline.embedding.forward(&inputs);
                
                self.pipeline.mingru.forward_with_cache(&emb, &mut scratch_history[t]);
                let h_1 = self.pipeline.mingru.hidden_state.clone();
                h_1_history[t] = h_1.clone();
                
                let h_1_norm = self.pipeline.rmsnorm1.forward(&h_1);
                h_1_norm_history[t] = h_1_norm.clone();
                
                let mut query_polarity = 0u64;
                for &tok in &inputs {
                    if self.pipeline.negation_ids.contains(&tok) {
                        query_polarity = 1;
                        break;
                    }
                }
                
                let h_2 = self.pipeline.memory_attention.forward(
                    &h_1_norm,
                    &h_1,
                    &self.pipeline.base_memory,
                    &self.pipeline.user_memory,
                    query_polarity,
                    &mut attn_cache_history[t]
                );
                h_2_history[t] = h_2.clone();
                
                let h_2_norm = self.pipeline.rmsnorm2.forward(&h_2);
                h_2_norm_history[t] = h_2_norm.clone();
                
                let mut deltanet_out = BatchTensor::new(batch_size, hidden_size);
                self.pipeline.deltanet2.forward_with_cache(&h_2_norm, query_polarity == 1, &mut deltanet_scratch_history[t]);
                deltanet_out.data.copy_from(&deltanet_scratch_history[t].y.data);
                deltanet_out_history[t] = deltanet_out.clone();
                
                let mut h_3 = BatchTensor::new(batch_size, hidden_size);
                for b_idx in 0..batch_size {
                    for j in 0..hidden_size {
                        h_3.data.write(b_idx, j, h_2.data.read(b_idx, j) + deltanet_out.data.read(b_idx, j));
                    }
                }
                h_3_history[t] = h_3.clone();
                
                let h_3_norm = self.pipeline.rmsnorm3.forward(&h_3);
                h_3_norm_history[t] = h_3_norm.clone();
                
                self.pipeline.gating.forward_with_cache(&h_3_norm, &mut gating_scratch_history[t]);
                let gated_hidden = self.pipeline.gating.hidden.clone();
                gating_hidden_history[t] = gated_hidden.clone();

                // Memory updates
                let h_cols = self.pipeline.gating.hidden.data.cols;
                let mut memory_writes = Vec::new();
                for b in 0..batch_size {
                    let mut gate_energy = 0.0;
                    for j in 0..h_cols {
                        gate_energy += self.pipeline.gating.hidden.data.read(b, j);
                    }
                    gate_energy /= h_cols as f32;

                    if gate_energy > 0.05 || self.pipeline.base_memory.current_size == 0 {
                        let mut key = crate::ant_core::tensor::Tensor1D::new(self.pipeline.embedding.embed_dim);
                        let mut val = crate::ant_core::tensor::Tensor1D::new(hidden_size);
                        for k_idx in 0..self.pipeline.embedding.embed_dim {
                            key.data[k_idx] = emb.data.read(b, k_idx);
                        }
                        for j in 0..hidden_size {
                            val.data[j] = h_1.data.read(b, j);
                        }
                        memory_writes.push((key, val));
                    }
                }
                for (k, v) in memory_writes {
                    self.pipeline.base_memory.add_memory(k, v, query_polarity);
                }

                if has_gpu {
                    for b in 0..batch_size {
                        for j in 0..hidden_size {
                            bulk_hidden.data.write(t * batch_size + b, j, gated_hidden.data.read(b, j));
                        }
                        bulk_targets[t * batch_size + b] = targets[b];
                    }
                } else {
                    self.pipeline.readout.forward(&gated_hidden, &self.pipeline.embedding, &mut readout_scratch_history[t], &mut self.pipeline.logits);
                    
                    let step_losses: Vec<f32> = (0..batch_size)
                        .map(|b| {
                            let target = targets[b];
                            let target_logit = self.pipeline.logits.data.read(b, target);
                            
                            let mut max_val = self.pipeline.logits.data.read(b, 0);
                            for j in 1..dataset.vocab_size {
                                let val = self.pipeline.logits.data.read(b, j);
                                if val > max_val { max_val = val; }
                            }
                            
                            let mut sum_exp = 0.0;
                            for j in 0..dataset.vocab_size {
                                sum_exp += (self.pipeline.logits.data.read(b, j) - max_val).exp();
                            }
                            
                            let loss_val = sum_exp.log(std::f32::consts::E) - (target_logit - max_val);
                            
                            for j in 0..dataset.vocab_size {
                                let p = (self.pipeline.logits.data.read(b, j) - max_val).exp() / (sum_exp + 1e-9);
                                let target_val = if j == target { 1.0 } else { 0.0 };
                                let grad_val = (p - target_val) / (batch_size * chunk_size) as f32;
                                loss_grad_history[t].grad.write(b, j, if grad_val.is_finite() { grad_val } else { 0.0 });
                            }
                            
                            if loss_val.is_finite() { loss_val } else { 0.0 }
                        })
                        .collect();
                    
                    chunk_loss += step_losses.iter().sum::<f32>();
                }
                
                pb.inc(1);
            }
            
            let bulk_d_hidden = if has_gpu {
                let gpu_readout = self.pipeline.gpu_readout.as_mut().unwrap();
                let (total_loss, _) = gpu_readout.forward_and_loss(
                    &bulk_hidden,
                    &bulk_targets,
                    chunk_size,
                );
                chunk_loss += total_loss;
                gpu_readout.backward(&bulk_hidden)
            } else {
                Vec::new()
            };
            
            epoch_total_loss += chunk_loss / (batch_size * chunk_size) as f32;
            epoch_steps += 1;
            
            // 2. TBPTT Backward Pass
            self.optimizer.zero_grad_embedding(&mut self.pipeline.embedding);
            self.optimizer.zero_grad_mingru(&mut self.pipeline.mingru);
            self.optimizer.zero_grad_rmsnorm(&mut self.pipeline.rmsnorm1);
            self.optimizer.zero_grad_memory_attention(&mut self.pipeline.memory_attention);
            self.optimizer.zero_grad_rmsnorm(&mut self.pipeline.rmsnorm2);
            self.optimizer.zero_grad_deltanet(&mut self.pipeline.deltanet2);
            self.optimizer.zero_grad_rmsnorm(&mut self.pipeline.rmsnorm3);
            self.optimizer.zero_grad_sparse_gating(&mut self.pipeline.gating);
            self.optimizer.zero_grad_readout(&mut self.pipeline.readout);

            self.dh_next.zero_grad();
            self.dh_next.data.fill(0.0);
            
            let mut d_S_next = vec![0.0; batch_size * hidden_size * hidden_size];
            
            for t in (0..chunk_size).rev() {
                let d_gating_out = if has_gpu {
                    let mut d_gating = BatchTensor::new(batch_size, hidden_size);
                    for b_idx in 0..batch_size {
                        for j in 0..hidden_size {
                            let val = bulk_d_hidden[(t * batch_size + b_idx) * hidden_size + j];
                            d_gating.grad.write(b_idx, j, val);
                        }
                    }
                    d_gating
                } else {
                    self.pipeline.readout.backward(&gating_hidden_history[t], &loss_grad_history[t], &mut self.pipeline.embedding, &mut readout_scratch_history[t]).clone()
                };
                
                // Backprop through SparseGating
                let d_h_3_norm = self.pipeline.gating.backward(&d_gating_out, &mut gating_scratch_history[t]);
                
                // Backprop through RMSNorm3
                let mut d_h_3 = BatchTensor::new(batch_size, hidden_size);
                self.pipeline.rmsnorm3.backward(&h_3_history[t], d_h_3_norm, &mut d_h_3);
                
                // Backprop GatedDeltaNet2
                let d_h_2_norm = self.pipeline.deltanet2.backward_step(&d_h_3, &mut d_S_next, &mut deltanet_scratch_history[t]);
                
                // Backprop RMSNorm2
                let mut d_h_2_attn_out = BatchTensor::new(batch_size, hidden_size);
                self.pipeline.rmsnorm2.backward(&h_2_history[t], &d_h_2_norm, &mut d_h_2_attn_out);
                
                // Accumulate direct residual gradient to h_2
                let mut d_h_2_total = BatchTensor::new(batch_size, hidden_size);
                for b_idx in 0..batch_size {
                    for j in 0..hidden_size {
                        let val = d_h_2_attn_out.grad.read(b_idx, j) + d_h_3.grad.read(b_idx, j);
                        d_h_2_total.grad.write(b_idx, j, val);
                    }
                }
                
                // Backprop MemoryAttention
                let mut grad_hidden_raw = BatchTensor::new(batch_size, hidden_size);
                let d_h_1_norm = self.pipeline.memory_attention.backward(
                    &mut h_1_norm_history[t],
                    &d_h_2_total,
                    &self.pipeline.base_memory,
                    &self.pipeline.user_memory,
                    &mut attn_cache_history[t],
                    &mut grad_hidden_raw,
                );
                
                // Backprop RMSNorm1
                let mut d_h_1 = BatchTensor::new(batch_size, hidden_size);
                self.pipeline.rmsnorm1.backward(&h_1_history[t], d_h_1_norm, &mut d_h_1);
                
                // Sum gradients for minGRU output
                self.dh_total.zero_grad();
                for b_idx in 0..batch_size {
                    for j in 0..hidden_size {
                        let val = d_h_1.grad.read(b_idx, j) + grad_hidden_raw.grad.read(b_idx, j) + self.dh_next.grad.read(b_idx, j);
                        self.dh_total.grad.write(b_idx, j, val);
                    }
                }
                
                // Backprop minGRU
                let (d_emb, prev_dh) = self.pipeline.mingru.backward(&self.dh_total, &mut scratch_history[t]);
                self.dh_next.grad.copy_from(&prev_dh.grad);
                self.dh_next.data.copy_from(&prev_dh.data);

                self.pipeline.embedding.backward(&input_history[t], d_emb);
                
                pb.inc(1);
            }

            if has_gpu {
                let emb = &mut self.pipeline.embedding;
                let ro = &mut self.pipeline.readout;
                self.pipeline.gpu_readout.as_mut().unwrap().sync_grads_back_to_host(emb, ro);
            }

            // 3. Update Weights
            if let Some(ref ewc) = self.ewc_snapshot {
                ewc.apply_penalty_grads(&mut self.pipeline.mingru);
            }

            self.optimizer.clip_embedding(&mut self.pipeline.embedding, 5.0);
            self.optimizer.clip_mingru(&mut self.pipeline.mingru, 5.0);
            self.optimizer.clip_rmsnorm(&mut self.pipeline.rmsnorm1, 5.0);
            self.optimizer.clip_memory_attention(&mut self.pipeline.memory_attention, 5.0);
            self.optimizer.clip_rmsnorm(&mut self.pipeline.rmsnorm2, 5.0);
            self.optimizer.clip_deltanet(&mut self.pipeline.deltanet2, 5.0);
            self.optimizer.clip_rmsnorm(&mut self.pipeline.rmsnorm3, 5.0);
            self.optimizer.clip_readout(&mut self.pipeline.readout, 5.0);
            self.optimizer.clip_sparse_gating(&mut self.pipeline.gating, 5.0);
            
            self.optimizer.step_embedding(&mut self.pipeline.embedding);
            self.optimizer.step_mingru(&mut self.pipeline.mingru);
            self.optimizer.step_rmsnorm(&mut self.pipeline.rmsnorm1);
            self.optimizer.step_memory_attention(&mut self.pipeline.memory_attention);
            self.optimizer.step_rmsnorm(&mut self.pipeline.rmsnorm2);
            self.optimizer.step_deltanet(&mut self.pipeline.deltanet2);
            self.optimizer.step_rmsnorm(&mut self.pipeline.rmsnorm3);
            self.optimizer.step_readout(&mut self.pipeline.readout);
            self.optimizer.step_sparse_gating(&mut self.pipeline.gating);
            
            let avg_loss = epoch_total_loss / epoch_steps as f32;
            pb.set_message(format!("{:.4}", avg_loss));
            
            i += chunk_size;
        }

        pb.finish_with_message("Epoch complete");
    }
}
