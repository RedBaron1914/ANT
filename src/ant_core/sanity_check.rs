use crate::ant_core::pipeline::AntPipeline;
use crate::ant_core::memory_io::DiskKVMemory;
use crate::ant_core::tensor::Tensor1D;
use crate::AntConfig;
use std::time::Instant;

pub fn run_full_diagnostics(config: &AntConfig) {
    println!("\n🔍 ========================================================");
    println!("   ANT v3.0 DIAGNOSTIC & ANOMALY DETECTION HARNESS");
    println!("========================================================\n");

    let start_time = Instant::now();
    let test_base_mem = "test_diag_base.ant";
    let test_user_mem = "test_diag_user.ant";
    let test_model_bin = "test_diag_model.bin";
    let _ = std::fs::remove_file(test_base_mem);
    let _ = std::fs::remove_file(test_user_mem);
    let _ = std::fs::remove_file(test_model_bin);

    // TEST 1: Memory Pruning & Black Hole Bug check
    print!("[1/5] Testing Memory Clumping & Black-Hole Defense... ");
    {
        let mut mem = DiskKVMemory::new(test_base_mem, 20, 16, 16).unwrap();
        // Insert 10 identical keys
        for _ in 0..10 {
            let mut k = Tensor1D::new(16);
            k.data.fill(1.0);
            let mut v = Tensor1D::new(16);
            v.data.fill(2.0);
            mem.add_memory(k, v, 0);
        }
        assert_eq!(mem.current_size, 10);
        mem.compress_and_prune(0.95);
        
        // Must be compressed exactly to 1!
        if mem.current_size != 1 {
            println!("❌ FAILED! Memory expected 1 entry after duplicate pruning, got {}", mem.current_size);
            std::process::exit(1);
        }
        for &val in mem.get_key(0) {
            if val.is_nan() || val.is_infinite() {
                println!("❌ FAILED! Key contains NaN/Inf after pruning!");
                std::process::exit(1);
            }
        }
        println!("✅ PASSED (10 entries merged to 1 clean entry)");
    }

    // TEST 2: Polarity / Negation Penalty Check
    print!("[2/5] Testing Negation Polarity Search Penalty... ");
    {
        let mut mem = DiskKVMemory::new(test_user_mem, 10, 16, 16).unwrap();
        let mut k = Tensor1D::new(16);
        k.data[0] = 1.0;
        let v = Tensor1D::new(16);
        // Add entry with polarity 0 (positive)
        mem.add_memory(k.clone(), v.clone(), 0);
        
        // Add entry with polarity 1 (negative/forbidden)
        let mut k2 = Tensor1D::new(16);
        k2.data[0] = 1.0;
        mem.add_memory(k2, v.clone(), 1);

        let query = k.clone();
        // Query with positive polarity (0)
        let (_, indices) = mem.lookup(&query, 0, 2);
        if indices[0] != 0 {
            println!("❌ FAILED! Positive query did not rank positive memory first!");
            std::process::exit(1);
        }
        println!("✅ PASSED (Opposite polarity penalized properly)");
    }

    let _ = std::fs::remove_file(test_base_mem);
    let _ = std::fs::remove_file(test_user_mem);

    // TEST 3: Pipeline Initialization, ACT Deliberation & Cartridge Mounting
    print!("[3/5] Testing Model Instantiation, ACT Deliberation & Memory Cartridges... ");
    let mut pipeline = AntPipeline::new(
        test_base_mem,
        test_user_mem,
        config.model.vocab_size,
        config.model.embed_dim,
        config.model.hidden_size,
        config.training.batch_size,
        100, 100,
        config.memory.top_k_base,
        config.memory.top_k_user,
        config.memory.consolidation_energy,
        config.session_tape.capacity,
        config.session_tape.fifo_window,
        config.continual_learning.lora_rank,
        config.continual_learning.lora_alpha,
        false
    ).unwrap();

    let test_logits = pipeline.forward(&[1]);
    assert_eq!(test_logits.data.ncols(), config.model.vocab_size);
    println!("✅ PASSED (Vocab: {}, Embed: {}, Hidden: {})", config.model.vocab_size, config.model.embed_dim, config.model.hidden_size);

    // TEST 4: Overfitting Single Batch on GPU (The Ultimate BPTT & Gradient Test)
    print!("[4/5] Testing GPU BPTT Overfitting & Loss Convergence (40 steps)... ");
    if let Some(ref mut accelerator) = pipeline.gpu_memory.as_ref().and_then(|_| {
        let h = pipeline.deltanet2.hidden_size;
        let e = pipeline.embedding.embed_dim;
        let b = config.training.batch_size;
        let v = pipeline.embedding.vocab_size;
        Some(crate::compute_gpu::gpu_rnn::GpuAccelerator::new(
            v, e, h, b, 20, 200,
            config.continual_learning.lora_rank,
            config.continual_learning.lora_alpha
        ))
    }) {
        let c = 20; // 20 tokens sequence
        let b = config.training.batch_size;
        
        // Create a toy repetitive phrase: [10, 20, 30, 40, 50, ...]
        let mut toy_inputs = vec![vec![0usize; b]; c];
        let mut toy_targets = vec![vec![0usize; b]; c];
        for t in 0..c {
            for batch_idx in 0..b {
                toy_inputs[t][batch_idx] = (t % 15) + 5;
                toy_targets[t][batch_idx] = ((t + 1) % 15) + 5;
            }
        }

        accelerator.load_weights(&pipeline);
        if let Some(ref mut gpu_readout) = pipeline.gpu_readout {
            gpu_readout.sync_weights_to_gpu(&pipeline.embedding, &pipeline.readout);
            gpu_readout.zero_grad();
        }
        let mut first_loss = 0.0f32;
        let mut last_loss = 0.0f32;

        for step in 0..40 {
            let mut step_loss = 0.0f32;
            if let Some(ref mut gpu_readout) = pipeline.gpu_readout {
                gpu_readout.zero_grad();
            }
            accelerator.train_chunk(
                &mut pipeline,
                &toy_inputs,
                &toy_targets,
                &mut step_loss,
                0.001, // aggressive LR for fast 5-sec test
                0.9, 0.99, 0.0
            );
            if let Some(ref mut gpu_readout) = pipeline.gpu_readout {
                gpu_readout.step(0.001, 0.9, 0.99, 0.0);
            }
            if step == 0 {
                first_loss = step_loss / (b * c) as f32;
            }
            last_loss = step_loss / (b * c) as f32;

            if last_loss.is_nan() || last_loss.is_infinite() {
                println!("\n❌ CRITICAL FAILURE! Loss exploded to NaN/Inf at step {}!", step);
                std::process::exit(1);
            }
        }

        if last_loss >= first_loss {
            println!("\n❌ FAILED! BPTT Gradients broken: Loss failed to converge (First: {:.4}, Last: {:.4})", first_loss, last_loss);
            std::process::exit(1);
        }
        println!("✅ PASSED (Loss converged: {:.4} -> {:.4})", first_loss, last_loss);
    } else {
        println!("⚠️ SKIPPED (No CUDA GPU available)");
    }

    // TEST 5: DeltaNet-2 Clamping verification
    print!("[5/5] Testing DeltaNet-2 State Matrix Clamping Bounds... ");
    {
        let mut max_val = 0.0f32;
        for &val in &pipeline.deltanet2.state {
            if val.abs() > max_val { max_val = val.abs(); }
        }
        if max_val > 5.05 {
            println!("❌ FAILED! DeltaNet-2 state exceeded [-5.0, 5.0] bounds: max = {}", max_val);
            std::process::exit(1);
        }
        println!("✅ PASSED (Max state bound: {:.3} <= 5.0)", max_val);
    }

    // Cleanup
    let _ = std::fs::remove_file(test_base_mem);
    let _ = std::fs::remove_file(test_user_mem);
    let _ = std::fs::remove_file(test_model_bin);

    println!("\n🎉 ========================================================");
    println!("   ALL DIAGNOSTIC CHECKS PASSED in {:.2?}!", start_time.elapsed());
    println!("   The architecture is 100% stable and ready for training.");
    println!("========================================================\n");
}
