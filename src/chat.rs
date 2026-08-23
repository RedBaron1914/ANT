use std::io::{self, Write};
use rand::Rng;
use tokenizers::Tokenizer;

use crate::ant_core::pipeline::AntPipeline;
use crate::ant_core::tensor::{BatchTensor, MatExt};
use crate::ChatSection;

/// Threshold Top-K Sampler for a single batch element
fn sample_token(logits: &BatchTensor, threshold: f32, top_k: usize, temp: f32) -> Option<usize> {
    let vocab_size = logits.data.ncols();
    let temp = if temp <= 0.0 { 1.0 } else { temp };
    
    let mut max_val = logits.data.read(0, 0);
    for j in 1..vocab_size {
        let v = logits.data.read(0, j);
        if v > max_val { max_val = v; }
    }
    
    let mut sum_exp = 0.0;
    let mut probs = vec![0.0; vocab_size];
    for j in 0..vocab_size {
        let e = ((logits.data.read(0, j) - max_val) / temp).exp();
        probs[j] = e;
        sum_exp += e;
    }
    
    if sum_exp > 0.0 {
        for j in 0..vocab_size {
            probs[j] /= sum_exp;
        }
    }
    
    let mut scores: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    if threshold > 0.0 && scores[0].1 < threshold {
        return None;
    }
    
    let k = std::cmp::min(top_k, scores.len());
    let top_scores = &scores[0..k];
    
    if k == 1 {
        return Some(top_scores[0].0);
    }
    
    let weights: Vec<f32> = top_scores.iter().map(|&(_, s)| f32::max(s, 1e-5)).collect();
    let sum: f32 = weights.iter().sum();
    if sum <= 0.0 {
        return Some(top_scores[0].0);
    }
    
    let mut rng = rand::thread_rng();
    let mut r = rng.gen_range(0.0..sum);
    
    for (i, &w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            return Some(top_scores[i].0);
        }
    }
    
    Some(top_scores[0].0)
}

pub fn run_chat(pipeline: &mut AntPipeline, prompt_text: Option<&str>, chat_cfg: &ChatSection, tokenizer_path: &str) {
    println!("=== ANT-rs Interactive Chat (8K Tokenizer) ===");
    println!("Type 'quit' or 'exit' to stop.");
    println!("Threshold sampling active. If unsure, the model will output '?'.");
    println!("-------------------------------");

    let vocab_size = pipeline.embedding.vocab_size;
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .expect("Failed to load tokenizer.json in chat mode");
    pipeline.negation_ids = crate::ant_core::pipeline::get_negation_ids(&tokenizer);
    pipeline.is_training = false;
    
    // If a prompt is given, "warm up" the network
    if let Some(prompt) = prompt_text {
        print!("{}", prompt);
        let _ = io::stdout().flush();
        
        pipeline.mingru.hidden_state.data.fill(0.0);
        pipeline.deltanet2.state.fill(0.0);
        
        let mut last_output = BatchTensor::new(1, vocab_size);
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let tokens = encoding.get_ids();
        for &token_id in tokens {
            last_output = pipeline.forward(&[token_id as usize]).clone();
        }
        
        let mut generated_warmup_tokens: Vec<u32> = Vec::new();
        let mut last_warmup_len = 0;
        
        // Auto-generate some tokens following the prompt
        for _ in 0..50 {
            if let Some(next_token) = sample_token(&last_output, 0.0, chat_cfg.top_k, chat_cfg.temperature) {
                if next_token == 1 || next_token == 0 {
                    println!();
                    break;
                }
                generated_warmup_tokens.push(next_token as u32);
                let full_text = tokenizer.decode(&generated_warmup_tokens, true).unwrap_or_default();
                if full_text.len() > last_warmup_len {
                    let new_part = &full_text[last_warmup_len..];
                    print!("{}", new_part);
                    let _ = io::stdout().flush();
                    last_warmup_len = full_text.len();
                }
                
                last_output = pipeline.forward(&[next_token]).clone();
            } else {
                let mut max_val = last_output.data.read(0, 0);
                for j in 0..vocab_size {
                    let val = last_output.data.read(0, j);
                    if val > max_val { max_val = val; }
                }
                let mut sum_exp = 0.0;
                for j in 0..vocab_size {
                    sum_exp += (last_output.data.read(0, j) - max_val).exp();
                }
                let max_prob = 1.0 / sum_exp;
                print!("?[max_conf:{:.3}]", max_prob);
                let _ = io::stdout().flush();
                break;
            }
        }
        println!();
    }
    
    // REPL
    loop {
        print!("> ");
        let _ = io::stdout().flush();
        
        let mut user_input = String::new();
        if io::stdin().read_line(&mut user_input).is_err() {
            break;
        }
        let trimmed = user_input.trim();
        
        if trimmed == "quit" || trimmed == "exit" {
            break;
        }
        
        if trimmed.is_empty() {
            continue;
        }
        
        pipeline.mingru.hidden_state.data.fill(0.0);
        pipeline.deltanet2.state.fill(0.0);
        
        let formatted_prompt = format!("User: {}\nAgent: ", trimmed);
        let mut last_output = BatchTensor::new(1, vocab_size);
        let encoding = tokenizer.encode(formatted_prompt.as_str(), false).unwrap();
        let tokens = encoding.get_ids();
        for &token_id in tokens {
            last_output = pipeline.forward(&[token_id as usize]).clone();
        }
        
        print!("ANT: ");
        let _ = io::stdout().flush();
        
        let mut generated_history: Vec<usize> = Vec::new();
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut last_decoded_len = 0;

        for _ in 0..100 {
            // Apply repetition penalty from config
            for &prev_token in generated_history.iter().rev().take(chat_cfg.repetition_window) {
                let val = last_output.data.read(0, prev_token);
                last_output.data.write(0, prev_token, val - chat_cfg.repetition_penalty); 
            }

            if let Some(next_token) = sample_token(&last_output, 0.0, chat_cfg.top_k, chat_cfg.temperature) {
                if next_token == 1 || next_token == 0 {
                    println!();
                    break;
                }
                
                generated_history.push(next_token);
                generated_tokens.push(next_token as u32);
                
                let full_text = tokenizer.decode(&generated_tokens, true).unwrap_or_default();
                if full_text.len() > last_decoded_len {
                    let new_part = &full_text[last_decoded_len..];
                    print!("{}", new_part);
                    let _ = io::stdout().flush();
                    last_decoded_len = full_text.len();
                }
                
                if full_text.contains("User:") {
                    println!();
                    break;
                }
                
                last_output = pipeline.forward(&[next_token]).clone();
            } else {
                let mut max_val = last_output.data.read(0, 0);
                for j in 0..vocab_size {
                    let val = last_output.data.read(0, j);
                    if val > max_val { max_val = val; }
                }
                let mut sum_exp = 0.0;
                for j in 0..vocab_size {
                    sum_exp += (last_output.data.read(0, j) - max_val).exp();
                }
                let max_prob = 1.0 / sum_exp;
                println!(" [Unsure/Threshold not met. Max conf: {:.3}]", max_prob);
                break;
            }
        }
    }
}
