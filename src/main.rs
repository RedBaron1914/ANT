pub mod format;
pub mod os_windows;
pub mod compute_cpu;
pub mod compute_gpu;
pub mod ant_core;
pub mod training;
pub mod chat;

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelSection {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub hidden_size: usize,
    pub tokenizer_path: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MemorySection {
    pub base_memory_path: String,
    pub user_memory_path: String,
    pub base_capacity: usize,
    pub user_capacity: usize,
    pub top_k_base: usize,
    pub top_k_user: usize,
    pub consolidation_energy: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SessionTapeSection {
    pub capacity: usize,
    pub fifo_window: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainingSection {
    pub dataset_path: String,
    pub model_path: String,
    pub batch_size: usize,
    pub seq_len: usize,
    pub epochs: usize,
    pub force_cpu: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OptimizerSection {
    pub optimizer_type: String,
    pub default_lr: f64,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatSection {
    pub temperature: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub repetition_window: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ContinualLearningSection {
    pub ewc_lambda: f32,
    pub fisher_decay: f32,
    pub lora_rank: usize,
    pub lora_alpha: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AntConfig {
    pub model: ModelSection,
    pub memory: MemorySection,
    pub session_tape: SessionTapeSection,
    pub training: TrainingSection,
    pub optimizer: OptimizerSection,
    pub chat: ChatSection,
    pub continual_learning: ContinualLearningSection,
}

impl Default for AntConfig {
    fn default() -> Self {
        Self {
            model: ModelSection {
                vocab_size: 8000,
                embed_dim: 256,
                hidden_size: 512,
                tokenizer_path: "tokenizer.json".to_string(),
            },
            memory: MemorySection {
                base_memory_path: "base_knowledge.ant".to_string(),
                user_memory_path: "user_experience.ant".to_string(),
                base_capacity: 10000,
                user_capacity: 5000,
                top_k_base: 16,
                top_k_user: 16,
                consolidation_energy: 0.05,
            },
            session_tape: SessionTapeSection {
                capacity: 1000,
                fifo_window: 128,
            },
            training: TrainingSection {
                dataset_path: "datasets/universal_brain.txt".to_string(),
                model_path: "model.bin".to_string(),
                batch_size: 32,
                seq_len: 70,
                epochs: 50,
                force_cpu: false,
            },
            optimizer: OptimizerSection {
                optimizer_type: "lion".to_string(),
                default_lr: 0.0001,
                beta1: 0.9,
                beta2: 0.99,
                weight_decay: 0.01,
            },
            chat: ChatSection {
                temperature: 0.7,
                top_k: 5,
                repetition_penalty: 1.2,
                repetition_window: 15,
            },
            continual_learning: ContinualLearningSection {
                ewc_lambda: 100.0,
                fisher_decay: 0.99,
                lora_rank: 8,
                lora_alpha: 16.0,
            },
        }
    }
}

impl AntConfig {
    pub fn load_or_default() -> Self {
        let path = "ant_config.toml";
        if Path::new(path).exists() {
            let content = fs::read_to_string(path).unwrap();
            toml::from_str(&content).expect("Failed to parse ant_config.toml")
        } else {
            let default_cfg = Self::default();
            let toml_str = toml::to_string_pretty(&default_cfg).unwrap();
            fs::write(path, toml_str).unwrap();
            println!("✨ Created default configuration file: {}", path);
            default_cfg
        }
    }

    pub fn save(&self) {
        let toml_str = toml::to_string_pretty(self).unwrap();
        fs::write("ant_config.toml", toml_str).unwrap();
    }
}

/// ANT-rs: Asynchronous Neural Runtime
#[derive(Parser)]
#[command(name = "ANT-rs", version = "0.3", about = "Cognitive Neuromorphic Runtime")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new project and generate default config
    Init,
    
    /// Train or fine-tune the model
    Train {
        /// Path to training dataset
        #[arg(short, long)]
        data: Option<String>,
        
        /// Number of epochs
        #[arg(short, long)]
        epochs: Option<usize>,
        
        /// Learning rate override
        #[arg(short, long)]
        lr: Option<f64>,
    },
    
    /// Start interactive chat session
    Chat {
        /// Initial prompt
        #[arg(short, long)]
        prompt: Option<String>,
    },
    
    /// Run memory consolidation and deduplication (sleep phase)
    Sleep,

    /// Run full neural integrity and diagnostic test suite
    Test,
}

#[tokio::main]
async fn main() {
    println!("=== ANT-rs Agent Runtime ===");
    os_windows::pin_thread_to_ccd0();

    let cli = Cli::parse();
    let mut config = AntConfig::load_or_default();

    match cli.command {
        Commands::Init => {
            config.save();
            println!("✅ Project initialized! Edit `ant_config.toml` to change base settings.");
        }

        Commands::Train { data, epochs, lr } => {
            if let Some(d) = data { config.training.dataset_path = d; }
            if let Some(e) = epochs { config.training.epochs = e; }
            if let Some(l) = lr { config.optimizer.default_lr = l; }
            config.save();

            let dataset = training::TextDataset::new(&config.training.dataset_path, &config.model.tokenizer_path).expect("Failed to load dataset!");
            
            let mut pipeline = if Path::new(&config.training.model_path).exists() {
                println!("🔄 Found existing model '{}'. Starting Fine-Tuning...", config.training.model_path);
                let header = format::AntHeader::read_from_file(&config.training.model_path)
                    .expect("Failed to read .ant header.");
                
                let mut p = ant_core::pipeline::AntPipeline::new(
                    &config.memory.base_memory_path,
                    &config.memory.user_memory_path,
                    header.vocab_size as usize, header.embed_dim as usize, header.hidden_size as usize,
                    config.training.batch_size, 
                    config.memory.base_capacity, config.memory.user_capacity,
                    config.memory.top_k_base, config.memory.top_k_user,
                    config.memory.consolidation_energy,
                    config.session_tape.capacity, config.session_tape.fifo_window,
                    config.continual_learning.lora_rank, config.continual_learning.lora_alpha,
                    config.training.force_cpu
                ).unwrap();
                p.load_weights(&config.training.model_path).unwrap();
                p
            } else {
                println!("🌱 Initializing FRESH model...");
                ant_core::pipeline::AntPipeline::new(
                    &config.memory.base_memory_path,
                    &config.memory.user_memory_path,
                    dataset.vocab_size, config.model.embed_dim, config.model.hidden_size,
                    config.training.batch_size, 
                    config.memory.base_capacity, config.memory.user_capacity,
                    config.memory.top_k_base, config.memory.top_k_user,
                    config.memory.consolidation_energy,
                    config.session_tape.capacity, config.session_tape.fifo_window,
                    config.continual_learning.lora_rank, config.continual_learning.lora_alpha,
                    config.training.force_cpu
                ).unwrap()
            };

            let tokenizer = tokenizers::Tokenizer::from_file(&config.model.tokenizer_path).unwrap();
            pipeline.negation_ids = ant_core::pipeline::get_negation_ids(&tokenizer);
            pipeline.is_training = true;

            let mut trainer = training::Trainer::new(pipeline, config.optimizer.clone(), config.continual_learning.clone(), config.training.model_path.clone());
            println!("🚀 Training on {} for {} epochs (LR: {})...", config.training.dataset_path, config.training.epochs, config.optimizer.default_lr);
            
            for epoch in 1..=config.training.epochs {
                println!("Epoch {}/{}", epoch, config.training.epochs);
                trainer.train_epoch(&dataset, config.training.seq_len);
                if let Err(e) = trainer.pipeline.save_weights(&config.training.model_path) {
                    println!("⚠️ Failed to save checkpoint: {}", e);
                } else {
                    println!("💾 Checkpoint saved: {}", config.training.model_path);
                }
            }
            println!("✅ Training complete! Weights saved.");
        }

        Commands::Chat { prompt } => {
            let (vocab, embed, hidden) = if Path::new(&config.training.model_path).exists() {
                let header = format::AntHeader::read_from_file(&config.training.model_path)
                    .expect("Failed to read .ant header.");
                (header.vocab_size as usize, header.embed_dim as usize, header.hidden_size as usize)
            } else {
                (config.model.vocab_size, config.model.embed_dim, config.model.hidden_size)
            };

            let mut pipeline = ant_core::pipeline::AntPipeline::new(
                &config.memory.base_memory_path,
                &config.memory.user_memory_path,
                vocab, embed, hidden,
                1, 
                config.memory.base_capacity, config.memory.user_capacity,
                config.memory.top_k_base, config.memory.top_k_user,
                config.memory.consolidation_energy,
                config.session_tape.capacity, config.session_tape.fifo_window,
                config.continual_learning.lora_rank, config.continual_learning.lora_alpha,
                config.training.force_cpu
            ).unwrap();
            
            if Path::new(&config.training.model_path).exists() {
                pipeline.load_weights(&config.training.model_path).unwrap();
            } else {
                println!("⚠️ WARNING: No trained model found at {}. Using random weights!", config.training.model_path);
            }
            
            chat::run_chat(&mut pipeline, prompt.as_deref(), &config.chat, &config.model.tokenizer_path);
        }

        Commands::Sleep => {
            println!("💤 Agent is going to sleep...");
            let (vocab, embed, hidden) = if Path::new(&config.training.model_path).exists() {
                let header = format::AntHeader::read_from_file(&config.training.model_path)
                    .expect("Failed to read .ant header.");
                (header.vocab_size as usize, header.embed_dim as usize, header.hidden_size as usize)
            } else {
                (config.model.vocab_size, config.model.embed_dim, config.model.hidden_size)
            };

            let mut pipeline = ant_core::pipeline::AntPipeline::new(
                &config.memory.base_memory_path,
                &config.memory.user_memory_path,
                vocab, embed, hidden,
                config.training.batch_size, 
                config.memory.base_capacity, config.memory.user_capacity,
                config.memory.top_k_base, config.memory.top_k_user,
                config.memory.consolidation_energy,
                config.session_tape.capacity, config.session_tape.fifo_window,
                config.continual_learning.lora_rank, config.continual_learning.lora_alpha,
                config.training.force_cpu
            ).unwrap();
            pipeline.load_weights(&config.training.model_path).unwrap();

            let dataset = training::TextDataset::new(&config.training.dataset_path, &config.model.tokenizer_path).unwrap();
            let tokenizer = tokenizers::Tokenizer::from_file(&config.model.tokenizer_path).unwrap();
            pipeline.negation_ids = ant_core::pipeline::get_negation_ids(&tokenizer);
            pipeline.is_training = true;

            let mut trainer = training::Trainer::new(pipeline, config.optimizer.clone(), config.continual_learning.clone(), config.training.model_path.clone());

            trainer.sleep_phase(&dataset, config.training.seq_len, 10);
            trainer.pipeline.save_weights(&config.training.model_path).unwrap();
        }

        Commands::Test => {
            crate::ant_core::sanity_check::run_full_diagnostics(&config);
        }
    }
}