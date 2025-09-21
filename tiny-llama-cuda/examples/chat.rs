use std::time::Instant;

use burn::tensor::{backend::Backend, Device};
use clap::Parser;
use tiny_llama_cuda::{
    llama::{TinyLlama, TinyLlamaConfig},
    sampling::{Sampler, TopP},
    tokenizer::Tokenizer,
};

const DEFAULT_PROMPT: &str = "How many helicopters can a human eat in one sitting?";

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Config {
    /// Top-p probability threshold.
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    /// Temperature value for controlling randomness in sampling.
    #[arg(long, default_value_t = 0.6)]
    temperature: f64,

    /// Maximum sequence length for input text.
    #[arg(long, default_value_t = 512)]
    max_seq_len: usize,

    /// The number of new tokens to generate (i.e., the number of generation steps to take).
    #[arg(long, short = 'n', default_value_t = 65)]
    sample_len: usize,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// The input prompt.
    #[arg(short, long, default_value_t = String::from(DEFAULT_PROMPT))]
    prompt: String,

    /// Model checkpoint path (supports .mpk and .safetensors)
    #[arg(long)]
    checkpoint: Option<String>,

    /// Tokenizer path
    #[arg(long)]
    tokenizer: Option<String>,
}

pub fn generate<B: Backend, T: Tokenizer>(
    llama: &mut TinyLlama<B, T>,
    prompt: &str,
    sample_len: usize,
    temperature: f64,
    sampler: &mut Sampler,
) {
    let now = Instant::now();
    let generated = llama.generate(prompt, sample_len, temperature, sampler);
    let elapsed = now.elapsed().as_secs();

    println!("> {}\n", generated.text);
    println!(
        "{} tokens generated ({:.4} tokens/s)\n",
        generated.tokens,
        generated.tokens as f64 / generated.time
    );

    println!(
        "Generation completed in {}m{}s",
        (elapsed / 60),
        elapsed % 60
    );
}

pub fn chat<B: Backend>(args: Config, device: Device<B>) {
    let mut prompt = args.prompt;

    // Sampling strategy
    let mut sampler = if args.temperature > 0.0 {
        Sampler::TopP(TopP::new(args.top_p, args.seed))
    } else {
        Sampler::Argmax
    };

    // Load TinyLlama model
    let mut llama = if let (Some(checkpoint), Some(tokenizer)) = (&args.checkpoint, &args.tokenizer) {
        // Use provided paths
        TinyLlamaConfig::load_auto::<B>(
            checkpoint,
            tokenizer,
            args.max_seq_len,
            &device,
        ).expect("Failed to load TinyLlama model from provided paths")
    } else {
        // Try cached model files
        let cache_dir = dirs::home_dir()
            .expect("Should be able to get home directory")
            .join(".cache")
            .join("llama-burn")
            .join("TinyLlama-1.1B");

        let model_path = cache_dir.join("model.mpk");
        let tokenizer_path = cache_dir.join("tokenizer.json");

        if model_path.exists() && tokenizer_path.exists() {
            TinyLlamaConfig::load_tiny_llama::<B>(
                model_path.to_str().unwrap(),
                tokenizer_path.to_str().unwrap(),
                args.max_seq_len,
                &device,
            ).expect("Failed to load cached TinyLlama model")
        } else {
            // Try safetensors format
            let model_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
                .join("models")
                .join("TinyLlama-1.1B-Chat-v1.0");

            let model_path = model_dir.join("model.safetensors");
            let tokenizer_path = model_dir.join("tokenizer.json");

            if model_path.exists() && tokenizer_path.exists() {
                TinyLlamaConfig::load_tiny_llama_safetensors::<B>(
                    model_path.to_str().unwrap(),
                    tokenizer_path.to_str().unwrap(),
                    args.max_seq_len,
                    &device,
                ).expect("Failed to load safetensors TinyLlama model")
            } else {
                panic!("No TinyLlama model found! Please provide --checkpoint and --tokenizer paths, or ensure model files exist in cache or ~/models/TinyLlama-1.1B-Chat-v1.0/");
            }
        }
    };

    println!("Processing prompt: {}", prompt);

    // Prompt formatting for chat model
    prompt = format!(
        "<|system|>\nYou are a friendly chatbot who always responds in the style of a pirate</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
    );

    generate(
        &mut llama,
        &prompt,
        args.sample_len,
        args.temperature,
        &mut sampler,
    );
}

mod cuda {
    use super::*;
    use burn::{
        backend::{cuda::CudaDevice, Cuda},
        tensor::f16,
    };

    pub fn run(args: Config) {
        let device = CudaDevice::default();
        chat::<Cuda<f16, i32>>(args, device);
    }
}

pub fn main() {
    // Parse arguments
    let args = Config::parse();
    cuda::run(args);
}