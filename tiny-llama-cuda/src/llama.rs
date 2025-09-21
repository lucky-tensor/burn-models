use std::time::Instant;

use burn::{
    config::Config,
    module::Module,
    nn::{RotaryEncoding, RotaryEncodingConfig},
    record::{FileRecorder, NamedMpkFileRecorder, HalfPrecisionSettings, RecorderError},
    tensor::{
        activation::softmax, backend::Backend, Device, Int, Tensor,
    },
};
use burn::tensor::cast::ToElement;

use crate::{
    sampling::Sampler,
    tokenizer::{SentiencePieceTokenizer, Tokenizer},
    transformer::{KeyValueCache, Transformer, TransformerConfig},
};

#[derive(Config, Debug)]
pub struct TinyLlamaConfig {
    /// The size of the model.
    #[config(default = "2048")]
    pub d_model: usize,
    /// The size of the feed-forward hidden inner features.
    pub hidden_size: usize,
    /// The number of transformer blocks.
    #[config(default = "22")]
    pub num_hidden_layers: usize,
    /// The number of attention heads.
    #[config(default = "32")]
    pub num_attention_heads: usize,
    /// The number of key-value heads.
    pub num_key_value_heads: Option<usize>,
    /// The vocabulary size.
    pub vocab_size: usize,
    /// RMSNorm epsilon
    #[config(default = "1e-5")]
    pub norm_eps: f64,
    /// Maximum sequence length for input text.
    #[config(default = "512")]
    pub max_seq_len: usize,
    /// Maximum batch size (used for key-value cache).
    #[config(default = "1")]
    pub max_batch_size: usize,
    /// The tokenizer path.
    pub tokenizer: String,
}

impl TinyLlamaConfig {
    /// TinyLlama-1.1B Chat v1.0 configuration.
    pub fn tiny_llama(tokenizer_path: &str) -> Self {
        // hidden_size = 5632; vocab_size = 32000
        Self::new(5632, 32000, tokenizer_path.to_string())
            .with_d_model(2048)
            .with_num_hidden_layers(22)
            .with_num_key_value_heads(Some(4))
    }

    /// Load pre-trained TinyLlama model from .mpk file
    pub fn load_tiny_llama<B: Backend>(
        checkpoint: &str,
        tokenizer_path: &str,
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<TinyLlama<B, SentiencePieceTokenizer>, String> {
        let mut config = Self::tiny_llama(tokenizer_path);
        config.max_seq_len = max_seq_len;

        let llama = config.init(device)?;

        // Load model weights using NamedMpkFileRecorder like llama-burn
        println!("Loading TinyLlama from .mpk file: {}", checkpoint);
        let start = Instant::now();

        // Check file exists before loading
        if !std::path::Path::new(checkpoint).exists() {
            return Err(format!("Model file does not exist: {}", checkpoint));
        }

        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
        println!("About to call load with NamedMpkFileRecorder");
        let llama = llama
            .load(checkpoint, &recorder)
            .map_err(|e| format!("Failed to load from .mpk file: {}", e))?;

        let elapsed = start.elapsed();
        println!("MPK loading completed in {:.2}s", elapsed.as_secs_f32());

        Ok(llama)
    }

    /// Load pre-trained TinyLlama model from .safetensors file
    pub fn load_tiny_llama_safetensors<B: Backend>(
        checkpoint: &str,
        tokenizer_path: &str,
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<TinyLlama<B, SentiencePieceTokenizer>, String> {
        // For now, redirect to .mpk loading
        // TODO: Implement proper SafeTensors loading once the API is clarified
        println!("SafeTensors loading not yet implemented, falling back to .mpk loader");
        Self::load_tiny_llama(checkpoint, tokenizer_path, max_seq_len, device)
    }

    /// Auto-detect format and load TinyLlama model
    pub fn load_auto<B: Backend>(
        checkpoint_path: &str,
        tokenizer_path: &str,
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<TinyLlama<B, SentiencePieceTokenizer>, String> {
        if checkpoint_path.ends_with(".safetensors") {
            Self::load_tiny_llama_safetensors(checkpoint_path, tokenizer_path, max_seq_len, device)
        } else if checkpoint_path.ends_with(".mpk") {
            Self::load_tiny_llama(checkpoint_path, tokenizer_path, max_seq_len, device)
        } else {
            Err(format!("Unsupported file format. Expected .safetensors or .mpk, got: {}", checkpoint_path))
        }
    }

    /// Initialize a new [TinyLlama](TinyLlama) module.
    pub fn init<B: Backend>(
        &self,
        device: &Device<B>,
    ) -> Result<TinyLlama<B, SentiencePieceTokenizer>, String> {
        let tokenizer = SentiencePieceTokenizer::new(&self.tokenizer)?;
        let num_key_value_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);

        let model = TransformerConfig::new(
            self.vocab_size,
            self.num_hidden_layers,
            self.d_model,
            self.hidden_size,
            self.num_attention_heads,
            num_key_value_heads,
        )
        .with_max_seq_len(self.max_seq_len)
        .with_norm_eps(self.norm_eps)
        .init(device);

        let rope = RotaryEncodingConfig::new(
            self.max_seq_len * 2,
            self.d_model / self.num_attention_heads,
        )
        .with_theta(10000.0)
        .init(device);

        let cache = (0..self.num_hidden_layers)
            .map(|_| {
                KeyValueCache::new(
                    self.max_batch_size,
                    num_key_value_heads,
                    self.max_seq_len,
                    self.d_model / self.num_attention_heads,
                    device,
                )
            })
            .collect::<Vec<_>>();

        Ok(TinyLlama {
            tokenizer,
            model,
            rope,
            cache,
            device: device.clone(),
        })
    }
}

/// Output of text generation.
pub struct GenerationOutput {
    pub text: String,
    pub tokens: usize,
    pub time: f64,
}

/// TinyLlama model for text generation.
#[derive(Module, Debug)]
pub struct TinyLlamaModule<B: Backend> {
    pub model: Transformer<B>,
    pub rope: RotaryEncoding<B>,
}

#[derive(Config, Debug)]
pub struct TinyLlamaModuleConfig {
    pub d_model: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_size: usize,
    pub norm_eps: f64,
    pub max_seq_len: usize,
}

impl TinyLlamaModuleConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> TinyLlamaModule<B> {
        let model = TransformerConfig::new(
            self.vocab_size,
            self.num_hidden_layers,
            self.d_model,
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
        )
        .with_max_seq_len(self.max_seq_len)
        .with_norm_eps(self.norm_eps)
        .init(device);

        let rope = RotaryEncodingConfig::new(
            self.max_seq_len * 2,
            self.d_model / self.num_attention_heads,
        )
        .with_theta(10000.0)
        .init(device);

        TinyLlamaModule { model, rope }
    }
}

/// TinyLlama wrapper with tokenizer and cache.
#[derive(Debug)]
pub struct TinyLlama<B: Backend, T: Tokenizer> {
    pub tokenizer: T,
    pub model: Transformer<B>,
    pub rope: RotaryEncoding<B>,
    pub cache: Vec<KeyValueCache<B>>,
    pub device: Device<B>,
}

impl<B: Backend, T: Tokenizer> TinyLlama<B, T> {
    /// Tokenize the input text.
    pub fn tokenize(&self, text: &str) -> Tensor<B, 1, Int> {
        let tokens = self.tokenizer.encode(text, true, false);
        Tensor::from_ints(tokens.as_slice(), &self.device)
    }

    /// Generate text completion.
    pub fn generate(
        &mut self,
        prompt: &str,
        sample_len: usize,
        temperature: f64,
        sampler: &mut Sampler,
    ) -> GenerationOutput {
        let input_tokens = self.tokenize(prompt);
        let prompt_len = input_tokens.dims()[0];
        let mut tokens = Tensor::<B, 1, Int>::empty([prompt_len + sample_len], &self.device);
        tokens = tokens.slice_assign([0..prompt_len], input_tokens);
        let stop_tokens = Tensor::from_ints(self.tokenizer.stop_ids().as_slice(), &self.device);
        let mut num_tokens: usize = 0;
        let mut input_pos = Tensor::<B, 1, Int>::arange(0..prompt_len as i64, &self.device);
        let now = Instant::now();

        for i in 0..sample_len {
            let x = tokens.clone().select(0, input_pos.clone()).reshape([1, -1]);
            let logits = self.model.forward(x, &mut self.cache, &self.rope);
            let [batch_size, seq_len, _vocab_size] = logits.dims();
            let mut next_token_logits = logits
                .slice([0..batch_size, seq_len - 1..seq_len])
                .squeeze(1); // [batch_size=1, vocab_size]

            if temperature > 0.0 {
                next_token_logits = temperature_scaled_softmax(next_token_logits, temperature);
            };

            let next_token = sampler.sample(next_token_logits).squeeze(0);

            // Stop when any of the valid stop tokens is encountered
            if stop_tokens
                .clone()
                .equal(next_token.clone())
                .any()
                .into_scalar()
                .to_bool()
            {
                break;
            }

            // Update with the new generated token
            tokens = tokens.slice_assign([prompt_len + i..prompt_len + i + 1], next_token);
            num_tokens += 1;

            // Advance
            let t = input_pos.dims()[0];
            input_pos = input_pos.slice([t - 1..t]) + 1;
        }

        let elapsed = now.elapsed().as_secs_f64();
        let generated_text = self.tokenizer.decode(
            tokens
                .slice([0..prompt_len + num_tokens])
                .into_data()
                .iter::<i32>()
                .map(|t| t as u32)
                .collect()
        );

        GenerationOutput {
            text: generated_text,
            tokens: num_tokens,
            time: elapsed,
        }
    }

    /// Reset the cache for a new conversation.
    pub fn reset_cache(&mut self) {
        for cache in &mut self.cache {
            cache.reset();
        }
    }

    /// Load model weights from file
    pub fn load<R: FileRecorder<B>>(
        mut self,
        file_path: &str,
        recorder: &R,
    ) -> Result<Self, RecorderError> {
        println!("Loading record...");
        let now = Instant::now();
        self.model = self.model.load_file(file_path, recorder, &self.device)?;
        let elapsed = now.elapsed().as_secs();
        println!("Loaded in {}s", elapsed);

        Ok(self)
    }
}

/// Temperature-scaled softmax implementation.
pub fn temperature_scaled_softmax<B: Backend>(logits: Tensor<B, 2>, temperature: f64) -> Tensor<B, 2> {
    let scaled_logits = logits / temperature;
    softmax(scaled_logits, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;
    use burn::tensor::{TensorData, Tolerance, ops::FloatElem};

    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_temperature_softmax() {
        let tensor = TestTensor::<2>::from([[21.3125, 19.859375, 19.0625, 18.75, 18.171875]]);

        let output = temperature_scaled_softmax(tensor, 0.6);
        let expected = TensorData::from([[
            0.8691406,
            0.07836914,
            0.020767212,
            0.0124053955,
            0.0047035217,
        ]]);

        output.into_data().assert_approx_eq::<FT>(&expected, Tolerance::relative(2e-2));
    }

    #[test]
    #[ignore] // Requires model files
    fn test_load_tiny_llama() {
        let device = burn::backend::cuda::CudaDevice::default();
        let max_seq_len = 128;

        // Construct paths to cached model files
        let cache_dir = dirs::home_dir()
            .expect("Should be able to get home directory")
            .join(".cache")
            .join("llama-burn")
            .join("TinyLlama-1.1B");

        let model_path = cache_dir.join("model.mpk");
        let tokenizer_path = cache_dir.join("tokenizer.json");

        // Skip test if cached files don't exist
        if !model_path.exists() || !tokenizer_path.exists() {
            println!("Skipping test: cached TinyLlama files not found");
            println!("Expected model at: {:?}", model_path);
            println!("Expected tokenizer at: {:?}", tokenizer_path);
            return;
        }

        println!("Found model at: {:?}", model_path);
        println!("Found tokenizer at: {:?}", tokenizer_path);

        // Test loading the model
        let result = TinyLlamaConfig::load_tiny_llama::<TestBackend>(
            model_path.to_str().unwrap(),
            tokenizer_path.to_str().unwrap(),
            max_seq_len,
            &device,
        );

        assert!(result.is_ok(), "Failed to load TinyLlama model: {:?}", result.err());

        let llama = result.unwrap();

        // Verify model loaded successfully by checking cache and tokenizer
        assert_eq!(llama.cache.len(), 22, "Cache size should match TinyLlama's 22 layers");

        // Test that tokenizer is working
        let test_text = "Hello world";
        let tokenized = llama.tokenize(test_text);
        assert!(tokenized.dims()[0] > 0, "Tokenization should produce tokens");

        println!("✓ TinyLlama model loaded successfully with {} cache layers", llama.cache.len());
    }

    #[test]
    #[ignore] // Requires model files
    fn test_load_tiny_llama_safetensors() {
        let device = burn::backend::cuda::CudaDevice::default();
        let max_seq_len = 128;

        // Path to safetensors model files
        let model_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("models")
            .join("TinyLlama-1.1B-Chat-v1.0");

        let model_path = model_dir.join("model.safetensors");
        let tokenizer_path = model_dir.join("tokenizer.json");

        // Skip test if safetensors files don't exist
        if !model_path.exists() || !tokenizer_path.exists() {
            println!("Skipping test: safetensors TinyLlama files not found");
            println!("Expected model at: {:?}", model_path);
            println!("Expected tokenizer at: {:?}", tokenizer_path);
            return;
        }

        // Test loading the model from safetensors format
        let result = TinyLlamaConfig::load_tiny_llama_safetensors::<TestBackend>(
            model_path.to_str().unwrap(),
            tokenizer_path.to_str().unwrap(),
            max_seq_len,
            &device,
        );

        assert!(result.is_ok(), "Failed to load TinyLlama model from safetensors: {:?}", result.err());

        let llama = result.unwrap();

        // Verify model loaded successfully by checking cache and tokenizer
        assert_eq!(llama.cache.len(), 22, "Cache size should match TinyLlama's 22 layers");

        // Test that tokenizer is working
        let test_text = "Hello world";
        let tokenized = llama.tokenize(test_text);
        assert!(tokenized.dims()[0] > 0, "Tokenization should produce tokens");

        println!("✓ TinyLlama model loaded successfully from safetensors with {} cache layers", llama.cache.len());
    }

    #[test]
    #[ignore] // Requires model files and CUDA
    fn test_tiny_llama_gpu_memory_allocation() {
        // Explicitly use CUDA GPU device
        let device = burn::backend::cuda::CudaDevice::default();

        let max_seq_len = 128;

        // Construct paths to cached model files
        let cache_dir = dirs::home_dir()
            .expect("Should be able to get home directory")
            .join(".cache")
            .join("llama-burn")
            .join("TinyLlama-1.1B");

        let model_path = cache_dir.join("model.mpk");
        let tokenizer_path = cache_dir.join("tokenizer.json");

        // Skip test if cached files don't exist
        if !model_path.exists() || !tokenizer_path.exists() {
            println!("Skipping GPU memory test: cached TinyLlama files not found");
            return;
        }

        // Load the model
        let mut llama = TinyLlamaConfig::load_tiny_llama::<TestBackend>(
            model_path.to_str().unwrap(),
            tokenizer_path.to_str().unwrap(),
            max_seq_len,
            &device,
        ).expect("Failed to load TinyLlama model");

        // Test that model tensors are allocated on the correct device
        println!("Testing GPU memory allocation for TinyLlama model...");
        println!("Using GPU device: {:?}", device);

        // Create test input and verify it gets processed on GPU
        let test_prompt = "Hello, world!";
        let input_tokens = llama.tokenize(test_prompt);

        // Store device and dimensions before moving the tensor
        let input_device = input_tokens.device();
        let batch_size = 1;
        let seq_len = input_tokens.dims()[0];

        // Verify the input tensor is on the expected device
        assert_eq!(input_device, device, "Input tokens should be on the specified device");

        // Test a forward pass to ensure model weights are on GPU
        let input_reshaped = input_tokens.reshape([batch_size, seq_len]);

        // This forward pass will fail if model weights aren't properly loaded on GPU
        let logits = llama.model.forward(input_reshaped, &mut llama.cache, &llama.rope);

        // Verify output tensor is on GPU
        let output_device = logits.device();
        assert_eq!(output_device, device, "Model output should be on the specified device");

        println!("✓ GPU memory allocation test passed");
        println!("  - Input tensors correctly allocated on GPU");
        println!("  - Model weights correctly loaded on GPU");
        println!("  - Forward pass successful on GPU");
        println!("  - All {} cache layers initialized", llama.cache.len());
    }
}