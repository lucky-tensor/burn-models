pub mod cache;
pub mod llama;
pub mod tokenizer;
pub mod transformer;
pub mod sampling;

pub use llama::TinyLlama;
pub use tokenizer::SentiencePieceTokenizer;
pub use sampling::{Sampler, TopP};

// Type alias for CUDA backend with f16 precision
pub type CudaBackend = burn::backend::Cuda<burn::tensor::f16, i32>;

#[cfg(test)]
mod tests {
    use burn::{backend::Cuda, tensor::f16};
    pub type TestBackend = Cuda<f16, i32>;

    pub type TestTensor<const D: usize> = burn::tensor::Tensor<TestBackend, D>;
}
