use burn_tensor::Tensor;
use burn_tensor::activation::softmax;
use burn_wgpu::Wgpu;

/// Логический attention на Burn (CPU/WGPU)
pub fn logical_attention(query: &Tensor<Wgpu, 2>, keys: &Tensor<Wgpu, 3>) -> Tensor<Wgpu, 2> {
    let query = query.clone().unsqueeze(); // [1, batch, hidden]
    let keys = keys.clone().permute([0, 2, 1]); // [batch, hidden, seq_len]
    let scores = query.matmul(keys); // [batch, 1, seq_len]
    let scores = scores.squeeze(1); // [batch, seq_len]
    softmax(scores, 1)
} 