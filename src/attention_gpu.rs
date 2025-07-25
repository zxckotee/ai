use burn_tensor::{Tensor as BurnTensor, Distribution, activation::softmax};
use burn_wgpu::{Wgpu, WgpuDevice};

/// Attention на GPU с помощью Burn
pub fn gpu_attention(query: &BurnTensor<Wgpu, 2>, keys: &BurnTensor<Wgpu, 2>) -> BurnTensor<Wgpu, 2> {
    // query: [batch, hidden], keys: [seq_len, hidden]
    // Attention: query * keys^T -> softmax
    let device = WgpuDevice::default();
    
    // Клонируем тензоры для операций, которые потребляют владение
    let keys_t = keys.clone().transpose();
    
    // Вычисляем attention scores (клонируем query для matmul)
    let scores = query.clone().matmul(keys_t);
    
    // Применяем softmax по последней размерности
    // Используем правильный импорт из burn_tensor::activation
    softmax(scores, 1) // 1 - это последняя размерность (индексация с 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_tensor::Distribution;
    
    #[test]
    fn test_gpu_attention() {
        let device = WgpuDevice::default();
        let query = BurnTensor::<Wgpu, 2>::random([1, 64], Distribution::Default, &device);
        let keys = BurnTensor::<Wgpu, 2>::random([10, 64], Distribution::Default, &device);
        let scores = gpu_attention(&query, &keys);
        assert_eq!(scores.shape(), [1, 10]);
    }
} 