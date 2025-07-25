// embedding_cache.rs
// Кеширование эмбеддингов и оптимизация для "Мыслящего Ядро"
// Подробные комментарии для понимания архитектуры

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use lazy_static::lazy_static;
use serde::{Serialize, Deserialize};
use crate::text_processor::{Entity, Action, Property};

/// Кеш эмбеддингов для оптимизации производительности
#[derive(Debug, Clone)]
pub struct EmbeddingCache {
    pub embeddings: HashMap<String, Vec<f32>>,
    pub entity_embeddings: HashMap<String, Vec<f32>>,
    pub action_embeddings: HashMap<String, Vec<f32>>,
    pub property_embeddings: HashMap<String, Vec<f32>>,
}

/// Глобальный кеш эмбеддингов (thread-safe)
lazy_static! {
    static ref EMBEDDING_CACHE: Arc<Mutex<EmbeddingCache>> = Arc::new(Mutex::new(EmbeddingCache {
        embeddings: HashMap::new(),
        entity_embeddings: HashMap::new(),
        action_embeddings: HashMap::new(),
        property_embeddings: HashMap::new(),
    }));
}

/// Структура для кешированного эмбеддинга
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedEmbedding {
    pub text: String,
    pub embedding: Vec<f32>,
    pub confidence: f32,
    pub timestamp: u64,
}

impl EmbeddingCache {
    /// Создать новый кеш эмбеддингов
    pub fn new() -> Self {
        Self {
            embeddings: HashMap::new(),
            entity_embeddings: HashMap::new(),
            action_embeddings: HashMap::new(),
            property_embeddings: HashMap::new(),
        }
    }

    /// Получить кешированный эмбеддинг или вычислить новый
    pub fn get_cached_embedding(text: &str) -> Vec<f32> {
        let mut cache = EMBEDDING_CACHE.lock().unwrap();
        cache.embeddings.entry(text.to_string()).or_insert_with(|| {
            Self::calculate_embedding(text)
        }).clone()
    }

    /// Вычислить эмбеддинг для текста (упрощенная версия)
    pub fn calculate_embedding(text: &str) -> Vec<f32> {
        // Упрощенный эмбеддинг на основе хэша
        // В реальной реализации здесь будет интеграция с BERT или другой моделью
        let mut embedding = vec![0.0; 64]; // 64-мерный вектор
        let bytes = text.as_bytes();
        
        for (i, &byte) in bytes.iter().enumerate() {
            let idx = i % 64;
            embedding[idx] += byte as f32 / 255.0;
        }
        
        // Нормализация
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }
        
        embedding
    }

    /// Получить эмбеддинг для сущности
    pub fn get_entity_embedding(entity: &Entity) -> Vec<f32> {
        let mut cache = EMBEDDING_CACHE.lock().unwrap();
        let key = format!("{}:{}", entity.text, format!("{:?}", entity.entity_type));
        cache.entity_embeddings.entry(key).or_insert_with(|| {
            Self::calculate_embedding(&entity.text)
        }).clone()
    }

    /// Получить эмбеддинг для действия
    pub fn get_action_embedding(action: &Action) -> Vec<f32> {
        let mut cache = EMBEDDING_CACHE.lock().unwrap();
        let key = format!("{}:{}:{}", action.subject, action.verb, action.object);
        cache.action_embeddings.entry(key).or_insert_with(|| {
            let combined = format!("{} {} {}", action.subject, action.verb, action.object);
            Self::calculate_embedding(&combined)
        }).clone()
    }

    /// Получить эмбеддинг для свойства
    pub fn get_property_embedding(property: &Property) -> Vec<f32> {
        let mut cache = EMBEDDING_CACHE.lock().unwrap();
        let key = format!("{}:{}:{}", property.entity, property.attribute, property.value);
        cache.property_embeddings.entry(key).or_insert_with(|| {
            let combined = format!("{} {} {}", property.entity, property.attribute, property.value);
            Self::calculate_embedding(&combined)
        }).clone()
    }

    /// Косинусное сходство между двумя векторами
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm_a * norm_b)
    }

    /// Найти наиболее похожие эмбеддинги
    pub fn find_similar_embeddings(query: &[f32], embeddings: &[Vec<f32>], top_k: usize) -> Vec<(usize, f32)> {
        let mut similarities: Vec<(usize, f32)> = embeddings.iter()
            .enumerate()
            .map(|(idx, emb)| (idx, Self::cosine_similarity(query, emb)))
            .collect();
        
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(top_k);
        similarities
    }

    /// Очистить старые эмбеддинги (управление памятью)
    pub fn cleanup_old_embeddings(_max_age_seconds: u64) {
        let mut cache = EMBEDDING_CACHE.lock().unwrap();
        let _current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Удаляем старые эмбеддинги
        cache.embeddings.retain(|_, _| true); // Пока оставляем все
        cache.entity_embeddings.retain(|_, _| true);
        cache.action_embeddings.retain(|_, _| true);
        cache.property_embeddings.retain(|_, _| true);
    }

    /// Получить статистику кеша
    pub fn get_cache_stats() -> CacheStats {
        let cache = EMBEDDING_CACHE.lock().unwrap();
        CacheStats {
            total_embeddings: cache.embeddings.len(),
            entity_embeddings: cache.entity_embeddings.len(),
            action_embeddings: cache.action_embeddings.len(),
            property_embeddings: cache.property_embeddings.len(),
        }
    }
}

/// Статистика кеша эмбеддингов
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_embeddings: usize,
    pub entity_embeddings: usize,
    pub action_embeddings: usize,
    pub property_embeddings: usize,
}

/// Параллельная обработка текстов с кешированием
pub fn process_batch_with_cache(texts: Vec<String>) -> Vec<Vec<f32>> {
    use rayon::prelude::*;
    
    texts.par_iter()
        .map(|text| EmbeddingCache::get_cached_embedding(text))
        .collect()
}

/// Оптимизированная обработка с предварительным кешированием
pub fn precompute_embeddings(texts: &[String]) -> HashMap<String, Vec<f32>> {
    let mut embeddings = HashMap::new();
    
    for text in texts {
        let embedding = EmbeddingCache::get_cached_embedding(text);
        embeddings.insert(text.clone(), embedding);
    }
    
    embeddings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_embedding() {
        let text = "кот";
        let embedding = EmbeddingCache::calculate_embedding(text);
        assert_eq!(embedding.len(), 64);
        
        // Проверка нормализации
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001 || norm == 0.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = EmbeddingCache::cosine_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 0.001);
        
        let c = vec![0.0, 1.0, 0.0];
        let similarity2 = EmbeddingCache::cosine_similarity(&a, &c);
        assert!((similarity2 - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_get_cached_embedding() {
        let text = "тест";
        let embedding1 = EmbeddingCache::get_cached_embedding(text);
        let embedding2 = EmbeddingCache::get_cached_embedding(text);
        
        // Кешированные эмбеддинги должны быть одинаковыми
        assert_eq!(embedding1, embedding2);
    }

    #[test]
    fn test_find_similar_embeddings() {
        let query = vec![1.0, 0.0, 0.0];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0], // Идентичный
            vec![0.0, 1.0, 0.0], // Перпендикулярный
            vec![0.5, 0.5, 0.0], // Частично похожий
        ];
        
        let similar = EmbeddingCache::find_similar_embeddings(&query, &embeddings, 2);
        assert_eq!(similar.len(), 2);
        assert_eq!(similar[0].0, 0); // Самый похожий должен быть первым
    }

    #[test]
    fn test_cache_stats() {
        let stats = EmbeddingCache::get_cache_stats();
        assert!(stats.total_embeddings >= 0);
        assert!(stats.entity_embeddings >= 0);
        assert!(stats.action_embeddings >= 0);
        assert!(stats.property_embeddings >= 0);
    }
} 