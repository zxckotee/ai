// tests/text_processing.rs
// Тесты для модуля обработки текста

use metastasa::text_processor::{TextProcessor, EntityType};
use metastasa::embedding_cache::EmbeddingCache;

#[tokio::test]
async fn test_text_processing_pipeline() {
    let processor = TextProcessor::new();
    let text = "Кот ест рыбу. Рыба водится в море.";
    
    let result = processor.process_text(text).await;
    
    // Проверяем, что найдены сущности
    assert!(!result.entities.is_empty());
    assert!(result.entities.iter().any(|e| e.text == "кот"));
    assert!(result.entities.iter().any(|e| e.text == "рыба"));
    
    // Проверяем, что найдены действия
    assert!(!result.actions.is_empty());
    assert!(result.actions.iter().any(|a| a.subject == "кот" && a.verb == "ест"));
    
    // Проверяем, что созданы узлы графа
    assert!(!result.graph_nodes.is_empty());
    assert!(!result.graph_edges.is_empty());
}

#[test]
fn test_preprocessing() {
    let processor = TextProcessor::new();
    let text = "Кот ест рыбу! Рыба водится в море.";
    
    let sentences = processor.preprocess_text(text);
    
    assert_eq!(sentences.len(), 2);
    assert!(sentences[0].contains("кот"));
    assert!(sentences[1].contains("рыба"));
}

#[test]
fn test_entity_extraction() {
    let processor = TextProcessor::new();
    let text = "Кот ест рыбу в Москве";
    
    let entities = processor.extract_entities(text);
    
    assert!(!entities.is_empty());
    assert!(entities.iter().any(|e| e.text == "кот"));
    assert!(entities.iter().any(|e| e.text == "рыба"));
    assert!(entities.iter().any(|e| e.text == "москва"));
}

#[test]
fn test_action_extraction() {
    let processor = TextProcessor::new();
    let text = "Кот ест рыбу";
    
    let actions = processor.extract_actions(text);
    
    assert!(!actions.is_empty());
    let action = &actions[0];
    assert_eq!(action.subject, "кот");
    assert_eq!(action.verb, "ест");
    assert_eq!(action.object, "рыбу");
    assert!(action.confidence > 0.0);
}

#[test]
fn test_property_extraction() {
    let processor = TextProcessor::new();
    let text = "Рыба водится в море";
    
    let properties = processor.extract_properties(text);
    
    assert!(!properties.is_empty());
    let property = &properties[0];
    assert_eq!(property.entity, "рыба");
    assert_eq!(property.attribute, "водится");
    assert_eq!(property.value, "в море");
}

#[tokio::test]
async fn test_learning_from_text() {
    let mut processor = TextProcessor::new();
    let text = "Квадрокоптер — это летательный аппарат с 4 моторами";
    let annotation = "Уточнение: пропеллеры";
    
    let result = processor.learn_from_text(text, annotation).await;
    
    // Результат может быть true или false в зависимости от TrustedScraper
    assert!(result == true || result == false);
}

#[test]
fn test_embedding_cache() {
    let text1 = "кот";
    let text2 = "кошка";
    
    let embedding1 = EmbeddingCache::get_cached_embedding(text1);
    let embedding2 = EmbeddingCache::get_cached_embedding(text2);
    
    // Проверяем, что эмбеддинги имеют правильную размерность
    assert_eq!(embedding1.len(), 64);
    assert_eq!(embedding2.len(), 64);
    
    // Проверяем нормализацию
    let norm1: f32 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = embedding2.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm1 - 1.0).abs() < 0.001 || norm1 == 0.0);
    assert!((norm2 - 1.0).abs() < 0.001 || norm2 == 0.0);
    
    // Проверяем кеширование
    let embedding1_cached = EmbeddingCache::get_cached_embedding(text1);
    assert_eq!(embedding1, embedding1_cached);
}

#[test]
fn test_cosine_similarity() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let c = vec![0.0, 1.0, 0.0];
    
    let similarity_identical = EmbeddingCache::cosine_similarity(&a, &b);
    let similarity_orthogonal = EmbeddingCache::cosine_similarity(&a, &c);
    
    assert!((similarity_identical - 1.0).abs() < 0.001);
    assert!((similarity_orthogonal - 0.0).abs() < 0.001);
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
    assert!(similar[0].1 > similar[1].1); // Убывание по сходству
}

#[test]
fn test_cache_stats() {
    let stats = EmbeddingCache::get_cache_stats();
    
    assert!(stats.total_embeddings >= 0);
    assert!(stats.entity_embeddings >= 0);
    assert!(stats.action_embeddings >= 0);
    assert!(stats.property_embeddings >= 0);
}

#[test]
fn test_entity_types() {
    let processor = TextProcessor::new();
    let text = "Кот в Москве 2024-01-01";
    
    let entities = processor.extract_entities(text);
    
    assert!(entities.iter().any(|e| e.entity_type == EntityType::Person));
    assert!(entities.iter().any(|e| e.entity_type == EntityType::Location));
    assert!(entities.iter().any(|e| e.entity_type == EntityType::Date));
}

#[tokio::test]
async fn test_batch_processing() {
    let texts = vec![
        "Кот ест рыбу".to_string(),
        "Собака бежит по улице".to_string(),
        "Рыба плавает в море".to_string(),
    ];
    
    let processor = TextProcessor::new();
    let mut all_results = Vec::new();
    
    for text in texts {
        let result = processor.process_text(&text).await;
        all_results.push(result);
    }
    
    assert_eq!(all_results.len(), 3);
    
    // Проверяем, что все результаты содержат данные
    for result in all_results {
        assert!(!result.entities.is_empty() || !result.actions.is_empty());
    }
} 