mod attention_gpu;
mod attention_logic;
mod knowledge_graph;
mod trusted_scraper;
mod segment;
mod text_processor;
mod embedding_cache;

use burn_tensor::{Tensor as BurnTensor, Distribution};
use burn_wgpu::{Wgpu, WgpuDevice};
use knowledge_graph::KnowledgeGraph;
use sled;
use tokio::runtime::Runtime;
use segment::Segment;
use text_processor::TextProcessor;
use embedding_cache::EmbeddingCache;

#[tokio::main]
async fn main() {
    println!("=== Metastasa - AI Knowledge System ===");
    
    // 1. Attention на GPU (Burn)
    println!("\n=== Attention на GPU (Burn) ===");
    let device = WgpuDevice::default();
    let query = BurnTensor::<Wgpu, 2>::random([1, 64], Distribution::Default, &device);
    let keys = BurnTensor::<Wgpu, 2>::random([10, 64], Distribution::Default, &device);
    let scores = attention_gpu::gpu_attention(&query, &keys);
    println!("GPU Attention scores: {:?}", scores);

    // 2. Логический attention (Burn)
    println!("\n=== Логический attention (Burn) ===");
    let query_logic = BurnTensor::<Wgpu, 2>::random([1, 8], Distribution::Default, &device);
    let keys_logic = BurnTensor::<Wgpu, 3>::random([1, 4, 8], Distribution::Default, &device);
    let logic_scores = attention_logic::logical_attention(&query_logic, &keys_logic);
    println!("Logic Attention scores: {:?}", logic_scores);

    // 3. Граф знаний (petgraph + sled) с вложенными сегментами
    println!("\n=== Граф знаний (petgraph + sled, вложенные сегменты) ===");
    let db = sled::Config::new().temporary(true).open().expect("sled db");
    let mut kg = KnowledgeGraph::new(db);
    // Пример вложенной структуры: Кот -> [Голова -> [Усы, Глаза], Лапы -> [Когти]]
    let cat = Segment::Composite(vec![
        Segment::Composite(vec![
            Segment::Primitive("Усы".into()),
            Segment::Primitive("Глаза".into()),
        ]),
        Segment::Primitive("Лапы".into()),
    ]);
    let cat_idx = kg.add_node(cat);
    println!("KnowledgeGraph создан: {} узлов", kg.graph.node_count());
    println!("Обход вложенных сегментов:");
    kg.traverse_segment(cat_idx);

    // 4. TrustedScraper (асинхронная проверка)
    println!("\n=== TrustedScraper (async) ===");
    let scraper = trusted_scraper::TrustedScraper::new();
    let result = scraper.check("Коты имеют 4 ноги").await;
    println!("TrustedScraper результат: {}", result);
    
    // 5. Обработка текста (новый функционал)
    println!("\n=== Обработка текста (TextProcessor) ===");
    let mut processor = TextProcessor::new();
    let text = "Кот ест рыбу. Рыба водится в море.";
    let result = processor.process_text(text).await;
    println!("Обработано предложений: {}", result.entities.len());
    println!("Найдено действий: {}", result.actions.len());
    println!("Найдено свойств: {}", result.properties.len());
    println!("Создано узлов графа: {}", result.graph_nodes.len());
    println!("Создано связей: {}", result.graph_edges.len());
    
    // 6. Кеширование эмбеддингов (новый функционал)
    println!("\n=== Кеширование эмбеддингов ===");
    let embedding1 = EmbeddingCache::get_cached_embedding("кот");
    let embedding2 = EmbeddingCache::get_cached_embedding("кошка");
    let similarity = EmbeddingCache::cosine_similarity(&embedding1, &embedding2);
    println!("Сходство 'кот' и 'кошка': {:.3}", similarity);
    
    let cache_stats = EmbeddingCache::get_cache_stats();
    println!("Статистика кеша:");
    println!("  Всего эмбеддингов: {}", cache_stats.total_embeddings);
    println!("  Эмбеддингов сущностей: {}", cache_stats.entity_embeddings);
    println!("  Эмбеддингов действий: {}", cache_stats.action_embeddings);
    println!("  Эмбеддингов свойств: {}", cache_stats.property_embeddings);
    
    // 7. Обучение на новых данных
    println!("\n=== Обучение на новых данных ===");
    let learned = processor.learn_from_text("Квадрокоптер — это летательный аппарат с 4 моторами", "Уточнение: пропеллеры").await;
    println!("Результат обучения: {}", if learned { "✅ Добавлено в граф" } else { "⚠️ Отправлено на модерацию" });
    
    // 8. Поиск похожих эмбеддингов
    println!("\n=== Поиск похожих эмбеддингов ===");
    let query_embedding = EmbeddingCache::get_cached_embedding("животное");
    let sample_embeddings = vec![
        EmbeddingCache::get_cached_embedding("кот"),
        EmbeddingCache::get_cached_embedding("собака"),
        EmbeddingCache::get_cached_embedding("рыба"),
        EmbeddingCache::get_cached_embedding("дерево"),
    ];
    let similar = EmbeddingCache::find_similar_embeddings(&query_embedding, &sample_embeddings, 3);
    println!("Наиболее похожие на 'животное':");
    for (idx, similarity) in similar {
        let text = match idx {
            0 => "кот",
            1 => "собака",
            2 => "рыба",
            3 => "дерево",
            _ => "неизвестно"
        };
        println!("  {}: {:.3}", text, similarity);
    }
    
    println!("\n=== Все компоненты работают! ===");
    println!("🚀 Система готова к использованию!");
    println!("📚 Для запуска REST API используйте: cargo run --bin api");
}
