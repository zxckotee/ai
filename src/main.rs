mod attention_gpu;
mod attention_logic;
mod knowledge_graph;
mod trusted_scraper;
mod segment;

use burn_tensor::{Tensor as BurnTensor, Distribution};
use burn_wgpu::{Wgpu, WgpuDevice};
use knowledge_graph::KnowledgeGraph;
use sled;
use tokio::runtime::Runtime;
use segment::Segment;

fn main() {
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
    let rt = Runtime::new().unwrap();
    rt.block_on(async {
        let scraper = trusted_scraper::TrustedScraper::new();
        let result = scraper.check("Коты имеют 4 ноги").await;
        println!("TrustedScraper результат: {}", result);
    });
    
    println!("\n=== Все компоненты работают! ===");
}
