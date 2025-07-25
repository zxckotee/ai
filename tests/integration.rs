use metastasa::trusted_scraper::TrustedScraper;
use metastasa::active_learning::{ActiveLearning, Correction, CorrectionResult};
use metastasa::logic_attention::{logical_attention};
use metastasa::segment::{Segment, KnowledgeNode};
use metastasa::graph_cache::GraphCache;
use std::sync::Arc;

#[tokio::test]
async fn test_trusted_scraper_and_active_learning() {
    let scraper = TrustedScraper::new();
    let al = ActiveLearning::new(&scraper);
    let corr = Correction {
        claim: "Кошка".to_string(),
        user: "user1".to_string(),
        justification: "Вижу в энциклопедии".to_string(),
    };
    let res = al.check_correction(&corr).await;
    assert!(matches!(res, CorrectionResult::NeedsModeration | CorrectionResult::AutoAccepted));
}

#[test]
fn test_logical_attention() {
    let query = Segment::Primitive("Кошка".into());
    let node = KnowledgeNode {
        id: uuid::Uuid::new_v4(),
        data: Segment::Primitive("Кошка".into()),
        depth: 0,
        edges: vec![],
    };
    let scores = logical_attention(&query, &[node]);
    assert_eq!(scores.len(), 1);
}

#[test]
fn test_graph_cache() {
    let mut cache = GraphCache::new(2);
    let graph = Arc::new(metastasa::knowledge_graph::KnowledgeGraph::new(sled::Config::new().temporary(true).open().unwrap()));
    cache.put("biology.cats".to_string(), graph.clone());
    let got = cache.get("biology.cats");
    assert!(got.is_some());
}

#[test]
fn test_logic_rule_lapa_animal() {
    let query = Segment::Primitive("лапа".into());
    let node = KnowledgeNode {
        id: uuid::Uuid::new_v4(),
        data: Segment::Primitive("кот".into()),
        depth: 0,
        edges: vec![],
        tags: vec!["животное".into()],
    };
    let score = metastasa::logic_attention::check_logic_rules(&query, &node);
    assert_eq!(score, 1.0);
} 