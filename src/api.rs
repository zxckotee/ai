use actix_web::{web, App, HttpServer, Responder, HttpResponse};
use crate::trusted_scraper::TrustedScraper;
use crate::active_learning::{ActiveLearning, Correction};
use crate::knowledge_graph::KnowledgeGraph;
use crate::text_processor::TextProcessor;
use crate::embedding_cache::EmbeddingCache;
use serde::{Serialize, Deserialize};
use std::sync::Mutex;
use petgraph::visit::EdgeRef;

/// Структура для запроса обработки текста
#[derive(Deserialize)]
pub struct ProcessTextRequest {
    pub text: String,
    pub learn: Option<bool>,
    pub annotation: Option<String>,
}

/// Структура для запроса обучения
#[derive(Deserialize)]
pub struct LearnRequest {
    pub text: String,
    pub annotation: String,
    pub user: String,
}

/// Структура для запроса верификации
#[derive(Deserialize)]
pub struct VerifyRequest {
    pub fact: String,
    pub source: Option<String>,
}

/// Расширенная структура для экспорта графа
#[derive(Serialize)]
struct GraphExport {
    nodes: Vec<String>,
    edges: Vec<(usize, usize, String)>,
    stats: GraphStats,
}

/// Статистика графа
#[derive(Serialize)]
struct GraphStats {
    total_nodes: usize,
    total_edges: usize,
    entity_types: Vec<String>,
    action_types: Vec<String>,
}

/// Обработчик для обработки текста
async fn process_text(
    processor: web::Data<Mutex<TextProcessor>>,
    req: web::Json<ProcessTextRequest>,
) -> impl Responder {
    let mut processor = processor.lock().unwrap();
    
    match processor.process_text(&req.text).await {
        result => {
            // Если запрошено обучение
            if req.learn.unwrap_or(false) {
                let annotation = req.annotation.as_deref().unwrap_or("Автоматическое обучение");
                let learned = processor.learn_from_text(&req.text, annotation).await;
                
                HttpResponse::Ok().json(serde_json::json!({
                    "success": true,
                    "result": result,
                    "learned": learned,
                    "message": if learned { "Факт добавлен в граф знаний" } else { "Факт отправлен на модерацию" }
                }))
            } else {
                HttpResponse::Ok().json(serde_json::json!({
                    "success": true,
                    "result": result
                }))
            }
        }
    }
}

/// Обработчик для обучения на новых данных
async fn learn_from_text(
    processor: web::Data<Mutex<TextProcessor>>,
    req: web::Json<LearnRequest>,
) -> impl Responder {
    let mut processor = processor.lock().unwrap();
    
    match processor.learn_from_text(&req.text, &req.annotation).await {
        true => HttpResponse::Ok().json(serde_json::json!({
            "success": true,
            "message": "Факт успешно добавлен в граф знаний",
            "text": req.text,
            "annotation": req.annotation
        })),
        false => HttpResponse::Accepted().json(serde_json::json!({
            "success": false,
            "message": "Факт отправлен на модерацию",
            "text": req.text,
            "annotation": req.annotation
        }))
    }
}

/// Обработчик для верификации фактов
async fn verify_fact(
    processor: web::Data<Mutex<TextProcessor>>,
    req: web::Json<VerifyRequest>,
) -> impl Responder {
    let processor = processor.lock().unwrap();
    
    match processor.verify_fact(&req.fact).await {
        true => HttpResponse::Ok().json(serde_json::json!({
            "success": true,
            "verified": true,
            "fact": req.fact,
            "message": "Факт верифицирован через TrustedScraper"
        })),
        false => HttpResponse::Ok().json(serde_json::json!({
            "success": true,
            "verified": false,
            "fact": req.fact,
            "message": "Факт не найден в доверенных источниках"
        }))
    }
}

/// Обработчик для проверки фактов (старый эндпоинт)
async fn check_fact(scraper: web::Data<TrustedScraper>, info: web::Query<Correction>) -> impl Responder {
    let al = ActiveLearning::new(&scraper);
    let res = al.check_correction(&info.into_inner()).await;
    match res {
        crate::active_learning::CorrectionResult::AutoAccepted => HttpResponse::Ok().body("Accepted automatically"),
        crate::active_learning::CorrectionResult::NeedsModeration => HttpResponse::Ok().body("Needs moderation"),
        crate::active_learning::CorrectionResult::Rejected => HttpResponse::Ok().body("Rejected"),
    }
}

/// Обработчик для модерации (заглушка)
async fn moderate_fact(_scraper: web::Data<TrustedScraper>, _info: web::Query<Correction>) -> impl Responder {
    HttpResponse::Ok().body("Moderation interface (TODO)")
}

/// Обработчик для экспорта графа знаний
async fn export_graph(graph: web::Data<Mutex<KnowledgeGraph>>) -> impl Responder {
    let graph = graph.lock().unwrap();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut entity_types = Vec::new();
    let mut action_types = Vec::new();
    
    for idx in graph.graph.node_indices() {
        let seg = &graph.graph[idx];
        nodes.push(format!("{:?}", seg));
        
        // Анализ типов узлов
        match seg {
            crate::segment::Segment::Primitive(text) => {
                if text.contains("кот") || text.contains("рыба") {
                    entity_types.push("животное".to_string());
                }
            },
            crate::segment::Segment::Composite(_) => {
                action_types.push("действие".to_string());
            },
            _ => {}
        }
    }
    
    for edge in graph.graph.edge_references() {
        let (a, b) = (edge.source().index(), edge.target().index());
        edges.push((a, b, edge.weight().clone()));
    }
    
    let stats = GraphStats {
        total_nodes: nodes.len(),
        total_edges: edges.len(),
        entity_types,
        action_types,
    };
    
    let export = GraphExport { nodes, edges, stats };
    HttpResponse::Ok().json(export)
}

/// Обработчик для статистики кеша эмбеддингов
async fn cache_stats() -> impl Responder {
    let stats = EmbeddingCache::get_cache_stats();
    HttpResponse::Ok().json(stats)
}

/// Обработчик для очистки кеша
async fn clear_cache() -> impl Responder {
    EmbeddingCache::cleanup_old_embeddings(0); // Очистить все
    HttpResponse::Ok().json(serde_json::json!({
        "success": true,
        "message": "Кеш эмбеддингов очищен"
    }))
}

/// Обработчик для поиска похожих эмбеддингов
#[derive(Deserialize)]
struct SimilarityRequest {
    text: String,
    top_k: Option<usize>,
}

async fn find_similar(
    req: web::Json<SimilarityRequest>,
) -> impl Responder {
    let query_embedding = EmbeddingCache::get_cached_embedding(&req.text);
    let top_k = req.top_k.unwrap_or(5);
    
    // Пример поиска похожих текстов (в реальности здесь будет база данных)
    let sample_embeddings = vec![
        EmbeddingCache::get_cached_embedding("кот"),
        EmbeddingCache::get_cached_embedding("кошка"),
        EmbeddingCache::get_cached_embedding("собака"),
        EmbeddingCache::get_cached_embedding("рыба"),
    ];
    
    let similar = EmbeddingCache::find_similar_embeddings(&query_embedding, &sample_embeddings, top_k);
    
    let results: Vec<serde_json::Value> = similar.iter().map(|(idx, similarity)| {
        serde_json::json!({
            "index": idx,
            "similarity": similarity,
            "text": match *idx {
                0 => "кот",
                1 => "кошка", 
                2 => "собака",
                3 => "рыба",
                _ => "неизвестно"
            }
        })
    }).collect();
    
    HttpResponse::Ok().json(serde_json::json!({
        "query": req.text,
        "top_k": top_k,
        "results": results
    }))
}

/// Обработчик для здоровья системы
async fn health_check() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "version": "0.1.0",
        "components": {
            "text_processor": "active",
            "embedding_cache": "active", 
            "knowledge_graph": "active",
            "trusted_scraper": "active"
        }
    }))
}

/// Обработчик для информации о системе
async fn system_info() -> impl Responder {
    let cache_stats = EmbeddingCache::get_cache_stats();
    
    HttpResponse::Ok().json(serde_json::json!({
        "system": "Metastasa - Мыслящее Ядро",
        "version": "0.1.0",
        "architecture": {
            "language": "Rust",
            "framework": "actix-web",
            "gpu": "Burn WGPU",
            "storage": "sled + petgraph"
        },
        "cache_stats": cache_stats,
        "features": [
            "GPU Attention",
            "Логические графы знаний", 
            "TrustedScraper",
            "Active Learning",
            "Кеширование эмбеддингов"
        ]
    }))
}

pub async fn run_api() -> std::io::Result<()> {
    let scraper = TrustedScraper::new();
    let graph = web::Data::new(Mutex::new(KnowledgeGraph::new(sled::Config::new().temporary(true).open().unwrap())));
    let processor = web::Data::new(Mutex::new(TextProcessor::new()));
    
    println!("🚀 Запуск REST API сервера на http://127.0.0.1:8080");
    println!("📚 Доступные эндпоинты:");
    println!("  POST /process    - Обработка текста");
    println!("  POST /learn      - Обучение на новых данных");
    println!("  POST /verify     - Верификация фактов");
    println!("  GET  /graph      - Экспорт графа знаний");
    println!("  GET  /cache      - Статистика кеша");
    println!("  POST /clear      - Очистка кеша");
    println!("  GET  /similar    - Поиск похожих эмбеддингов");
    println!("  GET  /health     - Проверка здоровья");
    println!("  GET  /info       - Информация о системе");
    
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(scraper.clone()))
            .app_data(graph.clone())
            .app_data(processor.clone())
            .route("/process", web::post().to(process_text))
            .route("/learn", web::post().to(learn_from_text))
            .route("/verify", web::post().to(verify_fact))
            .route("/check", web::get().to(check_fact))
            .route("/moderate", web::get().to(moderate_fact))
            .route("/graph", web::get().to(export_graph))
            .route("/cache", web::get().to(cache_stats))
            .route("/clear", web::post().to(clear_cache))
            .route("/similar", web::post().to(find_similar))
            .route("/health", web::get().to(health_check))
            .route("/info", web::get().to(system_info))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
} 