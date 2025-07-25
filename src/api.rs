use actix_web::{web, App, HttpServer, Responder, HttpResponse};
use crate::trusted_scraper::TrustedScraper;
use crate::active_learning::{ActiveLearning, Correction};
use crate::knowledge_graph::KnowledgeGraph;
use serde::Serialize;
use std::sync::Mutex;
use petgraph::visit::EdgeRef;

async fn check_fact(scraper: web::Data<TrustedScraper>, info: web::Query<Correction>) -> impl Responder {
    let al = ActiveLearning::new(&scraper);
    let res = al.check_correction(&info.into_inner()).await;
    match res {
        crate::active_learning::CorrectionResult::AutoAccepted => HttpResponse::Ok().body("Accepted automatically"),
        crate::active_learning::CorrectionResult::NeedsModeration => HttpResponse::Ok().body("Needs moderation"),
        crate::active_learning::CorrectionResult::Rejected => HttpResponse::Ok().body("Rejected"),
    }
}

async fn moderate_fact(_scraper: web::Data<TrustedScraper>, _info: web::Query<Correction>) -> impl Responder {
    // TODO: реализовать модерацию
    HttpResponse::Ok().body("Moderation interface (TODO)")
}

#[derive(Serialize)]
struct GraphExport {
    nodes: Vec<String>,
    edges: Vec<(usize, usize, String)>,
}

async fn export_graph(graph: web::Data<Mutex<KnowledgeGraph>>) -> impl Responder {
    let graph = graph.lock().unwrap();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for idx in graph.graph.node_indices() {
        let seg = &graph.graph[idx];
        nodes.push(format!("{:?}", seg));
    }
    for edge in graph.graph.edge_references() {
        let (a, b) = (edge.source().index(), edge.target().index());
        edges.push((a, b, edge.weight().clone()));
    }
    let export = GraphExport { nodes, edges };
    HttpResponse::Ok().json(export)
}

pub async fn run_api() -> std::io::Result<()> {
    let scraper = TrustedScraper::new();
    let graph = web::Data::new(Mutex::new(KnowledgeGraph::new(sled::Config::new().temporary(true).open().unwrap())));
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(scraper.clone()))
            .app_data(graph.clone())
            .route("/check", web::get().to(check_fact))
            .route("/moderate", web::get().to(moderate_fact))
            .route("/graph", web::get().to(export_graph))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
} 