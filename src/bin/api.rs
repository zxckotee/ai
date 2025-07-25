// bin/api.rs
// REST API сервер для "Мыслящего Ядро"

use metastasa::api;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    println!("🚀 Запуск REST API сервера Metastasa...");
    println!("📚 Система: Мыслящее Ядро v0.1.0");
    println!("🌐 Адрес: http://127.0.0.1:8080");
    println!("📖 Документация: см. DEVELOPMENT_PLAN.md");
    
    // Запуск API сервера
    api::run_api().await
} 