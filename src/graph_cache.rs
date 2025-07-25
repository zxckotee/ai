use std::sync::Arc;
use lru::LruCache;
use crate::knowledge_graph::KnowledgeGraph;
use std::num::NonZeroUsize;

/// Кеш подграфов знаний (по теме)
pub struct GraphCache {
    pub cache: LruCache<String, Arc<KnowledgeGraph>>,
    pub pending_updates: Vec<String>, // Для синхронизации с основным графом
}

impl GraphCache {
    pub fn new(size: usize) -> Self {
        Self {
            cache: LruCache::new(NonZeroUsize::new(size).unwrap()),
            pending_updates: Vec::new(),
        }
    }
    /// Получить подграф по ключу (теме)
    pub fn get(&mut self, key: &str) -> Option<Arc<KnowledgeGraph>> {
        self.cache.get(key).cloned()
    }
    /// Добавить/обновить подграф
    pub fn put(&mut self, key: String, graph: Arc<KnowledgeGraph>) {
        self.cache.put(key, graph);
    }
} 