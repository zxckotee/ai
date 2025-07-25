use petgraph::graph::{Graph, NodeIndex};
use sled::Db;
use uuid::Uuid;
use crate::segment::{Segment, KnowledgeNode as SegKnowledgeNode};

/// Граф знаний: petgraph + sled для хранения
pub struct KnowledgeGraph {
    pub graph: Graph<Segment, String>, // Узлы теперь содержат Segment
    pub db: Db,
}

impl KnowledgeGraph {
    /// Создать новый граф знаний
    pub fn new(db: Db) -> Self {
        Self {
            graph: Graph::new(),
            db,
        }
    }

    /// Добавить узел с вложенным сегментом
    pub fn add_node(&mut self, segment: Segment) -> NodeIndex {
        self.graph.add_node(segment)
    }

    /// Добавить ребро между двумя узлами
    pub fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, label: String) {
        self.graph.add_edge(a, b, label);
    }

    /// Найти узел по предикату (например, по id)
    pub fn find_node_by<F>(&self, mut pred: F) -> Option<NodeIndex>
    where
        F: FnMut(&Segment) -> bool,
    {
        self.graph.node_indices().find(|&idx| pred(&self.graph[idx]))
    }

    /// Рекурсивный обход сегмента в узле
    pub fn traverse_segment(&self, idx: NodeIndex) {
        if let Some(segment) = self.graph.node_weight(idx) {
            self.traverse_segment_rec(segment, 0);
        }
    }

    fn traverse_segment_rec(&self, segment: &Segment, depth: usize) {
        if depth > 100 { return; }
        match segment {
            Segment::Primitive(s) => println!("{}Primitive: {}", "  ".repeat(depth), s),
            Segment::Composite(children) => {
                println!("{}Composite:", "  ".repeat(depth));
                for child in children {
                    self.traverse_segment_rec(child, depth + 1);
                }
            },
            Segment::Link(_node) => println!("{}Link to KnowledgeNode", "  ".repeat(depth)),
        }
    }
} 