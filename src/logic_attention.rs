use crate::segment::{Segment, KnowledgeNode};

/// Пример логического attention: совмещает эмбеддинги и логику
pub fn logical_attention(query: &Segment, nodes: &[KnowledgeNode]) -> Vec<f32> {
    nodes.iter().map(|node| {
        // 1. Проверка прямых связей (если query — часть node)
        if node.data == *query {
            return 1.0;
        }
        // 2. Семантическое сходство (заглушка, обычно cosine_similarity)
        let sim = 0.5; // TODO: интеграция с FastText/эмбеддингами
        let logic_score = check_logic_rules(query, node);
        (sim + logic_score) / 2.0
    }).collect()
}

/// Расширяемые логические правила: пары (ключевое слово, тег)
const LOGIC_RULES: &[(&str, &str)] = &[
    ("лапа", "животное"),
    ("крыло", "птица"),
    ("корень", "растение"),
    ("плавник", "рыба"),
];

/// Пример логического правила (расширено)
pub fn check_logic_rules(query: &Segment, node: &KnowledgeNode) -> f32 {
    let query_text = segment_to_text(query);
    let node_tags = &node.tags;
    for (keyword, tag) in LOGIC_RULES {
        if query_text.contains(keyword) && node_tags.iter().any(|t| t == tag) {
            return 1.0;
        }
    }
    0.5
}

/// Вспомогательная функция: извлечь текст из Segment (упрощённо)
fn segment_to_text(segment: &Segment) -> String {
    match segment {
        Segment::Primitive(s) => s.clone(),
        Segment::Composite(children) => children.iter().map(segment_to_text).collect::<Vec<_>>().join(" "),
        Segment::Link(node) => segment_to_text(&node.data),
    }
} 