use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;
use std::error::Error;
use bumpalo::Bump;

/// Максимальная глубина вложенности для защиты от переполнения
const MAX_DEPTH: usize = 100;

/// Рекурсивный enum для вложенных сегментов
#[derive(Debug, Clone, PartialEq)]
pub enum Segment {
    Primitive(String), // Текст, числа
    Composite(Vec<Segment>), // Вложенные сегменты
    Link(Box<KnowledgeNode>), // Ссылка на узел графа
}

/// Узел графа знаний с вложенными сегментами
#[derive(Debug, Clone, PartialEq)]
pub struct KnowledgeNode {
    pub id: Uuid,
    pub data: Segment, // Может быть вложенным!
    pub depth: usize,  // Текущая глубина (для предотвращения переполнения)
    pub edges: Vec<Edge>,
    pub tags: Vec<String>, // Новое поле: теги (например, "животное", "биология")
}

/// Тип связи (заглушка)
#[derive(Debug, Clone, PartialEq)]
pub struct Edge;

/// Ошибка переполнения рекурсии
#[derive(Debug)]
pub enum SegmentError {
    RecursionLimitExceeded,
}

impl fmt::Display for SegmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SegmentError::RecursionLimitExceeded => write!(f, "Превышена максимальная глубина рекурсии"),
        }
    }
}

impl Error for SegmentError {}

/// Рекурсивная обработка сегмента с ограничением глубины
pub fn process_segment(segment: &Segment, depth: usize) -> Result<(), SegmentError> {
    if depth > MAX_DEPTH {
        return Err(SegmentError::RecursionLimitExceeded);
    }
    match segment {
        Segment::Primitive(text) => analyze_text(text),
        Segment::Composite(children) => {
            for child in children {
                process_segment(child, depth + 1)?;
            }
            Ok(())
        },
        Segment::Link(node) => process_knowledge_node(node),
    }
}

/// Итеративная обработка вложенных сегментов (для очень глубоких структур)
pub fn process_segment_iterative(root: &Segment) -> Result<(), SegmentError> {
    let mut stack = vec![(root, 0)];
    while let Some((segment, depth)) = stack.pop() {
        if depth > MAX_DEPTH {
            return Err(SegmentError::RecursionLimitExceeded);
        }
        match segment {
            Segment::Primitive(text) => analyze_text(text)?,
            Segment::Composite(children) => {
                for child in children.iter().rev() {
                    stack.push((child, depth + 1));
                }
            },
            Segment::Link(node) => process_knowledge_node(node)?,
        }
    }
    Ok(())
}

/// Пример анализа текста (заглушка)
fn analyze_text(_text: &str) -> Result<(), SegmentError> {
    // Здесь может быть логика анализа
    Ok(())
}

/// Пример обработки KnowledgeNode (заглушка)
fn process_knowledge_node(_node: &KnowledgeNode) -> Result<(), SegmentError> {
    // Здесь может быть логика работы с графом
    Ok(())
}

/// Arena-аллокатор для частого создания/удаления сегментов
pub fn arena_example() {
    let arena = Bump::new();
    let _cat = arena.alloc(Segment::Composite(vec![
        Segment::Primitive("Усы".into()),
        Segment::Primitive("Глаза".into()),
    ]));
}

/// Сжатие дубликатов (deduplication) по ключу
pub fn get_cached_segment<'a>(cache: &mut HashMap<String, &'a Segment>, segment: &'a Segment) -> &'a Segment {
    let key = segment_to_key(segment);
    cache.entry(key).or_insert(segment)
}

/// Генерация ключа для сегмента (например, хэш или сериализация)
fn segment_to_key(segment: &Segment) -> String {
    match segment {
        Segment::Primitive(s) => format!("P:{}", s),
        Segment::Composite(children) => {
            let mut key = String::from("C:[");
            for child in children {
                key.push_str(&segment_to_key(child));
                key.push(',');
            }
            key.push(']');
            key
        },
        Segment::Link(node) => format!("L:node:{}", node.id),
    }
} 