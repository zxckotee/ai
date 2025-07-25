// Основные структуры ядра "Мыслящее Ядро"
// Подробные комментарии для понимания архитектуры

use std::borrow::Cow;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use lazy_static::lazy_static;
use regex::Regex;

/// Мультимодальный сегмент
#[derive(Debug, Clone)]
pub enum Segment {
    Text(String),
    Image(Vec<u8>),
    // Для будущей интеграции аудио
    // Audio(whisper_rs::AudioBuffer),
}

#[derive(Debug, Clone)]
pub enum SegmentType {
    Text,
    Json,
    Problem { desc: String, prio: u8 },
}

/// Тип связи между узлами графа
#[derive(Debug, Clone)]
pub enum EdgeType {
    Cause,
    PartOf,
    Example,
}

/// Источник знания
#[derive(Debug, Clone)]
pub enum Source {
    ArXiv,
    Wikipedia,
    StackOverflow,
    UserCorrected,
}

#[derive(Debug, Clone)]
pub struct AttentionLayer {
    /// Веса внимания: логические и нейросетевые
    pub logic_weight: f32,
    pub nn_weight: f32,
}

/// Сегмент мысли с поддержкой мультимодальности и zero-copy JSON
#[derive(Debug, Clone)]
pub struct ThoughtSegment<'a> {
    pub content: Cow<'a, str>,
    pub segment_type: SegmentType,
    pub segment: Segment,
}

/// Узел графа знаний с расширенной структурой
#[derive(Debug, Clone)]
pub struct KnowledgeNode {
    pub id: Uuid,
    pub data: String, // Текст или JSON
    pub edges: Vec<(EdgeType, Uuid)>,
    pub source: Source,
    pub last_verified: DateTime<Utc>,
}

/// Граф знаний
#[derive(Debug, Clone)]
pub struct KnowledgeGraph {
    pub nodes: Vec<KnowledgeNode>,
}

/// Ядро системы
#[derive(Debug, Clone)]
pub struct Core<'a> {
    pub segments: Vec<ThoughtSegment<'a>>,
    pub attention: AttentionLayer,
    pub knowledge: KnowledgeGraph,
}

// Кешируем Regex для парсинга проблемных сегментов
lazy_static! {
    static ref PROBLEM_REGEX: Regex = Regex::new(r"ПРОБЛЕМА:(?P<desc>.+?)ПРИОРИТЕТ:(?P<prio>\\d+)").unwrap();
}

impl<'a> ThoughtSegment<'a> {
    /// Универсальный парсер входа: определяет тип сегмента (Text, Json, Problem)
    pub fn from_input(input: &'a str) -> Vec<ThoughtSegment<'a>> {
        // Проверка на Problem через Regex
        if let Some(caps) = PROBLEM_REGEX.captures(input) {
            let desc = caps.name("desc").map(|m| m.as_str().trim().to_string()).unwrap_or_default();
            let prio = caps.name("prio").and_then(|m| m.as_str().parse::<u8>().ok()).unwrap_or(0);
            return vec![ThoughtSegment {
                content: Cow::Borrowed(input),
                segment_type: SegmentType::Problem { desc, prio },
                segment: Segment::Text(input.to_string()),
            }];
        }
        // Проверка на JSON (zero-copy)
        let trimmed = input.trim();
        if trimmed.starts_with('{') && trimmed.ends_with('}') {
            return vec![ThoughtSegment {
                content: Cow::Borrowed(input),
                segment_type: SegmentType::Json,
                segment: Segment::Text(input.to_string()),
            }];
        }
        // По умолчанию — текст
        vec![ThoughtSegment {
            content: Cow::Borrowed(input),
            segment_type: SegmentType::Text,
            segment: Segment::Text(input.to_string()),
        }]
    }
} 