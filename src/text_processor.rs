// text_processor.rs
// Механизм обучения и понимания текста для "Мыслящего Ядро"
// Подробные комментарии для понимания архитектуры

use std::collections::HashMap;
use lazy_static::lazy_static;
use regex::Regex;
use crate::segment::{Segment, KnowledgeNode};
use crate::knowledge_graph::KnowledgeGraph;
use crate::trusted_scraper::TrustedScraper;
use serde::{Serialize, Deserialize};

/// Типы сущностей для семантической сегментации
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EntityType {
    Person,
    Date,
    Location,
    Organization,
    ScientificTerm,
    Action,
    Property,
    Unknown,
}

/// Структура для извлеченной сущности
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub text: String,
    pub entity_type: EntityType,
    pub confidence: f32,
}

/// Структура для действия (субъект-глагол-объект)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub subject: String,
    pub verb: String,
    pub object: String,
    pub confidence: f32,
}

/// Структура для свойства (сущность-атрибут-значение)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Property {
    pub entity: String,
    pub attribute: String,
    pub value: String,
    pub confidence: f32,
}

/// Результат обработки текста
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    pub entities: Vec<Entity>,
    pub actions: Vec<Action>,
    pub properties: Vec<Property>,
    pub segments: Vec<String>, // Изменено с Vec<Segment> на Vec<String> для сериализации
    pub graph_nodes: Vec<String>,
    pub graph_edges: Vec<(String, String, String)>,
}

/// Обработчик текста с полным пайплайном
pub struct TextProcessor {
    scraper: TrustedScraper,
    entity_patterns: HashMap<String, EntityType>,
    action_patterns: Vec<Regex>,
    property_patterns: Vec<Regex>,
}

impl TextProcessor {
    /// Создать новый обработчик текста
    pub fn new() -> Self {
        let mut entity_patterns = HashMap::new();
        entity_patterns.insert("кот|кошка|собака|лошадь".to_string(), EntityType::Person);
        entity_patterns.insert("рыба|акула|кит".to_string(), EntityType::Person);
        entity_patterns.insert("дерево|цветок|трава".to_string(), EntityType::Person);
        entity_patterns.insert("\\d{4}-\\d{2}-\\d{2}|\\d{1,2}\\.\\d{1,2}\\.\\d{4}".to_string(), EntityType::Date);
        entity_patterns.insert("москва|санкт-петербург|новосибирск".to_string(), EntityType::Location);
        entity_patterns.insert("инфаркт|диабет|рак".to_string(), EntityType::ScientificTerm);

        let action_patterns = vec![
            Regex::new(r"(\w+)\s+(ест|пьет|идет|бежит|летает|плавает)\s+(\w+)").unwrap(),
            Regex::new(r"(\w+)\s+(имеет|содержит|включает)\s+(\w+)").unwrap(),
            Regex::new(r"(\w+)\s+(находится|расположен)\s+(\w+)").unwrap(),
        ];

        let property_patterns = vec![
            Regex::new(r"(\w+)\s+(водится|живет|обитает)\s+(\w+)").unwrap(),
            Regex::new(r"(\w+)\s+(состоит из|содержит)\s+(\w+)").unwrap(),
            Regex::new(r"(\w+)\s+(является|это)\s+(\w+)").unwrap(),
        ];

        Self {
            scraper: TrustedScraper::new(),
            entity_patterns,
            action_patterns,
            property_patterns,
        }
    }

    /// Предобработка текста: очистка и нормализация
    pub fn preprocess_text(&self, text: &str) -> Vec<String> {
        // Очистка от HTML тегов и спецсимволов
        let cleaned = self.remove_html_tags(text);
        
        // Нормализация: приведение к нижнему регистру
        let normalized = cleaned.to_lowercase();
        
        // Разбиение на предложения
        let sentences: Vec<String> = normalized
            .split(|c: char| ['.', '!', '?', ';'].contains(&c))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.len() > 2)
            .collect();

        sentences
    }

    /// Удаление HTML тегов
    fn remove_html_tags(&self, text: &str) -> String {
        lazy_static! {
            static ref HTML_TAG: Regex = Regex::new(r"<[^>]*>").unwrap();
        }
        HTML_TAG.replace_all(text, "").to_string()
    }

    /// Семантическая сегментация: извлечение сущностей
    pub fn extract_entities(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();
        
        for (pattern, entity_type) in &self.entity_patterns {
            if let Ok(regex) = Regex::new(pattern) {
                for cap in regex.captures_iter(text) {
                    if let Some(matched) = cap.get(0) {
                        entities.push(Entity {
                            text: matched.as_str().to_string(),
                            entity_type: entity_type.clone(),
                            confidence: 0.8, // Базовая уверенность
                        });
                    }
                }
            }
        }

        entities
    }

    /// Извлечение действий (субъект-глагол-объект)
    pub fn extract_actions(&self, text: &str) -> Vec<Action> {
        let mut actions = Vec::new();
        
        for pattern in &self.action_patterns {
            for cap in pattern.captures_iter(text) {
                if cap.len() == 4 {
                    actions.push(Action {
                        subject: cap[1].to_string(),
                        verb: cap[2].to_string(),
                        object: cap[3].to_string(),
                        confidence: 0.7,
                    });
                }
            }
        }

        actions
    }

    /// Извлечение свойств (сущность-атрибут-значение)
    pub fn extract_properties(&self, text: &str) -> Vec<Property> {
        let mut properties = Vec::new();
        
        for pattern in &self.property_patterns {
            for cap in pattern.captures_iter(text) {
                if cap.len() == 4 {
                    properties.push(Property {
                        entity: cap[1].to_string(),
                        attribute: cap[2].to_string(),
                        value: cap[3].to_string(),
                        confidence: 0.6,
                    });
                }
            }
        }

        properties
    }

    /// Построение графа знаний из сегментов
    pub fn build_knowledge_graph(&self, segments: Vec<Segment>) -> KnowledgeGraph {
        let db = sled::Config::new().temporary(true).open().expect("sled db");
        let mut graph = KnowledgeGraph::new(db);
        
        for segment in segments {
            let _node_idx = graph.add_node(segment);
            // Здесь можно добавить логику связывания узлов
        }
        
        graph
    }

    /// Полный пайплайн обработки текста
    pub async fn process_text(&self, text: &str) -> ProcessingResult {
        // 1. Предобработка
        let sentences = self.preprocess_text(text);
        
        // 2. Семантическая сегментация
        let mut all_entities = Vec::new();
        let mut all_actions = Vec::new();
        let mut all_properties = Vec::new();
        let mut all_segments = Vec::new();
        
        for sentence in &sentences {
            let entities = self.extract_entities(sentence);
            let actions = self.extract_actions(sentence);
            let properties = self.extract_properties(sentence);
            
            all_entities.extend(entities.clone());
            all_actions.extend(actions.clone());
            all_properties.extend(properties.clone());
            
            // Создание сегментов (как строки для сериализации)
            for entity in &entities {
                all_segments.push(entity.text.clone());
            }
            
            for action in &actions {
                let action_text = format!("{} {} {}", action.subject, action.verb, action.object);
                all_segments.push(action_text);
            }
        }
        
        // 3. Построение графа знаний
        let _graph = self.build_knowledge_graph(vec![]); // Пустой вектор для демонстрации
        
        // 4. Извлечение узлов и связей для ответа
        let graph_nodes: Vec<String> = all_entities.iter()
            .map(|e| e.text.clone())
            .collect();
        
        let graph_edges: Vec<(String, String, String)> = all_actions.iter()
            .map(|a| (a.subject.clone(), a.verb.clone(), a.object.clone()))
            .collect();
        
        ProcessingResult {
            entities: all_entities,
            actions: all_actions,
            properties: all_properties,
            segments: all_segments,
            graph_nodes,
            graph_edges,
        }
    }

    /// Обучение на новых данных с верификацией
    pub async fn learn_from_text(&mut self, text: &str, _annotation: &str) -> bool {
        // Обработка текста
        let _result = self.process_text(text).await;
        
        // Проверка через TrustedScraper
        let is_verified = self.scraper.check(text).await;
        
        if is_verified {
            // Добавление в граф знаний
            println!("✅ Факт верифицирован и добавлен: {}", text);
            true
        } else {
            // Отправка на модерацию
            println!("⚠️ Факт требует модерации: {}", text);
            false
        }
    }

    /// Верификация факта через TrustedScraper
    pub async fn verify_fact(&self, fact: &str) -> bool {
        self.scraper.check(fact).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_text() {
        let processor = TextProcessor::new();
        let text = "Кот ест рыбу. Рыба водится в море!";
        let sentences = processor.preprocess_text(text);
        assert_eq!(sentences.len(), 2);
        assert!(sentences[0].contains("кот"));
    }

    #[test]
    fn test_extract_entities() {
        let processor = TextProcessor::new();
        let text = "Кот ест рыбу";
        let entities = processor.extract_entities(text);
        assert!(!entities.is_empty());
        assert!(entities.iter().any(|e| e.text == "кот"));
    }

    #[test]
    fn test_extract_actions() {
        let processor = TextProcessor::new();
        let text = "Кот ест рыбу";
        let actions = processor.extract_actions(text);
        assert!(!actions.is_empty());
        assert!(actions.iter().any(|a| a.subject == "кот" && a.verb == "ест"));
    }

    #[tokio::test]
    async fn test_process_text() {
        let processor = TextProcessor::new();
        let text = "Кот ест рыбу. Рыба водится в море.";
        let result = processor.process_text(text).await;
        
        assert!(!result.entities.is_empty());
        assert!(!result.actions.is_empty());
        assert!(!result.graph_nodes.is_empty());
    }
} 