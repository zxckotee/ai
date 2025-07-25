# API.md
## Документация REST API для Metastasa

**Дата обновления:** Декабрь 2024  
**Версия API:** 1.0  
**Базовый URL:** `http://localhost:8080`

---

## 🚀 Быстрый старт

### Запуск API сервера
```bash
cargo run --bin api
```

### Проверка доступности
```bash
curl http://localhost:8080/health
```

---

## 📋 Обзор эндпоинтов

| Метод | Эндпоинт | Описание |
|-------|----------|----------|
| `POST` | `/process` | Обработка текста с обучением |
| `POST` | `/learn` | Обучение на новых данных |
| `POST` | `/verify` | Верификация фактов |
| `GET` | `/cache` | Статистика кеша эмбеддингов |
| `POST` | `/clear` | Очистка кеша |
| `POST` | `/similar` | Поиск похожих эмбеддингов |
| `GET` | `/health` | Проверка здоровья системы |
| `GET` | `/info` | Информация о системе |
| `GET` | `/graph` | Экспорт графа знаний |

---

## 🔧 Обработка текста

### POST /process
Обрабатывает текст, извлекает сущности, действия и свойства, опционально обучается на новых данных.

**Запрос:**
```json
{
  "text": "Кот ест рыбу. Рыба водится в море.",
  "learn": true
}
```

**Ответ:**
```json
{
  "success": true,
  "result": {
    "entities": [
      {
        "text": "кот",
        "entity_type": "Animal",
        "confidence": 0.95
      },
      {
        "text": "рыба",
        "entity_type": "Animal",
        "confidence": 0.92
      },
      {
        "text": "море",
        "entity_type": "Location",
        "confidence": 0.88
      }
    ],
    "actions": [
      {
        "subject": "кот",
        "verb": "ест",
        "object": "рыбу"
      }
    ],
    "properties": [
      {
        "entity": "рыба",
        "attribute": "водится",
        "value": "в море"
      }
    ],
    "segments": [
      "Кот ест рыбу.",
      "Рыба водится в море."
    ],
    "graph_nodes": ["кот", "рыба", "море"],
    "graph_edges": [
      ["кот", "ест", "рыбу"],
      ["рыба", "водится", "в море"]
    ]
  },
  "learned": true,
  "message": "Факт добавлен в граф знаний"
}
```

**Параметры:**
- `text` (string, обязательный): Текст для обработки
- `learn` (boolean, опциональный): Включить обучение на новых данных

---

## 🎓 Обучение

### POST /learn
Добавляет новые знания в граф с проверкой через TrustedScraper.

**Запрос:**
```json
{
  "text": "Квадрокоптер — это летательный аппарат с 4 моторами",
  "annotation": "Уточнение: пропеллеры",
  "user": "user1"
}
```

**Ответ:**
```json
{
  "success": true,
  "verified": true,
  "added_to_graph": true,
  "message": "Знание добавлено в граф после проверки"
}
```

**Параметры:**
- `text` (string, обязательный): Новое знание
- `annotation` (string, опциональный): Пояснение или уточнение
- `user` (string, опциональный): Идентификатор пользователя

---

## ✅ Верификация

### POST /verify
Проверяет факт через TrustedScraper.

**Запрос:**
```json
{
  "fact": "Коты имеют 4 ноги"
}
```

**Ответ:**
```json
{
  "success": true,
  "verified": true,
  "confidence": 0.95,
  "sources": ["Wikipedia", "Veterinary Database"],
  "message": "Факт подтвержден"
}
```

**Параметры:**
- `fact` (string, обязательный): Факт для проверки

---

## 💾 Кеш эмбеддингов

### GET /cache
Получает статистику кеша эмбеддингов.

**Ответ:**
```json
{
  "success": true,
  "stats": {
    "total_embeddings": 150,
    "entity_embeddings": 45,
    "action_embeddings": 30,
    "property_embeddings": 25,
    "cache_hits": 1200,
    "cache_misses": 150,
    "hit_rate": 0.89
  }
}
```

### POST /clear
Очищает кеш эмбеддингов.

**Ответ:**
```json
{
  "success": true,
  "message": "Кеш очищен",
  "cleared_embeddings": 150
}
```

### POST /similar
Находит похожие эмбеддинги.

**Запрос:**
```json
{
  "text": "животное",
  "top_k": 5
}
```

**Ответ:**
```json
{
  "success": true,
  "similar": [
    {
      "text": "кот",
      "similarity": 0.847
    },
    {
      "text": "собака",
      "similarity": 0.823
    },
    {
      "text": "рыба",
      "similarity": 0.789
    },
    {
      "text": "птица",
      "similarity": 0.756
    },
    {
      "text": "лошадь",
      "similarity": 0.734
    }
  ]
}
```

**Параметры:**
- `text` (string, обязательный): Текст для поиска похожих
- `top_k` (integer, опциональный): Количество результатов (по умолчанию 5)

---

## 📊 Мониторинг

### GET /health
Проверяет здоровье системы.

**Ответ:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-15T10:30:00Z",
  "uptime": "2h 15m 30s",
  "components": {
    "knowledge_graph": "ok",
    "text_processor": "ok",
    "embedding_cache": "ok",
    "trusted_scraper": "ok"
  }
}
```

### GET /info
Получает информацию о системе.

**Ответ:**
```json
{
  "name": "Metastasa",
  "version": "0.1.0",
  "description": "Гибридная AI система с логическим мышлением",
  "features": [
    "GPU Attention (Burn)",
    "Knowledge Graph",
    "TrustedScraper",
    "Active Learning",
    "TextProcessor",
    "EmbeddingCache"
  ],
  "endpoints": [
    "POST /process",
    "POST /learn",
    "POST /verify",
    "GET /cache",
    "POST /clear",
    "POST /similar",
    "GET /health",
    "GET /info",
    "GET /graph"
  ]
}
```

### GET /graph
Экспортирует граф знаний со статистикой.

**Ответ:**
```json
{
  "success": true,
  "graph": {
    "nodes": [
      {
        "id": "кот",
        "type": "Animal",
        "properties": {
          "legs": 4,
          "family": "Felidae"
        }
      },
      {
        "id": "рыба",
        "type": "Animal",
        "properties": {
          "habitat": "water"
        }
      }
    ],
    "edges": [
      {
        "from": "кот",
        "to": "рыба",
        "relation": "ест"
      }
    ]
  },
  "stats": {
    "total_nodes": 150,
    "total_edges": 300,
    "entity_types": {
      "Animal": 45,
      "Location": 30,
      "Person": 25,
      "ScientificTerm": 20
    },
    "action_types": {
      "eats": 15,
      "lives_in": 20,
      "studies": 10
    }
  }
}
```

---

## 🔧 Примеры использования

### Обработка научного текста
```bash
curl -X POST http://localhost:8080/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Квантовая механика описывает поведение частиц на атомном уровне. Электроны обладают волновыми свойствами.",
    "learn": true
  }'
```

### Обучение на новых данных
```bash
curl -X POST http://localhost:8080/learn \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Нейронные сети используют градиентный спуск для оптимизации",
    "annotation": "Машинное обучение",
    "user": "researcher1"
  }'
```

### Поиск похожих концепций
```bash
curl -X POST http://localhost:8080/similar \
  -H "Content-Type: application/json" \
  -d '{
    "text": "искусственный интеллект",
    "top_k": 10
  }'
```

### Проверка факта
```bash
curl -X POST http://localhost:8080/verify \
  -H "Content-Type: application/json" \
  -d '{
    "fact": "Солнце является звездой"
  }'
```

---

## ⚠️ Обработка ошибок

### Общий формат ошибки
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Неверный формат запроса",
    "details": {
      "field": "text",
      "issue": "Поле обязательно для заполнения"
    }
  }
}
```

### Коды ошибок
- `VALIDATION_ERROR`: Ошибка валидации входных данных
- `PROCESSING_ERROR`: Ошибка обработки текста
- `VERIFICATION_ERROR`: Ошибка верификации факта
- `CACHE_ERROR`: Ошибка работы с кешем
- `GRAPH_ERROR`: Ошибка работы с графом знаний
- `INTERNAL_ERROR`: Внутренняя ошибка сервера

---

## 📈 Производительность

### Рекомендации
- **Размер текста:** До 10,000 символов для оптимальной производительности
- **Частота запросов:** До 100 запросов в минуту
- **Параллельные запросы:** До 10 одновременных соединений

### Мониторинг
- Используйте `/health` для проверки состояния системы
- Мониторьте `/cache` для оптимизации производительности
- Проверяйте `/info` для информации о версии и возможностях

---

## 🔐 Безопасность

### Текущие меры
- Валидация всех входных данных
- Ограничение размера запросов
- Логирование всех операций
- Проверка фактов через TrustedScraper

### Планируемые улучшения
- Аутентификация пользователей
- Авторизация по ролям
- Шифрование данных
- Rate limiting

---

## 📚 Дополнительные ресурсы

- **[README.md](README.md)** — Общая документация проекта
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** — Статус разработки
- **[DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)** — План развития

---

*API документация обновляется по мере развития проекта.* 