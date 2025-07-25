# QUICK_START.md
## Быстрый старт с Metastasa

**Дата обновления:** Декабрь 2024  
**Версия:** 0.1.0

---

## 🚀 Запуск проекта

### 1. Основная демонстрация
```bash
# Сборка и запуск основной демонстрации
cargo run --bin metastasa
```

**Ожидаемый вывод:**
```
=== Metastasa - AI Knowledge System ===

=== Attention на GPU (Burn) ===
GPU Attention scores: Tensor { ... }

=== Логический attention (Burn) ===
Logic Attention scores: Tensor { ... }

=== Граф знаний (petgraph + sled, вложенные сегменты) ===
KnowledgeGraph создан: 1 узлов

=== TrustedScraper (async) ===
TrustedScraper результат: true

=== Обработка текста (TextProcessor) ===
Обработано предложений: 2
Найдено действий: 1
Найдено свойств: 1
Создано узлов графа: 3
Создано связей: 1

=== Кеширование эмбеддингов ===
Сходство 'кот' и 'кошка': 0.847
Статистика кеша: ...

=== Обучение на новых данных ===
Результат обучения: ✅ Добавлено в граф

=== Поиск похожих эмбеддингов ===
Наиболее похожие на 'животное': ...

=== Все компоненты работают! ===
🚀 Система готова к использованию!
📚 Для запуска REST API используйте: cargo run --bin api
```

### 2. REST API сервер
```bash
# Запуск REST API сервера
cargo run --bin api
```

**Ожидаемый вывод:**
```
🚀 Запуск REST API сервера на http://127.0.0.1:8080

📋 Доступные эндпоинты:
  POST /process    - Обработка текста с обучением
  POST /learn      - Обучение на новых данных
  POST /verify     - Верификация фактов
  GET  /cache      - Статистика кеша эмбеддингов
  POST /clear      - Очистка кеша
  POST /similar    - Поиск похожих эмбеддингов
  GET  /health     - Проверка здоровья системы
  GET  /info       - Информация о системе
  GET  /graph      - Экспорт графа знаний

🌐 Сервер запущен на http://127.0.0.1:8080
```

### 3. Тестирование
```bash
# Запуск всех тестов
cargo test

# Запуск конкретных тестов
cargo test --test text_processing
cargo test --test integration
cargo test --test fuzzing
```

---

## 🔧 Примеры использования API

### Обработка текста
```bash
curl -X POST http://localhost:8080/process \
  -H "Content-Type: application/json" \
  -d '{"text": "Кот ест рыбу. Рыба водится в море.", "learn": true}'
```

### Обучение на новых данных
```bash
curl -X POST http://localhost:8080/learn \
  -H "Content-Type: application/json" \
  -d '{"text": "Квадрокоптер — это летательный аппарат", "annotation": "Уточнение: пропеллеры", "user": "user1"}'
```

### Поиск похожих эмбеддингов
```bash
curl -X POST http://localhost:8080/similar \
  -H "Content-Type: application/json" \
  -d '{"text": "животное", "top_k": 5}'
```

### Проверка здоровья системы
```bash
curl http://localhost:8080/health
```

---

## 📊 Мониторинг

### Статистика кеша
```bash
curl http://localhost:8080/cache
```

### Информация о системе
```bash
curl http://localhost:8080/info
```

### Экспорт графа знаний
```bash
curl http://localhost:8080/graph
```

---

## ⚠️ Устранение неполадок

### Проблема: "cargo run" не знает какой бинарный файл запустить
**Решение:** Используйте флаг `--bin`:
```bash
cargo run --bin metastasa  # Основная демонстрация
cargo run --bin api        # REST API сервер
```

### Проблема: Ошибки компиляции
**Решение:** Проверьте версию Rust:
```bash
rustc --version  # Должно быть 1.70+
```

### Проблема: API не отвечает
**Решение:** Проверьте, что сервер запущен:
```bash
curl http://localhost:8080/health
```

---

## 📚 Дополнительная документация

- **[README.md](README.md)** — Полная документация проекта
- **[API.md](API.md)** — Документация REST API
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** — Статус разработки
- **[DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)** — План развития

---

## 🎯 Следующие шаги

1. **Изучите API:** Попробуйте все эндпоинты
2. **Добавьте данные:** Обучите систему на новых фактах
3. **Интегрируйте:** Подключите к своему приложению
4. **Внесите вклад:** Присоединитесь к разработке

---

*Готово к использованию! 🚀* 