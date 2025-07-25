use crate::trusted_scraper::TrustedScraper;

#[derive(serde::Deserialize)]
/// Структура для пользовательской правки
pub struct Correction {
    pub claim: String,
    pub user: String,
    pub justification: String,
}

/// Результат проверки
pub enum CorrectionResult {
    AutoAccepted,
    NeedsModeration,
    Rejected,
}

/// Active Learning: автоматическая проверка и модерация
pub struct ActiveLearning<'a> {
    pub scraper: &'a TrustedScraper,
}

impl<'a> ActiveLearning<'a> {
    pub fn new(scraper: &'a TrustedScraper) -> Self {
        Self { scraper }
    }
    /// Проверить правку: если найдено 2+ подтверждения — принять, иначе на модерацию
    pub async fn check_correction(&self, corr: &Correction) -> CorrectionResult {
        if self.scraper.check_multi(&corr.claim).await >= 2 {
            CorrectionResult::AutoAccepted
        } else {
            CorrectionResult::NeedsModeration
        }
    }
    /// Модерация (заглушка)
    pub async fn moderate(&self, _corr: &Correction) -> CorrectionResult {
        // TODO: добавить интерфейс модератора
        CorrectionResult::Rejected
    }
} 