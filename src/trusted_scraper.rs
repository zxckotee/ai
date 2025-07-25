use reqwest::Client;
use tokio;

#[derive(Clone)]
/// TrustedScraper: асинхронный сбор и проверка фактов
pub struct TrustedScraper {
    client: reqwest::Client,
}

impl TrustedScraper {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }
    /// Асинхронная проверка утверждения через Wikipedia
    pub async fn check_wikipedia(&self, claim: &str) -> bool {
        let url = format!("https://ru.wikipedia.org/w/api.php?action=query&list=search&srsearch={}&format=json", claim);
        match self.client.get(&url).send().await {
            Ok(resp) => {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    let hits = json["query"]["search"].as_array().map(|a| !a.is_empty()).unwrap_or(false);
                    return hits;
                }
                false
            },
            Err(_) => false,
        }
    }
    /// Проверка по arXiv (реальный запрос)
    pub async fn check_arxiv(&self, claim: &str) -> bool {
        let url = format!("http://export.arxiv.org/api/query?search_query=all:{}&max_results=1", claim);
        match self.client.get(&url).send().await {
            Ok(resp) => {
                if let Ok(text) = resp.text().await {
                    // arXiv возвращает XML, ищем <entry>
                    text.contains("<entry>")
                } else { false }
            },
            Err(_) => false,
        }
    }
    /// Проверка по PubMed (реальный запрос)
    pub async fn check_pubmed(&self, claim: &str) -> bool {
        let url = format!("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={}&retmax=1", claim);
        match self.client.get(&url).send().await {
            Ok(resp) => {
                if let Ok(text) = resp.text().await {
                    // PubMed возвращает XML, ищем <IdList><Id>
                    text.contains("<IdList>") && text.contains("<Id>")
                } else { false }
            },
            Err(_) => false,
        }
    }
    /// Проверка по нескольким источникам (Wikipedia + arXiv + PubMed)
    pub async fn check_multi(&self, claim: &str) -> usize {
        let mut count = 0;
        if self.check_wikipedia(claim).await { count += 1; }
        if self.check_arxiv(claim).await { count += 1; }
        if self.check_pubmed(claim).await { count += 1; }
        count
    }
    /// Главная проверка: найдено ли 2+ подтверждения
    pub async fn check(&self, claim: &str) -> bool {
        self.check_multi(claim).await >= 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_check_wikipedia() {
        let scraper = TrustedScraper::new();
        let found = scraper.check_wikipedia("Кошка").await;
        assert!(found);
    }
} 