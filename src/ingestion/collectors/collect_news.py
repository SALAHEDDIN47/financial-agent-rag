import yfinance as yf
import json
from pathlib import Path

def download_news(tickers: list, output_dir: str = "data/raw/news"):
    """Télécharge les actualités récentes pour une liste d'entreprises via Yahoo Finance."""
    target_path = Path(output_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Récupération des actualités pour {tickers}")
    
    for ticker in tickers:
        print(f"⏳ Téléchargement pour {ticker}...")
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if news:
            file_path = target_path / f"{ticker}_news.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(news, f, indent=4, ensure_ascii=False)
            print(f"✅ Actualités sauvegardées dans {file_path.name}")
        else:
            print(f"⚠️ Aucune actualité trouvée pour {ticker}.")

if __name__ == "__main__":
    download_news(["AAPL", "TSLA"])