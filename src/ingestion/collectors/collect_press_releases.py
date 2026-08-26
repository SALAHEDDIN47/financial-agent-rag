from sec_edgar_downloader import Downloader
from pathlib import Path

def download_8k(tickers: list, limit: int = 3, output_dir: str = "data/raw/sec"):
    """Télécharge les derniers formulaires 8-K (Événements majeurs / Communiqués)."""
    target_path = Path(output_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    dl = Downloader("Projet_RAG_Financier", "etudiant.projet@universite.fr", target_path)
    
    for ticker in tickers:
        print(f"⏳ Téléchargement de {limit} formulaires 8-K pour {ticker}...")
        dl.get("8-K", ticker, limit=limit)
        print(f"✅ 8-K récupérés pour {ticker}.")

if __name__ == "__main__":
    download_8k(["AAPL", "TSLA"], limit=2)