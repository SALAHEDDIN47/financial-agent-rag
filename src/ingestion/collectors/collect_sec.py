import os
from pathlib import Path
from sec_edgar_downloader import Downloader

def download_sec_reports(tickers: list, doc_type: str = "10-K", limit: int = 1, output_dir: str = "data/raw/sec"):
    """
    Télécharge les rapports financiers de la SEC pour une liste d'entreprises.
    """
    target_path = Path(output_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # La SEC exige une identification stricte pour éviter le blocage IP.
    # Modifiez ces valeurs pour éviter d'utiliser des paramètres génériques.
    company_name = "Projet_RAG_Financier"
    email = "etudiant.projet@universite.fr"
    
    dl = Downloader(company_name, email, target_path)
    
    print(f"📥 Démarrage de l'acquisition des rapports {doc_type} pour {tickers}")
    
    for ticker in tickers:
        print(f"⏳ Téléchargement du dernier rapport {doc_type} pour {ticker}...")
        try:
            # Récupère les "limit" derniers documents du type demandé
            dl.get(doc_type, ticker, limit=limit)
            print(f"✅ {ticker} terminé avec succès.")
        except Exception as e:
            print(f"❌ Erreur lors de la récupération pour {ticker}: {e}")

if __name__ == "__main__":
    # Liste de tests (Apple, Tesla)
    test_tickers = ["AAPL", "TSLA", "GOOGL", "MSFT"]
    download_sec_reports(tickers=test_tickers, doc_type="10-K", limit=1)