from datasets import load_dataset
import json
from pathlib import Path

def download_transcripts(tickers: list, limit: int = 2, output_dir: str = "data/raw/transcripts"):
    target_path = Path(output_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Récupération des Transcriptions pour {tickers}...")
    
    # Mode streaming pour parcourir les données sans tout télécharger
    dataset = load_dataset("Rogersurf/earnings-call-transcripts", split="train", streaming=True)
    counts = {ticker: 0 for ticker in tickers}
    
    for row in dataset:
        ticker = row.get("ticker")
        if ticker in tickers and counts[ticker] < limit:
            file_name = f"{ticker}_Q{row.get('quarter')}_{row.get('earnings_year')}_transcript.json"
            file_path = target_path / file_name
            
            data = {
                "ticker": ticker,
                "quarter": row.get("quarter"),
                "year": row.get("earnings_year"),
                "transcript": row.get("transcript")
            }
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                
            print(f"✅ {file_name} sauvegardé.")
            counts[ticker] += 1
            
        if all(c >= limit for c in counts.values()):
            break

if __name__ == "__main__":
    download_transcripts(["AAPL", "TSLA"], limit=2)