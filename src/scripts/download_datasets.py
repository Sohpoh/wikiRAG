import os
import requests
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASETS = {
    "squad": {
        "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
        "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
    },
    "hotpotqa": {
        "train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
        "dev": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        "dev_fullwiki": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"
    }
}

def download_file(url: str, dest_path: Path):
    """Download a file with progress bar."""
    if dest_path.exists():
        logger.info(f"File already exists: {dest_path}")
        return

    logger.info(f"Downloading {url} to {dest_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            bar.update(size)

def main():
    base_dir = Path("data/datasets")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Download SQuAD
    squad_dir = base_dir / "squad"
    squad_dir.mkdir(exist_ok=True)
    
    for split, url in DATASETS["squad"].items():
        filename = f"{split}-v1.1.json"
        download_file(url, squad_dir / filename)
        
    # Download HotpotQA
    hotpot_dir = base_dir / "hotpotqa"
    hotpot_dir.mkdir(exist_ok=True)
    
    # Train
    download_file(DATASETS["hotpotqa"]["train"], hotpot_dir / "hotpot_train_v1.json")
    
    # Dev (Distractor) - standard dev set
    download_file(DATASETS["hotpotqa"]["dev"], hotpot_dir / "hotpot_dev_v1.json")
    
    # Dev (Fullwiki) - useful for RAG
    download_file(DATASETS["hotpotqa"]["dev_fullwiki"], hotpot_dir / "hotpot_dev_fullwiki_v1.json")
    
    logger.info("Download complete!")

if __name__ == "__main__":
    main()
