import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.preprocessing.dataset_loader import SQuADLoader, HotpotQALoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    data_path = Path("data/datasets")
    
    print("="*50)
    print("VERIFYING DATASETS")
    print("="*50)
    
    # Verify SQuAD
    print("\n1. Verifying SQuAD...")
    try:
        loader = SQuADLoader(data_path / "squad")
        train = loader.load_split("train")
        dev = loader.load_split("dev")
        print(f"✓ SQuAD Train: {len(train)} examples")
        print(f"✓ SQuAD Dev: {len(dev)} examples")
    except Exception as e:
        print(f"✗ SQuAD Verification Failed: {e}")
        
    # Verify HotpotQA
    print("\n2. Verifying HotpotQA...")
    try:
        loader = HotpotQALoader(data_path / "hotpotqa")
        train = loader.load_split("train")
        dev = loader.load_split("dev")
        print(f"✓ HotpotQA Train: {len(train)} examples")
        print(f"✓ HotpotQA Dev: {len(dev)} examples")
        
        # Check fullwiki if available
        try:
            dev_full = loader.load_split("dev_fullwiki")
            print(f"✓ HotpotQA Dev (FullWiki): {len(dev_full)} examples")
        except:
            print("- HotpotQA Dev (FullWiki) not found (optional)")
            
    except Exception as e:
        print(f"✗ HotpotQA Verification Failed: {e}")

if __name__ == "__main__":
    main()
