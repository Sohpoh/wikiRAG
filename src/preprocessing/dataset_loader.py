"""Dataset loaders for SQuAD and HotpotQA."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SQuADLoader:
    """Load SQuAD dataset."""
    
    def __init__(self, data_dir: str):
        """Initialize SQuAD loader."""
        self.data_dir = Path(data_dir)
        logger.info(f"SQuADLoader initialized: {data_dir}")
    
    def load_split(self, split: str = "dev") -> List[Dict]:
        """
        Load SQuAD split.
        
        Args:
            split: Split name (train/dev)
            
        Returns:
            List of examples with format:
                {
                    "id": question_id,
                    "question": question_text,
                    "answers": [answer1, answer2, ...],
                    "context": context_text
                }
        """
        file_path = self.data_dir / f"{split}-v1.1.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"SQuAD file not found: {file_path}")
        
        logger.info(f"Loading SQuAD {split} split from {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        examples = []
        
        for article in data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                
                for qa in paragraph['qas']:
                    # Get all answer texts
                    answers = [ans['text'] for ans in qa['answers']]
                    
                    example = {
                        "id": qa['id'],
                        "question": qa['question'],
                        "answers": answers,
                        "context": context
                    }
                    
                    examples.append(example)
        
        logger.info(f"Loaded {len(examples)} SQuAD examples")
        
        return examples


class HotpotQALoader:
    """Load HotpotQA dataset."""
    
    def __init__(self, data_dir: str):
        """Initialize HotpotQA loader."""
        self.data_dir = Path(data_dir)
        logger.info(f"HotpotQALoader initialized: {data_dir}")
    
    def load_split(self, split: str = "dev") -> List[Dict]:
        """
        Load HotpotQA split.
        
        Args:
            split: Split name (train/dev/test)
            
        Returns:
            List of examples with format:
                {
                    "id": question_id,
                    "question": question_text,
                    "answer": answer_text,
                    "supporting_facts": [[title, sent_idx], ...],
                    "context": list of paragraphs
                }
        """
        file_path = self.data_dir / f"hotpot_{split}_v1.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"HotpotQA file not found: {file_path}")
        
        logger.info(f"Loading HotpotQA {split} split from {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        examples = []
        
        for item in data:
            example = {
                "id": item['_id'],
                "question": item['question'],
                "answer": item['answer'],
                "supporting_facts": item.get('supporting_facts', []),
                "context": item.get('context', [])
            }
            
            examples.append(example)
        
        logger.info(f"Loaded {len(examples)} HotpotQA examples")
        
        return examples