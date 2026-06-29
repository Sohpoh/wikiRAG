
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator
import logging
from dataclasses import dataclass, asdict
from tqdm import tqdm
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dump_path = Path("data/raw/")
if not dump_path.exists():
    raise FileNotFoundError(f"Dump directory not found: {dump_path}")

chunk_size = 256,
stride = 128,
min_chunk_size = 50


@dataclass
class Passage:
    """Represents a passage chunk with metadata."""
    passage_id: str
    doc_id: str
    title: str
    section: str
    text: str
    start_offset: int
    end_offset: int
    token_count: int

    def to_dict(self) -> Dict:
        return asdict(self)

def _parse_wiki_file(self, file_path: Path) -> Generator[Dict, None, None]:
    """
    Parse a single wiki_* file from WikiExtractor.
    
    Each line is a JSON object with: {"id": ..., "url": ..., "title": ..., "text": ...}
    
    Args:
        file_path: Path to wiki_* file
        
    Yields:
        Document dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                doc = json.loads(line)
                
                # WikiExtractor format has 'id', 'title', 'text', 'url'
                if 'id' in doc and 'title' in doc and 'text' in doc:
                    yield {
                        'doc_id': str(doc['id']),
                        'title': doc['title'],
                        'text': doc['text'],
                        'url': doc.get('url', '')
                    }
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {line_num} in {file_path.name}: {e}")
                continue
    
def load_from_wikiextractor_streaming(self, max_docs) -> Generator[Dict, None, None]:
        """
        Stream Wikipedia articles from WikiExtractor output (memory efficient).
        
        Args:
            max_docs: Maximum number of documents to load
            
        Yields:
            Document dictionaries
        """
        wiki_files = []
        
        # Search for all wiki_* files recursively
        for wiki_file in self.dump_path.rglob("wiki_*"):
            if wiki_file.is_file():
                wiki_files.append(wiki_file)
        
        wiki_files.sort() 
        logger.info(f"Found {len(wiki_files)} wiki files")
        
        if not wiki_files:
            raise ValueError(f"No wiki_* files found in {self.dump_path}")
        
        logger.info(f"Streaming Wikipedia articles from {len(wiki_files)} files")
        
        doc_count = 0
        
        for wiki_file in wiki_files:
            for doc in _parse_wiki_file(wiki_file):
                yield doc
                doc_count += 1
                
                if max_docs and doc_count >= max_docs:
                    logger.info(f"Reached max_docs limit: {max_docs}")
                    return
        
        logger.info(f"Streamed {doc_count} documents")
def chunk_document(
        self,
        doc_id: str,
        title: str,
        text: str
    ) -> List[Passage]:
        """
        Split a document into overlapping passages.
        
        Args:
            doc_id: Document identifier
            title: Document title
            text: Document text
            
        Returns:
            List of Passage objects
        """
        passages = []
        
        # Extract sections
        sections = self.extract_sections(text, title)
        
        passage_counter = 0
        
        for section_title, section_text in sections:
            # Clean the section text
            cleaned_text = self.clean_text(section_text)
            
            if not cleaned_text:
                continue
            
            # Tokenize
            tokens = self.simple_tokenize(cleaned_text)
            
            # Create overlapping chunks
            start_idx = 0
            while start_idx < len(tokens):
                end_idx = min(start_idx + chunk_size, len(tokens))
                chunk_tokens = tokens[start_idx:end_idx]
                
                # Skip if chunk is too small (unless it's the last chunk)
                if len(chunk_tokens) < min_chunk_size and end_idx < len(tokens):
                    start_idx += stride
                    continue
                
                # Reconstruct text from tokens
                chunk_text = ' '.join(chunk_tokens)
                
                # Create passage
                passage = Passage(
                    passage_id=f"{doc_id}_p{passage_counter}",
                    doc_id=doc_id,
                    title=title,
                    section=section_title,
                    text=chunk_text,
                    start_offset=start_idx,
                    end_offset=end_idx,
                    token_count=len(chunk_tokens)
                )
                
                passages.append(passage)
                passage_counter += 1
                
                # Move to next chunk
                if end_idx >= len(tokens):
                    break
                start_idx += stride
        
        return passages
def chunk_corpus_streaming(
        self,
        documents: Generator[Dict, None, None],
        output_path: str,
        batch_size: int = 10000
    ) -> int:
        """
        Chunk corpus using streaming (most memory efficient).
        
        Args:
            documents: Generator yielding document dicts
            output_path: Path to save passages as JSONL
            batch_size: Number of passages to buffer before writing
            
        Returns:
            Total number of passages created
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Chunking documents (streaming mode)...")
        
        total_passages = 0
        batch_passages = []
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for doc in tqdm(documents, desc="Chunking documents"):
                passages = self.chunk_document(
                    doc_id=doc['doc_id'],
                    title=doc['title'],
                    text=doc['text']
                )
                
                batch_passages.extend(passages)
                total_passages += len(passages)
                
                # Write batch when it reaches batch_size
                if len(batch_passages) >= batch_size:
                    for p in batch_passages:
                        f_out.write(json.dumps(p.to_dict()) + '\n')
                    batch_passages = []
            
            # Write remaining passages
            for p in batch_passages:
                f_out.write(json.dumps(p.to_dict()) + '\n')
        
        logger.info(f"Created {total_passages} passages, saved to {output_path}")
        return total_passages

"""
loader = WikipediaLoader("data/raw/")
documents = loader.load_from_wikiextractor(max_docs=10000)

chunker = DocumentChunker(chunk_size=256, stride=128)
passages = chunker.chunk_corpus(
    documents,
    output_path="data/processed/passages.jsonl"
)
"""





doc_stream = load_from_wikiextractor_streaming()
total_passages = chunk_corpus_streaming(
    doc_stream,
    output_path="data/processed/passages.jsonl"
)


metadata_store = PassageMetadataStore()

metadata_store.load_from_jsonl("data/processed/passages.jsonl")
metadata_store.save("data/processed/metadata.pkl")

print(f"Processing complete!")
print(f"Total passages: {len(metadata_store.passage_metadata)}")


