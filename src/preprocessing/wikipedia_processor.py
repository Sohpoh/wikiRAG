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


class WikipediaLoader:
    """Load and parse Wikipedia dump files."""
    
    def __init__(self, dump_path: str):
        """
        Initialize Wikipedia loader.
        
        Args:
            dump_path: Path to Wikipedia dump directory (WikiExtractor output)
        """
        self.dump_path = Path(dump_path)
        if not self.dump_path.exists():
            raise FileNotFoundError(f"Dump directory not found: {dump_path}")
    
    def _find_wiki_files(self) -> List[Path]:
        """
        Find all wiki_* files in the WikiExtractor output directory.
        
        WikiExtractor creates structure like:
        data/raw/
          AA/
            wiki_00
            wiki_01
          AB/
            wiki_00
            ...
            
        Returns:
            List of paths to wiki_* files
        """
        wiki_files = []
        
        # Search for all wiki_* files recursively
        for wiki_file in self.dump_path.rglob("wiki_*"):
            if wiki_file.is_file():
                wiki_files.append(wiki_file)
        
        wiki_files.sort()  # Sort for consistent ordering
        logger.info(f"Found {len(wiki_files)} wiki files")
        return wiki_files
    
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
    
    def load_from_wikiextractor(
        self, 
        max_docs: Optional[int] = None,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Load Wikipedia articles from WikiExtractor output directory.
        
        Args:
            max_docs: Maximum number of documents to load
            show_progress: Show progress bar
            
        Returns:
            List of document dictionaries
        """
        documents = []
        wiki_files = self._find_wiki_files()
        
        if not wiki_files:
            raise ValueError(f"No wiki_* files found in {self.dump_path}")
        
        logger.info(f"Loading Wikipedia articles from {len(wiki_files)} files")
        
        doc_count = 0
        file_iterator = tqdm(wiki_files, desc="Processing wiki files") if show_progress else wiki_files
        
        for wiki_file in file_iterator:
            for doc in self._parse_wiki_file(wiki_file):
                documents.append(doc)
                doc_count += 1
                
                if max_docs and doc_count >= max_docs:
                    logger.info(f"Reached max_docs limit: {max_docs}")
                    return documents
        
        logger.info(f"Loaded {len(documents)} documents from {len(wiki_files)} files")
        return documents
    
    def load_from_wikiextractor_streaming(
        self, 
        max_docs: Optional[int] = None
    ) -> Generator[Dict, None, None]:
        """
        Stream Wikipedia articles from WikiExtractor output (memory efficient).
        
        Args:
            max_docs: Maximum number of documents to load
            
        Yields:
            Document dictionaries
        """
        wiki_files = self._find_wiki_files()
        
        if not wiki_files:
            raise ValueError(f"No wiki_* files found in {self.dump_path}")
        
        logger.info(f"Streaming Wikipedia articles from {len(wiki_files)} files")
        
        doc_count = 0
        
        for wiki_file in wiki_files:
            for doc in self._parse_wiki_file(wiki_file):
                yield doc
                doc_count += 1
                
                if max_docs and doc_count >= max_docs:
                    logger.info(f"Reached max_docs limit: {max_docs}")
                    return
        
        logger.info(f"Streamed {doc_count} documents")
    
    # Keep the old method for backward compatibility
    def load_from_jsonl(self, max_docs: Optional[int] = None) -> List[Dict]:
        """
        Load Wikipedia articles from a single JSONL file.
        
        Args:
            max_docs: Maximum number of documents to load
            
        Returns:
            List of document dictionaries
        """
        if self.dump_path.is_dir():
            logger.warning("Path is a directory. Use load_from_wikiextractor() instead.")
            return self.load_from_wikiextractor(max_docs=max_docs)
        
        documents = []
        
        logger.info(f"Loading Wikipedia dump from {self.dump_path}")
        
        with open(self.dump_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(tqdm(f, desc="Loading documents")):
                if max_docs and idx >= max_docs:
                    break
                    
                try:
                    doc = json.loads(line.strip())
                    if 'id' in doc and 'title' in doc and 'text' in doc:
                        documents.append({
                            'doc_id': str(doc['id']),
                            'title': doc['title'],
                            'text': doc['text']
                        })
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed line {idx}")
                    continue
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents


class DocumentChunker:
    """Split documents into overlapping passages."""
    
    def __init__(
        self,
        chunk_size: int = 256,
        stride: int = 128,
        min_chunk_size: int = 50
    ):
        """
        Initialize document chunker.
        
        Args:
            chunk_size: Target number of tokens per passage
            stride: Number of tokens to overlap between passages
            min_chunk_size: Minimum tokens for a valid passage
        """
        self.chunk_size = chunk_size
        self.stride = stride
        self.min_chunk_size = min_chunk_size
        
    def simple_tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace tokenizer (approximate).
        For production, use a proper tokenizer (e.g., from transformers).
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Simple split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        return tokens
    
    def extract_sections(self, text: str, title: str) -> List[Tuple[str, str]]:
        """
        Extract sections from Wikipedia text.
        
        WikiExtractor preserves section structure with newlines.
        
        Args:
            text: Full document text
            title: Document title
            
        Returns:
            List of (section_title, section_text) tuples
        """
        sections = []
        
        # WikiExtractor uses simple newline separation for paragraphs
        # Try to detect section-like structures
        lines = text.split('\n')
        
        current_section = "Introduction"
        current_text = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Heuristic: if line is short and doesn't end with punctuation, might be header
            # This is imperfect but works reasonably well
            if len(line) < 100 and not line.endswith(('.', '!', '?', ';', ':')):
                # Save previous section
                if current_text:
                    sections.append((current_section, '\n'.join(current_text)))
                    current_text = []
                current_section = line
            else:
                current_text.append(line)
        
        # Add final section
        if current_text:
            sections.append((current_section, '\n'.join(current_text)))
        
        # If no sections found, treat entire text as one section
        if not sections:
            sections.append((title, text))
        
        return sections
    
    def clean_text(self, text: str) -> str:
        """
        Clean Wikipedia markup and special characters.
        
        WikiExtractor already does most cleaning, but we do a bit more.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove any remaining HTML entities
        text = re.sub(r'&[a-z]+;', ' ', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove any remaining brackets or special chars
        text = re.sub(r'[\[\]{}]', '', text)
        
        return text.strip()
    
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
                end_idx = min(start_idx + self.chunk_size, len(tokens))
                chunk_tokens = tokens[start_idx:end_idx]
                
                # Skip if chunk is too small (unless it's the last chunk)
                if len(chunk_tokens) < self.min_chunk_size and end_idx < len(tokens):
                    start_idx += self.stride
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
                start_idx += self.stride
        
        return passages
    
    def chunk_corpus(
        self,
        documents: List[Dict],
        output_path: Optional[str] = None,
        batch_save: bool = False,
        batch_size: int = 10000
    ) -> List[Passage]:
        """
        Chunk entire corpus of documents.
        
        Args:
            documents: List of document dicts with 'doc_id', 'title', 'text'
            output_path: Optional path to save passages as JSONL
            batch_save: Save passages in batches (memory efficient)
            batch_size: Number of passages per batch when batch_save=True
            
        Returns:
            List of all passages (empty if batch_save=True)
        """
        all_passages = []
        batch_passages = []
        
        logger.info(f"Chunking {len(documents)} documents...")
        
        if output_path and batch_save:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            f_out = open(output_file, 'w', encoding='utf-8')
        
        for doc in tqdm(documents, desc="Chunking documents"):
            passages = self.chunk_document(
                doc_id=doc['doc_id'],
                title=doc['title'],
                text=doc['text']
            )
            
            if batch_save and output_path:
                # Save in batches to avoid memory issues
                batch_passages.extend(passages)
                if len(batch_passages) >= batch_size:
                    for p in batch_passages:
                        f_out.write(json.dumps(p.to_dict()) + '\n')
                    batch_passages = []
            else:
                all_passages.extend(passages)
        
        # Save remaining batch
        if batch_save and output_path and batch_passages:
            for p in batch_passages:
                f_out.write(json.dumps(p.to_dict()) + '\n')
            f_out.close()
            logger.info(f"Saved passages to {output_path}")
            return []  # Don't return passages to save memory
        
        logger.info(f"Created {len(all_passages)} passages from {len(documents)} documents")
        
        # Save if output path provided
        if output_path and not batch_save:
            self.save_passages(all_passages, output_path)
        
        return all_passages
    
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
    
    def save_passages(self, passages: List[Passage], output_path: str):
        """Save passages to JSONL file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving passages to {output_path}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for passage in passages:
                f.write(json.dumps(passage.to_dict()) + '\n')
        
        logger.info(f"Saved {len(passages)} passages")


class PassageMetadataStore:
    """Store and retrieve passage metadata mappings."""
    
    def __init__(self):
        self.passage_to_doc = {}  # passage_id -> doc_id
        self.doc_to_passages = {}  # doc_id -> [passage_ids]
        self.passage_metadata = {}  # passage_id -> Passage object
        
    def add_passage(self, passage: Passage):
        """Add a passage to the metadata store."""
        self.passage_to_doc[passage.passage_id] = passage.doc_id
        
        if passage.doc_id not in self.doc_to_passages:
            self.doc_to_passages[passage.doc_id] = []
        self.doc_to_passages[passage.doc_id].append(passage.passage_id)
        
        self.passage_metadata[passage.passage_id] = passage
    
    def load_from_passages(self, passages: List[Passage]):
        """Load metadata from a list of passages."""
        logger.info(f"Loading metadata for {len(passages)} passages")
        for passage in tqdm(passages, desc="Building metadata"):
            self.add_passage(passage)
    
    def load_from_jsonl(self, filepath: str):
        """Load metadata from passages JSONL file (memory efficient)."""
        logger.info(f"Loading metadata from {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading metadata"):
                passage_dict = json.loads(line)
                passage = Passage(**passage_dict)
                self.add_passage(passage)
    
    def get_doc_id(self, passage_id: str) -> Optional[str]:
        """Get document ID for a passage."""
        return self.passage_to_doc.get(passage_id)
    
    def get_passages_for_doc(self, doc_id: str) -> List[str]:
        """Get all passage IDs for a document."""
        return self.doc_to_passages.get(doc_id, [])
    
    def get_passage(self, passage_id: str) -> Optional[Passage]:
        """Get full passage metadata."""
        return self.passage_metadata.get(passage_id)
    
    def save(self, filepath: str):
        """Save metadata store to disk."""
        logger.info(f"Saving metadata store to {filepath}")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'passage_to_doc': self.passage_to_doc,
                'doc_to_passages': self.doc_to_passages,
                'passage_metadata': self.passage_metadata
            }, f)
    
    def load(self, filepath: str):
        """Load metadata store from disk."""
        logger.info(f"Loading metadata store from {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.passage_to_doc = data['passage_to_doc']
            self.doc_to_passages = data['doc_to_passages']
            self.passage_metadata = data['passage_metadata']


# Example usage
if __name__ == "__main__":
    # Example: Load and chunk Wikipedia documents from WikiExtractor output
    
    # Option 1: Load all into memory (for smaller datasets)
    loader = WikipediaLoader("data/raw/")
    documents = loader.load_from_wikiextractor(max_docs=10000)
    
    chunker = DocumentChunker(chunk_size=256, stride=128)
    passages = chunker.chunk_corpus(
        documents,
        output_path="data/processed/passages.jsonl"
    )
    
    # Option 2: Streaming mode (for large datasets - RECOMMENDED for 5GB dump)
    # This is most memory efficient
    """
    loader = WikipediaLoader("data/raw/")
    chunker = DocumentChunker(chunk_size=256, stride=128)
    
    doc_stream = loader.load_from_wikiextractor_streaming()
    total_passages = chunker.chunk_corpus_streaming(
        doc_stream,
        output_path="data/processed/passages.jsonl"
    )
    """
    
    # Build metadata store
    metadata_store = PassageMetadataStore()
    metadata_store.load_from_jsonl("data/processed/passages.jsonl")
    metadata_store.save("data/processed/metadata.pkl")
    
    print(f"Processing complete!")
    print(f"Total passages: {len(metadata_store.passage_metadata)}")


    