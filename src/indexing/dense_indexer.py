import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import numpy as np
import torch
from tqdm import tqdm


from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DenseSearchResult:
    """Represents a single dense search result."""
    passage_id: str
    score: float
    text: str
    doc_id: Optional[str] = None
    title: Optional[str] = None
    section: Optional[str] = None


class FAISSIndexer:
    """Dense retrieval using FAISS with sentence transformers."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True
    ):
        """
        Initialize FAISSIndexer with sentence transformer model.
        
        Args:
            model_name: HuggingFace model name for embeddings
            device: 'cpu', 'cuda', 'mps', or None (auto-detect)
            normalize_embeddings: Normalize embeddings for cosine similarity
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        
        logger.info(f"Loading sentence transformer model: {model_name} on device: {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
        
        # FAISS index (will be initialized later)
        self.index = None
        self.index_type = None
        
        # Passage metadata
        self.passage_ids = []  # Maps FAISS index position to passage_id
        self.passage_metadata = {}  # Maps passage_id to full metadata
    
    def encode_passages(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts into dense embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings (n_texts, dimension)
        """
        logger.info(f"Encoding {len(texts)} texts with batch size {batch_size}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        
        return embeddings.astype('float32')
    
    def build_index(
        self,
        passages: List[Dict],
        batch_size: int = 32,
        index_type: str = "flat",
        nlist: int = 100,
        show_progress: bool = True
    ):
        """
        Build FAISS index from passages.
        
        Args:
            passages: List of passage dictionaries with 'text', 'passage_id', etc.
            batch_size: Batch size for encoding
            index_type: Type of FAISS index
                - "flat": Exact search (IndexFlatIP for cosine similarity)
                - "ivf": Inverted file index (faster, approximate)
                - "hnsw": Hierarchical NSW (fastest, approximate)
            nlist: Number of clusters for IVF index
            show_progress: Show progress bar
        """
        import faiss
        logger.info(f"Building FAISS index for {len(passages)} passages")
        
        # Extract texts and metadata
        texts = [p['text'] for p in passages]
        self.passage_ids = [p['passage_id'] for p in passages]
        
        # Store full metadata for each passage
        for passage in passages:
            self.passage_metadata[passage['passage_id']] = {
                'passage_id': passage['passage_id'],
                'doc_id': passage.get('doc_id'),
                'title': passage.get('title'),
                'section': passage.get('section'),
                'text': passage['text']
            }
        
        # Encode passages
        embeddings = self.encode_passages(texts, batch_size=batch_size, show_progress=show_progress)
        
        # Create FAISS index
        self.index_type = index_type
        
        if index_type == "flat":
            # Exact search using inner product (cosine similarity if normalized)
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("Created IndexFlatIP (exact search)")
            
        elif index_type == "ivf":
            # IVF index for faster approximate search
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
            # Train the index
            logger.info(f"Training IVF index with {nlist} clusters")
            self.index.train(embeddings)
            logger.info("IVF index trained")
            
        elif index_type == "hnsw":
            # HNSW index (fastest approximate search)
            M = 32  # Number of connections per layer
            self.index = faiss.IndexHNSWFlat(self.dimension, M)
            self.index.hnsw.efConstruction = 40
            logger.info(f"Created HNSW index (M={M})")
            
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add embeddings to index
        logger.info("Adding embeddings to FAISS index")
        self.index.add(embeddings)
        logger.info(f"Index built with {self.index.ntotal} vectors")
    
    def build_index_from_file(
        self,
        passages_file: str,
        batch_size: int = 32,
        index_type: str = "flat",
        nlist: int = 100,
        show_progress: bool = True,
        max_passages: Optional[int] = None
    ):
        """
        Build FAISS index directly from JSONL file (memory efficient for encoding).
        
        Args:
            passages_file: Path to passages JSONL file
            batch_size: Batch size for encoding
            index_type: Type of FAISS index
            nlist: Number of clusters for IVF index
            show_progress: Show progress bar
            max_passages: Maximum number of passages to index
        """
        import faiss
        logger.info(f"Building FAISS index from file: {passages_file}")
        
        # First pass: collect texts in batches and encode
        texts_batch = []
        passages_batch = []
        all_embeddings = []
        
        passage_count = 0
        
        with open(passages_file, 'r', encoding='utf-8') as f:
            iterator = tqdm(f, desc="Loading and encoding passages") if show_progress else f
            
            for line in iterator:
                if not line.strip():
                    continue
                
                passage = json.loads(line)
                texts_batch.append(passage['text'])
                passages_batch.append(passage)
                passage_count += 1
                
                # Encode in batches
                if len(texts_batch) >= batch_size * 10:  # Encode 10 batches at once
                    embeddings = self.encode_passages(
                        texts_batch,
                        batch_size=batch_size,
                        show_progress=False
                    )
                    all_embeddings.append(embeddings)
                    
                    # Store metadata
                    for p in passages_batch:
                        self.passage_ids.append(p['passage_id'])
                        self.passage_metadata[p['passage_id']] = {
                            'passage_id': p['passage_id'],
                            'doc_id': p.get('doc_id'),
                            'title': p.get('title'),
                            'section': p.get('section'),
                            'text': p['text']
                        }
                    
                    texts_batch = []
                    passages_batch = []
                
                if max_passages and passage_count >= max_passages:
                    break
        
        # Encode remaining passages
        if texts_batch:
            embeddings = self.encode_passages(
                texts_batch,
                batch_size=batch_size,
                show_progress=False
            )
            all_embeddings.append(embeddings)
            
            for p in passages_batch:
                self.passage_ids.append(p['passage_id'])
                self.passage_metadata[p['passage_id']] = {
                    'passage_id': p['passage_id'],
                    'doc_id': p.get('doc_id'),
                    'title': p.get('title'),
                    'section': p.get('section'),
                    'text': p['text']
                }
        
        # Concatenate all embeddings
        logger.info("Concatenating embeddings")
        all_embeddings = np.vstack(all_embeddings)
        
        # Create FAISS index
        self.index_type = index_type
        
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("Created IndexFlatIP (exact search)")
            
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
            logger.info(f"Training IVF index with {nlist} clusters")
            self.index.train(all_embeddings)
            logger.info("IVF index trained")
            
        elif index_type == "hnsw":
            M = 32
            self.index = faiss.IndexHNSWFlat(self.dimension, M)
            self.index.hnsw.efConstruction = 40
            logger.info(f"Created HNSW index (M={M})")
            
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add embeddings to index
        logger.info("Adding embeddings to FAISS index")
        self.index.add(all_embeddings)
        logger.info(f"Index built with {self.index.ntotal} vectors")
    
    def search(
        self,
        query: Union[str, List[str]],
        top_k: int = 100,
        nprobe: int = 10
    ) -> Union[List[DenseSearchResult], List[List[DenseSearchResult]]]:
        """
        Search the FAISS index.
        
        Args:
            query: Query string or list of query strings
            top_k: Number of results to return
            nprobe: Number of clusters to search (for IVF index)
            
        Returns:
            List of DenseSearchResult objects (or list of lists for multiple queries)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Handle single query vs batch
        is_single_query = isinstance(query, str)
        queries = [query] if is_single_query else query
        
        # Encode queries
        query_embeddings = self.encode_passages(queries, show_progress=False)
        
        # Set nprobe for IVF index
        if self.index_type == "ivf":
            self.index.nprobe = nprobe
        
        # Search
        distances, indices = self.index.search(query_embeddings, top_k)
        
        # Parse results
        all_results = []
        for query_distances, query_indices in zip(distances, indices):
            results = []
            for distance, idx in zip(query_distances, query_indices):
                if idx == -1:  # No more results
                    break
                
                passage_id = self.passage_ids[idx]
                metadata = self.passage_metadata[passage_id]
                
                result = DenseSearchResult(
                    passage_id=passage_id,
                    score=float(distance),  # Inner product score
                    text=metadata['text'],
                    doc_id=metadata.get('doc_id'),
                    title=metadata.get('title'),
                    section=metadata.get('section')
                )
                results.append(result)
            
            all_results.append(results)
        
        # Return single list if single query
        return all_results[0] if is_single_query else all_results
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 100,
        batch_size: int = 32,
        nprobe: int = 10
    ) -> List[List[DenseSearchResult]]:
        """
        Perform batch search efficiently.
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
            batch_size: Batch size for encoding queries
            nprobe: Number of clusters to search (for IVF index)
            
        Returns:
            List of result lists (one per query)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        logger.info(f"Performing batch search for {len(queries)} queries")
        
        # Encode all queries
        query_embeddings = self.encode_passages(queries, batch_size=batch_size, show_progress=True)
        
        # Set nprobe for IVF index
        if self.index_type == "ivf":
            self.index.nprobe = nprobe
        
        # Search
        distances, indices = self.index.search(query_embeddings, top_k)
        
        # Parse results
        all_results = []
        for query_distances, query_indices in zip(tqdm(distances, desc="Parsing results"), indices):
            results = []
            for distance, idx in zip(query_distances, query_indices):
                if idx == -1:
                    break
                
                passage_id = self.passage_ids[idx]
                metadata = self.passage_metadata[passage_id]
                
                result = DenseSearchResult(
                    passage_id=passage_id,
                    score=float(distance),
                    text=metadata['text'],
                    doc_id=metadata.get('doc_id'),
                    title=metadata.get('title'),
                    section=metadata.get('section')
                )
                results.append(result)
            
            all_results.append(results)
        
        return all_results
    
    def save_index(self, index_path: str, metadata_path: str):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index_path: Path to save FAISS index (.index file)
            metadata_path: Path to save metadata (.pkl file)
        """
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        import faiss
        
        # Create directories if needed
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        logger.info(f"Saving metadata to {metadata_path}")
        metadata = {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'passage_ids': self.passage_ids,
            'passage_metadata': self.passage_metadata,
            'normalize_embeddings': self.normalize_embeddings
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info("Index and metadata saved successfully")
    
    def load_index(self, index_path: str, metadata_path: str):
        """
        Load FAISS index and metadata from disk.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata file
        """
        # Load metadata
        import faiss
        logger.info(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.model_name = metadata['model_name']
        self.dimension = metadata['dimension']
        self.index_type = metadata['index_type']
        self.passage_ids = metadata['passage_ids']
        self.passage_metadata = metadata['passage_metadata']
        self.normalize_embeddings = metadata['normalize_embeddings']
        
        # Load FAISS index
        logger.info(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)
        
        logger.info(f"Loaded index with {self.index.ntotal} vectors")
    
    def get_index_stats(self) -> Dict:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {"error": "Index not built"}
        
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "num_vectors": self.index.ntotal,
            "num_passages": len(self.passage_ids),
            "normalize_embeddings": self.normalize_embeddings
        }


# Example usage
if __name__ == "__main__":
    # Initialize indexer
    """
    indexer = FAISSIndexer(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        # device=None,  # Auto-detect (uses MPS/CUDA if available)
        normalize_embeddings=True
    )
    """
    # Option 1: Build index from list
    """
    passages = []
    with open("data/processed/passages.jsonl", 'r') as f:
        for line in f:
            passages.append(json.loads(line))
    
    indexer.build_index(
        passages,
        batch_size=32,
        index_type="flat"  # or "ivf" or "hnsw"
    )
    """
    """
    # Option 2: Build from file (RECOMMENDED for large datasets)
    indexer.build_index_from_file(
        "data/processed/passages.jsonl",
        batch_size=32,
        index_type="flat",  # Use "ivf" for faster search on large datasets
        max_passages=100000  # Optional: limit for testing
    )
    """
    """
    # Save index
    indexer.save_index(
        index_path="indices/faiss/wikipedia.index",
        metadata_path="indices/faiss/wikipedia_metadata.pkl"
    )
    """
    indexer = FAISSIndexer()
    indexer.load_index(
        index_path="indices/faiss/wikipedia.index",
        metadata_path="indices/faiss/wikipedia_metadata.pkl"
    )
    # Get index stats
    stats = indexer.get_index_stats()
    print(f"Index stats: {stats}")
    
    # Test search
    query = "What is machine learning?"
    results = indexer.search(query, top_k=10)
    
    print(f"\nTop results for query: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.score:.4f}")
        print(f"   Title: {result.title}")
        print(f"   Text: {result.text[:200]}...")
        print()
    """
    # Test batch search
    queries = [
        "artificial intelligence",
        "neural networks",
        "deep learning"
    ]
    batch_results = indexer.batch_search(queries, top_k=5)
    
    print(f"\nBatch search results for {len(queries)} queries")
    for query, results in zip(queries, batch_results):
        print(f"Query: '{query}' - Found {len(results)} results")
    
    # Later: Load index
    """
    """
    indexer_new = FAISSIndexer()
    indexer_new.load_index(
        index_path="indices/faiss/wikipedia.index",
        metadata_path="indices/faiss/wikipedia_metadata.pkl"
    )
    """
    