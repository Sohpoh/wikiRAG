import logging
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
import torch
from tqdm import tqdm

from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RerankedResult:
    """Represents a reranked search result."""
    passage_id: str
    text: str
    original_score: float
    rerank_score: float
    original_rank: int
    rerank_rank: int
    doc_id: Optional[str] = None
    title: Optional[str] = None
    section: Optional[str] = None


class CrossEncoderReranker:
    """
    Rerank passages using a cross-encoder model.
    Cross-encoders jointly encode query and passage for more accurate scoring.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace cross-encoder model name
            device: 'cpu', 'cuda', 'mps', or None (auto-detect)
            max_length: Maximum sequence length for cross-encoder
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
        self.max_length = max_length
        
        logger.info(f"Loading cross-encoder model: {model_name} on device: {device}")
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        logger.info(f"Cross-encoder loaded on {device}")
    
    def rerank(
        self,
        query: str,
        passages: List,
        top_k: int = 10,
        batch_size: int = 32,
        show_progress: bool = False,
        text_attr: str = "text",
        score_attr: str = "combined_score"
    ) -> List[RerankedResult]:
        """
        Rerank passages using cross-encoder.
        
        Args:
            query: Search query
            passages: List of passage objects
            top_k: Number of top results to return after reranking
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            text_attr: Attribute name for passage text
            score_attr: Attribute name for original score
            
        Returns:
            List of RerankedResult objects sorted by rerank score
        """
        if not passages:
            logger.warning("No passages to rerank")
            return []
        
        logger.info(f"Reranking {len(passages)} passages with cross-encoder")
        
        # Prepare query-passage pairs
        pairs = []
        for passage in passages:
            text = getattr(passage, text_attr, "")
            pairs.append([query, text])
        
        # Score with cross-encoder
        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Create reranked results
        reranked_results = []
        for original_rank, (passage, score) in enumerate(zip(passages, scores), 1):
            original_score = getattr(passage, score_attr, 0.0)
            
            result = RerankedResult(
                passage_id=getattr(passage, "passage_id", f"unknown_{original_rank}"),
                text=getattr(passage, text_attr, ""),
                original_score=original_score,
                rerank_score=float(score),
                original_rank=original_rank,
                rerank_rank=0,  # Will be set after sorting
                doc_id=getattr(passage, "doc_id", None),
                title=getattr(passage, "title", None),
                section=getattr(passage, "section", None)
            )
            reranked_results.append(result)
        
        # Sort by rerank score (descending)
        reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Set rerank ranks
        for rank, result in enumerate(reranked_results, 1):
            result.rerank_rank = rank
        
        # Return top-k
        top_results = reranked_results[:top_k]
        
        logger.info(f"Reranking complete. Returning top {len(top_results)} results")
        
        return top_results
    
    def rerank_with_bundles(
        self,
        query: str,
        passages: List,
        bundle_size: int = 2,
        top_k: int = 10,
        batch_size: int = 32,
        show_progress: bool = False,
        text_attr: str = "text"
    ) -> List[RerankedResult]:
        """
        Rerank passages where each "passage" is a bundle of 2-3 sentences.
        Useful for providing more context to the cross-encoder.
        
        Args:
            query: Search query
            passages: List of passage objects
            bundle_size: Number of consecutive passages to bundle together
            top_k: Number of top bundles to return
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            text_attr: Attribute name for passage text
            
        Returns:
            List of RerankedResult objects for top bundles
        """
        if not passages:
            logger.warning("No passages to rerank")
            return []
        
        logger.info(f"Reranking with bundles of size {bundle_size}")
        
        # Create bundles
        bundles = []
        bundle_metadata = []
        
        for i in range(0, len(passages), bundle_size):
            bundle_passages = passages[i:i + bundle_size]
            
            # Concatenate texts
            bundle_text = " ".join([
                getattr(p, text_attr, "") for p in bundle_passages
            ])
            
            bundles.append([query, bundle_text])
            bundle_metadata.append({
                "passages": bundle_passages,
                "start_idx": i,
                "size": len(bundle_passages)
            })
        
        # Score bundles
        scores = self.model.predict(
            bundles,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Create results for bundles
        reranked_bundles = []
        for idx, (metadata, score) in enumerate(zip(bundle_metadata, scores)):
            # Use first passage in bundle for metadata
            first_passage = metadata["passages"][0]
            
            result = RerankedResult(
                passage_id=f"bundle_{idx}",
                text=bundles[idx][1],  # Bundle text
                original_score=0.0,
                rerank_score=float(score),
                original_rank=idx + 1,
                rerank_rank=0,
                doc_id=getattr(first_passage, "doc_id", None),
                title=getattr(first_passage, "title", None),
                section=getattr(first_passage, "section", None)
            )
            reranked_bundles.append(result)
        
        # Sort by score
        reranked_bundles.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Set ranks
        for rank, result in enumerate(reranked_bundles, 1):
            result.rerank_rank = rank
        
        return reranked_bundles[:top_k]
    
    def rerank_batch(
        self,
        queries: List[str],
        passages_list: List[List],
        top_k: int = 10,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[List[RerankedResult]]:
        """
        Rerank multiple queries and their passages in batch.
        
        Args:
            queries: List of query strings
            passages_list: List of passage lists (one per query)
            top_k: Number of top results per query
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            List of reranked result lists (one per query)
        """
        logger.info(f"Batch reranking for {len(queries)} queries")
        
        all_results = []
        
        iterator = zip(queries, passages_list)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Reranking queries")
        
        for query, passages in iterator:
            results = self.rerank(
                query=query,
                passages=passages,
                top_k=top_k,
                batch_size=batch_size,
                show_progress=False
            )
            all_results.append(results)
        
        return all_results
    
    def analyze_reranking_effect(
        self,
        query: str,
        passages: List,
        top_k: int = 10,
        text_attr: str = "text",
        score_attr: str = "combined_score"
    ) -> Dict:
        """
        Analyze how reranking changes the ranking.
        Useful for understanding reranking behavior.
        
        Args:
            query: Search query
            passages: List of passage objects
            top_k: Number of results to analyze
            text_attr: Attribute name for passage text
            score_attr: Attribute name for original score
            
        Returns:
            Dictionary with reranking statistics
        """
        reranked = self.rerank(
            query=query,
            passages=passages,
            top_k=top_k,
            text_attr=text_attr,
            score_attr=score_attr
        )
        
        # Calculate metrics
        original_ranks = [r.original_rank for r in reranked]
        rerank_ranks = [r.rerank_rank for r in reranked]
        
        # Rank changes
        rank_changes = [orig - rerank for orig, rerank in zip(original_ranks, rerank_ranks)]
        avg_rank_change = np.mean(np.abs(rank_changes))
        
        # Passages that moved up significantly
        moved_up = [i for i, change in enumerate(rank_changes) if change > 5]
        moved_down = [i for i, change in enumerate(rank_changes) if change < -5]
        
        # Score correlation
        original_scores = [r.original_score for r in reranked]
        rerank_scores = [r.rerank_score for r in reranked]
        score_correlation = np.corrcoef(original_scores, rerank_scores)[0, 1]
        
        analysis = {
            "query": query,
            "num_passages": len(passages),
            "top_k": top_k,
            "avg_rank_change": float(avg_rank_change),
            "max_rank_improvement": max(rank_changes) if rank_changes else 0,
            "max_rank_decline": min(rank_changes) if rank_changes else 0,
            "passages_moved_up_significantly": len(moved_up),
            "passages_moved_down_significantly": len(moved_down),
            "score_correlation": float(score_correlation),
            "top_original_in_reranked": sum(1 for r in reranked if r.original_rank <= top_k),
            "new_in_top_k": sum(1 for r in reranked if r.original_rank > top_k)
        }
        
        return analysis
    
    def score_passages(
        self,
        query: str,
        passages: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Score query-passage pairs directly (returns raw scores).
        
        Args:
            query: Query string
            passages: List of passage texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Numpy array of scores
        """
        pairs = [[query, passage] for passage in passages]
        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return scores


# Example usage with real data
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    sys.path.append(str(Path(__file__).parent.parent))
    
    from indexing.sparse_indexer import ElasticsearchIndexer
    from indexing.dense_indexer import FAISSIndexer
    from retrieval.hybrid_retriever import HybridRetriever
    
    print("=" * 80)
    print("CROSS-ENCODER RERANKING TEST WITH REAL DATA")
    print("=" * 80)
    
    # Initialize sparse indexer
    print("\n1. Initializing Elasticsearch indexer...")
    sparse_indexer = ElasticsearchIndexer(
        index_name="wikipedia_bm25",
        host="localhost",
        port=9200
    )
    
    # Check if index exists
    stats = sparse_indexer.get_index_stats()
    if "error" in stats:
        print(f"Error: {stats['error']}")
        print("Please make sure Elasticsearch is running and the index is created.")
        sys.exit(1)
    
    print(f"   ✓ Connected to Elasticsearch")
    print(f"   ✓ Index has {stats['document_count']} documents")
    
    # Initialize dense indexer
    print("\n2. Initializing FAISS indexer...")
    dense_indexer = FAISSIndexer(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )
    
    # Load FAISS index
    try:
        dense_indexer.load_index(
            index_path="indices/faiss/wikipedia.index",
            metadata_path="indices/faiss/wikipedia_metadata.pkl"
        )
        dense_stats = dense_indexer.get_index_stats()
        print(f"   ✓ Loaded FAISS index")
        print(f"   ✓ Index has {dense_stats['num_vectors']} vectors")
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        print("Please make sure the FAISS index is built.")
        sys.exit(1)
    
    # Initialize hybrid retriever
    print("\n3. Initializing hybrid retriever...")
    hybrid_retriever = HybridRetriever(
        sparse_indexer=sparse_indexer,
        dense_indexer=dense_indexer,
        sparse_weight=0.5,
        dense_weight=0.5
    )
    print("   ✓ Hybrid retriever ready")
    
    # Initialize cross-encoder reranker
    print("\n4. Initializing cross-encoder reranker...")
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        # device=None  # Auto-detect
    )
    print("   ✓ Cross-encoder ready")
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain quantum computing",
    ]
    
    for query_idx, query in enumerate(test_queries, 1):
        print("\n" + "=" * 80)
        print(f"QUERY {query_idx}: {query}")
        print("=" * 80)
        
        # Step 1: Hybrid retrieval
        print("\n--- Hybrid Retrieval (Top 50 passages) ---")
        hybrid_results = hybrid_retriever.hybrid_search(
            query=query,
            top_k_sparse=100,
            top_k_dense=100,
            final_top_k=50,
            fusion_method="rrf"
        )
        
        print(f"Retrieved {len(hybrid_results)} passages")
        
        # Show top 5 from hybrid retrieval
        print("\nTop 5 by Hybrid Score:")
        for i, result in enumerate(hybrid_results[:5], 1):
            print(f"\n{i}. Passage ID: {result.passage_id}")
            print(f"   Title: {result.title}")
            print(f"   Combined Score: {result.combined_score:.4f}")
            print(f"   Sparse: {result.sparse_score:.4f} (rank {result.sparse_rank})")
            print(f"   Dense: {result.dense_score:.4f} (rank {result.dense_rank})")
            print(f"   Text: {result.text[:150]}...")
        
        # Step 2: Cross-encoder reranking
        print("\n\n--- Cross-Encoder Reranking (Top 10) ---")
        reranked_results = reranker.rerank(
            query=query,
            passages=hybrid_results,
            top_k=10,
            batch_size=32,
            show_progress=True
        )
        
        print(f"\nReranked to {len(reranked_results)} passages")
        
        # Show reranked results
        print("\nTop 10 after Reranking:")
        for i, result in enumerate(reranked_results, 1):
            rank_change = result.original_rank - result.rerank_rank
            change_symbol = "↑" if rank_change > 0 else "↓" if rank_change < 0 else "="
            
            print(f"\n{i}. Passage ID: {result.passage_id}")
            print(f"   Title: {result.title}")
            print(f"   Rerank Score: {result.rerank_score:.4f}")
            print(f"   Original Score: {result.original_score:.4f}")
            print(f"   Rank Change: {result.original_rank} → {result.rerank_rank} "
                  f"({change_symbol} {abs(rank_change)})")
            print(f"   Text: {result.text[:150]}...")
        
        # Step 3: Reranking analysis
        print("\n\n--- Reranking Analysis ---")
        analysis = reranker.analyze_reranking_effect(
            query=query,
            passages=hybrid_results,
            top_k=10
        )
        
        print(f"Average rank change: {analysis['avg_rank_change']:.2f}")
        print(f"Max rank improvement: +{analysis['max_rank_improvement']}")
        print(f"Max rank decline: {analysis['max_rank_decline']}")
        print(f"Score correlation (original vs rerank): {analysis['score_correlation']:.4f}")
        print(f"Original top-10 still in reranked top-10: {analysis['top_original_in_reranked']}/10")
        print(f"New passages in top-10: {analysis['new_in_top_k']}")
    """
    # Batch reranking test
    print("\n\n" + "=" * 80)
    print("BATCH RERANKING TEST")
    print("=" * 80)
    
    print("\nRetrieving passages for all queries...")
    all_hybrid_results = []
    for query in test_queries:
        results = hybrid_retriever.hybrid_search(
            query=query,
            final_top_k=50,
            fusion_method="rrf"
        )
        all_hybrid_results.append(results)
    
    print(f"Performing batch reranking for {len(test_queries)} queries...")
    batch_reranked = reranker.rerank_batch(
        queries=test_queries,
        passages_list=all_hybrid_results,
        top_k=5,
        show_progress=True
    )
    
    print("\nBatch Reranking Results:")
    for query, results in zip(test_queries, batch_reranked):
        print(f"\nQuery: {query}")
        print(f"Top result: {results[0].title}")
        print(f"Rerank score: {results[0].rerank_score:.4f}")
        print(f"Text preview: {results[0].text[:100]}...")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE!")
    print("=" * 80)
    """