import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from indexing.sparse_indexer2 import ElasticsearchIndexer, SearchResult
from indexing.dense_indexer import FAISSIndexer, DenseSearchResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """Represents a hybrid search result with combined scores."""
    passage_id: str
    sparse_score: float
    dense_score: float
    combined_score: float
    text: str
    doc_id: Optional[str] = None
    title: Optional[str] = None
    section: Optional[str] = None
    sparse_rank: Optional[int] = None
    dense_rank: Optional[int] = None


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 (sparse) and dense vector search.
    Uses Reciprocal Rank Fusion (RRF) to combine rankings.
    """
    
    def __init__(
        self,
        sparse_indexer: ElasticsearchIndexer,
        dense_indexer: FAISSIndexer,
        sparse_weight: float = 0.5,
        dense_weight: float = 0.5
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            sparse_indexer: Elasticsearch BM25 indexer
            dense_indexer: FAISS dense indexer
            sparse_weight: Weight for sparse retrieval (0-1)
            dense_weight: Weight for dense retrieval (0-1)
        """
        self.sparse_indexer = sparse_indexer
        self.dense_indexer = dense_indexer
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        
        logger.info(f"HybridRetriever initialized (sparse_weight={sparse_weight}, dense_weight={dense_weight})")
    
    def retrieve_sparse(
        self,
        query: str,
        top_k: int = 100
    ) -> List[SearchResult]:
        """
        Retrieve using BM25 sparse retrieval.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        return self.sparse_indexer.search(query, top_k=top_k)
    
    def retrieve_dense(
        self,
        query: str,
        top_k: int = 100
    ) -> List[DenseSearchResult]:
        """
        Retrieve using dense vector search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of DenseSearchResult objects
        """
        return self.dense_indexer.search(query, top_k=top_k)
    
    def reciprocal_rank_fusion(
        self,
        sparse_results: List[SearchResult],
        dense_results: List[DenseSearchResult],
        k: int = 60
    ) -> List[HybridSearchResult]:
        """
        Combine sparse and dense results using Reciprocal Rank Fusion (RRF).
        
        RRF formula: score(d) = Σ(1 / (k + rank(d)))
        
        Args:
            sparse_results: Results from sparse retrieval
            dense_results: Results from dense retrieval
            k: RRF constant (typically 60)
            
        Returns:
            List of HybridSearchResult objects sorted by combined score
        """
        # Create mappings: passage_id -> (rank, score)
        sparse_map = {
            result.passage_id: (rank + 1, result.score)
            for rank, result in enumerate(sparse_results)
        }
        
        dense_map = {
            result.passage_id: (rank + 1, result.score)
            for rank, result in enumerate(dense_results)
        }
        
        # Get all unique passage IDs
        all_passage_ids = set(sparse_map.keys()) | set(dense_map.keys())
        
        # Calculate RRF scores
        hybrid_results = []
        
        for passage_id in all_passage_ids:
            # Get ranks (use large rank if not present)
            sparse_rank = sparse_map.get(passage_id, (len(sparse_results) + 1000, 0.0))[0]
            dense_rank = dense_map.get(passage_id, (len(dense_results) + 1000, 0.0))[0]
            
            # Get original scores
            sparse_score = sparse_map.get(passage_id, (0, 0.0))[1]
            dense_score = dense_map.get(passage_id, (0, 0.0))[1]
            
            # Calculate RRF score
            sparse_rrf = self.sparse_weight / (k + sparse_rank)
            dense_rrf = self.dense_weight / (k + dense_rank)
            combined_score = sparse_rrf + dense_rrf
            
            # Get metadata (prefer sparse results as they have same info)
            metadata = None
            text = ""
            doc_id = None
            title = None
            section = None
            
            if passage_id in sparse_map:
                # Find the result in sparse_results
                for result in sparse_results:
                    if result.passage_id == passage_id:
                        text = result.text
                        doc_id = result.doc_id
                        title = result.title
                        section = result.section
                        break
            elif passage_id in dense_map:
                # Find the result in dense_results
                for result in dense_results:
                    if result.passage_id == passage_id:
                        text = result.text
                        doc_id = result.doc_id
                        title = result.title
                        section = result.section
                        break
            
            hybrid_result = HybridSearchResult(
                passage_id=passage_id,
                sparse_score=sparse_score,
                dense_score=dense_score,
                combined_score=combined_score,
                text=text,
                doc_id=doc_id,
                title=title,
                section=section,
                sparse_rank=sparse_rank if passage_id in sparse_map else None,
                dense_rank=dense_rank if passage_id in dense_map else None
            )
            
            hybrid_results.append(hybrid_result)
        
        # Sort by combined score (descending)
        hybrid_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return hybrid_results
    
    def weighted_score_fusion(
        self,
        sparse_results: List[SearchResult],
        dense_results: List[DenseSearchResult],
        normalize: bool = True
    ) -> List[HybridSearchResult]:
        """
        Combine sparse and dense results using weighted score fusion.
        
        Args:
            sparse_results: Results from sparse retrieval
            dense_results: Results from dense retrieval
            normalize: Normalize scores before combining
            
        Returns:
            List of HybridSearchResult objects sorted by combined score
        """
        # Create mappings: passage_id -> score
        sparse_map = {result.passage_id: result.score for result in sparse_results}
        dense_map = {result.passage_id: result.score for result in dense_results}
        
        # Normalize scores if requested
        if normalize:
            if sparse_map:
                sparse_scores = list(sparse_map.values())
                sparse_min, sparse_max = min(sparse_scores), max(sparse_scores)
                if sparse_max > sparse_min:
                    sparse_map = {
                        pid: (score - sparse_min) / (sparse_max - sparse_min)
                        for pid, score in sparse_map.items()
                    }
            
            if dense_map:
                dense_scores = list(dense_map.values())
                dense_min, dense_max = min(dense_scores), max(dense_scores)
                if dense_max > dense_min:
                    dense_map = {
                        pid: (score - dense_min) / (dense_max - dense_min)
                        for pid, score in dense_map.items()
                    }
        
        # Get all unique passage IDs
        all_passage_ids = set(sparse_map.keys()) | set(dense_map.keys())
        
        # Calculate weighted scores
        hybrid_results = []
        
        for passage_id in all_passage_ids:
            sparse_score = sparse_map.get(passage_id, 0.0)
            dense_score = dense_map.get(passage_id, 0.0)
            
            # Weighted combination
            combined_score = (
                self.sparse_weight * sparse_score +
                self.dense_weight * dense_score
            )
            
            # Get metadata
            text = ""
            doc_id = None
            title = None
            section = None
            
            if passage_id in sparse_map:
                for result in sparse_results:
                    if result.passage_id == passage_id:
                        text = result.text
                        doc_id = result.doc_id
                        title = result.title
                        section = result.section
                        break
            elif passage_id in dense_map:
                for result in dense_results:
                    if result.passage_id == passage_id:
                        text = result.text
                        doc_id = result.doc_id
                        title = result.title
                        section = result.section
                        break
            
            hybrid_result = HybridSearchResult(
                passage_id=passage_id,
                sparse_score=sparse_score,
                dense_score=dense_score,
                combined_score=combined_score,
                text=text,
                doc_id=doc_id,
                title=title,
                section=section
            )
            
            hybrid_results.append(hybrid_result)
        
        # Sort by combined score (descending)
        hybrid_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return hybrid_results
    
    def hybrid_search(
        self,
        query: str,
        top_k_sparse: int = 100,
        top_k_dense: int = 100,
        final_top_k: int = 50,
        fusion_method: str = "rrf",
        rrf_k: int = 60
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search combining sparse and dense retrieval.
        
        Args:
            query: Search query
            top_k_sparse: Number of results from sparse retrieval
            top_k_dense: Number of results from dense retrieval
            final_top_k: Number of final results to return
            fusion_method: Fusion method - "rrf" or "weighted"
            rrf_k: RRF constant (only used if fusion_method="rrf")
            
        Returns:
            List of top-k HybridSearchResult objects
        """
        logger.info(f"Hybrid search for query: '{query[:50]}...'")
        
        # Retrieve from both indices
        logger.info(f"Retrieving {top_k_sparse} results from sparse index")
        sparse_results = self.retrieve_sparse(query, top_k=top_k_sparse)
        
        logger.info(f"Retrieving {top_k_dense} results from dense index")
        dense_results = self.retrieve_dense(query, top_k=top_k_dense)
        
        # Combine results
        logger.info(f"Combining results using {fusion_method} fusion")
        
        if fusion_method == "rrf":
            hybrid_results = self.reciprocal_rank_fusion(
                sparse_results,
                dense_results,
                k=rrf_k
            )
        elif fusion_method == "weighted":
            hybrid_results = self.weighted_score_fusion(
                sparse_results,
                dense_results,
                normalize=True
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Return top-k results
        final_results = hybrid_results[:final_top_k]
        
        logger.info(f"Returning top {len(final_results)} hybrid results")
        
        return final_results
    
    def batch_hybrid_search(
        self,
        queries: List[str],
        top_k_sparse: int = 100,
        top_k_dense: int = 100,
        final_top_k: int = 50,
        fusion_method: str = "rrf",
        rrf_k: int = 60
    ) -> List[List[HybridSearchResult]]:
        """
        Perform batch hybrid search for multiple queries.
        
        Args:
            queries: List of search queries
            top_k_sparse: Number of results from sparse retrieval per query
            top_k_dense: Number of results from dense retrieval per query
            final_top_k: Number of final results to return per query
            fusion_method: Fusion method - "rrf" or "weighted"
            rrf_k: RRF constant
            
        Returns:
            List of result lists (one per query)
        """
        logger.info(f"Batch hybrid search for {len(queries)} queries")
        
        all_results = []
        
        for query in queries:
            results = self.hybrid_search(
                query,
                top_k_sparse=top_k_sparse,
                top_k_dense=top_k_dense,
                final_top_k=final_top_k,
                fusion_method=fusion_method,
                rrf_k=rrf_k
            )
            all_results.append(results)
        
        return all_results
    
    def analyze_retrieval_overlap(
        self,
        query: str,
        top_k: int = 100
    ) -> Dict:
        """
        Analyze the overlap between sparse and dense retrieval.
        Useful for understanding retrieval behavior.
        
        Args:
            query: Search query
            top_k: Number of results to compare
            
        Returns:
            Dictionary with overlap statistics
        """
        sparse_results = self.retrieve_sparse(query, top_k=top_k)
        dense_results = self.retrieve_dense(query, top_k=top_k)
        
        sparse_ids = set(r.passage_id for r in sparse_results)
        dense_ids = set(r.passage_id for r in dense_results)
        
        overlap = sparse_ids & dense_ids
        sparse_only = sparse_ids - dense_ids
        dense_only = dense_ids - sparse_ids
        
        analysis = {
            "query": query,
            "top_k": top_k,
            "sparse_count": len(sparse_ids),
            "dense_count": len(dense_ids),
            "overlap_count": len(overlap),
            "overlap_percentage": len(overlap) / top_k * 100 if top_k > 0 else 0,
            "sparse_only_count": len(sparse_only),
            "dense_only_count": len(dense_only),
            "sparse_ids": list(sparse_ids)[:10],  # First 10 for inspection
            "dense_ids": list(dense_ids)[:10],
            "overlap_ids": list(overlap)[:10]
        }
        
        return analysis


# Example usage
if __name__ == "__main__":
    # Initialize sparse indexer
    sparse_indexer = ElasticsearchIndexer(
        index_name="wikipedia_bm25",
        host="localhost",
        port=9200
    )
    
    # Initialize dense indexer
    dense_indexer = FAISSIndexer(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )
    
    # Load dense index
    dense_indexer.load_index(
        index_path="indices/faiss/wikipedia.index",
        metadata_path="indices/faiss/wikipedia_metadata.pkl"
    )
    
    # Initialize hybrid retriever
    hybrid_retriever = HybridRetriever(
        sparse_indexer=sparse_indexer,
        dense_indexer=dense_indexer,
        sparse_weight=0.5,
        dense_weight=0.5
    )
    
    # Test hybrid search
    query = "What is machine learning?"
    
    print("=" * 80)
    print(f"Query: {query}")
    print("=" * 80)
    
    # Perform hybrid search with RRF
    results_rrf = hybrid_retriever.hybrid_search(
        query,
        top_k_sparse=100,
        top_k_dense=100,
        final_top_k=10,
        fusion_method="rrf",
        rrf_k=60
    )
    
    print("\n--- RRF Fusion Results ---")
    for i, result in enumerate(results_rrf, 1):
        print(f"\n{i}. Passage ID: {result.passage_id}")
        print(f"   Combined Score: {result.combined_score:.4f}")
        print(f"   Sparse Score: {result.sparse_score:.4f} (rank: {result.sparse_rank})")
        print(f"   Dense Score: {result.dense_score:.4f} (rank: {result.dense_rank})")
        print(f"   Title: {result.title}")
        print(f"   Text: {result.text[:150]}...")
    
    # Perform hybrid search with weighted fusion
    results_weighted = hybrid_retriever.hybrid_search(
        query,
        top_k_sparse=100,
        top_k_dense=100,
        final_top_k=10,
        fusion_method="weighted"
    )
    
    print("\n\n--- Weighted Fusion Results ---")
    for i, result in enumerate(results_weighted, 1):
        print(f"\n{i}. Passage ID: {result.passage_id}")
        print(f"   Combined Score: {result.combined_score:.4f}")
        print(f"   Sparse Score: {result.sparse_score:.4f}")
        print(f"   Dense Score: {result.dense_score:.4f}")
        print(f"   Title: {result.title}")
        print(f"   Text: {result.text[:150]}...")
    
    # Analyze retrieval overlap
    print("\n\n--- Retrieval Overlap Analysis ---")
    analysis = hybrid_retriever.analyze_retrieval_overlap(query, top_k=100)
    print(f"Sparse results: {analysis['sparse_count']}")
    print(f"Dense results: {analysis['dense_count']}")
    print(f"Overlap: {analysis['overlap_count']} ({analysis['overlap_percentage']:.1f}%)")
    print(f"Sparse only: {analysis['sparse_only_count']}")
    print(f"Dense only: {analysis['dense_only_count']}")
    """
    # Batch search
    print("\n\n--- Batch Hybrid Search ---")
    queries = [
        "What is artificial intelligence?",
        "How do neural networks work?",
        "Explain quantum computing"
    ]
    
    batch_results = hybrid_retriever.batch_hybrid_search(
        queries,
        final_top_k=5,
        fusion_method="rrf"
    )
    
    for query, results in zip(queries, batch_results):
        print(f"\nQuery: {query}")
        print(f"Top result: {results[0].title if results else 'No results'}")
        print(f"Score: {results[0].combined_score:.4f}" if results else "")
    """