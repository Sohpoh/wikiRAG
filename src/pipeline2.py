import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.indexing.sparse_indexer2 import ElasticsearchIndexer
from src.indexing.dense_indexer import FAISSIndexer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_reformulation import QueryReformulator
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.generation.prompt_builder import PromptBuilder
from src.generation.answer_generator import AnswerGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Complete RAG pipeline result."""
    query: str
    answer: str
    
    # Retrieval info
    num_sparse_results: int
    num_dense_results: int
    num_hybrid_results: int
    num_reranked_results: int
    
    # Generation info
    num_passages_used: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    # Timing
    retrieval_time: float
    reranking_time: float
    generation_time: float
    total_time: float
    
    # Optional detailed info
    top_passages: Optional[List[Dict]] = None
    reformulated_queries: Optional[List[str]] = None


class RAGPipeline:
    """
    End-to-end RAG pipeline for question answering.
    
    Pipeline stages:
    1. Query reformulation (optional)
    2. Hybrid retrieval (BM25 + Dense)
    3. Cross-encoder reranking
    4. Prompt building
    5. Answer generation
    """
    
    def __init__(
        self,
        # Indexers
        sparse_indexer: ElasticsearchIndexer,
        dense_indexer: FAISSIndexer,
        
        # Pipeline components
        query_reformulator: Optional[QueryReformulator] = None,
        cross_encoder_reranker: Optional[CrossEncoderReranker] = None,
        
        # Generation
        prompt_builder: Optional[PromptBuilder] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        
        # Pipeline config
        use_query_reformulation: bool = False,
        top_k_sparse: int = 100,
        top_k_dense: int = 100,
        top_k_hybrid: int = 50,
        top_k_rerank: int = 10,
        fusion_method: str = "rrf",
        prompt_template: str = "default"
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            sparse_indexer: Elasticsearch BM25 indexer
            dense_indexer: FAISS dense indexer
            query_reformulator: Optional query reformulator
            cross_encoder_reranker: Optional cross-encoder reranker
            prompt_builder: Optional prompt builder
            answer_generator: Optional answer generator
            use_query_reformulation: Whether to use query reformulation
            top_k_sparse: Number of results from sparse retrieval
            top_k_dense: Number of results from dense retrieval
            top_k_hybrid: Number of results after hybrid fusion
            top_k_rerank: Number of results after reranking
            fusion_method: Hybrid fusion method ("rrf" or "weighted")
            prompt_template: Prompt template to use
        """
        # Indexers
        self.sparse_indexer = sparse_indexer
        self.dense_indexer = dense_indexer
        
        # Initialize hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            sparse_indexer=sparse_indexer,
            dense_indexer=dense_indexer,
            sparse_weight=0.5,
            dense_weight=0.5
        )
        
        # Optional components
        self.query_reformulator = query_reformulator
        self.cross_encoder_reranker = cross_encoder_reranker
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.answer_generator = answer_generator
        
        # Config
        self.use_query_reformulation = use_query_reformulation
        self.top_k_sparse = top_k_sparse
        self.top_k_dense = top_k_dense
        self.top_k_hybrid = top_k_hybrid
        self.top_k_rerank = top_k_rerank
        self.fusion_method = fusion_method
        self.prompt_template = prompt_template
        
        logger.info("RAG Pipeline initialized")
        logger.info(f"  Hybrid retrieval: top_k={top_k_hybrid}")
        logger.info(f"  Reranking: top_k={top_k_rerank}")
        logger.info(f"  Query reformulation: {use_query_reformulation}")
    
    def answer_question(
        self,
        query: str,
        return_passages: bool = False,
        custom_prompt_template: Optional[str] = None
    ) -> RAGResult:
        """
        Answer a question using the full RAG pipeline.
        
        Args:
            query: Question to answer
            return_passages: Whether to return top passages in result
            custom_prompt_template: Optional custom prompt template
            
        Returns:
            RAGResult object with answer and metadata
        """
        start_time = time.time()
        
        logger.info(f"Processing query: '{query}'")
        
        # Stage 1: Query reformulation (optional)
        reformulated_queries = None
        if self.use_query_reformulation and self.query_reformulator:
            logger.info("Stage 1: Query reformulation")
            reformulation_result = self.query_reformulator.expand_query(
                query,
                strategy="synonyms"
            )
            reformulated_queries = reformulation_result.reformulated_queries
            logger.info(f"  Generated {len(reformulated_queries)} query variations")
        
        # Stage 2: Hybrid retrieval
        logger.info("Stage 2: Hybrid retrieval")
        retrieval_start = time.time()
        
        hybrid_results = self.hybrid_retriever.hybrid_search(
            query=query,
            top_k_sparse=self.top_k_sparse,
            top_k_dense=self.top_k_dense,
            final_top_k=self.top_k_hybrid,
            fusion_method=self.fusion_method
        )
        
        retrieval_time = time.time() - retrieval_start
        logger.info(f"  Retrieved {len(hybrid_results)} passages ({retrieval_time:.2f}s)")
        
        # Stage 3: Reranking (optional)
        reranking_time = 0.0
        if self.cross_encoder_reranker:
            logger.info("Stage 3: Cross-encoder reranking")
            reranking_start = time.time()
            
            reranked_results = self.cross_encoder_reranker.rerank(
                query=query,
                passages=hybrid_results,
                top_k=self.top_k_rerank,
                batch_size=32
            )
            
            reranking_time = time.time() - reranking_start
            logger.info(f"  Reranked to {len(reranked_results)} passages ({reranking_time:.2f}s)")
            
            final_passages = reranked_results
        else:
            logger.info("Stage 3: Skipping reranking (not configured)")
            final_passages = hybrid_results[:self.top_k_rerank]
        
        # Stage 4: Prompt building
        logger.info("Stage 4: Building prompt")
        
        template = custom_prompt_template or self.prompt_template
        rag_prompt = self.prompt_builder.build_rag_prompt(
            query=query,
            passages=final_passages,
            prompt_template=template
        )
        
        logger.info(f"  Prompt: {rag_prompt.num_passages} passages, "
                   f"~{rag_prompt.total_tokens} tokens")
        
        # Stage 5: Answer generation
        if self.answer_generator:
            logger.info("Stage 5: Generating answer")
            generation_start = time.time()
            
            generated = self.answer_generator.generate_from_rag_prompt(rag_prompt)
            
            generation_time = time.time() - generation_start
            logger.info(f"  Generated answer ({generation_time:.2f}s, "
                       f"{generated.total_tokens} tokens)")
            
            answer = generated.answer
            prompt_tokens = generated.prompt_tokens
            completion_tokens = generated.completion_tokens
            total_tokens = generated.total_tokens
        else:
            logger.info("Stage 5: Skipping generation (not configured)")
            answer = "[Answer generation not configured]"
            generation_time = 0.0
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
        
        total_time = time.time() - start_time
        
        # Prepare result
        result = RAGResult(
            query=query,
            answer=answer,
            num_sparse_results=self.top_k_sparse,
            num_dense_results=self.top_k_dense,
            num_hybrid_results=len(hybrid_results),
            num_reranked_results=len(final_passages),
            num_passages_used=rag_prompt.num_passages,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            retrieval_time=retrieval_time,
            reranking_time=reranking_time,
            generation_time=generation_time,
            total_time=total_time,
            reformulated_queries=reformulated_queries
        )
        
        # Add top passages if requested
        if return_passages:
            result.top_passages = [
                {
                    'passage_id': getattr(p, 'passage_id', ''),
                    'title': getattr(p, 'title', ''),
                    'text': getattr(p, 'text', ''),
                    'score': getattr(p, 'rerank_score', getattr(p, 'combined_score', 0.0))
                }
                for p in final_passages
            ]
        
        logger.info(f"Pipeline complete: {total_time:.2f}s total")
        
        return result
    
    def batch_answer(
        self,
        queries: List[str],
        show_progress: bool = True
    ) -> List[RAGResult]:
        """
        Answer multiple questions.
        
        Args:
            queries: List of questions
            show_progress: Show progress bar
            
        Returns:
            List of RAGResult objects
        """
        logger.info(f"Batch processing {len(queries)} queries")
        
        results = []
        
        iterator = queries
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(queries, desc="Answering questions")
        
        for query in iterator:
            try:
                result = self.answer_question(query)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process query '{query}': {e}")
                # Create error result
                error_result = RAGResult(
                    query=query,
                    answer="[Error during processing]",
                    num_sparse_results=0,
                    num_dense_results=0,
                    num_hybrid_results=0,
                    num_reranked_results=0,
                    num_passages_used=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    retrieval_time=0.0,
                    reranking_time=0.0,
                    generation_time=0.0,
                    total_time=0.0
                )
                results.append(error_result)
        
        return results
    
    def get_pipeline_stats(self, results: List[RAGResult]) -> Dict[str, Any]:
        """
        Get statistics from multiple pipeline runs.
        
        Args:
            results: List of RAGResult objects
            
        Returns:
            Dictionary with statistics
        """
        if not results:
            return {}
        
        stats = {
            "total_queries": len(results),
            "avg_retrieval_time": sum(r.retrieval_time for r in results) / len(results),
            "avg_reranking_time": sum(r.reranking_time for r in results) / len(results),
            "avg_generation_time": sum(r.generation_time for r in results) / len(results),
            "avg_total_time": sum(r.total_time for r in results) / len(results),
            "avg_passages_used": sum(r.num_passages_used for r in results) / len(results),
            "avg_tokens": sum(r.total_tokens for r in results) / len(results),
            "total_tokens": sum(r.total_tokens for r in results)
        }
        
        # Estimate cost if using OpenAI
        if self.answer_generator:
            total_cost = sum(
                self.answer_generator.estimate_cost(r.prompt_tokens, r.completion_tokens)
                for r in results
            )
            stats["estimated_cost_usd"] = total_cost
        
        return stats


# Example usage
if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path
    
    print("=" * 80)
    print("RAG PIPELINE EXAMPLE")
    print("=" * 80)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nWarning: OPENAI_API_KEY not found.")
        print("Set it to enable answer generation:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("\nContinuing with retrieval only...\n")
    
    # Initialize components
    print("\n1. Initializing components...")
    
    # Sparse indexer
    sparse_indexer = ElasticsearchIndexer(
        index_name="wikipedia_bm25",
        host="localhost",
        port=9200
    )
    
    # Check if index exists
    stats = sparse_indexer.get_index_stats()
    if "error" in stats:
        print(f"Error: {stats['error']}")
        print("Please ensure Elasticsearch is running and indices are built.")
        sys.exit(1)
    
    print(f"   ✓ Elasticsearch: {stats['document_count']} documents")
    
    # Dense indexer
    dense_indexer = FAISSIndexer(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )
    
    try:
        dense_indexer.load_index(
            index_path="indices/faiss/wikipedia.index",
            metadata_path="indices/faiss/wikipedia_metadata.pkl"
        )
        dense_stats = dense_indexer.get_index_stats()
        print(f"   ✓ FAISS: {dense_stats['num_vectors']} vectors")
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        sys.exit(1)
    
    # Cross-encoder reranker
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu"
    )
    print(f"   ✓ Cross-encoder loaded")
    
    # Prompt builder
    prompt_builder = PromptBuilder(
        max_context_length=4000,
        passage_format="numbered",
        include_metadata=True
    )
    print(f"   ✓ Prompt builder ready")
    
    # Answer generator (if API key available)
    if os.getenv("OPENAI_API_KEY"):
        answer_generator = AnswerGenerator(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=256
        )
        print(f"   ✓ Answer generator ready")
    else:
        answer_generator = None
        print(f"   ⚠ Answer generator skipped (no API key)")
    
    # Initialize pipeline
    print("\n2. Initializing pipeline...")
    
    pipeline = RAGPipeline(
        sparse_indexer=sparse_indexer,
        dense_indexer=dense_indexer,
        cross_encoder_reranker=reranker,
        prompt_builder=prompt_builder,
        answer_generator=answer_generator,
        use_query_reformulation=False,  # Set to True if you have OpenAI key
        top_k_sparse=100,
        top_k_dense=100,
        top_k_hybrid=50,
        top_k_rerank=10,
        fusion_method="rrf",
        prompt_template="default"
    )
    
    print("   ✓ Pipeline ready")
    
    # Interactive loop
    print("\n" + "=" * 80)
    print("INTERACTIVE RAG PIPELINE")
    print("=" * 80)
    print("Type 'exit', 'quit', or press Ctrl+C to stop.")
    
    try:
        while True:
            print("\n" + "-" * 50)
            try:
                query = input("Enter your question: ").strip()
            except EOFError:
                break
            
            if not query:
                continue
                
            if query.lower() in ('exit', 'quit'):
                break
            
            print(f"\nProcessing query: {query}...\n")
            
            try:
                result = pipeline.answer_question(query, return_passages=True)
                
                print(f"Answer:\n{result.answer}\n")
                
                print("Pipeline Statistics:")
                print(f"  Retrieval: {result.retrieval_time:.2f}s")
                print(f"  Reranking: {result.reranking_time:.2f}s")
                print(f"  Generation: {result.generation_time:.2f}s")
                print(f"  Total: {result.total_time:.2f}s")
                
                print(f"\nRetrieval Info:")
                print(f"  Hybrid results: {result.num_hybrid_results}")
                print(f"  Reranked results: {result.num_reranked_results}")
                print(f"  Passages used: {result.num_passages_used}")
                
                if result.total_tokens > 0:
                    print(f"\nToken Usage:")
                    print(f"  Prompt: {result.prompt_tokens}")
                    print(f"  Completion: {result.completion_tokens}")
                    print(f"  Total: {result.total_tokens}")
                    
                    if answer_generator:
                        cost = answer_generator.estimate_cost(result.prompt_tokens, result.completion_tokens)
                        print(f"  Estimated cost: ${cost:.6f}")
                
                if result.top_passages:
                    print(f"\nTop 3 Passages:")
                    for i, passage in enumerate(result.top_passages[:3], 1):
                        print(f"\n  {i}. {passage['title']}")
                        print(f"     Score: {passage['score']:.4f}")
                        print(f"     Text: {passage['text'][:150]}...")
                        
            except Exception as e:
                print(f"Error processing query: {e}")
                import traceback
                traceback.print_exc()
                
    except KeyboardInterrupt:
        print("\n\nStopping RAG pipeline. Goodbye!")