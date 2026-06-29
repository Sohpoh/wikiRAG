import logging
import time
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from indexing.sparse_indexer2 import ElasticsearchIndexer
from indexing.dense_indexer import FAISSIndexer
from retrieval.hybrid_retriever import HybridRetriever, HybridSearchResult
from reranking.cross_encoder_reranker import CrossEncoderReranker, RerankedResult
from generation.prompt_builder import PromptBuilder, RAGPrompt
from generation.answer_generator import AnswerGenerator, GeneratedAnswer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Result of a RAG pipeline run."""
    query: str
    answer: str
    generated_answer: GeneratedAnswer
    retrieved_passages: List[HybridSearchResult]
    reranked_passages: List[RerankedResult]
    rag_prompt: RAGPrompt
    execution_time: float
    metadata: Dict[str, Any]


class RAGPipeline:
    """
    End-to-end RAG pipeline orchestrating retrieval, reranking, and generation.
    """
    
    def __init__(
        self,
        sparse_index_name: str = "wikipedia_bm25",
        dense_index_path: str = "indices/faiss/wikipedia.index",
        dense_metadata_path: str = "indices/faiss/wikipedia_metadata.pkl",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        openai_model: str = "gpt-3.5-turbo",
        device: Optional[str] = None,
        es_host: str = "localhost",
        es_port: int = 9200
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            sparse_index_name: Elasticsearch index name
            dense_index_path: Path to FAISS index file
            dense_metadata_path: Path to FAISS metadata file
            embedding_model: SentenceTransformer model name for dense retrieval
            cross_encoder_model: CrossEncoder model name for reranking
            openai_model: OpenAI model name for generation
            device: Device for models ('cpu', 'cuda', 'mps', or None for auto)
            es_host: Elasticsearch host
            es_port: Elasticsearch port
        """
        self.device = device
        
        logger.info("Initializing RAG Pipeline components...")
        
        # 1. Initialize Indexers
        self.sparse_indexer = ElasticsearchIndexer(
            index_name=sparse_index_name,
            host=es_host,
            port=es_port
        )
        
        self.dense_indexer = FAISSIndexer(
            model_name=embedding_model,
            device=device
        )
        
        # Load dense index if paths exist
        if Path(dense_index_path).exists() and Path(dense_metadata_path).exists():
            logger.info(f"Loading dense index from {dense_index_path}")
            self.dense_indexer.load_index(dense_index_path, dense_metadata_path)
        else:
            logger.warning(f"Dense index not found at {dense_index_path}. Dense retrieval will fail.")
        
        # 2. Initialize Retriever
        self.retriever = HybridRetriever(
            sparse_indexer=self.sparse_indexer,
            dense_indexer=self.dense_indexer,
            sparse_weight=0.5,
            dense_weight=0.5
        )
        
        # 3. Initialize Reranker
        self.reranker = CrossEncoderReranker(
            model_name=cross_encoder_model,
            device=device
        )
        
        # 4. Initialize Generator Components
        self.prompt_builder = PromptBuilder(
            max_context_length=4000,
            passage_format="numbered"
        )
        
        self.answer_generator = AnswerGenerator(
            model_name=openai_model,
            temperature=0.1
        )
        
        logger.info("RAG Pipeline initialized successfully")
    
    def run(
        self,
        query: str,
        top_k_retrieval: int = 50,
        top_k_rerank: int = 10,
        prompt_template: str = "default",
        return_supporting_facts: bool = True
    ) -> RAGResult:
        """
        Run the full RAG pipeline for a single query.
        
        Args:
            query: User query
            top_k_retrieval: Number of passages to retrieve
            top_k_rerank: Number of passages to keep after reranking
            prompt_template: Template for prompt builder
            return_supporting_facts: Whether to ask model for citations
            
        Returns:
            RAGResult object
        """
        start_time = time.time()
        logger.info(f"Processing query: '{query}'")
        
        # 1. Retrieval
        retrieval_start = time.time()
        retrieved_results = self.retriever.hybrid_search(
            query=query,
            final_top_k=top_k_retrieval,
            fusion_method="rrf"
        )
        retrieval_time = time.time() - retrieval_start
        logger.info(f"Retrieved {len(retrieved_results)} passages in {retrieval_time:.2f}s")
        
        # 2. Reranking
        rerank_start = time.time()
        reranked_results = self.reranker.rerank(
            query=query,
            passages=retrieved_results,
            top_k=top_k_rerank
        )
        rerank_time = time.time() - rerank_start
        logger.info(f"Reranked to {len(reranked_results)} passages in {rerank_time:.2f}s")
        
        # 3. Prompt Building
        if return_supporting_facts:
            rag_prompt = self.prompt_builder.build_with_supporting_facts(
                query=query,
                passages=reranked_results
            )
        else:
            rag_prompt = self.prompt_builder.build_rag_prompt(
                query=query,
                passages=reranked_results,
                prompt_template=prompt_template
            )
        
        # 4. Generation
        gen_start = time.time()
        if return_supporting_facts:
            generated_answer = self.answer_generator.generate_with_citations(rag_prompt)
        else:
            generated_answer = self.answer_generator.generate_from_rag_prompt(rag_prompt)
        gen_time = time.time() - gen_start
        logger.info(f"Generated answer in {gen_time:.2f}s")
        
        total_time = time.time() - start_time
        
        return RAGResult(
            query=query,
            answer=generated_answer.answer,
            generated_answer=generated_answer,
            retrieved_passages=retrieved_results,
            reranked_passages=reranked_results,
            rag_prompt=rag_prompt,
            execution_time=total_time,
            metadata={
                "retrieval_time": retrieval_time,
                "rerank_time": rerank_time,
                "generation_time": gen_time,
                "total_tokens": generated_answer.total_tokens
            }
        )
    
    def batch_run(
        self,
        queries: List[str],
        show_progress: bool = True
    ) -> List[RAGResult]:
        """
        Run pipeline for multiple queries.
        
        Args:
            queries: List of queries
            show_progress: Show progress bar
            
        Returns:
            List of RAGResult objects
        """
        results = []
        iterator = queries
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(queries, desc="Running RAG Pipeline")
            
        for query in iterator:
            try:
                result = self.run(query)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                # Append None or error result? For now, skip or handle gracefully
                continue
                
        return results


if __name__ == "__main__":
    # Example usage
    import os
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Generation step will fail.")
    
    pipeline = RAGPipeline(
        sparse_index_name="wikipedia_bm25",
        dense_index_path="indices/faiss/wikipedia.index",
        dense_metadata_path="indices/faiss/wikipedia_metadata.pkl",
        device=None # Auto-detect
    )
    
    print(f"\nRAG Pipeline Initialized. Ready for queries.")
    print("Type 'exit', 'quit', or press Ctrl+C to stop.")
    
    try:
        while True:
            print("\n" + "=" * 50)
            try:
                query = input("Enter your question: ").strip()
            except EOFError:
                break
                
            if not query:
                continue
                
            if query.lower() in ('exit', 'quit'):
                break
                
            print("-" * 50)
            
            try:
                result = pipeline.run(query)
                
                print(f"\nAnswer:\n{result.answer}\n")
                print("-" * 50)
                print(f"Retrieved: {len(result.retrieved_passages)}")
                print(f"Reranked: {len(result.reranked_passages)}")
                print(f"Time: {result.execution_time:.2f}s")
                
                if result.generated_answer.supporting_passages:
                    print(f"Citations: {result.generated_answer.supporting_passages}")
                    
            except Exception as e:
                print(f"Pipeline run failed: {e}")
                
    except KeyboardInterrupt:
        print("\n\nStopping RAG pipeline. Goodbye!")
