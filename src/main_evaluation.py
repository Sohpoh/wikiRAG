"""
Main evaluation script for RAG system.

Evaluates the RAG pipeline on SQuAD and HotpotQA datasets using:
- Retrieval metrics (Recall@k, MRR, nDCG)
- QA metrics (EM, F1)
- RAGAS metrics (Faithfulness, Relevancy, etc.)

Usage:
    python main_evaluation.py --dataset squad --split dev --max_samples 100
    python main_evaluation.py --dataset hotpotqa --split dev --output results/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import time
from datetime import datetime
import sys

from tqdm import tqdm
import pandas as pd
sys.path.append(str(Path(__file__).parent.parent))
# Import pipeline components
from src.indexing.sparse_indexer import ElasticsearchIndexer
from src.indexing.dense_indexer import FAISSIndexer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.generation.prompt_builder import PromptBuilder
from src.generation.answer_generator import AnswerGenerator
from src.pipeline2 import RAGPipeline

# Import evaluation modules
from src.evaluation.retrieval_metrics import RetrievalMetrics
from src.evaluation.qa_metrics import SQuADEvaluator, HotpotQAEvaluator
from src.evaluation.ragas_evaluator import RAGASEvaluator
from src.preprocessing.dataset_loader import SQuADLoader, HotpotQALoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Main evaluation class for RAG pipeline."""
    
    def __init__(
        self,
        pipeline: RAGPipeline,
        output_dir: str = "results",
        save_predictions: bool = True,
        save_details: bool = True
    ):
        """
        Initialize RAG evaluator.
        
        Args:
            pipeline: RAG pipeline to evaluate
            output_dir: Directory to save results
            save_predictions: Save prediction files
            save_details: Save detailed per-example results
        """
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_predictions = save_predictions
        self.save_details = save_details
        
        # Initialize evaluators
        self.retrieval_metrics = RetrievalMetrics()
        self.squad_evaluator = SQuADEvaluator()
        self.hotpotqa_evaluator = HotpotQAEvaluator()
        self.ragas_evaluator = RAGASEvaluator()
        
        logger.info(f"RAGEvaluator initialized. Output dir: {output_dir}")
    
    def evaluate_retrieval(
        self,
        dataset: List[Dict],
        k_values: List[int] = [5, 10, 20, 50, 100]
    ) -> Dict:
        """
        Evaluate retrieval performance.
        
        Args:
            dataset: List of examples with ground truth passages
            k_values: k values for Recall@k
            
        Returns:
            Dictionary with retrieval metrics
        """
        logger.info("Evaluating retrieval performance...")
        
        retrieval_results = []
        
        for example in tqdm(dataset, desc="Retrieval evaluation"):
            query = example["question"]
            relevant_passage_ids = set(example.get("relevant_passages", []))
            
            if not relevant_passage_ids:
                logger.warning(f"No relevant passages for query: {query[:50]}...")
                continue
            
            # Get hybrid retrieval results
            hybrid_results = self.pipeline.hybrid_retriever.hybrid_search(
                query=query,
                top_k_sparse=self.pipeline.top_k_sparse,
                top_k_dense=self.pipeline.top_k_dense,
                final_top_k=100,  # Get more for evaluation
                fusion_method=self.pipeline.fusion_method
            )
            
            # Extract passage IDs
            retrieved_ids = [r.passage_id for r in hybrid_results]
            
            retrieval_results.append((retrieved_ids, relevant_passage_ids))
        
        # Calculate metrics
        metrics = self.retrieval_metrics.evaluate_multiple_queries(
            results=retrieval_results,
            k_values=k_values
        )
        
        return metrics
    
    def evaluate_qa(
        self,
        dataset: List[Dict],
        dataset_type: str = "squad"
    ) -> Dict:
        """
        Evaluate QA performance.
        
        Args:
            dataset: List of examples
            dataset_type: "squad" or "hotpotqa"
            
        Returns:
            Dictionary with QA metrics
        """
        logger.info(f"Evaluating QA performance on {dataset_type}...")
        
        predictions = {}
        all_results = []
        
        for example in tqdm(dataset, desc="QA evaluation"):
            qid = example["id"]
            query = example["question"]
            
            # Run pipeline
            try:
                result = self.pipeline.answer_question(query, return_passages=True)
                predictions[qid] = result.answer
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing question {qid}: {e}")
                predictions[qid] = "[Error]"
        
        # Evaluate
        if dataset_type == "squad":
            metrics = self.squad_evaluator.evaluate(predictions, dataset)
        elif dataset_type == "hotpotqa":
            metrics = self.hotpotqa_evaluator.evaluate_answer(predictions, dataset)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Add pipeline statistics
        pipeline_stats = self.pipeline.get_pipeline_stats(all_results)
        metrics.update(pipeline_stats)
        
        # Save predictions
        if self.save_predictions:
            pred_file = self.output_dir / f"predictions_{dataset_type}.json"
            with open(pred_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            logger.info(f"Saved predictions to {pred_file}")
        
        return metrics
    
    def evaluate_ragas(
        self,
        dataset: List[Dict],
        max_samples: Optional[int] = None
    ) -> Dict:
        """
        Evaluate using RAGAS metrics.
        
        Args:
            dataset: List of examples
            max_samples: Optional limit on number of samples
            
        Returns:
            Dictionary with RAGAS metrics
        """
        logger.info("Evaluating with RAGAS metrics...")
        
        if max_samples:
            dataset = dataset[:max_samples]
        
        questions = []
        answers = []
        contexts_list = []
        ground_truths = []
        
        for example in tqdm(dataset, desc="RAGAS evaluation"):
            query = example["question"]
            
            # Run pipeline
            try:
                result = self.pipeline.answer_question(query, return_passages=True)
                
                questions.append(query)
                answers.append(result.answer)
                contexts_list.append([p['text'] for p in result.top_passages])
                
                # Get ground truth
                if "answer" in example:
                    ground_truths.append(example["answer"])
                elif "answers" in example and example["answers"]:
                    ground_truths.append(example["answers"][0])
                else:
                    ground_truths.append("")
                
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                continue
        
        # Evaluate
        metrics = self.ragas_evaluator.evaluate_batch(
            questions=questions,
            answers=answers,
            contexts_list=contexts_list,
            ground_truths=ground_truths
        )
        
        return metrics
    
    def evaluate_end_to_end(
        self,
        dataset: List[Dict],
        dataset_type: str = "squad",
        evaluate_retrieval: bool = True,
        evaluate_ragas: bool = True,
        ragas_max_samples: int = 100
    ) -> Dict:
        """
        Run complete evaluation pipeline.
        
        Args:
            dataset: List of examples
            dataset_type: "squad" or "hotpotqa"
            evaluate_retrieval: Whether to evaluate retrieval
            evaluate_ragas: Whether to evaluate with RAGAS
            ragas_max_samples: Max samples for RAGAS evaluation
            
        Returns:
            Dictionary with all metrics
        """
        logger.info("=" * 80)
        logger.info(f"STARTING END-TO-END EVALUATION: {dataset_type.upper()}")
        logger.info(f"Dataset size: {len(dataset)}")
        logger.info("=" * 80)
        
        all_metrics = {
            "dataset": dataset_type,
            "num_examples": len(dataset),
            "timestamp": datetime.now().isoformat()
        }
        
        # 1. QA Evaluation (always run)
        logger.info("\n[1/3] QA Evaluation")
        logger.info("-" * 80)
        qa_metrics = self.evaluate_qa(dataset, dataset_type)
        all_metrics["qa_metrics"] = qa_metrics
        
        logger.info("\nQA Results:")
        logger.info(f"  Exact Match: {qa_metrics['em']:.4f} ({qa_metrics['em']*100:.2f}%)")
        logger.info(f"  F1 Score: {qa_metrics['f1']:.4f}")
        
        # 2. Retrieval Evaluation (if requested and data available)
        if evaluate_retrieval:
            logger.info("\n[2/3] Retrieval Evaluation")
            logger.info("-" * 80)
            
            # Check if dataset has retrieval ground truth
            has_retrieval_gt = any("relevant_passages" in ex for ex in dataset)
            
            if has_retrieval_gt:
                retrieval_metrics = self.evaluate_retrieval(dataset)
                all_metrics["retrieval_metrics"] = retrieval_metrics
                
                logger.info("\nRetrieval Results:")
                logger.info(f"  Recall@5: {retrieval_metrics.get('recall@5', 0):.4f}")
                logger.info(f"  Recall@10: {retrieval_metrics.get('recall@10', 0):.4f}")
                logger.info(f"  Recall@20: {retrieval_metrics.get('recall@20', 0):.4f}")
                logger.info(f"  MRR: {retrieval_metrics.get('mrr', 0):.4f}")
            else:
                logger.warning("No retrieval ground truth available. Skipping retrieval evaluation.")
        
        # 3. RAGAS Evaluation (if requested)
        if evaluate_ragas:
            logger.info("\n[3/3] RAGAS Evaluation")
            logger.info("-" * 80)
            logger.info(f"Evaluating on {ragas_max_samples} samples (for speed)...")
            
            ragas_metrics = self.evaluate_ragas(dataset, max_samples=ragas_max_samples)
            all_metrics["ragas_metrics"] = ragas_metrics
            
            logger.info("\nRAGAS Results:")
            logger.info(f"  Faithfulness: {ragas_metrics['faithfulness']:.4f}")
            logger.info(f"  Answer Relevancy: {ragas_metrics['answer_relevancy']:.4f}")
            logger.info(f"  Context Precision: {ragas_metrics['context_precision']:.4f}")
            logger.info(f"  Context Recall: {ragas_metrics['context_recall']:.4f}")
            logger.info(f"  Overall: {ragas_metrics['overall']:.4f}")
        
        # Save complete results
        results_file = self.output_dir / f"evaluation_results_{dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        logger.info(f"\n✓ Complete results saved to: {results_file}")
        
        # Save summary
        self._save_summary(all_metrics, dataset_type)
        
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 80)
        
        return all_metrics
    
    def _save_summary(self, metrics: Dict, dataset_type: str):
        """Save evaluation summary to CSV."""
        summary = {
            "dataset": dataset_type,
            "timestamp": metrics["timestamp"],
            "num_examples": metrics["num_examples"],
        }
        
        # QA metrics
        if "qa_metrics" in metrics:
            summary["em"] = metrics["qa_metrics"]["em"]
            summary["f1"] = metrics["qa_metrics"]["f1"]
            summary["avg_time"] = metrics["qa_metrics"].get("avg_total_time", 0)
        
        # Retrieval metrics
        if "retrieval_metrics" in metrics:
            summary["recall@10"] = metrics["retrieval_metrics"].get("recall@10", 0)
            summary["mrr"] = metrics["retrieval_metrics"].get("mrr", 0)
        
        # RAGAS metrics
        if "ragas_metrics" in metrics:
            summary["faithfulness"] = metrics["ragas_metrics"]["faithfulness"]
            summary["answer_relevancy"] = metrics["ragas_metrics"]["answer_relevancy"]
        
        # Save to CSV
        summary_file = self.output_dir / "evaluation_summary.csv"
        df = pd.DataFrame([summary])
        
        # Append if file exists
        if summary_file.exists():
            df_existing = pd.read_csv(summary_file)
            df = pd.concat([df_existing, df], ignore_index=True)
        
        df.to_csv(summary_file, index=False)
        logger.info(f"✓ Summary saved to: {summary_file}")


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    
    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["squad", "hotpotqa"],
        required=True,
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Dataset split (train/dev/test)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/datasets",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--skip_retrieval_eval",
        action="store_true",
        help="Skip retrieval evaluation"
    )
    parser.add_argument(
        "--skip_ragas",
        action="store_true",
        help="Skip RAGAS evaluation"
    )
    parser.add_argument(
        "--ragas_samples",
        type=int,
        default=100,
        help="Number of samples for RAGAS evaluation"
    )
    
    # Pipeline arguments
    parser.add_argument(
        "--top_k_rerank",
        type=int,
        default=10,
        help="Number of passages after reranking"
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="default",
        choices=["default", "strict", "cot", "factoid", "hotpotqa"],
        help="Prompt template to use"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--no_save_predictions",
        action="store_true",
        help="Don't save prediction files"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("RAG PIPELINE EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Max samples: {args.max_samples or 'all'}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)
    
    # 1. Load dataset
    logger.info("\n[SETUP] Loading dataset...")
    data_path = Path(args.data_path)
    
    if args.dataset == "squad":
        loader = SQuADLoader(data_path / "squad")
        dataset = loader.load_split(args.split)
    elif args.dataset == "hotpotqa":
        loader = HotpotQALoader(data_path / "hotpotqa")
        dataset = loader.load_split(args.split)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    if args.max_samples:
        dataset = dataset[:args.max_samples]
    
    logger.info(f"✓ Loaded {len(dataset)} examples")
    
    # 2. Initialize pipeline
    logger.info("\n[SETUP] Initializing pipeline components...")
    
    # Check indices
    logger.info("  Checking indices...")
    
    sparse_indexer = ElasticsearchIndexer(
        index_name="wikipedia_bm25",
        host="localhost",
        port=9200
    )
    stats = sparse_indexer.get_index_stats()
    if "error" in stats:
        logger.error(f"Elasticsearch error: {stats['error']}")
        logger.error("Please ensure Elasticsearch is running with indexed data.")
        sys.exit(1)
    logger.info(f"    ✓ Elasticsearch: {stats['document_count']} documents")
    
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
        logger.info(f"    ✓ FAISS: {dense_stats['num_vectors']} vectors")
    except Exception as e:
        logger.error(f"FAISS error: {e}")
        logger.error("Please ensure FAISS index is built.")
        sys.exit(1)
    
    # Initialize components
    logger.info("  Initializing components...")
    
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu"
    )
    logger.info("    ✓ Cross-encoder loaded")
    
    prompt_builder = PromptBuilder(
        max_context_length=4000,
        passage_format="numbered",
        include_metadata=True
    )
    logger.info("    ✓ Prompt builder ready")
    
    # Check for OpenAI API key
    import os
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set!")
        logger.error("Please set it to run evaluation with answer generation:")
        logger.error("  export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    answer_generator = AnswerGenerator(
        model_name="gpt-3.5-turbo",
        temperature=0.0,  # Deterministic for evaluation
        max_tokens=150 if args.dataset == "squad" else 300
    )
    logger.info("    ✓ Answer generator ready")
    
    # Create pipeline
    prompt_template = args.prompt_template
    if args.dataset == "hotpotqa" and prompt_template == "default":
        prompt_template = "hotpotqa"  # Use HotpotQA template by default
    
    pipeline = RAGPipeline(
        sparse_indexer=sparse_indexer,
        dense_indexer=dense_indexer,
        cross_encoder_reranker=reranker,
        prompt_builder=prompt_builder,
        answer_generator=answer_generator,
        use_query_reformulation=False,
        top_k_sparse=100,
        top_k_dense=100,
        top_k_hybrid=50,
        top_k_rerank=args.top_k_rerank,
        fusion_method="rrf",
        prompt_template=prompt_template
    )
    
    logger.info("  ✓ Pipeline initialized")
    
    # 3. Run evaluation
    evaluator = RAGEvaluator(
        pipeline=pipeline,
        output_dir=args.output_dir,
        save_predictions=not args.no_save_predictions,
        save_details=True
    )
    
    results = evaluator.evaluate_end_to_end(
        dataset=dataset,
        dataset_type=args.dataset,
        evaluate_retrieval=not args.skip_retrieval_eval,
        evaluate_ragas=not args.skip_ragas,
        ragas_max_samples=args.ragas_samples
    )
    
    logger.info("\n✓ Evaluation complete! Check results directory for details.")
    
    return results


if __name__ == "__main__":
    main()