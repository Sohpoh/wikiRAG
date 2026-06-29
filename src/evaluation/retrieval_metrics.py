import logging
from typing import List, Dict, Optional, Set, Tuple, Union
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalMetrics:
    """
    Evaluate retrieval performance using standard IR metrics.
    Supports Recall@k, Precision@k, MRR, MAP, and nDCG.
    """
    
    def __init__(self):
        """Initialize retrieval metrics calculator."""
        logger.info("RetrievalMetrics initialized")
    
    def recall_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        Calculate Recall@k.
        
        Recall@k = (# relevant items in top-k) / (total # relevant items)
        
        Args:
            retrieved: List of retrieved passage IDs (ordered by relevance)
            relevant: Set of relevant passage IDs (ground truth)
            k: Number of top results to consider
            
        Returns:
            Recall@k score (0-1)
        """
        if not relevant:
            return 0.0
        
        top_k = set(retrieved[:k])
        relevant_in_top_k = len(top_k & relevant)
        
        return relevant_in_top_k / len(relevant)
    
    def precision_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        Calculate Precision@k.
        
        Precision@k = (# relevant items in top-k) / k
        
        Args:
            retrieved: List of retrieved passage IDs
            relevant: Set of relevant passage IDs (ground truth)
            k: Number of top results to consider
            
        Returns:
            Precision@k score (0-1)
        """
        if k == 0:
            return 0.0
        
        top_k = set(retrieved[:k])
        relevant_in_top_k = len(top_k & relevant)
        
        return relevant_in_top_k / k
    
    def mean_reciprocal_rank(
        self,
        retrieved: List[str],
        relevant: Set[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        MRR = 1 / rank_of_first_relevant_item
        
        Args:
            retrieved: List of retrieved passage IDs
            relevant: Set of relevant passage IDs
            
        Returns:
            MRR score (0-1)
        """
        for rank, passage_id in enumerate(retrieved, 1):
            if passage_id in relevant:
                return 1.0 / rank
        
        return 0.0
    
    def average_precision(
        self,
        retrieved: List[str],
        relevant: Set[str]
    ) -> float:
        """
        Calculate Average Precision (AP).
        
        AP = (sum of P@k for each relevant item) / (total relevant items)
        
        Args:
            retrieved: List of retrieved passage IDs
            relevant: Set of relevant passage IDs
            
        Returns:
            Average Precision score (0-1)
        """
        if not relevant:
            return 0.0
        
        num_relevant = 0
        sum_precisions = 0.0
        
        for rank, passage_id in enumerate(retrieved, 1):
            if passage_id in relevant:
                num_relevant += 1
                precision_at_rank = num_relevant / rank
                sum_precisions += precision_at_rank
        
        return sum_precisions / len(relevant)
    
    def ndcg_at_k(
        self,
        retrieved: List[str],
        relevant: Dict[str, float],
        k: int
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (nDCG@k).
        
        DCG@k = sum(rel_i / log2(i+1)) for i in 1..k
        nDCG@k = DCG@k / IDCG@k
        
        Args:
            retrieved: List of retrieved passage IDs
            relevant: Dict mapping passage_id -> relevance score (0-1 or 0-2 or 0-3)
            k: Number of top results to consider
            
        Returns:
            nDCG@k score (0-1)
        """
        def dcg(scores: List[float], k: int) -> float:
            """Calculate DCG."""
            dcg_sum = 0.0
            for i, score in enumerate(scores[:k], 1):
                dcg_sum += score / np.log2(i + 1)
            return dcg_sum
        
        # Calculate DCG for retrieved items
        retrieved_scores = []
        for passage_id in retrieved[:k]:
            retrieved_scores.append(relevant.get(passage_id, 0.0))
        
        dcg_value = dcg(retrieved_scores, k)
        
        # Calculate IDCG (ideal DCG with perfect ranking)
        ideal_scores = sorted(relevant.values(), reverse=True)
        idcg_value = dcg(ideal_scores, k)
        
        # Return nDCG
        if idcg_value == 0:
            return 0.0
        
        return dcg_value / idcg_value
    
    def f1_score(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        Calculate F1 score at k.
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Args:
            retrieved: List of retrieved passage IDs
            relevant: Set of relevant passage IDs
            k: Number of top results to consider
            
        Returns:
            F1 score (0-1)
        """
        precision = self.precision_at_k(retrieved, relevant, k)
        recall = self.recall_at_k(retrieved, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def evaluate_single_query(
        self,
        retrieved: List[str],
        relevant: Union[Set[str], Dict[str, float]],
        k_values: List[int] = [5, 10, 20, 50, 100]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval for a single query across multiple metrics.
        
        Args:
            retrieved: List of retrieved passage IDs
            relevant: Set of relevant passage IDs or dict with relevance scores
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary of metric -> score
        """
        # Convert to set if needed for binary relevance
        if isinstance(relevant, dict):
            relevant_set = set(relevant.keys())
            has_scores = True
        else:
            relevant_set = relevant
            has_scores = False
        
        metrics = {}
        
        # Recall@k
        for k in k_values:
            metrics[f"recall@{k}"] = self.recall_at_k(retrieved, relevant_set, k)
        
        # Precision@k
        for k in k_values:
            metrics[f"precision@{k}"] = self.precision_at_k(retrieved, relevant_set, k)
        
        # F1@k
        for k in k_values:
            metrics[f"f1@{k}"] = self.f1_score(retrieved, relevant_set, k)
        
        # MRR (doesn't depend on k)
        metrics["mrr"] = self.mean_reciprocal_rank(retrieved, relevant_set)
        
        # Average Precision
        metrics["ap"] = self.average_precision(retrieved, relevant_set)
        
        # nDCG@k (if we have relevance scores)
        if has_scores:
            for k in k_values:
                metrics[f"ndcg@{k}"] = self.ndcg_at_k(retrieved, relevant, k)
        
        return metrics
    
    def evaluate_multiple_queries(
        self,
        results: List[Tuple[List[str], Union[Set[str], Dict[str, float]]]],
        k_values: List[int] = [5, 10, 20, 50, 100]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval across multiple queries (macro-averaged).
        
        Args:
            results: List of (retrieved, relevant) tuples
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary of metric -> average score
        """
        logger.info(f"Evaluating {len(results)} queries")
        
        all_metrics = defaultdict(list)
        
        for retrieved, relevant in results:
            query_metrics = self.evaluate_single_query(retrieved, relevant, k_values)
            
            for metric, value in query_metrics.items():
                all_metrics[metric].append(value)
        
        # Calculate averages
        avg_metrics = {}
        for metric, values in all_metrics.items():
            avg_metrics[metric] = np.mean(values)
        
        return avg_metrics
    
    def evaluate_with_document_level(
        self,
        retrieved_passages: List[str],
        relevant_passages: Set[str],
        passage_to_doc: Dict[str, str],
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        Evaluate at both passage and document level.
        Useful for understanding if relevant documents are retrieved even if
        exact passages don't match.
        
        Args:
            retrieved_passages: List of retrieved passage IDs
            relevant_passages: Set of relevant passage IDs
            passage_to_doc: Mapping from passage_id to doc_id
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary with passage-level and document-level metrics
        """
        metrics = {}
        
        # Passage-level metrics
        for k in k_values:
            metrics[f"passage_recall@{k}"] = self.recall_at_k(
                retrieved_passages, relevant_passages, k
            )
        
        # Convert to document IDs
        retrieved_docs = []
        for passage_id in retrieved_passages:
            doc_id = passage_to_doc.get(passage_id)
            if doc_id and doc_id not in retrieved_docs:
                retrieved_docs.append(doc_id)
        
        relevant_docs = set()
        for passage_id in relevant_passages:
            doc_id = passage_to_doc.get(passage_id)
            if doc_id:
                relevant_docs.add(doc_id)
        
        # Document-level metrics
        for k in k_values:
            metrics[f"doc_recall@{k}"] = self.recall_at_k(
                retrieved_docs, relevant_docs, k
            )
        
        return metrics
    
    def success_rate_at_k(
        self,
        results: List[Tuple[List[str], Set[str]]],
        k: int
    ) -> float:
        """
        Calculate success rate: percentage of queries with at least one
        relevant item in top-k.
        
        Args:
            results: List of (retrieved, relevant) tuples
            k: Number of top results to consider
            
        Returns:
            Success rate (0-1)
        """
        successes = 0
        
        for retrieved, relevant in results:
            top_k = set(retrieved[:k])
            if top_k & relevant:  # If intersection is non-empty
                successes += 1
        
        return successes / len(results) if results else 0.0


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("RETRIEVAL METRICS EXAMPLES")
    print("=" * 80)
    
    # Initialize metrics calculator
    metrics_calc = RetrievalMetrics()
    
    # Example 1: Single query evaluation (binary relevance)
    print("\n1. SINGLE QUERY EVALUATION (Binary Relevance)")
    print("-" * 80)
    
    # Simulated retrieval results
    retrieved = [
        "doc1_p1",  # relevant
        "doc2_p1",  # not relevant
        "doc1_p2",  # relevant
        "doc3_p1",  # not relevant
        "doc4_p1",  # relevant
        "doc5_p1",  # not relevant
        "doc1_p3",  # relevant
        "doc6_p1",  # not relevant
        "doc7_p1",  # not relevant
        "doc8_p1",  # not relevant
    ]
    
    # Ground truth relevant passages
    relevant = {"doc1_p1", "doc1_p2", "doc1_p3", "doc4_p1", "doc9_p1"}  # 5 relevant, 1 not retrieved
    
    print(f"Retrieved: {len(retrieved)} passages")
    print(f"Relevant: {len(relevant)} passages")
    print(f"Overlap: {len(set(retrieved) & relevant)} passages\n")
    
    # Evaluate at different k values
    query_metrics = metrics_calc.evaluate_single_query(
        retrieved=retrieved,
        relevant=relevant,
        k_values=[5, 10, 20]
    )
    
    print("Metrics:")
    for metric, value in sorted(query_metrics.items()):
        print(f"  {metric}: {value:.4f}")
    
    # Example 2: Single query with graded relevance (for nDCG)
    print("\n\n2. SINGLE QUERY WITH GRADED RELEVANCE")
    print("-" * 80)
    
    # Graded relevance: 0 (not relevant), 1 (somewhat), 2 (relevant), 3 (highly relevant)
    graded_relevant = {
        "doc1_p1": 3,  # highly relevant
        "doc1_p2": 2,  # relevant
        "doc1_p3": 2,  # relevant
        "doc4_p1": 1,  # somewhat relevant
        "doc9_p1": 3,  # highly relevant (not retrieved)
    }
    
    query_metrics_graded = metrics_calc.evaluate_single_query(
        retrieved=retrieved,
        relevant=graded_relevant,
        k_values=[5, 10]
    )
    
    print("Metrics with graded relevance:")
    for metric, value in sorted(query_metrics_graded.items()):
        print(f"  {metric}: {value:.4f}")
    
    # Example 3: Multiple queries evaluation
    print("\n\n3. MULTIPLE QUERIES EVALUATION")
    print("-" * 80)
    
    # Simulate results for 5 queries
    multiple_results = [
        # Query 1: Good retrieval (4/5 relevant in top 10)
        (
            ["r1", "n1", "r2", "n2", "r3", "n3", "r4", "n4", "n5", "n6"],
            {"r1", "r2", "r3", "r4", "r5"}
        ),
        # Query 2: Perfect retrieval
        (
            ["r1", "r2", "r3", "n1", "n2", "n3", "n4", "n5", "n6", "n7"],
            {"r1", "r2", "r3"}
        ),
        # Query 3: Poor retrieval (1/4 relevant in top 10)
        (
            ["n1", "n2", "r1", "n3", "n4", "n5", "n6", "n7", "n8", "n9"],
            {"r1", "r2", "r3", "r4"}
        ),
        # Query 4: Medium retrieval (2/3 relevant in top 10)
        (
            ["r1", "n1", "n2", "r2", "n3", "n4", "n5", "n6", "n7", "n8"],
            {"r1", "r2", "r3"}
        ),
        # Query 5: Good retrieval (3/3 relevant in top 10)
        (
            ["n1", "r1", "r2", "n2", "r3", "n3", "n4", "n5", "n6", "n7"],
            {"r1", "r2", "r3"}
        ),
    ]
    
    avg_metrics = metrics_calc.evaluate_multiple_queries(
        results=multiple_results,
        k_values=[5, 10, 20]
    )
    
    print(f"Averaged metrics over {len(multiple_results)} queries:\n")
    for metric, value in sorted(avg_metrics.items()):
        print(f"  {metric}: {value:.4f}")
    
    # Example 4: Document-level vs Passage-level
    print("\n\n4. DOCUMENT-LEVEL vs PASSAGE-LEVEL EVALUATION")
    print("-" * 80)
    
    retrieved_passages = [
        "doc1_p1", "doc2_p1", "doc1_p2", "doc3_p1", "doc4_p1",
        "doc5_p1", "doc1_p3", "doc6_p1", "doc2_p2", "doc7_p1"
    ]
    
    relevant_passages = {"doc1_p1", "doc1_p2", "doc1_p3", "doc4_p1", "doc8_p1"}
    
    # Mapping passages to documents
    passage_to_doc = {
        "doc1_p1": "doc1", "doc1_p2": "doc1", "doc1_p3": "doc1",
        "doc2_p1": "doc2", "doc2_p2": "doc2",
        "doc3_p1": "doc3",
        "doc4_p1": "doc4",
        "doc5_p1": "doc5",
        "doc6_p1": "doc6",
        "doc7_p1": "doc7",
        "doc8_p1": "doc8"
    }
    
    doc_metrics = metrics_calc.evaluate_with_document_level(
        retrieved_passages=retrieved_passages,
        relevant_passages=relevant_passages,
        passage_to_doc=passage_to_doc,
        k_values=[5, 10]
    )
    
    print("Passage-level vs Document-level:\n")
    for metric, value in sorted(doc_metrics.items()):
        print(f"  {metric}: {value:.4f}")
    
    # Example 5: Success rate
    print("\n\n5. SUCCESS RATE")
    print("-" * 80)
    
    success_rate_5 = metrics_calc.success_rate_at_k(multiple_results, k=5)
    success_rate_10 = metrics_calc.success_rate_at_k(multiple_results, k=10)
    
    print(f"Success@5: {success_rate_5:.4f} ({success_rate_5*100:.1f}% queries with ≥1 relevant in top-5)")
    print(f"Success@10: {success_rate_10:.4f} ({success_rate_10*100:.1f}% queries with ≥1 relevant in top-10)")
    
    # Example 6: Detailed breakdown for one query
    print("\n\n6. DETAILED BREAKDOWN FOR ONE QUERY")
    print("-" * 80)
    
    retrieved_detailed = ["r1", "n1", "r2", "n2", "n3", "r3", "n4", "n5", "r4", "n6"]
    relevant_detailed = {"r1", "r2", "r3", "r4", "r5"}
    
    print(f"Retrieved: {retrieved_detailed}")
    print(f"Relevant: {relevant_detailed}\n")
    
    k_values_detailed = [1, 3, 5, 10]
    
    print(f"{'k':<5} {'Recall':<10} {'Precision':<10} {'F1':<10}")
    print("-" * 40)
    
    for k in k_values_detailed:
        recall = metrics_calc.recall_at_k(retrieved_detailed, relevant_detailed, k)
        precision = metrics_calc.precision_at_k(retrieved_detailed, relevant_detailed, k)
        f1 = metrics_calc.f1_score(retrieved_detailed, relevant_detailed, k)
        
        print(f"{k:<5} {recall:<10.4f} {precision:<10.4f} {f1:<10.4f}")
    
    mrr = metrics_calc.mean_reciprocal_rank(retrieved_detailed, relevant_detailed)
    ap = metrics_calc.average_precision(retrieved_detailed, relevant_detailed)
    
    print(f"\nMRR: {mrr:.4f}")
    print(f"AP: {ap:.4f}")
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)