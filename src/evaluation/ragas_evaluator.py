import logging
from typing import List, Dict, Optional, Union
import numpy as np
from dataclasses import dataclass
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGASScore:
    """Represents RAGAS evaluation scores."""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall: float


class RAGASEvaluator:
    """
    Evaluate RAG systems using RAGAS metrics.
    
    RAGAS (Retrieval Augmented Generation Assessment) metrics:
    1. Faithfulness: Is the answer grounded in the retrieved context?
    2. Answer Relevancy: Is the answer relevant to the question?
    3. Context Precision: Are the retrieved contexts relevant?
    4. Context Recall: Does the context contain information to answer the question?
    
    Note: Full RAGAS implementation requires LLM calls for evaluation.
    This is a simplified version with rule-based approximations.
    For production, use the official RAGAS library with LLM-based evaluation.
    """
    
    def __init__(
        self,
        use_llm_evaluation: bool = False,
        llm_evaluator=None
    ):
        """
        Initialize RAGAS evaluator.
        
        Args:
            use_llm_evaluation: Whether to use LLM for evaluation (recommended)
            llm_evaluator: Optional LLM evaluator (e.g., OpenAI client)
        """
        self.use_llm_evaluation = use_llm_evaluation
        self.llm_evaluator = llm_evaluator
        
        if use_llm_evaluation and llm_evaluator is None:
            logger.warning("LLM evaluation enabled but no evaluator provided. "
                          "Using rule-based evaluation instead.")
            self.use_llm_evaluation = False
        
        logger.info(f"RAGASEvaluator initialized (use_llm={self.use_llm_evaluation})")
    
    def faithfulness(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Evaluate faithfulness: Is the answer grounded in the context?
        
        Simplified approach:
        - Check if key facts/claims in the answer appear in contexts
        - Higher score if answer statements are found in contexts
        
        Full RAGAS uses LLM to:
        1. Extract statements from answer
        2. Verify each statement against contexts
        3. Score = (verified statements) / (total statements)
        
        Args:
            answer: Generated answer
            contexts: List of context passages
            
        Returns:
            Faithfulness score (0-1)
        """
        if not answer or not contexts:
            return 0.0
        
        # Combine all contexts
        combined_context = " ".join(contexts).lower()
        answer_lower = answer.lower()
        
        # Split answer into sentences (simple approximation)
        sentences = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Check how many sentences have support in context
        supported_count = 0
        
        for sentence in sentences:
            # Extract key words (simplified - remove stop words)
            stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for'}
            words = [w for w in sentence.split() if w.lower() not in stop_words and len(w) > 3]
            
            # Check if majority of key words appear in context
            if words:
                found_words = sum(1 for word in words if word.lower() in combined_context)
                if found_words / len(words) > 0.6:  # 60% threshold
                    supported_count += 1
        
        faithfulness_score = supported_count / len(sentences)
        
        return faithfulness_score
    
    def answer_relevancy(
        self,
        question: str,
        answer: str
    ) -> float:
        """
        Evaluate answer relevancy: Is the answer relevant to the question?
        
        Simplified approach:
        - Check if answer contains key terms from question
        - Check if answer is a reasonable length (not too short/long)
        
        Full RAGAS uses LLM to:
        1. Generate potential questions from the answer
        2. Compare with original question
        3. Score based on similarity
        
        Args:
            question: Original question
            answer: Generated answer
            
        Returns:
            Answer relevancy score (0-1)
        """
        if not question or not answer:
            return 0.0
        
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Extract key terms from question
        stop_words = {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'was', 'were', 
                     'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of'}
        question_words = [w for w in question_lower.split() if w not in stop_words and len(w) > 3]
        
        if not question_words:
            return 0.5  # Neutral score if no key words
        
        # Check how many question terms appear in answer
        found_count = sum(1 for word in question_words if word in answer_lower)
        term_overlap_score = found_count / len(question_words)
        
        # Check answer length (penalize very short or very long answers)
        answer_length = len(answer.split())
        if answer_length < 5:
            length_score = 0.5
        elif answer_length > 200:
            length_score = 0.7
        else:
            length_score = 1.0
        
        # Check if answer is not just a copy of question
        question_in_answer = question_lower in answer_lower
        if question_in_answer:
            repetition_penalty = 0.8
        else:
            repetition_penalty = 1.0
        
        relevancy_score = (term_overlap_score * 0.6 + length_score * 0.4) * repetition_penalty
        
        return min(relevancy_score, 1.0)
    
    def context_precision(
        self,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        question: Optional[str] = None
    ) -> float:
        """
        Evaluate context precision: Are the retrieved contexts relevant?
        
        Simplified approach:
        - Check if contexts contain terms from question
        - Higher ranked contexts should be more relevant
        
        Full RAGAS uses LLM to:
        1. Check if each context is useful for answering the question
        2. Weight by rank position
        3. Score = sum(relevance_i * precision_at_i)
        
        Args:
            contexts: List of retrieved contexts (ordered by relevance)
            ground_truth: Optional ground truth answer
            question: Optional question text
            
        Returns:
            Context precision score (0-1)
        """
        if not contexts:
            return 0.0
        
        # If we have ground truth, check if contexts contain answer information
        if ground_truth:
            gt_lower = ground_truth.lower()
            precision_scores = []
            
            for i, context in enumerate(contexts):
                context_lower = context.lower()
                
                # Extract key words from ground truth
                gt_words = [w for w in gt_lower.split() if len(w) > 3]
                
                if gt_words:
                    # Check overlap with context
                    found = sum(1 for word in gt_words if word in context_lower)
                    relevance = found / len(gt_words)
                else:
                    relevance = 0.0
                
                # Weight by position (earlier contexts should be more relevant)
                position_weight = 1.0 / (i + 1)
                precision_scores.append(relevance * position_weight)
            
            # Normalize
            if sum(precision_scores) > 0:
                avg_precision = sum(precision_scores) / len(contexts)
            else:
                avg_precision = 0.0
            
            return avg_precision
        
        # If we have question, check if contexts are relevant to question
        elif question:
            question_lower = question.lower()
            question_words = [w for w in question_lower.split() if len(w) > 3]
            
            if not question_words:
                return 0.5
            
            precision_scores = []
            
            for i, context in enumerate(contexts):
                context_lower = context.lower()
                found = sum(1 for word in question_words if word in context_lower)
                relevance = found / len(question_words)
                
                position_weight = 1.0 / (i + 1)
                precision_scores.append(relevance * position_weight)
            
            avg_precision = sum(precision_scores) / len(contexts)
            return avg_precision
        
        else:
            # No ground truth or question, can't evaluate
            return 0.5
    
    def context_recall(
        self,
        contexts: List[str],
        ground_truth: str
    ) -> float:
        """
        Evaluate context recall: Does context contain information to answer?
        
        Simplified approach:
        - Check if ground truth information appears in contexts
        
        Full RAGAS uses LLM to:
        1. Extract key information from ground truth
        2. Check if each piece appears in contexts
        3. Score = (found information) / (total information)
        
        Args:
            contexts: List of retrieved contexts
            ground_truth: Ground truth answer or reference
            
        Returns:
            Context recall score (0-1)
        """
        if not contexts or not ground_truth:
            return 0.0
        
        combined_context = " ".join(contexts).lower()
        gt_lower = ground_truth.lower()
        
        # Extract key terms from ground truth
        gt_words = [w for w in gt_lower.split() if len(w) > 3]
        
        if not gt_words:
            return 0.0
        
        # Check how many ground truth terms appear in contexts
        found_count = sum(1 for word in gt_words if word in combined_context)
        
        recall_score = found_count / len(gt_words)
        
        return recall_score
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> RAGASScore:
        """
        Evaluate a single RAG output.
        
        Args:
            question: Input question
            answer: Generated answer
            contexts: Retrieved contexts used for generation
            ground_truth: Optional ground truth answer
            
        Returns:
            RAGASScore object with all metrics
        """
        # Calculate individual metrics
        faithfulness_score = self.faithfulness(answer, contexts)
        answer_relevancy_score = self.answer_relevancy(question, answer)
        context_precision_score = self.context_precision(contexts, ground_truth, question)
        
        # Context recall requires ground truth
        if ground_truth:
            context_recall_score = self.context_recall(contexts, ground_truth)
        else:
            context_recall_score = 0.0
            logger.warning("Context recall requires ground_truth. Skipping.")
        
        # Calculate overall score (average of available metrics)
        available_scores = [faithfulness_score, answer_relevancy_score, 
                           context_precision_score]
        if ground_truth:
            available_scores.append(context_recall_score)
        
        overall_score = np.mean(available_scores)
        
        return RAGASScore(
            faithfulness=faithfulness_score,
            answer_relevancy=answer_relevancy_score,
            context_precision=context_precision_score,
            context_recall=context_recall_score,
            overall=overall_score
        )
    
    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate multiple RAG outputs.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts_list: List of context lists
            ground_truths: Optional list of ground truth answers
            
        Returns:
            Dictionary with averaged metrics
        """
        if len(questions) != len(answers) != len(contexts_list):
            raise ValueError("Lengths of questions, answers, and contexts must match")
        
        if ground_truths and len(ground_truths) != len(questions):
            raise ValueError("Length of ground_truths must match questions")
        
        logger.info(f"Evaluating {len(questions)} examples with RAGAS")
        
        all_scores = {
            "faithfulness": [],
            "answer_relevancy": [],
            "context_precision": [],
            "context_recall": [],
            "overall": []
        }
        
        for i in range(len(questions)):
            gt = ground_truths[i] if ground_truths else None
            
            score = self.evaluate_single(
                question=questions[i],
                answer=answers[i],
                contexts=contexts_list[i],
                ground_truth=gt
            )
            
            all_scores["faithfulness"].append(score.faithfulness)
            all_scores["answer_relevancy"].append(score.answer_relevancy)
            all_scores["context_precision"].append(score.context_precision)
            all_scores["context_recall"].append(score.context_recall)
            all_scores["overall"].append(score.overall)
        
        # Calculate averages
        avg_scores = {
            metric: np.mean(scores) for metric, scores in all_scores.items()
        }
        
        avg_scores["count"] = len(questions)
        
        logger.info(f"RAGAS Evaluation: Faithfulness={avg_scores['faithfulness']:.4f}, "
                   f"Answer Relevancy={avg_scores['answer_relevancy']:.4f}, "
                   f"Context Precision={avg_scores['context_precision']:.4f}, "
                   f"Context Recall={avg_scores['context_recall']:.4f}, "
                   f"Overall={avg_scores['overall']:.4f}")
        
        return avg_scores
    
    def evaluate_with_details(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> tuple[Dict[str, float], List[Dict]]:
        """
        Evaluate with per-example details.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts_list: List of context lists
            ground_truths: Optional list of ground truth answers
            
        Returns:
            Tuple of (average_scores, per_example_scores)
        """
        per_example = []
        
        for i in range(len(questions)):
            gt = ground_truths[i] if ground_truths else None
            
            score = self.evaluate_single(
                question=questions[i],
                answer=answers[i],
                contexts=contexts_list[i],
                ground_truth=gt
            )
            
            per_example.append({
                "question": questions[i],
                "answer": answers[i],
                "num_contexts": len(contexts_list[i]),
                "faithfulness": score.faithfulness,
                "answer_relevancy": score.answer_relevancy,
                "context_precision": score.context_precision,
                "context_recall": score.context_recall,
                "overall": score.overall
            })
        
        # Calculate averages
        avg_scores = self.evaluate_batch(questions, answers, contexts_list, ground_truths)
        
        return avg_scores, per_example


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("RAGAS EVALUATOR EXAMPLES")
    print("=" * 80)
    
    print("\nNote: This is a simplified RAGAS implementation using rule-based metrics.")
    print("For production use, consider the official RAGAS library with LLM-based evaluation.")
    print("Install: pip install ragas")
    
    # Initialize evaluator
    ragas_eval = RAGASEvaluator(use_llm_evaluation=False)
    
    # Example 1: Single evaluation
    print("\n\n1. SINGLE EXAMPLE EVALUATION")
    print("-" * 80)
    
    question = "What is machine learning?"
    answer = "Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve their performance without being explicitly programmed."
    contexts = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
        "Machine learning systems can learn from data and improve their performance over time.",
        "Deep learning is a type of machine learning based on neural networks."
    ]
    ground_truth = "Machine learning is a subset of AI that allows systems to learn from data."
    
    score = ragas_eval.evaluate_single(question, answer, contexts, ground_truth)
    
    print(f"Question: {question}")
    print(f"\nAnswer: {answer}")
    print(f"\nNumber of contexts: {len(contexts)}")
    print(f"\nScores:")
    print(f"  Faithfulness: {score.faithfulness:.4f}")
    print(f"  Answer Relevancy: {score.answer_relevancy:.4f}")
    print(f"  Context Precision: {score.context_precision:.4f}")
    print(f"  Context Recall: {score.context_recall:.4f}")
    print(f"  Overall: {score.overall:.4f}")
    
    # Example 2: Faithfulness comparison
    print("\n\n2. FAITHFULNESS COMPARISON")
    print("-" * 80)
    
    contexts_base = [
        "Python is a high-level programming language.",
        "It was created by Guido van Rossum and released in 1991."
    ]
    
    # Faithful answer
    answer_faithful = "Python is a high-level programming language created by Guido van Rossum in 1991."
    faith_score_1 = ragas_eval.faithfulness(answer_faithful, contexts_base)
    
    # Unfaithful answer (hallucination)
    answer_unfaithful = "Python is a low-level programming language created by James Gosling in 1995."
    faith_score_2 = ragas_eval.faithfulness(answer_unfaithful, contexts_base)
    
    print("Contexts:")
    for i, ctx in enumerate(contexts_base, 1):
        print(f"  [{i}] {ctx}")
    
    print(f"\nFaithful answer: {answer_faithful}")
    print(f"Faithfulness: {faith_score_1:.4f}")
    
    print(f"\nUnfaithful answer: {answer_unfaithful}")
    print(f"Faithfulness: {faith_score_2:.4f}")
    
    # Example 3: Answer relevancy comparison
    print("\n\n3. ANSWER RELEVANCY COMPARISON")
    print("-" * 80)
    
    question_rel = "Who invented the telephone?"
    
    # Relevant answer
    answer_relevant = "Alexander Graham Bell invented the telephone in 1876."
    rel_score_1 = ragas_eval.answer_relevancy(question_rel, answer_relevant)
    
    # Irrelevant answer
    answer_irrelevant = "The telephone is a device used for communication over long distances."
    rel_score_2 = ragas_eval.answer_relevancy(question_rel, answer_irrelevant)
    
    print(f"Question: {question_rel}")
    print(f"\nRelevant answer: {answer_relevant}")
    print(f"Answer Relevancy: {rel_score_1:.4f}")
    
    print(f"\nIrrelevant answer: {answer_irrelevant}")
    print(f"Answer Relevancy: {rel_score_2:.4f}")
    
    # Example 4: Batch evaluation
    print("\n\n4. BATCH EVALUATION")
    print("-" * 80)
    
    questions_batch = [
        "What is machine learning?",
        "Who coined the term machine learning?",
        "When was the term coined?"
    ]
    
    answers_batch = [
        "Machine learning is a type of artificial intelligence that learns from data.",
        "Arthur Samuel coined the term machine learning.",
        "The term was coined in 1959."
    ]
    
    contexts_batch = [
        [
            "Machine learning is a subset of AI.",
            "It involves algorithms that learn from data."
        ],
        [
            "Arthur Samuel, an American computer scientist, coined the term.",
            "He did pioneering work in AI and machine learning."
        ],
        [
            "The term machine learning was first used in 1959.",
            "Arthur Samuel used this term in his research paper."
        ]
    ]
    
    ground_truths_batch = [
        "Machine learning is AI that learns from data",
        "Arthur Samuel",
        "1959"
    ]
    
    batch_scores = ragas_eval.evaluate_batch(
        questions_batch,
        answers_batch,
        contexts_batch,
        ground_truths_batch
    )
    
    print(f"Evaluated {batch_scores['count']} examples\n")
    print("Average Scores:")
    for metric, score in batch_scores.items():
        if metric != "count":
            print(f"  {metric}: {score:.4f}")
    
    # Example 5: Detailed evaluation
    print("\n\n5. DETAILED PER-EXAMPLE EVALUATION")
    print("-" * 80)
    
    avg_scores, per_example = ragas_eval.evaluate_with_details(
        questions_batch,
        answers_batch,
        contexts_batch,
        ground_truths_batch
    )
    
    for i, ex in enumerate(per_example, 1):
        print(f"\nExample {i}:")
        print(f"  Question: {ex['question']}")
        print(f"  Answer: {ex['answer'][:80]}...")
        print(f"  Faithfulness: {ex['faithfulness']:.4f}")
        print(f"  Answer Relevancy: {ex['answer_relevancy']:.4f}")
        print(f"  Context Precision: {ex['context_precision']:.4f}")
        print(f"  Context Recall: {ex['context_recall']:.4f}")
        print(f"  Overall: {ex['overall']:.4f}")
    
    # Example 6: Context quality comparison
    print("\n\n6. CONTEXT QUALITY COMPARISON")
    print("-" * 80)
    
    question_ctx = "What is deep learning?"
    gt_ctx = "Deep learning is a type of machine learning using neural networks"
    
    # High quality contexts (relevant and ordered)
    contexts_good = [
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
        "These neural networks can learn hierarchical representations of data.",
        "Python is a programming language often used for machine learning."  # Less relevant
    ]
    
    # Low quality contexts (less relevant)
    contexts_poor = [
        "Machine learning is a broad field of artificial intelligence.",
        "There are many applications of AI in various industries.",
        "Deep learning uses neural networks."  # Relevant but at the end
    ]
    
    precision_good = ragas_eval.context_precision(contexts_good, gt_ctx, question_ctx)
    precision_poor = ragas_eval.context_precision(contexts_poor, gt_ctx, question_ctx)
    
    recall_good = ragas_eval.context_recall(contexts_good, gt_ctx)
    recall_poor = ragas_eval.context_recall(contexts_poor, gt_ctx)
    
    print(f"Question: {question_ctx}")
    print(f"Ground Truth: {gt_ctx}\n")
    
    print("Good Contexts (relevant and well-ordered):")
    print(f"  Context Precision: {precision_good:.4f}")
    print(f"  Context Recall: {recall_good:.4f}")
    
    print("\nPoor Contexts (less relevant and poorly ordered):")
    print(f"  Context Precision: {precision_poor:.4f}")
    print(f"  Context Recall: {recall_poor:.4f}")
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nFor production use with LLM-based evaluation:")
    print("  pip install ragas")
    print("  from ragas import evaluate")
    print("  from ragas.metrics import faithfulness, answer_relevancy, context_precision")