import logging
import re
import string
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QAMetrics:
    """
    Evaluate question answering performance using standard QA metrics.
    Supports Exact Match (EM), F1, and metrics for SQuAD and HotpotQA.
    """
    
    def __init__(self):
        """Initialize QA metrics calculator."""
        logger.info("QAMetrics initialized")
    
    @staticmethod
    def normalize_answer(text: str) -> str:
        """
        Normalize answer text for comparison.
        
        Normalization steps:
        1. Lowercase
        2. Remove punctuation
        3. Remove articles (a, an, the)
        4. Remove extra whitespace
        
        Args:
            text: Answer text
            
        Returns:
            Normalized text
        """
        def remove_articles(text: str) -> str:
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text: str) -> str:
            return ' '.join(text.split())
        
        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text: str) -> str:
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(text))))
    
    def exact_match(
        self,
        prediction: str,
        ground_truth: str
    ) -> float:
        """
        Calculate Exact Match (EM) score.
        
        EM = 1 if normalized prediction equals normalized ground truth, else 0
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            EM score (0 or 1)
        """
        return float(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))
    
    def exact_match_any(
        self,
        prediction: str,
        ground_truths: List[str]
    ) -> float:
        """
        Calculate EM with multiple ground truth answers.
        Returns 1 if prediction matches any ground truth.
        
        Args:
            prediction: Predicted answer
            ground_truths: List of acceptable ground truth answers
            
        Returns:
            EM score (0 or 1)
        """
        return float(any(self.exact_match(prediction, gt) for gt in ground_truths))
    
    def f1_score(
        self,
        prediction: str,
        ground_truth: str
    ) -> float:
        """
        Calculate token-level F1 score.
        
        F1 = 2 * (precision * recall) / (precision + recall)
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            F1 score (0-1)
        """
        pred_tokens = self.normalize_answer(prediction).split()
        truth_tokens = self.normalize_answer(ground_truth).split()
        
        # If either is empty
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return float(pred_tokens == truth_tokens)
        
        # Calculate token overlap
        common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
        num_common = sum(common_tokens.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(truth_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1
    
    def f1_score_max(
        self,
        prediction: str,
        ground_truths: List[str]
    ) -> float:
        """
        Calculate maximum F1 score over multiple ground truths.
        
        Args:
            prediction: Predicted answer
            ground_truths: List of acceptable ground truth answers
            
        Returns:
            Maximum F1 score (0-1)
        """
        return max(self.f1_score(prediction, gt) for gt in ground_truths)
    
    def evaluate_single(
        self,
        prediction: str,
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate a single prediction against ground truths.
        
        Args:
            prediction: Predicted answer
            ground_truths: List of acceptable ground truth answers
            
        Returns:
            Dictionary with EM and F1 scores
        """
        em = self.exact_match_any(prediction, ground_truths)
        f1 = self.f1_score_max(prediction, ground_truths)
        
        return {
            "em": em,
            "f1": f1
        }
    
    def evaluate_batch(
        self,
        predictions: List[str],
        ground_truths_list: List[List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate multiple predictions (macro-averaged).
        
        Args:
            predictions: List of predicted answers
            ground_truths_list: List of ground truth answer lists
            
        Returns:
            Dictionary with average EM and F1 scores
        """
        if len(predictions) != len(ground_truths_list):
            raise ValueError("Number of predictions and ground truths must match")
        
        logger.info(f"Evaluating {len(predictions)} predictions")
        
        em_scores = []
        f1_scores = []
        
        for pred, gts in zip(predictions, ground_truths_list):
            scores = self.evaluate_single(pred, gts)
            em_scores.append(scores["em"])
            f1_scores.append(scores["f1"])
        
        return {
            "em": np.mean(em_scores),
            "f1": np.mean(f1_scores),
            "count": len(predictions)
        }


class SQuADEvaluator:
    """
    Evaluate on SQuAD-style datasets.
    SQuAD format: each example has a question and list of acceptable answers.
    """
    
    def __init__(self):
        """Initialize SQuAD evaluator."""
        self.qa_metrics = QAMetrics()
        logger.info("SQuADEvaluator initialized")
    
    def evaluate(
        self,
        predictions: Dict[str, str],
        dataset: List[Dict]
    ) -> Dict[str, float]:
        """
        Evaluate predictions on SQuAD-style dataset.
        
        Args:
            predictions: Dict mapping question_id -> predicted_answer
            dataset: List of examples with format:
                {
                    "id": question_id,
                    "question": question_text,
                    "answers": [answer1, answer2, ...]  # Multiple acceptable answers
                }
            
        Returns:
            Dictionary with EM and F1 scores
        """
        pred_list = []
        gt_list = []
        
        for example in dataset:
            qid = example["id"]
            if qid not in predictions:
                logger.warning(f"No prediction for question ID: {qid}")
                continue
            
            pred_list.append(predictions[qid])
            gt_list.append(example["answers"])
        
        results = self.qa_metrics.evaluate_batch(pred_list, gt_list)
        
        logger.info(f"SQuAD Evaluation: EM={results['em']:.4f}, F1={results['f1']:.4f}")
        
        return results
    
    def evaluate_with_details(
        self,
        predictions: Dict[str, str],
        dataset: List[Dict]
    ) -> Tuple[Dict[str, float], List[Dict]]:
        """
        Evaluate with per-example details.
        
        Args:
            predictions: Dict mapping question_id -> predicted_answer
            dataset: List of examples
            
        Returns:
            Tuple of (overall_scores, per_example_scores)
        """
        pred_list = []
        gt_list = []
        per_example = []
        
        for example in dataset:
            qid = example["id"]
            if qid not in predictions:
                continue
            
            prediction = predictions[qid]
            ground_truths = example["answers"]
            
            scores = self.qa_metrics.evaluate_single(prediction, ground_truths)
            
            per_example.append({
                "id": qid,
                "question": example["question"],
                "prediction": prediction,
                "ground_truths": ground_truths,
                "em": scores["em"],
                "f1": scores["f1"]
            })
            
            pred_list.append(prediction)
            gt_list.append(ground_truths)
        
        overall = self.qa_metrics.evaluate_batch(pred_list, gt_list)
        
        return overall, per_example


class HotpotQAEvaluator:
    """
    Evaluate on HotpotQA dataset.
    HotpotQA includes both answer evaluation and supporting facts evaluation.
    """
    
    def __init__(self):
        """Initialize HotpotQA evaluator."""
        self.qa_metrics = QAMetrics()
        logger.info("HotpotQAEvaluator initialized")
    
    def evaluate_answer(
        self,
        predictions: Dict[str, str],
        dataset: List[Dict]
    ) -> Dict[str, float]:
        """
        Evaluate answer prediction only (same as SQuAD).
        
        Args:
            predictions: Dict mapping question_id -> predicted_answer
            dataset: List of examples with format:
                {
                    "id": question_id,
                    "question": question_text,
                    "answer": ground_truth_answer,
                    "supporting_facts": [[title1, sent_idx1], [title2, sent_idx2], ...]
                }
            
        Returns:
            Dictionary with EM and F1 scores
        """
        pred_list = []
        gt_list = []
        
        for example in dataset:
            qid = example["id"]
            if qid not in predictions:
                logger.warning(f"No prediction for question ID: {qid}")
                continue
            
            pred_list.append(predictions[qid])
            # HotpotQA has single answer, but we wrap in list for consistency
            gt_list.append([example["answer"]])
        
        results = self.qa_metrics.evaluate_batch(pred_list, gt_list)
        
        logger.info(f"HotpotQA Answer Evaluation: EM={results['em']:.4f}, F1={results['f1']:.4f}")
        
        return results
    
    def evaluate_supporting_facts(
        self,
        predictions: Dict[str, List[List]],
        dataset: List[Dict]
    ) -> Dict[str, float]:
        """
        Evaluate supporting facts prediction.
        
        Supporting facts metrics:
        - Precision: # correct supporting facts / # predicted supporting facts
        - Recall: # correct supporting facts / # true supporting facts
        - F1: Harmonic mean of precision and recall
        
        Args:
            predictions: Dict mapping question_id -> predicted_supporting_facts
                Format: [[title1, sent_idx1], [title2, sent_idx2], ...]
            dataset: List of examples with supporting_facts field
            
        Returns:
            Dictionary with precision, recall, and F1 for supporting facts
        """
        precisions = []
        recalls = []
        f1_scores = []
        
        for example in dataset:
            qid = example["id"]
            if qid not in predictions:
                continue
            
            pred_facts = set(tuple(fact) for fact in predictions[qid])
            true_facts = set(tuple(fact) for fact in example["supporting_facts"])
            
            if len(pred_facts) == 0:
                precision = 0.0
            else:
                correct = len(pred_facts & true_facts)
                precision = correct / len(pred_facts)
            
            if len(true_facts) == 0:
                recall = 0.0
            else:
                correct = len(pred_facts & true_facts)
                recall = correct / len(true_facts)
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        results = {
            "sp_precision": np.mean(precisions),
            "sp_recall": np.mean(recalls),
            "sp_f1": np.mean(f1_scores),
            "count": len(precisions)
        }
        
        logger.info(f"HotpotQA Supporting Facts: P={results['sp_precision']:.4f}, "
                   f"R={results['sp_recall']:.4f}, F1={results['sp_f1']:.4f}")
        
        return results
    
    def evaluate_joint(
        self,
        answer_predictions: Dict[str, str],
        sp_predictions: Dict[str, List[List]],
        dataset: List[Dict]
    ) -> Dict[str, float]:
        """
        Evaluate both answer and supporting facts (joint score).
        
        Joint EM: Both answer EM=1 and supporting facts F1=1
        Joint F1: Average of answer F1 and supporting facts F1
        
        Args:
            answer_predictions: Dict mapping question_id -> predicted_answer
            sp_predictions: Dict mapping question_id -> predicted_supporting_facts
            dataset: List of examples
            
        Returns:
            Dictionary with all metrics including joint scores
        """
        # Evaluate answers
        answer_results = self.evaluate_answer(answer_predictions, dataset)
        
        # Evaluate supporting facts
        sp_results = self.evaluate_supporting_facts(sp_predictions, dataset)
        
        # Calculate joint metrics
        joint_em_count = 0
        joint_f1_scores = []
        total_count = 0
        
        for example in dataset:
            qid = example["id"]
            if qid not in answer_predictions or qid not in sp_predictions:
                continue
            
            # Answer scores
            answer_scores = self.qa_metrics.evaluate_single(
                answer_predictions[qid],
                [example["answer"]]
            )
            
            # Supporting facts scores
            pred_facts = set(tuple(fact) for fact in sp_predictions[qid])
            true_facts = set(tuple(fact) for fact in example["supporting_facts"])
            
            if len(pred_facts) == 0 or len(true_facts) == 0:
                sp_f1 = 0.0
            else:
                correct = len(pred_facts & true_facts)
                sp_precision = correct / len(pred_facts)
                sp_recall = correct / len(true_facts)
                
                if sp_precision + sp_recall == 0:
                    sp_f1 = 0.0
                else:
                    sp_f1 = 2 * (sp_precision * sp_recall) / (sp_precision + sp_recall)
            
            # Joint EM: both answer and supporting facts must be perfect
            if answer_scores["em"] == 1.0 and sp_f1 == 1.0:
                joint_em_count += 1
            
            # Joint F1: average of answer F1 and supporting facts F1
            joint_f1 = (answer_scores["f1"] + sp_f1) / 2
            joint_f1_scores.append(joint_f1)
            
            total_count += 1
        
        joint_results = {
            **answer_results,
            **sp_results,
            "joint_em": joint_em_count / total_count if total_count > 0 else 0.0,
            "joint_f1": np.mean(joint_f1_scores) if joint_f1_scores else 0.0
        }
        
        logger.info(f"HotpotQA Joint Evaluation: Joint EM={joint_results['joint_em']:.4f}, "
                   f"Joint F1={joint_results['joint_f1']:.4f}")
        
        return joint_results


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("QA METRICS EXAMPLES")
    print("=" * 80)
    
    # Initialize metrics
    qa_metrics = QAMetrics()
    
    # Example 1: Basic EM and F1
    print("\n1. BASIC EXACT MATCH AND F1")
    print("-" * 80)
    
    prediction = "The capital of France is Paris"
    ground_truth = "Paris"
    
    em = qa_metrics.exact_match(prediction, ground_truth)
    f1 = qa_metrics.f1_score(prediction, ground_truth)
    
    print(f"Prediction: {prediction}")
    print(f"Ground Truth: {ground_truth}")
    print(f"EM: {em:.4f}")
    print(f"F1: {f1:.4f}")
    
    # Example 2: Multiple ground truths
    print("\n\n2. MULTIPLE GROUND TRUTHS")
    print("-" * 80)
    
    prediction = "1959"
    ground_truths = ["1959", "the year 1959", "in 1959"]
    
    em = qa_metrics.exact_match_any(prediction, ground_truths)
    f1 = qa_metrics.f1_score_max(prediction, ground_truths)
    
    print(f"Prediction: {prediction}")
    print(f"Ground Truths: {ground_truths}")
    print(f"EM: {em:.4f}")
    print(f"F1: {f1:.4f}")
    
    # Example 3: Normalization effects
    print("\n\n3. NORMALIZATION EFFECTS")
    print("-" * 80)
    
    test_cases = [
        ("Paris", "paris"),  # Case
        ("The answer is Paris", "Paris"),  # Articles
        ("Paris!", "Paris"),  # Punctuation
        ("Paris  ", "Paris"),  # Whitespace
        ("The Paris!", "paris"),  # All combined
    ]
    
    print(f"{'Prediction':<25} {'Ground Truth':<20} {'EM':<5} {'F1':<5}")
    print("-" * 60)
    
    for pred, gt in test_cases:
        em = qa_metrics.exact_match(pred, gt)
        f1 = qa_metrics.f1_score(pred, gt)
        print(f"{pred:<25} {gt:<20} {em:<5.0f} {f1:<5.2f}")
    
    # Example 4: SQuAD evaluation
    print("\n\n4. SQUAD EVALUATION")
    print("-" * 80)
    
    squad_evaluator = SQuADEvaluator()
    
    # Mock SQuAD dataset
    squad_dataset = [
        {
            "id": "q1",
            "question": "What is machine learning?",
            "answers": ["a subset of AI", "subset of artificial intelligence"]
        },
        {
            "id": "q2",
            "question": "Who coined the term machine learning?",
            "answers": ["Arthur Samuel", "Samuel"]
        },
        {
            "id": "q3",
            "question": "When was the term coined?",
            "answers": ["1959", "in 1959"]
        },
        {
            "id": "q4",
            "question": "What are the types of ML?",
            "answers": ["supervised, unsupervised, and reinforcement learning"]
        }
    ]
    
    # Mock predictions
    squad_predictions = {
        "q1": "a type of artificial intelligence",  # Partial match
        "q2": "Arthur Samuel",  # Exact match
        "q3": "1959",  # Exact match
        "q4": "supervised and unsupervised learning"  # Partial match
    }
    
    squad_results = squad_evaluator.evaluate(squad_predictions, squad_dataset)
    
    print(f"Overall SQuAD Results:")
    print(f"  Exact Match: {squad_results['em']:.4f} ({squad_results['em']*100:.1f}%)")
    print(f"  F1 Score: {squad_results['f1']:.4f}")
    print(f"  Count: {squad_results['count']} examples")
    
    # Detailed results
    print("\nPer-example breakdown:")
    overall, per_example = squad_evaluator.evaluate_with_details(squad_predictions, squad_dataset)
    
    for ex in per_example:
        print(f"\n  Q: {ex['question']}")
        print(f"  Prediction: {ex['prediction']}")
        print(f"  Ground Truth: {ex['ground_truths'][0]}")
        print(f"  EM: {ex['em']:.0f}, F1: {ex['f1']:.4f}")
    
    # Example 5: HotpotQA evaluation
    print("\n\n5. HOTPOTQA EVALUATION")
    print("-" * 80)
    
    hotpot_evaluator = HotpotQAEvaluator()
    
    # Mock HotpotQA dataset
    hotpot_dataset = [
        {
            "id": "h1",
            "question": "What year was the person who coined machine learning born?",
            "answer": "1901",
            "supporting_facts": [["Arthur Samuel", 0], ["Arthur Samuel", 1]]
        },
        {
            "id": "h2",
            "question": "Which university did the inventor of backpropagation attend?",
            "answer": "MIT",
            "supporting_facts": [["Geoffrey Hinton", 0], ["MIT", 2]]
        }
    ]
    
    # Mock predictions (answers only)
    hotpot_answer_predictions = {
        "h1": "1901",  # Correct
        "h2": "Massachusetts Institute of Technology"  # Correct but different form
    }
    
    # Mock supporting facts predictions
    hotpot_sp_predictions = {
        "h1": [["Arthur Samuel", 0], ["Arthur Samuel", 1]],  # Perfect
        "h2": [["Geoffrey Hinton", 0]]  # Missing one
    }
    
    # Evaluate answers only
    answer_results = hotpot_evaluator.evaluate_answer(hotpot_answer_predictions, hotpot_dataset)
    print(f"Answer Evaluation:")
    print(f"  EM: {answer_results['em']:.4f}")
    print(f"  F1: {answer_results['f1']:.4f}")
    
    # Evaluate supporting facts
    sp_results = hotpot_evaluator.evaluate_supporting_facts(hotpot_sp_predictions, hotpot_dataset)
    print(f"\nSupporting Facts Evaluation:")
    print(f"  Precision: {sp_results['sp_precision']:.4f}")
    print(f"  Recall: {sp_results['sp_recall']:.4f}")
    print(f"  F1: {sp_results['sp_f1']:.4f}")
    
    # Joint evaluation
    joint_results = hotpot_evaluator.evaluate_joint(
        hotpot_answer_predictions,
        hotpot_sp_predictions,
        hotpot_dataset
    )
    print(f"\nJoint Evaluation:")
    print(f"  Joint EM: {joint_results['joint_em']:.4f}")
    print(f"  Joint F1: {joint_results['joint_f1']:.4f}")
    
    # Example 6: Batch evaluation
    print("\n\n6. BATCH EVALUATION")
    print("-" * 80)
    
    predictions = [
        "Paris",
        "1959",
        "supervised learning",
        "neural networks",
        "Arthur Samuel"
    ]
    
    ground_truths_list = [
        ["Paris", "the city of Paris"],
        ["1959"],
        ["supervised, unsupervised, reinforcement"],
        ["neural networks", "artificial neural networks"],
        ["Samuel", "Arthur Samuel"]
    ]
    
    batch_results = qa_metrics.evaluate_batch(predictions, ground_truths_list)
    
    print(f"Batch Results ({batch_results['count']} examples):")
    print(f"  Average EM: {batch_results['em']:.4f} ({batch_results['em']*100:.1f}%)")
    print(f"  Average F1: {batch_results['f1']:.4f}")
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)