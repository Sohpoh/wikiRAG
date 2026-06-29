import logging
import time
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReformulatedQuery:
    """Represents a reformulated query with variations."""
    original_query: str
    reformulated_queries: List[str]
    key_entities: List[str]
    method: str


class QueryReformulator:
    """
    Reformulate queries using LLMs to improve retrieval.
    Supports multiple reformulation strategies.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        max_reformulations: int = 3
    ):
        """
        Initialize query reformulator.
        
        Args:
            model_name: OpenAI model to use
            api_key: OpenAI API key (uses env var OPENAI_API_KEY if None)
            max_reformulations: Maximum number of query variations
        """
        self.model_name = model_name
        self.max_reformulations = max_reformulations
        
        # Set up OpenAI client
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            
        if not api_key:
            logger.warning("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
            
        self.client = OpenAI(api_key=api_key)
        
        logger.info(f"QueryReformulator initialized with model: {model_name}")
    
    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 200,
        max_attempts: int = 3
    ) -> str:
        """
        Call OpenAI API with simple retry logic (no Tenacity).
        """
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            
            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {e}")
                
                if attempt == max_attempts:
                    logger.error("Max retry attempts reached. Raising error.")
                    raise
                
                sleep_time = min(2 ** attempt, 10)  # exponential backoff capped at 10s
                logger.info(f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
    
    def expand_query(
        self,
        query: str,
        strategy: str = "synonyms"
    ) -> ReformulatedQuery:
        """
        Expand query with additional terms/variations.
        """
        logger.info(f"Expanding query with strategy: {strategy}")
        
        if strategy == "synonyms":
            prompt = f"""Given this search query, generate {self.max_reformulations} alternative versions using synonyms and related terms. Keep the same meaning but use different words.

Original query: {query}

Generate {self.max_reformulations} alternative queries (one per line):"""
        
        elif strategy == "decompose":
            prompt = f"""Given this complex query, break it down into {self.max_reformulations} simpler sub-questions that would help answer the original question.

Original query: {query}

Generate {self.max_reformulations} sub-questions (one per line):"""
        
        elif strategy == "perspective":
            prompt = f"""Rephrase this query from {self.max_reformulations} different perspectives or angles while maintaining the same information need.

Original query: {query}

Generate {self.max_reformulations} rephrased queries (one per line):"""
        
        else:  # general
            prompt = f"""Expand and reformulate this search query in {self.max_reformulations} different ways to improve information retrieval.

Original query: {query}

Generate {self.max_reformulations} query variations (one per line):"""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that reformulates search queries to improve information retrieval."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_llm(messages, temperature=0.7)
        
        reformulated = [
            q.lstrip('0123456789.-) ').strip()
            for q in response.split('\n')
            if q.strip() and q.strip() != query
        ]
        
        return ReformulatedQuery(
            original_query=query,
            reformulated_queries=reformulated[:self.max_reformulations],
            key_entities=[],
            method=f"expand_{strategy}"
        )
    
    def extract_entities(
        self,
        query: str
    ) -> List[str]:
        """
        Extract key entities from query.
        """
        prompt = f"""Extract only the key terms and important concepts from this query.

Query: {query}

List key terms (one per line):"""
        
        messages = [
            {"role": "system", "content": "You extract key entities from search queries."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_llm(messages, temperature=0.3)
        
        entities = [
            e.lstrip('0123456789.-) ').strip()
            for e in response.split('\n')
            if e.strip()
        ]
        
        return entities
    
    def generate_hypothetical_document(
        self,
        query: str
    ) -> str:
        """
        HyDE: Create a short hypothetical answer passage.
        """
        logger.info("Generating hypothetical document (HyDE)")
        
        prompt = f"""Write a short encyclopedia-style passage answering the question.

Question: {query}

Passage:"""
        
        messages = [
            {"role": "system", "content": "You write concise, informative passages."},
            {"role": "user", "content": prompt}
        ]
        
        return self._call_llm(messages, temperature=0.7, max_tokens=150)
    
    def reformulate_with_context(
        self,
        query: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[str]] = None
    ) -> ReformulatedQuery:
        """
        Reformulate a query while including previous context.
        """
        logger.info("Reformulating with context")
        
        prompt_parts = []
        
        if conversation_history:
            prompt_parts.append("Previous conversation:")
            for turn in conversation_history[-3:]:
                prompt_parts.append(f"- {turn}")
            prompt_parts.append("")
        
        if context:
            prompt_parts.append(f"Context: {context}\n")
        
        prompt_parts.append(f"Current query: {query}")
        prompt_parts.append(f"Generate {self.max_reformulations} clear context-aware reformulations:")
        
        prompt = "\n".join(prompt_parts)
        
        messages = [
            {"role": "system", "content": "You make search queries self-contained and explicit."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_llm(messages, temperature=0.7)
        
        reformulated = [
            q.lstrip('0123456789.-) ').strip()
            for q in response.split('\n')
            if q.strip() and q.strip() != query
        ]
        
        return ReformulatedQuery(
            original_query=query,
            reformulated_queries=reformulated[:self.max_reformulations],
            key_entities=[],
            method="context_aware"
        )
    
    def multi_strategy_reformulation(
        self,
        query: str,
        strategies: Optional[List[str]] = None
    ) -> ReformulatedQuery:
        """
        Use multiple reformulation strategies and combine results.
        """
        if strategies is None:
            strategies = ["synonyms", "perspective"]
        
        logger.info(f"Multi-strategy reformulation: {strategies}")
        
        all_reformulations = []
        
        for strategy in strategies:
            try:
                result = self.expand_query(query, strategy=strategy)
                all_reformulations.extend(result.reformulated_queries)
            except Exception as e:
                logger.warning(f"{strategy} failed: {e}")
        
        seen = set()
        unique_reformulations = []
        for q in all_reformulations:
            q_lower = q.lower()
            if q_lower not in seen and q_lower != query.lower():
                seen.add(q_lower)
                unique_reformulations.append(q)
        
        entities = self.extract_entities(query)
        
        return ReformulatedQuery(
            original_query=query,
            reformulated_queries=unique_reformulations[:self.max_reformulations * len(strategies)],
            key_entities=entities,
            method="multi_strategy"
        )
    
    def simple_expansion(
        self,
        query: str
    ) -> List[str]:
        """
        Fallback non-LLM query expansion.
        """
        expansions = [query]
        
        question_words = ["what", "why", "how", "when", "where", "who", "which"]
        query_lower = query.lower()
        
        for qword in question_words:
            if query_lower.startswith(qword):
                keyword_version = query_lower[len(qword):].strip(" ?")
                if keyword_version:
                    expansions.append(keyword_version)
                break
        
        if len(query.split()) <= 4:
            expansions.append(f'"{query}"')
        
        return expansions



# Example usage
if __name__ == "__main__":
    # Set your OpenAI API key
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # Initialize reformulator
    reformulator = QueryReformulator(
        model_name="gpt-3.5-turbo",
        max_reformulations=3
    )
    
    # Test queries
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Who invented the telephone and when?",
        "Explain quantum computing in simple terms"
    ]
    
    print("=" * 80)
    print("QUERY REFORMULATION EXAMPLES")
    print("=" * 80)
    
    for query in queries:
        print(f"\nOriginal Query: {query}")
        print("-" * 80)
        
        # Strategy 1: Synonym expansion
        print("\n1. Synonym Expansion:")
        result = reformulator.expand_query(query, strategy="synonyms")
        for i, q in enumerate(result.reformulated_queries, 1):
            print(f"   {i}. {q}")
        
        # Strategy 2: Perspective rephrasing
        print("\n2. Perspective Rephrasing:")
        result = reformulator.expand_query(query, strategy="perspective")
        for i, q in enumerate(result.reformulated_queries, 1):
            print(f"   {i}. {q}")
        
        # Strategy 3: Entity extraction
        print("\n3. Key Entities:")
        entities = reformulator.extract_entities(query)
        print(f"   {', '.join(entities)}")
        
        # Strategy 4: Hypothetical document (HyDE)
        print("\n4. Hypothetical Document (HyDE):")
        hyde_doc = reformulator.generate_hypothetical_document(query)
        print(f"   {hyde_doc}")
        
        print("\n" + "=" * 80)
    """
    # Example: Multi-strategy reformulation
    print("\n\nMULTI-STRATEGY REFORMULATION")
    print("=" * 80)
    
    query = "What are the benefits of deep learning?"
    print(f"Original Query: {query}\n")
    
    result = reformulator.multi_strategy_reformulation(
        query,
        strategies=["synonyms", "perspective"]
    )
    
    print(f"Generated {len(result.reformulated_queries)} reformulations:")
    for i, q in enumerate(result.reformulated_queries, 1):
        print(f"{i}. {q}")
    
    print(f"\nKey Entities: {', '.join(result.key_entities)}")
    
    # Example: Context-aware reformulation
    print("\n\nCONTEXT-AWARE REFORMULATION")
    print("=" * 80)
    
    conversation_history = [
        "What is neural network?",
        "How does backpropagation work?"
    ]
    current_query = "What about the vanishing gradient problem?"
    
    print("Conversation History:")
    for turn in conversation_history:
        print(f"  - {turn}")
    print(f"\nCurrent Query: {current_query}\n")
    
    result = reformulator.reformulate_with_context(
        current_query,
        conversation_history=conversation_history
    )
    
    print("Reformulated (with context):")
    for i, q in enumerate(result.reformulated_queries, 1):
        print(f"{i}. {q}")
    
    # Example: Simple expansion (no LLM)
    print("\n\nSIMPLE EXPANSION (No LLM)")
    print("=" * 80)
    
    query = "How does photosynthesis work?"
    print(f"Original Query: {query}\n")
    
    expansions = reformulator.simple_expansion(query)
    print("Expansions:")
    for i, q in enumerate(expansions, 1):
        print(f"{i}. {q}")
    """