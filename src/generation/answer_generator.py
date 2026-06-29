import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from openai import OpenAI
import os
from tenacity import retry, stop_after_attempt, wait_exponential
import time
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GeneratedAnswer:
    """Represents a generated answer with metadata."""
    answer: str
    query: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    generation_time: float
    num_passages_used: int
    supporting_passages: Optional[List[int]] = None


class AnswerGenerator:
    """
    Generate answers using LLMs based on retrieved context.
    Supports OpenAI models and customizable generation parameters.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 256,
        top_p: float = 1.0
    ):
        """
        Initialize answer generator.
        
        Args:
            model_name: OpenAI model to use
                - "gpt-3.5-turbo" (fast, cheap)
                - "gpt-3.5-turbo-16k" (longer context)
                - "gpt-4" (best quality)
                - "gpt-4-turbo-preview" (long context, good quality)
            api_key: OpenAI API key (uses env var OPENAI_API_KEY if None)
            temperature: Sampling temperature (0.0-1.0, lower = more deterministic)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        
        self.top_p = top_p
        
        # Set up OpenAI client
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            
        if not api_key:
            logger.warning("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
            
        self.client = OpenAI(api_key=api_key)
        
        logger.info(f"AnswerGenerator initialized with model: {model_name}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> GeneratedAnswer:
        """
        Generate answer using OpenAI API.
        
        Args:
            prompt: Complete prompt with context and query
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stop_sequences: Optional stop sequences
            
        Returns:
            GeneratedAnswer object
        """
        start_time = time.time()
        
        # Use defaults if not specified
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            logger.info(f"Generating answer with {self.model_name}")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=max_tok,
                top_p=self.top_p,
                stop=stop_sequences
            )
            
            # Extract answer
            answer = response.choices[0].message.content.strip()
            
            # Extract token usage
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            
            generation_time = time.time() - start_time
            
            logger.info(f"Answer generated in {generation_time:.2f}s "
                       f"({total_tokens} tokens)")
            
            # Create GeneratedAnswer object
            generated = GeneratedAnswer(
                answer=answer,
                query="",  # Will be set by caller
                model=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                generation_time=generation_time,
                num_passages_used=0  # Will be set by caller
            )
            
            return generated
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    def generate_from_rag_prompt(
        self,
        rag_prompt,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> GeneratedAnswer:
        """
        Generate answer from RAGPrompt object.
        
        Args:
            rag_prompt: RAGPrompt object from PromptBuilder
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            GeneratedAnswer object
        """
        generated = self.generate(
            prompt=rag_prompt.prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Add query and passage info
        generated.query = rag_prompt.query
        generated.num_passages_used = rag_prompt.num_passages
        
        return generated
    
    def generate_with_citations(
        self,
        rag_prompt,
        citation_format: str = "numbered"
    ) -> GeneratedAnswer:
        """
        Generate answer and extract citations to passages.
        
        Args:
            rag_prompt: RAGPrompt object
            citation_format: Format for citations
                - "numbered": [1], [2], etc.
                - "extract": Extract passage numbers from answer
                
        Returns:
            GeneratedAnswer with supporting_passages filled
        """
        # Modify prompt to encourage citations
        prompt_with_citations = rag_prompt.prompt + "\n\nCite your sources by including passage numbers [1], [2], etc. in your answer."
        
        generated = self.generate(prompt=prompt_with_citations)
        generated.query = rag_prompt.query
        generated.num_passages_used = rag_prompt.num_passages
        
        # Extract citation numbers from answer
        import re
        citation_pattern = r'\[(\d+)\]'
        citations = re.findall(citation_pattern, generated.answer)
        generated.supporting_passages = [int(c) for c in citations]
        
        logger.info(f"Extracted {len(generated.supporting_passages)} citations: {generated.supporting_passages}")
        
        return generated
    
    def generate_with_confidence(
        self,
        rag_prompt,
        request_confidence: bool = True
    ) -> tuple[GeneratedAnswer, Optional[str]]:
        """
        Generate answer and ask model to indicate confidence.
        
        Args:
            rag_prompt: RAGPrompt object
            request_confidence: Whether to ask for confidence
            
        Returns:
            Tuple of (GeneratedAnswer, confidence_explanation)
        """
        if request_confidence:
            confidence_prompt = rag_prompt.prompt + """\n\nAfter your answer, on a new line, indicate your confidence level (High/Medium/Low) and briefly explain why."""
            
            generated = self.generate(prompt=confidence_prompt, max_tokens=300)
            generated.query = rag_prompt.query
            generated.num_passages_used = rag_prompt.num_passages
            
            # Try to split answer and confidence
            lines = generated.answer.split('\n')
            answer_lines = []
            confidence_lines = []
            found_confidence = False
            
            for line in lines:
                if any(conf in line.lower() for conf in ['confidence:', 'high', 'medium', 'low']) and not found_confidence:
                    found_confidence = True
                    confidence_lines.append(line)
                elif found_confidence:
                    confidence_lines.append(line)
                else:
                    answer_lines.append(line)
            
            if confidence_lines:
                generated.answer = '\n'.join(answer_lines).strip()
                confidence_explanation = '\n'.join(confidence_lines).strip()
            else:
                confidence_explanation = None
            
            return generated, confidence_explanation
        else:
            generated = self.generate_from_rag_prompt(rag_prompt)
            return generated, None
    
    def batch_generate(
        self,
        rag_prompts: List,
        show_progress: bool = True
    ) -> List[GeneratedAnswer]:
        """
        Generate answers for multiple prompts.
        
        Args:
            rag_prompts: List of RAGPrompt objects
            show_progress: Show progress bar
            
        Returns:
            List of GeneratedAnswer objects
        """
        logger.info(f"Batch generating {len(rag_prompts)} answers")
        
        generated_answers = []
        
        iterator = rag_prompts
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(rag_prompts, desc="Generating answers")
        
        for rag_prompt in iterator:
            try:
                generated = self.generate_from_rag_prompt(rag_prompt)
                generated_answers.append(generated)
            except Exception as e:
                logger.error(f"Failed to generate answer for query '{rag_prompt.query}': {e}")
                # Create empty answer on failure
                generated = GeneratedAnswer(
                    answer="[Generation failed]",
                    query=rag_prompt.query,
                    model=self.model_name,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    generation_time=0.0,
                    num_passages_used=rag_prompt.num_passages
                )
                generated_answers.append(generated)
        
        return generated_answers
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """
        Estimate cost of API call based on model and tokens.
        Prices as of 2024 (approximate).
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            
        Returns:
            Estimated cost in USD
        """
        # Pricing per 1K tokens (approximate, check OpenAI pricing for exact rates)
        pricing = {
            "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
            "gpt-3.5-turbo-16k": {"prompt": 0.003, "completion": 0.004},
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
            "gpt-4-turbo-preview": {"prompt": 0.01, "completion": 0.03},
        }
        
        model_pricing = pricing.get(self.model_name, {"prompt": 0.001, "completion": 0.002})
        
        prompt_cost = (prompt_tokens / 1000) * model_pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * model_pricing["completion"]
        
        return prompt_cost + completion_cost


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.generation.prompt_builder import PromptBuilder
    from dataclasses import dataclass
    
    print("=" * 80)
    print("ANSWER GENERATOR EXAMPLES")
    print("=" * 80)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nWarning: OPENAI_API_KEY not found in environment variables.")
        print("Please set it to run this example:")
        print("  export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Mock reranked results
    @dataclass
    class MockRerankedResult:
        passage_id: str
        text: str
        title: str
        section: str
        doc_id: str
        rerank_score: float
    
    mock_passages = [
        MockRerankedResult(
            "1_p0",
            "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
            "Machine Learning",
            "Introduction",
            "1",
            0.92
        ),
        MockRerankedResult(
            "1_p1",
            "The term 'machine learning' was coined by Arthur Samuel in 1959. Samuel was an American pioneer in the field of computer gaming and artificial intelligence, and he defined machine learning as a 'field of study that gives computers the ability to learn without being explicitly programmed.'",
            "Machine Learning",
            "History",
            "1",
            0.89
        ),
        MockRerankedResult(
            "2_p0",
            "Machine learning algorithms build a mathematical model based on sample data, known as training data, to make predictions or decisions without being explicitly programmed to perform the task. The algorithms use statistical techniques to learn patterns from data.",
            "Machine Learning Algorithms",
            "Overview",
            "2",
            0.85
        ),
    ]
    
    # Initialize prompt builder
    prompt_builder = PromptBuilder(
        max_context_length=4000,
        passage_format="numbered",
        include_metadata=True
    )
    
    # Initialize answer generator
    generator = AnswerGenerator(
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=256
    )
    
    query = "What is machine learning and who coined the term?"
    
    # Example 1: Basic answer generation
    print("\n1. BASIC ANSWER GENERATION")
    print("-" * 80)
    print(f"Query: {query}\n")
    
    rag_prompt = prompt_builder.build_rag_prompt(
        query=query,
        passages=mock_passages,
        prompt_template="default"
    )
    
    generated = generator.generate_from_rag_prompt(rag_prompt)
    
    print(f"Answer: {generated.answer}\n")
    print(f"Metadata:")
    print(f"  Model: {generated.model}")
    print(f"  Generation time: {generated.generation_time:.2f}s")
    print(f"  Tokens: {generated.prompt_tokens} prompt + {generated.completion_tokens} completion = {generated.total_tokens} total")
    print(f"  Passages used: {generated.num_passages_used}")
    
    # Estimate cost
    cost = generator.estimate_cost(generated.prompt_tokens, generated.completion_tokens)
    print(f"  Estimated cost: ${cost:.6f}")
    
    # Example 2: Answer with citations
    print("\n\n2. ANSWER WITH CITATIONS")
    print("-" * 80)
    
    rag_prompt_cite = prompt_builder.build_rag_prompt(
        query=query,
        passages=mock_passages,
        prompt_template="default"
    )
    
    generated_cite = generator.generate_with_citations(rag_prompt_cite)
    
    print(f"Answer: {generated_cite.answer}\n")
    print(f"Supporting passages cited: {generated_cite.supporting_passages}")
    
    # Example 3: Answer with confidence
    print("\n\n3. ANSWER WITH CONFIDENCE ESTIMATION")
    print("-" * 80)
    
    rag_prompt_conf = prompt_builder.build_rag_prompt(
        query=query,
        passages=mock_passages,
        prompt_template="default"
    )
    
    generated_conf, confidence = generator.generate_with_confidence(rag_prompt_conf)
    
    print(f"Answer: {generated_conf.answer}\n")
    if confidence:
        print(f"Confidence: {confidence}")
    
    # Example 4: Different prompt templates
    print("\n\n4. DIFFERENT PROMPT TEMPLATES")
    print("-" * 80)
    
    templates = ["strict", "cot", "factoid"]
    
    for template in templates:
        print(f"\nTemplate: {template}")
        print("-" * 40)
        
        rag_prompt_temp = prompt_builder.build_rag_prompt(
            query=query,
            passages=mock_passages,
            prompt_template=template
        )
        
        generated_temp = generator.generate_from_rag_prompt(
            rag_prompt_temp,
            max_tokens=200 if template == "cot" else 150
        )
        
        print(f"Answer: {generated_temp.answer[:300]}...")
    
    # Example 5: Batch generation
    print("\n\n5. BATCH ANSWER GENERATION")
    print("-" * 80)
    
    queries = [
        "What is machine learning?",
        "Who coined the term machine learning?",
        "How do machine learning algorithms work?"
    ]
    
    rag_prompts = []
    for q in queries:
        rp = prompt_builder.build_rag_prompt(
            query=q,
            passages=mock_passages,
            prompt_template="default"
        )
        rag_prompts.append(rp)
    
    print(f"Generating answers for {len(queries)} queries...\n")
    
    batch_results = generator.batch_generate(rag_prompts, show_progress=True)
    
    total_cost = 0
    for q, result in zip(queries, batch_results):
        cost = generator.estimate_cost(result.prompt_tokens, result.completion_tokens)
        total_cost += cost
        print(f"\nQuery: {q}")
        print(f"Answer: {result.answer[:150]}...")
        print(f"Cost: ${cost:.6f}")
    
    print(f"\nTotal cost for batch: ${total_cost:.6f}")
    
    # Example 6: Temperature comparison
    print("\n\n6. TEMPERATURE COMPARISON")
    print("-" * 80)
    
    temperatures = [0.0, 0.5, 1.0]
    
    rag_prompt_temp_test = prompt_builder.build_rag_prompt(
        query="Explain machine learning in simple terms.",
        passages=mock_passages,
        prompt_template="default"
    )
    
    for temp in temperatures:
        print(f"\nTemperature: {temp}")
        print("-" * 40)
        
        generated_temp_test = generator.generate_from_rag_prompt(
            rag_prompt_temp_test,
            temperature=temp
        )
        
        print(f"Answer: {generated_temp_test.answer[:200]}...")
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)