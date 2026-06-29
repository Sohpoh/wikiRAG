import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGPrompt:
    """Represents a complete RAG prompt with metadata."""
    prompt: str
    query: str
    passages: List[Dict]
    num_passages: int
    total_tokens: int
    truncated: bool


class PromptBuilder:
    """
    Build prompts for RAG by combining query and retrieved passages.
    Handles token limits and various prompt formats.
    """
    
    def __init__(
        self,
        max_context_length: int = 4000,
        passage_format: str = "numbered",
        include_metadata: bool = True
    ):
        """
        Initialize prompt builder.
        
        Args:
            max_context_length: Maximum number of tokens for context
            passage_format: Format for passages
                - "numbered": [1] passage1 [2] passage2
                - "xml": <passage id="1">passage1</passage>
                - "markdown": ## Passage 1\npassage1
                - "simple": passage1\n\npassage2
            include_metadata: Include title/section in passages
        """
        self.max_context_length = max_context_length
        self.passage_format = passage_format
        self.include_metadata = include_metadata
        
        logger.info(f"PromptBuilder initialized (max_context_length={max_context_length}, "
                   f"format={passage_format})")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate number of tokens in text.
        Simple approximation: ~4 characters per token.
        For production, use tiktoken or transformers tokenizer.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    def format_passage(
        self,
        passage: Union[Dict, object],
        index: int,
        include_citations: bool = False
    ) -> str:
        """
        Format a single passage according to the specified format.
        
        Args:
            passage: Passage object or dictionary
            index: Passage index (1-based)
            include_citations: Include citation markers
            
        Returns:
            Formatted passage string
        """
        # Extract passage information
        if isinstance(passage, dict):
            text = passage.get('text', '')
            title = passage.get('title', '')
            section = passage.get('section', '')
            passage_id = passage.get('passage_id', '')
        else:
            text = getattr(passage, 'text', '')
            title = getattr(passage, 'title', '')
            section = getattr(passage, 'section', '')
            passage_id = getattr(passage, 'passage_id', '')
        
        # Build metadata prefix
        metadata = ""
        if self.include_metadata and title:
            if section:
                metadata = f"{title} - {section}: "
            else:
                metadata = f"{title}: "
        
        # Format according to style
        if self.passage_format == "numbered":
            if include_citations:
                return f"[{index}] {metadata}{text}"
            else:
                return f"[{index}] {metadata}{text}"
        
        elif self.passage_format == "xml":
            attrs = f' id="{index}"'
            if title:
                attrs += f' title="{title}"'
            if section:
                attrs += f' section="{section}"'
            return f"<passage{attrs}>\n{text}\n</passage>"
        
        elif self.passage_format == "markdown":
            header = f"## Passage {index}"
            if self.include_metadata and title:
                header += f": {title}"
                if section:
                    header += f" - {section}"
            return f"{header}\n\n{text}"
        
        else:  # simple
            if metadata:
                return f"{metadata}{text}"
            else:
                return text
    
    def build_context(
        self,
        passages: List,
        max_tokens: Optional[int] = None
    ) -> tuple[str, List, bool]:
        """
        Build context from passages, respecting token limits.
        
        Args:
            passages: List of passage objects
            max_tokens: Maximum tokens (uses self.max_context_length if None)
            
        Returns:
            Tuple of (context_string, included_passages, was_truncated)
        """
        if max_tokens is None:
            max_tokens = self.max_context_length
        
        context_parts = []
        included_passages = []
        current_tokens = 0
        truncated = False
        
        for idx, passage in enumerate(passages, 1):
            formatted = self.format_passage(passage, idx)
            passage_tokens = self.estimate_tokens(formatted)
            
            # Check if adding this passage exceeds limit
            if current_tokens + passage_tokens > max_tokens:
                logger.warning(f"Context limit reached. Including {idx-1}/{len(passages)} passages.")
                truncated = True
                break
            
            context_parts.append(formatted)
            included_passages.append(passage)
            current_tokens += passage_tokens
        
        # Join passages with appropriate separator
        if self.passage_format == "simple":
            context = "\n\n".join(context_parts)
        elif self.passage_format == "markdown":
            context = "\n\n".join(context_parts)
        else:
            context = "\n\n".join(context_parts)
        
        return context, included_passages, truncated
    
    def build_rag_prompt(
        self,
        query: str,
        passages: List,
        instruction: Optional[str] = None,
        prompt_template: str = "default"
    ) -> RAGPrompt:
        """
        Build complete RAG prompt with query and context.
        
        Args:
            query: User query
            passages: List of retrieved passages
            instruction: Optional system instruction
            prompt_template: Template to use
                - "default": Standard QA prompt
                - "strict": Emphasize using only provided context
                - "cot": Chain-of-thought reasoning
                - "factoid": For short factoid answers
                
        Returns:
            RAGPrompt object
        """
        logger.info(f"Building RAG prompt for query: '{query[:50]}...'")
        
        # Build context from passages
        context, included_passages, truncated = self.build_context(passages)
        context_tokens = self.estimate_tokens(context)
        
        # Select prompt template
        if prompt_template == "default":
            template = """Use the following passages to answer the question. If the answer cannot be found in the passages, say "I cannot find the answer in the provided context."

Context:
{context}

Question: {query}

Answer:"""
        
        elif prompt_template == "strict":
            template = """Answer the question using ONLY the information from the passages below. Do not use any external knowledge. If the passages do not contain enough information to answer the question, explicitly state that.

Passages:
{context}

Question: {query}

Answer based only on the passages above:"""
        
        elif prompt_template == "cot":
            template = """Use the following passages to answer the question. Think step by step and explain your reasoning.

Context:
{context}

Question: {query}

Let's approach this step by step:
1) First, identify the relevant information from the passages
2) Then, synthesize the information to form a complete answer

Answer:"""
        
        elif prompt_template == "factoid":
            template = """Based on the passages below, provide a concise answer to the question. Be brief and direct.

Passages:
{context}

Question: {query}

Short Answer:"""
        
        elif prompt_template == "hotpotqa":
            template = """Answer the question using the provided passages. This question may require combining information from multiple passages.

Passages:
{context}

Question: {query}

Answer (combining relevant information):"""
        
        else:
            raise ValueError(f"Unknown prompt template: {prompt_template}")
        
        # Add custom instruction if provided
        if instruction:
            template = f"{instruction}\n\n{template}"
        
        # Fill in template
        prompt = template.format(context=context, query=query)
        total_tokens = self.estimate_tokens(prompt)
        
        logger.info(f"Built prompt: {len(included_passages)} passages, "
                   f"~{total_tokens} tokens, truncated={truncated}")
        
        # Create RAGPrompt object
        rag_prompt = RAGPrompt(
            prompt=prompt,
            query=query,
            passages=[self._passage_to_dict(p) for p in included_passages],
            num_passages=len(included_passages),
            total_tokens=total_tokens,
            truncated=truncated
        )
        
        return rag_prompt
    
    def build_multi_turn_prompt(
        self,
        query: str,
        passages: List,
        conversation_history: List[Dict[str, str]],
        max_history: int = 3
    ) -> RAGPrompt:
        """
        Build prompt for multi-turn conversation with history.
        
        Args:
            query: Current query
            passages: Retrieved passages
            conversation_history: List of previous turns
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            max_history: Maximum number of previous turns to include
            
        Returns:
            RAGPrompt object
        """
        # Build context from passages
        context, included_passages, truncated = self.build_context(passages)
        
        # Build conversation history
        history_text = ""
        recent_history = conversation_history[-max_history*2:] if conversation_history else []
        
        for turn in recent_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                history_text += f"User: {content}\n"
            else:
                history_text += f"Assistant: {content}\n"
        
        # Build prompt
        template = """Conversation History:
{history}

Use the following passages to answer the current question:

Context:
{context}

Current Question: {query}

Answer:"""
        
        prompt = template.format(
            history=history_text if history_text else "No previous conversation.",
            context=context,
            query=query
        )
        
        total_tokens = self.estimate_tokens(prompt)
        
        rag_prompt = RAGPrompt(
            prompt=prompt,
            query=query,
            passages=[self._passage_to_dict(p) for p in included_passages],
            num_passages=len(included_passages),
            total_tokens=total_tokens,
            truncated=truncated
        )
        
        return rag_prompt
    
    def build_with_supporting_facts(
        self,
        query: str,
        passages: List,
        require_supporting_facts: bool = True
    ) -> RAGPrompt:
        """
        Build prompt that asks for supporting facts (for HotpotQA-style evaluation).
        
        Args:
            query: Query string
            passages: Retrieved passages
            require_supporting_facts: Whether to ask for supporting facts
            
        Returns:
            RAGPrompt object
        """
        context, included_passages, truncated = self.build_context(passages)
        
        if require_supporting_facts:
            template = """Answer the question using the passages below. After your answer, list the passage numbers that support your answer.

Passages:
{context}

Question: {query}

Answer: [Your answer here]

Supporting Passages: [List passage numbers that support your answer, e.g., [1, 3]]"""
        else:
            template = """Answer the question using the passages below.

Passages:
{context}

Question: {query}

Answer:"""
        
        prompt = template.format(context=context, query=query)
        total_tokens = self.estimate_tokens(prompt)
        
        rag_prompt = RAGPrompt(
            prompt=prompt,
            query=query,
            passages=[self._passage_to_dict(p) for p in included_passages],
            num_passages=len(included_passages),
            total_tokens=total_tokens,
            truncated=truncated
        )
        
        return rag_prompt
    
    def _passage_to_dict(self, passage: Union[Dict, object]) -> Dict:
        """Convert passage object to dictionary."""
        if isinstance(passage, dict):
            return passage
        else:
            return {
                'passage_id': getattr(passage, 'passage_id', ''),
                'text': getattr(passage, 'text', ''),
                'title': getattr(passage, 'title', ''),
                'section': getattr(passage, 'section', ''),
                'doc_id': getattr(passage, 'doc_id', ''),
                'score': getattr(passage, 'rerank_score', getattr(passage, 'score', 0.0))
            }


# Example usage
if __name__ == "__main__":
    from dataclasses import dataclass
    
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
            "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "Machine Learning",
            "Introduction",
            "1",
            0.92
        ),
        MockRerankedResult(
            "1_p1",
            "The term 'machine learning' was coined by Arthur Samuel in 1959. He defined it as a field of study that gives computers the ability to learn without being explicitly programmed.",
            "Machine Learning",
            "History",
            "1",
            0.87
        ),
        MockRerankedResult(
            "2_p0",
            "Machine learning algorithms build a mathematical model based on sample data, known as training data, to make predictions or decisions without being explicitly programmed to do so.",
            "Machine Learning Algorithms",
            "Overview",
            "2",
            0.85
        ),
        MockRerankedResult(
            "3_p0",
            "There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Each type is suited for different kinds of problems.",
            "Types of Machine Learning",
            "Categories",
            "3",
            0.83
        ),
        MockRerankedResult(
            "1_p2",
            "Applications of machine learning include computer vision, natural language processing, speech recognition, email filtering, and recommendation systems.",
            "Machine Learning",
            "Applications",
            "1",
            0.80
        ),
    ]
    
    query = "What is machine learning and who coined the term?"
    
    print("=" * 80)
    print("PROMPT BUILDER EXAMPLES")
    print("=" * 80)
    
    # Example 1: Default prompt format
    print("\n1. DEFAULT PROMPT FORMAT (numbered)")
    print("-" * 80)
    
    builder_default = PromptBuilder(
        max_context_length=4000,
        passage_format="numbered",
        include_metadata=True
    )
    
    rag_prompt = builder_default.build_rag_prompt(
        query=query,
        passages=mock_passages,
        prompt_template="default"
    )
    
    print(rag_prompt.prompt)
    print(f"\nMetadata:")
    print(f"  Passages included: {rag_prompt.num_passages}")
    print(f"  Estimated tokens: {rag_prompt.total_tokens}")
    print(f"  Truncated: {rag_prompt.truncated}")
    
    # Example 2: XML format
    print("\n\n2. XML PROMPT FORMAT")
    print("-" * 80)
    
    builder_xml = PromptBuilder(
        max_context_length=4000,
        passage_format="xml",
        include_metadata=True
    )
    
    rag_prompt_xml = builder_xml.build_rag_prompt(
        query=query,
        passages=mock_passages[:3],
        prompt_template="default"
    )
    
    print(rag_prompt_xml.prompt[:800] + "...")
    
    # Example 3: Markdown format
    print("\n\n3. MARKDOWN PROMPT FORMAT")
    print("-" * 80)
    
    builder_md = PromptBuilder(
        max_context_length=4000,
        passage_format="markdown",
        include_metadata=True
    )
    
    rag_prompt_md = builder_md.build_rag_prompt(
        query=query,
        passages=mock_passages[:2],
        prompt_template="default"
    )
    
    print(rag_prompt_md.prompt[:800] + "...")
    
    # Example 4: Strict template (emphasizes using only context)
    print("\n\n4. STRICT PROMPT TEMPLATE")
    print("-" * 80)
    
    rag_prompt_strict = builder_default.build_rag_prompt(
        query=query,
        passages=mock_passages[:3],
        prompt_template="strict"
    )
    
    print(rag_prompt_strict.prompt[:600] + "...")
    
    # Example 5: Chain-of-thought template
    print("\n\n5. CHAIN-OF-THOUGHT PROMPT")
    print("-" * 80)
    
    rag_prompt_cot = builder_default.build_rag_prompt(
        query=query,
        passages=mock_passages[:3],
        prompt_template="cot"
    )
    
    print(rag_prompt_cot.prompt[:600] + "...")
    
    # Example 6: Factoid template (short answers)
    print("\n\n6. FACTOID PROMPT (Short Answers)")
    print("-" * 80)
    
    rag_prompt_factoid = builder_default.build_rag_prompt(
        query="Who coined the term machine learning?",
        passages=mock_passages[:3],
        prompt_template="factoid"
    )
    
    print(rag_prompt_factoid.prompt[:600] + "...")
    
    # Example 7: HotpotQA template
    print("\n\n7. HOTPOTQA PROMPT (Multi-hop)")
    print("-" * 80)
    
    rag_prompt_hotpot = builder_default.build_rag_prompt(
        query="What year was the term coined and by whom?",
        passages=mock_passages[:4],
        prompt_template="hotpotqa"
    )
    
    print(rag_prompt_hotpot.prompt[:600] + "...")
    
    # Example 8: With supporting facts
    print("\n\n8. PROMPT WITH SUPPORTING FACTS")
    print("-" * 80)
    
    rag_prompt_sf = builder_default.build_with_supporting_facts(
        query=query,
        passages=mock_passages[:4],
        require_supporting_facts=True
    )
    
    print(rag_prompt_sf.prompt[:700] + "...")
    
    # Example 9: Multi-turn conversation
    print("\n\n9. MULTI-TURN CONVERSATION PROMPT")
    print("-" * 80)
    
    conversation_history = [
        {"role": "user", "content": "What is artificial intelligence?"},
        {"role": "assistant", "content": "Artificial intelligence is the simulation of human intelligence by machines."},
        {"role": "user", "content": "What about machine learning?"}
    ]
    
    rag_prompt_conv = builder_default.build_multi_turn_prompt(
        query="How does it relate to AI?",
        passages=mock_passages[:3],
        conversation_history=conversation_history,
        max_history=2
    )
    
    print(rag_prompt_conv.prompt[:800] + "...")
    
    # Example 10: Token limit truncation
    print("\n\n10. TOKEN LIMIT TRUNCATION")
    print("-" * 80)
    
    builder_small = PromptBuilder(
        max_context_length=400,  # Very small limit
        passage_format="numbered",
        include_metadata=True
    )
    
    rag_prompt_trunc = builder_small.build_rag_prompt(
        query=query,
        passages=mock_passages,  # All 5 passages
        prompt_template="default"
    )
    
    print(f"Requested: {len(mock_passages)} passages")
    print(f"Included: {rag_prompt_trunc.num_passages} passages")
    print(f"Truncated: {rag_prompt_trunc.truncated}")
    print(f"Estimated tokens: {rag_prompt_trunc.total_tokens}")
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)