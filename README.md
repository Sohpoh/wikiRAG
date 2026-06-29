# WikiRAG

A Retrieval-Augmented Generation (RAG) system that answers questions using the full English Wikipedia as a knowledge base. Given a question, the pipeline retrieves relevant passages from Wikipedia and uses an OpenAI language model to synthesize an answer grounded in those passages.

## How it works

### 1. Indexing Wikipedia

The Wikipedia XML dump is parsed and split into passages. Each passage is indexed two ways:

- **Sparse index (Elasticsearch BM25)** — keyword-based inverted index. Fast and good at exact term matching.
- **Dense index (FAISS)** — each passage is embedded using a sentence-transformer model (`all-MiniLM-L6-v2`) and stored as a vector. Good at semantic/paraphrase matching where keywords differ.

### 2. Hybrid retrieval

At query time, the question is sent to both indices:
- BM25 returns a ranked list of passages by keyword overlap.
- FAISS returns a ranked list by cosine similarity of embeddings.

The two lists are merged using **Reciprocal Rank Fusion (RRF)**, which rewards passages that rank highly in both lists without requiring score scales to match.

### 3. Cross-encoder reranking

The top ~50 fused candidates are re-scored by a cross-encoder (`ms-marco-MiniLM-L-6-v2`). Unlike the bi-encoder used for FAISS retrieval, the cross-encoder sees the query and passage together, producing more accurate relevance scores. The top 10 passages after reranking are passed to the generator.

### 4. Prompt building & generation

The top passages are formatted into a prompt and sent to an OpenAI model (default: `gpt-3.5-turbo`). Several prompt templates are available depending on the task (standard, chain-of-thought, factoid, etc.). The model returns an answer along with citations to supporting passages when requested.

## Pipeline stages

```
Query
  ├─ 1. Query Reformulation (optional)   synonym / HyDE expansion
  ├─ 2. Hybrid Retrieval                 BM25 + FAISS → RRF fusion (top 50)
  ├─ 3. Cross-Encoder Reranking          ms-marco (top 10)
  ├─ 4. Prompt Building                  numbered passages + template
  └─ 5. Answer Generation                OpenAI GPT
```

## Repository structure

```
src/
├── pipeline.py              # Interactive REPL (simple entry point)
├── pipeline2.py             # Full configurable pipeline used by evaluation
├── main_evaluation.py       # Evaluation harness (CLI)
├── indexing/
│   ├── sparse_indexer.py    # Elasticsearch indexing
│   └── dense_indexer.py     # FAISS indexing + embedding
├── retrieval/
│   ├── hybrid_retriever.py  # RRF / weighted fusion
│   └── query_reformulation.py
├── reranking/
│   └── cross_encoder_reranker.py
├── generation/
│   ├── prompt_builder.py    # Prompt templates
│   └── answer_generator.py  # OpenAI API wrapper
├── evaluation/
│   ├── retrieval_metrics.py # Recall@k, MRR, nDCG
│   ├── qa_metrics.py        # Exact Match + F1
│   └── ragas_evaluator.py   # Faithfulness, Answer Relevancy, Context Precision/Recall
├── preprocessing/
│   ├── wikipedia_processor.py  # Parse & chunk Wikipedia XML dump
│   └── dataset_loader.py       # SQuAD / HotpotQA loaders
└── scripts/
    ├── download_datasets.py
    └── verify_datasets.py
```

## Requirements

- Python 3.10+
- Elasticsearch 8.x running locally on port 9200
- OpenAI API key
- Wikipedia XML dump (`enwiki-latest-pages-articles-multistream.xml.bz2`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# set OPENAI_API_KEY in .env
```

## Building the indices

These steps only need to be run once.

```bash
# 1. Parse and chunk the Wikipedia dump
python src/preprocessing/wikipedia_processor.py

# 2. Index into Elasticsearch (must be running)
python src/indexing/sparse_indexer.py

# 3. Build the FAISS vector index
python src/indexing/dense_indexer.py
```

Built index files land at `indices/faiss/wikipedia.index` and `indices/faiss/wikipedia_metadata.pkl`.

## Running the pipeline

```bash
python src/pipeline.py
```

```
Enter your question: Who invented the telephone?

Answer: Alexander Graham Bell is credited with inventing the telephone...

Retrieved: 50 | Reranked: 10 | Time: 1.8s
```

## Evaluation

The evaluation harness runs the full pipeline on SQuAD or HotpotQA and reports retrieval, QA, and RAGAS metrics.

```bash
# Evaluate on SQuAD dev (100 samples)
python src/main_evaluation.py --dataset squad --split dev --max_samples 100

# Evaluate on HotpotQA, skip RAGAS for speed
python src/main_evaluation.py --dataset hotpotqa --split dev --skip_ragas

# Use chain-of-thought prompting
python src/main_evaluation.py --dataset squad --prompt_template cot --output_dir results/
```

### Metrics collected

| Category   | Metrics                                                          |
|------------|------------------------------------------------------------------|
| Retrieval  | Recall@5/10/20/50/100, MRR, nDCG                                 |
| QA         | Exact Match (EM), F1                                             |
| RAGAS      | Faithfulness, Answer Relevancy, Context Precision, Context Recall |

Results are written to `results/` as timestamped JSON files and appended to `results/evaluation_summary.csv`.

### Prompt templates

| Template    | Use case                                        |
|-------------|------------------------------------------------|
| `default`   | General-purpose                                |
| `strict`    | Stay grounded in provided context              |
| `cot`       | Chain-of-thought reasoning                     |
| `factoid`   | Short factoid answers                          |
| `hotpotqa`  | Multi-hop reasoning (default for HotpotQA)     |

## Key configuration

| Parameter         | Default              | Description                        |
|-------------------|----------------------|------------------------------------|
| `top_k_sparse`    | 100                  | BM25 candidates per query          |
| `top_k_dense`     | 100                  | Dense retrieval candidates         |
| `top_k_hybrid`    | 50                   | Passages kept after RRF fusion     |
| `top_k_rerank`    | 10                   | Passages kept after cross-encoding |
| `fusion_method`   | `rrf`                | `rrf` or `weighted`                |
| `embedding_model` | `all-MiniLM-L6-v2`   | SentenceTransformer for FAISS      |
| `openai_model`    | `gpt-3.5-turbo`      | OpenAI model for generation        |

## Environment variables

```
OPENAI_API_KEY=sk-...   # Required for answer generation
```
