import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import ConnectionError, RequestError

@dataclass
class SearchResult:
    """Represents a single search result."""
    passage_id: str
    score: float
    text: str
    doc_id: Optional[str] = None
    title: Optional[str] = None
    section: Optional[str] = None


class ElasticsearchIndexer:
    """BM25-based sparse retrieval using Elasticsearch."""
    
    def __init__(
        self,
        index_name: str = "wikipedia_bm25",
        host: str = "localhost",
        port: int = 9200,
        scheme: str = "http"

    ):

        self.index_name = index_name
        
        # Build connection URL
        es_url = f"{scheme}://{host}:{port}"
        
        # Connection parameters for newer elasticsearch-py client
        connection_params = {
            'request_timeout': 60,
            'max_retries': 3,
            'retry_on_timeout': True
        }
        
        try:
            # Use URL string format for newer client versions
            self.es = Elasticsearch([es_url], **connection_params)
            
            # Test connection with a simple info request
            try:
                info = self.es.info()
                print(f"Connected to Elasticsearch at {host}:{port} (version: {info.get('version', {}).get('number', 'unknown')})")
            except Exception as e:
                print(f"Failed to connect to Elasticsearch: {e}")
                raise
        except Exception as e:
            print(f"Failed to initialize Elasticsearch client: {e}")
            raise
    
    def create_index(
        self,
        delete_if_exists: bool = False,
        k1: float = 1.2,
        b: float = 0.75
    ):
        """
        Create Elasticsearch index with BM25 configuration.
        
        Args:
            delete_if_exists: Delete index if it already exists
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
        """
        # Check if index exists
        if self.es.indices.exists(index=self.index_name):
            if delete_if_exists:
                print(f"Deleting existing index: {self.index_name}")
                self.es.indices.delete(index=self.index_name)
            else:
                print(f"Index {self.index_name} already exists. Use delete_if_exists=True to recreate.")
                return
        
        # Define index settings and mappings
        index_config = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index": {
                    "similarity": {
                        "default": {
                            "type": "BM25",
                            "k1": k1,
                            "b": b
                        }
                    }
                },
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "passage_id": {
                        "type": "keyword"
                    },
                    "doc_id": {
                        "type": "keyword"
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "section": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "text": {
                        "type": "text",
                        "analyzer": "standard",
                        "similarity": "default"
                    },
                    "token_count": {
                        "type": "integer"
                    }
                }
            }
        }
        
        print(f"Creating index: {self.index_name} with BM25 (k1={k1}, b={b})")
        self.es.indices.create(index=self.index_name, body=index_config)
        print(f"Index {self.index_name} created successfully")
    
    def index_passages(
        self,
        passages: List[Dict],
        batch_size: int = 1000,
        show_progress: bool = True
    ):

        if not self.es.indices.exists(index=self.index_name):
            raise ValueError(f"Index {self.index_name} does not exist. Create it first with create_index()")
        
        print(f"Indexing {len(passages)} passages to {self.index_name}")
        
        def generate_actions():
            for passage in passages:
                yield {
                    "_index": self.index_name,
                    "_id": passage.get("passage_id"),
                    "_source": {
                        "passage_id": passage.get("passage_id"),
                        "doc_id": passage.get("doc_id"),
                        "title": passage.get("title"),
                        "section": passage.get("section"),
                        "text": passage.get("text"),
                        "token_count": passage.get("token_count", 0)
                    }
                }
        
        # Use bulk helper for efficient indexing
        success_count = 0
        error_count = 0
        
        if show_progress:
            progress_bar = tqdm(total=len(passages), desc="Indexing passages")
        
        for ok, response in helpers.streaming_bulk(
            self.es,
            generate_actions(),
            chunk_size=batch_size,
            raise_on_error=False
        ):
            if ok:
                success_count += 1
            else:
                error_count += 1
                print(f"Indexing error: {response}")
            
            if show_progress:
                progress_bar.update(1)
        
        if show_progress:
            progress_bar.close()
        
        # Refresh index to make documents searchable
        self.es.indices.refresh(index=self.index_name)
        
        print(f"Indexing complete: {success_count} successful, {error_count} errors")
    
    def index_passages_from_file(
        self,
        passages_file: str,
        batch_size: int = 1000,
        show_progress: bool = True
    ):
        """
        Index passages directly from JSONL file (memory efficient).
        
        Args:
            passages_file: Path to passages JSONL file
            batch_size: Number of documents per bulk request
            show_progress: Show progress bar
        """
        if not self.es.indices.exists(index=self.index_name):
            raise ValueError(f"Index {self.index_name} does not exist. Create it first with create_index()")
        
        print(f"Indexing passages from {passages_file} to {self.index_name}")
        
        def generate_actions():
            """Generate bulk indexing actions from file."""
            with open(passages_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    passage = json.loads(line)
                    yield {
                        "_index": self.index_name,
                        "_id": passage.get("passage_id"),
                        "_source": {
                            "passage_id": passage.get("passage_id"),
                            "doc_id": passage.get("doc_id"),
                            "title": passage.get("title"),
                            "section": passage.get("section"),
                            "text": passage.get("text"),
                            "token_count": passage.get("token_count", 0)
                        }
                    }
        
        # Count total lines for progress bar
        total_lines = 0
        if show_progress:
            with open(passages_file, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for line in f if line.strip())
        
        # Use bulk helper for efficient indexing
        success_count = 0
        error_count = 0
        
        if show_progress:
            progress_bar = tqdm(total=total_lines, desc="Indexing passages")
        
        for ok, response in helpers.streaming_bulk(
            self.es,
            generate_actions(),
            chunk_size=batch_size,
            raise_on_error=False
        ):
            if ok:
                success_count += 1
            else:
                error_count += 1
                print(f"Indexing error: {response}")
            
            if show_progress:
                progress_bar.update(1)
        
        if show_progress:
            progress_bar.close()
        
        # Refresh index to make documents searchable
        self.es.indices.refresh(index=self.index_name)
        
        print(f"Indexing complete: {success_count} successful, {error_count} errors")
    
    def search(
        self,
        query: str,
        top_k: int = 100,
        return_fields: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Search using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            return_fields: Fields to return (default: all)
            
        Returns:
            List of SearchResult objects
        """
        if not self.es.indices.exists(index=self.index_name):
            raise ValueError(f"Index {self.index_name} does not exist")
        
        # Build search query
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text^1.0", "title^0.5", "section^0.3"],
                    "type": "best_fields"
                }
            },
            "size": top_k
        }
        
        # Specify fields to return
        if return_fields:
            search_body["_source"] = return_fields
        
        # Execute search
        try:
            response = self.es.search(index=self.index_name, body=search_body)
        except RequestError as e:
            print(f"Search error: {e}")
            return []
        
        # Parse results
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            result = SearchResult(
                passage_id=source.get('passage_id', hit['_id']),
                score=hit['_score'],
                text=source.get('text', ''),
                doc_id=source.get('doc_id'),
                title=source.get('title'),
                section=source.get('section')
            )
            results.append(result)
        
        return results
    
    def search_with_filters(
        self,
        query: str,
        top_k: int = 100,
        doc_ids: Optional[List[str]] = None,
        title_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search with additional filters.
        
        Args:
            query: Search query
            top_k: Number of results to return
            doc_ids: Filter by specific document IDs
            title_filter: Filter by title (partial match)
            
        Returns:
            List of SearchResult objects
        """
        if not self.es.indices.exists(index=self.index_name):
            raise ValueError(f"Index {self.index_name} does not exist")
        
        # Build base query
        must_clauses = [
            {
                "multi_match": {
                    "query": query,
                    "fields": ["text^1.0", "title^0.5", "section^0.3"],
                    "type": "best_fields"
                }
            }
        ]
        
        # Add filters
        filter_clauses = []
        
        if doc_ids:
            filter_clauses.append({
                "terms": {
                    "doc_id": doc_ids
                }
            })
        
        if title_filter:
            filter_clauses.append({
                "match": {
                    "title": title_filter
                }
            })
        
        # Build complete query
        search_body = {
            "query": {
                "bool": {
                    "must": must_clauses,
                    "filter": filter_clauses
                }
            },
            "size": top_k
        }
        
        # Execute search
        try:
            response = self.es.search(index=self.index_name, body=search_body)
        except RequestError as e:
            print(f"Search error: {e}")
            return []
        
        # Parse results
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            result = SearchResult(
                passage_id=source.get('passage_id', hit['_id']),
                score=hit['_score'],
                text=source.get('text', ''),
                doc_id=source.get('doc_id'),
                title=source.get('title'),
                section=source.get('section')
            )
            results.append(result)
        
        return results
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 100
    ) -> List[List[SearchResult]]:
        """
        Perform multiple searches efficiently.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            
        Returns:
            List of result lists (one per query)
        """
        all_results = []
        
        print(f"Performing batch search for {len(queries)} queries")
        
        for query in tqdm(queries, desc="Batch searching"):
            results = self.search(query, top_k=top_k)
            all_results.append(results)
        
        return all_results
    
    def get_index_stats(self) -> Dict:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics
        """
        if not self.es.indices.exists(index=self.index_name):
            return {"error": f"Index {self.index_name} does not exist"}
        
        stats = self.es.indices.stats(index=self.index_name)
        count = self.es.count(index=self.index_name)
        
        return {
            "index_name": self.index_name,
            "document_count": count['count'],
            "size_in_bytes": stats['_all']['primaries']['store']['size_in_bytes']
        }
    
    def delete_index(self):
        """Delete the index."""
        if self.es.indices.exists(index=self.index_name):
            print(f"Deleting index: {self.index_name}")
            self.es.indices.delete(index=self.index_name)
            print(f"Index {self.index_name} deleted")
        else:
            print(f"Index {self.index_name} does not exist")
    
    def close(self):
        """Close Elasticsearch connection."""
        self.es.close()


if __name__ == "__main__":
    # Initialize indexer
    indexer = ElasticsearchIndexer(
        index_name="wikipedia_bm25",
        host="127.0.0.1",
        port=9200
    )

    #indexer.create_index(delete_if_exists=True)
    #indexer.index_passages_from_file("data/processed/passages.jsonl")
    query = "What is a bull"
    results = indexer.search(query, top_k=10)
    
    print(f"\nTop results for query: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.score:.4f}")
        print(f"   Title: {result.title}")
        print(f"   Text: {result.text[:200]}...")
        print()

    """
    # Create index
    indexer.create_index(delete_existing=False)
    
    # Index passages from JSONL file
    passages_path = "data/processed/passages.jsonl"
    if Path(passages_path).exists():
        indexer.index_passages_from_jsonl(passages_path, batch_size=1000)
        
        # Print index statistics
        stats = indexer.get_index_stats()
        print(f"\nIndex Statistics:")
        print(f"  Documents indexed: {stats.get('document_count', 0):,}")
        print(f"  Index size: {stats.get('index_size', 0):,} bytes")
    else:
        print(f"Warning: {passages_path} not found. Please run preprocessing first.")
    """
