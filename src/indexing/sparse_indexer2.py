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
        """
        Initialize Elasticsearch indexer.
        
        Args:
            index_name: Name of the Elasticsearch index
            host: Elasticsearch host
            port: Elasticsearch port
            scheme: http or https
            username: Optional username for authentication
            password: Optional password for authentication
        """
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
                logging.info(f"Connected to Elasticsearch at {host}:{port} (version: {info.get('version', {}).get('number', 'unknown')})")
                print(f"Connected to Elasticsearch at {host}:{port} (version: {info.get('version', {}).get('number', 'unknown')})")
            except Exception as e:
                logging.error(f"Failed to connect to Elasticsearch: {e}")
                print(f"Failed to connect to Elasticsearch: {e}")
                raise
        except Exception as e:
            logging.error(f"Failed to initialize Elasticsearch client: {e}")
            print(f"Failed to initialize Elasticsearch client: {e}")
            raise
    
    def create_index(self, delete_existing: bool = False):
        """
        Create Elasticsearch index with BM25 settings.
        
        Args:
            delete_existing: If True, delete existing index before creating new one
        """
        if delete_existing and self.es.indices.exists(index=self.index_name):
            logging.info(f"Deleting existing index: {self.index_name}")
            self.es.indices.delete(index=self.index_name)
        
        if self.es.indices.exists(index=self.index_name):
            logging.info(f"Index {self.index_name} already exists")
            return
        
        # Define index mapping with BM25 analyzer
        index_mapping = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard"
                        }
                    }
                },
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "passage_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "section": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "text": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "start_offset": {"type": "integer"},
                    "end_offset": {"type": "integer"},
                    "token_count": {"type": "integer"}
                }
            }
        }
        
        try:
            # For elasticsearch-py 8.x, use body parameter
            self.es.indices.create(index=self.index_name, body=index_mapping)
            logging.info(f"Created index: {self.index_name}")
            print(f"Created index: {self.index_name}")
        except RequestError as e:
            if hasattr(e, 'error') and e.error == 'resource_already_exists_exception':
                logging.info(f"Index {self.index_name} already exists")
            elif 'resource_already_exists_exception' in str(e):
                logging.info(f"Index {self.index_name} already exists")
            else:
                raise
    
    def index_passages_from_jsonl(self, jsonl_path: str, batch_size: int = 1000):
        """
        Index passages from a JSONL file.
        
        Args:
            jsonl_path: Path to JSONL file containing passages
            batch_size: Number of documents to index per batch
        """
        jsonl_file = Path(jsonl_path)
        if not jsonl_file.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        
        logging.info(f"Indexing passages from {jsonl_path}")
        print(f"Indexing passages from {jsonl_path}")
        
        def passage_generator():
            """Generator that yields passages for bulk indexing."""
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        passage = json.loads(line)
                        yield {
                            "_index": self.index_name,
                            "_id": passage.get("passage_id"),
                            "_source": passage
                        }
                    except json.JSONDecodeError as e:
                        logging.warning(f"Skipping malformed line {line_num}: {e}")
                        continue
        
        # Use bulk helper to index passages
        success_count = 0
        error_count = 0
        
        for ok, response in helpers.streaming_bulk(
            self.es,
            passage_generator(),
            chunk_size=batch_size,
            raise_on_error=False,
            request_timeout=60
        ):
            if not ok:
                error_count += 1
                logging.warning(f"Failed to index document: {response}")
            else:
                success_count += 1
        
        logging.info(f"Indexed {success_count} passages, {error_count} errors")
        print(f"Indexed {success_count} passages, {error_count} errors")
        
        # Refresh index to make documents searchable
        self.es.indices.refresh(index=self.index_name)
        logging.info(f"Index refreshed")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        fields: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Search the index using BM25.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            fields: Optional list of fields to search (default: ["text"])
            
        Returns:
            List of SearchResult objects
        """
        if fields is None:
            fields = ["text"]
        
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": fields,
                    "type": "best_fields"
                }
            },
            "size": top_k
        }
        
        try:
            # Newer elasticsearch-py uses direct parameters
            response = self.es.search(index=self.index_name, **search_body)
            
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                result = SearchResult(
                    passage_id=source.get("passage_id", ""),
                    score=hit["_score"],
                    text=source.get("text", ""),
                    doc_id=source.get("doc_id"),
                    title=source.get("title"),
                    section=source.get("section")
                )
                results.append(result)
            
            return results
        except Exception as e:
            logging.error(f"Search failed: {e}")
            raise
    
    def delete_index(self):
        """Delete the Elasticsearch index."""
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
            logging.info(f"Deleted index: {self.index_name}")
            print(f"Deleted index: {self.index_name}")
        else:
            logging.info(f"Index {self.index_name} does not exist")
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the index."""
        if not self.es.indices.exists(index=self.index_name):
            return {"exists": False}
        
        stats = self.es.indices.stats(index=self.index_name)
        count = self.es.count(index=self.index_name)
        
        return {
            "exists": True,
            "document_count": count["count"],
            "index_size": stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"]
        }


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Initialize indexer
    indexer = ElasticsearchIndexer(
        index_name="wikipedia_bm25",
        host="127.0.0.1",
        port=9200
    )
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
