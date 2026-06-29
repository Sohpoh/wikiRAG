import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator, Iterable
import logging
from dataclasses import dataclass, asdict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------
# Core data structure (keep)
# ---------------------------
@dataclass
class Passage:
    """Represents a passage chunk with metadata."""
    passage_id: str
    doc_id: str
    title: str
    section: str
    text: str
    start_offset: int
    end_offset: int
    token_count: int

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------
# Streaming I/O over dumps
# ---------------------------
def iter_wikiextractor_files(dump_path: str) -> Iterable[Path]:
    """
    Yield all wiki_* files under a WikiExtractor output directory.
    The directory usually contains shard folders like AA/, AB/ with files wiki_00, wiki_01, ...
    """
    base = Path(dump_path)
    if not base.exists():
        raise FileNotFoundError(f"Dump directory not found: {dump_path}")
    files = sorted(p for p in base.rglob("wiki_*") if p.is_file())
    if not files:
        raise ValueError(f"No wiki_* files found in {dump_path}")
    logger.info(f"Found {len(files)} wiki files")
    for f in files:
        yield f


def iter_docs_from_wikiextractor(
    dump_path: str,
    max_docs: Optional[int] = None,
    show_progress: bool = True
) -> Generator[Dict, None, None]:
    """
    Stream documents from WikiExtractor JSONL shards.
    Each line is a JSON object with {id, title, text, url?}.
    Yields dictionaries with keys: doc_id, title, text, url
    """
    count = 0
    files = list(iter_wikiextractor_files(dump_path))
    file_iter = tqdm(files, desc="Scanning wiki files") if show_progress else files

    for file_path in file_iter:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line {line_num} in {file_path.name}: {e}")
                    continue

                if "id" in obj and "title" in obj and "text" in obj:
                    yield {
                        "doc_id": str(obj["id"]),
                        "title": obj["title"],
                        "text": obj["text"],
                        "url": obj.get("url", "")
                    }
                    count += 1
                    if max_docs and count >= max_docs:
                        logger.info(f"Reached max_docs limit: {max_docs}")
                        return


# ---------------------------
# Lightweight text utilities
# ---------------------------
_WORD_RE = re.compile(r"\b\w+\b|[^\w\s]")

def simple_tokenize(text: str) -> List[str]:
    """Very simple whitespace + punctuation tokenizer."""
    return _WORD_RE.findall(text)


def clean_text(text: str) -> str:
    """Minor cleanup on top of WikiExtractor output."""
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\[\]{}]", "", text)
    return text.strip()


def extract_sections(text: str, title: str) -> List[Tuple[str, str]]:
    """
    Heuristic section splitter: treats short, non-terminal lines as headers.
    Falls back to a single 'Introduction' if nothing is found.
    """
    sections: List[Tuple[str, str]] = []
    lines = text.split("\n")
    current_section = "Introduction"
    current_text: List[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if len(line) < 100 and not line.endswith(('.', '!', '?', ';', ':')):
            if current_text:
                sections.append((current_section, "\n".join(current_text)))
                current_text = []
            current_section = line
        else:
            current_text.append(line)

    if current_text:
        sections.append((current_section, "\n".join(current_text)))

    if not sections:
        sections.append((title or "Article", text))

    return sections


# ---------------------------
# Chunking (generator-based)
# ---------------------------
def iter_passages_from_doc(
    doc: Dict,
    chunk_size: int = 256,
    stride: int = 128,
    min_chunk_size: int = 50
) -> Generator[Passage, None, None]:
    """
    Stream Passage objects for a single document.
    """
    doc_id = doc["doc_id"]
    title = doc.get("title", "")
    text = doc.get("text", "")

    sections = extract_sections(text, title)
    pcount = 0

    for section_title, section_text in sections:
        section_text = clean_text(section_text)
        if not section_text:
            continue

        tokens = simple_tokenize(section_text)
        if not tokens:
            continue

        start = 0
        n = len(tokens)
        while start < n:
            end = min(start + chunk_size, n)
            chunk_tokens = tokens[start:end]

            # Skip undersized non-final chunks
            if len(chunk_tokens) < min_chunk_size and end < n:
                start += stride
                continue

            passage = Passage(
                passage_id=f"{doc_id}_p{pcount}",
                doc_id=doc_id,
                title=title,
                section=section_title,
                text=" ".join(chunk_tokens),
                start_offset=start,
                end_offset=end,
                token_count=len(chunk_tokens),
            )
            yield passage
            pcount += 1

            if end >= n:
                break
            start += stride


def iter_passages_from_docs(
    docs: Iterable[Dict],
    chunk_size: int = 256,
    stride: int = 128,
    min_chunk_size: int = 50
) -> Generator[Passage, None, None]:
    """
    Stream Passage objects from a stream of document dicts.
    """
    for doc in docs:
        yield from iter_passages_from_doc(
            doc,
            chunk_size=chunk_size,
            stride=stride,
            min_chunk_size=min_chunk_size
        )


# ---------------------------
# Streaming writers
# ---------------------------
def write_passages_jsonl_stream(
    passages: Iterable[Passage],
    output_path: str,
    batch_size: int = 10_000,
    show_progress: bool = True
) -> int:
    """
    Stream passages to a JSONL file. Buffers only a small batch in memory.
    Returns the total number of passages written.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    buffer: List[Passage] = []
    total = 0

    with open(out, "w", encoding="utf-8") as f:
        iterator = passages
        iterator = tqdm(iterator, desc="Writing passages") if show_progress else iterator

        for p in iterator:
            buffer.append(p)
            if len(buffer) >= batch_size:
                for bp in buffer:
                    f.write(json.dumps(bp.to_dict()) + "\n")
                total += len(buffer)
                buffer.clear()

        # Flush remaining
        if buffer:
            for bp in buffer:
                f.write(json.dumps(bp.to_dict()) + "\n")
            total += len(buffer)

    logger.info(f"Wrote {total} passages to {output_path}")
    return total


# ---------------------------
# Minimal example usage
# ---------------------------
if __name__ == "__main__":
    """
    Example: fully streaming pipeline
      1) stream docs from WikiExtractor shards
      2) stream-chunk each doc to passages
      3) stream-write passages to JSONL
    Nothing loads the full corpus or all passages into memory.
    """

    DUMP_DIR = "../../data/raw"  # path to WikiExtractor output root (folders like AA/wiki_00, AB/wiki_00, ...)
    OUT_PATH = "../../data/processed/passages.jsonl"

    # (A) Stream documents
    doc_stream = iter_docs_from_wikiextractor(
        DUMP_DIR,
        max_docs=None,          # or set a cap for quick tests
        show_progress=True
    )

    # (B) Stream passages (tune chunk params as needed)
    passage_stream = iter_passages_from_docs(
        doc_stream,
        chunk_size=256,
        stride=128,
        min_chunk_size=50
    )

    # (C) Stream-write to disk
    total = write_passages_jsonl_stream(
        passage_stream,
        output_path=OUT_PATH,
        batch_size=10_000,
        show_progress=True
    )

    print(f"Done. Total passages written: {total}")