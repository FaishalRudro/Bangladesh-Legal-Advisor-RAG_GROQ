"""
rag_pipeline.py  —  HuggingFace Spaces Version
================================================
Based on: bangladesh_legal_rag.py FIXED VERSION (1st code / BGE-M3 model)

Changes from Kaggle version for HuggingFace Spaces:
=====================================================
HF-1  PATH CHANGES
      /kaggle/working  →  /tmp
      /kaggle/input/.. →  data/bangladesh_laws.json

HF-2  GROQ API KEY
      Kaggle UserSecretsClient removed.
      Uses os.environ["GROQ_API_KEY"] only (set in HF Space Secrets).

HF-3  _load_cache() FIXED
      Patches sys.modules['__main__'] before pickle.load() so that classes
      stored as '__main__.LawChunk' etc. resolve correctly when uvicorn is
      the __main__ process.  Always restores __main__ in a finally block.

HF-4  QueryExpander CACHE PATH
      /kaggle/working/expansion_cache.json  →  /tmp/expansion_cache.json

HF-5  chat() stream mode
      Removed print() calls inside stream loop (Gradio handles output directly).

All FIX-1..8 from the original fixed version are preserved as-is.
Model and retrieval approach identical to 1st code (BAAI/bge-m3, dim=1024).
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import gc
import json
import time
import pickle
import hashlib
import warnings
import logging
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from groq import Groq
from rank_bm25 import BM25Okapi

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── Dataset ──
    # HF-1: path changed from /kaggle/input/... to data/ (local repo folder)
    dataset_path: str = "data/bangladesh_laws.json"

    # ── Embedding model (strongest multilingual, handles BN/EN/mixed) ──
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024

    # ── Reranker: multilingual mMiniLM (100+ languages including Bangla) ──
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    use_reranker: bool = True

    # ── Chunking: section-preserving ──
    section_max_size: int = 1200
    chunk_overlap: int = 150
    min_chunk_size: int = 60

    # ── Retrieval ──
    top_k_retrieve: int = 20
    top_k_bm25: int = 20
    top_k_fused: int = 30
    top_k_rerank: int = 6
    rrf_k: int = 60
    bm25_weight: float = 0.5
    dense_weight: float = 0.5
    neighbour_window: int = 1

    # ── Repeal chain: how many chunks to inject from replacement law ──
    repeal_chain_inject_k: int = 4

    # ── Context length per source in prompt ──
    context_chars_per_source: int = 1500

    # ── FAISS ──
    faiss_nlist: int = 64
    faiss_nprobe: int = 16
    # HF-1: cache path changed from /kaggle/working to /tmp
    index_cache_path: str = "/tmp/rag_index.pkl"

    # ── Generation ──
    groq_model: str = "llama-3.3-70b-versatile"
    groq_max_tokens: int = 2048
    groq_temperature: float = 0.1

    # ── Embedding batch size ──
    embed_batch_size: int = 32

    # HF-1: mmap path changed from /kaggle/working to /tmp
    embed_mmap_path: str = "/tmp/_embeddings_tmp.npy"

    # ── Reranker candidate pool multiplier ──
    rerank_pool_multiplier: int = 5


CONFIG = Config()

# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

class RepealStatus(str, Enum):
    ACTIVE   = "ACTIVE"
    REPEALED = "REPEALED"
    REPLACED = "REPLACED"
    AMENDED  = "AMENDED"
    UNKNOWN  = "UNKNOWN"


@dataclass
class LawChunk:
    chunk_id: int
    law_idx: int
    law_title: str
    law_year: str
    law_link: str
    section_title: str
    text: str
    char_start: int
    repeal_status: RepealStatus = RepealStatus.UNKNOWN
    repealed_by: str = ""
    replaces: str = ""
    repeal_note: str = ""
    amendment_notes: list = field(default_factory=list)
    chunk_seq: int = 0
    # FIX-2: Added section_number field — extracted from pipe-section header
    section_number: str = ""

    @property
    def is_repealed(self) -> bool:
        return self.repeal_status in (RepealStatus.REPEALED, RepealStatus.REPLACED)


@dataclass
class RetrievedChunk:
    chunk: LawChunk
    score: float
    match_type: str = "hit"
    rerank_score: float = 0.0
    injected_via_repeal_chain: bool = False
    chain_depth: int = 0


@dataclass
class SearchResult:
    query: str
    lang: str
    chunks: list
    latency_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# Title Normalizer
# ─────────────────────────────────────────────────────────────────────────────

class TitleNormalizer:
    _BN_DIGIT = r'[\u09E6-\u09EF]'
    _OCR_SPACE_RE = re.compile(
        r'([\u09E6-\u09EF\d])\s{1,2}([\u09E6-\u09EF\d])',
        re.UNICODE
    )
    _ACT_NUM_SUFFIX_RE = re.compile(
        r'\s*\(\s*(?:'
        r'\d{4}\s*(?:সনের|সালের)?\s*\d+\s*(?:নং|নম্বর)\s*(?:আইন|অধ্যাদেশ|বিধি)'
        r'|Act\s+No\.?\s*\d+\s+of\s+\d{4}'
        r'|Ordinance\s+No\.?\s*\w+\s+OF\s+\d{4}'
        r'|[\u09E6-\u09EF\d]+\s+(?:নং|নম্বর)\s+(?:আইন|অধ্যাদেশ)'
        r')[^)]*\)',
        re.IGNORECASE | re.UNICODE
    )
    _NOISE_RE = re.compile(r'[,;।\s]+$')

    @classmethod
    def normalize(cls, title: str) -> str:
        t = title.strip()
        for _ in range(4):
            t_new = cls._OCR_SPACE_RE.sub(r'\1\2', t)
            if t_new == t:
                break
            t = t_new
        t = cls._ACT_NUM_SUFFIX_RE.sub(' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        t = cls._NOISE_RE.sub('', t)
        return t.lower()

    @classmethod
    def build_index(cls, law_titles: List[str]) -> Dict[str, Tuple[int, str]]:
        index = {}
        for i, title in enumerate(law_titles):
            norm = cls.normalize(title)
            if norm not in index:
                index[norm] = (i, title)
        return index

    @classmethod
    def find_best_match(
        cls,
        query_title: str,
        norm_index: Dict[str, Tuple[int, str]],
        min_token_overlap: float = 0.45,
    ) -> Optional[Tuple[int, str]]:
        norm_q = cls.normalize(query_title)
        if not norm_q:
            return None
        if norm_q in norm_index:
            return norm_index[norm_q]
        prefix = norm_q[:25]
        for k, v in norm_index.items():
            if k.startswith(prefix) or prefix in k:
                return v
        q_tokens = set(norm_q.split())
        if len(q_tokens) < 2:
            return None
        best_score, best_match = 0.0, None
        for k, v in norm_index.items():
            k_tokens = set(k.split())
            overlap = len(q_tokens & k_tokens)
            jaccard = overlap / max(len(q_tokens | k_tokens), 1)
            if jaccard > best_score:
                best_score, best_match = jaccard, v
        if best_score >= min_token_overlap:
            return best_match
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Repeal Chain Detector
# ─────────────────────────────────────────────────────────────────────────────

class RepealChainDetector:
    _BRACKET_RE = re.compile(
        r'\[REPEAL(?:ED)?[:\s]*(.+?)\]',
        re.IGNORECASE | re.DOTALL
    )
    _BN_REPEALED_BY_RE = re.compile(
        r'(?:এই\s+(?:আইন|আইনটি|অধ্যাদেশ|বিধিমালা|আইন\s+এবং\s+বিধি)\s+)'
        r'(.+?)'
        r'\s+দ্বারা\s+রহিত',
        re.UNICODE
    )
    _BN_PASSIVE_REPEAL_RE = re.compile(
        r'রহিত\s*(?:কর(?:া|িয়া|ে)\s*হ(?:য়েছে|ইয়াছে|ইয়া|য়)|হ(?:য়েছে|ইয়াছে))',
        re.UNICODE
    )
    _BN_RAHITOKROME_RE = re.compile(
        r'([^|]{5,150}?)\s+রহিতক্রমে',
        re.UNICODE
    )
    _EN_REPEALED_BY_RE = re.compile(
        r'Repealed?\s+by\s+(?:section\s+[\d\w]+\s+(?:and\s+\w+\s+)?of\s+(?:the\s+)?)?'
        r'(.+?)(?:\s*\(|\s*\]|\s*\.)',
        re.IGNORECASE
    )
    _EN_PROSE_REPEAL_RE = re.compile(
        r'(?:this\s+(?:act|ordinance|law)\s+)'
        r'(?:is|was|has\s+been)\s+repealed?',
        re.IGNORECASE
    )
    _AMENDMENT_RE = re.compile(
        r'\[(?:Amended?|Amendment|সংশোধিত)[^\]]{0,200}\]',
        re.IGNORECASE | re.UNICODE
    )

    @classmethod
    def analyze(cls, law_full_text: str) -> dict:
        result = {
            "repeal_status": RepealStatus.UNKNOWN,
            "repealed_by": "",
            "replaces": "",
            "repeal_note": "",
            "amendment_notes": [],
        }
        if not law_full_text:
            return result
        preamble = law_full_text[:800]
        bracket_m = cls._BRACKET_RE.search(preamble)
        if bracket_m:
            annotation = bracket_m.group(0)
            inner = bracket_m.group(1).strip()
            result["repeal_note"] = annotation[:400]
            result["repeal_status"] = RepealStatus.REPEALED
            bn_m = cls._BN_REPEALED_BY_RE.search(inner)
            if bn_m:
                replacing = bn_m.group(1).strip()
                if len(replacing) > 3:
                    result["repealed_by"] = replacing[:250]
                    result["repeal_status"] = RepealStatus.REPLACED
            else:
                en_m = cls._EN_REPEALED_BY_RE.search(inner)
                if en_m:
                    replacing = en_m.group(1).strip()
                    if len(replacing) > 3:
                        result["repealed_by"] = replacing[:250]
                        result["repeal_status"] = RepealStatus.REPLACED
            return result
        if cls._BN_PASSIVE_REPEAL_RE.search(preamble):
            result["repeal_status"] = RepealStatus.REPEALED
            bn_m = cls._BN_REPEALED_BY_RE.search(preamble)
            if bn_m:
                replacing = bn_m.group(1).strip()
                if len(replacing) > 3:
                    result["repealed_by"] = replacing[:250]
                    result["repeal_status"] = RepealStatus.REPLACED
        if cls._EN_PROSE_REPEAL_RE.search(preamble):
            if result["repeal_status"] == RepealStatus.UNKNOWN:
                result["repeal_status"] = RepealStatus.REPEALED
        rahit_m = cls._BN_RAHITOKROME_RE.search(preamble)
        if rahit_m:
            candidate = rahit_m.group(1).strip()
            candidate = re.sub(r'[|[\]]', '', candidate).strip()
            segments = [s.strip() for s in candidate.split(',') if s.strip()]
            if segments:
                for seg in reversed(segments):
                    if len(seg) > 8 and any(c.isalpha() for c in seg):
                        result["replaces"] = seg[:200]
                        break
        amendments = cls._AMENDMENT_RE.findall(law_full_text[:5000])
        if amendments:
            result["amendment_notes"] = amendments[:10]
            if result["repeal_status"] == RepealStatus.UNKNOWN:
                result["repeal_status"] = RepealStatus.AMENDED
        if result["repeal_status"] == RepealStatus.UNKNOWN:
            result["repeal_status"] = RepealStatus.ACTIVE
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Repeal Chain Linker
# ─────────────────────────────────────────────────────────────────────────────

class RepealChainLinker:
    def __init__(self):
        self._title_to_chunks: Dict[str, List[LawChunk]] = {}
        self._norm_index: Dict[str, Tuple[int, str]] = {}
        self._repeal_chain: Dict[str, str] = {}

    def build(self, chunks: List[LawChunk]):
        for c in chunks:
            key = c.law_title.strip()
            if key not in self._title_to_chunks:
                self._title_to_chunks[key] = []
            self._title_to_chunks[key].append(c)
        all_titles = list(self._title_to_chunks.keys())
        self._norm_index = TitleNormalizer.build_index(all_titles)
        seen_titles = set()
        for c in chunks:
            if c.law_title in seen_titles:
                continue
            seen_titles.add(c.law_title)
            if c.is_repealed and c.repealed_by:
                norm_old = TitleNormalizer.normalize(c.law_title)
                match = TitleNormalizer.find_best_match(c.repealed_by, self._norm_index)
                if match:
                    norm_new = TitleNormalizer.normalize(match[1])
                    self._repeal_chain[norm_old] = norm_new
            if c.replaces:
                match = TitleNormalizer.find_best_match(c.replaces, self._norm_index)
                if match:
                    norm_old_replaced = TitleNormalizer.normalize(match[1])
                    norm_this = TitleNormalizer.normalize(c.law_title)
                    if norm_old_replaced not in self._repeal_chain:
                        self._repeal_chain[norm_old_replaced] = norm_this
        print(f"  ✅ Repeal chain linker built. "
              f"{len(self._repeal_chain)} chain links, "
              f"{len(self._title_to_chunks)} unique laws indexed.")

    def get_chunks_for_title(self, title: str, top_k: int = 4) -> List[LawChunk]:
        if title in self._title_to_chunks:
            return self._title_to_chunks[title][:top_k]
        match = TitleNormalizer.find_best_match(title, self._norm_index)
        if match:
            orig_title = match[1]
            return self._title_to_chunks.get(orig_title, [])[:top_k]
        return []

    def get_current_law(self, chunk: LawChunk) -> Optional[Tuple[str, List[LawChunk]]]:
        if not chunk.is_repealed:
            return None
        visited = set()
        norm_current = TitleNormalizer.normalize(chunk.law_title)
        for _ in range(8):
            if norm_current in visited:
                break
            visited.add(norm_current)
            norm_next = self._repeal_chain.get(norm_current)
            if not norm_next:
                break
            next_entry = self._norm_index.get(norm_next)
            if not next_entry:
                break
            next_title = next_entry[1]
            next_chunks = self._title_to_chunks.get(next_title, [])
            if not next_chunks:
                break
            sample_chunk = next_chunks[0]
            if not sample_chunk.is_repealed:
                return (next_title, next_chunks)
            norm_current = norm_next
        return None

    def get_replacement_chunks(self, chunk: LawChunk, top_k: int = 4) -> List[Tuple[LawChunk, int]]:
        result = []
        if chunk.repealed_by:
            direct_chunks = self.get_chunks_for_title(chunk.repealed_by, top_k)
            for c in direct_chunks:
                result.append((c, 1))
        current = self.get_current_law(chunk)
        if current:
            current_title, current_chunks = current
            direct_title = chunk.repealed_by
            if direct_title and TitleNormalizer.normalize(current_title) != \
               TitleNormalizer.normalize(direct_title):
                for c in current_chunks[:top_k]:
                    result.append((c, 2))
        seen = set()
        deduped = []
        for c, depth in result:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                deduped.append((c, depth))
        return deduped[:top_k * 2]


# ─────────────────────────────────────────────────────────────────────────────
# Advanced Chunker
# ─────────────────────────────────────────────────────────────────────────────

class LawChunker:
    """
    Section-preserving chunker.
    FIX-3: Extracts section_number from the pipe-section header
    and stores it in every LawChunk.
    """

    def __init__(self, config: Config):
        self.cfg = config

    def chunk_law(self, law_idx: int, raw: dict, global_chunk_id_start: int) -> List[LawChunk]:
        text = raw.get("law_full_text", "").strip()
        year = str(raw.get("year", ""))
        link = raw.get("link", "")
        if not text:
            return []
        pipe_parts = [p.strip() for p in text.split(" | ") if p.strip()]
        if not pipe_parts:
            return []
        law_title = pipe_parts[0]
        repeal_info = RepealChainDetector.analyze(text)
        chunks = []
        cid = global_chunk_id_start
        seq = 0
        for part_idx, part in enumerate(pipe_parts):
            if len(part) < self.cfg.min_chunk_size:
                continue
            section_title = self._extract_section_title(part)
            # FIX-3: extract section number from the header of this pipe segment
            section_number = self._extract_section_number(part)
            sub_chunks = self._split_section(part)
            for sc in sub_chunks:
                if len(sc.strip()) < self.cfg.min_chunk_size:
                    continue
                chunk = LawChunk(
                    chunk_id=cid,
                    law_idx=law_idx,
                    law_title=law_title,
                    law_year=year,
                    law_link=link,
                    section_title=section_title,
                    text=sc.strip(),
                    char_start=text.find(sc[:40]) if sc else 0,
                    repeal_status=repeal_info["repeal_status"],
                    repealed_by=repeal_info["repealed_by"],
                    replaces=repeal_info["replaces"],
                    repeal_note=repeal_info["repeal_note"],
                    amendment_notes=repeal_info["amendment_notes"],
                    chunk_seq=seq,
                    section_number=section_number,  # FIX-3
                )
                chunks.append(chunk)
                cid += 1
                seq += 1
        if not chunks and law_title:
            chunks.append(LawChunk(
                chunk_id=global_chunk_id_start,
                law_idx=law_idx,
                law_title=law_title,
                law_year=year,
                law_link=link,
                section_title="(title only)",
                text=law_title,
                char_start=0,
                repeal_status=repeal_info["repeal_status"],
                repealed_by=repeal_info["repealed_by"],
                replaces=repeal_info["replaces"],
                repeal_note=repeal_info["repeal_note"],
                amendment_notes=repeal_info["amendment_notes"],
                chunk_seq=0,
                section_number="",
            ))
        return chunks

    # FIX-3: Extracts section number from pipe-section header.
    @staticmethod
    def _extract_section_number(pipe_text: str) -> str:
        # EN: 'Title: NUM[A-Z]. text' or 'Title: NUM( text'
        m = re.match(
            r'^[^|\n]{3,120}:\s*(\d+[A-Za-z]?)\s*[\.\(]',
            pipe_text
        )
        if m:
            return m.group(1)
        # BN: 'Title: BNNUM৷ text' or 'Title: BNNUM। text' or 'Title: BNNUM (1)'
        m = re.match(
            r'^[^|\n]{3,120}:\s*([\u09E6-\u09EF]+[\u0995-\u09B9]?)\s*'
            r'[\u09F7\u09F8\u0964\u0965\(\[]',
            pipe_text
        )
        if m:
            return m.group(1).rstrip()
        # Direct-number format: '376. text' or '৪৬৷ text'
        m = re.match(r'^(\d+[A-Za-z]?)\s*[\.\(]', pipe_text)
        if m:
            return m.group(1)
        m = re.match(
            r'^([\u09E6-\u09EF]+[\u0995-\u09B9]?)\s*'
            r'[\u09F7\u09F8\u0964\u0965\(\[]',
            pipe_text
        )
        if m:
            return m.group(1)
        return ""

    def _extract_section_title(self, text: str) -> str:
        m = re.match(r'^([^\n:।]{3,70})(?::\s*\d|।)', text)
        if m:
            return m.group(1).strip()
        return text[:60].split('\n')[0].strip()

    def _split_section(self, text: str) -> List[str]:
        if len(text) <= self.cfg.section_max_size:
            return [text]
        chunks = []
        start = 0
        size = self.cfg.section_max_size
        overlap = self.cfg.chunk_overlap
        while start < len(text):
            end = start + size
            window = text[start:end]
            if end < len(text):
                for sep in ['।\n', '\n', '। ', '. ', ' ']:
                    last_sep = window.rfind(sep)
                    if last_sep > size * 0.6:
                        window = text[start:start + last_sep + len(sep)]
                        end = start + last_sep + len(sep)
                        break
            chunks.append(window.strip())
            start = end - overlap
            if start >= len(text):
                break
        return [c for c in chunks if c]


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Model
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingModel:
    def __init__(self, model_name: str):
        print(f"  Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"  Embedding model loaded on {self.device}")

    def _mean_pool(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * input_mask).sum(1) / input_mask.sum(1).clamp(min=1e-9)

    def encode(self, texts: list, batch_size: int = 32, normalize: bool = True,
               mmap_path: str = "/tmp/_embeddings_tmp.npy") -> np.ndarray:
        total = len(texts)
        total_batches = (total + batch_size - 1) // batch_size
        # HF-1: dim=1024 for BAAI/bge-m3
        fp = np.memmap(mmap_path, dtype="float32", mode="w+", shape=(total, 1024))
        for batch_idx, i in enumerate(range(0, total, batch_size)):
            batch = texts[i: i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                output = self.model(**encoded)
            embeds = self._mean_pool(output, encoded["attention_mask"])
            embeds = embeds.cpu().numpy().astype("float32")
            if normalize:
                norms = np.linalg.norm(embeds, axis=1, keepdims=True)
                embeds = embeds / np.maximum(norms, 1e-9)
            fp[i: i + len(embeds)] = embeds
            fp.flush()
            del encoded, output, embeds
            if self.device == "cuda":
                torch.cuda.empty_cache()
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_batches:
                done = min(i + batch_size, total)
                print(f"    Embedding: {done}/{total} ({done/total*100:.1f}%) — "
                      f"batch {batch_idx+1}/{total_batches}")
                gc.collect()
        result = np.array(fp)
        del fp
        gc.collect()
        return result

    def encode_one(self, text: str) -> np.ndarray:
        encoded = self.tokenizer(
            [text], padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            output = self.model(**encoded)
        embeds = self._mean_pool(output, encoded["attention_mask"])
        embeds = embeds.cpu().numpy().astype("float32")
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        embeds = embeds / np.maximum(norms, 1e-9)
        return embeds[0]


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Encoder Reranker
# ─────────────────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    def __init__(self, model_name: str):
        print(f"  Loading reranker: {model_name}")
        from transformers import AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"  Reranker loaded on {self.device}")

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        if not chunks:
            return []
        pairs = []
        for rc in chunks:
            c = rc.chunk
            passage = f"{c.law_title} | {c.section_title} | {c.text}"
            pairs.append((query, passage[:512]))
        scores = []
        batch_size = 32
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i: i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**encoded).logits
            scores.extend(logits.squeeze(-1).cpu().numpy().tolist())
        for rc, sc in zip(chunks, scores):
            rc.rerank_score = float(sc)
        chunks.sort(key=lambda x: x.rerank_score, reverse=True)
        return chunks[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# FAISS Index
# ─────────────────────────────────────────────────────────────────────────────

class VectorIndex:
    def __init__(self, dim: int, nlist: int = 64, nprobe: int = 16):
        quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = nprobe
        self._trained = False

    def build(self, embeddings: np.ndarray):
        if not self._trained:
            print(f"  Training FAISS IVF index on {len(embeddings)} vectors...")
            self.index.train(embeddings)
            self._trained = True
        self.index.add(embeddings)
        print(f"  Index built. Total vectors: {self.index.ntotal}")

    def search(self, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        q = query_vec.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(q, top_k)
        return scores[0], indices[0]


# ─────────────────────────────────────────────────────────────────────────────
# BM25 Index
# ─────────────────────────────────────────────────────────────────────────────

class BM25Index:
    _PUNCT_RE = re.compile(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~।॥।]')

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_ids: List[int] = []
        self._tokenized_corpus: List[List[str]] = []

    def build(self, chunks: List[LawChunk]):
        print(f"  Building BM25 index over {len(chunks)} chunks...")
        tokenized = []
        self._corpus_ids = []
        for c in chunks:
            # FIX-4: section_number included for better keyword matching
            text = f"{c.law_title} {c.section_title} {c.section_number} {c.text}"
            tokens = self._tokenise(text)
            tokenized.append(tokens)
            self._corpus_ids.append(c.chunk_id)
        self._tokenized_corpus = tokenized
        self._bm25 = BM25Okapi(tokenized)
        print(f"  BM25 ready. Corpus: {self._bm25.corpus_size}, Vocab: {len(self._bm25.idf)}")

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self._bm25 is None:
            return []
        tokens = self._tokenise(query)
        scores = self._bm25.get_scores(tokens)
        ranked = np.argsort(scores)[::-1][:top_k * 2]
        results = [
            (int(pos), float(scores[pos]))
            for pos in ranked if scores[pos] > 0
        ]
        return results[:top_k]

    @classmethod
    def _tokenise(cls, text: str) -> List[str]:
        lowered = text.lower()
        cleaned = cls._PUNCT_RE.sub(" ", lowered)
        tokens = [t for t in cleaned.split() if len(t) > 1]
        return tokens if tokens else ["<empty>"]

    def serialise(self) -> dict:
        return {
            "corpus_ids": self._corpus_ids,
            "tokenized_corpus": self._tokenized_corpus,
            "k1": self._bm25.k1,
            "b": self._bm25.b,
            "epsilon": self._bm25.epsilon,
        }

    @classmethod
    def deserialise(cls, data: dict) -> "BM25Index":
        obj = cls()
        obj._corpus_ids = data["corpus_ids"]
        obj._tokenized_corpus = data["tokenized_corpus"]
        print(f"  Reconstructing BM25 from {len(obj._tokenized_corpus)} docs...")
        obj._bm25 = BM25Okapi(
            obj._tokenized_corpus,
            k1=data.get("k1", 1.5),
            b=data.get("b", 0.75),
            epsilon=data.get("epsilon", 0.25),
        )
        print(f"  BM25 reconstructed. Vocab: {len(obj._bm25.idf)}")
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ─────────────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_hits: List[Tuple[int, float]],
    bm25_hits: List[Tuple[int, float]],
    all_chunks_by_id: Dict[int, LawChunk],
    rrf_k: int = 60,
    dense_weight: float = 0.5,
    bm25_weight: float = 0.5,
    top_k: int = 30,
) -> List[RetrievedChunk]:
    dense_rank = {cid: rank + 1 for rank, (cid, _) in enumerate(dense_hits)}
    bm25_rank  = {cid: rank + 1 for rank, (cid, _) in enumerate(bm25_hits)}
    dense_fallback = len(dense_hits) + 1
    bm25_fallback  = len(bm25_hits) + 1
    all_ids = set(dense_rank) | set(bm25_rank)
    fused = []
    for cid in all_ids:
        chunk = all_chunks_by_id.get(cid)
        if chunk is None:
            continue
        dr = dense_rank.get(cid, dense_fallback)
        br = bm25_rank.get(cid, bm25_fallback)
        rrf_score = (
            dense_weight * (1.0 / (rrf_k + dr)) +
            bm25_weight  * (1.0 / (rrf_k + br))
        )
        source = (
            "dense+bm25" if (cid in dense_rank and cid in bm25_rank)
            else "dense" if cid in dense_rank
            else "bm25"
        )
        fused.append(RetrievedChunk(chunk=chunk, score=rrf_score, match_type=source))
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Query Expander with Cross-Lingual Translation
# ─────────────────────────────────────────────────────────────────────────────

class QueryExpander:
    _EXPAND_MODEL = "llama-3.1-8b-instant"
    _EXPAND_MAX_TOKENS = 200
    _EXPAND_TEMPERATURE = 0.0
    _EXPAND_TIMEOUT_S = 5.0
    _CACHE_SIZE = 512
    # HF-4: cache path changed from /kaggle/working to /tmp
    _CACHE_PATH = "/tmp/expansion_cache.json"

    _EXPAND_SYSTEM = (
        "You are a Bangladesh legal search assistant. "
        "Your only task is to expand a user's query into BM25 search tokens. "
        "Output ONLY a single line of space-separated tokens — no explanation, "
        "no JSON, no bullet points. Include: synonyms, related Bangladesh act names "
        "(short form), relevant section numbers, formal legal terms, colloquial equivalents. "
        "Maximum 40 tokens total."
    )
    _EXPAND_USER = (
        "Query: {query}\n"
        "Expand into BM25 tokens (flat space-separated, ≤40 tokens):"
    )
    _TRANSLATE_SYSTEM = (
        "You are a precise translator specializing in Bangladesh legal text. "
        "Translate the given query to {target_lang}. "
        "Output ONLY the translated text — no explanation, no extra words."
    )
    _TRANSLATE_USER = "Translate to {target_lang}: {query}"

    def __init__(self, groq_client=None):
        self._client = groq_client
        self._cache: Dict[str, dict] = {}
        self._cache_order: List[str] = []
        self._lock = __import__("threading").Lock()
        self._load_cache()

    def set_client(self, groq_client):
        self._client = groq_client

    def expand(self, query: str) -> str:
        return self._get_cache_entry(query).get("expanded", query)

    def translate(self, query: str, source_lang: str) -> str:
        return self._get_cache_entry(query, source_lang).get("translated", "")

    def _get_cache_entry(self, query: str, source_lang: str = None) -> dict:
        if not query or not query.strip():
            return {"expanded": query, "translated": ""}
        cache_key = self._normalise(query)
        with self._lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            needs_translation = (
                source_lang is not None
                and not cached.get("translated", "")
                and self._client is not None
            )
            if not needs_translation:
                return cached
            target_lang = "English" if source_lang == "bn" else "Bengali"
            translation = self._call_translate(query, target_lang)
            if translation:
                with self._lock:
                    self._cache[cache_key]["translated"] = translation
                self._persist_cache()
                return self._cache[cache_key]
            return cached
        entry = {"expanded": query, "translated": ""}
        if self._client is not None:
            expansion = self._call_expand(query)
            if expansion:
                entry["expanded"] = query + " " + expansion
            if source_lang is not None:
                target_lang = "English" if source_lang == "bn" else "Bengali"
                translation = self._call_translate(query, target_lang)
                if translation:
                    entry["translated"] = translation
        with self._lock:
            self._cache[cache_key] = entry
            self._cache_order.append(cache_key)
            if len(self._cache_order) > self._CACHE_SIZE:
                evict = self._cache_order.pop(0)
                self._cache.pop(evict, None)
        self._persist_cache()
        return entry

    def _call_expand(self, query: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._EXPAND_MODEL,
                messages=[
                    {"role": "system", "content": self._EXPAND_SYSTEM},
                    {"role": "user",   "content": self._EXPAND_USER.format(query=query)},
                ],
                max_tokens=self._EXPAND_MAX_TOKENS,
                temperature=self._EXPAND_TEMPERATURE,
                timeout=self._EXPAND_TIMEOUT_S,
            )
            raw = response.choices[0].message.content.strip()
            return self._sanitise_tokens(raw)
        except Exception:
            return ""

    def _call_translate(self, query: str, target_lang: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._EXPAND_MODEL,
                messages=[
                    {"role": "system", "content": self._TRANSLATE_SYSTEM.format(target_lang=target_lang)},
                    {"role": "user",   "content": self._TRANSLATE_USER.format(target_lang=target_lang, query=query)},
                ],
                max_tokens=150,
                temperature=0.0,
                timeout=self._EXPAND_TIMEOUT_S,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return ""

    @staticmethod
    def _sanitise_tokens(raw: str) -> str:
        cleaned = re.sub(r'[^\w\u0980-\u09FF\s]', ' ', raw, flags=re.UNICODE)
        tokens = cleaned.split()
        seen = set()
        unique = []
        for t in tokens:
            t_low = t.lower()
            if t_low not in seen and len(t) > 1:
                seen.add(t_low)
                unique.append(t)
        return " ".join(unique[:60])

    @staticmethod
    def _normalise(query: str) -> str:
        return re.sub(r'\s+', ' ', query.lower().strip())

    def _load_cache(self):
        try:
            if Path(self._CACHE_PATH).exists():
                with open(self._CACHE_PATH, encoding="utf-8") as f:
                    data = json.load(f)
                raw_cache = data.get("cache", {})
                for k, v in raw_cache.items():
                    if isinstance(v, str):
                        self._cache[k] = {"expanded": v, "translated": ""}
                    else:
                        self._cache[k] = v
                self._cache_order = list(self._cache.keys())
                print(f"  QueryExpander: loaded {len(self._cache)} cached expansions.")
        except Exception:
            self._cache = {}
            self._cache_order = []

    def _persist_cache(self):
        def _write():
            try:
                with self._lock:
                    snapshot = dict(self._cache)
                Path(self._CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
                with open(self._CACHE_PATH, "w", encoding="utf-8") as f:
                    json.dump({"cache": snapshot}, f, ensure_ascii=False)
            except Exception:
                pass
        import threading
        threading.Thread(target=_write, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Language Detector
# ─────────────────────────────────────────────────────────────────────────────

class LanguageDetector:
    @staticmethod
    def detect(text: str) -> str:
        bangla = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
        latin  = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        if bangla > 3:
            return "bn"
        if latin > bangla:
            return "en"
        try:
            from langdetect import detect as ld_detect
            lang = ld_detect(text)
            return "bn" if lang in ("bn", "bd") else "en"
        except Exception:
            return "en"


# ─────────────────────────────────────────────────────────────────────────────
# Citation Extractor
# ─────────────────────────────────────────────────────────────────────────────

class CitationExtractor:
    """
    FIX-6: Uses chunk.section_number (always accurate, extracted at index-build time)
    instead of scanning chunk text for keywords that never appeared in section headers.
    """

    _EN_XREF_RE = re.compile(
        r'(?:section|sec\.|s\.)\s*(\d+[A-Za-z]?)(?:\s*\((\d+)\))?',
        re.IGNORECASE
    )
    _BN_XREF_RE = re.compile(
        r'(?:ধারা|অনুচ্ছেদ|বিভাগ|উপ-ধারা)\s*([\d\u09E6-\u09EF]+[ক-হ]?)'
        r'(?:\s*\(([\d\u09E6-\u09EF]+)\))?',
        re.UNICODE
    )

    @classmethod
    def extract(cls, retrieved_chunks: List[RetrievedChunk]) -> List[dict]:
        citations = []
        for i, rc in enumerate(retrieved_chunks):
            c = rc.chunk
            # FIX-6: primary section number from chunk (never empty when extractable)
            primary_sec = c.section_number

            # Cross-references found inside the body text (informational)
            xrefs_en = cls._EN_XREF_RE.findall(c.text)
            xrefs_bn = cls._BN_XREF_RE.findall(c.text)
            cross_refs = []
            for s, sub in xrefs_en[:4]:
                ref = f"Section {s}" + (f"({sub})" if sub else "")
                cross_refs.append(ref)
            for s, sub in xrefs_bn[:4]:
                ref = f"ধারা {s}" + (f"({sub})" if sub else "")
                cross_refs.append(ref)

            citations.append({
                "source_num": i + 1,
                "law_title": c.law_title,
                "law_year": c.law_year,
                "law_link": c.law_link,
                "section_title": c.section_title,
                "section_number": primary_sec,
                "section_refs": cross_refs,
                "repeal_status": c.repeal_status.value,
                "is_repealed": c.is_repealed,
                "repealed_by": c.repealed_by,
                "replaces": c.replaces,
                "repeal_note": c.repeal_note,
                "text": c.text,
                "match_type": rc.match_type,
                "injected_via_repeal_chain": rc.injected_via_repeal_chain,
                "chain_depth": rc.chain_depth,
            })
        return citations


# ─────────────────────────────────────────────────────────────────────────────
# Conversational Query Detector  (LLM-based)
# ─────────────────────────────────────────────────────────────────────────────

# FIX-7: Use the Groq LLM (llama-3.1-8b-instant) to detect greetings,
# farewells, identity questions, and small-talk that do not require law
# retrieval — matching the approach used in rag.py's Generator system prompt
# OFF-TOPIC DETECTION block.
#
# The groq_client is wired lazily via set_conversational_client(), called
# from BangladeshLegalRAG.__init__() and set_groq_key() once the
# GroqGenerator is initialised.

_conversational_groq_client = None
_CONV_DETECT_MODEL   = "llama-3.1-8b-instant"
_CONV_DETECT_TIMEOUT = 4.0

_CONV_DETECT_SYSTEM = (
    "You are a classifier. Your ONLY job is to decide whether a message is:\n"
    "  A) conversational/small-talk (greeting, farewell, identity question, thanks,\n"
    "     'how are you', 'what can you do', etc.) that does NOT require any legal\n"
    "     database lookup, OR\n"
    "  B) a legal or substantive question that DOES require a legal database lookup.\n\n"
    "Output EXACTLY one token: YES if it is conversational/small-talk (case A),\n"
    "or NO if it is a legal/substantive question (case B).\n"
    "No explanation. No punctuation. Just YES or NO."
)
_CONV_DETECT_USER = "Message: {query}\nIs this conversational/small-talk?"


def set_conversational_client(groq_client) -> None:
    """Called once the Groq client is available so the detector can use it."""
    global _conversational_groq_client
    _conversational_groq_client = groq_client


def is_conversational(query: str) -> bool:
    """
    Returns True if the query is a greeting, farewell, identity question, or
    small-talk phrase that does not require law retrieval.

    Uses the Groq LLM (llama-3.1-8b-instant) for detection — matching the
    approach in rag.py's Generator system prompt — instead of hardcoded regex.

    Falls back to False (treat as legal query) if the LLM call fails or times
    out, so retrieval is never accidentally skipped due to an API error.
    """
    q = query.strip()
    if not q:
        return True

    if _conversational_groq_client is None:
        # No client available yet — treat as legal query to be safe
        return False

    try:
        response = _conversational_groq_client.chat.completions.create(
            model=_CONV_DETECT_MODEL,
            messages=[
                {"role": "system", "content": _CONV_DETECT_SYSTEM},
                {"role": "user",   "content": _CONV_DETECT_USER.format(query=q)},
            ],
            max_tokens=5,
            temperature=0.0,
            timeout=_CONV_DETECT_TIMEOUT,
        )
        answer = response.choices[0].message.content.strip().upper()
        return answer.startswith("YES")
    except Exception:
        # On any failure (timeout, rate limit, etc.) fall back to treating
        # the query as a legal question so retrieval still runs.
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Builder
# ─────────────────────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    FIX-5: Context header now includes 'Section Number: X'.
    FIX-7: Added build_conversational() for greeting/small-talk queries.
    """

    SYSTEM_EN = """You are an expert Bangladesh Legal Advisor AI. You explain the law clearly, like a knowledgeable friend — not a document dumper.
STRICT RULES:
1. ANSWER LANGUAGE: You MUST respond in English. The user asked in English. Do not use any Bangla in your answer except for law titles or proper nouns that have no English equivalent.
2. NO REPETITION: Never say the same thing twice. Each sentence must add new information.
3. COMPLETENESS: If the source contains a specific penalty, duration, amount, or condition — state it verbatim. Never omit concrete legal details.
4. SECTION NUMBERS: Every source has a "Section Number" header. Use THAT exact number when citing sections. NEVER invent or guess section numbers from your training memory.
5. REPEAL CONTEXT — only when relevant: If the user asks about a law that has been replaced, mention this naturally: "That law was replaced by X in [year]. Under the current law, ..." Do NOT open every answer with a repeal warning banner.
6. NATURAL LANGUAGE: Summarize in plain English first, then give legal details. Cite [Source N] after each fact.
7. NEVER FABRICATE: Do not invent any information not present in the source documents. If something is unclear, say so.
8. REFERENCES: End with a clean **References** section as a numbered markdown list:
   1. [Law Title] (Year) — Section X — [bdlaws.minlaw.gov.bd](url)
   Do NOT list the same law multiple times. Always format URLs as clickable markdown links.
9. IF SOURCES ARE INSUFFICIENT: Say what you found and what's missing, then suggest checking bdlaws.minlaw.gov.bd."""

    SYSTEM_BN = """আপনি বাংলাদেশের একজন বিশেষজ্ঞ আইনি উপদেষ্টা AI। আপনি আইন পরিষ্কারভাবে ব্যাখ্যা করেন — একজন জ্ঞানী বন্ধুর মতো।
কঠোর নিয়মাবলী:
১. উত্তরের ভাষা: আপনাকে অবশ্যই বাংলায় উত্তর দিতে হবে। ইংরেজি আইনের নাম বা পরিভাষা ছাড়া কোনো ইংরেজি ব্যবহার করবেন না।
২. পুনরাবৃত্তি নেই: একই কথা দুইবার বলবেন না। প্রতিটি বাক্য নতুন তথ্য যোগ করবে।
৩. সম্পূর্ণতা: উৎসে নির্দিষ্ট শাস্তি, সময়কাল, পরিমাণ বা শর্ত থাকলে হুবহু বলুন। গুরুত্বপূর্ণ বিবরণ বাদ দেবেন না।
৪. ধারা নম্বর: প্রতিটি উৎসে "Section Number" হেডার আছে। সেই EXACT নম্বরটি ব্যবহার করুন। প্রশিক্ষণ স্মৃতি থেকে ধারা নম্বর অনুমান বা বানাবেন না।
৫. রহিতকরণ প্রসঙ্গ — শুধুমাত্র প্রাসঙ্গিক হলে: যদি ব্যবহারকারী কোনো রহিত আইন সম্পর্কে জিজ্ঞেস করেন, স্বাভাবিকভাবে উল্লেখ করুন। প্রতিটি উত্তর ব্যানার দিয়ে শুরু করবেন না।
৬. সহজ ভাষা: প্রথমে সহজ বাংলায় সারসংক্ষেপ, তারপর আইনি বিবরণ। প্রতিটি তথ্যের পর [উৎস N] উল্লেখ করুন।
৭. কিছু বানাবেন না: উৎসে নেই এমন কোনো তথ্য বানাবেন না।
৮. তথ্যসূত্র: শেষে **তথ্যসূত্র** বিভাগ দিন numbered markdown list হিসেবে:
   ১. [আইনের নাম] (সাল) — ধারা X — [bdlaws.minlaw.gov.bd](url)
   একই আইন একাধিকবার লিখবেন না। URL সবসময় clickable markdown link হিসেবে দিন।
৯. উৎস অপর্যাপ্ত হলে: কী পেয়েছেন তা বলুন এবং bdlaws.minlaw.gov.bd দেখতে বলুন।"""

    # FIX-7: System prompt for conversational queries — no law chunks injected
    SYSTEM_CONVERSATIONAL = """You are a friendly, helpful Bangladesh Legal Advisor AI named 'BD Legal AI'.
When users greet you or ask about you, respond naturally and warmly in the same language they used.
Keep your response brief (2-4 sentences). Introduce yourself as a Bangladesh legal advisor AI
and invite them to ask any legal question about Bangladesh law.
Do NOT fabricate any legal information. Do NOT cite any laws unless the user asks a legal question."""

    @classmethod
    def build_conversational(cls, query: str, lang: str) -> Tuple[str, str]:
        """FIX-7: Builds a prompt for non-legal conversational queries."""
        user_msg = (
            f"User message: {query}\n\n"
            f"Respond naturally in {'Bangla' if lang == 'bn' else 'English'}."
        )
        return cls.SYSTEM_CONVERSATIONAL, user_msg

    @classmethod
    def build(
        cls,
        query: str,
        citations: List[dict],
        lang: str,
        chat_history: List[dict] = None,
    ) -> Tuple[str, str]:
        system = cls.SYSTEM_BN if lang == "bn" else cls.SYSTEM_EN
        context_block = cls._format_context(citations, lang)
        history_block = cls._format_history(chat_history or [], lang)
        if lang == "bn":
            user_msg = (
                f"পূর্ববর্তী কথোপকথন:\n{history_block}\n\n"
                f"উৎস দলিলসমূহ:\n{'='*60}\n{context_block}\n{'='*60}\n\n"
                f"প্রশ্ন: {query}\n\n"
                f"নির্দেশনা: উপরের উৎস দলিল ব্যবহার করে বাংলায় উত্তর দিন। "
                f"প্রতিটি উৎসের 'Section Number' ফিল্ড থেকে ধারা নম্বর নিন — অনুমান করবেন না। "
                f"একই কথা দুইবার বলবেন না। প্রতিটি তথ্যের পর [উৎস N] উল্লেখ করুন।"
            )
        else:
            user_msg = (
                f"Previous conversation:\n{history_block}\n\n"
                f"Source Documents:\n{'='*60}\n{context_block}\n{'='*60}\n\n"
                f"Question: {query}\n\n"
                f"Instructions: Answer in English using the sources above. "
                f"Use the 'Section Number' field from each source for section citations — never guess. "
                f"Never repeat the same point. Cite [Source N] after each fact."
            )
        return system, user_msg

    @classmethod
    def _format_context(cls, citations: List[dict], lang: str) -> str:
        parts = []
        for c in citations:
            status = c["repeal_status"]
            chain_note = ""
            if c["injected_via_repeal_chain"]:
                depth = c.get("chain_depth", 1)
                if lang == "bn":
                    chain_note = " ⛓️ [বর্তমান আইন - স্বয়ংক্রিয়ভাবে যুক্ত]" if depth == 2 else " ⛓️ [সরাসরি প্রতিস্থাপন]"
                else:
                    chain_note = " ⛓️ [CURRENT LAW - auto-linked]" if depth == 2 else " ⛓️ [DIRECT REPLACEMENT]"

            if status == RepealStatus.REPLACED.value:
                by_name = c['repealed_by'][:80] if c['repealed_by'] else ('অজানা' if lang == 'bn' else 'unknown')
                status_line = (
                    f"⚠️ {'রহিত ও প্রতিস্থাপিত' if lang=='bn' else 'REPEALED & REPLACED'}"
                    f" → {by_name}{chain_note}"
                )
            elif status == RepealStatus.REPEALED.value:
                status_line = f"⚠️ {'রহিত আইন' if lang=='bn' else 'REPEALED'}{chain_note}"
            elif status == RepealStatus.AMENDED.value:
                status_line = f"📝 {'সংশোধিত আইন' if lang=='bn' else 'AMENDED LAW'}{chain_note}"
            elif status == RepealStatus.ACTIVE.value:
                replaces_note = ""
                if c["replaces"]:
                    replaces_note = (
                        f" (প্রতিস্থাপিত করেছে: {c['replaces'][:60]})"
                        if lang == "bn"
                        else f" (Replaced: {c['replaces'][:60]})"
                    )
                status_line = f"✅ {'বর্তমান আইন' if lang=='bn' else 'CURRENT LAW'}{replaces_note}{chain_note}"
            else:
                status_line = f"❓ {'অজানা অবস্থা' if lang=='bn' else 'STATUS UNKNOWN'}{chain_note}"

            # FIX-5: Show real section number prominently in header
            sec_num = c.get("section_number", "")
            sec_num_line = ""
            if sec_num:
                label = "ধারা নম্বর" if lang == "bn" else "Section Number"
                sec_num_line = f"{label}: {sec_num}\n"

            cross_refs = c.get("section_refs", [])
            cross_refs_str = " | ".join(cross_refs[:5]) if cross_refs else ""

            if lang == "bn":
                header = (
                    f"[উৎস {c['source_num']}]\n"
                    f"আইন: {c['law_title']}\n"
                    f"বছর: {c['law_year']}\n"
                    f"বিভাগ: {c['section_title']}\n"
                    f"{sec_num_line}"
                    f"অবস্থা: {status_line}\n"
                )
            else:
                header = (
                    f"[Source {c['source_num']}]\n"
                    f"Law: {c['law_title']}\n"
                    f"Year: {c['law_year']}\n"
                    f"Section: {c['section_title']}\n"
                    f"{sec_num_line}"
                    f"Status: {status_line}\n"
                )

            if cross_refs_str:
                label = "অভ্যন্তরীণ রেফারেন্স" if lang == "bn" else "Cross-references in text"
                header += f"{label}: {cross_refs_str}\n"

            link_label = "লিংক" if lang == "bn" else "Link"
            header += f"{link_label}: {c['law_link']}\n"

            if c["is_repealed"] and c["repeal_note"]:
                label = "রহিতকরণ বিবরণ" if lang == "bn" else "Repeal note"
                header += f"{label}: {c['repeal_note'][:250]}\n"

            content_label = "বিষয়বস্তু" if lang == "bn" else "Content"
            content_len = CONFIG.context_chars_per_source
            header += f"\n{content_label}:\n{c['text'][:content_len]}"
            parts.append(header)

        sep = "\n\n" + "─" * 50 + "\n\n"
        return sep.join(parts)

    @classmethod
    def _format_history(cls, history: List[dict], lang: str) -> str:
        if not history:
            return "কোনো পূর্ববর্তী কথোপকথন নেই।" if lang == "bn" else "No previous conversation."
        _SENTENCE_END_RE = re.compile(r'(?<=[।.!?])\s')
        _HISTORY_ANSWER_LIMIT = 400
        parts = []
        role_q = "ব্যবহারকারী" if lang == "bn" else "User"
        role_a = "সহকারী" if lang == "bn" else "Assistant"
        for turn in history[-4:]:
            answer = turn["answer"]
            if len(answer) > _HISTORY_ANSWER_LIMIT:
                window = answer[:_HISTORY_ANSWER_LIMIT]
                m = _SENTENCE_END_RE.search(window[::-1])
                if m:
                    cut = _HISTORY_ANSWER_LIMIT - m.start()
                    answer = answer[:cut].rstrip() + "…"
                else:
                    answer = window.rstrip() + "…"
            parts.append(f"{role_q}: {turn['query']}\n{role_a}: {answer}")
        return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Generator (Groq)
# ─────────────────────────────────────────────────────────────────────────────

class GroqGenerator:
    def __init__(self, api_key: str, model: str, max_tokens: int, temperature: float):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        print(f"  Groq generator: {model}")

    def generate(self, system: str, user_msg: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    def generate_stream(self, system: str, user_msg: str):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ─────────────────────────────────────────────────────────────────────────────
# Main RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class BangladeshLegalRAG:
    """
    End-to-end RAG pipeline for Bangladesh law advisory.
    HuggingFace Spaces version — all Kaggle-specific code removed.
    Model: BAAI/bge-m3 (dim=1024), same as 1st code.
    """

    def __init__(self, config: Config = None, groq_api_key: str = None):
        self.cfg = config or CONFIG
        self._chunks: List[LawChunk] = []
        self._chunks_by_id: Dict[int, LawChunk] = {}
        self._embeddings: Optional[np.ndarray] = None
        self._index: Optional[VectorIndex] = None
        self._bm25: Optional[BM25Index] = None
        self._repeal_linker: Optional[RepealChainLinker] = None
        self._embedder: Optional[EmbeddingModel] = None
        self._reranker: Optional[CrossEncoderReranker] = None
        self._generator: Optional[GroqGenerator] = None
        self._lang_detector = LanguageDetector()
        self._citation_extractor = CitationExtractor()
        self._prompt_builder = PromptBuilder()
        self._query_expander = QueryExpander(groq_client=None)
        self._chat_history: List[dict] = []
        self._neighbour_index: Dict[Tuple[int, int], LawChunk] = {}

        print("=" * 65)
        print("  Bangladesh Legal RAG — Loading models...")
        print("=" * 65)
        self._embedder = EmbeddingModel(self.cfg.embed_model)
        if self.cfg.use_reranker:
            self._reranker = CrossEncoderReranker(self.cfg.rerank_model)
        if groq_api_key:
            self._generator = GroqGenerator(
                api_key=groq_api_key,
                model=self.cfg.groq_model,
                max_tokens=self.cfg.groq_max_tokens,
                temperature=self.cfg.groq_temperature,
            )
            self._query_expander.set_client(self._generator.client)
            # Wire up LLM-based conversational detector with the same Groq client
            set_conversational_client(self._generator.client)
        else:
            print("  ⚠️  No Groq API key provided. Use .set_groq_key() before chatting.")
        print("  ✅ Models loaded.\n")

    def set_groq_key(self, api_key: str):
        self._generator = GroqGenerator(
            api_key=api_key,
            model=self.cfg.groq_model,
            max_tokens=self.cfg.groq_max_tokens,
            temperature=self.cfg.groq_temperature,
        )
        self._query_expander.set_client(self._generator.client)
        # Wire up LLM-based conversational detector with the new Groq client
        set_conversational_client(self._generator.client)

    # ──────────────────────── BUILD INDEX ────────────────────────────────────

    def build_index(self, dataset_path: str = None, force_rebuild: bool = False):
        path = dataset_path or self.cfg.dataset_path
        cache = self.cfg.index_cache_path
        print(f"🔑 Computing file hash for {path}...", flush=True)
        ds_hash = self._file_hash(path)
        print(f"🔑 Hash done: {ds_hash[:8]}...", flush=True)
        cache_meta_path = cache + ".meta"

        if not force_rebuild and Path(cache).exists() and Path(cache_meta_path).exists():
            try:
                with open(cache_meta_path) as f:
                    meta = json.load(f)
                if meta.get("hash") == ds_hash:
                    print("  Loading cached index...")
                    self._load_cache(cache)
                    self._build_repeal_linker()
                    self._build_neighbour_index()
                    unknown_ratio = sum(
                        1 for c in self._chunks if c.repeal_status == RepealStatus.UNKNOWN
                    ) / max(len(self._chunks), 1)
                    chain_links = len(self._repeal_linker._repeal_chain)
                    if unknown_ratio > 0.9 and chain_links == 0:
                        print("  ⚠️  Stale cache detected. Rebuilding...")
                        force_rebuild = True
                    else:
                        print(f"  ✅ Loaded {len(self._chunks)} chunks from cache.")
                        return
            except Exception as e:
                print(f"  Cache load failed ({e}), rebuilding...")

        print(f"  Building index from: {path}")
        with open(path, encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"  Dataset: {len(dataset)} laws")

        chunker = LawChunker(self.cfg)
        all_chunks = []
        cid = 0
        for idx, law in enumerate(dataset):
            chunks = chunker.chunk_law(idx, law, cid)
            all_chunks.extend(chunks)
            cid += len(chunks)
            if idx % 200 == 0:
                print(f"    Chunked {idx}/{len(dataset)} laws, {len(all_chunks)} chunks...")

        self._chunks = all_chunks
        self._chunks_by_id = {c.chunk_id: c for c in self._chunks}
        print(f"  Total chunks: {len(self._chunks)}")

        replaced_count = sum(1 for c in self._chunks if c.repeal_status == RepealStatus.REPLACED)
        repealed_count = sum(1 for c in self._chunks if c.is_repealed)
        print(f"  Laws with REPLACED status chunks: {replaced_count}")
        print(f"  Total repealed chunks: {repealed_count} ({repealed_count*100//max(len(self._chunks),1)}%)")

        print("  Computing embeddings (memmap)...")
        texts = [self._chunk_to_embed_text(c) for c in self._chunks]
        self._embeddings = self._embedder.encode(
            texts,
            batch_size=self.cfg.embed_batch_size,
            normalize=True,
            mmap_path=self.cfg.embed_mmap_path,
        )
        del texts
        gc.collect()
        print(f"  Embeddings shape: {self._embeddings.shape}")

        nlist = min(self.cfg.faiss_nlist, max(1, len(self._chunks) // 40))
        self._index = VectorIndex(
            dim=self._embeddings.shape[1],
            nlist=nlist,
            nprobe=self.cfg.faiss_nprobe,
        )
        self._index.build(self._embeddings)

        self._bm25 = BM25Index()
        self._bm25.build(self._chunks)

        self._build_repeal_linker()
        self._build_neighbour_index()

        self._save_cache(cache)
        with open(cache_meta_path, "w") as f:
            json.dump({"hash": ds_hash, "num_chunks": len(self._chunks)}, f)

        print(f"  ✅ Index built and cached. {len(self._chunks)} chunks ready.")

    def _build_repeal_linker(self):
        self._repeal_linker = RepealChainLinker()
        self._repeal_linker.build(self._chunks)

    def _build_neighbour_index(self):
        self._neighbour_index = {}
        for c in self._chunks:
            self._neighbour_index[(c.law_idx, c.chunk_seq)] = c

    # FIX-4: Include section_number in embed text
    def _chunk_to_embed_text(self, c: LawChunk) -> str:
        sec_part = f"section {c.section_number}" if c.section_number else ""
        return f"{c.law_title} {c.section_title} {sec_part} {c.text}".strip()

    def _save_cache(self, path: str):
        print("  Saving index cache...")
        with open(path, "wb") as f:
            pickle.dump({
                "chunks": self._chunks,
                "embeddings": self._embeddings,
                "faiss_index": faiss.serialize_index(self._index.index),
                "faiss_trained": self._index._trained,
                "bm25": self._bm25.serialise(),
            }, f)
        print(f"  Cache saved to {path}")

    def _load_cache(self, path: str):
        # HF-3 FIXED: patch sys.modules['__main__'] before pickle.load() so
        # that class references stored as '__main__.LawChunk', '__main__.RepealStatus',
        # etc. (written when the cache was built outside uvicorn) resolve correctly.
        # The finally block guarantees __main__ is always restored even on error.
        import sys
        import rag_pipeline as _rp

        _orig_main = sys.modules.get('__main__')
        sys.modules['__main__'] = _rp

        try:
            print(f"📦 Loading cache from {path}...", flush=True)
            with open(path, "rb") as f:
                data = pickle.load(f)
            print(f"📦 Cache loaded. Deserializing...", flush=True)
        finally:
            if _orig_main is not None:
                sys.modules['__main__'] = _orig_main
            else:
                sys.modules.pop('__main__', None)

        self._chunks = data["chunks"]
        self._chunks_by_id = {c.chunk_id: c for c in self._chunks}
        self._embeddings = data["embeddings"]
        raw_index = faiss.deserialize_index(data["faiss_index"])
        self._index = VectorIndex.__new__(VectorIndex)
        self._index.index = raw_index
        self._index._trained = data.get("faiss_trained", True)
        if "bm25" in data:
            self._bm25 = BM25Index.deserialise(data["bm25"])
        else:
            print("  BM25 not in cache — rebuilding from chunks...")
            self._bm25 = BM25Index()
            self._bm25.build(self._chunks)

    @staticmethod
    def _file_hash(path: str) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        return h.hexdigest()

    # ──────────────────────── RETRIEVAL ──────────────────────────────────────

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        if self._index is None or self._bm25 is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        lang = self._lang_detector.detect(query)

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            expand_future = pool.submit(
                self._query_expander._get_cache_entry, query, lang
            )
            q_vec = self._embedder.encode_one(query)
            dense_scores, dense_indices = self._index.search(q_vec, self.cfg.top_k_retrieve)
            cache_entry = expand_future.result()

        expanded_query = cache_entry.get("expanded", query)
        translated_query = cache_entry.get("translated", "")

        dense_hits = [
            (self._chunks[idx].chunk_id, float(score))
            for score, idx in zip(dense_scores, dense_indices)
            if 0 <= idx < len(self._chunks)
        ]

        bm25_raw_orig = self._bm25.search(expanded_query, self.cfg.top_k_bm25)
        bm25_hits_orig = [
            (self._bm25._corpus_ids[pos], score)
            for pos, score in bm25_raw_orig
            if 0 <= pos < len(self._bm25._corpus_ids)
        ]

        bm25_hits_translated = []
        if translated_query:
            bm25_raw_trans = self._bm25.search(translated_query, self.cfg.top_k_bm25)
            bm25_hits_translated = [
                (self._bm25._corpus_ids[pos], score)
                for pos, score in bm25_raw_trans
                if 0 <= pos < len(self._bm25._corpus_ids)
            ]

        merged_bm25: Dict[int, float] = {}
        for cid, score in bm25_hits_orig:
            merged_bm25[cid] = score
        for cid, score in bm25_hits_translated:
            if cid not in merged_bm25:
                merged_bm25[cid] = score * 0.85
            else:
                merged_bm25[cid] = max(merged_bm25[cid], score)

        bm25_hits_merged = sorted(merged_bm25.items(), key=lambda x: x[1], reverse=True)

        fused = reciprocal_rank_fusion(
            dense_hits=dense_hits,
            bm25_hits=bm25_hits_merged,
            all_chunks_by_id=self._chunks_by_id,
            rrf_k=self.cfg.rrf_k,
            dense_weight=self.cfg.dense_weight,
            bm25_weight=self.cfg.bm25_weight,
            top_k=self.cfg.top_k_fused,
        )

        seen_ids = {rc.chunk.chunk_id for rc in fused}
        injected: List[RetrievedChunk] = []

        if self._repeal_linker:
            for rc in fused:
                if rc.chunk.is_repealed:
                    replacements = self._repeal_linker.get_replacement_chunks(
                        rc.chunk, top_k=self.cfg.repeal_chain_inject_k
                    )
                    for repl_chunk, depth in replacements:
                        if repl_chunk.chunk_id not in seen_ids:
                            injected.append(RetrievedChunk(
                                chunk=repl_chunk,
                                score=rc.score * (0.95 if depth == 1 else 0.90),
                                match_type="repeal_chain",
                                injected_via_repeal_chain=True,
                                chain_depth=depth,
                            ))
                            seen_ids.add(repl_chunk.chunk_id)

        combined = fused + injected
        expanded_chunks = self._expand_neighbours(combined, seen_ids)

        rerank_pool_size = self.cfg.top_k_rerank * self.cfg.rerank_pool_multiplier
        if self._reranker and self.cfg.use_reranker:
            reranked = self._reranker.rerank(query, expanded_chunks, rerank_pool_size)
        else:
            reranked = sorted(expanded_chunks, key=lambda x: x.score, reverse=True)
            reranked = reranked[:rerank_pool_size]

        final = self._deduplicate(reranked, self.cfg.top_k_rerank)
        return final

    def _deduplicate(self, hits: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        seen = set()
        result = []
        for rc in hits:
            key = (rc.chunk.law_title, rc.chunk.chunk_seq)
            if key not in seen:
                seen.add(key)
                result.append(rc)
            if len(result) >= top_k:
                break
        return result

    def _expand_neighbours(self, hits: List[RetrievedChunk], seen_ids: set) -> List[RetrievedChunk]:
        expanded = list(hits)
        window = self.cfg.neighbour_window
        for hit in hits:
            c = hit.chunk
            for delta in range(-window, window + 1):
                if delta == 0:
                    continue
                neighbour = self._neighbour_index.get((c.law_idx, c.chunk_seq + delta))
                if neighbour and neighbour.chunk_id not in seen_ids:
                    mtype = f"prev_{abs(delta)}" if delta < 0 else f"next_{delta}"
                    expanded.append(RetrievedChunk(
                        chunk=neighbour,
                        score=hit.score * 0.85,
                        match_type=mtype,
                    ))
                    seen_ids.add(neighbour.chunk_id)
        return expanded

    # ──────────────────────── CHAT ────────────────────────────────────────

    def chat(self, query: str, stream: bool = False, verbose: bool = False) -> str:
        """
        Main chat method. verbose=False by default for HF Spaces
        (Gradio handles output, not stdout).
        stream=True returns a generator for Gradio streaming support.
        """
        if not self._generator:
            raise RuntimeError("Groq API key not set. Call .set_groq_key('your_key') first.")

        t0 = time.perf_counter()
        lang = self._lang_detector.detect(query)

        # FIX-7: Handle conversational queries without retrieval (LLM-based detection)
        if is_conversational(query):
            system, user_msg = self._prompt_builder.build_conversational(query, lang)
            if stream:
                # HF-5: return generator for Gradio streaming
                return self._stream_and_store(system, user_msg, query)
            else:
                full_answer = self._generator.generate(system, user_msg)
                self._chat_history.append({"query": query, "answer": full_answer})
                return full_answer

        retrieved = self.retrieve(query)

        if verbose:
            print(f"📚 Retrieved {len(retrieved)} chunks")
            for rc in retrieved:
                sec_str = f"sec={rc.chunk.section_number}" if rc.chunk.section_number else ""
                print(f"   {'⚠️' if rc.chunk.is_repealed else '✅'} [{rc.match_type}] "
                      f"{rc.chunk.law_title[:45]} | {sec_str}")

        citations = self._citation_extractor.extract(retrieved)
        system, user_msg = self._prompt_builder.build(query, citations, lang, self._chat_history)

        if stream:
            return self._stream_and_store(system, user_msg, query)
        else:
            full_answer = self._generator.generate(system, user_msg)
            self._chat_history.append({"query": query, "answer": full_answer})
            elapsed = (time.perf_counter() - t0) * 1000
            if verbose:
                print(f"⏱️  {elapsed:.0f} ms")
            return full_answer

    def _stream_and_store(self, system: str, user_msg: str, query: str):
        """HF-5: Generator for Gradio streaming — yields tokens, stores answer at end."""
        full_answer = ""
        for token in self._generator.generate_stream(system, user_msg):
            full_answer += token
            yield token
        self._chat_history.append({"query": query, "answer": full_answer})

    def search_only(self, query: str) -> List[dict]:
        retrieved = self.retrieve(query)
        return self._citation_extractor.extract(retrieved)

    def clear_history(self):
        self._chat_history = []

    def get_stats(self) -> dict:
        if not self._chunks:
            return {"status": "Index not built"}
        by_status: Dict[str, int] = {}
        for c in self._chunks:
            k = c.repeal_status.value
            by_status[k] = by_status.get(k, 0) + 1
        unique_laws = len(set(c.law_idx for c in self._chunks))
        bm25_vocab = len(self._bm25._bm25.idf) if self._bm25 and self._bm25._bm25 else 0
        chain_links = len(self._repeal_linker._repeal_chain) if self._repeal_linker else 0
        chunks_with_sec = sum(1 for c in self._chunks if c.section_number)
        return {
            "total_chunks": len(self._chunks),
            "unique_laws": unique_laws,
            "chunks_by_status": by_status,
            "chunks_with_section_number": chunks_with_sec,
            "section_number_coverage_pct": round(chunks_with_sec * 100 / max(len(self._chunks), 1), 1),
            "embedding_dim": self._embeddings.shape[1] if self._embeddings is not None else 0,
            "embed_model": self.cfg.embed_model,
            "bm25_vocabulary_size": bm25_vocab,
            "repeal_chain_links": chain_links,
            "retrieval": (
                f"Hybrid RRF (dense={self.cfg.dense_weight}, "
                f"bm25={self.cfg.bm25_weight}, k={self.cfg.rrf_k}) "
                f"+ cross-lingual BM25 + multi-hop repeal_chain_injection"
            ),
            "reranker": self.cfg.rerank_model if self.cfg.use_reranker else "disabled",
            "groq_model": self.cfg.groq_model,
        }


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def get_groq_key() -> str:
    """
    HF-2: Reads GROQ_API_KEY from environment only.
    Set it in HuggingFace Space: Settings → Variables and secrets → GROQ_API_KEY
    """
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError(
            "Groq API key not found.\n"
            "HuggingFace Spaces: Settings → Variables and secrets → Add GROQ_API_KEY"
        )
    return key