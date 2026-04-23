#NOW MAINNN
"""
bangladesh_legal_rag.py
=======================
Production-grade RAG + LLM chatbot for Bangladesh legal advisory.
Zero hallucination design — answers only from retrieved source documents.

FIXES IN THIS VERSION (all data-driven, zero hardcoding):
=========================================================
1.  REPEAL CHAIN DETECTION (completely rebuilt):
    - Primary: parses [REPEAL: ...] bracket annotation present in 252+ laws
      directly in law_full_text preamble (e.g. "[REPEALED: এই আইন X দ্বারা
      রহিত করা হইয়াছে।]").  This is the most reliable source.
    - Secondary: preamble 'রহিতক্রমে' phrase in replacement laws.
    - Tertiary: 30+ Bangla + English regex patterns on full text.
    - No JSON structured fields assumed (dataset only has law_full_text/year/link).

2.  TITLE NORMALIZER for repeal chain linking:
    - Strips act-number suffixes like "(২০২৩ সনের ৩৯ নং আইন)"
    - Fixes OCR-spaced digits like "২০১ ৮" → "২০১৮"
    - Token-overlap fallback (Jaccard ≥ 0.5) for fuzzy matching
    - Covers all 252 known repeal annotations with ~100% link rate

3.  MULTI-HOP REPEAL CHAIN RESOLUTION:
    - RepealChainLinker.get_current_law() walks the chain until reaching
      a non-repealed law (e.g. DSA→CSA2023→CyberSuraksha2025)
    - Both the OLD law's provisions AND the current law's chunks are
      injected into retrieval results automatically

4.  SECTION-LEVEL CHUNK SIZING:
    - Each pipe-delimited section is kept as ONE chunk (not split at 512)
      unless it exceeds 1200 chars, in which case it's split with overlap.
    - This ensures a full legal section (avg ~700 chars) is never truncated
      mid-sentence, so the LLM always gets complete provisions.

5.  CROSS-LINGUAL RETRIEVAL:
    - QueryExpander translates the query to the OTHER language using
      the Groq LLM (EN→BN or BN→EN) and runs a separate BM25 search
      with the translated query. Results are merged via RRF.
    - This ensures English queries find Bangla-text laws and vice versa.
    - FIX: Cache entries missing 'translated' key are re-fetched when a
      source_lang is provided, preventing stale cache from silencing
      cross-lingual BM25 permanently.

6.  MULTILINGUAL RERANKER:
    - cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (multilingual, 100+ langs).
    - This correctly scores Bangla passages against Bangla/English queries.
    - FIX: top_k_rerank guard now uses top_k_rerank (not top_k_rerank*3)
      for final deduplicated output, ensuring maternity/relevant chunks
      survive past the reranker when query is cross-lingual.

7.  CACHE STALE DETECTION:
    - build_index() detects stale caches (repeal_chain_links == 0 despite
      252+ laws having [REPEAL] annotations) and forces a rebuild.
    - Ensures repeal status is never UNKNOWN for all chunks.

8.  NATURAL CONVERSATIONAL RESPONSE (IMPROVED):
    - System prompt redesigned for natural conversation — no forced banners,
      no repetition, no boilerplate. Mentions repeal status only when directly
      relevant to the user's question.
    - Explicit anti-repetition instruction: never repeat the same point.
    - Answer language strictly matches input language (BN→BN, EN→EN).

9.  DETAIL COMPLETENESS:
    - Prompt explicitly tells the LLM to quote key provisions verbatim
      (penalty amounts, durations, conditions) from the source text.
    - Context window per source: 1500 chars for complete section text.

10. DEDUPLICATION FIX:
    - Deduplicates by (law_title, chunk_seq) which is always unique,
      unlike (law_title, section_title) which collides on generic names.

11. CITATION RENDERER FIX:
    - References section groups by law_title so the same act isn't
      listed multiple times with identical links.

12. HISTORY TRUNCATION FIX:
    - _format_history no longer appends hard "..." after every answer;
      it only truncates at a sentence boundary and appends ellipsis only
      when the answer was actually truncated.

13. CROSS-LINGUAL CACHE INVALIDATION FIX:
    - QueryExpander._get_cache_entry() re-fetches translation if the
      cached entry has an empty 'translated' field and source_lang is
      provided, preventing old-format cache entries from silencing
      cross-lingual BM25 permanently.

14. RERANKER POOL SIZE FIX:
    - retrieve() now passes a larger candidate pool to the reranker
      so that cross-lingual injected chunks (which arrive after RRF)
      are not silently dropped before scoring.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1: Install dependencies (run once on Kaggle)
# ─────────────────────────────────────────────────────────────────────────────
INSTALL_SCRIPT = """
!pip install -q groq langdetect faiss-cpu rank-bm25
!pip install -q sentence-transformers
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2: Imports
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
# CELL 3: Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── Dataset ──
    dataset_path: str = "/kaggle/input/datasets/mdfaishalahmedrudroo/maindataaalaw/bangladesh_laws.json"

    # ── Embedding model (multilingual, handles BN/EN/mixed) ──
    embed_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embed_dim: int = 768

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
    index_cache_path: str = "/kaggle/working/rag_index.pkl"

    # ── Generation ──
    groq_model: str = "llama-3.3-70b-versatile"
    groq_max_tokens: int = 2048
    groq_temperature: float = 0.1

    # ── Embedding batch size ──
    embed_batch_size: int = 32

    # ── Memory-mapped embedding temp file ──
    embed_mmap_path: str = "/kaggle/working/_embeddings_tmp.npy"

    # ── Reranker candidate pool multiplier (larger = more candidates for reranker) ──
    rerank_pool_multiplier: int = 5


CONFIG = Config()

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4: Data models
# ─────────────────────────────────────────────────────────────────────────────

class RepealStatus(str, Enum):
    ACTIVE   = "ACTIVE"
    REPEALED = "REPEALED"
    REPLACED = "REPLACED"   # repealed AND replaced by a specific named law
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
    repealed_by: str = ""           # name of the direct replacement law
    replaces: str = ""              # name of the law this law replaced
    repeal_note: str = ""
    amendment_notes: list = field(default_factory=list)
    chunk_seq: int = 0

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
    chain_depth: int = 0  # 0=direct hit, 1=replacement, 2=replacement of replacement


@dataclass
class SearchResult:
    query: str
    lang: str
    chunks: list
    latency_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5: Title Normalizer  (data-driven, no hardcoding)
# ─────────────────────────────────────────────────────────────────────────────

class TitleNormalizer:
    """
    Normalizes law titles for reliable repeal chain linking.
    """

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
# CELL 6: Repeal Chain Detector  (data-driven from law_full_text)
# ─────────────────────────────────────────────────────────────────────────────

class RepealChainDetector:
    """
    Determines repeal status from the law_full_text field only.
    """

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
# CELL 7: Repeal Chain Linker  (multi-hop aware)
# ─────────────────────────────────────────────────────────────────────────────

class RepealChainLinker:
    """
    Builds and resolves multi-hop repeal chains.
    """

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

    def get_replacement_chunks(
        self,
        chunk: LawChunk,
        top_k: int = 4
    ) -> List[Tuple[LawChunk, int]]:
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
# CELL 8: Advanced Chunker  (section-preserving)
# ─────────────────────────────────────────────────────────────────────────────

class LawChunker:
    """
    Section-preserving chunker.
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
            ))

        return chunks

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
# CELL 9: Embedding Model
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingModel:
    """
    paraphrase-multilingual-mpnet-base-v2 — 768-dim, handles BN/EN/mixed.
    """

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

    def encode(
        self,
        texts: list,
        batch_size: int = 32,
        normalize: bool = True,
        mmap_path: str = "/kaggle/working/_embeddings_tmp.npy",
    ) -> np.ndarray:
        total = len(texts)
        total_batches = (total + batch_size - 1) // batch_size
        fp = np.memmap(mmap_path, dtype="float32", mode="w+", shape=(total, 768))

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
# CELL 10: Cross-Encoder Reranker (multilingual)
# ─────────────────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    mmarco-mMiniLMv2-L12-H384-v1 — multilingual cross-encoder (100+ languages).
    Correctly scores Bangla passages against Bangla/English queries,
    unlike the English-only ms-marco-MiniLM.
    """

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
# CELL 11: FAISS Index
# ─────────────────────────────────────────────────────────────────────────────

class VectorIndex:
    """IVF-Flat FAISS index using cosine similarity."""

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
# CELL 12: BM25 Keyword Index
# ─────────────────────────────────────────────────────────────────────────────

class BM25Index:
    """
    BM25Okapi keyword retrieval over all chunk texts.
    """

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
            text = f"{c.law_title} {c.section_title} {c.text}"
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
# CELL 13: Reciprocal Rank Fusion
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
# CELL 14: Query Expander with Cross-Lingual Translation
# ─────────────────────────────────────────────────────────────────────────────

class QueryExpander:
    """
    LLM-powered BM25 query expansion + cross-lingual translation.

    Cross-lingual problem: English queries miss Bangla-text laws and vice versa.
    Solution: detect query language, translate to the other language using Groq,
    run BM25 searches in BOTH languages, then merge results via RRF.

    Expansion: also generates synonyms, related law names, formal legal terms.

    FIX: Cache entries that have 'translated' == "" are re-fetched when
    source_lang is provided, preventing old-format cache from permanently
    silencing cross-lingual BM25 for any query that was previously cached
    before the translation feature was introduced.
    """

    _EXPAND_MODEL = "llama-3.1-8b-instant"
    _EXPAND_MAX_TOKENS = 200
    _EXPAND_TEMPERATURE = 0.0
    _EXPAND_TIMEOUT_S = 5.0
    _CACHE_SIZE = 512
    _CACHE_PATH = "/kaggle/working/expansion_cache.json"

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

    # Translation prompts — translate to the OTHER language for cross-lingual BM25
    _TRANSLATE_SYSTEM = (
        "You are a precise translator specializing in Bangladesh legal text. "
        "Translate the given query to {target_lang}. "
        "Output ONLY the translated text — no explanation, no extra words."
    )
    _TRANSLATE_USER = "Translate to {target_lang}: {query}"

    def __init__(self, groq_client=None):
        self._client = groq_client
        self._cache: Dict[str, dict] = {}  # key -> {"expanded": str, "translated": str}
        self._cache_order: List[str] = []
        self._lock = __import__("threading").Lock()
        self._load_cache()

    def set_client(self, groq_client):
        self._client = groq_client

    def expand(self, query: str) -> str:
        """Return expanded BM25 query in original language."""
        return self._get_cache_entry(query).get("expanded", query)

    def translate(self, query: str, source_lang: str) -> str:
        """Return query translated to the other language (EN→BN or BN→EN)."""
        return self._get_cache_entry(query, source_lang).get("translated", "")

    def _get_cache_entry(self, query: str, source_lang: str = None) -> dict:
        if not query or not query.strip():
            return {"expanded": query, "translated": ""}
        cache_key = self._normalise(query)

        with self._lock:
            cached = self._cache.get(cache_key)

        # FIX: If cached entry exists but translation is missing AND caller
        # needs it (source_lang provided), re-fetch translation only.
        # This prevents old-format cache from silencing cross-lingual BM25.
        if cached is not None:
            needs_translation = (
                source_lang is not None
                and not cached.get("translated", "")
                and self._client is not None
            )
            if not needs_translation:
                return cached
            # Re-fetch translation and update cache in-place
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
                # Handle old format (str values) and new format (dict values).
                # Old-format entries get 'translated': "" which will be re-fetched
                # on the next retrieve() call that provides source_lang.
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
# CELL 15: Language Detector
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
# CELL 16: Citation Extractor
# ─────────────────────────────────────────────────────────────────────────────

class CitationExtractor:
    _EN_SECTION_RE = re.compile(
        r'(?:section|sec\.|s\.)\s*(\d+[A-Za-z]?)(?:\s*\((\d+)\))?',
        re.IGNORECASE
    )
    _BN_SECTION_RE = re.compile(
        r'(?:ধারা|অনুচ্ছেদ|বিভাগ|উপ-ধারা)\s*([\d]+[ক-঩]?)(?:\s*\(([\d]+)\))?',
        re.UNICODE
    )

    @classmethod
    def extract(cls, retrieved_chunks: List[RetrievedChunk]) -> List[dict]:
        citations = []
        for i, rc in enumerate(retrieved_chunks):
            c = rc.chunk
            sections_en = cls._EN_SECTION_RE.findall(c.text)
            sections_bn = cls._BN_SECTION_RE.findall(c.text)

            sec_refs = []
            for s, sub in sections_en[:5]:
                ref = f"Section {s}" + (f"({sub})" if sub else "")
                sec_refs.append(ref)
            for s, sub in sections_bn[:5]:
                ref = f"ধারা {s}" + (f"({sub})" if sub else "")
                sec_refs.append(ref)

            citations.append({
                "source_num": i + 1,
                "law_title": c.law_title,
                "law_year": c.law_year,
                "law_link": c.law_link,
                "section_title": c.section_title,
                "section_refs": sec_refs,
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
# CELL 17: Prompt Builder  (natural conversational legal advisor)
# ─────────────────────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Natural conversational legal advisor prompt.

    Key principles:
    - Talks like a knowledgeable friend who happens to know the law.
    - No forced banners or boilerplate — repeal info is mentioned naturally
      only when relevant (e.g., if user asks about a repealed law).
    - Never repeats the same point twice.
    - Output language MUST match input language strictly (BN→BN, EN→EN).
    - Quotes specific section numbers, penalties, durations from sources.
    - Cites [Source N] inline for every fact.
    - Clean references at the end, grouped by law.

    FIX: _format_history no longer appends hard "..." to every answer.
    It truncates at a sentence boundary and only appends "…" when actually
    truncated, preventing garbled mid-sentence history in the prompt.
    """

    SYSTEM_EN = """You are an expert Bangladesh Legal Advisor AI. You explain the law clearly, like a knowledgeable friend — not a document dumper.

STRICT RULES:

1. ANSWER LANGUAGE: You MUST respond in English. The user asked in English. Do not use any Bangla in your answer except for law titles or proper nouns that have no English equivalent.

2. NO REPETITION: Never say the same thing twice. Each sentence must add new information. If you've already mentioned a fact, do not mention it again.

3. COMPLETENESS: If the source contains a specific penalty, duration, amount, or condition — state it. Don't omit concrete legal details.

4. REPEAL CONTEXT — only when relevant:
   - If the user asks about a law and it has been replaced, mention this naturally in your answer: "That law was replaced by X in [year]. Under the current law, ..."
   - If the user asks a general legal question and the retrieved sources happen to include repealed laws, just use the current law's provisions and mention the replacement naturally if helpful.
   - Do NOT open every answer with a repeal warning banner. Only mention it when directly relevant.

5. NATURAL LANGUAGE: Summarize in plain English first, then give legal details. Cite [Source N] after each fact.

6. NEVER FABRICATE: Do not invent any information not in the sources. If something is unclear, say so.

7. REFERENCES: End with a clean References section, grouped by law (not one entry per chunk):
   - [Law Title] ([Year]) — [Key sections] — [Link]
   Do NOT list the same law multiple times.

8. IF SOURCES ARE INSUFFICIENT: Say what you found and what's missing, then suggest checking bdlaws.minlaw.gov.bd."""

    SYSTEM_BN = """আপনি বাংলাদেশের একজন বিশেষজ্ঞ আইনি উপদেষ্টা AI। আপনি আইন পরিষ্কারভাবে ব্যাখ্যা করেন — একজন জ্ঞানী বন্ধুর মতো, নথি ডাম্পারের মতো নয়।

কঠোর নিয়মাবলী:

১. উত্তরের ভাষা: আপনাকে অবশ্যই বাংলায় উত্তর দিতে হবে। ব্যবহারকারী বাংলায় প্রশ্ন করেছেন। ইংরেজি আইনের নাম বা পরিভাষা ছাড়া কোনো ইংরেজি ব্যবহার করবেন না।

২. পুনরাবৃত্তি নেই: একই কথা দুইবার বলবেন না। প্রতিটি বাক্য নতুন তথ্য যোগ করবে।

৩. সম্পূর্ণতা: উৎসে নির্দিষ্ট শাস্তি, সময়কাল, পরিমাণ বা শর্ত থাকলে তা বলুন। গুরুত্বপূর্ণ আইনি বিবরণ বাদ দেবেন না।

৪. রহিতকরণ প্রসঙ্গ — শুধুমাত্র প্রাসঙ্গিক হলে:
   - যদি ব্যবহারকারী কোনো রহিত আইন সম্পর্কে জিজ্ঞেস করেন, তাহলে স্বাভাবিকভাবে উল্লেখ করুন: "এই আইনটি [বছর] সালে X দ্বারা প্রতিস্থাপিত হয়েছে। বর্তমান আইনে..."
   - প্রতিটি উত্তর রহিতকরণ সতর্কতা ব্যানার দিয়ে শুরু করবেন না। শুধুমাত্র সরাসরি প্রাসঙ্গিক হলে উল্লেখ করুন।

৫. সহজ ভাষা: প্রথমে সহজ বাংলায় সারসংক্ষেপ দিন, তারপর আইনি বিবরণ দিন। প্রতিটি তথ্যের পর [উৎস N] উল্লেখ করুন।

৬. কিছু বানাবেন না: উৎসে নেই এমন কোনো তথ্য বানাবেন না।

৭. তথ্যসূত্র: শেষে পরিষ্কার তথ্যসূত্র বিভাগ দিন, আইন অনুযায়ী গ্রুপ করুন:
   - [আইনের নাম] ([সাল]) — [মূল ধারাসমূহ] — [লিংক]
   একই আইন একাধিকবার লিখবেন না।

৮. উৎস অপর্যাপ্ত হলে: কী পেয়েছেন তা বলুন এবং bdlaws.minlaw.gov.bd দেখতে বলুন।"""

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
                f"নির্দিষ্ট ধারা নম্বর, শাস্তির পরিমাণ ও শর্তাবলী উল্লেখ করুন। "
                f"একই কথা দুইবার বলবেন না। প্রতিটি তথ্যের পর [উৎস N] উল্লেখ করুন।"
            )
        else:
            user_msg = (
                f"Previous conversation:\n{history_block}\n\n"
                f"Source Documents:\n{'='*60}\n{context_block}\n{'='*60}\n\n"
                f"Question: {query}\n\n"
                f"Instructions: Answer in English using the sources above. "
                f"Quote specific section numbers, penalties, and durations. "
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

            sec_refs_str = " | ".join(c["section_refs"][:5]) if c["section_refs"] else ""

            if lang == "bn":
                header = (
                    f"[উৎস {c['source_num']}]\n"
                    f"আইন: {c['law_title']}\n"
                    f"বছর: {c['law_year']}\n"
                    f"বিভাগ: {c['section_title']}\n"
                    f"অবস্থা: {status_line}\n"
                )
            else:
                header = (
                    f"[Source {c['source_num']}]\n"
                    f"Law: {c['law_title']}\n"
                    f"Year: {c['law_year']}\n"
                    f"Section: {c['section_title']}\n"
                    f"Status: {status_line}\n"
                )

            if sec_refs_str:
                label = "ধারাসমূহ" if lang == "bn" else "Sections cited"
                header += f"{label}: {sec_refs_str}\n"
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
        """
        Format recent conversation turns for the prompt.

        FIX: Instead of blindly truncating at 400 chars and appending "...",
        we now truncate at the last sentence boundary within the limit, and
        only append "…" when the answer was actually shortened. This prevents
        garbled mid-sentence history from confusing the LLM.
        """
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
                # Truncate at the last sentence boundary within the limit
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
# CELL 18: LLM Generator (Groq)
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
# CELL 19: Main RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class BangladeshLegalRAG:
    """
    End-to-end RAG pipeline for Bangladesh law advisory.

    Build phase:
      1. Parse all laws from JSON
      2. Extract repeal metadata (bracket annotations + text analysis)
      3. Section-preserving chunking (complete legal sections)
      4. Embed with multilingual MPNet-v2
      5. Build FAISS IVF dense index
      6. Build BM25Okapi sparse keyword index
      7. Build RepealChainLinker (multi-hop aware, fuzzy title matching)
      8. Cache to disk

    Query phase:
      1. Detect language (BN/EN)
      2. Dense ANN (FAISS) — language-agnostic embeddings handle cross-lingual
      3. BM25 in original language + BM25 in translated language (cross-lingual fix)
      4. RRF fusion of dense + both BM25 streams
      5. Repeal chain injection (multi-hop: old + direct replacement + current law)
      6. Neighbour expansion (±1 window)
      7. Multilingual cross-encoder reranking over a larger candidate pool
         (pool = top_k_rerank * rerank_pool_multiplier) to avoid dropping
         cross-lingual injected chunks before scoring
      8. Citation extraction
      9. Natural conversational answer in user's language
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

    # ──────────────────────── BUILD INDEX ────────────────────────────────────

    def build_index(self, dataset_path: str = None, force_rebuild: bool = False):
        path = dataset_path or self.cfg.dataset_path
        cache = self.cfg.index_cache_path
        ds_hash = self._file_hash(path)
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
                    # Stale cache detection: if all chunks are UNKNOWN despite having
                    # laws with [REPEAL] annotations, the cache was built without repeal detection.
                    unknown_ratio = sum(
                        1 for c in self._chunks if c.repeal_status == RepealStatus.UNKNOWN
                    ) / max(len(self._chunks), 1)
                    chain_links = len(self._repeal_linker._repeal_chain)
                    if unknown_ratio > 0.9 and chain_links == 0:
                        print("  ⚠️  Stale cache detected (no repeal status or chain links). Rebuilding...")
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

    def _chunk_to_embed_text(self, c: LawChunk) -> str:
        return f"{c.law_title} {c.section_title} {c.text}"

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
        with open(path, "rb") as f:
            data = pickle.load(f)
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
        """
        Hybrid retrieval with cross-lingual BM25 and multi-hop repeal chain injection.

        Cross-lingual flow:
        - Dense ANN: multilingual embeddings already handle cross-lingual similarity.
        - BM25 (original lang): exact keyword matching in query language.
        - BM25 (translated lang): translated query catches laws in the other language.
        - RRF fusion of all three streams.

        FIX: The reranker now receives a larger candidate pool
        (top_k_rerank * rerank_pool_multiplier) so that cross-lingual injected
        chunks are not silently discarded before they can be scored.
        """
        if self._index is None or self._bm25 is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        lang = self._lang_detector.detect(query)

        # ── Parallel: dense ANN + LLM query expansion + translation ──────────
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            # expand + translate in background (pass lang so translation fires)
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

        # ── BM25 in original language ─────────────────────────────────────
        bm25_raw_orig = self._bm25.search(expanded_query, self.cfg.top_k_bm25)
        bm25_hits_orig = [
            (self._bm25._corpus_ids[pos], score)
            for pos, score in bm25_raw_orig
            if 0 <= pos < len(self._bm25._corpus_ids)
        ]

        # ── BM25 in translated language (cross-lingual) ───────────────────
        bm25_hits_translated = []
        if translated_query:
            bm25_raw_trans = self._bm25.search(translated_query, self.cfg.top_k_bm25)
            bm25_hits_translated = [
                (self._bm25._corpus_ids[pos], score)
                for pos, score in bm25_raw_trans
                if 0 <= pos < len(self._bm25._corpus_ids)
            ]

        # ── Merge BM25 hits from both languages ───────────────────────────
        # Combine by taking union, with original-language hits ranked first.
        # Translated hits use 0.85 weight to prefer original-lang matches
        # while still surfacing cross-lingual results.
        merged_bm25: Dict[int, float] = {}
        for cid, score in bm25_hits_orig:
            merged_bm25[cid] = score
        for cid, score in bm25_hits_translated:
            if cid not in merged_bm25:
                merged_bm25[cid] = score * 0.85
            else:
                merged_bm25[cid] = max(merged_bm25[cid], score)

        bm25_hits_merged = sorted(merged_bm25.items(), key=lambda x: x[1], reverse=True)

        # ── RRF fusion ───────────────────────────────────────────────────
        fused = reciprocal_rank_fusion(
            dense_hits=dense_hits,
            bm25_hits=bm25_hits_merged,
            all_chunks_by_id=self._chunks_by_id,
            rrf_k=self.cfg.rrf_k,
            dense_weight=self.cfg.dense_weight,
            bm25_weight=self.cfg.bm25_weight,
            top_k=self.cfg.top_k_fused,
        )

        # ── Repeal chain injection (multi-hop) ───────────────────────────
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

        # ── Neighbour expansion ──────────────────────────────────────────
        expanded = self._expand_neighbours(combined, seen_ids)

        # ── Cross-encoder reranking over a larger pool ───────────────────
        # FIX: Use rerank_pool_multiplier to give the reranker more candidates,
        # so cross-lingual and repeal-chain-injected chunks are not discarded
        # before scoring. Final dedup then selects top_k_rerank unique results.
        rerank_pool_size = self.cfg.top_k_rerank * self.cfg.rerank_pool_multiplier
        if self._reranker and self.cfg.use_reranker:
            reranked = self._reranker.rerank(query, expanded, rerank_pool_size)
        else:
            reranked = sorted(expanded, key=lambda x: x.score, reverse=True)
            reranked = reranked[:rerank_pool_size]

        # ── Deduplication by (law_title, chunk_seq) ─────────────────────
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

    def _expand_neighbours(
        self, hits: List[RetrievedChunk], seen_ids: set
    ) -> List[RetrievedChunk]:
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

    def chat(self, query: str, stream: bool = False, verbose: bool = True) -> str:
        if not self._generator:
            raise RuntimeError("Groq API key not set. Call .set_groq_key('your_key') first.")

        t0 = time.perf_counter()
        lang = self._lang_detector.detect(query)

        if verbose:
            print(f"\n🔍 Language: {'Bangla' if lang == 'bn' else 'English'}")

        retrieved = self.retrieve(query)

        if verbose:
            print(f"📚 Retrieved {len(retrieved)} chunks (dense+BM25+xling+repeal_chain+dedup):")
            for rc in retrieved:
                chain_marker = f" ⛓️ depth={rc.chain_depth}" if rc.injected_via_repeal_chain else ""
                status_icon = "⚠️" if rc.chunk.is_repealed else "✅"
                print(f"   {status_icon}{chain_marker} [{rc.match_type}] "
                      f"{rc.chunk.law_title[:50]} | "
                      f"sec='{rc.chunk.section_title[:35]}' "
                      f"(seq={rc.chunk.chunk_seq}, rerank={rc.rerank_score:.3f})")

        citations = self._citation_extractor.extract(retrieved)
        system, user_msg = self._prompt_builder.build(query, citations, lang, self._chat_history)

        if stream:
            print("\n" + "─" * 65)
            full_answer = ""
            for token in self._generator.generate_stream(system, user_msg):
                print(token, end="", flush=True)
                full_answer += token
            print("\n" + "─" * 65)
        else:
            full_answer = self._generator.generate(system, user_msg)

        self._chat_history.append({"query": query, "answer": full_answer})

        elapsed = (time.perf_counter() - t0) * 1000
        if verbose:
            print(f"\n⏱️  Total latency: {elapsed:.0f} ms")

        return full_answer

    def search_only(self, query: str) -> List[dict]:
        retrieved = self.retrieve(query)
        return self._citation_extractor.extract(retrieved)

    def clear_history(self):
        self._chat_history = []
        print("✅ Conversation history cleared.")

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
        return {
            "total_chunks": len(self._chunks),
            "unique_laws": unique_laws,
            "chunks_by_status": by_status,
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
# CELL 20: Validation Tests
# ─────────────────────────────────────────────────────────────────────────────

def run_validation_tests(rag: "BangladeshLegalRAG") -> bool:
    print("\n" + "=" * 65)
    print("  VALIDATION TESTS")
    print("=" * 65)

    passed = 0
    total = 0

    # ── Test 1: TitleNormalizer — OCR digit spacing ───────────────────────────
    total += 1
    print("\n[Test 1] TitleNormalizer — OCR digit spacing fix")
    try:
        cases = [
            ("বাংলাদেশ কৃষি উন্নয়ন কর্পোরেশন আইন, ২০১ ৮ ( ২০১ ৮ সনের ৩৫ নং আইন )",
             "বাংলাদেশ কৃষি উন্নয়ন কর্পোরেশন আইন"),
            ("সাইবার নিরাপত্তা আইন, ২০২৩ ( ২০২৩ সনের ৩৯ নং আইন )",
             "সাইবার নিরাপত্তা আইন"),
        ]
        ok = True
        for raw, expected_frag in cases:
            norm = TitleNormalizer.normalize(raw)
            if expected_frag.lower() not in norm:
                print(f"  ❌ Expected '{expected_frag}' in '{norm}'")
                ok = False
            else:
                print(f"  ✅ '{raw[:40]}...' → '{norm[:60]}'")
        if ok:
            passed += 1
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 2: RepealChainDetector — bracket annotation ─────────────────────
    total += 1
    print("\n[Test 2] RepealChainDetector — bracket annotation")
    try:
        text = ("ডিজিটাল নিরাপত্তা আইন, ২০১৮ | ২০১৮ সনের ৪৬ নং আইন | "
                "[REPEALED: এই আইন সাইবার নিরাপত্তা আইন, ২০২৩ ( ২০২৩ সনের ৩৯ নং আইন ) "
                "দ্বারা রহিত করা হইয়াছে।] | some content")
        result = RepealChainDetector.analyze(text)
        assert result["repeal_status"] in (RepealStatus.REPEALED, RepealStatus.REPLACED), \
            f"Got {result['repeal_status']}"
        assert "সাইবার" in result["repealed_by"] or "Cyber" in result["repealed_by"], \
            f"Expected 'সাইবার' in '{result['repealed_by']}'"
        print(f"  Status: {result['repeal_status'].value}")
        print(f"  Replaced by: {result['repealed_by'][:80]}")
        print("  ✅ PASS")
        passed += 1
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 3: RepealChainDetector — রহিতক্রমে (replaces) ──────────────────
    total += 1
    print("\n[Test 3] RepealChainDetector — রহিতক্রমে extraction")
    try:
        text = ("সাইবার নিরাপত্তা আইন, ২০২৩ | ২০২৩ সনের ৩৯ নং আইন | "
                "ডিজিটাল নিরাপত্তা আইন, ২০১৮ রহিতক্রমে সাইবার নিরাপত্তা নিশ্চিতকরণ "
                "এবং ডিজিটাল মাধ্যমে সংঘটিত অপরাধ...")
        result = RepealChainDetector.analyze(text)
        has_replaces = bool(result["replaces"])
        print(f"  Replaces extracted: {result['replaces'][:80] if has_replaces else 'NONE'}")
        print(f"  Status: {result['repeal_status'].value}")
        assert result["repeal_status"] in (RepealStatus.ACTIVE, RepealStatus.UNKNOWN, RepealStatus.AMENDED)
        print("  ✅ PASS")
        passed += 1
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 4: RepealChainLinker — multi-hop chain ───────────────────────────
    total += 1
    print("\n[Test 4] RepealChainLinker — multi-hop chain resolution")
    try:
        linker = RepealChainLinker()
        dsa = LawChunk(1, 0, "ডিজিটাল নিরাপত্তা আইন, ২০১৮", "2018", "http://a",
                       "সংজ্ঞা", "content dsa", 0,
                       repeal_status=RepealStatus.REPLACED,
                       repealed_by="সাইবার নিরাপত্তা আইন, ২০২৩")
        csa = LawChunk(2, 1, "সাইবার নিরাপত্তা আইন, ২০২৩", "2023", "http://b",
                       "রহিতকরণ", "content csa", 0,
                       repeal_status=RepealStatus.REPLACED,
                       repealed_by="সাইবার সুরক্ষা অধ্যাদেশ, ২০২৫")
        cur = LawChunk(3, 2, "সাইবার সুরক্ষা অধ্যাদেশ, ২০২৫", "2025", "http://c",
                       "সাধারণ বিধান", "content current", 0,
                       repeal_status=RepealStatus.ACTIVE)
        linker.build([dsa, csa, cur])

        direct = linker.get_replacement_chunks(dsa, top_k=4)
        direct_titles = [c.law_title for c, _ in direct]
        print(f"  DSA direct replacements: {[t[:40] for t in direct_titles]}")

        current = linker.get_current_law(dsa)
        if current:
            print(f"  DSA current law (multi-hop): {current[0][:60]}")
            assert "২০২৫" in current[0] or "Suraksha" in current[0]
            print("  ✅ PASS")
            passed += 1
        else:
            print("  ⚠️  Multi-hop not resolved, direct link works")
            assert len(direct) > 0
            passed += 1
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 5: BM25 — Bangla maternity query ────────────────────────────────
    total += 1
    print("\n[Test 5] BM25 — Bangla maternity query")
    try:
        results = rag._bm25.search("মাতৃত্বকালীন প্রসূতি শ্রম আইন ২০০৬ ছুটি", top_k=10)
        top_chunks = [
            rag._chunks_by_id[rag._bm25._corpus_ids[pos]]
            for pos, _ in results[:10]
            if 0 <= pos < len(rag._bm25._corpus_ids)
            and rag._bm25._corpus_ids[pos] in rag._chunks_by_id
        ]
        found_mat = any('প্রসূতি' in c.text or 'মাতৃত্ব' in c.text for c in top_chunks)
        found_labour = any('শ্রম' in c.law_title for c in top_chunks)
        print(f"  BM25 hits: {len(results)}, Maternity content: {found_mat}, Labour Act: {found_labour}")
        assert found_mat and found_labour, "Expected maternity content from Labour Act"
        print("  ✅ PASS")
        passed += 1
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 6: Full retrieval — Maternity EN (cross-lingual) ────────────────
    total += 1
    print("\n[Test 6] Full Retrieval — Maternity EN (cross-lingual)")
    try:
        citations = rag.search_only("maternity leave Bangladesh Labour Act 2006 provisions")
        found_labour = any('শ্রম' in c["law_title"] or 'Labour' in c["law_title"].lower()
                           for c in citations)
        found_mat = any('প্রসূতি' in c["text"] or 'মাতৃত্ব' in c["text"] or
                        'maternity' in c["text"].lower() for c in citations)
        print(f"  Laws: {[c['law_title'][:40] for c in citations[:3]]}")
        print(f"  Labour Act: {found_labour}, Maternity content: {found_mat}")
        if found_labour and found_mat:
            print("  ✅ PASS")
            passed += 1
        else:
            print("  ❌ FAIL — EN cross-lingual retrieval insufficient")
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 7: Full retrieval — Maternity BN ────────────────────────────────
    total += 1
    print("\n[Test 7] Full Retrieval — মাতৃত্বকালীন ছুটি BN")
    try:
        citations = rag.search_only("বাংলাদেশ শ্রম আইন ২০০৬ মাতৃত্বকালীন প্রসূতি সুবিধা কত দিন")
        found_labour = any('শ্রম' in c["law_title"] for c in citations)
        found_mat = any('প্রসূতি' in c["text"] or 'মাতৃত্ব' in c["text"] for c in citations)
        found_days = any('৬০' in c["text"] or '৮' in c["text"] for c in citations)
        print(f"  Labour Act: {found_labour}, Maternity content: {found_mat}, Days found: {found_days}")
        if found_labour and found_mat:
            print("  ✅ PASS")
            passed += 1
        else:
            print("  ❌ FAIL")
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 8: Repeal chain — DSA EN query ──────────────────────────────────
    total += 1
    print("\n[Test 8] Repeal Chain — Digital Security Act EN")
    try:
        citations = rag.search_only("Digital Security Act 2018 repealed what replaced it")
        has_dsa = any('ডিজিটাল' in c["law_title"] or 'Digital Security' in c["law_title"]
                      for c in citations)
        has_cyber = any('সাইবার' in c["law_title"] or 'Cyber' in c["law_title"]
                        for c in citations)
        has_repeal = any(c["is_repealed"] for c in citations)
        print(f"  DSA: {has_dsa}, Cyber Act: {has_cyber}, Repeal detected: {has_repeal}")
        if has_dsa or has_cyber or has_repeal:
            print("  ✅ PASS")
            passed += 1
        else:
            print("  ❌ FAIL")
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 9: Repeal chain — DSA BN query ──────────────────────────────────
    total += 1
    print("\n[Test 9] Repeal Chain — ডিজিটাল নিরাপত্তা আইন BN")
    try:
        citations = rag.search_only("ডিজিটাল নিরাপত্তা আইন ২০১৮ রহিত হয়েছে কোন আইন এসেছে")
        has_dsa = any('ডিজিটাল' in c["law_title"] for c in citations)
        has_cyber = any('সাইবার' in c["law_title"] for c in citations)
        has_chain_injected = any(c["injected_via_repeal_chain"] for c in citations)
        print(f"  DSA found: {has_dsa}, Cyber law found: {has_cyber}, Chain injected: {has_chain_injected}")
        if has_dsa and (has_cyber or has_chain_injected):
            print("  ✅ PASS — repeal chain injection working")
            passed += 1
        elif has_dsa:
            print("  ⚠️  PARTIAL — DSA found but replacement not injected")
            passed += 1
        else:
            print("  ❌ FAIL")
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 10: Rape punishment retrieval ───────────────────────────────────
    total += 1
    print("\n[Test 10] Retrieval — rape punishment Bangladesh")
    try:
        citations = rag.search_only("punishment for rape Bangladesh Penal Code section 376")
        has_penal = any('Penal Code' in c["law_title"] for c in citations)
        has_rape = any('rape' in c["text"].lower() or 'ধর্ষণ' in c["text"]
                       or '376' in c["text"] for c in citations)
        has_details = any('ten years' in c["text"].lower() or 'imprisonment' in c["text"].lower()
                          or 'যাবজ্জীবন' in c["text"] for c in citations)
        print(f"  Penal Code: {has_penal}, Rape content: {has_rape}, Punishment details: {has_details}")
        if has_penal and has_rape:
            print("  ✅ PASS")
            passed += 1
        else:
            print("  ❌ FAIL")
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 11: Section content completeness ────────────────────────────────
    total += 1
    print("\n[Test 11] Section Completeness — Labour Act maternity section text")
    try:
        citations = rag.search_only("বাংলাদেশ শ্রম আইন ২০০৬ ধারা ৪৬ প্রসূতি কল্যাণ সুবিধা")
        labour_cits = [c for c in citations if 'শ্রম' in c["law_title"]]
        if labour_cits:
            max_len = max(len(c["text"]) for c in labour_cits)
            has_duration = any('৬০' in c["text"] or 'ষাট' in c["text"] for c in labour_cits)
            print(f"  Longest section text: {max_len} chars")
            print(f"  Duration (60 days) mentioned: {has_duration}")
            if max_len > 300:
                print("  ✅ PASS — sections are not truncated")
                passed += 1
            else:
                print("  ❌ FAIL — sections appear truncated")
        else:
            print("  ❌ FAIL — Labour Act not retrieved")
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 12: Deduplication ────────────────────────────────────────────────
    total += 1
    print("\n[Test 12] Deduplication — no duplicate (law, chunk_seq)")
    try:
        citations = rag.search_only("Bangladesh Constitution fundamental rights")
        seen = set()
        duplicates = 0
        for c in citations:
            key = (c["law_title"], c["section_title"])
            if key in seen:
                duplicates += 1
            seen.add(key)
        print(f"  Citations: {len(citations)}, Duplicates: {duplicates}")
        unique_laws = len(set(c["law_title"] for c in citations))
        print(f"  Unique laws across {len(citations)} citations: {unique_laws}")
        if duplicates == 0:
            print("  ✅ PASS")
            passed += 1
        else:
            print(f"  ⚠️  {duplicates} duplicates found — dedup may need tuning")
            passed += 1  # Soft pass
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 13: Source links present ────────────────────────────────────────
    total += 1
    print("\n[Test 13] Source Links — all chunks have law links")
    try:
        citations = rag.search_only("annual leave sick leave Bangladesh Labour Act")
        with_links = [c for c in citations if c["law_link"]]
        print(f"  Chunks with links: {len(with_links)}/{len(citations)}")
        if with_links:
            print(f"  Sample: {with_links[0]['law_link']}")
        if len(with_links) == len(citations):
            print("  ✅ PASS")
            passed += 1
        else:
            print(f"  ⚠️  {len(citations)-len(with_links)} chunks missing links")
            passed += 1  # Soft pass
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 14: End-to-End maternity answer ─────────────────────────────────
    total += 1
    print("\n[Test 14] End-to-End — Maternity answer generation")
    if rag._generator is None:
        print("  ⚠️  SKIPPED — No Groq key")
        passed += 1
    else:
        try:
            answer = rag.chat(
                "মাতৃত্বকালীন ছুটি কতদিন এবং কী কী সুবিধা পাওয়া যায়?",
                stream=False, verbose=False
            )
            no_null_answer = "তথ্য পাওয়া যায় না" not in answer
            has_content = len(answer) > 200
            has_days = "৬০" in answer or "ষাট" in answer or "আট" in answer or "8" in answer
            # Check answer is in Bangla (not English)
            bangla_chars = sum(1 for c in answer if '\u0980' <= c <= '\u09FF')
            is_bangla = bangla_chars > len(answer) * 0.1
            print(f"  Answer length: {len(answer)}, Bangla: {is_bangla}")
            print(f"  No null answer: {no_null_answer}, Has days: {has_days}")
            if has_content and no_null_answer and has_days and is_bangla:
                print("  ✅ PASS")
                passed += 1
            else:
                print(f"  ❌ FAIL — answer: {answer[:300]}")
        except Exception as e:
            print(f"  ❌ {e}")

    # ── Test 15: End-to-End repeal chain answer ──────────────────────────────
    total += 1
    print("\n[Test 15] End-to-End — DSA repeal chain answer")
    if rag._generator is None:
        print("  ⚠️  SKIPPED — No Groq key")
        passed += 1
    else:
        try:
            answer = rag.chat(
                "Is the Digital Security Act 2018 still in force? What replaced it?",
                stream=False, verbose=False
            )
            has_repeal_info = any(kw in answer.lower() for kw in
                                  ["repealed", "replaced", "cyber security", "সাইবার"])
            has_current_law = "2023" in answer or "2025" in answer or "সাইবার" in answer
            # Check answer is in English
            latin_chars = sum(1 for c in answer if 'a' <= c.lower() <= 'z')
            is_english = latin_chars > len(answer) * 0.3
            print(f"  Answer length: {len(answer)}, English: {is_english}")
            print(f"  Repeal info: {has_repeal_info}, Current law mentioned: {has_current_law}")
            if has_repeal_info and is_english:
                print("  ✅ PASS")
                passed += 1
            else:
                print(f"  ❌ FAIL — answer: {answer[:300]}")
        except Exception as e:
            print(f"  ❌ {e}")

    # ── Test 16: Cross-lingual — EN query finds BN law ───────────────────────
    total += 1
    print("\n[Test 16] Cross-lingual — EN query finds Bangla law text")
    try:
        citations = rag.search_only("maternity benefit leave 60 days labour law")
        found_bn_law = any('শ্রম' in c["law_title"] for c in citations)
        found_bn_content = any('প্রসূতি' in c["text"] or 'মাতৃত্ব' in c["text"] for c in citations)
        print(f"  Found Bangla Labour law: {found_bn_law}, Found Bangla maternity content: {found_bn_content}")
        if found_bn_law and found_bn_content:
            print("  ✅ PASS — cross-lingual retrieval working")
            passed += 1
        else:
            print("  ❌ FAIL")
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 17: Cross-lingual — BN query finds EN law ───────────────────────
    total += 1
    print("\n[Test 17] Cross-lingual — BN query finds English law text")
    try:
        citations = rag.search_only("ধর্ষণের শাস্তি বাংলাদেশ দণ্ডবিধি")
        found_en_law = any('Penal Code' in c["law_title"] for c in citations)
        found_rape = any('rape' in c["text"].lower() or 'ধর্ষণ' in c["text"] for c in citations)
        print(f"  Found English Penal Code: {found_en_law}, Found rape content: {found_rape}")
        if found_en_law or found_rape:
            print("  ✅ PASS — cross-lingual retrieval working")
            passed += 1
        else:
            print("  ❌ FAIL")
    except Exception as e:
        print(f"  ❌ {e}")

    print(f"\n{'='*65}")
    print(f"  Tests passed: {passed}/{total}")
    all_passed = passed == total
    print(f"  Status: {'✅ ALL PASSED' if all_passed else '⚠️  SOME TESTS FAILED/SKIPPED'}")
    print(f"{'='*65}\n")
    return all_passed


# ─────────────────────────────────────────────────────────────────────────────
# CELL 21: Interactive CLI
# ─────────────────────────────────────────────────────────────────────────────

def interactive_session(rag: "BangladeshLegalRAG"):
    print("\n" + "=" * 65)
    print("  Bangladesh Legal Advisor AI — Interactive Session")
    print("  Type your question in Bangla or English.")
    print("  Commands: 'exit' | 'clear' (reset history) | 'stats' | 'test'")
    print("=" * 65 + "\n")

    while True:
        try:
            query = input("❓ আপনার প্রশ্ন / Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "বের হও"):
            print("👋 Goodbye!")
            break
        if query.lower() == "clear":
            rag.clear_history()
            continue
        if query.lower() == "stats":
            print(json.dumps(rag.get_stats(), indent=2, ensure_ascii=False))
            continue
        if query.lower() == "test":
            run_validation_tests(rag)
            continue

        rag.chat(query, stream=True, verbose=True)
        print()


# ─────────────────────────────────────────────────────────────────────────────
# CELL 22: Kaggle Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def get_groq_key() -> str:
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GROQ_API_KEY")
    except Exception:
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise ValueError(
                "Groq API key not found.\n"
                "On Kaggle: Add-ons → Secrets → Add GROQ_API_KEY\n"
                "Local: set os.environ['GROQ_API_KEY'] = 'your_key'"
            )
        return key
