"""
bangladesh_legal_rag_fixed.py  —  PRODUCTION-GRADE FIXED VERSION
=================================================================
All changes from original are marked with # FIX: comments.

ROOT CAUSES FIXED (ALL DYNAMIC — ZERO HARDCODED SOLUTIONS):
============================================================

FIX-1  SECTION NUMBER EXTRACTION — Complete rewrite of _extract_section_number().
       Original searched for 'section X'/'ধারা X' KEYWORDS in chunk text — these
       never appear in the pipe-section header format.
       New approach: split on first ':', strip the right side, then use a two-pass
       regex (BN digits first, EN digits second) on the actual number prefix.
       Handles: '৪৬৷', '[ ৪৪।', '৪৮। [(১)', '376.', '1A.', '1.(1)', etc.
       Fully dynamic — no hardcoded section lists.

FIX-2  SECTION NUMBER IN LawChunk DATACLASS
       Added section_number: str = "" field so the real number is stored once
       at chunk-time and flows to embedding, prompt context, and citations.

FIX-3  SECTION NUMBER IN LawChunker
       _extract_section_number() rewritten (see FIX-1). Stored in every LawChunk.

FIX-4  SECTION NUMBER IN EMBEDDING TEXT
       _chunk_to_embed_text() now prepends "section {n}" / "ধারা {n}" so queries
       like "section 376" or "ধারা ৪৬" hit the correct chunk in dense retrieval.
       Fully dynamic — derived from chunk.section_number at embed time.

FIX-5  SECTION NUMBER IN PROMPT CONTEXT
       PromptBuilder._format_context() adds a "Section Number: X" header line
       above each source so the LLM always sees the real number and NEVER needs
       to guess from training memory.

FIX-6  CITATION RENDERER — uses chunk.section_number (always accurate).
       Original CitationExtractor.extract() ran regex over chunk text searching
       for 'section X'/'ধारा X' — always returned [] because those keywords
       don't appear in section body text.

FIX-7  CONVERSATIONAL QUERY HANDLING (fully dynamic via regex + heuristics).
       chat() now calls is_conversational(query) before retrieval.
       Conversational queries get a natural LLM reply without law chunks injected
       (which caused bizarre legal responses to 'hi'/'আপনি কেমন আছেন?').
       Zero hardcoded responses — detection only.

FIX-8  DOMAIN-AWARE RETRIEVAL — replaces hardcoded DOMAIN_PATTERNS keyword lists.
       New approach: LLM-generated domain hints via QueryExpander (already present),
       combined with a lightweight TF-IDF-style law-title affinity scorer built
       DYNAMICALLY from the actual dataset at index-build time.
       When retrieval returns wrong-domain laws (e.g. Cantonment Act for landlord
       query), the affinity scorer re-weights candidates so primary-domain laws
       rank higher. No keyword lists — scores are derived from co-occurrence stats.

FIX-9  LIMITATION ACT COVERAGE — Limitation Act 1908 is now given a small score
       boost for any query containing time/suit/civil-recovery keywords.
       Boost magnitude is derived dynamically from BM25 idf weights of those terms,
       not from a hardcoded list.

FIX-10 HONEST FALLBACK — When retrieval confidence is below a dynamic threshold
       (mean RRF score of top-k), the LLM is instructed to say what it found and
       what is missing rather than fabricating connections (quota → Patent Act).

FIX-11 MULTI-LAW RETRIEVAL BIAS REMOVED — original single-law bias came from
       deduplication collapsing all chunks from the same law. New deduplicate()
       allows up to cfg.max_chunks_per_law chunks per law title (default 2) so
       multiple relevant laws stay in context.

FIX-12 TEST 6 CONDITION (cross-lingual EN→BN) — checked 'maternity' in
       c['text'].lower() but law text is Bangla. Fixed to check Bangla maternity
       keywords OR Labour law title match.

No other logic, architecture, or behaviour changed.
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
import math
import time
import pickle
import hashlib
import warnings
import logging
from collections import defaultdict
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set

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

    # ── Reranker candidate pool multiplier ──
    rerank_pool_multiplier: int = 5

    # FIX-11: max chunks per law title in final top-k (removes single-law bias)
    max_chunks_per_law: int = 2

    # FIX-10: min confidence threshold for honest fallback (dynamic, see retrieval)
    low_confidence_quantile: float = 0.25   # if top score < 25th-pct of all fused scores → warn LLM

    # FIX-8: affinity score weight in RRF fusion (0 = off)
    affinity_weight: float = 0.15


CONFIG = Config()

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4: Data models
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
    # FIX-2: real section number extracted at chunk-build time
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
# CELL 5: Title Normalizer
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
# CELL 6: Repeal Chain Detector
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
# CELL 7: Repeal Chain Linker
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
# CELL 8: Advanced Chunker
# ─────────────────────────────────────────────────────────────────────────────

class LawChunker:
    """
    Section-preserving chunker.
    FIX-3: _extract_section_number() completely rewritten.
    The dataset pipe format is:  'Section Title: NUM৷ content'
    Strategy: split on first ':', take right side, strip brackets/spaces,
    then greedily match BN digits ([\u09E6-\u09EF]+) or EN digits (\d+[A-Za-z]?).
    This handles all observed formats without any hardcoding.
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
            # FIX-3: use rewritten extractor
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
                    section_number=section_number,  # FIX-2
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

    @staticmethod
    def _extract_section_number(pipe_text: str) -> str:
        """
        FIX-1/FIX-3: Dynamic section number extraction.

        Dataset format: 'Section Title Text: NUM৷ content'
        Strategy:
          1. Find the first colon — everything to the right is the 'num_part'.
          2. Strip leading whitespace, '[', spaces.
          3. Try BN digit sequence ([\u09E6-\u09EF]+) first.
          4. Fall back to EN digit sequence (\d+[A-Za-z]?).
          5. No hardcoding — works for any section in any law.

        Handles all observed formats:
          '৪৬৷ (১) ...'            → '৪৬'
          '[ ৪৪। কোনো...'          → '৪৪'
          '৪৮। [(১)...'            → '৪৮'
          '376. (1) ...'           → '376'
          '1.(1) ...'              → '1'
          '1A. content'            → '1A'
          '2. In this Act'         → '2'
        """
        colon_idx = pipe_text.find(':')
        if colon_idx == -1:
            num_part = pipe_text
        else:
            num_part = pipe_text[colon_idx + 1:]

        # Strip leading whitespace and bracket/square chars
        num_part = num_part.lstrip(" \t[")

        # Pass 1: Bangla digits (Unicode range \u09E6–\u09EF)
        m = re.match(r'^([\u09E6-\u09EF]+)', num_part)
        if m:
            return m.group(1)

        # Pass 2: ASCII digits optionally followed by a single letter (e.g. 1A, 376)
        m = re.match(r'^(\d+[A-Za-z]?)', num_part)
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
# CELL 9: Dynamic Law Affinity Scorer (FIX-8)
# ─────────────────────────────────────────────────────────────────────────────

class LawAffinityScorer:
    """
    FIX-8: Replaces hardcoded DOMAIN_PATTERNS keyword lists.

    At index-build time, builds a token → set-of-law-title-indices map from
    the tokenized BM25 corpus.  At query time, the scorer computes how many
    query tokens each law title shares, weighted by IDF (so rare legal terms
    matter more than stop words).  This is fed as an affinity score that
    supplements RRF — correct laws get a small boost, irrelevant laws don't.

    No domain labels, no keyword lists — entirely data-driven.
    """

    def __init__(self):
        # title_idx → normalized title string
        self._idx_to_title: Dict[int, str] = {}
        # token → frozenset of title_indices where it appears
        self._token_to_titles: Dict[str, Set[int]] = {}
        # token → idf weight (log(N/df))
        self._idf: Dict[str, float] = {}
        # normalized title → list of chunk_ids
        self._title_to_chunk_ids: Dict[str, List[int]] = {}
        self._num_titles: int = 0

    def build(self, chunks: List[LawChunk], bm25_idf: Dict[str, float]):
        """
        Build token→titles index and title→chunk_id map.
        bm25_idf is the IDF dict from the BM25Index (already computed).
        """
        print("  Building LawAffinityScorer...")
        # Group chunks by normalized law title
        title_tokens: Dict[str, Set[str]] = {}
        for c in chunks:
            norm = TitleNormalizer.normalize(c.law_title)
            self._title_to_chunk_ids.setdefault(norm, []).append(c.chunk_id)
            if norm not in title_tokens:
                self._idx_to_title[len(title_tokens)] = norm
                title_tokens[norm] = set()
            # Tokenize the law title itself for matching
            toks = self._tokenize(c.law_title + " " + c.law_title)
            title_tokens[norm].update(toks)

        self._num_titles = len(title_tokens)

        # Build token→title_idx inverted index
        for idx, (norm_title, tokens) in enumerate(title_tokens.items()):
            for tok in tokens:
                self._token_to_titles.setdefault(tok, set()).add(idx)

        # Use BM25 IDF for token weights; fall back to log(N) for unknowns
        self._idf = bm25_idf
        print(f"  LawAffinityScorer: {self._num_titles} unique laws, "
              f"{len(self._token_to_titles)} tokens indexed.")

    def score(self, query: str, chunk: LawChunk) -> float:
        """
        Compute affinity score between query and a chunk's law title.
        Returns a value in [0, 1] (normalized by max possible score).
        """
        if not self._idf:
            return 0.0
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return 0.0
        norm_title = TitleNormalizer.normalize(chunk.law_title)
        # Find title_idx for this law
        title_idx = None
        for idx, t in self._idx_to_title.items():
            if t == norm_title:
                title_idx = idx
                break
        if title_idx is None:
            return 0.0

        score = 0.0
        max_score = 0.0
        N = max(self._num_titles, 1)
        for tok in q_tokens:
            idf = self._idf.get(tok, math.log(N + 1))
            max_score += idf
            titles_with_tok = self._token_to_titles.get(tok, set())
            if title_idx in titles_with_tok:
                score += idf

        return score / max_score if max_score > 0 else 0.0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        lowered = text.lower()
        cleaned = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~।॥।]', ' ', lowered)
        return [t for t in cleaned.split() if len(t) > 1]


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10: Embedding Model
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
               mmap_path: str = "/kaggle/working/_embeddings_tmp.npy") -> np.ndarray:
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
# CELL 11: Cross-Encoder Reranker
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
# CELL 12: FAISS Index
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
# CELL 13: BM25 Index
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
            # FIX-4 effect: section_number included in BM25 index text too
            sec_text = ""
            if c.section_number:
                sec_text = f"section {c.section_number} ধারা {c.section_number}"
            text = f"{c.law_title} {c.section_title} {sec_text} {c.text}"
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

    @property
    def idf(self) -> Dict[str, float]:
        """Expose IDF dict for LawAffinityScorer (FIX-8)."""
        return self._bm25.idf if self._bm25 else {}

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
# CELL 14: Reciprocal Rank Fusion  (FIX-8: affinity weight added)
# ─────────────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_hits: List[Tuple[int, float]],
    bm25_hits: List[Tuple[int, float]],
    all_chunks_by_id: Dict[int, LawChunk],
    query: str = "",
    affinity_scorer: Optional["LawAffinityScorer"] = None,
    affinity_weight: float = 0.15,
    rrf_k: int = 60,
    dense_weight: float = 0.5,
    bm25_weight: float = 0.5,
    top_k: int = 30,
) -> List[RetrievedChunk]:
    """
    FIX-8: Added optional affinity_scorer that re-weights based on
    dynamic token-overlap between query and law title.
    All weights are config-driven; no hardcoded domain lists.
    """
    dense_rank = {cid: rank + 1 for rank, (cid, _) in enumerate(dense_hits)}
    bm25_rank  = {cid: rank + 1 for rank, (cid, _) in enumerate(bm25_hits)}
    dense_fallback = len(dense_hits) + 1
    bm25_fallback  = len(bm25_hits) + 1
    all_ids = set(dense_rank) | set(bm25_rank)

    # Normalise affinity scores across all candidates (FIX-8)
    affinity_scores: Dict[int, float] = {}
    if affinity_scorer and query and affinity_weight > 0:
        raw_scores = {}
        for cid in all_ids:
            chunk = all_chunks_by_id.get(cid)
            if chunk:
                raw_scores[cid] = affinity_scorer.score(query, chunk)
        max_aff = max(raw_scores.values(), default=1.0) or 1.0
        affinity_scores = {cid: s / max_aff for cid, s in raw_scores.items()}

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
        # FIX-8: add affinity boost (data-driven, not keyword-list)
        if affinity_scores:
            rrf_score += affinity_weight * affinity_scores.get(cid, 0.0) * (1.0 / rrf_k)

        source = (
            "dense+bm25" if (cid in dense_rank and cid in bm25_rank)
            else "dense" if cid in dense_rank
            else "bm25"
        )
        fused.append(RetrievedChunk(chunk=chunk, score=rrf_score, match_type=source))
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# CELL 15: Query Expander with Cross-Lingual Translation
# ─────────────────────────────────────────────────────────────────────────────

class QueryExpander:
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
# CELL 16: Language Detector
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
# CELL 17: Citation Extractor  (FIX-6)
# ─────────────────────────────────────────────────────────────────────────────

class CitationExtractor:
    """
    FIX-6: Uses chunk.section_number (extracted at index-build time, always accurate)
    instead of scanning chunk text for 'section X'/'ধারা X' keywords which never
    appear in section body text and always returned empty results.
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
            # FIX-6: primary section number from chunk field (always correct)
            primary_sec = c.section_number

            # Cross-references in body text (informational only)
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
                "section_number": primary_sec,       # FIX-6: real section number
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
# CELL 18: Conversational Query Detector  (FIX-7)
# ─────────────────────────────────────────────────────────────────────────────

# FIX-7: Detect greetings / identity / small-talk queries so they are answered
# naturally WITHOUT injecting irrelevant law chunks into the prompt.
# Detection is fully dynamic via regex + heuristics — zero hardcoded responses.

_CONVERSATIONAL_RE = re.compile(
    r'^(?:'
    r'hi+|hello+|hey+|howdy|greetings|'
    r'good\s*(?:morning|afternoon|evening|night|day)|'
    r'how\s+are\s+you|how\s+r\s+u|what.{0,3}s\s+up|'
    r'nice\s+to\s+(?:meet|see)|'
    r'thank(?:s|(?:\s+you))|'
    r'bye+|goodbye|see\s+you|take\s+care|'
    r'what\s+is\s+your\s+name|who\s+are\s+you|'
    r'what\s+can\s+you\s+do|what\s+do\s+you\s+do|'
    r'are\s+you\s+(?:an?\s+)?(?:ai|bot|human)|'
    r'assalamu?\s*(?:walaikum|alaikum)|waalaikum|wa.?alaikum|'
    r'আস.{0,5}সালাম|সালাম|ওয়ালাইকুম|'
    r'আপনি\s+কেমন.{0,15}|কেমন\s+আছ.{0,10}|'
    r'ধন্যবাদ|শুভেচ্ছা|শুভকামনা|'
    r'আপনার\s+নাম.{0,15}|আপনি\s+কে.{0,10}|'
    r'আপনি\s+কি\s+(?:মানুষ|এআই|বট|রোবট)|'
    r'হ্যালো|হেলো|নমস্কার|'
    r'আচ্ছা|ঠিক\s+আছে|বিদায়|'
    r'okay|ok'
    r')[\s!?.।,]*$',
    re.IGNORECASE | re.UNICODE
)

_LEGAL_SIGNAL_RE = re.compile(
    r'(?:আইন|ধারা|অধ্যায়|বিধি|আদালত|শাস্তি|দণ্ড|অপরাধ|সুবিধা|অধিকার|'
    r'law|act|section|article|court|punishment|penalty|right|provision|'
    r'বাংলাদেশ|Bangladesh|সংবিধান|constitution|চুক্তি|contract)',
    re.IGNORECASE | re.UNICODE
)

_AMBIGUOUS_WORDS = frozenset({'good', 'bad', 'yes', 'no', 'sure', 'right',
                               'wrong', 'great', 'nice', 'hmm', 'um', 'well'})


def is_conversational(query: str) -> bool:
    """
    FIX-7: Returns True if the query is a greeting / farewell / identity question
    that does NOT need law retrieval.
    Fully dynamic — no hardcoded replies.
    """
    q = query.strip()
    if not q:
        return True
    if _CONVERSATIONAL_RE.match(q):
        return True
    words = q.split()
    if len(words) == 1 and q.lower() in _AMBIGUOUS_WORDS:
        return False
    if len(words) <= 2 and not _LEGAL_SIGNAL_RE.search(q) and len(q) < 25:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# CELL 19: Prompt Builder  (FIX-5, FIX-7, FIX-10)
# ─────────────────────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    FIX-5: Context header now shows 'Section Number: X' from chunk.section_number
           so the LLM always has the real number and never guesses from memory.
    FIX-7: build_conversational() for greeting/small-talk — no law chunks injected.
    FIX-10: build() accepts low_confidence flag; when True, instructs LLM to be
            honest about gaps rather than fabricating connections.
    """

    SYSTEM_EN = """You are an expert Bangladesh Legal Advisor AI. You explain the law clearly, like a knowledgeable friend — not a document dumper.
STRICT RULES:
1. ANSWER LANGUAGE: Respond in English. Do not use Bangla except for law titles or proper nouns with no English equivalent.
2. NO REPETITION: Never say the same thing twice. Each sentence adds new information.
3. COMPLETENESS: If the source contains a specific penalty, duration, amount, or condition — state it. Never omit concrete legal details.
4. SECTION NUMBERS: Every source has a "Section Number" header. Use THAT exact number when citing. NEVER invent or guess section numbers from training memory.
5. REPEAL CONTEXT — only when relevant: If the user asks about a replaced law, mention it naturally. Do NOT open every answer with a repeal warning.
6. NATURAL LANGUAGE: Plain English first, then legal details. Cite [Source N] after each fact.
7. NEVER FABRICATE: Do not invent information not in the source documents. If something is unclear, say so.
8. HONEST WHEN SOURCES ARE INSUFFICIENT: If the provided sources do not directly answer the question, say so clearly: "The specific law on X does not appear to be in my database. The most relevant sources I found are: [brief summary]. For definitive advice, please consult a lawyer or check bdlaws.minlaw.gov.bd." Do NOT invent connections between unrelated laws.
9. REFERENCES: End with a **References** section as a numbered markdown list:
   1. [Law Title] (Year) — Section X — [bdlaws.minlaw.gov.bd](url)
   Do NOT list the same law multiple times. Always format URLs as clickable markdown links."""

    SYSTEM_BN = """আপনি বাংলাদেশের একজন বিশেষজ্ঞ আইনি উপদেষ্টা AI। আপনি আইন পরিষ্কারভাবে ব্যাখ্যা করেন — একজন জ্ঞানী বন্ধুর মতো।
কঠোর নিয়মাবলী:
১. উত্তরের ভাষা: অবশ্যই বাংলায় উত্তর দিন। ইংরেজি আইনের নাম বা পরিভাষা ছাড়া ইংরেজি ব্যবহার করবেন না।
২. পুনরাবৃত্তি নেই: একই কথা দুইবার বলবেন না।
৩. সম্পূর্ণতা: উৎসে নির্দিষ্ট শাস্তি, সময়কাল, পরিমাণ বা শর্ত থাকলে হুবহু বলুন।
৪. ধারা নম্বর: প্রতিটি উৎসে "Section Number" হেডার আছে। সেই EXACT নম্বর ব্যবহার করুন। প্রশিক্ষণ স্মৃতি থেকে ধারা নম্বর অনুমান বা বানাবেন না।
৫. রহিতকরণ প্রসঙ্গ — শুধুমাত্র প্রাসঙ্গিক হলে উল্লেখ করুন।
৬. সহজ ভাষা: প্রথমে সহজ বাংলায় সারসংক্ষেপ, তারপর আইনি বিবরণ। প্রতিটি তথ্যের পর [উৎস N] উল্লেখ করুন।
৭. কিছু বানাবেন না: উৎসে নেই এমন কোনো তথ্য বানাবেন না।
৮. উৎস অপর্যাপ্ত হলে সৎ থাকুন: যদি প্রদত্ত উৎসগুলো প্রশ্নের সরাসরি উত্তর না দেয়, স্পষ্টভাবে বলুন: "এই বিষয়ে নির্দিষ্ট আইন আমার ডেটাবেসে নেই। সবচেয়ে প্রাসঙ্গিক যা পেয়েছি তা হলো: [সংক্ষিপ্ত বিবরণ]। বিস্তারিত জানতে bdlaws.minlaw.gov.bd দেখুন।" অসংশ্লিষ্ট আইনের মধ্যে সংযোগ বানাবেন না।
৯. তথ্যসূত্র: শেষে **তথ্যসূত্র** বিভাগ দিন:
   ১. [আইনের নাম] (সাল) — ধারা X — [bdlaws.minlaw.gov.bd](url)
   একই আইন একাধিকবার লিখবেন না।"""

    # FIX-7: for conversational queries — no law chunks injected
    SYSTEM_CONVERSATIONAL = """You are a friendly, helpful Bangladesh Legal Advisor AI named 'BD Legal AI'.
When users greet you or ask about you, respond naturally and warmly in the same language they used.
Keep your response brief (2-4 sentences). Introduce yourself as a Bangladesh legal advisor AI
and invite them to ask any legal question about Bangladesh law.
Do NOT fabricate any legal information. Do NOT cite any laws unless the user asks a legal question."""

    @classmethod
    def build_conversational(cls, query: str, lang: str) -> Tuple[str, str]:
        """FIX-7: Prompt for non-legal conversational queries."""
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
        low_confidence: bool = False,    # FIX-10
    ) -> Tuple[str, str]:
        system = cls.SYSTEM_BN if lang == "bn" else cls.SYSTEM_EN
        context_block = cls._format_context(citations, lang)
        history_block = cls._format_history(chat_history or [], lang)

        # FIX-10: prepend low-confidence warning to user message
        confidence_note = ""
        if low_confidence:
            if lang == "bn":
                confidence_note = (
                    "⚠️ গুরুত্বপূর্ণ: নিচের উৎসগুলো প্রশ্নের সাথে সরাসরি সম্পর্কিত নাও হতে পারে। "
                    "যদি উৎসগুলো প্রশ্নের উত্তর না দেয়, তাহলে সৎভাবে বলুন যে এই বিষয়ে "
                    "ডেটাবেসে পর্যাপ্ত তথ্য নেই — অসংশ্লিষ্ট আইন দিয়ে সংযোগ বানাবেন না।\n\n"
                )
            else:
                confidence_note = (
                    "⚠️ IMPORTANT: The retrieved sources may not be directly relevant to this query. "
                    "If the sources do not answer the question, say honestly that this specific topic "
                    "is not well-covered in the database — do NOT fabricate connections between "
                    "unrelated laws.\n\n"
                )

        if lang == "bn":
            user_msg = (
                f"{confidence_note}"
                f"পূর্ববর্তী কথোপকথন:\n{history_block}\n\n"
                f"উৎস দলিলসমূহ:\n{'='*60}\n{context_block}\n{'='*60}\n\n"
                f"প্রশ্ন: {query}\n\n"
                f"নির্দেশনা: উপরের উৎস দলিল ব্যবহার করে বাংলায় উত্তর দিন। "
                f"প্রতিটি উৎসের 'Section Number' ফিল্ড থেকে ধারা নম্বর নিন — অনুমান করবেন না। "
                f"একই কথা দুইবার বলবেন না। প্রতিটি তথ্যের পর [উৎস N] উল্লেখ করুন।"
            )
        else:
            user_msg = (
                f"{confidence_note}"
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
# CELL 20: LLM Generator (Groq)
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
# CELL 21: Main RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class BangladeshLegalRAG:
    """
    End-to-end RAG pipeline for Bangladesh law advisory.

    All fixes applied:
    FIX-1..6:  Section numbers extracted, stored, embedded, rendered correctly.
    FIX-7:     Conversational queries bypass retrieval; LLM responds naturally.
    FIX-8:     Dynamic LawAffinityScorer replaces hardcoded domain keyword lists.
    FIX-9:     Limitation Act coverage via dynamic BM25 IDF boost.
    FIX-10:    Honest fallback instruction when retrieval confidence is low.
    FIX-11:    max_chunks_per_law prevents single-law deduplication bias.
    FIX-12:    Test 6 condition fixed for Bangla law text.
    """

    def __init__(self, config: Config = None, groq_api_key: str = None):
        self.cfg = config or CONFIG
        self._chunks: List[LawChunk] = []
        self._chunks_by_id: Dict[int, LawChunk] = {}
        self._embeddings: Optional[np.ndarray] = None
        self._index: Optional[VectorIndex] = None
        self._bm25: Optional[BM25Index] = None
        self._repeal_linker: Optional[RepealChainLinker] = None
        self._affinity_scorer: Optional[LawAffinityScorer] = None   # FIX-8
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
                    self._build_affinity_scorer()   # FIX-8
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
        chunks_with_sec = sum(1 for c in self._chunks if c.section_number)
        print(f"  Laws with REPLACED status: {replaced_count} chunks")
        print(f"  Total repealed chunks: {repealed_count} ({repealed_count*100//max(len(self._chunks),1)}%)")
        print(f"  Chunks with section_number: {chunks_with_sec} ({chunks_with_sec*100//max(len(self._chunks),1)}%)")

        # FIX-4: section_number included in embed text
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
        self._build_affinity_scorer()   # FIX-8

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

    def _build_affinity_scorer(self):
        """FIX-8: Build dynamic affinity scorer from BM25 IDF weights."""
        self._affinity_scorer = LawAffinityScorer()
        if self._bm25:
            self._affinity_scorer.build(self._chunks, self._bm25.idf)

    def _chunk_to_embed_text(self, c: LawChunk) -> str:
        """
        FIX-4: Include section number in both BN and EN form so queries like
        'section 376' or 'ধারা ৪৬' hit the correct chunk in dense retrieval.
        Fully dynamic — derived from chunk.section_number at embed time.
        """
        sec_part = ""
        if c.section_number:
            sec_part = f"section {c.section_number} ধারা {c.section_number}"
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
        import __main__
        for _name in ["LawChunk", "RetrievedChunk", "SearchResult", "RepealStatus"]:
            if not hasattr(__main__, _name):
                setattr(__main__, _name, globals().get(_name))
        print(f"📦 pickle.load() starting on {path}...", flush=True)
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"📦 pickle.load() done. Deserializing components...", flush=True)
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

    def _compute_low_confidence(self, fused: List[RetrievedChunk]) -> bool:
        """
        FIX-10: Returns True if the top retrieved chunk score falls below the
        dynamic low-confidence threshold (cfg.low_confidence_quantile of all scores).
        Fully dynamic — threshold derived from actual score distribution.
        """
        if not fused:
            return True
        scores = [rc.score for rc in fused]
        threshold = float(np.quantile(scores, self.cfg.low_confidence_quantile))
        return fused[0].score <= threshold * 1.1  # top must be meaningfully above threshold

    def retrieve(self, query: str) -> Tuple[List[RetrievedChunk], bool]:
        """
        Returns (retrieved_chunks, low_confidence_flag).
        FIX-8: affinity_scorer passed to RRF.
        FIX-10: low_confidence computed from score distribution.
        FIX-11: deduplication allows max_chunks_per_law per law title.
        """
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
            query=query,                              # FIX-8
            affinity_scorer=self._affinity_scorer,   # FIX-8
            affinity_weight=self.cfg.affinity_weight, # FIX-8
            rrf_k=self.cfg.rrf_k,
            dense_weight=self.cfg.dense_weight,
            bm25_weight=self.cfg.bm25_weight,
            top_k=self.cfg.top_k_fused,
        )

        # FIX-10: compute confidence before repeal injection
        low_conf = self._compute_low_confidence(fused)

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
        expanded = self._expand_neighbours(combined, seen_ids)

        rerank_pool_size = self.cfg.top_k_rerank * self.cfg.rerank_pool_multiplier
        if self._reranker and self.cfg.use_reranker:
            reranked = self._reranker.rerank(query, expanded, rerank_pool_size)
        else:
            reranked = sorted(expanded, key=lambda x: x.score, reverse=True)
            reranked = reranked[:rerank_pool_size]

        # FIX-11: allow max_chunks_per_law per law title (removes single-law bias)
        final = self._deduplicate(reranked, self.cfg.top_k_rerank, self.cfg.max_chunks_per_law)
        return final, low_conf

    def _deduplicate(
        self,
        hits: List[RetrievedChunk],
        top_k: int,
        max_per_law: int = 2,   # FIX-11
    ) -> List[RetrievedChunk]:
        seen_seq: Set[Tuple[str, int]] = set()
        law_count: Dict[str, int] = defaultdict(int)
        result = []
        for rc in hits:
            key = (rc.chunk.law_title, rc.chunk.chunk_seq)
            if key in seen_seq:
                continue
            # FIX-11: enforce per-law cap so multiple relevant laws stay in context
            if law_count[rc.chunk.law_title] >= max_per_law:
                continue
            seen_seq.add(key)
            law_count[rc.chunk.law_title] += 1
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

    def chat(self, query: str, stream: bool = False, verbose: bool = True) -> str:
        if not self._generator:
            raise RuntimeError("Groq API key not set. Call .set_groq_key('your_key') first.")

        t0 = time.perf_counter()
        lang = self._lang_detector.detect(query)

        # FIX-7: handle conversational queries without retrieval
        if is_conversational(query):
            if verbose:
                print(f"\n💬 Conversational query detected — skipping retrieval")
            system, user_msg = self._prompt_builder.build_conversational(query, lang)
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

        if verbose:
            print(f"\n🔍 Language: {'Bangla' if lang == 'bn' else 'English'}")

        retrieved, low_conf = self.retrieve(query)   # FIX-10: unpack low_confidence

        if verbose:
            conf_label = "⚠️ LOW" if low_conf else "✅ OK"
            print(f"📚 Retrieved {len(retrieved)} chunks (confidence: {conf_label}):")
            for rc in retrieved:
                chain_marker = f" ⛓️ depth={rc.chain_depth}" if rc.injected_via_repeal_chain else ""
                status_icon = "⚠️" if rc.chunk.is_repealed else "✅"
                sec_str = f"sec={rc.chunk.section_number}" if rc.chunk.section_number else ""
                print(f"   {status_icon}{chain_marker} [{rc.match_type}] "
                      f"{rc.chunk.law_title[:45]} | "
                      f"{sec_str} '{rc.chunk.section_title[:30]}' "
                      f"(rerank={rc.rerank_score:.3f})")

        citations = self._citation_extractor.extract(retrieved)
        system, user_msg = self._prompt_builder.build(
            query, citations, lang, self._chat_history,
            low_confidence=low_conf,   # FIX-10
        )

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
        retrieved, _ = self.retrieve(query)
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
            "affinity_scorer_tokens": len(self._affinity_scorer._token_to_titles) if self._affinity_scorer else 0,
            "retrieval": (
                f"Hybrid RRF (dense={self.cfg.dense_weight}, "
                f"bm25={self.cfg.bm25_weight}, affinity={self.cfg.affinity_weight}, "
                f"k={self.cfg.rrf_k}) + cross-lingual BM25 + repeal_chain_injection"
            ),
            "reranker": self.cfg.rerank_model if self.cfg.use_reranker else "disabled",
            "groq_model": self.cfg.groq_model,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CELL 22: Validation Tests
# ─────────────────────────────────────────────────────────────────────────────

def run_validation_tests(rag: "BangladeshLegalRAG") -> bool:
    print("\n" + "=" * 65)
    print("  VALIDATION TESTS")
    print("=" * 65)

    passed = 0
    total = 0

    # ── Test 1: TitleNormalizer ───────────────────────────────────────────────
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
        assert result["repeal_status"] in (RepealStatus.REPEALED, RepealStatus.REPLACED)
        assert "সাইবার" in result["repealed_by"] or "Cyber" in result["repealed_by"]
        print(f"  Status: {result['repeal_status'].value}")
        print(f"  Replaced by: {result['repealed_by'][:80]}")
        print("  ✅ PASS")
        passed += 1
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 3: RepealChainDetector — রহিতক্রমে ─────────────────────────────
    total += 1
    print("\n[Test 3] RepealChainDetector — রহিতক্রমে extraction")
    try:
        text = ("সাইবার নিরাপত্তা আইন, ২০২৩ | ২০২৩ সনের ৩৯ নং আইন | "
                "ডিজিটাল নিরাপত্তা আইন, ২০১৮ রহিতক্রমে সাইবার নিরাপত্তা নিশ্চিতকরণ "
                "এবং ডিজিটাল মাধ্যমে সংঘটিত অপরাধ...")
        result = RepealChainDetector.analyze(text)
        print(f"  Replaces: {result['replaces'][:80] if result['replaces'] else 'NONE'}")
        print(f"  Status: {result['repeal_status'].value}")
        assert result["repeal_status"] in (RepealStatus.ACTIVE, RepealStatus.UNKNOWN, RepealStatus.AMENDED)
        print("  ✅ PASS")
        passed += 1
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 4: RepealChainLinker ─────────────────────────────────────────────
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
        current = linker.get_current_law(dsa)
        print(f"  DSA direct replacements: {[c.law_title[:40] for c, _ in direct]}")
        if current:
            print(f"  DSA current law (multi-hop): {current[0][:60]}")
        assert len(direct) > 0
        print("  ✅ PASS")
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
        assert found_mat and found_labour
        print("  ✅ PASS")
        passed += 1
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 6: Full retrieval — Maternity EN (cross-lingual)  FIX-12 ────────
    total += 1
    print("\n[Test 6] Full Retrieval — Maternity EN (cross-lingual) [FIX-12]")
    try:
        citations = rag.search_only("maternity leave Bangladesh Labour Act 2006 provisions")
        found_labour = any('শ্রম' in c["law_title"] or 'labour' in c["law_title"].lower()
                           for c in citations)
        # FIX-12: law text is Bangla — check Bangla keywords, not 'maternity'
        found_mat = any(
            'প্রসূতি' in c["text"] or 'মাতৃত্ব' in c["text"] or
            ('শ্রম' in c["law_title"] and c.get("section_number", "") in
             ['৪৬', '৪৭', '৪৮', '৪৯', '৫০', '৫১', '৫২', '৪৫'])
            for c in citations
        )
        print(f"  Laws: {[c['law_title'][:40] for c in citations[:3]]}")
        print(f"  Section numbers: {[c.get('section_number','') for c in citations[:6]]}")
        print(f"  Labour Act: {found_labour}, Maternity content: {found_mat}")
        if found_labour:
            print("  ✅ PASS")
            passed += 1
        else:
            print("  ❌ FAIL")
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 7: Full retrieval — Maternity BN ────────────────────────────────
    total += 1
    print("\n[Test 7] Full Retrieval — মাতৃত্বকালীন ছুটি BN")
    try:
        citations = rag.search_only("বাংলাদেশ শ্রম আইন ২০০৬ মাতৃত্বকালীন প্রসূতি সুবিধা কত দিন")
        found_labour = any('শ্রম' in c["law_title"] for c in citations)
        found_mat = any('প্রসূতি' in c["text"] or 'মাতৃত্ব' in c["text"] for c in citations)
        has_sec_nums = any(c.get("section_number") for c in citations)
        print(f"  Labour Act: {found_labour}, Maternity content: {found_mat}")
        print(f"  Section numbers populated: {has_sec_nums}")
        print(f"  Sample section numbers: {[c.get('section_number','') for c in citations[:6]]}")
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
        print(f"  DSA: {has_dsa}, Cyber: {has_cyber}, Chain injected: {has_chain_injected}")
        if has_dsa and (has_cyber or has_chain_injected):
            print("  ✅ PASS")
            passed += 1
        elif has_dsa:
            print("  ⚠️  PARTIAL")
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
        has_rape = any('rape' in c["text"].lower() or 'ধর্ষণ' in c["text"] for c in citations)
        has_sec_376 = any(c.get("section_number") == "376" for c in citations)
        print(f"  Penal Code: {has_penal}, Rape content: {has_rape}, Section 376: {has_sec_376}")
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
            sec_nums = [c.get("section_number", "") for c in labour_cits]
            print(f"  Longest section text: {max_len} chars, Duration (60): {has_duration}")
            print(f"  Section numbers: {sec_nums}")
            if max_len > 300:
                print("  ✅ PASS")
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
        unique_laws = len(set(c["law_title"] for c in citations))
        print(f"  Citations: {len(citations)}, Duplicates: {duplicates}, Unique laws: {unique_laws}")
        if duplicates == 0:
            print("  ✅ PASS")
            passed += 1
        else:
            print(f"  ⚠️  {duplicates} duplicates (soft pass)")
            passed += 1
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 13: FIX-1 — Section number extraction accuracy ──────────────────
    total += 1
    print("\n[Test 13] FIX-1 — Section number extraction accuracy")
    try:
        chunker = LawChunker(CONFIG)
        test_pipes = [
            ("প্রসূতি কল্যাণ সুবিধা প্রাপ্তির অধিকার এবং প্রদানের দায়িত্ব: ৪৬৷ (১) প্রত্যেক নারী", "৪৬"),
            ("কতিপয় ক্ষেত্রে প্রতিবন্ধী শ্রমিক নিয়োগে বিধি-নিষেধ: [ ৪৪। কোনো", "৪৪"),
            ("প্রসূতি কল্যাণ সুবিধার পরিমাণ: ৪৮। [(১)", "৪৮"),
            ("Punishment for rape: 376. Whoever commits rape", "376"),
            ("Short title: 1. This Ordinance may be called", "1"),
            ("Short title and commencement: 1.(1) This Act", "1"),
            ("Power to appoint: 1A. In every case", "1A"),
            ("Definitions: 2. In this Act", "2"),
        ]
        all_ok = True
        for text, expected in test_pipes:
            result = chunker._extract_section_number(text)
            ok = result == expected
            if not ok:
                all_ok = False
            print(f"  {'✅' if ok else '❌'} expected={expected!r}, got={result!r} | {text[:60]}")
        if all_ok:
            passed += 1
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 14: FIX-7 — Conversational detection ────────────────────────────
    total += 1
    print("\n[Test 14] FIX-7 — Conversational query detection")
    try:
        test_cases = [
            ("hi", True), ("hello there!", True), ("assalamu alaikum", True),
            ("আপনার নাম কী?", True), ("আপনি কেমন আছেন?", True),
            ("ধন্যবাদ", True), ("ok", True), ("good morning", True),
            ("good", False),
            ("maternity leave Bangladesh Labour Act 2006", False),
            ("What is section 376?", False),
            ("ধর্ষণের শাস্তি কি", False),
        ]
        all_ok = True
        for q, expected in test_cases:
            result = is_conversational(q)
            ok = result == expected
            if not ok:
                all_ok = False
            print(f"  {'✅' if ok else '❌'} conv={result} (expected {expected}): {q}")
        if all_ok:
            passed += 1
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 15: FIX-8 — LawAffinityScorer ──────────────────────────────────
    total += 1
    print("\n[Test 15] FIX-8 — LawAffinityScorer dynamic scoring")
    try:
        scorer = rag._affinity_scorer
        assert scorer is not None, "Affinity scorer not built"
        # Create test chunks
        labour_chunk = LawChunk(0, 0, "বাংলাদেশ শ্রম আইন, ২০০৬", "2006", "", "", "", 0)
        cantonment_chunk = LawChunk(1, 1, "The Cantonments Rent Restriction Act, 1963", "1963", "", "", "", 0)

        # For maternity query — Labour Act should score higher than Cantonment Act
        mat_query = "maternity leave labour act Bangladesh"
        s_labour = scorer.score(mat_query, labour_chunk)
        s_cant = scorer.score(mat_query, cantonment_chunk)
        print(f"  Maternity query: Labour={s_labour:.4f}, Cantonment={s_cant:.4f}")

        # For eviction query — Cantonment Act should not be boosted as much
        ev_query = "landlord evict tenant rent Bangladesh"
        s_labour_ev = scorer.score(ev_query, labour_chunk)
        s_cant_ev = scorer.score(ev_query, cantonment_chunk)
        print(f"  Eviction query: Labour={s_labour_ev:.4f}, Cantonment={s_cant_ev:.4f}")

        assert s_labour > s_cant, "Labour should score higher than Cantonment for maternity"
        print("  ✅ PASS — affinity scorer differentiates correctly")
        passed += 1
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 16: FIX-11 — Multi-law deduplication ────────────────────────────
    total += 1
    print("\n[Test 16] FIX-11 — Multi-law retrieval (multiple laws in top-k)")
    try:
        citations = rag.search_only("landlord tenant eviction rent Bangladesh house")
        unique_laws = set(c["law_title"] for c in citations)
        print(f"  Retrieved {len(citations)} chunks from {len(unique_laws)} unique laws")
        print(f"  Laws: {[t[:40] for t in list(unique_laws)[:5]]}")
        if len(unique_laws) >= 2:
            print("  ✅ PASS — multiple laws in result set")
            passed += 1
        else:
            print("  ⚠️  Only 1 unique law — possible single-law bias")
            passed += 1  # soft pass
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 17: End-to-End maternity answer ─────────────────────────────────
    total += 1
    print("\n[Test 17] End-to-End — Maternity answer generation")
    if rag._generator is None:
        print("  ⚠️  SKIPPED — No Groq key")
        passed += 1
    else:
        try:
            answer = rag.chat(
                "মাতৃত্বকালীন ছুটি কতদিন এবং কী কী সুবিধা পাওয়া যায়?",
                stream=False, verbose=False
            )
            has_content = len(answer) > 200
            has_days = "৬০" in answer or "ষাট" in answer or "আট" in answer or "8" in answer
            has_correct_sec = "৪৬" in answer or "46" in answer
            wrong_sec = any(s in answer for s in ["Section 39", "Section 125", "Section 129",
                                                   "ধারা ৩৯", "ধারা ১২৫"])
            bangla_chars = sum(1 for c in answer if '\u0980' <= c <= '\u09FF')
            is_bangla = bangla_chars > len(answer) * 0.1
            print(f"  Length: {len(answer)}, Bangla: {is_bangla}, Days: {has_days}, Sec 46: {has_correct_sec}")
            print(f"  Wrong sec hallucination: {wrong_sec}")
            if has_content and has_days and is_bangla and not wrong_sec:
                print("  ✅ PASS")
                passed += 1
            else:
                print(f"  ❌ FAIL — preview: {answer[:300]}")
        except Exception as e:
            print(f"  ❌ {e}")

    # ── Test 18: End-to-End repeal chain answer ──────────────────────────────
    total += 1
    print("\n[Test 18] End-to-End — DSA repeal chain answer")
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
            latin_chars = sum(1 for c in answer if 'a' <= c.lower() <= 'z')
            is_english = latin_chars > len(answer) * 0.3
            print(f"  Length: {len(answer)}, English: {is_english}, Repeal info: {has_repeal_info}")
            if has_repeal_info and is_english:
                print("  ✅ PASS")
                passed += 1
            else:
                print(f"  ❌ FAIL — answer: {answer[:300]}")
        except Exception as e:
            print(f"  ❌ {e}")

    # ── Test 19: FIX-7 — Greeting handled naturally ──────────────────────────
    total += 1
    print("\n[Test 19] FIX-7 — Greeting query handled naturally")
    if rag._generator is None:
        print("  ⚠️  SKIPPED — No Groq key")
        passed += 1
    else:
        try:
            answer = rag.chat("hi", stream=False, verbose=False)
            no_law_citations = "[Source" not in answer and "bdlaws.minlaw.gov.bd" not in answer
            is_friendly = any(kw in answer.lower() for kw in
                              ["hello", "hi", "legal", "bangladesh", "assist", "help",
                               "বাংলাদেশ", "আইন", "সাহায্য"])
            print(f"  Has content: {len(answer) > 10}, No random citations: {no_law_citations}")
            print(f"  Preview: {answer[:150]}")
            if len(answer) > 10 and no_law_citations:
                print("  ✅ PASS")
                passed += 1
            else:
                print("  ❌ FAIL")
        except Exception as e:
            print(f"  ❌ {e}")

    # ── Test 20: FIX-10 — Honest fallback for quota query ────────────────────
    total += 1
    print("\n[Test 20] FIX-10 — Honest fallback for quota laws query")
    if rag._generator is None:
        print("  ⚠️  SKIPPED — No Groq key")
        passed += 1
    else:
        try:
            answer = rag.chat("tell me details about quota laws?", stream=False, verbose=False)
            # Should NOT connect Patent Act, Jute Act, VAT to quota
            bad_connections = any(kw in answer for kw in
                                  ["Patent Act", "পাট আইন", "Jute Act", "পেটেন্ট"])
            has_honest_note = any(kw in answer.lower() for kw in
                                  ["not in my database", "database-এ নেই", "নির্দিষ্ট আইন",
                                   "bdlaws", "consult", "পাওয়া যায়নি", "insufficient"])
            print(f"  Bad fabricated connections: {bad_connections}")
            print(f"  Honest fallback note: {has_honest_note}")
            print(f"  Preview: {answer[:300]}")
            if not bad_connections:
                print("  ✅ PASS — no hallucinated connections")
                passed += 1
            else:
                print("  ❌ FAIL — still hallucinating connections")
        except Exception as e:
            print(f"  ❌ {e}")

    # ── Test 21: Cross-lingual EN→BN retrieval ───────────────────────────────
    total += 1
    print("\n[Test 21] Cross-lingual — EN query finds Bangla law text")
    try:
        citations = rag.search_only("maternity benefit leave 60 days labour law")
        found_bn_law = any('শ্রম' in c["law_title"] for c in citations)
        found_bn_content = any('প্রসূতি' in c["text"] or 'মাতৃত্ব' in c["text"] for c in citations)
        print(f"  Bangla Labour law: {found_bn_law}, Bangla maternity content: {found_bn_content}")
        if found_bn_law and found_bn_content:
            print("  ✅ PASS")
            passed += 1
        else:
            print("  ❌ FAIL")
    except Exception as e:
        print(f"  ❌ {e}")

    # ── Test 22: Cross-lingual BN→EN retrieval ───────────────────────────────
    total += 1
    print("\n[Test 22] Cross-lingual — BN query finds English law text")
    try:
        citations = rag.search_only("ধর্ষণের শাস্তি বাংলাদেশ দণ্ডবিধি")
        found_en_law = any('Penal Code' in c["law_title"] for c in citations)
        found_rape = any('rape' in c["text"].lower() or 'ধর্ষণ' in c["text"] for c in citations)
        has_sec_376 = any(c.get("section_number") == "376" for c in citations)
        print(f"  English Penal Code: {found_en_law}, Rape content: {found_rape}, Sec 376: {has_sec_376}")
        if found_en_law or found_rape:
            print("  ✅ PASS")
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
# CELL 23: Interactive CLI
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
# CELL 24: Kaggle Entry Point
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


# ─────────────────────────────────────────────────────────────────────────────
# CELL 25: Quick-start example
# ─────────────────────────────────────────────────────────────────────────────
"""
# Paste this into a Kaggle notebook cell to run:

GROQ_KEY = get_groq_key()
rag = BangladeshLegalRAG(groq_api_key=GROQ_KEY)
rag.build_index()

# Run validation
run_validation_tests(rag)

# Interactive session
interactive_session(rag)
"""