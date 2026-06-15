import re
from app.core.config import settings

class Chunker:
    """
    Section-preserving chunker for legal documents.
    """
    def __init__(self, section_max_size=1200, chunk_overlap=150, min_chunk_size=60):
        self.section_max_size = section_max_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_law(self, law_dict: dict) -> list[dict]:
        text = law_dict.get("law_full_text", "").strip()
        year = str(law_dict.get("year", ""))
        link = law_dict.get("link", "")
        if not text:
            return []
        
        pipe_parts = [p.strip() for p in text.split(" | ") if p.strip()]
        if not pipe_parts:
            return []
            
        law_title = pipe_parts[0]
        chunks = []
        seq = 0
        
        # Simple extraction of repeal status
        repealed = "REPEALED" if "[REPEALED:" in text or "[REPEAL" in text else "ACTIVE"
        
        for part_idx, part in enumerate(pipe_parts):
            if len(part) < self.min_chunk_size:
                continue
            section_title = self._extract_section_title(part)
            sub_chunks = self._split_section(part)
            
            for sc in sub_chunks:
                if len(sc.strip()) < self.min_chunk_size:
                    continue
                chunks.append({
                    "text": sc.strip(),
                    "metadata": {
                        "law_title": law_title,
                        "year": year,
                        "link": link,
                        "section_title": section_title,
                        "repealed": repealed,
                        "chunk_seq": seq,
                        "word_count": len(sc.strip().split())
                    }
                })
                seq += 1
                
        # If no chunks but there's a title
        if not chunks and law_title:
            chunks.append({
                "text": law_title,
                "metadata": {
                    "law_title": law_title,
                    "year": year,
                    "link": link,
                    "section_title": "(title only)",
                    "repealed": repealed,
                    "chunk_seq": 0,
                    "word_count": len(law_title.split())
                }
            })
            
        return chunks

    def _extract_section_title(self, text: str) -> str:
        m = re.match(r'^([^\n:।]{3,70})(?::\s*\d|।)', text)
        if m:
            return m.group(1).strip()
        return text[:60].split('\n')[0].strip()

    def _split_section(self, text: str) -> list[str]:
        if len(text) <= self.section_max_size:
            return [text]
        chunks = []
        start = 0
        size = self.section_max_size
        overlap = self.chunk_overlap
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
