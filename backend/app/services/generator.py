import json
import re
import logging
from google import genai
from google.genai import types
from app.core.config import settings
from app.prompts.generation_prompt import SCHOLARLY_SYSTEM_PROMPT_BN, SCHOLARLY_SYSTEM_PROMPT_EN
from app.formatters.chunk_formatter import ChunkFormatter
from pydantic import BaseModel, Field
from typing import Literal

logger = logging.getLogger("app.generator")

# ---------------------------------------------------------------------------
# Safety net — strips any "অর্থ:" / "Meaning:" line the LLM sneaks in.
# ---------------------------------------------------------------------------
_MEANING_RE = re.compile(
    r'(?m)^[^\S\n]*(?:'
    r'অর্থ|অনুবাদ|সহজ\s*অর্থ|অর্থাৎ|এর\s*অর্থ\s*হলো|'
    r'Meaning|Translation|In\s+other\s+words|This\s+means'
    r')\s*[:\-][\s\S]*?(?=\n\n|\Z)',
    re.IGNORECASE,
)

def _clean(text: str) -> str:
    if not text:
        return ""
    result = re.sub(_MEANING_RE, "", text)
    result = re.sub(r'\n{3,}', '\n\n', result).strip()
    if result != text.strip():
        logger.info("Generator: stripped a forbidden translation block.")
    return result


# ---------------------------------------------------------------------------
# Schema — with Chain-of-Thought reasoning fields (logged, not shown to user)
# ---------------------------------------------------------------------------

class ReasoningTrace(BaseModel):
    """Internal CoT — logged for debugging, never shown in the final response."""
    question_intent: str = Field(
        description="One sentence: what exactly is the user asking? Is it quantitative, procedural, substantive, or comparative?"
    )
    relevant_context_found: str = Field(
        description="Which parts of the retrieved context directly address the query? List section titles or key phrases."
    )
    direct_answer_available: bool = Field(
        description="True if the context contains a direct, specific answer. False if only partial or indirect info is available."
    )
    gaps: str = Field(
        description="What specific information is missing from the context that would be needed for a complete answer? Write 'none' if fully answered."
    )


class LegalPoint(BaseModel):
    analysis: str = Field(
        description=(
            "Legal analysis prose explaining WHY this provision is relevant to the query. "
            "If the answer is only partial, explicitly state what this provision covers and what it does not. "
            "Distinguish between procedural (how) and substantive (what right/obligation) aspects where relevant. "
            "Do NOT write 'অর্থ:', 'অনুবাদ:', 'Meaning:', or 'Translation:' anywhere."
        )
    )
    legal_text: str = Field(
        description=(
            "Exact verbatim legal text from the Context. No paraphrase. No translation appended."
        )
    )
    citation: str = Field(
        description="Law Name, Year, Section Title. Example: পারিবারিক আদালত আইন, ২০২৩, ধারা ৫"
    )


class ScholarlyEssay(BaseModel):
    reasoning: ReasoningTrace = Field(
        description="Internal reasoning trace. Fill this FIRST before writing the answer."
    )
    summary: str = Field(
        description=(
            "3-4 sentences: situation overview and direct answer based ONLY on retrieved context. "
            "If context lacks the specific answer, write exactly: "
            "'দুঃখিত, আপনার প্রশ্নের উত্তর প্রদত্ত আইনি রেফারেন্সে পাওয়া যায়নি।' (Bengali) or "
            "'Sorry, the answer to your question was not found in the provided legal references.' (English). "
            "For quantitative questions (how many, what number): if the exact number is not in context, "
            "say so explicitly — do NOT estimate or infer."
        )
    )
    points: list[LegalPoint] = Field(
        description=(
            "Detailed legal evidence from the context. "
            "Empty array [] if context does not contain relevant specific provisions. "
            "Never fabricate a legal_text — it must exist verbatim in the context."
        )
    )
    conclusion: str = Field(
        description=(
            "Final practical ruling starting with 'মোটকথা,' (Bengali) or 'In summary,' (English). "
            "If context lacks the answer, leave this as empty string ''."
        )
    )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_name = settings.GENERATION_MODEL
        self.chunk_formatter = ChunkFormatter()

    async def generate(self, query: str, language: str, chunks: list[dict]) -> str:
        if not chunks:
            return (
                "দুঃখিত, আপনার প্রশ্নের উত্তর প্রদত্ত আইনি রেফারেন্সে পাওয়া যায়নি।"
                if language == "bn" else
                "Sorry, the answer to your question was not found in the provided legal references."
            )

        system_prompt = (
            SCHOLARLY_SYSTEM_PROMPT_BN if language == "bn" else SCHOLARLY_SYSTEM_PROMPT_EN
        )
        context = self.chunk_formatter.format_for_prompt(chunks)
        full_prompt = (
            f"=== SOURCE PASSAGES ===\n{context}\n=== END SOURCE PASSAGES ===\n\n"
            f"User Query: {query}\n\n"
        )

        def bn_digits(text: str) -> str:
            if not text:
                return text
            for en, bn in zip("0123456789", "০১২৩৪৫৬৭৮৯"):
                text = text.replace(en, bn)
            for ar, bn in zip("٠١٢٣٤٥٦٧٨٩", "০১২৩৪৫৬৭৮৯"):
                text = text.replace(ar, bn)
            return text

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=ScholarlyEssay,
                    temperature=0.1,
                ),
            )

            data = json.loads(response.text)

            # --- Log CoT reasoning trace (never shown to user) ---
            r = data.get("reasoning", {})
            logger.info(
                f"Generator CoT | intent: {r.get('question_intent', '')} | "
                f"direct_answer: {r.get('direct_answer_available', '')} | "
                f"gaps: {r.get('gaps', '')}"
            )

            header = "বাংলাদেশ লিগ্যাল এডভাইজার" if language == "bn" else "Bangladesh Legal Advisor"

            # --- Summary ---
            summary = _clean(data.get("summary", ""))
            out = f"{header}\n\n**{summary}**\n\n"

            # --- Points ---
            for pt in data.get("points", []):
                analysis = pt.get("analysis", "")
                law_text = _clean(pt.get("legal_text", ""))
                citation = pt.get("citation", "").strip()

                if citation.startswith("[") and citation.endswith("]"):
                    citation = citation[1:-1]

                # Strip any quote blocks LLM put inside analysis
                analysis = re.sub(r'\*\*"[\s\S]*?"\*\*', '', analysis)
                analysis = _clean(analysis)

                if language == "bn":
                    analysis = bn_digits(analysis)
                    citation = bn_digits(citation)

                if analysis:
                    out += f"{analysis}\n\n"
                if law_text:
                    out += f"**\"{law_text}\"**\n"
                if citation:
                    out += f"[{citation}]\n\n"

            # --- Conclusion ---
            conclusion = _clean(data.get("conclusion", ""))
            if conclusion:
                if language == "bn":
                    conclusion = bn_digits(conclusion)
                out += f"**{conclusion}**\n\n"

            # --- References from chunk metadata ---
            seen: dict = {}
            for chunk in chunks:
                m = chunk.get("metadata", {})
                name = (
                    m.get("law_title") or m.get("source_file")
                    or ("অজ্ঞাত আইন" if language == "bn" else "Unknown Law")
                )
                year = m.get("year") or ""
                link  = m.get("link") or ""
                seen[(name, year, link)] = True

            if seen:
                label = "**রেফারেন্স:**" if language == "bn" else "**References:**"
                refs = []
                for name, year, link in seen:
                    yr = bn_digits(str(year)) if language == "bn" else str(year)
                    entry = f"{name} ({yr})"
                    if link:
                        entry += f" - [Link]({link})"
                    refs.append(entry)
                out += label + "\n" + "\n".join(f"- {r}" for r in refs)

            return out.strip()

        except Exception as e:
            logger.error(f"Generator error: {e}", exc_info=True)
            return (
                "দুঃখিত, উত্তর জেনারেট করতে সমস্যা হয়েছে।"
                if language == "bn" else
                "Sorry, there was an error generating the response."
            )
