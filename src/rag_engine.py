"""
Phase 5: RAG and LLM Integration Engine
Performs grounded question answering over AUTOSAR HLD documents.
Enforces strict citations, confidence indicators, and dual LLM support
(Google Gemini API with seamless offline local synthesis fallback).
"""

import os
import re
from typing import Dict, Any, List, Optional
from src.vector_store import AUTOSARVectorStore

# Prompt template enforcing strict automotive citations and evidence grounding
SYSTEM_PROMPT = """You are an expert AUTOSAR High-Level Design (HLD) Architecture Specialist for Automotive OEM systems.
Your mission is to answer engineering questions with rigorous accuracy, grounded strictly in the provided architectural context.

STRICT GROUNDING & CITATION RULES:
1. Ground every claim strictly in the provided context chunks.
2. For EVERY technical statement, cite the exact source page and section using format: [Page X - Section Title].
3. If an answer cannot be determined or is not present in the provided context, state clearly: "The provided High-Level Design document does not contain information regarding this topic." Do NOT hallucinate.
4. Highlight key automotive architectural metrics: Component names (SWC), Port types (P-Port / R-Port), Interfaces (S/R or C/S), ASIL levels (QM / ASIL-A to D), and BSW modules (DEM, NvM, COM, etc.).
"""

class AUTOSARRAGEngine:
    """
    RAG orchestrator for AUTOSAR document analysis.
    """

    def __init__(self, vector_store: AUTOSARVectorStore, api_key: Optional[str] = None):
        self.vector_store = vector_store
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.gemini_model = None

        if self.api_key:
            self._init_gemini(self.api_key)

    def _init_gemini(self, api_key: str):
        """Initializes Google Generative AI client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            self.gemini_model = None

    def set_api_key(self, api_key: str):
        """Updates the Gemini API key dynamically."""
        self.api_key = api_key
        self._init_gemini(api_key)

    def answer_question(
        self,
        question: str,
        n_results: int = 4,
        doc_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieves relevant chunks and generates a grounded response with page citations.
        """
        where_filter = {"source": doc_filter} if doc_filter else None
        retrieved = self.vector_store.query(question, n_results=n_results, where_filter=where_filter)

        if not retrieved:
            return {
                "answer": "No relevant architectural context could be found in the indexed HLD document for this query.",
                "citations": [],
                "confidence": "LOW",
                "retrieved_chunks": []
            }

        # Build context block
        context_blocks = []
        citations = []
        seen_pages = set()

        for idx, r in enumerate(retrieved):
            meta = r.get("metadata", {})
            page = meta.get("page", "?")
            sec = meta.get("section", "General")
            src = meta.get("source", "HLD")
            
            context_blocks.append(
                f"[Source Chunk {idx+1} | Document: {src} | Page: {page} | Section: {sec}]\n{r['content']}"
            )
            
            if (src, page) not in seen_pages:
                citations.append({
                    "document": src,
                    "page": page,
                    "section": sec,
                    "similarity": r.get("similarity", 0.0),
                    "confidence": r.get("confidence", "MEDIUM"),
                    "snippet": r["content"][:250] + "..."
                })
                seen_pages.add((src, page))

        combined_context = "\n\n".join(context_blocks)

        # Check for query keyword relevance to prevent hallucination on unsupported topics
        query_significant_words = [
            w for w in re.findall(r"\b[A-Za-z0-9_]{3,}\b", question.lower())
            if w not in {"what", "which", "where", "when", "does", "have", "with", "from", "that", "this", "these", "those", "explain", "describe", "show", "list", "tell", "used", "under"}
        ]
        
        # Count occurrences of significant query terms across retrieved chunks
        def _has_term(word: str, content: str) -> bool:
            c_low = content.lower()
            if word in c_low:
                return True
            # Check without whitespace to handle line wraps in PDF tables
            c_nowhite = re.sub(r"\s+", "", c_low)
            return word in c_nowhite

        term_hits = sum(
            1 for w in query_significant_words
            if any(_has_term(w, r["content"]) for r in retrieved)
        )
        
        top_similarity = retrieved[0].get("similarity", 0.0) if retrieved else 0.0
        avg_score = sum(r.get("similarity", 0.0) for r in retrieved) / len(retrieved)

        # Classify as unsupported if zero significant terms match or very low unique keyword overlap
        is_unsupported = False
        if query_significant_words:
            unique_words = set(query_significant_words)
            matched_unique = {w for w in unique_words if any(_has_term(w, r["content"]) for r in retrieved)}
            unique_overlap = len(matched_unique) / len(unique_words)

            if len(matched_unique) == 0:
                is_unsupported = True
            elif unique_overlap < 0.35:
                is_unsupported = True
            elif unique_overlap < 0.50 and top_similarity < 0.65:
                is_unsupported = True
        elif top_similarity < 0.50:
            is_unsupported = True

        if is_unsupported:
            return {
                "answer": (
                    f"### Architectural Analysis Notice\n\n"
                    f"**Out-of-Scope / Unsupported Query**: The provided High-Level Design document does not contain "
                    f"specifications, requirements, or architecture components related to **\"{question.strip()}\"**.\n\n"
                    f"*Strict Grounding Policy: The assistant only generates answers backed by explicit evidence in the loaded HLD.*"
                ),
                "citations": [],
                "confidence": "LOW",
                "retrieved_chunks": retrieved,
                "mode": "Grounding Verifier"
            }
        overall_confidence = "HIGH" if avg_score > 0.68 else ("MEDIUM" if avg_score > 0.50 else "LOW")

        # Check if Gemini model is available
        if self.gemini_model and self.api_key:
            try:
                full_prompt = (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"--- ARCHITECTURAL CONTEXT ---\n{combined_context}\n\n"
                    f"--- USER QUESTION ---\n{question}\n\n"
                    f"Answer with explicit page citations [Page X] for every fact:"
                )
                response = self.gemini_model.generate_content(full_prompt)
                answer_text = response.text
                return {
                    "answer": answer_text,
                    "citations": citations,
                    "confidence": overall_confidence,
                    "retrieved_chunks": retrieved,
                    "mode": "Gemini LLM"
                }
            except Exception as e:
                # Fallback to local synthesis if API fails or rate limited
                pass

        # Offline Local Synthesizer Fallback
        local_answer = self._synthesize_local_answer(question, retrieved, citations)
        return {
            "answer": local_answer,
            "citations": citations,
            "confidence": overall_confidence,
            "retrieved_chunks": retrieved,
            "mode": "Local Offline RAG Engine"
        }

    def _synthesize_local_answer(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        citations: List[Dict[str, Any]]
    ) -> str:
        """
        High-precision local deterministic synthesizer that extracts relevant facts,
        tables, and quotes from retrieved chunks with citations.
        Ensures 100% offline functionality.
        """
        q_lower = question.lower()
        ans_parts = []
        ans_parts.append(f"### Architectural Analysis Summary\n")

        # Check for specific questions
        found_relevant_info = False

        for c in chunks:
            text = c["content"]
            page = c["metadata"].get("page", 1)
            sec = c["metadata"].get("section", "Overview")
            matched_rows = []

            # Check if chunk contains tables
            if "| --- |" in text:
                table_lines = text.split("\n")
                headers = []
                for line in table_lines:
                    if line.startswith("|") and ("---" not in line):
                        if not headers:
                            headers.append(line)
                        else:
                            words = [w for w in re.findall(r"\w+", q_lower) if len(w) > 3]
                            line_low = line.lower()
                            line_nowhite = re.sub(r"\s+", "", line_low)
                            if any(w in line_low or w in line_nowhite for w in words):
                                matched_rows.append(line)

                if matched_rows:
                    found_relevant_info = True
                    ans_parts.append(f"**Found in [Page {page} - {sec}]:**\n")
                    ans_parts.append("\n".join(headers + ["| --- " * (headers[0].count("|") - 1) + "|"] + matched_rows))
                    ans_parts.append(f"\n*Citation: [Page {page} - {sec}]*\n")

            # Extract paragraphs / lines instead of breaking on numbering periods
            lines = [l.strip() for l in re.split(r"\n|(?<=[a-z]{2}\.)\s+(?=[A-Z])", text) if l.strip()]
            relevant_lines = []
            words = [w for w in re.findall(r"\w+", q_lower) if len(w) > 3 and w not in {"what", "which", "where", "under"}]
            for l in lines:
                if any(w in l.lower() for w in words) and len(l) > 20 and not l.startswith("|"):
                    relevant_lines.append(l)

            if relevant_lines and not matched_rows:
                found_relevant_info = True
                ans_parts.append(f"**Evidence from [Page {page} - {sec}]:**")
                for l in relevant_lines[:3]:
                    ans_parts.append(f"- {l} `[Page {page}]`")
                ans_parts.append("")

        if not found_relevant_info:
            # Fallback to top chunk excerpt
            top_c = chunks[0]
            ans_parts.append(
                f"Based on the most relevant section **[Page {top_c['metadata'].get('page')} - {top_c['metadata'].get('section')}]**:\n\n"
                f"{top_c['content']}\n"
            )

        ans_parts.append("\n---")
        ans_parts.append("**Source Page Citations:** " + ", ".join([f"`Page {c['page']}: {c['section']}`" for c in citations]))

        return "\n".join(ans_parts)


if __name__ == "__main__":
    from src.pdf_parser import AUTOSARDocumentParser
    from src.chunker import AUTOSARChunker

    pdf_file = "data/AUTOSAR_BodyControlModule_HLD.pdf"
    parser = AUTOSARDocumentParser(pdf_file)
    doc_data = parser.extract_document()
    chunker = AUTOSARChunker()
    chunks = chunker.chunk_document(doc_data)

    store = AUTOSARVectorStore()
    store.index_chunks(chunks)

    engine = AUTOSARRAGEngine(store)
    result = engine.answer_question("What is the periodicity and ASIL level of SWC_ExteriorLighting?")
    print("ANSWER:\n", result["answer"])
    print("\nConfidence:", result["confidence"])
    print("Mode:", result["mode"])
