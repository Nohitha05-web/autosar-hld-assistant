"""
Phase 3: Document Structuring and Semantic Chunking Engine
Divides extracted content into meaningful sections and chunks while retaining
component names, headings, metadata, and source-page references.
"""

import re
from typing import List, Dict, Any, Optional

# Automotive / AUTOSAR entity regex patterns for tagging
AUTOSAR_PATTERNS = {
    "swc": re.compile(r"\b(SWC_[A-Za-z0-9_]+|CDD_[A-Za-z0-9_]+)\b"),
    "port": re.compile(r"\b([PR]_[A-Za-z0-9_]+)\b"),
    "interface": re.compile(r"\b(If_[A-Za-z0-9_]+)\b"),
    "signal": re.compile(r"\b(Sig_[A-Za-z0-9_]+)\b"),
    "runnable": re.compile(r"\b(Runnable_[A-Za-z0-9_]+)\b"),
    "bsw_module": re.compile(r"\b(DEM|DCM|FIM|CanIf|CanTp|PduR|COM|NvM|MemIf|Fee|EcuM|BswM|OS|WdgM)\b"),
    "asil": re.compile(r"\b(ASIL-[ABCD]|QM)\b"),
    "task": re.compile(r"\b(Task_[A-Za-z0-9_]+)\b"),
}

SECTION_HEADING_REGEX = re.compile(
    r"(?:^|\n)\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][A-Za-z0-9\s,&/\-()]+)(?:\n|$)"
)

class AUTOSARChunker:
    """
    Splits an extracted AUTOSAR HLD document into semantically rich chunks
    with domain-specific entity detection and complete source-page attribution.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes pages and tables from doc_data into structured chunks.
        """
        chunks = []
        doc_name = doc_data["document_name"]
        doc_meta = doc_data.get("metadata", {})
        
        current_section_num = "0"
        current_section_title = "Document Overview"

        for page in doc_data["pages"]:
            page_num = page["page_number"]
            page_text = page["text"]
            tables = page.get("tables", [])

            # Check if this page contains section headings
            headings = self._find_headings(page_text)
            
            # 1. Chunk tables directly into structured Markdown table chunks
            for t_idx, table in enumerate(tables):
                table_md = self._format_table_as_markdown(table)
                # Find entities inside table
                entities = self._extract_entities(table_md)
                
                # Check if table has a title/context from nearest heading
                chunk_id = f"{doc_name}_p{page_num}_tbl{t_idx+1}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_name": doc_name,
                    "page_number": page_num,
                    "section_number": current_section_num,
                    "section_title": current_section_title,
                    "content_type": "table",
                    "content": f"[Table from Page {page_num} - {current_section_title}]\n" + table_md,
                    "entities": entities,
                    "metadata": {
                        "source": doc_name,
                        "page": page_num,
                        "section": current_section_title,
                        "type": "table",
                        "swcs": ", ".join(entities.get("swc", [])) or "None",
                        "ecu": doc_meta.get("Target ECU", "Unknown")
                    }
                })

            # 2. Chunk text semantically
            text_blocks = self._split_by_sections_or_paragraphs(page_text)
            for block_idx, block in enumerate(text_blocks):
                # Update section context if heading found in block
                found_h = self._find_headings(block)
                if found_h:
                    current_section_num, current_section_title = found_h[0]

                # If block is large, break into overlapping windows
                sub_chunks = self._sliding_window_split(block)
                for s_idx, text_chunk in enumerate(sub_chunks):
                    if len(text_chunk.strip()) < 30:
                        continue  # Skip trivial headers/footers

                    entities = self._extract_entities(text_chunk)
                    chunk_id = f"{doc_name}_p{page_num}_txt{block_idx+1}_{s_idx+1}"
                    
                    # Prefix with context to enrich vector representation
                    enriched_content = (
                        f"Document: {doc_name} | Page: {page_num} | Section: {current_section_num}. {current_section_title}\n"
                        f"{text_chunk.strip()}"
                    )

                    chunks.append({
                        "chunk_id": chunk_id,
                        "document_name": doc_name,
                        "page_number": page_num,
                        "section_number": current_section_num,
                        "section_title": current_section_title,
                        "content_type": "text",
                        "content": enriched_content,
                        "entities": entities,
                        "metadata": {
                            "source": doc_name,
                            "page": page_num,
                            "section": current_section_title,
                            "type": "text",
                            "swcs": ", ".join(entities.get("swc", [])) or "None",
                            "ecu": doc_meta.get("Target ECU", "Unknown")
                        }
                    })

        return chunks

    def _find_headings(self, text: str) -> List[tuple]:
        """
        Locates numbered section headings in the text.
        """
        matches = []
        for line in text.split("\n"):
            line = line.strip()
            m = re.match(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][A-Za-z0-9\s,&/\-()]+)$", line)
            if m:
                matches.append((m.group(1), m.group(2).strip()))
        return matches

    def _split_by_sections_or_paragraphs(self, text: str) -> List[str]:
        """
        Splits text by double newlines or section boundaries.
        """
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return paragraphs

    def _sliding_window_split(self, text: str) -> List[str]:
        """
        Splits text into chunks of roughly chunk_size characters with chunk_overlap.
        """
        if len(text) <= self.chunk_size:
            return [text]

        words = text.split()
        chunks = []
        curr_words = []
        curr_len = 0

        target_word_count = max(40, self.chunk_size // 6)
        overlap_word_count = max(10, self.chunk_overlap // 6)

        i = 0
        while i < len(words):
            end_idx = min(i + target_word_count, len(words))
            chunk_slice = " ".join(words[i:end_idx])
            chunks.append(chunk_slice)
            if end_idx == len(words):
                break
            i += (target_word_count - overlap_word_count)

        return chunks

    def _format_table_as_markdown(self, table: List[List[str]]) -> str:
        """
        Converts 2D table list into GitHub-flavored Markdown table.
        """
        if not table or len(table) < 1:
            return ""

        header = table[0]
        md_lines = []
        md_lines.append("| " + " | ".join(header) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        for row in table[1:]:
            # Pad row if needed
            padded_row = row + [""] * (len(header) - len(row))
            md_lines.append("| " + " | ".join(padded_row[:len(header)]) + " |")

        return "\n".join(md_lines)

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Scans text chunk for AUTOSAR specific components, ports, interfaces, signals, and BSW modules.
        """
        entities = {}
        for entity_type, pattern in AUTOSAR_PATTERNS.items():
            matches = list(set(pattern.findall(text)))
            if matches:
                entities[entity_type] = sorted(matches)
        return entities


if __name__ == "__main__":
    from src.pdf_parser import AUTOSARDocumentParser
    parser = AUTOSARDocumentParser("data/AUTOSAR_BodyControlModule_HLD.pdf")
    doc_data = parser.extract_document()
    chunker = AUTOSARChunker(chunk_size=400, chunk_overlap=60)
    chunks = chunker.chunk_document(doc_data)
    print(f"Total structured chunks generated: {len(chunks)}")
    for i, c in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ({c['chunk_id']}) ---")
        print(f"Page: {c['page_number']}, Section: {c['section_title']}")
        print(f"Entities: {c['entities']}")
        print(f"Content Preview: {c['content'][:150]}...")
