"""
Phase 2: PDF Document Processing Engine
Extracts page-by-page text, tabular data, headings, and metadata
from AUTOSAR High-Level Design (HLD) PDF documents.
"""

import os
import re
from typing import List, Dict, Any, Optional
import pdfplumber
import pypdf

class AUTOSARDocumentParser:
    """
    Robust parser for AUTOSAR HLD documents.
    Preserves page numbers, table structures, and section hierarchies.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF document not found at: {pdf_path}")
        self.doc_name = os.path.basename(pdf_path)

    def extract_document(self) -> Dict[str, Any]:
        """
        Extracts all pages, tables, raw text, and detected metadata.
        Returns a structured document dictionary.
        """
        pages_data = []
        raw_full_text = []

        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for idx, page in enumerate(pdf.pages):
                    page_num = idx + 1
                    text = page.extract_text() or ""
                    
                    # Extract tables with header normalization
                    tables = []
                    extracted_tables = page.extract_tables()
                    for t in extracted_tables:
                        cleaned_table = self._clean_table(t)
                        if cleaned_table and len(cleaned_table) > 1:
                            tables.append(cleaned_table)

                    pages_data.append({
                        "page_number": page_num,
                        "text": text,
                        "tables": tables,
                        "char_count": len(text)
                    })
                    raw_full_text.append(text)

        except Exception as e:
            # Fallback to pypdf if pdfplumber encounters an unexpected issue
            pages_data = []
            reader = pypdf.PdfReader(self.pdf_path)
            for idx, page in enumerate(reader.pages):
                page_num = idx + 1
                text = page.extract_text() or ""
                pages_data.append({
                    "page_number": page_num,
                    "text": text,
                    "tables": [],
                    "char_count": len(text)
                })
                raw_full_text.append(text)

        combined_text = "\n\n".join(raw_full_text)
        metadata = self._extract_metadata(combined_text)

        return {
            "document_name": self.doc_name,
            "file_path": self.pdf_path,
            "total_pages": len(pages_data),
            "metadata": metadata,
            "pages": pages_data,
            "combined_text": combined_text
        }

    def _clean_table(self, table_raw: List[List[Optional[str]]]) -> List[List[str]]:
        """
        Cleans table cells, removing newlines and excessive whitespace.
        """
        cleaned = []
        for row in table_raw:
            cleaned_row = []
            has_content = False
            for cell in row:
                if cell is not None:
                    txt = " ".join(str(cell).split())
                    if txt:
                        has_content = True
                    cleaned_row.append(txt)
                else:
                    cleaned_row.append("")
            if has_content:
                cleaned.append(cleaned_row)
        return cleaned

    def _extract_metadata(self, full_text: str) -> Dict[str, str]:
        """
        Extracts key AUTOSAR document metadata using pattern matching.
        """
        metadata = {
            "Document ID": "N/A",
            "Target ECU": "N/A",
            "AUTOSAR Release": "N/A",
            "Safety Level": "N/A",
            "Author / Lead": "N/A",
            "Status": "N/A",
            "Subsystem": "N/A"
        }

        # Regex patterns for standard automotive HLD headers
        patterns = {
            "Document ID": r"Document ID\s*[:\n]?\s*([A-Za-z0-9_\-]+)",
            "Target ECU": r"Target ECU\s*[:\n]?\s*([A-Za-z0-9_\-]+)",
            "AUTOSAR Release": r"AUTOSAR Release\s*[:\n]?\s*([^\n\r]+)",
            "Safety Level": r"Safety Level\s*[:\n]?\s*([^\n\r]+)",
            "Author / Lead": r"Author\s*(?:/\s*Lead)?\s*[:\n]?\s*([^\n\r]+)",
            "Status": r"Status\s*[:\n]?\s*(APPROVED|DRAFT|IN REVIEW|REVIEW)",
            "Subsystem": r"Subsystem\s*[:\n]?\s*([^\n\r]+)"
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                metadata[key] = match.group(1).strip()

        # Fallback heuristic for Title / Subsystem if not found in control block
        if metadata["Subsystem"] == "N/A":
            sub_match = re.search(r"System Architecture:\s*([^\n|]+)", full_text, re.IGNORECASE)
            if sub_match:
                metadata["Subsystem"] = sub_match.group(1).strip()

        return metadata


if __name__ == "__main__":
    # Test parser on sample HLD
    test_file = "data/AUTOSAR_BodyControlModule_HLD.pdf"
    if os.path.exists(test_file):
        parser = AUTOSARDocumentParser(test_file)
        doc = parser.extract_document()
        print(f"Parsed {doc['document_name']} successfully!")
        print(f"Total Pages: {doc['total_pages']}")
        print(f"Metadata: {doc['metadata']}")
        for p in doc["pages"]:
            print(f"Page {p['page_number']}: {p['char_count']} chars, {len(p['tables'])} tables")
