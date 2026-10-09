"""
Phase 6: Streamlit Engineering Application
AUTOSAR HLD Document Analysis Assistant
Tata Engineering AI Project
"""

import os
import time
import json
import streamlit as st
import pandas as pd

from src.pdf_parser import AUTOSARDocumentParser
from src.chunker import AUTOSARChunker
from src.vector_store import AUTOSARVectorStore
from src.rag_engine import AUTOSARRAGEngine
from src.architecture_extractor import AUTOSARArchitectureExtractor
from src.inconsistency_checker import AUTOSARInconsistencyChecker
from src.visualizer import AUTOSARVisualizer
from src.revision_diff import AUTOSARRevisionDiff

# Page setup
st.set_page_config(
    page_title="AUTOSAR HLD Analysis Assistant | Tata AI Project",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1A365D;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4A5568;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F7FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .citation-badge {
        display: inline-block;
        background-color: #EBF8FF;
        color: #2B6CB0;
        border: 1px solid #BEE3F8;
        border-radius: 4px;
        padding: 2px 8px;
        margin-right: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-critical {
        background-color: #FED7D7;
        color: #9B2C2C;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-high {
        background-color: #FEEBC8;
        color: #9C4221;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-medium {
        background-color: #FEFCBF;
        color: #744210;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to cache document parsing & extraction
@st.cache_resource
def get_vector_store():
    return AUTOSARVectorStore()

@st.cache_data
def parse_and_extract(file_path: str):
    parser = AUTOSARDocumentParser(file_path)
    doc_data = parser.extract_document()
    extractor = AUTOSARArchitectureExtractor(doc_data)
    arch = extractor.extract_all()
    chunker = AUTOSARChunker()
    chunks = chunker.chunk_document(doc_data)
    return doc_data, arch, chunks

# App State
vector_store = get_vector_store()

# Sidebar: Document Management & Config
st.sidebar.title("🚗 Navigation & Ingestion")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
available_files = []
if os.path.exists(DATA_DIR):
    available_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]

# Document Selection
st.sidebar.subheader("Select HLD Document")
selected_doc_name = st.sidebar.selectbox(
    "Active Architecture Document",
    options=available_files,
    index=0 if available_files else None
)

# Custom File Upload
uploaded_file = st.sidebar.file_uploader("Upload New AUTOSAR HLD (PDF)", type=["pdf"])
if uploaded_file is not None:
    custom_save_path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(custom_save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Saved: {uploaded_file.name}")
    selected_doc_name = uploaded_file.name

# Model Backend Configuration
st.sidebar.divider()
st.sidebar.subheader("⚙️ AI Reasoning Backend")
ai_mode = st.sidebar.radio(
    "Inference Engine",
    options=["Local Offline RAG Engine", "Google Gemini Cloud API"],
    index=0
)

gemini_key = ""
if ai_mode == "Google Gemini Cloud API":
    gemini_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Enter Google Gemini API key to enable cloud generation"
    )

# Load Selected Document Data
active_pdf_path = os.path.join(DATA_DIR, selected_doc_name) if selected_doc_name else None

if not active_pdf_path or not os.path.exists(active_pdf_path):
    st.warning("No AUTOSAR HLD document found. Please upload a PDF or run `python generate_sample_hld.py`.")
    st.stop()

doc_data, arch, chunks = parse_and_extract(active_pdf_path)

# Ensure Chunks are indexed
if not vector_store.is_document_indexed(selected_doc_name):
    with st.spinner(f"Indexing {selected_doc_name} into ChromaDB vector store..."):
        vector_store.index_chunks(chunks)

# Initialize RAG Engine
rag_engine = AUTOSARRAGEngine(vector_store, api_key=gemini_key if ai_mode == "Google Gemini Cloud API" else None)

# Ingestion Stats in Sidebar
st.sidebar.divider()
st.sidebar.markdown(f"**Document:** `{selected_doc_name}`")
st.sidebar.markdown(f"- **Total Pages:** {doc_data['total_pages']}")
st.sidebar.markdown(f"- **Indexed Chunks:** {len(chunks)}")
st.sidebar.markdown(f"- **SWCs Catalogued:** {arch['stats']['total_swcs']}")
st.sidebar.markdown(f"- **Vector DB:** ChromaDB (Cosine)")

if st.sidebar.button("🔄 Force Re-index Document"):
    vector_store.delete_document(selected_doc_name)
    vector_store.index_chunks(chunks)
    st.sidebar.success("Vector store re-indexed!")

# Header Banner
st.markdown("<p class='main-header'>AUTOSAR HLD Document Analysis Assistant</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'><b>Primary Value:</b> Faster architecture understanding and reusable design knowledge | Tata Automotive Engineering AI</p>", unsafe_allow_html=True)

# Top KPI Summary Cards
meta = arch.get("metadata", {})
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Target ECU", meta.get("Target ECU", "Unknown"))
with kpi2:
    st.metric("AUTOSAR Release", meta.get("AUTOSAR Release", "4.4.0"))
with kpi3:
    st.metric("Safety Rating", meta.get("Safety Level", "ASIL-B"))
with kpi4:
    st.metric("Atomic SWCs", arch["stats"]["total_swcs"])
with kpi5:
    st.metric("Port Connections", arch["stats"]["total_interfaces"])

# Main Application Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Architecture Overview",
    "💬 Grounded RAG Assistant",
    "🧩 Component & Interface Explorer",
    "⚠️ Inconsistency & Audit",
    "🔄 Revision Comparison",
    "📑 Knowledge Export"
])

# ==========================================
# TAB 1: ARCHITECTURE OVERVIEW
# ==========================================
with tab1:
    st.subheader("Architectural System Blueprint")
    col_desc, col_graph = st.columns([1, 1])

    with col_desc:
        st.markdown("### Executive Architecture Summary")
        # Extract executive summary text from first page
        page1_text = doc_data["pages"][0]["text"] if doc_data["pages"] else ""
        st.info(
            f"**Target System:** {meta.get('Subsystem', 'Automotive Subsystem')}\n\n"
            f"**ECU Node:** `{meta.get('Target ECU', 'ECU_NODE')}` running **{meta.get('AUTOSAR Release', 'Classic 4.4.0')}**.\n\n"
            f"**Safety Integrity:** Allocated at **{meta.get('Safety Level', 'ASIL-B')}** under ISO 26262.\n\n"
            f"**Approval Status:** `{meta.get('Status', 'APPROVED')}` by {meta.get('Author / Lead', 'Engineering Architecture Team')}."
        )

        st.markdown("### Integrated Basic Software (BSW) Stacks")
        bsw_stacks = arch.get("bsw_stack", {})
        if bsw_stacks:
            for stack_name, modules in bsw_stacks.items():
                st.markdown(f"**{stack_name}**: " + ", ".join([f"`{m}`" for m in modules]))
        else:
            st.write("No BSW modules explicitly parsed.")

    with col_graph:
        st.markdown("### AUTOSAR Layered Topology")
        viz = AUTOSARVisualizer(arch)
        mermaid_code = viz.generate_mermaid_diagram()
        st.code(mermaid_code, language="mermaid")
        st.caption("Diagram automatically generated from parsed Sender-Receiver and Client-Server port topologies.")

# ==========================================
# TAB 2: GROUNDED RAG ASSISTANT
# ==========================================
with tab2:
    st.subheader("Automotive Architectural Q&A Assistant")
    st.markdown("Ask natural language technical questions grounded strictly in the loaded AUTOSAR HLD document.")

    # Sample Quick Questions
    st.markdown("**Sample Quick Queries:**")
    quick_queries = [
        "What is the periodicity and ASIL level of SWC_ExteriorLighting?",
        "Which interface connects SWC_ExteriorLighting to CDD_SmartSmartActuator?",
        "What is the trigger event and max WCET of Runnable_Lighting_Step?",
        "What diagnostic stack modules and DTC reporting interfaces are used?",
        "What safety goals are defined under ISO 26262 for this ECU?",
        "What architectural open issues or dangling ports were recorded?"
    ]

    selected_quick_q = None
    cols = st.columns(3)
    for i, q in enumerate(quick_queries):
        if cols[i % 3].button(q, key=f"quick_btn_{i}"):
            selected_quick_q = q

    user_query = st.text_input(
        "Enter your technical architecture question:",
        value=selected_quick_q or "",
        placeholder="e.g., What are the runnable entities and OS task mappings in SWC_ExteriorLighting?"
    )

    if user_query:
        with st.spinner("Analyzing architecture context and synthesizing cited response..."):
            t_start = time.time()
            res = rag_engine.answer_question(user_query, n_results=4, doc_filter=selected_doc_name)
            latency_ms = round((time.time() - t_start) * 1000, 1)

        # Confidence & Engine Badge
        conf = res.get("confidence", "MEDIUM")
        conf_color = "green" if conf == "HIGH" else ("orange" if conf == "MEDIUM" else "red")
        
        st.markdown(
            f"**Engine:** `{res.get('mode', 'RAG Engine')}` | "
            f"**Confidence:** :{conf_color}[**{conf}**] | "
            f"**Response Latency:** `{latency_ms} ms`"
        )

        st.markdown(res["answer"])

        # Display Citations
        citations = res.get("citations", [])
        if citations:
            st.markdown("#### Source Page Citations")
            c_cols = st.columns(min(len(citations), 4))
            for idx, cit in enumerate(citations):
                with c_cols[idx % 4]:
                    st.markdown(f"<span class='citation-badge'>Page {cit['page']}</span> **{cit['section']}**", unsafe_allow_html=True)
                    st.caption(f"Doc: {cit['document']}")

        # Expandable Evidence Chunks
        with st.expander("🔍 Inspect Retrieved Grounding Context & Evidence Chunks"):
            for i, chunk in enumerate(res.get("retrieved_chunks", [])):
                meta_c = chunk.get("metadata", {})
                st.markdown(f"**Chunk {i+1} | Source:** `{meta_c.get('source')}` | **Page:** `{meta_c.get('page')}` | **Similarity:** `{chunk.get('similarity')}`")
                st.text(chunk.get("content"))
                st.divider()

# ==========================================
# TAB 3: COMPONENT & INTERFACE EXPLORER
# ==========================================
with tab3:
    st.subheader("Architecture Repository & Data Dictionary")
    
    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "Software Components (SWC)",
        "Ports & Interfaces",
        "Signals & Bus Mapping",
        "Runnables & RTE Tasks"
    ])

    with subtab1:
        swcs = arch.get("swc_inventory", [])
        if swcs:
            df_swc = pd.DataFrame(swcs)
            st.dataframe(df_swc, use_container_width=True)
            csv_data = df_swc.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download SWC Catalogue (CSV)", csv_data, f"{selected_doc_name}_swcs.csv", "text/csv")
        else:
            st.info("No SWCs found in document.")

    with subtab2:
        ports = arch.get("ports_interfaces", [])
        if ports:
            df_ports = pd.DataFrame(ports)
            st.dataframe(df_ports, use_container_width=True)
            csv_ports = df_ports.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Ports & Interfaces (CSV)", csv_ports, f"{selected_doc_name}_ports.csv", "text/csv")
        else:
            st.info("No Ports & Interfaces found in document.")

    with subtab3:
        signals = arch.get("signals", [])
        if signals:
            df_signals = pd.DataFrame(signals)
            st.dataframe(df_signals, use_container_width=True)
            csv_sig = df_signals.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Signals Dictionary (CSV)", csv_sig, f"{selected_doc_name}_signals.csv", "text/csv")
        else:
            st.info("No Signals defined in document.")

    with subtab4:
        runnables = arch.get("runnables", [])
        if runnables:
            df_run = pd.DataFrame(runnables)
            st.dataframe(df_run, use_container_width=True)
        else:
            st.info("No Runnable entities extracted.")

# ==========================================
# TAB 4: INCONSISTENCY & AUDIT
# ==========================================
with tab4:
    st.subheader("Architectural Completeness & Inconsistency Audit")
    st.markdown("Automated scan for dangling ports, signal width mismatches, missing BSW handlers, and ASIL safety conflicts.")

    checker = AUTOSARInconsistencyChecker(arch)
    findings = checker.check_all()

    if not findings:
        st.success("✅ No architectural defects or inconsistencies detected!")
    else:
        st.warning(f"⚠️ {len(findings)} architectural finding(s) detected during automated review:")
        for idx, item in enumerate(findings):
            sev = item.get("severity", "MEDIUM")
            badge_class = "badge-critical" if sev == "CRITICAL" else ("badge-high" if sev == "HIGH" else "badge-medium")
            
            with st.container():
                st.markdown(
                    f"#### Finding #{idx+1}: `{item['rule']}` | <span class='{badge_class}'>{sev}</span>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**Component:** `{item.get('component')}` | **Source Page:** `Page {item.get('page')}`")
                st.markdown(f"**Description:** {item.get('description')}")
                st.info(f"**Recommended Remediation:** {item.get('remediation')}")
                st.divider()

# ==========================================
# TAB 5: REVISION COMPARISON
# ==========================================
with tab5:
    st.subheader("Document Revision Comparison & Architecture Delta")
    st.markdown("Compare two versions of an AUTOSAR HLD document to identify added SWCs, modified periodicity, and core allocation upgrades.")

    rev_col1, rev_col2 = st.columns(2)
    with rev_col1:
        doc_a_name = st.selectbox("Baseline Document (v1)", options=available_files, index=0 if len(available_files) > 1 else 0)
    with rev_col2:
        default_idx = min(1, len(available_files)-1) if len(available_files) > 1 else 0
        doc_b_name = st.selectbox("Updated Document (v2)", options=available_files, index=default_idx)

    if st.button("🔍 Compare Revisions"):
        if doc_a_name == doc_b_name:
            st.warning("Please select two distinct documents to perform revision comparison.")
        else:
            path_a = os.path.join(DATA_DIR, doc_a_name)
            path_b = os.path.join(DATA_DIR, doc_b_name)
            _, arch_a, _ = parse_and_extract(path_a)
            _, arch_b, _ = parse_and_extract(path_b)

            differ = AUTOSARRevisionDiff(arch_a, arch_b)
            diff_res = differ.compute_diff()

            st.markdown(f"### Delta Summary: `{doc_a_name}` ➔ `{doc_b_name}`")
            d1, d2, d3 = st.columns(3)
            d1.metric("SWCs Added", diff_res["summary"]["total_added"])
            d2.metric("SWCs Removed", diff_res["summary"]["total_removed"])
            d3.metric("SWCs Modified", diff_res["summary"]["total_modified"])

            if diff_res["swcs_added"]:
                st.markdown("#### ✨ Newly Added SWCs in Revision:")
                st.table(pd.DataFrame(diff_res["swcs_added"]))

            if diff_res["swcs_modified"]:
                st.markdown("#### 🔄 Modified SWC Attributes:")
                for m in diff_res["swcs_modified"]:
                    st.markdown(f"**{m['swc_name']}:**")
                    for c in m["changes"]:
                        st.markdown(f"- {c}")

# ==========================================
# TAB 6: KNOWLEDGE EXPORT
# ==========================================
with tab6:
    st.subheader("Structured Knowledge Export & Governance Reports")
    st.markdown("Export structured architectural artifacts for downstream compliance, safety review, and test case authoring.")

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        st.markdown("### Architecture JSON Export")
        st.caption("Machine-readable export of all SWCs, ports, interfaces, and signals.")
        json_str = json.dumps(arch, indent=2)
        st.download_button(
            "📥 Download Architecture Knowledge JSON",
            json_str,
            f"{selected_doc_name}_architecture.json",
            "application/json"
        )
        with st.expander("Preview JSON"):
            st.json(arch)

    with col_exp2:
        st.markdown("### Engineering Review Report (Markdown)")
        st.caption("Formatted architecture review summary ready for Jira, Confluence, or Tata technical committee.")
        
        def df_to_md(df):
            try:
                return df.to_markdown(index=False)
            except Exception:
                return df.to_string()

        swc_md = df_to_md(pd.DataFrame(arch.get('swc_inventory', []))) if arch.get('swc_inventory') else 'None'
        findings_md = df_to_md(pd.DataFrame(findings)) if 'findings' in locals() and findings else 'No issues detected.'

        md_report = f"""# AUTOSAR Architecture Review Report: {selected_doc_name}
Generated by AUTOSAR HLD Document Analysis Assistant (Tata AI Project)
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

## System Overview
- **Target ECU:** {meta.get('Target ECU')}
- **AUTOSAR Release:** {meta.get('AUTOSAR Release')}
- **Safety Level:** {meta.get('Safety Level')}
- **Approval Status:** {meta.get('Status')}

## Component Statistics
- Total Application SWCs: {arch['stats']['total_swcs']}
- Total Ports & Interfaces: {arch['stats']['total_interfaces']}
- Total Signals: {arch['stats']['total_signals']}

## Software Components
{swc_md}

## Open Issues and Action Items
{findings_md}
"""
        st.download_button(
            "📥 Download Audit Report (Markdown)",
            md_report,
            f"{selected_doc_name}_Review_Report.md",
            "text/markdown"
        )
        with st.expander("Preview Report"):
            st.markdown(md_report)
