"""
Phase 3/5: AUTOSAR Architecture Extractor
Extracts structured architecture catalogues: SWCs, Ports, Interfaces,
Signals, Runnables, and BSW mappings from parsed document tables and text.
"""

import re
from typing import Dict, Any, List
import pandas as pd

class AUTOSARArchitectureExtractor:
    """
    Extracts structured domain representations from parsed AUTOSAR HLD documents.
    """

    def __init__(self, doc_data: Dict[str, Any]):
        self.doc_data = doc_data
        self.doc_name = doc_data["document_name"]
        self.metadata = doc_data.get("metadata", {})
        self.pages = doc_data.get("pages", [])
        self.combined_text = doc_data.get("combined_text", "")

    def extract_all(self) -> Dict[str, Any]:
        """
        Extracts all architectural artifacts into a single structured dictionary.
        """
        swcs = self.extract_swc_inventory()
        ports_interfaces = self.extract_ports_and_interfaces()
        signals = self.extract_signals()
        runnables = self.extract_runnables()
        bsw_stack = self.extract_bsw_modules()
        issues = self.extract_recorded_issues()

        return {
            "document_name": self.doc_name,
            "metadata": self.metadata,
            "swc_inventory": swcs,
            "ports_interfaces": ports_interfaces,
            "signals": signals,
            "runnables": runnables,
            "bsw_stack": bsw_stack,
            "issues": issues,
            "stats": {
                "total_swcs": len(swcs),
                "total_interfaces": len(ports_interfaces),
                "total_signals": len(signals),
                "total_runnables": len(runnables),
                "total_issues": len(issues)
            }
        }

    def extract_swc_inventory(self) -> List[Dict[str, Any]]:
        """
        Extracts SWCs from tables or text mentioning ApplicationSWC / ComplexDeviceDriver.
        """
        swcs = []
        seen = set()

        for page in self.pages:
            p_num = page["page_number"]
            for table in page.get("tables", []):
                if not table or len(table) < 2:
                    continue
                header = [c.lower() for c in table[0]]
                if any("swc" in h or "component" in h for h in header):
                    # Identified SWC table
                    for row in table[1:]:
                        if len(row) >= 4 and row[0] not in seen:
                            name = row[0].strip()
                            # Clean potential OCR/wrap artifacts
                            name = re.sub(r"\s+", "", name)
                            if name.startswith("SWC_") or name.startswith("CDD_"):
                                swc_type = row[1].strip() if len(row) > 1 else "ApplicationSWC"
                                periodicity = row[2].strip() if len(row) > 2 else "N/A"
                                asil = row[3].strip() if len(row) > 3 else "QM"
                                desc = row[4].strip() if len(row) > 4 else ""
                                swcs.append({
                                    "swc_name": name,
                                    "swc_type": swc_type,
                                    "periodicity": periodicity,
                                    "asil_level": asil,
                                    "description": desc,
                                    "source_page": p_num
                                })
                                seen.add(name)

        return swcs

    def extract_ports_and_interfaces(self) -> List[Dict[str, Any]]:
        """
        Extracts port and interface connections between SWCs and BSW.
        """
        connections = []

        for page in self.pages:
            p_num = page["page_number"]
            for table in page.get("tables", []):
                if not table or len(table) < 2:
                    continue
                header = [c.lower() for c in table[0]]
                if any("interface" in h for h in header) and any("port" in h for h in header):
                    for row in table[1:]:
                        if len(row) >= 6:
                            src_port = re.sub(r"\s+", "", row[0].strip())
                            prov_swc = re.sub(r"\s+", "", row[1].strip())
                            if_name = re.sub(r"\s+", "", row[2].strip())
                            if_type = row[3].strip()
                            tgt_port = re.sub(r"\s+", "", row[4].strip())
                            cons_swc = row[5].strip()
                            
                            connections.append({
                                "source_port": src_port,
                                "provider_swc": prov_swc,
                                "interface_name": if_name,
                                "interface_type": if_type,
                                "target_port": tgt_port,
                                "consumer_swc": cons_swc,
                                "source_page": p_num
                            })

        return connections

    def extract_signals(self) -> List[Dict[str, Any]]:
        """
        Extracts signals, data types, value ranges, and CAN/LIN bus mapping.
        """
        signals = []

        for page in self.pages:
            p_num = page["page_number"]
            for table in page.get("tables", []):
                if not table or len(table) < 2:
                    continue
                header = [c.lower() for c in table[0]]
                if any("signal" in h for h in header) and any("data type" in h or "range" in h for h in header):
                    for row in table[1:]:
                        if len(row) >= 5:
                            sig_name = re.sub(r"\s+", "", row[0].strip())
                            if_name = re.sub(r"\s+", "", row[1].strip())
                            dtype = row[2].strip()
                            v_range = row[3].strip()
                            default_val = row[4].strip() if len(row) > 4 else "N/A"
                            bus_map = row[5].strip() if len(row) > 5 else "RTE Internal"

                            signals.append({
                                "signal_name": sig_name,
                                "interface": if_name,
                                "data_type": dtype,
                                "valid_range": v_range,
                                "default_value": default_val,
                                "bus_mapping": bus_map,
                                "source_page": p_num
                            })

        return signals

    def extract_runnables(self) -> List[Dict[str, Any]]:
        """
        Extracts Runnables, trigger events, OS tasks, and WCET.
        """
        runnables = []

        for page in self.pages:
            p_num = page["page_number"]
            for table in page.get("tables", []):
                if not table or len(table) < 2:
                    continue
                header = [c.lower() for c in table[0]]
                if any("runnable" in h for h in header):
                    for row in table[1:]:
                        if len(row) >= 4:
                            swc_name = re.sub(r"\s+", "", row[0].strip())
                            runnable = re.sub(r"\s+", "", row[1].strip())
                            trigger = row[2].strip()
                            task = row[3].strip()
                            wcet = row[4].strip() if len(row) > 4 else "N/A"

                            runnables.append({
                                "swc_name": swc_name,
                                "runnable_name": runnable,
                                "trigger_event": trigger,
                                "os_task": task,
                                "wcet": wcet,
                                "source_page": p_num
                            })

        return runnables

    def extract_bsw_modules(self) -> Dict[str, List[str]]:
        """
        Extracts mapped BSW stacks and modules mentioned in the design.
        """
        stacks = {
            "Communication Stack (ComStack)": ["CanIf", "CanTp", "PduR", "COM", "LinIf"],
            "Diagnostic Stack (DiagStack)": ["DEM", "DCM", "FIM", "DET"],
            "Memory Stack (MemStack)": ["NvM", "MemIf", "Fee", "Ea"],
            "System & Mode Management": ["EcuM", "BswM", "WdgM", "OS"],
            "Hardware Abstraction / Drivers": ["Mcu", "Dio", "Pwm", "Adc", "Can", "Lin"]
        }

        detected = {}
        for stack_name, modules in stacks.items():
            found_in_text = []
            for mod in modules:
                if re.search(r"\b" + re.escape(mod) + r"\b", self.combined_text):
                    found_in_text.append(mod)
            if found_in_text:
                detected[stack_name] = found_in_text

        return detected

    def extract_recorded_issues(self) -> List[Dict[str, Any]]:
        """
        Extracts documented architectural issues, open points, or review findings.
        """
        issues = []
        pattern = re.compile(r"(?:Issue|Note)-[A-Za-z0-9\-]+:\s*([^\n\r]+)", re.IGNORECASE)

        for page in self.pages:
            p_num = page["page_number"]
            matches = pattern.finditer(page["text"])
            for m in matches:
                full_issue = m.group(0).strip()
                issues.append({
                    "issue_text": full_issue,
                    "page_number": p_num
                })

        return issues
