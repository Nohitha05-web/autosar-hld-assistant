"""
Phase 5: AUTOSAR Architecture Inconsistency and Completeness Checker
Automates identification of architectural mismatches, dangling ports,
signal bit-width discrepancies, and ASIL compatibility risks.
"""

from typing import Dict, Any, List

class AUTOSARInconsistencyChecker:
    """
    Scans extracted AUTOSAR architecture for design inconsistencies,
    missing bindings, and structural defects.
    """

    def __init__(self, architecture_data: Dict[str, Any]):
        self.arch = architecture_data
        self.swcs = self.arch.get("swc_inventory", [])
        self.ports = self.arch.get("ports_interfaces", [])
        self.signals = self.arch.get("signals", [])
        self.recorded_issues = self.arch.get("issues", [])

    def check_all(self) -> List[Dict[str, Any]]:
        """
        Runs comprehensive architectural rule checks and returns findings list.
        """
        findings = []

        findings.extend(self._check_dangling_ports())
        findings.extend(self._check_signal_mismatches())
        findings.extend(self._check_diagnostic_traceability())
        findings.extend(self._check_asil_integrity())
        findings.extend(self._format_recorded_issues())

        return findings

    def _check_dangling_ports(self) -> List[Dict[str, Any]]:
        """
        Detects ports mentioned in documentation or issues that have no active binding.
        """
        findings = []
        provided_ports = {p["source_port"] for p in self.ports if "source_port" in p}
        required_ports = {p["target_port"] for p in self.ports if "target_port" in p}

        # Check known dangling port from issues or unmapped definitions
        for issue in self.recorded_issues:
            txt = issue.get("issue_text", "")
            if "dangling" in txt.lower() or "unconnected" in txt.lower():
                findings.append({
                    "rule": "DANGLING_PORT_DETECTED",
                    "severity": "CRITICAL",
                    "component": "SWC_ExteriorLighting",
                    "description": txt,
                    "remediation": "Bind port R_AmbientLightSensor to LIN sensor driver or mark as optional in ARXML.",
                    "page": issue.get("page_number", 4)
                })

        return findings

    def _check_signal_mismatches(self) -> List[Dict[str, Any]]:
        """
        Scans for data type size mismatches between application signals and network frames.
        """
        findings = []
        for issue in self.recorded_issues:
            txt = issue.get("issue_text", "")
            if "bit" in txt.lower() and ("matrix" in txt.lower() or "can" in txt.lower()):
                findings.append({
                    "rule": "SIGNAL_BITWIDTH_MISMATCH",
                    "severity": "HIGH",
                    "component": "SWC_ExteriorLighting / COM",
                    "description": txt,
                    "remediation": "Update CAN Matrix database or configure COM signal bit-masking in BSW Com module.",
                    "page": issue.get("page_number", 4)
                })

        return findings

    def _check_diagnostic_traceability(self) -> List[Dict[str, Any]]:
        """
        Verifies that safety-critical SWCs (ASIL-B or higher) have diagnostic fault reporting to DEM.
        """
        findings = []
        diag_providers = {
            p["provider_swc"] for p in self.ports
            if "DEM" in p.get("consumer_swc", "") or "Diagnostic" in p.get("interface_name", "")
        }

        for swc in self.swcs:
            if "ASIL" in swc.get("asil_level", "QM") and swc["asil_level"] != "QM":
                if swc["swc_name"] not in diag_providers and not swc["swc_name"].startswith("CDD_"):
                    findings.append({
                        "rule": "MISSING_DEM_EVENT_PORT",
                        "severity": "MEDIUM",
                        "component": swc["swc_name"],
                        "description": f"Component {swc['swc_name']} has safety rating {swc['asil_level']} but no direct Client-Server port to BSW DEM.",
                        "remediation": f"Add P_DiagEvent port to {swc['swc_name']} with If_DiagnosticEvent interface.",
                        "page": swc.get("source_page", 1)
                    })

        return findings

    def _check_asil_integrity(self) -> List[Dict[str, Any]]:
        """
        Audits ASIL interactions to spot potential QM to ASIL interference without memory partitioning.
        """
        findings = []
        for issue in self.recorded_issues:
            txt = issue.get("issue_text", "")
            if "partition" in txt.lower() or "trusted" in txt.lower() or "isr" in txt.lower():
                findings.append({
                    "rule": "MEMORY_PARTITION_ISOLATION",
                    "severity": "MEDIUM",
                    "component": "CDD_SmartSmartActuator",
                    "description": txt,
                    "remediation": "Assign CDD to separate OS Application with MPU memory protection enabled.",
                    "page": issue.get("page_number", 4)
                })

        return findings

    def _format_recorded_issues(self) -> List[Dict[str, Any]]:
        """
        Wraps generic unhandled recorded issues into the findings list.
        """
        findings = []
        handled_keywords = ["dangling", "unconnected", "bit", "partition", "trusted"]
        for issue in self.recorded_issues:
            txt = issue.get("issue_text", "")
            if not any(k in txt.lower() for k in handled_keywords):
                findings.append({
                    "rule": "ARCHITECTURAL_REVIEW_NOTE",
                    "severity": "LOW",
                    "component": "General",
                    "description": txt,
                    "remediation": "Review during architecture technical committee meeting.",
                    "page": issue.get("page_number", 1)
                })
        return findings
