"""
Revision Comparison Engine
Compares two versions of AUTOSAR High-Level Design documents (e.g. v1.0 vs v2.0)
and produces a structured architectural delta report.
"""

from typing import Dict, Any, List

class AUTOSARRevisionDiff:
    """
    Computes architectural diffs between baseline and updated HLD designs.
    """

    def __init__(self, baseline_arch: Dict[str, Any], updated_arch: Dict[str, Any]):
        self.base = baseline_arch
        self.updated = updated_arch

    def compute_diff(self) -> Dict[str, Any]:
        """
        Computes differences in SWCs, Interfaces, and metadata.
        """
        base_swcs = {s["swc_name"]: s for s in self.base.get("swc_inventory", [])}
        updated_swcs = {s["swc_name"]: s for s in self.updated.get("swc_inventory", [])}

        added_swcs = [s for name, s in updated_swcs.items() if name not in base_swcs]
        removed_swcs = [s for name, s in base_swcs.items() if name not in updated_swcs]
        
        modified_swcs = []
        for name in base_swcs:
            if name in updated_swcs:
                b = base_swcs[name]
                u = updated_swcs[name]
                changes = []
                if b.get("periodicity") != u.get("periodicity"):
                    changes.append(f"Periodicity changed: {b.get('periodicity')} -> {u.get('periodicity')}")
                if b.get("asil_level") != u.get("asil_level"):
                    changes.append(f"ASIL Level changed: {b.get('asil_level')} -> {u.get('asil_level')}")
                if changes:
                    modified_swcs.append({
                        "swc_name": name,
                        "changes": changes,
                        "baseline": b,
                        "updated": u
                    })

        return {
            "baseline_doc": self.base.get("document_name", "Baseline"),
            "updated_doc": self.updated.get("document_name", "Updated"),
            "swcs_added": added_swcs,
            "swcs_removed": removed_swcs,
            "swcs_modified": modified_swcs,
            "summary": {
                "total_added": len(added_swcs),
                "total_removed": len(removed_swcs),
                "total_modified": len(modified_swcs)
            }
        }
