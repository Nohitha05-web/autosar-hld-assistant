"""
Architecture Visualizer Engine
Generates interactive Mermaid diagrams showing AUTOSAR layered architecture,
SWC-to-SWC connections, and BSW service integrations.
"""

from typing import Dict, Any, List

class AUTOSARVisualizer:
    """
    Renders Mermaid diagrams representing AUTOSAR software components,
    interfaces, ports, and BSW mappings.
    """

    def __init__(self, architecture_data: Dict[str, Any]):
        self.arch = architecture_data
        self.swcs = self.arch.get("swc_inventory", [])
        self.ports = self.arch.get("ports_interfaces", [])
        self.bsw = self.arch.get("bsw_stack", {})

    def generate_mermaid_diagram(self) -> str:
        """
        Constructs an AUTOSAR layered Mermaid flowchart.
        """
        lines = [
            "flowchart TD",
            "    %% AUTOSAR Layer Styling",
            "    classDef asw fill:#EBF8FF,stroke:#2B6CB0,stroke-width:2px,color:#1A365D;",
            "    classDef cdd fill:#FEFCBF,stroke:#B7791F,stroke-width:2px,color:#744210;",
            "    classDef bsw fill:#EDF2F7,stroke:#4A5568,stroke-width:2px,color:#2D3748;",
            "    classDef rte fill:#E2E8F0,stroke:#718096,stroke-width:1.5px,stroke-dasharray: 5 5,color:#1A202C;",
            "",
            "    subgraph ASW[\"Application Software Layer (ASW)\"]"
        ]

        # Add SWCs
        for swc in self.swcs:
            name = swc.get("swc_name", "")
            period = swc.get("periodicity", "")
            asil = swc.get("asil_level", "")
            label = f"\"{name}<br/><small>{period} | {asil}</small>\""
            
            if name.startswith("CDD_"):
                lines.append(f"        {name}[{label}]:::cdd")
            else:
                lines.append(f"        {name}[{label}]:::asw")

        lines.append("    end")
        lines.append("")

        # Add BSW Layer
        lines.append("    subgraph BSW[\"Basic Software Layer (BSW)\"]")
        lines.append("        BSW_DEM[\"Diagnostic Event Mgr (DEM)\"]:::bsw")
        lines.append("        BSW_NvM[\"Non-Volatile Memory (NvM)\"]:::bsw")
        lines.append("        BSW_COM[\"ComStack (CanIf / PduR / COM)\"]:::bsw")
        lines.append("    end")
        lines.append("")

        # Connections via Ports & Interfaces
        lines.append("    %% Interface Connections")
        for p in self.ports:
            src = p.get("provider_swc", "")
            tgt = p.get("consumer_swc", "")
            if_name = p.get("interface_name", "")
            if_type = p.get("interface_type", "S/R")

            # Clean names for Mermaid node IDs
            tgt_clean = "BSW_DEM" if "DEM" in tgt else ("BSW_NvM" if "NvM" in tgt else tgt)
            src_clean = "BSW_DEM" if "DEM" in src else ("BSW_NvM" if "NvM" in src else src)

            if src_clean and tgt_clean and src_clean != tgt_clean:
                connector = "-->" if if_type == "S/R" else "==>"
                lines.append(f"    {src_clean} {connector}|\"{if_name} ({if_type})\"| {tgt_clean}")

        return "\n".join(lines)


if __name__ == "__main__":
    test_arch = {
        "swc_inventory": [
            {"swc_name": "SWC_ExteriorLighting", "periodicity": "10 ms", "asil_level": "ASIL-B"},
            {"swc_name": "SWC_CentralLocking", "periodicity": "20 ms", "asil_level": "ASIL-A"},
            {"swc_name": "CDD_SmartSmartActuator", "periodicity": "5 ms", "asil_level": "ASIL-B"}
        ],
        "ports_interfaces": [
            {"provider_swc": "SWC_ExteriorLighting", "consumer_swc": "CDD_SmartSmartActuator", "interface_name": "If_LightingCommand", "interface_type": "S/R"},
            {"provider_swc": "SWC_ExteriorLighting", "consumer_swc": "BSW_DEM", "interface_name": "If_DiagnosticEvent", "interface_type": "C/S"}
        ]
    }
    viz = AUTOSARVisualizer(test_arch)
    print(viz.generate_mermaid_diagram())
