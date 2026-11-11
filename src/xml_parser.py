"""
XML parser for LIDC-IDRI radiologist markup files.

Parses official XML annotation files to extract nodule boundaries,
reader sessions, contour coordinate points (xCoord, yCoord),
slice SOPInstanceUIDs, and nodule diagnostic characteristics.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import xml.etree.ElementTree as ET


class NoduleContourROI:
    """Represents a single 2D contour slice of a nodule marked by one radiologist."""
    def __init__(self, z_position: Optional[float], sop_uid: Optional[str], coords: List[Tuple[int, int]], inclusion: bool = True):
        self.z_position = z_position
        self.sop_uid = sop_uid
        self.coords = coords  # List of (x, y) pixel coordinates on 512x512 grid
        self.inclusion = inclusion


class NoduleAnnotation:
    """Represents a nodule annotated by one reader session, composed of multiple slice ROIs."""
    def __init__(self, nodule_id: str, reader_id: int):
        self.nodule_id = nodule_id
        self.reader_id = reader_id
        self.rois: List[NoduleContourROI] = []
        self.characteristics: Dict[str, int] = {}

    def add_roi(self, roi: NoduleContourROI):
        self.rois.append(roi)


class LIDCXMLParser:
    """
    Parses LIDC-IDRI XML annotation documents into structured Python objects.
    Extracts all reading sessions, unblindedReadNodules, and contour edgeMaps.
    """

    def __init__(self, xml_path: Path):
        self.xml_path = Path(xml_path)
        self.reading_sessions: List[Dict[str, Any]] = []
        self.nodules: List[NoduleAnnotation] = []
        self.non_nodules: List[Dict[str, Any]] = []
        self.num_reading_sessions: int = 4

    def parse(self) -> List[NoduleAnnotation]:
        """Parse the XML file and return all annotated nodule objects."""
        if not self.xml_path.is_file():
            raise FileNotFoundError(f"XML file not found: {self.xml_path}")

        try:
            tree = ET.parse(str(self.xml_path))
            root = tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Malformed XML in {self.xml_path}: {e}")

        # The XML uses a namespace usually: http://www.nih.gov
        # Strip namespaces for simpler querying
        for elem in root.iter():
            if "}" in elem.tag:
                elem.tag = elem.tag.split("}", 1)[1]

        reader_idx = 0
        all_nodules = []
        reading_sessions = root.findall("readingSession")
        self.num_reading_sessions = len(reading_sessions)

        for session in reading_sessions:
            reader_idx += 1
            session_reader_id = reader_idx

            # 1. Process unblindedReadNodule (nodules >= 3mm with contours)
            for nodule_elem in session.findall("unblindedReadNodule"):
                nodule_id_elem = nodule_elem.find("noduleID")
                nodule_id = nodule_id_elem.text.strip() if nodule_id_elem is not None and nodule_id_elem.text else f"nodule_{len(all_nodules)}"
                
                nodule = NoduleAnnotation(nodule_id=nodule_id, reader_id=session_reader_id)

                # Extract nodule characteristics if present
                char_elem = nodule_elem.find("characteristics")
                if char_elem is not None:
                    for child in char_elem:
                        try:
                            if child.text:
                                nodule.characteristics[child.tag] = int(child.text.strip())
                        except ValueError:
                            pass

                # Extract ROIs and edgeMaps (contours)
                for roi_elem in nodule_elem.findall("roi"):
                    z_elem = roi_elem.find("imageZposition")
                    sop_elem = roi_elem.find("imageSOP_UID")
                    inc_elem = roi_elem.find("inclusion")

                    z_pos = float(z_elem.text.strip()) if z_elem is not None and z_elem.text else None
                    sop_uid = sop_elem.text.strip() if sop_elem is not None and sop_elem.text else None
                    inclusion = (inc_elem.text.strip().lower() == "true") if inc_elem is not None and inc_elem.text else True

                    coords = []
                    for edge_map in roi_elem.findall("edgeMap"):
                        x_elem = edge_map.find("xCoord")
                        y_elem = edge_map.find("yCoord")
                        if x_elem is not None and y_elem is not None and x_elem.text and y_elem.text:
                            coords.append((int(x_elem.text.strip()), int(y_elem.text.strip())))

                    if coords:
                        roi = NoduleContourROI(
                            z_position=z_pos,
                            sop_uid=sop_uid,
                            coords=coords,
                            inclusion=inclusion
                        )
                        nodule.add_roi(roi)

                if nodule.rois:
                    all_nodules.append(nodule)

        self.nodules = all_nodules
        return self.nodules

    def get_summary(self) -> Dict[str, Any]:
        """Returns statistics of the parsed XML."""
        readers = set(n.reader_id for n in self.nodules)
        total_rois = sum(len(n.rois) for n in self.nodules)
        return {
            "xml_path": str(self.xml_path),
            "num_readers": len(readers),
            "num_reading_sessions": getattr(self, "num_reading_sessions", len(readers)),
            "num_nodules": len(self.nodules),
            "total_rois": total_rois
        }


def find_matching_ct_xml(xml_or_raw_dir: Path, series_uid: str) -> Optional[Path]:
    """
    Finds the specific CT XML annotation file matching a CT SeriesInstanceUID.
    Parses XML headers to verify the SeriesInstanceUid match and presence of CT nodule markups.
    """
    xml_or_raw_dir = Path(xml_or_raw_dir)
    if not xml_or_raw_dir.exists():
        return None

    xml_candidates = list(xml_or_raw_dir.rglob("*.xml")) if xml_or_raw_dir.is_dir() else [xml_or_raw_dir]
    for xml_p in xml_candidates:
        try:
            tree = ET.parse(str(xml_p))
            root = tree.getroot()
            for elem in root.iter():
                tag = elem.tag.split("}", 1)[-1] if "}" in elem.tag else elem.tag
                if tag == "SeriesInstanceUid" and elem.text and elem.text.strip() == series_uid:
                    # Verify it has CT nodule markups
                    for child in root.iter():
                        c_tag = child.tag.split("}", 1)[-1] if "}" in child.tag else child.tag
                        if c_tag == "unblindedReadNodule":
                            return xml_p
        except Exception:
            pass
    return None
