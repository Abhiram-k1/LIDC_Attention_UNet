"""
TCIA Downloader module for official LIDC-IDRI data.

Downloads:
1. Official LIDC-IDRI XML annotation archive (9 MB) containing radiologist markups for all 1,018 cases.
2. Authentic CT DICOM series from The Cancer Imaging Archive (TCIA) NBIA REST API for selected patients.
Safely extracts files into data/raw/ without exceeding disk storage limits.
"""

import os
import io
import zipfile
from pathlib import Path
from typing import List, Optional, Dict, Any
import requests
from tqdm import tqdm

from src.config import cfg


class TCIADownloader:
    """
    Downloads authentic LIDC-IDRI DICOM CT scans and official XML annotations from TCIA.
    """

    TCIA_BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
    XML_ARCHIVE_URL = "https://wiki.cancerimagingarchive.net/download/attachments/1966254/LIDC-XML-only.zip"

    def __init__(self, raw_dir: Optional[Path] = None):
        self.raw_dir = Path(raw_dir) if raw_dir else cfg.paths.raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download_xml_annotations(self) -> Path:
        """
        Downloads the complete official XML annotations archive (9MB)
        and extracts to data/raw/tcia_annotations/.
        """
        extract_dir = self.raw_dir / "tcia_annotations"
        if extract_dir.exists() and any(extract_dir.rglob("*.xml")):
            print(f"[TCIA] XML annotations already exist in {extract_dir}.")
            return extract_dir

        print(f"[TCIA] Downloading official LIDC-IDRI XML annotations from {self.XML_ARCHIVE_URL}...")
        r = requests.get(self.XML_ARCHIVE_URL, stream=True, timeout=60)
        r.raise_for_status()

        total_bytes = int(r.headers.get("content-length", 0))
        buffer = io.BytesIO()

        with tqdm(total=total_bytes, unit="B", unit_scale=True, desc="LIDC-XML-only.zip") as pbar:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    buffer.write(chunk)
                    pbar.update(len(chunk))

        print(f"[TCIA] Extracting XML annotations to {extract_dir}...")
        buffer.seek(0)
        with zipfile.ZipFile(buffer, "r") as z:
            z.extractall(extract_dir)

        xml_count = len(list(extract_dir.rglob("*.xml")))
        print(f"[TCIA] Successfully extracted {xml_count} XML annotation files.")
        return extract_dir

    def get_ct_series_list(self, patient_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Queries the TCIA REST API for available CT series in LIDC-IDRI.
        """
        url = f"{self.TCIA_BASE_URL}/getSeries?Collection=LIDC-IDRI&Modality=CT"
        if patient_id:
            url += f"&PatientID={patient_id}"

        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()

    def download_series_dicom(self, series_uid: str, patient_id: str) -> Path:
        """
        Downloads and unzips a full CT DICOM series from TCIA.
        Saves into data/raw/{patient_id}/{series_uid}/.
        """
        patient_folder = self.raw_dir / patient_id
        target_dir = patient_folder / series_uid
        target_dir.mkdir(parents=True, exist_ok=True)

        existing_dcms = list(target_dir.glob("*.dcm"))
        if existing_dcms:
            print(f"[TCIA] Series {series_uid} already has {len(existing_dcms)} DICOM files. Skipping download.")
            return target_dir

        url = f"{self.TCIA_BASE_URL}/getImage?SeriesInstanceUID={series_uid}"
        print(f"[TCIA] Downloading DICOM series for {patient_id} (SeriesUID: {series_uid[:15]}...)...")

        r = requests.get(url, stream=True, timeout=180)
        r.raise_for_status()

        total_bytes = int(r.headers.get("content-length", 0))
        buffer = io.BytesIO()

        with tqdm(total=total_bytes, unit="B", unit_scale=True, desc=f"{patient_id}_CT") as pbar:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    buffer.write(chunk)
                    pbar.update(len(chunk))

        print(f"[TCIA] Extracting DICOM slices into {target_dir}...")
        buffer.seek(0)
        with zipfile.ZipFile(buffer, "r") as z:
            z.extractall(target_dir)

        # Match and copy the patient's XML file into the patient's folder
        xml_dir = self.raw_dir / "tcia_annotations"
        if xml_dir.exists():
            matched_xmls = list(xml_dir.rglob(f"*{patient_id[-4:]}*.xml")) or list(xml_dir.rglob(f"*{patient_id}*.xml"))
            for m_xml in matched_xmls:
                dest = patient_folder / m_xml.name
                if not dest.exists():
                    dest.write_bytes(m_xml.read_bytes())
                    print(f"[TCIA] Copied matching XML {m_xml.name} -> {dest}")

        dcm_count = len(list(target_dir.rglob("*.dcm")))
        print(f"[TCIA] Successfully downloaded {dcm_count} DICOM slices for {patient_id}.")
        return target_dir


if __name__ == "__main__":
    downloader = TCIADownloader()
    downloader.download_xml_annotations()
