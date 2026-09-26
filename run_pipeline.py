"""
End-to-End Orchestrator Pipeline for LIDC_Attention_UNet.

Executes all 26 research project steps in rigorous sequence:
1. Environment verification
2. Authentic LIDC-IDRI data acquisition from TCIA (safe footprint)
3. Dataset inspection & DICOM header extraction
4. XML radiologist markup parsing & mask generation
5. Patient-level train/val/test splitting
6. CT preprocessing, lung windowing, and slice selection
7. Baseline U-Net training & evaluation
8. Proposed SSLA U-Net training & evaluation
9. Qualitative visualization & attention map extraction
10. Final research results compilation
"""

import sys
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.tcia_downloader import TCIADownloader
from src.dataset_inspection import DatasetInspector
from src.preprocessing import CTPreprocessor
from src.dataset import PatientSplitter, build_dataloaders
from src.unet import StandardUNet
from src.proposed_model import CrossFeatureSpatialSparseLinearAttentionUNet
from src.train import Trainer
from src.evaluate import evaluate_model_on_test_set
from src.visualize import ResultVisualizer


def find_matching_ct_xml(raw_dir: Path, series_uid: str) -> Path:
    """Finds the specific CT XML annotation file matching a CT SeriesInstanceUID."""
    for xml_p in raw_dir.rglob("*.xml"):
        try:
            tree = ET.parse(xml_p)
            root = tree.getroot()
            for elem in root.iter():
                if elem.tag.endswith("SeriesInstanceUid") and elem.text and elem.text.strip() == series_uid:
                    # Verify it has CT nodule markups
                    if any(e.tag.endswith("unblindedReadNodule") for e in root.iter()):
                        return xml_p
        except Exception:
            pass
    return None


def run():
    print("================================================================================")
    print("  Cross-Feature Spatial Sparse Linear Attention U-Net for Lung Nodule Segmentation")
    print("  Official Dataset: LIDC-IDRI (DICOM CT Scans + XML Radiologist Markups)")
    print("================================================================================\n")

    # ---------------------------------------------------------
    # STEP 1: Environment Verification
    # ---------------------------------------------------------
    print(">>> STEP 1: Verifying Environment & Hardware Device...")
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  Active Compute Device: {cfg.device}")
    print(f"  Project Root Directory: {cfg.paths.project_root}\n")

    # ---------------------------------------------------------
    # DATA ACQUISITION: Download authentic LIDC-IDRI data from TCIA
    # ---------------------------------------------------------
    print(">>> ACQUISITION: Ensuring Authentic LIDC-IDRI Data from TCIA...")
    downloader = TCIADownloader(cfg.paths.raw_dir)
    # 1. Official XML archive (9MB)
    xml_dir = downloader.download_xml_annotations()

    # 2. Download sample authentic patients
    target_patients = ["LIDC-IDRI-0001", "LIDC-IDRI-0003", "LIDC-IDRI-0005"]
    downloaded_series = []

    for pid in target_patients:
        try:
            patient_folder = cfg.paths.raw_dir / pid
            # Check if patient DICOM already exists
            existing_series_dirs = [d for d in patient_folder.iterdir() if d.is_dir()] if patient_folder.exists() else []
            if existing_series_dirs:
                s_dir = existing_series_dirs[0]
                s_uid = s_dir.name
                downloaded_series.append({
                    "patient_id": pid,
                    "series_uid": s_uid,
                    "series_dir": s_dir
                })
                print(f"  Found existing CT series for {pid} in {s_dir.name}")
            else:
                series_list = downloader.get_ct_series_list(patient_id=pid)
                ct_candidates = [s for s in series_list if s.get("Modality") == "CT"]
                if ct_candidates:
                    selected_s = ct_candidates[0]
                    s_uid = selected_s["SeriesInstanceUID"]
                    dcm_dir = downloader.download_series_dicom(series_uid=s_uid, patient_id=pid)
                    downloaded_series.append({
                        "patient_id": pid,
                        "series_uid": s_uid,
                        "series_dir": dcm_dir
                    })
        except Exception as e:
            print(f"[Warning] Error acquiring {pid}: {e}")

    # ---------------------------------------------------------
    # STEP 2-5: Dataset Inspection & DICOM Extraction
    # ---------------------------------------------------------
    print("\n>>> STEPS 2-5: Running Empirical Dataset Inspection...")
    inspector = DatasetInspector(cfg.paths.raw_dir)
    inspection_report = inspector.scan_directory()
    md_report = inspector.generate_report_markdown(cfg.paths.results_dir / "dataset_inspection_report.md")
    print(f"  Patients Discovered: {inspection_report['num_patients']}")
    print(f"  CT Series Discovered: {inspection_report['num_ct_series']}")
    print(f"  Total Slices Discovered: {inspection_report['total_slices']}")
    print(f"  Total Annotated Nodules: {inspection_report['total_annotated_nodules']}\n")

    # ---------------------------------------------------------
    # STEP 6-10: DICOM -> HU, XML -> Masks, Preprocessing
    # ---------------------------------------------------------
    print(">>> STEPS 6-10: Preprocessing CT Slices & Rasterizing XML Contours...")
    preprocessor = CTPreprocessor(
        window=cfg.preprocessing.window,
        target_size=cfg.preprocessing.target_size,
        normalization=cfg.preprocessing.normalization,
        neighbor_margin=cfg.preprocessing.neighbor_slices,
        negative_ratio=cfg.preprocessing.negative_sample_ratio
    )

    all_processed_records = []
    for p_info in downloaded_series:
        pid = p_info["patient_id"]
        s_dir = p_info["series_dir"]
        s_uid = p_info["series_uid"]

        xml_path = find_matching_ct_xml(cfg.paths.raw_dir, s_uid)
        print(f"  Processing Patient {pid} with matching CT XML: {xml_path.name if xml_path else 'None'} ...")

        records = preprocessor.process_series(
            series_dir=s_dir,
            xml_path=xml_path,
            output_dir=cfg.paths.processed_dir
        )
        all_processed_records.extend(records)

    print(f"  Total preprocessed slices saved to disk: {len(all_processed_records)}")
    pos_count = sum(r["is_positive"] for r in all_processed_records)
    print(f"  Positive Nodule Slices: {pos_count} | Negative Slices: {len(all_processed_records) - pos_count}\n")

    # ---------------------------------------------------------
    # STEP 9: Patient-Level Splitting (Strict Zero-Leakage)
    # ---------------------------------------------------------
    print(">>> STEP 9: Patient-Level Train / Val / Test Partitioning...")
    unique_patients = list(set(r["patient_id"] for r in all_processed_records))
    splitter = PatientSplitter()
    split_dict = splitter.create_split(
        patient_ids=unique_patients,
        train_ratio=cfg.splits.train_ratio,
        val_ratio=cfg.splits.val_ratio,
        test_ratio=cfg.splits.test_ratio,
        seed=cfg.splits.random_seed
    )
    splitter.save_split(split_dict, cfg.paths.splits_dir)

    # ---------------------------------------------------------
    # STEP 11: PyTorch DataLoaders
    # ---------------------------------------------------------
    print("\n>>> STEP 11: Constructing DataLoaders with Medical Augmentations...")
    train_loader, val_loader, test_loader = build_dataloaders(
        processed_dir=cfg.paths.processed_dir,
        splits_dict=split_dict,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers
    )

    # ---------------------------------------------------------
    # STEP 12-14: Baseline Standard U-Net Training & Evaluation
    # ---------------------------------------------------------
    print("\n>>> STEPS 12-14: Training Baseline Standard U-Net...")
    baseline_model = StandardUNet(base_channels=cfg.model.base_channels)
    baseline_trainer = Trainer(
        model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name="baseline_standard_unet",
        epochs=min(10, cfg.training.epochs),
        early_stopping_patience=4
    )
    baseline_trainer.fit()

    baseline_test_results = evaluate_model_on_test_set(
        model=baseline_model,
        test_loader=test_loader,
        model_name="baseline_standard_unet"
    )

    # ---------------------------------------------------------
    # STEP 15-21: Proposed Model Training & Evaluation
    # ---------------------------------------------------------
    print("\n>>> STEPS 15-21: Training Proposed SSLA U-Net Architecture...")
    proposed_model = CrossFeatureSpatialSparseLinearAttentionUNet(
        base_channels=cfg.model.base_channels,
        sparse_k=cfg.model.sparse_top_k
    )
    proposed_trainer = Trainer(
        model=proposed_model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name="proposed_ssla_unet",
        epochs=min(10, cfg.training.epochs),
        early_stopping_patience=4
    )
    proposed_trainer.fit()

    proposed_test_results = evaluate_model_on_test_set(
        model=proposed_model,
        test_loader=test_loader,
        model_name="proposed_ssla_unet"
    )

    # ---------------------------------------------------------
    # STEP 23 & 28: Visualizations & Attention Maps
    # ---------------------------------------------------------
    print("\n>>> STEPS 23 & 28: Generating Visualizations & Attention Heatmaps...")
    visualizer = ResultVisualizer()
    vis_path = visualizer.plot_comparative_predictions(
        baseline_model=baseline_model,
        proposed_model=proposed_model,
        test_loader=test_loader,
        num_examples=4
    )
    attn_path = visualizer.plot_attention_maps(
        model=proposed_model,
        test_loader=test_loader
    )

    # ---------------------------------------------------------
    # STEP 25-26: Final Report Compilation
    # ---------------------------------------------------------
    print("\n================================================================================")
    print("  FINAL SCIENTIFIC RESULTS SUMMARY")
    print("================================================================================")
    bm = baseline_test_results["metrics"]
    pm = proposed_test_results["metrics"]

    print(f"{'Metric':<20} | {'Baseline Standard U-Net':<25} | {'Proposed SSLA U-Net':<25}")
    print("-" * 76)
    print(f"{'Dice Score':<20} | {bm['dice']['mean']:.4f} ± {bm['dice']['ci_95']:.4f}             | {pm['dice']['mean']:.4f} ± {pm['dice']['ci_95']:.4f}")
    print(f"{'IoU Score':<20} | {bm['iou']['mean']:.4f} ± {bm['iou']['ci_95']:.4f}             | {pm['iou']['mean']:.4f} ± {pm['iou']['ci_95']:.4f}")
    print(f"{'Precision':<20} | {bm['precision']['mean']:.4f}                    | {pm['precision']['mean']:.4f}")
    print(f"{'Recall / Sens':<20} | {bm['recall']['mean']:.4f}                    | {pm['recall']['mean']:.4f}")
    print(f"{'Specificity':<20} | {bm['specificity']['mean']:.4f}                    | {pm['specificity']['mean']:.4f}")
    print(f"{'F1 Score':<20} | {bm['f1']['mean']:.4f}                    | {pm['f1']['mean']:.4f}")
    print(f"{'Parameters':<20} | {baseline_test_results['total_parameters']:,}                | {proposed_test_results['total_parameters']:,}")
    print(f"{'Latency (ms)':<20} | {baseline_test_results['avg_latency_ms']:.2f} ms                  | {proposed_test_results['avg_latency_ms']:.2f} ms")
    print("=" * 76)
    print("\nPipeline execution completed successfully!")


if __name__ == "__main__":
    run()
