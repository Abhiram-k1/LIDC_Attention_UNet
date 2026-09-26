"""
Standalone runner for the 6-Model Systematic Ablation Study.

Can be run directly from terminal, Colab, or Kaggle:
python run_ablation.py --epochs 15 --batch_size 8
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is at the head of sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.dataset import build_dataloaders, PatientSplitter
from src.ablation import run_ablation_study
from run_pipeline import run as run_data_prep


def main():
    parser = argparse.ArgumentParser(description="Run 6-Model Systematic Ablation Study on LIDC-IDRI")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs per ablation model")
    parser.add_argument("--batch_size", type=int, default=cfg.training.batch_size, help="Batch size")
    args = parser.parse_args()

    splits_file = cfg.paths.splits_dir / "patient_splits.json"
    processed_files = list(cfg.paths.processed_dir.glob("*.npz"))

    # Auto-prepare data if not already preprocessed
    if not splits_file.exists() or len(processed_files) == 0:
        print("[run_ablation] Preprocessed data or patient splits not found. Running data prep...")
        run_data_prep()

    print(f"\n[run_ablation] Loading patient splits from {splits_file}...")
    splits = PatientSplitter.load_split(splits_file)

    train_loader, val_loader, test_loader = build_dataloaders(
        processed_dir=cfg.paths.processed_dir,
        splits_dict=splits,
        batch_size=args.batch_size,
        num_workers=cfg.training.num_workers
    )

    print(f"\n[run_ablation] Starting 6-model ablation benchmark ({args.epochs} epochs each)...")
    df = run_ablation_study(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=args.epochs
    )

    print("\n================================================================================")
    print("  COMPLETED ABLATION BENCHMARK RESULTS")
    print("================================================================================")
    print(df.to_string())
    print("================================================================================\n")


if __name__ == "__main__":
    main()
