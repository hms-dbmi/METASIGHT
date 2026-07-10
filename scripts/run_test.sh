#!/bin/bash
# Quick test run of METASIGHT pipeline with real deidentified data

# Load environment
module load gcc/14.2.0 cuda/12.8 conda/miniforge3/24.11.3-0
conda activate fairtune

# Navigate to repo root (this script lives in scripts/)
cd "$(dirname "$0")/.."

echo "========================================="
echo "METASIGHT Pipeline Test - POOLED MULTI-CANCER"
echo "Data: 30 deidentified samples from 4 cancers"
echo "  - 8 HNSC LRR (Locoregional Recurrence)"
echo "  - 8 LUAD DM (Distant Metastasis)"
echo "  - 7 BRCA Stable, 7 STAD Stable"
echo "  Status: 14 M0, 16 M1"
echo "  Trajectory: 14 Stable, 8 LRR, 8 DM"
echo "  Using 2-fold CV (balanced classes)"
echo "========================================="

# Test 1: Status Prediction (2 epochs for quick test)
echo -e "\n=== Test 1: Status Prediction Module ==="
python scripts/train_status_prediction.py \
  --foundation_model CHIEF \
  --slide_type FS \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --dropout 0.3 \
  --num_epochs 3 \
  --fold_n 3 \
  --loss_type combined \
  --scheduler_type cosine \
  --cancer_list BRCA HNSC LUAD STAD \
  --feature_root example_data/features/ \
  --clinical_root example_data/clinical \
  --label_file example_data/labels/metastasis_status_label.csv \
  --output_dir test_run/results_status

# Test 2: Trajectory Prediction (2 epochs for quick test)
echo -e "\n=== Test 2: Trajectory Prediction Module ==="
python scripts/train_trajectory_prediction.py \
  --foundation_model CHIEF \
  --slide_type FS \
  --batch_size 4 \
  --learning_rate 5e-5 \
  --dropout 0.2 \
  --hidden_dim 128 \
  --num_epochs 3 \
  --fold_n 2 \
  --loss_type ce \
  --cutoffs 730 1095 \
  --use_cross_fit_ipcw \
  --use_class_weight \
  --stratified_cv \
  --cancer_list BRCA HNSC LUAD STAD \
  --feature_root example_data/features/ \
  --clinical_root example_data/clinical \
  --label_file example_data/labels/future_trajectory_label.csv \
  --output_dir test_run/results_trajectory

echo -e "\n========================================="
echo "Tests complete!"
echo "Status results: test_run/results_status/"
echo "Trajectory results: test_run/results_trajectory/"
echo "========================================="

