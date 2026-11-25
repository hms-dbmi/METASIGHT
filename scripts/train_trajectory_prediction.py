"""
METASIGHT: Future Trajectory Prediction Module

Multi-class classification for predicting patient outcome at time horizons:
- Class 0: Stable disease (no event)
- Class 1: Locoregional recurrence  
- Class 2: Distant metastasis

Uses IPCW (Inverse Probability of Censoring Weighting) to handle censored data.
"""

import os
import sys
import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TrajectoryMILNet, OrdinalDistanceLoss, get_loss_function
from data import (WSIDataset, collate_fn, load_trajectory_clinical_data, cross_fit_ipcw,
                  load_feature_file, get_feature_dim, FEATURE_SUBDIR)
from training import get_scheduler, step_scheduler
from evaluation import aggregate_to_patient, patient_level_metrics, evaluate_per_cancer, format_ci

PROB_COLS = ['pred_prob_class_0', 'pred_prob_class_1', 'pred_prob_class_2']
METRICS = ('auroc', 'brier')

# IPCW covariates for censoring model
IPCW_COVARIATES = [
    'age_at_diagnosis', 'gender', 'race', 'ethnicity',
    'ajcc_pathologic_stage', 'ajcc_pathologic_t', 'ajcc_pathologic_n', 'ajcc_pathologic_m'
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="METASIGHT: Future Trajectory Prediction Module")
    
    # Model configuration
    parser.add_argument("--foundation_model", type=str, required=True,
                       choices=['CHIEF', 'GIGAPATH', 'KEEP', 'MUSK'])
    parser.add_argument("--slide_type", type=str, required=True,
                       choices=['FS', 'PM', 'MIX'])
    parser.add_argument("--feature_level", type=str, default="patch",
                       choices=['patch', 'slide'],
                       help="patch: tile embeddings (CHIEF, GIGAPATH, KEEP, MUSK); "
                            "slide: whole-slide encoder embedding (CHIEF, GIGAPATH)")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--fold_n", type=int, default=5)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip_max_norm", type=float, default=1.0)
    
    # Loss function
    parser.add_argument("--loss_type", type=str, default="ordinal",
                       choices=["ordinal", "ce", "focal", "combined"])
    
    # Scheduler
    parser.add_argument("--scheduler_type", type=str, default="cosine",
                       choices=["none", "cosine", "plateau"])
    parser.add_argument("--scheduler_T0", type=int, default=10)
    
    # Time cutoffs for trajectory prediction (days)
    parser.add_argument("--cutoffs", nargs="+", type=int, default=[365, 730, 1095],
                       help="Time horizons in days (e.g., 365 730 1095 for 1, 2, 3 years)")
    
    # IPCW settings
    parser.add_argument("--use_cross_fit_ipcw", action="store_true",
                       help="Use rigorous cross-fit IPCW (recommended)")
    parser.add_argument("--use_class_weight", action="store_true",
                       help="Combine class weights with IPCW weights")
    
    # Data paths
    parser.add_argument("--feature_root", type=str,
                       default="/path/to/foundation_model_features/WSI_features/")
    parser.add_argument("--clinical_root", type=str,
                       default="/path/to/clinical_data/")
    parser.add_argument("--label_file", type=str,
                       default="/path/to/labels/future_trajectory_label.csv")
    
    # Cancer types (trains on all cancers in list)
    parser.add_argument("--cancer_list", nargs="+", 
                       default=["BRCA", "LUAD", "KIRC"],
                       help="List of cancer types to train on")
    parser.add_argument("--min_samples_per_cancer", type=int, default=30,
                       help="Minimum patients for a cancer type to be scored separately")
    
    # Evaluation
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                       help="Patient-level bootstrap resamples for 95%% CIs")
    
    # Stratified CV
    parser.add_argument("--stratified_cv", action="store_true",
                       help="Use StratifiedGroupKFold instead of GroupKFold")
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True)
    
    return parser.parse_args()


def load_features_for_cancer(cancer, foundation_model, slide_type, feature_root,
                             feature_level='patch'):
    """Load feature files for a single cancer type (supports both TCGA and non-TCGA formats)."""
    import glob
    
    model_dir = FEATURE_SUBDIR[feature_level].format(fm=foundation_model)
    feature_paths = []
    # Try both TCGA-prefixed and non-prefixed paths for compatibility
    for prefix in [f'TCGA-{cancer}', cancer]:
        if slide_type == "MIX":
            fs_path = f"{feature_root}{prefix}-FS/{model_dir}/20X/pt_files(stain_norm)/*.pt"
            pm_path = f"{feature_root}{prefix}-PM/{model_dir}/20X/pt_files(stain_norm)/*.pt"
            feature_paths.extend(glob.glob(fs_path))
            feature_paths.extend(glob.glob(pm_path))
        else:
            path = f"{feature_root}{prefix}-{slide_type}/{model_dir}/20X/pt_files(stain_norm)/*.pt"
            feature_paths.extend(glob.glob(path))
    
    # Remove duplicates
    feature_paths = list(set(feature_paths))
    
    feature_dict = {}
    for fpath in feature_paths:
        slide_id = os.path.basename(fpath).replace('.pt', '')
        feature_dict[slide_id] = load_feature_file(fpath)
    
    return feature_dict


def compute_ipcw_weights(clinical_df, covariate_cols, fold_n, use_cross_fit, seed=42):
    """Compute IPCW weights using appropriate method."""
    if use_cross_fit:
        print("Computing cross-fit IPCW weights...")
        clinical_df = cross_fit_ipcw(
            clinical_df, 
            covariate_cols=covariate_cols,
            fold_n=fold_n,
            censor_col='censored',
            seed=seed
        )
    else:
        print("Computing simple IPCW weights...")
        from data import compute_ipcw
        ipcw_series, _, _, _ = compute_ipcw(
            clinical_df,
            covariate_cols=covariate_cols,
            censor_col='censored',
            seed=seed,
            n_imputations=5,
            trunc_q=0.95
        )
        clinical_df['ipcw'] = ipcw_series
    
    return clinical_df


def train_one_fold(model, train_loader, val_loader, optimizer, scheduler, loss_fn,
                   use_ipcw, use_class_weight, args, device, fold_idx):
    """Train model for one fold."""
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    scaler = GradScaler()
    
    for epoch in range(args.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            features = batch['features'].to(device).float()
            mask = batch.get('mask')
            if mask is not None:
                mask = mask.to(device)
            labels = batch['tumor_event_label'].to(device).long()
            
            # Get IPCW weights if using
            if use_ipcw and 'ipcw' in batch:
                sample_weights = batch['ipcw'].to(device).float()
            else:
                sample_weights = torch.ones(labels.size(0), device=device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(features, mask)
                
                # Compute weighted loss
                if isinstance(loss_fn, OrdinalDistanceLoss):
                    loss = loss_fn(outputs, labels, sample_weights)
                elif isinstance(loss_fn, nn.CrossEntropyLoss):
                    loss_per_sample = nn.functional.cross_entropy(
                        outputs, labels, reduction='none'
                    )
                    loss = (loss_per_sample * sample_weights).mean()
                else:
                    loss = loss_fn(outputs, labels)
            
            scaler.scale(loss).backward()
            
            if args.grad_clip_max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                               max_norm=args.grad_clip_max_norm)
            
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device).float()
                mask = batch.get('mask')
                if mask is not None:
                    mask = mask.to(device)
                labels = batch['tumor_event_label'].to(device).long()
                
                if use_ipcw and 'ipcw' in batch:
                    sample_weights = batch['ipcw'].to(device).float()
                else:
                    sample_weights = torch.ones(labels.size(0), device=device)
                
                outputs = model(features, mask)
                
                if isinstance(loss_fn, OrdinalDistanceLoss):
                    loss = loss_fn(outputs, labels, sample_weights)
                elif isinstance(loss_fn, nn.CrossEntropyLoss):
                    loss_per_sample = nn.functional.cross_entropy(
                        outputs, labels, reduction='none'
                    )
                    loss = (loss_per_sample * sample_weights).mean()
                else:
                    loss = loss_fn(outputs, labels)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                if best_state is not None:
                    model.load_state_dict(best_state)
                break
        
        # Step scheduler
        step_scheduler(scheduler, val_loss=val_loss)
    
    return best_state, best_val_loss


def evaluate_fold(model, test_loader, device):
    """Slide-level class probabilities on the test set (metrics are computed per patient later)."""
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device).float()
            mask = batch.get('mask')
            if mask is not None:
                mask = mask.to(device)
            
            outputs = model(features, mask)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            test_preds.append(probs)
            test_labels.extend(batch['tumor_event_label'].cpu().numpy())
    
    return np.vstack(test_preds), np.array(test_labels)


def main():
    args = parse_args()
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_len = get_feature_dim(args.foundation_model, args.feature_level)
    
    print(f"METASIGHT - Future Trajectory Prediction Module")
    print(f"Foundation Model: {args.foundation_model} ({args.feature_level} level, {feature_len}D)")
    print(f"Slide Type: {args.slide_type}")
    print(f"Time Horizons: {args.cutoffs} days")
    print(f"Output: {args.output_dir}")
    
    # Process each time cutoff
    for cutoff in args.cutoffs:
        print(f"\n{'='*60}")
        print(f"Training for {cutoff}-day horizon ({cutoff/365:.1f} years)")
        print(f"{'='*60}")
        
        cutoff_dir = os.path.join(args.output_dir, f"cutoff_{cutoff}days")
        os.makedirs(cutoff_dir, exist_ok=True)
        
        # Load data for this cutoff (all cancers in cancer_list)
        print(f"Loading {len(args.cancer_list)} cancer type(s): {', '.join(args.cancer_list)}")
        all_clinical = []
        feature_dict = {}
        
        for cancer in args.cancer_list:
            try:
                # Load clinical data
                cancer_df = load_trajectory_clinical_data(
                    cancer, cutoff, args.clinical_root,
                    args.label_file, slide_type_filter=args.slide_type if args.slide_type != "MIX" else None
                )
                if cancer_df is None:
                    continue
                
                cancer_df['cancer_type'] = cancer
                all_clinical.append(cancer_df)
                
                # Load features
                cancer_features = load_features_for_cancer(
                    cancer, args.foundation_model, args.slide_type, args.feature_root,
                    args.feature_level
                )
                feature_dict.update(cancer_features)
                
                print(f"  {cancer}: {len(cancer_df)} slides")
            except Exception as e:
                print(f"  {cancer}: Failed - {e}")
                continue
        
        if not all_clinical:
            print(f"No data loaded for cutoff {cutoff}, skipping...")
            continue
        
        res_df = pd.concat(all_clinical, ignore_index=True)
        cancer_name = "_".join(args.cancer_list) if len(args.cancer_list) <= 3 else f"{len(args.cancer_list)}cancers"
        
        # Filter to slides with features
        res_df = res_df[res_df['folder_id'].isin(feature_dict.keys())].reset_index(drop=True)
        print(f"Total: {len(res_df)} slides with features")
        
        # Check class distribution
        print(f"Class distribution:")
        for cls in [0, 1, 2]:
            n = (res_df['tumor_event_label'] == cls).sum()
            print(f"  Class {cls}: {n} ({100*n/len(res_df):.1f}%)")
        
        # Compute IPCW weights
        available_covariates = [col for col in IPCW_COVARIATES if col in res_df.columns]
        if len(available_covariates) > 0:
            res_df = compute_ipcw_weights(
                res_df, available_covariates, args.fold_n,
                args.use_cross_fit_ipcw, args.seed
            )
            print(f"IPCW weights computed using {len(available_covariates)} covariates")
        else:
            print("Warning: No IPCW covariates available, using uniform weights")
            res_df['ipcw'] = 1.0
        
        # Cross-validation setup
        if args.stratified_cv:
            cv_splitter = StratifiedGroupKFold(n_splits=args.fold_n, shuffle=True, 
                                               random_state=args.seed)
        else:
            cv_splitter = GroupKFold(n_splits=args.fold_n)
        
        res_df['fold'] = -1
        for fold_idx, (_, test_idx) in enumerate(
            cv_splitter.split(X=res_df, y=res_df['tumor_event_label'], 
                            groups=res_df['case_submitter_id'])
        ):
            res_df.loc[test_idx, 'fold'] = fold_idx
        
        # Train each fold
        fold_preds = []
        
        for fold_idx in range(args.fold_n):
            print(f"\nFold {fold_idx + 1}/{args.fold_n}")
            
            # Split data
            outer_train_df = res_df[res_df['fold'] != fold_idx].reset_index(drop=True)
            test_df = res_df[res_df['fold'] == fold_idx].reset_index(drop=True)
            
            # Inner validation split
            inner_splitter = GroupKFold(n_splits=5)
            inner_train_idx, val_idx = next(inner_splitter.split(
                X=outer_train_df, y=outer_train_df['tumor_event_label'],
                groups=outer_train_df['case_submitter_id']
            ))
            train_df = outer_train_df.iloc[inner_train_idx].reset_index(drop=True)
            val_df = outer_train_df.iloc[val_idx].reset_index(drop=True)
            
            # Create datasets
            train_dataset = WSIDataset(train_df, feature_dict=feature_dict)
            val_dataset = WSIDataset(val_df, feature_dict=feature_dict)
            test_dataset = WSIDataset(test_df, feature_dict=feature_dict)
            
            # Create dataloaders
            batch_size = min(args.batch_size, len(train_df))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=collate_fn, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                   collate_fn=collate_fn, num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     collate_fn=collate_fn, num_workers=4, pin_memory=True)
            
            # Initialize model
            use_batchnorm = len(train_df) >= 16
            model = TrajectoryMILNet(
                feature_dim=feature_len,
                n_classes=3,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                use_batchnorm=use_batchnorm
            ).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            scheduler = get_scheduler(
                optimizer,
                scheduler_type=args.scheduler_type,
                scheduler_params={'T_0': args.scheduler_T0, 'T_mult': 2}
            )
            
            # Setup loss function
            if args.use_class_weight:
                class_weights = compute_class_weight(
                    class_weight="balanced",
                    classes=np.unique(train_df['tumor_event_label']),
                    y=train_df['tumor_event_label']
                )
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
                loss_fn = get_loss_function(args.loss_type, class_weights, device)
            else:
                loss_fn = get_loss_function(args.loss_type, None, device)
            
            # Train
            start_time = time.time()
            best_state, best_val_loss = train_one_fold(
                model, train_loader, val_loader, optimizer, scheduler,
                loss_fn, 'ipcw' in res_df.columns, args.use_class_weight,
                args, device, fold_idx
            )
            train_time = time.time() - start_time
            
            # Evaluate
            test_preds, test_labels = evaluate_fold(model, test_loader, device)
            
            # Save checkpoint
            checkpoint_dir = os.path.join(cutoff_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'fold': fold_idx,
                'cutoff': cutoff,
                'model_state_dict': best_state,
                'val_loss': best_val_loss,
                'hyperparameters': vars(args),
                'foundation_model': args.foundation_model
            }, os.path.join(checkpoint_dir, f"fold{fold_idx}_best.pt"))
            
            # Save predictions
            pred_dir = os.path.join(cutoff_dir, "predictions")
            os.makedirs(pred_dir, exist_ok=True)
            pred_df = pd.DataFrame({
                'patient_id': test_df['case_submitter_id'].values,
                'slide_id': test_df['folder_id'].values,
                'fold_id': fold_idx,
                'true_label': test_labels,
                'pred_prob_class_0': test_preds[:, 0],
                'pred_prob_class_1': test_preds[:, 1],
                'pred_prob_class_2': test_preds[:, 2],
            })
            if 'cancer_type' in test_df.columns:
                pred_df['cancer_type'] = test_df['cancer_type'].values
            pred_df.to_csv(os.path.join(pred_dir, f"fold{fold_idx}_predictions.csv"), index=False)
            fold_preds.append(pred_df)
            
            fold_res = patient_level_metrics(
                aggregate_to_patient(pred_df, PROB_COLS), PROB_COLS, METRICS, n_boot=0
            )
            print(f"Fold {fold_idx + 1}: patient-level AUROC={format_ci(fold_res, 'auroc')}, "
                  f"Brier={format_ci(fold_res, 'brier')} "
                  f"({fold_res['n_patients']} patients), Time={train_time:.1f}s")
        
        # Patient-level evaluation over all out-of-fold predictions
        patient_df = aggregate_to_patient(pd.concat(fold_preds, ignore_index=True), PROB_COLS)
        patient_df.to_csv(os.path.join(cutoff_dir, "patient_predictions.csv"), index=False)
        
        results = patient_level_metrics(patient_df, PROB_COLS, METRICS,
                                        n_boot=args.n_bootstrap, seed=args.seed)
        
        if 'cancer_type' in patient_df.columns:
            per_cancer = evaluate_per_cancer(patient_df, PROB_COLS, METRICS,
                                             min_patients=args.min_samples_per_cancer,
                                             n_boot=args.n_bootstrap, seed=args.seed)
            per_cancer.to_csv(os.path.join(cutoff_dir, "per_cancer_metrics.csv"), index=False)
        
        result_summary = {
            "cancer": cancer_name,
            "cutoff_days": cutoff,
            "foundation_model": args.foundation_model,
            "feature_level": args.feature_level,
            "slide_type": args.slide_type,
            "n_folds": args.fold_n,
            "evaluation": (f"patient level (slide probabilities averaged per patient); AUROC = "
                           f"mean of per-fold macro one-vs-rest AUROC, Brier = macro one-vs-rest, "
                           f"pooled; 95% CI = percentile bootstrap, {args.n_bootstrap} resamples "
                           f"of patients within folds"),
            **results,
            "hyperparameters": vars(args)
        }
        
        # Save results
        with open(os.path.join(cutoff_dir, "results_summary.json"), 'w') as f:
            json.dump(result_summary, f, indent=2)
        
        print(f"\n=== Results for {cutoff}-day horizon (patient level, "
              f"{results['n_patients']} patients) ===")
        print(f"AUROC: {format_ci(results, 'auroc')}")
        print(f"Brier: {format_ci(results, 'brier')}")
    
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

