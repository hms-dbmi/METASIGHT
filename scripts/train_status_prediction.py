"""
METASIGHT: Metastasis Status Prediction Module

This module trains a MIL model to predict the presence/absence of distant 
metastasis from whole-slide images using foundation model features.
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, average_precision_score

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MILNet, get_loss_function
from data import WSIDataset, collate_fn, load_clinical_data, load_pooled_features
from training import get_scheduler, step_scheduler
from evaluation import compute_fold_aware_metrics, evaluate_per_cancer


# Foundation model dimensions
FOUNDATION_MODELS = {
    'CHIEF': 768,
    'UNI': 1024,
    'GIGAPATH': 1536,
    'VIRCHOW2': 2560
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="METASIGHT: Metastasis Status Prediction Module")
    
    # Model configuration
    parser.add_argument("--foundation_model", type=str, required=True,
                       choices=['CHIEF', 'UNI', 'GIGAPATH', 'VIRCHOW2'])
    parser.add_argument("--slide_type", type=str, required=True,
                       choices=['FS', 'PM', 'MIX'])
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--fold_n", type=int, default=3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip_max_norm", type=float, default=1.0)
    
    # Loss function
    parser.add_argument("--loss_type", type=str, default="combined",
                       choices=["ce", "focal", "combined"])
    
    # Scheduler
    parser.add_argument("--scheduler_type", type=str, default="none",
                       choices=["none", "cosine", "plateau"])
    parser.add_argument("--scheduler_T0", type=int, default=10)
    
    # Data paths
    parser.add_argument("--feature_root", type=str,
                       default="/n/data2/hms/dbmi/kyu/lab/NCKU/foundation_model_features/WSI_features/")
    parser.add_argument("--clinical_root", type=str,
                       default="/n/data2/hms/dbmi/kyu/lab/pet200/clinical_gdc")
    parser.add_argument("--label_file", type=str,
                       default="/n/data2/hms/dbmi/kyu/lab/tik161/Metastasis_STpath/data/labels/metastasis_status_label.csv")
    
    # Cancer types (trains on all cancers in list)
    parser.add_argument("--cancer_list", nargs="+", default=["BRCA", "LUAD", "KIRC"],
                       help="List of cancer types to train on")
    parser.add_argument("--min_samples_per_cancer", type=int, default=30)
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True)
    
    return parser.parse_args()


def train_one_fold(model, train_loader, val_loader, optimizer, scheduler, loss_fn, 
                   args, device, fold_idx):
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
            labels = batch['metastasis_label'].to(device).long()
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(features, mask)
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
                labels = batch['metastasis_label'].to(device).long()
                
                outputs = model(features, mask)
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
    """Evaluate model on test set."""
    model.eval()
    test_preds = []
    test_labels = []
    test_cancers = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device).float()
            mask = batch.get('mask')
            if mask is not None:
                mask = mask.to(device)
            
            outputs = model(features, mask)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            
            test_preds.extend(probs)
            test_labels.extend(batch['metastasis_label'].cpu().numpy())
            
            if 'cancer_types' in batch:
                test_cancers.extend(batch['cancer_types'])
    
    # Compute metrics
    try:
        auroc = roc_auc_score(test_labels, test_preds)
    except ValueError:
        auroc = None
    
    try:
        auprc = average_precision_score(test_labels, test_preds)
    except ValueError:
        auprc = None
    
    return test_preds, test_labels, test_cancers, auroc, auprc


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_len = FOUNDATION_MODELS[args.foundation_model]
    
    print(f"METASIGHT - Status Prediction Module")
    print(f"Foundation Model: {args.foundation_model} ({feature_len}D)")
    print(f"Slide Type: {args.slide_type}")
    print(f"Output: {args.output_dir}")
    
    # Load data for all cancers in cancer_list
    print(f"Training on {len(args.cancer_list)} cancer type(s): {', '.join(args.cancer_list)}")
    
    feature_dict, max_len = load_pooled_features(
        args.cancer_list, args.foundation_model, args.slide_type, args.feature_root
    )
    
    # Load and merge clinical data for all cancers
    label_df = pd.read_csv(args.label_file)
    all_clinical = []
    for cancer in args.cancer_list:
        try:
            cancer_df = load_clinical_data(cancer, args.clinical_root, args.label_file)
            cancer_df['cancer_type'] = cancer
            all_clinical.append(cancer_df)
        except:
            continue
    
    if not all_clinical:
        raise ValueError("No clinical data loaded")
    
    res_df = pd.concat(all_clinical, ignore_index=True)
    res_df = res_df[res_df['folder_id'].isin(feature_dict.keys())]
    cancer_name = "_".join(args.cancer_list) if len(args.cancer_list) <= 3 else f"{len(args.cancer_list)}cancers"
    
    print(f"Loaded {len(res_df)} slides, max patches: {max_len}")
    
    # Cross-validation
    sgkf = StratifiedGroupKFold(n_splits=args.fold_n, shuffle=True, random_state=args.seed)
    res_df['fold'] = -1
    for fold_idx, (_, test_idx) in enumerate(
        sgkf.split(X=res_df, y=res_df['metastasis_label'], groups=res_df['case_submitter_id'])
    ):
        res_df.loc[test_idx, 'fold'] = fold_idx
    
    # Train each fold
    fold_results = []
    
    for fold_idx in range(args.fold_n):
        print(f"\nFold {fold_idx + 1}/{args.fold_n}")
        
        # Split data
        outer_train_df = res_df[res_df['fold'] != fold_idx].reset_index(drop=True)
        test_df = res_df[res_df['fold'] == fold_idx].reset_index(drop=True)
        
        # Inner validation split
        inner_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
        inner_train_idx, val_idx = next(inner_splitter.split(
            X=outer_train_df, y=outer_train_df['metastasis_label'],
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
        model = MILNet(
            feature_dim=feature_len,
            n_output=2,
            hidden_dim=128,
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
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_df['metastasis_label']),
            y=train_df['metastasis_label']
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        loss_fn = get_loss_function(args.loss_type, class_weights, device)
        
        # Train
        start_time = time.time()
        best_state, best_val_loss = train_one_fold(
            model, train_loader, val_loader, optimizer, scheduler,
            loss_fn, args, device, fold_idx
        )
        train_time = time.time() - start_time
        
        # Evaluate
        test_preds, test_labels, test_cancers, auroc, auprc = evaluate_fold(
            model, test_loader, device
        )
        
        print(f"Fold {fold_idx + 1}: AUROC={auroc:.4f if auroc else 'N/A'}, "
              f"AUPRC={auprc:.4f if auprc else 'N/A'}, Time={train_time:.1f}s")
        
        fold_results.append({'auroc': auroc, 'auprc': auprc})
        
        # Save checkpoint
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            'fold': fold_idx,
            'model_state_dict': best_state,
            'val_loss': best_val_loss,
            'hyperparameters': vars(args),
            'foundation_model': args.foundation_model
        }, os.path.join(checkpoint_dir, f"fold{fold_idx}_best.pt"))
        
        # Save predictions
        pred_dir = os.path.join(args.output_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        pred_df = pd.DataFrame({
            'patient_id': test_df['case_submitter_id'].values,
            'slide_id': test_df['folder_id'].values,
            'fold_id': fold_idx,
            'true_label': test_labels,
            'pred_prob_metastasis': test_preds,
        })
        if 'cancer_type' in test_df.columns:
            pred_df['cancer_type'] = test_df['cancer_type'].values
        pred_df.to_csv(os.path.join(pred_dir, f"fold{fold_idx}_predictions.csv"), index=False)
    
    # Aggregate results
    fold_aware_results = compute_fold_aware_metrics(fold_results, metric_names=['auroc', 'auprc'])
    
    result_summary = {
        "cancer": cancer_name,
        "foundation_model": args.foundation_model,
        "slide_type": args.slide_type,
        "n_folds": args.fold_n,
        "mean_auroc": fold_aware_results['auroc_mean'],
        "std_auroc": fold_aware_results['auroc_std'],
        "mean_auprc": fold_aware_results['auprc_mean'],
        "std_auprc": fold_aware_results['auprc_std'],
        "hyperparameters": vars(args)
    }
    
    # Save results
    with open(os.path.join(args.output_dir, "results_summary.json"), 'w') as f:
        json.dump(result_summary, f, indent=2)
    
    print(f"\n=== Final Results ===")
    print(f"AUROC: {fold_aware_results['auroc_mean']:.4f} ± {fold_aware_results['auroc_std']:.4f}")
    print(f"AUPRC: {fold_aware_results['auprc_mean']:.4f} ± {fold_aware_results['auprc_std']:.4f}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

