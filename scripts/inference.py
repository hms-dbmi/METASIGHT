"""
METASIGHT Inference Script

Standalone script for running inference on new whole-slide images
using trained METASIGHT models.
"""

import os
import sys
import argparse
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MILNet, TrajectoryMILNet
from data import load_feature_file, get_feature_dim
from evaluation import aggregate_to_patient, patient_level_metrics, evaluate_per_cancer, format_ci


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="METASIGHT Inference")
    
    # Model configuration
    parser.add_argument("--checkpoint", type=str, nargs='+', required=True,
                       help="Trained model checkpoint(s) (.pt). With several checkpoints (e.g. the "
                            "five fold models), their predicted probabilities are averaged")
    parser.add_argument("--module", type=str, required=True,
                       choices=['status', 'trajectory'],
                       help="Prediction module: status (binary) or trajectory (multi-class)")
    parser.add_argument("--feature_level", type=str, default=None,
                       choices=['patch', 'slide'],
                       help="Feature level of the inputs; defaults to the level stored in the checkpoint")
    
    # Input data
    parser.add_argument("--feature_file", type=str, default=None,
                       help="Path to single feature file (.pt)")
    parser.add_argument("--feature_dir", type=str, default=None,
                       help="Path to directory containing multiple feature files")
    parser.add_argument("--slide_list", type=str, default=None,
                       help="Path to CSV with slide_id column")
    
    # Output
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output CSV file for predictions")
    parser.add_argument("--output_format", type=str, default="csv",
                       choices=['csv', 'json'],
                       help="Output format")
    
    # Optional patient-level evaluation
    parser.add_argument("--label_csv", type=str, default=None,
                       help="CSV with slide_id, patient_id, true_label (optional cancer_type); "
                            "if given, predictions are averaged per patient and scored with 95%% CIs")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                       help="Patient-level bootstrap resamples for 95%% CIs")
    parser.add_argument("--min_patients_per_cancer", type=int, default=30)
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                       choices=['cuda', 'cpu'],
                       help="Device for inference")
    
    return parser.parse_args()


def load_checkpoint(checkpoint_path):
    """Load model checkpoint and extract metadata."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract hyperparameters
    if 'hyperparameters' in checkpoint:
        hparams = checkpoint['hyperparameters']
    else:
        # Try to infer from checkpoint keys
        hparams = {}
        if 'foundation_model' in checkpoint:
            hparams['foundation_model'] = checkpoint['foundation_model']
    
    return checkpoint, hparams


def load_model(checkpoint, module, device, feature_level=None):
    """Initialize model and load weights from checkpoint."""
    hparams = checkpoint.get('hyperparameters', {})
    
    # Get foundation model info
    foundation_model = checkpoint.get('foundation_model', 
                                      hparams.get('foundation_model', 'CHIEF'))
    feature_level = feature_level or hparams.get('feature_level', 'patch')
    feature_dim = get_feature_dim(foundation_model, feature_level)
    # Training disables batch norm for folds with < 16 training slides
    use_batchnorm = any(k.endswith('running_mean') for k in checkpoint['model_state_dict'])
    
    # Initialize model based on module type
    if module == 'status':
        model = MILNet(
            feature_dim=feature_dim,
            n_output=2,
            hidden_dim=hparams.get('hidden_dim', 128),
            dropout=hparams.get('dropout', 0.3),
            use_batchnorm=use_batchnorm
        )
    else:  # trajectory
        model = TrajectoryMILNet(
            feature_dim=feature_dim,
            n_classes=3,
            hidden_dim=hparams.get('hidden_dim', 128),
            dropout=hparams.get('dropout', 0.2),
            use_batchnorm=use_batchnorm
        )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, foundation_model, feature_level, feature_dim


def load_features(feature_path, feature_dim):
    """Load feature tensor from file: [n_patches, dim] (patch) or [1, dim] (slide)."""
    features = load_feature_file(feature_path)
    
    if features.dim() != 2 or features.shape[1] != feature_dim:
        raise ValueError(f"Expected 2D tensor [n_instances, {feature_dim}], got shape "
                         f"{tuple(features.shape)}")
    
    return features


def predict_single(models, features, device, module):
    """Run inference on a single slide, averaging the probabilities of all models."""
    features = features.to(device).float()
    
    # Add batch dimension
    features = features.unsqueeze(0)  # [1, n_patches, feature_dim]
    
    with torch.no_grad():
        probs = np.mean([torch.softmax(m(features, mask=None), dim=1).cpu().numpy()[0]
                         for m in models], axis=0)
    
    # Format results based on module
    if module == 'status':
        result = {
            'prob_no_metastasis': float(probs[0]),
            'prob_metastasis': float(probs[1]),
            'prediction': int(probs[1] > 0.5)
        }
    else:  # trajectory
        result = {
            'prob_no_event': float(probs[0]),
            'prob_locoregional': float(probs[1]),
            'prob_metastasis': float(probs[2]),
            'prediction': int(np.argmax(probs))
        }
    
    return result


def run_inference(args):
    """Main inference pipeline."""
    
    # Load checkpoint and model
    models = []
    for checkpoint_path in args.checkpoint:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint, hparams = load_checkpoint(checkpoint_path)
        model, foundation_model, feature_level, feature_dim = load_model(
            checkpoint, args.module, args.device, args.feature_level
        )
        models.append(model)
    print(f"Model loaded: {args.module} module, {foundation_model} {feature_level}-level "
          f"features ({feature_dim}D), {len(models)} checkpoint(s) averaged")
    
    # Collect feature files to process
    feature_files = []
    
    if args.feature_file is not None:
        # Single file
        if not os.path.exists(args.feature_file):
            raise FileNotFoundError(f"Feature file not found: {args.feature_file}")
        feature_files.append(args.feature_file)
        
    elif args.feature_dir is not None:
        # Directory of files
        feature_dir = Path(args.feature_dir)
        if not feature_dir.exists():
            raise FileNotFoundError(f"Feature directory not found: {args.feature_dir}")
        
        if args.slide_list is not None:
            # Use slide list to filter
            slide_df = pd.read_csv(args.slide_list)
            if 'slide_id' not in slide_df.columns:
                raise ValueError("Slide list CSV must have 'slide_id' column")
            
            for slide_id in slide_df['slide_id']:
                fpath = feature_dir / f"{slide_id}.pt"
                if fpath.exists():
                    feature_files.append(str(fpath))
                else:
                    print(f"Warning: Feature file not found for {slide_id}")
        else:
            # Process all .pt files in directory
            feature_files = list(feature_dir.glob("*.pt"))
    else:
        raise ValueError("Must provide either --feature_file or --feature_dir")
    
    if len(feature_files) == 0:
        raise ValueError("No feature files found to process")
    
    print(f"Processing {len(feature_files)} slide(s)...")
    
    # Run inference on each slide
    results = []
    
    for i, fpath in enumerate(feature_files):
        slide_id = Path(fpath).stem
        
        try:
            # Load features
            features = load_features(fpath, feature_dim)
            
            # Run prediction
            prediction = predict_single(models, features, args.device, args.module)
            
            # Add metadata
            prediction['slide_id'] = slide_id
            prediction['feature_file'] = str(fpath)
            prediction['n_patches'] = features.shape[0]
            
            results.append(prediction)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(feature_files)} slides")
                
        except Exception as e:
            print(f"Error processing {slide_id}: {e}")
            continue
    
    print(f"Successfully processed {len(results)}/{len(feature_files)} slides")
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    
    if args.output_format == 'csv':
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output_file, index=False)
        print(f"Results saved to: {args.output_file}")
        
    else:  # json
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output_file}")
    
    # Print summary statistics
    print("\n=== Prediction Summary ===")
    results_df = pd.DataFrame(results)
    
    if args.module == 'status':
        n_positive = (results_df['prediction'] == 1).sum()
        print(f"Predicted metastasis: {n_positive}/{len(results_df)} "
              f"({100*n_positive/len(results_df):.1f}%)")
        print(f"Mean metastasis probability: {results_df['prob_metastasis'].mean():.3f}")
        
    else:  # trajectory
        pred_counts = results_df['prediction'].value_counts().sort_index()
        print("Predicted outcomes:")
        labels = ['No event', 'Locoregional recurrence', 'Distant metastasis']
        for cls, count in pred_counts.items():
            print(f"  {labels[cls]}: {count} ({100*count/len(results_df):.1f}%)")
    
    if args.label_csv is not None:
        evaluate_patient_level(results_df, args)


def evaluate_patient_level(results_df, args):
    """Average slide predictions per patient and score them with bootstrap 95% CIs."""
    labels_df = pd.read_csv(args.label_csv)
    missing = {'slide_id', 'patient_id', 'true_label'} - set(labels_df.columns)
    if missing:
        raise ValueError(f"--label_csv is missing columns: {sorted(missing)}")
    
    if args.module == 'status':
        prob_cols, metrics = ['prob_metastasis'], ('auroc', 'auprc', 'brier')
    else:
        prob_cols, metrics = ['prob_no_event', 'prob_locoregional', 'prob_metastasis'], ('auroc', 'brier')
    
    keep = [c for c in ('slide_id', 'patient_id', 'true_label', 'cancer_type') if c in labels_df.columns]
    merged = results_df.merge(labels_df[keep].astype({'slide_id': str}),
                              on='slide_id', how='inner')
    print(f"\n=== Patient-level evaluation ({len(merged)}/{len(results_df)} slides with labels) ===")
    if merged.empty:
        return
    
    patient_df = aggregate_to_patient(merged, prob_cols)
    stem = os.path.splitext(args.output_file)[0]
    patient_df.to_csv(f"{stem}_patient_predictions.csv", index=False)
    
    res = patient_level_metrics(patient_df, prob_cols, metrics, n_boot=args.n_bootstrap)
    print(f"Patients: {res['n_patients']}")
    for m in metrics:
        print(f"{m.upper()}: {format_ci(res, m)}")
    
    summary = {'module': args.module,
               'evaluation': (f"patient level (slide probabilities averaged per patient); "
                              f"95% CI = percentile bootstrap, {args.n_bootstrap} resamples of patients"),
               **res}
    if 'cancer_type' in patient_df.columns:
        per_cancer = evaluate_per_cancer(patient_df, prob_cols, metrics,
                                         min_patients=args.min_patients_per_cancer,
                                         n_boot=args.n_bootstrap)
        per_cancer.to_csv(f"{stem}_per_cancer_metrics.csv", index=False)
    with open(f"{stem}_patient_metrics.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Patient-level results saved with prefix: {stem}_")


def main():
    args = parse_args()
    
    # Validate arguments
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Run inference
    try:
        run_inference(args)
    except Exception as e:
        print(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    main()

