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

# Foundation model dimensions
FOUNDATION_MODELS = {
    'CHIEF': 768,
    'GIGAPATH': 1536,
    'KEEP': 768,
    'MUSK': 1024
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="METASIGHT Inference")
    
    # Model configuration
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint (.pt file)")
    parser.add_argument("--module", type=str, required=True,
                       choices=['status', 'trajectory'],
                       help="Prediction module: status (binary) or trajectory (multi-class)")
    
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


def load_model(checkpoint, module, device):
    """Initialize model and load weights from checkpoint."""
    hparams = checkpoint.get('hyperparameters', {})
    
    # Get foundation model info
    foundation_model = checkpoint.get('foundation_model', 
                                      hparams.get('foundation_model', 'CHIEF'))
    feature_dim = FOUNDATION_MODELS[foundation_model]
    
    # Initialize model based on module type
    if module == 'status':
        model = MILNet(
            feature_dim=feature_dim,
            n_output=2,
            hidden_dim=hparams.get('hidden_dim', 128),
            dropout=hparams.get('dropout', 0.3),
            use_batchnorm=True
        )
    else:  # trajectory
        model = TrajectoryMILNet(
            feature_dim=feature_dim,
            n_classes=3,
            hidden_dim=hparams.get('hidden_dim', 128),
            dropout=hparams.get('dropout', 0.2),
            use_batchnorm=True
        )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, foundation_model


def load_features(feature_path):
    """Load feature tensor from file."""
    features = torch.load(feature_path)
    
    # Validate
    if not isinstance(features, torch.Tensor):
        raise ValueError(f"Feature file must contain a tensor, got {type(features)}")
    
    if len(features.shape) != 2:
        raise ValueError(f"Expected 2D tensor [n_patches, dim], got shape {features.shape}")
    
    return features


def predict_single(model, features, device, module):
    """Run inference on a single slide."""
    features = features.to(device).float()
    
    # Add batch dimension
    features = features.unsqueeze(0)  # [1, n_patches, feature_dim]
    
    with torch.no_grad():
        logits = model(features, mask=None)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
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
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint, hparams = load_checkpoint(args.checkpoint)
    model, foundation_model = load_model(checkpoint, args.module, args.device)
    print(f"Model loaded: {args.module} module, {foundation_model} features")
    
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
            features = load_features(fpath)
            
            # Run prediction
            prediction = predict_single(model, features, args.device, args.module)
            
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
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
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

