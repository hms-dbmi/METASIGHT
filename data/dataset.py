"""Dataset class for loading tile-level foundation model features."""

import torch
from torch.utils.data import Dataset
import os


class WSIDataset(Dataset):
    """
    Dataset for loading tile-level foundation model features.
    
    Features are pre-extracted from whole-slide images using foundation models
    (CHIEF, UNI, GIGAPATH, VIRCHOW2) and stored as tensors [n_tiles, feature_dim].
    
    Args:
        df: DataFrame with slide metadata and labels
        feature_dict: Preloaded dictionary of {slide_id: tile_features}
        feature_path: Path to .pt feature files (if feature_dict not provided)
    """
    
    def __init__(self, df, feature_dict=None, feature_path=None):
        self.df = df.reset_index(drop=True)
        self.feature_dict = feature_dict
        self.feature_path = feature_path
        
        if feature_dict is None and feature_path is None:
            raise ValueError("Either feature_dict or feature_path must be provided")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = row['folder_id']
        
        # Load features
        if self.feature_dict is not None:
            features = self.feature_dict.get(slide_id)
        else:
            # Construct path for pooled mode if cancer_type column exists
            if 'cancer_type' in row and 'slide_type' in row:
                cancer_type = row['cancer_type']
                slide_type = row['slide_type']
                # Replace cancer in path
                path_parts = self.feature_path.split('/')
                new_parts = []
                for part in path_parts:
                    if part.startswith('TCGA-'):
                        new_parts.append(f"TCGA-{cancer_type}-{slide_type}")
                    else:
                        new_parts.append(part)
                feature_path = os.path.join('/'.join(new_parts), f"{slide_id}.pt")
            else:
                feature_path = os.path.join(self.feature_path, f"{slide_id}.pt")
            
            if os.path.exists(feature_path):
                features = torch.load(feature_path)
            else:
                features = None
        
        # Collect labels
        labels = {'slide_id': slide_id}
        
        if 'metastasis_label' in row:
            labels['metastasis_label'] = int(row['metastasis_label'])
        if 'tumor_event_label' in row:
            labels['tumor_event_label'] = int(row['tumor_event_label'])
        if 'ipcw' in row:
            labels['ipcw'] = float(row['ipcw'])
        if 'days' in row:
            labels['days'] = float(row['days'])
        if 'event' in row:
            labels['event'] = int(row['event'])
        if 'cancer_type' in row:
            labels['cancer_type'] = row['cancer_type']
        
        return {
            'features': features,
            **labels
        }


def collate_fn(batch):
    """
    Collate function for variable-length tile features.
    
    Pads all slides in batch to max tile count and creates attention masks.
    Each slide has variable number of tiles with foundation model features.
    
    Args:
        batch: List of dicts from WSIDataset
    
    Returns:
        dict with batched tensors and masks
    """
    feature_list = [b['features'] for b in batch if b['features'] is not None]
    slide_ids = [b['slide_id'] for b in batch]
    
    result = {'slide_ids': slide_ids}
    
    if feature_list:
        max_patches = max([f.shape[0] for f in feature_list])
        feature_dim = feature_list[0].shape[1]
        
        feature_batch = []
        mask_batch = []
        
        for b in batch:
            if b['features'] is not None:
                feats = b['features']
                n_patches = feats.shape[0]
                
                # Pad to max length
                if n_patches < max_patches:
                    padding = torch.zeros(max_patches - n_patches, feature_dim, dtype=feats.dtype)
                    feats = torch.cat([feats, padding], dim=0)
                
                feature_batch.append(feats)
                
                # Create mask (1 for real patches, 0 for padding)
                mask = torch.ones(max_patches, dtype=torch.float32)
                mask[n_patches:] = 0
                mask_batch.append(mask)
            else:
                # Handle missing data
                feature_batch.append(torch.zeros(max_patches, feature_dim, dtype=torch.float32))
                mask_batch.append(torch.zeros(max_patches, dtype=torch.float32))
        
        result['features'] = torch.stack(feature_batch)
        result['mask'] = torch.stack(mask_batch)
    else:
        result['features'] = None
        result['mask'] = None
    
    # Batch labels
    if 'metastasis_label' in batch[0]:
        result['metastasis_label'] = torch.tensor(
            [b['metastasis_label'] for b in batch], dtype=torch.long
        )
    if 'tumor_event_label' in batch[0]:
        result['tumor_event_label'] = torch.tensor(
            [b['tumor_event_label'] for b in batch], dtype=torch.long
        )
    if 'ipcw' in batch[0]:
        result['ipcw'] = torch.tensor(
            [b['ipcw'] for b in batch], dtype=torch.float32
        )
    if 'days' in batch[0]:
        import numpy as np
        result['days'] = np.array([b['days'] for b in batch], dtype=np.float32)
    if 'event' in batch[0]:
        import numpy as np
        result['event'] = np.array([b['event'] for b in batch], dtype=np.int64)
    if 'cancer_type' in batch[0]:
        result['cancer_types'] = [b['cancer_type'] for b in batch]
    
    return result

