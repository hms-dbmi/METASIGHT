"""Data preprocessing and loading utilities."""

import torch
import glob
import os
import pandas as pd


def load_features(cancer, foundation_model, slide_type, feature_root):
    """
    Load foundation model features for a single cancer type.
    
    Args:
        cancer: Cancer type (e.g., 'BRCA', 'custom_cancer')
        foundation_model: Foundation model name (e.g., 'CHIEF')
        slide_type: 'FS' | 'PM' | 'MIX'
        feature_root: Root directory for features
    
    Returns:
        feature_dict: {slide_id: tensor}
        max_len: Maximum number of patches across slides
    """
    mag = '20X'
    
    # Try with TCGA prefix first, then without (for custom datasets)
    if slide_type == "MIX":
        # Try TCGA paths first
        fs_path = f"{feature_root}TCGA-{cancer}-FS/{foundation_model}/{mag}/pt_files(stain_norm)/*.pt"
        pm_path = f"{feature_root}TCGA-{cancer}-PM/{foundation_model}/{mag}/pt_files(stain_norm)/*.pt"
        feature_paths = glob.glob(fs_path) + glob.glob(pm_path)
        
        # If no TCGA paths found, try without prefix
        if not feature_paths:
            fs_path = f"{feature_root}{cancer}-FS/{foundation_model}/{mag}/pt_files(stain_norm)/*.pt"
            pm_path = f"{feature_root}{cancer}-PM/{foundation_model}/{mag}/pt_files(stain_norm)/*.pt"
            feature_paths = glob.glob(fs_path) + glob.glob(pm_path)
    else:
        # Try TCGA path first
        path = f"{feature_root}TCGA-{cancer}-{slide_type}/{foundation_model}/{mag}/pt_files(stain_norm)/*.pt"
        feature_paths = glob.glob(path)
        
        # If no TCGA path found, try without prefix
        if not feature_paths:
            path = f"{feature_root}{cancer}-{slide_type}/{foundation_model}/{mag}/pt_files(stain_norm)/*.pt"
            feature_paths = glob.glob(path)
    
    feature_dict = {}
    max_len = 0
    
    for path in feature_paths:
        slide_id = os.path.basename(path).replace('.pt', '')
        tensor = torch.load(path)
        feature_dict[slide_id] = tensor
        max_len = max(max_len, tensor.shape[0])
    
    return feature_dict, max_len


def load_pooled_features(cancer_list, foundation_model, slide_type, feature_root):
    """
    Load foundation model features for multiple cancer types (pooled mode).
    
    Args:
        cancer_list: List of cancer types
        foundation_model: Foundation model name
        slide_type: Slide type
        feature_root: Root directory for features
    
    Returns:
        feature_dict: Combined dictionary for all cancers
        max_len: Maximum number of patches across all slides
    """
    feature_dict = {}
    max_len = 0
    
    for cancer in cancer_list:
        cancer_features, cancer_max_len = load_features(
            cancer, foundation_model, slide_type, feature_root
        )
        feature_dict.update(cancer_features)
        max_len = max(max_len, cancer_max_len)
    
    return feature_dict, max_len


def load_clinical_data(cancer, clinical_root, label_file):
    """
    Load metastasis labels with folder_id for status prediction.
    
    Args:
        cancer: Cancer type
        clinical_root: Root directory for clinical data (not used - kept for compatibility)
        label_file: Path to label CSV (must include folder_id column)
    
    Returns:
        DataFrame with labels and folder_id
    """
    # Load labels - they now contain folder_id
    label_df = pd.read_csv(label_file)
    # Accept multiple project_id formats: TCGA-{CANCER}, TEST-{CANCER}, or just {CANCER}
    cancer_upper = cancer.upper()
    label_df = label_df[
        (label_df['project_id'] == f'TCGA-{cancer_upper}') | 
        (label_df['project_id'] == f'TEST-{cancer_upper}') |
        (label_df['project_id'] == cancer_upper)
    ]
    
    # Verify folder_id is present
    if 'folder_id' not in label_df.columns:
        raise KeyError(f"'folder_id' column missing in label file. Labels must include folder_id to link to features.")
    
    # Add slide type and cancer indicator based on folder_id (if TCGA format)
    # For non-TCGA data, these columns won't filter anything
    def safe_get_slide_type(folder_id):
        try:
            return 'PM' if len(folder_id) > 21 and folder_id[20:22] == 'DX' else 'FS'
        except:
            return 'FS'  # Default for non-TCGA data
    
    def safe_get_cancer_slide(folder_id):
        # Check if TCGA format (starts with "TCGA-")
        if folder_id.startswith('TCGA-'):
            try:
                return 1 if len(folder_id) > 13 and folder_id[13] == '0' else 0
            except:
                return 0
        else:
            return 1  # Non-TCGA data - treat all as cancer slides
    
    label_df['slide_type'] = label_df['folder_id'].apply(safe_get_slide_type)
    label_df['cancer_slide'] = label_df['folder_id'].apply(safe_get_cancer_slide)
    
    # Filter for cancer slides only and valid metastasis labels
    label_df = label_df[label_df['cancer_slide'] == 1]
    # Support both column names: metastasis_status (new) and metastasis_label (legacy)
    label_col = 'metastasis_status' if 'metastasis_status' in label_df.columns else 'metastasis_label'
    label_df = label_df[label_df[label_col].isin([0, 1])]
    
    return label_df.reset_index(drop=True)


def load_trajectory_clinical_data(cancer, cutoff, clinical_root, event_data_path, slide_type_filter=None):
    """
    Load clinical data for trajectory prediction (multi-class classification).
    
    Labels patients at a specific time cutoff:
    - Class 0: Stable disease (no event by cutoff)
    - Class 1: Locoregional recurrence
    - Class 2: Distant metastasis
    
    Args:
        cancer: Cancer type
        cutoff: Time cutoff in days (e.g., 365, 730, 1095)
        clinical_root: Root directory for clinical data
        event_data_path: Path to event data CSV
        slide_type_filter: Optional filter for slide type ('FS' or 'PM')
    
    Returns:
        DataFrame with clinical data, tumor_event_label, and censored indicator
    """
    cancer_upper = cancer.upper()
    
    # Load clinical data for IPCW
    # Try loading from simple structure first (single file for all cancers)
    clinical_path = os.path.join(clinical_root, "clinical_for_ipcw.csv")
    
    if not os.path.exists(clinical_path):
        # Fall back to cancer-specific subdirectory structure
        cancer_dirs = [d for d in os.listdir(clinical_root) if cancer_upper in d]
        if len(cancer_dirs) == 0:
            raise FileNotFoundError(f"Clinical data not found for {cancer_upper}. "
                                    f"Expected either {clinical_root}/clinical_for_ipcw.csv "
                                    f"or {clinical_root}/{{CANCER_DIR}}/clinical_for_ipcw.csv")
        cancer_dir = cancer_dirs[0]
        clinical_path = os.path.join(clinical_root, cancer_dir, "clinical_for_ipcw.csv")
    
    clinical_df = pd.read_csv(clinical_path)
    
    # Filter by cancer type if project_id column exists (support TCGA, TEST, and plain formats)
    if 'project_id' in clinical_df.columns:
        clinical_df = clinical_df[
            (clinical_df['project_id'] == f'TCGA-{cancer_upper}') | 
            (clinical_df['project_id'] == f'TEST-{cancer_upper}') |
            (clinical_df['project_id'] == cancer_upper)
        ].copy()
    
    # Load event data
    event_df = pd.read_csv(event_data_path)
    event_df = event_df[
        (event_df["project_id"] == f"TCGA-{cancer_upper}") | 
        (event_df["project_id"] == f"TEST-{cancer_upper}") |
        (event_df["project_id"] == cancer_upper)
    ]
    event_df["days"] = pd.to_numeric(event_df["days"], errors="coerce")
    event_df = event_df.dropna(subset=["days"])
    
    # Filter for relevant event types
    event_df = event_df[
        event_df["new_tumor_event_type"].isin([
            "No Meta No Recur",
            "Distant Metastasis",
            "Locoregional Recurrence"
        ])
    ].copy()
    
    # Assign labels based on cutoff
    def assign_label(row):
        if row["days"] <= cutoff:
            if row["new_tumor_event_type"] == "Distant Metastasis":
                return 2
            elif row["new_tumor_event_type"] == "Locoregional Recurrence":
                return 1
            else:
                return 0
        else:
            return 0
    
    event_df["tumor_event_label"] = event_df.apply(assign_label, axis=1)
    
    # Censored indicator: patients with "No Meta No Recur" before cutoff
    event_df["censored"] = (
        (event_df["new_tumor_event_type"] == "No Meta No Recur") & 
        (event_df["days"] < cutoff)
    ).astype(int)
    
    # Merge with clinical data for IPCW covariates
    merged_df = pd.merge(
        clinical_df,
        event_df[["case_submitter_id", "tumor_event_label", "censored", "days"]],
        on="case_submitter_id",
        how="inner",
    )
    
    # Get folder_id from event data (labels should have it)
    if "folder_id" in event_df.columns:
        merged_df = pd.merge(
            merged_df,
            event_df[["case_submitter_id", "folder_id"]],
            on="case_submitter_id",
            how="inner",
        )
    else:
        raise KeyError(f"'folder_id' column missing in event data (future_trajectory_label.csv). Labels must include folder_id.")
    
    merged_df = merged_df[merged_df["folder_id"].notnull()].copy()
    
    # Add slide type (generic for both TCGA and non-TCGA data)
    def safe_get_slide_type(folder_id):
        try:
            return 'PM' if len(folder_id) > 21 and folder_id[20:22] == 'DX' else 'FS'
        except:
            return 'FS'  # Default for non-TCGA data
    
    merged_df["slide_type"] = merged_df["folder_id"].apply(safe_get_slide_type)
    
    # Filter by slide type if specified
    if slide_type_filter:
        merged_df = merged_df[merged_df["slide_type"] == slide_type_filter]
    
    return merged_df.reset_index(drop=True)

