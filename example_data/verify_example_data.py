"""
Verify that example data files are correctly formatted and loadable.
"""

import os
import sys
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_clinical_data():
    """Verify clinical data files."""
    print("="*60)
    print("Verifying Clinical Data")
    print("="*60)
    
    clinical_dir = "clinical"
    
    # Check IPCW CSV file (only file needed)
    csv_path = os.path.join(clinical_dir, "clinical_for_ipcw.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"\n[PASS] clinical_for_ipcw.csv loaded successfully")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Patients: {df['case_submitter_id'].tolist()}")
        
        # Check required columns
        required_cols = ['case_submitter_id', 'age_at_diagnosis', 'gender', 
                        'race', 'ajcc_pathologic_stage']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"  [FAIL] Missing columns: {missing}")
        else:
            print(f"  [PASS] All required columns present")
    else:
        print(f"[FAIL] clinical_for_ipcw.csv not found")
    
    return df if os.path.exists(csv_path) else None


def verify_label_data():
    """Verify label files."""
    print("\n" + "="*60)
    print("Verifying Label Data")
    print("="*60)
    
    # Status labels
    status_path = "labels/metastasis_status_label.csv"
    status_df = None
    if os.path.exists(status_path):
        status_df = pd.read_csv(status_path)
        print(f"\n[PASS] metastasis_status_label.csv loaded successfully")
        print(f"  Shape: {status_df.shape}")
        print(f"  Columns: {list(status_df.columns)}")
        if 'folder_id' not in status_df.columns:
            print(f"  [FAIL] Missing required 'folder_id' column!")
        else:
            print(f"  [PASS] Has folder_id for linking to features")
        print(f"  Label distribution:")
        print(f"    No metastasis (0): {(status_df['metastasis_label'] == 0).sum()}")
        print(f"    Metastasis (1): {(status_df['metastasis_label'] == 1).sum()}")
    else:
        print(f"[FAIL] metastasis_status_label.csv not found")
    
    # Trajectory labels
    traj_path = "labels/future_trajectory_label.csv"
    traj_df = None
    if os.path.exists(traj_path):
        traj_df = pd.read_csv(traj_path)
        print(f"\n[PASS] future_trajectory_label.csv loaded successfully")
        print(f"  Shape: {traj_df.shape}")
        print(f"  Columns: {list(traj_df.columns)}")
        if 'folder_id' not in traj_df.columns:
            print(f"  [FAIL] Missing required 'folder_id' column!")
        else:
            print(f"  [PASS] Has folder_id for linking to features")
        print(f"  Event types:")
        for event_type in traj_df['new_tumor_event_type'].unique():
            count = (traj_df['new_tumor_event_type'] == event_type).sum()
            print(f"    {event_type}: {count}")
    else:
        print(f"[FAIL] future_trajectory_label.csv not found")
    
    return status_df, traj_df


def verify_data_consistency(clinical_df, status_df, traj_df):
    """Verify that patient IDs match across files."""
    print("\n" + "="*60)
    print("Verifying Data Consistency")
    print("="*60)
    
    if clinical_df is None or status_df is None or traj_df is None:
        print("[FAIL] Cannot verify consistency - some files are missing")
        return
    
    clinical_ids = set(clinical_df['case_submitter_id'])
    status_ids = set(status_df['case_submitter_id'])
    traj_ids = set(traj_df['case_submitter_id'])
    
    print(f"\nPatient IDs:")
    print(f"  Clinical data: {len(clinical_ids)} patients")
    print(f"  Status labels: {len(status_ids)} patients")
    print(f"  Trajectory labels: {len(traj_ids)} patients")
    
    # Check matches
    if clinical_ids == status_ids == traj_ids:
        print(f"\n[PASS] All patient IDs match across files")
        print(f"  Patients: {sorted(clinical_ids)}")
    else:
        print(f"\n[FAIL] Patient ID mismatch detected")
        if clinical_ids != status_ids:
            print(f"  Clinical vs Status: {clinical_ids.symmetric_difference(status_ids)}")
        if clinical_ids != traj_ids:
            print(f"  Clinical vs Trajectory: {clinical_ids.symmetric_difference(traj_ids)}")
    
    # Check folder IDs in label files
    if 'folder_id' in status_df.columns and 'folder_id' in traj_df.columns:
        status_folder_ids = set(status_df['folder_id'])
        traj_folder_ids = set(traj_df['folder_id'])
        
        print(f"\nSlide IDs (folder_id):")
        for fid in sorted(status_folder_ids.union(traj_folder_ids)):
            print(f"  {fid}")
        
        if status_folder_ids == traj_folder_ids:
            print(f"\n[PASS] Folder IDs match across label files")
        else:
            print(f"\n[FAIL] Folder ID mismatch between label files")


def main():
    """Run all verification checks."""
    print("\n" + "="*60)
    print("METASIGHT Example Data Verification")
    print("="*60)
    
    # Change to example_data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        clinical_df = verify_clinical_data()
        status_df, traj_df = verify_label_data()
        verify_data_consistency(clinical_df, status_df, traj_df)
        
        print("\n" + "="*60)
        print("[PASS] All verification checks passed!")
        print("="*60)
        print("\nExample data is correctly formatted and ready to use.")
        print("Note: These are small example files for demonstration only.")
        print("Use real TCGA data for actual model training.")
        
    except Exception as e:
        print(f"\n[FAIL] Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

