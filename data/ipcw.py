"""
Inverse Probability of Censoring Weighting (IPCW) for handling censored data in multi-class outcome prediction.

This module implements cross-fit IPCW to prevent data leakage in cross-validation.
Weights are computed separately for each CV fold using only training data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


def compute_ipcw(
    clinical_df: pd.DataFrame,
    covariate_cols: List[str],
    censor_col: str = "censored",
    seed: int = 42,
    n_imputations: int = 5,
    imputer_max_iter: int = 20,
    trunc_q: float = 0.95,
    lr_kwargs: dict = None,
):
    """
    Compute stabilized IPCW using Bayesian MICE and ridge logistic regression.
    
    Args:
        clinical_df: DataFrame with clinical covariates and censoring indicator
        covariate_cols: List of covariate column names
        censor_col: Column name for censoring (0=observed, 1=censored)
        seed: Random seed for reproducibility
        n_imputations: Number of multiple imputations
        imputer_max_iter: Maximum iterations for MICE
        trunc_q: Quantile for weight truncation (default: 0.95)
        lr_kwargs: Additional kwargs for LogisticRegression
    
    Returns:
        weights: pd.Series of IPCW values (mean-scaled to 1)
        imputer: Fitted IterativeImputer object
        X_all: Original covariate matrix (may have missing values)
        imputed_arrays: List of imputed covariate matrices
    """
    if censor_col not in clinical_df.columns:
        raise KeyError(f"'{censor_col}' not found in dataframe")
    
    X_all = clinical_df[covariate_cols]
    y = clinical_df[censor_col].astype(int)
    n = len(clinical_df)
    
    # Bayesian MICE for missing data
    imputer = IterativeImputer(
        random_state=seed,
        max_iter=imputer_max_iter,
        sample_posterior=True,
    ).fit(X_all)
    
    # Logistic regression parameters
    lr_args = dict(
        max_iter=500,
        solver="lbfgs",
        class_weight="balanced",
        penalty="l2",
        C=1
    )
    if lr_kwargs:
        lr_args.update(lr_kwargs)
    
    # Marginal probabilities for stabilization
    p0 = (y == 0).mean()  # P(uncensored)
    p1 = 1 - p0           # P(censored)
    
    w_accum = np.zeros(n)
    imputed_arrays = []
    
    # Multiple imputations
    for k in range(n_imputations):
        X_imp = imputer.transform(X_all)
        logit = LogisticRegression(**lr_args).fit(X_imp, y)
        
        # P(censored | covariates)
        p_cens = logit.predict_proba(X_imp)[:, 1]
        
        # Stabilized IPCW formula
        w_k = np.where(
            y == 0,
            p0 / (1.0 - p_cens + 1e-3),  # Uncensored
            p1 / (p_cens + 1e-3),         # Censored
        )
        
        w_accum += w_k
        imputed_arrays.append(X_imp)
    
    # Average over imputations
    weights = w_accum / n_imputations
    
    # Truncate extreme values
    upper = np.quantile(weights, trunc_q)
    weights = np.clip(weights, None, upper)
    
    # Rescale to mean = 1
    weights /= weights.mean()
    
    return (
        pd.Series(weights, index=clinical_df.index, name="ipcw"),
        imputer,
        X_all,
        imputed_arrays,
    )


def cross_fit_ipcw(
    clinical_df: pd.DataFrame,
    covariate_cols: List[str],
    fold_n: int,
    censor_col: str = "censored",
    seed: int = 42,
):
    """
    Cross-fit IPCW: Compute weights separately for each CV fold.
    
    This prevents data leakage by ensuring test data is never used in
    fitting the censoring model. For each fold:
    - Fit IPCW model on training patients only
    - Predict weights for test patients using the trained model
    
    Args:
        clinical_df: DataFrame with clinical covariates and censoring
        covariate_cols: List of covariate column names
        fold_n: Number of cross-validation folds
        censor_col: Column name for censoring indicator
        seed: Random seed
    
    Returns:
        DataFrame with 'ipcw' column added
    """
    groups = clinical_df["case_submitter_id"]
    folds = GroupKFold(n_splits=fold_n)
    
    clinical_df = clinical_df.copy()
    clinical_df["ipcw"] = 1.0  # Initialize
    
    for fold_id, (train_idx, test_idx) in enumerate(
        folds.split(clinical_df, clinical_df["tumor_event_label"], groups)
    ):
        print(f"  Computing IPCW for fold {fold_id + 1}/{fold_n}")
        
        train_df = clinical_df.iloc[train_idx].copy()
        test_df = clinical_df.iloc[test_idx].copy()
        
        # Fit IPCW on training data only
        ipcw_train, imputer, X_full, _ = compute_ipcw(
            train_df,
            covariate_cols=covariate_cols,
            censor_col=censor_col,
            seed=seed,
            n_imputations=5,
            trunc_q=0.95,
            lr_kwargs=dict(penalty="l2", C=1),
        )
        
        # Predict IPCW for test data using trained model
        lr_args = dict(
            max_iter=500,
            solver="lbfgs",
            class_weight="balanced",
            penalty="l2",
            C=1
        )
        
        X_train_imp = imputer.transform(X_full)
        lr_model = LogisticRegression(**lr_args).fit(
            X_train_imp, train_df[censor_col]
        )
        
        # Transform and predict for test set
        X_test = test_df[covariate_cols]
        X_test_imp = imputer.transform(X_test)
        p_cens_test = lr_model.predict_proba(X_test_imp)[:, 1]
        
        # Compute stabilized weights for test set
        p0 = (train_df[censor_col] == 0).mean()
        p1 = 1 - p0
        
        w_test = np.where(
            test_df[censor_col] == 0,
            p0 / (1.0 - p_cens_test + 1e-3),
            p1 / (p_cens_test + 1e-3),
        )
        
        # Truncate and rescale
        upper = np.quantile(w_test, 0.95)
        w_test = np.clip(w_test, None, upper)
        w_test /= w_test.mean()
        
        # Assign weights
        clinical_df.loc[test_df.index, "ipcw"] = w_test
        clinical_df.loc[train_df.index, "ipcw"] = ipcw_train
    
    return clinical_df


def compute_smd(x1, x2):
    """
    Compute standardized mean difference between two groups.
    
    Args:
        x1: First group values
        x2: Second group values
    
    Returns:
        Standardized mean difference
    """
    mu1, mu2 = np.mean(x1), np.mean(x2)
    sd_pooled = np.sqrt((np.var(x1) + np.var(x2)) / 2)
    return np.abs(mu1 - mu2) / (sd_pooled + 1e-6)


def check_covariate_balance(
    df: pd.DataFrame,
    covariate_cols: List[str],
    weight_col: str = "ipcw",
    censor_col: str = "censored",
) -> pd.DataFrame:
    """
    Check covariate balance before and after IPCW weighting.
    
    Returns DataFrame with standardized mean differences (SMD)
    before and after weighting. SMD < 0.1 indicates good balance.
    
    Args:
        df: DataFrame with covariates, weights, and censoring
        covariate_cols: List of covariate names
        weight_col: Column name for weights
        censor_col: Column name for censoring
    
    Returns:
        DataFrame with balance statistics
    """
    results = []
    
    for col in covariate_cols:
        if df[col].dtype == "object":
            # Categorical variable - check each level
            dummies = pd.get_dummies(df[col], drop_first=True)
            for level in dummies.columns:
                x = dummies[level]
                smd_before = compute_smd(
                    x[df[censor_col] == 1],
                    x[df[censor_col] == 0]
                )
                
                # Weighted SMD (simplified)
                w = df[weight_col]
                c = df[censor_col] == 1
                wc, wnc = w[c], w[~c]
                xc, xnc = x[c], x[~c]
                
                mu_c_w = np.dot(wc, xc) / wc.sum()
                mu_nc_w = np.dot(wnc, xnc) / wnc.sum()
                
                var_c_w = np.dot(wc, (xc - mu_c_w)**2) / wc.sum()
                var_nc_w = np.dot(wnc, (xnc - mu_nc_w)**2) / wnc.sum()
                
                smd_after = np.abs(mu_c_w - mu_nc_w) / np.sqrt((var_c_w + var_nc_w) / 2)
                
                results.append({
                    'covariate': f"{col}_{level}",
                    'smd_before': smd_before,
                    'smd_after': smd_after,
                    'balanced': smd_after < 0.1
                })
        else:
            # Continuous variable
            smd_before = compute_smd(
                df.loc[df[censor_col] == 1, col],
                df.loc[df[censor_col] == 0, col]
            )
            
            # Weighted SMD
            w = df[weight_col]
            c = df[censor_col] == 1
            wc, wnc = w[c], w[~c]
            xc, xnc = df.loc[c, col], df.loc[~c, col]
            
            mu_c_w = np.dot(wc, xc) / wc.sum()
            mu_nc_w = np.dot(wnc, xnc) / wnc.sum()
            
            var_c_w = np.dot(wc, (xc - mu_c_w)**2) / wc.sum()
            var_nc_w = np.dot(wnc, (xnc - mu_nc_w)**2) / wnc.sum()
            
            smd_after = np.abs(mu_c_w - mu_nc_w) / np.sqrt((var_c_w + var_nc_w) / 2)
            
            results.append({
                'covariate': col,
                'smd_before': smd_before,
                'smd_after': smd_after,
                'balanced': smd_after < 0.1
            })
    
    return pd.DataFrame(results)

