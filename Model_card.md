# METASIGHT Model Card

## Specification

|  |  |
| ---- | ---- |
| **Description:** | METASIGHT is an agentic artificial intelligence framework developed to predict (i) metastatic status at diagnosis and (ii) future clinical trajectory (stable disease, locoregional recurrence, or distant metastasis) from hematoxylin and eosin (H&E) whole-slide images of primary tumors. The framework integrates pretrained pathology foundation model embeddings, attention-based multiple instance learning (AB-MIL), inverse probability of censoring weighting (IPCW) for trajectory modeling, and LLM-enabled ensemble selection to improve robustness across cancer types and institutions.|
| **Model Type:** | Supervised deep learning ensemble model combining pretrained pathology foundation model embeddings with attention-based multiple instance learning and adaptive ensemble integration.|
| **Developed By:** | Yu Lab, Department of Biomedical Informatics, Harvard Medical School |
| **Status** | Development complete (research use only)|
| **Launch Date:** | 2026 |
| **Version** | v1.0 |

## Intended Use

**Development Background:** METASIGHT was developed to evaluate whether diagnostic primary tumor pathology encodes reproducible signals of metastatic competence and future disease trajectory. The framework was designed for retrospective multi-cohort research studies spanning diverse cancer types and institutional settings.

**Scope:** Prediction from digitized H&E whole-slide images of primary tumors:
- Binary metastatic status at diagnosis (M0 vs M1)
- Time-horizon–specific trajectory (1-, 2-, and 3-year risk of stable disease, locoregional recurrence, or distant metastasis)
<br>Outputs are patient-level probabilistic risk scores.

**Intended Users:** 
- Computational pathology researchers
- Translational oncology researchers
- Clinical AI research teams
<br>Not intended for direct clinical deployment without prospective validation.

**Use cases out of scope:** 
- Intraoperative decision-making
- Pediatric populations
- Non-H&E imaging modalities

## Data

**Data Overview:** Model development used whole-slide images and clinical metadata from TCGA (5,847 patients). External validation included 10,405 pathology samples across six independent cohorts (DFCI, NHS, HPFS, HANCOCK, CPTAC, PLCO), spanning 23 cancer types. The full study comprised 13,867 WSIs and 2,895 TMA cores from 10,353 patients. <br>All datasets were retrospectively collected and de-identified.

**Sensitive Data:** All patient data were de-identified and accessed under institutional review board approvals and data use agreements. Raw pathology images are not redistributed.

**Pre-processing and cleaning:** Slides underwent standardized quality control, tissue detection, artifact exclusion, and color normalization. Gigapixel images were tiled, and patch-level embeddings were extracted using pretrained pathology foundation models. Standardized preprocessing was applied across cohorts to minimize distribution shift.

**Data Split:** 
| Type | Split | Description |
| ---- | ---- | ---- |
| **Training/Validation:** | 10-fold cross-validation (TCGA)| Model development performed using stratified 10-fold cross-validation within TCGA |
| **Testing:** | External cohorts |Fully independent multi-institutional cohorts used for external validation |


## Methodology and Training

**Model Type:** Supervised deep learning ensemble using pretrained pathology foundation model embeddings with attention-based multiple instance learning and adaptive ensemble integration.

**Models Used:** 
- Pretrained pathology foundation models (e.g., CHIEF, UNI, GigaPath, Virchow2)
- Attention-based MIL pooling
- Fully connected classification heads
- LLM-enabled agentic ensemble selection
- IPCW for trajectory modeling under censoring

**Justification:** Pretrained foundation models capture high-dimensional morphologic representations from large-scale pathology data. Attention-based aggregation enables modeling of regional heterogeneity without manual annotation. Agentic ensemble integration mitigates backbone-specific failure modes and improves stability across cancer types and class imbalance. IPCW accounts for variable follow-up and censoring in trajectory prediction.

**Feature Engineering:** No manual handcrafted features were used for prediction. Slide-level risk scores were derived from learned embeddings and attention-weighted aggregation. Nuclear morphometric and tissue composition features were used for interpretability analyses but not as primary predictive inputs.



### Training Methods:

**Training Process:** Models were trained using supervised learning with cross-entropy loss under 10-fold cross-validation within TCGA. Trajectory prediction incorporated IPCW to account for censoring.

**Hyperparameter/Fine Tuning:** All models were trained using the Adam optimizer (learning rate 1×10⁻⁵, batch size 32, 50 epochs) with a cosine annealing learning rate schedule. Weighted cross-entropy loss was applied to account for class imbalance. Hyperparameters were fixed across foundation model backbones. Internal performance was evaluated using outcome-stratified 10-fold cross-validation before independent external validation.

## Evaluation and Performance


### Model Evaluation

**Evaluation Process:** 

**Evaluation Focus:** 
- Primary metric: AUROC (macro-AUROC for trajectory tasks), as AUROC provides threshold-independent discrimination assessment and is robust under class imbalance. 
- Calibration metric: Brier score


**Performance breakdown:** 
- Metastasis status prediction: average AUROC 0.730 (development), median AUROC 0.732 across external cohorts
- Trajectory prediction: pan-cancer macro-AUROC 0.721–0.759 across 1-, 2-, and 3-year horizons
Stable calibration across cohorts (Brier score)
- Significant improvement over baseline pooling models and clinicopathologic-variable models
- Performance preserved in early-stage–restricted analyses

**Performance in Deployment**: Model evaluation was conducted in retrospective research settings using GPU infrastructure. The model has not been prospectively validated or deployed in live clinical workflows.

### Ethical Considerations

**Bias and fairness analysis:** Robustness was evaluated through multi-institutional external validation, calibration assessment, bootstrap uncertainty quantification, and sensitivity analyses (including early-stage restriction and cancer-type embedding tests). Most patients were of European ancestry; further evaluation across broader populations is warranted.

**Implications for human safety:** False negatives may underestimate metastatic risk, while false positives may increase surveillance burden. METASIGHT is intended as a decision-support research tool and not as a standalone clinical decision-maker.

### Caveats

**Caveats and Limitations:** 
- Majority European-ancestry representation
- Trajectory modeling limited to 3-year horizon
- Treatment heterogeneity not fully modeled
- Not approved for clinical use