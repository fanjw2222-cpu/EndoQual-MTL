# EndoQual-MTL

Code for degradation-aware self-supervised pretraining, supervised multitask fine-tuning, and evaluation of endocytoscopic image quality. The primary output is overall quality (Poor, Fair, Good); auxiliary outputs are structure clarity, stain wash, focus and brightness.

## Files and implementation status

| File                    | Purpose                                                                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------------------------- |
| `pretrain.py`           | Stage 1, adapted from retained pretraining source                                                             |
| `finetune.py`           | Stage 2, reconstructed from retained baseline code and the documented model design                            |
| `evaluate.py`           | Final-model inference and statistical evaluation                                                              |
| `configs/pretrain.json` | Stage-1 paths and training settings                                                                           |
| `configs/finetune.json` | Stage-2 paths and training settings                                                                           |
| `configs/evaluate.json` | Test cohorts and evaluation settings                                                                          |
| `environment.json`      | Exported verification environment: Python and package versions, GPU information, and checkpoint SHA256 hashes |
| `requirements.lock.txt` | Exact installed Python distribution versions exported from the same verification environment                  |

## Weights, environment and paths

Run commands from the directory containing this README. All configured paths are relative to that directory. Add the separately supplied weights without modifying their contents:

- `weights/endoqual_final.pth`: complete final multitask model used for evaluation.
- `weights/ssl_encoder.pth`: pretrained encoder used to initialize stage 2.

Evaluation requires only the final weight. Retraining stage 1 additionally requires `weights/imagenet_encoder.safetensors.

Use the Python environment in which the model was validated. Dependencies include PyTorch, torchvision, timm, NumPy, pandas, Pillow, Albumentations, OpenCV, safetensors, TensorBoard, tqdm, scikit-learn, matplotlib and openpyxl. Openpyxl is optional for CSV-only evaluation. 

The environment record identifies package versions, available GPU hardware and SHA256 hashes of the retained weights. The lock file records installed distributions; GPU package installation also depends on the platform and PyTorch CUDA build recorded in the environment file. The snapshot describes the current verification environment, not necessarily the historical training environment.

## Data and execution

Edit JSON paths to match the locally retained data. The default locations are `data/unlabeled/`, `data/labeled/`, `data/internal/`, `data/external/` and `data/annotations/`.

Annotation columns:

```text
filename,main_class,structure_clarity,stain_wash,blurriness,brightness,split,patient_id,lesion_id
```

`filename` identifies an image relative to its image root. Main labels are 0/1/2 or Poor/Fair/Good (Chinese equivalents are also accepted). Auxiliary labels are 0/1/2, with higher values indicating better quality. `blurriness` supplies the focus label without reversing its direction. Use consistent pseudonymous patient and lesion identifiers. Patient identifiers are required for patient-clustered intervals; lesion identifiers are required for lesion-clustered intervals.

The supervised script uses the existing `train`/`val` assignments in `train_validation.csv`; it does not create a replacement split. Internal evaluation reads `internal_test.csv` and filters `split=test`; external evaluation reads `external_test.csv` without a split filter by default. Check these settings against the actual files. Missing labels or identifiers are not imputed.

```bash
# Evaluate the retained final model; creates a new run directory.
python evaluate.py --config configs/evaluate.json --stage all

# Check supervised data and paths without starting training.
python finetune.py --config configs/finetune.json --mode check-inputs

# Start a new supervised run from the retained SSL encoder.
python finetune.py --config configs/finetune.json --mode train

# Optional: retrain stage 1 after supplying its ImageNet initializer.
python pretrain.py --config configs/pretrain.json --run
```

To reuse predictions, run `python evaluate.py --stage analyze --run-dir runs/evaluation/ACTUAL_RUN_DIRECTORY`, substituting the directory printed by inference. New training outputs are separate from the retained final checkpoint.

## Configuration

| Setting                      | Stage 1          | Stage 2          |
| ---------------------------- | ---------------- | ---------------- |
| Training batch size          | 64 image pairs   | 64 images        |
| Maximum epochs / seed        | 30 / 42          | 30 / 42          |
| Optimizer                    | AdamW            | AdamW            |
| Encoder / head learning rate | 1e-5 / 1e-4      | 2e-5 / 2e-4      |
| Weight decay                 | 0.05             | 0.10             |
| Warm-up                      | 3 epochs, linear | 3 epochs, linear |
| Subsequent schedule          | Cosine           | Cosine           |
| Initial encoder freeze       | None             | 3 epochs         |
| Early-stopping patience      | 8 epochs         | 15 epochs        |
| Drop-path rate               | 0.30             | 0.30             |

Evaluation uses FP32 and batch size 8. A batch size of 1, if used in a separate single-image latency experiment, describes that benchmark rather than batched statistical evaluation.

## Preprocessing and augmentation

Images are RGB, 224 × 224, normalized using means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. Validation/test resizing is bilinear. Stage-1 degradations are applied before resizing to two independently generated views: each view is clean with probability 0.20; otherwise degradation type and nonzero severity are sampled uniformly.

| Stage-1 degradation            | Mild                 | Moderate               | Severe                 |
| ------------------------------ | -------------------- | ---------------------- | ---------------------- |
| Gaussian blur (sigma=0)        | Kernel 3             | Kernel 5               | Kernel 7               |
| Brightness multiplier; offset  | 0.9 or 1.1; −8 or +8 | 0.8 or 1.2; −18 or +18 | 0.7 or 1.3; −30 or +30 |
| Stain-field amplitude, uniform | 0.10–0.18            | 0.18–0.28              | 0.28–0.40              |
| Occlusion patches; scale       | 1; 0.08              | 2; 0.12                | 3; 0.16                |

Brightness multiplier and offset are sampled independently. The Gaussian stain field uses coordinates −1 to 1, independently sampled center coordinates in [−0.4, 0.4] and sigma in [0.4, 0.8]. Min–max normalization includes 1e-8 in the denominator. RGB values are multiplied by `1 + amplitude * (2 * field - 1)`. Each occlusion width/height fraction is sampled independently from `[0.6 * scale, scale]`, converted to integer pixels and placed within the image. Fill intensity is sampled from integers 0–30 or 220–255, with equal probability. Transformed intensities are clipped to [0,255] and converted to uint8.

Stage-1 training adds horizontal flip, vertical flip and random 90-degree rotation, each with probability 0.5. Validation uses resize/normalization, but its generated degradation pairs remain stochastic.

Stage-2 training uses random resized crop (scale 0.85–1, ratio 0.75–4/3), horizontal/vertical flips and random 90-degree rotation (each p=0.5), coarse dropout (1–8 holes, width/height 8–16 pixels, zero fill, p=0.3), motion blur (kernel 3 or 5, p=0.15), and brightness/contrast limits ±0.1 (p=0.25). Validation/test has no stochastic augmentation. Use the exported package versions; cross-version augmentation equivalence is not assumed.

## Losses and decoding

Stage 1 minimizes degradation-type cross-entropy + severity cross-entropy + 0.5 × pairwise ranking BCE. Classification losses are summed across both views. Ranking uses the difference of scalar quality outputs and target 1 when view A has lower degradation severity; otherwise the target is 0, including equal-severity pairs. Checkpoint selection maximizes `0.35 * type_accuracy + 0.35 * severity_accuracy + 0.30 * ranking_accuracy`; type/severity accuracy uses view A.

Stage 2 uses mean BCE-with-logits for main targets `[y > 0, y > 1]` and focal loss with gamma=2, no class weights, for each auxiliary task. Five task losses are combined as `sum(exp(-s_t) * L_t + s_t)`, with learned log-variances initially zero. Added losses are 0.25 × Smooth L1 (beta=1) between the main expected grade and the mean auxiliary expected score, and 0.02 × center loss (mean squared Euclidean distance between normalized features and the normalized true-class center). Checkpoint selection uses validation main-task macro-F1. No test-set checkpoint selection is performed.

For main logits, let `q0 = sigmoid(logit0)` and `q1 = sigmoid(logit1)`. Compute `[1-q0, max(q0-q1, 1e-8), max(q1, 1e-8)]`, normalize to sum to one, then use argmax. The condition `q1 > q0` is audited separately; clipping is not a monotonic model constraint. Auxiliary outputs use three-class softmax and argmax.

## Evaluation outputs

Outputs include per-image predictions, class distributions, accuracy, macro-F1, class-specific metrics, OvR AUC, Brier score, log loss, ECE and calibration plots, plus ordinal diagnostics. The multiclass Brier score sums three squared errors per image (range 0–2). Calibration uses five fixed equal-width bins; ECE weights absolute bin discrepancies by bin counts. Classwise ECE is computed one-versus-rest and averaged across classes.

The default is 1,000 bootstrap replicates. Patient resampling retains all images of each sampled patient and is the primary uncertainty analysis; lesion and image resampling are sensitivity analyses. These intervals describe image-level performance with clustered uncertainty, not a separate patient-level diagnostic endpoint. Calibration is evaluated without fitting recalibration parameters or selecting thresholds on the test sets. Resolve any incomplete-input status before reporting results.

## Availability and controlled data access

Code, configuration files, environment records and the two retained EndoQual weight files are intended to be supplied as review attachments or through a repository accessible to the reviewers. The accompanying environment record identifies the weight files by SHA256.

Patient images and original split/annotation records are held by the study data custodians. Requests should be directed to the corresponding author using the contact in the manuscript, specifying institutional affiliation, research purpose and requested data. Release is subject to custodian approval, applicable ethics permissions and a data-use agreement. Where authorized, de-identified images and permitted split records can be provided through an institution-approved authenticated, encrypted transfer or controlled analysis environment. 
