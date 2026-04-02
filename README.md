# fMRI HRF Analysis — LMM ANCOVA + Group Characterization

Reference: McDonough et al. (2025), _Behavioral Sciences_, 15(11), 1457.

> "Interpreting fMRI Studies in Populations with Cerebrovascular Risk: The Use of a Subject-Specific Hemodynamic Response Function"

---

## Project Structure

```
ANCOVA_apply/
├── dataset/                      # 33 subjects (SUB701–SUB826)
│   ├── SUB701_hrf.mat
│   └── ...
├── ancova_hrf.py                 # Script 1: LMM-ANCOVA (Cond1 vs Cond2)
├── hrf_group_analysis.py         # Script 2: HRF Group Analysis (Young vs T2DM)
├── ancova_input_data.csv         # Script 1 input DataFrame
├── ancova_result_summary.csv     # Script 1 results
├── hrf_group_metrics.csv         # Script 2 per-subject HRF metrics
├── hrf_group_lmm_summary.csv     # Script 2 group LMM results
├── ancova_hrf_results.png        # Script 1 visualization
├── hrf_group_analysis.png        # Script 2 visualization
└── README.md
```

---

## Dataset Overview

### Primary HRF Dataset (`dataset/`, 33 subjects)

| Group        | N   | Subject IDs                                                          |
| ------------ | --- | -------------------------------------------------------------------- |
| Young (<=35) | 14  | 701, 702, 703, 705, 709, 712, 714, 715, 716, 718, 720, 721, 722, 725 |
| T2DM         | 5   | 717, 808, 809, 811, 812                                              |
| Mid (36~54)  | 1   | 801 (excluded from group stats, N=1)                                 |
| Unknown      | 13  | 719, 724, 726, 728, 729, 730, 732, 734, 735, 736, 737, 738, 826      |

Each `SUBxxx_hrf.mat` contains `hrf_ROI` struct:

| Field  | Shape      | Description                                                                     |
| ------ | ---------- | ------------------------------------------------------------------------------- |
| `cnt`  | (7, 2)     | Voxel count per ROI per condition (not used as covariate — post-treatment bias) |
| `data` | (7, 2, 13) | HRF time series [ROI x Condition x TimePoint]                                   |
| `all`  | (1, 13)    | Grand-average HRF (condition-independent) — source of shape covariates          |

## Analysis 1 — LMM-ANCOVA: Condition Effect (`ancova_hrf.py`)

### Research Question

Within the same subjects, does HRF peak amplitude differ between **Cond1 and Cond2**, after controlling for individual neurovascular reactivity?

### Model

```
Peak_amp ~ PeakAmp_grand + TTP_grand + C(Condition)   [groups = Subject, REML]
```

| Role                  | Variable        | Description                                                           |
| --------------------- | --------------- | --------------------------------------------------------------------- |
| Dependent variable    | `Peak_amp`      | Baseline-corrected HRF peak (window t=3–6, ±1 tp smoothed, minus t=0) |
| Categorical predictor | `Condition`     | Cond1 vs Cond2                                                        |
| Covariate 1           | `PeakAmp_grand` | Grand-avg HRF peak — neurovascular response strength                  |
| Covariate 2           | `TTP_grand`     | Grand-avg HRF latency — vascular delay index                          |
| Random effect         | `Subject`       | Random intercept (within-subject repeated measures)                   |

> `PeakAmp_grand` and `TTP_grand` are derived from the `all` field (averaged over all ROIs AND conditions), making them **condition-independent** — zero post-treatment bias.

> `Voxel_cnt` was **removed**: activation voxel count varies with Condition (it is a product of the manipulation), so including it would regress out the Condition effect itself (over-correction / post-treatment bias).

### Improvements over OLS ANCOVA

| #   | Problem                                     | Fix                                                      |
| --- | ------------------------------------------- | -------------------------------------------------------- |
| ①   | Within-subject design violates independence | `smf.mixedlm` with Subject random intercept              |
| ②   | `np.max` vulnerable to noise spikes         | Windowed peak (t=3–6) + ±1 tp average                    |
| ③   | Absolute peak ignores baseline offset       | `Peak_amp = smoothed_peak − baseline(t=0)`               |
| ④   | 7 simultaneous tests inflate Type I error   | FDR correction (Benjamini-Hochberg) across ROIs          |
| ⑤   | `Voxel_cnt` varies with Condition           | Replaced with condition-independent HRF shape covariates |

### Results (33 subjects × 7 ROIs × 2 conditions = 462 rows)

| ROI   | z(PeakAmp) | p(PeakAmp)  | z(TTP) | p(TTP) | z(Cond)   | p(Cond)     | p_FDR | Sig |
| ----- | ---------- | ----------- | ------ | ------ | --------- | ----------- | ----- | --- |
| ROI 1 | 8.364      | <0.001 \*\* | -0.243 | 0.808  | -0.170    | 0.865       | 0.865 | No  |
| ROI 2 | 1.644      | 0.100       | 0.773  | 0.439  | -0.321    | 0.748       | 0.865 | No  |
| ROI 3 | 3.153      | 0.002 \*\*  | -0.357 | 0.721  | **2.037** | **0.042\*** | 0.292 | No  |
| ROI 4 | 3.834      | <0.001 \*\* | 0.243  | 0.808  | -1.586    | 0.113       | 0.394 | No  |
| ROI 5 | 0.927      | 0.354       | 0.262  | 0.793  | 0.893     | 0.372       | 0.651 | No  |
| ROI 6 | 5.143      | <0.001 \*\* | -0.095 | 0.925  | -1.255    | 0.210       | 0.489 | No  |
| ROI 7 | 8.479      | <0.001 \*\* | -1.158 | 0.247  | 0.256     | 0.798       | 0.865 | No  |

**Key findings:**

- No ROI reached significance after FDR correction
- `PeakAmp_grand` was a significant covariate in ROIs 1, 3, 4, 6, 7 — confirms individual vascular reactivity strongly predicts HRF peak
- ROI 3 showed a marginal Condition effect (p=0.042 uncorrected; Cond2 M=1.976 > Cond1 M=1.480)
- `TTP_grand` did not significantly predict Peak_amp in any ROI (consistent with paper)

---

## Analysis 2 — HRF Group Characterization (`hrf_group_analysis.py`)

### Research Question

Do HRF shape metrics (peak, latency, FWHM) differ between **Young** and **T2DM** groups? (Analogous to paper Section 3.3)

### HRF Shape Metrics (from `all` field, condition-independent)

| Metric          | Description                                | Paper equivalent           |
| --------------- | ------------------------------------------ | -------------------------- |
| `PeakAmp_grand` | Baseline-corrected peak amplitude          | "Maximum value of the HRF" |
| `TTP_grand`     | Time-to-peak index                         | "Latency"                  |
| `FWHM_grand`    | Full-width at half-maximum (in timepoints) | "FWHM"                     |

### Canonical HRF Reference

SPM12-style double-gamma HRF (TR=1.72 s, normalized to peak=1) — used as overlay in Figure 1 equivalent.

### Group Comparison: Young (n=14) vs T2DM (n=5)

| Metric        | Young M | Young SD | T2DM M | T2DM SD | t      | p     | Cohen's d |
| ------------- | ------- | -------- | ------ | ------- | ------ | ----- | --------- |
| PeakAmp_grand | 1.653   | 0.521    | 1.236  | 0.383   | 1.833  | 0.084 | 1.002     |
| TTP_grand     | —       | —        | —      | —       | 0.449  | 0.659 | —         |
| FWHM_grand    | —       | —        | —      | —       | -0.482 | 0.636 | —         |

**Comparison with paper (McDonough et al., 2025):**

| Paper finding                              | Our result                          | Match                                      |
| ------------------------------------------ | ----------------------------------- | ------------------------------------------ |
| Older age → smaller max HRF peak (p<0.001) | Young > T2DM trend (d=1.0, p=0.084) | Direction matches, underpowered (T2DM n=5) |
| No age effect on latency (p>0.05)          | TTP p=0.659                         | Replicated                                 |
| No age effect on FWHM (p>0.05)             | FWHM p=0.636                        | Replicated                                 |

### Group LMM per ROI (Young vs T2DM)

```
Peak_amp ~ PeakAmp_grand + TTP_grand + C(Condition) + C(Group)   [groups = Subject, REML]
```

| ROI   | z(Group) | p(Group) | p_FDR | M_Young | M_T2DM |
| ----- | -------- | -------- | ----- | ------- | ------ |
| ROI 1 | 1.304    | 0.192    | 0.681 | 2.137   | 1.376  |
| ROI 2 | 0.975    | 0.330    | 0.681 | 2.212   | 1.495  |
| ROI 3 | 1.094    | 0.274    | 0.681 | 1.644   | 1.054  |
| ROI 4 | 0.083    | 0.934    | 0.934 | 2.018   | 1.469  |
| ROI 5 | 0.798    | 0.425    | 0.681 | 1.359   | 0.944  |
| ROI 6 | -0.257   | 0.797    | 0.930 | 2.174   | 1.297  |
| ROI 7 | -0.696   | 0.486    | 0.681 | 1.921   | 1.795  |

Young consistently shows higher mean Peak_amp than T2DM across ROIs (except ROI 7), consistent with paper's direction. No ROI reached significance — limited by T2DM n=5.

---

## Methodology Implementation Status vs Paper

| Block                                          | Paper Section | Status                       | Bottleneck                      |
| ---------------------------------------------- | ------------- | ---------------------------- | ------------------------------- |
| sHRF extraction (FIR-based)                    | 2.3.3         | Complete                     | data pre-extracted in .mat      |
| HRF metrics: max, TTP, FWHM                    | 2.3.3         | Complete                     |                                 |
| Figure 1: sHRF variability + canonical overlay | 3.3/Fig1      | Complete                     |                                 |
| Figure 2-4: HRF by group (3 age groups)        | 3.3/Fig2-4    | Partial (Young vs T2DM only) | Old/Mid .mat files needed       |
| Age (continuous) regression on HRF metrics     | 3.3           | Not possible                 | No age metadata                 |
| EFA: 3 vascular risk factors                   | 2.3.1         | Not possible                 | No BMI/BP/clinical data         |
| LMM: Cond1 vs Cond2 per ROI                    | 3.4           | Complete                     |                                 |
| LMM: Group effect per ROI                      | 3.4           | Complete (Young vs T2DM)     |                                 |
| Canonical vs sHRF GLM comparison               | 3.4-3.5       | Not possible                 | No raw task BOLD data           |
| Age x Vascular Risk whole-brain                | 3.6           | Not possible                 | No clinical metadata + raw BOLD |
| Brain-behavior correlations                    | 3.7           | Not possible                 | No memory performance scores    |

**Overall: ~35%**

---

## Requirements

```bash
pip install numpy pandas scipy statsmodels matplotlib nibabel
```

## Usage

```bash
python ancova_hrf.py          # Analysis 1
python hrf_group_analysis.py  # Analysis 2
```

---

## Next Steps (data needed to advance)

| What is needed                        | Enables                                                 |
| ------------------------------------- | ------------------------------------------------------- |
| Age, BMI, BP per subject (SUB701–826) | Age continuous regression; EFA; Age x Risk interactions |
| Old (>=55) group `.mat` files         | Full 3-group comparison as in paper                     |
| Memory task performance scores        | Brain-behavior correlations (Block D)                   |
| Raw task BOLD (encoding session)      | Canonical vs sHRF GLM comparison (Block B)              |
