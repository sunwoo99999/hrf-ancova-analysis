"""
=============================================================
LMM-based ANCOVA Analysis: fMRI HRF Data
(7 ROIs x 2 Conditions x 26 Subjects)
=============================================================
[Analysis Design]
- Dependent variable (Y) : Baseline-corrected HRF peak amplitude
                           (peak averaged ±1 tp in window t=3~6, minus t=0 baseline)
- Group (categorical)    : Condition 1 vs Condition 2
- Covariates             : HRF shape features from per-subject grand-average HRF ('all' field)
                           ① TTP_grand     — time-to-peak of grand-average HRF (vascular latency index)
                           ② PeakAmp_grand — baseline-corrected peak of grand-average HRF
                                             (subject-level neurovascular response strength)
                           Both are extracted from 'all', which averages across all ROIs and
                           conditions → entirely condition-independent → no post-treatment bias
- Analysis unit          : Linear Mixed Model (LMM) per ROI (7 ROIs total)
- Random effect          : Subject — corrects for within-subject repeated-measures design
- Multiple comparisons   : FDR correction (Benjamini–Hochberg) across 7 ROIs

[Why Voxel_cnt was removed]
  Voxel_cnt varies by Condition (activation size is a direct output of the experimental
  manipulation), so including it as a covariate regresses out part of the Condition effect
  itself — a textbook Over-correction / Post-treatment bias.

[Improvements over plain OLS ANCOVA]
  ① mixedlm        : Subject random intercept removes between-subject variance
  ② Windowed peak (t=3~6) + ±1 tp average : robust against transient noise spikes
  ③ Baseline correction : removes per-session offset at t=0
  ④ FDR (BH)       : controls false-discovery rate across 7 simultaneous tests
  ⑤ Signal-derived covariates (TTP_grand, PeakAmp_grand) : condition-independent HRF
                     shape features that capture individual vascular reactivity differences
                     without introducing post-treatment bias
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------ #
# 1. Load data and build DataFrame
# ------------------------------------------------------------------ #
DATASET_DIR    = 'dataset'
PEAK_WIN_START = 3    # expected HRF peak window start index (≈3 s post-stimulus)
PEAK_WIN_END   = 7    # expected HRF peak window end index  (exclusive; covers t=3~6)

files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('_hrf.mat')])

# ⑤ Per-subject HRF shape covariates — derived from the 'all' field (grand-average
#    HRF across all ROIs and conditions).  Because 'all' is condition-agnostic, these
#    covariates represent individual vascular reactivity and carry no post-treatment bias.
def extract_grand_hrf_features(hrf_all, peak_win_start, peak_win_end):
    """Return (TTP_grand, PeakAmp_grand) from the grand-average HRF vector."""
    n_tp      = len(hrf_all)
    baseline  = float(hrf_all[0])
    win_end   = min(peak_win_end, n_tp)
    local_idx = int(np.argmax(hrf_all[peak_win_start:win_end]))
    ttp       = local_idx + peak_win_start              # time-to-peak (global index)
    t0 = max(0,      ttp - 1)
    t1 = min(n_tp-1, ttp + 1)
    peak_smooth = float(np.mean(hrf_all[t0:t1+1]))     # ±1 tp average
    peak_amp    = peak_smooth - baseline                 # baseline-corrected
    return ttp, peak_amp

records = []
for fname in files:
    subject_id = fname.replace('_hrf.mat', '')           # e.g. SUB701
    mat = sio.loadmat(os.path.join(DATASET_DIR, fname))
    inner = mat['hrf_ROI'][0, 0]

    data    = inner['data']          # shape (7, 2, 13) — HRF time series
    hrf_all = inner['all'].flatten() # shape (13,)  — grand-average HRF

    # ⑤ Extract per-subject HRF shape covariates (condition-independent)
    ttp_grand, peakamp_grand = extract_grand_hrf_features(
        hrf_all, PEAK_WIN_START, PEAK_WIN_END
    )

    n_roi, n_cond, n_tp = data.shape

    for roi in range(n_roi):
        for cond in range(n_cond):
            hrf_curve = data[roi, cond, :]          # HRF time series (length 13)
            baseline  = float(hrf_curve[0])         # pre-stimulus baseline (t=0)

            # ② Robust peak: argmax within expected window (t=3~6),
            #    then average ±1 time point to suppress transient noise spikes
            win_end   = min(PEAK_WIN_END, n_tp)
            local_idx = int(np.argmax(hrf_curve[PEAK_WIN_START:win_end]))
            peak_time = local_idx + PEAK_WIN_START   # global index
            t0 = max(0,        peak_time - 1)
            t1 = min(n_tp - 1, peak_time + 1)
            peak_amp_smooth = float(np.mean(hrf_curve[t0:t1 + 1]))  # ±1 tp average

            # ③ Baseline correction: removes per-session offset at t=0
            peak_amp = peak_amp_smooth - baseline

            records.append({
                'Subject'        : subject_id,
                'ROI'            : roi + 1,               # 1~7
                'Condition'      : f'Cond{cond + 1}',     # 'Cond1' or 'Cond2'
                'TTP_grand'      : ttp_grand,             # covariate ⑤: HRF latency
                'PeakAmp_grand'  : peakamp_grand,         # covariate ⑤: overall HRF strength
                'Baseline'       : baseline,              # stored for reference
                'Peak_amp'       : peak_amp,              # DV: baseline-corrected
                'Peak_time'      : peak_time,             # reference only
            })

df = pd.DataFrame(records)

print("=" * 65)
print("■ Input DataFrame (top 10 rows)")
print("=" * 65)
print(df.head(10).to_string(index=False))
print(f"\nTotal rows: {len(df)}  (26 subjects x 7 ROIs x 2 conditions)")
print(f"Columns: {list(df.columns)}")
print("Note: Peak_amp = baseline-corrected  (windowed peak +/-1 tp) - baseline(t=0)")
print("      TTP_grand / PeakAmp_grand = per-subject grand-average HRF shape features")
print("      (condition-independent → no post-treatment bias)")

# ------------------------------------------------------------------ #
# 2. Run Linear Mixed Model (LMM) per ROI
#    ① Subject as random intercept — corrects within-subject design
#    ⑤ Covariates: TTP_grand, PeakAmp_grand (condition-independent HRF shape)
# ------------------------------------------------------------------ #
print("\n" + "=" * 65)
print("■ LMM Results per ROI")
print("   [Model: Peak_amp ~ C(Condition) + PeakAmp_grand + TTP_grand]")
print("   [Random intercept per Subject  |  REML estimation]")
print("=" * 65)

summary_rows  = []
p_vals_group  = []   # collect for FDR correction

for roi_idx in range(1, 8):
    roi_df = df[df['ROI'] == roi_idx].copy()

    # ① Linear Mixed Model: fixed effects = PeakAmp_grand + TTP_grand + Condition
    #    random effect = random intercept per Subject
    model = smf.mixedlm(
        'Peak_amp ~ PeakAmp_grand + TTP_grand + C(Condition)',
        data=roi_df,
        groups=roi_df['Subject']
    ).fit(reml=True)

    cov_peak_z = model.tvalues.get('PeakAmp_grand',               np.nan)
    cov_ttp_z  = model.tvalues.get('TTP_grand',                   np.nan)
    group_z    = model.tvalues.get('C(Condition)[T.Cond2]',       np.nan)
    cov_peak_p = model.pvalues.get('PeakAmp_grand',               np.nan)
    cov_ttp_p  = model.pvalues.get('TTP_grand',                   np.nan)
    group_p    = model.pvalues.get('C(Condition)[T.Cond2]',       np.nan)

    p_vals_group.append(group_p)

    print(f"\n--- ROI {roi_idx} ---")
    print(model.summary().tables[1])
    sig = "★ Significant" if group_p < 0.05 else "  Not significant"
    print(f"-> [Group effect] z={group_z:.3f}, p={group_p:.4f}  {sig} (uncorrected)")

    means = roi_df.groupby('Condition')['Peak_amp'].mean()

    summary_rows.append({
        'ROI'            : f'ROI {roi_idx}',
        'z(PeakAmp_grd)' : round(cov_peak_z, 3),
        'p(PeakAmp_grd)' : round(cov_peak_p, 4),
        'z(TTP_grd)'     : round(cov_ttp_z,  3),
        'p(TTP_grd)'     : round(cov_ttp_p,  4),
        'z(Group)'       : round(group_z,     3),
        'p(Group)'       : round(group_p,     4),
        'Mean_Cond1'     : round(means.get('Cond1', np.nan), 4),
        'Mean_Cond2'     : round(means.get('Cond2', np.nan), 4),
    })

# ------------------------------------------------------------------ #
# 3. FDR correction across 7 ROIs (④ Benjamini–Hochberg)
# ------------------------------------------------------------------ #
reject_fdr, pvals_fdr, _, _ = multipletests(p_vals_group, alpha=0.05, method='fdr_bh')

for i, row in enumerate(summary_rows):
    row['p_FDR']   = round(float(pvals_fdr[i]), 4)
    row['Sig_FDR'] = 'Yes' if reject_fdr[i] else 'No'

# ------------------------------------------------------------------ #
# 4. Print summary table
# ------------------------------------------------------------------ #
summary_df = pd.DataFrame(summary_rows)

print("\n" + "=" * 65)
print("■ LMM Summary Table -- all ROIs  (with FDR correction, BH method)")
print("=" * 65)
print(summary_df.to_string(index=False))

sig_raw = summary_df[summary_df['p(Group)'] < 0.05]['ROI'].tolist()
sig_fdr = summary_df[summary_df['Sig_FDR'] == 'Yes']['ROI'].tolist()
print(f"\nSignificant ROIs (uncorrected p < 0.05) : {sig_raw if sig_raw else 'None'}")
print(f"Significant ROIs (FDR-corrected p < 0.05): {sig_fdr if sig_fdr else 'None'}")

# ------------------------------------------------------------------ #
# 5. Visualization
# ------------------------------------------------------------------ #
fig = plt.figure(figsize=(18, 14))
fig.suptitle('LMM-based ANCOVA: Baseline-Corrected HRF Peak Amplitude\n'
             '(Covariates: Grand-avg HRF Shape [PeakAmp_grand, TTP_grand]  |  '
             'Random effect: Subject  |  FDR corrected)',
             fontsize=12, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)

colors = {'Cond1': '#2196F3', 'Cond2': '#F44336'}

for idx, roi_idx in enumerate(range(1, 8)):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])
    roi_df = df[df['ROI'] == roi_idx]

    for cond, color in colors.items():
        subset = roi_df[roi_df['Condition'] == cond]
        ax.scatter(subset['PeakAmp_grand'], subset['Peak_amp'],
                   alpha=0.6, color=color, label=cond, s=35, zorder=3)
        # Regression line
        x_sorted = np.sort(subset['PeakAmp_grand'].values)
        coefs = np.polyfit(subset['PeakAmp_grand'], subset['Peak_amp'], 1)
        ax.plot(x_sorted, np.polyval(coefs, x_sorted),
                color=color, linewidth=1.5, alpha=0.8)

    row = summary_df[summary_df['ROI'] == f'ROI {roi_idx}'].iloc[0]
    # Show both uncorrected and FDR-corrected p-value; star only if FDR significant
    star   = ' ★' if row['Sig_FDR'] == 'Yes' else ''
    bold   = 'bold' if row['Sig_FDR'] == 'Yes' else 'normal'
    title  = (f'ROI {roi_idx}{star}  '
              f'p={row["p(Group)"]:.3f} / p_FDR={row["p_FDR"]:.3f}')
    ax.set_title(title, fontsize=9, fontweight=bold)
    ax.set_xlabel('Grand-avg Peak Amp (PeakAmp_grand)', fontsize=8)
    ax.set_ylabel('Peak Amp (baseline-corrected)', fontsize=8)
    ax.legend(fontsize=7, loc='upper left')
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

# Summary: dual p-value bar chart (raw vs FDR)
ax_bar = fig.add_subplot(gs[2, 2])
x      = np.arange(1, 8)
width  = 0.35
raw_p  = summary_df['p(Group)'].values
fdr_p  = summary_df['p_FDR'].values

bars1 = ax_bar.bar(x - width / 2, raw_p, width,
                   color='#90CAF9', edgecolor='black', linewidth=0.6, label='p (uncorrected)')
bars2 = ax_bar.bar(x + width / 2, fdr_p, width,
                   color=['#4CAF50' if p < 0.05 else '#9E9E9E' for p in fdr_p],
                   edgecolor='black', linewidth=0.6, label='p_FDR (BH)')

ax_bar.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, label='α=0.05')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels([f'R{i}' for i in range(1, 8)], fontsize=8)
ax_bar.set_xlabel('ROI', fontsize=9)
ax_bar.set_ylabel('p-value', fontsize=9)
ax_bar.set_title('Group Effect p-values\n(Blue=raw, Green=FDR sig.)', fontsize=9, fontweight='bold')
ax_bar.legend(fontsize=7)
ax_bar.tick_params(labelsize=8)
ax_bar.set_ylim(0, max(max(raw_p), max(fdr_p)) * 1.3 + 0.01)

plt.savefig('ancova_hrf_results.png', dpi=150, bbox_inches='tight')
print("\nPlot saved: ancova_hrf_results.png")
plt.show()

# ------------------------------------------------------------------ #
# 6. Save results to CSV
# ------------------------------------------------------------------ #
df.to_csv('ancova_input_data.csv', index=False, encoding='utf-8-sig')
summary_df.to_csv('ancova_result_summary.csv', index=False, encoding='utf-8-sig')
print("CSV saved: ancova_input_data.csv, ancova_result_summary.csv")
print("\n[NOTE] Covariates TTP_grand and PeakAmp_grand are extracted from the 'all'")
print("       (grand-average HRF) field in each subject's .mat file.")
print("       They are constant within subject across all ROI/Condition rows,")
print("       so they carry NO post-treatment / over-correction bias.")
