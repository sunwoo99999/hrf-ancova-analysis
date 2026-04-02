"""
=============================================================
HRF Group Analysis: Young vs T2DM
Based on: McDonough et al. (2025), Behavioral Sciences, 15, 1457

Implements:
  - Section 3.3: Age/Risk Effects on HRF Metrics (max value, TTP, FWHM)
  - Figure 1 equivalent: sHRF variability + canonical HRF overlay
  - Figure 2 equivalent: HRF curves by group
  - Group-level t-tests on HRF metrics (Young vs T2DM)
  - Updated LMM with AgeGroup as between-subject factor

Available groups from dataset (33 subjects):
  Young (<=35): 14 subjects ??701~725
  T2DM        :  5 subjects ??717, 808, 809, 811, 812
  Mid (36~54) :  1 subject  ??801 (excluded from group stats, N too small)
  Unknown     : 13 subjects ??group not confirmed
=============================================================
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats
from scipy.stats import gamma as scipy_gamma
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------ #
# 0. Configuration
# ------------------------------------------------------------------ #
DATASET_DIR    = 'dataset'
PEAK_WIN_START = 3
PEAK_WIN_END   = 7

# Group assignments from subject classification image
# Young (<=35) and T2DM confirmed; Mid only 1 subject (excluded from stats)
GROUP_MAP = {
    '701': 'Young', '702': 'Young', '703': 'Young', '705': 'Young',
    '709': 'Young', '712': 'Young', '714': 'Young', '715': 'Young',
    '716': 'Young', '718': 'Young', '720': 'Young', '721': 'Young',
    '722': 'Young', '725': 'Young',
    '717': 'T2DM',  '808': 'T2DM',  '809': 'T2DM',
    '811': 'T2DM',  '812': 'T2DM',
    '801': 'Mid',
}
GROUP_COLORS = {'Young': '#2196F3', 'T2DM': '#F44336', 'Mid': '#FF9800', 'Unknown': '#9E9E9E'}

# ------------------------------------------------------------------ #
# 1. Canonical HRF (SPM double-gamma, normalized to peak = 1)
#    Matches Figure 1 black line in the paper
# ------------------------------------------------------------------ #
def canonical_hrf(n_tp, tr=1.72):
    """SPM canonical HRF: double-gamma difference, normalized.
    Default TR=1.72 s (same as paper's EPI acquisition)."""
    t = np.arange(n_tp) * tr
    h = (scipy_gamma.pdf(t, 6, scale=1) -
         0.167 * scipy_gamma.pdf(t, 16, scale=1))
    peak = h.max()
    if peak > 0:
        h = h / peak
    return h

# ------------------------------------------------------------------ #
# 2. HRF metrics extraction (max value, TTP, FWHM)
#    Mirrors paper Section 2.3.3
# ------------------------------------------------------------------ #
def extract_hrf_metrics(hrf, peak_win_start, peak_win_end):
    """
    Returns (peak_amp, ttp, fwhm) from a baseline-corrected HRF.
    - peak_amp : windowed peak (짹1 tp average) minus baseline(t=0)
    - ttp      : time-to-peak (global index)
    - fwhm     : full-width at half-maximum (in timepoints)
    """
    n_tp     = len(hrf)
    baseline = float(hrf[0])

    # Peak amplitude (windowed + smoothed, baseline-corrected)
    win_end   = min(peak_win_end, n_tp)
    local_idx = int(np.argmax(hrf[peak_win_start:win_end]))
    ttp       = local_idx + peak_win_start
    t0  = max(0,      ttp - 1)
    t1  = min(n_tp-1, ttp + 1)
    peak_amp = float(np.mean(hrf[t0:t1+1])) - baseline

    # FWHM: search around peak for half-maximum crossings
    hrf_bc   = np.array(hrf) - baseline
    half_max = peak_amp / 2.0
    rise_idx, fall_idx = 0, n_tp - 1
    for i in range(ttp):
        if hrf_bc[i] >= half_max:
            rise_idx = i
            break
    for i in range(ttp, n_tp):
        if hrf_bc[i] <= half_max:
            fall_idx = i
            break
    fwhm = fall_idx - rise_idx

    return peak_amp, ttp, fwhm

# ------------------------------------------------------------------ #
# 3. Load data and compute metrics
# ------------------------------------------------------------------ #
files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('_hrf.mat')])

metrics_records = []   # per-subject grand-average HRF metrics
roi_records     = []   # per-subject 횞 ROI 횞 Condition (for LMM)

for fname in files:
    subject_id  = fname.replace('_hrf.mat', '')
    sub_num_str = subject_id.replace('SUB', '')
    group       = GROUP_MAP.get(sub_num_str, 'Unknown')

    mat   = sio.loadmat(os.path.join(DATASET_DIR, fname))
    inner = mat['hrf_ROI'][0, 0]
    data    = inner['data'].squeeze()    # (7, 2, 13)
    hrf_all = inner['all'].flatten()     # (13,) grand-average HRF

    peak_amp, ttp, fwhm = extract_hrf_metrics(hrf_all, PEAK_WIN_START, PEAK_WIN_END)

    metrics_records.append({
        'Subject'       : subject_id,
        'Group'         : group,
        'PeakAmp_grand' : peak_amp,
        'TTP_grand'     : ttp,
        'FWHM_grand'    : fwhm,
        'hrf_all'       : hrf_all.tolist(),   # keep raw curve for plotting
    })

    n_roi, n_cond, n_tp = data.shape
    for roi in range(n_roi):
        for cond in range(n_cond):
            hrf_curve = data[roi, cond, :]
            baseline  = float(hrf_curve[0])
            win_end   = min(PEAK_WIN_END, n_tp)
            local_idx = int(np.argmax(hrf_curve[PEAK_WIN_START:win_end]))
            peak_time = local_idx + PEAK_WIN_START
            t0 = max(0,      peak_time - 1)
            t1 = min(n_tp-1, peak_time + 1)
            roi_peak_amp = float(np.mean(hrf_curve[t0:t1+1])) - baseline

            roi_records.append({
                'Subject'       : subject_id,
                'Group'         : group,
                'ROI'           : roi + 1,
                'Condition'     : f'Cond{cond + 1}',
                'PeakAmp_grand' : peak_amp,
                'TTP_grand'     : ttp,
                'FWHM_grand'    : fwhm,
                'Baseline'      : baseline,
                'Peak_amp'      : roi_peak_amp,
                'Peak_time'     : peak_time,
            })

metrics_df = pd.DataFrame(metrics_records)
roi_df     = pd.DataFrame(roi_records)

n_total = len(metrics_df)
n_young = (metrics_df['Group'] == 'Young').sum()
n_t2dm  = (metrics_df['Group'] == 'T2DM').sum()
n_mid   = (metrics_df['Group'] == 'Mid').sum()
n_unk   = (metrics_df['Group'] == 'Unknown').sum()

print("=" * 65)
print("??HRF Group Analysis ??Dataset Overview")
print("=" * 65)
print(f"Total subjects : {n_total}")
print(f"  Young (<=35) : {n_young}")
print(f"  T2DM         : {n_t2dm}")
print(f"  Mid (36~54)  : {n_mid}  (excluded from group stats, N=1)")
print(f"  Unknown      : {n_unk}  (group not confirmed)")
print(f"\nHRF metrics extracted: PeakAmp_grand, TTP_grand, FWHM_grand")
print(metrics_df[['Subject','Group','PeakAmp_grand','TTP_grand','FWHM_grand']].to_string(index=False))

# ------------------------------------------------------------------ #
# 4. Group statistics: Young vs T2DM on HRF metrics
#    Analogous to paper Section 3.3 regression analyses
# ------------------------------------------------------------------ #
print("\n" + "=" * 65)
print("??Young vs T2DM: HRF Metric Comparison (Independent t-test)")
print("  (Paper Section 3.3 ??Age/Risk effects on max value, TTP, FWHM)")
print("=" * 65)

group_df = metrics_df[metrics_df['Group'].isin(['Young', 'T2DM'])].copy()
young_df = group_df[group_df['Group'] == 'Young']
t2dm_df  = group_df[group_df['Group'] == 'T2DM']

stat_rows = []
for metric in ['PeakAmp_grand', 'TTP_grand', 'FWHM_grand']:
    y_vals = young_df[metric].values
    d_vals = t2dm_df[metric].values
    t_stat, p_val = stats.ttest_ind(y_vals, d_vals)
    cohens_d = (y_vals.mean() - d_vals.mean()) / np.sqrt(
        ((len(y_vals)-1)*y_vals.std()**2 + (len(d_vals)-1)*d_vals.std()**2) /
        (len(y_vals)+len(d_vals)-2)
    )
    stat_rows.append({
        'Metric'    : metric,
        'Young_M'   : round(y_vals.mean(), 3),
        'Young_SD'  : round(y_vals.std(),  3),
        'T2DM_M'    : round(d_vals.mean(), 3),
        'T2DM_SD'   : round(d_vals.std(),  3),
        't'         : round(t_stat, 3),
        'p'         : round(p_val,  4),
        'd'         : round(cohens_d, 3),
        'Sig'       : '*' if p_val < 0.05 else '',
    })
    print(f"\n  {metric}:")
    print(f"    Young (n={len(y_vals)}): M={y_vals.mean():.3f}, SD={y_vals.std():.3f}")
    print(f"    T2DM  (n={len(d_vals)}): M={d_vals.mean():.3f}, SD={d_vals.std():.3f}")
    print(f"    t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={cohens_d:.3f}  {'*' if p_val<0.05 else ''}")

# ------------------------------------------------------------------ #
# 5. Updated LMM (known groups only):
#    Peak_amp ~ PeakAmp_grand + TTP_grand + C(Condition) + C(Group)
#    groups = Subject (random intercept)
# ------------------------------------------------------------------ #
print("\n" + "=" * 65)
print("??LMM with AgeGroup Effect (Young + T2DM only)")
print("   [Model: Peak_amp ~ PeakAmp_grand + TTP_grand + C(Condition) + C(Group)]")
print("   [Random intercept per Subject | REML]")
print("=" * 65)

lmm_df = roi_df[roi_df['Group'].isin(['Young', 'T2DM'])].copy()
p_group_vals = []
lmm_rows     = []

for roi_idx in range(1, 8):
    sub = lmm_df[lmm_df['ROI'] == roi_idx].copy()
    model = smf.mixedlm(
        'Peak_amp ~ PeakAmp_grand + TTP_grand + C(Condition) + C(Group)',
        data=sub,
        groups=sub['Subject']
    ).fit(reml=True)

    grp_z = model.tvalues.get('C(Group)[T.Young]', np.nan)
    grp_p = model.pvalues.get('C(Group)[T.Young]', np.nan)
    cnd_z = model.tvalues.get('C(Condition)[T.Cond2]', np.nan)
    cnd_p = model.pvalues.get('C(Condition)[T.Cond2]', np.nan)
    p_group_vals.append(grp_p)

    means = sub.groupby('Group')['Peak_amp'].mean()
    lmm_rows.append({
        'ROI'        : f'ROI {roi_idx}',
        'z(Group)'   : round(grp_z, 3),
        'p(Group)'   : round(grp_p, 4),
        'z(Cond)'    : round(cnd_z, 3),
        'p(Cond)'    : round(cnd_p, 4),
        'M_Young'    : round(means.get('Young', np.nan), 4),
        'M_T2DM'     : round(means.get('T2DM',  np.nan), 4),
    })
    sig = '*' if grp_p < 0.05 else ' '
    print(f"  ROI {roi_idx}  Group: z={grp_z:.3f}, p={grp_p:.4f} {sig}  "
          f"| Cond: z={cnd_z:.3f}, p={cnd_p:.4f}")

reject_fdr, pvals_fdr, _, _ = multipletests(p_group_vals, alpha=0.05, method='fdr_bh')
for i, row in enumerate(lmm_rows):
    row['p_FDR']   = round(float(pvals_fdr[i]), 4)
    row['Sig_FDR'] = 'Yes' if reject_fdr[i] else 'No'

lmm_summary = pd.DataFrame(lmm_rows)
print("\n  LMM Summary (AgeGroup effect, FDR corrected):")
print(lmm_summary.to_string(index=False))

# ------------------------------------------------------------------ #
# 6. Visualization
# ------------------------------------------------------------------ #
fig = plt.figure(figsize=(20, 16))
fig.suptitle('HRF Group Analysis: Young vs T2DM\n'
             '(Based on McDonough et al., 2025  |  sHRF = all-field grand-average)',
             fontsize=13, fontweight='bold', y=0.99)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

# ------ Panel 1: Figure 1 equivalent ??all sHRF curves + canonical ------ #
ax1 = fig.add_subplot(gs[0, :2])
n_tp_ex = len(metrics_df.iloc[0]['hrf_all'])
can_hrf = canonical_hrf(n_tp_ex, tr=1.72)

for _, row in metrics_df.iterrows():
    hrf = np.array(row['hrf_all'])
    # normalize to peak=1 for shape comparison (mirrors paper Figure 1)
    pk = hrf.max()
    if pk > 0:
        hrf = hrf / pk
    color = GROUP_COLORS.get(row['Group'], '#BDBDBD')
    alpha = 0.6 if row['Group'] in ('Young', 'T2DM') else 0.2
    lw    = 1.2 if row['Group'] in ('Young', 'T2DM') else 0.7
    ax1.plot(hrf, color=color, alpha=alpha, linewidth=lw)

ax1.plot(can_hrf, color='black', linewidth=2.5, label='Canonical HRF (SPM)', zorder=5)
from matplotlib.lines import Line2D
legend_els = [
    Line2D([0],[0], color=GROUP_COLORS['Young'], lw=2, label=f'Young (n={n_young})'),
    Line2D([0],[0], color=GROUP_COLORS['T2DM'],  lw=2, label=f'T2DM  (n={n_t2dm})'),
    Line2D([0],[0], color='black',               lw=2.5, label='Canonical HRF'),
]
ax1.legend(handles=legend_els, fontsize=8, loc='upper right')
ax1.set_xlabel('Time (timepoints)', fontsize=9)
ax1.set_ylabel('Normalized HRF', fontsize=9)
ax1.set_title('Figure 1 Equivalent: sHRF Variability Across Subjects\n'
              '(Normalized to peak=1 per subject)', fontsize=9, fontweight='bold')
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax1.grid(True, alpha=0.3)

# ------ Panel 2: Figure 2 equivalent ??mean HRF by group ------ #
ax2 = fig.add_subplot(gs[0, 2:])
for grp, color in [('Young', GROUP_COLORS['Young']), ('T2DM', GROUP_COLORS['T2DM'])]:
    grp_hrfs = np.array([
        np.array(r['hrf_all']) / max(np.array(r['hrf_all']).max(), 1e-8)
        for _, r in metrics_df[metrics_df['Group'] == grp].iterrows()
    ])
    mean_hrf = grp_hrfs.mean(axis=0)
    se_hrf   = grp_hrfs.std(axis=0) / np.sqrt(len(grp_hrfs))
    x = np.arange(n_tp_ex)
    ax2.plot(x, mean_hrf, color=color, linewidth=2, label=grp)
    ax2.fill_between(x, mean_hrf - se_hrf, mean_hrf + se_hrf,
                     color=color, alpha=0.2)
ax2.plot(can_hrf, color='black', linewidth=2, linestyle='--', label='Canonical')
ax2.legend(fontsize=8)
ax2.set_xlabel('Time (timepoints)', fontsize=9)
ax2.set_ylabel('Normalized HRF (Mean 짹 SE)', fontsize=9)
ax2.set_title('Figure 2 Equivalent: Mean sHRF by Group\n'
              '(Shaded = 짹1 SE)', fontsize=9, fontweight='bold')
ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax2.grid(True, alpha=0.3)

# ------ Panels 3-5: Metric comparison bar charts ------ #
metrics_info = [
    ('PeakAmp_grand', 'Max HRF Value\n(baseline-corrected)', gs[1, 0]),
    ('TTP_grand',     'Latency (TTP)\n(timepoints)',          gs[1, 1]),
    ('FWHM_grand',    'FWHM\n(timepoints)',                   gs[1, 2]),
]
for metric, ylabel, gspec in metrics_info:
    ax = fig.add_subplot(gspec)
    for gi, (grp, color) in enumerate([('Young', GROUP_COLORS['Young']),
                                        ('T2DM',  GROUP_COLORS['T2DM'])]):
        vals = group_df[group_df['Group'] == grp][metric].values
        ax.bar(gi, vals.mean(), yerr=vals.std()/np.sqrt(len(vals)),
               color=color, alpha=0.7, capsize=5, label=grp)
        ax.scatter([gi]*len(vals), vals, color=color, s=20, zorder=5, alpha=0.8)
    # p-value annotation
    row_stat = next(r for r in stat_rows if r['Metric'] == metric)
    p_ann = row_stat['p']
    star  = '***' if p_ann<0.001 else ('**' if p_ann<0.01 else ('*' if p_ann<0.05 else 'n.s.'))
    y_max = group_df[metric].max()
    ax.annotate(f'p={p_ann:.4f} {star}', xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', fontsize=8, color='black')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Young', 'T2DM'], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(metric, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

# ------ Panel 6: Group effect in LMM (p-value per ROI) ------ #
ax_lmm = fig.add_subplot(gs[1, 3])
rois  = np.arange(1, 8)
col   = ['#4CAF50' if s == 'Yes' else '#9E9E9E' for s in lmm_summary['Sig_FDR']]
ax_lmm.bar(rois, lmm_summary['p(Group)'].values, color='#90CAF9',
           edgecolor='black', linewidth=0.6, label='p (raw)')
ax_lmm.bar(rois, lmm_summary['p_FDR'].values, color=col,
           edgecolor='black', linewidth=0.6, alpha=0.7, label='p_FDR (BH)')
ax_lmm.axhline(0.05, color='red', linestyle='--', linewidth=1.5, label='alpha=0.05')
ax_lmm.set_xticks(rois)
ax_lmm.set_xticklabels([f'R{i}' for i in rois], fontsize=8)
ax_lmm.set_ylabel('p-value', fontsize=8)
ax_lmm.set_title('LMM: Young vs T2DM\nGroup Effect per ROI', fontsize=9, fontweight='bold')
ax_lmm.legend(fontsize=7)
ax_lmm.grid(True, alpha=0.3, axis='y')

# ------ Panel 7: Scatter PeakAmp_grand by group (paper-style) ------ #
ax_sc = fig.add_subplot(gs[2, :2])
for grp, color in [('Young', GROUP_COLORS['Young']), ('T2DM', GROUP_COLORS['T2DM']),
                   ('Unknown', GROUP_COLORS['Unknown'])]:
    sub = metrics_df[metrics_df['Group'] == grp]
    ax_sc.scatter(sub.index, sub['PeakAmp_grand'], color=color, label=grp,
                  s=50, alpha=0.8, zorder=3)
ax_sc.axhline(metrics_df[metrics_df['Group']=='Young']['PeakAmp_grand'].mean(),
              color=GROUP_COLORS['Young'], linestyle='--', linewidth=1.5, label='Young mean')
ax_sc.axhline(metrics_df[metrics_df['Group']=='T2DM']['PeakAmp_grand'].mean(),
              color=GROUP_COLORS['T2DM'],  linestyle='--', linewidth=1.5, label='T2DM mean')
ax_sc.set_ylabel('PeakAmp_grand', fontsize=9)
ax_sc.set_xlabel('Subject index', fontsize=9)
ax_sc.set_title('Grand-average HRF Peak Amplitude per Subject\n(Group means shown as dashed lines)',
                fontsize=9, fontweight='bold')
ax_sc.legend(fontsize=7, ncol=2)
ax_sc.grid(True, alpha=0.3)

# ------ Panel 8: Correlation PeakAmp vs FWHM ------ #
ax_corr = fig.add_subplot(gs[2, 2:])
for grp, color in [('Young', GROUP_COLORS['Young']), ('T2DM', GROUP_COLORS['T2DM']),
                   ('Unknown', GROUP_COLORS['Unknown'])]:
    sub = metrics_df[metrics_df['Group'] == grp]
    ax_corr.scatter(sub['PeakAmp_grand'], sub['FWHM_grand'],
                    color=color, label=grp, s=50, alpha=0.8)
# Overall regression line
x_all = metrics_df['PeakAmp_grand'].values
y_all = metrics_df['FWHM_grand'].values
r_val, p_val = stats.pearsonr(x_all, y_all)
coefs = np.polyfit(x_all, y_all, 1)
x_line = np.linspace(x_all.min(), x_all.max(), 100)
ax_corr.plot(x_line, np.polyval(coefs, x_line), 'k--', linewidth=1.5,
             label=f'All: r={r_val:.3f}, p={p_val:.4f}')
ax_corr.set_xlabel('PeakAmp_grand', fontsize=9)
ax_corr.set_ylabel('FWHM_grand (timepoints)', fontsize=9)
ax_corr.set_title('HRF Peak Amplitude vs FWHM\n(Paper: AUC highly correlated with max value)',
                  fontsize=9, fontweight='bold')
ax_corr.legend(fontsize=7)
ax_corr.grid(True, alpha=0.3)

plt.savefig('hrf_group_analysis.png', dpi=150, bbox_inches='tight')
print("\nPlot saved: hrf_group_analysis.png")
plt.show()

# ------------------------------------------------------------------ #
# 7. Save outputs
# ------------------------------------------------------------------ #
metrics_out = metrics_df.drop(columns=['hrf_all'])
metrics_out.to_csv('hrf_group_metrics.csv', index=False, encoding='utf-8-sig')
lmm_summary.to_csv('hrf_group_lmm_summary.csv', index=False, encoding='utf-8-sig')
print("CSV saved: hrf_group_metrics.csv, hrf_group_lmm_summary.csv")

print("\n" + "=" * 65)
print("??Summary vs Paper (McDonough et al., 2025)")
print("=" * 65)
print(f"  Paper finding: 'Older age associated with smaller max HRF peak'")
print(f"  Our result   : Young M={young_df['PeakAmp_grand'].mean():.3f}, "
      f"T2DM M={t2dm_df['PeakAmp_grand'].mean():.3f}")
r = next(r for r in stat_rows if r['Metric']=='PeakAmp_grand')
print(f"                 t={r['t']}, p={r['p']}, d={r['d']} {r['Sig']}")
print(f"\n  Paper finding: 'No age effect on TTP or FWHM' (all p>0.05)")
r_ttp  = next(r for r in stat_rows if r['Metric']=='TTP_grand')
r_fwhm = next(r for r in stat_rows if r['Metric']=='FWHM_grand')
print(f"  Our TTP      : t={r_ttp['t']}, p={r_ttp['p']} {r_ttp['Sig']}")
print(f"  Our FWHM     : t={r_fwhm['t']}, p={r_fwhm['p']} {r_fwhm['Sig']}")
print(f"\n  Note: Paper compared Young(20-30) vs Middle+Old(50-74).")
print(f"        We compare Young(<=35) vs T2DM(n=5) ??limited power.")
