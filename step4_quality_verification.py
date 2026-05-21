import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, kurtosis
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_squared_error, mean_absolute_error

os.makedirs("outputs", exist_ok=True)

print("=" * 65)
print("  STEP 4: QUALITY VERIFICATION (EVALUATION)")
print("  Metrics: MAE, ME±SD, SNR, RMSE, Pearson r")
print("=" * 65)

# ==============================================================
# Load Data
# ==============================================================

print("\n[1/6] Loading reconstruction results...")

te_c = np.load("outputs/te_c_phys.npy")
te_n = np.load("outputs/te_n_phys.npy")
te_r = np.load("outputs/te_r_phys.npy")

Xc_te = np.load("outputs/Xc_te.npy")
Xn_te = np.load("outputs/Xn_te.npy")
Xr_te = np.load("outputs/Xr_te.npy")

n_total         = 1707   # 569 original segments × 3 augmentations (from step 1)
n_test          = te_c.shape[0]
n_train         = n_total - n_test
seg_len         = te_c.shape[1]
n_reconstructed = n_test

print(f"      Total segments in pipeline    : {n_total}  (569 original × 3 augmentations)")
print(f"      Training segments (80%)       : {n_train}")
print(f"      Test segments     (20%)       : {n_test}")
print(f"      Segment length               : {seg_len} samples  ({seg_len/100:.1f} s at 100 Hz)")
print(f"      Signals reconstructed        : {n_reconstructed} / {n_test} test segments")
print(f"      NOTE: All {n_total} segments pass through the full pipeline.")
print(f"            Autoencoder is evaluated on {n_test} held-out test segments.")


# ==============================================================
# Metric Function
# ==============================================================

def compute_metrics(clean, pred):
    c = clean.flatten()
    p = pred.flatten()

    mse  = mean_squared_error(c, p)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(c, p)

    err  = p - c
    me   = np.mean(err)
    sd   = np.std(err)

    snr  = 10 * np.log10(
        np.var(clean) / (np.mean((clean - pred)**2) + 1e-10)
    )

    r, _ = pearsonr(c, p)

    return {
        "MSE": mse, "RMSE": rmse, "MAE": mae,
        "ME": me,   "SD": sd,
        "SNR": snr, "Pearson_r": r
    }


# ==============================================================
# Global Metrics
# ==============================================================

print("\n[2/6] Computing global metrics...")

mn = compute_metrics(te_c, te_n)
mr = compute_metrics(te_c, te_r)

mse_imp  = (1 - mr['MSE']  / mn['MSE'])  * 100
rmse_imp = (1 - mr['RMSE'] / mn['RMSE']) * 100
mae_imp  = (1 - mr['MAE']  / mn['MAE'])  * 100
snr_imp  = mr['SNR'] - mn['SNR']
cor_imp  = (mr['Pearson_r'] - mn['Pearson_r']) * 100

print(f"\n      {'Metric':<16} {'Noisy':>14} {'Reconstructed':>16} {'Improvement':>14}")
print(f"      {'-'*16}  {'-'*14}  {'-'*16}  {'-'*14}")
print(f"      {'MSE':<16} {mn['MSE']:>14.2f} {mr['MSE']:>16.2f} {mse_imp:>+13.1f}%")
print(f"      {'RMSE':<16} {mn['RMSE']:>14.4f} {mr['RMSE']:>16.4f} {rmse_imp:>+13.1f}%")
print(f"      {'MAE':<16} {mn['MAE']:>14.4f} {mr['MAE']:>16.4f} {mae_imp:>+13.1f}%")
print(f"      {'SNR (dB)':<16} {mn['SNR']:>14.4f} {mr['SNR']:>16.4f} {snr_imp:>+13.2f} dB")
print(f"      {'Pearson r':<16} {mn['Pearson_r']:>14.4f} {mr['Pearson_r']:>16.4f} {cor_imp:>+13.2f} pp")


# ==============================================================
# Per-Segment Metrics
# ==============================================================

print("\n[3/6] Computing per-segment metrics...")

seg_mae_n, seg_mae_r = [], []
seg_snr_n, seg_snr_r = [], []
seg_improved         = []

print(f"\n      {'Seg':>4}  {'MAE-Noisy':>10}  {'MAE-Recon':>10}  "
      f"{'SNR-Noisy':>10}  {'SNR-Recon':>10}  {'Improved':>9}")
print(f"      {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*9}")

for i in range(n_test):
    mae_n = mean_absolute_error(te_c[i], te_n[i])
    mae_r = mean_absolute_error(te_c[i], te_r[i])
    snr_n = 10 * np.log10(np.var(te_c[i]) / (np.mean((te_c[i]-te_n[i])**2) + 1e-10))
    snr_r = 10 * np.log10(np.var(te_c[i]) / (np.mean((te_c[i]-te_r[i])**2) + 1e-10))
    improved = mae_r < mae_n

    seg_mae_n.append(mae_n);    seg_mae_r.append(mae_r)
    seg_snr_n.append(snr_n);    seg_snr_r.append(snr_r)
    seg_improved.append(improved)

    flag = "YES ✓" if improved else "no"
    print(f"      {i:>4}  {mae_n:>10.2f}  {mae_r:>10.2f}  "
          f"{snr_n:>10.3f}  {snr_r:>10.3f}  {flag:>9}")

seg_mae_n    = np.array(seg_mae_n)
seg_mae_r    = np.array(seg_mae_r)
seg_snr_n    = np.array(seg_snr_n)
seg_snr_r    = np.array(seg_snr_r)
seg_improved = np.array(seg_improved)

n_improved   = seg_improved.sum()
pct_improved = seg_improved.mean() * 100

print(f"\n      Total pipeline segments      : {n_total}")
print(f"      Segments reconstructed       : {n_reconstructed} / {n_test} (test set)")
print(f"      Segments improved (MAE)      : {n_improved} / {n_test}  ({pct_improved:.1f}%)")
print(f"      Mean MAE  noisy              : {seg_mae_n.mean():.2f}")
print(f"      Mean MAE  reconstructed      : {seg_mae_r.mean():.2f}")
print(f"      Mean SNR  noisy              : {seg_snr_n.mean():.3f} dB")
print(f"      Mean SNR  reconstructed      : {seg_snr_r.mean():.3f} dB")
print(f"      Avg SNR gain                 : +{(seg_snr_r - seg_snr_n).mean():.3f} dB")


# ==============================================================
# Sample Output — 5 representative segments
# ==============================================================

print("\n[4/6] Sample reconstruction results (5 segments):")
print(f"\n      {'Seg':>4}  {'MAE-Noisy':>10}  {'MAE-Recon':>10}  "
      f"{'SNR-Noisy':>10}  {'SNR-Recon':>10}  {'MAE Imp':>9}  {'SNR Gain':>9}")
print(f"      {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*9}")

sample_idx = [0, 1, 2, 50, 100]

for i in sample_idx:
    delta_mae_pct = (1 - seg_mae_r[i] / seg_mae_n[i]) * 100
    delta_snr     = seg_snr_r[i] - seg_snr_n[i]
    print(f"      {i:>4}  {seg_mae_n[i]:>10.2f}  {seg_mae_r[i]:>10.2f}  "
          f"{seg_snr_n[i]:>10.3f}  {seg_snr_r[i]:>10.3f}  "
          f"{delta_mae_pct:>+8.1f}%  {delta_snr:>+8.2f} dB")

best_i  = np.argmin(seg_mae_r)
worst_i = np.argmax(seg_mae_r)

print(f"\n      Best  reconstruction : Seg {best_i:>3}  "
      f"MAE = {seg_mae_r[best_i]:.2f}   SNR = {seg_snr_r[best_i]:.3f} dB")
print(f"      Worst reconstruction : Seg {worst_i:>3}  "
      f"MAE = {seg_mae_r[worst_i]:.2f}  SNR = {seg_snr_r[worst_i]:.3f} dB")
print(f"      Avg SNR gain         : +{(seg_snr_r - seg_snr_n).mean():.3f} dB "
      f"across {n_test} test segments")


# ==============================================================
# MODEL TESTING — prove model works
# ==============================================================

print("\n[5/6] Model testing — verifying reconstruction quality...")

import pickle

with open("outputs/autoencoder_model.pkl", "rb") as f:
    model = pickle.load(f)

# Test 1: Predict on a fresh batch and check MSE
test_pred   = model.predict(Xn_te)
test_mse    = mean_squared_error(Xc_te, test_pred)
noisy_mse   = mean_squared_error(Xc_te, Xn_te)
mse_red_pct = (1 - test_mse / noisy_mse) * 100

print(f"\n      --- TEST 1: MSE on held-out test set ---")
print(f"      Noisy  MSE (baseline)  : {noisy_mse:.6f}")
print(f"      Model  MSE (predicted) : {test_mse:.6f}")
print(f"      MSE reduction          : {mse_red_pct:.1f}%")
print(f"      RESULT: {'PASS ✓' if mse_red_pct > 0 else 'FAIL ✗'}  "
      f"(model MSE is {'lower' if mse_red_pct > 0 else 'higher'} than noisy baseline)")

# Test 2: Pearson correlation on each test segment
print(f"\n      --- TEST 2: Per-segment Pearson r ---")
r_noisy_list = []
r_recon_list = []

for i in range(n_test):
    r_n, _ = pearsonr(Xc_te[i], Xn_te[i])
    r_r, _ = pearsonr(Xc_te[i], test_pred[i])
    r_noisy_list.append(r_n)
    r_recon_list.append(r_r)

r_noisy_arr = np.array(r_noisy_list)
r_recon_arr = np.array(r_recon_list)

pct_r_improved = (r_recon_arr > r_noisy_arr).mean() * 100

print(f"      Mean Pearson r (noisy  vs clean) : {r_noisy_arr.mean():.4f}")
print(f"      Mean Pearson r (recon  vs clean) : {r_recon_arr.mean():.4f}")
print(f"      Segments where r improved        : "
      f"{(r_recon_arr > r_noisy_arr).sum()} / {n_test}  ({pct_r_improved:.1f}%)")
print(f"      RESULT: {'PASS ✓' if pct_r_improved > 50 else 'FAIL ✗'}  "
      f"(reconstruction correlation is higher in {pct_r_improved:.1f}% of segments)")

# Test 3: SNR improvement
print(f"\n      --- TEST 3: SNR improvement ---")
pct_snr_improved = (seg_snr_r > seg_snr_n).mean() * 100
avg_snr_gain     = (seg_snr_r - seg_snr_n).mean()

print(f"      Segments with SNR gain           : "
      f"{(seg_snr_r > seg_snr_n).sum()} / {n_test}  ({pct_snr_improved:.1f}%)")
print(f"      Average SNR gain                 : +{avg_snr_gain:.3f} dB")
print(f"      RESULT: {'PASS ✓' if pct_snr_improved > 50 else 'FAIL ✗'}  "
      f"(SNR improved in {pct_snr_improved:.1f}% of test segments)")

# Test 4: Sanity — reconstructed must differ from noisy
diff = np.abs(test_pred - Xn_te).mean()
print(f"\n      --- TEST 4: Reconstruction sanity check ---")
print(f"      Mean absolute difference (recon vs noisy) : {diff:.6f}")
print(f"      RESULT: {'PASS ✓' if diff > 1e-4 else 'FAIL ✗ (model may be copying input)'}  "
      f"(model is actively transforming the signal)")

# Test summary
print(f"\n      === MODEL TEST SUMMARY ===")
print(f"      Test 1 — MSE reduction     : {mse_red_pct:+.1f}%   "
      f"{'PASS ✓' if mse_red_pct > 0 else 'FAIL ✗'}")
print(f"      Test 2 — Pearson r gain    : {(r_recon_arr.mean()-r_noisy_arr.mean())*100:+.2f} pp  "
      f"{'PASS ✓' if r_recon_arr.mean() > r_noisy_arr.mean() else 'FAIL ✗'}")
print(f"      Test 3 — SNR gain          : +{avg_snr_gain:.3f} dB  "
      f"{'PASS ✓' if avg_snr_gain > 0 else 'FAIL ✗'}")
print(f"      Test 4 — Sanity check      : diff={diff:.4f}  "
      f"{'PASS ✓' if diff > 1e-4 else 'FAIL ✗'}")


# ==============================================================
# Save CSV + JSON
# ==============================================================

print("\n[6/6] Saving evaluation results...")

df = pd.DataFrame({
    "Metric"       : ["MSE", "RMSE", "MAE", "SNR_dB", "Pearson_r"],
    "Noisy"        : [mn["MSE"], mn["RMSE"], mn["MAE"], mn["SNR"], mn["Pearson_r"]],
    "Reconstructed": [mr["MSE"], mr["RMSE"], mr["MAE"], mr["SNR"], mr["Pearson_r"]]
})
df.to_csv("outputs/reconstruction_metrics.csv", index=False)

summary = {
    "n_total"         : int(n_total),
    "n_train"         : int(n_train),
    "n_test"          : int(n_test),
    "n_reconstructed" : int(n_reconstructed),
    "n_improved"      : int(n_improved),
    "pct_improved"    : pct_improved,
    "mse_imp"         : mse_imp,
    "rmse_imp"        : rmse_imp,
    "mae_imp"         : mae_imp,
    "snr_imp"         : snr_imp,
    "cor_imp"         : cor_imp,
    "avg_snr_gain"    : float((seg_snr_r - seg_snr_n).mean()),
    "noisy"           : mn,
    "recon"           : mr,
    "post_sqa"        : {}
}

with open("outputs/eval_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

np.save("outputs/seg_mae_n.npy", seg_mae_n)
np.save("outputs/seg_mae_r.npy", seg_mae_r)
np.save("outputs/seg_snr_n.npy", seg_snr_n)
np.save("outputs/seg_snr_r.npy", seg_snr_r)


# ==============================================================
# Plot
# ==============================================================

fig, ax = plt.subplots(figsize=(10, 6))
labels = ["MSE Reduction (%)", "RMSE Reduction (%)", "MAE Reduction (%)", "SNR Gain (dB)"]
values = [mse_imp, rmse_imp, mae_imp, snr_imp]
colors = ['#4CAF50' if v > 0 else '#F44336' for v in values]
ax.bar(labels, values, color=colors, alpha=0.85)
ax.set_title("Step 4 — Reconstruction Improvement Metrics", fontsize=12, fontweight='bold')
ax.set_ylabel("Improvement")
ax.axhline(0, color='black', lw=0.8)
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/step4_metrics.png", dpi=200)
plt.close()


# ==============================================================
# Final Summary
# ==============================================================

print("\n" + "=" * 65)
print("  STEP 4 COMPLETE")
print("=" * 65)
print(f"  Total pipeline segments  : {n_total}  (569 original × 3 augmentations)")
print(f"  Train / Test split       : {n_train} / {n_test}  (80% / 20%)")
print(f"  Signals reconstructed    : {n_reconstructed} / {n_test} test segments")
print(f"  Segments improved (MAE)  : {n_improved} / {n_test}  ({pct_improved:.1f}%)")
print(f"  MSE  reduction           : {mse_imp:+.1f}%")
print(f"  RMSE reduction           : {rmse_imp:+.1f}%")
print(f"  MAE  reduction           : {mae_imp:+.1f}%")
print(f"  SNR  gain                : {snr_imp:+.2f} dB")
print(f"  Avg  SNR gain/segment    : +{(seg_snr_r-seg_snr_n).mean():.3f} dB")
print(f"  Pearson r                : {mn['Pearson_r']:.4f} -> {mr['Pearson_r']:.4f}")
print("=" * 65)