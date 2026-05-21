
import os
import json
import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

print("=" * 65)
print("  STEP 5: VISUALIZATION")
print("  Generating final plots")
print("=" * 65)


print("\n[1/3] Loading results...")

te_c = np.load("outputs/te_c_phys.npy")
te_n = np.load("outputs/te_n_phys.npy")
te_r = np.load("outputs/te_r_phys.npy")

seg_mae_n = np.load("outputs/seg_mae_n.npy")
seg_mae_r = np.load("outputs/seg_mae_r.npy")
seg_snr_n = np.load("outputs/seg_snr_n.npy")
seg_snr_r = np.load("outputs/seg_snr_r.npy")

with open("outputs/eval_summary.json") as f:
    ev = json.load(f)

n_total         = ev.get("n_total", 1707)
n_train         = ev.get("n_train", 1365)
n_test          = te_c.shape[0]
n_reconstructed = ev.get("n_reconstructed", n_test)
n_improved      = ev.get("n_improved", 0)
pct_improved    = ev.get("pct_improved", 0.0)
avg_snr_gain    = ev.get("avg_snr_gain", 0.0)

FS   = 100
t_ms = np.arange(te_c.shape[1]) / FS * 1000

# Auto-pick best reconstructed segment
idx = int(np.argmin(seg_mae_r))
print(f"      Best reconstructed segment   : index {idx}")
print(f"        MAE  noisy  -> recon       : {seg_mae_n[idx]:.2f} -> {seg_mae_r[idx]:.2f}")
print(f"        SNR  noisy  -> recon       : {seg_snr_n[idx]:.3f} -> {seg_snr_r[idx]:.3f} dB")
print(f"      Total pipeline segments      : {n_total}")
print(f"      Train / Test split           : {n_train} / {n_test}  (80/20)")
print(f"      Signals reconstructed        : {n_reconstructed} / {n_test}")
print(f"      Segments improved            : {n_improved} / {n_test}  ({pct_improved:.1f}%)")
print(f"      Avg SNR gain                 : +{avg_snr_gain:.3f} dB")


print("\n[2/3] Plotting signal comparison...")

fig1, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, facecolor="#F8F9FA")
fig1.suptitle("PPG Signal Reconstruction: Clean vs Noisy vs Reconstructed",
              fontsize=13, fontweight='bold')

axes[0].plot(t_ms, te_c[idx], color='#1565C0', lw=2)
axes[0].set_title("Clean PPG Signal", fontsize=11, fontweight='bold')
axes[0].set_ylabel("Amplitude (ADC)")
axes[0].grid(True, alpha=0.35)
axes[0].set_facecolor("#F8F9FA")

axes[1].plot(t_ms, te_n[idx], color='#C62828', lw=1.3, alpha=0.9)
axes[1].set_title("Noisy PPG Signal", fontsize=11, fontweight='bold')
axes[1].set_ylabel("Amplitude (ADC)")
axes[1].grid(True, alpha=0.35)
axes[1].set_facecolor("#F8F9FA")

axes[2].plot(t_ms, te_r[idx], color='#2E7D32', lw=2.2, label='Reconstructed')
axes[2].plot(t_ms, te_c[idx], color='#1565C0', lw=1.2, ls='--', alpha=0.6, label='Clean reference')
axes[2].set_title("Reconstructed PPG Signal", fontsize=11, fontweight='bold')
axes[2].set_xlabel("Time (ms)")
axes[2].set_ylabel("Amplitude (ADC)")
axes[2].legend(fontsize=10, loc='upper right')
axes[2].grid(True, alpha=0.35)
axes[2].set_facecolor("#F8F9FA")

fig1.tight_layout()
fig1.savefig("outputs/signal_reconstruction_comparison.png", dpi=200, bbox_inches='tight')
print("      Saved : outputs/signal_reconstruction_comparison.png")

print("\n[3/3] Plotting error comparison...")

err_before = te_n[idx] - te_c[idx]
err_after  = te_r[idx] - te_c[idx]

fig2, ax = plt.subplots(figsize=(13, 5), facecolor="#F8F9FA")
ax.set_facecolor("#F8F9FA")

ax.plot(t_ms, err_before, color='#1565C0', lw=1.2, alpha=0.85,
        label="Error Before Reconstruction")
ax.plot(t_ms, err_after,  color='#E65100', lw=2.0, alpha=0.90,
        label="Error After Reconstruction")

ax.set_title("Time-Domain Error Signal Comparison", fontsize=12, fontweight='bold')
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Error Amplitude")
ax.legend(fontsize=10)
ax.axhline(0, color='black', lw=0.8, ls='--')
ax.grid(True, alpha=0.35)

fig2.tight_layout()
fig2.savefig("outputs/time_domain_error_comparison.png", dpi=200, bbox_inches='tight')
print("      Saved : outputs/time_domain_error_comparison.png")


plt.show()
plt.close('all')

print("\n" + "=" * 65)
print("  STEP 5 COMPLETE")
print("=" * 65)
print(f"  Best segment displayed   : #{idx}  "
      f"(MAE {seg_mae_n[idx]:.1f} -> {seg_mae_r[idx]:.1f})")
print(f"  Total pipeline segments  : {n_total}  (569 × 3 augmentations)")
print(f"  Train / Test             : {n_train} / {n_test}  (80% / 20%)")
print(f"  Signals reconstructed    : {n_reconstructed} / {n_test} test segments")
print(f"  Segments improved        : {n_improved} / {n_test}  ({pct_improved:.1f}%)")
print(f"  Avg SNR gain             : +{avg_snr_gain:.3f} dB")
print("  Output files saved:")
print("   - signal_reconstruction_comparison.png")
print("   - time_domain_error_comparison.png")
print("=" * 65)