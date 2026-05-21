"""
STEP 1: PREPROCESSING
Ref: Zhang et al. 2025, IEEE JBHI
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as sp_signal

np.random.seed(42)
os.makedirs("outputs", exist_ok=True)

DATA_PATH = "201902020222_Data.csv"
FS = 100
SEG_LEN = 200
HOP = 25
N_AUG = 3

print("=" * 65)
print("  STEP 1: PREPROCESSING")
print("=" * 65)

# ── LOAD ─────────────────────────────────────────────────────────
print("\n[1/7] Loading raw data...")
df = pd.read_csv(DATA_PATH, sep='\t')
ppg_raw = df['ppg'].values.astype(float)
print(f"      Rows loaded         : {len(df):,}")
print(f"      Columns             : {list(df.columns)}")
print(f"      Sampling rate       : {FS} Hz")

# ── MISSING VALUES ────────────────────────────────────────────────
print("\n[2/7] Checking data quality...")
missing = df.isnull().sum()
has_missing = missing[missing > 0]
if len(has_missing) == 0:
    print("      Missing values      : 0 (all columns clean)")
else:
    for col, cnt in has_missing.items():
        print(f"      WARNING {col}: {cnt} missing values")
dupes = df.duplicated().sum()
print(f"      Duplicate rows      : {dupes}")
print(f"      PPG min             : {ppg_raw.min():,.2f}")
print(f"      PPG max             : {ppg_raw.max():,.2f}")
print(f"      PPG mean            : {ppg_raw.mean():,.2f}")
print(f"      PPG std dev         : {ppg_raw.std():,.2f}")

# ── FILTER ────────────────────────────────────────────────────────
print("\n[3/7] Applying Butterworth bandpass filter (0.5-8 Hz, order=4)...")
sos = sp_signal.butter(4, [0.5, 8.0], btype='bandpass', fs=FS, output='sos')
ppg_filt = sp_signal.sosfiltfilt(sos, ppg_raw)
print(f"      Filter type         : Butterworth (order=4, zero-phase)")
print(f"      Passband            : 0.5 Hz - 8.0 Hz (cardiac band)")
print(f"      After filter min    : {ppg_filt.min():,.2f}")
print(f"      After filter max    : {ppg_filt.max():,.2f}")
print(f"      Std before filter   : {ppg_raw.std():,.2f}")
print(f"      Std after filter    : {ppg_filt.std():,.2f}")

print("\n[4/7] Applying Savitzky-Golay smoothing (window=51, polyorder=3)...")
ppg_smooth = sp_signal.savgol_filter(ppg_filt, window_length=51, polyorder=3)
noise_removed = ppg_raw.std() - ppg_smooth.std()
print(f"      Window length       : 51 samples")
print(f"      Polynomial order    : 3")
print(f"      Std after smoothing : {ppg_smooth.std():,.2f}")
print(f"      Noise std removed   : {noise_removed:,.2f}")

# ── SEGMENTATION ─────────────────────────────────────────────────
print("\n[5/7] Segmenting signal into windows...")
print(f"      Segment length      : {SEG_LEN} samples = {SEG_LEN/FS*1000:.0f} ms")
print(f"      Hop size            : {HOP} samples = {HOP/FS*1000:.0f} ms")
print(f"      Overlap             : {(1-HOP/SEG_LEN)*100:.0f}%")
clean_segs = []
for i in range(0, len(df) - SEG_LEN, HOP):
    seg = ppg_smooth[i:i + SEG_LEN]
    if len(seg) == SEG_LEN:
        clean_segs.append(seg)
clean_segs = np.array(clean_segs)
print(f"      Segments extracted  : {len(clean_segs)}")
print(f"      Total data points   : {len(clean_segs)*SEG_LEN:,}")

def make_noisy(seg, seed=None):
    rng = np.random.RandomState(seed)
    std = np.std(seg)
    t = np.linspace(0, 2*np.pi, len(seg))
    n = rng.normal(0, 0.25*std, len(seg))
    n += 0.08*std * np.sin(0.5*t + rng.uniform(0, np.pi))
    if rng.random() < 0.4:
        pos = rng.randint(20, len(seg)-20)
        w = rng.randint(3, 8)
        n[pos:pos+w] += rng.choice([-1,1]) * 0.35*std * rng.randn(w)
    return seg + n

print("\n[6/7] Data augmentation (3x noisy versions per segment)...")
clean_aug, noisy_aug = [], []
for ci, seg in enumerate(clean_segs):
    for s in range(N_AUG):
        clean_aug.append(seg)
        noisy_aug.append(make_noisy(seg, seed=ci*10+s))
clean_aug = np.array(clean_aug)
noisy_aug = np.array(noisy_aug)
noise_diff = noisy_aug - clean_aug
snr_noisy = 10*np.log10(np.var(clean_aug)/(np.var(noise_diff)+1e-10))
print(f"      Original segments   : {len(clean_segs)}")
print(f"      After augmentation  : {len(clean_aug)} pairs ({len(clean_segs)} x {N_AUG})")
print(f"      Noise type          : Gaussian + baseline wander + impulse artifacts")
print(f"      Mean noise level    : {noise_diff.std():.4f} (normalized)")
print(f"      Avg SNR of noisy    : {snr_noisy:.2f} dB")

print("\n[7/7] Normalizing (z-score using clean segment stats for both)...")
mu_c = clean_aug.mean(axis=1, keepdims=True)
std_c = clean_aug.std(axis=1, keepdims=True) + 1e-8
Xc = (clean_aug - mu_c) / std_c
Xn = (noisy_aug - mu_c) / std_c
print(f"      Method              : z-score per segment")
print(f"      KEY FIX             : Noisy uses CLEAN stats -> consistent inverse-transform")
print(f"      Xc mean             : {Xc.mean():.6f}  (ideal: 0)")
print(f"      Xc std              : {Xc.std():.6f}   (ideal: 1)")
print(f"      Xn mean             : {Xn.mean():.6f}")
print(f"      Xn std              : {Xn.std():.6f}")

np.save("outputs/Xc.npy",         Xc)
np.save("outputs/Xn.npy",         Xn)
np.save("outputs/mu_c.npy",       mu_c)
np.save("outputs/std_c.npy",      std_c)
np.save("outputs/clean_segs.npy", clean_segs)
np.save("outputs/ppg_raw.npy",    ppg_raw)
np.save("outputs/ppg_smooth.npy", ppg_smooth)

# ── PLOT 1: Raw vs Filtered ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4), facecolor="#F8F9FA")
ax.set_facecolor("#F8F9FA")
ax.plot(ppg_raw[:600],    color='#BDBDBD', lw=0.9, alpha=0.9, label="Raw PPG")
ax.plot(ppg_smooth[:600], color='#2196F3', lw=2.0, label="Filtered PPG (Bandpass 0.5-8Hz + Savitzky-Golay)")
ax.set_title("Step 1 — Raw vs Filtered PPG Signal\n",
             fontsize=12, fontweight='bold')
ax.set_xlabel("Sample Index"); ax.set_ylabel("Amplitude (ADC)")
ax.legend(fontsize=10); ax.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig("outputs/step1a_raw_vs_filtered.png", dpi=200, bbox_inches='tight')
plt.close()
print("\n      Saved: outputs/step1a_raw_vs_filtered.png")

# ── PLOT 2: Clean vs Noisy segment ───────────────────────────────
t_ms = np.arange(SEG_LEN)/FS*1000
fig, ax = plt.subplots(figsize=(12, 4), facecolor="#F8F9FA")
ax.set_facecolor("#F8F9FA")
ax.plot(t_ms, clean_segs[0],                   color='#2196F3', lw=2.2, label="Clean segment")
ax.plot(t_ms, make_noisy(clean_segs[0], seed=0),color='#F44336', lw=1.4, alpha=0.85, label="Noisy (augmented)")
ax.set_title("Step 1 — Sample Segment: Clean vs Noisy Augmentation",
             fontsize=12, fontweight='bold')
ax.set_xlabel("Time (ms)"); ax.set_ylabel("Amplitude (ADC)")
ax.legend(fontsize=10); ax.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig("outputs/step1b_clean_vs_noisy_segment.png", dpi=200, bbox_inches='tight')
plt.close()
print("      Saved: outputs/step1b_clean_vs_noisy_segment.png")

print("\n" + "=" * 65)
print("  STEP 1 COMPLETE")
print("=" * 65)
print(f"  Raw samples         : {len(ppg_raw):,}")
print(f"  Missing values      : 0")
print(f"  Duplicate rows      : {dupes}")
print(f"  Segments created    : {len(clean_segs)}")
print(f"  Augmented pairs     : {len(clean_aug)}")
print(f"  Output shape (Xc)   : {Xc.shape}")
print(f"  Noisy SNR           : {snr_noisy:.2f} dB")
print("=" * 65)
