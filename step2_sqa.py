
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis
from scipy.ndimage import gaussian_filter1d
from collections import Counter

np.random.seed(42)
os.makedirs("outputs", exist_ok=True)

FEAT_NAMES = ['ApEn','SampEn','Shannon','Spectral','PermEn','Skewness','Kurtosis','Higuchi FD','SNR (dB)']

print("=" * 65)
print("  STEP 2: SIGNAL QUALITY ASSESSMENT (SQA)")

print("=" * 65)

print("\n[1/4] Loading preprocessed data...")
Xc         = np.load("outputs/Xc.npy")
Xn         = np.load("outputs/Xn.npy")
clean_segs = np.load("outputs/clean_segs.npy")
print(f"      Clean shape : {Xc.shape}")
print(f"      Noisy shape : {Xn.shape}")
print(f"      Segments    : {len(clean_segs)}")

print("\n[2/4] SQA feature definitions:")
print("      Feature 1  ApEn           — signal regularity (lower = cleaner)")
print("      Feature 2  SampEn         — sample entropy (less biased)")
print("      Feature 3  Shannon Ent    — amplitude distribution uncertainty")
print("      Feature 4  Spectral Ent   — frequency domain complexity")
print("      Feature 5  PermEn         — ordinal pattern randomness")
print("      Feature 6  Skewness       — amplitude asymmetry")
print("      Feature 7  Kurtosis       — tail heaviness of distribution")
print("      Feature 8  Higuchi FD     — nonlinear fractal complexity")
print("      Feature 9  SNR (dB)       — signal-to-noise ratio estimate")
print("\n      Grading rules:")
print("      Class 0 (clean)       : SNR >= 10 dB  AND  ApEn < 0.6")
print("      Class 1 (partly clean): SNR >= 5 dB   OR  (ApEn < 0.8 AND |Kurt| < 5)")
print("      Class 2 (corrupted)   : all other cases")

def approx_entropy(sig, m=2, r=0.2):
    r *= np.std(sig)+1e-10; N = len(sig)
    def phi(m_):
        xm = np.array([sig[i:i+m_] for i in range(N-m_)])
        C  = np.sum(np.max(np.abs(xm[:,None]-xm[None,:]),axis=2)<=r,axis=1)/(N-m_)
        return np.mean(np.log(C+1e-10))
    return abs(phi(m)-phi(m+1))

def samp_entropy(sig, m=2, r=0.2):
    r *= np.std(sig)+1e-10; N = len(sig)
    def cnt(m_):
        xm = np.array([sig[i:i+m_] for i in range(N-m_)])
        return np.sum(np.max(np.abs(xm[:,None]-xm[None,:]),axis=2)<=r)-(N-m_)
    return -np.log((cnt(m+1)+1e-10)/(cnt(m)+1e-10))

def shannon_ent(sig, bins=32):
    h,_ = np.histogram(sig, bins=bins, density=True); h=h[h>0]
    return -np.sum(h*np.log2(h+1e-10))

def spectral_ent(sig, fs=100):
    _,psd = sp_signal.welch(sig, fs=fs, nperseg=min(64,len(sig)))
    p = psd/(np.sum(psd)+1e-10); p=p[p>0]
    return -np.sum(p*np.log2(p+1e-10))

def perm_ent(sig, m=3, tau=1):
    N=len(sig)
    pats=[tuple(np.argsort([sig[i+j*tau] for j in range(m)])) for i in range(N-(m-1)*tau)]
    cnt=Counter(pats); tot=sum(cnt.values())
    prb=np.array([v/tot for v in cnt.values()])
    return -np.sum(prb*np.log2(prb+1e-10))

def higuchi_fd(sig, k_max=8):
    N=len(sig); L=[]
    for k in range(1,k_max+1):
        Lk=[]
        for m in range(1,k+1):
            idx=list(range(m-1,N,k))
            if len(idx)<2: continue
            Lk.append(np.sum(np.abs(np.diff(sig[idx])))*(N-1)/(len(idx)*k))
        if Lk: L.append(np.mean(Lk))
    if len(L)<2: return 1.0
    return np.polyfit(np.log(np.arange(1,len(L)+1)),np.log(np.array(L)+1e-10),1)[0]

def seg_snr(seg):
    ref=gaussian_filter1d(seg,2)
    return 10*np.log10(np.var(seg)/(np.var(seg-ref)+1e-10))

def compute_features(seg):
    return np.array([approx_entropy(seg), samp_entropy(seg), shannon_ent(seg),
                     spectral_ent(seg), perm_ent(seg), skew(seg), kurtosis(seg),
                     higuchi_fd(seg), seg_snr(seg)])

def sqa_grade(feat):
    snr, apen, kurt = feat[8], feat[0], feat[6]
    if   snr >= 10 and apen < 0.6:                    return 'clean'
    elif snr >=  5 or (apen < 0.8 and abs(kurt) < 5): return 'partly_clean'
    else:                                              return 'corrupted'

def make_noisy_local(seg, seed):
    rng=np.random.RandomState(seed+100); std=np.std(seg)
    return seg + rng.normal(0, 0.25*std, len(seg))

N_SQA   = min(len(clean_segs), 80)
sqa_idx = np.random.choice(len(clean_segs), N_SQA, replace=False)

print(f"\n[3/4] Computing SQA on {N_SQA} segments...")
print(f"\n      {'Seg':>4}  {'ApEn':>7}  {'SampEn':>7}  {'SNR(dB)':>8}  {'Kurt':>7}  {'Grade-Clean':>13}  {'Grade-Noisy':>13}")
print(f"      {'-'*4}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*13}  {'-'*13}")

gc, gn, feats_c, feats_n = [], [], [], []
for rank, i in enumerate(sqa_idx):
    fc = compute_features(clean_segs[i])
    fn = compute_features(make_noisy_local(clean_segs[i], i))
    gc.append(sqa_grade(fc)); feats_c.append(fc)
    gn.append(sqa_grade(fn)); feats_n.append(fn)
    print(f"      {i:>4}  {fc[0]:>7.4f}  {fc[1]:>7.4f}  {fc[8]:>8.2f}  "
          f"{fc[6]:>7.3f}  {gc[-1]:>13}  {gn[-1]:>13}")

feats_c = np.array(feats_c)
feats_n = np.array(feats_n)
dist_c  = {g: gc.count(g) for g in ['clean','partly_clean','corrupted']}
dist_n  = {g: gn.count(g) for g in ['clean','partly_clean','corrupted']}

print(f"\n      --- GRADE SUMMARY ---")
print(f"      {'Grade':<15}  {'Clean signals':>15}  {'Noisy signals':>15}")
print(f"      {'-'*15}  {'-'*15}  {'-'*15}")
for g in ['clean','partly_clean','corrupted']:
    print(f"      {g:<15}  {dist_c[g]:>15}  {dist_n[g]:>15}")

print(f"\n      --- FEATURE STATISTICS (clean segments) ---")
print(f"      {'Feature':<15}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}")
print(f"      {'-'*15}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")
for fi, name in enumerate(FEAT_NAMES):
    print(f"      {name:<15}  {feats_c[:,fi].mean():>9.4f}  {feats_c[:,fi].std():>9.4f}"
          f"  {feats_c[:,fi].min():>9.4f}  {feats_c[:,fi].max():>9.4f}")

np.save("outputs/feats_clean.npy",  feats_c)
np.save("outputs/feats_noisy.npy",  feats_n)
np.save("outputs/grades_clean.npy", np.array(gc))
np.save("outputs/grades_noisy.npy", np.array(gn))
with open("outputs/sqa_dist.json","w") as f:
    json.dump({"clean": dist_c, "noisy": dist_n}, f)

print("\n[4/4] Saving SQA plots...")

# Image 1: Grade distribution
fig, ax = plt.subplots(figsize=(8, 5), facecolor="#F8F9FA")
ax.set_facecolor("#F8F9FA")
lbs = ['Clean','Partly Clean','Corrupted']
vc  = [dist_c['clean'], dist_c['partly_clean'], dist_c['corrupted']]
vn  = [dist_n['clean'], dist_n['partly_clean'], dist_n['corrupted']]
x = np.arange(3); w = 0.35
b1 = ax.bar(x-w/2, vc, w, label='Clean signals',  color='#4CAF50', alpha=0.85)
b2 = ax.bar(x+w/2, vn, w, label='Noisy signals',  color='#F44336', alpha=0.85)
for bar, v in zip(list(b1)+list(b2), vc+vn):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
            str(v), ha='center', fontsize=11, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(lbs, fontsize=11)
ax.set_title("Step 2 — SQA Grade Distribution\n",
             fontsize=12, fontweight='bold')
ax.set_ylabel("Segment Count"); ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("outputs/step2a_sqa_grades.png", dpi=200, bbox_inches='tight')
plt.close()
print("      Saved: outputs/step2a_sqa_grades.png")

# Image 2: ApEn vs SNR scatter
fig, ax = plt.subplots(figsize=(8, 6), facecolor="#F8F9FA")
ax.set_facecolor("#F8F9FA")
ax.scatter(feats_c[:,0], feats_c[:,8], color='#2196F3', alpha=0.8, s=55, label='Clean signals', zorder=3)
ax.scatter(feats_n[:,0], feats_n[:,8], color='#F44336', alpha=0.8, s=55, label='Noisy signals', zorder=3)
ax.axvline(0.6, color='#555', ls='--', lw=1.5, label='ApEn threshold = 0.6')
ax.axhline(10,  color='#555', ls=':',  lw=1.5, label='SNR threshold = 10 dB')
ax.set_xlabel("Approximate Entropy (ApEn)", fontsize=11)
ax.set_ylabel("SNR (dB)", fontsize=11)
ax.set_title("Step 2 — ApEn vs SNR: Grade Decision Boundary\n",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/step2b_apen_vs_snr.png", dpi=200, bbox_inches='tight')
plt.close()
print("      Saved: outputs/step2b_apen_vs_snr.png")

# Image 3: Radar chart
fc_m = feats_c.mean(axis=0)
fn_m = feats_n.mean(axis=0)
fmin = np.minimum(fc_m, fn_m); fmax = np.maximum(fc_m, fn_m)+1e-10
fc_r = (fc_m-fmin)/(fmax-fmin); fn_r = (fn_m-fmin)/(fmax-fmin)
angles = np.linspace(0, 2*np.pi, len(FEAT_NAMES), endpoint=False).tolist()
angles += angles[:1]
fc_r_p = fc_r.tolist()+[fc_r[0]]; fn_r_p = fn_r.tolist()+[fn_r[0]]
fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True), facecolor="#F8F9FA")
ax.set_facecolor("#F8F9FA")
ax.plot(angles, fc_r_p, color='#2196F3', lw=2.5, label='Clean signals')
ax.fill(angles, fc_r_p, color='#2196F3', alpha=0.18)
ax.plot(angles, fn_r_p, color='#F44336', lw=2.5, label='Noisy signals')
ax.fill(angles, fn_r_p, color='#F44336', alpha=0.18)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(FEAT_NAMES, fontsize=10, fontweight='bold')
ax.set_ylim(0,1)
ax.set_title("Step 2 — SQA Feature Radar: Clean vs Noisy\n",
             fontsize=12, fontweight='bold', pad=25)
ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.1), fontsize=10)
plt.tight_layout()
plt.savefig("outputs/step2c_sqa_radar.png", dpi=200, bbox_inches='tight')
plt.close()
print("      Saved: outputs/step2c_sqa_radar.png")

print("\n" + "=" * 65)
print("  STEP 2 COMPLETE")
print("=" * 65)
print(f"  Segments assessed   : {N_SQA}")
print(f"  Clean  -> clean     : {dist_c['clean']} / {N_SQA}")
print(f"  Noisy  -> partly    : {dist_n['partly_clean']} / {N_SQA}")
print(f"  Mean clean SNR      : {feats_c[:,8].mean():.2f} dB")
print(f"  Mean noisy SNR      : {feats_n[:,8].mean():.2f} dB")
print("=" * 65)
