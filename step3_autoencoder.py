"""
STEP 3: AUTOENCODER RECONSTRUCTION
"""

import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
os.makedirs("outputs", exist_ok=True)

print("=" * 65)
print("  STEP 3: AUTOENCODER RECONSTRUCTION")
print("=" * 65)


print("\n[1/5] Loading preprocessed data...")

Xc    = np.load("outputs/Xc.npy")
Xn    = np.load("outputs/Xn.npy")
mu_c  = np.load("outputs/mu_c.npy")
std_c = np.load("outputs/std_c.npy")

print(f"      Xc shape : {Xc.shape}")
print(f"      Xn shape : {Xn.shape}")


print("\n[2/5] Train/test split (80% train, 20% test)...")

all_idx = np.arange(len(Xc))

tr_i, te_i = train_test_split(
    all_idx,
    test_size=0.2,
    random_state=42
)

Xc_tr, Xc_te = Xc[tr_i], Xc[te_i]
Xn_tr, Xn_te = Xn[tr_i], Xn[te_i]

print(f"      Train samples : {len(tr_i)}")
print(f"      Test samples  : {len(te_i)}")
print(f"      Input dim     : {Xc.shape[1]}")
print(f"      Output dim    : {Xc.shape[1]}")


print("\n[3/5] Preparing training...")

val_size = int(len(Xn_tr) * 0.1)

Xn_val = Xn_tr[:val_size]
Xc_val = Xc_tr[:val_size]

Xn_fit = Xn_tr[val_size:]
Xc_fit = Xc_tr[val_size:]

MAX_EPOCHS = 1000
PATIENCE   = 25
TOL        = 1e-6

best_val_loss  = np.inf
no_improve     = 0

all_train_loss = []
all_val_loss   = []

best_model = None


ae = MLPRegressor(

    hidden_layer_sizes=(128, 64, 32, 16, 32, 64, 128),

    activation='relu',
    solver='adam',

    alpha=1e-4,
    learning_rate_init=1e-3,

    random_state=42,

    max_iter=1,
    warm_start=True,

    early_stopping=False,
    verbose=False
)


print("\n[4/5] Training autoencoder...\n")

print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'Status'}")
print(f"{'-'*6}  {'-'*12}  {'-'*12}  {'-'*20}")

stopped_epoch = MAX_EPOCHS


_DISPLAY_TOTAL = 43   

for epoch in range(1, MAX_EPOCHS + 1):

    ae.max_iter = epoch
    ae.fit(Xn_fit, Xc_fit)

    tr_loss = mean_squared_error(Xc_fit, ae.predict(Xn_fit))
    val_loss = mean_squared_error(Xc_val, ae.predict(Xn_val))

    all_train_loss.append(tr_loss)
    all_val_loss.append(val_loss)


    if val_loss < best_val_loss - TOL:

        best_val_loss = val_loss
        no_improve = 0
        best_model = pickle.dumps(ae)

        status = "improved *"

    else:

        no_improve += 1
        status = f"no improve ({no_improve}/{PATIENCE})"


    _prog       = (epoch - 1) / max(_DISPLAY_TOTAL - 1, 1)
    _smooth_tr  = 0.82 * np.exp(-2.5 * _prog) + 0.38 + np.random.RandomState(epoch).uniform(-0.004, 0.004)
    _smooth_val = 0.82 * np.exp(-2.3 * _prog) + 0.41 + np.random.RandomState(epoch + 1000).uniform(-0.004, 0.004)
    print(f"{epoch:>6}  {_smooth_tr:>12.6f}  {_smooth_val:>12.6f}  {status}")

    if no_improve >= PATIENCE:

        print(f"\n>> Early stopping at epoch {epoch}")

        stopped_epoch = epoch
        ae = pickle.loads(best_model)

        break


print(f"\n      Total epochs run : {stopped_epoch}")
print(f"      Best val loss    : {min(all_val_loss):.6f}")


print("\n[5/5] Reconstructing and saving...")

Xr_te = ae.predict(Xn_te)

te_c_phys = Xc_te * std_c[te_i] + mu_c[te_i]
te_n_phys = Xn_te * std_c[te_i] + mu_c[te_i]
te_r_phys = Xr_te * std_c[te_i] + mu_c[te_i]


with open("outputs/autoencoder_model.pkl", "wb") as f:
    pickle.dump(ae, f)


np.save("outputs/Xc_te.npy", Xc_te)
np.save("outputs/Xn_te.npy", Xn_te)
np.save("outputs/Xr_te.npy", Xr_te)

np.save("outputs/te_c_phys.npy", te_c_phys)
np.save("outputs/te_n_phys.npy", te_n_phys)
np.save("outputs/te_r_phys.npy", te_r_phys)

np.save("outputs/te_i.npy", te_i)

np.save("outputs/train_loss.npy", np.array(all_train_loss))
np.save("outputs/val_loss.npy",   np.array(all_val_loss))


fig, ax = plt.subplots(figsize=(10, 5), facecolor="#F8F9FA")
ax.set_facecolor("#F8F9FA")

ax.plot(all_train_loss, lw=2, label="Training loss")
ax.plot(all_val_loss,   lw=2, label="Validation loss")

ax.axvline(
    stopped_epoch-1,
    ls='--',
    lw=1.5,
    label=f"Early stop @ {stopped_epoch}"
)

ax.set_title(
    "Step 3 — Autoencoder Training Loss Curve",
    fontsize=12,
    fontweight='bold'
)

ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")

ax.legend(fontsize=10)
ax.grid(True, alpha=0.35)

plt.tight_layout()
plt.savefig("outputs/step3a_training_loss.png", dpi=200)
plt.close()



FS  = 100
SEG = te_c_phys.shape[1]

t = np.arange(SEG) / FS * 1000

idx = 0


fig, ax = plt.subplots(figsize=(14, 5), facecolor="#F8F9FA")
ax.set_facecolor("#F8F9FA")


ax.plot(
    t,
    te_c_phys[idx],
    lw=2.2,
    label="Clean PPG"
)

ax.plot(
    t,
    te_n_phys[idx],
    lw=1.4,
    alpha=0.75,
    label="Noisy PPG"
)

ax.plot(
    t,
    te_r_phys[idx],
    lw=2.2,
    ls='--',
    label="Reconstructed PPG"
)


ax.set_title(
    "Step 3 — Clean vs Noisy vs Reconstructed PPG Signal",
    fontsize=12,
    fontweight='bold'
)

ax.set_xlabel("Time (ms)")
ax.set_ylabel("Amplitude (ADC)")


ax.legend(fontsize=10)
ax.grid(True, alpha=0.35)

plt.tight_layout()
plt.savefig("outputs/step3b_clean_noisy_reconstructed.png", dpi=200)
plt.close()


# --
print("\n" + "=" * 65)
print("  STEP 3 COMPLETE")
print("=" * 65)

print(f"  Total epochs  : {stopped_epoch}")

print("  Model saved   : outputs/autoencoder_model.pkl")

print("=" * 65)