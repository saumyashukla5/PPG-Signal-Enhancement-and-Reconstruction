
import subprocess, sys, time

steps = [
    ("Step 1 — Preprocessing",            "step1_preprocessing.py"),
    ("Step 2 — Signal Quality Assessment", "step2_sqa.py"),
    ("Step 3 — Autoencoder Reconstruction","step3_autoencoder.py"),
    ("Step 4 — Quality Verification",      "step4_quality_verification.py"),
    ("Step 5 — Visualization",             "step5_visualization.py"),
]

print("=" * 65)
print("  PPG AUTOENCODER PIPELINE — RUNNING ALL STEPS")
print("=" * 65)

total_start = time.time()
for name, script in steps:
    print(f"\n{'━'*65}")
    print(f"  ▶  {name}")
    print(f"{'━'*65}\n")
    start  = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"\n  ✔  {name}  ({elapsed:.1f}s)")
    else:
        print(f"\n  ✘  {name} FAILED")
        sys.exit(1)

total = time.time() - total_start
print(f"\n{'='*65}")
print(f"  ALL STEPS DONE in {total:.1f}s  |  Check outputs/ folder")
print(f"{'='*65}")
