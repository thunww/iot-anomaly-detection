import os
import sys
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

STEPS = [
    "src.preprocessing.prepare_dataset",
    "src.preprocessing.build_scaler",

    "src.autoencoder.train_autoencoder",
    "src.autoencoder.tune_threshold",
    "src.autoencoder.extract_latent",

    "src.xgboost.train_xgb",

    "src.hybrid.evaluate_hybrid"
]

def run_step(module_name):
    print("\n======================================")
    print(f" RUNNING STEP → {module_name}")
    print("======================================")

    result = subprocess.run(["python", "-m", module_name])

    if result.returncode != 0:
        print(f"[ERROR] Step FAILED: {module_name}")
        print("Stopping pipeline...")
        sys.exit(1)

    print(f"[OK] Completed: {module_name}")

def main():
    print("========== FULL HYBRID TRAINING PIPELINE ==========")

    for step in STEPS:
        run_step(step)

    print("\n========== PIPELINE COMPLETED SUCCESSFULLY ==========")

if __name__ == "__main__":
    main()
