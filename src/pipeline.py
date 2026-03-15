"""
pipeline.py
────────────────────────────────────────────────────
End-to-End Customer Churn & LTV Pipeline

Runs the complete project pipeline in order:
    1. Feature Engineering  — validate data
    2. Train Churn Model    — LightGBM
    3. Train LTV Model      — RandomForest
    4. Model Comparison     — benchmark both
    5. Customer Segmentation — segment + export

Usage:
    cd src && python pipeline.py
────────────────────────────────────────────────────
"""

import subprocess
import sys
import os
import time
from datetime import datetime

# ── Constants ─────────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    {
        "name"   : "Feature Engineering",
        "script" : "feature_engineering.py",
        "desc"   : "Validate data + create features",
    },
    {
        "name"   : "Train Churn Model",
        "script" : "train_churn_model.py",
        "desc"   : "Train LightGBM churn model",
    },
    {
        "name"   : "Train LTV Model",
        "script" : "train_ltv_model.py",
        "desc"   : "Train RandomForest LTV model",
    },
    {
        "name"   : "Model Comparison",
        "script" : "model_comparision.py",
        "desc"   : "Benchmark all models",
    },
    {
        "name"   : "Customer Segmentation",
        "script" : "customer_segmentation.py",
        "desc"   : "Segment + export customers",
    },
]


# ── Helpers ───────────────────────────────────────────
def separator(char="=", width=65):
    print(char * width)


def print_header():
    separator()
    print("   CUSTOMER CHURN & LTV — FULL PIPELINE")
    separator()
    print(f"\n  Started : "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Scripts : {len(STEPS)} steps\n")


def print_footer(total_time, passed, failed):
    separator()
    if failed == 0:
        print("   PIPELINE COMPLETE ✅")
    else:
        print("   PIPELINE COMPLETED WITH ERRORS ⚠️")
    separator()
    print(f"\n  Steps Passed : {passed}/{len(STEPS)}")
    print(f"  Steps Failed : {failed}/{len(STEPS)}")
    print(f"  Total Runtime: {total_time:.1f}s\n")
    separator()


def extract_metrics(output: str, script: str) -> str:
    """Extract key metrics from script output."""
    lines   = output.split("\n")
    metrics = []

    if "train_churn_model" in script:
        for line in lines:
            if "F1 Score" in line and "Test" not in line:
                metrics.append(line.strip())
            if "ROC-AUC" in line and "Test" not in line:
                metrics.append(line.strip())
            if "Caught" in line:
                metrics.append(line.strip())

    elif "train_ltv_model" in script:
        for line in lines:
            if "R²" in line or "RMSE" in line:
                metrics.append(line.strip())

    elif "customer_segmentation" in script:
        for line in lines:
            if "Champion" in line or \
               "Revenue At Risk" in line or \
               "COMPLETE" in line:
                metrics.append(line.strip())

    elif "model_comparision" in script:
        for line in lines:
            if "Winner" in line or \
               "winner" in line or \
               "Best" in line:
                metrics.append(line.strip())

    elif "feature_engineering" in script:
        for line in lines:
            if "column" in line.lower() or \
               "feature" in line.lower() or \
               "shape" in line.lower() or \
               "created" in line.lower():
                metrics.append(line.strip())

    # Return top 3 metric lines max
    return "\n".join(
        f"       {m}" for m in metrics[:3]
        if m.strip()
    )


# ── Run Single Step ───────────────────────────────────
def run_step(step: dict, index: int) -> tuple:
    """
    Run a single pipeline step.

    Returns:
        (success: bool, duration: float,
         output: str)
    """
    script_path = os.path.join(SRC_DIR, step["script"])

    print(f"  [{index}/{len(STEPS)}] "
          f"{step['name']}")
    print(f"         {step['desc']}")

    if not os.path.exists(script_path):
        print(f"  ❌ Script not found: "
              f"{step['script']}\n")
        return False, 0.0, ""

    start = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=SRC_DIR,
        )
        duration = time.time() - start

        if result.returncode == 0:
            metrics = extract_metrics(
                result.stdout, step["script"]
            )
            print(f"  ✅ Done — {duration:.1f}s")
            if metrics:
                print(metrics)
            print()
            return True, duration, result.stdout

        else:
            duration = time.time() - start
            print(f"  ❌ Failed — {duration:.1f}s")
            # Print last 5 lines of error
            err_lines = [
                l for l in
                result.stderr.split("\n")
                if l.strip()
            ][-5:]
            for line in err_lines:
                print(f"       {line}")
            print()
            return False, duration, result.stderr

    except Exception as e:
        duration = time.time() - start
        print(f"  ❌ Exception — {str(e)}\n")
        return False, duration, str(e)


# ── Main Pipeline ─────────────────────────────────────
def run_pipeline():
    print_header()

    total_start = time.time()
    passed      = 0
    failed      = 0
    step_times  = []

    for i, step in enumerate(STEPS, 1):
        success, duration, output = run_step(
            step, i
        )
        step_times.append((step["name"], duration))

        if success:
            passed += 1
        else:
            failed += 1
            print(f"  ⚠️  Pipeline stopped at step "
                  f"{i}: {step['name']}")
            print(f"  Fix the error and re-run "
                  f"pipeline.py\n")
            break

    total_time = time.time() - total_start

    # Step timing summary
    separator("-")
    print("  STEP TIMING SUMMARY")
    separator("-")
    for name, t in step_times:
        bar = "█" * int(t * 2)
        print(f"  {name:<30} {t:>5.1f}s  {bar}")

    print_footer(total_time, passed, failed)


# ── Entry Point ───────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()
