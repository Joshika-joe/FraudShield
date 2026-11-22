#!/usr/bin/env bash
# FraudShield - run the complete Member 2 pipeline end to end.
#
#   ./run_pipeline.sh            full dataset (slow: Random Forest takes a while)
#   ./run_pipeline.sh --sample   quick run on a 2 % sample, for debugging
#
# Member 1's data engineering scripts are NOT re-run: this pipeline starts from
# the engineered dataset they produced.

set -euo pipefail

cd "$(dirname "$0")"
# shellcheck disable=SC1091
source env.sh

SAMPLE_ARG=""
if [[ "${1:-}" == "--sample" ]]; then
  SAMPLE_ARG="--sample 0.02"
  echo ">>> SAMPLE MODE: results are for debugging only, not for the report"
fi

mkdir -p outputs/logs
LOG="outputs/logs/pipeline_$(date +%Y%m%d_%H%M%S).log"

step() { echo; echo "=============================================================="; echo ">>> $1"; echo "=============================================================="; }

{
  step "1/9  Convert the engineered CSV output to Parquet (once)"
  if [[ -f outputs/engineered_parquet/_SUCCESS ]]; then
    echo "Parquet copy already exists - skipping"
  else
    python src/convert_to_parquet.py
  fi

  step "2/9  Verify the engineered dataset in Spark"
  python src/verify_dataset.py

  step "3/9  Train Logistic Regression"
  # shellcheck disable=SC2086
  python src/train_logistic_regression.py $SAMPLE_ARG

  step "4/9  Train Random Forest"
  # shellcheck disable=SC2086
  python src/train_random_forest.py --num-trees 100 $SAMPLE_ARG

  step "5/9  Tune the fraud threshold on the validation split"
  python src/threshold_optimization.py

  step "6/9  Final evaluation on the test split and model comparison"
  python src/model_comparison.py

  step "7/9  Generate fraud alerts and the dashboard aggregates"
  python src/fraud_alerts.py
  python src/prepare_dashboard_data.py

  step "8/9  Feature ablation and the deterministic-rule sanity check"
  # shellcheck disable=SC2086
  python src/feature_ablation.py --models lr rf --num-trees 60 --max-depth 8 $SAMPLE_ARG
  python src/rule_baseline.py

  step "9/9  Generate the results documentation from the measured outputs"
  python src/generate_report.py

  step "Integration tests"
  python tests/test_integration.py

  echo
  echo "Pipeline complete. Start the dashboard with:"
  echo "    source env.sh && streamlit run dashboard/app.py"
} 2>&1 | tee "$LOG"

echo
echo "Full log saved to $LOG"
