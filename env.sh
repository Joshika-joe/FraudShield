#!/usr/bin/env bash
# FraudShield - environment setup.
#
#   source env.sh          (run this once per terminal session)
#
# Spark is a JVM application driven from Python, so BOTH a Java 17 runtime and
# the project virtual environment must be on the path before any script runs.

FRAUDSHIELD_HOME="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export FRAUDSHIELD_HOME

# --- Java 17 (required by PySpark 4.3.0) ---
export JAVA_HOME="$HOME/Library/Java/JavaVirtualMachines/jdk-17.0.21.1+2/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"

# --- Project virtual environment (Python 3.11) ---
export PATH="$FRAUDSHIELD_HOME/venv/bin:$PATH"

# Spark must launch the SAME interpreter for the driver and the workers,
# otherwise it falls back to the system Python and cannot import pyspark.
export PYSPARK_PYTHON="$FRAUDSHIELD_HOME/venv/bin/python"
export PYSPARK_DRIVER_PYTHON="$FRAUDSHIELD_HOME/venv/bin/python"

# Keep Spark's temporary shuffle files off the small system temp volume.
export SPARK_LOCAL_DIRS="${SPARK_LOCAL_DIRS:-$FRAUDSHIELD_HOME/.spark-tmp}"
mkdir -p "$SPARK_LOCAL_DIRS"

echo "FraudShield environment ready"
echo "  JAVA_HOME : $JAVA_HOME"
echo "  Python    : $(python --version 2>&1)"
echo "  Java      : $(java -version 2>&1 | head -1)"
