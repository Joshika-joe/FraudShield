# FraudShield - Setup and Execution Guide

Everything here was actually run on the development machine (macOS on Intel,
8 GB RAM, 8 logical cores). Commands are copy-paste ready.

---

## 1. What the project needs

| Component | Version used | Why |
|---|---|---|
| Python | 3.11.18 | PySpark 4.3.0 is tested against 3.11; newer interpreters have no prebuilt wheels for pandas/pyarrow yet |
| Java (JDK) | OpenJDK 17 (Temurin 17.0.21.1) | **Mandatory.** Spark is a JVM application; PySpark is a Python wrapper around it |
| PySpark | 4.3.0 | Big Data processing and MLlib |
| Streamlit | 1.42.0 | Dashboard |
| Plotly | 6.0.0 | Charts |
| pandas / pyarrow | 2.2.3 / 19.0.0 | Reading the small precomputed summaries, Parquet support |

---

## 2. Installing Java 17

Spark will not start without a JVM. Check first:

```bash
java -version
```

If that prints a version starting with `17`, skip to §3.

### macOS

The straightforward route is Homebrew's prebuilt cask:

```bash
brew install --cask temurin@17
```

**If Homebrew refuses** (on Intel Macs it no longer ships prebuilt bottles and
will try to compile from source, which needs the full Xcode app), install the
JDK without admin rights by unpacking the official Adoptium archive into your
home folder - this is what was done on the development machine:

```bash
curl -L -o /tmp/jdk17.tar.gz "https://api.adoptium.net/v3/binary/latest/17/ga/mac/x64/jdk/hotspot/normal/eclipse"
```

```bash
mkdir -p ~/Library/Java/JavaVirtualMachines && tar -xzf /tmp/jdk17.tar.gz -C ~/Library/Java/JavaVirtualMachines
```

```bash
ls ~/Library/Java/JavaVirtualMachines
```

The last command prints the JDK folder name (for example `jdk-17.0.21.1+2`).
Put that name into `JAVA_HOME` inside `env.sh` if it differs from what is
already there.

### Linux

```bash
sudo apt install openjdk-17-jdk
```

### Windows

Install "Eclipse Temurin 17 (JDK)" from <https://adoptium.net>, tick *Set
JAVA_HOME*, and use `venv\Scripts\activate` instead of `env.sh`.

---

## 3. Installing Python 3.11 and the project environment

If `python3.11 --version` already works, use it directly in the next step.
Otherwise, a prebuilt interpreter can be installed without admin rights:

```bash
python3 -m pip install --user uv && ~/Library/Python/*/bin/uv python install 3.11
```

Then create the virtual environment and install the dependencies:

```bash
cd /path/to/FraudShield && python3.11 -m venv venv && ./venv/bin/pip install -r requirements.txt
```

(If `python3.11` is not on your PATH, use the full path that `uv python list
--only-installed` prints for CPython 3.11.)

---

## 4. Pointing the project at the dataset

The engineered dataset (557 MB, 16 Spark CSV part files) is not in the git
repository. The project expects it at `outputs/engineered_data`. If the folder
lives somewhere else - a Downloads folder, an external drive - link it:

```bash
ln -sfn /absolute/path/to/engineered_data outputs/engineered_data
```

Or set an environment variable instead, which takes priority:

```bash
export FRAUDSHIELD_ENGINEERED_DATA=/absolute/path/to/engineered_data
```

---

## 5. Loading the environment (every new terminal)

```bash
cd /path/to/FraudShield && source env.sh
```

This sets `JAVA_HOME`, puts the virtual environment on the PATH, and - important
- sets `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` so Spark's driver and its
workers run the *same* interpreter. Without that, Spark can fall back to the
system Python, which has no `pyspark` installed.

Verify:

```bash
python src/test_spark.py
```

Expected output: `Spark Version: 4.3.0` and `Spark Session Created Successfully!`

---

## 6. Running everything

One command runs the whole Member 2 pipeline in the correct order:

```bash
./run_pipeline.sh
```

For a fast debugging run on a 2 % sample (results are *not* for the report):

```bash
./run_pipeline.sh --sample
```

### Or run the stages individually

```bash
python src/convert_to_parquet.py
```

```bash
python src/verify_dataset.py
```

```bash
python src/train_logistic_regression.py
```

```bash
python src/train_random_forest.py --num-trees 100
```

```bash
python src/threshold_optimization.py
```

```bash
python src/model_comparison.py
```

```bash
python src/fraud_alerts.py
```

```bash
python src/prepare_dashboard_data.py
```

```bash
python tests/test_integration.py
```

### Starting the dashboard

```bash
source env.sh && streamlit run dashboard/app.py
```

It opens at <http://localhost:8501>. Stop it with Ctrl-C.

---

## 7. Useful flags

| Flag | Script | Effect |
|---|---|---|
| `--sample 0.02` | both training scripts, `feature_ablation.py` | Train on a 2 % random sample for quick debugging |
| `--no-tune` | both training scripts | Skip the hyperparameter search, fit once |
| `--no-mismatch` | both training scripts | Drop the two balance-mismatch features (ablation) |
| `--num-trees`, `--max-depth` | `train_random_forest.py` | Fix the forest shape instead of searching |
| `--model lr\|rf\|final` | `fraud_alerts.py` | Which saved model to score with |
| `--source test\|validation\|train\|all` | `fraud_alerts.py` | Which transactions to score |
| `--input <path>` | `fraud_alerts.py` | Score an external CSV/Parquet file |

Environment variables: `FRAUDSHIELD_ENGINEERED_DATA`,
`FRAUDSHIELD_ENGINEERED_PARQUET`, `FRAUDSHIELD_DRIVER_MEMORY` (default `5g`),
`FRAUDSHIELD_SHUFFLE_PARTITIONS` (default `16`), `FRAUDSHIELD_SPARK_MASTER`
(default `local[*]`), `FRAUDSHIELD_SPARK_LOGLEVEL` (default `WARN`).

---

## 8. Troubleshooting

**`Unable to locate a Java Runtime` / `JAVA_HOME is not set`**
Java 17 is missing or `env.sh` points at the wrong folder. Re-check §2 and make
sure the folder name in `env.sh` matches what is in
`~/Library/Java/JavaVirtualMachines`.

**`ModuleNotFoundError: No module named 'pyspark'`**
You did not `source env.sh` in this terminal, or the virtual environment was
created with the wrong interpreter.

**`Python in worker has different version than that in driver`**
`PYSPARK_PYTHON` is not set - `source env.sh` handles this.

**`WARN MemoryStore: Not enough space to cache ...` / `Persisting block to disk`**
Harmless on a small machine. Spark is spilling cached data to disk because 8 GB
is not enough to hold 4.46 million feature vectors in memory. Training still
completes; it is just slower. Lower `FRAUDSHIELD_DRIVER_MEMORY` if the machine
starts swapping heavily, or raise it on a bigger machine.

**`java.lang.OutOfMemoryError: Java heap space`**
Reduce the work per task:
`FRAUDSHIELD_DRIVER_MEMORY=6g python src/train_random_forest.py --num-trees 60 --max-depth 8`

**`Engineered dataset not found at ...`**
The symlink or `FRAUDSHIELD_ENGINEERED_DATA` is wrong - see §4.

**Dashboard shows "not generated yet" messages**
The corresponding Spark script has not been run. Each message names the exact
command to run.

**Dashboard shows stale numbers**
The cache keys include each file's modification time, so re-running a script is
enough. If it still looks stale, press `R` in the browser or restart Streamlit.

**`py4j.protocol.Py4JJavaError` with `_SUCCESS` or path errors**
An output directory from a previous interrupted run is half-written. Delete that
specific folder under `outputs/` and re-run the stage.

**Spark's console is very noisy**
`export FRAUDSHIELD_SPARK_LOGLEVEL=ERROR` before running.

---

## 9. What is safe to delete and regenerate

| Path | Regenerate with |
|---|---|
| `outputs/engineered_parquet/` | `python src/convert_to_parquet.py` |
| `models/` | the two training scripts |
| `outputs/model_results/` | training, threshold optimisation, comparison |
| `outputs/fraud_alerts/` | `python src/fraud_alerts.py` |
| `outputs/dashboard_data/` | `python src/prepare_dashboard_data.py` |
| `.spark-tmp/`, `outputs/logs/` | created automatically |

`outputs/engineered_data` (Member 1's dataset) and everything in `src/` that
Member 1 wrote must **not** be deleted - the dataset cannot be regenerated
without the original `dataset/paysim.csv`, which is not part of this repository.
