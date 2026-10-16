#!/bin/bash
# Generic Slurm template for the TabPFN Tecator benchmark
# (benchmarks/run_tabpfn_tecator.py).
#
# Adapt to your cluster:
#   1. Uncomment and set the -p partition line (or delete it to use the
#      cluster's default partition).
#   2. Make sure `python3` resolves to an interpreter where `tabpfn` is
#      installed (module load / conda activate), or edit PYTHON below.
#   3. sbatch hpc/sbatch_tabpfn.sh
#
#SBATCH -J tabpfn-tecator
#SBATCH -t 02:00:00
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -o slurm-%j.out
# #SBATCH -p cpu

set -e
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
# Keep pip's cache off the home filesystem (quota-friendly on shared clusters).
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${TMPDIR:-/tmp}/spectra-screen-pip}"

python3 -m pip install -q -e .
echo "=== env ==="
python3 -c "import torch, sklearn, tabpfn; print('torch', torch.__version__, '| sklearn', sklearn.__version__, '| tabpfn', tabpfn.__version__)"
echo "=== benchmark ==="
python3 -u benchmarks/run_tabpfn_tecator.py
echo "=== done ==="
