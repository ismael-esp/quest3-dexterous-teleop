# Sourced by the run_* scripts: activate the conda env that has Isaac Lab 2.3 (+ dex_retargeting, pyzmq).
# ISAACLAB_ENV = env name (default env_isaaclab); CONDA_BASE = conda install dir (default: `conda info --base`).
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")}"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${ISAACLAB_ENV:-env_isaaclab}"
