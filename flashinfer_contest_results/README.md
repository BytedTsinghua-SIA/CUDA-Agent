# CUDA-Agent FlashInfer Contest Results

This directory contains the CUDA-Agent kernels with corresponding FlashInfer-Bench Modal
runners, and agent trajectories used to reproduce the published MLSys-2026
FlashInfer contest results for:

- `gdn/decode`: GDN decode, 54 official workloads
- `moe/fp8`: MoE FP8 block-scale, 19 official workloads
- `dsa/sparse_mla`: DSA sparse MLA, 23 official workloads

The local runtime driver only needs the Modal client. The setup script also
installs `huggingface_hub` to download the official contest data. CUDA, PyTorch,
Triton, FlashInfer-Bench, FlashInfer-Python, and DeepGEMM are installed inside
the Modal image built by each runner.

`contest_data` is intentionally not tracked by git because it contains large
workload tensors. `setup_env.sh` downloads and materializes the corresponding
minimal TraceSet at:

- `gdn/decode/contest_data`
- `moe/fp8/contest_data`
- `dsa/topk_indexer/contest_data`
- `dsa/sparse_mla/contest_data`

## Setup

Use Python 3.12 and run the setup script:

```bash
cd flashinfer_contest_results
./setup_env.sh
```

The script creates `.venv`, installs the local Python dependencies, downloads
the minimal GDN decode, MoE FP8, and DSA data from the official Hugging Face
dataset `flashinfer-ai/mlsys26-contest`, and re-materializes `contest_data`
from that downloaded snapshot. 

Authenticate Modal before running the benchmarks:

```bash
.venv/bin/modal setup
```

The scripts build a CUDA 13.2 FlashInfer-Bench environment inside Modal and
request an NVIDIA B200 GPU. Local GPU access is not required for the driver
process. The runner scripts use `flashinfer_contest_results/.venv/bin/modal`
by default. 

## Smoke Tests To Verify The Environment is Set Up Correctly

Run one workload with short timing settings:

```bash
cd gdn/decode
./run_modal_official_cli.sh --max-workloads 1 --warmup-runs 1 --iterations 10 --num-trials 1
```

```bash
cd ../../moe/fp8
./run_modal_official_cli.sh --max-workloads 1 --warmup-runs 1 --iterations 10 --num-trials 1
```

```bash
cd ../sparse_mla
./run_modal_official_cli.sh --max-workloads 1 --warmup-runs 1 --iterations 10 --num-trials 1
```

## Full Evaluation

Use all official workloads. Timing follows the FlashInfer starter kit
`flashinfer-bench run` defaults: `warmup_runs=10`, `iterations=50`, and
`num_trials=3`. Timeout is intentionally not aligned to the starter kit's 300s
limit so that full runs are not
truncated.

```bash
cd flashinfer_contest_results/gdn/decode
./run_modal_official_cli.sh
```

```bash
cd flashinfer_contest_results/moe/fp8
./run_modal_official_cli.sh
```

```bash
cd flashinfer_contest_results/dsa/sparse_mla
./run_modal_official_cli.sh
```

The scripts print both per-workload `official_pair` speedups and
`get_author_score avg_speedup`. FlashInfer-Bench's official score is the
arithmetic mean of per-workload speedups. The MoE runner includes a live
FlashInfer baseline by default.

The expected full-run arithmetic mean is approximately:

- GDN decode: `1.30x`
- MoE FP8: `1.46x`
- DSA sparse MLA: `16.55x`

The bar chart uses these full-run scores; AKO4X artifact values were rerun
under the same timing setting. The running logs are available in the `logs/` directory.

## Files List

- `setup_env.sh`: creates the local virtualenv and invokes `setup_env.py`.
- `setup_env.py`: downloads the official minimal contest data and rebuilds
  `contest_data` from the downloaded snapshot.
- `gdn/decode/decode/kernel.py`: CUDA-Agent GDN decode kernel.
- `moe/fp8/kernel.py`: CUDA-Agent MoE FP8 kernel.
- `dsa/topk_indexer/kernel.py`: CUDA-Agent DSA top-k indexer kernel.
- `dsa/sparse_mla/kernel.py`: CUDA-Agent DSA sparse MLA kernel.
- `*/run_modal_official_cli.py`: self-contained Modal runners that package the
  local kernel and execute official FlashInfer-Bench.
- `*/contest_data`: minimal contest TraceSet data needed by each runner.
- `*/agent_traj.txt`: agent trajectory.
