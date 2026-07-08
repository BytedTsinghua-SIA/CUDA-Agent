"""用官方 flashinfer-bench CLI 在 Modal B200 上跑 GDN decode。

功能：
  构造只包含 GDN decode 的最小 TraceSet，加入指定 solution，
  然后执行官方形式的 `flashinfer-bench run`。

参数：
  --probe：只打印远端 Python/CUDA/FlashInfer-Bench 环境，不跑 benchmark。
  --solution：选择 ours 或 ako；默认 ours。
  --max-workloads：跑前 N 个 workload；默认 0，表示跑全部 54 个。
  --warmup-runs/--iterations/--num-trials：传给 `flashinfer-bench run`。
  --timeout-seconds：传给 `flashinfer-bench run --timeout`；默认 7200，避免全量评估被 CLI timeout 截断。
  --no-baseline：只跑 CUDA-Agent solution，不跑 FlashInfer baseline。

示例：
  ../../.venv/bin/modal run run_modal_official_cli.py --probe
  ../../.venv/bin/modal run run_modal_official_cli.py --max-workloads 1 --warmup-runs 1 --iterations 1 --num-trials 1
  ../../.venv/bin/modal run run_modal_official_cli.py --solution ako --max-workloads 1 --warmup-runs 1 --iterations 1 --num-trials 1
  ../../.venv/bin/modal run run_modal_official_cli.py --max-workloads 0
"""

from __future__ import annotations

import json
from pathlib import Path
import selectors
import shutil
import subprocess
import time

import modal


SCRIPT_DIR = Path(__file__).resolve().parent
CONTEST_DATA_DIR = SCRIPT_DIR / "contest_data"
KERNEL_PATH = SCRIPT_DIR / "decode" / "kernel.py"
AKO_KERNEL_PATH = SCRIPT_DIR / "decode" / "kernel_ako4x_cuda_bv32_register_resident.py"

REMOTE_ROOT = Path("/root/gdn_decode_official_cli")
REMOTE_INPUT_DATA = REMOTE_ROOT / "input_contest_data"
REMOTE_KERNEL = REMOTE_ROOT / "kernel.py"
REMOTE_AKO_KERNEL = REMOTE_ROOT / "kernel_ako4x_cuda_bv32_register_resident.py"
REMOTE_RUN_DATA = Path("/tmp/gdn_decode_official_cli_dataset")

DEFINITION = "gdn_decode_qk4_v8_d128_k_last"
OP_TYPE = "gdn"
BASELINE_SOLUTION = "flashinfer_wrapper_9b7f1e"
SOLUTION_CONFIGS = {
    "ours": {
        "name": "cuda_agent_gdn_decode",
        "author": "cuda_agent",
        "entry_point": "kernel.py::run",
        "language": "triton",
        "source_path": "kernel.py",
        "remote_kernel": REMOTE_KERNEL,
        "description": "CUDA-Agent GDN decode Triton kernel.",
    },
    "ako": {
        "name": "ako4x_gdn_decode_cuda_bv32",
        "author": "ako4x",
        "entry_point": "kernel_ako4x_cuda_bv32_register_resident.py::run",
        "language": "python",
        "source_path": "kernel_ako4x_cuda_bv32_register_resident.py",
        "remote_kernel": REMOTE_AKO_KERNEL,
        "description": "AKO4X cuda_bv32_register_resident GDN decode kernel.",
    },
}
CLI_EXTRA_ARGS: list[str] = []

MODAL_IMAGE_REGISTRY = "flashinfer/flashinfer-ci-cu132:20260401-2c675fb"
MODAL_PYTHON = "3.12"
MODAL_PACKAGE_PIN = (
    "flashinfer-bench @ https://github.com/flashinfer-ai/flashinfer-bench/"
    "archive/f7b4d8d185625ab2d609233a1a06e99ee18a0c6b.tar.gz"
)
MODAL_EXTRA_PIN = "flashinfer-python==0.6.8"

app = modal.App("cuda-agent-gdn-decode-official-cli")

image = (
    modal.Image.from_registry(MODAL_IMAGE_REGISTRY, add_python=MODAL_PYTHON)
    .env({"CUDA_HOME": "/usr/local/cuda"})
    # 官方 CUDA13.2 base + Modal Python 层补齐 FIB runtime。
    .pip_install("numpy", "cupti-python", "apache-tvm-ffi")
    .pip_install(MODAL_PACKAGE_PIN)
    .pip_install(MODAL_EXTRA_PIN)
    .pip_install("torch-c-dlpack-ext")
    .run_commands(
        "apt-get update && apt-get install -y git clang && pip install wheel && "
        "pip install git+https://github.com/deepseek-ai/DeepGEMM.git@main --no-build-isolation"
    )
    .run_commands(
        "pip uninstall -y nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base "
        "nvidia-cutlass-dsl-libs-core nvidia-cutlass-dsl-libs-cu12 && "
        "pip install 'nvidia-cutlass-dsl[cu13]==4.4.2'"
    )
    .run_commands(
        "FI_DATA=$(python3 -c 'import flashinfer, os; print(os.path.dirname(flashinfer.__file__))')/data && "
        "if [ -f $FI_DATA/cutlass/include/cutlass/cutlass.h ]; then "
        "echo 'CUTLASS headers already present in flashinfer install, skipping patch'; exit 0; fi && "
        "mkdir -p $FI_DATA/cutlass && cd $FI_DATA/cutlass && "
        "git init -q && git remote add origin https://github.com/NVIDIA/cutlass.git && "
        "git fetch --depth 1 -q origin da5e086dab31d63815acafdac9a9c5893b1c69e2 && "
        "git checkout -q FETCH_HEAD && rm -rf $FI_DATA/cutlass/.git"
    )
    .add_local_file(str(KERNEL_PATH), str(REMOTE_KERNEL))
    .add_local_file(str(AKO_KERNEL_PATH), str(REMOTE_AKO_KERNEL))
    .add_local_dir(str(CONTEST_DATA_DIR), str(REMOTE_INPUT_DATA))
)


def _run_streaming(cmd: list[str], cwd: Path | None = None) -> str:
    """运行子进程并流式转发输出；返回完整日志文本。"""
    header = "running: " + " ".join(cmd) + "\n"
    print(header, end="", flush=True)
    output = [header]
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert proc.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    last_output = time.monotonic()
    while proc.poll() is None:
        events = selector.select(timeout=30.0)
        if not events:
            elapsed = int(time.monotonic() - last_output)
            line = f"[heartbeat] still running; no output for {elapsed}s\n"
            print(line, end="", flush=True)
            output.append(line)
            continue
        for key, _ in events:
            line = key.fileobj.readline()
            if not line:
                continue
            last_output = time.monotonic()
            print(line, end="", flush=True)
            output.append(line)
    rest = proc.stdout.read()
    if rest:
        print(rest, end="", flush=True)
        output.append(rest)
    selector.close()
    returncode = proc.wait()
    text = "".join(output)
    if returncode != 0:
        raise RuntimeError(text)
    return text


def _get_solution_config(solution_key: str) -> dict[str, str | Path]:
    """校验并返回指定 solution 的元信息。"""
    if solution_key not in SOLUTION_CONFIGS:
        raise ValueError(f"unknown solution {solution_key!r}; expected one of {sorted(SOLUTION_CONFIGS)}")
    return SOLUTION_CONFIGS[solution_key]


def _write_solution(dataset_dir: Path, solution_key: str) -> Path:
    """把指定 kernel 打包成 flashinfer-bench Solution JSON。"""
    config = _get_solution_config(solution_key)
    solution = {
        "name": config["name"],
        "definition": DEFINITION,
        "author": config["author"],
        "spec": {
            "language": config["language"],
            "target_hardware": ["cuda"],
            "entry_point": config["entry_point"],
            "dependencies": [],
            "destination_passing_style": False,
        },
        "description": config["description"],
        "sources": [{"path": config["source_path"], "content": config["remote_kernel"].read_text()}],
    }
    out_dir = dataset_dir / "solutions" / str(config["author"]) / OP_TYPE / DEFINITION
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{config['name']}.json"
    out_path.write_text(json.dumps(solution, indent=2))
    return out_path


def _prepare_dataset(max_workloads: int | None, solution_key: str) -> Path:
    """复制最小 TraceSet，并按需截断 workload jsonl。"""
    if REMOTE_RUN_DATA.exists():
        shutil.rmtree(REMOTE_RUN_DATA)
    shutil.copytree(REMOTE_INPUT_DATA, REMOTE_RUN_DATA)
    traces_dir = REMOTE_RUN_DATA / "traces"
    if traces_dir.exists():
        shutil.rmtree(traces_dir)
    _write_solution(REMOTE_RUN_DATA, solution_key)

    if max_workloads is not None and max_workloads > 0:
        workload_path = REMOTE_RUN_DATA / "workloads" / OP_TYPE / f"{DEFINITION}.jsonl"
        lines = workload_path.read_text().splitlines()
        workload_path.write_text("\n".join(lines[:max_workloads]) + "\n")

    return REMOTE_RUN_DATA


def _normalize_baseline_author(trace_set) -> None:
    """把官方 baseline author 归一为 flashinfer，便于 get_author_score 对齐。"""
    for sols in trace_set.solutions.values():
        for i, sol in enumerate(sols):
            if sol.name == BASELINE_SOLUTION or (
                sol.author and sol.author.startswith("flashinfer") and sol.author != "flashinfer"
            ):
                sols[i] = sol.model_copy(update={"author": "flashinfer"})
                trace_set._solution_by_name[sol.name] = sols[i]


def _summarize_results(dataset_dir: Path, solution_key: str) -> str:
    """读取官方 trace 结果，输出 baseline/solution latency 和 speedup。"""
    from flashinfer_bench.data import TraceSet

    config = _get_solution_config(solution_key)
    solution_name = str(config["name"])
    author = str(config["author"])
    trace_set = TraceSet.from_path(str(dataset_dir))
    _normalize_baseline_author(trace_set)
    traces = trace_set.traces.get(DEFINITION, [])
    lines = ["\nsummary:"]
    latencies: dict[tuple[str, str], float] = {}
    statuses: dict[tuple[str, str], str] = {}

    for trace in traces:
        if trace.evaluation is None:
            continue
        uuid = trace.workload.uuid
        status = trace.evaluation.status.value
        statuses[(trace.solution, uuid)] = status
        perf = trace.evaluation.performance
        if perf is None:
            lines.append(f"  {trace.solution} {uuid[:8]} status={status}")
            log = trace.evaluation.log.strip()
            if log:
                tail = "\n".join(log.splitlines()[-20:])
                lines.append("    log_tail:\n" + "\n".join(f"      {line}" for line in tail.splitlines()))
            continue
        latencies[(trace.solution, uuid)] = perf.latency_ms
        lines.append(
            f"  {trace.solution} {uuid[:8]} status={status} "
            f"latency_ms={perf.latency_ms:.6f} ref_speedup={perf.speedup_factor:.6f}"
        )

    common = sorted(
        uuid
        for solution, uuid in latencies
        if solution == solution_name and (BASELINE_SOLUTION, uuid) in latencies
    )
    if common:
        speedups = []
        for uuid in common:
            base = latencies[(BASELINE_SOLUTION, uuid)]
            solution_ms = latencies[(solution_name, uuid)]
            speedup = base / solution_ms
            speedups.append(speedup)
            lines.append(
                f"  official_pair {uuid[:8]} baseline_ms={base:.6f} "
                f"solution_ms={solution_ms:.6f} speedup={speedup:.6f} "
                f"baseline_status={statuses.get((BASELINE_SOLUTION, uuid))} "
                f"solution_status={statuses.get((solution_name, uuid))}"
            )
        lines.append(f"  official_pair_avg_speedup={sum(speedups) / len(speedups):.6f}")

    score = trace_set.get_author_score(
        author,
        baseline_author="flashinfer",
        definition_name=DEFINITION,
    )
    if score is None:
        lines.append("  get_author_score=None")
    else:
        lines.append(
            "  get_author_score "
            f"avg_speedup={score.avg_speedup:.6f} "
            f"definitions={score.definitions} workloads={score.workloads} "
            f"success_rate={score.success_rate:.6f} win_rate={score.win_rate:.6f}"
        )
    return "\n".join(lines) + "\n"


@app.function(image=image, gpu="B200:1", timeout=900)
def probe_env() -> str:
    """返回远端官方 CLI runner 的环境信息。"""
    code = r"""
import sys
from importlib.metadata import PackageNotFoundError, version

def maybe_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"

print("python", sys.version.split()[0])
print("torch", maybe_version("torch"))
print("triton", maybe_version("triton"))
print("flashinfer-bench", maybe_version("flashinfer-bench"))
print("flashinfer-python", maybe_version("flashinfer-python"))
print("cupti-python", maybe_version("cupti-python"))
print("deep_gemm", maybe_version("deep-gemm"))
print("apache-tvm-ffi", maybe_version("apache-tvm-ffi"))
print("nvidia-cutlass-dsl", maybe_version("nvidia-cutlass-dsl"))
print("nvidia-cutlass-dsl-libs-base", maybe_version("nvidia-cutlass-dsl-libs-base"))
print("nvidia-cutlass-dsl-libs-core", maybe_version("nvidia-cutlass-dsl-libs-core"))
print("nvidia-cutlass-dsl-libs-cu12", maybe_version("nvidia-cutlass-dsl-libs-cu12"))
print("tilelang", maybe_version("tilelang"))
import torch
print("torch_cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("which_flashinfer_bench", __import__("shutil").which("flashinfer-bench"))
import traceback
import flashinfer
print("flashinfer_file", flashinfer.__file__)
try:
    import flashinfer.gdn_decode as gd
    print("gdn_decode_file", gd.__file__)
    print("_PRETRANSPOSE_AVAILABLE", getattr(gd, "_PRETRANSPOSE_AVAILABLE", None))
    print("run_pretranspose_decode", getattr(gd, "run_pretranspose_decode", None))
except Exception:
    print("flashinfer.gdn_decode import failed")
    traceback.print_exc()
try:
    import flashinfer.gdn_kernels.gdn_decode_pretranspose as pre
    print("gdn_decode_pretranspose_file", pre.__file__)
    print("pre.run_pretranspose_decode", getattr(pre, "run_pretranspose_decode", None))
except Exception:
    print("gdn_decode_pretranspose import failed")
    traceback.print_exc()
"""
    proc = subprocess.run(
        ["python3", "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    cli = subprocess.run(
        ["flashinfer-bench", "run", "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return f"probe_returncode={proc.returncode}\n{proc.stdout}\ncli_returncode={cli.returncode}\n{cli.stdout}"


@app.function(image=image, gpu="B200:1", timeout=7200)
def run_official_cli(
    solution: str,
    max_workloads: int,
    warmup_runs: int,
    iterations: int,
    num_trials: int,
    timeout_seconds: int,
    include_baseline: bool,
) -> str:
    """执行官方 flashinfer-bench CLI，并返回完整日志和结果摘要。"""
    config = _get_solution_config(solution)
    solution_name = str(config["name"])
    dataset_dir = _prepare_dataset(None if max_workloads <= 0 else max_workloads, solution)
    solutions = [solution_name]
    if include_baseline:
        solutions.append(BASELINE_SOLUTION)

    cmd = [
        "flashinfer-bench",
        "run",
        "--local",
        str(dataset_dir),
        "--definitions",
        DEFINITION,
        "--solutions",
        *solutions,
        "--save-results",
        "--use-isolated-runner",
        "--log-level",
        "INFO",
        "--resume",
        "--timeout",
        str(timeout_seconds),
        "--warmup-runs",
        str(warmup_runs),
        "--iterations",
        str(iterations),
        "--num-trials",
        str(num_trials),
        *CLI_EXTRA_ARGS,
    ]
    output = _run_streaming(cmd)
    summary = _summarize_results(dataset_dir, solution)
    print(summary, end="", flush=True)
    return output + summary


@app.local_entrypoint()
def main(
    probe: bool = False,
    solution: str = "ours",
    max_workloads: int = 0,
    warmup_runs: int = 10,
    iterations: int = 50,
    num_trials: int = 3,
    timeout_seconds: int = 7200,
    no_baseline: bool = False,
):
    """Modal local entrypoint。"""
    if probe:
        print(probe_env.remote(), end="")
        return
    run_official_cli.remote(
        solution=solution,
        max_workloads=max_workloads,
        warmup_runs=warmup_runs,
        iterations=iterations,
        num_trials=num_trials,
        timeout_seconds=timeout_seconds,
        include_baseline=not no_baseline,
    )
