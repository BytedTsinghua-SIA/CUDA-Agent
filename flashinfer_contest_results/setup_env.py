# 功能：从官方 Hugging Face dataset 下载并重建 FlashInfer contest result 复现数据。
# 参数：
#   FORCE=1        强制重新从 Hugging Face 拉取缓存文件。
#   HF_TOKEN=...   如 Hugging Face 下载需要鉴权，可传入 token。
# 示例：
#   cd flashinfer_contest_results
#   .venv/bin/python setup_env.py
#   FORCE=1 .venv/bin/python setup_env.py

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parent
REPO_ID = "flashinfer-ai/mlsys26-contest"
FORCE = os.environ.get("FORCE") == "1"

TASKS = [
    {
        "op_type": "gdn",
        "definition": "gdn_decode_qk4_v8_d128_k_last",
        "target": ROOT / "gdn" / "decode" / "contest_data",
        "baseline_name": "flashinfer_wrapper_9b7f1e",
    },
    {
        "op_type": "moe",
        "definition": "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
        "target": ROOT / "moe" / "fp8" / "contest_data",
        "baseline_name": "flashinfer_wrapper_9sdjf3",
    },
    {
        "op_type": "dsa_paged",
        "baseline_op_type": "dsa",
        "definition": "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64",
        "target": ROOT / "dsa" / "topk_indexer" / "contest_data",
        "baseline_name": "flashinfer_deepgemm_wrapper_2ba145",
    },
    {
        "op_type": "dsa_paged",
        "baseline_op_type": "dsa",
        "definition": "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64",
        "target": ROOT / "dsa" / "sparse_mla" / "contest_data",
        "baseline_name": "flashinfer_wrapper_5af199",
    },
]

GDN_BASELINE_SOURCE = """
import math
import torch
from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose


def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    else:
        scale = float(scale)
    if scale == 0.0:
        scale = 1.0 / math.sqrt(q.shape[-1])

    B, T, num_v_heads, head_size = v.shape
    output = torch.empty(B, T, num_v_heads, head_size, dtype=q.dtype, device=q.device)

    out, new_state = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=scale,
        output=output,
        use_qk_l2norm=False,
    )

    return out, new_state
""".strip() + "\n"

MOE_BASELINE_SOURCE = """
import torch
from flashinfer.fused_moe import trtllm_fp8_block_scale_moe


NUM_EXPERTS_GLOBAL = 256
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
BLOCK_SIZE = 128


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    seq_len, num_experts = routing_logits.shape
    local_num_experts = gemm1_weights.shape[0]

    assert num_experts == NUM_EXPERTS_GLOBAL
    assert hidden_states.shape == (seq_len, HIDDEN_SIZE)
    assert hidden_states_scale.shape == (HIDDEN_SIZE // BLOCK_SIZE, seq_len)
    assert gemm1_weights.shape == (local_num_experts, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
    assert gemm1_weights_scale.shape == (
        local_num_experts,
        (2 * INTERMEDIATE_SIZE) // BLOCK_SIZE,
        HIDDEN_SIZE // BLOCK_SIZE,
    )
    assert gemm2_weights.shape == (local_num_experts, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    assert gemm2_weights_scale.shape == (
        local_num_experts,
        HIDDEN_SIZE // BLOCK_SIZE,
        INTERMEDIATE_SIZE // BLOCK_SIZE,
    )
    assert routing_bias is None or routing_bias.shape[-1] == NUM_EXPERTS_GLOBAL

    if isinstance(local_expert_offset, torch.Tensor):
        local_expert_offset = int(local_expert_offset.item())
    else:
        local_expert_offset = int(local_expert_offset)

    if isinstance(routed_scaling_factor, torch.Tensor):
        routed_scaling_factor = float(routed_scaling_factor.item())
    else:
        routed_scaling_factor = float(routed_scaling_factor)

    return trtllm_fp8_block_scale_moe(
        routing_logits.to(torch.float32).contiguous(),
        None if routing_bias is None else routing_bias.contiguous(),
        hidden_states.contiguous(),
        hidden_states_scale.to(torch.float32).contiguous(),
        gemm1_weights.contiguous(),
        gemm1_weights_scale.to(torch.float32).contiguous(),
        gemm2_weights.contiguous(),
        gemm2_weights_scale.to(torch.float32).contiguous(),
        NUM_EXPERTS_GLOBAL,
        TOP_K,
        N_GROUP,
        TOPK_GROUP,
        INTERMEDIATE_SIZE,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type=2,
        use_shuffled_weight=False,
    )
""".strip() + "\n"


def sanitize_no_proxy_env() -> None:
    """清理 httpx 无法解析的 IPv6/CIDR no_proxy 项，保留普通域名和 IPv4 项。"""
    for key in ("NO_PROXY", "no_proxy"):
        value = os.environ.get(key)
        if not value:
            continue
        parts = [part.strip() for part in value.split(",") if part.strip()]
        cleaned_parts = [part for part in parts if ":" not in part and "/" not in part]
        cleaned = ",".join(cleaned_parts)
        if cleaned != value:
            print(f"[setup] sanitized {key} for huggingface_hub/httpx")
            os.environ[key] = cleaned


def copy_tree(src: Path, dst: Path) -> None:
    """复制目录树；输入为源目录和目标目录，目标存在时先删除。"""
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_file(src: Path, dst: Path) -> None:
    """复制单个文件；输入为源文件和目标文件，自动创建父目录。"""
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_baseline(path: Path, task: dict[str, object]) -> None:
    """写入 FlashInfer baseline solution；输入为目标 JSON 路径和任务配置。"""
    op_type = str(task["op_type"])
    definition = str(task["definition"])
    baseline_name = str(task["baseline_name"])
    if op_type == "gdn":
        target_hardware = ["NVIDIA H20", "NVIDIA H100", "NVIDIA H200"]
        source = GDN_BASELINE_SOURCE
        description = (
            "Solution using FlashInfer gated_delta_rule_decode_pretranspose for "
            "GDN single-token decode (qk4_v8_d128, k-last state layout)."
        )
    elif op_type == "moe":
        target_hardware = ["NVIDIA B200"]
        source = MOE_BASELINE_SOURCE
        description = "Solution using flashinfer.fused_moe.trtllm_fp8_block_scale_moe."
    else:
        raise RuntimeError(f"No baseline solution fallback for {op_type}/{definition}")

    baseline = {
        "name": baseline_name,
        "definition": definition,
        "author": "flashinfer",
        "spec": {
            "language": "python",
            "target_hardware": target_hardware,
            "entry_point": "main.py::run",
            "dependencies": ["flashinfer"],
            "destination_passing_style": False,
        },
        "sources": [{"path": "main.py", "content": source}],
        "description": description,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2) + "\n")


def prepare_task(task: dict[str, object], snapshot: Path) -> None:
    """从下载 snapshot 重建单个任务的 contest_data；输入为任务配置和 snapshot 路径。"""
    op_type = str(task["op_type"])
    definition = str(task["definition"])
    target = Path(task["target"])

    if target.exists():
        print(f"[setup] replacing {target.relative_to(ROOT)} from downloaded snapshot")
        shutil.rmtree(target)

    print(f"[setup] preparing {target.relative_to(ROOT)}")
    copy_tree(
        snapshot / "blob" / "workloads" / op_type / definition,
        target / "blob" / "workloads" / op_type / definition,
    )
    copy_file(
        snapshot / "workloads" / op_type / f"{definition}.jsonl",
        target / "workloads" / op_type / f"{definition}.jsonl",
    )
    copy_file(
        snapshot / "definitions" / op_type / f"{definition}.json",
        target / "definitions" / op_type / f"{definition}.json",
    )

    baseline_name = str(task["baseline_name"])
    baseline_op_type = str(task.get("baseline_op_type", op_type))
    baseline_path = (
        target / "solutions" / "baseline" / baseline_op_type / definition / f"{baseline_name}.json"
    )
    snapshot_baseline = (
        snapshot
        / "solutions"
        / "baseline"
        / baseline_op_type
        / definition
        / f"{baseline_name}.json"
    )
    if snapshot_baseline.exists():
        copy_file(snapshot_baseline, baseline_path)
    else:
        write_baseline(baseline_path, task)


def build_allow_patterns() -> list[str]:
    """生成 Hugging Face 最小下载白名单；输出 snapshot_download 的 allow_patterns。"""
    allow_patterns: list[str] = []
    for task in TASKS:
        op_type = str(task["op_type"])
        definition = str(task["definition"])
        baseline_name = str(task["baseline_name"])
        baseline_op_type = str(task.get("baseline_op_type", op_type))
        allow_patterns.extend(
            [
                f"blob/workloads/{op_type}/{definition}/*",
                f"workloads/{op_type}/{definition}.jsonl",
                f"definitions/{op_type}/{definition}.json",
                f"solutions/baseline/{baseline_op_type}/{definition}/{baseline_name}.json",
            ]
        )
    return allow_patterns


def setup_hf_cache() -> Path:
    """准备 Hugging Face 缓存目录；输出 snapshot_download 使用的 cache_dir。"""
    cache_dir = ROOT / ".hf_cache"
    hf_home = ROOT / ".hf_home"
    xet_cache = ROOT / ".hf_xet"
    if FORCE and cache_dir.exists():
        print(f"[setup] removing Hugging Face cache {cache_dir.relative_to(ROOT)}")
        shutil.rmtree(cache_dir)
    for path in (cache_dir, hf_home, xet_cache):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir))
    os.environ.setdefault("HF_XET_CACHE", str(xet_cache))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    return cache_dir


def download_snapshot(cache_dir: Path) -> Path:
    """下载官方 dataset 的最小 snapshot；输入为缓存目录，输出本地 snapshot 路径。"""
    sanitize_no_proxy_env()
    print(f"[setup] downloading minimal contest data from {REPO_ID}")
    return Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=build_allow_patterns(),
            cache_dir=str(cache_dir),
            force_download=FORCE,
            local_files_only=False,
            token=os.environ.get("HF_TOKEN"),
        )
    )


def main() -> None:
    """执行完整 setup：下载官方数据，并用下载结果重建所有 contest_data 目录。"""
    cache_dir = setup_hf_cache()
    snapshot = download_snapshot(cache_dir)
    for task in TASKS:
        prepare_task(task, snapshot)
    print("[setup] done")


if __name__ == "__main__":
    main()
