"""Temporary utility to inspect whether trained residual models look collapsed.

This script is meant for quick sanity checks during development.
It summarizes:
- static residual layer scales
- prompt-conditioned MLP parameter statistics
- prompt-conditioned predicted scale variability on a sample split
- how close the prompt-conditioned average scale vector is to the static vector
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from _eval_utils import create_runtime, load_prompt_conditioned_model_for_eval
from _training_utils import create_train_dataloader, move_batch_to_device


DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_BATCHES = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick inspection script for residual-scale collapse."
    )
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--static-checkpoint",
        default=str(repo_root / "artifact" / "checkpoints" / "static_residual" / "trained_model.pt"),
        help="Path to the static residual checkpoint.",
    )
    parser.add_argument(
        "--prompt-checkpoint",
        default=str(repo_root / "artifact" / "checkpoints" / "prompt_conditioned" / "trained_model.pt"),
        help="Path to the prompt-conditioned checkpoint.",
    )
    parser.add_argument(
        "--data-path",
        default=str(repo_root / "artifact" / "data" / "prepared" / "validation.arrow"),
        help="Prepared Arrow split used for the scale-variability probe.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Probe batch size. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=DEFAULT_NUM_BATCHES,
        help=f"How many batches to inspect. Default: {DEFAULT_NUM_BATCHES}",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional path to save the summary as JSON.",
    )
    return parser.parse_args()


def summarize_tensor(tensor: torch.Tensor) -> dict:
    tensor = tensor.float()
    return {
        "shape": list(tensor.shape),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def inspect_static_checkpoint(checkpoint_path: str | Path) -> tuple[torch.Tensor, dict]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    raw_scales = checkpoint["trainable_state_dict"]["raw_layer_scales"].float()
    actual_scales = 1.0 + torch.tanh(raw_scales)
    summary = {
        "raw_scales": summarize_tensor(raw_scales),
        "actual_scales": summarize_tensor(actual_scales),
        "first_10_actual_scales": [float(x) for x in actual_scales[:10]],
    }
    return actual_scales, summary


def inspect_prompt_parameters(checkpoint_path: str | Path) -> dict:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["trainable_state_dict"]
    return {
        name: summarize_tensor(tensor)
        for name, tensor in state_dict.items()
    }


def inspect_prompt_scale_variability(
    checkpoint_path: str | Path,
    data_path: str | Path,
    batch_size: int,
    num_batches: int,
) -> dict:
    device, model_dtype = create_runtime()
    model, _ = load_prompt_conditioned_model_for_eval(
        checkpoint_path=checkpoint_path,
        model_dir=None,
        device=device,
        model_dtype=model_dtype,
        attn_implementation=None,
    )
    dataloader = create_train_dataloader(
        arrow_path=data_path,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    all_scales: list[torch.Tensor] = []
    all_sources: list[str] = []

    progress_bar = tqdm(
        dataloader,
        total=num_batches,
        desc="Inspect prompt scales",
        leave=False,
    )
    with torch.inference_mode():
        for batch_index, batch in enumerate(progress_bar):
            if batch_index >= num_batches:
                break

            batch = move_batch_to_device(batch, device)
            scales = model.get_layer_scales(
                prompt_input_ids=batch["prompt_input_ids"],
                example_id=batch["example_id"],
            )
            scales = scales.detach().cpu().float()
            all_scales.append(scales)
            all_sources.extend(batch["source"])

            progress_bar.set_postfix(
                batch=batch_index + 1,
                examples=sum(x.shape[0] for x in all_scales),
            )
    progress_bar.close()

    all_scales_tensor = torch.cat(all_scales, dim=0)
    layer_mean = all_scales_tensor.mean(dim=0)
    layer_std = all_scales_tensor.std(dim=0)
    global_mean = layer_mean.unsqueeze(0)
    dist_to_global_mean = torch.norm(all_scales_tensor - global_mean, dim=1)

    source_to_indices: dict[str, list[int]] = {}
    for index, source in enumerate(all_sources):
        source_to_indices.setdefault(source, []).append(index)

    by_source = {}
    for source, indices in source_to_indices.items():
        source_scales = all_scales_tensor[indices]
        by_source[source] = {
            "num_examples": len(indices),
            "overall_mean": float(source_scales.mean()),
            "overall_std": float(source_scales.std()),
            "centroid_first_10": [float(x) for x in source_scales.mean(dim=0)[:10]],
        }

    centroid_distance = None
    if set(source_to_indices) >= {"wikitext", "medical"}:
        w_centroid = all_scales_tensor[source_to_indices["wikitext"]].mean(dim=0)
        m_centroid = all_scales_tensor[source_to_indices["medical"]].mean(dim=0)
        centroid_distance = float(torch.norm(w_centroid - m_centroid))

    return {
        "num_examples_checked": int(all_scales_tensor.shape[0]),
        "overall_mean": float(all_scales_tensor.mean()),
        "overall_std": float(all_scales_tensor.std()),
        "layer_mean_first_10": [float(x) for x in layer_mean[:10]],
        "layer_std_first_10": [float(x) for x in layer_std[:10]],
        "layer_std_mean": float(layer_std.mean()),
        "layer_std_min": float(layer_std.min()),
        "layer_std_max": float(layer_std.max()),
        "dist_to_global_mean_mean": float(dist_to_global_mean.mean()),
        "dist_to_global_mean_std": float(dist_to_global_mean.std()),
        "dist_to_global_mean_min": float(dist_to_global_mean.min()),
        "dist_to_global_mean_max": float(dist_to_global_mean.max()),
        "by_source": by_source,
        "wikitext_medical_centroid_distance": centroid_distance,
        "mean_scale_vector": layer_mean,
    }


def compare_prompt_mean_to_static(
    prompt_scale_summary: dict,
    static_scales: torch.Tensor,
) -> dict:
    prompt_mean = prompt_scale_summary["mean_scale_vector"]
    return {
        "l2_distance": float(torch.norm(prompt_mean - static_scales)),
        "mean_abs_diff": float((prompt_mean - static_scales).abs().mean()),
    }


def make_json_safe(summary: dict) -> dict:
    safe_summary = dict(summary)
    variability = dict(safe_summary["prompt_scale_variability"])
    variability.pop("mean_scale_vector", None)
    safe_summary["prompt_scale_variability"] = variability
    return safe_summary


def main() -> None:
    args = parse_args()

    print("[inspect] static checkpoint")
    static_scales, static_summary = inspect_static_checkpoint(args.static_checkpoint)

    print("[inspect] prompt-conditioned parameters")
    prompt_param_summary = inspect_prompt_parameters(args.prompt_checkpoint)

    print("[inspect] prompt-conditioned variability")
    prompt_scale_summary = inspect_prompt_scale_variability(
        checkpoint_path=args.prompt_checkpoint,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
    )

    comparison = compare_prompt_mean_to_static(
        prompt_scale_summary=prompt_scale_summary,
        static_scales=static_scales,
    )

    full_summary = {
        "static": static_summary,
        "prompt_parameters": prompt_param_summary,
        "prompt_scale_variability": prompt_scale_summary,
        "prompt_mean_vs_static": comparison,
    }
    json_safe_summary = make_json_safe(full_summary)

    print("\n=== Quick Summary ===")
    print(
        "static actual scale std:",
        round(json_safe_summary["static"]["actual_scales"]["std"], 6),
    )
    print(
        "prompt layer std mean/min/max:",
        round(json_safe_summary["prompt_scale_variability"]["layer_std_mean"], 6),
        round(json_safe_summary["prompt_scale_variability"]["layer_std_min"], 6),
        round(json_safe_summary["prompt_scale_variability"]["layer_std_max"], 6),
    )
    print(
        "prompt mean vs static mean_abs_diff:",
        round(json_safe_summary["prompt_mean_vs_static"]["mean_abs_diff"], 6),
    )
    print(
        "wikitext vs medical centroid distance:",
        json_safe_summary["prompt_scale_variability"]["wikitext_medical_centroid_distance"],
    )

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(json_safe_summary, indent=2), encoding="utf-8")
        print(f"[done] saved summary to: {output_path}")


if __name__ == "__main__":
    main()
