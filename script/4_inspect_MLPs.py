"""Inspect prompt-conditioned MLP outputs on sampled validation/test subsets.

This script samples small subsets from validation/test splits, extracts the
raw MLP outputs from the two prompt-conditioned models, and computes simple
collapse-oriented summary statistics:
1. collapse cosine over all sampled examples
2. domain-separation cosine gap between WikiText and medical outputs
3. per-layer variance of the raw outputs
4. per-layer domain-mean difference

The script writes only JSON summaries. Plotting is intentionally left for stage 5.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from _eval_utils import (
    create_runtime,
    load_prompt_conditioned_reaggregation_model_for_eval,
    load_prompt_conditioned_write_strength_model_for_eval,
)
from _residual_modules import PromptConditionedResidualStreamReaggregationModel
from _training_utils import (
    PreparedArrowDataset,
    collate_prepared_batch,
    move_batch_to_device,
)


DEFAULT_SAMPLE_SIZE = 256
DEFAULT_BATCH_SIZE = 16
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect prompt-conditioned MLP outputs on sampled subsets."
    )
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--write-strength-checkpoint",
        default=str(
            repo_root
            / "artifact"
            / "checkpoints"
            / "prompt_conditioned_write_strength"
            / "trained_model.pt"
        ),
        help="Path to the prompt-conditioned write-strength checkpoint.",
    )
    parser.add_argument(
        "--reaggregation-checkpoint",
        default=str(
            repo_root
            / "artifact"
            / "checkpoints"
            / "prompt_conditioned_residual_stream_reaggregation"
            / "trained_model.pt"
        ),
        help="Path to the prompt-conditioned re-aggregation checkpoint.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional override for the frozen base model directory.",
    )
    parser.add_argument(
        "--validation-data",
        default=str(repo_root / "artifact" / "data" / "prepared" / "validation.arrow"),
        help="Path to the prepared validation Arrow file.",
    )
    parser.add_argument(
        "--test-data",
        default=str(repo_root / "artifact" / "data" / "prepared" / "test.arrow"),
        help="Path to the prepared test Arrow file.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Maximum number of examples to sample per split. Default: {DEFAULT_SAMPLE_SIZE}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for MLP-output inspection. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for deterministic sampling. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional override for the attention backend used by the base model.",
    )
    parser.add_argument(
        "--output-path",
        default=str(repo_root / "result" / "analysis" / "mlp_inspection.json"),
        help="Path to save the JSON summary.",
    )
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sample_indices(dataset_size: int, sample_size: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    final_size = min(sample_size, dataset_size)
    if final_size >= dataset_size:
        return list(range(dataset_size))
    return sorted(rng.sample(range(dataset_size), final_size))


def create_subset_dataloader(
    arrow_path: str | Path,
    sample_size: int,
    batch_size: int,
    seed: int,
    pin_memory: bool,
) -> tuple[DataLoader, list[int]]:
    dataset = PreparedArrowDataset(arrow_path=arrow_path)
    indices = sample_indices(
        dataset_size=len(dataset),
        sample_size=sample_size,
        seed=seed,
    )
    subset = Subset(dataset, indices)
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_prepared_batch,
        pin_memory=pin_memory,
    )
    return dataloader, indices


def extract_raw_write_strength_outputs(
    model,
    batch: dict,
) -> torch.Tensor:
    prompt_representation = model.get_prompt_representation(
        prompt_input_ids=batch["prompt_input_ids"],
        example_id=batch["example_id"],
    )
    return model.conditioner(prompt_representation)


def extract_raw_reaggregation_outputs(
    model,
    batch: dict,
) -> torch.Tensor:
    prompt_representation = model.get_prompt_representation(
        prompt_input_ids=batch["prompt_input_ids"],
        example_id=batch["example_id"],
    )
    return model.conditioner(prompt_representation)


def collect_sampled_outputs(
    model,
    dataloader: DataLoader,
    device: torch.device,
    split_name: str,
    model_label: str,
) -> tuple[torch.Tensor, list[str], list[str]]:
    output_batches: list[torch.Tensor] = []
    sources: list[str] = []
    example_ids: list[str] = []

    progress_bar = tqdm(
        dataloader,
        desc=f"Inspect {model_label} {split_name}",
        leave=False,
    )
    with torch.inference_mode():
        for batch in progress_bar:
            batch = move_batch_to_device(batch, device=device)
            if isinstance(model, PromptConditionedResidualStreamReaggregationModel):
                raw_outputs = extract_raw_reaggregation_outputs(model=model, batch=batch)
            else:
                raw_outputs = extract_raw_write_strength_outputs(model=model, batch=batch)

            output_batches.append(raw_outputs.detach().float().cpu())
            sources.extend(batch["source"])
            example_ids.extend(batch["example_id"])

    progress_bar.close()
    return torch.cat(output_batches, dim=0), sources, example_ids


def compute_mean_pairwise_cosine_similarity(vectors: torch.Tensor) -> float | None:
    num_examples = vectors.shape[0]
    if num_examples < 2:
        return None

    normalized_vectors = F.normalize(vectors, p=2, dim=1, eps=1e-12)
    similarity_matrix = normalized_vectors @ normalized_vectors.T
    upper_triangle_indices = torch.triu_indices(
        row=num_examples,
        col=num_examples,
        offset=1,
    )
    pairwise_values = similarity_matrix[
        upper_triangle_indices[0],
        upper_triangle_indices[1],
    ]
    return float(pairwise_values.mean().item())


def compute_cross_cosine_similarity(
    left_vectors: torch.Tensor,
    right_vectors: torch.Tensor,
) -> float | None:
    if left_vectors.numel() == 0 or right_vectors.numel() == 0:
        return None

    left_normalized = F.normalize(left_vectors, p=2, dim=1, eps=1e-12)
    right_normalized = F.normalize(right_vectors, p=2, dim=1, eps=1e-12)
    similarity_matrix = left_normalized @ right_normalized.T
    return float(similarity_matrix.mean().item())


def summarize_write_strength_per_layer_variance(vectors: torch.Tensor) -> tuple[list[float], float]:
    per_layer_variance = vectors.var(dim=0, unbiased=False)
    return per_layer_variance.tolist(), float(per_layer_variance.mean().item())


def summarize_reaggregation_per_layer_variance(
    vectors: torch.Tensor,
    num_layers: int,
) -> tuple[list[float], float]:
    per_dimension_variance = vectors.var(dim=0, unbiased=False)
    per_layer_variance: list[float] = []
    start_index = 0
    for layer_index in range(num_layers):
        num_valid_entries = layer_index + 1
        layer_slice = per_dimension_variance[start_index : start_index + num_valid_entries]
        per_layer_variance.append(float(layer_slice.mean().item()))
        start_index += num_valid_entries

    return per_layer_variance, float(sum(per_layer_variance) / max(len(per_layer_variance), 1))


def summarize_write_strength_domain_mean_difference(
    wiki_vectors: torch.Tensor,
    medical_vectors: torch.Tensor,
) -> tuple[list[float], float | None]:
    if wiki_vectors.numel() == 0 or medical_vectors.numel() == 0:
        return [], None

    per_layer_difference = (wiki_vectors.mean(dim=0) - medical_vectors.mean(dim=0)).abs()
    return per_layer_difference.tolist(), float(per_layer_difference.mean().item())


def summarize_reaggregation_domain_mean_difference(
    wiki_vectors: torch.Tensor,
    medical_vectors: torch.Tensor,
    num_layers: int,
) -> tuple[list[float], float | None]:
    if wiki_vectors.numel() == 0 or medical_vectors.numel() == 0:
        return [], None

    per_dimension_difference = (wiki_vectors.mean(dim=0) - medical_vectors.mean(dim=0)).abs()
    per_layer_difference: list[float] = []
    start_index = 0
    for layer_index in range(num_layers):
        num_valid_entries = layer_index + 1
        layer_slice = per_dimension_difference[start_index : start_index + num_valid_entries]
        per_layer_difference.append(float(layer_slice.mean().item()))
        start_index += num_valid_entries

    return per_layer_difference, float(sum(per_layer_difference) / max(len(per_layer_difference), 1))


def summarize_outputs(
    vectors: torch.Tensor,
    sources: list[str],
    model_kind: str,
    num_layers: int,
) -> dict:
    wiki_indices = [index for index, source in enumerate(sources) if source == "wikitext"]
    medical_indices = [index for index, source in enumerate(sources) if source == "medical"]
    wiki_vectors = vectors[wiki_indices]
    medical_vectors = vectors[medical_indices]

    collapse_cosine = compute_mean_pairwise_cosine_similarity(vectors)
    within_wiki_cosine = compute_mean_pairwise_cosine_similarity(wiki_vectors)
    within_medical_cosine = compute_mean_pairwise_cosine_similarity(medical_vectors)
    between_domain_cosine = compute_cross_cosine_similarity(wiki_vectors, medical_vectors)

    if (
        within_wiki_cosine is None
        or within_medical_cosine is None
        or between_domain_cosine is None
    ):
        domain_separation_cosine_gap = None
    else:
        mean_within_domain_cosine = 0.5 * (within_wiki_cosine + within_medical_cosine)
        domain_separation_cosine_gap = mean_within_domain_cosine - between_domain_cosine

    if model_kind == "write_strength":
        per_layer_variance, overall_variance_mean = summarize_write_strength_per_layer_variance(
            vectors
        )
        (
            per_layer_domain_mean_difference,
            overall_domain_mean_difference,
        ) = summarize_write_strength_domain_mean_difference(
            wiki_vectors=wiki_vectors,
            medical_vectors=medical_vectors,
        )
    else:
        per_layer_variance, overall_variance_mean = (
            summarize_reaggregation_per_layer_variance(
                vectors=vectors,
                num_layers=num_layers,
            )
        )
        (
            per_layer_domain_mean_difference,
            overall_domain_mean_difference,
        ) = summarize_reaggregation_domain_mean_difference(
            wiki_vectors=wiki_vectors,
            medical_vectors=medical_vectors,
            num_layers=num_layers,
        )

    return {
        "collapse_cosine": collapse_cosine,
        "domain_separation_cosine_gap": domain_separation_cosine_gap,
        "per_layer_variance": per_layer_variance,
        "overall_variance_mean": overall_variance_mean,
        "per_layer_domain_mean_difference": per_layer_domain_mean_difference,
        "overall_domain_mean_difference": overall_domain_mean_difference,
    }


def inspect_model_on_split(
    model,
    dataloader: DataLoader,
    device: torch.device,
    split_name: str,
    model_label: str,
    model_kind: str,
    num_layers: int,
) -> dict:
    vectors, sources, _example_ids = collect_sampled_outputs(
        model=model,
        dataloader=dataloader,
        device=device,
        split_name=split_name,
        model_label=model_label,
    )
    return summarize_outputs(
        vectors=vectors,
        sources=sources,
        model_kind=model_kind,
        num_layers=num_layers,
    )


def main() -> None:
    args = parse_args()
    device, model_dtype = create_runtime()
    print(f"[setup] device={device} model_dtype={model_dtype}")

    write_strength_model, write_strength_checkpoint = (
        load_prompt_conditioned_write_strength_model_for_eval(
            checkpoint_path=args.write_strength_checkpoint,
            model_dir=args.model_dir,
            device=device,
            model_dtype=model_dtype,
            attn_implementation=args.attn_implementation,
        )
    )
    reaggregation_model, reaggregation_checkpoint = (
        load_prompt_conditioned_reaggregation_model_for_eval(
            checkpoint_path=args.reaggregation_checkpoint,
            model_dir=args.model_dir,
            device=device,
            model_dtype=model_dtype,
            attn_implementation=args.attn_implementation,
        )
    )

    split_to_path = {
        "validation": args.validation_data,
        "test": args.test_data,
    }
    split_to_seed = {
        "validation": args.seed,
        "test": args.seed + 1,
    }

    split_to_dataloader: dict[str, DataLoader] = {}
    split_to_indices: dict[str, list[int]] = {}
    for split_name, arrow_path in split_to_path.items():
        dataloader, indices = create_subset_dataloader(
            arrow_path=arrow_path,
            sample_size=args.sample_size,
            batch_size=args.batch_size,
            seed=split_to_seed[split_name],
            pin_memory=device.type == "cuda",
        )
        split_to_dataloader[split_name] = dataloader
        split_to_indices[split_name] = indices
        print(
            f"[sample] split={split_name} sampled_examples={len(indices)} "
            f"source={arrow_path}"
        )

    results = {
        "config": {
            "sample_size": args.sample_size,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": str(device),
            "model_dtype": str(model_dtype),
            "splits": {
                split_name: {
                    "path": split_to_path[split_name],
                    "sampled_num_examples": len(split_to_indices[split_name]),
                }
                for split_name in split_to_path
            },
        },
        "prompt_conditioned_write_strength": {
            "checkpoint_path": args.write_strength_checkpoint,
            "train_config": write_strength_checkpoint["train_config"],
            "splits": {},
        },
        "prompt_conditioned_residual_stream_reaggregation": {
            "checkpoint_path": args.reaggregation_checkpoint,
            "train_config": reaggregation_checkpoint["train_config"],
            "splits": {},
        },
    }

    for split_name, dataloader in split_to_dataloader.items():
        results["prompt_conditioned_write_strength"]["splits"][split_name] = (
            inspect_model_on_split(
                model=write_strength_model,
                dataloader=dataloader,
                device=device,
                split_name=split_name,
                model_label="write_strength",
                model_kind="write_strength",
                num_layers=write_strength_model.num_layers,
            )
        )
        results["prompt_conditioned_residual_stream_reaggregation"]["splits"][
            split_name
        ] = inspect_model_on_split(
            model=reaggregation_model,
            dataloader=dataloader,
            device=device,
            split_name=split_name,
            model_label="reaggregation",
            model_kind="reaggregation",
            num_layers=reaggregation_model.num_layers,
        )

    output_path = Path(args.output_path)
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] Saved inspection summary to: {output_path}")


if __name__ == "__main__":
    main()
