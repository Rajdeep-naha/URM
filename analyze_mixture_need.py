"""
Offline validation check for whether a 3-component Logistic mixture is warranted.

This script follows the recommended first investigation step for the URM MDN:
take held-out validation targets, fit a single Logistic and a 3-component
Logistic mixture offline, and compare held-out log-likelihood.

It supports two sources of validation values:
1. A saved diagnostics dump from `train_attribute_mdn.py`
2. Recomputing validation labels or residuals from a trained head checkpoint
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from scipy.optimize import minimize
from scipy.special import logsumexp
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.modeling_mdn_urm import LlamaForSequenceClassificationWithMDN
from train_attribute_mdn import ATTRIBUTES, HelpSteer2Dataset, remove_hooks_and_materialize_meta_parameters


EPS = 1e-8


@dataclass(frozen=True)
class FitResult:
    avg_nll: float
    total_log_likelihood: float
    bic: float
    aic: float
    converged: bool
    parameters: dict[str, list[float] | float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare 1x Logistic vs 3x Logistic mixture on URM validation targets.")
    parser.add_argument("--uncertainty_target", choices=("label", "residual"), required=True)
    parser.add_argument("--diagnostic_dump", type=str, default=None, help="Optional path to best_validation_diagnostics.pt from training.")
    parser.add_argument("--model_name_or_path", type=str, default="LxzGordon/URM-LLaMa-3.1-8B")
    parser.add_argument("--mdn_head_weights", type=str, default="checkpoints/stage1/best_mdn_head.pt")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--num_components", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on validation rows for faster analysis.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/mixture_need")
    return parser.parse_args()


def logistic_logpdf(values: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (values - loc) / scale
    return -np.log(scale) - np.logaddexp(0.0, z) - np.logaddexp(0.0, -z)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def fit_single_logistic(values: np.ndarray) -> FitResult:
    mean = float(np.mean(values))
    std = float(np.std(values))
    init = np.array([mean, math.log(max(std * math.sqrt(3.0) / math.pi, 1e-2))], dtype=np.float64)

    def objective(theta: np.ndarray) -> float:
        loc = theta[0]
        scale = np.exp(theta[1]) + EPS
        return float(-logistic_logpdf(values, loc, scale).sum())

    result = minimize(objective, init, method="L-BFGS-B")
    loc = float(result.x[0])
    scale = float(np.exp(result.x[1]) + EPS)
    total_ll = -float(result.fun)
    num_params = 2
    n = len(values)
    return FitResult(
        avg_nll=-total_ll / n,
        total_log_likelihood=total_ll,
        bic=num_params * math.log(n) - 2.0 * total_ll,
        aic=2.0 * num_params - 2.0 * total_ll,
        converged=bool(result.success),
        parameters={"loc": loc, "scale": scale},
    )


def fit_logistic_mixture(values: np.ndarray, num_components: int = 3, num_restarts: int = 6, seed: int = 42) -> FitResult:
    rng = np.random.default_rng(seed)
    mean = float(np.mean(values))
    std = float(np.std(values))
    std = max(std, 1e-2)

    best_result = None
    best_objective = float("inf")

    base_means = np.linspace(mean - std, mean + std, num_components, dtype=np.float64)
    base_scales = np.full(num_components, max(std * math.sqrt(3.0) / math.pi, 1e-2), dtype=np.float64)

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weight_logits = theta[:num_components]
        means = theta[num_components: 2 * num_components]
        log_scales = theta[2 * num_components:]
        weights = softmax(weight_logits)
        scales = np.exp(log_scales) + EPS
        order = np.argsort(means)
        return weights[order], means[order], scales[order]

    def objective(theta: np.ndarray) -> float:
        weights, means, scales = unpack(theta)
        component_ll = np.stack(
            [np.log(weights[k] + EPS) + logistic_logpdf(values, means[k], scales[k]) for k in range(num_components)],
            axis=0,
        )
        return float(-logsumexp(component_ll, axis=0).sum())

    for restart in range(num_restarts):
        if restart == 0:
            init_means = base_means.copy()
            init_scales = base_scales.copy()
            init_logits = np.zeros(num_components, dtype=np.float64)
        else:
            init_means = base_means + rng.normal(scale=0.35 * std, size=num_components)
            init_scales = np.clip(base_scales * np.exp(rng.normal(scale=0.2, size=num_components)), 1e-2, None)
            init_logits = rng.normal(scale=0.2, size=num_components)

        init = np.concatenate([init_logits, init_means, np.log(init_scales)])
        result = minimize(objective, init, method="L-BFGS-B")
        if result.fun < best_objective:
            best_objective = float(result.fun)
            best_result = result

    if best_result is None:
        raise RuntimeError("Mixture fitting failed before optimization started.")

    weights, means, scales = unpack(best_result.x)
    total_ll = -best_objective
    num_params = (num_components - 1) + num_components + num_components
    n = len(values)
    return FitResult(
        avg_nll=-total_ll / n,
        total_log_likelihood=total_ll,
        bic=num_params * math.log(n) - 2.0 * total_ll,
        aic=2.0 * num_params - 2.0 * total_ll,
        converged=bool(best_result.success),
        parameters={
            "weights": weights.tolist(),
            "means": means.tolist(),
            "scales": scales.tolist(),
        },
    )


def load_diagnostic_values(path: str, uncertainty_target: str) -> dict[str, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("uncertainty_target") != uncertainty_target:
        raise ValueError(
            f"Diagnostic dump target {payload.get('uncertainty_target')!r} does not match requested {uncertainty_target!r}."
        )

    key = "labels_orig" if uncertainty_target == "label" else "residuals_orig"
    values = payload[key].to(torch.float32).numpy()
    return {attr: values[:, idx] for idx, attr in enumerate(payload["attributes"])}


def compute_validation_values(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.diagnostic_dump:
        return load_diagnostic_values(args.diagnostic_dump, args.uncertainty_target)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("nvidia/HelpSteer2")
    train_data = dataset["train"]
    val_data = dataset["validation"]
    if args.max_samples is not None:
        val_data = val_data.select(range(min(args.max_samples, len(val_data))))

    attr_means = []
    attr_stds = []
    for attr in ATTRIBUTES:
        train_values = np.array([float(row[attr]) for row in train_data], dtype=np.float32)
        attr_means.append(float(train_values.mean()))
        attr_stds.append(float(max(train_values.std(), 1e-6)))

    attr_means_t = torch.tensor(attr_means, dtype=torch.float32)
    attr_stds_t = torch.tensor(attr_stds, dtype=torch.float32)
    val_dataset = HelpSteer2Dataset(val_data, tokenizer, max_length=args.max_length, means=attr_means_t, stds=attr_stds_t)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    if args.uncertainty_target == "label":
        labels_orig = []
        for batch in val_loader:
            labels_norm = batch["labels"].to(torch.float32)
            labels_orig.append(labels_norm * attr_stds_t + attr_means_t)
        labels_orig_all = torch.cat(labels_orig, dim=0).numpy()
        return {attr: labels_orig_all[:, idx] for idx, attr in enumerate(ATTRIBUTES)}

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = LlamaForSequenceClassificationWithMDN.from_pretrained(
        args.model_name_or_path,
        ignore_mismatched_sizes=True,
        torch_dtype=dtype,
        device_map=args.device_map,
        num_components=args.num_components,
        uncertainty_target=args.uncertainty_target,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model.score.proj.weight.device.type == "meta":
        remove_hooks_and_materialize_meta_parameters(model, device)
    state_dict = torch.load(args.mdn_head_weights, map_location="cpu", weights_only=False)
    model.score.load_state_dict(state_dict, assign=True)
    model.eval()

    if not hasattr(model, "hf_device_map"):
        model.to(device)

    labels_orig_batches = []
    residuals_orig_batches = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_norm = batch["labels"].to(device).to(dtype)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
            expected_rewards_norm = outputs[2].to(torch.float32).cpu()

            labels_norm_cpu = labels_norm.to(torch.float32).cpu()
            labels_orig = labels_norm_cpu * attr_stds_t + attr_means_t
            expected_rewards_orig = expected_rewards_norm * attr_stds_t + attr_means_t

            labels_orig_batches.append(labels_orig)
            residuals_orig_batches.append(labels_orig - expected_rewards_orig)

    labels_orig_all = torch.cat(labels_orig_batches, dim=0).numpy()
    residuals_orig_all = torch.cat(residuals_orig_batches, dim=0).numpy()
    source = labels_orig_all if args.uncertainty_target == "label" else residuals_orig_all
    return {attr: source[:, idx] for idx, attr in enumerate(ATTRIBUTES)}


def summarize_preference(single: FitResult, mixture: FitResult) -> str:
    if mixture.bic + 10.0 < single.bic:
        return "3-component mixture clearly preferred"
    if single.bic + 10.0 < mixture.bic:
        return "single Logistic sufficient"
    if mixture.avg_nll + 0.01 < single.avg_nll:
        return "mixture modestly better"
    return "no strong evidence for mixture"


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    values_by_attr = compute_validation_values(args)
    rows = []
    report_lines = [
        f"# Mixture Need Analysis ({args.uncertainty_target} target)",
        "",
        "Held-out validation values were fit offline with:",
        "- one Logistic distribution",
        "- one 3-component Logistic mixture",
        "",
    ]

    for attr in ATTRIBUTES:
        values = np.asarray(values_by_attr[attr], dtype=np.float64)
        single = fit_single_logistic(values)
        mixture = fit_logistic_mixture(values, num_components=args.num_components, seed=args.seed)
        conclusion = summarize_preference(single, mixture)

        rows.append(
            {
                "attribute": attr,
                "n_samples": len(values),
                "single_avg_nll": single.avg_nll,
                "mix3_avg_nll": mixture.avg_nll,
                "delta_avg_nll": single.avg_nll - mixture.avg_nll,
                "single_bic": single.bic,
                "mix3_bic": mixture.bic,
                "single_aic": single.aic,
                "mix3_aic": mixture.aic,
                "single_converged": single.converged,
                "mix3_converged": mixture.converged,
                "conclusion": conclusion,
            }
        )

        report_lines.extend(
            [
                f"## {attr}",
                f"- Samples: {len(values)}",
                f"- Single Logistic avg NLL: {single.avg_nll:.4f}",
                f"- 3-component mixture avg NLL: {mixture.avg_nll:.4f}",
                f"- Delta avg NLL (single - mix3): {single.avg_nll - mixture.avg_nll:.4f}",
                f"- BIC: single={single.bic:.2f}, mix3={mixture.bic:.2f}",
                f"- Conclusion: {conclusion}",
                "",
            ]
        )

    results = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, f"{args.uncertainty_target}_mixture_need.csv")
    md_path = os.path.join(args.output_dir, f"{args.uncertainty_target}_mixture_need.md")
    results.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines) + "\n")

    print(results.to_string(index=False))
    print(f"\nSaved CSV to {csv_path}")
    print(f"Saved report to {md_path}")


if __name__ == "__main__":
    main()
