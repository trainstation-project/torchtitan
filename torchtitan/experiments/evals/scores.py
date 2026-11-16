# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scores derived from lm-eval results: gold BPB and per-step composites.

Gold BPB is the bits per byte of the correct answer's text given the prompt:
the negative log-likelihood lm-eval already computed for the correct choice,
divided by ln 2 and by the UTF-8 bytes of exactly the text whose
log-likelihood was summed (the continuation with its leading space, see
``scored_text``).
It changes smoothly with training, so small runs can be compared on it long
before accuracy leaves chance (Signal and Noise, arXiv 2508.13144).
"""

import json
import math
import os
import statistics
from typing import Any

GOLD_BPB = "gold_bpb,none"
BPB = "bits_per_byte,none"

# Per-step composites: the mean gold BPB of each group of members, and the
# mean cloze accuracy, each computed once a step has results for every member.
KNOWLEDGE = [
    ("arc_challenge", GOLD_BPB),
    ("arc_easy", GOLD_BPB),
    ("hellaswag", GOLD_BPB),
    ("trainstation_mmlu_cloze", GOLD_BPB),
    ("triviaqa_gold_bpb", BPB),
]
MATH = [("gsm8k_gold_bpb", BPB), ("math500_gold_bpb", BPB)]
CODE = [("humaneval_gold_bpb", BPB), ("mbpp_gold_bpb", BPB)]
CLOZE_ACCURACY = [
    "arc_challenge",
    "arc_easy",
    "hellaswag",
    "piqa",
    "winogrande",
    "openbookqa",
    "commonsense_qa_cloze",
    "social_iqa",
    "trainstation_mmlu_cloze",
]
SUMMARY_FILE = "summary.json"


def gold_index(sample: dict[str, Any]) -> int:
    """Index of the correct choice of a multiple-choice sample.

    Resolved as lm-eval's ``process_results`` resolves it. lm-eval's
    ``doc_to_target`` already turns a digit string into an integer, which is
    the index; a string target is the text of the correct choice. WinoGrande is
    lm-eval's multiple-input task: the choices differ in context and share one
    continuation, and the correct option (1 or 2) is in the document.
    """
    target = sample["target"]
    continuations = [continuation for _, continuation in sample["arguments"]]
    if len(set(continuations)) == 1 and "option1" in sample["doc"]:
        return int(sample["doc"]["answer"]) - 1
    if isinstance(target, int) and not isinstance(target, bool):
        if not 0 <= target < len(continuations):
            raise ValueError(f"Target {target} is not a choice index")
        return target
    if isinstance(target, str):
        matches = [
            i for i, c in enumerate(continuations) if c.strip() == target.strip()
        ]
        if len(matches) == 1:
            return matches[0]
    raise ValueError(f"Target {target!r} does not name exactly one choice")


def scored_text(context: str, continuation: str) -> str:
    """The text whose log-likelihood lm-eval sums for one request.

    ``TemplateLM._encode_pair`` moves the context's trailing whitespace onto
    the continuation before scoring it.
    """
    return context[len(context.rstrip()) :] + continuation


def _is_option_label(choice: str) -> bool:
    label = choice.strip().strip("().:")
    return len(label) == 1 and label.isascii() and label.isalpha()


def task_gold_nats_and_bytes(
    samples: list[dict[str, Any]],
) -> tuple[float, int] | None:
    """Summed NLL (nats) and UTF-8 bytes of the correct choices of one task.

    None for a lettered task, whose choices are option labels ("A", "(B)"):
    the label's BPB says nothing about the answer.
    """
    # In memory, lm-eval keeps each choice's request as (context, continuation).
    if all(
        _is_option_label(continuation)
        for sample in samples
        for _, continuation in sample["arguments"]
    ):
        return None
    nats = 0.0
    num_bytes = 0
    for sample in samples:
        gold = gold_index(sample)
        nats -= float(sample["resps"][gold][0][0])
        num_bytes += len(scored_text(*sample["arguments"][gold]).encode("utf-8"))
    return nats, num_bytes


def add_gold_bpb(results: dict[str, Any]) -> None:
    """Add ``gold_bpb`` to each multiple-choice task in lm-eval ``results``.

    Needs lm-eval's ``samples`` (``log_samples=True``). A group gets the mean
    of its tasks' gold BPB weighted by their number of questions, as lm-eval
    weights MMLU's accuracy.
    """
    configs = results.get("configs", {})
    task_bpb: dict[str, tuple[float, int]] = {}
    for task, samples in results.get("samples", {}).items():
        if not samples or configs.get(task, {}).get("output_type") != "multiple_choice":
            continue
        total = task_gold_nats_and_bytes(samples)
        if total is None:
            continue
        nats, num_bytes = total
        bpb = nats / (math.log(2) * num_bytes)
        results["results"][task][GOLD_BPB] = bpb
        task_bpb[task] = (bpb, len(samples))

    group_subtasks: dict[str, list[str]] = results.get("group_subtasks", {})

    def leaves(name: str) -> list[str]:
        if name not in group_subtasks:
            return [name]
        return [leaf for child in group_subtasks[name] for leaf in leaves(child)]

    for group in group_subtasks:
        members = [task_bpb[t] for t in dict.fromkeys(leaves(group)) if t in task_bpb]
        if members and group in results["results"]:
            results["results"][group][GOLD_BPB] = sum(
                bpb * n for bpb, n in members
            ) / sum(n for _, n in members)


def step_metrics(step_folder: str, units: list[str]) -> dict[str, dict[str, Any]]:
    """The metrics of ``units`` in a step's results files, keyed by task.

    Only the suite's own units are read, so results files written into the
    same folder by other suites never enter its summary.
    """
    metrics: dict[str, dict[str, Any]] = {}
    for unit in units:
        path = os.path.join(step_folder, f"{unit}.json")
        if os.path.exists(path):
            with open(path) as f:
                metrics.update(json.load(f).get("results", {}))
    return metrics


def _mean(
    metrics: dict[str, dict[str, Any]], members: list[tuple[str, str]]
) -> float | None:
    values = [metrics.get(task, {}).get(key) for task, key in members]
    if any(v is None for v in values):
        return None
    return statistics.fmean(values)


def step_summary(step_folder: str, units: list[str]) -> dict[str, float]:
    """The step's summary.json, updated with the composites of ``units``.

    Composites are merged into the existing summary, so a later run of another
    suite on the same step (a milestone suite, a decontaminated pass) keeps the
    composites it cannot compute itself.
    """
    path = os.path.join(step_folder, SUMMARY_FILE)
    summary: dict[str, float] = {}
    if os.path.exists(path):
        with open(path) as f:
            summary = json.load(f)["summary"]
    summary.update(summarize(step_metrics(step_folder, units)))
    return summary


def summarize(metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
    """The composites a step has every member for."""
    summary = {
        "knowledge_gold_bpb": _mean(metrics, KNOWLEDGE),
        "math_gold_bpb": _mean(metrics, MATH),
        "code_gold_bpb": _mean(metrics, CODE),
        "cloze_acc": _mean(
            metrics,
            [
                (
                    task,
                    "acc_norm,none"
                    if "acc_norm,none" in metrics.get(task, {})
                    else "acc,none",
                )
                for task in CLOZE_ACCURACY
            ],
        ),
        "paloma_bpb": metrics.get("trainstation_paloma", {}).get(BPB),
        "openwebmath_bpb": metrics.get("trainstation_openwebmath", {}).get(BPB),
    }
    return {name: value for name, value in summary.items() if value is not None}
